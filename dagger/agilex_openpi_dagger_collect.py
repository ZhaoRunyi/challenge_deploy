from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import signal
import termios
import threading
import time
import tty

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.buffer import StreamActionBuffer
from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.constants import CAMERA_NAMES
from challenge_deploy.dataset import EpisodeCollector, delete_episode_artifacts
from challenge_deploy.observation import build_policy_payload
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.policy import OpenPiPolicyClient
from challenge_deploy.realsense import RealSenseRig
from challenge_deploy.runtime import DualPiperObservationSource


shutdown_event = threading.Event()
dagger_mode_active = False
save_data_requested = False
collection_active = False
delete_data_requested = False
waiting_for_user_confirm = False

dagger_mode_lock = threading.Lock()
save_data_lock = threading.Lock()
collection_lock = threading.Lock()
delete_data_lock = threading.Lock()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROS-free Challenge Deploy Agilex DAgger collection.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", type=str, default=None)
    parser.add_argument("--right-can", type=str, default=None)
    parser.add_argument("--master-left-can", type=str, default=None)
    parser.add_argument("--master-right-can", type=str, default=None)
    parser.add_argument("--camera-front-serial", type=str, default=None)
    parser.add_argument("--camera-left-serial", type=str, default=None)
    parser.add_argument("--camera-right-serial", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--enable-master-rig", action="store_true")
    parser.add_argument("--no-policy", action="store_true", help="Skip remote policy inference and only collect DAgger episodes.")
    return parser


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.left_can:
        set_by_dotted_path(config, "robot.left.can_name", args.left_can)
    if args.right_can:
        set_by_dotted_path(config, "robot.right.can_name", args.right_can)
    if args.master_left_can:
        set_by_dotted_path(config, "robot.master_left.can_name", args.master_left_can)
    if args.master_right_can:
        set_by_dotted_path(config, "robot.master_right.can_name", args.master_right_can)
    if args.camera_front_serial:
        set_by_dotted_path(config, "cameras.serials.cam_high", args.camera_front_serial)
    if args.camera_right_serial:
        set_by_dotted_path(config, "cameras.serials.cam_right_wrist", args.camera_right_serial)
    if args.camera_left_serial:
        set_by_dotted_path(config, "cameras.serials.cam_left_wrist", args.camera_left_serial)
    if args.host:
        set_by_dotted_path(config, "policy.host", args.host)
    if args.port is not None:
        set_by_dotted_path(config, "policy.port", args.port)
    if args.prompt:
        set_by_dotted_path(config, "policy.prompt", args.prompt)
    if args.dataset_dir:
        set_by_dotted_path(config, "dataset.dataset_dir", args.dataset_dir)
    if args.dataset_name:
        set_by_dotted_path(config, "dataset.dataset_name", args.dataset_name)
    return config


def keyboard_monitor_thread() -> None:
    global dagger_mode_active, save_data_requested, collection_active
    global delete_data_requested, waiting_for_user_confirm

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not shutdown_event.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                char = sys.stdin.read(1)
                with delete_data_lock:
                    is_waiting = waiting_for_user_confirm
                if is_waiting:
                    if char.lower() == "w":
                        with delete_data_lock:
                            delete_data_requested = True
                            waiting_for_user_confirm = False
                    else:
                        with delete_data_lock:
                            waiting_for_user_confirm = False
                    continue

                if char.lower() == "d":
                    with dagger_mode_lock:
                        dagger_mode_active = True
                    print("DAgger mode activated.")
                elif char.lower() == "r":
                    with dagger_mode_lock:
                        dagger_mode_active = False
                    print("Returned to inference mode.")
                elif char.lower() == "s":
                    with save_data_lock:
                        save_data_requested = True
                    print("Save requested.")
                elif char == " ":
                    with collection_lock:
                        collection_active = True
                    print("Collection started.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def confirm_or_delete(saved_path: Path) -> None:
    global delete_data_requested, waiting_for_user_confirm
    with delete_data_lock:
        waiting_for_user_confirm = True
        delete_data_requested = False
    print(f"Saved {saved_path}. Press 'w' within 10 seconds to delete it, or any other key to keep it.")
    start = time.time()
    while time.time() - start < 10.0:
        with delete_data_lock:
            if delete_data_requested:
                delete_episode_artifacts(saved_path.with_suffix(""), list(CAMERA_NAMES))
                print(f"Deleted {saved_path}.")
                delete_data_requested = False
                waiting_for_user_confirm = False
                return
            if not waiting_for_user_confirm:
                return
        time.sleep(0.1)
    with delete_data_lock:
        waiting_for_user_confirm = False


def start_policy_thread(
    *,
    source: DualPiperObservationSource,
    prompt: str,
    policy: OpenPiPolicyClient,
    buffer: StreamActionBuffer,
    inference_rate: float,
    latency_k: int,
    min_smooth_steps: int,
) -> threading.Thread:
    def runner() -> None:
        while not shutdown_event.is_set():
            with dagger_mode_lock:
                if dagger_mode_active:
                    time.sleep(0.05)
                    continue
            snapshot = source.capture_snapshot()
            payload = build_policy_payload(snapshot, prompt)
            actions = policy.infer(payload)["actions"]
            if actions is not None and len(actions) > 0:
                buffer.integrate_new_chunk(np.asarray(actions, dtype=np.float64), latency_k, min_smooth_steps)
            time.sleep(max(0.0, 1.0 / inference_rate))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread


def main() -> None:
    global collection_active, save_data_requested

    args = build_parser().parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    commands_enabled = True

    def _shutdown(_signum=None, _frame=None) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    slave_robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=commands_enabled,
        name="slave_dual_piper",
    )
    master_robot = None
    if args.enable_master_rig:
        master_robot = DualPiperSystem(
            left_can_name=config["robot"]["master_left"]["can_name"],
            right_can_name=config["robot"]["master_right"]["can_name"],
            commands_enabled=commands_enabled,
            prefer_joint_ctrl=True,
            name="master_dual_piper",
        )

    cameras = RealSenseRig(
        config["cameras"]["serials"],
        width=int(config["cameras"]["width"]),
        height=int(config["cameras"]["height"]),
        fps=int(config["cameras"]["fps"]),
        warmup_frames=int(config["cameras"]["warmup_frames"]),
    )
    source = DualPiperObservationSource(robot=slave_robot, cameras=cameras)

    slave_robot.connect(read_only=not commands_enabled)
    if master_robot is not None:
        master_robot.connect(read_only=not commands_enabled)

    inference_collector = EpisodeCollector(
        camera_names=list(CAMERA_NAMES),
        dataset_dir=Path(config["dataset"]["dataset_dir"]) / f"{config['dataset']['dataset_name']}_inference_hdf5",
        dataset_name="aloha_mobile_dummy",
    )
    dagger_collector = EpisodeCollector(
        camera_names=list(CAMERA_NAMES),
        dataset_dir=Path(config["dataset"]["dataset_dir"]) / f"{config['dataset']['dataset_name']}_dagger_hdf5",
        dataset_name="aloha_mobile_dummy",
    )

    policy = None
    buffer = StreamActionBuffer(
        max_chunks=int(config["policy"]["buffer_max_chunks"]),
        state_dim=14,
        smooth_method="temporal",
    )

    try:
        cameras.start()
        if not source.wait_until_ready(timeout_s=15.0):
            raise RuntimeError("Timed out waiting for slave robot / camera data")
        print("Initial slave state:")
        print(json.dumps(source.capture_snapshot().state.to_dict(), indent=2))
        if not args.no_policy:
            policy = OpenPiPolicyClient(config["policy"]["host"], int(config["policy"]["port"]))
            print("Server metadata:", policy.get_server_metadata())
            start_policy_thread(
                source=source,
                prompt=str(config["policy"]["prompt"]),
                policy=policy,
                buffer=buffer,
                inference_rate=float(config["policy"]["inference_rate"]),
                latency_k=int(config["policy"]["latency_k"]),
                min_smooth_steps=int(config["policy"]["min_smooth_steps"]),
            )

        keyboard_thread = threading.Thread(target=keyboard_monitor_thread, daemon=True)
        keyboard_thread.start()

        if not args.no_policy:
            inference_collector.start()

        prepared_dagger = False
        publish_rate = float(config["runtime"]["publish_rate"])
        gripper_offset = float(config["runtime"]["right_gripper_offset"])

        while not shutdown_event.is_set():
            with dagger_mode_lock:
                in_dagger = dagger_mode_active

            if not in_dagger:
                prepared_dagger = False
                if not args.no_policy and not inference_collector.is_collecting:
                    inference_collector.start()

                action = buffer.pop_next_action() if policy is not None else None
                if action is not None:
                    snapshot = source.capture_snapshot()
                    action = np.asarray(action, dtype=np.float64).copy()
                    action[6] = max(0.0, action[6] - gripper_offset)
                    action[13] = max(0.0, action[13] - gripper_offset)
                    inference_collector.add_frame(snapshot.to_collector_observation(), action)
                    slave_robot.set_joint_positions(action, speed_percent=100)
                time.sleep(1.0 / publish_rate)
                continue

            if not prepared_dagger:
                if inference_collector.has_data():
                    saved = inference_collector.save_current_episode(
                        export_video=bool(config["dataset"]["export_video"]),
                        video_fps=int(config["dataset"]["video_fps"]),
                    )
                    print(f"Saved inference-phase episode to {saved}")
                if master_robot is not None:
                    master_robot.configure_masters_for_teaching(align_to=source.capture_snapshot().state.qpos)
                prepared_dagger = True

            with collection_lock:
                collecting = collection_active
            if collecting and not dagger_collector.is_collecting:
                dagger_collector.start()

            if collecting:
                snapshot = source.capture_snapshot()
                if master_robot is not None:
                    action_state = master_robot.read_state(prefer_joint_ctrl=True)
                    action = action_state.qpos
                else:
                    action = snapshot.state.qpos
                dagger_collector.add_frame(snapshot.to_collector_observation(), action)

            with save_data_lock:
                do_save = save_data_requested
                if do_save:
                    save_data_requested = False
            if do_save and dagger_collector.has_data():
                saved = dagger_collector.save_current_episode(
                    export_video=bool(config["dataset"]["export_video"]),
                    video_fps=int(config["dataset"]["video_fps"]),
                )
                with collection_lock:
                    collection_active = False
                print(f"Saved DAgger episode to {saved}")
                confirm_or_delete(saved)

            time.sleep(1.0 / publish_rate)
    finally:
        shutdown_event.set()
        cameras.stop()
        slave_robot.disconnect()
        if master_robot is not None:
            master_robot.disconnect()


if __name__ == "__main__":
    main()
