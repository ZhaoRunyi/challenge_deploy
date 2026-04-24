from __future__ import annotations

import argparse
from pathlib import Path
import signal
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.buffer import StreamActionBuffer
from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.constants import CAMERA_NAMES
from challenge_deploy.observation import build_policy_payload
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.policy import OpenPiPolicyClient
from challenge_deploy.realsense import RealSenseRig
from challenge_deploy.runtime import DualPiperObservationSource


shutdown_event = threading.Event()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROS-free Challenge Deploy Agilex inference with temporal smoothing.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", type=str, default=None)
    parser.add_argument("--right-can", type=str, default=None)
    parser.add_argument("--camera-front-serial", type=str, default=None)
    parser.add_argument("--camera-left-serial", type=str, default=None)
    parser.add_argument("--camera-right-serial", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--probe-only", action="store_true", help="Only verify robot/camera connectivity, then exit.")
    parser.add_argument("--max_publish_step", type=int, default=None)
    parser.add_argument("--publish_rate", type=float, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--latency_k", type=int, default=None)
    parser.add_argument("--inference_rate", type=float, default=None)
    parser.add_argument("--min_smooth_steps", type=int, default=None)
    parser.add_argument("--buffer_max_chunks", type=int, default=None)
    parser.add_argument("--right_offset", type=float, default=None)
    parser.add_argument("--no-cameras", action="store_true")
    return parser


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.left_can:
        set_by_dotted_path(config, "robot.left.can_name", args.left_can)
    if args.right_can:
        set_by_dotted_path(config, "robot.right.can_name", args.right_can)
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
    if args.max_publish_step is not None:
        set_by_dotted_path(config, "runtime.max_publish_step", args.max_publish_step)
    if args.publish_rate is not None:
        set_by_dotted_path(config, "runtime.publish_rate", args.publish_rate)
    if args.chunk_size is not None:
        set_by_dotted_path(config, "policy.chunk_size", args.chunk_size)
    if args.latency_k is not None:
        set_by_dotted_path(config, "policy.latency_k", args.latency_k)
    if args.inference_rate is not None:
        set_by_dotted_path(config, "policy.inference_rate", args.inference_rate)
    if args.min_smooth_steps is not None:
        set_by_dotted_path(config, "policy.min_smooth_steps", args.min_smooth_steps)
    if args.buffer_max_chunks is not None:
        set_by_dotted_path(config, "policy.buffer_max_chunks", args.buffer_max_chunks)
    if args.right_offset is not None:
        set_by_dotted_path(config, "runtime.right_gripper_offset", args.right_offset)
    if args.no_cameras:
        set_by_dotted_path(config, "cameras.enabled", False)
    return config


def start_inference_thread(
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
            start = time.time()
            snapshot = source.capture_snapshot()
            payload = build_policy_payload(snapshot, prompt)
            actions = policy.infer(payload)["actions"]
            if actions is not None and len(actions) > 0:
                buffer.integrate_new_chunk(
                    np.asarray(actions, dtype=np.float64),
                    max_k=latency_k,
                    min_m=min_smooth_steps,
                )
            elapsed = time.time() - start
            sleep_s = max(0.0, (1.0 / inference_rate) - elapsed)
            time.sleep(sleep_s)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread


def main() -> None:
    args = build_parser().parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    commands_enabled = not args.probe_only

    robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=commands_enabled,
        name="agilex_inference",
    )
    cameras = None
    if config["cameras"]["enabled"]:
        cameras = RealSenseRig(
            config["cameras"]["serials"],
            width=int(config["cameras"]["width"]),
            height=int(config["cameras"]["height"]),
            fps=int(config["cameras"]["fps"]),
            warmup_frames=int(config["cameras"]["warmup_frames"]),
        )

    source = DualPiperObservationSource(robot=robot, cameras=cameras)

    def _shutdown(_signum=None, _frame=None) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    robot.connect(read_only=not commands_enabled)
    try:
        if cameras is not None:
            cameras.start()
        if not source.wait_until_ready(timeout_s=15.0):
            raise RuntimeError("Timed out waiting for robot/camera data")

        snapshot = source.capture_snapshot()
        print("Initial qpos:", np.array2string(snapshot.state.qpos, precision=4))
        print("Cameras:", sorted(snapshot.images.keys()))

        if args.probe_only:
            return

        policy = OpenPiPolicyClient(
            host=config["policy"]["host"],
            port=int(config["policy"]["port"]),
        )
        print("Server metadata:", policy.get_server_metadata())

        buffer = StreamActionBuffer(
            max_chunks=int(config["policy"]["buffer_max_chunks"]),
            state_dim=14,
            smooth_method="temporal",
        )
        start_inference_thread(
            source=source,
            prompt=str(config["policy"]["prompt"]),
            policy=policy,
            buffer=buffer,
            inference_rate=float(config["policy"]["inference_rate"]),
            latency_k=int(config["policy"]["latency_k"]),
            min_smooth_steps=int(config["policy"]["min_smooth_steps"]),
        )

        publish_rate = float(config["runtime"]["publish_rate"])
        gripper_offset = float(config["runtime"]["right_gripper_offset"])
        max_publish_step = int(config["runtime"]["max_publish_step"])
        step = 0
        while not shutdown_event.is_set() and step < max_publish_step:
            action = buffer.pop_next_action()
            if action is None:
                time.sleep(1.0 / publish_rate)
                continue
            action = np.asarray(action, dtype=np.float64).copy()
            action[6] = max(0.0, action[6] - gripper_offset)
            action[13] = max(0.0, action[13] - gripper_offset)
            robot.set_joint_positions(action, speed_percent=100)
            step += 1
            time.sleep(1.0 / publish_rate)
    finally:
        shutdown_event.set()
        if cameras is not None:
            cameras.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
