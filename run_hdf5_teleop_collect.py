from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import Any, Callable

from data.worker import HDF5TeleopDataWorker, HDF5TeleopSaveConfig
from hardware.config import load_config, set_by_dotted_path
from hardware.piper import DualPiperSystem
from hardware.realsense import RealSenseRig
from rollout.recording import RuntimeExecutionWindow, RecordingSchema
from teleop.hdf5_teleop import (
    HDF5TeleopCollectionSource,
    HDF5_TELEOP_VECTOR_NAMES,
    collect_hdf5_teleop_episode,
    episode_base_path,
    infer_language_instruction,
    next_episode_index,
    running_sentinel_path,
)
from teleop.worker import TeleopWorker


DEPLOY_ROOT = Path(__file__).resolve().parent
DEFAULT_JPEG_QUALITY = 95
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IDLE_PROMPT = "Idle: press c to start an episode, or q to quit."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect HDF5 teleop episodes from dual Piper master/puppet arms and RealSense cameras."
    )
    parser.add_argument("--dataset-dir", default=None, help="Root directory that contains task folders.")
    parser.add_argument("--task-name", default=None, help="Task folder name under dataset-dir.")
    parser.add_argument("--episode-idx", type=int, default=None, help="First episode index. Default: next available.")
    parser.add_argument("--language-instruction", default=None, help="Episode-level language instruction.")
    parser.add_argument("--max-timesteps", type=int, default=None, help="Optional per-episode data frame cap. Default: no cap.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--arm-sample-hz", type=float, default=200.0, help="Polling rate for piper_sdk cached arm states before timestamp alignment.")
    parser.add_argument("--queue-maxlen", type=int, default=2000, help="Max async samples to keep per source, matching the original ROS deque cap by default.")
    parser.add_argument("--countdown-seconds", type=int, default=0)
    parser.add_argument("--alignment-plot-frames", type=int, default=16, help="Number of evenly spaced selected frames to draw in the alignment plot.")
    parser.add_argument("--record", action="store_true", help="Render a Motus/OpenPI-style deploy video from each saved HDF5 episode.")
    parser.add_argument("--window", nargs="?", const=1, type=int, default=0, help="Show live camera/action-state window; optional value selects display index.")
    parser.add_argument("--action-from-state", action="store_true", help="Save action[t] from puppet state[t+1] instead of master control[t+1].")
    parser.add_argument("--record-dir", default=str(DEPLOY_ROOT / "artifacts" / "hdf5_teleop_records"))
    parser.add_argument("--skip-idle", action=argparse.BooleanOptionalAction, default=True, help="Skip frames when master arms stay within the idle tolerance. Use --no-skip-idle to keep them.")
    parser.add_argument("--use-depth-image", action="store_true")
    parser.add_argument("--config", default=str(DEPLOY_ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--master-left-can", default=None)
    parser.add_argument("--master-right-can", default=None)
    parser.add_argument("--camera-high-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument("--extra-camera-names", nargs="*", default=())
    parser.add_argument("--extra-camera-serials", nargs="*", default=())
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    return parser


def apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.left_can:
        set_by_dotted_path(config, "robot.left.can_name", args.left_can)
    if args.right_can:
        set_by_dotted_path(config, "robot.right.can_name", args.right_can)
    master_left_can = args.master_left_can or config["robot"]["left"]["can_name"]
    master_right_can = args.master_right_can or config["robot"]["right"]["can_name"]
    set_by_dotted_path(config, "robot.master_left.can_name", master_left_can)
    set_by_dotted_path(config, "robot.master_right.can_name", master_right_can)
    if args.camera_high_serial:
        set_by_dotted_path(config, "cameras.serials.cam_high", args.camera_high_serial)
    if args.camera_left_serial:
        set_by_dotted_path(config, "cameras.serials.cam_left_wrist", args.camera_left_serial)
    if args.camera_right_serial:
        set_by_dotted_path(config, "cameras.serials.cam_right_wrist", args.camera_right_serial)
    for camera_name, serial in zip(args.extra_camera_names, args.extra_camera_serials):
        config["cameras"]["serials"][camera_name] = serial
    return config


def camera_names_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        camera_name
        for camera_name, serial in config["cameras"]["serials"].items()
        if serial
    )


def make_collection_source(
    config: dict[str, Any],
    *,
    enable_depth: bool,
    arm_sample_hz: float,
    queue_maxlen: int,
) -> tuple[DualPiperSystem, DualPiperSystem, RealSenseRig, HDF5TeleopCollectionSource]:
    puppet_robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=False,
        name="hdf5_teleop_puppet_reader",
    )
    master_robot = DualPiperSystem(
        left_can_name=config["robot"]["master_left"]["can_name"],
        right_can_name=config["robot"]["master_right"]["can_name"],
        commands_enabled=False,
        prefer_joint_ctrl=True,
        name="hdf5_teleop_master_reader",
    )
    cameras = RealSenseRig(
        config["cameras"]["serials"],
        width=int(config["cameras"]["width"]),
        height=int(config["cameras"]["height"]),
        fps=int(config["cameras"]["fps"]),
        warmup_frames=int(config["cameras"]["warmup_frames"]),
        enable_depth=enable_depth,
    )
    source = HDF5TeleopCollectionSource(
        master_robot=master_robot,
        puppet_robot=puppet_robot,
        cameras=cameras,
        arm_sample_hz=arm_sample_hz,
        queue_maxlen=queue_maxlen,
    )
    return master_robot, puppet_robot, cameras, source


def make_teleop_worker(config: dict[str, Any], args: argparse.Namespace) -> TeleopWorker:
    master_robot, puppet_robot, cameras, source = make_collection_source(
        config,
        enable_depth=args.use_depth_image,
        arm_sample_hz=args.arm_sample_hz,
        queue_maxlen=args.queue_maxlen,
    )

    def connect_master_robot() -> None:
        master_robot.connect(read_only=True)

    def connect_puppet_robot() -> None:
        puppet_robot.connect(read_only=True)

    return TeleopWorker(
        source=source,
        ready_timeout_s=args.ready_timeout,
        start_callbacks=(connect_master_robot, connect_puppet_robot),
        stop_callbacks=(
            ("cameras", cameras.stop),
            ("master robot", master_robot.disconnect),
            ("puppet robot", puppet_robot.disconnect),
        ),
    )


def make_data_worker(
    args: argparse.Namespace,
    language_instruction: str,
    camera_names: tuple[str, ...],
) -> HDF5TeleopDataWorker:
    config = HDF5TeleopSaveConfig(
        camera_names=camera_names,
        language_instruction=language_instruction,
        include_depth_images=args.use_depth_image,
        jpeg_quality=DEFAULT_JPEG_QUALITY,
        action_from_state=args.action_from_state,
        alignment_plot_frames=args.alignment_plot_frames,
        record_video=args.record,
        fps=args.fps,
    )
    return HDF5TeleopDataWorker(config=config)


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, writable, errors = select.select([sys.stdin], [], [], 0.0)
    del writable, errors
    if not ready:
        return None
    return sys.stdin.read(1).lower()


def wait_for_key(valid_keys: set[str], poll: Callable[[], None] | None = None) -> str:
    while True:
        if poll is not None:
            poll()
        key = read_key()
        if key in valid_keys:
            return key
        time.sleep(0.05)


def print_json(key: str, value: dict[str, Any]) -> None:
    print(json.dumps({key: value}, indent=2), flush=True)


def print_save_result(result: dict[str, Any]) -> None:
    print_json("hdf5_teleop_collection_result", result)


def print_save_error(exc: Exception) -> None:
    print(f"HDF5 teleop save failed: {exc}", flush=True)


def print_idle_prompt() -> None:
    print(IDLE_PROMPT, flush=True)


def print_episode_decision_prompt() -> None:
    print("Episode stopped: press c to save and continue, or d to discard and continue.", flush=True)


def validate_camera_name(camera_name: str) -> None:
    if not camera_name or not all(char.isalnum() or char == "_" for char in camera_name):
        raise ValueError(
            f"Camera name {camera_name!r} must be non-empty and contain only letters, digits, or underscores"
        )


def validate_args(args: argparse.Namespace) -> None:
    if len(args.extra_camera_names) != len(args.extra_camera_serials):
        raise ValueError("--extra-camera-names and --extra-camera-serials must have the same length")
    seen_camera_names = set(CAMERA_NAMES)
    for camera_name in args.extra_camera_names:
        validate_camera_name(camera_name)
        if camera_name in seen_camera_names:
            raise ValueError(f"Extra camera name {camera_name!r} is duplicated or already reserved")
        seen_camera_names.add(camera_name)
    for serial in args.extra_camera_serials:
        if not str(serial).strip():
            raise ValueError("--extra-camera-serials cannot contain empty values")
    if args.max_timesteps is not None and args.max_timesteps <= 0:
        raise ValueError("--max-timesteps must be positive when set")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.arm_sample_hz <= 0.0:
        raise ValueError("--arm-sample-hz must be positive")
    if args.queue_maxlen <= 0:
        raise ValueError("--queue-maxlen must be positive")
    if args.alignment_plot_frames <= 0:
        raise ValueError("--alignment-plot-frames must be positive")
    if args.window < 0:
        raise ValueError("--window display index must be non-negative")
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive HDF5 teleop collection requires a TTY for c/s/q controls")


def episode_start_payload(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    episode_idx: int,
    episode_path: Path,
    language_instruction: str,
    camera_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "dataset_root": str(dataset_root),
        "episode_idx": episode_idx,
        "episode_path": str(episode_path.with_suffix(".hdf5")),
        "episode_dir": str(episode_path.parent),
        "language_instruction": language_instruction,
        "max_timesteps": args.max_timesteps,
        "fps": args.fps,
        "arm_sample_hz": args.arm_sample_hz,
        "queue_maxlen": args.queue_maxlen,
        "alignment_plot_frames": args.alignment_plot_frames,
        "record": args.record,
        "record_dir": str(Path(args.record_dir).expanduser()),
        "action_from_state": args.action_from_state,
        "skip_idle": args.skip_idle,
        "use_depth_image": args.use_depth_image,
        "camera_names": list(camera_names),
        "running_sentinel": str(running_sentinel_path(dataset_root, episode_idx)),
    }


def stop_requested() -> bool:
    key_pressed = read_key()
    if key_pressed == "s":
        print("Stop requested for current episode.", flush=True)
        return True
    if key_pressed == "q":
        print("Recording is active; press s to save this episode, then q to quit from idle.", flush=True)
    return False


def collect_kwargs(
    args: argparse.Namespace,
    dataset_root: Path,
    episode_idx: int,
    runtime_window: RuntimeExecutionWindow | None,
) -> dict[str, Any]:
    return {
        "max_timesteps": args.max_timesteps,
        "fps": args.fps,
        "countdown_seconds": args.countdown_seconds,
        "ready_timeout_s": args.ready_timeout,
        "running_sentinel": running_sentinel_path(dataset_root, episode_idx),
        "stop_requested": stop_requested,
        "start_source": False,
        "skip_stationary": args.skip_idle,
        "runtime_window": runtime_window,
        "action_from_state": args.action_from_state,
    }


def next_episode_path(dataset_root: Path, episode_idx: int) -> Path:
    return episode_base_path(dataset_root, episode_idx)


def run_once(args: argparse.Namespace) -> None:
    validate_args(args)
    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    camera_names = camera_names_from_config(runtime_config)
    dataset_dir = Path(args.dataset_dir or runtime_config["dataset"]["dataset_dir"]).expanduser().resolve()
    task_name = str(args.task_name or runtime_config["dataset"]["dataset_name"])
    dataset_root = dataset_dir / task_name
    language_instruction = infer_language_instruction(task_name, args.language_instruction)
    next_manual_episode = args.episode_idx

    teleop_worker = make_teleop_worker(runtime_config, args)
    data_worker = make_data_worker(args, language_instruction, camera_names)
    window_schema = RecordingSchema(
        camera_names=camera_names,
        action_names=HDF5_TELEOP_VECTOR_NAMES,
        state_names=HDF5_TELEOP_VECTOR_NAMES,
        used_action_names=frozenset(HDF5_TELEOP_VECTOR_NAMES),
    )
    runtime_window = (
        RuntimeExecutionWindow(schema=window_schema, display_index=args.window)
        if args.window
        else None
    )
    last_window_refresh_s = 0.0
    terminal_settings = termios.tcgetattr(sys.stdin.fileno())

    def drain_data_worker(*, repeat_prompt: Callable[[], None] | None = None) -> None:
        completed_count = data_worker.drain(on_result=print_save_result, on_error=print_save_error)
        if repeat_prompt is not None and completed_count > 0:
            repeat_prompt()

    def refresh_idle_window() -> None:
        nonlocal last_window_refresh_s
        if runtime_window is None:
            return
        now_s = time.monotonic()
        if now_s - last_window_refresh_s < 0.1:
            return
        images = teleop_worker.source.latest_images()
        if images is not None:
            runtime_window.show_images(images)
        last_window_refresh_s = now_s

    def poll_idle_wait() -> None:
        drain_data_worker(repeat_prompt=print_idle_prompt)
        refresh_idle_window()

    def poll_episode_decision_wait() -> None:
        drain_data_worker(repeat_prompt=print_episode_decision_prompt)
        refresh_idle_window()

    try:
        teleop_worker.start()
        tty.setcbreak(sys.stdin.fileno())
        print("Interactive controls: idle c starts an episode, recording s stops it, idle q quits.", flush=True)
        while True:
            drain_data_worker()
            refresh_idle_window()
            print_idle_prompt()
            key = wait_for_key({"c", "q"}, poll=poll_idle_wait)
            if key == "q":
                break

            episode_idx = next_manual_episode if next_manual_episode is not None else next_episode_index(dataset_root)
            if next_manual_episode is not None:
                next_manual_episode += 1
            episode_path = next_episode_path(dataset_root, episode_idx)
            print_json(
                "hdf5_teleop_collection",
                episode_start_payload(
                    args=args,
                    dataset_root=dataset_root,
                    episode_idx=episode_idx,
                    episode_path=episode_path,
                    language_instruction=language_instruction,
                    camera_names=camera_names,
                ),
            )

            if runtime_window is not None:
                runtime_window.reset()
            episode = teleop_worker.collect_episode(
                episode_index=episode_idx,
                episode_path=episode_path,
                collect_fn=collect_hdf5_teleop_episode,
                collect_kwargs=collect_kwargs(args, dataset_root, episode_idx, runtime_window),
            )
            if runtime_window is not None:
                runtime_window.reset()
            if len(episode.frames) < 2:
                print(f"Discarded episode {episode_idx}: need at least 2 frames, got {len(episode.frames)}.", flush=True)
                continue
            refresh_idle_window()
            print_episode_decision_prompt()
            decision = wait_for_key({"c", "d"}, poll=poll_episode_decision_wait)
            if decision == "d":
                print(f"Discarded episode {episode_idx}: user requested delete.", flush=True)
                continue
            print_json("hdf5_teleop_collection_queued", data_worker.submit(episode))
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_settings)
        if runtime_window is not None:
            runtime_window.close()
        teleop_worker.stop()
        data_worker.stop(on_result=print_save_result, on_error=print_save_error)


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
