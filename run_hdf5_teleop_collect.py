from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import Any

from hardware.config import load_config, set_by_dotted_path
from hardware.piper import DualPiperSystem
from hardware.realsense import RealSenseRig
from teleop.hdf5_teleop import (
    HDF5TeleopCollectionSource,
    collect_hdf5_teleop_episode,
    infer_language_instruction,
    next_episode_index,
    running_sentinel_path,
    save_alignment_diagnostics,
    save_hdf5_teleop_episode,
    save_hdf5_teleop_record_video,
)


DEPLOY_ROOT = Path(__file__).resolve().parent
DEFAULT_JPEG_QUALITY = 95


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
    parser.add_argument("--record-dir", default=str(DEPLOY_ROOT / "artifacts" / "hdf5_teleop_records"))
    parser.add_argument("--skip-idle", action=argparse.BooleanOptionalAction, default=True, help="Skip frames when master arms stay within the idle tolerance. Use --no-skip-idle to keep them.")
    parser.add_argument("--use-depth-image", action="store_true")
    parser.add_argument("--config", default=str(DEPLOY_ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--master-left-can", default=None)
    parser.add_argument("--master-right-can", default=None)
    parser.add_argument("--camera-front-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    return parser


def apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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
    if args.camera_left_serial:
        set_by_dotted_path(config, "cameras.serials.cam_left_wrist", args.camera_left_serial)
    if args.camera_right_serial:
        set_by_dotted_path(config, "cameras.serials.cam_right_wrist", args.camera_right_serial)
    return config


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
    return master_robot, puppet_robot, cameras, HDF5TeleopCollectionSource(
        master_robot=master_robot,
        puppet_robot=puppet_robot,
        cameras=cameras,
        arm_sample_hz=arm_sample_hz,
        queue_maxlen=queue_maxlen,
    )


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, writable, errors = select.select([sys.stdin], [], [], 0.0)
    del writable, errors
    if not ready:
        return None
    return sys.stdin.read(1).lower()


def wait_for_key(valid_keys: set[str]) -> str:
    while True:
        key = read_key()
        if key in valid_keys:
            return key
        time.sleep(0.05)


def validate_args(args: argparse.Namespace) -> None:
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
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive HDF5 teleop collection requires a TTY for c/s/q controls")


def run_once(args: argparse.Namespace) -> None:
    validate_args(args)
    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    dataset_dir = Path(args.dataset_dir or runtime_config["dataset"]["dataset_dir"]).expanduser().resolve()
    task_name = str(args.task_name or runtime_config["dataset"]["dataset_name"])
    dataset_root = dataset_dir / task_name
    language_instruction = infer_language_instruction(task_name, args.language_instruction)
    next_manual_episode = args.episode_idx

    master_robot, puppet_robot, cameras, source = make_collection_source(
        runtime_config,
        enable_depth=args.use_depth_image,
        arm_sample_hz=args.arm_sample_hz,
        queue_maxlen=args.queue_maxlen,
    )

    master_robot.connect(read_only=True)
    puppet_robot.connect(read_only=True)
    terminal_settings = termios.tcgetattr(sys.stdin.fileno())
    try:
        source.start()
        if not source.wait_until_ready(timeout_s=args.ready_timeout):
            detail = f": {source.last_error}" if source.last_error is not None else ""
            raise RuntimeError(f"Timed out waiting for master/puppet/camera async queues{detail}")
        tty.setcbreak(sys.stdin.fileno())
        print("Interactive controls: idle c starts an episode, recording s stops it, idle q quits.", flush=True)
        quit_requested = False
        while not quit_requested:
            print("Idle: press c to start an episode, or q to quit.", flush=True)
            key = wait_for_key({"c", "q"})
            if key == "q":
                break
            episode_idx = next_manual_episode if next_manual_episode is not None else next_episode_index(dataset_root)
            if next_manual_episode is not None:
                next_manual_episode += 1
            episode_path = dataset_root / f"episode_{episode_idx}"
            sentinel_path = running_sentinel_path(dataset_root, episode_idx)
            source.reset_trace()
            print(json.dumps({"hdf5_teleop_collection": {
                "dataset_root": str(dataset_root),
                "episode_idx": episode_idx,
                "episode_path": str(episode_path.with_suffix(".hdf5")),
                "language_instruction": language_instruction,
                "max_timesteps": args.max_timesteps,
                "fps": args.fps,
                "arm_sample_hz": args.arm_sample_hz,
                "queue_maxlen": args.queue_maxlen,
                "alignment_plot_frames": args.alignment_plot_frames,
                "record": args.record,
                "record_dir": str(Path(args.record_dir).expanduser()),
                "skip_idle": args.skip_idle,
                "use_depth_image": args.use_depth_image,
                "running_sentinel": str(sentinel_path),
            }}, indent=2), flush=True)

            def stop_requested() -> bool:
                key_pressed = read_key()
                if key_pressed == "s":
                    print("Stop requested for current episode.", flush=True)
                    return True
                if key_pressed == "q":
                    print("Recording is active; press s to save this episode, then q to quit from idle.", flush=True)
                return False

            frames = collect_hdf5_teleop_episode(
                source=source,
                max_timesteps=args.max_timesteps,
                fps=args.fps,
                countdown_seconds=args.countdown_seconds,
                ready_timeout_s=args.ready_timeout,
                running_sentinel=sentinel_path,
                stop_requested=stop_requested,
                start_source=False,
                skip_stationary=args.skip_idle,
            )
            if len(frames) < 2:
                print(f"Discarded episode {episode_idx}: need at least 2 frames, got {len(frames)}.", flush=True)
                continue
            output_path = save_hdf5_teleop_episode(
                output_path=episode_path,
                frames=frames,
                camera_names=("cam_high", "cam_left_wrist", "cam_right_wrist"),
                language_instruction=language_instruction,
                include_depth_images=args.use_depth_image,
                jpeg_quality=DEFAULT_JPEG_QUALITY,
            )
            alignment_json_path, alignment_image_path = save_alignment_diagnostics(
                output_path,
                frames,
                source.alignment_trace(),
                plot_frames=args.alignment_plot_frames,
            )
            record_video_path = None
            if args.record:
                print("Rendering record video from saved episode frames...", flush=True)
                record_video_path = save_hdf5_teleop_record_video(
                    frames=frames,
                    output_dir=Path(args.record_dir).expanduser(),
                    fps=args.fps,
                    name_prefix=f"hdf5_teleop_{task_name}_episode_{episode_idx}",
                )
                if record_video_path is not None:
                    print(f"Recording saved to {record_video_path}", flush=True)
            print(json.dumps({"hdf5_teleop_collection_result": {
                "saved_path": str(output_path),
                "alignment_json_path": str(alignment_json_path),
                "alignment_image_path": str(alignment_image_path),
                "record_video_path": None if record_video_path is None else str(record_video_path),
                "captured_frames": len(frames),
                "saved_steps": max(0, len(frames) - 1),
            }}, indent=2), flush=True)
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_settings)
        try:
            source.stop()
        except Exception as exc:
            print(f"Failed to stop async teleop source cleanly: {exc}", flush=True)
        try:
            cameras.stop()
        except Exception as exc:
            print(f"Failed to stop cameras cleanly: {exc}", flush=True)
        try:
            master_robot.disconnect()
        except Exception as exc:
            print(f"Failed to disconnect master robot cleanly: {exc}", flush=True)
        try:
            puppet_robot.disconnect()
        except Exception as exc:
            print(f"Failed to disconnect puppet robot cleanly: {exc}", flush=True)


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
