from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.constants import CAMERA_NAMES
from challenge_deploy.dataset import EpisodeCollector
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.realsense import RealSenseRig
from challenge_deploy.runtime import DualPiperObservationSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROS-free raw collect_data equivalent: save Piper state + RealSense images as Challenge Deploy HDF5/video."
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="aloha_mobile_dummy")
    parser.add_argument("--episode_idx", type=int, default=None)
    parser.add_argument("--max_timesteps", type=int, default=500)
    parser.add_argument("--frame_rate", type=float, default=30.0)
    parser.add_argument("--left-can", type=str, default=None)
    parser.add_argument("--right-can", type=str, default=None)
    parser.add_argument("--camera-front-serial", type=str, default=None)
    parser.add_argument("--camera-left-serial", type=str, default=None)
    parser.add_argument("--camera-right-serial", type=str, default=None)
    parser.add_argument("--action-source", choices=["state", "zeros"], default="state")
    parser.add_argument("--no-video", action="store_true")
    return parser


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.dataset_dir:
        set_by_dotted_path(config, "dataset.dataset_dir", args.dataset_dir)
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
    return config


def main() -> None:
    args = build_parser().parse_args()
    if args.max_timesteps < 1:
        raise ValueError("--max_timesteps must be at least 1 because the first frame is the initial observation.")
    config = apply_cli_overrides(load_config(args.config), args)

    robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=False,
        name="raw_collect_data",
    )
    cameras = RealSenseRig(
        config["cameras"]["serials"],
        width=int(config["cameras"]["width"]),
        height=int(config["cameras"]["height"]),
        fps=int(config["cameras"]["fps"]),
        warmup_frames=int(config["cameras"]["warmup_frames"]),
    )
    source = DualPiperObservationSource(robot=robot, cameras=cameras)

    collector = EpisodeCollector(
        camera_names=list(CAMERA_NAMES),
        dataset_dir=config["dataset"]["dataset_dir"],
        dataset_name=args.task_name,
    )
    if args.episode_idx is not None:
        collector.episode_idx = args.episode_idx

    robot.connect(read_only=True)
    try:
        cameras.start()
        if not source.wait_until_ready(timeout_s=15.0):
            raise RuntimeError("Timed out waiting for robot/camera data")

        collector.start()
        print(f"Collecting raw episode {collector.episode_idx} into {collector.dataset_root}")
        print("Press Ctrl+C to stop early.")
        interval = 1.0 / float(args.frame_rate)

        for frame_idx in range(args.max_timesteps + 1):
            loop_start = time.time()
            snapshot = source.capture_snapshot()
            observation = snapshot.to_collector_observation()
            if args.action_source == "state":
                action = snapshot.state.qpos.copy()
            else:
                action = np.zeros(14, dtype=np.float64)
            collector.add_frame(observation, action)
            if frame_idx % 50 == 0:
                print(f"Collected frame {frame_idx}/{args.max_timesteps}")
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, interval - elapsed))
    except KeyboardInterrupt:
        print("Interrupted; saving collected frames.")
    finally:
        cameras.stop()
        robot.disconnect()

    if not collector.has_data():
        raise RuntimeError("No raw frames were collected; nothing saved.")

    saved_path = collector.save_current_episode(
        export_video=not args.no_video,
        video_fps=int(config["dataset"]["video_fps"]),
    )
    metadata_path = saved_path.with_suffix(".metadata.json")
    metadata = {
        "script": "deploy/dagger/collect_data.py",
        "dataset_path": str(saved_path),
        "task_name": args.task_name,
        "episode_idx": collector.episode_idx - 1,
        "max_timesteps": args.max_timesteps,
        "frame_rate": args.frame_rate,
        "action_source": args.action_source,
        "left_can": config["robot"]["left"]["can_name"],
        "right_can": config["robot"]["right"]["can_name"],
        "camera_serials": config["cameras"]["serials"],
        "camera_names": list(CAMERA_NAMES),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved raw HDF5: {saved_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
