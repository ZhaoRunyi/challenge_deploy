from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.realsense import RealSenseRig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture one RGB snapshot from the configured RealSense rig.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--camera-front-serial", type=str, default=None)
    parser.add_argument("--camera-left-serial", type=str, default=None)
    parser.add_argument("--camera-right-serial", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "snapshots"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.camera_front_serial:
        set_by_dotted_path(config, "cameras.serials.cam_high", args.camera_front_serial)
    if args.camera_right_serial:
        set_by_dotted_path(config, "cameras.serials.cam_right_wrist", args.camera_right_serial)
    if args.camera_left_serial:
        set_by_dotted_path(config, "cameras.serials.cam_left_wrist", args.camera_left_serial)

    with RealSenseRig(
        config["cameras"]["serials"],
        width=int(config["cameras"]["width"]),
        height=int(config["cameras"]["height"]),
        fps=int(config["cameras"]["fps"]),
        warmup_frames=int(config["cameras"]["warmup_frames"]),
    ) as rig:
        saved_paths = rig.save_snapshot(args.output_dir)
    print(json.dumps(saved_paths, indent=2))


if __name__ == "__main__":
    main()
