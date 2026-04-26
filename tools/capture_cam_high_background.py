from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.lerobot_assets import default_cam_high_background_path
from challenge_deploy.realsense import RealSenseRig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a clean cam_high background image for train-distribution fusion.")
    parser.add_argument("--config", default="/home/edemlab/challenge_ws/deploy/configs/dual_piper_example.yaml")
    parser.add_argument("--camera-front-serial", default=None)
    parser.add_argument("--output", default=None, help="Defaults to deploy/artifacts/backgrounds/cam_high_clean.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.camera_front_serial:
        set_by_dotted_path(config, "cameras.serials.cam_high", args.camera_front_serial)

    output_path = Path(args.output) if args.output else default_cam_high_background_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rig = RealSenseRig(
        {"cam_high": config["cameras"]["serials"]["cam_high"]},
        width=int(config["cameras"]["width"]),
        height=int(config["cameras"]["height"]),
        fps=int(config["cameras"]["fps"]),
        warmup_frames=int(config["cameras"]["warmup_frames"]),
    )
    try:
        rig.start()
        image = rig.capture()["cam_high"]
    finally:
        rig.stop()

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write background image: {output_path}")
    print(output_path)


if __name__ == "__main__":
    main()
