from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.realsense import RealSenseRig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan direct Piper gripper openings on hardware and save wrist-camera snapshots."
    )
    parser.add_argument(
        "--openings",
        nargs="+",
        type=float,
        default=[0.05, 0.052, 0.055, 0.058, 0.06],
        help="Physical gripper openings in meters to command sequentially.",
    )
    parser.add_argument("--speed-percent", type=int, default=20)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    parser.add_argument("--config", default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--camera-front-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "piper_gripper_scan"))
    return parser


def _apply_runtime_overrides(config: dict, args: argparse.Namespace) -> dict:
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


def _save_images(images: dict[str, np.ndarray], output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, image in images.items():
        path = output_dir / f"{stem}_{name}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to save image: {path}")
        saved[name] = str(path)
    return saved


def _capture_step(
    *,
    cameras: RealSenseRig,
    robot: DualPiperSystem,
    output_dir: Path,
    stem: str,
) -> dict:
    images = cameras.capture()
    image_paths = _save_images(images, output_dir, stem)
    state = robot.read_state()
    return {
        "image_paths": image_paths,
        "qpos": state.qpos.tolist(),
        "left_gripper": float(state.left.qpos[6]),
        "right_gripper": float(state.right.qpos[6]),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    runtime_config = _apply_runtime_overrides(load_config(args.config), args)

    robot = DualPiperSystem(
        left_can_name=runtime_config["robot"]["left"]["can_name"],
        right_can_name=runtime_config["robot"]["right"]["can_name"],
        commands_enabled=True,
        name="scan_piper_gripper_opening",
    )
    cameras = RealSenseRig(
        runtime_config["cameras"]["serials"],
        width=int(runtime_config["cameras"]["width"]),
        height=int(runtime_config["cameras"]["height"]),
        fps=int(runtime_config["cameras"]["fps"]),
        warmup_frames=int(runtime_config["cameras"]["warmup_frames"]),
    )

    robot.connect(read_only=False)
    try:
        cameras.start()
        if not robot.enable():
            raise RuntimeError("Failed to enable dual Piper")

        deadline = time.time() + args.ready_timeout
        while time.time() < deadline:
            try:
                cameras.capture()
                robot.read_state()
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("Timed out waiting for cameras/robot")

        baseline = robot.read_state().qpos.astype(np.float64)
        results: dict[str, object] = {
            "requested_openings_m": list(args.openings),
            "baseline_qpos": baseline.tolist(),
            "steps": {},
        }

        results["steps"]["before"] = _capture_step(
            cameras=cameras,
            robot=robot,
            output_dir=output_dir,
            stem="before",
        )

        for index, opening in enumerate(args.openings):
            target = baseline.copy()
            target[6] = float(opening)
            target[13] = float(opening)
            robot.set_joint_positions(target, speed_percent=args.speed_percent)
            time.sleep(args.settle_seconds)
            key = f"open_{index}_{opening:.4f}m"
            results["steps"][key] = _capture_step(
                cameras=cameras,
                robot=robot,
                output_dir=output_dir,
                stem=key,
            )

        robot.set_joint_positions(baseline, speed_percent=args.speed_percent)
        time.sleep(args.settle_seconds)
        results["steps"]["restored"] = _capture_step(
            cameras=cameras,
            robot=robot,
            output_dir=output_dir,
            stem="restored",
        )
        print(json.dumps(results, indent=2))
    finally:
        try:
            cameras.stop()
        finally:
            robot.disconnect()


if __name__ == "__main__":
    main()
