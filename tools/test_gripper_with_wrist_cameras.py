from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.realsense import RealSenseRig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command a small dual-gripper open/close test and save wrist-camera before/after frames."
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", type=str, default=None)
    parser.add_argument("--right-can", type=str, default=None)
    parser.add_argument("--camera-left-serial", type=str, default=None)
    parser.add_argument("--camera-right-serial", type=str, default=None)
    parser.add_argument("--camera-front-serial", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "gripper_camera_tests"),
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.03,
        help="Opening delta to test on each gripper, in deploy qpos units.",
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=20,
        help="Joint-mode speed percent used while holding current joints and moving only the gripper.",
    )
    parser.add_argument(
        "--command-repeat",
        type=int,
        default=8,
        help="How many times to resend the same target to avoid missing a single CAN command.",
    )
    parser.add_argument(
        "--command-interval",
        type=float,
        default=0.08,
        help="Seconds between repeated commands.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Seconds to wait after each test command before grabbing the next image.",
    )
    return parser


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.left_can:
        set_by_dotted_path(config, "robot.left.can_name", args.left_can)
    if args.right_can:
        set_by_dotted_path(config, "robot.right.can_name", args.right_can)
    if args.camera_front_serial:
        set_by_dotted_path(config, "cameras.serials.cam_high", args.camera_front_serial)
    if args.camera_left_serial:
        set_by_dotted_path(config, "cameras.serials.cam_left_wrist", args.camera_left_serial)
    if args.camera_right_serial:
        set_by_dotted_path(config, "cameras.serials.cam_right_wrist", args.camera_right_serial)
    return config


def _clamp_gripper(value: float) -> float:
    return float(np.clip(value, 0.0, 0.11))


def _save_stage(
    output_dir: Path,
    stage_name: str,
    *,
    images: dict[str, np.ndarray],
) -> dict[str, str]:
    saved: dict[str, str] = {}
    for camera_name, image in images.items():
        path = output_dir / f"{stage_name}_{camera_name}.png"
        cv2.imwrite(str(path), image)
        saved[camera_name] = str(path)
    return saved


def _roi(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return image[int(height * 0.35) : int(height * 0.98), int(width * 0.10) : int(width * 0.90)]


def _diff_metric(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    full = cv2.absdiff(before, after)
    roi = cv2.absdiff(_roi(before), _roi(after))
    return {
        "full_mean_absdiff": float(np.mean(full)),
        "roi_mean_absdiff": float(np.mean(roi)),
        "roi_max_absdiff": float(np.max(roi)),
    }


def _save_diff(output_dir: Path, label: str, before: np.ndarray, after: np.ndarray) -> str:
    diff = cv2.absdiff(before, after)
    path = output_dir / f"{label}_diff.png"
    cv2.imwrite(str(path), diff)
    return str(path)


def _state_summary(robot: DualPiperSystem) -> dict:
    state = robot.read_state()
    return {
        "left": {
            "qpos": state.left.qpos.tolist(),
            "qpos_feedback": state.left.qpos_feedback.tolist(),
            "qpos_command": state.left.qpos_command.tolist(),
            "enabled": state.left.enabled,
            "feedback_hz": state.left.feedback_hz,
            "command_hz": state.left.command_hz,
        },
        "right": {
            "qpos": state.right.qpos.tolist(),
            "qpos_feedback": state.right.qpos_feedback.tolist(),
            "qpos_command": state.right.qpos_command.tolist(),
            "enabled": state.right.enabled,
            "feedback_hz": state.right.feedback_hz,
            "command_hz": state.right.command_hz,
        },
    }


def _command_target(
    robot: DualPiperSystem,
    *,
    left_target: np.ndarray,
    right_target: np.ndarray,
    speed_percent: int,
    command_repeat: int,
    command_interval: float,
    settle_seconds: float,
) -> None:
    target = np.concatenate((left_target, right_target), axis=0)
    for _ in range(command_repeat):
        robot.set_joint_positions(target, speed_percent=speed_percent)
        time.sleep(command_interval)
    time.sleep(settle_seconds)


def main() -> None:
    args = build_parser().parse_args()
    config = _apply_overrides(load_config(args.config), args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=True,
        name="gripper_camera_test",
    )
    cameras = RealSenseRig(
        config["cameras"]["serials"],
        width=int(config["cameras"]["width"]),
        height=int(config["cameras"]["height"]),
        fps=int(config["cameras"]["fps"]),
        warmup_frames=int(config["cameras"]["warmup_frames"]),
    )

    report: dict[str, object] = {
        "output_dir": str(output_dir),
        "params": {
            "delta": args.delta,
            "speed_percent": args.speed_percent,
            "command_repeat": args.command_repeat,
            "command_interval": args.command_interval,
            "settle_seconds": args.settle_seconds,
        },
    }

    robot.connect(read_only=False)
    try:
        cameras.start()
        enabled = robot.enable()
        time.sleep(1.0)
        report["enable_success"] = bool(enabled)

        before_state = robot.read_state()
        before_images = cameras.capture()
        before_paths = _save_stage(output_dir, "before", images=before_images)

        left_current = before_state.left.qpos.copy()
        right_current = before_state.right.qpos.copy()

        if float(left_current[6] + right_current[6]) >= args.delta:
            stage1_name = "after_close"
            stage2_name = "after_reopen"
            left_stage1 = left_current.copy()
            right_stage1 = right_current.copy()
            left_stage1[6] = _clamp_gripper(float(left_current[6] - args.delta))
            right_stage1[6] = _clamp_gripper(float(right_current[6] - args.delta))
            left_stage2 = left_current.copy()
            right_stage2 = right_current.copy()
        else:
            stage1_name = "after_open"
            stage2_name = "after_reclose"
            left_stage1 = left_current.copy()
            right_stage1 = right_current.copy()
            left_stage1[6] = _clamp_gripper(float(left_current[6] + args.delta))
            right_stage1[6] = _clamp_gripper(float(right_current[6] + args.delta))
            left_stage2 = left_current.copy()
            right_stage2 = right_current.copy()

        _command_target(
            robot,
            left_target=left_stage1,
            right_target=right_stage1,
            speed_percent=args.speed_percent,
            command_repeat=args.command_repeat,
            command_interval=args.command_interval,
            settle_seconds=args.settle_seconds,
        )
        stage1_state = robot.read_state()
        stage1_images = cameras.capture()
        stage1_paths = _save_stage(output_dir, stage1_name, images=stage1_images)

        _command_target(
            robot,
            left_target=left_stage2,
            right_target=right_stage2,
            speed_percent=args.speed_percent,
            command_repeat=args.command_repeat,
            command_interval=args.command_interval,
            settle_seconds=args.settle_seconds,
        )
        stage2_state = robot.read_state()
        stage2_images = cameras.capture()
        stage2_paths = _save_stage(output_dir, stage2_name, images=stage2_images)

        diff_metrics: dict[str, dict[str, float]] = {}
        diff_paths: dict[str, str] = {}
        for camera_name in ("cam_left_wrist", "cam_right_wrist"):
            if camera_name not in before_images or camera_name not in stage1_images or camera_name not in stage2_images:
                continue
            diff_metrics[f"{camera_name}_before_to_stage1"] = _diff_metric(before_images[camera_name], stage1_images[camera_name])
            diff_metrics[f"{camera_name}_stage1_to_stage2"] = _diff_metric(stage1_images[camera_name], stage2_images[camera_name])
            diff_metrics[f"{camera_name}_before_to_stage2"] = _diff_metric(before_images[camera_name], stage2_images[camera_name])
            diff_paths[f"{camera_name}_before_to_stage1"] = _save_diff(
                output_dir,
                f"{camera_name}_before_to_stage1",
                before_images[camera_name],
                stage1_images[camera_name],
            )
            diff_paths[f"{camera_name}_stage1_to_stage2"] = _save_diff(
                output_dir,
                f"{camera_name}_stage1_to_stage2",
                stage1_images[camera_name],
                stage2_images[camera_name],
            )
            diff_paths[f"{camera_name}_before_to_stage2"] = _save_diff(
                output_dir,
                f"{camera_name}_before_to_stage2",
                before_images[camera_name],
                stage2_images[camera_name],
            )

        report["stages"] = {
            "before": {
                "images": before_paths,
                "state": _state_summary(robot) if False else {
                    "left": {
                        "qpos": before_state.left.qpos.tolist(),
                        "qpos_feedback": before_state.left.qpos_feedback.tolist(),
                        "qpos_command": before_state.left.qpos_command.tolist(),
                        "enabled": before_state.left.enabled,
                        "feedback_hz": before_state.left.feedback_hz,
                        "command_hz": before_state.left.command_hz,
                    },
                    "right": {
                        "qpos": before_state.right.qpos.tolist(),
                        "qpos_feedback": before_state.right.qpos_feedback.tolist(),
                        "qpos_command": before_state.right.qpos_command.tolist(),
                        "enabled": before_state.right.enabled,
                        "feedback_hz": before_state.right.feedback_hz,
                        "command_hz": before_state.right.command_hz,
                    },
                },
            },
            stage1_name: {
                "images": stage1_paths,
                "state": {
                    "left": {
                        "qpos": stage1_state.left.qpos.tolist(),
                        "qpos_feedback": stage1_state.left.qpos_feedback.tolist(),
                        "qpos_command": stage1_state.left.qpos_command.tolist(),
                        "enabled": stage1_state.left.enabled,
                        "feedback_hz": stage1_state.left.feedback_hz,
                        "command_hz": stage1_state.left.command_hz,
                    },
                    "right": {
                        "qpos": stage1_state.right.qpos.tolist(),
                        "qpos_feedback": stage1_state.right.qpos_feedback.tolist(),
                        "qpos_command": stage1_state.right.qpos_command.tolist(),
                        "enabled": stage1_state.right.enabled,
                        "feedback_hz": stage1_state.right.feedback_hz,
                        "command_hz": stage1_state.right.command_hz,
                    },
                },
            },
            stage2_name: {
                "images": stage2_paths,
                "state": {
                    "left": {
                        "qpos": stage2_state.left.qpos.tolist(),
                        "qpos_feedback": stage2_state.left.qpos_feedback.tolist(),
                        "qpos_command": stage2_state.left.qpos_command.tolist(),
                        "enabled": stage2_state.left.enabled,
                        "feedback_hz": stage2_state.left.feedback_hz,
                        "command_hz": stage2_state.left.command_hz,
                    },
                    "right": {
                        "qpos": stage2_state.right.qpos.tolist(),
                        "qpos_feedback": stage2_state.right.qpos_feedback.tolist(),
                        "qpos_command": stage2_state.right.qpos_command.tolist(),
                        "enabled": stage2_state.right.enabled,
                        "feedback_hz": stage2_state.right.feedback_hz,
                        "command_hz": stage2_state.right.command_hz,
                    },
                },
            },
        }
        report["diff_metrics"] = diff_metrics
        report["diff_images"] = diff_paths

        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        try:
            cameras.stop()
        finally:
            robot.disconnect()


if __name__ == "__main__":
    main()
