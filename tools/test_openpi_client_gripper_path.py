from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from openpi.policies import slai_piper_policy

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.constants import KAI0_GRIPPER_UNIT_SCALE, LEGACY_PIPER_DATA_GRIPPER_UNIT_SCALE
from challenge_deploy.openpi_client import OpenPiPiperClient
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.realsense import RealSenseRig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use the same OpenPI client command path to fully open/close both grippers and save wrist-camera images."
    )
    parser.add_argument("--train-config", default="pi05_slai_piper_dock_tubes_H30_Ajointgripper_Sjointgripper_0423")
    parser.add_argument("--config", default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--camera-front-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "openpi_client_gripper_tests"))
    parser.add_argument("--open-value", type=float, default=0.105)
    parser.add_argument("--close-value", type=float, default=0.0)
    parser.add_argument("--joint-speed-percent", type=int, default=20)
    parser.add_argument("--command-repeat", type=int, default=10)
    parser.add_argument("--command-interval", type=float, default=0.08)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
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


def _save_stage(output_dir: Path, stage_name: str, images: dict[str, np.ndarray]) -> dict[str, str]:
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


def _hardware_opening_to_model_raw(value: float) -> float:
    return float(value) * (KAI0_GRIPPER_UNIT_SCALE / LEGACY_PIPER_DATA_GRIPPER_UNIT_SCALE)


def _make_action(client: OpenPiPiperClient, robot: DualPiperSystem, *, left_gripper: float, right_gripper: float) -> np.ndarray:
    state = robot.read_state()
    action = np.zeros(client.spec.action_dim, dtype=np.float64)
    action_space = slai_piper_policy._space_from_action_config(client.spec.action_space)
    slices = slai_piper_policy._field_slices_from_space(action_space)
    fields = set(slai_piper_policy._fields_from_action_config(client.spec.action_space))
    if "joint" not in fields:
        raise ValueError(f"{client.train_config_name} action_space does not expose joint control")

    action[slices["left_joint"]] = state.left.qpos[:6]
    action[slices["left_gripper"]] = np.array([_hardware_opening_to_model_raw(left_gripper)], dtype=np.float64)
    action[slices["right_joint"]] = state.right.qpos[:6]
    action[slices["right_gripper"]] = np.array([_hardware_opening_to_model_raw(right_gripper)], dtype=np.float64)
    return action


def _command_via_client(
    client: OpenPiPiperClient,
    robot: DualPiperSystem,
    *,
    left_gripper: float,
    right_gripper: float,
    command_repeat: int,
    command_interval: float,
    settle_seconds: float,
) -> None:
    action = _make_action(client, robot, left_gripper=left_gripper, right_gripper=right_gripper)
    for _ in range(command_repeat):
        client.command_action(robot, action)
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
        name="openpi_client_gripper_test",
    )
    cameras = RealSenseRig(
        config["cameras"]["serials"],
        width=int(config["cameras"]["width"]),
        height=int(config["cameras"]["height"]),
        fps=int(config["cameras"]["fps"]),
        warmup_frames=int(config["cameras"]["warmup_frames"]),
    )
    client = OpenPiPiperClient(
        args.train_config,
        control_mode="joints",
        joint_speed_percent=args.joint_speed_percent,
    )

    report: dict[str, object] = {
        "output_dir": str(output_dir),
        "train_config": args.train_config,
        "params": {
            "open_value": args.open_value,
            "close_value": args.close_value,
            "joint_speed_percent": args.joint_speed_percent,
            "command_repeat": args.command_repeat,
            "command_interval": args.command_interval,
            "settle_seconds": args.settle_seconds,
        },
    }

    robot.connect(read_only=False)
    try:
        cameras.start()
        report["enable_success"] = bool(robot.enable())
        time.sleep(1.0)

        before_images = cameras.capture()
        before_paths = _save_stage(output_dir, "before", before_images)
        before_state = _state_summary(robot)

        _command_via_client(
            client,
            robot,
            left_gripper=max(args.open_value, before_state["left"]["qpos"][6]),
            right_gripper=max(args.open_value, before_state["right"]["qpos"][6]),
            command_repeat=args.command_repeat,
            command_interval=args.command_interval,
            settle_seconds=args.settle_seconds,
        )
        open_images = cameras.capture()
        open_paths = _save_stage(output_dir, "after_full_open", open_images)
        open_state = _state_summary(robot)

        _command_via_client(
            client,
            robot,
            left_gripper=args.close_value,
            right_gripper=args.close_value,
            command_repeat=args.command_repeat,
            command_interval=args.command_interval,
            settle_seconds=args.settle_seconds,
        )
        close_images = cameras.capture()
        close_paths = _save_stage(output_dir, "after_full_close", close_images)
        close_state = _state_summary(robot)

        _command_via_client(
            client,
            robot,
            left_gripper=max(args.open_value, before_state["left"]["qpos"][6]),
            right_gripper=max(args.open_value, before_state["right"]["qpos"][6]),
            command_repeat=args.command_repeat,
            command_interval=args.command_interval,
            settle_seconds=args.settle_seconds,
        )
        reopen_images = cameras.capture()
        reopen_paths = _save_stage(output_dir, "after_reopen", reopen_images)
        reopen_state = _state_summary(robot)

        diff_metrics: dict[str, dict[str, float]] = {}
        diff_paths: dict[str, str] = {}
        for camera_name in ("cam_left_wrist", "cam_right_wrist"):
            if camera_name not in before_images:
                continue
            diff_metrics[f"{camera_name}_before_to_open"] = _diff_metric(before_images[camera_name], open_images[camera_name])
            diff_metrics[f"{camera_name}_open_to_close"] = _diff_metric(open_images[camera_name], close_images[camera_name])
            diff_metrics[f"{camera_name}_close_to_reopen"] = _diff_metric(close_images[camera_name], reopen_images[camera_name])
            diff_paths[f"{camera_name}_before_to_open"] = _save_diff(
                output_dir, f"{camera_name}_before_to_open", before_images[camera_name], open_images[camera_name]
            )
            diff_paths[f"{camera_name}_open_to_close"] = _save_diff(
                output_dir, f"{camera_name}_open_to_close", open_images[camera_name], close_images[camera_name]
            )
            diff_paths[f"{camera_name}_close_to_reopen"] = _save_diff(
                output_dir, f"{camera_name}_close_to_reopen", close_images[camera_name], reopen_images[camera_name]
            )

        report["stages"] = {
            "before": {"images": before_paths, "state": before_state},
            "after_full_open": {"images": open_paths, "state": open_state},
            "after_full_close": {"images": close_paths, "state": close_state},
            "after_reopen": {"images": reopen_paths, "state": reopen_state},
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
