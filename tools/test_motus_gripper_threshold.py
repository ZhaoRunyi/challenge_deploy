from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.motus_client import MotusPiperClient, load_motus_policy_spec
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.realsense import RealSenseRig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise Motus gripper-threshold decode on hardware and save wrist-camera snapshots."
    )
    parser.add_argument("--train-config", required=True, help="Motus YAML config path.")
    parser.add_argument("--threshold", type=float, required=True, help="Executable gripper threshold in meters.")
    parser.add_argument("--below-opening", type=float, default=0.01, help="Opening to encode below threshold.")
    parser.add_argument("--above-opening", type=float, default=0.03, help="Opening to encode above threshold.")
    parser.add_argument("--speed-percent", type=int, default=30)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    parser.add_argument("--config", default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--camera-front-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "motus_gripper_threshold_test"))
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


def _make_decode_only_client(train_config: str, threshold: float, speed_percent: int) -> MotusPiperClient:
    client = object.__new__(MotusPiperClient)
    client.spec = load_motus_policy_spec(train_config)
    client.control_mode = "joints"
    client.joint_speed_percent = speed_percent
    client.ee_speed_percent = speed_percent
    client.gripper_threshold = threshold
    client._default_session_id = None
    return client


def _hardware_opening_to_action_value(opening_m: float, gripper_cfg: object | None) -> float:
    opening = max(0.0, float(opening_m))
    if gripper_cfg is not None and getattr(gripper_cfg, "type", None) == "01":
        full_width = float(getattr(gripper_cfg, "full_width", 0.05))
        return 1.0 if opening >= (full_width * 0.5) else 0.0
    return opening * (1_000_000.0 / 70_000.0)


def _current_joint_action_vector(robot: DualPiperSystem, client: MotusPiperClient, opening_m: float) -> np.ndarray:
    qpos = robot.read_state().qpos.astype(np.float64)
    action = qpos.copy()
    action[6] = _hardware_opening_to_action_value(opening_m, client.spec.action_space.gripper)
    action[13] = _hardware_opening_to_action_value(opening_m, client.spec.action_space.gripper)
    return action


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
    paths = _save_images(images, output_dir, stem)
    state = robot.read_state()
    return {
        "image_paths": paths,
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
        name="test_motus_gripper_threshold",
    )
    cameras = RealSenseRig(
        runtime_config["cameras"]["serials"],
        width=int(runtime_config["cameras"]["width"]),
        height=int(runtime_config["cameras"]["height"]),
        fps=int(runtime_config["cameras"]["fps"]),
        warmup_frames=int(runtime_config["cameras"]["warmup_frames"]),
    )
    client = _make_decode_only_client(args.train_config, args.threshold, args.speed_percent)

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

        baseline_state = robot.read_state()
        below_action = _current_joint_action_vector(robot, client, args.below_opening)
        above_action = _current_joint_action_vector(robot, client, args.above_opening)
        below_decoded = client.decode_action(below_action)
        above_decoded = client.decode_action(above_action)

        result = {
            "threshold_m": args.threshold,
            "below_opening_input_m": args.below_opening,
            "above_opening_input_m": args.above_opening,
            "baseline_qpos": baseline_state.qpos.tolist(),
            "below_decoded_gripper": {
                "left": float(below_decoded.arms["left"].gripper),
                "right": float(below_decoded.arms["right"].gripper),
            },
            "above_decoded_gripper": {
                "left": float(above_decoded.arms["left"].gripper),
                "right": float(above_decoded.arms["right"].gripper),
            },
            "steps": {},
        }

        result["steps"]["before"] = _capture_step(cameras=cameras, robot=robot, output_dir=output_dir, stem="before")

        client.command_action(robot, below_action)
        time.sleep(args.settle_seconds)
        result["steps"]["below"] = _capture_step(cameras=cameras, robot=robot, output_dir=output_dir, stem="below")

        client.command_action(robot, above_action)
        time.sleep(args.settle_seconds)
        result["steps"]["above"] = _capture_step(cameras=cameras, robot=robot, output_dir=output_dir, stem="above")

        robot.set_joint_positions(baseline_state.qpos, speed_percent=args.speed_percent)
        time.sleep(args.settle_seconds)
        result["steps"]["restored"] = _capture_step(cameras=cameras, robot=robot, output_dir=output_dir, stem="restored")

        print(json.dumps(result, indent=2))
    finally:
        try:
            cameras.stop()
        finally:
            robot.disconnect()


if __name__ == "__main__":
    main()
