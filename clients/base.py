from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hardware.constants import PIPER_GRIPPER_FULL_OPEN_METERS
from hardware.conversions import (
    legacy_piper_raw_gripper_to_opening,
    normalized_gripper_to_opening,
    opening_to_legacy_piper_raw_gripper,
    opening_to_normalized_gripper,
)
from hardware.schemas import PiperArmState, RobotSnapshot
from . import slai_piper_policy


ControlMode = Literal["joints", "ee_pose"]


@dataclass(frozen=True)
class DecodedArmAction:
    joint: np.ndarray | None
    gripper: float
    ee_pose: np.ndarray | None
    binary_gripper: bool = False


@dataclass(frozen=True)
class DecodedPiperAction:
    arms: dict[str, DecodedArmAction]
    control_mode: ControlMode


def rpy_to_rotation(rpy: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation_format = slai_piper_policy.resolve_rotation_format(rotation_format)
    rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
    if rotation_format == "rpy":
        return rpy
    rot = Rotation.from_euler("xyz", rpy, degrees=False)
    if rotation_format == "quat":
        return rot.as_quat().astype(np.float64)
    return rot.as_matrix()[:, :2].reshape(-1).astype(np.float64)


def rotation_to_rpy(values: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation_format = slai_piper_policy.resolve_rotation_format(rotation_format)
    values = np.asarray(values, dtype=np.float64)
    if rotation_format == "rpy":
        return values.reshape(3)
    rotation_cls = Rotation
    if rotation_format == "quat":
        return rotation_cls.from_quat(values.reshape(4)).as_euler("xyz", degrees=False)
    columns = values.reshape(3, 2)
    x_axis = columns[:, 0]
    y_axis = columns[:, 1]
    if np.linalg.norm(x_axis) < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-9)
    y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
    if np.linalg.norm(y_axis) < 1e-9:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(x_axis, fallback))) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = fallback - np.dot(x_axis, fallback) * x_axis
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    matrix = np.stack((x_axis, y_axis, z_axis), axis=1)
    return rotation_cls.from_matrix(matrix).as_euler("xyz", degrees=False)


def stabilize_rpy(rpy: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is None:
        return rpy
    return rpy + 2 * np.pi * np.round((previous - rpy) / (2 * np.pi))


def hardware_gripper_to_model_raw(value: float, *, old_gripper: bool) -> float:
    if old_gripper:
        return opening_to_legacy_piper_raw_gripper(value)
    return opening_to_normalized_gripper(value)


def model_raw_gripper_to_hardware(value: float, *, old_gripper: bool) -> float:
    if old_gripper:
        return legacy_piper_raw_gripper_to_opening(value)
    return normalized_gripper_to_opening(value)


def state_gripper_for_policy(value: float, gripper_config: Any, *, old_gripper: bool) -> float:
    value = float(value)
    if gripper_config is not None and gripper_config.type == "01":
        return value / gripper_config.full_width if gripper_config.full_width > 0 else value
    return hardware_gripper_to_model_raw(value, old_gripper=old_gripper)


def action_gripper_for_piper(value: float, gripper_config: Any, *, old_gripper: bool) -> float:
    value = float(value)
    if gripper_config is not None and gripper_config.type == "01":
        return gripper_config.full_width if value >= 0.5 else 0.0
    return model_raw_gripper_to_hardware(value, old_gripper=old_gripper)


def bounded_gripper_for_piper(
    value: float,
    threshold: float | None,
    lower: float | None = None,
    upper: float | None = None,
    *,
    full_open_value: float = PIPER_GRIPPER_FULL_OPEN_METERS,
) -> float:
    value = max(0.0, float(value))
    if threshold is not None:
        return full_open_value if value >= threshold else 0.0
    if upper is not None and value > upper:
        return full_open_value
    return 0.0 if lower is not None and value < lower else value


def arm_full_state(arm: PiperArmState, *, ee_rotation: str, gripper_config: Any, old_gripper: bool) -> np.ndarray:
    return np.concatenate(
        (
            arm.qpos[:6],
            np.array([state_gripper_for_policy(arm.qpos[6], gripper_config, old_gripper=old_gripper)], dtype=np.float64),
            arm.end_pose[:3],
            rpy_to_rotation(arm.end_pose[3:6], ee_rotation),
        ),
        axis=0,
    ).astype(np.float64)


def build_full_piper_state(snapshot: RobotSnapshot, spec: Any, *, old_gripper: bool = False) -> np.ndarray:
    return np.concatenate(
        (
            arm_full_state(snapshot.state.left, ee_rotation=spec.state_space.ee_rotation, gripper_config=spec.state_space.gripper, old_gripper=old_gripper),
            arm_full_state(snapshot.state.right, ee_rotation=spec.state_space.ee_rotation, gripper_config=spec.state_space.gripper, old_gripper=old_gripper),
        ),
        axis=0,
    )


def build_configured_piper_state(snapshot: RobotSnapshot, spec: Any, *, old_gripper: bool = False, dtype: Any = np.float64) -> np.ndarray:
    full_state = build_full_piper_state(snapshot, spec, old_gripper=old_gripper)
    state_space = slai_piper_policy.space_from_state_config(spec.state_space)
    return np.asarray(slai_piper_policy.extract_vec(full_state, state_space, spec.state_space.gripper), dtype=dtype)


def image_to_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def action_array_from_response(response: dict[str, Any], keys: tuple[str, ...] = ("action", "actions")) -> np.ndarray:
    for key in keys:
        if key in response:
            return np.asarray(response[key], dtype=np.float64)
    raise KeyError(f"Policy response does not contain any action key {keys}: {sorted(response)}")


def used_action_names(spec: Any, control_mode: ControlMode) -> frozenset[str]:
    action_space = slai_piper_policy.space_from_action_config(spec.action_space)
    names = slai_piper_policy.get_vector_names(spec.action_space)
    slices = slai_piper_policy.field_slices_from_space(action_space)
    used_fields = {"gripper"}
    if control_mode == "joints":
        used_fields.add("joint")
    else:
        used_fields.update(("ee_pos", "ee_rot"))
    used: set[str] = set()
    for arm in action_space["arms"]:
        for field in used_fields:
            field_slice = slices.get(f"{arm}_{field}")
            if field_slice is None:
                continue
            used.update(names[index] for index in range(field_slice.start, field_slice.stop))
    return frozenset(used)


class SlaiPiperClient:
    def __init__(
        self,
        *,
        spec: Any,
        policy_client: Any,
        control_mode: ControlMode = "joints",
        joint_speed_percent: int = 50,
        ee_speed_percent: int = 50,
        gripper_threshold: float | None = None,
        gripper_lower: float | None = None,
        gripper_upper: float | None = None,
        old_gripper: bool = False,
    ) -> None:
        self.spec = spec
        self.client = policy_client
        self.control_mode = control_mode
        self.joint_speed_percent = joint_speed_percent
        self.ee_speed_percent = ee_speed_percent
        self.gripper_threshold = gripper_threshold
        self.gripper_lower = gripper_lower
        self.gripper_upper = gripper_upper
        self.old_gripper = old_gripper
        self.previous_ee_rpy: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.validate_control_mode()

    @property
    def train_config_name(self) -> str:
        return str(getattr(self.spec, "train_config_name", getattr(self.spec, "config_path", "")))

    def spec_label(self) -> str:
        return self.train_config_name or type(self.spec).__name__

    def validate_control_mode(self) -> None:
        fields = set(slai_piper_policy.fields_from_action_config(self.spec.action_space))
        if "gripper" not in fields:
            raise ValueError(f"{self.spec_label()}: deploy requires action_space to include gripper")
        if self.control_mode == "joints":
            if "joint" not in fields:
                raise ValueError(f"{self.spec_label()}: control_mode='joints' requires action_space to include joint")
            return
        if self.control_mode == "ee_pose":
            missing = {"ee_pos", "ee_rot"} - fields
            if missing:
                raise ValueError(f"{self.spec_label()}: control_mode='ee_pose' requires action_space fields {sorted(missing)}")
            return
        raise ValueError(f"Unsupported control_mode: {self.control_mode}")

    def get_server_metadata(self) -> Any:
        return self.client.get_server_metadata()

    def build_payload(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def infer(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        return self.client.infer(self.build_payload(snapshot, prompt=prompt, **kwargs))

    def infer_actions(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> np.ndarray:
        return action_array_from_response(self.infer(snapshot, prompt=prompt, **kwargs))

    def decode_gripper_for_piper(self, value: float, arm_name: str) -> tuple[float, bool]:
        arm_threshold = getattr(self, f"{arm_name}_gripper_threshold", None)
        arm_lower = getattr(self, f"{arm_name}_gripper_lower", None)
        arm_upper = getattr(self, f"{arm_name}_gripper_upper", None)
        threshold = arm_threshold if arm_threshold is not None else self.gripper_threshold
        lower = arm_lower if arm_lower is not None else self.gripper_lower
        upper = arm_upper if arm_upper is not None else self.gripper_upper
        gripper = bounded_gripper_for_piper(
            action_gripper_for_piper(value, self.spec.action_space.gripper, old_gripper=self.old_gripper),
            threshold,
            lower,
            upper,
        )
        return gripper, False

    def decode_action(self, action: np.ndarray) -> DecodedPiperAction:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1:
            raise ValueError(f"Expected one action vector, got shape {action.shape}")
        if action.shape[0] < self.spec.action_dim:
            raise ValueError(f"Action dim {action.shape[0]} is smaller than expected {self.spec.action_dim} for {self.spec_label()}")
        action_space = slai_piper_policy.space_from_action_config(self.spec.action_space)
        slices = slai_piper_policy.field_slices_from_space(action_space)
        fields = set(slai_piper_policy.fields_from_action_config(self.spec.action_space))
        decoded: dict[str, DecodedArmAction] = {}
        for arm in action_space["arms"]:
            gripper, binary_gripper = self.decode_gripper_for_piper(float(action[slices[f"{arm}_gripper"]][0]), arm)
            joint = None
            ee_pose = None
            if "joint" in fields:
                joint = np.concatenate((action[slices[f"{arm}_joint"]], np.array([gripper])), axis=0)
            if {"ee_pos", "ee_rot"}.issubset(fields):
                ee_rpy = rotation_to_rpy(action[slices[f"{arm}_ee_rot"]], self.spec.action_space.ee_rotation)
                ee_pose = np.concatenate((action[slices[f"{arm}_ee_pos"]], ee_rpy, np.array([gripper])), axis=0)
            decoded[arm] = DecodedArmAction(joint=joint, gripper=gripper, ee_pose=ee_pose, binary_gripper=binary_gripper)
        return DecodedPiperAction(arms=decoded, control_mode=self.control_mode)

    def command_decoded(self, robot: Any, decoded: DecodedPiperAction) -> None:
        for arm_name, arm_action in decoded.arms.items():
            arm = robot.left if arm_name == "left" else robot.right
            if decoded.control_mode == "joints":
                if arm_action.joint is None:
                    raise ValueError(f"Decoded action for {arm_name} has no joint block")
                arm.command_joint_positions(arm_action.joint, speed_percent=self.joint_speed_percent)
            else:
                if arm_action.ee_pose is None:
                    raise ValueError(f"Decoded action for {arm_name} has no ee_pose block")
                pose = arm_action.ee_pose.copy()
                pose[3:6] = stabilize_rpy(pose[3:6], self.previous_ee_rpy[arm_name])
                self.previous_ee_rpy[arm_name] = pose[3:6].copy()
                arm.command_end_pose(pose, speed_percent=self.ee_speed_percent)

    def command_action(self, robot: Any, action: np.ndarray) -> None:
        self.command_decoded(robot, self.decode_action(action))

    def command_first_action(self, robot: Any, response_or_actions: dict[str, Any] | np.ndarray) -> None:
        actions = action_array_from_response(response_or_actions) if isinstance(response_or_actions, dict) else response_or_actions
        actions = np.asarray(actions, dtype=np.float64)
        self.command_action(robot, actions[0] if actions.ndim == 2 else actions)
