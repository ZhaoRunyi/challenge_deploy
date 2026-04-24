from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from openpi.policies import slai_piper_policy
from openpi.training import config as openpi_config
from openpi_client import websocket_client_policy

from .schemas import PiperArmState, RobotSnapshot

try:
    from scipy.spatial.transform import Rotation
except ImportError:  # pragma: no cover - scipy exists in the intended OpenPI venv.
    Rotation = None


ControlMode = Literal["joints", "ee_pose"]


@dataclass(frozen=True)
class PiperPolicySpec:
    train_config_name: str
    train_config: Any
    state_space: Any
    action_space: Any
    image_space: Any
    state_dim: int
    action_dim: int
    model_action_dim: int | None
    action_horizon: int | None
    image_ids: tuple[str, ...]
    image_key_map: dict[str, str]


@dataclass(frozen=True)
class DecodedArmAction:
    joint: np.ndarray | None
    gripper: float
    ee_pose: np.ndarray | None


@dataclass(frozen=True)
class DecodedPiperAction:
    arms: dict[str, DecodedArmAction]
    control_mode: ControlMode


def load_piper_policy_spec(train_config_name: str) -> PiperPolicySpec:
    train_config = openpi_config.get_config(train_config_name)
    data_config = train_config.data

    missing = [
        name
        for name in ("state_space", "action_space", "image_space")
        if not hasattr(data_config, name)
    ]
    if missing:
        raise TypeError(
            f"OpenPI train config {train_config_name!r} is not a SLAI Piper config; "
            f"missing data fields: {missing}"
        )

    return PiperPolicySpec(
        train_config_name=train_config_name,
        train_config=train_config,
        state_space=data_config.state_space,
        action_space=data_config.action_space,
        image_space=data_config.image_space,
        state_dim=int(slai_piper_policy.get_space_dim(data_config.state_space)),
        action_dim=int(slai_piper_policy.get_space_dim(data_config.action_space)),
        model_action_dim=getattr(train_config.model, "action_dim", None),
        action_horizon=getattr(train_config.model, "action_horizon", None),
        image_ids=tuple(slai_piper_policy.get_image_ids(data_config.image_space)),
        image_key_map=slai_piper_policy.get_image_key_map(data_config.image_space),
    )


def _require_rotation() -> Any:
    if Rotation is None:
        raise RuntimeError("scipy is required for ee rotation conversion")
    return Rotation


def _resolve_rotation(rotation: str) -> str:
    return slai_piper_policy._resolve_rotation_format(rotation)


def rpy_to_rotation(rpy: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation_format = _resolve_rotation(rotation_format)
    rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
    if rotation_format == "rpy":
        return rpy
    rot = _require_rotation().from_euler("xyz", rpy, degrees=False)
    if rotation_format == "quat":
        return rot.as_quat().astype(np.float64)
    matrix = rot.as_matrix()
    return matrix[:, :2].reshape(-1).astype(np.float64)


def rotation_to_rpy(values: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation_format = _resolve_rotation(rotation_format)
    values = np.asarray(values, dtype=np.float64)
    if rotation_format == "rpy":
        return values.reshape(3)
    rotation_cls = _require_rotation()
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


def _state_gripper_for_openpi(value: float, gripper_cfg: Any) -> float:
    value = float(value)
    if gripper_cfg is not None and gripper_cfg.type == "01":
        return value / gripper_cfg.full_width if gripper_cfg.full_width > 0 else value
    return value


def _action_gripper_for_piper(value: float, gripper_cfg: Any) -> float:
    value = float(value)
    if gripper_cfg is not None and gripper_cfg.type == "01":
        return gripper_cfg.full_width if value >= 0.5 else 0.0
    return value


def _arm_full_state(arm: PiperArmState, *, ee_rotation: str, gripper_cfg: Any) -> np.ndarray:
    return np.concatenate(
        (
            arm.qpos[:6],
            np.array([_state_gripper_for_openpi(arm.qpos[6], gripper_cfg)], dtype=np.float64),
            arm.end_pose[:3],
            rpy_to_rotation(arm.end_pose[3:6], ee_rotation),
        ),
        axis=0,
    ).astype(np.float64)


def build_full_piper_state(snapshot: RobotSnapshot, spec: PiperPolicySpec) -> np.ndarray:
    """Build the full all-fields/all-arms vector expected before SLAIPiperInputs.

    SLAIPiperInputs owns the final configured state extraction. The client sends
    the same full Piper vector layout that the OpenPI transform indexes from.
    """

    return np.concatenate(
        (
            _arm_full_state(
                snapshot.state.left,
                ee_rotation=spec.state_space.ee_rotation,
                gripper_cfg=spec.state_space.gripper,
            ),
            _arm_full_state(
                snapshot.state.right,
                ee_rotation=spec.state_space.ee_rotation,
                gripper_cfg=spec.state_space.gripper,
            ),
        ),
        axis=0,
    )


def build_configured_piper_state(snapshot: RobotSnapshot, spec: PiperPolicySpec) -> np.ndarray:
    full_state = build_full_piper_state(snapshot, spec)
    state_space = slai_piper_policy._space_from_state_config(spec.state_space)
    return np.asarray(
        slai_piper_policy._extract_vec(full_state, state_space, spec.state_space.gripper),
        dtype=np.float64,
    )


def _image_to_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_policy_payload(
    snapshot: RobotSnapshot,
    *,
    prompt: str,
    spec: PiperPolicySpec,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "observation.state": build_full_piper_state(snapshot, spec),
        "prompt": prompt,
    }
    for image_id, dataset_key in spec.image_key_map.items():
        if image_id not in snapshot.images:
            raise KeyError(f"Snapshot is missing required image {image_id}")
        payload[dataset_key] = _image_to_rgb(snapshot.images[image_id])
    return payload


def _action_array_from_response(response: dict[str, Any]) -> np.ndarray:
    if "action" in response:
        return np.asarray(response["action"], dtype=np.float64)
    if "actions" in response:
        return np.asarray(response["actions"], dtype=np.float64)
    raise KeyError(f"Policy response does not contain 'action' or 'actions': {sorted(response)}")


class OpenPiPiperClient:
    """OpenPI websocket client for SLAI Piper train configs.

    The OpenPI server can run the unmodified `scripts/serve_policy.py`.
    This deploy-side client receives the same train config name, imports
    OpenPI's own SLAI Piper space definitions, builds the pre-transform payload,
    and decodes the post-transform action into Piper control calls.
    """

    def __init__(
        self,
        train_config_name: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        control_mode: ControlMode = "joints",
        api_key: str | None = None,
        joint_speed_percent: int = 50,
        ee_speed_percent: int = 50,
    ) -> None:
        self.spec = load_piper_policy_spec(train_config_name)
        self.control_mode = control_mode
        self.joint_speed_percent = joint_speed_percent
        self.ee_speed_percent = ee_speed_percent
        self._validate_control_mode()
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port, api_key=api_key)

    def _validate_control_mode(self) -> None:
        fields = set(slai_piper_policy._fields_from_action_config(self.spec.action_space))
        if "gripper" not in fields:
            raise ValueError(f"{self.spec.train_config_name}: deploy requires action_space to include gripper")
        if self.control_mode == "joints":
            if "joint" not in fields:
                raise ValueError(
                    f"{self.spec.train_config_name}: control_mode='joints' requires action_space to include joint"
                )
            return
        if self.control_mode == "ee_pose":
            missing = {"ee_pos", "ee_rot"} - fields
            if missing:
                raise ValueError(
                    f"{self.spec.train_config_name}: control_mode='ee_pose' requires action_space fields "
                    f"{sorted(missing)}"
                )
            return
        raise ValueError(f"Unsupported control_mode: {self.control_mode}")

    @property
    def train_config_name(self) -> str:
        return self.spec.train_config_name

    def get_server_metadata(self) -> Any:
        return self._client.get_server_metadata()

    def build_payload(self, snapshot: RobotSnapshot, prompt: str) -> dict[str, Any]:
        return build_policy_payload(snapshot, prompt=prompt, spec=self.spec)

    def infer(self, snapshot: RobotSnapshot, prompt: str) -> dict[str, Any]:
        return self._client.infer(self.build_payload(snapshot, prompt))

    def infer_actions(self, snapshot: RobotSnapshot, prompt: str) -> np.ndarray:
        return _action_array_from_response(self.infer(snapshot, prompt))

    def decode_action(self, action: np.ndarray) -> DecodedPiperAction:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1:
            raise ValueError(f"Expected one action vector, got shape {action.shape}")
        if action.shape[0] < self.spec.action_dim:
            raise ValueError(
                f"Action dim {action.shape[0]} is smaller than expected {self.spec.action_dim} "
                f"for {self.spec.train_config_name}"
            )

        action_space = slai_piper_policy._space_from_action_config(self.spec.action_space)
        slices = slai_piper_policy._field_slices_from_space(action_space)
        fields = set(slai_piper_policy._fields_from_action_config(self.spec.action_space))
        decoded: dict[str, DecodedArmAction] = {}
        for arm in action_space["arms"]:
            gripper = _action_gripper_for_piper(
                float(action[slices[f"{arm}_gripper"]][0]),
                self.spec.action_space.gripper,
            )
            joint = None
            ee_pose = None
            if "joint" in fields:
                joint = np.concatenate((action[slices[f"{arm}_joint"]], np.array([gripper])), axis=0)
            if {"ee_pos", "ee_rot"}.issubset(fields):
                ee_rpy = rotation_to_rpy(action[slices[f"{arm}_ee_rot"]], self.spec.action_space.ee_rotation)
                ee_pose = np.concatenate((action[slices[f"{arm}_ee_pos"]], ee_rpy, np.array([gripper])), axis=0)
            decoded[arm] = DecodedArmAction(joint=joint, gripper=gripper, ee_pose=ee_pose)
        return DecodedPiperAction(arms=decoded, control_mode=self.control_mode)

    def command_action(self, robot: Any, action: np.ndarray) -> None:
        decoded = self.decode_action(action)
        for arm_name, arm_action in decoded.arms.items():
            arm = robot.left if arm_name == "left" else robot.right
            if decoded.control_mode == "joints":
                if arm_action.joint is None:
                    raise ValueError(f"Decoded action for {arm_name} has no joint block")
                arm.command_joint_positions(arm_action.joint, speed_percent=self.joint_speed_percent)
            else:
                if arm_action.ee_pose is None:
                    raise ValueError(f"Decoded action for {arm_name} has no ee_pose block")
                arm.command_end_pose(arm_action.ee_pose, speed_percent=self.ee_speed_percent)

    def command_first_action(self, robot: Any, response_or_actions: dict[str, Any] | np.ndarray) -> None:
        actions = (
            _action_array_from_response(response_or_actions)
            if isinstance(response_or_actions, dict)
            else response_or_actions
        )
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 2:
            self.command_action(robot, actions[0])
            return
        self.command_action(robot, actions)


def _space_summary(space: Any) -> dict[str, Any]:
    gripper = getattr(space, "gripper", None)
    return {
        "ids": getattr(space, "ids", None),
        "arms": getattr(space, "arms", None),
        "ee_rotation": getattr(space, "ee_rotation", None),
        "gripper": None
        if gripper is None
        else {
            "type": getattr(gripper, "type", None),
            "threshold": getattr(gripper, "threshold", None),
            "full_width": getattr(gripper, "full_width", None),
        },
    }


def spec_summary(spec: PiperPolicySpec) -> dict[str, Any]:
    return {
        "train_config_name": spec.train_config_name,
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "model_action_dim": spec.model_action_dim,
        "action_horizon": spec.action_horizon,
        "image_ids": list(spec.image_ids),
        "image_key_map": spec.image_key_map,
        "state_space": _space_summary(spec.state_space),
        "action_space": _space_summary(spec.action_space),
        "image_space": {"ids": getattr(spec.image_space, "ids", None)},
    }


def decoded_action_summary(decoded: DecodedPiperAction) -> dict[str, Any]:
    return {
        "control_mode": decoded.control_mode,
        "arms": {
            arm_name: {
                "has_joint": arm_action.joint is not None,
                "has_ee_pose": arm_action.ee_pose is not None,
                "gripper": arm_action.gripper,
            }
            for arm_name, arm_action in decoded.arms.items()
        },
    }
