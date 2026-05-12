from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from openpi.training import config as openpi_config
from openpi_client import image_tools, websocket_client_policy

from .constants import PIPER_GRIPPER_FULL_OPEN_METERS
from .conversions import (
    legacy_piper_raw_gripper_to_opening,
    normalized_gripper_to_opening,
    opening_to_legacy_piper_raw_gripper,
    opening_to_normalized_gripper,
)
from .schemas import PiperArmState, RobotSnapshot


ControlMode = Literal["joints"]

SIM_IMAGE_IDS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SIM_STATE_NAMES = (
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
)
SIM_ACTION_NAMES = SIM_STATE_NAMES
SIM_ACTION_DIM = 14
SIM_IMAGE_SIZE = 224
SIM_GRIPPER_FULL_OPEN_M = PIPER_GRIPPER_FULL_OPEN_METERS


@dataclass(frozen=True)
class OpenPiSimPolicySpec:
    train_config_name: str
    train_config: Any
    state_dim: int
    action_dim: int
    model_action_dim: int | None
    action_horizon: int | None
    image_ids: tuple[str, ...]
    default_prompt: str | None


def _hardware_gripper_to_model_raw(value: float, *, old_gripper: bool) -> float:
    if old_gripper:
        return opening_to_legacy_piper_raw_gripper(value)
    return opening_to_normalized_gripper(value)


def _model_raw_gripper_to_hardware(value: float, *, old_gripper: bool) -> float:
    if old_gripper:
        return legacy_piper_raw_gripper_to_opening(value)
    return normalized_gripper_to_opening(value)


def _state_gripper_for_openpi(value: float, gripper_cfg: Any, *, old_gripper: bool) -> float:
    value = float(value)
    if gripper_cfg is not None and gripper_cfg.type == "01":
        return value / gripper_cfg.full_width if gripper_cfg.full_width > 0 else value
    return _hardware_gripper_to_model_raw(value, old_gripper=old_gripper)


def _action_gripper_for_piper(value: float, gripper_cfg: Any, *, old_gripper: bool) -> float:
    value = float(value)
    if gripper_cfg is not None and gripper_cfg.type == "01":
        return gripper_cfg.full_width if value >= 0.5 else 0.0
    return _model_raw_gripper_to_hardware(value, old_gripper=old_gripper)


def _bounded_gripper_for_piper(value: float, threshold: float | None, lower: float | None = None, upper: float | None = None) -> float:
    value = max(0.0, float(value))
    if threshold is not None:
        return PIPER_GRIPPER_FULL_OPEN_METERS if value >= threshold else 0.0
    if upper is not None and value > upper:
        return PIPER_GRIPPER_FULL_OPEN_METERS
    return 0.0 if lower is not None and value < lower else value


@dataclass(frozen=True)
class DecodedSimArmAction:
    joint: np.ndarray
    gripper: float
    ee_pose: None = None


@dataclass(frozen=True)
class DecodedSimPiperAction:
    arms: dict[str, DecodedSimArmAction]
    control_mode: ControlMode


def load_openpi_sim_policy_spec(train_config_name: str) -> OpenPiSimPolicySpec:
    train_config = openpi_config.get_config(train_config_name)
    data_config_name = type(train_config.data).__name__
    if "EmbodiChain" not in data_config_name:
        raise TypeError(
            f"{train_config_name!r} uses {data_config_name}, but this client is EmbodiChain-only."
        )

    return OpenPiSimPolicySpec(
        train_config_name=train_config_name,
        train_config=train_config,
        state_dim=SIM_ACTION_DIM,
        action_dim=SIM_ACTION_DIM,
        model_action_dim=getattr(train_config.model, "action_dim", None),
        action_horizon=getattr(train_config.model, "action_horizon", None),
        image_ids=SIM_IMAGE_IDS,
        default_prompt=getattr(train_config.data, "default_prompt", None),
    )


def _sim_gripper_to_model_raw(value: float, *, old_gripper: bool) -> float:
    full_open = _hardware_gripper_to_model_raw(SIM_GRIPPER_FULL_OPEN_M, old_gripper=old_gripper)
    return float(np.clip(float(value), 0.0, 1.0) * full_open)


def sim_gripper_to_piper(
    value: float,
    threshold: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    *,
    old_gripper: bool = False,
) -> float:
    return _bounded_gripper_for_piper(
        _action_gripper_for_piper(
            _sim_gripper_to_model_raw(value, old_gripper=old_gripper),
            None,
            old_gripper=old_gripper,
        ),
        threshold,
        lower,
        upper,
    )


def _piper_gripper_to_sim(value: float, *, old_gripper: bool) -> float:
    full_open = _hardware_gripper_to_model_raw(SIM_GRIPPER_FULL_OPEN_M, old_gripper=old_gripper)
    return float(
        np.clip(
            _state_gripper_for_openpi(value, None, old_gripper=old_gripper) / full_open,
            0.0,
            1.0,
        )
    )


def _arm_state_for_openpi_sim(arm: PiperArmState, *, old_gripper: bool) -> np.ndarray:
    return np.concatenate(
        (
            arm.qpos[:6],
            np.array([_piper_gripper_to_sim(arm.qpos[6], old_gripper=old_gripper)], dtype=np.float64),
        ),
        axis=0,
    ).astype(np.float64)


def build_configured_piper_state(
    snapshot: RobotSnapshot,
    spec: OpenPiSimPolicySpec,
    *,
    old_gripper: bool = False,
) -> np.ndarray:
    """Build EmbodiChain's fixed qpos state: left 6+gripper01, right 6+gripper01."""

    return np.concatenate(
        (
            _arm_state_for_openpi_sim(snapshot.state.left, old_gripper=old_gripper),
            _arm_state_for_openpi_sim(snapshot.state.right, old_gripper=old_gripper),
        ),
        axis=0,
    )


def _image_to_embodichain_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_tools.resize_with_pad(image_rgb, SIM_IMAGE_SIZE, SIM_IMAGE_SIZE)
    return image_tools.convert_to_uint8(image_rgb)


def build_policy_payload(
    snapshot: RobotSnapshot,
    *,
    prompt: str,
    spec: OpenPiSimPolicySpec,
    old_gripper: bool = False,
) -> dict[str, Any]:
    missing = [image_id for image_id in spec.image_ids if image_id not in snapshot.images]
    if missing:
        raise KeyError(f"EmbodiChainInputs requires camera images {spec.image_ids}; missing {missing}")
    return {
        "observation/image": _image_to_embodichain_rgb(snapshot.images["cam_high"]),
        "observation/left_wrist_image": _image_to_embodichain_rgb(snapshot.images["cam_left_wrist"]),
        "observation/right_wrist_image": _image_to_embodichain_rgb(snapshot.images["cam_right_wrist"]),
        "observation/state": build_configured_piper_state(snapshot, spec, old_gripper=old_gripper),
        "prompt": prompt,
    }


def _action_array_from_response(response: dict[str, Any]) -> np.ndarray:
    if "actions" in response:
        return np.asarray(response["actions"], dtype=np.float64)
    if "action" in response:
        return np.asarray(response["action"], dtype=np.float64)
    raise KeyError(f"Policy response does not contain 'actions' or 'action': {sorted(response)}")


class OpenPiSimPiperClient:
    """OpenPI-sim EmbodiChain websocket client for dual Piper joint deployment.

    openpi_sim's transform chain owns model normalization and delta-joint recovery.
    The server response is already EmbodiChain runtime action:
    [left 6 joints, left gripper 0/1, right 6 joints, right gripper 0/1].
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
        gripper_threshold: float | None = None,
        old_gripper: bool = False,
    ) -> None:
        if control_mode != "joints":
            raise ValueError("openpi_sim only exposes joint+gripper actions; use control_mode='joints'")
        self.spec = load_openpi_sim_policy_spec(train_config_name)
        self.control_mode = control_mode
        self.joint_speed_percent = joint_speed_percent
        self.gripper_threshold = gripper_threshold
        self.old_gripper = old_gripper
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port, api_key=api_key)

    @property
    def train_config_name(self) -> str:
        return self.spec.train_config_name

    def get_server_metadata(self) -> Any:
        return self._client.get_server_metadata()

    def build_payload(self, snapshot: RobotSnapshot, prompt: str) -> dict[str, Any]:
        return build_policy_payload(snapshot, prompt=prompt, spec=self.spec, old_gripper=self.old_gripper)

    def infer(self, snapshot: RobotSnapshot, prompt: str) -> dict[str, Any]:
        return self._client.infer(self.build_payload(snapshot, prompt))

    def infer_actions(self, snapshot: RobotSnapshot, prompt: str) -> np.ndarray:
        return _action_array_from_response(self.infer(snapshot, prompt))

    def decode_action(self, action: np.ndarray) -> DecodedSimPiperAction:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1:
            raise ValueError(f"Expected one action vector, got shape {action.shape}")
        if action.shape[0] < SIM_ACTION_DIM:
            raise ValueError(f"openpi_sim action dim {action.shape[0]} is smaller than expected {SIM_ACTION_DIM}")

        left_threshold, left_lower, left_upper = getattr(self, "left_gripper_threshold", None), getattr(self, "left_gripper_lower", None), getattr(self, "left_gripper_upper", None)
        left_gripper = sim_gripper_to_piper(
            float(action[6]),
            left_threshold if left_threshold is not None else self.gripper_threshold,
            left_lower if left_lower is not None else getattr(self, "gripper_lower", None),
            left_upper if left_upper is not None else getattr(self, "gripper_upper", None),
            old_gripper=self.old_gripper,
        )
        right_threshold, right_lower, right_upper = getattr(self, "right_gripper_threshold", None), getattr(self, "right_gripper_lower", None), getattr(self, "right_gripper_upper", None)
        right_gripper = sim_gripper_to_piper(
            float(action[13]),
            right_threshold if right_threshold is not None else self.gripper_threshold,
            right_lower if right_lower is not None else getattr(self, "gripper_lower", None),
            right_upper if right_upper is not None else getattr(self, "gripper_upper", None),
            old_gripper=self.old_gripper,
        )
        return DecodedSimPiperAction(
            control_mode=self.control_mode,
            arms={
                "left": DecodedSimArmAction(
                    joint=np.concatenate((action[:6], np.array([left_gripper])), axis=0),
                    gripper=left_gripper,
                ),
                "right": DecodedSimArmAction(
                    joint=np.concatenate((action[7:13], np.array([right_gripper])), axis=0),
                    gripper=right_gripper,
                ),
            },
        )

    def command_action(self, robot: Any, action: np.ndarray) -> None:
        decoded = self.decode_action(action)
        robot.left.command_joint_positions(decoded.arms["left"].joint, speed_percent=self.joint_speed_percent)
        robot.right.command_joint_positions(decoded.arms["right"].joint, speed_percent=self.joint_speed_percent)

    def command_first_action(self, robot: Any, response_or_actions: dict[str, Any] | np.ndarray) -> None:
        actions = (
            _action_array_from_response(response_or_actions)
            if isinstance(response_or_actions, dict)
            else response_or_actions
        )
        actions = np.asarray(actions, dtype=np.float64)
        self.command_action(robot, actions[0] if actions.ndim == 2 else actions)


def spec_summary(spec: OpenPiSimPolicySpec) -> dict[str, Any]:
    return {
        "train_config_name": spec.train_config_name,
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "model_action_dim": spec.model_action_dim,
        "action_horizon": spec.action_horizon,
        "image_ids": list(spec.image_ids),
        "default_prompt": spec.default_prompt,
        "state_space": {
            "layout": "left_joints6,left_gripper01,right_joints6,right_gripper01",
            "names": list(SIM_STATE_NAMES),
            "gripper_full_open_m": SIM_GRIPPER_FULL_OPEN_M,
            "gripper_physical_layer": "EmbodiChain gripper01 is adapted through the same raw Piper conversion as openpi_client",
        },
        "action_space": {
            "layout": "left_joints6,left_gripper01,right_joints6,right_gripper01",
            "names": list(SIM_ACTION_NAMES),
            "normalization_note": "server-side EmbodiChainOutputs returns executable-scale joints; gripper01=1 maps to the calibrated full-open raw Piper value",
        },
    }


def decoded_action_summary(decoded: DecodedSimPiperAction) -> dict[str, Any]:
    return {
        "control_mode": decoded.control_mode,
        "arms": {
            arm_name: {
                "has_joint": True,
                "has_ee_pose": False,
                "gripper": arm_action.gripper,
            }
            for arm_name, arm_action in decoded.arms.items()
        },
    }
