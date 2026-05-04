from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from openpi.training import config as openpi_config
from openpi_client import image_tools, websocket_client_policy

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
DEPLOY_SUPPORTED_IMAGE_IDS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


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
    gripper_full_width: float


@dataclass(frozen=True)
class DecodedSimArmAction:
    joint: np.ndarray
    gripper: float
    gripper_01: float


@dataclass(frozen=True)
class DecodedSimPiperAction:
    arms: dict[str, DecodedSimArmAction]
    control_mode: ControlMode


def load_openpi_sim_policy_spec(
    train_config_name: str,
    *,
    gripper_full_width: float = 0.05,
) -> OpenPiSimPolicySpec:
    train_config = openpi_config.get_config(train_config_name)
    if gripper_full_width <= 0.0:
        raise ValueError("gripper_full_width must be positive")

    return OpenPiSimPolicySpec(
        train_config_name=train_config_name,
        train_config=train_config,
        state_dim=SIM_ACTION_DIM,
        action_dim=SIM_ACTION_DIM,
        model_action_dim=getattr(train_config.model, "action_dim", None),
        action_horizon=getattr(train_config.model, "action_horizon", None),
        image_ids=_image_ids_from_train_config(train_config),
        default_prompt=getattr(train_config.data, "default_prompt", None),
        gripper_full_width=float(gripper_full_width),
    )


def _image_ids_from_train_config(train_config: Any) -> tuple[str, ...]:
    repack_transforms = getattr(train_config.data, "repack_transforms", None)
    for transform in getattr(repack_transforms, "inputs", ()):
        images = getattr(transform, "structure", {}).get("images", None)
        if isinstance(images, dict):
            image_ids = tuple(name for name in DEPLOY_SUPPORTED_IMAGE_IDS if name in images)
            if image_ids:
                return image_ids
    return SIM_IMAGE_IDS


def _gripper_opening_to_01(value: float, full_width: float) -> float:
    return float(np.clip(float(value) / full_width, 0.0, 1.0))


def _gripper_01_to_opening(value: float, full_width: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0) * full_width)


def _thresholded_gripper_for_piper(value: float, threshold: float | None) -> float:
    value = float(value)
    if threshold is None:
        return value
    return value if value >= threshold else 0.0


def _arm_state_for_openpi_sim(arm: PiperArmState, *, gripper_full_width: float) -> np.ndarray:
    return np.concatenate(
        (
            arm.qpos[:6],
            np.array([_gripper_opening_to_01(arm.qpos[6], gripper_full_width)], dtype=np.float64),
        ),
        axis=0,
    ).astype(np.float64)


def build_configured_piper_state(snapshot: RobotSnapshot, spec: OpenPiSimPolicySpec) -> np.ndarray:
    """Build openpi_sim's fixed Aloha runtime state: left 6+gripper01, right 6+gripper01."""

    return np.concatenate(
        (
            _arm_state_for_openpi_sim(snapshot.state.left, gripper_full_width=spec.gripper_full_width),
            _arm_state_for_openpi_sim(snapshot.state.right, gripper_full_width=spec.gripper_full_width),
        ),
        axis=0,
    )


def _image_to_aloha_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_tools.resize_with_pad(image_rgb, SIM_IMAGE_SIZE, SIM_IMAGE_SIZE)
    image_rgb = image_tools.convert_to_uint8(image_rgb)
    return np.transpose(image_rgb, (2, 0, 1))


def build_policy_payload(
    snapshot: RobotSnapshot,
    *,
    prompt: str,
    spec: OpenPiSimPolicySpec,
) -> dict[str, Any]:
    images: dict[str, np.ndarray] = {}
    for image_id in spec.image_ids:
        if image_id in snapshot.images:
            images[image_id] = _image_to_aloha_chw(snapshot.images[image_id])
    if "cam_high" not in images:
        raise KeyError("openpi_sim AlohaInputs requires a cam_high image")

    return {
        "state": build_configured_piper_state(snapshot, spec),
        "images": images,
        "prompt": prompt,
    }


def _action_array_from_response(response: dict[str, Any]) -> np.ndarray:
    if "actions" in response:
        return np.asarray(response["actions"], dtype=np.float64)
    if "action" in response:
        return np.asarray(response["action"], dtype=np.float64)
    raise KeyError(f"Policy response does not contain 'actions' or 'action': {sorted(response)}")


class OpenPiSimPiperClient:
    """OpenPI-sim/Aloha websocket client for dual Piper joint deployment.

    openpi_sim's transform chain owns model normalization and delta-joint recovery.
    The server response is already standard Aloha runtime action:
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
        gripper_full_width: float = 0.05,
        gripper_threshold: float | None = None,
    ) -> None:
        if control_mode != "joints":
            raise ValueError("openpi_sim only exposes joint+gripper actions; use control_mode='joints'")
        if gripper_threshold is not None and gripper_threshold < 0.0:
            raise ValueError("gripper_threshold must be non-negative")
        self.spec = load_openpi_sim_policy_spec(train_config_name, gripper_full_width=gripper_full_width)
        self.control_mode = control_mode
        self.joint_speed_percent = joint_speed_percent
        self.gripper_threshold = gripper_threshold
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port, api_key=api_key)

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

    def decode_action(self, action: np.ndarray) -> DecodedSimPiperAction:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1:
            raise ValueError(f"Expected one action vector, got shape {action.shape}")
        if action.shape[0] < SIM_ACTION_DIM:
            raise ValueError(f"openpi_sim action dim {action.shape[0]} is smaller than expected {SIM_ACTION_DIM}")

        left_gripper_01 = float(np.clip(action[6], 0.0, 1.0))
        right_gripper_01 = float(np.clip(action[13], 0.0, 1.0))
        left_gripper = _thresholded_gripper_for_piper(
            _gripper_01_to_opening(left_gripper_01, self.spec.gripper_full_width),
            self.gripper_threshold,
        )
        right_gripper = _thresholded_gripper_for_piper(
            _gripper_01_to_opening(right_gripper_01, self.spec.gripper_full_width),
            self.gripper_threshold,
        )
        return DecodedSimPiperAction(
            control_mode=self.control_mode,
            arms={
                "left": DecodedSimArmAction(
                    joint=np.concatenate((action[:6], np.array([left_gripper])), axis=0),
                    gripper=left_gripper,
                    gripper_01=left_gripper_01,
                ),
                "right": DecodedSimArmAction(
                    joint=np.concatenate((action[7:13], np.array([right_gripper])), axis=0),
                    gripper=right_gripper,
                    gripper_01=right_gripper_01,
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
        "gripper_full_width": spec.gripper_full_width,
        "state_space": {
            "layout": "left_joints6,left_gripper01,right_joints6,right_gripper01",
            "names": list(SIM_STATE_NAMES),
        },
        "action_space": {
            "layout": "left_joints6,left_gripper01,right_joints6,right_gripper01",
            "names": list(SIM_ACTION_NAMES),
            "normalization_note": "server-side AlohaOutputs already returns executable-scale joints; grippers stay 0/1",
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
                "gripper_01": arm_action.gripper_01,
            }
            for arm_name, arm_action in decoded.arms.items()
        },
    }
