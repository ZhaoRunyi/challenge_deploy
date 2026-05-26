from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from openpi.training import config as openpi_config
from openpi_client import image_tools, websocket_client_policy

from hardware.constants import PIPER_GRIPPER_FULL_OPEN_METERS
from hardware.schemas import PiperArmState, RobotSnapshot
from .base import (
    DecodedArmAction,
    DecodedPiperAction,
    SlaiPiperClient,
    action_gripper_for_piper,
    bounded_gripper_for_piper,
    decoded_action_summary,
    hardware_gripper_to_model_raw,
    state_gripper_for_policy,
)

ControlMode = Literal["joints"]
SIM_IMAGE_IDS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SIM_STATE_NAMES = (
    "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
    "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper",
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


def load_openpi_sim_policy_spec(train_config_name: str) -> OpenPiSimPolicySpec:
    train_config = openpi_config.get_config(train_config_name)
    data_config_name = type(train_config.data).__name__
    if "EmbodiChain" not in data_config_name:
        raise TypeError(f"{train_config_name!r} uses {data_config_name}, but this client is EmbodiChain-only.")
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


def sim_gripper_to_model_raw(value: float, *, old_gripper: bool) -> float:
    full_open = hardware_gripper_to_model_raw(SIM_GRIPPER_FULL_OPEN_M, old_gripper=old_gripper)
    return float(np.clip(float(value), 0.0, 1.0) * full_open)


def sim_gripper_to_piper(value: float, threshold: float | None = None, lower: float | None = None, upper: float | None = None, *, old_gripper: bool = False) -> float:
    return bounded_gripper_for_piper(
        action_gripper_for_piper(sim_gripper_to_model_raw(value, old_gripper=old_gripper), None, old_gripper=old_gripper),
        threshold,
        lower,
        upper,
    )


def piper_gripper_to_sim(value: float, *, old_gripper: bool) -> float:
    full_open = hardware_gripper_to_model_raw(SIM_GRIPPER_FULL_OPEN_M, old_gripper=old_gripper)
    return float(np.clip(state_gripper_for_policy(value, None, old_gripper=old_gripper) / full_open, 0.0, 1.0))


def arm_state_for_openpi_sim(arm: PiperArmState, *, old_gripper: bool) -> np.ndarray:
    return np.concatenate((arm.qpos[:6], np.array([piper_gripper_to_sim(arm.qpos[6], old_gripper=old_gripper)], dtype=np.float64)), axis=0).astype(np.float64)


def build_configured_piper_state(snapshot: RobotSnapshot, spec: OpenPiSimPolicySpec, *, old_gripper: bool = False) -> np.ndarray:
    return np.concatenate((arm_state_for_openpi_sim(snapshot.state.left, old_gripper=old_gripper), arm_state_for_openpi_sim(snapshot.state.right, old_gripper=old_gripper)), axis=0)


def image_to_embodichain_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image_rgb = image[..., ::-1]
    image_rgb = image_tools.resize_with_pad(image_rgb, SIM_IMAGE_SIZE, SIM_IMAGE_SIZE)
    return image_tools.convert_to_uint8(image_rgb)


def build_policy_payload(snapshot: RobotSnapshot, *, prompt: str | None, spec: OpenPiSimPolicySpec, old_gripper: bool = False) -> dict[str, Any]:
    if prompt is None:
        raise ValueError("OpenPI-sim policy payload requires a prompt")
    missing = [image_id for image_id in spec.image_ids if image_id not in snapshot.images]
    if missing:
        raise KeyError(f"EmbodiChainInputs requires camera images {spec.image_ids}; missing {missing}")
    return {
        "observation/image": image_to_embodichain_rgb(snapshot.images["cam_high"]),
        "observation/left_wrist_image": image_to_embodichain_rgb(snapshot.images["cam_left_wrist"]),
        "observation/right_wrist_image": image_to_embodichain_rgb(snapshot.images["cam_right_wrist"]),
        "observation/state": build_configured_piper_state(snapshot, spec, old_gripper=old_gripper),
        "prompt": prompt,
    }


class OpenPiSimPiperClient(SlaiPiperClient):
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
        gripper_lower: float | None = None,
        gripper_upper: float | None = None,
        num_steps: int | None = None,
        old_gripper: bool = False,
        bad_sim: bool = False,
    ) -> None:
        if control_mode != "joints":
            raise ValueError("openpi_sim only exposes joint+gripper actions; use control_mode='joints'")
        self.num_steps = num_steps
        self.bad_sim = bad_sim
        spec = load_openpi_sim_policy_spec(train_config_name)
        policy_client = websocket_client_policy.WebsocketClientPolicy(host, port, api_key=api_key)
        super().__init__(
            spec=spec,
            policy_client=policy_client,
            control_mode=control_mode,
            joint_speed_percent=joint_speed_percent,
            ee_speed_percent=0,
            gripper_threshold=gripper_threshold,
            gripper_lower=gripper_lower,
            gripper_upper=gripper_upper,
            old_gripper=old_gripper,
        )

    def validate_control_mode(self) -> None:
        if self.control_mode != "joints":
            raise ValueError("openpi_sim only supports control_mode='joints'")

    def build_payload(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        payload = build_policy_payload(snapshot, prompt=prompt, spec=self.spec, old_gripper=self.old_gripper)
        if self.num_steps is not None:
            payload["num_steps"] = self.num_steps
        return payload

    def decode_action(self, action: np.ndarray) -> DecodedPiperAction:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1:
            raise ValueError(f"Expected one action vector, got shape {action.shape}")
        if action.shape[0] < SIM_ACTION_DIM:
            raise ValueError(f"openpi_sim action dim {action.shape[0]} is smaller than expected {SIM_ACTION_DIM}")
        left_threshold = getattr(self, "left_gripper_threshold", None)
        left_lower = getattr(self, "left_gripper_lower", None)
        left_upper = getattr(self, "left_gripper_upper", None)
        right_threshold = getattr(self, "right_gripper_threshold", None)
        right_lower = getattr(self, "right_gripper_lower", None)
        right_upper = getattr(self, "right_gripper_upper", None)
        left_gripper = sim_gripper_to_piper(
            float(action[6]) / 0.05 if self.bad_sim else float(action[6]),
            left_threshold if left_threshold is not None else self.gripper_threshold,
            left_lower if left_lower is not None else self.gripper_lower,
            left_upper if left_upper is not None else self.gripper_upper,
            old_gripper=self.old_gripper,
        )
        right_gripper = sim_gripper_to_piper(
            float(action[13]) / 0.05 if self.bad_sim else float(action[13]),
            right_threshold if right_threshold is not None else self.gripper_threshold,
            right_lower if right_lower is not None else self.gripper_lower,
            right_upper if right_upper is not None else self.gripper_upper,
            old_gripper=self.old_gripper,
        )
        return DecodedPiperAction(
            control_mode="joints",
            arms={
                "left": DecodedArmAction(joint=np.concatenate((action[:6], np.array([left_gripper])), axis=0), gripper=left_gripper, ee_pose=None),
                "right": DecodedArmAction(joint=np.concatenate((action[7:13], np.array([right_gripper])), axis=0), gripper=right_gripper, ee_pose=None),
            },
        )

    def command_action(self, robot: Any, action: np.ndarray) -> None:
        decoded = self.decode_action(action)
        robot.left.command_joint_positions(decoded.arms["left"].joint, speed_percent=self.joint_speed_percent)
        robot.right.command_joint_positions(decoded.arms["right"].joint, speed_percent=self.joint_speed_percent)


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
