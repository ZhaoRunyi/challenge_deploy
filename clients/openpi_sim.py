from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

try:
    from openpi.training import config as openpi_config
except ModuleNotFoundError as error:
    openpi_config = None
    OPENPI_IMPORT_ERROR = error
else:
    OPENPI_IMPORT_ERROR = None

try:
    from openpi_client import image_tools
except ModuleNotFoundError as error:
    image_tools = None
    OPENPI_CLIENT_IMPORT_ERROR = error
else:
    OPENPI_CLIENT_IMPORT_ERROR = None

from hardware.constants import PIPER_GRIPPER_FULL_OPEN_METERS
from hardware.schemas import PiperArmState, RobotSnapshot
from . import websocket_client_policy
from .base import (
    ActionGripperEncoding,
    DecodedArmAction,
    DecodedPiperAction,
    SlaiPiperClient,
    StateGripperEncoding,
    action_gripper_for_piper,
    bounded_gripper_for_piper,
    hardware_gripper_to_model_raw,
    quiet_close_policy_transport_on_construction_error,
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
    if openpi_config is None:
        raise RuntimeError(
            "OpenPI-sim train-config support requires the OpenPI source package "
            "in this environment"
        ) from OPENPI_IMPORT_ERROR
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


def sim_gripper_to_model_raw(
    value: float,
    *,
    action_gripper_encoding: ActionGripperEncoding = "policy",
) -> float:
    value = float(value)
    if action_gripper_encoding == "meters":
        return max(0.0, value)
    if action_gripper_encoding == "binary":
        return value
    state_encoding: StateGripperEncoding = "old" if action_gripper_encoding == "old" else "policy"
    full_open = hardware_gripper_to_model_raw(
        SIM_GRIPPER_FULL_OPEN_M,
        state_gripper_encoding=state_encoding,
    )
    return float(np.clip(value, 0.0, 1.0) * full_open)


def sim_gripper_to_piper(
    value: float,
    threshold: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    *,
    action_gripper_encoding: ActionGripperEncoding = "policy",
    full_open_value: float = PIPER_GRIPPER_FULL_OPEN_METERS,
) -> tuple[float, bool]:
    raw_gripper = action_gripper_for_piper(
        sim_gripper_to_model_raw(value, action_gripper_encoding=action_gripper_encoding),
        None,
        action_gripper_encoding=action_gripper_encoding,
        full_open_value=full_open_value,
    )
    bounded_binary = (
        action_gripper_encoding == "binary"
        or threshold is not None
        or (upper is not None and raw_gripper > upper)
        or (lower is not None and raw_gripper < lower)
    )
    return (
        bounded_gripper_for_piper(
            raw_gripper,
            threshold,
            lower,
            upper,
            full_open_value=full_open_value,
        ),
        bounded_binary,
    )


def piper_gripper_to_sim(
    value: float,
    *,
    state_gripper_encoding: StateGripperEncoding = "policy",
) -> float:
    if state_gripper_encoding == "meters":
        return float(value)
    full_open = hardware_gripper_to_model_raw(
        SIM_GRIPPER_FULL_OPEN_M,
        state_gripper_encoding=state_gripper_encoding,
    )
    if abs(full_open) < 1e-9:
        return 0.0
    encoded = state_gripper_for_policy(
        value,
        None,
        state_gripper_encoding=state_gripper_encoding,
    )
    return float(np.clip(encoded / full_open, 0.0, 1.0))


def arm_state_for_openpi_sim(
    arm: PiperArmState,
    *,
    state_gripper_encoding: StateGripperEncoding = "policy",
) -> np.ndarray:
    return np.concatenate(
        (
            arm.qpos[:6],
            np.array(
                [piper_gripper_to_sim(arm.qpos[6], state_gripper_encoding=state_gripper_encoding)],
                dtype=np.float64,
            ),
        ),
        axis=0,
    ).astype(np.float64)


def build_configured_piper_state(
    snapshot: RobotSnapshot,
    spec: OpenPiSimPolicySpec,
    *,
    state_gripper_encoding: StateGripperEncoding = "policy",
) -> np.ndarray:
    del spec
    return np.concatenate(
        (
            arm_state_for_openpi_sim(snapshot.state.left, state_gripper_encoding=state_gripper_encoding),
            arm_state_for_openpi_sim(snapshot.state.right, state_gripper_encoding=state_gripper_encoding),
        ),
        axis=0,
    )


def image_to_embodichain_rgb(image: np.ndarray) -> np.ndarray:
    if image_tools is None:
        raise RuntimeError(
            "OpenPI-sim image preprocessing requires the openpi-client package"
        ) from OPENPI_CLIENT_IMPORT_ERROR
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image_rgb = image[..., ::-1]
    image_rgb = image_tools.resize_with_pad(image_rgb, SIM_IMAGE_SIZE, SIM_IMAGE_SIZE)
    return image_tools.convert_to_uint8(image_rgb)


def build_policy_payload(
    snapshot: RobotSnapshot,
    *,
    prompt: str | None,
    spec: OpenPiSimPolicySpec,
    state_gripper_encoding: StateGripperEncoding = "policy",
) -> dict[str, Any]:
    if prompt is None:
        raise ValueError("OpenPI-sim policy payload requires a prompt")
    missing = [image_id for image_id in spec.image_ids if image_id not in snapshot.images]
    if missing:
        raise KeyError(f"EmbodiChainInputs requires camera images {spec.image_ids}; missing {missing}")
    return {
        "observation/image": image_to_embodichain_rgb(snapshot.images["cam_high"]),
        "observation/left_wrist_image": image_to_embodichain_rgb(snapshot.images["cam_left_wrist"]),
        "observation/right_wrist_image": image_to_embodichain_rgb(snapshot.images["cam_right_wrist"]),
        "observation/state": build_configured_piper_state(
            snapshot,
            spec,
            state_gripper_encoding=state_gripper_encoding,
        ),
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
        state_gripper_encoding: StateGripperEncoding = "policy",
        action_gripper_encoding: ActionGripperEncoding = "policy",
        bad_sim: bool = False,
    ) -> None:
        if control_mode != "joints":
            raise ValueError("openpi_sim only exposes joint+gripper actions; use control_mode='joints'")
        self.num_steps = num_steps
        self.bad_sim = bad_sim
        spec = load_openpi_sim_policy_spec(train_config_name)
        policy_client = websocket_client_policy.WebsocketClientPolicy(host, port, api_key=api_key)
        try:
            super().__init__(
                spec=spec,
                policy_client=policy_client,
                control_mode=control_mode,
                joint_speed_percent=joint_speed_percent,
                ee_speed_percent=0,
                gripper_threshold=gripper_threshold,
                gripper_lower=gripper_lower,
                gripper_upper=gripper_upper,
                state_gripper_encoding=state_gripper_encoding,
                action_gripper_encoding=action_gripper_encoding,
            )
        except BaseException:
            quiet_close_policy_transport_on_construction_error(policy_client)
            raise

    def validate_control_mode(self) -> None:
        if self.control_mode != "joints":
            raise ValueError("openpi_sim only supports control_mode='joints'")

    def build_payload(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        payload = build_policy_payload(
            snapshot,
            prompt=prompt,
            spec=self.spec,
            state_gripper_encoding=self.state_gripper_encoding,
        )
        if self.num_steps is not None:
            payload["num_steps"] = self.num_steps
        return payload

    def _action_gripper_value(self, value: float) -> float:
        value = float(value)
        return value / 0.05 if self.bad_sim else value

    def _decode_sim_gripper(
        self,
        value: float,
        arm_name: str,
        *,
        full_open_value: float = PIPER_GRIPPER_FULL_OPEN_METERS,
    ) -> tuple[float, bool]:
        arm_threshold = getattr(self, f"{arm_name}_gripper_threshold", None)
        arm_lower = getattr(self, f"{arm_name}_gripper_lower", None)
        arm_upper = getattr(self, f"{arm_name}_gripper_upper", None)
        return sim_gripper_to_piper(
            self._action_gripper_value(value),
            arm_threshold if arm_threshold is not None else self.gripper_threshold,
            arm_lower if arm_lower is not None else self.gripper_lower,
            arm_upper if arm_upper is not None else self.gripper_upper,
            action_gripper_encoding=self.action_gripper_encoding,
            full_open_value=full_open_value,
        )

    def decode_action(
        self,
        action: np.ndarray,
        *,
        gripper_full_openings: Mapping[str, float] | None = None,
    ) -> DecodedPiperAction:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1:
            raise ValueError(f"Expected one action vector, got shape {action.shape}")
        if action.shape[0] < SIM_ACTION_DIM:
            raise ValueError(f"openpi_sim action dim {action.shape[0]} is smaller than expected {SIM_ACTION_DIM}")
        full_openings = gripper_full_openings or {
            "left": PIPER_GRIPPER_FULL_OPEN_METERS,
            "right": PIPER_GRIPPER_FULL_OPEN_METERS,
        }
        left_gripper, left_binary = self._decode_sim_gripper(
            float(action[6]),
            "left",
            full_open_value=float(full_openings["left"]),
        )
        right_gripper, right_binary = self._decode_sim_gripper(
            float(action[13]),
            "right",
            full_open_value=float(full_openings["right"]),
        )
        return DecodedPiperAction(
            control_mode="joints",
            arms={
                "left": DecodedArmAction(
                    joint=np.concatenate((action[:6], np.array([left_gripper])), axis=0),
                    gripper=left_gripper,
                    ee_pose=None,
                    binary_gripper=left_binary,
                ),
                "right": DecodedArmAction(
                    joint=np.concatenate((action[7:13], np.array([right_gripper])), axis=0),
                    gripper=right_gripper,
                    ee_pose=None,
                    binary_gripper=right_binary,
                ),
            },
        )


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
            "gripper_physical_layer": "EmbodiChain gripper01 defaults to policy encoding; --state-gripper meters/old selects the explicit adapter encoding.",
        },
        "action_space": {
            "layout": "left_joints6,left_gripper01,right_joints6,right_gripper01",
            "names": list(SIM_ACTION_NAMES),
            "normalization_note": "server-side EmbodiChainOutputs returns executable-scale joints; gripper encoding is selected by --action-gripper.",
        },
    }
