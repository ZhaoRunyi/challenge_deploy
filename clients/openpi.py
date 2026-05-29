from __future__ import annotations

from typing import Any

from openpi.training import config as openpi_config
from openpi_client import websocket_client_policy

from hardware.schemas import RobotSnapshot
from . import slai_piper_policy
from .base import (
    ControlMode,
    SlaiPiperClient,
    build_full_piper_state,
    image_to_rgb,
)
from .specs import SlaiPolicySpec, slai_policy_spec_summary


class PiperPolicySpec(SlaiPolicySpec):
    pass


def load_piper_policy_spec(train_config_name: str) -> PiperPolicySpec:
    train_config = openpi_config.get_config(train_config_name)
    data_config = train_config.data
    missing = [name for name in ("state_space", "action_space", "image_space") if not hasattr(data_config, name)]
    if missing:
        raise TypeError(f"OpenPI train config {train_config_name!r} is not a SLAI Piper config; missing data fields: {missing}")
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


def build_policy_payload(snapshot: RobotSnapshot, *, prompt: str | None, spec: PiperPolicySpec, old_gripper: bool = False) -> dict[str, Any]:
    if prompt is None:
        raise ValueError("OpenPI policy payload requires a prompt")
    payload: dict[str, Any] = {
        "observation.state": build_full_piper_state(snapshot, spec, old_gripper=old_gripper),
        "prompt": prompt,
    }
    for image_id, dataset_key in spec.image_key_map.items():
        if image_id not in snapshot.images:
            raise KeyError(f"Snapshot is missing required image {image_id}")
        payload[dataset_key] = image_to_rgb(snapshot.images[image_id])
    return payload


class OpenPiPiperClient(SlaiPiperClient):
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
        gripper_threshold: float | None = None,
        gripper_lower: float | None = None,
        gripper_upper: float | None = None,
        old_gripper: bool = False,
    ) -> None:
        spec = load_piper_policy_spec(train_config_name)
        policy_client = websocket_client_policy.WebsocketClientPolicy(host, port, api_key=api_key)
        super().__init__(
            spec=spec,
            policy_client=policy_client,
            control_mode=control_mode,
            joint_speed_percent=joint_speed_percent,
            ee_speed_percent=ee_speed_percent,
            gripper_threshold=gripper_threshold,
            gripper_lower=gripper_lower,
            gripper_upper=gripper_upper,
            old_gripper=old_gripper,
        )

    def build_payload(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        return build_policy_payload(snapshot, prompt=prompt, spec=self.spec, old_gripper=self.old_gripper)


def spec_summary(spec: PiperPolicySpec) -> dict[str, Any]:
    return slai_policy_spec_summary(spec)
