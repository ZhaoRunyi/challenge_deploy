from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hardware.schemas import RobotSnapshot
from . import websocket_client_policy
from . import slai_piper_policy
from .base import (
    ActionGripperEncoding,
    ControlMode,
    SlaiPiperClient,
    StateGripperEncoding,
    action_array_from_response,
    build_full_piper_state,
    image_to_rgb,
)
from .specs import space_summary


@dataclass(frozen=True)
class DreamZeroPolicySpec:
    train_config_name: str
    config_path: str
    config: dict[str, Any]
    train_config: Any
    state_space: Any
    action_space: Any
    image_space: Any
    state_dim: int
    action_dim: int
    model_action_dim: int | None
    action_horizon: int
    image_ids: tuple[str, ...]
    image_key_map: dict[str, str]
    train_data_paths: str | list[str] | None
    video_size: tuple[int, int]
    prompt: str | None = None
    distribution_name: str | None = None
    distribution_aliases: tuple[str, ...] = ()


def _actor_model(config: dict[str, Any]) -> dict[str, Any]:
    actor = dict(config.get("actor") or {})
    model = actor.get("model") or config.get("model") or {}
    if not isinstance(model, dict):
        raise ValueError("DreamZero config must contain actor.model or model")
    return model


def _gripper_config(model: dict[str, Any]) -> slai_piper_policy.GripperConfig:
    return slai_piper_policy.GripperConfig(
        type=str(model.get("gripper_type", "01")),
        threshold=float(model.get("gripper_threshold", 0.01)),
        full_width=float(model.get("gripper_full_width", 0.05)),
    )


def load_dreamzero_policy_spec(config_path: str | Path) -> DreamZeroPolicySpec:
    config_file = Path(config_path).expanduser().resolve()
    with open(config_file, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    model = _actor_model(config)
    data = dict(config.get("data") or {})
    gripper = _gripper_config(model)
    state_space = slai_piper_policy.StateSpaceConfig(ids="joint_gripper", arms="dual", gripper=gripper)
    action_space = slai_piper_policy.ActionSpaceConfig(ids="joint_gripper", arms="dual", gripper=gripper)
    image_space = slai_piper_policy.ImageSpaceConfig(ids="all")
    train_data_paths = data.get("train_data_paths")
    dataset_aliases = _dataset_aliases(train_data_paths)
    default_prompt = str(model.get("default_instruction", "") or "").strip() or None
    return DreamZeroPolicySpec(
        train_config_name=str(config_file),
        config_path=str(config_file),
        config=config,
        train_config=config,
        state_space=state_space,
        action_space=action_space,
        image_space=image_space,
        state_dim=int(slai_piper_policy.get_space_dim(state_space)),
        action_dim=int(slai_piper_policy.get_space_dim(action_space)),
        model_action_dim=int(model.get("max_action_dim", 32)),
        action_horizon=int(model.get("action_horizon", model.get("num_action_per_block", 24))),
        image_ids=tuple(slai_piper_policy.get_image_ids(image_space)),
        image_key_map=slai_piper_policy.get_image_key_map(image_space),
        train_data_paths=train_data_paths,
        video_size=(
            int(model.get("target_video_height", model.get("view_height", 160))),
            int(model.get("target_video_width", model.get("view_width", 320))),
        ),
        prompt=default_prompt,
        distribution_name=Path(str(config_file)).stem,
        distribution_aliases=dataset_aliases,
    )


def _dataset_aliases(train_data_paths: str | list[str] | None) -> tuple[str, ...]:
    raw_paths: list[str]
    if train_data_paths is None:
        raw_paths = []
    elif isinstance(train_data_paths, str):
        raw_paths = [train_data_paths]
    else:
        raw_paths = [str(path) for path in train_data_paths]
    aliases: list[str] = []
    for raw_path in raw_paths:
        name = Path(raw_path).name
        if "*" in raw_path:
            aliases.append(name.replace("*", ""))
        else:
            aliases.append(name)
    return tuple(alias for alias in aliases if alias)


def build_policy_payload(
    snapshot: RobotSnapshot,
    *,
    prompt: str | None,
    spec: DreamZeroPolicySpec,
    session_id: str | None = None,
    num_inference_timesteps: int | None = None,
    state_gripper_encoding: StateGripperEncoding = "policy",
) -> dict[str, Any]:
    if prompt is None:
        raise ValueError("DreamZero policy payload requires a prompt")
    images = {}
    for image_id in spec.image_ids:
        if image_id not in snapshot.images:
            raise KeyError(f"Snapshot is missing required image {image_id}")
        images[image_id] = image_to_rgb(snapshot.images[image_id])
    payload: dict[str, Any] = {
        "images": images,
        "state": build_full_piper_state(
            snapshot,
            spec,
            state_gripper_encoding=state_gripper_encoding,
        ).astype(np.float32),
        "prompt": prompt,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    if num_inference_timesteps is not None:
        payload["num_inference_timesteps"] = num_inference_timesteps
    return payload


class DreamZeroPiperClient(SlaiPiperClient):
    def __init__(
        self,
        config_path: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        control_mode: ControlMode = "joints",
        api_key: str | None = None,
        joint_speed_percent: int = 50,
        ee_speed_percent: int = 50,
        gripper_effort: int | None = None,
        gripper_action_frames: int = 5,
        gripper_threshold: float | None = None,
        gripper_lower: float | None = None,
        gripper_upper: float | None = None,
        num_inference_timesteps: int | None = None,
        state_gripper_encoding: StateGripperEncoding = "policy",
        action_gripper_encoding: ActionGripperEncoding = "policy",
    ) -> None:
        for name, value in (("gripper_threshold", gripper_threshold), ("gripper_lower", gripper_lower), ("gripper_upper", gripper_upper)):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.num_inference_timesteps = num_inference_timesteps
        spec = load_dreamzero_policy_spec(config_path)
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
            state_gripper_encoding=state_gripper_encoding,
            action_gripper_encoding=action_gripper_encoding,
            gripper_effort=gripper_effort,
            gripper_action_frames=gripper_action_frames,
        )
        self.validate_server_metadata()

    def validate_server_metadata(self) -> None:
        metadata = self.get_server_metadata()
        action_dim = metadata.get("action_dim")
        action_chunk_size = metadata.get("action_chunk_size")
        if action_dim is not None and int(action_dim) != self.spec.action_dim:
            raise ValueError(f"DreamZero server action_dim={action_dim}, local config action_dim={self.spec.action_dim}")
        if action_chunk_size is not None and int(action_chunk_size) != self.spec.action_horizon:
            raise ValueError(
                f"DreamZero server action_chunk_size={action_chunk_size}, local config action_horizon={self.spec.action_horizon}"
            )

    def build_payload(
        self,
        snapshot: RobotSnapshot,
        prompt: str | None = None,
        *,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        return build_policy_payload(
            snapshot,
            prompt=prompt,
            spec=self.spec,
            session_id=session_id or self.default_session_id,
            num_inference_timesteps=self.num_inference_timesteps,
            state_gripper_encoding=self.state_gripper_encoding,
        )

    def infer(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        response = dict(self.client.infer(self.build_payload(snapshot, prompt, **kwargs)))
        response["actions"] = action_array_from_response(response, keys=("actions", "action")).astype(np.float64)
        return response

    def infer_actions(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> np.ndarray:
        return np.asarray(self.infer(snapshot, prompt, **kwargs)["actions"], dtype=np.float64)



def spec_summary(spec: DreamZeroPolicySpec) -> dict[str, Any]:
    return {
        "config_path": spec.config_path,
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "model_action_dim": spec.model_action_dim,
        "action_horizon": spec.action_horizon,
        "video_size": list(spec.video_size),
        "image_ids": list(spec.image_ids),
        "image_key_map": spec.image_key_map,
        "train_data_paths": spec.train_data_paths,
        "state_space": space_summary(spec.state_space),
        "action_space": space_summary(spec.action_space),
        "image_space": {"ids": getattr(spec.image_space, "ids", None)},
    }
