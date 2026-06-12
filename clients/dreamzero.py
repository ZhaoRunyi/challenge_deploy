from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hardware.constants import PIPER_GRIPPER_FULL_OPEN_METERS
from hardware.schemas import RobotSnapshot
from . import slai_piper_policy
from .base import (
    ControlMode,
    DecodedArmAction,
    DecodedPiperAction,
    SlaiPiperClient,
    action_array_from_response,
    action_gripper_for_piper,
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
    old_gripper: bool = False,
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
        "state": build_full_piper_state(snapshot, spec, old_gripper=old_gripper).astype(np.float32),
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
        gripper_effort: int = 1000,
        gripper_action_frames: int = 5,
        gripper_threshold: float | None = None,
        gripper_lower: float | None = None,
        gripper_upper: float | None = None,
        num_inference_timesteps: int | None = None,
        old_gripper: bool = False,
    ) -> None:
        if not 0 <= gripper_effort <= 5000:
            raise ValueError("gripper_effort must be in [0, 5000]")
        if gripper_action_frames <= 0:
            raise ValueError("gripper_action_frames must be positive")
        self.default_session_id: str | None = None
        self.gripper_effort = gripper_effort
        self.gripper_action_frames = gripper_action_frames
        self.num_inference_timesteps = num_inference_timesteps
        self.last_commanded: DecodedPiperAction | None = None
        self.gripper_transition: tuple[DecodedPiperAction, DecodedPiperAction, int] | None = None
        spec = load_dreamzero_policy_spec(config_path)
        from . import websocket_client_policy

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

    def set_default_session_id(self, session_id: str | None) -> None:
        self.default_session_id = session_id

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
            old_gripper=self.old_gripper,
        )

    def infer(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        response = dict(self.client.infer(self.build_payload(snapshot, prompt, **kwargs)))
        response["actions"] = action_array_from_response(response, keys=("actions", "action")).astype(np.float64)
        return response

    def infer_actions(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> np.ndarray:
        return np.asarray(self.infer(snapshot, prompt, **kwargs)["actions"], dtype=np.float64)

    def get_predicted_video(self, session_id: str) -> dict[str, Any]:
        return dict(self.client.infer({"_request": "get_predicted_video", "session_id": session_id}))

    def save_predicted_video(self, *, session_id: str, output_dir: str | Path, file_stem: str) -> Path | None:
        response = self.get_predicted_video(session_id)
        video_bytes = response.get("predicted_video_bytes")
        if video_bytes is None:
            return None
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{file_stem}_predicted_video.mp4"
        output_path.write_bytes(video_bytes)
        return output_path

    def decode_gripper_for_piper(self, value: float, arm_name: str) -> tuple[float, bool]:
        arm_threshold = getattr(self, f"{arm_name}_gripper_threshold", None)
        arm_lower = getattr(self, f"{arm_name}_gripper_lower", None)
        arm_upper = getattr(self, f"{arm_name}_gripper_upper", None)
        threshold = arm_threshold if arm_threshold is not None else self.gripper_threshold
        lower = arm_lower if arm_lower is not None else self.gripper_lower
        upper = arm_upper if arm_upper is not None else self.gripper_upper
        gripper_config = self.spec.action_space.gripper
        value = action_gripper_for_piper(value, gripper_config, old_gripper=self.old_gripper)
        binary_gripper = bool(gripper_config is not None and getattr(gripper_config, "type", None) == "01")
        if threshold is not None:
            return (PIPER_GRIPPER_FULL_OPEN_METERS if value >= threshold else 0.0), True
        if upper is not None and value > upper:
            return PIPER_GRIPPER_FULL_OPEN_METERS, True
        if lower is not None and value < lower:
            return 0.0, True
        return value, binary_gripper

    def command_decoded(self, robot: Any, decoded: DecodedPiperAction) -> None:
        for arm_name, arm_action in decoded.arms.items():
            arm = robot.left if arm_name == "left" else robot.right
            if decoded.control_mode == "joints":
                if arm_action.joint is None:
                    raise ValueError(f"Decoded action for {arm_name} has no joint block")
                arm.command_joint_positions(arm_action.joint, speed_percent=self.joint_speed_percent, gripper_effort=self.gripper_effort)
            else:
                if arm_action.ee_pose is None:
                    raise ValueError(f"Decoded action for {arm_name} has no ee_pose block")
                arm.command_end_pose(arm_action.ee_pose, speed_percent=self.ee_speed_percent, gripper_effort=self.gripper_effort)

    def current_decoded_from_robot(self, robot: Any) -> DecodedPiperAction:
        state = robot.read_state()
        arms = {
            "left": DecodedArmAction(
                joint=np.asarray(state.left.qpos, dtype=np.float64).copy() if self.control_mode == "joints" else None,
                gripper=float(state.left.qpos[6]),
                ee_pose=None if self.control_mode == "joints" else np.asarray(state.left.end_pose, dtype=np.float64).copy(),
            ),
            "right": DecodedArmAction(
                joint=np.asarray(state.right.qpos, dtype=np.float64).copy() if self.control_mode == "joints" else None,
                gripper=float(state.right.qpos[6]),
                ee_pose=None if self.control_mode == "joints" else np.asarray(state.right.end_pose, dtype=np.float64).copy(),
            ),
        }
        return DecodedPiperAction(arms=arms, control_mode=self.control_mode)

    def command_transition_step(self, robot: Any, start: DecodedPiperAction, target: DecodedPiperAction, step: int) -> None:
        ratio = float(step) / float(self.gripper_action_frames)
        arms: dict[str, DecodedArmAction] = {}
        for arm_name, start_arm in start.arms.items():
            target_arm = target.arms[arm_name]
            gripper = start_arm.gripper
            if target_arm.binary_gripper:
                gripper = start_arm.gripper + (target_arm.gripper - start_arm.gripper) * ratio
            if self.control_mode == "joints":
                if start_arm.joint is None:
                    raise ValueError(f"Transition start for {arm_name} has no joint block")
                arms[arm_name] = DecodedArmAction(
                    joint=np.concatenate((start_arm.joint[:6], np.array([gripper], dtype=np.float64))),
                    gripper=gripper,
                    ee_pose=None,
                    binary_gripper=target_arm.binary_gripper,
                )
            else:
                if start_arm.ee_pose is None:
                    raise ValueError(f"Transition start for {arm_name} has no ee_pose block")
                arms[arm_name] = DecodedArmAction(
                    joint=None,
                    gripper=gripper,
                    ee_pose=np.concatenate((start_arm.ee_pose[:6], np.array([gripper], dtype=np.float64))),
                    binary_gripper=target_arm.binary_gripper,
                )
        decoded = DecodedPiperAction(arms=arms, control_mode=self.control_mode)
        self.command_decoded(robot, decoded)
        self.last_commanded = decoded

    def command_action(self, robot: Any, action: np.ndarray) -> None:
        if self.gripper_transition is not None:
            start, target, step = self.gripper_transition
            self.command_transition_step(robot, start, target, step)
            self.gripper_transition = None if step >= self.gripper_action_frames else (start, target, step + 1)
            return
        decoded = self.decode_action(action)
        if self.last_commanded is None:
            self.last_commanded = self.current_decoded_from_robot(robot)
        if (
            self.last_commanded is not None
            and self.gripper_action_frames > 1
            and any(arm.binary_gripper and not np.isclose(arm.gripper, self.last_commanded.arms[name].gripper, atol=1e-6) for name, arm in decoded.arms.items())
        ):
            start = self.last_commanded
            self.command_transition_step(robot, start, decoded, 1)
            self.gripper_transition = (start, decoded, 2)
            return
        self.command_decoded(robot, decoded)
        self.last_commanded = decoded


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
