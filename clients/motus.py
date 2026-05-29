from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from hardware.constants import PIPER_GRIPPER_FULL_OPEN_METERS
from hardware.schemas import RobotSnapshot
from . import slai_piper_policy
from . import websocket_client_policy
from .base import (
    ControlMode,
    DecodedArmAction,
    DecodedPiperAction,
    SlaiPiperClient,
    action_array_from_response,
    action_gripper_for_piper,
    build_configured_piper_state as build_slai_configured_piper_state,
    image_to_rgb,
)
from .specs import space_summary

MOTUS_STATS_PATH = Path(
    os.environ.get("MOTUS_STATS_PATH", "/workspace/Motus/data/utils/stat.json")
).expanduser().resolve()


@dataclass(frozen=True)
class MotusPolicySpec:
    config_path: str
    config: dict[str, Any]
    state_space: Any
    action_space: Any
    image_space: Any
    state_dim: int
    action_dim: int
    model_action_dim: int | None
    action_horizon: int
    num_video_frames: int
    video_action_freq_ratio: int
    video_size: tuple[int, int]
    embodiment_type: str
    normalization_stats_name: str
    normalization_embodiment_types: tuple[str, ...]
    repo_id: str | None
    image_ids: tuple[str, ...]
    image_key_map: dict[str, str]
    model_image_key_map: dict[str, str]
    action_min: np.ndarray
    action_max: np.ndarray
    per_task_gripper_thresholds: dict[str, float]


def motus_stats_path() -> Path:
    return MOTUS_STATS_PATH


def motus_stats_payload() -> dict[str, Any]:
    with open(motus_stats_path(), "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {motus_stats_path()}, got {type(payload).__name__}")
    return payload


def load_single_motus_normalization_stats(payload: dict[str, Any], embodiment_type: str) -> tuple[np.ndarray, np.ndarray]:
    if embodiment_type not in payload:
        raise KeyError(f"Normalization stats for {embodiment_type!r} not found in {motus_stats_path()}")
    stats = payload[embodiment_type]
    return np.asarray(stats["min"], dtype=np.float32), np.asarray(stats["max"], dtype=np.float32)


def load_motus_normalization_stats(
    embodiment_type: str,
    embodiment_types: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, str, tuple[str, ...]]:
    payload = motus_stats_payload()
    if not embodiment_types:
        action_min, action_max = load_single_motus_normalization_stats(payload, embodiment_type)
        return action_min, action_max, embodiment_type, (embodiment_type,)
    resolved_types = tuple(str(name) for name in embodiment_types)
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for name in resolved_types:
        action_min, action_max = load_single_motus_normalization_stats(payload, name)
        if expected_shape is None:
            expected_shape = action_min.shape
        if action_min.shape != expected_shape or action_max.shape != expected_shape:
            raise ValueError(f"Inconsistent normalization dims for {name!r}: min={action_min.shape}, max={action_max.shape}, expected {expected_shape}")
        mins.append(action_min)
        maxs.append(action_max)
    merged_min = np.minimum.reduce(mins).astype(np.float32)
    merged_max = np.maximum.reduce(maxs).astype(np.float32)
    return merged_min, merged_max, f"merged[{len(resolved_types)}]", resolved_types


def resolve_gripper_config(params: dict[str, Any], *, enabled: bool, prefix: str) -> Any | None:
    if not enabled:
        return None
    gripper_type = params.get(f"{prefix}_gripper_type", params.get("gripper_type", "raw"))
    threshold = float(params.get(f"{prefix}_gripper_threshold", params.get("gripper_threshold", 0.01)))
    full_width = float(params.get(f"{prefix}_gripper_full_width", params.get("gripper_full_width", PIPER_GRIPPER_FULL_OPEN_METERS)))
    return slai_piper_policy.GripperConfig(type=gripper_type, threshold=threshold, full_width=full_width)


def resolve_image_space(params: dict[str, Any]) -> Any:
    ids = params["image_ids"] if "image_ids" in params else params.get("image_space", "all")
    return slai_piper_policy.ImageSpaceConfig(ids=ids)


def resolve_rotation(params: dict[str, Any], key: str) -> str:
    return str(params.get(key, "rot6d"))


def space_with_gripper_threshold(space: Any, threshold: float) -> Any:
    if getattr(space, "gripper", None) is None or space.gripper.type != "01":
        return space
    return replace(space, gripper=replace(space.gripper, threshold=threshold))


def load_motus_policy_spec(config_path: str | Path) -> MotusPolicySpec:
    config_file = Path(config_path).expanduser().resolve()
    with open(config_file, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    common = dict(config.get("common") or {})
    dataset = dict(config.get("dataset") or {})
    params = dict(dataset.get("params") or {})
    state_ids = params.get("state_space", params.get("state_action_space", "joints"))
    action_ids = params.get("action_space", params.get("state_action_space", state_ids))
    arms = params.get("state_action_arms", params.get("arms", "dual"))
    state_rotation = resolve_rotation(params, "state_ee_rotation")
    action_rotation = resolve_rotation(params, "action_ee_rotation")
    state_fields = slai_piper_policy.resolve_ids(state_ids)
    action_fields = slai_piper_policy.resolve_ids(action_ids)
    state_space = slai_piper_policy.StateSpaceConfig(ids=state_ids, arms=arms, ee_rotation=state_rotation, gripper=resolve_gripper_config(params, enabled="gripper" in state_fields, prefix="state"))
    action_space = slai_piper_policy.ActionSpaceConfig(ids=action_ids, arms=arms, ee_rotation=action_rotation, gripper=resolve_gripper_config(params, enabled="gripper" in action_fields, prefix="action"))
    image_space = resolve_image_space(params)
    state_dim = int(slai_piper_policy.get_space_dim(state_space))
    action_dim = int(slai_piper_policy.get_space_dim(action_space))
    model_action_dim = int(common["action_dim"]) if "action_dim" in common else None
    model_state_dim = int(common["state_dim"]) if "state_dim" in common else None
    if model_action_dim is not None and model_action_dim != action_dim:
        raise ValueError(f"{config_file}: common.action_dim={model_action_dim}, action_space dim={action_dim}")
    if model_state_dim is not None and model_state_dim != state_dim:
        raise ValueError(f"{config_file}: common.state_dim={model_state_dim}, state_space dim={state_dim}")
    embodiment_type = str(params["embodiment_type"])
    raw_embodiment_types = params.get("embodiment_types")
    if raw_embodiment_types is None:
        embodiment_types = None
    elif isinstance(raw_embodiment_types, str):
        embodiment_types = [raw_embodiment_types]
    else:
        embodiment_types = [str(name) for name in raw_embodiment_types]
    action_min, action_max, normalization_stats_name, normalization_embodiment_types = load_motus_normalization_stats(embodiment_type, embodiment_types)
    if action_min.shape[0] != action_dim or action_max.shape[0] != action_dim:
        stats_context = normalization_stats_name
        if normalization_embodiment_types != (embodiment_type,):
            stats_context = f"{normalization_stats_name} from {list(normalization_embodiment_types)}"
        raise ValueError(f"{config_file}: normalization stats {stats_context!r} have dim {action_min.shape[0]}, expected {action_dim}")
    gripper_stats = motus_stats_payload().get(embodiment_type, {})
    if "gripper_threshold" not in params and gripper_stats.get("gripper_threshold") is not None:
        state_space = space_with_gripper_threshold(state_space, float(gripper_stats["gripper_threshold"]))
        action_space = space_with_gripper_threshold(action_space, float(gripper_stats["gripper_threshold"]))
    per_task_gripper_thresholds = {}
    if params.get("per_task_gripper_01"):
        grasp_values = gripper_stats.get("gripper_grasp_values", {})
        for task_name, value in grasp_values.get("task", {}).items():
            if value is None:
                continue
            threshold = float(value) * float(gripper_stats.get("gripper_full_width", PIPER_GRIPPER_FULL_OPEN_METERS))
            per_task_gripper_thresholds[str(task_name)] = threshold
            per_task_gripper_thresholds[str(grasp_values.get("prompt", {}).get(task_name, task_name))] = threshold
    return MotusPolicySpec(
        config_path=str(config_file),
        config=config,
        state_space=state_space,
        action_space=action_space,
        image_space=image_space,
        state_dim=state_dim,
        action_dim=action_dim,
        model_action_dim=model_action_dim,
        action_horizon=int(common["num_video_frames"]) * int(common["video_action_freq_ratio"]),
        num_video_frames=int(common["num_video_frames"]),
        video_action_freq_ratio=int(common["video_action_freq_ratio"]),
        video_size=(int(common["video_height"]), int(common["video_width"])),
        embodiment_type=embodiment_type,
        normalization_stats_name=normalization_stats_name,
        normalization_embodiment_types=normalization_embodiment_types,
        repo_id=params.get("repo_id"),
        image_ids=tuple(slai_piper_policy.get_image_ids(image_space)),
        image_key_map=slai_piper_policy.get_image_key_map(image_space),
        model_image_key_map=slai_piper_policy.get_model_image_key_map(image_space),
        action_min=action_min,
        action_max=action_max,
        per_task_gripper_thresholds=per_task_gripper_thresholds,
    )


def normalize_actions(actions: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    action_range = np.where((action_max - action_min) == 0.0, 1.0, action_max - action_min)
    return (actions - action_min) / action_range


def denormalize_actions(actions: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    return actions * (action_max - action_min) + action_min


def build_configured_piper_state(snapshot: RobotSnapshot, spec: MotusPolicySpec, *, old_gripper: bool = False) -> np.ndarray:
    return build_slai_configured_piper_state(snapshot, spec, old_gripper=old_gripper, dtype=np.float32)


def resolve_per_task_gripper_threshold(spec: MotusPolicySpec, prompt: str | None) -> float | None:
    if not prompt or not spec.per_task_gripper_thresholds:
        return None
    prompt = prompt.strip()
    if not prompt:
        return None
    if prompt in spec.per_task_gripper_thresholds:
        return spec.per_task_gripper_thresholds[prompt]
    prompt_lower = prompt.lower()
    for task_prompt, threshold in spec.per_task_gripper_thresholds.items():
        task_prompt_lower = task_prompt.lower()
        if task_prompt and (task_prompt_lower in prompt_lower or prompt_lower in task_prompt_lower):
            return threshold
    return None


def build_normalized_policy_state(snapshot: RobotSnapshot, spec: MotusPolicySpec, *, prompt: str | None = None, old_gripper: bool = False) -> np.ndarray:
    threshold = resolve_per_task_gripper_threshold(spec, prompt)
    if threshold is not None and spec.state_space.gripper is not None:
        gripper = replace(spec.state_space.gripper, threshold=threshold)
        spec = replace(spec, state_space=replace(spec.state_space, gripper=gripper))
    configured = build_configured_piper_state(snapshot, spec, old_gripper=old_gripper)
    return normalize_actions(configured, spec.action_min, spec.action_max).astype(np.float32)


def resize_with_padding(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_size
    original_height, original_width = frame.shape[:2]
    scale = min(target_height / original_height, target_width / original_width)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    resized = cv2.resize(frame, (new_width, new_height))
    padded = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    padded[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return padded


def camera_frame_from_snapshot(snapshot: RobotSnapshot, image_id: str) -> np.ndarray:
    if image_id not in snapshot.images:
        raise KeyError(f"Snapshot is missing required image {image_id}")
    return image_to_rgb(snapshot.images[image_id])


def build_standard_three_view_frame(cam_high: np.ndarray, cam_left: np.ndarray, cam_right: np.ndarray) -> np.ndarray:
    top_h, target_w = cam_high.shape[:2]
    bottom_h = max(1, top_h // 2)
    left_w = target_w // 2
    right_w = target_w - left_w
    cam_left = cv2.resize(cam_left, (left_w, bottom_h))
    cam_right = cv2.resize(cam_right, (right_w, bottom_h))
    stitched = np.zeros((top_h + bottom_h, target_w, 3), dtype=np.uint8)
    stitched[:top_h, :target_w] = cam_high
    stitched[top_h:, :left_w] = cam_left
    stitched[top_h:, left_w:] = cam_right
    return stitched


def build_policy_frame(snapshot: RobotSnapshot, spec: MotusPolicySpec) -> np.ndarray:
    image_ids = tuple(spec.image_ids)
    if image_ids == ("cam_high",):
        return resize_with_padding(camera_frame_from_snapshot(snapshot, "cam_high"), spec.video_size)
    if image_ids == ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        cam_high = camera_frame_from_snapshot(snapshot, "cam_high")
        cam_left = camera_frame_from_snapshot(snapshot, "cam_left_wrist")
        cam_right = camera_frame_from_snapshot(snapshot, "cam_right_wrist")
        return resize_with_padding(build_standard_three_view_frame(cam_high, cam_left, cam_right), spec.video_size)
    if len(image_ids) == 1:
        return resize_with_padding(camera_frame_from_snapshot(snapshot, image_ids[0]), spec.video_size)
    raise NotImplementedError(f"Motus deploy currently supports one-view or standard three-view Piper layouts; got {image_ids}")


def build_policy_payload(
    snapshot: RobotSnapshot,
    *,
    prompt: str | None,
    spec: MotusPolicySpec,
    session_id: str | None = None,
    t5_embeds: np.ndarray | None = None,
    num_inference_timesteps: int | None = None,
    old_gripper: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image": build_policy_frame(snapshot, spec),
        "state": build_normalized_policy_state(snapshot, spec, prompt=prompt, old_gripper=old_gripper),
    }
    if prompt is not None:
        payload["prompt"] = prompt
    if session_id is not None:
        payload["session_id"] = session_id
    if t5_embeds is not None:
        payload["t5_embeds"] = t5_embeds
    if num_inference_timesteps is not None:
        payload["num_inference_timesteps"] = num_inference_timesteps
    return payload


class MotusPiperClient(SlaiPiperClient):
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
        for name, value in (("gripper_threshold", gripper_threshold), ("gripper_lower", gripper_lower), ("gripper_upper", gripper_upper)):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.gripper_effort = gripper_effort
        self.gripper_action_frames = gripper_action_frames
        self.num_inference_timesteps = num_inference_timesteps
        self.default_session_id: str | None = None
        self.last_commanded: DecodedPiperAction | None = None
        self.gripper_transition: tuple[DecodedPiperAction, DecodedPiperAction, int] | None = None
        spec = load_motus_policy_spec(config_path)
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

    @property
    def config_path(self) -> str:
        return self.spec.config_path

    def validate_server_metadata(self) -> None:
        metadata = self.get_server_metadata()
        state_dim = metadata.get("state_dim")
        action_dim = metadata.get("action_dim")
        action_chunk_size = metadata.get("action_chunk_size")
        if state_dim is not None and int(state_dim) != self.spec.state_dim:
            raise ValueError(f"Motus server state_dim={state_dim}, local config state_dim={self.spec.state_dim}")
        if action_dim is not None and int(action_dim) != self.spec.action_dim:
            raise ValueError(f"Motus server action_dim={action_dim}, local config action_dim={self.spec.action_dim}")
        if action_chunk_size is not None and int(action_chunk_size) != self.spec.action_horizon:
            raise ValueError(f"Motus server action_chunk_size={action_chunk_size}, local config action_horizon={self.spec.action_horizon}")

    def set_default_session_id(self, session_id: str | None) -> None:
        self.default_session_id = session_id

    def build_payload(
        self,
        snapshot: RobotSnapshot,
        prompt: str | None = None,
        *,
        session_id: str | None = None,
        t5_embeds: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if session_id is None:
            session_id = self.default_session_id
        return build_policy_payload(
            snapshot,
            prompt=prompt,
            spec=self.spec,
            session_id=session_id,
            t5_embeds=t5_embeds,
            num_inference_timesteps=self.num_inference_timesteps,
            old_gripper=self.old_gripper,
        )

    def infer(
        self,
        snapshot: RobotSnapshot,
        prompt: str | None = None,
        *,
        session_id: str | None = None,
        t5_embeds: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = self.client.infer(self.build_payload(snapshot, prompt, session_id=session_id, t5_embeds=t5_embeds))
        actions = action_array_from_response(response, keys=("actions", "action"))
        response = dict(response)
        response["normalized_actions"] = actions
        response["actions"] = denormalize_actions(actions, self.spec.action_min, self.spec.action_max).astype(np.float64)
        return response

    def infer_actions(
        self,
        snapshot: RobotSnapshot,
        prompt: str | None = None,
        *,
        session_id: str | None = None,
        t5_embeds: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.asarray(self.infer(snapshot, prompt, session_id=session_id, t5_embeds=t5_embeds)["actions"], dtype=np.float64)

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
                arms[arm_name] = DecodedArmAction(joint=np.concatenate((start_arm.joint[:6], np.array([gripper], dtype=np.float64))), gripper=gripper, ee_pose=None, binary_gripper=target_arm.binary_gripper)
            else:
                if start_arm.ee_pose is None:
                    raise ValueError(f"Transition start for {arm_name} has no ee_pose block")
                arms[arm_name] = DecodedArmAction(joint=None, gripper=gripper, ee_pose=np.concatenate((start_arm.ee_pose[:6], np.array([gripper], dtype=np.float64))), binary_gripper=target_arm.binary_gripper)
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


def spec_summary(spec: MotusPolicySpec) -> dict[str, Any]:
    return {
        "config_path": spec.config_path,
        "repo_id": spec.repo_id,
        "embodiment_type": spec.embodiment_type,
        "normalization_stats_name": spec.normalization_stats_name,
        "normalization_embodiment_types": list(spec.normalization_embodiment_types),
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "model_action_dim": spec.model_action_dim,
        "action_horizon": spec.action_horizon,
        "num_video_frames": spec.num_video_frames,
        "video_action_freq_ratio": spec.video_action_freq_ratio,
        "video_size": list(spec.video_size),
        "image_ids": list(spec.image_ids),
        "image_key_map": spec.image_key_map,
        "model_image_key_map": spec.model_image_key_map,
        "state_space": space_summary(spec.state_space),
        "action_space": space_summary(spec.action_space),
        "image_space": {"ids": getattr(spec.image_space, "ids", None)},
    }
