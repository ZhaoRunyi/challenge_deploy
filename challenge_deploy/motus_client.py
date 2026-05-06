from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Literal

import cv2
import numpy as np
import yaml

from .constants import PIPER_GRIPPER_FULL_OPEN_METERS
from .conversions import (
    legacy_piper_raw_gripper_to_opening,
    normalized_gripper_to_opening,
    opening_to_legacy_piper_raw_gripper,
    opening_to_normalized_gripper,
)
from .schemas import PiperArmState, RobotSnapshot

try:
    from scipy.spatial.transform import Rotation
except ImportError:  # pragma: no cover - scipy exists in the intended deploy env.
    Rotation = None


ControlMode = Literal["joints", "ee_pose"]


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


@dataclass(frozen=True)
class DecodedArmAction:
    joint: np.ndarray | None
    gripper: float
    ee_pose: np.ndarray | None


@dataclass(frozen=True)
class DecodedPiperAction:
    arms: dict[str, DecodedArmAction]
    control_mode: ControlMode


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _motus_root() -> Path:
    return _repo_root() / "baselines" / "Motus"


def _motus_policy_path() -> Path:
    return _motus_root() / "data" / "lerobot" / "slai_piper_policy.py"


def _motus_websocket_client_policy_path() -> Path:
    return _motus_root() / "inference" / "challenge_deploy" / "websocket_client_policy.py"


def _motus_stats_path() -> Path:
    return _motus_root() / "data" / "utils" / "stat.json"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _motus_slai_policy() -> Any:
    return _load_module("challenge_deploy_motus_slai_piper_policy", _motus_policy_path())


@lru_cache(maxsize=1)
def _motus_websocket_client_policy_module() -> Any:
    return _load_module(
        "challenge_deploy_motus_websocket_client_policy",
        _motus_websocket_client_policy_path(),
    )


@lru_cache(maxsize=1)
def _motus_stats_payload() -> dict[str, Any]:
    with open(_motus_stats_path(), "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {_motus_stats_path()}, got {type(payload).__name__}")
    return payload


def _load_single_motus_normalization_stats(
    payload: dict[str, Any],
    embodiment_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    if embodiment_type not in payload:
        raise KeyError(f"Normalization stats for {embodiment_type!r} not found in {_motus_stats_path()}")
    stats = payload[embodiment_type]
    return (
        np.asarray(stats["min"], dtype=np.float32),
        np.asarray(stats["max"], dtype=np.float32),
    )


def load_motus_normalization_stats(
    embodiment_type: str,
    embodiment_types: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, str, tuple[str, ...]]:
    payload = _motus_stats_payload()
    if not embodiment_types:
        action_min, action_max = _load_single_motus_normalization_stats(payload, embodiment_type)
        return action_min, action_max, embodiment_type, (embodiment_type,)

    resolved_types = tuple(str(name) for name in embodiment_types)
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for name in resolved_types:
        action_min, action_max = _load_single_motus_normalization_stats(payload, name)
        if expected_shape is None:
            expected_shape = action_min.shape
        if action_min.shape != expected_shape or action_max.shape != expected_shape:
            raise ValueError(
                f"Inconsistent normalization dims for {name!r}: "
                f"min={action_min.shape}, max={action_max.shape}, expected {expected_shape}"
            )
        mins.append(action_min)
        maxs.append(action_max)

    merged_min = np.minimum.reduce(mins).astype(np.float32)
    merged_max = np.maximum.reduce(maxs).astype(np.float32)
    return merged_min, merged_max, f"merged[{len(resolved_types)}]", resolved_types


def _resolve_gripper_config(
    slai_policy: Any,
    params: dict[str, Any],
    *,
    enabled: bool,
    prefix: str,
) -> Any | None:
    if not enabled:
        return None
    gripper_type = params.get(f"{prefix}_gripper_type", params.get("gripper_type", "raw"))
    threshold = float(params.get(f"{prefix}_gripper_threshold", params.get("gripper_threshold", 0.01)))
    full_width = float(params.get(f"{prefix}_gripper_full_width", params.get("gripper_full_width", 0.05)))
    return slai_policy.GripperConfig(type=gripper_type, threshold=threshold, full_width=full_width)


def _resolve_image_space(slai_policy: Any, params: dict[str, Any]) -> Any:
    if "image_ids" in params:
        ids = params["image_ids"]
    else:
        ids = params.get("image_space", "all")
    return slai_policy.ImageSpaceConfig(ids=ids)


def _resolve_rotation(params: dict[str, Any], key: str) -> str:
    return str(params.get(key, "rot6d"))


def load_motus_policy_spec(config_path: str | Path) -> MotusPolicySpec:
    config_file = Path(config_path).expanduser().resolve()
    with open(config_file, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    common = dict(config.get("common") or {})
    dataset = dict(config.get("dataset") or {})
    params = dict(dataset.get("params") or {})
    slai_policy = _motus_slai_policy()

    state_ids = params.get("state_space", params.get("state_action_space", "joints"))
    action_ids = params.get("action_space", params.get("state_action_space", state_ids))
    arms = params.get("state_action_arms", params.get("arms", "dual"))
    state_rotation = _resolve_rotation(params, "state_ee_rotation")
    action_rotation = _resolve_rotation(params, "action_ee_rotation")
    state_fields = slai_policy._resolve_ids(state_ids)
    action_fields = slai_policy._resolve_ids(action_ids)

    state_space = slai_policy.StateSpaceConfig(
        ids=state_ids,
        arms=arms,
        ee_rotation=state_rotation,
        gripper=_resolve_gripper_config(slai_policy, params, enabled="gripper" in state_fields, prefix="state"),
    )
    action_space = slai_policy.ActionSpaceConfig(
        ids=action_ids,
        arms=arms,
        ee_rotation=action_rotation,
        gripper=_resolve_gripper_config(slai_policy, params, enabled="gripper" in action_fields, prefix="action"),
    )
    image_space = _resolve_image_space(slai_policy, params)

    state_dim = int(slai_policy.get_space_dim(state_space))
    action_dim = int(slai_policy.get_space_dim(action_space))
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
    action_min, action_max, normalization_stats_name, normalization_embodiment_types = (
        load_motus_normalization_stats(embodiment_type, embodiment_types)
    )
    if action_min.shape[0] != action_dim or action_max.shape[0] != action_dim:
        stats_context = normalization_stats_name
        if normalization_embodiment_types != (embodiment_type,):
            stats_context = f"{normalization_stats_name} from {list(normalization_embodiment_types)}"
        raise ValueError(
            f"{config_file}: normalization stats {stats_context!r} have dim "
            f"{action_min.shape[0]}, expected {action_dim}"
        )

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
        image_ids=tuple(slai_policy.get_image_ids(image_space)),
        image_key_map=slai_policy.get_image_key_map(image_space),
        model_image_key_map=slai_policy.get_model_image_key_map(image_space),
        action_min=action_min,
        action_max=action_max,
    )


def _normalize_actions(actions: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    action_range = np.where((action_max - action_min) == 0.0, 1.0, action_max - action_min)
    return (actions - action_min) / action_range


def _denormalize_actions(actions: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    return actions * (action_max - action_min) + action_min


def _require_rotation() -> Any:
    if Rotation is None:
        raise RuntimeError("scipy is required for ee rotation conversion")
    return Rotation


def _resolve_rotation_format(rotation: str) -> str:
    return _motus_slai_policy()._resolve_rotation_format(rotation)


def rpy_to_rotation(rpy: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation_format = _resolve_rotation_format(rotation_format)
    rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
    if rotation_format == "rpy":
        return rpy
    rot = _require_rotation().from_euler("xyz", rpy, degrees=False)
    if rotation_format == "quat":
        return rot.as_quat().astype(np.float64)
    return rot.as_matrix()[:, :2].reshape(-1).astype(np.float64)


def rotation_to_rpy(values: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation_format = _resolve_rotation_format(rotation_format)
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


def _hardware_gripper_to_model_raw(value: float, *, old_gripper: bool) -> float:
    if old_gripper:
        return opening_to_legacy_piper_raw_gripper(value)
    return opening_to_normalized_gripper(value)


def _model_raw_gripper_to_hardware(value: float, *, old_gripper: bool) -> float:
    if old_gripper:
        return legacy_piper_raw_gripper_to_opening(value)
    return normalized_gripper_to_opening(value)


def _state_gripper_for_motus(value: float, gripper_cfg: Any, *, old_gripper: bool) -> float:
    value = float(value)
    if gripper_cfg is not None and gripper_cfg.type == "01":
        return value / gripper_cfg.full_width if gripper_cfg.full_width > 0 else value
    return _hardware_gripper_to_model_raw(value, old_gripper=old_gripper)


def _action_gripper_for_piper(value: float, gripper_cfg: Any, *, old_gripper: bool) -> float:
    value = float(value)
    if gripper_cfg is not None and gripper_cfg.type == "01":
        return gripper_cfg.full_width if value >= 0.5 else 0.0
    return _model_raw_gripper_to_hardware(value, old_gripper=old_gripper)


def _thresholded_gripper_for_piper(
    value: float,
    threshold: float | None,
    gripper_cfg: Any | None,
    lower: float | None = None,
    upper: float | None = None,
    *,
    old_gripper: bool,
) -> float:
    value = _action_gripper_for_piper(value, gripper_cfg, old_gripper=old_gripper)
    if threshold is not None:
        return value if value >= threshold else 0.0
    if upper is not None and value > upper:
        return PIPER_GRIPPER_FULL_OPEN_METERS
    return 0.0 if lower is not None and value < lower else value


def _arm_full_state(
    arm: PiperArmState,
    *,
    ee_rotation: str,
    gripper_cfg: Any,
    old_gripper: bool,
) -> np.ndarray:
    return np.concatenate(
        (
            arm.qpos[:6],
            np.array([_state_gripper_for_motus(arm.qpos[6], gripper_cfg, old_gripper=old_gripper)], dtype=np.float64),
            arm.end_pose[:3],
            rpy_to_rotation(arm.end_pose[3:6], ee_rotation),
        ),
        axis=0,
    ).astype(np.float64)


def build_full_piper_state(
    snapshot: RobotSnapshot,
    spec: MotusPolicySpec,
    *,
    old_gripper: bool = False,
) -> np.ndarray:
    return np.concatenate(
        (
            _arm_full_state(
                snapshot.state.left,
                ee_rotation=spec.state_space.ee_rotation,
                gripper_cfg=spec.state_space.gripper,
                old_gripper=old_gripper,
            ),
            _arm_full_state(
                snapshot.state.right,
                ee_rotation=spec.state_space.ee_rotation,
                gripper_cfg=spec.state_space.gripper,
                old_gripper=old_gripper,
            ),
        ),
        axis=0,
    )


def build_configured_piper_state(
    snapshot: RobotSnapshot,
    spec: MotusPolicySpec,
    *,
    old_gripper: bool = False,
) -> np.ndarray:
    slai_policy = _motus_slai_policy()
    full_state = build_full_piper_state(snapshot, spec, old_gripper=old_gripper)
    state_space = slai_policy._space_from_state_config(spec.state_space)
    return np.asarray(
        slai_policy._extract_vec(full_state, state_space, spec.state_space.gripper),
        dtype=np.float32,
    )


def build_normalized_policy_state(
    snapshot: RobotSnapshot,
    spec: MotusPolicySpec,
    *,
    old_gripper: bool = False,
) -> np.ndarray:
    configured = build_configured_piper_state(snapshot, spec, old_gripper=old_gripper)
    return _normalize_actions(configured, spec.action_min, spec.action_max).astype(np.float32)


def _image_to_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_with_padding(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
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


def _camera_frame_from_snapshot(snapshot: RobotSnapshot, image_id: str) -> np.ndarray:
    if image_id not in snapshot.images:
        raise KeyError(f"Snapshot is missing required image {image_id}")
    return _image_to_rgb(snapshot.images[image_id])


def _build_standard_three_view_frame(
    cam_high: np.ndarray,
    cam_left: np.ndarray,
    cam_right: np.ndarray,
) -> np.ndarray:
    """Match Motus training-time T-shape concatenation.

    Head view stays at full size. Left/right wrist views are resized to half
    height and half width, then stacked side-by-side below the head view.
    """
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
        return _resize_with_padding(_camera_frame_from_snapshot(snapshot, "cam_high"), spec.video_size)

    if image_ids == ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        cam_high = _camera_frame_from_snapshot(snapshot, "cam_high")
        cam_left = _camera_frame_from_snapshot(snapshot, "cam_left_wrist")
        cam_right = _camera_frame_from_snapshot(snapshot, "cam_right_wrist")
        stitched = _build_standard_three_view_frame(cam_high, cam_left, cam_right)
        return _resize_with_padding(stitched, spec.video_size)

    if len(image_ids) == 1:
        return _resize_with_padding(_camera_frame_from_snapshot(snapshot, image_ids[0]), spec.video_size)

    raise NotImplementedError(
        f"Motus deploy currently supports one-view or standard three-view Piper layouts; got {image_ids}"
    )


def build_policy_payload(
    snapshot: RobotSnapshot,
    *,
    prompt: str | None,
    spec: MotusPolicySpec,
    session_id: str | None = None,
    t5_embeds: np.ndarray | None = None,
    old_gripper: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image": build_policy_frame(snapshot, spec),
        "state": build_normalized_policy_state(snapshot, spec, old_gripper=old_gripper),
    }
    if prompt is not None:
        payload["prompt"] = prompt
    if session_id is not None:
        payload["session_id"] = session_id
    if t5_embeds is not None:
        payload["t5_embeds"] = t5_embeds
    return payload


def _action_array_from_response(response: dict[str, Any]) -> np.ndarray:
    if "actions" in response:
        return np.asarray(response["actions"], dtype=np.float64)
    if "action" in response:
        return np.asarray(response["action"], dtype=np.float64)
    raise KeyError(f"Policy response does not contain 'actions' or 'action': {sorted(response)}")


class MotusPiperClient:
    """Remote Motus websocket client for SLAI Piper-style configs."""

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
        gripper_threshold: float | None = None,
        old_gripper: bool = False,
    ) -> None:
        self.spec = load_motus_policy_spec(config_path)
        self.control_mode = control_mode
        self.joint_speed_percent = joint_speed_percent
        self.ee_speed_percent = ee_speed_percent
        self.gripper_threshold = gripper_threshold
        self.old_gripper = old_gripper
        self._default_session_id: str | None = None
        self._validate_control_mode()
        self._client = _motus_websocket_client_policy_module().WebsocketClientPolicy(host, port, api_key=api_key)
        self._validate_server_metadata()

    def _validate_control_mode(self) -> None:
        slai_policy = _motus_slai_policy()
        fields = set(slai_policy._fields_from_action_config(self.spec.action_space))
        if "gripper" not in fields:
            raise ValueError(f"{self.spec.config_path}: deploy requires action_space to include gripper")
        if self.control_mode == "joints":
            if "joint" not in fields:
                raise ValueError(f"{self.spec.config_path}: control_mode='joints' requires action_space to include joint")
            return
        if self.control_mode == "ee_pose":
            missing = {"ee_pos", "ee_rot"} - fields
            if missing:
                raise ValueError(
                    f"{self.spec.config_path}: control_mode='ee_pose' requires action_space fields {sorted(missing)}"
                )
            return
        raise ValueError(f"Unsupported control_mode: {self.control_mode}")

    def _validate_server_metadata(self) -> None:
        metadata = self.get_server_metadata()
        state_dim = metadata.get("state_dim")
        action_dim = metadata.get("action_dim")
        action_chunk_size = metadata.get("action_chunk_size")
        if state_dim is not None and int(state_dim) != self.spec.state_dim:
            raise ValueError(f"Motus server state_dim={state_dim}, local config state_dim={self.spec.state_dim}")
        if action_dim is not None and int(action_dim) != self.spec.action_dim:
            raise ValueError(f"Motus server action_dim={action_dim}, local config action_dim={self.spec.action_dim}")
        if action_chunk_size is not None and int(action_chunk_size) != self.spec.action_horizon:
            raise ValueError(
                f"Motus server action_chunk_size={action_chunk_size}, local config action_horizon={self.spec.action_horizon}"
            )

    @property
    def config_path(self) -> str:
        return self.spec.config_path

    def get_server_metadata(self) -> dict[str, Any]:
        return self._client.get_server_metadata()

    def set_default_session_id(self, session_id: str | None) -> None:
        self._default_session_id = session_id

    def build_payload(
        self,
        snapshot: RobotSnapshot,
        prompt: str | None = None,
        *,
        session_id: str | None = None,
        t5_embeds: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if session_id is None:
            session_id = self._default_session_id
        return build_policy_payload(
            snapshot,
            prompt=prompt,
            spec=self.spec,
            session_id=session_id,
            t5_embeds=t5_embeds,
            old_gripper=self.old_gripper,
        )

    def infer(
        self,
        snapshot: RobotSnapshot,
        prompt: str | None = None,
        *,
        session_id: str | None = None,
        t5_embeds: np.ndarray | None = None,
    ) -> dict[str, Any]:
        response = self._client.infer(
            self.build_payload(snapshot, prompt, session_id=session_id, t5_embeds=t5_embeds)
        )
        actions = _action_array_from_response(response)
        response = dict(response)
        response["normalized_actions"] = actions
        response["actions"] = _denormalize_actions(actions, self.spec.action_min, self.spec.action_max).astype(np.float64)
        return response

    def infer_actions(
        self,
        snapshot: RobotSnapshot,
        prompt: str | None = None,
        *,
        session_id: str | None = None,
        t5_embeds: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.asarray(
            self.infer(snapshot, prompt, session_id=session_id, t5_embeds=t5_embeds)["actions"],
            dtype=np.float64,
        )

    def get_predicted_video(self, session_id: str) -> dict[str, Any]:
        response = self._client.infer({"_request": "get_predicted_video", "session_id": session_id})
        return dict(response)

    def save_predicted_video(
        self,
        *,
        session_id: str,
        output_dir: str | Path,
        file_stem: str,
    ) -> Path | None:
        response = self.get_predicted_video(session_id)
        video_bytes = response.get("predicted_video_bytes")
        if video_bytes is None:
            return None
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{file_stem}_predicted_video.mp4"
        output_path.write_bytes(video_bytes)
        return output_path

    def decode_action(self, action: np.ndarray) -> DecodedPiperAction:
        slai_policy = _motus_slai_policy()
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1:
            raise ValueError(f"Expected one action vector, got shape {action.shape}")
        if action.shape[0] < self.spec.action_dim:
            raise ValueError(
                f"Action dim {action.shape[0]} is smaller than expected {self.spec.action_dim} "
                f"for {self.spec.config_path}"
            )

        action_space = slai_policy._space_from_action_config(self.spec.action_space)
        slices = slai_policy._field_slices_from_space(action_space)
        fields = set(slai_policy._fields_from_action_config(self.spec.action_space))
        decoded: dict[str, DecodedArmAction] = {}
        for arm in action_space["arms"]:
            gripper = _thresholded_gripper_for_piper(
                float(action[slices[f"{arm}_gripper"]][0]),
                self.gripper_threshold,
                self.spec.action_space.gripper,
                getattr(self, "gripper_lower", None),
                getattr(self, "gripper_upper", None),
                old_gripper=self.old_gripper,
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
        actions = _action_array_from_response(response_or_actions) if isinstance(response_or_actions, dict) else response_or_actions
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
