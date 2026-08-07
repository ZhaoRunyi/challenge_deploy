from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import time
from typing import Any, Literal, Mapping

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hardware.constants import PIPER_GRIPPER_FULL_OPEN_METERS
from hardware.conversions import (
    legacy_piper_raw_gripper_to_opening,
    normalized_gripper_to_opening,
    opening_to_legacy_piper_raw_gripper,
    opening_to_normalized_gripper,
)
from hardware.schemas import DualPiperState, PiperArmState, RobotSnapshot
from . import slai_piper_policy


ControlMode = Literal["joints", "ee_pose"]
StateGripperEncoding = Literal["policy", "meters", "old"]
ActionGripperEncoding = Literal["policy", "meters", "binary", "old"]
STATE_GRIPPER_ENCODINGS: tuple[StateGripperEncoding, ...] = ("policy", "meters", "old")
ACTION_GRIPPER_ENCODINGS: tuple[ActionGripperEncoding, ...] = ("policy", "meters", "binary", "old")


class PolicySessionCapability(str, Enum):
    """How a concrete policy payload handles rollout session identity."""

    SESSION_ID = "session_id"
    STATELESS = "stateless"


class PolicyResponseFormatError(ValueError):
    """A deterministic model-response schema error that must not be retried."""


@dataclass(frozen=True)
class DecodedArmAction:
    joint: np.ndarray | None
    gripper: float
    ee_pose: np.ndarray | None
    binary_gripper: bool = False


@dataclass(frozen=True)
class DecodedPiperAction:
    arms: dict[str, DecodedArmAction]
    control_mode: ControlMode


def rpy_to_rotation(rpy: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation_format = slai_piper_policy.resolve_rotation_format(rotation_format)
    rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
    if rotation_format == "rpy":
        return rpy
    rot = Rotation.from_euler("xyz", rpy, degrees=False)
    if rotation_format == "quat":
        return rot.as_quat().astype(np.float64)
    return rot.as_matrix()[:, :2].reshape(-1).astype(np.float64)


def rotation_to_rpy(values: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation_format = slai_piper_policy.resolve_rotation_format(rotation_format)
    values = np.asarray(values, dtype=np.float64)
    if rotation_format == "rpy":
        return values.reshape(3)
    rotation_cls = Rotation
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


def stabilize_rpy(rpy: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is None:
        return rpy
    return rpy + 2 * np.pi * np.round((previous - rpy) / (2 * np.pi))


def validate_state_gripper_encoding(value: str) -> StateGripperEncoding:
    if value not in STATE_GRIPPER_ENCODINGS:
        raise ValueError(f"Unsupported state_gripper_encoding {value!r}; expected one of {STATE_GRIPPER_ENCODINGS}")
    return value  # type: ignore[return-value]


def validate_action_gripper_encoding(value: str) -> ActionGripperEncoding:
    if value not in ACTION_GRIPPER_ENCODINGS:
        raise ValueError(f"Unsupported action_gripper_encoding {value!r}; expected one of {ACTION_GRIPPER_ENCODINGS}")
    return value  # type: ignore[return-value]


def hardware_gripper_to_model_raw(value: float, *, state_gripper_encoding: StateGripperEncoding = "policy") -> float:
    if state_gripper_encoding == "old":
        return opening_to_legacy_piper_raw_gripper(value)
    if state_gripper_encoding == "meters":
        return float(value)
    return opening_to_normalized_gripper(value)


def model_raw_gripper_to_hardware(
    value: float,
    *,
    action_gripper_encoding: ActionGripperEncoding = "policy",
    full_open_value: float = PIPER_GRIPPER_FULL_OPEN_METERS,
) -> float:
    if action_gripper_encoding == "old":
        return legacy_piper_raw_gripper_to_opening(value)
    if action_gripper_encoding == "meters":
        return max(0.0, float(value))
    if action_gripper_encoding == "binary":
        return full_open_value if float(value) >= 0.5 else 0.0
    return normalized_gripper_to_opening(value)


def state_gripper_for_policy(
    value: float,
    gripper_config: Any,
    *,
    state_gripper_encoding: StateGripperEncoding = "policy",
) -> float:
    value = float(value)
    state_gripper_encoding = validate_state_gripper_encoding(state_gripper_encoding)
    if state_gripper_encoding == "meters":
        return value
    if state_gripper_encoding == "old":
        return hardware_gripper_to_model_raw(value, state_gripper_encoding=state_gripper_encoding)
    if gripper_config is not None and gripper_config.type == "01":
        return value / gripper_config.full_width if gripper_config.full_width > 0 else value
    return hardware_gripper_to_model_raw(value, state_gripper_encoding=state_gripper_encoding)


def action_gripper_for_piper(
    value: float,
    gripper_config: Any,
    *,
    action_gripper_encoding: ActionGripperEncoding = "policy",
    full_open_value: float | None = None,
) -> float:
    value = float(value)
    action_gripper_encoding = validate_action_gripper_encoding(action_gripper_encoding)
    resolved_full_open = (
        PIPER_GRIPPER_FULL_OPEN_METERS
        if full_open_value is None
        else float(full_open_value)
    )
    if action_gripper_encoding in {"meters", "binary", "old"}:
        return model_raw_gripper_to_hardware(
            value,
            action_gripper_encoding=action_gripper_encoding,
            full_open_value=resolved_full_open,
        )
    if gripper_config is not None and gripper_config.type == "01":
        binary_full_open = (
            gripper_config.full_width
            if full_open_value is None
            else resolved_full_open
        )
        return binary_full_open if value >= 0.5 else 0.0
    return model_raw_gripper_to_hardware(
        value,
        action_gripper_encoding=action_gripper_encoding,
        full_open_value=resolved_full_open,
    )


def bounded_gripper_for_piper(
    value: float,
    threshold: float | None,
    lower: float | None = None,
    upper: float | None = None,
    *,
    full_open_value: float = PIPER_GRIPPER_FULL_OPEN_METERS,
) -> float:
    value = max(0.0, float(value))
    if threshold is not None:
        return full_open_value if value >= threshold else 0.0
    if upper is not None and value > upper:
        return full_open_value
    return 0.0 if lower is not None and value < lower else value


def arm_full_state(
    arm: PiperArmState,
    *,
    ee_rotation: str,
    gripper_config: Any,
    state_gripper_encoding: StateGripperEncoding = "policy",
) -> np.ndarray:
    return np.concatenate(
        (
            arm.qpos[:6],
            np.array(
                [
                    state_gripper_for_policy(
                        arm.qpos[6],
                        gripper_config,
                        state_gripper_encoding=state_gripper_encoding,
                    )
                ],
                dtype=np.float64,
            ),
            arm.end_pose[:3],
            rpy_to_rotation(arm.end_pose[3:6], ee_rotation),
        ),
        axis=0,
    ).astype(np.float64)


def build_full_piper_state(
    snapshot: RobotSnapshot,
    spec: Any,
    *,
    state_gripper_encoding: StateGripperEncoding = "policy",
) -> np.ndarray:
    return np.concatenate(
        (
            arm_full_state(
                snapshot.state.left,
                ee_rotation=spec.state_space.ee_rotation,
                gripper_config=spec.state_space.gripper,
                state_gripper_encoding=state_gripper_encoding,
            ),
            arm_full_state(
                snapshot.state.right,
                ee_rotation=spec.state_space.ee_rotation,
                gripper_config=spec.state_space.gripper,
                state_gripper_encoding=state_gripper_encoding,
            ),
        ),
        axis=0,
    )


def build_configured_piper_state(
    snapshot: RobotSnapshot,
    spec: Any,
    *,
    state_gripper_encoding: StateGripperEncoding = "policy",
    dtype: Any = np.float64,
) -> np.ndarray:
    full_state = build_full_piper_state(snapshot, spec, state_gripper_encoding=state_gripper_encoding)
    state_space = slai_piper_policy.space_from_state_config(spec.state_space)
    return np.asarray(slai_piper_policy.extract_vec(full_state, state_space, spec.state_space.gripper), dtype=dtype)


def image_to_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def action_array_from_response(
    response: Mapping[str, Any],
    keys: tuple[str, ...] = ("action", "actions"),
) -> np.ndarray:
    if not isinstance(response, Mapping):
        raise PolicyResponseFormatError(
            "Model response must be a mapping containing an action array, "
            f"got {type(response).__name__}"
        )
    for key in keys:
        if key in response:
            try:
                return np.asarray(response[key], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise PolicyResponseFormatError(
                    f"Model response field {key!r} is not a numeric action array: {exc}"
                ) from exc
    raise PolicyResponseFormatError(
        f"Model response does not contain any action key {keys}: {sorted(response)}"
    )


def used_action_names(spec: Any, control_mode: ControlMode) -> frozenset[str]:
    action_space = slai_piper_policy.space_from_action_config(spec.action_space)
    names = slai_piper_policy.get_vector_names(spec.action_space)
    slices = slai_piper_policy.field_slices_from_space(action_space)
    used_fields = {"gripper"}
    if control_mode == "joints":
        used_fields.add("joint")
    else:
        used_fields.update(("ee_pos", "ee_rot"))
    used: set[str] = set()
    for arm in action_space["arms"]:
        for field in used_fields:
            field_slice = slices.get(f"{arm}_{field}")
            if field_slice is None:
                continue
            used.update(names[index] for index in range(field_slice.start, field_slice.stop))
    return frozenset(used)


def quiet_close_policy_transport_on_construction_error(
    policy_client: Any,
) -> None:
    """Release a transport without replacing the constructor's root error."""

    close = getattr(policy_client, "close", None)
    if not callable(close):
        close = getattr(policy_client, "close_inference_session", None)
    if not callable(close):
        return
    try:
        close()
    except BaseException:
        pass


class SlaiPiperClient:
    SESSION_CAPABILITY = PolicySessionCapability.STATELESS

    def __init__(
        self,
        *,
        spec: Any,
        policy_client: Any,
        control_mode: ControlMode = "joints",
        joint_speed_percent: int = 50,
        ee_speed_percent: int = 50,
        gripper_threshold: float | None = None,
        gripper_lower: float | None = None,
        gripper_upper: float | None = None,
        state_gripper_encoding: StateGripperEncoding = "policy",
        action_gripper_encoding: ActionGripperEncoding = "policy",
        gripper_effort: int | None = None,
        gripper_action_frames: int = 1,
    ) -> None:
        if gripper_action_frames <= 0:
            raise ValueError("gripper_action_frames must be positive")
        self.spec = spec
        self.client = policy_client
        self.control_mode = control_mode
        self.joint_speed_percent = joint_speed_percent
        self.ee_speed_percent = ee_speed_percent
        self.gripper_threshold = gripper_threshold
        self.gripper_lower = gripper_lower
        self.gripper_upper = gripper_upper
        self.state_gripper_encoding = validate_state_gripper_encoding(state_gripper_encoding)
        self.action_gripper_encoding = validate_action_gripper_encoding(action_gripper_encoding)
        self.gripper_effort = gripper_effort
        self.gripper_action_frames = int(gripper_action_frames)
        self.default_session_id: str | None = None
        self.last_commanded: DecodedPiperAction | None = None
        self.gripper_transition: tuple[DecodedPiperAction, DecodedPiperAction, int] | None = None
        self.previous_ee_rpy: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._inference_timeout_s: float | None = None
        self.validate_control_mode()

    @property
    def train_config_name(self) -> str:
        return str(getattr(self.spec, "train_config_name", getattr(self.spec, "config_path", "")))

    def spec_label(self) -> str:
        return self.train_config_name or type(self.spec).__name__

    def validate_control_mode(self) -> None:
        fields = set(slai_piper_policy.fields_from_action_config(self.spec.action_space))
        if "gripper" not in fields:
            raise ValueError(f"{self.spec_label()}: deploy requires action_space to include gripper")
        if self.control_mode == "joints":
            if "joint" not in fields:
                raise ValueError(f"{self.spec_label()}: control_mode='joints' requires action_space to include joint")
            return
        if self.control_mode == "ee_pose":
            missing = {"ee_pos", "ee_rot"} - fields
            if missing:
                raise ValueError(f"{self.spec_label()}: control_mode='ee_pose' requires action_space fields {sorted(missing)}")
            return
        raise ValueError(f"Unsupported control_mode: {self.control_mode}")

    def get_server_metadata(self) -> Any:
        return self.client.get_server_metadata()

    def set_default_session_id(self, session_id: str | None) -> None:
        self.default_session_id = session_id

    @property
    def policy_session_capability(self) -> PolicySessionCapability:
        """Return the concrete client's explicit server-session contract."""

        return self.SESSION_CAPABILITY

    @property
    def supports_policy_sessions(self) -> bool:
        return self.policy_session_capability is PolicySessionCapability.SESSION_ID

    def resync_after_authority_change(
        self,
        *,
        session_id: str | None = None,
        reset_policy: bool = True,
    ) -> PolicySessionCapability:
        """Drop command-side continuity whenever control authority changes.

        The first subsequent command will seed binary-gripper interpolation from
        fresh robot feedback.  EE angle unwrapping likewise restarts from the new
        physical pose instead of an action issued before the role transaction.
        Stateful servers receive a new session id when they explicitly support
        one; other clients are accurately reported as stateless.
        """

        self.last_commanded = None
        self.gripper_transition = None
        self.previous_ee_rpy = {"left": None, "right": None}
        capability = self.policy_session_capability
        self.set_default_session_id(
            session_id if capability is PolicySessionCapability.SESSION_ID else None
        )
        if reset_policy:
            reset = getattr(self.client, "reset", None)
            if callable(reset):
                reset()
        return capability

    def fork_rollout_inference_session(self) -> "SlaiPiperClient":
        """Create an independent transport lane after a hard request timeout.

        A blocked websocket must never share its connection with a retry.  The
        transport therefore owns the actual reconnection factory while this
        base class makes a shallow inference-only client facade around it.
        """

        policy_client = self.client.new_inference_session()
        forked = copy.copy(self)
        forked.client = policy_client
        forked.last_commanded = None
        forked.gripper_transition = None
        forked.previous_ee_rpy = {"left": None, "right": None}
        return forked

    def configure_inference_timeout(self, timeout_s: float) -> None:
        if timeout_s <= 0.0:
            raise ValueError("inference timeout must be positive")
        self._inference_timeout_s = float(timeout_s)
        configure = getattr(self.client, "set_inference_timeout", None)
        if callable(configure):
            configure(self._inference_timeout_s)

    def close_inference_session(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

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

    def build_payload(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def infer(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        return self.client.infer(self.build_payload(snapshot, prompt=prompt, **kwargs))

    def infer_actions(self, snapshot: RobotSnapshot, prompt: str | None = None, **kwargs: Any) -> np.ndarray:
        return action_array_from_response(self.infer(snapshot, prompt=prompt, **kwargs))

    def decode_gripper_for_piper(
        self,
        value: float,
        arm_name: str,
        *,
        full_open_value: float | None = None,
    ) -> tuple[float, bool]:
        arm_threshold = getattr(self, f"{arm_name}_gripper_threshold", None)
        arm_lower = getattr(self, f"{arm_name}_gripper_lower", None)
        arm_upper = getattr(self, f"{arm_name}_gripper_upper", None)
        threshold = arm_threshold if arm_threshold is not None else self.gripper_threshold
        lower = arm_lower if arm_lower is not None else self.gripper_lower
        upper = arm_upper if arm_upper is not None else self.gripper_upper
        gripper_config = self.spec.action_space.gripper
        resolved_full_open = (
            PIPER_GRIPPER_FULL_OPEN_METERS
            if full_open_value is None
            else float(full_open_value)
        )
        raw_gripper = action_gripper_for_piper(
            value,
            gripper_config,
            action_gripper_encoding=self.action_gripper_encoding,
            full_open_value=full_open_value,
        )
        bounded_binary = (
            threshold is not None
            or (upper is not None and raw_gripper > upper)
            or (lower is not None and raw_gripper < lower)
        )
        gripper = bounded_gripper_for_piper(
            raw_gripper,
            threshold,
            lower,
            upper,
            full_open_value=resolved_full_open,
        )
        binary_gripper = self.action_gripper_encoding == "binary" or bounded_binary or bool(
            gripper_config is not None
            and getattr(gripper_config, "type", None) == "01"
            and self.action_gripper_encoding == "policy"
        )
        return gripper, binary_gripper

    def decode_action(
        self,
        action: np.ndarray,
        *,
        gripper_full_openings: Mapping[str, float] | None = None,
    ) -> DecodedPiperAction:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1:
            raise ValueError(f"Expected one action vector, got shape {action.shape}")
        if action.shape[0] < self.spec.action_dim:
            raise ValueError(f"Action dim {action.shape[0]} is smaller than expected {self.spec.action_dim} for {self.spec_label()}")
        action_space = slai_piper_policy.space_from_action_config(self.spec.action_space)
        slices = slai_piper_policy.field_slices_from_space(action_space)
        fields = set(slai_piper_policy.fields_from_action_config(self.spec.action_space))
        decoded: dict[str, DecodedArmAction] = {}
        for arm in action_space["arms"]:
            full_open_value = (
                None
                if gripper_full_openings is None
                else float(gripper_full_openings[arm])
            )
            gripper, binary_gripper = self.decode_gripper_for_piper(
                float(action[slices[f"{arm}_gripper"]][0]),
                arm,
                full_open_value=full_open_value,
            )
            joint = None
            ee_pose = None
            if "joint" in fields:
                joint = np.concatenate((action[slices[f"{arm}_joint"]], np.array([gripper])), axis=0)
            if {"ee_pos", "ee_rot"}.issubset(fields):
                ee_rpy = rotation_to_rpy(action[slices[f"{arm}_ee_rot"]], self.spec.action_space.ee_rotation)
                ee_pose = np.concatenate((action[slices[f"{arm}_ee_pos"]], ee_rpy, np.array([gripper])), axis=0)
            decoded[arm] = DecodedArmAction(joint=joint, gripper=gripper, ee_pose=ee_pose, binary_gripper=binary_gripper)
        return DecodedPiperAction(arms=decoded, control_mode=self.control_mode)

    def command_decoded(self, robot: Any, decoded: DecodedPiperAction) -> None:
        if decoded.control_mode == "joints":
            command_bimanual = getattr(robot, "command_bimanual_joint_positions", None)
            if callable(command_bimanual) and {"left", "right"}.issubset(decoded.arms):
                left_joint = decoded.arms["left"].joint
                right_joint = decoded.arms["right"].joint
                if left_joint is None or right_joint is None:
                    raise ValueError("Decoded bimanual joint action is missing a joint block")
                command_bimanual(
                    left_joint,
                    right_joint,
                    speed_percent=self.joint_speed_percent,
                    gripper_effort=self.gripper_effort,
                )
                return

        if decoded.control_mode == "ee_pose":
            command_bimanual = getattr(robot, "command_bimanual_end_poses", None)
            if callable(command_bimanual) and {"left", "right"}.issubset(decoded.arms):
                poses: dict[str, np.ndarray] = {}
                for arm_name in ("left", "right"):
                    arm_pose = decoded.arms[arm_name].ee_pose
                    if arm_pose is None:
                        raise ValueError(
                            f"Decoded bimanual EE action for {arm_name} has no ee_pose block"
                        )
                    pose = arm_pose.copy()
                    pose[3:6] = stabilize_rpy(
                        pose[3:6],
                        self.previous_ee_rpy[arm_name],
                    )
                    self.previous_ee_rpy[arm_name] = pose[3:6].copy()
                    poses[arm_name] = pose
                command_bimanual(
                    poses["left"],
                    poses["right"],
                    speed_percent=self.ee_speed_percent,
                    gripper_effort=self.gripper_effort,
                )
                return

        for arm_name, arm_action in decoded.arms.items():
            arm = robot.left if arm_name == "left" else robot.right
            if decoded.control_mode == "joints":
                if arm_action.joint is None:
                    raise ValueError(f"Decoded action for {arm_name} has no joint block")
                arm.command_joint_positions(
                    arm_action.joint,
                    speed_percent=self.joint_speed_percent,
                    gripper_effort=self.gripper_effort,
                )
            else:
                if arm_action.ee_pose is None:
                    raise ValueError(f"Decoded action for {arm_name} has no ee_pose block")
                pose = arm_action.ee_pose.copy()
                pose[3:6] = stabilize_rpy(pose[3:6], self.previous_ee_rpy[arm_name])
                self.previous_ee_rpy[arm_name] = pose[3:6].copy()
                arm.command_end_pose(
                    pose,
                    speed_percent=self.ee_speed_percent,
                    gripper_effort=self.gripper_effort,
                )

    def validate_decoded_action_for_robot(
        self,
        robot: Any,
        decoded: DecodedPiperAction,
    ) -> None:
        validator = getattr(robot, "validate_bimanual_targets", None)
        if not callable(validator):
            return
        if decoded.control_mode == "joints":
            left = decoded.arms["left"].joint
            right = decoded.arms["right"].joint
            joint_positions = True
        else:
            left = decoded.arms["left"].ee_pose
            right = decoded.arms["right"].ee_pose
            joint_positions = False
        if left is None or right is None:
            raise ValueError("decoded bimanual action is missing its command target")
        validator(
            left,
            right,
            joint_positions=joint_positions,
        )

    def current_decoded_from_robot(self, robot: Any) -> DecodedPiperAction:
        state = robot.read_state()
        if self.control_mode == "ee_pose":
            self.previous_ee_rpy = {
                "left": np.asarray(state.left.end_pose[3:6], dtype=np.float64).copy(),
                "right": np.asarray(state.right.end_pose[3:6], dtype=np.float64).copy(),
            }
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

    def action_state_after_command(
        self,
        robot: Any,
        snapshot_before_command: RobotSnapshot,
    ) -> DualPiperState:
        """Return the canonical target that was actually submitted this step.

        Joint commands are built from the exact validated targets because a
        SocketCAN sender may not receive its own 0x155-0x157 frames and a shared
        bus may cache the physical master's control family instead.
        Cartesian commands have no equivalent joint target; their canonical
        state keeps fresh slave joints and replaces EE pose/gripper with the
        exact command-side values, including a binary-gripper ramp.
        """

        if self.control_mode == "joints":
            target_builder = getattr(robot, "action_state_for_joint_targets", None)
            if callable(target_builder):
                if self.last_commanded is None:
                    raise RuntimeError("no joint command is available for canonical action state")
                left_target = self.last_commanded.arms["left"].joint
                right_target = self.last_commanded.arms["right"].joint
                if left_target is None or right_target is None:
                    raise RuntimeError("last bimanual joint command is incomplete")
                feedback = getattr(snapshot_before_command, "state", None)
                if not isinstance(feedback, DualPiperState):
                    feedback = robot.read_state(prefer_joint_ctrl=False)
                return target_builder(left_target, right_target, feedback)
            state = robot.read_state(prefer_joint_ctrl=True)
            if not isinstance(state, DualPiperState):
                raise TypeError("joint action_state_after_command expects DualPiperState")
            return state
        if self.last_commanded is None:
            raise RuntimeError("no EE command is available for canonical action state")

        fresh = getattr(snapshot_before_command, "state", None)
        if not isinstance(fresh, DualPiperState):
            fresh = robot.read_state(prefer_joint_ctrl=False)
        if not isinstance(fresh, DualPiperState):
            raise TypeError("EE action_state_after_command expects DualPiperState")
        command_timestamp_s = time.time()

        def commanded_pose(arm_name: str) -> np.ndarray:
            arm_action = self.last_commanded.arms[arm_name]
            if arm_action.ee_pose is None:
                raise RuntimeError(f"last EE command for {arm_name} has no end pose")
            pose = np.asarray(arm_action.ee_pose, dtype=np.float64).copy()
            stabilized_rpy = self.previous_ee_rpy.get(arm_name)
            if stabilized_rpy is not None:
                pose[3:6] = np.asarray(stabilized_rpy, dtype=np.float64)
            pose[6] = float(arm_action.gripper)
            return pose

        left_pose = commanded_pose("left")
        right_pose = commanded_pose("right")
        isolated_builder = getattr(robot, "action_state_for_end_pose_targets", None)
        if callable(isolated_builder):
            return isolated_builder(left_pose, right_pose, fresh)

        def commanded_arm(
            arm_name: str,
            arm_state: PiperArmState,
            pose: np.ndarray,
        ) -> PiperArmState:
            arm_action = self.last_commanded.arms[arm_name]
            qpos = np.asarray(arm_state.qpos, dtype=np.float64).copy()
            qpos[6] = float(arm_action.gripper)
            qpos_command = np.asarray(arm_state.qpos_command, dtype=np.float64).copy()
            qpos_command[6] = float(arm_action.gripper)
            return replace(
                arm_state,
                qpos=qpos,
                qpos_command=qpos_command,
                end_pose=pose,
                timestamp_s=max(float(arm_state.timestamp_s), command_timestamp_s),
                end_pose_timestamp_s=command_timestamp_s,
                command_timestamp_s=command_timestamp_s,
            )

        return DualPiperState(
            left=commanded_arm("left", fresh.left, left_pose),
            right=commanded_arm("right", fresh.right, right_pose),
        )

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
        action = np.asarray(action, dtype=np.float64)
        full_opening_getter = getattr(robot, "gripper_full_opening_m", None)
        gripper_full_openings = None
        if callable(full_opening_getter):
            gripper_full_openings = {
                side: float(full_opening_getter(side))
                for side in ("left", "right")
            }
        decoded = self.decode_action(
            action,
            gripper_full_openings=gripper_full_openings,
        )
        self.validate_decoded_action_for_robot(robot, decoded)
        if self.gripper_transition is not None:
            start, target, step = self.gripper_transition
            self.command_transition_step(robot, start, target, step)
            self.gripper_transition = None if step >= self.gripper_action_frames else (start, target, step + 1)
            return
        if self.last_commanded is None:
            self.last_commanded = self.current_decoded_from_robot(robot)
        if (
            self.last_commanded is not None
            and self.gripper_action_frames > 1
            and any(
                arm.binary_gripper and not np.isclose(arm.gripper, self.last_commanded.arms[name].gripper, atol=1e-6)
                for name, arm in decoded.arms.items()
            )
        ):
            start = self.last_commanded
            self.command_transition_step(robot, start, decoded, 1)
            self.gripper_transition = (start, decoded, 2)
            return
        self.command_decoded(robot, decoded)
        self.last_commanded = decoded

    def command_first_action(self, robot: Any, response_or_actions: dict[str, Any] | np.ndarray) -> None:
        actions = action_array_from_response(response_or_actions) if isinstance(response_or_actions, dict) else response_or_actions
        actions = np.asarray(actions, dtype=np.float64)
        self.command_action(robot, actions[0] if actions.ndim == 2 else actions)
