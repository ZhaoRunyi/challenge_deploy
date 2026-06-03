from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from typing import Any, Sequence

import numpy as np

from clients import slai_piper_policy
from clients.base import build_configured_piper_state
from clients.specs import decoded_action_summary
from hardware.config import set_by_dotted_path
from hardware.constants import DUAL_PIPER_INIT_JOINTS
from hardware.piper import DualPiperSystem
from hardware.realsense import RealSenseRig
from hardware.runtime import DualPiperObservationSource
from .recording import RecordingSchema


def normalized_prompt(value: str | None) -> str | None:
    if value is None:
        return None
    prompt = value.strip()
    return prompt or None


def apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.left_can:
        set_by_dotted_path(config, "robot.left.can_name", args.left_can)
    if args.right_can:
        set_by_dotted_path(config, "robot.right.can_name", args.right_can)
    if args.camera_high_serial:
        set_by_dotted_path(config, "cameras.serials.cam_high", args.camera_high_serial)
    if args.camera_right_serial:
        set_by_dotted_path(config, "cameras.serials.cam_right_wrist", args.camera_right_serial)
    if args.camera_left_serial:
        set_by_dotted_path(config, "cameras.serials.cam_left_wrist", args.camera_left_serial)
    if args.no_cameras:
        set_by_dotted_path(config, "cameras.enabled", False)
    return config


def make_dual_piper_runtime(config: dict[str, Any], *, commands_enabled: bool, name: str) -> tuple[Any, Any, Any]:
    robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=commands_enabled,
        name=name,
    )
    cameras = None
    if config["cameras"]["enabled"]:
        cameras = RealSenseRig(
            config["cameras"]["serials"],
            width=int(config["cameras"]["width"]),
            height=int(config["cameras"]["height"]),
            fps=int(config["cameras"]["fps"]),
            warmup_frames=int(config["cameras"]["warmup_frames"]),
        )
    return robot, cameras, DualPiperObservationSource(robot=robot, cameras=cameras)


def used_slai_action_names(spec: Any, control_mode: str) -> frozenset[str]:
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


def make_slai_recording_schema(spec: Any, control_mode: str) -> Any:
    return RecordingSchema(
        camera_names=spec.image_ids,
        action_names=tuple(slai_piper_policy.get_vector_names(spec.action_space)),
        state_names=tuple(slai_piper_policy.get_vector_names(spec.state_space)),
        used_action_names=used_slai_action_names(spec, control_mode),
    )


def build_slai_recording_state(
    snapshot: Any,
    spec: Any,
    *,
    old_gripper: bool = False,
    dtype: Any = np.float64,
) -> np.ndarray:
    return build_configured_piper_state(snapshot, spec, old_gripper=old_gripper, dtype=dtype)


def resolve_dual_piper_init_joints(values: Sequence[float] | None) -> np.ndarray:
    joints = np.asarray(DUAL_PIPER_INIT_JOINTS if values is None else values, dtype=np.float64)
    if joints.shape != (14,):
        raise ValueError("--init-joints expects exactly 14 values")
    return joints.copy()


def server_ckpt_dir(server_metadata: dict[str, Any] | None) -> str | None:
    for key in ("ckpt_dir", "checkpoint_dir", "model_path"):
        value = server_metadata.get(key) if server_metadata is not None else None
        if isinstance(value, str) and value.strip():
            return value
    return None


def record_name_prefix(args: argparse.Namespace, server_metadata: dict[str, Any] | None = None) -> str:
    ckpt_dir = server_ckpt_dir(server_metadata)
    ckpt_name = Path(ckpt_dir).name if ckpt_dir else Path(args.train_config).stem
    execution_mode = getattr(args, "execution_mode", "chunk_sync")
    return f"{ckpt_name}_{args.control_mode}_{execution_mode}"


def install_record_signal_handlers() -> None:
    def raise_keyboard_interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is None:
            continue
        try:
            signal.signal(signal_value, raise_keyboard_interrupt)
        except (OSError, ValueError):
            pass


def ignore_record_signal_handlers() -> None:
    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is None:
            continue
        try:
            signal.signal(signal_value, signal.SIG_IGN)
        except (OSError, ValueError):
            pass


def install_recorder_signal_handlers(recorder: Any | None) -> None:
    if recorder is not None:
        install_record_signal_handlers()


def ignore_recorder_signal_handlers(recorder: Any | None) -> None:
    if recorder is not None:
        ignore_record_signal_handlers()


def print_rollout_chunk_summary(
    *,
    client: Any,
    chunk_index: int,
    action_count: int,
    executed_steps: int,
    rollout_steps: int,
    first_action: np.ndarray,
) -> None:
    target = "unlimited" if rollout_steps == 0 else str(rollout_steps)
    print(
        json.dumps(
            {
                "rollout_chunk": chunk_index,
                "actions_in_chunk": action_count,
                "executed_steps": executed_steps,
                "target_steps": target,
                "first_action": decoded_action_summary(client.decode_action(first_action)),
            },
            indent=2,
        ),
        flush=True,
    )
