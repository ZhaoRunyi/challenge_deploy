from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from typing import Any

import numpy as np

from clients import slai_piper_policy
from clients.base import ControlMode, decoded_action_summary, used_action_names
from hardware.config import set_by_dotted_path
from hardware import DualPiperObservationSource, DualPiperSystem, RealSenseRig
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
    if args.camera_front_serial:
        set_by_dotted_path(config, "cameras.serials.cam_high", args.camera_front_serial)
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


def make_slai_recording_schema(spec: Any, control_mode: ControlMode) -> RecordingSchema:
    return RecordingSchema(
        camera_names=spec.image_ids,
        action_names=tuple(slai_piper_policy.get_vector_names(spec.action_space)),
        state_names=tuple(slai_piper_policy.get_vector_names(spec.state_space)),
        used_action_names=used_action_names(spec, control_mode),
    )


def record_name_prefix(args: argparse.Namespace) -> str:
    ckpt_name = Path(args.ckpt_dir).name if args.ckpt_dir else Path(args.train_config).stem
    return f"{ckpt_name}_{args.control_mode}_{args.execution_mode}"


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
