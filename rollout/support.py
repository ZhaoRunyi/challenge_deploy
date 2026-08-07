from __future__ import annotations

import argparse
from importlib.resources import files
import json
import math
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np

from clients import slai_piper_policy
from clients.base import (
    ACTION_GRIPPER_ENCODINGS,
    STATE_GRIPPER_ENCODINGS,
    build_configured_piper_state,
    used_action_names,
)
from clients.specs import decoded_action_summary
from hardware.config import (
    set_by_dotted_path,
    validate_config,
)
from hardware.constants import (
    PIPER_ARM_IDS,
)
from hardware.factory import build_hardware
from hardware.realsense import RealSenseRig
from hardware.runtime import DualPiperObservationSource
from hardware.schemas import RobotSnapshot
from .coordinator import DynamicRolloutCoordinator
from .dry_run import (
    build_dry_run_hardware_plan,
    validated_dry_run_parameters,
    validated_initial_joints,
)
from .execution import resolve_record_steps
from .events import RuntimeEventLog
from .hardware_control import make_authority_hardware_controller
from .hdf5 import RolloutHDF5SessionCollector, RolloutHDF5WriterPool
from .interactive import InteractiveSessionResult, run_interactive_session
from .recording import RecordingSchema, set_distribution_overlap
from teleop.hdf5_teleop import infer_language_instruction, next_episode_index


DRY_RUN_HELP = (
    "Validate arguments, policy schema, runtime config, topology selection, and runner wiring; "
    "print one JSON plan without contacting a policy server or constructing hardware, cameras, "
    "recorders, windows, or HDF5 outputs."
)


def default_runtime_config_path() -> str:
    return str(files("configs").joinpath("dual_piper_example.yaml"))


def default_artifact_directory(name: str) -> str:
    if not name or Path(name).name != name:
        raise ValueError("artifact directory name must be one non-empty path component")
    return str(Path.cwd() / "artifacts" / name)


def normalized_prompt(value: str | None) -> str | None:
    if value is None:
        return None
    prompt = value.strip()
    return prompt or None


def validate_standard_rollout_args(args: argparse.Namespace) -> None:
    """Validate CLI contracts shared by every rollout entrypoint."""

    if args.save_sep and not args.record:
        raise ValueError("--save-sep requires --record")
    if args.record_steps is not None and not (args.record or args.window):
        raise ValueError("--record-steps requires --record or --window")
    if args.rollout_steps < 0:
        raise ValueError("--rollout-steps must be non-negative")
    resolve_record_steps(args.rollout_steps, args.record_steps)
    if args.fps < 0.0:
        raise ValueError("--fps must be non-negative")
    if args.inference_rate is not None and args.inference_rate < 0.0:
        raise ValueError("--inference-rate must be non-negative")
    if args.gripper_threshold is not None and args.gripper_threshold < 0.0:
        raise ValueError("--gripper_threshold must be non-negative")
    if args.gripper_threshold is not None and (
        args.gripper_lower is not None or args.gripper_upper is not None
    ):
        raise ValueError(
            "--gripper_threshold cannot be combined with "
            "--gripper_lower/--gripper_upper"
        )


def add_gripper_encoding_args(
    parser: argparse.ArgumentParser,
    *,
    default_state: str = "policy",
    default_action: str = "policy",
) -> None:
    parser.add_argument(
        "--state-gripper",
        choices=STATE_GRIPPER_ENCODINGS,
        default=default_state,
        help="State gripper encoding passed to the policy: policy, meters, or old.",
    )
    parser.add_argument(
        "--action-gripper",
        choices=ACTION_GRIPPER_ENCODINGS,
        default=default_action,
        help="Action gripper encoding returned by the policy: policy, meters, binary, or old.",
    )


def add_websocket_policy_args(
    parser: argparse.ArgumentParser,
    *,
    control_modes: tuple[str, ...] = ("joints", "ee_pose"),
    default_control_mode: str = "joints",
    default_speed_percent: int = 50,
) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--control-mode",
        choices=control_modes,
        default=default_control_mode,
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--joint-speed-percent",
        type=int,
        default=default_speed_percent,
    )
    parser.add_argument(
        "--ee-speed-percent",
        type=int,
        default=default_speed_percent,
    )


def add_gripper_bound_args(
    parser: argparse.ArgumentParser,
    *,
    per_arm: bool = True,
    threshold_help: str | None = None,
) -> None:
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=None,
        help=threshold_help,
    )
    if per_arm:
        for side in ("left", "right"):
            aliases = [f"--{side}_gripper_thrshold"] if side == "left" else []
            parser.add_argument(
                f"--{side}_gripper_threshold",
                *aliases,
                dest=f"{side}_gripper_threshold",
                type=float,
                default=None,
            )
            parser.add_argument(f"--{side}_gripper_lower", type=float, default=None)
            parser.add_argument(f"--{side}_gripper_upper", type=float, default=None)
    parser.add_argument("--gripper_lower", type=float, default=None)
    parser.add_argument("--gripper_upper", type=float, default=None)


def add_rollout_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Add topology-aware session flags shared by every real rollout entry."""
    parser.add_argument(
        "--intervention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable isolated four-arm ROLLOUT/INTERVENE switching.",
    )
    parser.add_argument("--save", action="store_true", help="Save stable episode steps as training HDF5.")
    parser.add_argument("--dataset-dir", default=None, help="Override the HDF5 dataset root from config.")
    parser.add_argument("--task-name", default=None, help="Override the HDF5 task directory name.")
    parser.add_argument("--language-instruction", default=None, help="Override the HDF5 episode instruction.")
    parser.add_argument("--master-left-can", default=None)
    parser.add_argument("--master-right-can", default=None)
    parser.add_argument("--event-log", default=None, help="Optional runtime JSONL event log path.")
    parser.add_argument(
        "--inference-timeout",
        type=float,
        default=15.0,
        help="Seconds before an asynchronous inference request is declared stale.",
    )


def make_rollout_argument_parser(
    client_name: str,
    *,
    allow_abbrev: bool = True,
) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            f"{client_name} interactive rollout client. Isolated runs use two slaves by default; "
            "--intervention enables four-arm ROLLOUT/INTERVENE switching."
        ),
        epilog=(
            "Keys: idle c=start, q=quit, k=kill oldest HDF5 writer; active s=stop, and with "
            "--intervention i=INTERVENE, r=ROLLOUT; at the step limit x=extend, s=stop; "
            "after a normal --save episode c=save, d=discard. At most two HDF5 writers run."
        ),
        allow_abbrev=allow_abbrev,
    )


def add_standard_rollout_args(
    parser: argparse.ArgumentParser,
    *,
    record_directory_name: str,
) -> None:
    """Add execution, output, topology, and observation flags shared by all runners."""

    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1000,
        help=(
            "Stable command ticks allowed per episode; 0 is unlimited. At the limit press x to add "
            "ceil(initial/2), or s to stop."
        ),
    )
    parser.add_argument(
        "--record-steps",
        type=int,
        default=None,
        help=(
            "Diagnostic capture ticks per episode; default follows --rollout-steps. This does not "
            "limit control or --save HDF5, and x does not extend it. Requires --record or --window."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Actions to execute from each policy chunk; default is train config action_horizon.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Action command frequency in Hz; 0 sends the chunk as fast as possible.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["streaming", "chunk_sync"],
        default="chunk_sync",
        help=(
            "Both modes infer asynchronously: streaming requests by cadence and smooths overlapping "
            "chunks; chunk_sync requests after the action buffer empties and does not cross-chunk smooth."
        ),
    )
    parser.add_argument(
        "--inference-rate",
        type=float,
        default=None,
        help="Streaming policy request frequency in Hz; default from config.",
    )
    parser.add_argument(
        "--latency-k",
        type=int,
        default=None,
        help="Max prefix actions to trim from a fresh chunk; default from config.",
    )
    parser.add_argument(
        "--min-smooth-steps",
        type=int,
        default=None,
        help="Minimum old-tail length for overlap smoothing; default from config.",
    )
    parser.add_argument(
        "--buffer-max-chunks",
        type=int,
        default=None,
        help="Action buffer chunk cap; default from config.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path to save rollout timing metrics as JSON.",
    )
    parser.add_argument(
        "--record",
        "--recording",
        action="store_true",
        help="Capture diagnostic video, action/state NPZ, frame1, and metrics; independent of --save HDF5.",
    )
    parser.add_argument(
        "--save-sep",
        action="store_true",
        help="Save one raw-camera video per recording camera; requires --record.",
    )
    parser.add_argument(
        "--record-dir",
        default=default_artifact_directory(record_directory_name),
        help="Diagnostic output directory used by --record.",
    )
    parser.add_argument(
        "--config",
        default=default_runtime_config_path(),
        help="Topology/runtime YAML. The default isolated file is provisional; pass a reviewed config for real hardware.",
    )
    add_rollout_runtime_args(parser)
    parser.add_argument("--left-can", default=None, help="Override robot.slave_left.can_name.")
    parser.add_argument("--right-can", default=None, help="Override robot.slave_right.can_name.")
    parser.add_argument(
        "--init-joints",
        nargs=14,
        type=float,
        default=None,
        help="Optional 14D dual-Piper initial qpos override: left 7 then right 7.",
    )
    parser.add_argument("--camera-high-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Do not construct cameras; incompatible with --save.",
    )
    parser.add_argument("--window", nargs="?", const=1, type=int, default=0)
    parser.add_argument(
        "--dist-overlap",
        action="store_true",
        help="Overlay train distribution on cam_high instead of stacking it above.",
    )
    parser.add_argument("--dry-run", action="store_true", help=DRY_RUN_HELP)
    parser.add_argument(
        "--spec-only",
        action="store_true",
        help="Only print the train-config-derived spaces; no server or hardware.",
    )
    parser.add_argument("--ready-timeout", type=float, default=15.0)


def apply_arm_gripper_overrides(client: Any, args: argparse.Namespace) -> None:
    for side in ("left", "right"):
        for field in ("threshold", "lower", "upper"):
            attr = f"{side}_gripper_{field}"
            if hasattr(args, attr):
                setattr(client, attr, getattr(args, attr))


def apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "left_can", None):
        set_by_dotted_path(config, "robot.slave_left.can_name", args.left_can)
    if getattr(args, "right_can", None):
        set_by_dotted_path(config, "robot.slave_right.can_name", args.right_can)
    if getattr(args, "master_left_can", None):
        set_by_dotted_path(config, "robot.master_left.can_name", args.master_left_can)
    if getattr(args, "master_right_can", None):
        set_by_dotted_path(config, "robot.master_right.can_name", args.master_right_can)
    if getattr(args, "camera_high_serial", None):
        set_by_dotted_path(config, "cameras.serials.cam_high", args.camera_high_serial)
    if getattr(args, "camera_right_serial", None):
        set_by_dotted_path(config, "cameras.serials.cam_right_wrist", args.camera_right_serial)
    if getattr(args, "camera_left_serial", None):
        set_by_dotted_path(config, "cameras.serials.cam_left_wrist", args.camera_left_serial)
    for camera_name, serial in zip(
        getattr(args, "extra_camera_names", ()),
        getattr(args, "extra_camera_serials", ()),
    ):
        config["cameras"]["serials"][camera_name] = serial
    if getattr(args, "no_cameras", False):
        set_by_dotted_path(config, "cameras.enabled", False)
    set_by_dotted_path(config, "runtime.intervention_enabled", bool(getattr(args, "intervention", False)))
    if bool(getattr(args, "save", False)) and not bool(config["cameras"]["enabled"]):
        raise ValueError("--save requires configured cameras; remove --no-cameras or disable --save")
    return config


def _validated_policy_schema(spec: Any, control_mode: str) -> dict[str, Any]:
    required_fields = ("state_dim", "action_dim", "image_ids")
    missing_fields = [field for field in required_fields if not hasattr(spec, field)]
    if missing_fields:
        raise ValueError(f"policy spec is missing required fields: {missing_fields}")

    state_dim = getattr(spec, "state_dim")
    action_dim = getattr(spec, "action_dim")
    if isinstance(state_dim, bool) or not isinstance(state_dim, int) or state_dim <= 0:
        raise ValueError(f"policy spec state_dim must be a positive integer, got {state_dim!r}")
    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
        raise ValueError(f"policy spec action_dim must be a positive integer, got {action_dim!r}")

    model_action_dim = getattr(spec, "model_action_dim", None)
    if model_action_dim is not None and (
        isinstance(model_action_dim, bool)
        or not isinstance(model_action_dim, int)
        or model_action_dim <= 0
    ):
        raise ValueError(
            "policy spec model_action_dim must be a positive integer or null, "
            f"got {model_action_dim!r}"
        )
    action_horizon = getattr(spec, "action_horizon", None)
    if action_horizon is not None and (
        isinstance(action_horizon, bool)
        or not isinstance(action_horizon, int)
        or action_horizon <= 0
    ):
        raise ValueError(
            "policy spec action_horizon must be a positive integer or null, "
            f"got {action_horizon!r}"
        )

    image_ids = tuple(getattr(spec, "image_ids"))
    if not image_ids or any(not isinstance(image_id, str) or not image_id for image_id in image_ids):
        raise ValueError("policy spec image_ids must contain non-empty strings")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"policy spec image_ids contains duplicates: {image_ids}")
    image_key_map = getattr(spec, "image_key_map", None)
    if image_key_map is not None and set(image_key_map) != set(image_ids):
        raise ValueError(
            "policy spec image_key_map keys must match image_ids; "
            f"keys={sorted(image_key_map)}, image_ids={sorted(image_ids)}"
        )

    state_space = getattr(spec, "state_space", None)
    action_space = getattr(spec, "action_space", None)
    if (state_space is None) != (action_space is None):
        raise ValueError("policy spec must define both state_space and action_space, or neither")
    if action_space is None:
        if control_mode != "joints":
            raise ValueError("a fixed-layout policy spec only supports control_mode='joints'")
        schema_kind = "fixed-layout"
    else:
        state_names = tuple(slai_piper_policy.get_vector_names(state_space))
        action_names = tuple(slai_piper_policy.get_vector_names(action_space))
        if len(state_names) != state_dim:
            raise ValueError(
                f"policy state schema has {len(state_names)} names but state_dim is {state_dim}"
            )
        if len(action_names) != action_dim:
            raise ValueError(
                f"policy action schema has {len(action_names)} names but action_dim is {action_dim}"
            )
        action_arms = tuple(
            slai_piper_policy.space_from_action_config(action_space)["arms"]
        )
        if action_arms != ("left", "right"):
            raise ValueError(
                "Piper rollout requires a dual-arm action schema with left and right; "
                f"resolved action arms are {action_arms}"
            )
        action_fields = set(slai_piper_policy.fields_from_action_config(action_space))
        if "gripper" not in action_fields:
            raise ValueError("policy action schema must include gripper")
        if control_mode == "joints" and "joint" not in action_fields:
            raise ValueError("control_mode='joints' requires joint actions in the policy schema")
        if control_mode == "ee_pose":
            missing_action_fields = {"ee_pos", "ee_rot"} - action_fields
            if missing_action_fields:
                raise ValueError(
                    "control_mode='ee_pose' requires policy action fields "
                    f"{sorted(missing_action_fields)}"
                )
        elif control_mode != "joints":
            raise ValueError(f"unsupported control_mode: {control_mode!r}")
        schema_kind = "configured-space"

    return {
        "schema_kind": schema_kind,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "model_action_dim": model_action_dim,
        "action_horizon": action_horizon,
        "image_ids": list(image_ids),
        "action_arms": ["left", "right"],
    }


def run_rollout_dry_run_plan(
    *,
    args: argparse.Namespace,
    runner_name: str,
    policy_transport_name: str,
    spec: Any,
    policy_spec_summary: dict[str, Any],
    runtime_config: dict[str, Any],
) -> bool:
    """Print the common zero-I/O rollout plan and report whether it handled the run."""

    if not getattr(args, "dry_run", False):
        return False

    hardware_plan = build_dry_run_hardware_plan(args, runtime_config)
    policy_schema = _validated_policy_schema(spec, str(args.control_mode))
    parameters = validated_dry_run_parameters(args, runtime_config, policy_schema)

    plan = {
        "dry_run_plan": {
            "result": "validation passed",
            "runner": {
                "entrypoint": runner_name,
                "policy_transport": policy_transport_name,
                "control_mode": args.control_mode,
            },
            "execution": {
                "passes": 1,
                "interactive": False,
                "requires_tty": False,
                "execution_mode": args.execution_mode,
                **parameters,
            },
            "policy": {
                "schema": policy_schema,
                "spec": policy_spec_summary,
                "prompt_supplied_by_cli": normalized_prompt(getattr(args, "prompt", None))
                is not None,
                "prompt_resolution": "deferred without server or dataset asset lookup",
                "operations": {
                    "transport_construction": "skipped",
                    "server_metadata": "skipped",
                    "server_preflight": "skipped",
                    "inference": "skipped",
                },
            },
            "hardware": hardware_plan,
            "outputs": {
                "recording_requested": bool(args.record),
                "window_requested": bool(args.window),
                "hdf5_requested": bool(args.save),
                "recorder_construction": "skipped",
                "window_construction": "skipped",
                "hdf5_creation": "skipped",
            },
            "validated": [
                "arguments",
                "policy spec",
                "policy schema",
                "runtime config",
                "topology selection",
                "runner wiring",
            ],
        }
    }
    print(json.dumps(plan, indent=2, allow_nan=False), flush=True)
    return True


def validate_rollout_runtime_preflight(
    *,
    args: argparse.Namespace,
    spec: Any,
    runtime_config: dict[str, Any],
) -> np.ndarray:
    """Run every pure real-run gate before policy, camera, or CAN construction."""

    intervention = bool(args.intervention)
    required_arm_ids = (
        tuple(PIPER_ARM_IDS)
        if intervention
        else ("slave_left", "slave_right")
    )
    validate_config(runtime_config, required_arm_ids=required_arm_ids)
    topology = str(runtime_config["can_topology"])
    if intervention and topology != "isolated":
        raise ValueError("--intervention requires can_topology='isolated'")
    if bool(
        runtime_config.get("runtime", {}).get("intervention_enabled", False)
    ) != intervention:
        raise ValueError(
            "runner wiring mismatch: runtime.intervention_enabled does not match --intervention"
        )
    policy_schema = _validated_policy_schema(spec, str(args.control_mode))
    validated_dry_run_parameters(args, runtime_config, policy_schema)
    return validated_initial_joints(getattr(args, "init_joints", None))


def make_dual_piper_runtime(
    config: dict[str, Any],
    *,
    name: str,
) -> tuple[Any, Any, Any]:
    intervention = bool(config.get("runtime", {}).get("intervention_enabled", False))
    assembly = build_hardware(
        config,
        intervention=intervention,
        commands_enabled=True,
        name=name,
    )
    robot = assembly.robot
    robot.hardware_assembly = assembly
    try:
        cameras = None
        if config["cameras"]["enabled"]:
            cameras = RealSenseRig(
                config["cameras"]["serials"],
                width=int(config["cameras"]["width"]),
                height=int(config["cameras"]["height"]),
                fps=int(config["cameras"]["fps"]),
                warmup_frames=int(config["cameras"]["warmup_frames"]),
            )
        if assembly.topology == "isolated":
            max_frame_age_s = float(robot.motion_watchdog_max_state_age_s)
            source = DualPiperObservationSource(
                robot=robot,
                cameras=cameras,
                camera_timeout_ms=max(1, math.ceil(max_frame_age_s * 1000.0)),
                parallel_camera_capture=True,
            )
        else:
            source = DualPiperObservationSource(robot=robot, cameras=cameras)
    except BaseException:
        if cameras is not None:
            try:
                cameras.stop()
            except Exception:
                pass
        gateway = getattr(assembly, "gateway", None)
        close_gateway = getattr(gateway, "close", None)
        if callable(close_gateway):
            try:
                close_gateway()
            except Exception:
                pass
        abort_robot = getattr(robot, "abort_construction", None)
        if callable(abort_robot):
            try:
                abort_robot()
            except Exception:
                pass
        raise
    return robot, cameras, source


def best_effort_hold_configured_robot(robot: Any) -> None:
    """Issue a reset-free current-position hold after pre-session failures."""

    try:
        controller = make_authority_hardware_controller(robot)
        controller.hold_position()
    except Exception as exc:
        print(f"Failed to submit a best-effort Piper hold: {exc}", flush=True)


def close_policy_transport(client: Any) -> None:
    """Close the runner-owned primary transport after postprocessing."""

    close = getattr(client, "close_inference_session", None)
    if not callable(close):
        close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        print(f"Failed to close policy transport cleanly: {exc}", flush=True)


def close_hardware_gateway(robot: Any) -> None:
    """Close the runner-owned isolated semantic gateway, when present."""

    if robot is None:
        return
    assembly = getattr(robot, "hardware_assembly", None)
    gateway = getattr(assembly, "gateway", None)
    close = getattr(gateway, "close", None)
    if callable(close):
        close()


def make_runtime_event_callback(event_log_path: str | Path | None = None) -> Any:
    """Create one best-effort JSONL sink before a real runner touches hardware."""

    try:
        path = (
            Path(event_log_path).expanduser()
            if event_log_path
            else Path.cwd()
            / "artifacts"
            / "runtime_events"
            / f"session_{time.time_ns()}.jsonl"
        )
        event_log = RuntimeEventLog(path)
    except Exception as exc:
        print(f"Runtime event log disabled: {exc}", flush=True)
        return lambda event, **fields: None

    error_reported = False

    def append_runtime_event(event: str, *, durable: bool = False, **fields: Any) -> None:
        nonlocal error_reported
        try:
            event_log.append(event, durable=durable, **fields)
        except Exception as exc:
            if not error_reported:
                print(f"Runtime event log disabled after an I/O failure: {exc}", flush=True)
                error_reported = True

    return append_runtime_event


def prepare_rollout_runtime(
    *,
    args: argparse.Namespace,
    spec: Any,
    runtime_config: dict[str, Any],
    runner_name: str,
) -> tuple[np.ndarray, Any]:
    initial_joints = validate_rollout_runtime_preflight(
        args=args,
        spec=spec,
        runtime_config=runtime_config,
    )
    runtime_event_callback = make_runtime_event_callback(args.event_log)
    runtime_event_callback(
        "runtime_preflight_passed",
        runner=runner_name,
        topology=runtime_config["can_topology"],
        durable=True,
    )
    set_distribution_overlap(args.dist_overlap)
    return initial_joints, runtime_event_callback


def print_server_metadata(server_metadata: dict[str, Any]) -> None:
    print(json.dumps({"server_metadata": server_metadata}, indent=2), flush=True)


def print_resolved_prompt(prompt: str, source: str) -> None:
    print(
        json.dumps(
            {
                "prompt": {
                    "value": prompt,
                    "source": source,
                }
            },
            indent=2,
        ),
        flush=True,
    )


def run_interactive_configured_rollout(
    *,
    args: argparse.Namespace,
    client: Any,
    source: Any,
    robot: Any,
    spec: Any,
    prompt: str,
    runtime_config: dict[str, Any],
    chunk_size: int | None,
    inference_rate: float,
    latency_k: int,
    min_smooth_steps: int,
    buffer_max_chunks: int,
    initial_joints: np.ndarray,
    initial_speed_percent: int,
    initial_gripper_effort: int | None = None,
    record_sink: Any | None = None,
    state_builder: Any | None = None,
    log_chunk: Any | None = None,
    key_reader: Any | None = None,
    require_tty: bool = True,
    runtime_event_callback: Any | None = None,
) -> InteractiveSessionResult:
    """Common six-client session path; resources are owned by the caller."""

    controller = make_authority_hardware_controller(robot)
    dataset_root: Path | None = None
    hdf5_collector: RolloutHDF5SessionCollector | None = None
    writer_pool: RolloutHDF5WriterPool | None = None
    if args.save:
        dataset_dir = Path(
            getattr(args, "dataset_dir", None)
            or runtime_config["dataset"]["dataset_dir"]
        ).expanduser().resolve()
        task_name = str(
            getattr(args, "task_name", None)
            or runtime_config["dataset"]["dataset_name"]
        )
        dataset_root = dataset_dir / task_name
        language_instruction = infer_language_instruction(
            task_name,
            getattr(args, "language_instruction", None) or prompt,
        )
        camera_names = tuple(
            name
            for name, serial in runtime_config["cameras"]["serials"].items()
            if serial
        )
        hdf5_collector = RolloutHDF5SessionCollector(
            dataset_root=dataset_root,
            language_instruction=language_instruction,
            camera_names=camera_names,
            control_mode=args.control_mode,
            can_topology=runtime_config["can_topology"],
            first_episode_index=next_episode_index(dataset_root),
        )
        writer_pool = RolloutHDF5WriterPool(max_writers=2)

    append_runtime_event = runtime_event_callback or make_runtime_event_callback(
        getattr(args, "event_log", None)
    )

    startup_operations = getattr(robot, "last_enable_operations", ())
    if startup_operations:
        append_runtime_event(
            "pre_session_hardware_enable",
            topology=runtime_config["can_topology"],
            operations=startup_operations,
            durable=True,
        )

    diagnostic_steps = 0
    diagnostics_active = record_sink is not None
    rollout_session_ids: list[str] = []
    diagnostic_limit = (
        args.rollout_steps
        if getattr(args, "record_steps", None) is None
        else int(args.record_steps)
    )

    def on_episode_start(episode_number: int) -> None:
        nonlocal diagnostic_steps
        diagnostic_steps = 0
        controller.require_healthy()
        if hdf5_collector is not None:
            preflight_snapshot = source.capture_snapshot()
            hdf5_collector.validate_snapshot(preflight_snapshot)
            append_runtime_event(
                "episode_save_preflight_passed",
                episode_number=episode_number,
            )
            hdf5_collector.begin_episode(episode_number)
        if episode_number != 1:
            robot.move_to_joint_positions(
                initial_joints,
                speed_percent=initial_speed_percent,
                gripper_effort=initial_gripper_effort,
            )
        append_runtime_event("episode_started", episode_number=episode_number)

    def stable_step_callback(**fields: Any) -> None:
        nonlocal diagnostic_steps, diagnostics_active
        session_id = fields.get("session_id")
        timestamp_s = float(fields["snapshot_before_command"].timestamp_s)
        if (
            not fields["is_intervention"]
            and session_id is not None
            and (not rollout_session_ids or rollout_session_ids[-1] != session_id)
        ):
            rollout_session_ids.append(str(session_id))
        if hdf5_collector is not None:
            hdf5_collector.record_stable_step(
                snapshot_before_command=fields["snapshot_before_command"],
                action_state=fields["action_state"],
                is_intervention=fields["is_intervention"],
            )
        within_diagnostic_limit = diagnostic_limit == 0 or diagnostic_steps < diagnostic_limit
        if within_diagnostic_limit:
            raw_action = fields.get("raw_action")
            if raw_action is None:
                raw_action = np.full(int(spec.action_dim), np.nan, dtype=np.float64)
            if diagnostics_active:
                try:
                    action_state_snapshot = RobotSnapshot(
                        timestamp_s=timestamp_s,
                        state=fields["action_state"],
                        images={},
                    )
                    if state_builder is None:
                        raise ValueError(
                            "state_builder is required for diagnostic rollout recording"
                        )
                    record_sink.record(
                        images=fields["snapshot_before_command"].images,
                        action=np.asarray(raw_action, dtype=np.float64),
                        state=state_builder(action_state_snapshot, spec),
                        timestamp_s=timestamp_s,
                    )
                except Exception as exc:
                    diagnostics_active = False
                    print(
                        f"Diagnostic recording disabled after a non-hardware failure: {exc}",
                        flush=True,
                    )
                    append_runtime_event(
                        "diagnostic_recording_failed",
                        error=repr(exc),
                        hardware_fault=False,
                        durable=True,
                    )
            diagnostic_steps += 1

    def on_episode_finish(episode_number: int, stop_reason: str, partial: bool) -> Any:
        episode_payload = None
        if hdf5_collector is not None:
            episode_payload = hdf5_collector.finish_episode(
                episode_number,
                stop_reason,
                partial,
            )
        append_runtime_event(
            "episode_finished",
            episode_number=episode_number,
            stop_reason=stop_reason,
            partial=partial,
            hdf5_output_path=(
                str(episode_payload.output_path)
                if episode_payload is not None
                else None
            ),
            durable=True,
        )
        return episode_payload

    def on_writer_result(writer_result: dict[str, Any]) -> None:
        print(json.dumps({"hdf5_writer": writer_result}, indent=2), flush=True)
        append_runtime_event("hdf5_writer_result", **writer_result, durable=True)

    intervention_sample_max_age_s = (
        float(robot.motion_watchdog_max_state_age_s)
        if args.intervention
        else None
    )

    coordinator: DynamicRolloutCoordinator | None = None
    session_result: InteractiveSessionResult | None = None
    interactive_session_owns_cleanup = False
    try:
        coordinator = DynamicRolloutCoordinator(
            client=client,
            source=source,
            robot=robot,
            spec=spec,
            hardware_controller=controller,
            prompt=prompt,
            execution_mode=args.execution_mode,
            fps=args.fps,
            chunk_size=chunk_size,
            inference_rate=inference_rate,
            latency_k=latency_k,
            min_smooth_steps=min_smooth_steps,
            buffer_max_chunks=buffer_max_chunks,
            request_timeout_s=float(args.inference_timeout),
            intervention_sample_max_age_s=intervention_sample_max_age_s,
            stable_step_callback=stable_step_callback,
            snapshot_validator=(
                hdf5_collector.validate_snapshot
                if hdf5_collector is not None
                else None
            ),
            runtime_event_callback=append_runtime_event,
            log_chunk=log_chunk,
        )
        session_kwargs: dict[str, Any] = {}
        if key_reader is not None:
            session_kwargs["key_reader"] = key_reader
        interactive_session_owns_cleanup = True
        session_result = run_interactive_session(
            coordinator=coordinator,
            rollout_steps=args.rollout_steps,
            fps=args.fps,
            intervention_enabled=bool(args.intervention),
            save_hdf5=bool(args.save),
            writer_pool=writer_pool,
            on_episode_start=on_episode_start,
            on_episode_finish=on_episode_finish,
            on_writer_result=on_writer_result,
            require_tty=require_tty,
            **session_kwargs,
        )
    finally:
        if coordinator is not None and not interactive_session_owns_cleanup:
            try:
                coordinator.close()
            except Exception as exc:
                append_runtime_event(
                    "coordinator_close_failed",
                    error=exc,
                    durable=True,
                )
        if writer_pool is not None and not interactive_session_owns_cleanup:
            try:
                remaining_writer_results = writer_pool.close(
                    terminate=session_result is None,
                )
            except Exception as exc:
                remaining_writer_results = [{"ok": False, "error": repr(exc)}]
            for writer_result in remaining_writer_results:
                on_writer_result(writer_result)
        assembly = getattr(robot, "hardware_assembly", None)
        gateway = getattr(assembly, "gateway", None)
        if gateway is not None:
            counters = {
                side: getattr(getattr(gateway, side, None), "counters", None)
                for side in ("left", "right")
            }
            append_runtime_event("semantic_gateway_counters", counters=counters, durable=True)
    if session_result is None:
        raise AssertionError("interactive session returned without a result")
    for episode in session_result.episodes:
        append_runtime_event(
            "episode_save_decision",
            episode_number=episode.episode_number,
            saved=episode.saved,
            partial=episode.partial,
            stable_steps=episode.stable_steps,
            stop_reason=episode.stop_reason,
            durable=True,
        )
    session_result.rollout_session_ids.extend(rollout_session_ids)
    return session_result


def make_slai_recording_schema(spec: Any, control_mode: str) -> Any:
    return RecordingSchema(
        camera_names=spec.image_ids,
        action_names=tuple(slai_piper_policy.get_vector_names(spec.action_space)),
        state_names=tuple(slai_piper_policy.get_vector_names(spec.state_space)),
        used_action_names=used_action_names(spec, control_mode),
    )


def build_slai_recording_state(
    snapshot: Any,
    spec: Any,
    *,
    state_gripper_encoding: str = "policy",
    dtype: Any = np.float64,
) -> np.ndarray:
    return build_configured_piper_state(
        snapshot,
        spec,
        state_gripper_encoding=state_gripper_encoding,
        dtype=dtype,
    )


def make_recording_state_builder(
    state_builder: Any,
    state_gripper_encoding: str,
) -> Any:
    def build_state(snapshot: Any, spec: Any) -> np.ndarray:
        return state_builder(
            snapshot,
            spec,
            state_gripper_encoding=state_gripper_encoding,
        )

    return build_state


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
