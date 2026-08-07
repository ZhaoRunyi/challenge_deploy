from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import Any, Callable, Mapping

from data.worker import HDF5TeleopDataWorker, HDF5TeleopSaveConfig
from hardware.config import load_config
from hardware.factory import build_hardware
from hardware.linkage_gateway import GatewayFaultLatchedError, LinkageGatewayError
from hardware.piper import DualPiperSystem
from hardware.realsense import RealSenseRig
from hardware.runtime import DualPiperArmView
from hardware.topology import (
    ArmId,
    MASTER_ARM_IDS,
    RoleTransactionError,
    RoleTransactionResult,
)
from rollout.recording import RecordingSchema
from rollout.support import (
    apply_runtime_overrides,
    default_runtime_config_path,
    make_runtime_event_callback,
)
from rollout.windowing import RuntimeExecutionWindow
from teleop.hdf5_teleop import (
    HDF5TeleopCollectionSource,
    HDF5_TELEOP_VECTOR_NAMES,
    collect_hdf5_teleop_episode,
    episode_base_path,
    infer_language_instruction,
    next_episode_index,
    running_sentinel_path,
)
from teleop.worker import TeleopEpisode, TeleopWorker


DEFAULT_JPEG_QUALITY = 95
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IDLE_PROMPT = "Idle: press c to start an episode, or q to quit."
RuntimeEventCallback = Callable[..., None]


def ignore_runtime_event(event: str, **fields: Any) -> None:
    del event, fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect HDF5 teleop episodes from dual Piper master/slave arms and RealSense cameras."
    )
    parser.add_argument("--dataset-dir", default=None, help="Root directory that contains task folders.")
    parser.add_argument("--task-name", default=None, help="Task folder name under dataset-dir.")
    parser.add_argument("--episode-idx", type=int, default=None, help="First episode index. Default: next available.")
    parser.add_argument("--language-instruction", default=None, help="Episode-level language instruction.")
    parser.add_argument("--max-timesteps", type=int, default=None, help="Optional per-episode data frame cap. Default: no cap.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--arm-sample-hz", type=float, default=200.0, help="Polling rate for piper_sdk cached arm states before timestamp alignment.")
    parser.add_argument("--queue-maxlen", type=int, default=2000, help="Max async samples to keep per source, matching the original ROS deque cap by default.")
    parser.add_argument("--countdown-seconds", type=int, default=0)
    parser.add_argument("--alignment-plot-frames", type=int, default=16, help="Number of evenly spaced selected frames to draw in the alignment plot.")
    parser.add_argument("--record", "--recording", action="store_true", help="Render a Motus/OpenPI-style deploy video from each saved HDF5 episode.")
    parser.add_argument("--save-sep", action="store_true", help="Save one raw-camera video per recording camera; requires --record.")
    parser.add_argument("--window", nargs="?", const=1, type=int, default=0, help="Show live camera/action-state window; optional value selects display index.")
    parser.add_argument("--action-from-state", action="store_true", help="Save action[t] from slave state[t+1] instead of master control[t+1].")
    parser.add_argument(
        "--record-dir",
        default=None,
        help=(
            "Optional directory for --record diagnostic and --save-sep per-camera "
            "videos. Default: the directory containing the corresponding HDF5 file."
        ),
    )
    parser.add_argument("--skip-idle", action=argparse.BooleanOptionalAction, default=True, help="Skip frames when master arms stay within the idle tolerance. Use --no-skip-idle to keep them.")
    parser.add_argument("--use-depth-image", action="store_true")
    parser.add_argument(
        "--config",
        default=default_runtime_config_path(),
    )
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--master-left-can", default=None)
    parser.add_argument("--master-right-can", default=None)
    parser.add_argument("--camera-high-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument("--extra-camera-names", nargs="*", default=())
    parser.add_argument("--extra-camera-serials", nargs="*", default=())
    parser.add_argument(
        "--event-log",
        default=None,
        help=(
            "Optional isolated-runtime JSONL audit path. Default: a timestamped file "
            "under artifacts/runtime_events. Shared topology does not create this log."
        ),
    )
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    return parser


def validate_collection_camera_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the complete static-collection camera schema before hardware opens."""

    cameras = config.get("cameras")
    if not isinstance(cameras, Mapping):
        raise ValueError("cameras must be a mapping for HDF5 teleop collection")
    if cameras.get("enabled") is not True:
        raise ValueError("HDF5 teleop collection requires cameras.enabled=true")

    serials = cameras.get("serials")
    if not isinstance(serials, Mapping) or not serials:
        raise ValueError("cameras.serials must contain at least one configured camera")

    camera_names: list[str] = []
    configured_serials: dict[str, str] = {}
    for raw_camera_name, raw_serial in serials.items():
        camera_name = str(raw_camera_name)
        validate_camera_name(camera_name)
        if not isinstance(raw_serial, str) or not raw_serial.strip():
            raise ValueError(
                f"cameras.serials.{camera_name} must be a non-empty device serial"
            )
        serial = raw_serial.strip()
        duplicate_name = configured_serials.get(serial)
        if duplicate_name is not None:
            raise ValueError(
                f"Configured cameras {duplicate_name!r} and {camera_name!r} share "
                f"device serial {serial!r}"
            )
        configured_serials[serial] = camera_name
        camera_names.append(camera_name)

    for field in ("width", "height", "fps"):
        value = cameras.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"cameras.{field} must be a positive integer")
    warmup_frames = cameras.get("warmup_frames")
    if (
        isinstance(warmup_frames, bool)
        or not isinstance(warmup_frames, int)
        or warmup_frames < 0
    ):
        raise ValueError("cameras.warmup_frames must be a non-negative integer")
    return tuple(camera_names)


def append_role_transaction_audit(
    event_callback: RuntimeEventCallback,
    transaction: RoleTransactionResult | RoleTransactionError,
    *,
    status: str,
) -> None:
    writes = tuple(transaction.writes)
    rollbacks = tuple(transaction.rollbacks)
    action_counts = {
        "request": sum(bool(result.written) for result in writes),
        "skip": sum(not bool(result.written) for result in writes),
        "success": sum(bool(result.ok) for result in writes),
        "motion_output_seed": sum(
            bool(result.motion_output_seeded)
            for result in writes + rollbacks
        ),
        "retry": max(0, int(transaction.attempts) - 1),
        "rollback": len(rollbacks),
    }
    event_callback(
        "isolated_role_transaction",
        status=status,
        target=transaction.target.short_name,
        attempts=int(transaction.attempts),
        actions=action_counts,
        writes=writes,
        rollbacks=rollbacks,
        durable=status != "success",
    )


def gateway_counter_snapshot(gateway: Any) -> dict[str, Any]:
    return {
        side: getattr(getattr(gateway, side, None), "counters", None)
        for side in ("left", "right")
    }


def audit_save_decision(
    event_callback: RuntimeEventCallback,
    episode_index: int | None,
    decision: str,
    *,
    reason: str | None = None,
    captured_frames: int | None = None,
    **fields: Any,
) -> None:
    if reason is not None:
        fields["reason"] = reason
    if captured_frames is not None:
        fields["captured_frames"] = captured_frames
    event_callback(
        "hdf5_episode_save_decision",
        episode_idx=episode_index,
        decision=decision,
        durable=True,
        **fields,
    )


@dataclass(frozen=True, slots=True)
class CollectionRuntime:
    master_robot: Any
    slave_robot: Any
    cameras: RealSenseRig
    source: HDF5TeleopCollectionSource
    start_callbacks: tuple[Callable[[], None], ...]
    stop_callbacks: tuple[tuple[str, Callable[[], None]], ...]
    emergency_callbacks: tuple[tuple[str, Callable[[], None]], ...] = ()
    wait_for_source_ready: bool = True


def make_cameras(config: dict[str, Any], *, enable_depth: bool) -> RealSenseRig:
    return RealSenseRig(
        config["cameras"]["serials"],
        width=int(config["cameras"]["width"]),
        height=int(config["cameras"]["height"]),
        fps=int(config["cameras"]["fps"]),
        warmup_frames=int(config["cameras"]["warmup_frames"]),
        enable_depth=enable_depth,
    )


def make_shared_collection_runtime(
    config: dict[str, Any],
    *,
    enable_depth: bool,
    arm_sample_hz: float,
    queue_maxlen: int,
    event_callback: RuntimeEventCallback | None = None,
) -> CollectionRuntime:
    del event_callback
    slave_robot = DualPiperSystem(
        left_can_name=config["robot"]["slave_left"]["can_name"],
        right_can_name=config["robot"]["slave_right"]["can_name"],
        commands_enabled=False,
        name="hdf5_teleop_slave_reader",
    )
    master_robot = DualPiperSystem(
        left_can_name=config["robot"]["master_left"]["can_name"],
        right_can_name=config["robot"]["master_right"]["can_name"],
        commands_enabled=False,
        prefer_joint_ctrl=True,
        name="hdf5_teleop_master_reader",
    )
    cameras = make_cameras(config, enable_depth=enable_depth)
    source = HDF5TeleopCollectionSource(
        master_robot=master_robot,
        slave_robot=slave_robot,
        cameras=cameras,
        arm_sample_hz=arm_sample_hz,
        queue_maxlen=queue_maxlen,
    )
    return CollectionRuntime(
        master_robot=master_robot,
        slave_robot=slave_robot,
        cameras=cameras,
        source=source,
        start_callbacks=(
            lambda: master_robot.connect(read_only=True),
            lambda: slave_robot.connect(read_only=True),
        ),
        stop_callbacks=(
            ("cameras", cameras.stop),
            ("master robot", master_robot.disconnect),
            ("slave robot", slave_robot.disconnect),
        ),
    )


def make_isolated_collection_runtime(
    config: dict[str, Any],
    *,
    enable_depth: bool,
    arm_sample_hz: float,
    queue_maxlen: int,
    event_callback: RuntimeEventCallback | None = None,
) -> CollectionRuntime:
    append_event = event_callback or ignore_runtime_event
    assembly = build_hardware(
        config,
        intervention=True,
        commands_enabled=True,
        name="hdf5_teleop_isolated",
    )
    robot = assembly.robot
    gateway = assembly.gateway
    max_state_age_s = float(robot.motion_watchdog_max_state_age_s)
    gateway_has_fresh_family = False

    def require_isolated_teleop_health() -> None:
        nonlocal gateway_has_fresh_family
        try:
            gateway.require_fresh_family()
            gateway_has_fresh_family = True
        except GatewayFaultLatchedError:
            raise
        except LinkageGatewayError:
            if gateway_has_fresh_family:
                raise
        robot.require_healthy(
            prefer_joint_ctrl_arm_ids=MASTER_ARM_IDS,
        )

    cameras = None
    try:
        master_robot = DualPiperArmView(
            left=robot.master_left,
            right=robot.master_right,
            prefer_joint_ctrl=True,
        )
        slave_robot = DualPiperArmView(
            left=robot.slave_left,
            right=robot.slave_right,
        )
        cameras = make_cameras(config, enable_depth=enable_depth)
        source = HDF5TeleopCollectionSource(
            master_robot=master_robot,
            slave_robot=slave_robot,
            cameras=cameras,
            arm_sample_hz=arm_sample_hz,
            queue_maxlen=queue_maxlen,
            health_check=require_isolated_teleop_health,
            max_sample_age_s=max_state_age_s,
        )
    except BaseException as exc:
        append_event(
            "isolated_static_teleop_runtime_build_failed",
            phase="camera_or_collection_source",
            stop_reason="runtime_build_failed",
            error=exc,
            counters=gateway_counter_snapshot(gateway),
            durable=True,
        )
        cleanup_steps: list[tuple[str, Callable[[], None]]] = []
        if cameras is not None:
            cleanup_steps.append(("cameras", cameras.stop))
        cleanup_steps.extend(
            (
                ("semantic gateway", gateway.close),
                ("isolated robot construction", robot.abort_construction),
            )
        )
        run_cleanup_steps(tuple(cleanup_steps), event_callback=append_event)
        raise
    slave_efforts = {
        ArmId.SLAVE_LEFT: int(
            config["robot"]["slave_left"].get("gripper_effort", 1000)
        ),
        ArmId.SLAVE_RIGHT: int(
            config["robot"]["slave_right"].get("gripper_effort", 1000)
        ),
    }

    def initialize_isolated_teleop() -> None:
        append_event("isolated_static_teleop_initialization_started")
        try:
            result = robot.initialize_static_teleop(
                gateway,
                gripper_efforts=slave_efforts,
            )
        except RoleTransactionError as exc:
            append_role_transaction_audit(append_event, exc, status="failed")
            raise
        for operation in result.operations:
            if isinstance(operation, RoleTransactionResult):
                append_role_transaction_audit(
                    append_event,
                    operation,
                    status="success",
                )
        append_event(
            "isolated_static_teleop_initialized",
            mode=result.mode,
            generation=result.generation,
            operations=result.operations,
            durable=True,
        )

    def close_isolated_gateway() -> None:
        append_event(
            "semantic_gateway_counters",
            counters=gateway_counter_snapshot(gateway),
            durable=True,
        )
        gateway.close()

    def hold_isolated_motion_output_arms_after_fault() -> None:
        robot.best_effort_hold_motion_output_arms(
            gripper_efforts=robot.effective_slave_gripper_efforts,
        )

    return CollectionRuntime(
        master_robot=master_robot,
        slave_robot=slave_robot,
        cameras=cameras,
        source=source,
        start_callbacks=(initialize_isolated_teleop,),
        stop_callbacks=(
            ("cameras", cameras.stop),
            ("semantic gateway", close_isolated_gateway),
            ("isolated robot", robot.disconnect),
        ),
        emergency_callbacks=(
            ("semantic gateway stop", gateway.stop),
            (
                "isolated motion-output arm hold",
                hold_isolated_motion_output_arms_after_fault,
            ),
        ),
        wait_for_source_ready=False,
    )


COLLECTION_RUNTIME_BUILDERS: dict[str, Callable[..., CollectionRuntime]] = {
    "shared": make_shared_collection_runtime,
    "isolated": make_isolated_collection_runtime,
}


def make_collection_runtime(
    config: dict[str, Any],
    *,
    enable_depth: bool,
    arm_sample_hz: float,
    queue_maxlen: int,
    event_callback: RuntimeEventCallback | None = None,
) -> CollectionRuntime:
    builder = COLLECTION_RUNTIME_BUILDERS[config["can_topology"]]
    return builder(
        config,
        enable_depth=enable_depth,
        arm_sample_hz=arm_sample_hz,
        queue_maxlen=queue_maxlen,
        event_callback=event_callback,
    )


def make_teleop_worker(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    event_callback: RuntimeEventCallback | None = None,
) -> TeleopWorker:
    runtime = make_collection_runtime(
        config,
        enable_depth=args.use_depth_image,
        arm_sample_hz=args.arm_sample_hz,
        queue_maxlen=args.queue_maxlen,
        event_callback=event_callback,
    )
    return TeleopWorker(
        source=runtime.source,
        ready_timeout_s=args.ready_timeout,
        wait_for_source_ready=runtime.wait_for_source_ready,
        start_callbacks=runtime.start_callbacks,
        stop_callbacks=runtime.stop_callbacks,
        emergency_callbacks=runtime.emergency_callbacks,
    )


def make_data_worker(
    args: argparse.Namespace,
    language_instruction: str,
    camera_names: tuple[str, ...],
) -> HDF5TeleopDataWorker:
    config = HDF5TeleopSaveConfig(
        camera_names=camera_names,
        language_instruction=language_instruction,
        include_depth_images=args.use_depth_image,
        jpeg_quality=DEFAULT_JPEG_QUALITY,
        action_from_state=args.action_from_state,
        alignment_plot_frames=args.alignment_plot_frames,
        record_video=args.record,
        save_separate_videos=args.save_sep,
        fps=args.fps,
        record_output_dir=(
            Path(args.record_dir).expanduser().resolve()
            if args.record and args.record_dir
            else None
        ),
    )
    return HDF5TeleopDataWorker(config=config)


def run_cleanup_steps(
    cleanup_steps: tuple[tuple[str, Callable[[], None]], ...],
    *,
    event_callback: RuntimeEventCallback = ignore_runtime_event,
) -> None:
    for cleanup_name, cleanup_callback in cleanup_steps:
        try:
            cleanup_callback()
        except Exception as exc:
            print(f"Failed to clean up {cleanup_name}: {exc}", flush=True)
            event_callback(
                "isolated_static_teleop_cleanup_failed",
                cleanup_step=cleanup_name,
                error=exc,
                durable=True,
            )


def make_collection_session_resources(
    runtime_config: dict[str, Any],
    args: argparse.Namespace,
    *,
    language_instruction: str,
    camera_names: tuple[str, ...],
    event_callback: RuntimeEventCallback | None,
) -> tuple[TeleopWorker, HDF5TeleopDataWorker, RuntimeExecutionWindow | None, Any]:
    """Construct the static session under one pre-start cleanup boundary."""

    terminal_settings = termios.tcgetattr(sys.stdin.fileno())
    data_worker = make_data_worker(args, language_instruction, camera_names)
    runtime_window: RuntimeExecutionWindow | None = None
    try:
        if args.window:
            window_schema = RecordingSchema(
                camera_names=camera_names,
                action_names=HDF5_TELEOP_VECTOR_NAMES,
                state_names=HDF5_TELEOP_VECTOR_NAMES,
                used_action_names=frozenset(HDF5_TELEOP_VECTOR_NAMES),
            )
            runtime_window = RuntimeExecutionWindow(
                schema=window_schema,
                display_index=args.window,
            )
        teleop_worker = make_teleop_worker(
            runtime_config,
            args,
            event_callback=event_callback,
        )
        return teleop_worker, data_worker, runtime_window, terminal_settings
    except BaseException:
        cleanup_steps: list[tuple[str, Callable[[], None]]] = []
        if runtime_window is not None:
            cleanup_steps.append(("runtime window", runtime_window.close))
        cleanup_steps.append(
            (
                "data worker",
                lambda: data_worker.stop(
                    on_result=print_save_result,
                    on_error=print_save_error,
                ),
            )
        )
        run_cleanup_steps(
            tuple(cleanup_steps),
            event_callback=event_callback or ignore_runtime_event,
        )
        raise


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, writable, errors = select.select([sys.stdin], [], [], 0.0)
    del writable, errors
    if not ready:
        return None
    return sys.stdin.read(1).lower()


def wait_for_key(valid_keys: set[str], poll: Callable[[], None] | None = None) -> str:
    while True:
        if poll is not None:
            poll()
        key = read_key()
        if key in valid_keys:
            return key
        time.sleep(0.05)


def print_json(key: str, value: dict[str, Any]) -> None:
    print(json.dumps({key: value}, indent=2), flush=True)


def print_save_result(result: dict[str, Any]) -> None:
    print_json("hdf5_teleop_collection_result", result)


def print_save_error(exc: Exception) -> None:
    print(f"HDF5 teleop save failed: {exc}", flush=True)


def print_idle_prompt() -> None:
    print(IDLE_PROMPT, flush=True)


def print_episode_decision_prompt() -> None:
    print("Episode stopped: press c to save and continue, or d to discard and continue.", flush=True)


def validate_camera_name(camera_name: str) -> None:
    if not camera_name or not all(char.isalnum() or char == "_" for char in camera_name):
        raise ValueError(
            f"Camera name {camera_name!r} must be non-empty and contain only letters, digits, or underscores"
        )


def validate_args(args: argparse.Namespace) -> None:
    if len(args.extra_camera_names) != len(args.extra_camera_serials):
        raise ValueError("--extra-camera-names and --extra-camera-serials must have the same length")
    seen_camera_names = set(CAMERA_NAMES)
    for camera_name in args.extra_camera_names:
        validate_camera_name(camera_name)
        if camera_name in seen_camera_names:
            raise ValueError(f"Extra camera name {camera_name!r} is duplicated or already reserved")
        seen_camera_names.add(camera_name)
    for serial in args.extra_camera_serials:
        if not str(serial).strip():
            raise ValueError("--extra-camera-serials cannot contain empty values")
    if args.max_timesteps is not None and args.max_timesteps <= 0:
        raise ValueError("--max-timesteps must be positive when set")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.arm_sample_hz <= 0.0:
        raise ValueError("--arm-sample-hz must be positive")
    if args.queue_maxlen <= 0:
        raise ValueError("--queue-maxlen must be positive")
    if args.alignment_plot_frames <= 0:
        raise ValueError("--alignment-plot-frames must be positive")
    if args.window < 0:
        raise ValueError("--window display index must be non-negative")
    if args.save_sep and not args.record:
        raise ValueError("--save-sep requires --record")
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive HDF5 teleop collection requires a TTY for c/s/q controls")


def episode_start_payload(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    episode_idx: int,
    episode_path: Path,
    language_instruction: str,
    camera_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "dataset_root": str(dataset_root),
        "episode_idx": episode_idx,
        "episode_path": str(episode_path.with_suffix(".hdf5")),
        "episode_dir": str(episode_path.parent),
        "language_instruction": language_instruction,
        "max_timesteps": args.max_timesteps,
        "fps": args.fps,
        "arm_sample_hz": args.arm_sample_hz,
        "queue_maxlen": args.queue_maxlen,
        "alignment_plot_frames": args.alignment_plot_frames,
        "record": args.record,
        "record_dir": (
            str(Path(args.record_dir).expanduser())
            if args.record_dir
            else str(episode_path.parent)
        ),
        "action_from_state": args.action_from_state,
        "skip_idle": args.skip_idle,
        "use_depth_image": args.use_depth_image,
        "camera_names": list(camera_names),
        "running_sentinel": str(running_sentinel_path(dataset_root, episode_idx)),
    }


def stop_requested() -> bool:
    key_pressed = read_key()
    if key_pressed == "s":
        print("Stop requested for current episode.", flush=True)
        return True
    if key_pressed == "q":
        print("Recording is active; press s to save this episode, then q to quit from idle.", flush=True)
    return False


def collect_kwargs(
    args: argparse.Namespace,
    dataset_root: Path,
    episode_idx: int,
    runtime_window: RuntimeExecutionWindow | None,
    stop_requested_callback: Callable[[], bool] = stop_requested,
    frame_sink: list[Any] | None = None,
) -> dict[str, Any]:
    kwargs = {
        "max_timesteps": args.max_timesteps,
        "fps": args.fps,
        "countdown_seconds": args.countdown_seconds,
        "ready_timeout_s": args.ready_timeout,
        "running_sentinel": running_sentinel_path(dataset_root, episode_idx),
        "stop_requested": stop_requested_callback,
        "start_source": False,
        "skip_stationary": args.skip_idle,
        "runtime_window": runtime_window,
        "action_from_state": args.action_from_state,
    }
    if frame_sink is not None:
        kwargs["frame_sink"] = frame_sink
    return kwargs


def default_event_log_path() -> Path:
    return (
        Path.cwd()
        / "artifacts"
        / "runtime_events"
        / f"hdf5_teleop_session_{time.time_ns()}.jsonl"
    )


@dataclass(slots=True)
class ActiveEpisode:
    index: int | None = None
    path: Path | None = None
    captured_frames: list[Any] | None = None
    completed: TeleopEpisode | None = None

    def begin(self, index: int, path: Path, *, retain_frames: bool) -> None:
        self.index = index
        self.path = path
        self.captured_frames = [] if retain_frames else None
        self.completed = None

    def clear(self) -> None:
        self.index = None
        self.path = None
        self.captured_frames = None
        self.completed = None


def preserve_isolated_partial_episode(
    *,
    teleop_worker: TeleopWorker,
    data_worker: HDF5TeleopDataWorker,
    active_episode: ActiveEpisode,
    stop_reason: str,
    error: BaseException,
    event_callback: RuntimeEventCallback,
) -> TeleopEpisode | None:
    """Fence isolated motion before asynchronously preserving completed frames."""

    emergency_failures = teleop_worker.emergency_stop()
    event_callback(
        "isolated_static_teleop_emergency_stop",
        stop_reason=stop_reason,
        failures=emergency_failures,
        durable=True,
    )

    if active_episode.completed is not None:
        partial_episode = active_episode.completed.as_partial(stop_reason)
    elif active_episode.index is not None and active_episode.path is not None:
        try:
            partial_episode = teleop_worker.episode_from_frames(
                episode_index=active_episode.index,
                episode_path=active_episode.path,
                frames=active_episode.captured_frames or (),
                partial=True,
                stop_reason=stop_reason,
            )
        except BaseException as partial_build_error:
            print(
                "Failed to package completed frames for partial recovery: "
                f"{partial_build_error}",
                flush=True,
            )
            audit_save_decision(
                event_callback,
                active_episode.index,
                "save_failed",
                reason="partial_packaging_failed",
                error=partial_build_error,
            )
            return None
    else:
        return None

    frame_count = len(partial_episode.frames)
    event_callback(
        "hdf5_episode_finished",
        episode_idx=partial_episode.episode_index,
        stop_reason=stop_reason,
        partial=True,
        captured_frames=frame_count,
        error=error,
        durable=True,
    )
    if frame_count < 2:
        print(
            f"Discarded partial episode {partial_episode.episode_index}: "
            f"need at least 2 frames, got {frame_count}.",
            flush=True,
        )
        audit_save_decision(
            event_callback,
            partial_episode.episode_index,
            "discard",
            reason="insufficient_partial_frames",
            partial=True,
            captured_frames=frame_count,
        )
        return partial_episode

    try:
        queued_result = data_worker.submit(partial_episode)
    except BaseException as submit_error:
        print_save_error(submit_error)
        audit_save_decision(
            event_callback,
            partial_episode.episode_index,
            "save_failed",
            reason="partial_enqueue_failed",
            partial=True,
            captured_frames=frame_count,
            error=submit_error,
        )
        return partial_episode

    print_json("hdf5_teleop_partial_queued", queued_result)
    audit_save_decision(
        event_callback,
        partial_episode.episode_index,
        "save",
        reason="automatic_partial_recovery",
        partial=True,
        captured_frames=frame_count,
        queued_result=queued_result,
    )
    return partial_episode


def run_once(args: argparse.Namespace) -> None:
    validate_args(args)
    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    camera_names = validate_collection_camera_config(runtime_config)
    dataset_dir = Path(args.dataset_dir or runtime_config["dataset"]["dataset_dir"]).expanduser().resolve()
    task_name = str(args.task_name or runtime_config["dataset"]["dataset_name"])
    dataset_root = dataset_dir / task_name
    language_instruction = infer_language_instruction(task_name, args.language_instruction)
    next_manual_episode = args.episode_idx
    isolated_runtime = runtime_config["can_topology"] == "isolated"

    event_callback: RuntimeEventCallback = ignore_runtime_event
    if isolated_runtime:
        event_log_path = Path(args.event_log).expanduser() if args.event_log else default_event_log_path()
        event_callback = make_runtime_event_callback(event_log_path)
        event_callback(
            "isolated_static_teleop_session_configured",
            config_path=str(Path(args.config).expanduser()),
            event_log_path=str(event_log_path),
            camera_names=camera_names,
            durable=True,
        )

    teleop_worker, data_worker, runtime_window, terminal_settings = (
        make_collection_session_resources(
            runtime_config,
            args,
            language_instruction=language_instruction,
            camera_names=camera_names,
            event_callback=event_callback,
        )
    )
    last_window_refresh_s = 0.0

    def report_save_result(result: dict[str, Any]) -> None:
        print_save_result(result)
        event_callback(
            "hdf5_save_result",
            status="saved",
            result=result,
            durable=True,
        )

    def report_save_error(exc: Exception) -> None:
        print_save_error(exc)
        event_callback(
            "hdf5_save_result",
            status="failed",
            error=exc,
            durable=True,
        )

    def drain_data_worker(*, repeat_prompt: Callable[[], None] | None = None) -> None:
        completed_count = data_worker.drain(
            on_result=report_save_result,
            on_error=report_save_error,
        )
        if repeat_prompt is not None and completed_count > 0:
            repeat_prompt()

    def refresh_idle_window() -> None:
        nonlocal last_window_refresh_s
        if runtime_window is None:
            return
        now_s = time.monotonic()
        if now_s - last_window_refresh_s < 0.1:
            return
        images = teleop_worker.source.latest_images()
        if images is not None:
            runtime_window.show_images(images)
        last_window_refresh_s = now_s

    def poll_idle_health() -> None:
        if not isolated_runtime:
            return
        teleop_worker.source.require_healthy()

    def poll_wait(repeat_prompt: Callable[[], None]) -> None:
        drain_data_worker(repeat_prompt=repeat_prompt)
        poll_idle_health()
        refresh_idle_window()

    active_episode = ActiveEpisode()
    session_stop_reason = "startup_failed"
    try:
        teleop_worker.start()
        event_callback("isolated_static_teleop_session_started", durable=True)
        tty.setcbreak(sys.stdin.fileno())
        print("Interactive controls: idle c starts an episode, recording s stops it, idle q quits.", flush=True)
        while True:
            drain_data_worker()
            poll_idle_health()
            refresh_idle_window()
            print_idle_prompt()
            key = wait_for_key(
                {"c", "q"},
                poll=lambda: poll_wait(print_idle_prompt),
            )
            if key == "q":
                session_stop_reason = "idle_q"
                break

            episode_idx = next_manual_episode if next_manual_episode is not None else next_episode_index(dataset_root)
            if next_manual_episode is not None:
                next_manual_episode += 1
            episode_path = episode_base_path(dataset_root, episode_idx)
            active_episode.begin(
                episode_idx,
                episode_path,
                retain_frames=isolated_runtime,
            )
            print_json(
                "hdf5_teleop_collection",
                episode_start_payload(
                    args=args,
                    dataset_root=dataset_root,
                    episode_idx=episode_idx,
                    episode_path=episode_path,
                    language_instruction=language_instruction,
                    camera_names=camera_names,
                ),
            )

            if runtime_window is not None:
                runtime_window.reset()
            episode_stop_reason: str | None = None

            def episode_stop_requested() -> bool:
                nonlocal episode_stop_reason
                requested = stop_requested()
                if requested:
                    episode_stop_reason = "operator_stop"
                return requested

            event_callback(
                "hdf5_episode_started",
                episode_idx=episode_idx,
                episode_path=str(episode_path.with_suffix(".hdf5")),
            )
            episode = teleop_worker.collect_episode(
                episode_index=episode_idx,
                episode_path=episode_path,
                collect_fn=collect_hdf5_teleop_episode,
                collect_kwargs=collect_kwargs(
                    args,
                    dataset_root,
                    episode_idx,
                    runtime_window,
                    stop_requested_callback=episode_stop_requested,
                    frame_sink=active_episode.captured_frames,
                ),
            )
            active_episode.completed = episode
            if episode_stop_reason is None:
                target_frame_count = (
                    None if args.max_timesteps is None else args.max_timesteps + 1
                )
                episode_stop_reason = (
                    "max_timesteps"
                    if target_frame_count is not None
                    and len(episode.frames) >= target_frame_count
                    else "running_sentinel_removed"
                )
            event_callback(
                "hdf5_episode_finished",
                episode_idx=episode_idx,
                stop_reason=episode_stop_reason,
                partial=False,
                captured_frames=len(episode.frames),
                durable=True,
            )
            if runtime_window is not None:
                runtime_window.reset()
            if len(episode.frames) < 2:
                print(f"Discarded episode {episode_idx}: need at least 2 frames, got {len(episode.frames)}.", flush=True)
                audit_save_decision(
                    event_callback,
                    episode_idx,
                    "discard",
                    reason="insufficient_frames",
                    captured_frames=len(episode.frames),
                )
                active_episode.clear()
                continue
            refresh_idle_window()
            print_episode_decision_prompt()
            decision = wait_for_key(
                {"c", "d"},
                poll=lambda: poll_wait(print_episode_decision_prompt),
            )
            if decision == "d":
                print(f"Discarded episode {episode_idx}: user requested delete.", flush=True)
                audit_save_decision(
                    event_callback,
                    episode_idx,
                    "discard",
                    reason="operator_discard",
                    captured_frames=len(episode.frames),
                )
                active_episode.clear()
                continue
            queued_result = data_worker.submit(episode)
            active_episode.clear()
            print_json("hdf5_teleop_collection_queued", queued_result)
            audit_save_decision(
                event_callback,
                episode_idx,
                "save",
                queued_result=queued_result,
            )
    except BaseException as exc:
        session_stop_reason = (
            "keyboard_interrupt" if isinstance(exc, KeyboardInterrupt) else "error"
        )
        if isolated_runtime:
            preserve_isolated_partial_episode(
                teleop_worker=teleop_worker,
                data_worker=data_worker,
                active_episode=active_episode,
                stop_reason=session_stop_reason,
                error=exc,
                event_callback=event_callback,
            )
        if not isinstance(exc, KeyboardInterrupt):
            event_callback(
                "isolated_static_teleop_session_error",
                error=exc,
                durable=True,
            )
        raise
    finally:
        event_callback(
            "isolated_static_teleop_session_finished",
            stop_reason=session_stop_reason,
            durable=True,
        )
        cleanup_steps: list[tuple[str, Callable[[], None]]] = [
            (
                "terminal settings",
                lambda: termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    terminal_settings,
                ),
            ),
        ]
        if runtime_window is not None:
            cleanup_steps.append(("runtime window", runtime_window.close))
        cleanup_steps.extend(
            (
                ("teleop worker", teleop_worker.stop),
                (
                    "data worker",
                    lambda: data_worker.stop(
                        on_result=report_save_result,
                        on_error=report_save_error,
                    ),
                ),
            )
        )
        run_cleanup_steps(tuple(cleanup_steps), event_callback=event_callback)


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
