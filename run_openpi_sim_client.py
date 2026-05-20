from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.lerobot_assets import dataset_asset_info, prepare_train_assets, resolve_prompt
from challenge_deploy.rollout_metrics import save_rollout_metrics_summary
from challenge_deploy.openpi_sim_client import (
    SIM_ACTION_NAMES,
    SIM_STATE_NAMES,
    OpenPiSimPiperClient,
    OpenPiSimPolicySpec,
    build_configured_piper_state,
    decoded_action_summary,
    load_openpi_sim_policy_spec,
    sim_gripper_to_piper,
    spec_summary,
)
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.realsense import RealSenseRig
from challenge_deploy.recording import OpenPiRolloutRecorder, RecordingSchema, preview_until_continue, save_frame1_image
from challenge_deploy.runtime import DualPiperObservationSource
from challenge_deploy.schemas import RobotSnapshot

DEPLOY_ROOT = Path(__file__).resolve().parent
INIT_EMBODICHAIN_QPOS = np.array(
    [
        -0.31471917033195496,
        0.9185937643051147,
        -1.1522988080978394,
        -0.06913353502750397,
        0.6181001663208008,
        -0.0012999551836401224,
        1.0000096559524536,
        0.3093342185020447,
        0.9417879581451416,
        -1.1448285579681396,
        -0.12809762358665466,
        0.6446187496185303,
        0.13154016435146332,
        1.0000003576278687,
    ],
    dtype=np.float64,
) # OLD

INIT_EMBODICHAIN_QPOS = np.array(
    [
        -0.05918411,
        0.00076794,
        -0.12870058,
        -0.13548991,
        0.29586821,
        0.13372713,
        0.0,
        0.08932595,
        0.00970403,
        -0.21027726,
        -0.08838347,
        0.39285615,
        0.08686504,
        0.0,
    ],
    dtype=np.float64,
)


def _initial_piper_joints(*, old_gripper: bool = False) -> np.ndarray:
    qpos = INIT_EMBODICHAIN_QPOS.copy()
    qpos[[6, 13]] = [
        sim_gripper_to_piper(qpos[6], old_gripper=old_gripper),
        sim_gripper_to_piper(qpos[13], old_gripper=old_gripper),
    ]
    return qpos


@dataclass
class RolloutMetrics:
    execution_mode: str = "chunk_sync"
    executed_steps: int = 0
    inferred_chunks: int = 0
    inference_seconds: list[float] = field(default_factory=list)
    command_period_seconds: list[float] = field(default_factory=list)
    command_seconds: list[float] = field(default_factory=list)
    rollout_started_at_s: float = field(default_factory=time.monotonic)
    interrupted: bool = False
    stop_reason: str | None = None

    def record_inference(self, seconds: float) -> None:
        self.inferred_chunks += 1
        self.inference_seconds.append(float(seconds))

    def record_command(self, *, period_seconds: float | None, command_seconds: float) -> None:
        self.executed_steps += 1
        if period_seconds is not None:
            self.command_period_seconds.append(float(period_seconds))
        self.command_seconds.append(float(command_seconds))

    def mark_interrupted(self, reason: str | None = None) -> None:
        self.interrupted = True
        self.stop_reason = reason or "KeyboardInterrupt"

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "p50": None, "p95": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "executed_steps": self.executed_steps,
            "inferred_chunks": self.inferred_chunks,
            "rollout_wall_seconds": float(time.monotonic() - self.rollout_started_at_s),
            "inference_seconds": self._stats(self.inference_seconds),
            "command_period_seconds": self._stats(self.command_period_seconds),
            "command_seconds": self._stats(self.command_seconds),
            "interrupted": self.interrupted,
            "stop_reason": self.stop_reason,
        }


def action_sequence(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        return actions.reshape(1, -1)
    if actions.ndim == 2:
        return actions
    raise ValueError(f"Expected action vector or action chunk, got shape {actions.shape}")


def resolve_chunk_size(spec: OpenPiSimPolicySpec, requested_chunk_size: int | None) -> int | None:
    if requested_chunk_size is not None:
        if requested_chunk_size <= 0:
            raise ValueError("--chunk-size must be positive when provided")
        return requested_chunk_size
    if spec.action_horizon is not None and spec.action_horizon > 0:
        return int(spec.action_horizon)
    return None


def trim_chunk(actions: np.ndarray, chunk_size: int | None) -> np.ndarray:
    actions = action_sequence(actions)
    if chunk_size is None:
        return actions
    return actions[: min(chunk_size, len(actions))]


def sleep_until_next_action(action_start_s: float, fps: float) -> None:
    if fps <= 0.0:
        return
    remaining_s = (1.0 / fps) - (time.monotonic() - action_start_s)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenPI-sim EmbodiChain dual Piper client: fixed 14D joints+gripper01 action space."
    )
    parser.add_argument("--train-config", required=True, help="OpenPI train config name, e.g. pi0_slai_piper_template.")
    parser.add_argument("--ckpt-dir", default=None, help="Checkpoint directory used only for record video filenames.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--control-mode", choices=["joints"], default="joints")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--joint-speed-percent", type=int, default=50)
    parser.add_argument("--ee-speed-percent", type=int, default=50)
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=None,
        help="Optional executable-scale gripper threshold. Final gripper values below this are clipped to 0.",
    )
    for side in ("left", "right"):
        parser.add_argument(f"--{side}_gripper_threshold", *([f"--{side}_gripper_thrshold"] if side == "left" else []), dest=f"{side}_gripper_threshold", type=float, default=None)
        parser.add_argument(f"--{side}_gripper_lower", type=float, default=None); parser.add_argument(f"--{side}_gripper_upper", type=float, default=None)
    parser.add_argument("--gripper_lower", type=float, default=None)
    parser.add_argument("--gripper_upper", type=float, default=None)
    parser.add_argument(
        "--old_gripper",
        action="store_true",
        help="Use the historical wrong Piper raw-gripper scaling for models trained before the 2lerobot fix.",
    )
    parser.add_argument("--bad-sim", action="store_true", help="Treat gripper outputs as 0-0.05 sim-width and renormalize them to 0-1 before decoding.")
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1000,
        help="Number of action frames to command; 0 means run until Ctrl-C.",
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
        help="streaming uses kai0-style async inference + temporal chunk-wise smoothing; chunk_sync preserves the older blocking loop.",
    )
    parser.add_argument("--inference-rate", type=float, default=None, help="Streaming policy request frequency in Hz; default from config.")
    parser.add_argument("--latency-k", type=int, default=None, help="Max prefix actions to trim from a fresh chunk; default from config.")
    parser.add_argument("--min-smooth-steps", type=int, default=None, help="Minimum old-tail length for overlap smoothing; default from config.")
    parser.add_argument("--buffer-max-chunks", type=int, default=None, help="Action buffer chunk cap; default from config.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override OpenPI-sim denoising steps.")
    parser.add_argument("--metrics-json", default=None, help="Optional path to save rollout timing metrics as JSON.")
    parser.add_argument("--record", action="store_true", help="Record cameras, actions, and states into one deploy video.")
    parser.add_argument("--record-dir", default=str(DEPLOY_ROOT / "artifacts" / "openpi_sim_records"))
    parser.add_argument("--config", default=str(DEPLOY_ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--camera-front-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Infer and decode the first action, but do not command Piper.")
    parser.add_argument("--spec-only", action="store_true", help="Only print the train-config-derived spaces; no server or hardware.")
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    return parser


def _apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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


def _make_runtime(config: dict[str, Any], *, commands_enabled: bool) -> tuple[Any, Any, Any]:
    robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=commands_enabled,
        name="openpi_sim_piper_client",
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


def _normalized_prompt(value: str | None) -> str | None:
    if value is None:
        return None
    prompt = value.strip()
    return prompt or None


def _make_recording_schema(spec: OpenPiSimPolicySpec) -> RecordingSchema:
    return RecordingSchema(
        camera_names=spec.image_ids,
        action_names=SIM_ACTION_NAMES,
        state_names=SIM_STATE_NAMES,
        used_action_names=frozenset(SIM_ACTION_NAMES),
    )


def _record_name_prefix(args: argparse.Namespace) -> str:
    ckpt_name = Path(args.ckpt_dir).name if args.ckpt_dir else args.train_config
    return f"{ckpt_name}_{args.control_mode}_{args.execution_mode}"


def _install_record_signal_handlers() -> None:
    def _raise_keyboard_interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is None:
            continue
        try:
            signal.signal(signal_value, _raise_keyboard_interrupt)
        except (OSError, ValueError):
            pass


def _ignore_record_signal_handlers() -> None:
    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is None:
            continue
        try:
            signal.signal(signal_value, signal.SIG_IGN)
        except (OSError, ValueError):
            pass


def _snapshot_with_grippers(snapshot: RobotSnapshot, grippers: np.ndarray) -> RobotSnapshot:
    grippers = np.asarray(grippers, dtype=np.float64)
    snapshot.state.left.qpos = snapshot.state.left.qpos.copy()
    snapshot.state.right.qpos = snapshot.state.right.qpos.copy()
    snapshot.state.left.qpos[6] = grippers[0]
    snapshot.state.right.qpos[6] = grippers[1]
    return snapshot


def _configured_state_after_command(
    robot: Any,
    spec: OpenPiSimPolicySpec,
    grippers: np.ndarray,
    *,
    old_gripper: bool = False,
) -> np.ndarray:
    snapshot = RobotSnapshot(timestamp_s=time.time(), state=robot.read_state(), images={})
    snapshot = _snapshot_with_grippers(snapshot, grippers)
    return build_configured_piper_state(snapshot, spec, old_gripper=old_gripper)


def _print_rollout_chunk_summary(
    *,
    client: OpenPiSimPiperClient,
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


def run_chunk_sync_rollout(
    *,
    client: OpenPiSimPiperClient,
    source: Any,
    robot: Any,
    spec: OpenPiSimPolicySpec,
    prompt: str,
    rollout_steps: int,
    chunk_size: int | None,
    fps: float,
    recorder: OpenPiRolloutRecorder | None = None,
    saved_actions: list[np.ndarray] | None = None,
    initial_snapshot: Any | None = None,
    initial_grippers: np.ndarray | None = None,
    old_gripper: bool = False,
) -> RolloutMetrics:
    metrics = RolloutMetrics()
    chunk_index = 0
    last_command_start_s: float | None = None
    next_snapshot = initial_snapshot
    last_grippers = np.asarray(
        initial_grippers if initial_grippers is not None else [0.0, 0.0],
        dtype=np.float64,
    )

    try:
        while rollout_steps == 0 or metrics.executed_steps < rollout_steps:
            inference_start_s = time.monotonic()
            chunk_snapshot = next_snapshot if next_snapshot is not None else source.capture_snapshot()
            chunk_snapshot = _snapshot_with_grippers(chunk_snapshot, last_grippers)
            next_snapshot = None
            actions = trim_chunk(client.infer_actions(chunk_snapshot, prompt=prompt), chunk_size)
            metrics.record_inference(time.monotonic() - inference_start_s)

            requested_actions = len(actions)
            if rollout_steps > 0:
                requested_actions = min(requested_actions, rollout_steps - metrics.executed_steps)
            if requested_actions <= 0:
                break

            _print_rollout_chunk_summary(
                client=client,
                chunk_index=chunk_index,
                action_count=requested_actions,
                executed_steps=metrics.executed_steps,
                rollout_steps=rollout_steps,
                first_action=actions[0],
            )
            for action_index, action in enumerate(actions[:requested_actions]):
                action_start_s = time.monotonic()
                period_seconds = None if last_command_start_s is None else action_start_s - last_command_start_s
                last_command_start_s = action_start_s
                frame_snapshot = chunk_snapshot if action_index == 0 else _snapshot_with_grippers(
                    source.capture_snapshot(),
                    last_grippers,
                )
                decoded = client.decode_action(action)
                command_start_s = time.monotonic()
                client.command_action(robot, action)
                if saved_actions is not None:
                    saved_actions.append(np.asarray(action, dtype=np.float64).copy())
                last_grippers = np.array(
                    [decoded.arms["left"].gripper, decoded.arms["right"].gripper],
                    dtype=np.float64,
                )
                if recorder is not None:
                    recorder.record(
                        images=frame_snapshot.images,
                        action=action,
                        state=_configured_state_after_command(
                            robot,
                            spec,
                            last_grippers,
                            old_gripper=old_gripper,
                        ),
                        timestamp_s=time.time(),
                    )
                metrics.record_command(period_seconds=period_seconds, command_seconds=time.monotonic() - command_start_s)
                if rollout_steps > 0 and metrics.executed_steps >= rollout_steps:
                    break
                sleep_until_next_action(action_start_s, fps)
            chunk_index += 1
    except KeyboardInterrupt as exc:
        metrics.mark_interrupted(repr(exc))

    return metrics


def _save_metrics(metrics_summary: dict[str, Any], *, args: argparse.Namespace, recorder: OpenPiRolloutRecorder | None) -> None:
    save_rollout_metrics_summary(
        metrics_summary,
        metrics_json_path=args.metrics_json,
        run_dir=recorder.run_dir if recorder is not None else None,
        record_stem=recorder.record_stem if recorder is not None else None,
    )


def run_once(args: argparse.Namespace) -> None:
    spec = load_openpi_sim_policy_spec(args.train_config)
    print(json.dumps(spec_summary(spec), indent=2), flush=True)
    if args.spec_only:
        return
    cli_prompt = _normalized_prompt(args.prompt)
    if args.rollout_steps < 0:
        raise ValueError("--rollout-steps must be non-negative")
    if args.fps < 0.0:
        raise ValueError("--fps must be non-negative")
    if args.gripper_threshold is not None and args.gripper_threshold < 0.0:
        raise ValueError("--gripper_threshold must be non-negative")
    if args.gripper_threshold is not None and (args.gripper_lower is not None or args.gripper_upper is not None):
        raise ValueError("--gripper_threshold cannot be combined with --gripper_lower/--gripper_upper")
    if args.inference_rate is not None and args.inference_rate < 0.0:
        raise ValueError("--inference-rate must be non-negative")
    if args.execution_mode != "chunk_sync":
        raise NotImplementedError("openpi_sim EmbodiChain runner currently supports --execution-mode chunk_sync only")

    asset_info = dataset_asset_info(args.train_config); record_assets = prepare_train_assets(train_config_name=args.train_config, cli_prompt=cli_prompt) if (args.record or args.window) else None
    resolved_prompt, prompt_source = (record_assets.prompt, record_assets.prompt_source) if record_assets is not None else resolve_prompt(train_config_name=args.train_config, cli_prompt=cli_prompt, dataset_dir=asset_info.dataset_dir)

    if resolved_prompt is None:
        raise RuntimeError(
            "No prompt available. Provide --prompt, or ensure the train config's LeRobot dataset exists "
            "and has a cached/discoverable task prompt."
        )
    print(
        json.dumps(
            {
                "prompt": {
                    "value": resolved_prompt,
                    "source": prompt_source,
                }
            },
            indent=2,
        ),
        flush=True,
    )

    client = OpenPiSimPiperClient(
        args.train_config,
        host=args.host,
        port=args.port,
        control_mode=args.control_mode,
        api_key=args.api_key,
        joint_speed_percent=args.joint_speed_percent,
        gripper_threshold=args.gripper_threshold,
        num_steps=args.num_steps,
        old_gripper=args.old_gripper,
        bad_sim=args.bad_sim,
    )
    client.left_gripper_threshold, client.right_gripper_threshold, client.left_gripper_lower, client.left_gripper_upper, client.right_gripper_lower, client.right_gripper_upper = args.left_gripper_threshold, args.right_gripper_threshold, args.left_gripper_lower, args.left_gripper_upper, args.right_gripper_lower, args.right_gripper_upper
    client.gripper_lower, client.gripper_upper = args.gripper_lower, args.gripper_upper

    runtime_config = _apply_runtime_overrides(load_config(args.config), args)
    robot, cameras, source = _make_runtime(runtime_config, commands_enabled=not args.dry_run)
    recording_schema = _make_recording_schema(spec)
    saved_actions: list[np.ndarray] | None = [] if args.record else None
    recorder = (
        OpenPiRolloutRecorder(
            output_dir=args.record_dir,
            schema=recording_schema,
            fps=args.fps,
            name_prefix=_record_name_prefix(args),
        )
        if args.record
        else None
    )
    if recorder is not None:
        _install_record_signal_handlers()

    first_obs_snapshot = None
    frame1_path = None
    metrics = None
    robot.connect(read_only=args.dry_run)
    try:
        if cameras is not None:
            cameras.start()
        if not source.wait_until_ready(timeout_s=args.ready_timeout):
            raise RuntimeError("Timed out waiting for Piper/RealSense data")

        if args.dry_run:
            snapshot = source.capture_snapshot()
            first_obs_snapshot = snapshot
            actions = action_sequence(client.infer_actions(snapshot, prompt=resolved_prompt))
            if recorder is not None:
                recorder.record(
                    images=snapshot.images,
                    action=actions[0],
                    state=build_configured_piper_state(snapshot, spec, old_gripper=args.old_gripper),
                    timestamp_s=snapshot.timestamp_s,
                )
            if saved_actions is not None:
                saved_actions.append(actions[0].copy())
            if recorder is not None:
                distribution_image_path = None if record_assets is None else record_assets.distribution_image_path
                frame1_path = save_frame1_image(
                    recorder,
                    first_obs_snapshot,
                    distribution_image_path=distribution_image_path,
                )
                if frame1_path is not None:
                    print(f"Frame1 image saved to {frame1_path}", flush=True)
            if args.window:
                distribution_image_path = None if record_assets is None else record_assets.distribution_image_path
                preview_until_continue(source, distribution_image_path=distribution_image_path)
            print(json.dumps(decoded_action_summary(client.decode_action(actions[0])), indent=2), flush=True)
            return

        print('{"hardware_init": "enable_dual_piper"}', flush=True)
        if not robot.enable():
            print("Warning: Piper arm enable check did not report success; continuing anyway.", flush=True)

        chunk_size = resolve_chunk_size(spec, args.chunk_size)
        initial_joints = _initial_piper_joints(old_gripper=args.old_gripper)
        print(json.dumps({"initial_pose": {"qpos": initial_joints.tolist()}}, indent=2), flush=True)
        robot.move_to_joint_positions(initial_joints, speed_percent=args.joint_speed_percent)
        first_obs_snapshot = _snapshot_with_grippers(source.capture_snapshot(), initial_joints[[6, 13]])
        if recorder is not None:
            distribution_image_path = None if record_assets is None else record_assets.distribution_image_path
            frame1_path = save_frame1_image(
                recorder,
                first_obs_snapshot,
                distribution_image_path=distribution_image_path,
            )
            if frame1_path is not None:
                print(f"Frame1 image saved to {frame1_path}", flush=True)
        if args.window:
            distribution_image_path = None if record_assets is None else record_assets.distribution_image_path
            preview_until_continue(source, distribution_image_path=distribution_image_path)
        print(
            json.dumps(
                {
                    "rollout": {
                        "execution_mode": args.execution_mode,
                        "rollout_steps": args.rollout_steps,
                        "chunk_size": chunk_size,
                        "fps": args.fps,
                        "joint_speed_percent": args.joint_speed_percent,
                        "ee_speed_percent": args.ee_speed_percent,
                        "gripper_threshold": args.gripper_threshold,
                        "old_gripper": args.old_gripper,
                    }
                },
                indent=2,
            ),
            flush=True,
        )

        metrics = run_chunk_sync_rollout(
            client=client,
            source=source,
            robot=robot,
            spec=spec,
            prompt=resolved_prompt,
            rollout_steps=args.rollout_steps,
            chunk_size=chunk_size,
            fps=args.fps,
            recorder=recorder,
            saved_actions=saved_actions,
            initial_snapshot=first_obs_snapshot,
            initial_grippers=initial_joints[[6, 13]],
            old_gripper=args.old_gripper,
        )
        if metrics.interrupted:
            print("Interrupted by user; stopping rollout.", flush=True)
        metrics_summary = metrics.summary()
        print(json.dumps({"rollout_metrics": metrics_summary}, indent=2), flush=True)
        _save_metrics(metrics_summary, args=args, recorder=recorder)
    except KeyboardInterrupt:
        print("Interrupted by user; stopping rollout.", flush=True)
    finally:
        if recorder is not None:
            _ignore_record_signal_handlers()
        if cameras is not None:
            try:
                cameras.stop()
            except Exception as exc:
                print(f"Failed to stop cameras cleanly: {exc}", flush=True)
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"Failed to disconnect robot cleanly: {exc}", flush=True)
        if recorder is not None:
            try:
                action_path = recorder.run_dir / f"{recorder.record_stem}_actions.npz"
                action_trajectory = np.stack(saved_actions, axis=0) if saved_actions else np.empty((0, len(recording_schema.action_names)), dtype=np.float64)
                np.savez_compressed(action_path, action_mean_trajectory=action_trajectory, action_names=np.asarray(recording_schema.action_names))
                print(f"Actions saved to {action_path}", flush=True)
            except Exception as exc:
                print(f"Failed to save actions: {exc}", flush=True)
            output_path = None
            try:
                output_path = recorder.finalize()
            except Exception as exc:
                print(f"Failed to finalize recording: {exc}", flush=True)
            if output_path is not None:
                print(f"Recording saved to {output_path}", flush=True)
                if frame1_path is None:
                    try:
                        frame1_path = save_frame1_image(recorder, first_obs_snapshot)
                        if frame1_path is not None:
                            print(f"Frame1 image saved to {frame1_path}", flush=True)
                    except Exception as exc:
                        print(f"Failed to save frame1 image: {exc}", flush=True)


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
