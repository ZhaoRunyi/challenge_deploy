from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from hardware.config import load_config
from rollout.assets import prepare_client_assets
from clients.openpi_sim import (
    SIM_ACTION_NAMES,
    SIM_STATE_NAMES,
    OpenPiSimPiperClient,
    OpenPiSimPolicySpec,
    build_configured_piper_state,
    load_openpi_sim_policy_spec,
    spec_summary,
)
from rollout.recording import RolloutVideoRecorder, RecordingSchema, preview_until_continue, save_frame1_image, save_recorded_actions
from hardware.schemas import RobotSnapshot
from rollout.execution import (
    RolloutMetrics,
    action_sequence,
    resolve_chunk_size,
    run_temporal_smoothing_rollout,
    save_rollout_metrics,
    sleep_until_next_action,
    trim_chunk,
)
from rollout.support import (
    apply_runtime_overrides,
    decoded_action_summary,
    ignore_recorder_signal_handlers,
    install_recorder_signal_handlers,
    make_dual_piper_runtime,
    normalized_prompt,
    print_rollout_chunk_summary,
    record_name_prefix,
    resolve_dual_piper_init_joints,
)

DEPLOY_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenPI-sim EmbodiChain dual Piper client: fixed 14D joints+gripper01 action space."
    )
    parser.add_argument("--train-config", required=True, help="OpenPI train config name, e.g. pi0_slai_piper_template.")
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
        typo_aliases = [f"--{side}_gripper_thrshold"] if side == "left" else []
        parser.add_argument(
            f"--{side}_gripper_threshold",
            *typo_aliases,
            dest=f"{side}_gripper_threshold",
            type=float,
            default=None,
        )
        parser.add_argument(f"--{side}_gripper_lower", type=float, default=None)
        parser.add_argument(f"--{side}_gripper_upper", type=float, default=None)
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
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Infer and decode the first action, but do not command Piper.")
    parser.add_argument("--spec-only", action="store_true", help="Only print the train-config-derived spaces; no server or hardware.")
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    return parser


def make_recording_schema(spec: OpenPiSimPolicySpec) -> RecordingSchema:
    return RecordingSchema(
        camera_names=spec.image_ids,
        action_names=SIM_ACTION_NAMES,
        state_names=SIM_STATE_NAMES,
        used_action_names=frozenset(SIM_ACTION_NAMES),
    )


def snapshot_with_grippers(snapshot: RobotSnapshot, grippers: np.ndarray) -> RobotSnapshot:
    grippers = np.asarray(grippers, dtype=np.float64)
    snapshot.state.left.qpos = snapshot.state.left.qpos.copy()
    snapshot.state.right.qpos = snapshot.state.right.qpos.copy()
    snapshot.state.left.qpos[6] = grippers[0]
    snapshot.state.right.qpos[6] = grippers[1]
    return snapshot


def configured_state_after_command(
    robot: Any,
    spec: OpenPiSimPolicySpec,
    grippers: np.ndarray,
    *,
    old_gripper: bool = False,
) -> np.ndarray:
    snapshot = RobotSnapshot(timestamp_s=time.time(), state=robot.read_state(), images={})
    snapshot = snapshot_with_grippers(snapshot, grippers)
    return build_configured_piper_state(snapshot, spec, old_gripper=old_gripper)


class OpenPiSimStreamingAdapter:
    def __init__(self, client: OpenPiSimPiperClient, source: Any, initial_grippers: np.ndarray) -> None:
        self.client = client
        self.source = source
        self.last_grippers = np.asarray(initial_grippers, dtype=np.float64)

    def capture_snapshot(self) -> RobotSnapshot:
        return snapshot_with_grippers(self.source.capture_snapshot(), self.last_grippers)

    def infer_actions(self, snapshot: RobotSnapshot, prompt: str) -> np.ndarray:
        return self.client.infer_actions(snapshot, prompt=prompt)

    def command_action(self, robot: Any, action: np.ndarray) -> None:
        decoded = self.client.decode_action(action)
        self.client.command_action(robot, action)
        self.last_grippers = np.array([decoded.arms["left"].gripper, decoded.arms["right"].gripper], dtype=np.float64)


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
    recorder: RolloutVideoRecorder | None = None,
    saved_actions: list[np.ndarray] | None = None,
    initial_snapshot: Any | None = None,
    initial_grippers: np.ndarray | None = None,
    old_gripper: bool = False,
) -> RolloutMetrics:
    metrics = RolloutMetrics(execution_mode="chunk_sync")
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
            chunk_snapshot = snapshot_with_grippers(chunk_snapshot, last_grippers)
            next_snapshot = None
            actions = trim_chunk(client.infer_actions(chunk_snapshot, prompt=prompt), chunk_size)
            metrics.record_inference(time.monotonic() - inference_start_s)

            requested_actions = len(actions)
            if rollout_steps > 0:
                requested_actions = min(requested_actions, rollout_steps - metrics.executed_steps)
            if requested_actions <= 0:
                break

            print_rollout_chunk_summary(
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
                frame_snapshot = chunk_snapshot if action_index == 0 else snapshot_with_grippers(
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
                        state=configured_state_after_command(
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


def run_once(args: argparse.Namespace) -> None:
    spec = load_openpi_sim_policy_spec(args.train_config)
    print(json.dumps(spec_summary(spec), indent=2), flush=True)
    if args.spec_only:
        return
    cli_prompt = normalized_prompt(args.prompt)
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

    client_assets = prepare_client_assets(
        client_kind="openpi_sim",
        train_config_name=args.train_config,
        cli_prompt=cli_prompt,
        need_distribution=args.record or args.window,
    )
    resolved_prompt = client_assets.prompt
    prompt_source = client_assets.prompt_source

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
    server_metadata = client.get_server_metadata()
    print(json.dumps({"server_metadata": server_metadata}, indent=2), flush=True)

    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    robot, cameras, source = make_dual_piper_runtime(
        runtime_config,
        commands_enabled=not args.dry_run,
        name="openpi_sim_piper_client",
    )
    recording_schema = make_recording_schema(spec)
    saved_actions: list[np.ndarray] | None = [] if args.record else None
    recorder = (
        RolloutVideoRecorder(
            output_dir=args.record_dir,
            schema=recording_schema,
            fps=args.fps,
            name_prefix=record_name_prefix(args, server_metadata),
        )
        if args.record
        else None
    )
    install_recorder_signal_handlers(recorder)

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
                frame1_path = save_frame1_image(
                    recorder,
                    first_obs_snapshot,
                    distribution_image_path=client_assets.distribution_image_path,
                )
                if frame1_path is not None:
                    print(f"Frame1 image saved to {frame1_path}", flush=True)
            if args.window:
                preview_until_continue(source, distribution_image_path=client_assets.distribution_image_path)
            print(json.dumps(decoded_action_summary(client.decode_action(actions[0])), indent=2), flush=True)
            return

        print('{"hardware_init": "enable_dual_piper"}', flush=True)
        if not robot.enable():
            print("Warning: Piper arm enable check did not report success; continuing anyway.", flush=True)

        chunk_size = resolve_chunk_size(spec, args.chunk_size)
        initial_joints = resolve_dual_piper_init_joints(args.init_joints)
        print(json.dumps({"initial_pose": {"qpos": initial_joints.tolist()}}, indent=2), flush=True)
        robot.move_to_joint_positions(initial_joints, speed_percent=args.joint_speed_percent)
        first_obs_snapshot = snapshot_with_grippers(source.capture_snapshot(), initial_joints[[6, 13]])
        if recorder is not None:
            frame1_path = save_frame1_image(
                recorder,
                first_obs_snapshot,
                distribution_image_path=client_assets.distribution_image_path,
            )
            if frame1_path is not None:
                print(f"Frame1 image saved to {frame1_path}", flush=True)
        if args.window:
            preview_until_continue(source, distribution_image_path=client_assets.distribution_image_path)
        inference_rate = float(args.inference_rate if args.inference_rate is not None else runtime_config["policy"]["inference_rate"])
        latency_k = int(args.latency_k if args.latency_k is not None else runtime_config["policy"]["latency_k"])
        min_smooth_steps = int(args.min_smooth_steps if args.min_smooth_steps is not None else runtime_config["policy"]["min_smooth_steps"])
        buffer_max_chunks = int(args.buffer_max_chunks if args.buffer_max_chunks is not None else runtime_config["policy"]["buffer_max_chunks"])
        print(
            json.dumps(
                {
                    "rollout": {
                        "execution_mode": args.execution_mode,
                        "rollout_steps": args.rollout_steps,
                        "chunk_size": chunk_size,
                        "fps": args.fps,
                        "inference_rate": inference_rate if args.execution_mode == "streaming" else None,
                        "latency_k": latency_k if args.execution_mode == "streaming" else None,
                        "min_smooth_steps": min_smooth_steps if args.execution_mode == "streaming" else None,
                        "buffer_max_chunks": buffer_max_chunks if args.execution_mode == "streaming" else None,
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

        if args.execution_mode == "streaming":
            adapter = OpenPiSimStreamingAdapter(client, source, initial_joints[[6, 13]])

            def log_chunk(chunk_index: int, action_count: int, executed_steps: int, first_action: np.ndarray) -> None:
                print_rollout_chunk_summary(
                    client=client,
                    chunk_index=chunk_index,
                    action_count=action_count,
                    executed_steps=executed_steps,
                    rollout_steps=args.rollout_steps,
                    first_action=first_action,
                )

            metrics = run_temporal_smoothing_rollout(
                client=adapter,
                source=adapter,
                robot=robot,
                spec=spec,
                prompt=resolved_prompt,
                rollout_steps=args.rollout_steps,
                chunk_size=chunk_size,
                fps=args.fps,
                inference_rate=inference_rate,
                latency_k=latency_k,
                min_smooth_steps=min_smooth_steps,
                buffer_max_chunks=buffer_max_chunks,
                recorder=recorder,
                saved_actions=saved_actions,
                log_chunk=log_chunk,
                initial_snapshot=first_obs_snapshot,
                state_builder=lambda snapshot, policy_spec: configured_state_after_command(
                    robot,
                    policy_spec,
                    adapter.last_grippers,
                    old_gripper=args.old_gripper,
                ),
            )
        else:
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
        metrics_summary, written_metric_paths = save_rollout_metrics(
            metrics,
            metrics_json_path=args.metrics_json,
            run_dir=recorder.run_dir if recorder is not None else None,
            record_stem=recorder.record_stem if recorder is not None else None,
        )
        print(json.dumps({"rollout_metrics": metrics_summary}, indent=2), flush=True)
        for metrics_path in written_metric_paths:
            print(f"Rollout metrics saved to {metrics_path}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted by user; stopping rollout.", flush=True)
    finally:
        ignore_recorder_signal_handlers(recorder)
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
                action_path = save_recorded_actions(recorder, saved_actions, recording_schema.action_names)
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
                        frame1_path = save_frame1_image(
                            recorder,
                            first_obs_snapshot,
                            distribution_image_path=client_assets.distribution_image_path,
                        )
                        if frame1_path is not None:
                            print(f"Frame1 image saved to {frame1_path}", flush=True)
                        elif client_assets.skip_reason is not None:
                            print(f"Skipped train-distribution frame1 image: {client_assets.skip_reason}", flush=True)
                    except Exception as exc:
                        print(f"Failed to save frame1 image: {exc}", flush=True)


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
