from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from hardware.config import load_config
from rollout.assets import prepare_client_assets
from clients.motus import (
    MotusPiperClient,
    build_configured_piper_state,
    load_motus_policy_spec,
    spec_summary,
)
from rollout.execution import (
    action_sequence,
    resolve_chunk_size,
    run_chunk_sync_rollout,
    save_rollout_metrics,
    run_temporal_smoothing_rollout,
)
from rollout.recording import RolloutVideoRecorder, preview_until_continue, save_frame1_image
from rollout.support import (
    apply_runtime_overrides,
    decoded_action_summary,
    ignore_record_signal_handlers,
    install_record_signal_handlers,
    make_dual_piper_runtime,
    make_slai_recording_schema,
    normalized_prompt,
    print_rollout_chunk_summary,
    record_name_prefix,
    resolve_dual_piper_init_joints,
)


DEPLOY_ROOT = Path(__file__).resolve().parent


def new_session_id() -> str:
    return f"motus_{time.strftime('%Y%m%d_%H%M%S')}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Motus SLAI Piper client: capture snapshots, infer chunks, and command Piper in a rollout."
    )
    parser.add_argument(
        "--train-config",
        required=True,
        help="Motus YAML config path, e.g. baselines/Motus/configs/piper_click_bell_0403_robotwin_like.yaml.",
    )
    parser.add_argument("--ckpt-dir", default=None, help="Checkpoint directory used only for record video filenames.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--control-mode", choices=["joints", "ee_pose"], default="joints")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--joint-speed-percent", type=int, default=100)
    parser.add_argument("--ee-speed-percent", type=int, default=100)
    parser.add_argument(
        "--gripper-effort",
        type=int,
        default=1000,
        help="Piper SDK GripperCtrl effort in [0, 5000].",
    )
    parser.add_argument(
        "--gripper-action-frames",
        type=int,
        default=3,
        help=(
            "Only used when gripper commands are binarized. Open/close transitions are executed linearly "
            "across this many command frames while the other joints stay frozen."
        ),
    )
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=None,
        help="Optional executable-scale gripper threshold in meters. Values below threshold close the gripper, and values above threshold command full open.",
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
    parser.add_argument("--num-inference-timesteps", type=int, default=None, help="Override Motus denoising steps for server inference.")
    parser.add_argument("--metrics-json", default=None, help="Optional path to save rollout timing metrics as JSON.")
    parser.add_argument("--record", action="store_true", help="Record cameras, actions, and states into one deploy video.")
    parser.add_argument("--record-dir", default=str(DEPLOY_ROOT / "artifacts" / "motus_records"))
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
    parser.add_argument("--camera-front-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Infer and decode the first action, but do not command Piper.")
    parser.add_argument("--spec-only", action="store_true", help="Only print the train-config-derived spaces; no server or hardware.")
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    return parser


def run_once(args: argparse.Namespace) -> None:
    spec = load_motus_policy_spec(args.train_config)
    print(json.dumps(spec_summary(spec), indent=2))
    if args.spec_only:
        return
    cli_prompt = normalized_prompt(args.prompt)
    if args.rollout_steps < 0:
        raise ValueError("--rollout-steps must be non-negative")
    if args.fps < 0.0:
        raise ValueError("--fps must be non-negative")
    if not 0 <= args.gripper_effort <= 5000:
        raise ValueError("--gripper-effort must be in [0, 5000]")
    if args.gripper_action_frames <= 0:
        raise ValueError("--gripper-action-frames must be positive")
    if args.gripper_threshold is not None and args.gripper_threshold < 0.0:
        raise ValueError("--gripper_threshold must be non-negative")
    if args.gripper_threshold is not None and (args.gripper_lower is not None or args.gripper_upper is not None):
        raise ValueError("--gripper_threshold cannot be combined with --gripper_lower/--gripper_upper")
    if args.inference_rate is not None and args.inference_rate < 0.0:
        raise ValueError("--inference-rate must be non-negative")
    client = MotusPiperClient(
        args.train_config,
        host=args.host,
        port=args.port,
        control_mode=args.control_mode,
        api_key=args.api_key,
        joint_speed_percent=args.joint_speed_percent,
        ee_speed_percent=args.ee_speed_percent,
        gripper_effort=args.gripper_effort,
        gripper_action_frames=args.gripper_action_frames,
        gripper_threshold=args.gripper_threshold,
        gripper_lower=args.gripper_lower,
        gripper_upper=args.gripper_upper,
        num_inference_timesteps=args.num_inference_timesteps,
        old_gripper=args.old_gripper,
    )
    client.left_gripper_threshold, client.right_gripper_threshold, client.left_gripper_lower, client.left_gripper_upper, client.right_gripper_lower, client.right_gripper_upper = args.left_gripper_threshold, args.right_gripper_threshold, args.left_gripper_lower, args.left_gripper_upper, args.right_gripper_lower, args.right_gripper_upper
    state_builder = lambda snapshot, policy_spec: build_configured_piper_state(
        snapshot,
        policy_spec,
        old_gripper=args.old_gripper,
    )
    server_metadata = client.get_server_metadata()
    print(json.dumps({"server_metadata": server_metadata}, indent=2), flush=True)

    client_assets = prepare_client_assets(
        client_kind="motus",
        train_config_name=args.train_config,
        cli_prompt=cli_prompt,
        need_distribution=args.record or args.window,
        spec=spec,
        server_metadata=server_metadata,
    )
    resolved_prompt = client_assets.prompt
    prompt_source = client_assets.prompt_source
    if resolved_prompt is None:
        raise RuntimeError(
            "No prompt available. Provide --prompt, or ensure the remote Motus server was started with --default_prompt."
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

    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    robot, cameras, source = make_dual_piper_runtime(
        runtime_config,
        commands_enabled=not args.dry_run,
        name="motus_piper_client",
    )
    recording_schema = make_slai_recording_schema(spec, args.control_mode)
    saved_actions: list[np.ndarray] | None = [] if args.record else None
    recorder = (
        RolloutVideoRecorder(
            output_dir=args.record_dir,
            schema=recording_schema,
            fps=args.fps,
            name_prefix=record_name_prefix(args),
        )
        if args.record
        else None
    )
    if recorder is not None:
        install_record_signal_handlers()

    first_obs_snapshot = None
    frame1_path = None
    distribution_image_path = client_assets.distribution_image_path
    distribution_skip_reason = client_assets.skip_reason
    session_id = new_session_id()
    client.set_default_session_id(session_id)
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
                    state=state_builder(snapshot, spec),
                    timestamp_s=snapshot.timestamp_s,
                )
            if saved_actions is not None:
                saved_actions.append(actions[0].copy())
            if recorder is not None:
                frame1_path = save_frame1_image(
                    recorder,
                    snapshot,
                    distribution_image_path=distribution_image_path,
                    preferred_names=("cam_high",) + tuple(spec.image_ids),
                )
                if frame1_path is not None:
                    print(f"Frame1 image saved to {frame1_path}", flush=True)
            if args.window:
                preview_until_continue(source, distribution_image_path=distribution_image_path)
            print(json.dumps(decoded_action_summary(client.decode_action(actions[0])), indent=2))
            return

        print('{"hardware_init": "enable_dual_piper"}', flush=True)
        if not robot.enable():
            print("Warning: Piper arm enable check did not report success; continuing anyway.", flush=True)

        initial_joints = resolve_dual_piper_init_joints(args.init_joints)
        print(json.dumps({"initial_pose": {"qpos": initial_joints.tolist()}}, indent=2), flush=True)
        robot.move_to_joint_positions(
            initial_joints,
            speed_percent=args.joint_speed_percent,
            gripper_effort=args.gripper_effort,
        )
        first_obs_snapshot = source.capture_snapshot()
        if recorder is not None:
            frame1_path = save_frame1_image(
                recorder,
                first_obs_snapshot,
                distribution_image_path=distribution_image_path,
                preferred_names=("cam_high",) + tuple(spec.image_ids),
            )
            if frame1_path is not None:
                print(f"Frame1 image saved to {frame1_path}", flush=True)
        if args.window:
            preview_until_continue(source, distribution_image_path=distribution_image_path)
        chunk_size = resolve_chunk_size(spec, args.chunk_size)
        inference_rate = (
            float(args.inference_rate)
            if args.inference_rate is not None
            else float(runtime_config["policy"]["inference_rate"])
        )
        latency_k = (
            int(args.latency_k)
            if args.latency_k is not None
            else int(runtime_config["policy"]["latency_k"])
        )
        min_smooth_steps = (
            int(args.min_smooth_steps)
            if args.min_smooth_steps is not None
            else int(runtime_config["policy"]["min_smooth_steps"])
        )
        buffer_max_chunks = (
            int(args.buffer_max_chunks)
            if args.buffer_max_chunks is not None
            else int(runtime_config["policy"]["buffer_max_chunks"])
        )

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
                        "gripper_effort": args.gripper_effort,
                        "gripper_action_frames": args.gripper_action_frames,
                        "gripper_threshold": args.gripper_threshold,
                        "gripper_lower": args.gripper_lower,
                        "gripper_upper": args.gripper_upper,
                        "old_gripper": args.old_gripper,
                    }
                },
                indent=2,
            ),
            flush=True,
        )

        def log_chunk(chunk_index: int, action_count: int, executed_steps: int, first_action: np.ndarray) -> None:
            print_rollout_chunk_summary(
                client=client,
                chunk_index=chunk_index,
                action_count=action_count,
                executed_steps=executed_steps,
                rollout_steps=args.rollout_steps,
                first_action=first_action,
            )

        if args.execution_mode == "streaming":
            metrics = run_temporal_smoothing_rollout(
                client=client,
                source=source,
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
                state_builder=state_builder,
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
                log_chunk=log_chunk,
                initial_snapshot=first_obs_snapshot,
                state_builder=state_builder,
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
        ignore_record_signal_handlers()
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
                try:
                    predicted_video_path = client.save_predicted_video(
                        session_id=session_id,
                        output_dir=recorder.run_dir,
                        file_stem=recorder.record_stem,
                    )
                    if predicted_video_path is not None:
                        print(f"Predicted video saved to {predicted_video_path}", flush=True)
                except Exception as exc:
                    print(f"Failed to save predicted video: {exc}", flush=True)
                try:
                    if frame1_path is None:
                        frame1_path = save_frame1_image(
                            recorder,
                            first_obs_snapshot,
                            distribution_image_path=distribution_image_path,
                            preferred_names=("cam_high",) + tuple(spec.image_ids),
                        )
                    if frame1_path is not None:
                        print(f"Frame1 image saved to {frame1_path}", flush=True)
                    elif distribution_skip_reason is not None:
                        print(f"Skipped train-distribution frame1 image: {distribution_skip_reason}", flush=True)
                except Exception as exc:
                    print(f"Failed to save frame1 image: {exc}", flush=True)


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
