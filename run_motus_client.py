from __future__ import annotations

import argparse
import cv2
import json
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.lerobot_assets import repo_id_distribution_image_path
from challenge_deploy.motus_client import (
    ControlMode,
    MotusPiperClient,
    MotusPolicySpec,
    _motus_slai_policy,
    build_configured_piper_state,
    decoded_action_summary,
    load_motus_policy_spec,
    spec_summary,
)
from challenge_deploy.openpi_rollout import (
    action_sequence,
    resolve_chunk_size,
    run_chunk_sync_rollout,
    run_temporal_smoothing_rollout,
)
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.realsense import RealSenseRig
from challenge_deploy.recording import OpenPiRolloutRecorder, RecordingSchema, stack_vertical
from challenge_deploy.runtime import DualPiperObservationSource


DEPLOY_ROOT = Path(__file__).resolve().parent
INIT_JOINTS = np.array(
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


def _used_action_names(spec: MotusPolicySpec, control_mode: ControlMode) -> frozenset[str]:
    slai_policy = _motus_slai_policy()
    action_space = slai_policy._space_from_action_config(spec.action_space)
    names = slai_policy.get_vector_names(spec.action_space)
    slices = slai_policy._field_slices_from_space(action_space)
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


def _make_recording_schema(spec: MotusPolicySpec, control_mode: ControlMode) -> RecordingSchema:
    slai_policy = _motus_slai_policy()
    return RecordingSchema(
        camera_names=spec.image_ids,
        action_names=tuple(slai_policy.get_vector_names(spec.action_space)),
        state_names=tuple(slai_policy.get_vector_names(spec.state_space)),
        used_action_names=_used_action_names(spec, control_mode),
    )


def _record_name_prefix(args: argparse.Namespace) -> str:
    ckpt_name = Path(args.ckpt_dir).name if args.ckpt_dir else Path(args.train_config).stem
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


def _new_session_id() -> str:
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
    parser.add_argument("--joint-speed-percent", type=int, default=50)
    parser.add_argument("--ee-speed-percent", type=int, default=50)
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=None,
        help="Optional executable-scale gripper threshold. Final gripper values below this are clipped to 0.",
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
    parser.add_argument("--metrics-json", default=None, help="Optional path to save rollout timing metrics as JSON.")
    parser.add_argument("--record", action="store_true", help="Record cameras, actions, and states into one deploy video.")
    parser.add_argument("--record-dir", default=str(DEPLOY_ROOT / "artifacts" / "motus_records"))
    parser.add_argument("--config", default=str(DEPLOY_ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--camera-front-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Infer and decode the first action, but do not command Piper.")
    parser.add_argument("--spec-only", action="store_true", help="Only print the train-config-derived spaces; no server or hardware.")
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    return parser


def _normalized_prompt(value: str | None) -> str | None:
    if value is None:
        return None
    prompt = value.strip()
    return prompt or None


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
        name="motus_piper_client",
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


def _print_rollout_chunk_summary(
    *,
    client: MotusPiperClient,
    chunk_index: int,
    action_count: int,
    executed_steps: int,
    rollout_steps: int,
    first_action: np.ndarray,
) -> None:
    summary = decoded_action_summary(client.decode_action(first_action))
    target = "unlimited" if rollout_steps == 0 else str(rollout_steps)
    print(
        json.dumps(
            {
                "rollout_chunk": chunk_index,
                "actions_in_chunk": action_count,
                "executed_steps": executed_steps,
                "target_steps": target,
                "first_action": summary,
            },
            indent=2,
        ),
        flush=True,
    )


def _save_first_frame_image(
    *,
    recorder: OpenPiRolloutRecorder,
    snapshot: Any | None,
    spec: MotusPolicySpec,
    distribution_image_path: Path | None = None,
) -> Path | None:
    if snapshot is None:
        return None
    if distribution_image_path is not None and "cam_high" in snapshot.images:
        distribution_image = cv2.imread(str(distribution_image_path), cv2.IMREAD_COLOR)
        if distribution_image is None:
            raise RuntimeError(f"Failed to read train distribution image: {distribution_image_path}")
        comparison = stack_vertical(distribution_image, snapshot.images["cam_high"])
        return recorder.save_extra_image(comparison, suffix="frame1")
    preferred_names = ("cam_high",) + tuple(spec.image_ids)
    for image_name in preferred_names:
        image = snapshot.images.get(image_name)
        if image is None:
            continue
        return recorder.save_extra_image(image, suffix="frame1")
    return None


def _record_repo_id_for_distribution(spec: MotusPolicySpec) -> str | None:
    if not spec.repo_id:
        return None
    owner, sep, dataset_name = str(spec.repo_id).partition("/")
    if sep and dataset_name.startswith("Motus_"):
        return f"{owner}/{dataset_name[len('Motus_'):]}"
    return str(spec.repo_id)


def _resolve_motus_distribution_image(spec: MotusPolicySpec) -> tuple[Path | None, str | None]:
    repo_id = _record_repo_id_for_distribution(spec)
    if repo_id is None:
        return None, "train config does not define dataset.params.repo_id"
    distribution_image_path = repo_id_distribution_image_path(repo_id)
    if distribution_image_path.exists():
        return distribution_image_path, None
    return None, f"train distribution image not found: {distribution_image_path}"


def run_once(args: argparse.Namespace) -> None:
    spec = load_motus_policy_spec(args.train_config)
    print(json.dumps(spec_summary(spec), indent=2))
    if args.spec_only:
        return
    cli_prompt = _normalized_prompt(args.prompt)
    if args.rollout_steps < 0:
        raise ValueError("--rollout-steps must be non-negative")
    if args.fps < 0.0:
        raise ValueError("--fps must be non-negative")
    if args.gripper_threshold is not None and args.gripper_threshold < 0.0:
        raise ValueError("--gripper_threshold must be non-negative")
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
        gripper_threshold=args.gripper_threshold,
    )
    server_metadata = client.get_server_metadata()
    print(json.dumps({"server_metadata": server_metadata}, indent=2), flush=True)

    resolved_prompt = cli_prompt or server_metadata.get("default_prompt")
    prompt_source = "cli" if cli_prompt is not None else "server_default"
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

    runtime_config = _apply_runtime_overrides(load_config(args.config), args)
    robot, cameras, source = _make_runtime(runtime_config, commands_enabled=not args.dry_run)
    recorder = (
        OpenPiRolloutRecorder(
            output_dir=args.record_dir,
            schema=_make_recording_schema(spec, args.control_mode),
            fps=args.fps,
            name_prefix=_record_name_prefix(args),
        )
        if args.record
        else None
    )
    if recorder is not None:
        _install_record_signal_handlers()

    first_obs_snapshot = None
    distribution_image_path = None
    distribution_skip_reason = None
    if recorder is not None:
        distribution_image_path, distribution_skip_reason = _resolve_motus_distribution_image(spec)
    session_id = _new_session_id()
    client.set_default_session_id(session_id)
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
                    state=build_configured_piper_state(snapshot, spec),
                    timestamp_s=snapshot.timestamp_s,
                )
            print(json.dumps(decoded_action_summary(client.decode_action(actions[0])), indent=2))
            return

        print('{"hardware_init": "enable_dual_piper"}', flush=True)
        if not robot.enable():
            print("Warning: Piper arm enable check did not report success; continuing anyway.", flush=True)

        print(json.dumps({"initial_pose": {"qpos": INIT_JOINTS.tolist()}}, indent=2), flush=True)
        robot.move_to_joint_positions(INIT_JOINTS, speed_percent=args.joint_speed_percent)
        first_obs_snapshot = source.capture_snapshot()
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
                        "gripper_threshold": args.gripper_threshold,
                    }
                },
                indent=2,
            ),
            flush=True,
        )

        def log_chunk(chunk_index: int, action_count: int, executed_steps: int, first_action: np.ndarray) -> None:
            _print_rollout_chunk_summary(
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
                log_chunk=log_chunk,
                initial_snapshot=first_obs_snapshot,
                state_builder=build_configured_piper_state,
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
                log_chunk=log_chunk,
                initial_snapshot=first_obs_snapshot,
                state_builder=build_configured_piper_state,
            )
        metrics_summary = metrics.summary()
        print(json.dumps({"rollout_metrics": metrics_summary}, indent=2), flush=True)
        if args.metrics_json:
            metrics_path = Path(args.metrics_json)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    except KeyboardInterrupt:
        print("Interrupted by user; stopping rollout.", flush=True)
    finally:
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
                    frame1_path = _save_first_frame_image(
                        recorder=recorder,
                        snapshot=first_obs_snapshot,
                        spec=spec,
                        distribution_image_path=distribution_image_path,
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
