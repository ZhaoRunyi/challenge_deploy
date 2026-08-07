from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .execution import resolve_chunk_size, resolve_record_steps, save_rollout_metrics
from .recording import (
    ExecutionRecordSink,
    RolloutVideoRecorder,
    save_frame1_image,
    save_recorded_actions,
)
from .support import (
    best_effort_hold_configured_robot,
    close_hardware_gateway,
    ignore_record_signal_handlers,
    install_record_signal_handlers,
    make_dual_piper_runtime,
    print_rollout_chunk_summary,
    record_name_prefix,
    run_interactive_configured_rollout,
)
from .windowing import RuntimeExecutionWindow, preview_until_continue


@dataclass(frozen=True)
class RolloutRuntimePlan:
    hardware_name: str
    prompt: str
    initial_joints: np.ndarray
    recording_schema: Any
    state_builder: Callable[[Any, Any], np.ndarray]
    server_metadata: dict[str, Any]
    runtime_event_callback: Callable[..., None]
    distribution_image_path: Path | None = None
    distribution_skip_reason: str | None = None
    preferred_frame_names: tuple[str, ...] = ("cam_high",)
    initial_gripper_effort: int | None = None
    save_predicted_video: bool = False
    session_client_factory: Callable[[Any], Any] | None = None


def _execution_settings(
    args: argparse.Namespace,
    spec: Any,
    runtime_config: dict[str, Any],
) -> tuple[int | None, float, int, int, int]:
    policy_config = runtime_config["policy"]
    chunk_size = resolve_chunk_size(spec, args.chunk_size)
    inference_rate = float(
        args.inference_rate
        if args.inference_rate is not None
        else policy_config["inference_rate"]
    )
    latency_k = int(
        args.latency_k if args.latency_k is not None else policy_config["latency_k"]
    )
    min_smooth_steps = int(
        args.min_smooth_steps
        if args.min_smooth_steps is not None
        else policy_config["min_smooth_steps"]
    )
    buffer_max_chunks = int(
        args.buffer_max_chunks
        if args.buffer_max_chunks is not None
        else policy_config["buffer_max_chunks"]
    )
    return (
        chunk_size,
        inference_rate,
        latency_k,
        min_smooth_steps,
        buffer_max_chunks,
    )


def _print_rollout_settings(
    args: argparse.Namespace,
    *,
    chunk_size: int | None,
    inference_rate: float,
    latency_k: int,
    min_smooth_steps: int,
    buffer_max_chunks: int,
) -> None:
    streaming = args.execution_mode == "streaming"
    print(
        json.dumps(
            {
                "rollout": {
                    "execution_mode": args.execution_mode,
                    "rollout_steps": args.rollout_steps,
                    "record_steps": resolve_record_steps(
                        args.rollout_steps,
                        args.record_steps,
                    ),
                    "chunk_size": chunk_size,
                    "fps": args.fps,
                    "inference_rate": inference_rate if streaming else None,
                    "latency_k": latency_k if streaming else None,
                    "min_smooth_steps": min_smooth_steps if streaming else None,
                    "buffer_max_chunks": buffer_max_chunks if streaming else None,
                }
            },
            indent=2,
        ),
        flush=True,
    )


def _finalize_recording(
    recorder: RolloutVideoRecorder | None,
    *,
    client: Any,
    session_id: str | None,
    save_predicted_video: bool,
) -> None:
    if recorder is None:
        return
    try:
        action_path = save_recorded_actions(recorder)
        print(f"Actions saved to {action_path}", flush=True)
    except Exception as exc:
        print(f"Failed to save actions: {exc}", flush=True)

    try:
        output_path = recorder.finalize()
    except Exception as exc:
        print(f"Failed to finalize recording: {exc}", flush=True)
        return
    if output_path is None:
        return

    print(f"Recording saved to {output_path}", flush=True)
    for separate_video_path in recorder.separate_video_paths:
        print(f"Separate camera video saved to {separate_video_path}", flush=True)
    if not save_predicted_video:
        return
    try:
        if session_id is None:
            raise RuntimeError("no completed ROLLOUT session is available")
        predicted_video_path = client.save_predicted_video(
            session_id=session_id,
            output_dir=recorder.run_dir,
            file_stem=recorder.record_stem,
        )
        if predicted_video_path is not None:
            print(f"Predicted video saved to {predicted_video_path}", flush=True)
    except Exception as exc:
        print(f"Failed to save predicted video: {exc}", flush=True)


def run_configured_rollout_runtime(
    *,
    args: argparse.Namespace,
    client: Any,
    spec: Any,
    runtime_config: dict[str, Any],
    plan: RolloutRuntimePlan,
) -> Any:
    robot = None
    cameras = None
    runtime_window = None
    recorder = None
    signal_handlers_installed = False
    robot_connected = False
    session_id: str | None = None

    try:
        robot, cameras, source = make_dual_piper_runtime(
            runtime_config,
            name=plan.hardware_name,
        )
        runtime_window = (
            RuntimeExecutionWindow(
                schema=plan.recording_schema,
                display_index=args.window,
            )
            if args.window
            else None
        )
        recorder = (
            RolloutVideoRecorder(
                output_dir=args.record_dir,
                schema=plan.recording_schema,
                fps=args.fps,
                name_prefix=record_name_prefix(args, plan.server_metadata),
                save_separate_videos=args.save_sep,
            )
            if args.record
            else None
        )
        record_sink = (
            ExecutionRecordSink(recorder=recorder, runtime_window=runtime_window)
            if recorder is not None or runtime_window is not None
            else None
        )
        install_record_signal_handlers()
        signal_handlers_installed = True

        robot.connect(read_only=False)
        robot_connected = True
        if cameras is not None:
            cameras.start()
        if not source.wait_until_ready(timeout_s=args.ready_timeout):
            raise RuntimeError("Timed out waiting for Piper/RealSense data")

        print('{"hardware_init": "enable_dual_piper"}', flush=True)
        if not robot.enable():
            print(
                "Warning: Piper arm enable check did not report success; continuing anyway.",
                flush=True,
            )

        print(
            json.dumps(
                {"initial_pose": {"qpos": plan.initial_joints.tolist()}},
                indent=2,
            ),
            flush=True,
        )
        robot.move_to_joint_positions(
            plan.initial_joints,
            speed_percent=args.joint_speed_percent,
            gripper_effort=plan.initial_gripper_effort,
        )
        if recorder is not None:
            try:
                first_snapshot = source.capture_snapshot()
                frame1_path = save_frame1_image(
                    recorder,
                    first_snapshot,
                    distribution_image_path=plan.distribution_image_path,
                    preferred_names=plan.preferred_frame_names,
                )
                if frame1_path is not None:
                    print(f"Frame1 image saved to {frame1_path}", flush=True)
                elif plan.distribution_skip_reason is not None:
                    print(
                        "Skipped train-distribution frame1 image: "
                        f"{plan.distribution_skip_reason}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"Failed to save frame1 image: {exc}", flush=True)
        if args.window:
            preview_until_continue(
                source,
                distribution_image_path=plan.distribution_image_path,
            )

        (
            chunk_size,
            inference_rate,
            latency_k,
            min_smooth_steps,
            buffer_max_chunks,
        ) = _execution_settings(args, spec, runtime_config)
        _print_rollout_settings(
            args,
            chunk_size=chunk_size,
            inference_rate=inference_rate,
            latency_k=latency_k,
            min_smooth_steps=min_smooth_steps,
            buffer_max_chunks=buffer_max_chunks,
        )

        def log_chunk(
            chunk_index: int,
            action_count: int,
            executed_steps: int,
            first_action: np.ndarray,
        ) -> None:
            print_rollout_chunk_summary(
                client=client,
                chunk_index=chunk_index,
                action_count=action_count,
                executed_steps=executed_steps,
                rollout_steps=args.rollout_steps,
                first_action=first_action,
            )

        session_client = (
            plan.session_client_factory(source)
            if plan.session_client_factory is not None
            else client
        )
        session_result = run_interactive_configured_rollout(
            args=args,
            client=session_client,
            source=source,
            robot=robot,
            spec=spec,
            prompt=plan.prompt,
            runtime_config=runtime_config,
            chunk_size=chunk_size,
            inference_rate=inference_rate,
            latency_k=latency_k,
            min_smooth_steps=min_smooth_steps,
            buffer_max_chunks=buffer_max_chunks,
            initial_joints=plan.initial_joints,
            initial_speed_percent=args.joint_speed_percent,
            initial_gripper_effort=plan.initial_gripper_effort,
            record_sink=record_sink,
            state_builder=plan.state_builder,
            log_chunk=log_chunk,
            runtime_event_callback=plan.runtime_event_callback,
        )
        metrics = session_result.metrics
        if session_result.rollout_session_ids:
            session_id = session_result.rollout_session_ids[-1]
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
        return session_result
    except KeyboardInterrupt as exc:
        plan.runtime_event_callback(
            "runner_interrupted",
            error=exc,
            robot_fault=getattr(robot, "fault", None),
            durable=True,
        )
        if robot is not None:
            best_effort_hold_configured_robot(robot)
        print("Interrupted by user; stopping rollout.", flush=True)
        return None
    except Exception as exc:
        plan.runtime_event_callback(
            "runner_failed",
            error=exc,
            robot_fault=getattr(robot, "fault", None),
            last_enable_operations=getattr(robot, "last_enable_operations", ()),
            durable=True,
        )
        if robot is not None:
            best_effort_hold_configured_robot(robot)
        raise
    finally:
        if signal_handlers_installed:
            ignore_record_signal_handlers()
        if robot is not None:
            try:
                close_hardware_gateway(robot)
            except Exception as exc:
                print(f"Failed to close semantic gateway cleanly: {exc}", flush=True)
        if cameras is not None:
            try:
                cameras.stop()
            except Exception as exc:
                print(f"Failed to stop cameras cleanly: {exc}", flush=True)
        if robot is not None:
            try:
                abort_construction = getattr(robot, "abort_construction", None)
                if not robot_connected and callable(abort_construction):
                    abort_construction()
                else:
                    robot.disconnect()
            except Exception as exc:
                print(f"Failed to disconnect robot cleanly: {exc}", flush=True)
        _finalize_recording(
            recorder,
            client=client,
            session_id=session_id,
            save_predicted_video=plan.save_predicted_video,
        )
        if runtime_window is not None:
            try:
                runtime_window.close()
            except Exception as exc:
                print(f"Failed to close runtime window: {exc}", flush=True)
