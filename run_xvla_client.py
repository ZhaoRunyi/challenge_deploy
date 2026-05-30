from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clients.xvla import (
    XVLAPiperClient,
    build_full_piper_state,
    load_piper_policy_spec,
    spec_summary,
)
from hardware.config import load_config
from rollout.assets import prepare_client_assets
from rollout.execution import (
    action_sequence,
    resolve_chunk_size,
    run_chunk_sync_rollout,
    save_rollout_metrics,
)
from rollout.recording import RolloutVideoRecorder, preview_until_continue, save_frame1_image, save_recorded_actions
from rollout.support import (
    apply_runtime_overrides,
    make_dual_piper_runtime,
    make_slai_recording_schema,
    normalized_prompt,
    print_rollout_chunk_summary,
    record_name_prefix,
    resolve_dual_piper_init_joints,
)

DEPLOY_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="X-VLA SLAI Piper client.")
    parser.add_argument("--train-config", default="slai_piper_items_hand_over_place_ee20_xvla_pt_bs256_400000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", required=False)
    parser.add_argument("--control-mode", choices=["joints", "ee_pose"], default="ee_pose")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--joint-speed-percent", type=int, default=50)
    parser.add_argument("--ee-speed-percent", type=int, default=50)
    parser.add_argument("--gripper_threshold", type=float, default=None)
    parser.add_argument("--gripper_lower", type=float, default=None)
    parser.add_argument("--gripper_upper", type=float, default=None)
    parser.add_argument("--old_gripper", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--metrics-json", default=None, help="Optional path to save rollout timing metrics as JSON.")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-dir", default=str(DEPLOY_ROOT / "artifacts" / "xvla_records"))
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--spec-only", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = load_piper_policy_spec(args.train_config)
    print(json.dumps(spec_summary(spec), indent=2), flush=True)
    if args.spec_only:
        return
    cli_prompt = normalized_prompt(args.prompt)
    client_assets = prepare_client_assets(
        client_kind="xvla",
        train_config_name=args.train_config,
        cli_prompt=cli_prompt,
        need_distribution=args.record or args.window,
        spec=spec,
    )
    resolved_prompt = client_assets.prompt
    if resolved_prompt is None:
        raise RuntimeError(
            "No prompt available for this train config. Provide --prompt, or add a matching entry to "
            "deploy/artifacts/trainconfig_prompts.json."
        )
    print(
        json.dumps(
            {"prompt": {"value": resolved_prompt, "source": client_assets.prompt_source}},
            indent=2,
        ),
        flush=True,
    )

    client = XVLAPiperClient(
        args.train_config,
        host=args.host,
        port=args.port,
        control_mode=args.control_mode,
        api_key=args.api_key,
        joint_speed_percent=args.joint_speed_percent,
        ee_speed_percent=args.ee_speed_percent,
        gripper_threshold=args.gripper_threshold,
        gripper_lower=args.gripper_lower,
        gripper_upper=args.gripper_upper,
        old_gripper=args.old_gripper,
    )

    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    robot, cameras, source = make_dual_piper_runtime(
        runtime_config,
        commands_enabled=not args.dry_run,
        name="xvla_piper_client",
    )
    recorder = (
        RolloutVideoRecorder(
            output_dir=args.record_dir,
            schema=make_slai_recording_schema(spec, args.control_mode),
            fps=args.fps,
            name_prefix=record_name_prefix(args),
        )
        if args.record
        else None
    )
    saved_actions: list[np.ndarray] | None = [] if recorder is not None else None
    state_builder = lambda snapshot, policy_spec: build_full_piper_state(snapshot, policy_spec, old_gripper=args.old_gripper)
    robot.connect(read_only=args.dry_run)
    try:
        if cameras is not None:
            cameras.start()
        if not source.wait_until_ready(timeout_s=args.ready_timeout):
            raise RuntimeError("Timed out waiting for Piper/RealSense data")
        if not args.dry_run and not robot.enable():
            print("Warning: Piper arm enable check did not report success; continuing anyway.", flush=True)
        chunk_size = resolve_chunk_size(spec, args.chunk_size)
        if args.dry_run:
            snapshot = source.capture_snapshot()
            actions = action_sequence(client.infer_actions(snapshot, prompt=resolved_prompt))[:chunk_size]
            print_rollout_chunk_summary(
                client=client,
                chunk_index=0,
                action_count=len(actions),
                executed_steps=0,
                rollout_steps=args.rollout_steps,
                first_action=actions[0],
            )
            if args.window:
                preview_until_continue(source, distribution_image_path=client_assets.distribution_image_path)
                if client_assets.skip_reason is not None:
                    print(f"Skipped train-distribution frame1 image: {client_assets.skip_reason}", flush=True)
            return
        initial_joints = resolve_dual_piper_init_joints(args.init_joints)
        print(json.dumps({"initial_pose": {"qpos": initial_joints.tolist()}}, indent=2), flush=True)
        robot.move_to_joint_positions(initial_joints, speed_percent=args.joint_speed_percent)
        if recorder is not None:
            frame1_path = save_frame1_image(
                recorder,
                source.capture_snapshot(),
                distribution_image_path=client_assets.distribution_image_path,
            )
            if frame1_path is not None:
                print(f"Frame1 comparison saved to {frame1_path}", flush=True)
        if args.window:
            preview_until_continue(source, distribution_image_path=client_assets.distribution_image_path)
            if client_assets.skip_reason is not None:
                print(f"Skipped train-distribution frame1 image: {client_assets.skip_reason}", flush=True)

        def log_chunk(chunk_index: int, action_count: int, executed_steps: int, first_action) -> None:
            print_rollout_chunk_summary(
                client=client,
                chunk_index=chunk_index,
                action_count=action_count,
                executed_steps=executed_steps,
                rollout_steps=args.rollout_steps,
                first_action=first_action,
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
            log_chunk=log_chunk,
            state_builder=state_builder,
        )
        if metrics.interrupted:
            print("Interrupted by user; stopping rollout.", flush=True)
        metrics_summary, written_metric_paths = save_rollout_metrics(metrics, metrics_json_path=args.metrics_json)
        print(json.dumps({"rollout_metrics": metrics_summary}, indent=2), flush=True)
        for metrics_path in written_metric_paths:
            print(f"Rollout metrics saved to {metrics_path}", flush=True)
    finally:
        if recorder is not None:
            try:
                action_path = save_recorded_actions(recorder, saved_actions, recorder.schema.action_names)
                print(f"Actions saved to {action_path}", flush=True)
            except Exception as exc:
                print(f"Failed to save actions: {exc}", flush=True)
            try:
                output_path = recorder.finalize()
                if output_path is not None:
                    print(f"Recording saved to {output_path}", flush=True)
            except Exception as exc:
                print(f"Failed to finalize recording: {exc}", flush=True)
        if cameras is not None:
            cameras.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
