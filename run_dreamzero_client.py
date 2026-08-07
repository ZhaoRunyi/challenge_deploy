from __future__ import annotations

import argparse
import json

from clients.dreamzero import (
    DreamZeroPiperClient,
    load_dreamzero_policy_spec,
    spec_summary,
)
from hardware.config import load_config
from rollout.dry_run import run_deferred_policy_dry_run_plan
from rollout.dreamzero_assets import prepare_dreamzero_client_assets
from rollout.runner import RolloutRuntimePlan, run_configured_rollout_runtime
from rollout.support import (
    add_gripper_bound_args,
    add_gripper_encoding_args,
    add_standard_rollout_args,
    add_websocket_policy_args,
    apply_arm_gripper_overrides,
    apply_runtime_overrides,
    build_slai_recording_state,
    close_policy_transport,
    make_recording_state_builder,
    make_rollout_argument_parser,
    make_slai_recording_schema,
    normalized_prompt,
    prepare_rollout_runtime,
    print_resolved_prompt,
    print_server_metadata,
    validate_standard_rollout_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = make_rollout_argument_parser("DreamZero SLAI Piper")
    parser.add_argument(
        "--train-config",
        required=True,
        help=(
            "DreamZero YAML config path, e.g. "
            "examples/sft/config/piper_sft_dreamzero_5b.yaml."
        ),
    )
    add_websocket_policy_args(parser, default_speed_percent=100)
    parser.add_argument(
        "--gripper-effort",
        type=int,
        default=None,
        help=(
            "Piper SDK GripperCtrl effort. "
            "Defaults to the hardware layer value."
        ),
    )
    parser.add_argument(
        "--gripper-action-frames",
        type=int,
        default=3,
        help=(
            "Only used when gripper commands are binarized. Open/close "
            "transitions are executed linearly across this many command frames."
        ),
    )
    add_gripper_bound_args(
        parser,
        threshold_help=(
            "Optional executable-scale gripper threshold in meters. Values below "
            "threshold close the gripper and values above it command full open."
        ),
    )
    add_gripper_encoding_args(parser)
    parser.add_argument(
        "--num-inference-timesteps",
        type=int,
        default=None,
        help="Forward an optional denoising step override to the server.",
    )
    add_standard_rollout_args(parser, record_directory_name="dreamzero_records")
    return parser


def run_once(args: argparse.Namespace) -> None:
    validate_standard_rollout_args(args)
    if not args.dry_run:
        spec = load_dreamzero_policy_spec(args.train_config)
        policy_spec_summary = spec_summary(spec)
        if args.spec_only:
            print(json.dumps(policy_spec_summary, indent=2))
            return
    cli_prompt = normalized_prompt(args.prompt)
    if args.gripper_action_frames <= 0:
        raise ValueError("--gripper-action-frames must be positive")
    if (
        args.num_inference_timesteps is not None
        and args.num_inference_timesteps <= 0
    ):
        raise ValueError("--num-inference-timesteps must be positive")

    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    if args.dry_run:
        run_deferred_policy_dry_run_plan(
            args=args,
            runner_name="run_dreamzero_client",
            policy_transport_name="DreamZeroPiperClient",
            runtime_config=runtime_config,
            configuration_kind="DreamZero deployment YAML",
        )
        return
    initial_joints, runtime_event_callback = prepare_rollout_runtime(
        args=args,
        spec=spec,
        runtime_config=runtime_config,
        runner_name="run_dreamzero_client",
    )
    print(json.dumps(policy_spec_summary, indent=2))
    client = DreamZeroPiperClient(
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
        state_gripper_encoding=args.state_gripper,
        action_gripper_encoding=args.action_gripper,
    )
    try:
        apply_arm_gripper_overrides(client, args)
        server_metadata = client.get_server_metadata()
        print_server_metadata(server_metadata)
        client_assets = prepare_dreamzero_client_assets(
            train_config_name=args.train_config,
            cli_prompt=cli_prompt,
            need_distribution=args.record or args.window,
            spec=spec,
            server_metadata=server_metadata,
        )
        resolved_prompt = client_assets.prompt
        if resolved_prompt is None:
            raise RuntimeError(
                "No prompt available. Provide --prompt, or ensure the remote "
                "DreamZero server was started with --default_prompt."
            )
        print_resolved_prompt(resolved_prompt, client_assets.prompt_source)
        run_configured_rollout_runtime(
            args=args,
            client=client,
            spec=spec,
            runtime_config=runtime_config,
            plan=RolloutRuntimePlan(
                hardware_name="dreamzero_piper_client",
                prompt=resolved_prompt,
                initial_joints=initial_joints,
                recording_schema=make_slai_recording_schema(spec, args.control_mode),
                state_builder=make_recording_state_builder(
                    build_slai_recording_state,
                    args.state_gripper,
                ),
                server_metadata=server_metadata,
                runtime_event_callback=runtime_event_callback,
                distribution_image_path=client_assets.distribution_image_path,
                distribution_skip_reason=client_assets.skip_reason,
                preferred_frame_names=("cam_high",) + tuple(spec.image_ids),
                initial_gripper_effort=args.gripper_effort,
                save_predicted_video=True,
            ),
        )
    finally:
        close_policy_transport(client)


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
