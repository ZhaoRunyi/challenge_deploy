from __future__ import annotations

import argparse
import json

from clients.fastwam import (
    FASTWAM_DEFAULT_ACTION_HORIZON,
    FASTWAM_DEFAULT_TRAIN_CONFIG,
    FastWAMPiperClient,
    load_fastwam_policy_spec,
    spec_summary,
)
from hardware.config import load_config
from rollout.assets import prepare_client_assets
from rollout.runner import RolloutRuntimePlan, run_configured_rollout_runtime
from rollout.support import (
    add_gripper_bound_args,
    add_gripper_encoding_args,
    add_standard_rollout_args,
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
    run_rollout_dry_run_plan,
    validate_standard_rollout_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = make_rollout_argument_parser(
        "FastWAM SLAI Piper HTTP",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--train-config",
        default=FASTWAM_DEFAULT_TRAIN_CONFIG,
        help=(
            "Deploy label used for summaries and record names. The FastWAM "
            "server owns the actual Hydra config."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--endpoint", default="/infer")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--server-preflight-timeout", type=float, default=8.0)
    parser.add_argument(
        "--skip-server-preflight",
        action="store_true",
        help="Skip the FastWAM preflight before touching hardware.",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--control-mode", choices=["joints"], default="joints")
    parser.add_argument("--joint-speed-percent", type=int, default=100)
    parser.add_argument("--ee-speed-percent", type=int, default=100)
    parser.add_argument("--gripper-effort", type=int, default=None)
    parser.add_argument("--gripper-action-frames", type=int, default=1)
    add_gripper_bound_args(parser)
    add_gripper_encoding_args(parser)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=FASTWAM_DEFAULT_ACTION_HORIZON,
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    add_standard_rollout_args(parser, record_directory_name="fastwam_records")
    return parser


def run_once(args: argparse.Namespace) -> None:
    validate_standard_rollout_args(args)
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    cli_prompt = normalized_prompt(args.prompt)
    spec = load_fastwam_policy_spec(
        args.train_config,
        action_horizon=args.action_horizon,
        prompt=cli_prompt,
    )
    policy_spec_summary = spec_summary(spec)
    if args.spec_only:
        print(json.dumps(policy_spec_summary, indent=2), flush=True)
        return
    if cli_prompt is None:
        raise RuntimeError("FastWAM requires --prompt")
    if args.request_timeout <= 0.0:
        raise ValueError("--request-timeout must be positive")
    if args.server_preflight_timeout <= 0.0:
        raise ValueError("--server-preflight-timeout must be positive")
    if args.gripper_action_frames <= 0:
        raise ValueError("--gripper-action-frames must be positive")
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")

    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    if run_rollout_dry_run_plan(
        args=args,
        runner_name="run_fastwam_client",
        policy_transport_name="FastWAMPiperClient",
        spec=spec,
        policy_spec_summary=policy_spec_summary,
        runtime_config=runtime_config,
    ):
        return
    initial_joints, runtime_event_callback = prepare_rollout_runtime(
        args=args,
        spec=spec,
        runtime_config=runtime_config,
        runner_name="run_fastwam_client",
    )
    print(json.dumps(policy_spec_summary, indent=2), flush=True)
    client_assets = prepare_client_assets(
        client_kind="fastwam",
        train_config_name=args.train_config,
        cli_prompt=cli_prompt,
        need_distribution=args.record or args.window,
        spec=spec,
    )
    resolved_prompt = client_assets.prompt
    if resolved_prompt is None:
        raise RuntimeError("FastWAM prompt resolution unexpectedly returned no prompt")

    client = FastWAMPiperClient(
        args.train_config,
        host=args.host,
        port=args.port,
        endpoint=args.endpoint,
        request_timeout_s=args.request_timeout,
        control_mode=args.control_mode,
        joint_speed_percent=args.joint_speed_percent,
        ee_speed_percent=args.ee_speed_percent,
        gripper_effort=args.gripper_effort,
        gripper_action_frames=args.gripper_action_frames,
        gripper_threshold=args.gripper_threshold,
        gripper_lower=args.gripper_lower,
        gripper_upper=args.gripper_upper,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        state_gripper_encoding=args.state_gripper,
        action_gripper_encoding=args.action_gripper,
    )
    try:
        apply_arm_gripper_overrides(client, args)
        server_metadata = client.get_server_metadata()
        print_server_metadata(server_metadata)
        print_resolved_prompt(resolved_prompt, client_assets.prompt_source)
        if not args.skip_server_preflight:
            print(
                json.dumps(
                    {
                        "server_preflight": client.probe_server(
                            timeout_s=args.server_preflight_timeout
                        )
                    },
                    indent=2,
                ),
                flush=True,
            )
        if client_assets.skip_reason is not None:
            print(
                f"Skipped FastWAM distribution image: {client_assets.skip_reason}",
                flush=True,
            )
        run_configured_rollout_runtime(
            args=args,
            client=client,
            spec=spec,
            runtime_config=runtime_config,
            plan=RolloutRuntimePlan(
                hardware_name="fastwam_piper_client",
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
                preferred_frame_names=tuple(spec.image_ids),
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
