from __future__ import annotations

import argparse
import json

from clients.xvla import (
    XVLA_TRAIN_CONFIGS,
    XVLAPiperClient,
    load_piper_policy_spec,
    spec_summary,
)
from hardware.config import load_config
from rollout.assets import prepare_client_assets
from rollout.dry_run import run_deferred_policy_dry_run_plan
from rollout.runner import RolloutRuntimePlan, run_configured_rollout_runtime
from rollout.support import (
    add_gripper_bound_args,
    add_gripper_encoding_args,
    add_standard_rollout_args,
    add_websocket_policy_args,
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
    parser = make_rollout_argument_parser("X-VLA SLAI Piper")
    parser.add_argument(
        "--train-config",
        default="slai_piper_items_hand_over_place_ee20_xvla_pt_bs256_400000",
    )
    add_websocket_policy_args(parser, default_control_mode="ee_pose")
    add_gripper_bound_args(parser, per_arm=False)
    add_gripper_encoding_args(parser, default_state="meters", default_action="binary")
    add_standard_rollout_args(parser, record_directory_name="xvla_records")
    return parser


def run_once(args: argparse.Namespace) -> None:
    validate_standard_rollout_args(args)
    defer_external_config = args.dry_run and args.train_config not in XVLA_TRAIN_CONFIGS
    if not defer_external_config:
        spec = load_piper_policy_spec(args.train_config)
        policy_spec_summary = spec_summary(spec)
        if args.spec_only:
            print(json.dumps(policy_spec_summary, indent=2), flush=True)
            return
    cli_prompt = normalized_prompt(args.prompt)
    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    if defer_external_config:
        run_deferred_policy_dry_run_plan(
            args=args,
            runner_name="run_xvla_client",
            policy_transport_name="XVLAPiperClient",
            runtime_config=runtime_config,
            configuration_kind="external X-VLA deployment JSON or registry name",
        )
        return
    initial_joints, runtime_event_callback = prepare_rollout_runtime(
        args=args,
        spec=spec,
        runtime_config=runtime_config,
        runner_name="run_xvla_client",
    )
    print(json.dumps(policy_spec_summary, indent=2), flush=True)
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
            "No prompt available for this train config. Provide --prompt, or "
            "add a matching prompt entry."
        )
    print_resolved_prompt(resolved_prompt, client_assets.prompt_source)

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
        state_gripper_encoding=args.state_gripper,
        action_gripper_encoding=args.action_gripper,
    )
    try:
        server_metadata = client.get_server_metadata()
        print_server_metadata(server_metadata)
        run_configured_rollout_runtime(
            args=args,
            client=client,
            spec=spec,
            runtime_config=runtime_config,
            plan=RolloutRuntimePlan(
                hardware_name="xvla_piper_client",
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
            ),
        )
    finally:
        close_policy_transport(client)


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
