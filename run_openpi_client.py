from __future__ import annotations

import argparse
import json

from clients.openpi import OpenPiPiperClient, load_piper_policy_spec, spec_summary
from hardware.config import load_config
from rollout.dry_run import run_deferred_policy_dry_run_plan
from rollout.lerobot_assets import prepare_lerobot_assets, repo_id_from_spec
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
    parser = make_rollout_argument_parser("OpenPI SLAI Piper")
    parser.add_argument(
        "--train-config",
        required=True,
        help="OpenPI train config name, e.g. pi0_slai_piper_template.",
    )
    add_websocket_policy_args(parser)
    add_gripper_bound_args(
        parser,
        threshold_help=(
            "Optional executable-scale gripper threshold. Final values below "
            "this are clipped to zero."
        ),
    )
    add_gripper_encoding_args(parser)
    add_standard_rollout_args(parser, record_directory_name="openpi_records")
    return parser


def run_once(args: argparse.Namespace) -> None:
    validate_standard_rollout_args(args)
    if not args.dry_run:
        spec = load_piper_policy_spec(args.train_config)
        policy_spec_summary = spec_summary(spec)
        if args.spec_only:
            print(json.dumps(policy_spec_summary, indent=2))
            return
    cli_prompt = normalized_prompt(args.prompt)
    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    if args.dry_run:
        run_deferred_policy_dry_run_plan(
            args=args,
            runner_name="run_openpi_client",
            policy_transport_name="OpenPiPiperClient",
            runtime_config=runtime_config,
            configuration_kind="OpenPI registered train config",
        )
        return
    initial_joints, runtime_event_callback = prepare_rollout_runtime(
        args=args,
        spec=spec,
        runtime_config=runtime_config,
        runner_name="run_openpi_client",
    )
    print(json.dumps(policy_spec_summary, indent=2))
    client_assets = prepare_lerobot_assets(
        train_config_name=args.train_config,
        cli_prompt=cli_prompt,
        need_distribution=args.record or args.window,
        repo_id=repo_id_from_spec(spec),
    )
    resolved_prompt = client_assets.prompt
    if resolved_prompt is None:
        raise RuntimeError(
            "No prompt available. Provide --prompt, or ensure the train config's "
            "LeRobot dataset has a cached task prompt."
        )
    print_resolved_prompt(resolved_prompt, client_assets.prompt_source)

    client = OpenPiPiperClient(
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
        apply_arm_gripper_overrides(client, args)
        server_metadata = client.get_server_metadata()
        print_server_metadata(server_metadata)
        run_configured_rollout_runtime(
            args=args,
            client=client,
            spec=spec,
            runtime_config=runtime_config,
            plan=RolloutRuntimePlan(
                hardware_name="openpi_piper_client",
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
