from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from clients.openpi_sim import (
    SIM_ACTION_DIM,
    SIM_ACTION_NAMES,
    SIM_IMAGE_IDS,
    SIM_STATE_NAMES,
    OpenPiSimPiperClient,
    OpenPiSimPolicySpec,
    build_configured_piper_state,
    load_openpi_sim_policy_spec,
    spec_summary,
)
from hardware.config import load_config
from hardware.schemas import RobotSnapshot
from rollout.lerobot_assets import prepare_lerobot_assets, repo_id_from_spec
from rollout.recording import RecordingSchema
from rollout.runner import RolloutRuntimePlan, run_configured_rollout_runtime
from rollout.support import (
    add_gripper_bound_args,
    add_gripper_encoding_args,
    add_standard_rollout_args,
    add_websocket_policy_args,
    apply_arm_gripper_overrides,
    apply_runtime_overrides,
    close_policy_transport,
    make_recording_state_builder,
    make_rollout_argument_parser,
    normalized_prompt,
    prepare_rollout_runtime,
    print_resolved_prompt,
    print_server_metadata,
    run_rollout_dry_run_plan,
    validate_standard_rollout_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = make_rollout_argument_parser("OpenPI-sim EmbodiChain Piper")
    parser.add_argument("--train-config", required=True)
    add_websocket_policy_args(parser, control_modes=("joints",))
    add_gripper_bound_args(parser)
    add_gripper_encoding_args(parser)
    parser.add_argument(
        "--bad-sim",
        action="store_true",
        help="Renormalize 0-0.05 simulation gripper outputs before decoding.",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    add_standard_rollout_args(parser, record_directory_name="openpi_sim_records")
    return parser


def make_recording_schema(spec: OpenPiSimPolicySpec) -> RecordingSchema:
    return RecordingSchema(
        camera_names=spec.image_ids,
        action_names=SIM_ACTION_NAMES,
        state_names=SIM_STATE_NAMES,
        used_action_names=frozenset(SIM_ACTION_NAMES),
    )


def snapshot_with_grippers(
    snapshot: RobotSnapshot,
    grippers: np.ndarray,
) -> RobotSnapshot:
    grippers = np.asarray(grippers, dtype=np.float64)
    snapshot.state.left.qpos = snapshot.state.left.qpos.copy()
    snapshot.state.right.qpos = snapshot.state.right.qpos.copy()
    snapshot.state.left.qpos[6] = grippers[0]
    snapshot.state.right.qpos[6] = grippers[1]
    return snapshot


class OpenPiSimCommandCache:
    def __init__(self, initial_grippers: np.ndarray) -> None:
        self.grippers = np.asarray(initial_grippers, dtype=np.float64).copy()


class OpenPiSimStreamingAdapter:
    def __init__(
        self,
        client: OpenPiSimPiperClient,
        source: Any,
        initial_grippers: np.ndarray,
        *,
        command_cache: OpenPiSimCommandCache | None = None,
    ) -> None:
        self.client = client
        self.source = source
        self.command_cache = command_cache or OpenPiSimCommandCache(initial_grippers)

    @property
    def last_grippers(self) -> np.ndarray:
        return self.command_cache.grippers

    @last_grippers.setter
    def last_grippers(self, values: np.ndarray) -> None:
        self.command_cache.grippers = np.asarray(values, dtype=np.float64).copy()

    @property
    def supports_policy_sessions(self) -> bool:
        return self.client.supports_policy_sessions

    def infer_actions(
        self,
        snapshot: RobotSnapshot,
        prompt: str,
        **kwargs: Any,
    ) -> np.ndarray:
        policy_snapshot = snapshot_with_grippers(snapshot, self.last_grippers)
        return self.client.infer_actions(policy_snapshot, prompt=prompt, **kwargs)

    def resync_after_authority_change(
        self,
        *,
        session_id: str | None = None,
        reset_policy: bool = True,
    ) -> Any:
        robot = getattr(self.source, "robot", None)
        if robot is None:
            raise RuntimeError("OpenPI-sim authority resync requires a source-owned robot")
        fresh = robot.read_state()
        self.last_grippers = np.array(
            [fresh.left.qpos[6], fresh.right.qpos[6]],
            dtype=np.float64,
        )
        return self.client.resync_after_authority_change(
            session_id=session_id,
            reset_policy=reset_policy,
        )

    def fork_rollout_inference_session(self) -> "OpenPiSimStreamingAdapter":
        return OpenPiSimStreamingAdapter(
            self.client.fork_rollout_inference_session(),
            self.source,
            self.last_grippers,
            command_cache=self.command_cache,
        )

    def configure_inference_timeout(self, timeout_s: float) -> None:
        self.client.configure_inference_timeout(timeout_s)

    def close_inference_session(self) -> None:
        self.client.close_inference_session()

    def action_state_after_command(
        self,
        robot: Any,
        snapshot_before_command: RobotSnapshot,
    ) -> Any:
        return self.client.action_state_after_command(robot, snapshot_before_command)

    def command_action(self, robot: Any, action: np.ndarray) -> None:
        decoded = self.client.decode_action(action)
        self.client.command_action(robot, action)
        actual = self.client.last_commanded or decoded
        self.last_grippers = np.array(
            [actual.arms["left"].gripper, actual.arms["right"].gripper],
            dtype=np.float64,
        )


def run_once(args: argparse.Namespace) -> None:
    validate_standard_rollout_args(args)
    if args.dry_run:
        spec = OpenPiSimPolicySpec(
            train_config_name=args.train_config,
            train_config=None,
            state_dim=SIM_ACTION_DIM,
            action_dim=SIM_ACTION_DIM,
            model_action_dim=None,
            action_horizon=None,
            image_ids=SIM_IMAGE_IDS,
            default_prompt=None,
        )
    else:
        spec = load_openpi_sim_policy_spec(args.train_config)
    policy_spec_summary = spec_summary(spec)
    if args.dry_run:
        policy_spec_summary["cold_start_contract"] = {
            "schema_source": "repository OpenPI-sim adapter constants",
            "fixed_fields": ["state_dim", "action_dim", "image_ids"],
            "external_train_config_fields_deferred": [
                "model_action_dim",
                "action_horizon",
                "default_prompt",
            ],
            "external_configuration_read": "skipped",
        }
    if args.spec_only:
        print(json.dumps(policy_spec_summary, indent=2), flush=True)
        return
    cli_prompt = normalized_prompt(args.prompt)
    if args.num_steps is not None and args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")

    runtime_config = apply_runtime_overrides(load_config(args.config), args)
    if run_rollout_dry_run_plan(
        args=args,
        runner_name="run_openpi_sim_client",
        policy_transport_name="OpenPiSimPiperClient",
        spec=spec,
        policy_spec_summary=policy_spec_summary,
        runtime_config=runtime_config,
    ):
        return
    initial_joints, runtime_event_callback = prepare_rollout_runtime(
        args=args,
        spec=spec,
        runtime_config=runtime_config,
        runner_name="run_openpi_sim_client",
    )
    print(json.dumps(policy_spec_summary, indent=2), flush=True)
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

    client = OpenPiSimPiperClient(
        args.train_config,
        host=args.host,
        port=args.port,
        control_mode=args.control_mode,
        api_key=args.api_key,
        joint_speed_percent=args.joint_speed_percent,
        gripper_threshold=args.gripper_threshold,
        gripper_lower=args.gripper_lower,
        gripper_upper=args.gripper_upper,
        num_steps=args.num_steps,
        state_gripper_encoding=args.state_gripper,
        action_gripper_encoding=args.action_gripper,
        bad_sim=args.bad_sim,
    )
    try:
        apply_arm_gripper_overrides(client, args)
        server_metadata = client.get_server_metadata()
        print_server_metadata(server_metadata)
        grippers = initial_joints[[6, 13]]
        run_configured_rollout_runtime(
            args=args,
            client=client,
            spec=spec,
            runtime_config=runtime_config,
            plan=RolloutRuntimePlan(
                hardware_name="openpi_sim_piper_client",
                prompt=resolved_prompt,
                initial_joints=initial_joints,
                recording_schema=make_recording_schema(spec),
                state_builder=make_recording_state_builder(
                    build_configured_piper_state,
                    args.state_gripper,
                ),
                server_metadata=server_metadata,
                runtime_event_callback=runtime_event_callback,
                distribution_image_path=client_assets.distribution_image_path,
                distribution_skip_reason=client_assets.skip_reason,
                session_client_factory=lambda source: OpenPiSimStreamingAdapter(
                    client,
                    source,
                    grippers,
                ),
            ),
        )
    finally:
        close_policy_transport(client)


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
