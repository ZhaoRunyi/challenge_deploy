from __future__ import annotations

import argparse
from importlib.resources import files
import time

from hardware.config import (
    discover_can_interfaces,
    load_config,
    validate_can_interface_serials,
    validate_config,
)
from hardware.piper import SinglePiperArm, gripper_effort_value
from hardware.topology import ArmId


def default_runtime_config_path() -> str:
    return str(files("configs").joinpath("dual_piper_example.yaml"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive Piper gripper zero calibration."
    )
    parser.add_argument("--config", default=str(default_runtime_config_path()))
    parser.add_argument(
        "--gripper",
        choices=[arm_id.value for arm_id in ArmId],
        default=ArmId.SLAVE_RIGHT.value,
        help="Physical gripper arm to connect. Default: slave_right.",
    )
    parser.add_argument(
        "--arm",
        choices=[arm_id.value for arm_id in ArmId],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--gripper-effort", type=int, default=None)
    parser.add_argument(
        "--no-command",
        action="store_true",
        help="Only sample the gripper feedback; do not run CPV calibration.",
    )
    return parser


def print_state(label: str, arm: SinglePiperArm) -> int:
    state = arm.read_state()
    gripper = arm.interface.GetArmGripperMsgs().gripper_state
    print(
        f"{label}: opening={state.qpos_feedback[6]:.6f} m, "
        f"angle={gripper.grippers_angle} um, effort={state.effort[6]:.3f}, "
        f"enabled={state.enabled}, ts={state.gripper_position_timestamp_s:.6f}",
        flush=True,
    )
    return int(gripper.grippers_angle)


def sample_gripper(label: str, arm: SinglePiperArm, samples: int, sample_hz: float) -> None:
    period_s = 1.0 / sample_hz
    for index in range(samples):
        print_state(f"{label} {index + 1:02d}/{samples:02d}", arm)
        time.sleep(period_s)


def run_once(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    arm_id = args.arm or args.gripper
    validate_config(config, required_arm_ids=(arm_id,))
    validate_can_interface_serials(
        config,
        required_arm_ids=(arm_id,),
        discovered=discover_can_interfaces(),
    )

    arm_config = config["robot"][arm_id]
    arm = SinglePiperArm(
        name=arm_id,
        can_name=arm_config["can_name"],
        commands_enabled=not args.no_command,
    )
    effort = gripper_effort_value(
        args.gripper_effort
        if args.gripper_effort is not None
        else arm_config.get("gripper_effort")
    )

    try:
        arm.connect(read_only=True)
        sample_gripper("before", arm, args.samples, args.sample_hz)
        if args.no_command:
            return

        arm.interface.GripperCtrl(0, effort, 0x00, 0)
        input(
            f"{arm_id} gripper is disabled. Squeeze it fully closed, keep holding it, "
            "then press ENTER..."
        )
        time.sleep(args.settle_seconds)
        print_state("closed before zeroing", arm)
        arm.interface.GripperCtrl(0, effort, 0x00, 0xAE)
        time.sleep(args.settle_seconds)
        zero_angle_um = print_state("closed after zeroing", arm)

        input("Zero command completed. Release the gripper slowly, then press ENTER...")
        sample_gripper("after", arm, args.samples, args.sample_hz)
        if abs(zero_angle_um) > 1000:
            raise RuntimeError(
                f"Gripper zeroing did not take effect: closed feedback is {zero_angle_um} um"
            )
    finally:
        arm.disconnect()


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
