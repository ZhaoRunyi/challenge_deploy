"""Two-slave isolated rollout mechanical path without ancillary systems."""

from __future__ import annotations

import unittest

import numpy as np

from hardware.factory import (
    HardwareAssembly,
    HardwareFactoryDependencies,
    build_hardware,
)
from hardware.isolated import IsolatedSlaveSystem
from hardware.topology import (
    ArmId,
    BatchOperationError,
    LatchedFaultError,
    SLAVE_ARM_IDS,
)
from rollout.hardware_control import (
    IsolatedSlaveRolloutHardwareController,
    make_authority_hardware_controller,
)
from tests.test_collect_data_robot import (
    FakeArm,
    FakeRoleObserver,
    confirmed_isolated_config,
    make_fake_arms,
)


def make_slave_system() -> tuple[IsolatedSlaveSystem, dict[ArmId, FakeArm]]:
    arms = make_fake_arms(SLAVE_ARM_IDS)
    system = IsolatedSlaveSystem(
        arms,
        role_observer=FakeRoleObserver(),
        role_verify_timeout_s=0.0,
        motion_watchdog_max_state_age_s=1.0,
        slave_gripper_efforts={
            ArmId.SLAVE_LEFT: 1111,
            ArmId.SLAVE_RIGHT: 2222,
        },
        sleeper=lambda _: None,
    )
    system.hardware_assembly = HardwareAssembly(
        robot=system,
        topology="isolated",
        constructed_arm_ids=("slave_left", "slave_right"),
        gateway=None,
    )
    return system, arms


class DualArmRolloutRobotTest(unittest.TestCase):
    def test_factory_constructs_only_the_two_slaves(self) -> None:
        config = confirmed_isolated_config()
        constructed: list[ArmId] = []
        fake_arms = make_fake_arms(SLAVE_ARM_IDS)

        def arm_factory(
            arm_id: ArmId,
            can_name: str,
            commands_enabled: bool,
        ) -> FakeArm:
            constructed.append(arm_id)
            source = fake_arms[arm_id]
            return FakeArm(
                arm_id,
                source.events,
                can_name=can_name,
                commands_enabled=commands_enabled,
                qpos=source.qpos,
            )

        discovered = {
            config["robot"][arm_id.value]["can_name"]: config["robot"][arm_id.value][
                "usb_serial"
            ]
            for arm_id in SLAVE_ARM_IDS
        }
        assembly = build_hardware(
            config,
            intervention=False,
            dependencies=HardwareFactoryDependencies(
                discovery_factory=lambda: discovered,
                isolated_arm_factory=arm_factory,
            ),
        )
        try:
            self.assertEqual(constructed, list(SLAVE_ARM_IDS))
            self.assertEqual(
                assembly.constructed_arm_ids,
                ("slave_left", "slave_right"),
            )
            self.assertIsNone(assembly.gateway)
            self.assertNotIn(ArmId.MASTER_LEFT, assembly.robot.physical_arms)
            self.assertNotIn(ArmId.MASTER_RIGHT, assembly.robot.physical_arms)
        finally:
            assembly.robot.abort_construction()

    def test_joint_and_end_pose_rollout_commands_reach_both_slaves(self) -> None:
        system, arms = make_slave_system()
        system.connect()
        system.enable(retries=1, sleep_s=0.0)
        controller = make_authority_hardware_controller(system)
        self.assertIsInstance(controller, IsolatedSlaveRolloutHardwareController)

        left_joint_target = np.array([0.1, 0.3, -0.1, 0.1, 0.1, 0.1, 0.025])
        right_joint_target = np.array([0.2, 0.4, 0.0, 0.2, 0.2, 0.2, 0.035])
        system.command_bimanual_joint_positions(
            left_joint_target,
            right_joint_target,
            speed_percent=60,
            gripper_efforts={
                ArmId.SLAVE_LEFT: 1500,
                ArmId.SLAVE_RIGHT: 2500,
            },
        )

        left_pose = np.array([0.25, 0.10, 0.30, 0.0, 0.0, 0.0, 0.026])
        right_pose = np.array([0.30, -0.10, 0.28, 0.0, 0.0, 0.0, 0.036])
        system.command_bimanual_end_poses(
            left_pose,
            right_pose,
            speed_percent=40,
        )
        state = system.read_state()
        controller.require_healthy()
        controller.hold_position()
        system.disconnect()

        np.testing.assert_allclose(arms[ArmId.SLAVE_LEFT].qpos, left_joint_target)
        np.testing.assert_allclose(arms[ArmId.SLAVE_RIGHT].qpos, right_joint_target)
        np.testing.assert_allclose(
            arms[ArmId.SLAVE_LEFT].end_pose,
            left_pose,
        )
        np.testing.assert_allclose(
            arms[ArmId.SLAVE_RIGHT].end_pose,
            right_pose,
        )
        self.assertEqual(state.left.name, "slave_left")
        self.assertEqual(state.right.name, "slave_right")
        self.assertEqual(arms[ArmId.SLAVE_LEFT].last_gripper_effort, 1500)
        self.assertEqual(arms[ArmId.SLAVE_RIGHT].last_gripper_effort, 2500)
        self.assertTrue(all(not arm.connected for arm in arms.values()))

    def test_one_slave_command_failure_fences_new_rollout_commands_but_keeps_hold(
        self,
    ) -> None:
        system, arms = make_slave_system()
        system.connect()
        system.require_healthy()
        arms[ArmId.SLAVE_RIGHT].failures["joint"] = 1
        target = np.array([0.05, 0.25, -0.15, 0.05, 0.05, 0.05, 0.02])

        with self.assertRaises(BatchOperationError):
            system.command_bimanual_joint_positions(target, target)

        self.assertTrue(system.faulted)
        left_command_count = len(arms[ArmId.SLAVE_LEFT].joint_commands)
        with self.assertRaises(LatchedFaultError):
            system.left.command_joint_positions(target)
        self.assertEqual(
            len(arms[ArmId.SLAVE_LEFT].joint_commands),
            left_command_count,
        )

        controller = make_authority_hardware_controller(system)
        before_hold = {
            arm_id: len(arm.joint_commands)
            for arm_id, arm in arms.items()
        }
        controller.hold_position()
        system.disconnect()

        for arm_id, arm in arms.items():
            self.assertEqual(len(arm.joint_commands), before_hold[arm_id] + 1)
            self.assertFalse(arm.connected)


if __name__ == "__main__":
    unittest.main()
