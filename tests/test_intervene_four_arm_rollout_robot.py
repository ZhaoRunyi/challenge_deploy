"""Standard isolated four-arm ROLLOUT/INTERVENE mechanical path."""

from __future__ import annotations

import struct
import time
import unittest

import can
import numpy as np

from hardware.factory import (
    HardwareAssembly,
    HardwareFactoryDependencies,
    build_hardware,
)
from hardware.linkage_gateway import (
    BimanualLinkageGateway,
    GatewayConfig,
    GatewayReadyTimeoutError,
    SemanticLinkageGateway,
)
from hardware.topology import (
    ALL_ARM_IDS,
    ArmId,
    LatchedFaultError,
    LinkageRole,
    RoleTransactionError,
)
from rollout.hardware_control import (
    IsolatedAuthorityHardwareController,
    IsolatedAuthorityMode,
    make_authority_hardware_controller,
)
from tests.test_collect_data_robot import (
    FakeArm,
    confirmed_isolated_config,
    make_fake_arms,
    make_four_arm_system,
)


class RecordingBus:
    def __init__(self) -> None:
        self.messages: list[can.Message] = []

    def recv(self, timeout: float | None = None) -> None:
        del timeout
        return None

    def send(self, message: can.Message, timeout: float | None = None) -> None:
        del timeout
        self.messages.append(message)

    def shutdown(self) -> None:
        return None


class CloseOnlySemanticGateway:
    def __init__(self, **kwargs: object) -> None:
        self.arguments = kwargs
        self.closed = False

    def close(self) -> None:
        self.closed = True


class CloseOnlyBimanualGateway:
    def __init__(
        self,
        left: CloseOnlySemanticGateway,
        right: CloseOnlySemanticGateway,
    ) -> None:
        self.left = left
        self.right = right

    def close(self) -> None:
        self.left.close()
        self.right.close()


class InterveneFourArmRolloutRobotTest(unittest.TestCase):
    def test_factory_constructs_four_unique_arms_and_two_gateway_sides(self) -> None:
        config = confirmed_isolated_config()
        source_arms = make_fake_arms()
        constructed: list[ArmId] = []
        semantic_gateways: list[CloseOnlySemanticGateway] = []

        def arm_factory(
            arm_id: ArmId,
            can_name: str,
            commands_enabled: bool,
        ) -> FakeArm:
            constructed.append(arm_id)
            return FakeArm(
                arm_id,
                source_arms[arm_id].events,
                can_name=can_name,
                commands_enabled=commands_enabled,
                qpos=source_arms[arm_id].qpos,
            )

        def semantic_factory(**kwargs: object) -> CloseOnlySemanticGateway:
            gateway = CloseOnlySemanticGateway(**kwargs)
            semantic_gateways.append(gateway)
            return gateway

        discovered = {
            config["robot"][arm_id.value]["can_name"]: config["robot"][arm_id.value][
                "usb_serial"
            ]
            for arm_id in ALL_ARM_IDS
        }
        assembly = build_hardware(
            config,
            intervention=True,
            dependencies=HardwareFactoryDependencies(
                discovery_factory=lambda: discovered,
                isolated_arm_factory=arm_factory,
                semantic_gateway_factory=semantic_factory,
                bimanual_gateway_factory=CloseOnlyBimanualGateway,
            ),
        )
        try:
            self.assertEqual(constructed, list(ALL_ARM_IDS))
            self.assertEqual(
                set(assembly.constructed_arm_ids),
                {arm_id.value for arm_id in ALL_ARM_IDS},
            )
            self.assertEqual(
                [gateway.arguments["arm_id"] for gateway in semantic_gateways],
                [ArmId.SLAVE_LEFT, ArmId.SLAVE_RIGHT],
            )
            self.assertEqual(
                [gateway.arguments["config"].gripper_effort for gateway in semantic_gateways],
                [1111, 2222],
            )
        finally:
            assembly.robot.abort_construction()
            assembly.gateway.close()

    def test_rollout_intervene_pause_resume_and_rollout_preserve_arm_positions(
        self,
    ) -> None:
        system, arms, gateway = make_four_arm_system()
        system.hardware_assembly = HardwareAssembly(
            robot=system,
            topology="isolated",
            constructed_arm_ids=tuple(arm_id.value for arm_id in ALL_ARM_IDS),
            gateway=gateway,
        )
        system.connect()
        system.enable(retries=1, sleep_s=0.0)
        left_target = np.array([0.05, 0.25, -0.15, 0.05, 0.05, 0.05, 0.021])
        right_target = np.array([0.15, 0.35, -0.05, 0.15, 0.15, 0.15, 0.031])
        system.command_bimanual_joint_positions(left_target, right_target)
        for arm_id, arm in arms.items():
            expected = left_target if arm_id.side == "left" else right_target
            np.testing.assert_allclose(arm.qpos, expected, atol=2e-5)

        controller = make_authority_hardware_controller(system)
        self.assertIsInstance(controller, IsolatedAuthorityHardwareController)
        controller.start_authority()
        controller.enter_intervention()
        self.assertIs(controller.mode, IsolatedAuthorityMode.INTERVENE)
        controller.pause_intervention()
        self.assertIs(controller.mode, IsolatedAuthorityMode.INTERVENE_PAUSED)
        controller.resume_intervention()
        controller.enter_rollout()
        self.assertIs(controller.mode, IsolatedAuthorityMode.ROLLOUT)

        role_writes = [
            (arm_id, value)
            for operation, arm_id, value in next(iter(arms.values())).events
            if operation == "role"
        ]
        self.assertEqual(
            role_writes,
            [
                (ArmId.MASTER_LEFT, LinkageRole.TEACHING_INPUT),
                (ArmId.MASTER_RIGHT, LinkageRole.TEACHING_INPUT),
                (ArmId.MASTER_LEFT, LinkageRole.MOTION_OUTPUT),
                (ArmId.MASTER_RIGHT, LinkageRole.MOTION_OUTPUT),
            ],
        )
        for arm_id, arm in arms.items():
            expected = left_target if arm_id.side == "left" else right_target
            np.testing.assert_allclose(arm.qpos, expected, atol=2e-5)
            self.assertIs(arm.role, LinkageRole.MOTION_OUTPUT)
        self.assertEqual(gateway.events.count("start:True"), 2)
        self.assertEqual(gateway.events.count("boundary"), 2)
        system.disconnect()
        self.assertTrue(all(not arm.connected for arm in arms.values()))

    def test_gateway_dispatches_only_a_complete_sanitized_left_right_pair(self) -> None:
        left_master = RecordingBus()
        right_master = RecordingBus()
        left_slave = RecordingBus()
        right_slave = RecordingBus()
        config = GatewayConfig(
            freshness_s=0.5,
            max_skew_s=0.05,
            dispatch_timeout_s=0.05,
        )
        left = SemanticLinkageGateway(
            arm_id=ArmId.SLAVE_LEFT,
            master_bus=left_master,
            slave_bus=left_slave,
            config=config,
        )
        right = SemanticLinkageGateway(
            arm_id=ArmId.SLAVE_RIGHT,
            master_bus=right_master,
            slave_bus=right_slave,
            config=config,
        )
        pair = BimanualLinkageGateway(left, right)

        def publish_family(
            gateway: SemanticLinkageGateway,
            generation: int,
            joint_value: int,
            gripper_angle_um: int | None = None,
        ) -> None:
            now_s = time.monotonic()
            messages = [
                can.Message(
                    arbitration_id=0x155,
                    data=struct.pack(">ii", joint_value, joint_value + 1),
                    dlc=8,
                    is_extended_id=False,
                ),
                can.Message(
                    arbitration_id=0x156,
                    data=struct.pack(">ii", joint_value + 2, joint_value + 3),
                    dlc=8,
                    is_extended_id=False,
                ),
                can.Message(
                    arbitration_id=0x157,
                    data=struct.pack(">ii", joint_value + 4, joint_value + 5),
                    dlc=8,
                    is_extended_id=False,
                ),
            ]
            if gripper_angle_um is not None:
                messages.insert(
                    0,
                    can.Message(
                        arbitration_id=0x159,
                        data=struct.pack(">iHBB", gripper_angle_um, 4999, 0x03, 0xAE),
                        dlc=8,
                        is_extended_id=False,
                    ),
                )
            for message in messages:
                gateway.process_message(
                    message,
                    generation=generation,
                    received_at_s=now_s,
                )

        def publish_gripper(
            gateway: SemanticLinkageGateway,
            generation: int,
            gripper_angle_um: int,
        ) -> None:
            gateway.process_message(
                can.Message(
                    arbitration_id=0x159,
                    data=struct.pack(">iHBB", gripper_angle_um, 4999, 0x03, 0xAE),
                    dlc=8,
                    is_extended_id=False,
                ),
                generation=generation,
                received_at_s=time.monotonic(),
            )

        try:
            generation = pair.begin_generation(
                seed_gripper_efforts={
                    ArmId.SLAVE_LEFT: 1111,
                    ArmId.SLAVE_RIGHT: 2222,
                },
                seed_gripper_angles_um={
                    ArmId.SLAVE_LEFT: 50_000,
                    ArmId.SLAVE_RIGHT: 60_000,
                },
            )
            local_frame = can.Message(
                arbitration_id=0x155,
                data=bytes(8),
                dlc=8,
                is_extended_id=False,
                is_rx=False,
            )
            left.process_message(local_frame, generation=generation)
            publish_family(left, generation, 10)
            with self.assertRaises(GatewayReadyTimeoutError):
                pair.wait_until_ready(timeout_s=0.01)
            self.assertEqual(left_slave.messages, [])
            self.assertEqual(right_slave.messages, [])

            publish_family(right, generation, 30)
            receipt = pair.wait_until_ready(timeout_s=0.05)

            self.assertEqual(receipt.dispatch_seq, 1)
            expected_ids = [0x151, 0x155, 0x156, 0x157]
            self.assertEqual(
                [message.arbitration_id for message in left_slave.messages],
                expected_ids,
            )
            self.assertEqual(
                [message.arbitration_id for message in right_slave.messages],
                expected_ids,
            )
            self.assertEqual(left_slave.messages[0].data[3], 0xAD)
            self.assertEqual(right_slave.messages[0].data[3], 0xAD)
            self.assertIsNone(receipt.left_commit.gripper_payload)
            self.assertIsNone(receipt.right_commit.gripper_payload)

            left_slave.messages.clear()
            right_slave.messages.clear()
            publish_gripper(left, generation, 20_000)
            publish_gripper(right, generation, 40_000)
            with self.assertRaises(GatewayReadyTimeoutError):
                pair.dispatch_next_pair(timeout_s=0.01)
            publish_gripper(left, generation, 25_000)
            publish_gripper(right, generation, 50_000)
            receipt = pair.dispatch_next_pair(timeout_s=0.05)

            self.assertEqual(receipt.dispatch_seq, 2)
            expected_ids = [0x151, 0x155, 0x156, 0x157, 0x159]
            self.assertEqual(
                [message.arbitration_id for message in left_slave.messages],
                expected_ids,
            )
            self.assertEqual(
                [message.arbitration_id for message in right_slave.messages],
                expected_ids,
            )
            self.assertEqual(
                struct.unpack(">iHBB", bytes(left_slave.messages[-1].data)),
                (55_000, 1111, 0x01, 0x00),
            )
            self.assertEqual(
                struct.unpack(">iHBB", bytes(right_slave.messages[-1].data)),
                (70_000, 2222, 0x01, 0x00),
            )
            self.assertEqual(left.counters.rejected, 1)
        finally:
            pair.close()

    def test_failed_master_role_transaction_rolls_back_and_fault_fences_commands(
        self,
    ) -> None:
        system, arms, _ = make_four_arm_system()
        system.connect()
        arms[ArmId.MASTER_RIGHT].failures["role_FA"] = 2

        with self.assertRaises(RoleTransactionError):
            system.configure_master_pair(LinkageRole.TEACHING_INPUT)

        self.assertTrue(system.faulted)
        self.assertIs(arms[ArmId.MASTER_LEFT].role, LinkageRole.MOTION_OUTPUT)
        self.assertIs(arms[ArmId.MASTER_RIGHT].role, LinkageRole.MOTION_OUTPUT)
        target = np.zeros(7, dtype=np.float64)
        with self.assertRaises(LatchedFaultError):
            system.command_bimanual_joint_positions(target, target)

        before_hold = {
            arm_id: len(arm.joint_commands)
            for arm_id, arm in arms.items()
        }
        system.best_effort_hold_motion_output_arms()
        system.disconnect()
        for arm_id, arm in arms.items():
            self.assertEqual(len(arm.joint_commands), before_hold[arm_id] + 1)
            self.assertFalse(arm.connected)


if __name__ == "__main__":
    unittest.main()
