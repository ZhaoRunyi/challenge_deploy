"""Isolated collect-data mechanical path.

This reduced suite intentionally has no shared-CAN coverage.  A shared
mechanical golden-path test must be added later without changing the isolated
checks below.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from hardware.config import default_config
from hardware.conversions import joints_rad_to_sdk, opening_to_sdk_gripper
from hardware.factory import HardwareAssembly
from hardware.isolated import IsolatedFourArmSystem
from hardware.linkage_gateway import LinkageGatewayError
from hardware.schemas import DualPiperState, PiperArmState
from hardware.topology import (
    ALL_ARM_IDS,
    SLAVE_ARM_IDS,
    ArmId,
    LinkageRole,
    RoleObservation,
)
import run_hdf5_teleop_collect as collector


class FakeRoleObserver:
    def observe(self, arm_id: ArmId, arm: "FakeArm") -> RoleObservation:
        return RoleObservation(
            arm_id=arm_id,
            role=arm.role,
            observed_at_s=time.time(),
            evidence="fake role traffic",
            fresh=True,
        )


class FakeArm:
    """Small public-API Piper arm fake shared by the three robot scenarios."""

    def __init__(
        self,
        arm_id: ArmId,
        events: list[tuple[str, ArmId, object]],
        *,
        can_name: str | None = None,
        commands_enabled: bool = True,
        role: LinkageRole = LinkageRole.MOTION_OUTPUT,
        qpos: np.ndarray | None = None,
    ) -> None:
        self.arm_id = arm_id
        self.name = arm_id.value
        self.can_name = can_name or f"can_{arm_id.value}"
        self.commands_enabled = commands_enabled
        self.role = role
        self.qpos = np.asarray(
            qpos if qpos is not None else np.zeros(7),
            dtype=np.float64,
        ).copy()
        self.end_pose = np.zeros(7, dtype=np.float64)
        self.enabled = True
        self.connected = False
        self.events = events
        self.failures: dict[str, int] = {}
        self.joint_commands: list[np.ndarray] = []
        self.gripper_command_flags: list[bool] = []
        self.last_gripper_effort: int | None = None

    def _record(self, operation: str, value: object = None) -> None:
        self.events.append((operation, self.arm_id, value))

    def _raise_if_requested(self, operation: str) -> None:
        remaining = self.failures.get(operation, 0)
        if remaining <= 0:
            return
        self.failures[operation] = remaining - 1
        raise RuntimeError(f"injected {self.arm_id.value} {operation} failure")

    def connect(self, *, read_only: bool = True) -> None:
        self._record("connect", read_only)
        self._raise_if_requested("connect")
        self.connected = True

    def disconnect(self) -> None:
        self._record("disconnect")
        self.connected = False

    def abort_construction(self) -> None:
        self._record("abort")
        self.connected = False

    def configure_linkage_role(self, role: LinkageRole) -> None:
        self._record("role", role)
        self._raise_if_requested(f"role_{role.short_name}")
        self.role = role

    def is_enabled(self) -> bool:
        return self.enabled

    def send_enable_without_reset(self) -> None:
        self._record("enable_once")

    def enable_without_reset(self, *, retries: int, sleep_s: float) -> bool:
        self._record("enable", (retries, sleep_s))
        self._raise_if_requested("enable")
        self.enabled = True
        return True

    def read_state(self, *, prefer_joint_ctrl: bool = False) -> PiperArmState:
        self._record("read", prefer_joint_ctrl)
        self._raise_if_requested("read")
        timestamp_s = time.time()
        return PiperArmState(
            name=self.name,
            can_name=self.can_name,
            qpos=self.qpos.copy(),
            qpos_feedback=self.qpos.copy(),
            qpos_command=self.qpos.copy(),
            qvel=np.zeros(7, dtype=np.float64),
            effort=np.zeros(7, dtype=np.float64),
            end_pose=self.end_pose.copy(),
            enabled=self.enabled,
            status={"errors": {}},
            feedback_hz=200.0,
            status_hz=200.0,
            command_hz=200.0,
            timestamp_s=timestamp_s,
            qpos_timestamp_s=timestamp_s,
            qvel_timestamp_s=timestamp_s,
            effort_timestamp_s=timestamp_s,
            end_pose_timestamp_s=timestamp_s,
            command_timestamp_s=timestamp_s,
            gripper_position_timestamp_s=timestamp_s,
        )

    def command_joint_positions(
        self,
        qpos: np.ndarray,
        *,
        speed_percent: int,
        gripper_effort: int | None = None,
        command_gripper: bool = True,
    ) -> None:
        target = np.asarray(qpos, dtype=np.float64).copy()
        self._record(
            "joint",
            (target, speed_percent, gripper_effort, command_gripper),
        )
        self._raise_if_requested("joint")
        self.qpos = target
        self.joint_commands.append(target)
        self.gripper_command_flags.append(command_gripper)
        if command_gripper:
            self.last_gripper_effort = gripper_effort

    def command_end_pose(
        self,
        pose: np.ndarray,
        *,
        speed_percent: int,
        gripper_effort: int | None = None,
        command_gripper: bool = True,
    ) -> None:
        target = np.asarray(pose, dtype=np.float64).copy()
        self._record(
            "end_pose",
            (target, speed_percent, gripper_effort, command_gripper),
        )
        self._raise_if_requested("end_pose")
        self.end_pose = target
        self.gripper_command_flags.append(command_gripper)
        if command_gripper:
            self.last_gripper_effort = gripper_effort


def confirmed_isolated_config() -> dict[str, object]:
    config = default_config()
    config["can_topology"] = "isolated"
    for index, (arm_id, arm_config) in enumerate(config["robot"].items(), start=1):
        arm_config["usb_serial"] = f"CONFIRMED-{index}-{arm_id}"
    config["robot"]["slave_left"]["gripper_effort"] = 1111
    config["robot"]["slave_right"]["gripper_effort"] = 2222
    return config


def make_fake_arms(
    arm_ids: tuple[ArmId, ...] = ALL_ARM_IDS,
) -> dict[ArmId, FakeArm]:
    events: list[tuple[str, ArmId, object]] = []
    side_positions = {
        "left": np.array([0.0, 0.2, -0.2, 0.0, 0.0, 0.0, 0.02]),
        "right": np.array([0.1, 0.3, -0.1, 0.1, 0.1, 0.1, 0.03]),
    }
    arms = {
        arm_id: FakeArm(
            arm_id,
            events,
            qpos=side_positions[arm_id.side],
        )
        for arm_id in arm_ids
    }
    return arms


class FakeGateway:
    """Gateway lifecycle fake whose targets follow current master states."""

    def __init__(self, arms: dict[ArmId, FakeArm]) -> None:
        self.arms = arms
        self.generation = 0
        self.dispatch_seq = 0
        self.events: list[str] = []
        self.last_dispatch_receipt: object | None = None
        self.transition_dispatch_receipt: object | None = None
        self.seed_gripper_angles_um: dict[ArmId, int | None] = {}
        counters = SimpleNamespace(received=0, rejected=0, transmitted_frames=0)
        self.left = SimpleNamespace(counters=counters)
        self.right = SimpleNamespace(counters=counters)

    def _target_commit(self, master_id: ArmId) -> SimpleNamespace:
        qpos = self.arms[master_id].qpos
        slave_id = (
            ArmId.SLAVE_LEFT
            if master_id is ArmId.MASTER_LEFT
            else ArmId.SLAVE_RIGHT
        )
        return SimpleNamespace(
            generation=self.generation,
            frame_group_seq=self.dispatch_seq,
            committed_at_s=time.monotonic(),
            complete_frame_group=True,
            joint_values_mdeg=tuple(joints_rad_to_sdk(qpos[:6])),
            gripper_angle_um=self.seed_gripper_angles_um.get(
                slave_id,
                opening_to_sdk_gripper(qpos[6]),
            ),
        )

    def _receipt(self) -> SimpleNamespace:
        self.dispatch_seq += 1
        receipt = SimpleNamespace(
            generation=self.generation,
            dispatch_seq=self.dispatch_seq,
            completed_at_s=time.monotonic(),
            left_commit=self._target_commit(ArmId.MASTER_LEFT),
            right_commit=self._target_commit(ArmId.MASTER_RIGHT),
        )
        self.last_dispatch_receipt = receipt
        self.transition_dispatch_receipt = receipt
        return receipt

    def begin_generation(
        self,
        *,
        seed_gripper_efforts: object,
        seed_gripper_angles_um: object = None,
    ) -> int:
        del seed_gripper_efforts
        self.seed_gripper_angles_um = dict(seed_gripper_angles_um or {})
        self.generation += 1
        self.events.append("begin")
        self.transition_dispatch_receipt = None
        return self.generation

    def start(self, *, automatic_dispatch: bool = True) -> int:
        self.events.append(f"start:{automatic_dispatch}")
        return self.generation

    def stop(self, *, timeout_s: float | None = None) -> None:
        del timeout_s
        self.events.append("stop")

    def stop_at_cycle_boundary(self, *, timeout_s: float) -> bool:
        del timeout_s
        self.events.append("boundary")
        return True

    def wait_for_target_pair(self, *, timeout_s: float) -> dict[ArmId, object]:
        del timeout_s
        self.events.append("targets")
        return {
            ArmId.SLAVE_LEFT: self._target_commit(ArmId.MASTER_LEFT),
            ArmId.SLAVE_RIGHT: self._target_commit(ArmId.MASTER_RIGHT),
        }

    def wait_until_ready(self, *, timeout_s: float) -> object:
        del timeout_s
        self.events.append("ready")
        return self._receipt()

    def require_fresh_family(self) -> object:
        self.events.append("health")
        if self.last_dispatch_receipt is None:
            raise LinkageGatewayError("fake gateway has no fresh target pair")
        return self.last_dispatch_receipt

    def close(self) -> None:
        self.events.append("close")


class StaticCommandArm(FakeArm):
    def __init__(
        self,
        arm_id: ArmId,
        events: list[tuple[str, ArmId, object]],
        *,
        command_timestamp_s: float,
    ) -> None:
        super().__init__(arm_id, events)
        self.command_timestamp_s = command_timestamp_s

    def read_state(self, *, prefer_joint_ctrl: bool = False) -> PiperArmState:
        state = super().read_state(prefer_joint_ctrl=prefer_joint_ctrl)
        if not prefer_joint_ctrl:
            return state
        return replace(
            state,
            qpos_timestamp_s=self.command_timestamp_s,
            command_timestamp_s=self.command_timestamp_s,
        )


class FakeCameraRig:
    def __init__(self) -> None:
        self.serials = {"cam_high": "FAKE-CAMERA"}
        self.enable_depth = False

    def start(self) -> None:
        pass

    def capture_camera_frame(self, camera_name: str) -> object:
        del camera_name
        return SimpleNamespace(
            timestamp_s=time.time(),
            color_image=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_image=None,
        )


def make_four_arm_system() -> tuple[
    IsolatedFourArmSystem,
    dict[ArmId, FakeArm],
    FakeGateway,
]:
    arms = make_fake_arms()
    system = IsolatedFourArmSystem(
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
    return system, arms, FakeGateway(arms)


class CollectDataRobotTest(unittest.TestCase):
    def make_runtime(
        self,
        system: IsolatedFourArmSystem,
        gateway: FakeGateway,
    ) -> tuple[collector.CollectionRuntime, dict[str, object]]:
        source_arguments: dict[str, object] = {}
        assembly = HardwareAssembly(
            robot=system,
            topology="isolated",
            constructed_arm_ids=tuple(arm_id.value for arm_id in ALL_ARM_IDS),
            gateway=gateway,
        )

        def source_factory(**kwargs: object) -> object:
            source_arguments.update(kwargs)
            return object()

        inert_cameras = SimpleNamespace(stop=lambda: None)
        with (
            mock.patch.object(collector, "build_hardware", return_value=assembly),
            mock.patch.object(collector, "make_cameras", return_value=inert_cameras),
            mock.patch.object(
                collector,
                "HDF5TeleopCollectionSource",
                side_effect=source_factory,
            ),
        ):
            runtime = collector.make_isolated_collection_runtime(
                confirmed_isolated_config(),
                enable_depth=False,
                arm_sample_hz=200.0,
                queue_maxlen=2000,
            )
        return runtime, source_arguments

    def test_collect_data_uses_the_four_arm_static_teleop_path(self) -> None:
        system, arms, gateway = make_four_arm_system()
        arms[ArmId.MASTER_LEFT].qpos[6] = 0.0
        arms[ArmId.MASTER_RIGHT].qpos[6] = 0.0
        arms[ArmId.SLAVE_LEFT].qpos[6] = 0.048
        arms[ArmId.SLAVE_RIGHT].qpos[6] = 0.057
        initial_slave_grippers = {
            arm_id: float(arms[arm_id].qpos[6])
            for arm_id in SLAVE_ARM_IDS
        }
        masters_by_slave = {
            ArmId.SLAVE_LEFT: ArmId.MASTER_LEFT,
            ArmId.SLAVE_RIGHT: ArmId.MASTER_RIGHT,
        }
        runtime, source_arguments = self.make_runtime(system, gateway)
        self.assertFalse(runtime.wait_for_source_ready)

        self.assertIs(source_arguments["master_robot"].left, arms[ArmId.MASTER_LEFT])
        self.assertIs(source_arguments["master_robot"].right, arms[ArmId.MASTER_RIGHT])
        self.assertIs(source_arguments["slave_robot"].left, arms[ArmId.SLAVE_LEFT])
        self.assertIs(source_arguments["slave_robot"].right, arms[ArmId.SLAVE_RIGHT])

        runtime.start_callbacks[0]()
        source_arguments["health_check"]()
        master_fallback = source_arguments["master_state_fallback"]()
        self.assertIsNotNone(master_fallback)
        np.testing.assert_allclose(
            master_fallback.qpos,
            np.concatenate(
                (arms[ArmId.SLAVE_LEFT].qpos, arms[ArmId.SLAVE_RIGHT].qpos)
            ),
        )

        self.assertEqual(
            {arm_id: arm.role for arm_id, arm in arms.items()},
            {
                ArmId.MASTER_LEFT: LinkageRole.TEACHING_INPUT,
                ArmId.MASTER_RIGHT: LinkageRole.TEACHING_INPUT,
                ArmId.SLAVE_LEFT: LinkageRole.MOTION_OUTPUT,
                ArmId.SLAVE_RIGHT: LinkageRole.MOTION_OUTPUT,
            },
        )
        self.assertIn("start:True", gateway.events)
        self.assertIn("health", gateway.events)
        self.assertNotIn("targets", gateway.events)
        self.assertNotIn("ready", gateway.events)
        for arm_id in SLAVE_ARM_IDS:
            np.testing.assert_allclose(
                arms[arm_id].qpos[:6],
                arms[masters_by_slave[arm_id]].qpos[:6],
                atol=2e-5,
            )
            self.assertEqual(float(arms[arm_id].qpos[6]), initial_slave_grippers[arm_id])
        self.assertTrue(
            all(
                all(
                    not command_gripper
                    for command_gripper in arms[arm_id].gripper_command_flags
                )
                for arm_id in SLAVE_ARM_IDS
            )
        )
        self.assertTrue(
            all(
                arms[arm_id].gripper_command_flags
                for arm_id in (ArmId.MASTER_LEFT, ArmId.MASTER_RIGHT)
            )
        )
        self.assertTrue(
            all(
                not command_gripper
                for arm_id in (ArmId.MASTER_LEFT, ArmId.MASTER_RIGHT)
                for command_gripper in arms[arm_id].gripper_command_flags
            )
        )

        for _, callback in runtime.emergency_callbacks:
            callback()
        for name, callback in runtime.stop_callbacks:
            if name != "cameras":
                callback()

        self.assertIn("stop", gateway.events)
        self.assertIn("close", gateway.events)
        self.assertTrue(all(not arm.connected for arm in arms.values()))

        master_role_writes = sum(
            operation == "role" and arm_id in (ArmId.MASTER_LEFT, ArmId.MASTER_RIGHT)
            for operation, arm_id, _ in arms[ArmId.MASTER_LEFT].events
        )
        restart_gateway = FakeGateway(arms)
        system.initialize_static_teleop(restart_gateway)
        self.assertEqual(
            sum(
                operation == "role"
                and arm_id in (ArmId.MASTER_LEFT, ArmId.MASTER_RIGHT)
                for operation, arm_id, _ in arms[ArmId.MASTER_LEFT].events
            ),
            master_role_writes,
        )
        self.assertNotIn("targets", restart_gateway.events)
        self.assertNotIn("ready", restart_gateway.events)
        self.assertEqual(restart_gateway.seed_gripper_angles_um, {})
        restart_gateway.close()
        system.disconnect()

    def test_collect_data_initialization_failure_has_a_complete_arm_cleanup_path(
        self,
    ) -> None:
        system, arms, gateway = make_four_arm_system()
        arms[ArmId.MASTER_RIGHT].failures["connect"] = 1
        runtime, _ = self.make_runtime(system, gateway)

        with self.assertRaisesRegex(RuntimeError, "master_right connect failure"):
            runtime.start_callbacks[0]()

        for _, callback in runtime.emergency_callbacks:
            callback()
        for name, callback in runtime.stop_callbacks:
            if name != "cameras":
                callback()

        self.assertTrue(system.faulted)
        self.assertIn("stop", gateway.events)
        self.assertIn("close", gateway.events)
        for arm in arms.values():
            cleanup_events = [
                event
                for event, arm_id, _ in arm.events
                if arm_id is arm.arm_id and event in ("disconnect", "abort")
            ]
            self.assertEqual(len(cleanup_events), 1, arm.arm_id.value)

    def test_static_master_fallback_does_not_block_source_readiness(
        self,
    ) -> None:
        events: list[tuple[str, ArmId, object]] = []
        master_robot = SimpleNamespace(
            left=StaticCommandArm(
                ArmId.MASTER_LEFT,
                events,
                command_timestamp_s=0.0,
            ),
            right=StaticCommandArm(
                ArmId.MASTER_RIGHT,
                events,
                command_timestamp_s=0.0,
            ),
        )
        fallback = DualPiperState(
            left=master_robot.left.read_state(prefer_joint_ctrl=True),
            right=master_robot.right.read_state(prefer_joint_ctrl=True),
        )
        source = collector.HDF5TeleopCollectionSource(
            master_robot=master_robot,
            slave_robot=SimpleNamespace(
                left=FakeArm(ArmId.SLAVE_LEFT, events),
                right=FakeArm(ArmId.SLAVE_RIGHT, events),
            ),
            cameras=FakeCameraRig(),
            arm_sample_hz=100.0,
            queue_maxlen=100,
            master_state_fallback=lambda: fallback,
        )

        try:
            source.start()
            self.assertTrue(
                source.wait_until_ready(timeout_s=1.0),
                source.last_sync_failure,
            )
            self.assertIsNotNone(source.get_frame())
            time.sleep(0.05)
            self.assertIsNotNone(source.get_frame())
        finally:
            source.stop()

        latest_master_sample_s = source.arm_joint_queues[
            "master_left"
        ].latest_timestamp_s()
        self.assertIsNotNone(latest_master_sample_s)
        self.assertGreater(latest_master_sample_s, time.time() - 1.0)

    def test_next_episode_index_ignores_discarded_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "episode_0").mkdir()
            (root / "episode_1").mkdir()
            (root / "episode_1" / "episode_1.hdf5").write_bytes(b"")

            self.assertEqual(collector.next_episode_index(root), 2)

            (root / "episode_2").mkdir()

            self.assertEqual(collector.next_episode_index(root), 2)


if __name__ == "__main__":
    unittest.main()
