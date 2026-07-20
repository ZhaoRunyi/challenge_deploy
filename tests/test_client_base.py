from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[1]

from clients import slai_piper_policy
from clients.base import (
    SlaiPiperClient,
    action_gripper_for_piper,
    hardware_gripper_to_model_raw,
    model_raw_gripper_to_hardware,
    state_gripper_for_policy,
)
from hardware.constants import PIPER_GRIPPER_FULL_OPEN_METERS
from hardware.conversions import (
    legacy_piper_raw_gripper_to_opening,
    normalized_gripper_to_opening,
    opening_to_legacy_piper_raw_gripper,
    opening_to_normalized_gripper,
)
from hardware.schemas import DualPiperState, PiperArmState
from rollout.support import add_gripper_encoding_args


class FakePolicyClient:
    def get_server_metadata(self) -> dict[str, object]:
        return {}

    def infer(self, payload: dict[str, object]) -> dict[str, object]:
        return {"action": np.zeros((1, 14), dtype=np.float64)}


class FakeArm:
    def __init__(self) -> None:
        self.joint_calls: list[dict[str, object]] = []
        self.pose_calls: list[dict[str, object]] = []

    def command_joint_positions(
        self,
        joint: np.ndarray,
        *,
        speed_percent: int,
        gripper_effort: int | None = None,
    ) -> None:
        self.joint_calls.append(
            {
                "joint": np.asarray(joint, dtype=np.float64).copy(),
                "speed_percent": speed_percent,
                "gripper_effort": gripper_effort,
            }
        )

    def command_end_pose(
        self,
        pose: np.ndarray,
        *,
        speed_percent: int,
        gripper_effort: int | None = None,
    ) -> None:
        self.pose_calls.append(
            {
                "pose": np.asarray(pose, dtype=np.float64).copy(),
                "speed_percent": speed_percent,
                "gripper_effort": gripper_effort,
            }
        )


class FakeRobot:
    def __init__(self, state: DualPiperState) -> None:
        self.left = FakeArm()
        self.right = FakeArm()
        self._state = state

    def read_state(self) -> DualPiperState:
        return self._state


def make_arm_state(name: str, gripper: float, offset: float = 0.0) -> PiperArmState:
    qpos = np.array([offset + 0.1 * i for i in range(6)] + [gripper], dtype=np.float64)
    end_pose = np.array([offset + 0.2 * i for i in range(6)] + [gripper], dtype=np.float64)
    return PiperArmState(
        name=name,
        can_name=f"can_{name}",
        qpos=qpos,
        qpos_feedback=qpos.copy(),
        qpos_command=qpos.copy(),
        qvel=np.zeros(7, dtype=np.float64),
        effort=np.zeros(7, dtype=np.float64),
        end_pose=end_pose,
        enabled=True,
        status={},
        feedback_hz=100.0,
        status_hz=100.0,
        command_hz=100.0,
    )


def make_state(left_gripper: float = 0.0, right_gripper: float = 0.0) -> DualPiperState:
    return DualPiperState(
        left=make_arm_state("left", left_gripper, offset=0.0),
        right=make_arm_state("right", right_gripper, offset=1.0),
    )


def make_spec(ids: str = "joint_gripper", gripper_type: str = "raw") -> SimpleNamespace:
    gripper = slai_piper_policy.GripperConfig(
        type=gripper_type,
        threshold=0.01,
        full_width=PIPER_GRIPPER_FULL_OPEN_METERS,
    )
    state_space = slai_piper_policy.StateSpaceConfig(ids=ids, arms="dual", ee_rotation="rpy", gripper=gripper)
    action_space = slai_piper_policy.ActionSpaceConfig(ids=ids, arms="dual", ee_rotation="rpy", gripper=gripper)
    return SimpleNamespace(
        train_config_name="fake",
        state_space=state_space,
        action_space=action_space,
        state_dim=slai_piper_policy.get_space_dim(state_space),
        action_dim=slai_piper_policy.get_space_dim(action_space),
        image_ids=(),
        image_key_map={},
    )


def make_action(left_gripper: float, right_gripper: float) -> np.ndarray:
    return np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, left_gripper, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, right_gripper],
        dtype=np.float64,
    )


class ClientBaseTest(unittest.TestCase):
    def test_gripper_effort_default_explicit_and_invalid(self) -> None:
        spec = make_spec()
        robot = FakeRobot(make_state())
        client = SlaiPiperClient(spec=spec, policy_client=FakePolicyClient())
        client.command_action(robot, make_action(0.5, 0.5))
        self.assertIsNone(robot.left.joint_calls[-1]["gripper_effort"])
        self.assertIsNone(robot.right.joint_calls[-1]["gripper_effort"])

        robot = FakeRobot(make_state())
        client = SlaiPiperClient(spec=spec, policy_client=FakePolicyClient(), gripper_effort=1234)
        client.command_action(robot, make_action(0.5, 0.5))
        self.assertEqual(robot.left.joint_calls[-1]["gripper_effort"], 1234)
        self.assertEqual(robot.right.joint_calls[-1]["gripper_effort"], 1234)

        with self.assertRaises(ValueError):
            SlaiPiperClient(spec=spec, policy_client=FakePolicyClient(), gripper_effort=5001)

    def test_binary_transition_uses_current_robot_start(self) -> None:
        spec = make_spec()
        robot = FakeRobot(make_state(left_gripper=0.0, right_gripper=0.0))
        client = SlaiPiperClient(
            spec=spec,
            policy_client=FakePolicyClient(),
            action_gripper_encoding="binary",
            gripper_action_frames=3,
        )
        action = make_action(1.0, 1.0)

        client.command_action(robot, action)
        self.assertAlmostEqual(robot.left.joint_calls[-1]["joint"][6], PIPER_GRIPPER_FULL_OPEN_METERS / 3.0)
        self.assertIsNotNone(client.gripper_transition)

        client.command_action(robot, action)
        self.assertAlmostEqual(robot.left.joint_calls[-1]["joint"][6], 2.0 * PIPER_GRIPPER_FULL_OPEN_METERS / 3.0)
        self.assertIsNotNone(client.gripper_transition)

        client.command_action(robot, action)
        self.assertAlmostEqual(robot.left.joint_calls[-1]["joint"][6], PIPER_GRIPPER_FULL_OPEN_METERS)
        self.assertIsNone(client.gripper_transition)

    def test_current_decoded_from_robot_reads_joint_and_ee_start(self) -> None:
        state = make_state(left_gripper=0.02, right_gripper=0.03)
        joint_client = SlaiPiperClient(spec=make_spec("joint_gripper"), policy_client=FakePolicyClient())
        decoded = joint_client.current_decoded_from_robot(FakeRobot(state))
        np.testing.assert_allclose(decoded.arms["left"].joint, state.left.qpos)
        self.assertIsNone(decoded.arms["left"].ee_pose)

        ee_client = SlaiPiperClient(
            spec=make_spec("ee_gripper"),
            policy_client=FakePolicyClient(),
            control_mode="ee_pose",
        )
        decoded = ee_client.current_decoded_from_robot(FakeRobot(state))
        np.testing.assert_allclose(decoded.arms["right"].ee_pose, state.right.end_pose)
        self.assertIsNone(decoded.arms["right"].joint)

    def test_state_and_action_gripper_encodings(self) -> None:
        opening = 0.05
        self.assertAlmostEqual(
            state_gripper_for_policy(opening, None, state_gripper_encoding="policy"),
            opening_to_normalized_gripper(opening),
        )
        self.assertAlmostEqual(
            state_gripper_for_policy(opening, None, state_gripper_encoding="meters"),
            opening,
        )
        self.assertAlmostEqual(
            state_gripper_for_policy(opening, None, state_gripper_encoding="old"),
            opening_to_legacy_piper_raw_gripper(opening),
        )
        self.assertAlmostEqual(
            hardware_gripper_to_model_raw(opening, state_gripper_encoding="old"),
            opening_to_legacy_piper_raw_gripper(opening),
        )

        normalized = 0.25
        legacy = opening_to_legacy_piper_raw_gripper(opening)
        self.assertAlmostEqual(
            action_gripper_for_piper(normalized, None, action_gripper_encoding="policy"),
            normalized_gripper_to_opening(normalized),
        )
        self.assertAlmostEqual(action_gripper_for_piper(opening, None, action_gripper_encoding="meters"), opening)
        self.assertEqual(action_gripper_for_piper(0.49, None, action_gripper_encoding="binary"), 0.0)
        self.assertEqual(
            action_gripper_for_piper(0.5, None, action_gripper_encoding="binary"),
            PIPER_GRIPPER_FULL_OPEN_METERS,
        )
        self.assertAlmostEqual(
            action_gripper_for_piper(legacy, None, action_gripper_encoding="old"),
            legacy_piper_raw_gripper_to_opening(legacy),
        )
        self.assertAlmostEqual(
            model_raw_gripper_to_hardware(legacy, action_gripper_encoding="old"),
            legacy_piper_raw_gripper_to_opening(legacy),
        )

    def test_old_encoding_overrides_gripper_01_policy_semantics(self) -> None:
        gripper = slai_piper_policy.GripperConfig(
            type="01",
            threshold=0.01,
            full_width=PIPER_GRIPPER_FULL_OPEN_METERS,
        )
        opening = 0.05
        legacy = opening_to_legacy_piper_raw_gripper(opening)
        self.assertAlmostEqual(
            state_gripper_for_policy(opening, gripper, state_gripper_encoding="old"),
            legacy,
        )
        self.assertAlmostEqual(
            action_gripper_for_piper(legacy, gripper, action_gripper_encoding="old"),
            opening,
        )
        self.assertEqual(action_gripper_for_piper(0.5, gripper, action_gripper_encoding="policy"), gripper.full_width)

    def test_parser_flags_are_explicit(self) -> None:
        parser = argparse.ArgumentParser()
        add_gripper_encoding_args(parser, default_state="meters", default_action="binary")
        args = parser.parse_args([])
        self.assertEqual(args.state_gripper, "meters")
        self.assertEqual(args.action_gripper, "binary")
        self.assertEqual(parser.parse_args(["--state-gripper", "old", "--action-gripper", "old"]).state_gripper, "old")

        entrypoints = (
            "run_openpi_client.py",
            "run_openpi_sim_client.py",
            "run_xvla_client.py",
            "run_motus_client.py",
            "run_dreamzero_client.py",
        )
        for entrypoint in entrypoints:
            source = (DEPLOY_ROOT / entrypoint).read_text(encoding="utf-8")
            self.assertIn("add_gripper_encoding_args(parser", source, entrypoint)
            self.assertNotIn("--old_gripper", source, entrypoint)
            self.assertNotIn("old_gripper", source, entrypoint)


if __name__ == "__main__":
    unittest.main()
