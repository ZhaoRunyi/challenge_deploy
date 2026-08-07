from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import time
from typing import Any

import numpy as np

from hardware.conversions import sdk_gripper_to_opening
from hardware.piper import joint_target_to_end_pose
from hardware.schemas import DualPiperState, PiperArmState
from hardware.topology import ArmId, MASTER_ARM_IDS, SLAVE_ARM_IDS

from .authority import InterventionActionSample


@dataclass(frozen=True, slots=True)
class AuthorityTransitionReceipt:
    """Expose the gateway epoch without conflating it with robot transactions."""

    generation: int
    operations: tuple[Any, ...]
    initial_dispatch_seq: int


class IsolatedAuthorityMode(str, Enum):
    """Hardware authority retained across an episode boundary."""

    ROLLOUT = "rollout"
    INTERVENE = "intervene"
    INTERVENE_PAUSED = "intervene_paused"


class StaticRolloutHardwareController:
    """Authority adapter for shared and isolated slave-only rollouts."""

    def __init__(self, robot: Any) -> None:
        self.robot = robot

    def start_authority(self) -> None:
        return None

    def require_healthy(self) -> Any:
        fault = getattr(self.robot, "fault", None)
        if fault is not None:
            raise RuntimeError(f"robot hardware fault is latched: {fault}")

    def enter_rollout(self) -> None:
        return None

    def enter_intervention(self) -> None:
        raise RuntimeError("INTERVENE is not available for this hardware runtime")

    def resume_intervention(self) -> None:
        raise RuntimeError("INTERVENE is not available for this hardware runtime")

    def pause_intervention(self) -> None:
        raise RuntimeError("INTERVENE is not available for this hardware runtime")

    def pause_episode(self) -> None:
        self.hold_position()

    def hold_position(self) -> Any:
        state = self.robot.read_state()
        failures: list[str] = []
        for side, arm, arm_state in (
            ("left", self.robot.left, state.left),
            ("right", self.robot.right, state.right),
        ):
            try:
                arm.command_joint_positions(arm_state.qpos, speed_percent=50)
            except Exception as exc:
                failures.append(f"{side}: {exc!r}")
        if failures:
            raise RuntimeError("failed to hold one or more shared arms: " + ", ".join(failures))
        return None

    def sample_intervention_action_state(self) -> None:
        return None


class IsolatedSlaveRolloutHardwareController(StaticRolloutHardwareController):
    """Two-slave controller with isolated-only freshness and fault gates."""

    def require_healthy(self) -> Any:
        return self.robot.require_healthy()

    def hold_position(self) -> Any:
        return self.robot.best_effort_hold_motion_output_arms()


class IsolatedAuthorityHardwareController:
    """Own isolated role transactions, gateway lifecycle, and health checks."""

    def __init__(self, *, robot: Any, gateway: Any) -> None:
        if gateway is None:
            raise ValueError("isolated dynamic control requires a bimanual semantic gateway")
        self.robot = robot
        self.gateway = gateway
        self.mode = IsolatedAuthorityMode.ROLLOUT
        self._last_intervention_sample: InterventionActionSample | None = None

    def _current_gripper_efforts(self) -> dict[ArmId, int]:
        effective = self.robot.effective_slave_gripper_efforts
        return {slave_id: int(effective[slave_id]) for slave_id in SLAVE_ARM_IDS}

    def _transition_kwargs(self) -> dict[str, Any]:
        return {
            "gripper_efforts": self._current_gripper_efforts(),
        }

    def _intervention_receipt(
        self,
        result: Any,
    ) -> AuthorityTransitionReceipt:
        transition_dispatch = self.gateway.transition_dispatch_receipt
        if transition_dispatch is None:
            raise RuntimeError(
                "isolated intervention transition has no paired dispatch receipt"
            )
        physical_states = getattr(result, "physical_states", None)
        if physical_states is not None:
            self._last_intervention_sample = self._sample_from_dispatch(
                physical_states,
                transition_dispatch,
            )
        return AuthorityTransitionReceipt(
            generation=int(self.gateway.generation),
            operations=tuple(result.operations),
            initial_dispatch_seq=int(transition_dispatch.dispatch_seq),
        )

    def start_authority(self) -> Any:
        """Adopt the runner-established FC session without repeating it."""

        self.gateway.stop()
        self.mode = IsolatedAuthorityMode.ROLLOUT
        self._last_intervention_sample = None
        return self.robot.generation

    def require_healthy(self) -> Any:
        gateway_fault = getattr(self.gateway, "fault", None)
        if gateway_fault:
            raise RuntimeError(f"semantic gateway fault is latched: {gateway_fault}")
        prefer_joint_ctrl_arm_ids: tuple[ArmId, ...] = (
            MASTER_ARM_IDS if self.mode is not IsolatedAuthorityMode.ROLLOUT else ()
        )
        return self.robot.require_healthy(
            prefer_joint_ctrl_arm_ids=prefer_joint_ctrl_arm_ids,
        )

    def enter_rollout(self) -> Any:
        if self.mode is not IsolatedAuthorityMode.ROLLOUT:
            result = self.robot.exit_intervention(
                self.gateway,
                **self._transition_kwargs(),
            )
        else:
            self.gateway.stop()
            self.robot.best_effort_hold_motion_output_arms(
                gripper_efforts=self._current_gripper_efforts(),
            )
            result = self.robot.generation
        self.mode = IsolatedAuthorityMode.ROLLOUT
        self._last_intervention_sample = None
        return result

    def enter_intervention(self) -> Any:
        self._last_intervention_sample = None
        result = self.robot.enter_intervention(
            self.gateway,
            **self._transition_kwargs(),
        )
        self.mode = IsolatedAuthorityMode.INTERVENE
        return self._intervention_receipt(result)

    def resume_intervention(self) -> AuthorityTransitionReceipt:
        """Resume a paused FA session without performing another role write."""

        self._last_intervention_sample = None
        result = self.robot.resume_intervention(
            self.gateway,
            **self._transition_kwargs(),
        )
        self.mode = IsolatedAuthorityMode.INTERVENE
        return self._intervention_receipt(result)

    def pause_episode(self) -> None:
        if self.mode is not IsolatedAuthorityMode.ROLLOUT:
            self.pause_intervention()
        else:
            self.robot.best_effort_hold_motion_output_arms(
                gripper_efforts=self._current_gripper_efforts(),
            )

    def pause_intervention(self) -> Any:
        """Fence gateway TX and hold slaves while leaving both masters in FA."""

        result = self.robot.pause_intervention(
            self.gateway,
            gripper_efforts=self._current_gripper_efforts(),
        )
        self.mode = IsolatedAuthorityMode.INTERVENE_PAUSED
        return result

    def hold_position(self) -> Any:
        try:
            self.gateway.stop()
        except Exception:
            pass
        gripper_efforts = self._current_gripper_efforts()
        return self.robot.best_effort_hold_motion_output_arms(
            gripper_efforts=gripper_efforts,
        )

    @staticmethod
    def _action_arm_state(state: PiperArmState, commit: Any) -> PiperArmState:
        joint_radians = np.deg2rad(np.asarray(commit.joint_values_mdeg, dtype=np.float64) / 1000.0)
        gripper = state.qpos[6]
        if commit.gripper_angle_um is not None:
            gripper = sdk_gripper_to_opening(int(commit.gripper_angle_um))
        qpos = np.concatenate((joint_radians, np.array([gripper], dtype=np.float64)))
        end_pose = joint_target_to_end_pose(qpos, state.end_pose)
        command_timestamp_s = time.time()
        return replace(
            state,
            qpos=qpos,
            qpos_command=qpos.copy(),
            end_pose=end_pose,
            timestamp_s=max(float(state.timestamp_s), command_timestamp_s),
            qpos_timestamp_s=command_timestamp_s,
            end_pose_timestamp_s=command_timestamp_s,
            command_timestamp_s=command_timestamp_s,
        )

    def _sample_from_dispatch(
        self,
        states: Any,
        receipt: Any,
    ) -> InterventionActionSample:
        left_commit = receipt.left_commit
        right_commit = receipt.right_commit
        if left_commit.generation != right_commit.generation:
            raise RuntimeError("left/right gateway generations do not match")
        if (
            left_commit.gripper_angle_um is None
            or right_commit.gripper_angle_um is None
        ):
            raise RuntimeError(
                "gateway dispatched an INTERVENE frame group without gripper targets"
            )
        action_state = DualPiperState(
            left=self._action_arm_state(states[ArmId.MASTER_LEFT], left_commit),
            right=self._action_arm_state(states[ArmId.MASTER_RIGHT], right_commit),
        )
        return InterventionActionSample(
            action_state=action_state,
            committed_monotonic_s=float(receipt.completed_at_s),
            frame_group_seq=int(receipt.dispatch_seq),
            generation=int(receipt.generation),
        )

    def sample_intervention_action_state(self) -> InterventionActionSample | None:
        if self.mode is not IsolatedAuthorityMode.INTERVENE:
            return None
        states = self.require_healthy()
        receipt = self.gateway.last_dispatch_receipt
        if receipt is None:
            raise RuntimeError(
                "active intervention gateway has no successful paired dispatch"
            )
        if (
            self._last_intervention_sample is not None
            and receipt.generation == self._last_intervention_sample.generation
            and receipt.dispatch_seq
            == self._last_intervention_sample.frame_group_seq
        ):
            return replace(
                self._last_intervention_sample,
                is_new_dispatch=False,
            )
        sample = self._sample_from_dispatch(states, receipt)
        self._last_intervention_sample = sample
        return sample


def make_authority_hardware_controller(robot: Any) -> Any:
    assembly = getattr(robot, "hardware_assembly", None)
    if assembly is None:
        return StaticRolloutHardwareController(robot)
    if assembly.topology == "isolated" and assembly.gateway is None:
        return IsolatedSlaveRolloutHardwareController(robot)
    if assembly.gateway is None:
        return StaticRolloutHardwareController(robot)
    return IsolatedAuthorityHardwareController(
        robot=robot,
        gateway=assembly.gateway,
    )


__all__ = [
    "AuthorityTransitionReceipt",
    "IsolatedAuthorityMode",
    "IsolatedAuthorityHardwareController",
    "IsolatedSlaveRolloutHardwareController",
    "StaticRolloutHardwareController",
    "make_authority_hardware_controller",
]
