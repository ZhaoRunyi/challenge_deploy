from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

import numpy as np
from piper_sdk import C_PiperInterface, LogLevel

from .constants import DEFAULT_ARM_STEP_LENGTH
from .conversions import (
    joints_feedback_to_rad,
    joints_rad_to_sdk,
    opening_to_sdk_gripper,
    sdk_gripper_to_opening,
    sdk_joint_to_rad,
    sdk_translation_to_meters,
)
from .schemas import DualPiperState, PiperArmState


class MotionNotAllowedError(RuntimeError):
    pass


DEFAULT_GRIPPER_EFFORT = 1000
MAX_GRIPPER_EFFORT = 5000


def _motor_field(container: Any, index: int) -> Any:
    return getattr(container, f"motor_{index}")


def _bool_attr(container: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(container, name, default))


def _joint_values_from_sdk(joint_state: Any) -> np.ndarray:
    return joints_feedback_to_rad(
        [
            joint_state.joint_1,
            joint_state.joint_2,
            joint_state.joint_3,
            joint_state.joint_4,
            joint_state.joint_5,
            joint_state.joint_6,
        ]
    )


def _gripper_effort(value: int | None) -> int:
    if value is None:
        return DEFAULT_GRIPPER_EFFORT
    effort = int(value)
    if not 0 <= effort <= MAX_GRIPPER_EFFORT:
        raise ValueError(f"gripper_effort must be in [0, {MAX_GRIPPER_EFFORT}], got {effort}")
    return effort


def _status_to_dict(status: Any) -> dict[str, Any]:
    err_status = getattr(status, "err_status", None)
    error_dict = {
        key: bool(getattr(err_status, key))
        for key in (
            "joint_1_angle_limit",
            "joint_2_angle_limit",
            "joint_3_angle_limit",
            "joint_4_angle_limit",
            "joint_5_angle_limit",
            "joint_6_angle_limit",
            "communication_status_joint_1",
            "communication_status_joint_2",
            "communication_status_joint_3",
            "communication_status_joint_4",
            "communication_status_joint_5",
            "communication_status_joint_6",
        )
        if err_status is not None and hasattr(err_status, key)
    }
    return {
        "ctrl_mode": int(getattr(status, "ctrl_mode", 0)),
        "arm_status": int(getattr(status, "arm_status", 0)),
        "mode_feed": int(getattr(status, "mode_feed", 0)),
        "teach_status": int(getattr(status, "teach_status", 0)),
        "motion_status": int(getattr(status, "motion_status", 0)),
        "trajectory_num": int(getattr(status, "trajectory_num", 0)),
        "errors": error_dict,
    }


@dataclass(slots=True)
class PiperProbeResult:
    arm_name: str
    can_name: str
    connected: bool
    samples: list[dict[str, Any]]


class SinglePiperArm:
    def __init__(
        self,
        *,
        name: str,
        can_name: str,
        commands_enabled: bool = True,
        logger_level: LogLevel = LogLevel.WARNING,
    ) -> None:
        self.name = name
        self.can_name = can_name
        self.commands_enabled = commands_enabled
        self.interface = C_PiperInterface(
            can_name=can_name,
            logger_level=logger_level,
            log_to_file=False,
        )
        self.connected = False

    def _require_motion_allowed(self, action: str) -> None:
        if not self.commands_enabled:
            raise MotionNotAllowedError(
                f"{self.name} is running in read-only mode; refusing to {action}. "
                "Use a read-only probe path only when you want to avoid sending commands."
            )

    def connect(self, *, read_only: bool = True) -> None:
        self.interface.ConnectPort(can_init=False, piper_init=not read_only, start_thread=True)
        self.connected = True

    def disconnect(self) -> None:
        if self.connected:
            try:
                self.interface.DisconnectPort()
            finally:
                self.connected = False

    def is_enabled(self) -> bool:
        low_spd = self.interface.GetArmLowSpdInfoMsgs()
        return all(
            _bool_attr(_motor_field(low_spd, idx).foc_status, "driver_enable_status")
            for idx in range(1, 7)
        )

    def enable(self, *, retries: int = 5, sleep_s: float = 0.5) -> bool:
        self._require_motion_allowed("enable the arm")
        for _ in range(retries):
            self.interface.EnableArm(7)
            # Match the legacy ROS deployment path: after reconnect or power-cycle,
            # the gripper often needs a reset-style pulse before normal enable.
            self.interface.GripperCtrl(0, 1000, 0x02, 0)
            self.interface.GripperCtrl(0, 1000, 0x01, 0)
            time.sleep(sleep_s)
            if self.is_enabled():
                return True
        return self.is_enabled()

    def disable(self) -> None:
        self._require_motion_allowed("disable the arm")
        self.interface.DisableArm(7)
        self.interface.GripperCtrl(0, 1000, 0x00, 0)

    def set_joint_mode(self, *, speed_percent: int = 100) -> None:
        self._require_motion_allowed("switch to joint control mode")
        self.interface.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)

    def set_cartesian_mode(self, *, speed_percent: int = 50) -> None:
        self._require_motion_allowed("switch to Cartesian control mode")
        self.interface.MotionCtrl_2(0x01, 0x00, speed_percent, 0x00)

    def configure_as_master_input(self) -> None:
        self._require_motion_allowed("configure the arm as a master input arm")
        self.interface.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00)

    def configure_as_slave_output(self) -> None:
        self._require_motion_allowed("configure the arm as a slave output arm")
        self.interface.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)

    def enter_teach_mode(self) -> None:
        self._require_motion_allowed("enter teach mode")
        self.interface.MotionCtrl_1(grag_teach_ctrl=0x01)

    def exit_teach_mode(self) -> None:
        self._require_motion_allowed("exit teach mode")
        self.interface.MotionCtrl_1(grag_teach_ctrl=0x02)

    def command_joint_positions(
        self,
        qpos: Iterable[float],
        *,
        speed_percent: int = 100,
        gripper_effort: int | None = None,
    ) -> None:
        self._require_motion_allowed("send joint commands")
        qpos_arr = np.asarray(list(qpos), dtype=np.float64)
        if qpos_arr.shape != (7,):
            raise ValueError(f"{self.name} expects 7 DoF input, got shape {qpos_arr.shape}")

        sdk_joints = joints_rad_to_sdk(qpos_arr[:6])
        sdk_gripper = opening_to_sdk_gripper(qpos_arr[6])
        effort = _gripper_effort(gripper_effort)
        self.set_joint_mode(speed_percent=speed_percent)
        self.interface.JointCtrl(
            int(sdk_joints[0]),
            int(sdk_joints[1]),
            int(sdk_joints[2]),
            int(sdk_joints[3]),
            int(sdk_joints[4]),
            int(sdk_joints[5]),
        )
        self.interface.GripperCtrl(abs(int(sdk_gripper)), effort, 0x01, 0)
        self.set_joint_mode(speed_percent=speed_percent)

    def command_end_pose(
        self,
        pose: Iterable[float],
        *,
        speed_percent: int = 50,
        gripper_effort: int | None = None,
    ) -> None:
        self._require_motion_allowed("send end-effector commands")
        pose_arr = np.asarray(list(pose), dtype=np.float64)
        if pose_arr.shape != (7,):
            raise ValueError(f"{self.name} expects 7-D pose input, got shape {pose_arr.shape}")

        x, y, z, rx, ry, rz, gripper = pose_arr
        effort = _gripper_effort(gripper_effort)
        self.set_cartesian_mode(speed_percent=speed_percent)
        self.interface.EndPoseCtrl(
            int(round(x * 1_000_000.0)),
            int(round(y * 1_000_000.0)),
            int(round(z * 1_000_000.0)),
            int(round(rx * 57_324.840764)),
            int(round(ry * 57_324.840764)),
            int(round(rz * 57_324.840764)),
        )
        self.interface.GripperCtrl(abs(opening_to_sdk_gripper(gripper)), effort, 0x01, 0)
        self.set_cartesian_mode(speed_percent=speed_percent)

    def read_state(self, *, prefer_joint_ctrl: bool = False) -> PiperArmState:
        joint_feedback_msgs = self.interface.GetArmJointMsgs()
        joint_ctrl_msgs = self.interface.GetArmJointCtrl()
        gripper_feedback_msgs = self.interface.GetArmGripperMsgs()
        gripper_ctrl_msgs = self.interface.GetArmGripperCtrl()
        end_pose_msgs = self.interface.GetArmEndPoseMsgs()
        status_msgs = self.interface.GetArmStatus()
        high_spd = self.interface.GetArmHighSpdInfoMsgs()

        qpos_feedback = np.concatenate(
            (
                _joint_values_from_sdk(joint_feedback_msgs.joint_state),
                np.array([sdk_gripper_to_opening(gripper_feedback_msgs.gripper_state.grippers_angle)]),
            ),
            axis=0,
        )
        qpos_command = np.concatenate(
            (
                _joint_values_from_sdk(joint_ctrl_msgs.joint_ctrl),
                np.array([sdk_gripper_to_opening(gripper_ctrl_msgs.gripper_ctrl.grippers_angle)]),
            ),
            axis=0,
        )

        use_command = prefer_joint_ctrl and not np.allclose(qpos_command[:6], 0.0, atol=1e-6)
        qpos = qpos_command if use_command else qpos_feedback
        qvel = np.array(
            [
                _motor_field(high_spd, 1).motor_speed / 1000.0,
                _motor_field(high_spd, 2).motor_speed / 1000.0,
                _motor_field(high_spd, 3).motor_speed / 1000.0,
                _motor_field(high_spd, 4).motor_speed / 1000.0,
                _motor_field(high_spd, 5).motor_speed / 1000.0,
                _motor_field(high_spd, 6).motor_speed / 1000.0,
                0.0,
            ],
            dtype=np.float64,
        )
        effort = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                gripper_feedback_msgs.gripper_state.grippers_effort / 1000.0,
            ],
            dtype=np.float64,
        )
        end_pose = np.array(
            [
                sdk_translation_to_meters(end_pose_msgs.end_pose.X_axis),
                sdk_translation_to_meters(end_pose_msgs.end_pose.Y_axis),
                sdk_translation_to_meters(end_pose_msgs.end_pose.Z_axis),
                sdk_joint_to_rad(end_pose_msgs.end_pose.RX_axis),
                sdk_joint_to_rad(end_pose_msgs.end_pose.RY_axis),
                sdk_joint_to_rad(end_pose_msgs.end_pose.RZ_axis),
                sdk_gripper_to_opening(gripper_feedback_msgs.gripper_state.grippers_angle),
            ],
            dtype=np.float64,
        )

        return PiperArmState(
            name=self.name,
            can_name=self.can_name,
            qpos=qpos.astype(np.float64),
            qpos_feedback=qpos_feedback.astype(np.float64),
            qpos_command=qpos_command.astype(np.float64),
            qvel=qvel,
            effort=effort,
            end_pose=end_pose,
            enabled=self.is_enabled(),
            status=_status_to_dict(status_msgs.arm_status),
            feedback_hz=float(getattr(joint_feedback_msgs, "Hz", 0.0)),
            status_hz=float(getattr(status_msgs, "Hz", 0.0)),
            command_hz=float(getattr(joint_ctrl_msgs, "Hz", 0.0)),
        )

    def move_to_joint_positions(
        self,
        target_qpos: Iterable[float],
        *,
        hz: float = 30.0,
        step_sizes: Iterable[float] = DEFAULT_ARM_STEP_LENGTH,
        speed_percent: int = 100,
        gripper_effort: int | None = None,
    ) -> None:
        self._require_motion_allowed("move the arm")
        current = self.read_state().qpos
        target = np.asarray(list(target_qpos), dtype=np.float64)
        step = np.asarray(list(step_sizes), dtype=np.float64)
        if target.shape != (7,):
            raise ValueError(f"{self.name} expects a 7-D target, got {target.shape}")
        if step.shape != (7,):
            raise ValueError(f"{self.name} expects 7 step sizes, got {step.shape}")

        while True:
            diff = target - current
            if np.all(np.abs(diff) <= step):
                self.command_joint_positions(
                    target,
                    speed_percent=speed_percent,
                    gripper_effort=gripper_effort,
                )
                break
            current = np.where(np.abs(diff) <= step, target, current + np.sign(diff) * step)
            self.command_joint_positions(
                current,
                speed_percent=speed_percent,
                gripper_effort=gripper_effort,
            )
            time.sleep(1.0 / hz)

    def probe(self, *, samples: int = 5, interval_s: float = 0.2, prefer_joint_ctrl: bool = False) -> PiperProbeResult:
        result_samples: list[dict[str, Any]] = []
        for _ in range(samples):
            state = self.read_state(prefer_joint_ctrl=prefer_joint_ctrl)
            result_samples.append(
                {
                    "qpos": state.qpos.tolist(),
                    "feedback_hz": state.feedback_hz,
                    "status_hz": state.status_hz,
                    "command_hz": state.command_hz,
                    "enabled": state.enabled,
                    "ctrl_mode": state.status.get("ctrl_mode"),
                    "arm_status": state.status.get("arm_status"),
                }
            )
            time.sleep(interval_s)
        return PiperProbeResult(
            arm_name=self.name,
            can_name=self.can_name,
            connected=self.connected,
            samples=result_samples,
        )


class DualPiperSystem:
    def __init__(
        self,
        *,
        left_can_name: str,
        right_can_name: str,
        commands_enabled: bool = True,
        prefer_joint_ctrl: bool = False,
        name: str = "dual_piper",
    ) -> None:
        self.name = name
        self.commands_enabled = commands_enabled
        self.prefer_joint_ctrl = prefer_joint_ctrl
        self.left = SinglePiperArm(
            name="left_arm",
            can_name=left_can_name,
            commands_enabled=commands_enabled,
        )
        self.right = SinglePiperArm(
            name="right_arm",
            can_name=right_can_name,
            commands_enabled=commands_enabled,
        )

    def connect(self, *, read_only: bool = True) -> None:
        self.left.connect(read_only=read_only)
        self.right.connect(read_only=read_only)

    def disconnect(self) -> None:
        self.left.disconnect()
        self.right.disconnect()

    def read_state(self, *, prefer_joint_ctrl: bool | None = None) -> DualPiperState:
        use_joint_ctrl = self.prefer_joint_ctrl if prefer_joint_ctrl is None else prefer_joint_ctrl
        left_state = self.left.read_state(prefer_joint_ctrl=use_joint_ctrl)
        right_state = self.right.read_state(prefer_joint_ctrl=use_joint_ctrl)
        return DualPiperState(left=left_state, right=right_state)

    def enable(self) -> bool:
        return self.left.enable() and self.right.enable()

    def disable(self) -> None:
        self.left.disable()
        self.right.disable()

    def set_joint_positions(
        self,
        qpos: Iterable[float],
        *,
        speed_percent: int = 100,
        gripper_effort: int | None = None,
    ) -> None:
        values = np.asarray(list(qpos), dtype=np.float64)
        if values.shape != (14,):
            raise ValueError(f"{self.name} expects a 14-D target, got {values.shape}")
        self.left.command_joint_positions(
            values[:7],
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
        )
        self.right.command_joint_positions(
            values[7:],
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
        )

    def move_to_joint_positions(
        self,
        qpos: Iterable[float],
        *,
        hz: float = 30.0,
        step_sizes: Iterable[float] = DEFAULT_ARM_STEP_LENGTH,
        speed_percent: int = 100,
        gripper_effort: int | None = None,
    ) -> None:
        values = np.asarray(list(qpos), dtype=np.float64)
        if values.shape != (14,):
            raise ValueError(f"{self.name} expects a 14-D target, got {values.shape}")
        self.left.move_to_joint_positions(
            values[:7],
            hz=hz,
            step_sizes=step_sizes,
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
        )
        self.right.move_to_joint_positions(
            values[7:],
            hz=hz,
            step_sizes=step_sizes,
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
        )

    def configure_masters_for_teaching(self, *, align_to: np.ndarray | None = None, hz: float = 10.0) -> None:
        self.left.configure_as_master_input()
        self.right.configure_as_master_input()
        time.sleep(0.2)
        self.enable()
        time.sleep(0.2)
        if align_to is not None:
            self.move_to_joint_positions(align_to, hz=hz)
            time.sleep(0.2)
        self.left.enter_teach_mode()
        self.right.enter_teach_mode()

    def configure_slaves_for_following(self) -> None:
        self.left.configure_as_slave_output()
        self.right.configure_as_slave_output()
        time.sleep(0.2)
        self.enable()

    def probe(self, *, samples: int = 5, interval_s: float = 0.2, prefer_joint_ctrl: bool | None = None) -> dict[str, Any]:
        use_joint_ctrl = self.prefer_joint_ctrl if prefer_joint_ctrl is None else prefer_joint_ctrl
        left_probe = self.left.probe(samples=samples, interval_s=interval_s, prefer_joint_ctrl=use_joint_ctrl)
        right_probe = self.right.probe(samples=samples, interval_s=interval_s, prefer_joint_ctrl=use_joint_ctrl)
        return {
            "name": self.name,
            "left": {
                "can_name": left_probe.can_name,
                "samples": left_probe.samples,
            },
            "right": {
                "can_name": right_probe.can_name,
                "samples": right_probe.samples,
            },
        }
