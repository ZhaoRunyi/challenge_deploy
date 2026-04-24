from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import CAMERA_NAMES


@dataclass(slots=True)
class PiperArmState:
    name: str
    can_name: str
    qpos: np.ndarray
    qpos_feedback: np.ndarray
    qpos_command: np.ndarray
    qvel: np.ndarray
    effort: np.ndarray
    end_pose: np.ndarray
    enabled: bool
    status: dict[str, Any]
    feedback_hz: float
    status_hz: float
    command_hz: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "can_name": self.can_name,
            "qpos": self.qpos.tolist(),
            "qpos_feedback": self.qpos_feedback.tolist(),
            "qpos_command": self.qpos_command.tolist(),
            "qvel": self.qvel.tolist(),
            "effort": self.effort.tolist(),
            "end_pose": self.end_pose.tolist(),
            "enabled": self.enabled,
            "status": self.status,
            "feedback_hz": self.feedback_hz,
            "status_hz": self.status_hz,
            "command_hz": self.command_hz,
        }


@dataclass(slots=True)
class DualPiperState:
    left: PiperArmState
    right: PiperArmState

    @property
    def qpos(self) -> np.ndarray:
        return np.concatenate((self.left.qpos, self.right.qpos), axis=0)

    @property
    def qvel(self) -> np.ndarray:
        return np.concatenate((self.left.qvel, self.right.qvel), axis=0)

    @property
    def effort(self) -> np.ndarray:
        return np.concatenate((self.left.effort, self.right.effort), axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "qpos": self.qpos.tolist(),
            "qvel": self.qvel.tolist(),
            "effort": self.effort.tolist(),
        }


@dataclass(slots=True)
class RobotSnapshot:
    timestamp_s: float
    state: DualPiperState
    images: dict[str, np.ndarray] = field(default_factory=dict)

    def to_collector_observation(self) -> OrderedDict[str, Any]:
        ordered_images = {
            camera_name: self.images[camera_name]
            for camera_name in CAMERA_NAMES
            if camera_name in self.images
        }
        observation = OrderedDict()
        observation["images"] = ordered_images
        observation["qpos"] = self.state.qpos.astype(np.float64)
        observation["qvel"] = self.state.qvel.astype(np.float64)
        observation["effort"] = self.state.effort.astype(np.float64)
        observation["base_vel"] = np.array([0.0, 0.0], dtype=np.float64)
        return observation
