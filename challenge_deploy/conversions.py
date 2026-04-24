from __future__ import annotations

import numpy as np

from .constants import (
    KAI0_GRIPPER_UNIT_SCALE,
    KAI0_MILLI_DEGREE_PER_RAD,
    KAI0_RAD_PER_MILLI_DEGREE,
    KAI0_TRANSLATION_UNIT_SCALE,
)


def sdk_joint_to_rad(value: int | float) -> float:
    return float(value) * KAI0_RAD_PER_MILLI_DEGREE


def rad_to_sdk_joint(value: int | float) -> int:
    return int(round(float(value) * KAI0_MILLI_DEGREE_PER_RAD))


def sdk_gripper_to_opening(value: int | float) -> float:
    return float(value) / KAI0_GRIPPER_UNIT_SCALE


def opening_to_sdk_gripper(value: int | float) -> int:
    return int(round(max(0.0, float(value)) * KAI0_GRIPPER_UNIT_SCALE))


def sdk_translation_to_meters(value: int | float) -> float:
    return float(value) / KAI0_TRANSLATION_UNIT_SCALE


def meters_to_sdk_translation(value: int | float) -> int:
    return int(round(float(value) * KAI0_TRANSLATION_UNIT_SCALE))


def joints_feedback_to_rad(joints: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
    values = np.asarray(joints, dtype=np.float64)
    return values * KAI0_RAD_PER_MILLI_DEGREE


def joints_rad_to_sdk(joints: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    values = np.asarray(joints, dtype=np.float64)
    return np.rint(values * KAI0_MILLI_DEGREE_PER_RAD).astype(np.int64)
