from __future__ import annotations

CAMERA_NAMES: tuple[str, str, str] = (
    "cam_high",
    "cam_right_wrist",
    "cam_left_wrist",
)

DEFAULT_CAMERA_SERIALS: dict[str, str] = {
    "cam_high": "323422071854",
    "cam_left_wrist": "344322073012",
    "cam_right_wrist": "335522070790",
}

DEFAULT_CAN_NAMES: dict[str, str] = {
    "left": "can0",
    "right": "can1",
    "master_left": "can_left_mas",
    "master_right": "can_right_mas",
}

DEFAULT_PROMPT = "fold the cloth"

# Keep the original deployment unit conversion constants instead of "fixing" them.
KAI0_RAD_PER_MILLI_DEGREE = 0.017444 / 1000.0
KAI0_MILLI_DEGREE_PER_RAD = 57324.840764
KAI0_GRIPPER_UNIT_SCALE = 1_000_000.0
KAI0_TRANSLATION_UNIT_SCALE = 1_000_000.0

DEFAULT_ARM_STEP_LENGTH: tuple[float, ...] = (
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.2,
)
