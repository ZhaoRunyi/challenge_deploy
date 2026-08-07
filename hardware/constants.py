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

PIPER_ARM_IDS: tuple[str, str, str, str] = (
    "master_left",
    "master_right",
    "slave_left",
    "slave_right",
)

CAN_TOPOLOGIES: tuple[str, str] = ("shared", "isolated")

# Stable interface names are installed only after a human has confirmed which
# USB-to-CAN adapter belongs to which physical arm.  The repository deliberately
# does not infer role assignment from the kernel's transient can0..can3 order.
DEFAULT_CAN_NAMES: dict[str, str] = {
    "master_left": "piper_ml",
    "master_right": "piper_mr",
    "slave_left": "piper_sl",
    "slave_right": "piper_sr",
}

UNCONFIRMED_CAN_SERIAL = "REPLACE_WITH_CONFIRMED_ID_SERIAL_SHORT"

DEFAULT_PROMPT = "fold the cloth"

DUAL_PIPER_INIT_JOINTS: tuple[float, ...] = (
    -0.05918411,
    0.00076794,
    -0.12870058,
    -0.13548991,
    0.29586821,
    0.13372713,
    1.0,
    0.08932595,
    0.00970403,
    -0.21027726,
    -0.08838347,
    0.39285615,
    0.08686504,
    1.0,
)

# Keep the original deployment unit conversion constants instead of "fixing" them.
KAI0_RAD_PER_MILLI_DEGREE = 0.017444 / 1000.0
KAI0_MILLI_DEGREE_PER_RAD = 57324.840764
KAI0_GRIPPER_UNIT_SCALE = 1_000_000.0
KAI0_TRANSLATION_UNIT_SCALE = 1_000_000.0
PIPER_GRIPPER_FULL_OPEN_METERS = 0.10

# Historical Piper HDF5 data in this workspace was collected through
# `control_your_robot`'s PiperController, which used:
#   state["gripper"] = grippers_angle * 0.001 / 70
#   set_gripper(x)   = int(x * 70 * 1000)
# Newly corrected LeRobot datasets instead use a normalized opening ratio where
# 0.0 = fully closed and 1.0 ~= 100 mm open. Deploy defaults to that corrected
# convention, while `--old_gripper` keeps compatibility with the historical
# 70_000-scale raw values below.
LEGACY_PIPER_DATA_GRIPPER_UNIT_SCALE = 70_000.0

DEFAULT_ARM_STEP_LENGTH: tuple[float, ...] = (
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.2,
)
