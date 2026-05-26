from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .constants import (
    DEFAULT_ARM_STEP_LENGTH,
    DEFAULT_CAMERA_SERIALS,
    DEFAULT_CAN_NAMES,
    DEFAULT_PROMPT,
)


def default_config() -> dict[str, Any]:
    return {
        "robot": {
            "left": {"can_name": DEFAULT_CAN_NAMES["left"]},
            "right": {"can_name": DEFAULT_CAN_NAMES["right"]},
            "master_left": {"can_name": DEFAULT_CAN_NAMES["master_left"]},
            "master_right": {"can_name": DEFAULT_CAN_NAMES["master_right"]},
        },
        "cameras": {
            "enabled": True,
            "width": 640,
            "height": 480,
            "fps": 30,
            "warmup_frames": 30,
            "serials": {
                "cam_high": DEFAULT_CAMERA_SERIALS["cam_high"],
                "cam_right_wrist": DEFAULT_CAMERA_SERIALS["cam_right_wrist"],
                "cam_left_wrist": DEFAULT_CAMERA_SERIALS["cam_left_wrist"],
            },
        },
        "policy": {
            "host": "127.0.0.1",
            "port": 8000,
            "prompt": DEFAULT_PROMPT,
            "inference_rate": 3.0,
            "chunk_size": 50,
            "latency_k": 8,
            "min_smooth_steps": 8,
            "buffer_max_chunks": 10,
        },
        "runtime": {
            "publish_rate": 30,
            "max_publish_step": 10000,
            "arm_steps_length": list(DEFAULT_ARM_STEP_LENGTH),
            "right_gripper_offset": 0.003,
        },
        "dataset": {
            "dataset_dir": "./data",
            "dataset_name": "aloha_mobile_dummy",
            "video_fps": 30,
            "export_video": True,
        },
    }


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = default_config()
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return merge_dicts(config, loaded)


def set_by_dotted_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value
