from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np

from .constants import CAMERA_NAMES
from .schemas import RobotSnapshot


def jpeg_mapping(image: np.ndarray) -> np.ndarray:
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    return cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)


def resize_with_pad(images: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    resized = []
    for image in images:
        height, width = image.shape[:2]
        scale = min(target_w / width, target_h / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = image_resized
        resized.append(canvas)
    return np.asarray(resized)


class ObservationWindow:
    def __init__(self, maxlen: int = 2) -> None:
        self.window: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.window.append(
            {
                "qpos": None,
                "images": {camera_name: None for camera_name in CAMERA_NAMES},
            }
        )

    def update(self, snapshot: RobotSnapshot) -> dict[str, Any]:
        images = {
            camera_name: jpeg_mapping(snapshot.images[camera_name])
            for camera_name in CAMERA_NAMES
            if camera_name in snapshot.images
        }
        item = {
            "qpos": snapshot.state.qpos.copy(),
            "images": images,
            "timestamp_s": snapshot.timestamp_s,
        }
        self.window.append(item)
        return item

    @property
    def latest(self) -> dict[str, Any]:
        return self.window[-1]


def build_policy_payload(snapshot: RobotSnapshot, prompt: str) -> dict[str, Any]:
    image_arrs = [snapshot.images[camera_name] for camera_name in CAMERA_NAMES]
    image_arrs = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_arrs]
    image_arrs = resize_with_pad(np.asarray(image_arrs), 224, 224)
    return {
        "state": snapshot.state.qpos.copy(),
        "images": {
            "top_head": image_arrs[0].transpose(2, 0, 1),
            "hand_right": image_arrs[1].transpose(2, 0, 1),
            "hand_left": image_arrs[2].transpose(2, 0, 1),
        },
        "prompt": prompt,
    }
