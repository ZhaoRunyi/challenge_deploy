from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .constants import CAMERA_NAMES

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - depends on hardware env
    rs = None


@dataclass(slots=True)
class RealSenseDeviceInfo:
    name: str
    serial: str
    physical_port: str


def require_realsense() -> Any:
    if rs is None:
        raise RuntimeError("pyrealsense2 is not installed in the active environment")
    return rs


def list_realsense_devices() -> list[RealSenseDeviceInfo]:
    rs_mod = require_realsense()
    context = rs_mod.context()
    devices = []
    for device in context.query_devices():
        devices.append(
            RealSenseDeviceInfo(
                name=device.get_info(rs_mod.camera_info.name),
                serial=device.get_info(rs_mod.camera_info.serial_number),
                physical_port=device.get_info(rs_mod.camera_info.physical_port),
            )
        )
    return devices


class RealSenseRig:
    """Minimal RealSense multi-camera wrapper following the challenge layout."""

    def __init__(
        self,
        serials: dict[str, str],
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 30,
        enable_depth: bool = False,
    ) -> None:
        self.serials = {
            camera_name: serials[camera_name]
            for camera_name in CAMERA_NAMES
            if serials.get(camera_name)
        }
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.enable_depth = enable_depth
        self.pipelines: dict[str, Any] = {}
        self.started = False

    def start(self) -> None:
        rs_mod = require_realsense()
        if self.started:
            return
        for camera_name, serial in self.serials.items():
            pipeline = rs_mod.pipeline()
            config = rs_mod.config()
            config.enable_device(serial)
            config.enable_stream(rs_mod.stream.color, self.width, self.height, rs_mod.format.bgr8, self.fps)
            if self.enable_depth:
                config.enable_stream(rs_mod.stream.depth, self.width, self.height, rs_mod.format.z16, self.fps)
            pipeline.start(config)
            self.pipelines[camera_name] = pipeline

        for _ in range(self.warmup_frames):
            for pipeline in self.pipelines.values():
                pipeline.wait_for_frames(timeout_ms=5000)
        self.started = True

    def stop(self) -> None:
        for pipeline in self.pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                pass
        self.pipelines.clear()
        self.started = False

    def capture(self, timeout_ms: int = 1000) -> dict[str, np.ndarray]:
        if not self.started:
            self.start()

        images: dict[str, np.ndarray] = {}
        for camera_name, pipeline in self.pipelines.items():
            frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError(f"No color frame available from {camera_name}")
            images[camera_name] = np.asanyarray(color_frame.get_data()).copy()
        return images

    def save_snapshot(self, output_dir: str | Path) -> dict[str, str]:
        image_map = self.capture()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_paths: dict[str, str] = {}
        for camera_name, image in image_map.items():
            path = output / f"{timestamp}_{camera_name}.png"
            cv2.imwrite(str(path), image)
            saved_paths[camera_name] = str(path)
        return saved_paths

    def __enter__(self) -> "RealSenseRig":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
