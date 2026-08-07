from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass(slots=True)
class RealSenseDeviceInfo:
    name: str
    serial: str
    physical_port: str


@dataclass(slots=True)
class RealSenseCapture:
    color_images: dict[str, np.ndarray]
    depth_images: dict[str, np.ndarray]
    color_timestamps_s: dict[str, float]
    depth_timestamps_s: dict[str, float]


@dataclass(slots=True)
class RealSenseCameraCapture:
    timestamp_s: float
    color_image: np.ndarray
    depth_image: np.ndarray | None
    color_frame_timestamp_ms: float | None
    depth_frame_timestamp_ms: float | None
    depth_timestamp_s: float | None = None


def list_realsense_devices() -> list[RealSenseDeviceInfo]:
    context = rs.context()
    devices = []
    for device in context.query_devices():
        devices.append(
            RealSenseDeviceInfo(
                name=device.get_info(rs.camera_info.name),
                serial=device.get_info(rs.camera_info.serial_number),
                physical_port=device.get_info(rs.camera_info.physical_port),
            )
        )
    return devices


def realsense_frame_timestamp_s(frame: Any, fallback_s: float) -> float:
    try:
        timestamp_ms = float(frame.get_timestamp())
        timestamp_domain = frame.get_frame_timestamp_domain()
        if timestamp_domain == rs.timestamp_domain.system_time and timestamp_ms > 0.0:
            return timestamp_ms / 1000.0
    except Exception:
        pass
    return fallback_s


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
        self.serials = {str(camera_name): str(serial) for camera_name, serial in serials.items() if serial}
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.enable_depth = enable_depth
        self.pipelines: dict[str, Any] = {}
        self.started = False
        self.last_capture: RealSenseCapture | None = None

    def start(self) -> None:
        if self.started:
            return
        try:
            for camera_name, serial in self.serials.items():
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(
                    rs.stream.color,
                    self.width,
                    self.height,
                    rs.format.bgr8,
                    self.fps,
                )
                if self.enable_depth:
                    config.enable_stream(
                        rs.stream.depth,
                        self.width,
                        self.height,
                        rs.format.z16,
                        self.fps,
                    )
                pipeline.start(config)
                self.pipelines[camera_name] = pipeline

            for _warmup_index in range(self.warmup_frames):
                for pipeline in self.pipelines.values():
                    pipeline.wait_for_frames(timeout_ms=5000)
            self.started = True
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        for pipeline in self.pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                pass
        self.pipelines.clear()
        self.started = False

    def capture_frames(
        self,
        timeout_ms: int = 1000,
        *,
        parallel: bool = False,
    ) -> RealSenseCapture:
        if not self.started:
            self.start()

        camera_names = tuple(self.pipelines)
        if parallel and len(camera_names) > 1:
            with ThreadPoolExecutor(
                max_workers=len(camera_names),
                thread_name_prefix="realsense_capture",
            ) as executor:
                futures = {
                    camera_name: executor.submit(
                        self.capture_camera_frame,
                        camera_name,
                        timeout_ms=timeout_ms,
                    )
                    for camera_name in camera_names
                }
                captures = {
                    camera_name: futures[camera_name].result()
                    for camera_name in camera_names
                }
        else:
            captures = {
                camera_name: self.capture_camera_frame(
                    camera_name,
                    timeout_ms=timeout_ms,
                )
                for camera_name in camera_names
            }

        color_images: dict[str, np.ndarray] = {}
        depth_images: dict[str, np.ndarray] = {}
        color_timestamps_s: dict[str, float] = {}
        depth_timestamps_s: dict[str, float] = {}
        for camera_name, capture in captures.items():
            color_images[camera_name] = capture.color_image
            color_timestamps_s[camera_name] = float(capture.timestamp_s)
            if self.enable_depth:
                if capture.depth_image is None:
                    raise RuntimeError(f"No depth frame available from {camera_name}")
                depth_images[camera_name] = capture.depth_image
                if capture.depth_timestamp_s is None:
                    raise RuntimeError(
                        f"No depth timestamp available from {camera_name}"
                    )
                depth_timestamps_s[camera_name] = float(capture.depth_timestamp_s)
        capture = RealSenseCapture(
            color_images=color_images,
            depth_images=depth_images,
            color_timestamps_s=color_timestamps_s,
            depth_timestamps_s=depth_timestamps_s,
        )
        self.last_capture = capture
        return capture

    def capture_camera_frame(self, camera_name: str, timeout_ms: int = 1000) -> RealSenseCameraCapture:
        if not self.started:
            self.start()
        if camera_name not in self.pipelines:
            raise KeyError(f"Camera {camera_name!r} is not configured or has not been started")

        frames = self.pipelines[camera_name].wait_for_frames(timeout_ms=timeout_ms)
        receive_timestamp_s = time.time()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"No color frame available from {camera_name}")

        depth_frame = frames.get_depth_frame() if self.enable_depth else None
        if self.enable_depth and not depth_frame:
            raise RuntimeError(f"No depth frame available from {camera_name}")

        color_timestamp_ms = float(color_frame.get_timestamp()) if hasattr(color_frame, "get_timestamp") else None
        depth_timestamp_ms = (
            float(depth_frame.get_timestamp())
            if depth_frame is not None and hasattr(depth_frame, "get_timestamp")
            else None
        )
        timestamp_s = realsense_frame_timestamp_s(color_frame, receive_timestamp_s)
        depth_timestamp_s = realsense_frame_timestamp_s(depth_frame, receive_timestamp_s) if depth_frame is not None else None
        return RealSenseCameraCapture(
            timestamp_s=timestamp_s,
            color_image=np.asanyarray(color_frame.get_data()).copy(),
            depth_image=np.asanyarray(depth_frame.get_data()).copy() if depth_frame is not None else None,
            color_frame_timestamp_ms=color_timestamp_ms,
            depth_frame_timestamp_ms=depth_timestamp_ms,
            depth_timestamp_s=depth_timestamp_s,
        )

    def capture(self, timeout_ms: int = 1000) -> dict[str, np.ndarray]:
        return self.capture_frames(timeout_ms=timeout_ms).color_images

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
