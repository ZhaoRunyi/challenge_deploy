from __future__ import annotations

import time
from typing import Any

from .piper import DualPiperSystem
from .realsense import RealSenseRig
from .schemas import DualPiperState, RobotSnapshot


class DualPiperArmView:
    """A non-owning left/right view over an existing pair of physical arms."""

    def __init__(self, *, left: Any, right: Any, prefer_joint_ctrl: bool = False) -> None:
        self.left = left
        self.right = right
        self.prefer_joint_ctrl = bool(prefer_joint_ctrl)

    def read_state(self, *, prefer_joint_ctrl: bool | None = None) -> DualPiperState:
        use_joint_control = (
            self.prefer_joint_ctrl
            if prefer_joint_ctrl is None
            else bool(prefer_joint_ctrl)
        )
        return DualPiperState(
            left=self.left.read_state(prefer_joint_ctrl=use_joint_control),
            right=self.right.read_state(prefer_joint_ctrl=use_joint_control),
        )


class DualPiperObservationSource:
    def __init__(
        self,
        *,
        robot: DualPiperSystem,
        cameras: RealSenseRig | None = None,
        camera_timeout_ms: int | None = None,
        parallel_camera_capture: bool = False,
    ) -> None:
        self.robot = robot
        self.cameras = cameras
        self.camera_timeout_ms = camera_timeout_ms
        self.parallel_camera_capture = bool(parallel_camera_capture)
        if camera_timeout_ms is not None and camera_timeout_ms <= 0:
            raise ValueError("camera_timeout_ms must be positive when provided")

    def wait_until_ready(self, timeout_s: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout_s:
            try:
                self.capture_snapshot()
                return True
            except Exception:
                time.sleep(0.2)
        return False

    def capture_snapshot(self) -> RobotSnapshot:
        source_timestamps: dict[str, float] = {}
        color_timestamps: dict[str, float] = {}
        if self.cameras is None:
            images = {}
        elif self.camera_timeout_ms is None:
            images = self.cameras.capture()
            capture = getattr(self.cameras, "last_capture", None)
            color_timestamps = getattr(capture, "color_timestamps_s", {})
        else:
            capture = self.cameras.capture_frames(
                timeout_ms=self.camera_timeout_ms,
                parallel=self.parallel_camera_capture,
            )
            images = capture.color_images
            color_timestamps = capture.color_timestamps_s
        source_timestamps.update(
            {
                f"camera_{name}": float(timestamp_s)
                for name, timestamp_s in color_timestamps.items()
            }
        )
        source_timestamps.update(
            {
                f"camera_{name}_color_time": float(timestamp_s)
                for name, timestamp_s in color_timestamps.items()
            }
        )
        state = self.robot.read_state()
        return RobotSnapshot(
            timestamp_s=time.time(),
            state=state,
            images=images,
            source_timestamps=source_timestamps,
        )

    def save_snapshot(self, output_dir: str) -> dict[str, Any]:
        snapshot = self.capture_snapshot()
        image_paths = self.cameras.save_snapshot(output_dir) if self.cameras is not None else {}
        return {
            "timestamp_s": snapshot.timestamp_s,
            "qpos": snapshot.state.qpos.tolist(),
            "qvel": snapshot.state.qvel.tolist(),
            "effort": snapshot.state.effort.tolist(),
            "images": image_paths,
        }
