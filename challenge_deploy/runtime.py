from __future__ import annotations

import time
from typing import Any

from .piper import DualPiperSystem
from .realsense import RealSenseRig
from .schemas import RobotSnapshot


class DualPiperObservationSource:
    def __init__(
        self,
        *,
        robot: DualPiperSystem,
        cameras: RealSenseRig | None = None,
    ) -> None:
        self.robot = robot
        self.cameras = cameras

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
        images = self.cameras.capture() if self.cameras is not None else {}
        state = self.robot.read_state()
        return RobotSnapshot(timestamp_s=time.time(), state=state, images=images)

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
