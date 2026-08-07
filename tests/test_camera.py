"""RealSense enumeration, parallel capture, and cleanup checks."""

from __future__ import annotations

import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import hardware.realsense as realsense
from hardware.realsense import RealSenseRig


class ColorFrame:
    def __init__(self, value: int, timestamp_s: float) -> None:
        self.value = value
        self.timestamp_s = timestamp_s

    def get_data(self) -> np.ndarray:
        return np.full((2, 3, 3), self.value, dtype=np.uint8)

    def get_timestamp(self) -> float:
        return self.timestamp_s * 1000.0


class FrameSet:
    def __init__(self, color_frame: ColorFrame) -> None:
        self.color_frame = color_frame

    def get_color_frame(self) -> ColorFrame:
        return self.color_frame


class ConcurrentPipeline:
    def __init__(self, barrier: threading.Barrier, frame: ColorFrame) -> None:
        self.barrier = barrier
        self.frame = frame
        self.timeouts: list[int] = []
        self.thread_ids: list[int] = []

    def wait_for_frames(self, *, timeout_ms: int) -> FrameSet:
        self.timeouts.append(timeout_ms)
        self.thread_ids.append(threading.get_ident())
        self.barrier.wait(timeout=0.5)
        return FrameSet(self.frame)


def realsense_module(pipelines: list[object]) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=mock.Mock(side_effect=pipelines),
        config=mock.Mock(side_effect=lambda: mock.Mock()),
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
    )


class CameraTest(unittest.TestCase):
    def test_realsense_device_enumeration_uses_the_module_binding(self) -> None:
        camera_info = SimpleNamespace(
            name="name",
            serial_number="serial",
            physical_port="physical_port",
        )
        device = mock.Mock()
        device.get_info.side_effect = {
            "name": "Intel RealSense",
            "serial": "camera_serial",
            "physical_port": "usb_port",
        }.__getitem__
        context = mock.Mock()
        context.query_devices.return_value = [device]
        module = SimpleNamespace(
            context=mock.Mock(return_value=context),
            camera_info=camera_info,
        )

        with mock.patch.object(realsense, "rs", module):
            devices = realsense.list_realsense_devices()

        self.assertEqual(devices[0].serial, "camera_serial")
        module.context.assert_called_once_with()

    def test_multiple_cameras_capture_in_parallel_with_independent_timestamps(self) -> None:
        barrier = threading.Barrier(2)
        left = ConcurrentPipeline(barrier, ColorFrame(1, 101.25))
        right = ConcurrentPipeline(barrier, ColorFrame(2, 101.50))
        rig = RealSenseRig(
            {"cam_left": "left_serial", "cam_right": "right_serial"},
            warmup_frames=0,
        )
        rig.pipelines = {"cam_left": left, "cam_right": right}
        rig.started = True

        with mock.patch.object(
            realsense,
            "realsense_frame_timestamp_s",
            side_effect=lambda frame, fallback_s: frame.timestamp_s,
        ):
            capture = rig.capture_frames(timeout_ms=37, parallel=True)

        self.assertEqual(left.timeouts, [37])
        self.assertEqual(right.timeouts, [37])
        self.assertNotEqual(left.thread_ids[0], right.thread_ids[0])
        self.assertEqual(
            capture.color_timestamps_s,
            {"cam_left": 101.25, "cam_right": 101.50},
        )
        np.testing.assert_array_equal(
            capture.color_images["cam_right"],
            np.full((2, 3, 3), 2, dtype=np.uint8),
        )

    def test_start_failure_stops_every_pipeline_that_was_acquired(self) -> None:
        for failure_stage in ("second_camera", "warmup"):
            with self.subTest(failure_stage=failure_stage):
                first = mock.Mock()
                pipelines = [first]
                serials = {"cam_high": "high_serial"}
                warmup_frames = 1
                if failure_stage == "second_camera":
                    second = mock.Mock()
                    second.start.side_effect = RuntimeError("second camera failed")
                    pipelines.append(second)
                    serials["cam_right"] = "right_serial"
                    warmup_frames = 0
                else:
                    first.wait_for_frames.side_effect = TimeoutError("warmup failed")

                rig = RealSenseRig(serials, warmup_frames=warmup_frames)
                with mock.patch.object(
                    realsense,
                    "rs",
                    realsense_module(pipelines),
                ):
                    with self.assertRaises((RuntimeError, TimeoutError)):
                        rig.start()

                first.stop.assert_called_once_with()
                self.assertEqual(rig.pipelines, {})
                self.assertFalse(rig.started)


if __name__ == "__main__":
    unittest.main()
