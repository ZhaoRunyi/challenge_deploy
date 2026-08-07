from __future__ import annotations

from pathlib import Path
import math
import select
import signal
import sys
import tkinter as tk
import time
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageTk

from .recording import (
    RecordingSchema,
    combine_distribution_image,
    draw_runtime_plot_canvas,
    load_distribution_image,
    resize_to_height,
    select_preview_frame,
    to_bgr_uint8,
)

WINDOW_CAPTURE_TIMEOUT_S = 2.0


class TkImageWindow:
    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        self.root = None
        self.label = None
        self.photo = None

    def show(self, frame: np.ndarray) -> None:
        if self.root is None:
            self.root = tk.Tk()
            self.root.title(self.window_name)
            self.label = tk.Label(self.root)
            self.label.pack()
        self.photo = ImageTk.PhotoImage(Image.fromarray(to_bgr_uint8(frame)[..., ::-1]))
        self.label.configure(image=self.photo)
        self.root.update()

    def close(self) -> None:
        if self.root is not None:
            self.root.destroy()
            self.root = None


class RuntimeExecutionWindow:
    def __init__(self, *, schema: RecordingSchema, display_index: int = 1, window_name: str = "execution_window") -> None:
        self.schema = schema
        self.display_index = max(1, int(display_index))
        self.window_name = window_name
        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.camera_height: int | None = None
        self.camera_width: int | None = None
        self.window_created = False
        self.window_disabled = False
        self.image_window = TkImageWindow(window_name)

    def reset(self) -> None:
        self.actions.clear()
        self.states.clear()
        self.camera_height = None
        self.camera_width = None

    def close(self) -> None:
        self.image_window.close()
        self.window_created = False

    def keep_window_on_top(self) -> None:
        return

    def show_frame(self, frame: np.ndarray) -> None:
        if self.window_disabled:
            return
        try:
            self.image_window.show(frame)
            self.window_created = True
        except Exception as exc:
            self.window_disabled = True
            print(f"Runtime window disabled because Tk window is unavailable: {exc}", flush=True)

    def compose_camera_row(self, images: Mapping[str, np.ndarray]) -> np.ndarray:
        panels = []
        for camera_name in self.schema.camera_names:
            if camera_name not in images:
                raise KeyError(f"Runtime window is missing camera image {camera_name!r}")
            image = to_bgr_uint8(images[camera_name])
            if self.camera_height is None:
                self.camera_height = int(image.shape[0])
            panels.append(resize_to_height(image, self.camera_height))
        row = np.concatenate(panels, axis=1)
        if self.camera_width is None:
            self.camera_width = int(row.shape[1])
        elif row.shape[1] != self.camera_width:
            row = cv2.resize(row, (self.camera_width, self.camera_height), interpolation=cv2.INTER_AREA)
        return row

    def show_images(self, images: Mapping[str, np.ndarray]) -> None:
        self.show_frame(self.compose_camera_row(images))

    def record(self, *, images: Mapping[str, np.ndarray], action: np.ndarray, state: np.ndarray, timestamp_s: float) -> None:
        del timestamp_s
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.states.append(np.asarray(state, dtype=np.float64).copy())
        camera_row = self.compose_camera_row(images)
        actions = np.stack(self.actions, axis=0)
        states = np.stack(self.states, axis=0)
        horizon = max(200, int(math.ceil(len(self.actions) / 200.0) * 200))
        cols = min(4, max(1, len(self.schema.plot_names)))
        rows = max(1, math.ceil(len(self.schema.plot_names) / cols))
        plot_row = draw_runtime_plot_canvas(
            width=camera_row.shape[1],
            height=rows * 80,
            cols=cols,
            names=self.schema.plot_names,
            schema=self.schema,
            actions=actions,
            states=states,
            x_horizon=horizon,
        )
        frame = np.concatenate((camera_row, plot_row), axis=0)
        self.show_frame(frame)


class WindowPreviewTimeout(TimeoutError):
    pass


def run_window_step(step_name: str, callback: Any, timeout_s: float = WINDOW_CAPTURE_TIMEOUT_S) -> Any:
    def raise_timeout(signum: int, frame: Any) -> None:
        raise WindowPreviewTimeout(f"window preview timed out after {timeout_s:.1f}s during {step_name}")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    start_time = time.monotonic()
    try:
        result = callback()
    except WindowPreviewTimeout:
        raise
    except Exception as exc:
        raise RuntimeError(f"window preview failed during {step_name}") from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
    elapsed_s = time.monotonic() - start_time
    if elapsed_s > timeout_s:
        raise WindowPreviewTimeout(f"window preview returned after {elapsed_s:.3f}s during {step_name}")
    return result


def capture_preview_images(source: Any) -> Mapping[str, np.ndarray]:
    cameras = getattr(source, "cameras", None)
    robot = getattr(source, "robot", None)
    if cameras is None or robot is None:
        snapshot = run_window_step("source.capture_snapshot", source.capture_snapshot)
        return snapshot.images
    images = run_window_step("cameras.capture", cameras.capture)
    run_window_step("robot.read_state", robot.read_state)
    return images


def preview_until_continue(
    source: Any,
    *,
    distribution_image_path: Path | None = None,
    image_name: str = "cam_high",
    window_name: str = "train_distribution",
) -> None:
    distribution_image = load_distribution_image(distribution_image_path)
    preview_window = TkImageWindow(window_name)
    print("Place the object to match the train distribution, then type c and press Enter to continue.", flush=True)
    try:
        while True:
            images = capture_preview_images(source)
            selected = run_window_step("select_preview_frame", lambda: select_preview_frame(images, (image_name,)))
            if selected is None:
                raise RuntimeError(f"window preview could not find image {image_name!r}; image keys={list(images)}")
            _, frame = selected
            preview_frame = combine_distribution_image(distribution_image, frame)
            run_window_step("tk preview update", lambda: preview_window.show(preview_frame))
            if select.select([sys.stdin], [], [], 0.05)[0] and sys.stdin.readline().strip().lower() == "c":
                return
    finally:
        preview_window.close()
