from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import shutil
import time
from typing import Mapping

import cv2
import numpy as np


ACTION_COLOR = (32, 32, 220)
USED_ACTION_COLOR = (0, 210, 255)
STATE_COLOR = (220, 90, 30)


def _safe_filename_part(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._-")


@dataclass(frozen=True)
class RecordingSchema:
    camera_names: tuple[str, ...]
    action_names: tuple[str, ...]
    state_names: tuple[str, ...]
    used_action_names: frozenset[str]

    @property
    def plot_names(self) -> tuple[str, ...]:
        names = list(self.action_names)
        action_name_set = set(self.action_names)
        names.extend(name for name in self.state_names if name not in action_name_set)
        return tuple(names)


class OpenPiRolloutRecorder:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        schema: RecordingSchema,
        fps: float,
        name_prefix: str = "openpi_record",
        plot_cols: int = 4,
        plot_cell_h: int = 80,
    ) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = _safe_filename_part(name_prefix) or "openpi_record"
        self.record_stem = f"{safe_prefix}_{timestamp}"
        self.run_dir = self.output_dir / self.record_stem
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.run_dir / f"{self.record_stem}_videos.mp4"
        self.frames_dir = self.run_dir / ".frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.schema = schema
        self.fps = fps if fps > 0.0 else 10.0
        self.plot_cols = max(1, plot_cols)
        self.plot_cell_h = max(48, plot_cell_h)
        self.frame_paths: list[Path] = []
        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.camera_height: int | None = None
        self.camera_width: int | None = None
        self._finalized = False

    def extra_image_path(self, suffix: str, extension: str = ".png") -> Path:
        clean_suffix = _safe_filename_part(suffix)
        return self.run_dir / f"{self.record_stem}_{clean_suffix}{extension}"

    def save_extra_image(self, image: np.ndarray, *, suffix: str, extension: str = ".png") -> Path:
        path = self.extra_image_path(suffix, extension=extension)
        image = _to_bgr_uint8(image)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write extra recording image: {path}")
        return path

    def record(
        self,
        *,
        images: Mapping[str, np.ndarray],
        action: np.ndarray,
        state: np.ndarray,
        timestamp_s: float,
    ) -> None:
        camera_row = self._compose_camera_row(images)
        frame_path = self.frames_dir / f"frame_{len(self.frame_paths):06d}.jpg"
        if not cv2.imwrite(str(frame_path), camera_row, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise RuntimeError(f"Failed to write recording frame: {frame_path}")

        self.frame_paths.append(frame_path)
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.states.append(np.asarray(state, dtype=np.float64).copy())
        self.timestamps.append(float(timestamp_s))

    def finalize(self) -> Path | None:
        if self._finalized:
            return self.output_path if self.output_path.exists() else None
        self._finalized = True
        if not self.frame_paths:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
            return None

        actions = np.stack(self.actions, axis=0)
        states = np.stack(self.states, axis=0)
        first_frame = cv2.imread(str(self.frame_paths[0]), cv2.IMREAD_COLOR)
        if first_frame is None:
            raise RuntimeError(f"Failed to read first recording frame: {self.frame_paths[0]}")

        camera_h, camera_w = first_frame.shape[:2]
        base_plot, final_plot, plot_rects = self._make_plot_canvases(
            width=camera_w,
            actions=actions,
            states=states,
        )
        plot_h = base_plot.shape[0]
        video_w, video_h = camera_w, camera_h + plot_h
        tmp_output = self.output_path.with_suffix(".tmp.mp4")

        writer = cv2.VideoWriter(
            str(tmp_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (video_w, video_h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open recording video writer: {tmp_output}")

        try:
            total = len(self.frame_paths)
            for index, frame_path in enumerate(self.frame_paths):
                camera_row = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if camera_row is None:
                    raise RuntimeError(f"Failed to read recording frame: {frame_path}")
                if camera_row.shape[:2] != (camera_h, camera_w):
                    camera_row = cv2.resize(camera_row, (camera_w, camera_h), interpolation=cv2.INTER_AREA)
                ratio = (index + 1) / total
                plot_row = _reveal_plot_frame(base_plot, final_plot, plot_rects, ratio)
                writer.write(np.concatenate((camera_row, plot_row), axis=0))
        finally:
            writer.release()

        tmp_output.replace(self.output_path)
        shutil.rmtree(self.frames_dir, ignore_errors=True)
        return self.output_path

    def _compose_camera_row(self, images: Mapping[str, np.ndarray]) -> np.ndarray:
        panels = []
        for camera_name in self.schema.camera_names:
            if camera_name not in images:
                raise KeyError(f"Recording is missing camera image {camera_name!r}")
            image = _to_bgr_uint8(images[camera_name])
            if self.camera_height is None:
                self.camera_height = int(image.shape[0])
            panels.append(_resize_to_height(image, self.camera_height))

        row = np.concatenate(panels, axis=1)
        if self.camera_width is None:
            self.camera_width = int(row.shape[1])
        elif row.shape[1] != self.camera_width:
            row = cv2.resize(row, (self.camera_width, self.camera_height), interpolation=cv2.INTER_AREA)
        return row

    def _make_plot_canvases(
        self,
        *,
        width: int,
        actions: np.ndarray,
        states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
        names = self.schema.plot_names
        cols = min(self.plot_cols, max(1, len(names)))
        rows = max(1, math.ceil(len(names) / cols))
        height = rows * self.plot_cell_h
        base = _draw_plot_canvas(
            width=width,
            height=height,
            cols=cols,
            names=names,
            schema=self.schema,
            actions=actions,
            states=states,
            draw_curves=False,
        )
        final = _draw_plot_canvas(
            width=width,
            height=height,
            cols=cols,
            names=names,
            schema=self.schema,
            actions=actions,
            states=states,
            draw_curves=True,
        )
        rects = _plot_rects(width=width, height=height, count=len(names), cols=cols)
        return base, final, rects


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    image = _to_bgr_uint8(image)
    if image.shape[1] == width:
        return image
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def stack_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    top_bgr = _to_bgr_uint8(top)
    bottom_bgr = _to_bgr_uint8(bottom)
    target_width = max(top_bgr.shape[1], bottom_bgr.shape[1])
    top_resized = resize_to_width(top_bgr, target_width)
    bottom_resized = resize_to_width(bottom_bgr, target_width)
    return np.concatenate((top_resized, bottom_resized), axis=0)


def _plot_rects(*, width: int, height: int, count: int, cols: int) -> list[tuple[int, int, int, int]]:
    rows = max(1, math.ceil(max(1, count) / cols))
    cell_w = width // cols
    cell_h = height // rows
    rects: list[tuple[int, int, int, int]] = []
    for index in range(count):
        row = index // cols
        col = index % cols
        cell_x = col * cell_w
        cell_y = row * cell_h
        rects.append((cell_x + 34, cell_y + 12, cell_x + cell_w - 8, cell_y + cell_h - 10))
    return rects


def _series_for_name(series: np.ndarray, names: tuple[str, ...], name: str) -> np.ndarray | None:
    try:
        index = names.index(name)
    except ValueError:
        return None
    if index >= series.shape[1]:
        return None
    return series[:, index]


def _short_label(name: str) -> str:
    label = name
    label = label.replace("left_", "L ")
    label = label.replace("right_", "R ")
    label = label.replace("joint_", "j ")
    label = label.replace("gripper", "grip")
    label = label.replace("ee_pos_", "ee ")
    label = label.replace("ee_rot6d_", "r6 ")
    label = label.replace("ee_rot_", "rot ")
    label = label.replace("ee_", "")
    label = label.replace("forearm_roll", "fore")
    label = label.replace("wrist_angle", "w_ang")
    label = label.replace("wrist_rotate", "w_rot")
    return label[:22]


def _put_small_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.27, (40, 40, 40), 1, cv2.LINE_AA)


def _draw_plot_canvas(
    *,
    width: int,
    height: int,
    cols: int,
    names: tuple[str, ...],
    schema: RecordingSchema,
    actions: np.ndarray,
    states: np.ndarray,
    draw_curves: bool,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    rows = max(1, math.ceil(max(1, len(names)) / cols))
    cell_w = width // cols
    cell_h = height // rows
    rects = _plot_rects(width=width, height=height, count=len(names), cols=cols)

    for index, name in enumerate(names):
        row = index // cols
        col = index % cols
        cell_x = col * cell_w
        cell_y = row * cell_h
        rect = rects[index]
        x0, y0, x1, y1 = rect

        action_values = _series_for_name(actions, schema.action_names, name)
        state_values = _series_for_name(states, schema.state_names, name)
        value_blocks = [values for values in (action_values, state_values) if values is not None and values.size]
        if value_blocks:
            values = np.concatenate(value_blocks)
            y_min = float(np.nanmin(values))
            y_max = float(np.nanmax(values))
        else:
            y_min, y_max = -1.0, 1.0
        if not np.isfinite(y_min) or not np.isfinite(y_max) or abs(y_max - y_min) < 1e-9:
            center = 0.0 if not np.isfinite(y_min) else y_min
            y_min, y_max = center - 1.0, center + 1.0
        margin = max((y_max - y_min) * 0.08, 1e-6)
        y_min -= margin
        y_max += margin

        cv2.rectangle(canvas, (cell_x, cell_y), (cell_x + cell_w - 1, cell_y + cell_h - 1), (220, 220, 220), 1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (138, 138, 138), 1)
        if y_min <= 0.0 <= y_max:
            zero_y = int(round(y1 - (0.0 - y_min) / (y_max - y_min) * (y1 - y0)))
            cv2.line(canvas, (x0, zero_y), (x1, zero_y), (218, 218, 218), 1)
        _put_small_label(canvas, _short_label(name), (cell_x + 3, cell_y + 10))

        if not draw_curves:
            continue

        if state_values is not None:
            state_points = _to_plot_points(state_values, rect, y_min, y_max)
            cv2.polylines(canvas, [state_points], False, STATE_COLOR, 1, cv2.LINE_AA)
        if action_values is not None:
            action_color = USED_ACTION_COLOR if name in schema.used_action_names else ACTION_COLOR
            action_points = _to_plot_points(action_values, rect, y_min, y_max)
            cv2.polylines(canvas, [action_points], False, action_color, 1, cv2.LINE_AA)

    return canvas


def _to_plot_points(values: np.ndarray, rect: tuple[int, int, int, int], y_min: float, y_max: float) -> np.ndarray:
    x0, y0, x1, y1 = rect
    if len(values) == 1:
        xs = np.array([x0], dtype=np.float64)
    else:
        xs = np.linspace(x0, x1 - 1, len(values), dtype=np.float64)
    ys = y1 - (values - y_min) / (y_max - y_min) * (y1 - y0)
    ys = np.clip(ys, y0, y1 - 1)
    return np.stack((xs, ys), axis=1).round().astype(np.int32)


def _reveal_plot_frame(
    base_plot: np.ndarray,
    final_plot: np.ndarray,
    plot_rects: list[tuple[int, int, int, int]],
    ratio: float,
) -> np.ndarray:
    frame = base_plot.copy()
    ratio = float(np.clip(ratio, 0.0, 1.0))
    for x0, y0, x1, y1 in plot_rects:
        reveal_x = x0 + int(round((x1 - x0) * ratio))
        if reveal_x > x0:
            frame[y0:y1, x0:reveal_x] = final_plot[y0:y1, x0:reveal_x]
    return frame
