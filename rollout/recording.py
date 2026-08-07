from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import shutil
import time
from typing import Any, Mapping

import cv2
import imageio
import imageio.v3 as iio
import numpy as np


ACTION_COLOR = (32, 32, 220)
USED_ACTION_COLOR = (0, 210, 255)
STATE_COLOR = (220, 90, 30)
DISTRIBUTION_OVERLAP = False


def safe_filename_part(value: str) -> str:
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


class ExecutionRecordSink:
    def __init__(self, *, recorder: Any | None = None, runtime_window: Any | None = None) -> None:
        self.recorder = recorder
        self.runtime_window = runtime_window

    def record(
        self,
        *,
        images: Mapping[str, np.ndarray],
        action: np.ndarray,
        state: np.ndarray,
        timestamp_s: float,
    ) -> None:
        if self.recorder is not None:
            self.recorder.record(
                images=images,
                action=action,
                state=state,
                timestamp_s=timestamp_s,
            )
        if self.runtime_window is not None:
            self.runtime_window.record(
                images=images,
                action=action,
                state=state,
                timestamp_s=timestamp_s,
            )


class RolloutVideoRecorder:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        schema: RecordingSchema,
        fps: float,
        name_prefix: str = "rollout_record",
        output_path: str | Path | None = None,
        plot_cols: int = 4,
        plot_cell_h: int = 80,
        keep_frames_in_memory: bool = False,
        video_codec: str = "libx264",
        video_output_params: tuple[str, ...] = ("-preset", "veryfast", "-crf", "18"),
        frame_jpeg_quality: int = 100,
        save_separate_videos: bool = False,
        separate_video_stem: str | None = None,
    ) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = safe_filename_part(name_prefix) or "rollout_record"
        self.record_stem = f"{safe_prefix}_{timestamp}"
        if output_path is None:
            self.run_dir = self.output_dir / self.record_stem
            self.output_path = self.run_dir / f"{self.record_stem}_videos.mp4"
        else:
            self.output_path = Path(output_path)
            self.run_dir = self.output_path.parent
            self.record_stem = self.output_path.stem
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_frames_in_memory = keep_frames_in_memory
        self.save_separate_videos = save_separate_videos
        self.separate_video_stem = safe_filename_part(separate_video_stem or self.record_stem) or self.record_stem
        self.frames_dir = self.run_dir / ".frames"
        if not self.keep_frames_in_memory:
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
        self.finalized = False
        self.frame_images: list[np.ndarray] = []
        self.separate_frame_images: dict[str, list[np.ndarray]] = {name: [] for name in self.schema.camera_names}
        self.separate_frame_paths: dict[str, list[Path]] = {name: [] for name in self.schema.camera_names}
        self.separate_video_paths: list[Path] = []
        self.video_codec = video_codec
        self.video_output_params = tuple(video_output_params)
        self.frame_jpeg_quality = int(frame_jpeg_quality)
        if self.save_separate_videos and not self.keep_frames_in_memory:
            for camera_name in self.schema.camera_names:
                (self.frames_dir / safe_filename_part(camera_name)).mkdir(parents=True, exist_ok=True)

    def extra_image_path(self, suffix: str, extension: str = ".png") -> Path:
        clean_suffix = safe_filename_part(suffix)
        return self.run_dir / f"{self.record_stem}_{clean_suffix}{extension}"

    def save_extra_image(self, image: np.ndarray, *, suffix: str, extension: str = ".png") -> Path:
        path = self.extra_image_path(suffix, extension=extension)
        image = to_bgr_uint8(image)
        iio.imwrite(path, image[..., ::-1])
        return path

    def record(
        self,
        *,
        images: Mapping[str, np.ndarray],
        action: np.ndarray,
        state: np.ndarray,
        timestamp_s: float,
    ) -> None:
        action_array = np.asarray(action, dtype=np.float64).copy()
        state_array = np.asarray(state, dtype=np.float64).copy()
        expected_action_dim = len(self.schema.action_names)
        expected_state_dim = len(self.schema.state_names)
        if action_array.ndim != 1 or action_array.shape[0] != expected_action_dim:
            raise ValueError(
                f"Recording action dim mismatch: got {action_array.shape}, expected ({expected_action_dim},)"
            )
        if state_array.ndim != 1 or state_array.shape[0] != expected_state_dim:
            raise ValueError(
                f"Recording state dim mismatch: got {state_array.shape}, expected ({expected_state_dim},)"
            )

        camera_row = self.compose_camera_row(images)
        if self.save_separate_videos:
            self.record_separate_frames(images)
        if self.keep_frames_in_memory:
            self.frame_images.append(camera_row.copy())
        else:
            frame_path = self.frames_dir / f"frame_{len(self.frame_paths):06d}.jpg"
            iio.imwrite(frame_path, camera_row[..., ::-1], quality=self.frame_jpeg_quality)
            self.frame_paths.append(frame_path)

        self.actions.append(action_array)
        self.states.append(state_array)
        self.timestamps.append(float(timestamp_s))

    def finalize(self) -> Path | None:
        if self.finalized:
            return self.output_path if self.output_path.exists() else None
        self.finalized = True
        total = len(self.frame_images) if self.keep_frames_in_memory else len(self.frame_paths)
        if total == 0:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
            return None

        actions = np.stack(self.actions, axis=0)
        states = np.stack(self.states, axis=0)
        first_frame = self.read_recorded_frame(0)

        camera_h, camera_w = first_frame.shape[:2]
        base_plot, final_plot, plot_rects = self.make_plot_canvases(
            width=camera_w,
            actions=actions,
            states=states,
        )
        tmp_output = self.output_path.with_suffix(".tmp.mp4")

        try:
            self.write_video_file(tmp_output, total, camera_h, camera_w, base_plot, final_plot, plot_rects)
        except Exception:
            if self.video_codec == "mpeg4":
                raise
            tmp_output.unlink(missing_ok=True)
            self.write_video_file(tmp_output, total, camera_h, camera_w, base_plot, final_plot, plot_rects, codec="mpeg4")

        tmp_output.replace(self.output_path)
        if self.save_separate_videos:
            self.separate_video_paths = self.write_separate_videos(total)
        shutil.rmtree(self.frames_dir, ignore_errors=True)
        return self.output_path

    def read_recorded_frame(self, index: int) -> np.ndarray:
        if self.keep_frames_in_memory:
            return self.frame_images[index].copy()
        return iio.imread(self.frame_paths[index])[..., ::-1]

    def record_separate_frames(self, images: Mapping[str, np.ndarray]) -> None:
        frame_index = len(self.actions)
        for camera_name in self.schema.camera_names:
            image = to_bgr_uint8(images[camera_name])
            if self.keep_frames_in_memory:
                self.separate_frame_images[camera_name].append(image.copy())
            else:
                frame_path = self.frames_dir / safe_filename_part(camera_name) / f"frame_{frame_index:06d}.jpg"
                iio.imwrite(frame_path, image[..., ::-1], quality=self.frame_jpeg_quality)
                self.separate_frame_paths[camera_name].append(frame_path)

    def read_separate_frame(self, camera_name: str, index: int) -> np.ndarray:
        if self.keep_frames_in_memory:
            return self.separate_frame_images[camera_name][index].copy()
        return iio.imread(self.separate_frame_paths[camera_name][index])[..., ::-1]

    def video_writer_kwargs(self, codec: str | None = None) -> dict[str, Any]:
        writer_kwargs: dict[str, Any] = {
            "fps": self.fps,
            "codec": codec or self.video_codec,
            "macro_block_size": 1,
            "ffmpeg_log_level": "error",
        }
        if codec is None and self.video_output_params:
            writer_kwargs["output_params"] = list(self.video_output_params)
        return writer_kwargs

    def write_video_file(
        self,
        output_path: Path,
        total: int,
        camera_h: int,
        camera_w: int,
        base_plot: np.ndarray,
        final_plot: np.ndarray,
        plot_rects: list[tuple[int, int, int, int]],
        *,
        codec: str | None = None,
    ) -> None:
        writer = imageio.get_writer(output_path, **self.video_writer_kwargs(codec))
        try:
            for index in range(total):
                camera_row = self.read_recorded_frame(index)
                if camera_row.shape[:2] != (camera_h, camera_w):
                    camera_row = cv2.resize(camera_row, (camera_w, camera_h), interpolation=cv2.INTER_AREA)
                ratio = (index + 1) / total
                plot_row = reveal_plot_frame(base_plot, final_plot, plot_rects, ratio)
                writer.append_data(np.concatenate((camera_row, plot_row), axis=0)[..., ::-1])
        finally:
            writer.close()

    def write_separate_videos(self, total: int) -> list[Path]:
        output_paths = []
        for camera_name in self.schema.camera_names:
            clean_name = safe_filename_part(camera_name)
            output_path = self.run_dir / f"{self.separate_video_stem}_{clean_name}.mp4"
            tmp_output = output_path.with_suffix(".tmp.mp4")
            try:
                self.write_separate_video_file(camera_name, tmp_output, total)
            except Exception:
                if self.video_codec == "mpeg4":
                    raise
                tmp_output.unlink(missing_ok=True)
                self.write_separate_video_file(camera_name, tmp_output, total, codec="mpeg4")
            tmp_output.replace(output_path)
            output_paths.append(output_path)
        return output_paths

    def write_separate_video_file(
        self,
        camera_name: str,
        output_path: Path,
        total: int,
        *,
        codec: str | None = None,
    ) -> None:
        first_frame = self.read_separate_frame(camera_name, 0)
        video_h, video_w = first_frame.shape[:2]
        writer = imageio.get_writer(output_path, **self.video_writer_kwargs(codec))
        try:
            for index in range(total):
                frame = self.read_separate_frame(camera_name, index)
                if frame.shape[:2] != (video_h, video_w):
                    frame = cv2.resize(frame, (video_w, video_h), interpolation=cv2.INTER_AREA)
                writer.append_data(frame[..., ::-1])
        finally:
            writer.close()

    def compose_camera_row(self, images: Mapping[str, np.ndarray]) -> np.ndarray:
        panels = []
        for camera_name in self.schema.camera_names:
            if camera_name not in images:
                raise KeyError(f"Recording is missing camera image {camera_name!r}")
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

    def make_plot_canvases(
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
        base = draw_record_plot_canvas(
            width=width,
            height=height,
            cols=cols,
            names=names,
            schema=self.schema,
            actions=actions,
            states=states,
            draw_curves=False,
        )
        final = draw_record_plot_canvas(
            width=width,
            height=height,
            cols=cols,
            names=names,
            schema=self.schema,
            actions=actions,
            states=states,
            draw_curves=True,
        )
        rects = plot_rects(width=width, height=height, count=len(names), cols=cols)
        return base, final, rects

def save_recorded_actions(
    recorder: RolloutVideoRecorder,
) -> Path:
    action_path = recorder.run_dir / f"{recorder.record_stem}_actions.npz"
    if recorder.actions:
        actions = np.stack(recorder.actions, axis=0)
    else:
        actions = np.empty((0, len(recorder.schema.action_names)), dtype=np.float64)
    if recorder.states:
        states = np.stack(recorder.states, axis=0)
    else:
        states = np.empty((0, len(recorder.schema.state_names)), dtype=np.float64)
    timestamps_s = np.asarray(recorder.timestamps, dtype=np.float64)
    np.savez_compressed(action_path, actions=actions, states=states, timestamps_s=timestamps_s)
    return action_path


def to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    image = to_bgr_uint8(image)
    if image.shape[1] == width:
        return image
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def stack_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    top_bgr = to_bgr_uint8(top)
    bottom_bgr = to_bgr_uint8(bottom)
    target_width = max(top_bgr.shape[1], bottom_bgr.shape[1])
    top_resized = resize_to_width(top_bgr, target_width)
    bottom_resized = resize_to_width(bottom_bgr, target_width)
    return np.concatenate((top_resized, bottom_resized), axis=0)


def set_distribution_overlap(enabled: bool) -> None:
    global DISTRIBUTION_OVERLAP
    DISTRIBUTION_OVERLAP = bool(enabled)


def load_distribution_image(distribution_image_path: Path | None) -> np.ndarray | None:
    if distribution_image_path is None or not distribution_image_path.exists():
        return None
    distribution_image = np.asarray(iio.imread(distribution_image_path))
    if distribution_image.ndim == 2:
        distribution_image = np.repeat(distribution_image[..., None], 3, axis=2)
    if distribution_image.ndim != 3 or distribution_image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HWC 3/4-channel image, got shape {distribution_image.shape}")
    if distribution_image.dtype != np.uint8:
        distribution_image = np.clip(distribution_image, 0, 255).astype(np.uint8)
    if distribution_image.shape[-1] == 4:
        return distribution_image[..., [2, 1, 0, 3]].copy()
    return distribution_image[..., ::-1].copy()


def overlay_distribution_image(
    distribution_image: np.ndarray,
    frame: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    frame_bgr = to_bgr_uint8(frame).astype(np.float32)
    distribution = np.asarray(distribution_image)
    if distribution.shape[:2] != frame_bgr.shape[:2]:
        distribution = cv2.resize(
            distribution,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    if distribution.ndim != 3 or distribution.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HWC 3/4-channel image, got shape {distribution.shape}")
    distribution_bgr = distribution[..., :3].astype(np.float32)
    if distribution.shape[-1] == 4:
        alpha_mask = distribution[..., 3:4].astype(np.float32) / 255.0
    else:
        alpha_mask = np.full((*frame_bgr.shape[:2], 1), float(alpha), dtype=np.float32)
    return np.clip(frame_bgr * (1.0 - alpha_mask) + distribution_bgr * alpha_mask, 0, 255).astype(np.uint8)


def combine_distribution_image(distribution_image: np.ndarray | None, frame: np.ndarray) -> np.ndarray:
    if distribution_image is None:
        return frame
    if DISTRIBUTION_OVERLAP:
        return overlay_distribution_image(distribution_image, frame)
    return stack_vertical(distribution_image[..., :3], frame)


def select_preview_frame(
    images: Mapping[str, np.ndarray],
    preferred_names: tuple[str, ...],
) -> tuple[str, np.ndarray] | None:
    for image_name in preferred_names + tuple(images):
        frame = images.get(image_name)
        if frame is not None:
            return image_name, to_bgr_uint8(frame)
    return None


def save_frame1_image(
    recorder: RolloutVideoRecorder,
    snapshot: Any | None,
    *,
    distribution_image_path: Path | None = None,
    preferred_names: tuple[str, ...] = ("cam_high",),
) -> Path | None:
    if snapshot is None:
        return None
    images = getattr(snapshot, "images", {}) or {}
    selected = select_preview_frame(images, preferred_names)
    if selected is None:
        return None
    image_name, frame = selected
    distribution_image = load_distribution_image(distribution_image_path)
    if distribution_image is not None and image_name == "cam_high":
        frame = combine_distribution_image(distribution_image, frame)
    return recorder.save_extra_image(frame, suffix="frame1")


def plot_rects(*, width: int, height: int, count: int, cols: int) -> list[tuple[int, int, int, int]]:
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


def series_for_name(series: np.ndarray, names: tuple[str, ...], name: str) -> np.ndarray | None:
    try:
        index = names.index(name)
    except ValueError:
        return None
    if index >= series.shape[1]:
        return None
    return series[:, index]


def short_label(name: str) -> str:
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


def put_small_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.27, (40, 40, 40), 1, cv2.LINE_AA)


def draw_record_plot_canvas(
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
    rects = plot_rects(width=width, height=height, count=len(names), cols=cols)

    for index, name in enumerate(names):
        row = index // cols
        col = index % cols
        cell_x = col * cell_w
        cell_y = row * cell_h
        rect = rects[index]
        x0, y0, x1, y1 = rect

        action_values = series_for_name(actions, schema.action_names, name)
        state_values = series_for_name(states, schema.state_names, name)
        value_blocks = []
        for values in (action_values, state_values):
            if values is None or not values.size:
                continue
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                value_blocks.append(finite_values)
        if value_blocks:
            values = np.concatenate(value_blocks)
            y_min = float(np.min(values))
            y_max = float(np.max(values))
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
        put_small_label(canvas, short_label(name), (cell_x + 3, cell_y + 10))

        if not draw_curves:
            continue

        if state_values is not None:
            draw_plot_segments(canvas, to_record_plot_segments(state_values, rect, y_min, y_max), STATE_COLOR)
        if action_values is not None:
            action_color = USED_ACTION_COLOR if name in schema.used_action_names else ACTION_COLOR
            draw_plot_segments(canvas, to_record_plot_segments(action_values, rect, y_min, y_max), action_color)

    return canvas


def draw_runtime_plot_canvas(
    *,
    width: int,
    height: int,
    cols: int,
    names: tuple[str, ...],
    schema: RecordingSchema,
    actions: np.ndarray,
    states: np.ndarray,
    x_horizon: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    rows = max(1, math.ceil(max(1, len(names)) / cols))
    cell_w = width // cols
    cell_h = height // rows
    rects = plot_rects(width=width, height=height, count=len(names), cols=cols)

    for index, name in enumerate(names):
        row = index // cols
        col = index % cols
        cell_x = col * cell_w
        cell_y = row * cell_h
        rect = rects[index]
        x0, y0, x1, y1 = rect

        action_values = series_for_name(actions, schema.action_names, name)
        state_values = series_for_name(states, schema.state_names, name)
        value_blocks = []
        for values in (action_values, state_values):
            if values is None or not values.size:
                continue
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                value_blocks.append(finite_values)
        if value_blocks:
            values = np.concatenate(value_blocks)
            y_min = float(np.min(values))
            y_max = float(np.max(values))
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
        put_small_label(canvas, short_label(name), (cell_x + 3, cell_y + 10))

        if state_values is not None:
            segments = to_runtime_plot_segments(state_values, rect, y_min, y_max, x_horizon)
            draw_plot_segments(canvas, segments, STATE_COLOR)
        if action_values is not None:
            action_color = USED_ACTION_COLOR if name in schema.used_action_names else ACTION_COLOR
            segments = to_runtime_plot_segments(action_values, rect, y_min, y_max, x_horizon)
            draw_plot_segments(canvas, segments, action_color)

    return canvas


def draw_plot_segments(canvas: np.ndarray, segments: list[np.ndarray], color: tuple[int, int, int]) -> None:
    for points in segments:
        if len(points) == 1:
            cv2.circle(canvas, tuple(points[0]), 1, color, -1, cv2.LINE_AA)
        elif len(points) > 1:
            cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)


def plot_segments_from_xy(
    xs: np.ndarray,
    values: np.ndarray,
    rect: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
) -> list[np.ndarray]:
    _x0, y0, _x1, y1 = rect
    values = np.asarray(values, dtype=np.float64)
    ys = y1 - (values - y_min) / (y_max - y_min) * (y1 - y0)
    valid = np.isfinite(values) & np.isfinite(ys)
    segments: list[np.ndarray] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = index
        if start is None:
            continue
        at_end = index == len(valid) - 1
        if not is_valid or at_end:
            end = index + 1 if is_valid and at_end else index
            if end > start:
                segment_y = np.clip(ys[start:end], y0, y1 - 1)
                segment = np.stack((xs[start:end], segment_y), axis=1).round().astype(np.int32)
                segments.append(segment)
            start = None
    return segments


def to_record_plot_segments(
    values: np.ndarray,
    rect: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
) -> list[np.ndarray]:
    x0, _y0, x1, _y1 = rect
    if len(values) == 1:
        xs = np.array([x0], dtype=np.float64)
    else:
        xs = np.linspace(x0, x1 - 1, len(values), dtype=np.float64)
    return plot_segments_from_xy(xs, values, rect, y_min, y_max)


def to_runtime_plot_segments(
    values: np.ndarray,
    rect: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
    x_horizon: int,
) -> list[np.ndarray]:
    x0, _y0, x1, _y1 = rect
    horizon = max(len(values), int(x_horizon))
    if horizon <= 1:
        xs = np.array([x0], dtype=np.float64)
    else:
        xs = np.linspace(x0, x1 - 1, horizon, dtype=np.float64)[:len(values)]
    return plot_segments_from_xy(xs, values, rect, y_min, y_max)


def reveal_plot_frame(
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
