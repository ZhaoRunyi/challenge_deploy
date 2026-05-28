from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from teleop.hdf5_teleop import (
    save_alignment_diagnostics,
    save_hdf5_teleop_episode,
    save_hdf5_teleop_record_video,
)


@dataclass(frozen=True)
class HDF5TeleopSaveConfig:
    camera_names: tuple[str, ...]
    language_instruction: str
    include_depth_images: bool = False
    jpeg_quality: int = 95
    action_from_state: bool = False
    alignment_plot_frames: int = 16
    record_video: bool = False
    fps: float = 30.0


class BaseDataWorker:
    def __init__(self, *, thread_name_prefix: str = "data_worker") -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self.futures: list[Any] = []

    def submit(self, item: Any) -> dict[str, Any]:
        future = self.executor.submit(self.save_item, item)
        self.futures.append(future)
        return self.planned_result(item)

    def save_item(self, item: Any) -> dict[str, Any]:
        raise NotImplementedError

    def planned_result(self, item: Any) -> dict[str, Any]:
        raise NotImplementedError

    def drain(
        self,
        *,
        block: bool = False,
        on_result: Callable[[dict[str, Any]], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> int:
        completed_count = 0
        pending = []
        for future in self.futures:
            if not block and not future.done():
                pending.append(future)
                continue
            completed_count += 1
            try:
                result = future.result()
            except Exception as exc:
                if on_error is not None:
                    on_error(exc)
                else:
                    raise
                continue
            if on_result is not None:
                on_result(result)
        self.futures = pending
        return completed_count

    def stop(
        self,
        *,
        on_result: Callable[[dict[str, Any]], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        try:
            self.drain(block=True, on_result=on_result, on_error=on_error)
        finally:
            self.executor.shutdown(wait=True)


class HDF5TeleopDataWorker(BaseDataWorker):
    def __init__(self, *, config: HDF5TeleopSaveConfig) -> None:
        super().__init__(thread_name_prefix="hdf5_teleop_save")
        self.config = config

    def save_item(self, item: Any) -> dict[str, Any]:
        output_path = save_hdf5_teleop_episode(
            output_path=item.episode_path,
            frames=item.frames,
            camera_names=self.config.camera_names,
            language_instruction=self.config.language_instruction,
            include_depth_images=self.config.include_depth_images,
            jpeg_quality=self.config.jpeg_quality,
            action_from_state=self.config.action_from_state,
        )
        alignment_json_path, alignment_image_path = save_alignment_diagnostics(
            output_path,
            item.frames,
            item.trace,
            plot_frames=self.config.alignment_plot_frames,
        )
        record_video_path = None
        if self.config.record_video:
            record_video_path = save_hdf5_teleop_record_video(
                frames=item.frames,
                output_dir=output_path.parent,
                fps=self.config.fps,
                name_prefix=f"episode_{item.episode_index}",
                action_from_state=self.config.action_from_state,
                output_path=output_path.with_name(f"episode_{item.episode_index}_video.mp4"),
            )
        return self.saved_result(
            output_path=output_path,
            alignment_json_path=alignment_json_path,
            alignment_image_path=alignment_image_path,
            record_video_path=record_video_path,
            frame_count=len(item.frames),
        )

    def planned_result(self, item: Any) -> dict[str, Any]:
        data_frames = max(0, len(item.frames) - 1)
        alignment_stem = f"{item.episode_path.name}_alignment_plot{self.config.alignment_plot_frames}_frames{data_frames}"
        record_video_path = None
        if self.config.record_video:
            record_video_path = item.episode_path.with_name(f"episode_{item.episode_index}_video.mp4")
        return {
            "saved_path": str(item.episode_path.with_suffix(".hdf5")),
            "alignment_json_path": str(item.episode_path.with_name(alignment_stem + ".json")),
            "alignment_image_path": str(item.episode_path.with_name(alignment_stem + ".png")),
            "record_video_path": None if record_video_path is None else str(record_video_path),
            "record_video_status": None if record_video_path is None else "queued",
            "captured_frames": len(item.frames),
            "saved_steps": data_frames,
        }

    def saved_result(
        self,
        *,
        output_path: Path,
        alignment_json_path: Path,
        alignment_image_path: Path,
        record_video_path: Path | None,
        frame_count: int,
    ) -> dict[str, Any]:
        return {
            "saved_path": str(output_path),
            "alignment_json_path": str(alignment_json_path),
            "alignment_image_path": str(alignment_image_path),
            "record_video_path": None if record_video_path is None else str(record_video_path),
            "record_video_status": None if record_video_path is None else "saved",
            "captured_frames": frame_count,
            "saved_steps": max(0, frame_count - 1),
        }
