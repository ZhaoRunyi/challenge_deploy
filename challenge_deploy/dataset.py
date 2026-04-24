from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


@dataclass(slots=True)
class EpisodeData:
    observations: list[dict[str, Any]]
    actions: list[np.ndarray]


def create_video_from_images(images: list[np.ndarray], output_path: str, fps: int = 30) -> None:
    if not images:
        raise ValueError("No image data available for video export")
    first = images[0]
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed to open {output_path}")
    try:
        for image in images:
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            writer.write(image)
    finally:
        writer.release()


def save_episode(
    *,
    dataset_path: str | Path,
    episode: EpisodeData,
    camera_names: list[str],
    export_video: bool = True,
    video_fps: int = 30,
) -> Path:
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    observations = episode.observations
    actions = episode.actions
    data_size = len(actions)
    if data_size == 0:
        raise ValueError("No action data to save")
    if len(observations) < data_size:
        raise ValueError("Expected at least as many observations as actions")

    qpos = []
    qvel = []
    effort = []
    base_action = []
    action_list = []
    video_images = {camera_name: [] for camera_name in camera_names}

    for index, action in enumerate(actions):
        observation = observations[index]
        qpos.append(observation["qpos"])
        qvel.append(observation["qvel"])
        effort.append(observation["effort"])
        base_action.append(observation.get("base_vel", np.array([0.0, 0.0], dtype=np.float64)))
        action_list.append(action)
        for camera_name in camera_names:
            image = observation["images"].get(camera_name)
            if image is not None:
                video_images[camera_name].append(image)

    with h5py.File(dataset_path.with_suffix(".hdf5"), "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = False
        obs_group = root.create_group("observations")
        obs_group.create_dataset("qpos", data=np.asarray(qpos, dtype=np.float64))
        obs_group.create_dataset("qvel", data=np.asarray(qvel, dtype=np.float64))
        obs_group.create_dataset("effort", data=np.asarray(effort, dtype=np.float64))
        root.create_dataset("action", data=np.asarray(action_list, dtype=np.float64))
        root.create_dataset("base_action", data=np.asarray(base_action, dtype=np.float64))

    if export_video:
        video_root = dataset_path.parent / "video"
        episode_idx = dataset_path.name.split("_")[-1]
        for camera_name, images in video_images.items():
            if not images:
                continue
            camera_dir = video_root / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            video_path = camera_dir / f"episode_{episode_idx}.mp4"
            create_video_from_images(images, str(video_path), fps=video_fps)
    return dataset_path.with_suffix(".hdf5")


def delete_episode_artifacts(dataset_path: str | Path, camera_names: list[str]) -> None:
    dataset_path = Path(dataset_path)
    hdf5_path = dataset_path.with_suffix(".hdf5")
    if hdf5_path.exists():
        hdf5_path.unlink()

    video_root = dataset_path.parent / "video"
    episode_idx = dataset_path.name.split("_")[-1]
    for camera_name in camera_names:
        video_path = video_root / camera_name / f"episode_{episode_idx}.mp4"
        if video_path.exists():
            video_path.unlink()


class EpisodeCollector:
    def __init__(self, *, camera_names: list[str], dataset_dir: str | Path, dataset_name: str) -> None:
        self.camera_names = camera_names
        self.dataset_root = Path(dataset_dir) / dataset_name
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.observations: list[dict[str, Any]] = []
        self.actions: list[np.ndarray] = []
        self.frame_count = 0
        self.episode_idx = self._find_next_episode_idx()
        self.is_collecting = False

    def _find_next_episode_idx(self) -> int:
        existing = sorted(self.dataset_root.glob("episode_*.hdf5"))
        if not existing:
            return 0
        return max(int(path.stem.split("_")[-1]) for path in existing) + 1

    def start(self) -> None:
        self.observations = []
        self.actions = []
        self.frame_count = 0
        self.is_collecting = True

    def stop(self) -> None:
        self.is_collecting = False

    def add_frame(self, observation: dict[str, Any], action: np.ndarray) -> None:
        if not self.is_collecting:
            return
        self.frame_count += 1
        if self.frame_count == 1:
            self.observations.append(observation)
            return
        self.observations.append(observation)
        self.actions.append(np.asarray(action, dtype=np.float64))

    def has_data(self) -> bool:
        return len(self.actions) > 0

    def save_current_episode(self, *, export_video: bool = True, video_fps: int = 30) -> Path:
        if not self.has_data():
            raise ValueError("No episode data available")
        dataset_path = self.dataset_root / f"episode_{self.episode_idx}"
        save_episode(
            dataset_path=dataset_path,
            episode=EpisodeData(observations=self.observations, actions=self.actions),
            camera_names=self.camera_names,
            export_video=export_video,
            video_fps=video_fps,
        )
        self.episode_idx += 1
        self.stop()
        return dataset_path.with_suffix(".hdf5")
