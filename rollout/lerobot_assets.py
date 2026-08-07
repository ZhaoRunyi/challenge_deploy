from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from lerobot.common.datasets.video_utils import decode_video_frames
except ModuleNotFoundError as error:
    decode_video_frames = None
    LEROBOT_IMPORT_ERROR = error
else:
    LEROBOT_IMPORT_ERROR = None

try:
    import pandas as pd
except ModuleNotFoundError as error:
    pd = None
    PANDAS_IMPORT_ERROR = error
else:
    PANDAS_IMPORT_ERROR = None

from .assets import (
    ARTIFACTS_ROOT,
    PROMPT_CACHE_PATH,
    DatasetAssetInfo,
    PreparedTrainAssets,
    cached_prompt_for_train_config,
    find_distribution_image_path,
    read_bgr_image,
    repo_id_distribution_image_path,
    safe_filename_part,
    set_cached_prompt,
    write_bgr_image,
)
from .task_segmentation import select_relevant_task_masks

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEROBOT_HOME = DEPLOY_ROOT.parent / "data"
CAM_HIGH_BACKGROUND_BY_REPO_ID = {
    "ZhaoRunyi/Piper_traffic_light_water_0612": "cam_high_background_344322073529.png",
    "ZhaoRunyi/Piper_traffic_light_water_0616": "cam_high_background_344322073529.png",
    "ZhaoRunyi/Piper_traffic_light_water_0624": "cam_high_background_344322073529.png",
    "ZhaoRunyi/Piper_traffic_light_water_color": "cam_high_background_344322073529.png",
}


def lerobot_home() -> Path:
    raw_value = os.environ.get("HF_LEROBOT_HOME")
    raw = Path(raw_value).expanduser() if raw_value else DEFAULT_LEROBOT_HOME
    return raw.resolve()


def normalized_repo_id(repo_id: Any) -> str | None:
    if not isinstance(repo_id, str):
        return None
    repo_id = repo_id.strip()
    return repo_id or None


def repo_id_from_spec(spec: Any | None) -> str | None:
    train_config = getattr(spec, "train_config", None)
    data_config = getattr(train_config, "data", None)
    return normalized_repo_id(getattr(data_config, "repo_id", None))


def dataset_dir_for_repo_id(repo_id: str | None, *, root: Path | None = None) -> Path | None:
    if not isinstance(repo_id, str) or not repo_id:
        return None
    dataset_dir = (root or lerobot_home()) / repo_id
    return dataset_dir if dataset_dir.exists() else None


def default_cam_high_background_path(*, artifacts_root: Path = ARTIFACTS_ROOT) -> Path:
    return artifacts_root / "train_distributions" / "cam_high_background.png"


def cam_high_background_path_for_repo_id(repo_id: str | None, *, artifacts_root: Path = ARTIFACTS_ROOT) -> Path:
    filename = CAM_HIGH_BACKGROUND_BY_REPO_ID.get(str(repo_id)) if repo_id else None
    if filename:
        return artifacts_root / "train_distributions" / filename
    return default_cam_high_background_path(artifacts_root=artifacts_root)


def dataset_asset_info(
    train_config_name: str,
    *,
    repo_id: str | None = None,
    artifacts_root: Path = ARTIFACTS_ROOT,
) -> DatasetAssetInfo:
    resolved_repo_id = normalized_repo_id(repo_id)
    dataset_dir = dataset_dir_for_repo_id(resolved_repo_id)
    distribution_image_path = (
        repo_id_distribution_image_path(resolved_repo_id, artifacts_root=artifacts_root)
        if resolved_repo_id
        else artifacts_root / "train_distributions" / f"{safe_filename_part(train_config_name)}.png"
    )
    return DatasetAssetInfo(
        train_config_name=train_config_name,
        repo_id=resolved_repo_id,
        dataset_dir=dataset_dir,
        distribution_image_path=distribution_image_path,
    )


def dataset_prompt(dataset_dir: Path) -> str | None:
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        for line in tasks_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            task = payload.get("task")
            if isinstance(task, str) and task.strip():
                return task.strip()

    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            tasks = payload.get("tasks")
            if isinstance(tasks, list):
                for task in tasks:
                    if isinstance(task, str) and task.strip():
                        return task.strip()
    return None


def episode_indices(dataset_dir: Path) -> list[int]:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return []
    indices: list[int] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        episode_index = payload.get("episode_index")
        if isinstance(episode_index, int):
            indices.append(episode_index)
    return indices


def info_json(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    return json.loads(info_path.read_text(encoding="utf-8"))


def parquet_path_for_episode(dataset_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    chunk_index = episode_index // chunks_size
    data_path = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    parquet_path = dataset_dir / data_path.format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
    )
    if parquet_path.exists():
        return parquet_path
    candidates = sorted(dataset_dir.rglob(f"episode_{episode_index:06d}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"episode_{episode_index:06d}.parquet not found under {dataset_dir}")
    return candidates[0]


def video_path_for_episode(dataset_dir: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    video_path = info.get("video_path")
    if not isinstance(video_path, str) or not video_path:
        raise FileNotFoundError(f"video_path is not configured in meta/info.json for {dataset_dir}")
    chunks_size = int(info.get("chunks_size", 1000))
    chunk_index = episode_index // chunks_size
    resolved = dataset_dir / video_path.format(
        episode_chunk=chunk_index,
        video_key=video_key,
        episode_index=episode_index,
    )
    if resolved.exists():
        return resolved
    candidates = [
        candidate
        for candidate in sorted(dataset_dir.rglob(f"episode_{episode_index:06d}.*"))
        if video_key in candidate.as_posix()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Video-backed frame source for {video_key!r} episode_{episode_index:06d} not found under {dataset_dir}"
        )
    return candidates[0]


def decode_image_value(value: Any, *, dataset_dir: Path) -> np.ndarray:
    if isinstance(value, np.ndarray):
        image = value
    elif isinstance(value, dict):
        raw_bytes = value.get("bytes")
        if raw_bytes:
            buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("cv2 failed to decode image bytes")
        else:
            path_value = value.get("path")
            if not path_value:
                raise RuntimeError(f"Unsupported image entry without bytes/path: {value}")
            image_path = Path(str(path_value))
            if not image_path.is_absolute():
                image_path = dataset_dir / image_path
            image = read_bgr_image(image_path)
    else:
        raise TypeError(f"Unsupported image entry type: {type(value)!r}")

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC/CHW 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def decode_video_frame_at_timestamp(
    dataset_dir: Path,
    info: dict[str, Any],
    episode_index: int,
    *,
    video_key: str,
    timestamp_s: float,
) -> np.ndarray:
    if decode_video_frames is None:
        raise RuntimeError(
            "LeRobot video asset extraction requires the optional lerobot package"
        ) from LEROBOT_IMPORT_ERROR
    video_path = video_path_for_episode(dataset_dir, info, episode_index, video_key)
    frames = decode_video_frames(video_path, [float(timestamp_s)], tolerance_s=1e-4, backend=None)
    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from {video_path} at timestamp {timestamp_s}")
    frame = frames[0]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected decoded RGB frame, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(frame.dtype, np.floating) and float(frame.max()) <= 1.0 + 1e-6 else 1.0
        frame = np.clip(frame * scale, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def load_cam_high_first_frame(
    dataset_dir: Path,
    info: dict[str, Any],
    episode_index: int,
    df: Any,
) -> np.ndarray:
    image_key = "observation.images.cam_high"
    if image_key in df.columns:
        return decode_image_value(df.iloc[0][image_key], dataset_dir=dataset_dir)
    feature = info.get("features", {}).get(image_key)
    if not isinstance(feature, dict) or feature.get("dtype") != "video":
        raise KeyError(image_key)
    timestamp_s = float(df.iloc[0].get("timestamp", 0.0))
    return decode_video_frame_at_timestamp(
        dataset_dir,
        info,
        episode_index,
        video_key=image_key,
        timestamp_s=timestamp_s,
    )


def load_cam_high_background_image(
    *,
    artifacts_root: Path = ARTIFACTS_ROOT,
    repo_id: str | None = None,
) -> np.ndarray | None:
    background_path = cam_high_background_path_for_repo_id(repo_id, artifacts_root=artifacts_root)
    if not background_path.exists():
        if repo_id is not None and str(repo_id) in CAM_HIGH_BACKGROUND_BY_REPO_ID:
            raise FileNotFoundError(
                f"Configured cam_high background image not found for repo_id={repo_id!r}: {background_path}"
            )
        return None
    return read_bgr_image(background_path)


def build_cam_high_first_frame_overlay(dataset_dir: Path, *, repo_id: str | None = None) -> np.ndarray:
    if pd is None:
        raise RuntimeError(
            "LeRobot distribution assets require the optional pandas dependency"
        ) from PANDAS_IMPORT_ERROR

    info = info_json(dataset_dir)
    indices = episode_indices(dataset_dir)
    if not indices:
        raise RuntimeError(f"No episodes found in {dataset_dir}")
    resolved_repo_id = repo_id or dataset_dir.relative_to(lerobot_home()).as_posix()

    background_image = load_cam_high_background_image(repo_id=resolved_repo_id)
    background_resized: np.ndarray | None = None
    foreground_accum: np.ndarray | None = None
    foreground_weight: np.ndarray | None = None
    count = 0
    target_shape: tuple[int, int] | None = None

    for episode_index in indices:
        parquet_path = parquet_path_for_episode(dataset_dir, info, episode_index)
        df = pd.read_parquet(parquet_path)
        if len(df) == 0:
            continue
        image = load_cam_high_first_frame(
            dataset_dir,
            info,
            episode_index,
            df,
        )
        if target_shape is None:
            target_shape = (int(image.shape[1]), int(image.shape[0]))
            if background_image is None:
                background_resized = image.copy()
            elif (background_image.shape[1], background_image.shape[0]) != target_shape:
                background_resized = cv2.resize(background_image, target_shape, interpolation=cv2.INTER_AREA)
            else:
                background_resized = background_image.copy()
            foreground_accum = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float64)
            foreground_weight = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
        elif (image.shape[1], image.shape[0]) != target_shape:
            image = cv2.resize(image, target_shape, interpolation=cv2.INTER_AREA)
        if background_resized is None or foreground_accum is None or foreground_weight is None:
            raise RuntimeError("Internal error: missing train distribution background image")
        selected_masks = select_relevant_task_masks(image, resolved_repo_id, background_image=background_resized)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
        for selected in selected_masks:
            mask = np.maximum(mask, selected.mask.astype(np.float64))
        foreground_accum += image.astype(np.float64) * mask[..., None]
        foreground_weight += mask
        count += 1

    if background_resized is None or foreground_accum is None or foreground_weight is None or count == 0:
        raise RuntimeError(f"No valid cam_high first frames found in {dataset_dir}")
    safe_weight = np.maximum(foreground_weight[..., None], 1.0)
    fused_foreground = foreground_accum / safe_weight
    occupancy = np.clip(foreground_weight / count, 0.0, 1.0)
    alpha = np.where(
        foreground_weight > 0.0,
        np.clip(0.60 + 0.35 * np.sqrt(occupancy), 0.0, 0.95),
        0.0,
    )[..., None]
    base = background_resized.astype(np.float64)
    return np.clip(base * (1.0 - alpha) + fused_foreground * alpha, 0.0, 255.0).astype(np.uint8)


def ensure_distribution_image(
    dataset_dir: Path,
    repo_id: str,
    *,
    artifacts_root: Path = ARTIFACTS_ROOT,
    force: bool = False,
) -> Path:
    output_path = repo_id_distribution_image_path(repo_id, artifacts_root=artifacts_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path
    image = build_cam_high_first_frame_overlay(dataset_dir, repo_id=repo_id)
    write_bgr_image(output_path, image)
    return output_path


def resolve_prompt(
    *,
    train_config_name: str,
    cli_prompt: str | None,
    dataset_dir: Path | None,
    prompt_cache_path: Path = PROMPT_CACHE_PATH,
) -> tuple[str | None, str | None]:
    if cli_prompt is not None:
        return cli_prompt, "cli"

    cached_prompt = cached_prompt_for_train_config(train_config_name, prompt_cache_path)
    if cached_prompt:
        return cached_prompt, "cache"

    if dataset_dir is not None:
        inferred_prompt = dataset_prompt(dataset_dir)
        if inferred_prompt:
            set_cached_prompt(train_config_name, inferred_prompt, prompt_cache_path)
            return inferred_prompt, "dataset"

    return None, None


def iter_valid_lerobot_datasets(root: Path | None = None) -> list[tuple[str, Path]]:
    dataset_root = (root or lerobot_home()).resolve()
    results: list[tuple[str, Path]] = []
    for info_path in sorted(dataset_root.rglob("meta/info.json")):
        dataset_dir = info_path.parent.parent
        try:
            repo_id = dataset_dir.relative_to(dataset_root).as_posix()
        except Exception:
            continue
        try:
            info = info_json(dataset_dir)
        except Exception:
            continue
        features = info.get("features", {})
        if "observation.images.cam_high" not in features:
            continue
        if not (dataset_dir / "meta" / "episodes.jsonl").exists():
            continue
        results.append((repo_id, dataset_dir))
    return results


def prepare_lerobot_assets(
    *,
    train_config_name: str,
    cli_prompt: str | None,
    need_distribution: bool,
    repo_id: str | None = None,
    artifacts_root: Path = ARTIFACTS_ROOT,
    prompt_cache_path: Path = PROMPT_CACHE_PATH,
) -> PreparedTrainAssets:
    asset_info = dataset_asset_info(train_config_name, repo_id=repo_id, artifacts_root=artifacts_root)
    prompt, prompt_source = resolve_prompt(
        train_config_name=train_config_name,
        cli_prompt=cli_prompt,
        dataset_dir=asset_info.dataset_dir,
        prompt_cache_path=prompt_cache_path,
    )
    if not need_distribution:
        return PreparedTrainAssets(
            prompt=prompt,
            prompt_source=prompt_source,
            distribution_image_path=None,
            distribution_ready=False,
            skip_reason=None,
        )
    existing_distribution_image_path = find_distribution_image_path(asset_info.repo_id, artifacts_root=artifacts_root)
    if asset_info.dataset_dir is None or asset_info.repo_id is None:
        ready_path = existing_distribution_image_path
        if ready_path is None and asset_info.distribution_image_path.exists():
            ready_path = asset_info.distribution_image_path
        return PreparedTrainAssets(
            prompt=prompt,
            prompt_source=prompt_source,
            distribution_image_path=ready_path,
            distribution_ready=ready_path is not None,
            skip_reason=f"dataset not found under {lerobot_home()} for repo_id={asset_info.repo_id!r}",
        )
    if existing_distribution_image_path is not None:
        return PreparedTrainAssets(
            prompt=prompt,
            prompt_source=prompt_source,
            distribution_image_path=existing_distribution_image_path,
            distribution_ready=True,
            skip_reason=None,
        )
    try:
        image_path = ensure_distribution_image(
            asset_info.dataset_dir,
            asset_info.repo_id,
            artifacts_root=artifacts_root,
        )
        return PreparedTrainAssets(
            prompt=prompt,
            prompt_source=prompt_source,
            distribution_image_path=image_path,
            distribution_ready=True,
            skip_reason=None,
        )
    except Exception as exc:
        return PreparedTrainAssets(
            prompt=prompt,
            prompt_source=prompt_source,
            distribution_image_path=None,
            distribution_ready=False,
            skip_reason=f"failed to build train distribution image from {asset_info.dataset_dir}: {exc}",
        )
