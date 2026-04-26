from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from openpi.training import config as openpi_config

from .task_segmentation import select_relevant_task_masks

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEROBOT_HOME = Path("/home/edemlab/challenge_ws/data")
ARTIFACTS_ROOT = DEPLOY_ROOT / "artifacts"
TRAIN_DISTRIBUTION_DIR = ARTIFACTS_ROOT / "train_distributions"
BACKGROUND_DIR = ARTIFACTS_ROOT / "backgrounds"
PROMPT_CACHE_PATH = ARTIFACTS_ROOT / "trainconfig_prompts.json"


def _safe_filename_part(value: str) -> str:
    value = value.strip()
    value = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("._-")


@dataclass(frozen=True)
class DatasetAssetInfo:
    train_config_name: str
    repo_id: str | None
    dataset_dir: Path | None
    distribution_image_path: Path


@dataclass(frozen=True)
class PreparedTrainAssets:
    prompt: str | None
    prompt_source: str | None
    distribution_image_path: Path | None
    distribution_ready: bool
    skip_reason: str | None = None


def lerobot_home() -> Path:
    raw_value = None
    try:
        import os

        raw_value = os.environ.get("HF_LEROBOT_HOME")
    except Exception:
        raw_value = None
    raw = Path(raw_value).expanduser() if raw_value else DEFAULT_LEROBOT_HOME
    return raw.resolve()


def get_train_config_repo_id(train_config_name: str) -> str | None:
    cfg = openpi_config.get_config(train_config_name)
    return getattr(cfg.data, "repo_id", None)


def dataset_dir_for_repo_id(repo_id: str | None, *, root: Path | None = None) -> Path | None:
    if not repo_id:
        return None
    dataset_dir = (root or lerobot_home()) / repo_id
    return dataset_dir if dataset_dir.exists() else None


def repo_id_distribution_image_path(repo_id: str, *, artifacts_root: Path = ARTIFACTS_ROOT) -> Path:
    safe_repo_id = _safe_filename_part(repo_id.replace("/", "__"))
    return artifacts_root / "train_distributions" / f"{safe_repo_id}_cam_high_first_frame_overlay.png"


def default_cam_high_background_path(*, artifacts_root: Path = ARTIFACTS_ROOT) -> Path:
    return artifacts_root / "backgrounds" / "cam_high_clean.png"


def dataset_asset_info(train_config_name: str, *, artifacts_root: Path = ARTIFACTS_ROOT) -> DatasetAssetInfo:
    repo_id = get_train_config_repo_id(train_config_name)
    dataset_dir = dataset_dir_for_repo_id(repo_id)
    distribution_image_path = (
        repo_id_distribution_image_path(repo_id, artifacts_root=artifacts_root)
        if repo_id
        else artifacts_root / "train_distributions" / f"{_safe_filename_part(train_config_name)}.png"
    )
    return DatasetAssetInfo(
        train_config_name=train_config_name,
        repo_id=repo_id,
        dataset_dir=dataset_dir,
        distribution_image_path=distribution_image_path,
    )


def load_prompt_cache(prompt_cache_path: Path = PROMPT_CACHE_PATH) -> dict[str, str]:
    if not prompt_cache_path.exists():
        return {}
    try:
        data = json.loads(prompt_cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


def save_prompt_cache(prompt_map: dict[str, str], prompt_cache_path: Path = PROMPT_CACHE_PATH) -> None:
    prompt_cache_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_cache_path.write_text(
        json.dumps(dict(sorted(prompt_map.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def set_cached_prompt(train_config_name: str, prompt: str, prompt_cache_path: Path = PROMPT_CACHE_PATH) -> None:
    prompt_map = load_prompt_cache(prompt_cache_path)
    prompt_map[train_config_name] = prompt
    save_prompt_cache(prompt_map, prompt_cache_path)


def cached_prompt_for_train_config(train_config_name: str, prompt_cache_path: Path = PROMPT_CACHE_PATH) -> str | None:
    return load_prompt_cache(prompt_cache_path).get(train_config_name)


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


def _episode_indices(dataset_dir: Path) -> list[int]:
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


def _info_json(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    return json.loads(info_path.read_text(encoding="utf-8"))


def _parquet_path_for_episode(dataset_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
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


def _decode_image_value(value: Any, *, dataset_dir: Path) -> np.ndarray:
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
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cv2 failed to read image path: {image_path}")
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


def load_cam_high_background_image(*, artifacts_root: Path = ARTIFACTS_ROOT) -> np.ndarray | None:
    background_path = default_cam_high_background_path(artifacts_root=artifacts_root)
    if not background_path.exists():
        return None
    image = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read clean cam_high background image: {background_path}")
    return image


def build_cam_high_first_frame_overlay(dataset_dir: Path, *, repo_id: str | None = None) -> np.ndarray:
    info = _info_json(dataset_dir)
    episode_indices = _episode_indices(dataset_dir)
    if not episode_indices:
        raise RuntimeError(f"No episodes found in {dataset_dir}")
    resolved_repo_id = repo_id or dataset_dir.relative_to(lerobot_home()).as_posix()

    background_image = load_cam_high_background_image()
    background_resized: np.ndarray | None = None
    foreground_accum: np.ndarray | None = None
    foreground_weight: np.ndarray | None = None
    count = 0
    target_shape: tuple[int, int] | None = None

    for episode_index in episode_indices:
        parquet_path = _parquet_path_for_episode(dataset_dir, info, episode_index)
        df = pd.read_parquet(parquet_path)
        if len(df) == 0:
            continue
        value = df.iloc[0]["observation.images.cam_high"]
        image = _decode_image_value(value, dataset_dir=dataset_dir)
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
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write train distribution image: {output_path}")
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


def prepare_train_assets(
    *,
    train_config_name: str,
    cli_prompt: str | None,
    artifacts_root: Path = ARTIFACTS_ROOT,
    prompt_cache_path: Path = PROMPT_CACHE_PATH,
) -> PreparedTrainAssets:
    asset_info = dataset_asset_info(train_config_name, artifacts_root=artifacts_root)
    prompt, prompt_source = resolve_prompt(
        train_config_name=train_config_name,
        cli_prompt=cli_prompt,
        dataset_dir=asset_info.dataset_dir,
        prompt_cache_path=prompt_cache_path,
    )

    if asset_info.dataset_dir is None or asset_info.repo_id is None:
        return PreparedTrainAssets(
            prompt=prompt,
            prompt_source=prompt_source,
            distribution_image_path=asset_info.distribution_image_path if asset_info.distribution_image_path.exists() else None,
            distribution_ready=asset_info.distribution_image_path.exists(),
            skip_reason=f"dataset not found under {lerobot_home()} for repo_id={asset_info.repo_id!r}",
        )

    if asset_info.distribution_image_path.exists():
        return PreparedTrainAssets(
            prompt=prompt,
            prompt_source=prompt_source,
            distribution_image_path=asset_info.distribution_image_path,
            distribution_ready=True,
            skip_reason=None,
        )

    try:
        image_path = ensure_distribution_image(asset_info.dataset_dir, asset_info.repo_id, artifacts_root=artifacts_root)
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
            info = _info_json(dataset_dir)
        except Exception:
            continue
        features = info.get("features", {})
        if "observation.images.cam_high" not in features:
            continue
        if not (dataset_dir / "meta" / "episodes.jsonl").exists():
            continue
        results.append((repo_id, dataset_dir))
    return results


def train_config_names() -> list[str]:
    names = getattr(openpi_config, "_CONFIGS", {})
    if isinstance(names, dict):
        return sorted(str(name) for name in names.keys())
    if isinstance(names, (list, tuple)):
        results: list[str] = []
        for item in names:
            name = getattr(item, "name", None)
            if isinstance(name, str):
                results.append(name)
        return sorted(results)
    return []
