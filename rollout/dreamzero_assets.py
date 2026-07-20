from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from .assets import (
    PreparedTrainAssets,
    artifact_keys,
    cached_prompt_for_keys,
    find_distribution_image_path,
)
from .lerobot_assets import dataset_prompt, ensure_distribution_image, lerobot_home


def dreamzero_dataset_dirs(spec: Any) -> list[Path]:
    raw_paths = getattr(spec, "train_data_paths", None)
    if raw_paths is None:
        return []
    if isinstance(raw_paths, str):
        patterns = [raw_paths]
    else:
        patterns = [str(path) for path in raw_paths]
    dirs: list[Path] = []
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)):
            path = Path(match)
            if (path / "meta" / "info.json").exists() and path not in dirs:
                dirs.append(path)
    return dirs


def dreamzero_repo_id_for_dataset(dataset_dir: Path) -> str:
    for root in (Path("/workspace/data"), lerobot_home()):
        try:
            return dataset_dir.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            pass
    return dataset_dir.name


def select_dreamzero_dataset_dir(dataset_dirs: list[Path], prompt: str | None) -> Path | None:
    if not dataset_dirs:
        return None
    if prompt:
        prompt_lower = prompt.lower()
        for dataset_dir in dataset_dirs:
            task = dataset_prompt(dataset_dir)
            if task and (task.lower() in prompt_lower or prompt_lower in task.lower()):
                return dataset_dir
    return dataset_dirs[0]


def resolve_dreamzero_distribution_image(spec: Any, prompt: str | None) -> tuple[Path | None, str | None]:
    aliases = tuple(str(alias) for alias in getattr(spec, "distribution_aliases", ()) if str(alias))
    image_path = find_distribution_image_path(Path(str(spec.config_path)).stem, aliases=aliases)
    if image_path is not None:
        return image_path, None
    dataset_dirs = dreamzero_dataset_dirs(spec)
    selected = select_dreamzero_dataset_dir(dataset_dirs, prompt)
    if selected is None:
        return None, f"no DreamZero LeRobot dataset found for train_data_paths={getattr(spec, 'train_data_paths', None)!r}"
    repo_id = dreamzero_repo_id_for_dataset(selected)
    image_path = find_distribution_image_path(repo_id, aliases=aliases)
    if image_path is not None:
        return image_path, None
    try:
        return ensure_distribution_image(selected, repo_id), None
    except Exception as exc:
        return None, f"failed to build DreamZero train distribution image from {selected}: {exc}"


def prepare_dreamzero_client_assets(
    *,
    train_config_name: str,
    cli_prompt: str | None,
    need_distribution: bool,
    spec: Any,
    server_metadata: dict[str, Any] | None,
) -> PreparedTrainAssets:
    aliases = tuple(str(alias) for alias in getattr(spec, "distribution_aliases", ()) if str(alias))
    if cli_prompt is not None:
        prompt, prompt_source = cli_prompt, "cli"
    else:
        prompt, prompt_source = cached_prompt_for_keys(artifact_keys(train_config_name, aliases))
        if prompt is None:
            prompt = (server_metadata or {}).get("default_prompt") or getattr(spec, "prompt", None)
            prompt_source = "server_default" if (server_metadata or {}).get("default_prompt") else ("train_config" if prompt else None)
        if prompt is None:
            selected = select_dreamzero_dataset_dir(dreamzero_dataset_dirs(spec), None)
            prompt = dataset_prompt(selected) if selected is not None else None
            prompt_source = "dataset" if prompt else None
    distribution_image_path = None
    distribution_skip_reason = None
    if need_distribution:
        distribution_image_path, distribution_skip_reason = resolve_dreamzero_distribution_image(spec, prompt)
    return PreparedTrainAssets(
        prompt=prompt,
        prompt_source=prompt_source,
        distribution_image_path=distribution_image_path,
        distribution_ready=distribution_image_path is not None,
        skip_reason=distribution_skip_reason,
    )


