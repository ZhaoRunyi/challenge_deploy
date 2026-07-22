from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = DEPLOY_ROOT / "artifacts"
TRAIN_DISTRIBUTION_DIR = ARTIFACTS_ROOT / "train_distributions"
PROMPT_CACHE_PATH = ARTIFACTS_ROOT / "trainconfig_prompts.json"
def safe_filename_part(value: str) -> str:
    value = value.strip()
    value = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("._-")


def read_bgr_image(path: Path) -> np.ndarray:
    image = np.asarray(iio.imread(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape} from {path}")
    if image.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(image.dtype, np.floating) and float(image.max()) <= 1.0 + 1e-6 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    return image[..., ::-1].copy()


def write_bgr_image(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    iio.imwrite(path, image[..., ::-1])


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


def repo_id_distribution_image_path(repo_id: str, *, artifacts_root: Path = ARTIFACTS_ROOT) -> Path:
    safe_repo_id = safe_filename_part(repo_id.replace("/", "__"))
    return artifacts_root / "train_distributions" / f"{safe_repo_id}_cam_high_first_frame_overlay.png"


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


def artifact_keys(train_config_name: str, aliases: tuple[str, ...] = ()) -> list[str]:
    keys: list[str] = []
    for value in (train_config_name, *aliases):
        key = str(value).strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def cached_prompt_for_keys(keys: list[str], prompt_cache_path: Path = PROMPT_CACHE_PATH) -> tuple[str | None, str | None]:
    prompt_map = load_prompt_cache(prompt_cache_path)
    for key in keys:
        prompt = prompt_map.get(key)
        if prompt:
            source = "cache" if key == keys[0] else f"cache:{key}"
            return prompt, source
    return None, None


def distribution_image_search_terms(repo_id: str, aliases: tuple[str, ...] = ()) -> list[str]:
    raw_terms = [repo_id, repo_id.replace("/", "__"), Path(repo_id).name, *aliases]
    terms: list[str] = []
    for raw_term in raw_terms:
        term = safe_filename_part(str(raw_term).replace("/", "__"))
        if term and term not in terms:
            terms.append(term)
        if "__" in term:
            tail = term.split("__")[-1]
            if tail and tail not in terms:
                terms.append(tail)
    return terms


def distribution_image_score(path: Path, terms: list[str]) -> tuple[int, int, int, str]:
    name = path.name.lower()
    score = sum(1 for term in terms if term.lower() in name)
    longest = max((len(term) for term in terms if term.lower() in name), default=0)
    source_priority = 0 if name.startswith("embodichain_sim_data") else 1
    return score, source_priority, longest, path.name


def find_distribution_image_path(repo_id: str | None, *, artifacts_root: Path = ARTIFACTS_ROOT, aliases: tuple[str, ...] = ()) -> Path | None:
    if not repo_id:
        return None
    exact_path = repo_id_distribution_image_path(repo_id, artifacts_root=artifacts_root)
    if exact_path.exists():
        return exact_path
    distribution_dir = artifacts_root / "train_distributions"
    if not distribution_dir.exists():
        return None
    terms = distribution_image_search_terms(repo_id, aliases)
    candidates: list[Path] = []
    for term in terms:
        candidates.extend(distribution_dir.glob(f"*{term}*_cam_high_first_frame_overlay.png"))
    candidates = sorted(set(path for path in candidates if path.is_file()))
    if not candidates:
        return None
    return max(candidates, key=lambda path: distribution_image_score(path, terms))


def record_repo_id_for_motus_distribution(spec: Any) -> str | None:
    if not spec.repo_id:
        return None
    owner, separator, dataset_name = str(spec.repo_id).partition("/")
    if separator and dataset_name.startswith("Motus_"):
        return f"{owner}/{dataset_name[len('Motus_') :]}"
    return str(spec.repo_id)


def record_dataset_name_for_motus_prompt(spec: Any, prompt: str | None) -> str | None:
    if prompt is None:
        return None
    task_names = spec.config.get("dataset", {}).get("task_name") or []
    if isinstance(task_names, str):
        task_names = [task_names]
    manifest_path = DEPLOY_ROOT.parent / "baselines" / "Motus" / "t5_prompt_cache" / "prompt_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for dataset in manifest.get("datasets", []):
        dataset_name = str(dataset.get("dataset_name") or Path(str(dataset.get("dataset_root", ""))).name)
        if task_names and dataset_name not in task_names:
            continue
        if any(str(item.get("prompt", "")).strip() == prompt for item in dataset.get("prompts", [])):
            return dataset_name
    return None


def resolve_motus_distribution_image(spec: Any, prompt: str | None = None) -> tuple[Path | None, str | None]:
    repo_id = record_repo_id_for_motus_distribution(spec)
    aliases: list[str] = []
    matched_task = record_dataset_name_for_motus_prompt(spec, prompt)
    if matched_task:
        aliases.append(matched_task.removeprefix("Motus_"))
        dataset_params = spec.config.get("dataset", {}).get("params", {})
        dataset_root = dataset_params.get("root")
        if dataset_root:
            repo_id = f"{Path(str(dataset_root)).name}/{matched_task.removeprefix('Motus_')}"
    if repo_id is None:
        return None, "train config does not define dataset.params.repo_id"
    distribution_image_path = find_distribution_image_path(repo_id, aliases=tuple(aliases))
    if distribution_image_path is not None:
        return distribution_image_path, None
    exact_path = repo_id_distribution_image_path(repo_id)
    return None, f"train distribution image not found for repo_id={repo_id!r}; exact path would be {exact_path}"


def resolve_named_artifact_distribution(
    train_config_name: str,
    *,
    artifacts_root: Path = ARTIFACTS_ROOT,
    aliases: tuple[str, ...] = (),
) -> tuple[Path | None, str | None]:
    image_path = find_distribution_image_path(
        train_config_name,
        artifacts_root=artifacts_root,
        aliases=aliases,
    )
    if image_path is not None:
        return image_path, None
    exact_path = repo_id_distribution_image_path(train_config_name, artifacts_root=artifacts_root)
    return None, f"train distribution image not found for train_config={train_config_name!r}; exact path would be {exact_path}"


def prepare_named_artifact_assets(
    *,
    train_config_name: str,
    cli_prompt: str | None,
    need_distribution: bool,
    artifacts_root: Path = ARTIFACTS_ROOT,
    prompt_cache_path: Path = PROMPT_CACHE_PATH,
    aliases: tuple[str, ...] = (),
    default_prompt: str | None = None,
) -> PreparedTrainAssets:
    if cli_prompt is not None:
        prompt, prompt_source = cli_prompt, "cli"
    else:
        keys = artifact_keys(train_config_name, aliases)
        prompt, prompt_source = cached_prompt_for_keys(
            keys,
            prompt_cache_path,
        )
        if prompt is None and default_prompt is not None:
            prompt, prompt_source = default_prompt, "train_config"
    distribution_image_path = None
    distribution_skip_reason = None
    if need_distribution:
        distribution_image_path, distribution_skip_reason = resolve_named_artifact_distribution(
            train_config_name,
            artifacts_root=artifacts_root,
            aliases=aliases,
        )
    return PreparedTrainAssets(
        prompt=prompt,
        prompt_source=prompt_source,
        distribution_image_path=distribution_image_path,
        distribution_ready=distribution_image_path is not None,
        skip_reason=distribution_skip_reason,
    )


def default_prompt_from_spec(spec: Any) -> str | None:
    train_config = getattr(spec, "train_config", None)
    prompt = getattr(train_config, "prompt", None)
    return prompt if isinstance(prompt, str) and prompt else None


def named_artifact_config_from_spec(spec: Any) -> tuple[str, tuple[str, ...], str | None]:
    train_config = getattr(spec, "train_config", None)
    distribution_name = getattr(train_config, "distribution_name", None) or getattr(spec, "train_config_name")
    distribution_aliases = getattr(train_config, "distribution_aliases", ())
    if not isinstance(distribution_aliases, tuple):
        distribution_aliases = tuple(distribution_aliases)
    return str(distribution_name), distribution_aliases, default_prompt_from_spec(spec)


def prepare_motus_client_assets(
    *,
    train_config_name: str,
    cli_prompt: str | None,
    need_distribution: bool,
    spec: Any,
    server_metadata: dict[str, Any] | None,
) -> PreparedTrainAssets:
    server_metadata = server_metadata or {}
    task_names = spec.config.get("dataset", {}).get("task_name") or []
    if isinstance(task_names, str):
        task_names = [task_names]
    aliases = [
        Path(train_config_name).stem,
        str(getattr(spec, "repo_id", "") or ""),
        Path(str(getattr(spec, "repo_id", "") or "")).name,
        *(str(task_name) for task_name in task_names),
    ]
    if cli_prompt is not None:
        prompt, prompt_source = cli_prompt, "cli"
    else:
        prompt, prompt_source = cached_prompt_for_keys(artifact_keys(train_config_name, tuple(aliases)))
        if prompt is None:
            prompt = server_metadata.get("default_prompt")
            prompt_source = "server_default" if prompt is not None else None
    distribution_image_path = None
    distribution_skip_reason = None
    if need_distribution:
        distribution_image_path, distribution_skip_reason = resolve_motus_distribution_image(spec, prompt)
    return PreparedTrainAssets(
        prompt=prompt,
        prompt_source=prompt_source,
        distribution_image_path=distribution_image_path,
        distribution_ready=distribution_image_path is not None,
        skip_reason=distribution_skip_reason,
    )


def prepare_client_assets(
    *,
    client_kind: str,
    train_config_name: str,
    cli_prompt: str | None,
    need_distribution: bool = False,
    spec: Any | None = None,
    server_metadata: dict[str, Any] | None = None,
) -> PreparedTrainAssets:
    if client_kind == "motus":
        if spec is None:
            raise ValueError("Motus assets require a loaded policy spec")
        return prepare_motus_client_assets(
            train_config_name=train_config_name,
            cli_prompt=cli_prompt,
            need_distribution=need_distribution,
            spec=spec,
            server_metadata=server_metadata,
        )
    if client_kind == "xvla":
        if spec is None:
            raise ValueError("X-VLA assets require a loaded policy spec")
        distribution_name, distribution_aliases, default_prompt = named_artifact_config_from_spec(spec)
        return prepare_named_artifact_assets(
            train_config_name=distribution_name,
            cli_prompt=cli_prompt,
            need_distribution=need_distribution,
            aliases=distribution_aliases,
            default_prompt=default_prompt,
        )
    if client_kind == "fastwam":
        if spec is None:
            raise ValueError("FastWAM assets require a loaded policy spec")
        distribution_name = getattr(spec, "distribution_name", None) or train_config_name
        distribution_aliases = getattr(spec, "distribution_aliases", ())
        if not isinstance(distribution_aliases, tuple):
            distribution_aliases = tuple(distribution_aliases)
        return prepare_named_artifact_assets(
            train_config_name=str(distribution_name),
            cli_prompt=cli_prompt,
            need_distribution=need_distribution,
            aliases=tuple(str(alias) for alias in distribution_aliases),
            default_prompt=getattr(spec, "prompt", None),
        )
    raise ValueError(f"Unsupported client_kind: {client_kind}")
