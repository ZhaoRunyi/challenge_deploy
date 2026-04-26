from __future__ import annotations

import argparse
import json

from challenge_deploy.lerobot_assets import (
    dataset_prompt,
    ensure_distribution_image,
    get_train_config_repo_id,
    iter_valid_lerobot_datasets,
    set_cached_prompt,
    train_config_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute train-distribution overlays and prompt cache for local LeRobot datasets."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate train-distribution overlays even if the target png already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_to_train_configs: dict[str, list[str]] = {}
    for train_config_name in train_config_names():
        repo_id = get_train_config_repo_id(train_config_name)
        if repo_id is None:
            continue
        repo_to_train_configs.setdefault(repo_id, []).append(train_config_name)

    results: list[dict[str, object]] = []
    for repo_id, dataset_dir in iter_valid_lerobot_datasets():
        item: dict[str, object] = {
            "repo_id": repo_id,
            "dataset_dir": str(dataset_dir),
            "train_configs": repo_to_train_configs.get(repo_id, []),
        }
        try:
            distribution_image_path = ensure_distribution_image(dataset_dir, repo_id, force=args.force)
            item["distribution_image_path"] = str(distribution_image_path)
        except Exception as exc:
            item["distribution_error"] = repr(exc)

        try:
            prompt = dataset_prompt(dataset_dir)
            item["prompt"] = prompt
            if prompt:
                for train_config_name in repo_to_train_configs.get(repo_id, []):
                    set_cached_prompt(train_config_name, prompt)
        except Exception as exc:
            item["prompt_error"] = repr(exc)
        results.append(item)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
