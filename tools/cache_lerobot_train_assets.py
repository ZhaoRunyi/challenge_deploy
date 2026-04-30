from __future__ import annotations

import argparse
import cv2
import json

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.lerobot_assets import (
    default_cam_high_background_path,
    dataset_prompt,
    ensure_distribution_image,
    get_train_config_repo_id,
    iter_valid_lerobot_datasets,
    set_cached_prompt,
    train_config_names,
)
from challenge_deploy.realsense import RealSenseRig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute train-distribution overlays and prompt cache for local LeRobot datasets."
    )
    parser.add_argument(
        "--capture-background",
        action="store_true",
        help="Capture a fresh clean cam_high background image before regenerating overlays.",
    )
    parser.add_argument(
        "--config",
        default="/home/edemlab/challenge_ws/deploy/configs/dual_piper_example.yaml",
        help="Runtime config used only when --capture-background is enabled.",
    )
    parser.add_argument(
        "--camera-front-serial",
        default=None,
        help="Optional cam_high serial override used only when --capture-background is enabled.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate train-distribution overlays even if the target png already exists.",
    )
    return parser


def _capture_background_image(*, config_path: str, camera_front_serial: str | None) -> str:
    config = load_config(config_path)
    if camera_front_serial:
        set_by_dotted_path(config, "cameras.serials.cam_high", camera_front_serial)

    output_path = default_cam_high_background_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rig = RealSenseRig(
        {"cam_high": config["cameras"]["serials"]["cam_high"]},
        width=int(config["cameras"]["width"]),
        height=int(config["cameras"]["height"]),
        fps=int(config["cameras"]["fps"]),
        warmup_frames=int(config["cameras"]["warmup_frames"]),
    )
    try:
        rig.start()
        image = rig.capture()["cam_high"]
    finally:
        rig.stop()

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write background image: {output_path}")
    return str(output_path)


def main() -> None:
    args = build_parser().parse_args()
    effective_force = bool(args.force or args.capture_background)
    captured_background_path: str | None = None
    if args.capture_background:
        captured_background_path = _capture_background_image(
            config_path=args.config,
            camera_front_serial=args.camera_front_serial,
        )

    repo_to_train_configs: dict[str, list[str]] = {}
    for train_config_name in train_config_names():
        repo_id = get_train_config_repo_id(train_config_name)
        if repo_id is None:
            continue
        repo_to_train_configs.setdefault(repo_id, []).append(train_config_name)

    results: list[dict[str, object]] = []
    if captured_background_path is not None:
        results.append(
            {
                "captured_background_path": captured_background_path,
            }
        )
    for repo_id, dataset_dir in iter_valid_lerobot_datasets():
        item: dict[str, object] = {
            "repo_id": repo_id,
            "dataset_dir": str(dataset_dir),
            "train_configs": repo_to_train_configs.get(repo_id, []),
        }
        try:
            distribution_image_path = ensure_distribution_image(dataset_dir, repo_id, force=effective_force)
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
