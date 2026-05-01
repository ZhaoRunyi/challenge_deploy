#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import imageio.v3 as iio


# Edit this list directly for repeated manual annotation. Each entry can be an
# eval result folder, an mp4 file, or a higher-level folder containing records.
HARDCODED_PATHS: list[Path] = [
    # Path("/home/edemlab/challenge_ws/deploy/artifacts/openpi_records/..."),
]

OUTPUT_NAME = "action_steps_real_time.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract rollout video frames and annotate action_step / real_time for eval records."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Optional eval dirs, mp4 files, or parent dirs.")
    parser.add_argument("--dir", action="append", type=Path, default=[], help="Eval dir, mp4 file, or parent dir.")
    parser.add_argument("--force-frames", action="store_true", help="Rewrite extracted png frames even if they exist.")
    parser.add_argument("--only-frame", "--only_frame", action="store_true", help="Only extract video frames; do not ask for action steps.")
    return parser


def is_actual_rollout_video(path: Path) -> bool:
    if path.suffix.lower() != ".mp4":
        return False
    name = path.name
    if name.endswith(".tmp.mp4") or name.endswith("_predicted_video.mp4"):
        return False
    return name.endswith("_videos.mp4") or "_videos" in path.stem


def discover_videos(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path] if is_actual_rollout_video(path) else []
    if not path.exists():
        raise FileNotFoundError(path)
    direct = [item for item in sorted(path.glob("*.mp4")) if is_actual_rollout_video(item)]
    if direct:
        return direct
    return [item for item in sorted(path.rglob("*.mp4")) if is_actual_rollout_video(item)]


def frames_dir_for_video(video_path: Path) -> Path:
    return video_path.parent / video_path.stem


def _readable_image(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        next(iter(iio.imiter(path)))
    except Exception:
        try:
            iio.imread(path)
        except Exception:
            return False
    return True


def extract_frames(video_path: Path, *, force: bool) -> tuple[Path, int, int]:
    frames_dir = frames_dir_for_video(video_path)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for old_frame in frames_dir.glob("frame_*.png"):
            old_frame.unlink()

    count = 0
    written = 0
    for count, frame in enumerate(iio.imiter(video_path), start=1):
        frame_path = frames_dir / f"frame_{count:06d}.png"
        if not force and _readable_image(frame_path):
            continue
        iio.imwrite(frame_path, frame)
        written += 1
    return frames_dir, count, written


def parse_action_step(raw_value: str, *, frame_count: int) -> int:
    value = raw_value.strip()
    match = re.search(r"frame_(\d+)", value)
    if match:
        step = int(match.group(1))
    else:
        step = int(value)
    if step < 1 or step > frame_count:
        raise ValueError(f"action step must be in [1, {frame_count}], got {step}")
    return step


def find_metrics_json(record_dir: Path) -> Path | None:
    candidates = sorted(record_dir.glob("*_rollout_metrics.json"))
    if candidates:
        return candidates[0]
    for name in ("rollout_metrics.json", "metrics.json"):
        path = record_dir / name
        if path.exists():
            return path
    return None


def estimate_real_time(record_dir: Path, action_step: int) -> tuple[float | None, str | None, str | None]:
    metrics_path = find_metrics_json(record_dir)
    if metrics_path is None:
        return None, None, "no rollout metrics json found"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    executed_steps = metrics.get("executed_steps")
    total_seconds = metrics.get("rollout_wall_seconds", metrics.get("total_wall_seconds"))
    if not isinstance(executed_steps, (int, float)) or executed_steps <= 0:
        return None, str(metrics_path), "metrics json has no positive executed_steps"
    if not isinstance(total_seconds, (int, float)) or total_seconds <= 0:
        return None, str(metrics_path), "metrics json has no positive rollout_wall_seconds/total_wall_seconds"

    clipped_step = min(float(action_step), float(executed_steps))
    return float(total_seconds) * clipped_step / float(executed_steps), str(metrics_path), None


def process_video(video_path: Path, *, force_frames: bool, only_frame: bool) -> None:
    record_dir = video_path.parent
    output_path = record_dir / OUTPUT_NAME
    frames_dir, frame_count, written_count = extract_frames(video_path, force=force_frames)
    if frame_count <= 0:
        print(f"Warning: no frames extracted from {video_path}")
        return
    print(f"Frames ready: {frames_dir.resolve()} ({frame_count} total, {written_count} written)")
    if only_frame:
        return
    if output_path.exists():
        print(f"Skip existing: {output_path}")
        print(output_path.read_text(encoding="utf-8"))
        return

    first_frame_path = (frames_dir / "frame_000001.png").resolve()
    print(f"First frame: {first_frame_path}")
    raw_step = input(f"Enter end action step [1-{frame_count}] for {record_dir.name}: ")
    action_step = parse_action_step(raw_step, frame_count=frame_count)
    real_time_seconds, metrics_path, warning = estimate_real_time(record_dir, action_step)
    if warning is not None:
        print(f"Warning: {warning}; real_time_seconds will be null.")

    payload: dict[str, Any] = {
        "action_step": action_step,
        "real_time_seconds": real_time_seconds,
        "frame_count": frame_count,
        "video_path": str(video_path.resolve()),
        "frames_dir": str(frames_dir.resolve()),
        "metrics_path": metrics_path,
        "warning": warning,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"action_step": action_step, "real_time_seconds": real_time_seconds}, indent=2))
    print(f"Saved: {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    roots = [*HARDCODED_PATHS, *args.dir, *args.paths]
    if not roots:
        raise SystemExit("No input path. Set HARDCODED_PATHS at the top, pass --dir, or pass a path argument.")

    videos: list[Path] = []
    for root in roots:
        videos.extend(discover_videos(root))
    unique_videos = sorted(dict.fromkeys(video.resolve() for video in videos))
    if not unique_videos:
        raise SystemExit("No actual rollout videos found. Expected files like *_videos.mp4.")

    for index, video_path in enumerate(unique_videos, start=1):
        print(f"[{index}/{len(unique_videos)}] {video_path}")
        process_video(video_path, force_frames=args.force_frames, only_frame=args.only_frame)


if __name__ == "__main__":
    main()
