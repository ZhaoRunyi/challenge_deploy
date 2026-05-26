from __future__ import annotations

import argparse
import json
from pathlib import Path

from teleop.hdf5_teleop import load_hdf5_teleop_episode, save_hdf5_teleop_episode_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an MP4 preview for an HDF5 teleop episode."
    )
    parser.add_argument("--input", required=True, help="Path to the .hdf5 episode.")
    parser.add_argument("--output", default=None, help="Optional output .mp4 path.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_once(args: argparse.Namespace) -> None:
    episode = load_hdf5_teleop_episode(args.input)
    output_path = save_hdf5_teleop_episode_preview(
        input_path=args.input,
        output_path=args.output,
        fps=args.fps,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "hdf5_teleop_episode": {
                    "input": str(Path(args.input).expanduser().resolve()),
                    "output": str(output_path),
                    "language_instruction": episode.language_instruction,
                    "camera_names": list(episode.camera_names),
                    "steps": int(len(episode.qpos)),
                    "duration_seconds": float(len(episode.qpos) / max(1, args.fps)),
                }
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
