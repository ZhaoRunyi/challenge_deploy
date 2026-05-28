from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from teleop.hdf5_teleop import save_hdf5_teleop_episode_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an MP4 preview for an HDF5 teleop episode."
    )
    parser.add_argument("--input", required=True, help="Path to the .hdf5 episode.")
    parser.add_argument("--output", default=None, help="Optional output .mp4 path.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def episode_metadata(path: str | Path) -> dict[str, object]:
    with h5py.File(path, "r") as root:
        language_raw = root["/language_instruction"][()] if "/language_instruction" in root else None
        if language_raw is None:
            language_instruction = None
        elif hasattr(language_raw, "__len__") and not isinstance(language_raw, (bytes, str)):
            language_instruction = language_raw[0].decode("utf-8") if len(language_raw) > 0 and isinstance(language_raw[0], bytes) else (str(language_raw[0]) if len(language_raw) > 0 else None)
        else:
            language_instruction = language_raw.decode("utf-8") if isinstance(language_raw, bytes) else str(language_raw)
        steps = int(root["/observations/qpos"].shape[0])
        camera_names = list(root["/observations/images"].keys())
    return {
        "language_instruction": language_instruction,
        "camera_names": camera_names,
        "steps": steps,
    }


def run_once(args: argparse.Namespace) -> None:
    metadata = episode_metadata(args.input)
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
                    "language_instruction": metadata["language_instruction"],
                    "camera_names": metadata["camera_names"],
                    "steps": metadata["steps"],
                    "duration_seconds": float(int(metadata["steps"]) / max(1, args.fps)),
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
