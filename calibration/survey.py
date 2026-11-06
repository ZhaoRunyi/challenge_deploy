"""Survey public high-camera videos for visible-arm reference poses; no hardware."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import HfApi

from .core import write_json
from .hf_reference import download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/calibration/history_cache"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        target = args.output / task
        if target.exists():
            raise ValueError(f"Existing survey: {target}")
        repo = "RoboSynChallenge/cobotmagic_Real_" + task
        print(f"Surveying {repo}, episode {args.episode}", flush=True)
        sha = HfApi(token=False).dataset_info(repo).sha
        cache = args.cache / repo.replace("/", "--") / sha
        download(repo, sha, ["meta/info.json"], cache)
        info = json.loads((cache / "meta/info.json").read_text())
        video = info["video_path"].format(episode_chunk=args.episode // info["chunks_size"],
                                         episode_index=args.episode, video_key="observation.images.cam_high")
        download(repo, sha, [video], cache)
        cap = cv2.VideoCapture(str(cache / video))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or count < 1:
                raise ValueError("Cannot decode survey video")
            indices = np.linspace(0, count - 1, 16, dtype=int).tolist()
            tiles = []
            target.mkdir()
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode frame {index}")
                cv2.imwrite(str(target / f"frame_{index:06d}.png"), image)
                tile = cv2.resize(image, (320, 240))
                cv2.rectangle(tile, (0, 0), (320, 24), (0, 0, 0), -1)
                cv2.putText(tile, f"{task} / {index}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                tiles.append(tile)
            cv2.imwrite(str(target / "survey.jpg"), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, 16, 4)]))
            write_json(target / "source.json", {"repo": repo, "revision": sha, "episode": args.episode,
                       "frames": indices, "camera": "cam_high", "selection": "uniform samples, not a safety assessment"})
        finally:
            cap.release()
        print(f"Saved {target / 'survey.jpg'}", flush=True)


if __name__ == "__main__":
    main()
