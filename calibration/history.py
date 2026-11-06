"""Extract synchronized LeRobot reference frames; never command robot hardware."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .core import write_json

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


def extract_joints(state, names):
    if names and isinstance(names[0], list):
        names = names[0]
    state = np.asarray(state, dtype=float)
    if not names or len(names) != len(state) or not np.isfinite(state).all():
        raise ValueError("Missing/inconsistent named observation.state schema")
    return {arm: [float(state[names.index(f"{arm}_joint_{joint}")]) for joint in JOINTS]
            for arm in ("left", "right")}


def extract(dataset, episode, output, frames=None, count=12, provenance=None):
    import pyarrow.parquet as pq
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Use a new reference output directory")
    info = json.loads((dataset / "meta/info.json").read_text())
    chunk = episode // info["chunks_size"]
    rows = pq.read_table(dataset / info["data_path"].format(episode_chunk=chunk, episode_index=episode)).to_pylist()
    if not rows or any(int(r["frame_index"]) != i or int(r["episode_index"]) != episode for i, r in enumerate(rows)):
        raise ValueError("Episode rows are missing, unordered or not zero-based")
    fps = float(info["fps"])
    if fps <= 0 or any(abs(float(r["timestamp"]) - i / fps) > .25 / fps for i, r in enumerate(rows)):
        raise ValueError("State timestamps do not match the episode video timeline")
    frames = frames if frames is not None else np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=int).tolist()
    if not frames or any(i < 0 or i >= len(rows) for i in frames):
        raise ValueError("Frame index outside episode")
    output.mkdir(parents=True)
    videos = {}
    sheets = []
    try:
        for cam in CAMERAS:
            path = dataset / info["video_path"].format(episode_chunk=chunk, episode_index=episode, video_key="observation.images." + cam)
            capture = cv2.VideoCapture(str(path))
            videos[cam] = capture
            if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - fps) > .01 or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
                raise ValueError(f"Video/state frame count or FPS mismatch: {cam}")
        for index in frames:
            row = rows[index]
            target = output / f"frame_{index:06d}"
            target.mkdir()
            write_json(target / "reference.json", {
                "source": provenance or {"local_dataset": str(dataset.resolve())},
                "episode": episode, "frame": index, "timestamp": row["timestamp"],
                "joint_angles": extract_joints(row["observation.state"], info["features"]["observation.state"]["names"]),
                "observation_state_raw": row["observation.state"],
                "state_names": info["features"]["observation.state"]["names"],
                "intrinsics_verified": False,
                "purpose": "Visual reference only. No robot motion. Confirm joint units/setup before use."})
            thumbnails = []
            for cam, video in videos.items():
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = video.read()
                if not ok or abs(video.get(cv2.CAP_PROP_POS_FRAMES) - index - 1) > .1:
                    raise ValueError(f"Cannot decode matching frame: {cam}/{index}")
                if not cv2.imwrite(str(target / f"{cam}.png"), image):
                    raise OSError("Could not save reference image")
                thumb = cv2.resize(image, (320, 240))
                cv2.putText(thumb, f"{index}: {cam}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                thumbnails.append(thumb)
            sheets.append(np.hstack(thumbnails))
        if not cv2.imwrite(str(output / "contact_sheet.jpg"), np.vstack(sheets)):
            raise OSError("Could not save contact sheet")
    finally:
        for video in videos.values():
            video.release()
    return frames


def compare(reference, sample, output, alpha=.5, joint_tolerance=.03):
    reference, sample, output = Path(reference), Path(sample), Path(output)
    if output.exists() or not 0 <= alpha <= 1 or joint_tolerance <= 0:
        raise ValueError("Use a new output and valid alpha/tolerance")
    old = json.loads((reference / "reference.json").read_text())
    new = json.loads((sample / "sample.json").read_text())
    delta = {arm: (np.asarray(new["arms"][arm]["qpos"][:6]) - old["joint_angles"][arm]).tolist()
             for arm in ("left", "right")}
    if max(abs(x) for values in delta.values() for x in values) > joint_tolerance:
        raise ValueError("Robot pose differs; do not interpret mismatch as camera error: " + json.dumps(delta))
    output.mkdir(parents=True)
    for cam in CAMERAS:
        a, b = cv2.imread(str(reference / f"{cam}.png")), cv2.imread(str(sample / f"{cam}_raw.png"))
        if a is None or b is None or a.shape != b.shape:
            raise ValueError(f"Missing/different image dimensions: {cam}; no implicit resizing")
        if not cv2.imwrite(str(output / f"{cam}_history_overlay.png"), cv2.addWeighted(a, 1 - alpha, b, alpha, 0)):
            raise OSError("Could not save comparison")
    write_json(output / "comparison.json", {"joint_delta_rad": delta, "reference": old,
        "warning": "Qualitative comparison only: historical intrinsics, distortion and base placement are not verified. No extrinsics changed."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    source = e.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--repo")
    e.add_argument("--revision", default="main")
    e.add_argument("--cache", type=Path, default=Path("artifacts/calibration/history_cache"))
    e.add_argument("--episode", type=int, default=0)
    e.add_argument("--frames", type=int, nargs="+")
    e.add_argument("--count", type=int, default=12)
    e.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--sample", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--alpha", type=float, default=.5)
    c.add_argument("--joint-tolerance", type=float, default=.03)
    args = p.parse_args()
    if args.command == "compare":
        compare(args.reference, args.sample, args.output, args.alpha, args.joint_tolerance)
        return
    if args.episode < 0 or args.count < 1:
        raise ValueError("Episode must be nonnegative and count positive")
    provenance = None
    if args.repo:
        from huggingface_hub import HfApi
        from .hf_reference import download
        sha = HfApi(token=False).dataset_info(args.repo, revision=args.revision).sha
        cache = args.cache / args.repo.replace("/", "--") / sha
        download(args.repo, sha, ["meta/info.json"], cache)
        info = json.loads((cache / "meta/info.json").read_text())
        chunk = args.episode // info["chunks_size"]
        paths = ["meta/info.json", "README.md", info["data_path"].format(episode_chunk=chunk, episode_index=args.episode)]
        paths += [info["video_path"].format(episode_chunk=chunk, episode_index=args.episode, video_key="observation.images." + c) for c in CAMERAS]
        args.dataset = download(args.repo, sha, paths, cache)
        provenance = {"repo": args.repo, "revision": sha}
    frames = extract(args.dataset, args.episode, args.output, args.frames, args.count, provenance)
    print(f"Exported frames {frames}; inspect {args.output / 'contact_sheet.jpg'}. No motion commands sent.")


if __name__ == "__main__":
    main()
