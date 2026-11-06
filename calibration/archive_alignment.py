"""Seal an alignment reference and operator-confirmed snapshot, without motion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml
from hardware.piper import SinglePiperArm
from .core import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--records", nargs="*", type=Path, default=[])
    p.add_argument("--config", type=Path, default=Path("configs/dual_piper_example.yaml"))
    args = p.parse_args()
    snapshots = sorted(args.session.glob("snapshot_*/snapshot.json"))
    if not snapshots:
        raise ValueError("Save a live snapshot before archiving")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.session / "reference", args.output / "reference")
    shutil.copytree(snapshots[-1].parent, args.output / "accepted_snapshot")
    shutil.copyfile(args.session / "session.json", args.output / "session.json")
    shutil.copyfile(args.config, args.output / "hardware_config.yaml")
    for name in ("calibration", "hardware"):
        shutil.copytree(name, args.output / "source" / name, ignore=shutil.ignore_patterns("__pycache__"))
    for index,path in enumerate(args.records):
        dest = args.output / "records" / f"{index}_{path.name}"
        dest.parent.mkdir(exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dest)
        else:
            shutil.copyfile(path, dest)
    cfg = yaml.safe_load(args.config.read_text())
    arms, feedback = {}, {}
    try:
        for n in ("left", "right"):
            arms[n] = SinglePiperArm(name=n, can_name=cfg["robot"][n]["can_name"], commands_enabled=False)
            arms[n].connect(read_only=True)
        time.sleep(.5)
        for n,a in arms.items():
            state = a.read_state()
            stamp = a.interface.GetArmJointMsgs().time_stamp
            if not 0 <= time.time() - stamp < .3:
                raise ValueError("Cannot archive stale robot state")
            feedback[n] = {"qpos": state.qpos.tolist(), "eef": state.end_pose.tolist(), "timestamp": stamp}
    finally:
        for a in arms.values():
            a.disconnect()
    write_json(args.output / "robot_feedback.json", feedback)
    write_json(args.output / "manifest.json", {
        "created_at": time.time(), "status": "user_confirmed_visual_alignment_cam_high",
        "not_a_calibration": True, "reference": json.loads((args.output / "reference/reference.json").read_text()),
        "scope": "High-camera visual alignment only; wrist alignment and multi-pose extrinsic calibration remain pending.",
        "feedback_note": "Robot feedback read after the accepted image, not an atomic synchronized capture.",
        "sha256": {str(f.relative_to(args.output)): hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in sorted(args.output.rglob("*")) if f.is_file()}})
    print(args.output.resolve())


if __name__ == "__main__":
    main()
