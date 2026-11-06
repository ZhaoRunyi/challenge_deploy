"""Restore documented historical gripper openings, without arm joint commands."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from hardware.piper import SinglePiperArm
from hardware.conversions import opening_to_sdk_gripper
from .core import write_json


def targets(reference, scale):
    names = reference["state_names"][0]
    result = {n: float(reference["observation_state_raw"][names.index(n + "_gripper")]) * scale for n in ("left", "right")}
    if not np.isfinite(scale) or scale <= 0 or any(not np.isfinite(v) or not 0 <= v <= .1 for v in result.values()):
        raise ValueError("Invalid gripper opening or scale")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", default="configs/dual_piper_example.yaml")
    p.add_argument("--raw-to-meters", type=float, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    ref = json.loads((args.reference / "reference.json").read_text())
    target = targets(ref, args.raw_to_meters)
    cfg = yaml.safe_load(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    arms = {}
    report = {"target_m": target, "source": ref, "execute": args.execute}
    def read(arm):
        stamp = arm.interface.GetArmGripperMsgs().time_stamp
        if not 0 <= time.time() - stamp < .3 or not arm.is_enabled():
            raise RuntimeError("Stale feedback or disabled arm")
        return arm.read_state()
    try:
        for n in target:
            arms[n] = SinglePiperArm(name=n, can_name=cfg["robot"][n]["can_name"], commands_enabled=args.execute)
            arms[n].connect(read_only=True)
        time.sleep(.5)
        start = {n: float(read(a).qpos[6]) for n,a in arms.items()}
        report["start_m"] = start
        print(json.dumps(report), flush=True)
        if not args.execute:
            return
        print("GRIPPER MOVEMENT IN 5 SECONDS: keep fingers and objects clear.", flush=True)
        time.sleep(5)
        for n,a in arms.items():
            if abs(read(a).qpos[6] - start[n]) > .003:
                raise RuntimeError("Gripper moved during countdown")
        for i in range(101):
            u = min(1., i/75)
            for n,a in arms.items():
                read(a)
                width = max(0., start[n] + u*u*(3-2*u)*(target[n]-start[n]))
                a.interface.GripperCtrl(abs(int(opening_to_sdk_gripper(width))), 500, 0x01, 0)
            time.sleep(.04)
        report["final_m"] = {n: float(read(a).qpos[6]) for n,a in arms.items()}
        report["error_mm"] = {n: abs(report["final_m"][n]-target[n])*1000 for n in target}
        if max(report["error_mm"].values()) > 2:
            raise RuntimeError("Gripper error exceeds 2 mm")
        report["status"] = "reached"
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(args.output / "gripper_report.json", report)
        for a in arms.values():
            a.disconnect()


if __name__ == "__main__":
    main()
