"""Supervised slow joint positioning. Requires an operator at the emergency stop."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from hardware.piper import SinglePiperArm
from hardware.conversions import joints_rad_to_sdk
from .core import write_json

LOW = np.array([-2.6179, 0, -2.967, -1.745, -1.22, -2.09439])
HIGH = np.array([2.6179, 3.14, 0, 1.745, 1.22, 2.09439])


def checked_state(arm):
    s = arm.read_state()
    stamp = arm.interface.GetArmJointMsgs().time_stamp
    if stamp <= 0 or not 0 <= time.time() - stamp < .3:
        raise ValueError(f"{arm.name}: stale joint feedback")
    if not arm.is_enabled() or not np.isfinite(s.qpos).all():
        raise ValueError(f"{arm.name}: disabled or invalid state")
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", default="configs/dual_piper_example.yaml")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-joint-speed", type=float, default=.15, help="Peak interpolation speed, rad/s (0.02 to 0.2)")
    args = p.parse_args()
    if not .02 <= args.max_joint_speed <= .2:
        raise ValueError("Joint speed must be between 0.02 and 0.2 rad/s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    ref = json.loads((args.reference / "reference.json").read_text())
    target = {n: np.asarray(ref["joint_angles"][n], dtype=float) for n in ("left", "right")}
    for q in target.values():
        if q.shape != (6,) or not np.isfinite(q).all() or np.any(q < LOW) or np.any(q > HIGH):
            raise ValueError("Target outside SDK joint limits")
    args.output.mkdir(parents=True, exist_ok=False)
    arms = {}
    started = False
    report = {"reference": ref, "execute": args.execute, "grippers": "unchanged", "samples": []}
    try:
        for name in target:
            arms[name] = SinglePiperArm(name=name, can_name=cfg["robot"][name]["can_name"], commands_enabled=args.execute)
            arms[name].connect(read_only=True)
        time.sleep(1)
        states = {n: checked_state(a) for n, a in arms.items()}
        start = {n: s.qpos[:6].copy() for n, s in states.items()}
        if any(np.any(q < LOW-.02) or np.any(q > HIGH+.02) for q in start.values()):
            raise ValueError("Start outside joint-limit tolerance")
        duration = max(5., max(np.max(abs(target[n]-start[n])) for n in start) * 1.5 / args.max_joint_speed)
        report.update(start={n:q.tolist() for n,q in start.items()}, duration_s=duration)
        write_json(args.output / "preflight.json", report)
        print(json.dumps({"duration_s": duration, "start": report["start"], "target": {n:q.tolist() for n,q in target.items()}}), flush=True)
        if not args.execute:
            return
        print("MOVEMENT IN 5 SECONDS. Operator must keep clear and have emergency stop ready.", flush=True)
        time.sleep(5)
        for n,a in arms.items():
            if np.max(abs(checked_state(a).qpos[:6] - start[n])) > .03:
                raise ValueError("Arm moved during countdown; refusing changed start")
        started = True
        for a in arms.values():
            a.set_joint_mode(speed_percent=15)
        previous = start
        begin = time.monotonic()
        for step in range(int(duration * 25) + 51):
            u = min(1., step / (duration * 25))
            easing = u*u*(3-2*u)
            states = {n: checked_state(a) for n,a in arms.items()}
            for n,s in states.items():
                if np.max(abs(s.qpos[:6] - previous[n])) > .15:
                    raise ValueError(f"{n}: tracking error exceeds 0.15 rad")
            command = {n: np.clip(start[n] + easing*(target[n]-start[n]), LOW, HIGH) for n in start}
            for n,a in arms.items():
                a.interface.JointCtrl(*[int(x) for x in joints_rad_to_sdk(command[n])])
            previous = command
            if step % 25 == 0:
                report["samples"].append({"time_s": time.monotonic()-begin, "qpos": {n:s.qpos.tolist() for n,s in states.items()}})
                print(f"Moving {min(100,100*u):.0f}%", flush=True)
            time.sleep(max(0, begin + (step+1)/25 - time.monotonic()))
        final = {n: checked_state(a).qpos.tolist() for n,a in arms.items()}
        error = {n: float(np.max(abs(np.asarray(final[n])[:6]-target[n]))) for n in final}
        report.update(final=final, max_joint_error_rad=error)
        if max(error.values()) > .025:
            raise ValueError("Final joint tolerance not reached")
        report["status"] = "reached"
        print("REACHED " + json.dumps(error), flush=True)
    except BaseException as exc:
        report.update(status="aborted", error=str(exc))
        if started:
            for arm in arms.values():
                try:
                    arm.interface.MotionCtrl_1(emergency_stop=0x01)
                except Exception:
                    pass
        raise
    finally:
        write_json(args.output / "motion_report.json", report)
        for arm in arms.values():
            arm.disconnect()


if __name__ == "__main__":
    main()
