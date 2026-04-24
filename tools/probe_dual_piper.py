from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.piper import DualPiperSystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read dual-Piper state without sending motion commands.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", type=str, default=None)
    parser.add_argument("--right-can", type=str, default=None)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--prefer-joint-ctrl", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.left_can:
        set_by_dotted_path(config, "robot.left.can_name", args.left_can)
    if args.right_can:
        set_by_dotted_path(config, "robot.right.can_name", args.right_can)

    robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=False,
        prefer_joint_ctrl=args.prefer_joint_ctrl,
        name="probe_dual_piper",
    )
    robot.connect(read_only=True)
    try:
        result = robot.probe(samples=args.samples, interval_s=args.interval)
        result["final_state"] = robot.read_state(prefer_joint_ctrl=args.prefer_joint_ctrl).to_dict()
        print(json.dumps(result, indent=2))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
