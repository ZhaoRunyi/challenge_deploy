from __future__ import annotations

import argparse
from pathlib import Path
import signal
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.observation import build_policy_payload
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.policy import OpenPiPolicyClient
from challenge_deploy.realsense import RealSenseRig
from challenge_deploy.runtime import DualPiperObservationSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROS-free Challenge Deploy Agilex synchronous inference.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_publish_step", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.host:
        set_by_dotted_path(config, "policy.host", args.host)
    if args.port is not None:
        set_by_dotted_path(config, "policy.port", args.port)
    if args.prompt:
        set_by_dotted_path(config, "policy.prompt", args.prompt)
    commands_enabled = True
    shutdown = {"value": False}

    def _handle(_signum=None, _frame=None) -> None:
        shutdown["value"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=commands_enabled,
        name="agilex_inference_sync",
    )
    cameras = RealSenseRig(
        config["cameras"]["serials"],
        width=int(config["cameras"]["width"]),
        height=int(config["cameras"]["height"]),
        fps=int(config["cameras"]["fps"]),
        warmup_frames=int(config["cameras"]["warmup_frames"]),
    )
    source = DualPiperObservationSource(robot=robot, cameras=cameras)

    robot.connect(read_only=not commands_enabled)
    try:
        cameras.start()
        if not source.wait_until_ready(timeout_s=15.0):
            raise RuntimeError("Timed out waiting for robot/camera data")
        policy = OpenPiPolicyClient(
            host=config["policy"]["host"],
            port=int(config["policy"]["port"]),
        )
        print("Server metadata:", policy.get_server_metadata())

        for step in range(args.max_publish_step):
            if shutdown["value"]:
                break
            snapshot = source.capture_snapshot()
            payload = build_policy_payload(snapshot, str(config["policy"]["prompt"]))
            actions = policy.infer(payload)["actions"]
            if actions is None or len(actions) == 0:
                time.sleep(1.0 / float(config["runtime"]["publish_rate"]))
                continue
            action = np.asarray(actions[0], dtype=np.float64)
            robot.set_joint_positions(action, speed_percent=100)
            time.sleep(1.0 / float(config["runtime"]["publish_rate"]))
    finally:
        cameras.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
