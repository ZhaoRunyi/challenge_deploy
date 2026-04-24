from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.can_tools import activate_can_interface


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Activate or rename a CAN interface for Piper usage.")
    parser.add_argument("desired_name", type=str, help="Interface name to end up with, e.g. can0 or can_left_slave")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--usb-bus-info", type=str, default=None, help="Optional ethtool bus-info selector")
    parser.add_argument("--sudo", action="store_true", help="Prefix ip commands with sudo")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    commands = activate_can_interface(
        desired_name=args.desired_name,
        bitrate=args.bitrate,
        usb_bus_info=args.usb_bus_info,
        use_sudo=args.sudo,
    )
    for command in commands:
        print(" ".join(command))


if __name__ == "__main__":
    main()
