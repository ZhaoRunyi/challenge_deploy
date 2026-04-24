from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.realsense import list_realsense_devices


def main() -> None:
    devices = list_realsense_devices()
    print(
        json.dumps(
            [
                {
                    "name": device.name,
                    "serial": device.serial,
                    "physical_port": device.physical_port,
                }
                for device in devices
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
