from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from challenge_deploy.can_tools import list_can_ports


def main() -> None:
    ports = list_can_ports()
    print(
        json.dumps(
            [
                {
                    "name": port.name,
                    "state": port.state,
                    "bus_info": port.bus_info,
                    "bitrate": port.bitrate,
                }
                for port in ports
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
