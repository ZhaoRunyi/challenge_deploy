from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
from typing import Iterable


@dataclass(slots=True)
class CanPortInfo:
    name: str
    state: str
    bus_info: str | None = None
    bitrate: int | None = None


def _run(args: Iterable[str], check: bool = True) -> str:
    completed = subprocess.run(
        list(args),
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def list_can_ports() -> list[CanPortInfo]:
    try:
        output = _run(["ip", "-br", "link", "show", "type", "can"], check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("'ip' command not found") from exc

    ports: list[CanPortInfo] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, state = parts[0], parts[1]
        bus_info = None
        bitrate = None
        try:
            eth = _run(["ethtool", "-i", name], check=False)
            match = re.search(r"bus-info:\s*(\S+)", eth)
            if match:
                bus_info = match.group(1)
        except FileNotFoundError:
            pass

        try:
            details = _run(["ip", "-details", "link", "show", name], check=False)
            match = re.search(r"bitrate\s+(\d+)", details)
            if match:
                bitrate = int(match.group(1))
        except FileNotFoundError:
            pass

        ports.append(CanPortInfo(name=name, state=state, bus_info=bus_info, bitrate=bitrate))
    return ports


def resolve_can_name(preferred_name: str, usb_bus_info: str | None) -> str:
    ports = list_can_ports()
    if usb_bus_info:
        for port in ports:
            if port.bus_info == usb_bus_info:
                return port.name
        raise ValueError(f"No CAN interface found for USB bus-info {usb_bus_info!r}")
    for port in ports:
        if port.name == preferred_name:
            return port.name
    raise ValueError(f"CAN interface {preferred_name!r} not found")


def activate_can_interface(
    desired_name: str,
    bitrate: int = 1_000_000,
    usb_bus_info: str | None = None,
    use_sudo: bool = False,
) -> list[list[str]]:
    current_name = resolve_can_name(desired_name, usb_bus_info)
    prefix = ["sudo"] if use_sudo else []
    commands: list[list[str]] = []

    if current_name != desired_name:
        commands.append(prefix + ["ip", "link", "set", current_name, "down"])
        commands.append(prefix + ["ip", "link", "set", current_name, "name", desired_name])
        current_name = desired_name
    else:
        commands.append(prefix + ["ip", "link", "set", current_name, "down"])

    commands.append(prefix + ["ip", "link", "set", current_name, "type", "can", "bitrate", str(bitrate)])
    commands.append(prefix + ["ip", "link", "set", current_name, "up"])

    for command in commands:
        subprocess.run(command, check=True)
    return commands
