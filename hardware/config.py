from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable, Mapping

import yaml

from .constants import (
    CAN_TOPOLOGIES,
    DEFAULT_CAMERA_SERIALS,
    DEFAULT_CAN_NAMES,
    DEFAULT_PROMPT,
    PIPER_ARM_IDS,
    UNCONFIRMED_CAN_SERIAL,
)


LEGACY_ROBOT_ARM_IDS = frozenset(("left", "right"))
CAN_NETWORK_INTERFACE_TYPE = "280"
PROVISIONAL_SERIAL_MARKERS = (
    "REPLACE_",
    "PROVISIONAL",
    "UNKNOWN",
    "UNCONFIRMED",
    "TODO",
    "TBD",
)


@dataclass(frozen=True, slots=True)
class CanInterfaceIdentity:
    interface: str
    serial: str | None
    id_serial: str | None = None
    id_path: str | None = None
    vendor_id: str | None = None
    model_id: str | None = None
    link_checked: bool = False
    is_up: bool | None = None
    bitrate: int | None = None


def default_config() -> dict[str, Any]:
    robot = {
        arm_id: {
            "can_name": DEFAULT_CAN_NAMES[arm_id],
            "usb_serial": UNCONFIRMED_CAN_SERIAL,
        }
        for arm_id in PIPER_ARM_IDS
    }
    robot["slave_left"]["gripper_effort"] = 1000
    robot["slave_right"]["gripper_effort"] = 1000
    return {
        "can_topology": "isolated",
        "robot": robot,
        "cameras": {
            "enabled": True,
            "width": 640,
            "height": 480,
            "fps": 30,
            "warmup_frames": 30,
            "serials": {
                "cam_high": DEFAULT_CAMERA_SERIALS["cam_high"],
                "cam_right_wrist": DEFAULT_CAMERA_SERIALS["cam_right_wrist"],
                "cam_left_wrist": DEFAULT_CAMERA_SERIALS["cam_left_wrist"],
            },
        },
        "policy": {
            "host": "127.0.0.1",
            "port": 8000,
            "prompt": DEFAULT_PROMPT,
            "inference_rate": 3.0,
            "chunk_size": 50,
            "latency_k": 8,
            "min_smooth_steps": 8,
            "buffer_max_chunks": 10,
        },
        "dataset": {
            "dataset_dir": "./artifacts/hdf5_data",
            "dataset_name": "dummy_task",
        },
    }


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = default_config()
    if path is None:
        validate_config(config, required_arm_ids=())
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    if "can_topology" not in loaded:
        raise ValueError(
            f"Config must explicitly set can_topology to one of {CAN_TOPOLOGIES}: {config_path}"
        )
    loaded_robot = _mapping(loaded.get("robot"), "robot")
    missing_arm_ids = [arm_id for arm_id in PIPER_ARM_IDS if arm_id not in loaded_robot]
    if missing_arm_ids:
        raise ValueError(
            "Config files must explicitly contain all canonical Piper arm entries; "
            f"missing: {missing_arm_ids}"
        )
    merged = merge_dicts(config, loaded)
    validate_config(merged, required_arm_ids=())
    return merged


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required_arm_ids(values: Iterable[str]) -> tuple[str, ...]:
    arm_ids = tuple(str(value) for value in values)
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError(f"required_arm_ids contains duplicates: {arm_ids}")
    invalid = sorted(set(arm_ids) - set(PIPER_ARM_IDS))
    if invalid:
        raise ValueError(f"Unknown Piper arm ids {invalid}; expected members of {PIPER_ARM_IDS}")
    return arm_ids


def is_provisional_can_serial(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    serial = value.strip()
    if not serial:
        return True
    upper = serial.upper()
    return any(marker in upper for marker in PROVISIONAL_SERIAL_MARKERS)


def validate_config(
    config: Mapping[str, Any],
    required_arm_ids: Iterable[str] = (),
) -> Mapping[str, Any]:
    """Validate topology and the physical arms that this process will construct.

    Shared CAN deployments intentionally receive no serial, uniqueness, role, or
    frame validation here.  They are an already deployed topology where a master
    and slave may deliberately use the same SocketCAN interface.
    """

    root = _mapping(config, "config")
    topology = root.get("can_topology")
    if topology not in CAN_TOPOLOGIES:
        raise ValueError(f"can_topology must be one of {CAN_TOPOLOGIES}, got {topology!r}")

    robot = _mapping(root.get("robot"), "robot")
    legacy_keys = sorted(LEGACY_ROBOT_ARM_IDS.intersection(robot))
    if legacy_keys:
        raise ValueError(
            "Legacy robot.left/robot.right keys are ambiguous and no longer accepted; "
            "use robot.slave_left/robot.slave_right explicitly "
            f"(found {legacy_keys})"
        )

    missing = [arm_id for arm_id in PIPER_ARM_IDS if arm_id not in robot]
    if missing:
        raise ValueError(f"robot is missing canonical Piper arm entries: {missing}")
    unknown_arm_ids = sorted(set(robot) - set(PIPER_ARM_IDS))
    if unknown_arm_ids:
        raise ValueError(
            "robot accepts only the four canonical master/slave arm entries; "
            f"unknown keys: {unknown_arm_ids}"
        )

    for arm_id in PIPER_ARM_IDS:
        arm = _mapping(robot[arm_id], f"robot.{arm_id}")
        can_name = arm.get("can_name")
        if not isinstance(can_name, str) or not can_name.strip():
            raise ValueError(f"robot.{arm_id}.can_name must be a non-empty string")

    for arm_id in ("slave_left", "slave_right"):
        effort = robot[arm_id].get("gripper_effort", 1000)
        if isinstance(effort, bool) or not isinstance(effort, int):
            raise ValueError(f"robot.{arm_id}.gripper_effort must be an integer")

    required = _required_arm_ids(required_arm_ids)
    if topology == "shared":
        return config

    required_can_names: dict[str, str] = {}
    required_serials: dict[str, str] = {}
    for arm_id in required:
        arm = _mapping(robot[arm_id], f"robot.{arm_id}")
        can_name = str(arm["can_name"]).strip()
        serial = arm.get("usb_serial")
        if is_provisional_can_serial(serial):
            raise ValueError(
                f"robot.{arm_id}.usb_serial is not confirmed; run the CAN mapping toolkit "
                "and replace the provisional value before constructing this arm"
            )
        serial_text = str(serial).strip()
        if can_name in required_can_names:
            raise ValueError(
                f"Isolated constructed arms {required_can_names[can_name]!r} and {arm_id!r} "
                f"share can_name {can_name!r}"
            )
        if serial_text in required_serials:
            raise ValueError(
                f"Isolated constructed arms {required_serials[serial_text]!r} and {arm_id!r} "
                f"share usb_serial {serial_text!r}"
            )
        required_can_names[can_name] = arm_id
        required_serials[serial_text] = arm_id
    return config


def validate_can_interface_serials(
    config: Mapping[str, Any],
    required_arm_ids: Iterable[str],
    discovered: Mapping[str, str | CanInterfaceIdentity],
) -> None:
    """Validate confirmed isolated arm serials against an injected discovery map.

    This function is pure.  Callers decide which arms they will construct and
    pass only those ids.  Shared topology deliberately bypasses serial checks.
    """

    required = _required_arm_ids(required_arm_ids)
    validate_config(config, required)
    if config["can_topology"] == "shared":
        return

    robot = _mapping(config["robot"], "robot")
    for arm_id in required:
        arm = _mapping(robot[arm_id], f"robot.{arm_id}")
        can_name = str(arm["can_name"]).strip()
        expected = str(arm["usb_serial"]).strip()
        identity = discovered.get(can_name)
        if identity is None:
            raise ValueError(
                f"Constructed arm {arm_id!r} expects SocketCAN interface {can_name!r}, "
                "but it was not discovered"
            )
        actual = identity.serial if isinstance(identity, CanInterfaceIdentity) else str(identity)
        if actual != expected:
            raise ValueError(
                f"CAN serial mismatch for {arm_id!r} on {can_name!r}: "
                f"expected {expected!r}, discovered {actual!r}"
            )
        if isinstance(identity, CanInterfaceIdentity) and identity.link_checked:
            if identity.is_up is not True:
                raise ValueError(
                    f"Constructed arm {arm_id!r} SocketCAN interface {can_name!r} "
                    "is not administratively up"
                )
            if identity.bitrate != 1_000_000:
                raise ValueError(
                    f"Constructed arm {arm_id!r} SocketCAN interface {can_name!r} "
                    f"must use 1000000 bit/s, discovered {identity.bitrate!r}"
                )


def parse_udev_properties(text: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        properties[key] = value
    return properties


def query_udev_properties(
    interface_path: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    try:
        result = run(
            ["udevadm", "info", "--query=property", f"--path={interface_path}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {}
    if result.returncode != 0:
        return {}
    return parse_udev_properties(result.stdout)


def query_can_link_details(
    interface_name: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[bool | None, int | None]:
    """Read administrative state and CAN bitrate without opening the bus."""

    try:
        result = run(
            ["ip", "-details", "-json", "link", "show", "dev", interface_name],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None, None
    if result.returncode != 0:
        return None, None
    try:
        records = json.loads(result.stdout)
        record = records[0]
        flags = record.get("flags", ())
        link_info = record.get("linkinfo", {})
        info_data = link_info.get("info_data", {})
        bit_timing = info_data.get("bittiming", {})
        bitrate = (
            bit_timing.get("bitrate")
            if isinstance(bit_timing, Mapping)
            else None
        )
        if bitrate is None:
            bitrate = info_data.get("bitrate")
        return "UP" in flags, int(bitrate) if bitrate is not None else None
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, None


def _parent_file(start: Path, filename: str) -> str | None:
    current = start.resolve()
    for directory in (current, *current.parents):
        candidate = directory / filename
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            continue
        if value:
            return value
    return None


def discover_can_interfaces(
    *,
    sys_class_net: str | Path = "/sys/class/net",
    property_reader: Callable[[Path], Mapping[str, str]] = query_udev_properties,
    link_reader: Callable[[str], tuple[bool | None, int | None]] = query_can_link_details,
) -> dict[str, CanInterfaceIdentity]:
    """Discover CAN identities without opening or transmitting on a CAN bus."""

    root = Path(sys_class_net)
    identities: dict[str, CanInterfaceIdentity] = {}
    if not root.exists():
        return identities
    for interface_path in sorted(root.iterdir(), key=lambda path: path.name):
        try:
            network_type = (interface_path / "type").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            continue
        if network_type != CAN_NETWORK_INTERFACE_TYPE:
            continue
        properties = dict(property_reader(interface_path))
        is_up, bitrate = link_reader(interface_path.name)
        device_path = interface_path / "device"
        serial = properties.get("ID_SERIAL_SHORT") or _parent_file(device_path, "serial")
        identities[interface_path.name] = CanInterfaceIdentity(
            interface=interface_path.name,
            serial=serial,
            id_serial=properties.get("ID_SERIAL"),
            id_path=properties.get("ID_PATH"),
            vendor_id=properties.get("ID_VENDOR_ID"),
            model_id=properties.get("ID_MODEL_ID"),
            link_checked=True,
            is_up=is_up,
            bitrate=bitrate,
        )
    return identities

def set_by_dotted_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value
