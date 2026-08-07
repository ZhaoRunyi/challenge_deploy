from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            str(key): _json_value(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, BaseException):
        return repr(value)
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, Enum) else key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "value"):
        return _json_value(value.value)
    return value


class RuntimeEventLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, event: str, *, durable: bool = False, **fields: Any) -> dict[str, Any]:
        payload = {
            "event": str(event),
            "wall_time_s": time.time(),
            "monotonic_ns": time.monotonic_ns(),
            **{key: _json_value(value) for key, value in fields.items()},
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.write("\n")
                handle.flush()
                if durable:
                    os.fsync(handle.fileno())
        return payload
