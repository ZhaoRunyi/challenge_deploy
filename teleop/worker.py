from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class TeleopEpisode:
    episode_index: int
    episode_path: Path
    frames: tuple[Any, ...]
    trace: dict[str, Any]


class TeleopWorker:
    def __init__(
        self,
        *,
        source: Any,
        ready_timeout_s: float,
        start_callbacks: Sequence[Callable[[], None]] = (),
        stop_callbacks: Sequence[tuple[str, Callable[[], None]]] = (),
    ) -> None:
        self.source = source
        self.ready_timeout_s = ready_timeout_s
        self.start_callbacks = tuple(start_callbacks)
        self.stop_callbacks = tuple(stop_callbacks)

    def start(self) -> None:
        for callback in self.start_callbacks:
            callback()
        self.source.start()
        if self.source.wait_until_ready(timeout_s=self.ready_timeout_s):
            return
        detail = f": {self.source.last_error}" if getattr(self.source, "last_error", None) is not None else ""
        raise RuntimeError(f"Timed out waiting for teleop source readiness{detail}")

    def collect_episode(
        self,
        *,
        episode_index: int,
        episode_path: Path,
        collect_fn: Callable[..., Sequence[Any]],
        collect_kwargs: dict[str, Any],
    ) -> TeleopEpisode:
        self.source.reset_trace()
        frames = collect_fn(source=self.source, **collect_kwargs)
        return TeleopEpisode(
            episode_index=episode_index,
            episode_path=episode_path,
            frames=tuple(frames),
            trace=self.source.alignment_trace(),
        )

    def stop(self) -> None:
        cleanup_steps = (("teleop source", self.source.stop),) + self.stop_callbacks
        for name, callback in cleanup_steps:
            try:
                callback()
            except Exception as exc:
                print(f"Failed to stop {name} cleanly: {exc}", flush=True)
