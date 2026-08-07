from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class TeleopEpisode:
    episode_index: int
    episode_path: Path
    frames: tuple[Any, ...]
    trace: dict[str, Any]
    partial: bool = False
    stop_reason: str | None = None

    def as_partial(self, stop_reason: str) -> "TeleopEpisode":
        return replace(self, partial=True, stop_reason=stop_reason)


class TeleopWorker:
    def __init__(
        self,
        *,
        source: Any,
        ready_timeout_s: float,
        wait_for_source_ready: bool = True,
        start_callbacks: Sequence[Callable[[], None]] = (),
        stop_callbacks: Sequence[tuple[str, Callable[[], None]]] = (),
        emergency_callbacks: Sequence[tuple[str, Callable[[], None]]] = (),
    ) -> None:
        self.source = source
        self.ready_timeout_s = ready_timeout_s
        self.wait_for_source_ready = wait_for_source_ready
        self.start_callbacks = tuple(start_callbacks)
        self.stop_callbacks = tuple(stop_callbacks)
        self.emergency_callbacks = tuple(emergency_callbacks)
        self.emergency_stop_called = False
        self.emergency_stop_failures: tuple[tuple[str, BaseException], ...] = ()

    def start(self) -> None:
        try:
            for callback in self.start_callbacks:
                callback()
            self.source.start()
            if not self.wait_for_source_ready:
                return
            if self.source.wait_until_ready(timeout_s=self.ready_timeout_s):
                return
            details = []
            last_error = getattr(self.source, "last_error", None)
            if last_error is not None:
                details.append(str(last_error))
            last_sync_failure = getattr(self.source, "last_sync_failure", None)
            if last_sync_failure:
                details.append(str(last_sync_failure))
            detail = f": {'; '.join(details)}" if details else ""
            raise RuntimeError(f"Timed out waiting for teleop source readiness{detail}")
        except BaseException:
            self.emergency_stop()
            raise

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
        return self.episode_from_frames(
            episode_index=episode_index,
            episode_path=episode_path,
            frames=frames,
        )

    def episode_from_frames(
        self,
        *,
        episode_index: int,
        episode_path: Path,
        frames: Sequence[Any],
        partial: bool = False,
        stop_reason: str | None = None,
    ) -> TeleopEpisode:
        return TeleopEpisode(
            episode_index=episode_index,
            episode_path=episode_path,
            frames=tuple(frames),
            trace=self.source.alignment_trace(),
            partial=partial,
            stop_reason=stop_reason,
        )

    def emergency_stop(self) -> tuple[tuple[str, BaseException], ...]:
        """Run the ordered fault-stop sequence once, attempting every step."""

        if self.emergency_stop_called:
            return self.emergency_stop_failures
        self.emergency_stop_called = True
        failures: list[tuple[str, BaseException]] = []
        for name, callback in self.emergency_callbacks:
            try:
                callback()
            except BaseException as exc:
                failures.append((name, exc))
                try:
                    print(f"Failed to run emergency step {name}: {exc}", flush=True)
                except Exception:
                    pass
        self.emergency_stop_failures = tuple(failures)
        return self.emergency_stop_failures

    def stop(self) -> None:
        cleanup_steps = (("teleop source", self.source.stop),) + self.stop_callbacks
        for name, callback in cleanup_steps:
            try:
                callback()
            except Exception as exc:
                print(f"Failed to stop {name} cleanly: {exc}", flush=True)
