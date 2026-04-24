from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Literal

import numpy as np

from .buffer import StreamActionBuffer
from .openpi_client import OpenPiPiperClient, PiperPolicySpec, build_configured_piper_state


ExecutionMode = Literal["streaming", "chunk_sync"]
ChunkLogger = Callable[[int, int, int, np.ndarray], None]


@dataclass
class RolloutMetrics:
    execution_mode: ExecutionMode
    executed_steps: int = 0
    inferred_chunks: int = 0
    empty_action_polls: int = 0
    inference_errors: int = 0
    last_inference_error: str | None = None
    inference_seconds: list[float] = field(default_factory=list)
    command_period_seconds: list[float] = field(default_factory=list)
    command_seconds: list[float] = field(default_factory=list)

    def record_inference(self, seconds: float) -> None:
        self.inferred_chunks += 1
        self.inference_seconds.append(float(seconds))

    def record_inference_error(self, exc: BaseException) -> None:
        self.inference_errors += 1
        self.last_inference_error = repr(exc)

    def record_command(self, *, period_seconds: float | None, command_seconds: float) -> None:
        self.executed_steps += 1
        if period_seconds is not None:
            self.command_period_seconds.append(float(period_seconds))
        self.command_seconds.append(float(command_seconds))

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "p50": None, "p95": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "executed_steps": self.executed_steps,
            "inferred_chunks": self.inferred_chunks,
            "empty_action_polls": self.empty_action_polls,
            "inference_errors": self.inference_errors,
            "last_inference_error": self.last_inference_error,
            "inference_seconds": self._stats(self.inference_seconds),
            "command_period_seconds": self._stats(self.command_period_seconds),
            "command_seconds": self._stats(self.command_seconds),
        }


def action_sequence(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        return actions.reshape(1, -1)
    if actions.ndim == 2:
        return actions
    raise ValueError(f"Expected action vector or action chunk, got shape {actions.shape}")


def resolve_chunk_size(spec: PiperPolicySpec, requested_chunk_size: int | None) -> int | None:
    if requested_chunk_size is not None:
        if requested_chunk_size <= 0:
            raise ValueError("--chunk-size must be positive when provided")
        return requested_chunk_size
    if spec.action_horizon is not None and spec.action_horizon > 0:
        return int(spec.action_horizon)
    return None


def trim_chunk(actions: np.ndarray, chunk_size: int | None) -> np.ndarray:
    actions = action_sequence(actions)
    if chunk_size is None:
        return actions
    return actions[: min(chunk_size, len(actions))]


def sleep_until_next_action(action_start_s: float, fps: float) -> None:
    if fps <= 0.0:
        return
    action_period_s = 1.0 / fps
    elapsed_s = time.monotonic() - action_start_s
    remaining_s = action_period_s - elapsed_s
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def run_chunk_sync_rollout(
    *,
    client: OpenPiPiperClient,
    source: Any,
    robot: Any,
    spec: PiperPolicySpec,
    prompt: str,
    rollout_steps: int,
    chunk_size: int | None,
    fps: float,
    recorder: Any | None = None,
    log_chunk: ChunkLogger | None = None,
) -> RolloutMetrics:
    metrics = RolloutMetrics(execution_mode="chunk_sync")
    chunk_index = 0
    last_command_start_s: float | None = None

    while rollout_steps == 0 or metrics.executed_steps < rollout_steps:
        inference_start_s = time.monotonic()
        chunk_snapshot = source.capture_snapshot()
        actions = trim_chunk(client.infer_actions(chunk_snapshot, prompt=prompt), chunk_size)
        metrics.record_inference(time.monotonic() - inference_start_s)

        requested_actions = len(actions)
        if rollout_steps > 0:
            requested_actions = min(requested_actions, rollout_steps - metrics.executed_steps)
        if requested_actions <= 0:
            break

        if log_chunk is not None:
            log_chunk(chunk_index, requested_actions, metrics.executed_steps, actions[0])

        for action_index, action in enumerate(actions[:requested_actions]):
            action_start_s = time.monotonic()
            period_seconds = None if last_command_start_s is None else action_start_s - last_command_start_s
            last_command_start_s = action_start_s
            frame_snapshot = chunk_snapshot if action_index == 0 else source.capture_snapshot()
            if recorder is not None:
                recorder.record(
                    images=frame_snapshot.images,
                    action=action,
                    state=build_configured_piper_state(frame_snapshot, spec),
                    timestamp_s=frame_snapshot.timestamp_s,
                )
            command_start_s = time.monotonic()
            client.command_action(robot, action)
            metrics.record_command(
                period_seconds=period_seconds,
                command_seconds=time.monotonic() - command_start_s,
            )
            if rollout_steps > 0 and metrics.executed_steps >= rollout_steps:
                break
            sleep_until_next_action(action_start_s, fps)
        chunk_index += 1

    return metrics


def run_temporal_smoothing_rollout(
    *,
    client: OpenPiPiperClient,
    source: Any,
    robot: Any,
    spec: PiperPolicySpec,
    prompt: str,
    rollout_steps: int,
    chunk_size: int | None,
    fps: float,
    inference_rate: float,
    latency_k: int,
    min_smooth_steps: int,
    buffer_max_chunks: int,
    recorder: Any | None = None,
    log_chunk: ChunkLogger | None = None,
    first_action_timeout_s: float = 15.0,
) -> RolloutMetrics:
    metrics = RolloutMetrics(execution_mode="streaming")
    buffer = StreamActionBuffer(
        max_chunks=buffer_max_chunks,
        state_dim=spec.action_dim,
        smooth_method="temporal",
    )
    capture_lock = threading.Lock()
    stop_event = threading.Event()

    def capture_snapshot() -> Any:
        with capture_lock:
            return source.capture_snapshot()

    def inference_loop() -> None:
        chunk_index = 0
        period_s = 1.0 / inference_rate if inference_rate > 0.0 else 0.0
        while not stop_event.is_set():
            inference_start_s = time.monotonic()
            try:
                snapshot = capture_snapshot()
                actions = trim_chunk(client.infer_actions(snapshot, prompt=prompt), chunk_size)
                if len(actions) > 0:
                    buffer.integrate_new_chunk(actions, max_k=latency_k, min_m=min_smooth_steps)
                    metrics.record_inference(time.monotonic() - inference_start_s)
                    if log_chunk is not None:
                        log_chunk(chunk_index, len(actions), metrics.executed_steps, actions[0])
                    chunk_index += 1
            except Exception as exc:
                metrics.record_inference_error(exc)

            elapsed_s = time.monotonic() - inference_start_s
            sleep_s = max(0.0, period_s - elapsed_s)
            if sleep_s > 0.0:
                stop_event.wait(sleep_s)

    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    inference_thread.start()
    wait_start_s = time.monotonic()
    last_command_start_s: float | None = None

    try:
        while rollout_steps == 0 or metrics.executed_steps < rollout_steps:
            action_start_s = time.monotonic()
            action = buffer.pop_next_action()
            if action is None:
                metrics.empty_action_polls += 1
                if (
                    metrics.executed_steps == 0
                    and time.monotonic() - wait_start_s > first_action_timeout_s
                    and metrics.inference_errors > 0
                ):
                    raise RuntimeError(
                        "Timed out waiting for the first smoothed action; "
                        f"last inference error: {metrics.last_inference_error}"
                    )
                time.sleep(0.001 if fps <= 0.0 else min(0.001, 1.0 / fps))
                continue

            period_seconds = None if last_command_start_s is None else action_start_s - last_command_start_s
            last_command_start_s = action_start_s
            frame_snapshot = capture_snapshot()
            if recorder is not None:
                recorder.record(
                    images=frame_snapshot.images,
                    action=action,
                    state=build_configured_piper_state(frame_snapshot, spec),
                    timestamp_s=frame_snapshot.timestamp_s,
                )
            command_start_s = time.monotonic()
            client.command_action(robot, action)
            metrics.record_command(
                period_seconds=period_seconds,
                command_seconds=time.monotonic() - command_start_s,
            )
            if rollout_steps > 0 and metrics.executed_steps >= rollout_steps:
                break
            sleep_until_next_action(action_start_s, fps)
    finally:
        stop_event.set()
        inference_thread.join(timeout=1.0)

    return metrics
