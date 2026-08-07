from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from .metrics import save_rollout_metrics_summary


ExecutionMode = Literal["streaming", "chunk_sync"]


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
    rollout_started_at_s: float = field(default_factory=time.monotonic)
    interrupted: bool = False
    stop_reason: str | None = None

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

    def mark_interrupted(self, reason: str | None = None) -> None:
        self.interrupted = True
        self.stop_reason = reason or "KeyboardInterrupt"

    @staticmethod
    def stats(values: list[float]) -> dict[str, float | None]:
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
            "rollout_wall_seconds": float(time.monotonic() - self.rollout_started_at_s),
            "inference_seconds": self.stats(self.inference_seconds),
            "command_period_seconds": self.stats(self.command_period_seconds),
            "command_seconds": self.stats(self.command_seconds),
            "interrupted": self.interrupted,
            "stop_reason": self.stop_reason,
        }


def save_rollout_metrics(
    metrics: RolloutMetrics,
    *,
    metrics_json_path: str | Path | None = None,
    run_dir: Path | None = None,
    record_stem: str | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    metrics_summary = metrics.summary()
    written_paths = save_rollout_metrics_summary(
        metrics_summary,
        metrics_json_path=metrics_json_path,
        run_dir=run_dir,
        record_stem=record_stem,
    )
    return metrics_summary, written_paths


def resolve_chunk_size(spec: Any, requested_chunk_size: int | None) -> int | None:
    if requested_chunk_size is not None:
        if requested_chunk_size <= 0:
            raise ValueError("--chunk-size must be positive when provided")
        return requested_chunk_size
    if spec.action_horizon is not None and spec.action_horizon > 0:
        return int(spec.action_horizon)
    return None


def resolve_record_steps(rollout_steps: int, record_steps: int | None) -> int:
    if record_steps is None:
        return rollout_steps
    if record_steps < 0:
        raise ValueError("--record-steps must be non-negative")
    return int(record_steps)
