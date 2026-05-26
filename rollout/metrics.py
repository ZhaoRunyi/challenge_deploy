from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_rollout_metrics_summary(
    metrics_summary: dict[str, Any],
    *,
    metrics_json_path: str | Path | None = None,
    run_dir: Path | None = None,
    record_stem: str | None = None,
) -> list[Path]:
    target_paths: list[Path] = []
    if metrics_json_path is not None:
        target_paths.append(Path(metrics_json_path))
    if run_dir is not None and record_stem:
        target_paths.append(run_dir / f"{record_stem}_rollout_metrics.json")

    written_paths: list[Path] = []
    seen_keys: set[str] = set()
    payload = json.dumps(metrics_summary, indent=2)
    for path in target_paths:
        dedupe_key = str(path.expanduser().resolve())
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        written_paths.append(path)
    return written_paths
