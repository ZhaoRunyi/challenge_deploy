from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from hardware.config import (
    is_provisional_can_serial,
    validate_config,
)
from hardware.constants import DUAL_PIPER_INIT_JOINTS, PIPER_ARM_IDS


def validated_initial_joints(values: Any) -> np.ndarray:
    joints = np.asarray(
        DUAL_PIPER_INIT_JOINTS if values is None else values,
        dtype=np.float64,
    )
    if joints.shape != (14,):
        raise ValueError("--init-joints expects exactly 14 values")
    if not np.all(np.isfinite(joints)):
        raise ValueError("--init-joints must contain only finite values")
    return joints.copy()


def build_dry_run_hardware_plan(
    args: Any,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate topology wiring and describe hardware that a real run would open."""

    validate_config(runtime_config, required_arm_ids=())
    topology = str(runtime_config["can_topology"])
    intervention = bool(args.intervention)
    configured_intervention = bool(
        runtime_config.get("runtime", {}).get("intervention_enabled", False)
    )
    if configured_intervention != intervention:
        raise ValueError(
            "runner wiring mismatch: runtime.intervention_enabled does not match --intervention"
        )
    if intervention and topology != "isolated":
        raise ValueError("--intervention requires can_topology='isolated'")

    construction_profile = "four-arm" if intervention else "slave-only"
    constructed_arm_ids = (
        tuple(PIPER_ARM_IDS)
        if construction_profile == "four-arm"
        else ("slave_left", "slave_right")
    )
    unconfirmed_serials: list[str] = []
    if topology == "isolated":
        unconfirmed_serials = [
            arm_id
            for arm_id in constructed_arm_ids
            if is_provisional_can_serial(
                runtime_config["robot"][arm_id].get("usb_serial")
            )
        ]

    return {
        "config_path": str(args.config),
        "can_topology": topology,
        "construction_profile": construction_profile,
        "constructed_arm_ids": list(constructed_arm_ids),
        "intervention_enabled": intervention,
        "cameras_enabled_for_real_run": bool(runtime_config["cameras"]["enabled"]),
        "unconfirmed_constructed_arm_serials": unconfirmed_serials,
        "operations": {
            "can_construction": "skipped",
            "can_connection": "skipped",
            "realsense_construction": "skipped",
            "realsense_connection": "skipped",
        },
    }


def validated_dry_run_parameters(
    args: Any,
    runtime_config: dict[str, Any],
    policy_schema: dict[str, Any],
) -> dict[str, Any]:
    if args.rollout_steps < 0:
        raise ValueError("--rollout-steps must be non-negative")
    fps = float(args.fps)
    if not math.isfinite(fps) or fps < 0.0:
        raise ValueError("--fps must be finite and non-negative")

    chunk_size = getattr(args, "chunk_size", None)
    if chunk_size is None:
        chunk_size = policy_schema["action_horizon"]
    elif chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer when provided")

    policy_config = runtime_config.get("policy", {})
    inference_rate = getattr(args, "inference_rate", None)
    if inference_rate is None:
        inference_rate = policy_config.get("inference_rate")
    inference_rate = float(inference_rate)
    if not math.isfinite(inference_rate) or inference_rate < 0.0:
        raise ValueError("--inference-rate must resolve to a finite non-negative value")

    latency_k = getattr(args, "latency_k", None)
    if latency_k is None:
        latency_k = policy_config.get("latency_k")
    if latency_k < 0:
        raise ValueError("--latency-k must resolve to a non-negative integer")

    min_smooth_steps = getattr(args, "min_smooth_steps", None)
    if min_smooth_steps is None:
        min_smooth_steps = policy_config.get("min_smooth_steps")
    if min_smooth_steps <= 0:
        raise ValueError("--min-smooth-steps must resolve to a positive integer")

    buffer_max_chunks = getattr(args, "buffer_max_chunks", None)
    if buffer_max_chunks is None:
        buffer_max_chunks = policy_config.get("buffer_max_chunks")
    if buffer_max_chunks <= 0:
        raise ValueError("--buffer-max-chunks must resolve to a positive integer")

    inference_timeout = float(getattr(args, "inference_timeout", 0.0))
    if not math.isfinite(inference_timeout) or inference_timeout <= 0.0:
        raise ValueError("--inference-timeout must be finite and positive")
    ready_timeout = float(getattr(args, "ready_timeout", 0.0))
    if not math.isfinite(ready_timeout) or ready_timeout <= 0.0:
        raise ValueError("--ready-timeout must be finite and positive")

    for name in ("joint_speed_percent", "ee_speed_percent"):
        value = getattr(args, name, None)
        if not 0 <= value <= 100:
            option = name.replace("_", "-")
            raise ValueError(f"--{option} must be an integer in [0, 100]")

    host = getattr(args, "host", None)
    if not host.strip():
        raise ValueError("--host must be a non-empty string")
    port = getattr(args, "port", None)
    if not 1 <= port <= 65535:
        raise ValueError("--port must be an integer in [1, 65535]")

    window = args.window
    if window < 0:
        raise ValueError("--window must be a non-negative display index")
    validated_initial_joints(getattr(args, "init_joints", None))

    return {
        "rollout_steps": int(args.rollout_steps),
        "chunk_size": chunk_size,
        "fps": fps,
        "inference_rate": inference_rate,
        "latency_k": latency_k,
        "min_smooth_steps": min_smooth_steps,
        "buffer_max_chunks": buffer_max_chunks,
        "inference_timeout_s": inference_timeout,
    }


def run_deferred_policy_dry_run_plan(
    *,
    args: Any,
    runner_name: str,
    policy_transport_name: str,
    runtime_config: dict[str, Any],
    configuration_kind: str,
) -> None:
    """Validate local wiring without importing an external policy config."""

    hardware_plan = build_dry_run_hardware_plan(args, runtime_config)
    parameters = validated_dry_run_parameters(
        args,
        runtime_config,
        {"action_horizon": None},
    )
    parameters["chunk_size_source"] = (
        "explicit --chunk-size"
        if getattr(args, "chunk_size", None) is not None
        else "external action_horizon deferred"
    )
    config_reference = str(getattr(args, "train_config", "")).strip()
    if not config_reference:
        raise ValueError(
            "--train-config must be a non-empty external configuration reference"
        )
    deferred_items = [
        "external policy configuration read",
        "policy spec",
        "policy schema",
    ]
    if getattr(args, "chunk_size", None) is None:
        deferred_items.append("default chunk size")
    prompt = getattr(args, "prompt", None)

    plan = {
        "dry_run_plan": {
            "result": "local validation passed; policy schema deferred",
            "runner": {
                "entrypoint": runner_name,
                "policy_transport": policy_transport_name,
                "control_mode": args.control_mode,
            },
            "execution": {
                "passes": 1,
                "interactive": False,
                "requires_tty": False,
                "execution_mode": args.execution_mode,
                **parameters,
            },
            "policy": {
                "schema": {
                    "schema_kind": "external-config-deferred",
                    "configuration_kind": configuration_kind,
                    "configuration_reference": config_reference,
                    "state_dim": None,
                    "action_dim": None,
                    "model_action_dim": None,
                    "action_horizon": None,
                    "image_ids": None,
                    "validation": "deferred to real-run preflight",
                },
                "prompt_supplied_by_cli": bool(
                    isinstance(prompt, str) and prompt.strip()
                ),
                "prompt_resolution": "deferred without server or dataset asset lookup",
                "operations": {
                    "external_configuration_read": "skipped",
                    "external_model_package_import": "skipped",
                    "transport_construction": "skipped",
                    "server_metadata": "skipped",
                    "server_preflight": "skipped",
                    "inference": "skipped",
                },
            },
            "hardware": hardware_plan,
            "outputs": {
                "recording_requested": bool(args.record),
                "window_requested": bool(args.window),
                "hdf5_requested": bool(args.save),
                "recorder_construction": "skipped",
                "window_construction": "skipped",
                "hdf5_creation": "skipped",
            },
            "validated": [
                "arguments",
                "external policy configuration wiring",
                "runtime config",
                "topology selection",
                "runner wiring",
            ],
            "deferred": deferred_items,
        }
    }
    print(json.dumps(plan, indent=2, allow_nan=False), flush=True)
