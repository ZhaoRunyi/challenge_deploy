from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SlaiPolicySpec:
    train_config_name: str
    train_config: Any
    state_space: Any
    action_space: Any
    image_space: Any
    state_dim: int
    action_dim: int
    model_action_dim: int | None
    action_horizon: int | None
    image_ids: tuple[str, ...]
    image_key_map: dict[str, str]


def space_summary(space: Any) -> dict[str, Any]:
    gripper = getattr(space, "gripper", None)
    return {
        "ids": getattr(space, "ids", None),
        "arms": getattr(space, "arms", None),
        "ee_rotation": getattr(space, "ee_rotation", None),
        "gripper": None if gripper is None else {
            "type": getattr(gripper, "type", None),
            "threshold": getattr(gripper, "threshold", None),
            "full_width": getattr(gripper, "full_width", None),
        },
    }


def slai_policy_spec_summary(spec: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = {
        "train_config_name": spec.train_config_name,
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "model_action_dim": spec.model_action_dim,
        "action_horizon": spec.action_horizon,
        "image_ids": list(spec.image_ids),
        "image_key_map": spec.image_key_map,
        "state_space": space_summary(spec.state_space),
        "action_space": space_summary(spec.action_space),
        "image_space": {"ids": getattr(spec.image_space, "ids", None)},
    }
    if extra:
        summary.update(extra)
    return summary


def decoded_action_summary(decoded: Any) -> dict[str, Any]:
    return {
        "control_mode": decoded.control_mode,
        "arms": {
            arm_name: {
                "has_joint": arm_action.joint is not None,
                "has_ee_pose": arm_action.ee_pose is not None,
                "gripper": arm_action.gripper,
            }
            for arm_name, arm_action in decoded.arms.items()
        },
    }
