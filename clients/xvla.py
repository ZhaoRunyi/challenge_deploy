from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from xvla_client import websocket_client_policy

from . import slai_piper_policy
from .specs import SlaiPolicySpec, slai_policy_spec_summary

ControlMode = Literal["joints", "ee_pose"]
PIPER_FULL_OPEN = 0.1
ACTION_MODE_IDS = {
    "slai_piper": "ee_gripper",
    "slai_piper_ee_gripper": "ee_gripper",
    "slai_piper_joint_gripper": "joint_gripper",
    "slai_piper_all": "all",
}
PIPER_TASKS = {
    "Piper_beaker_mixer_0421": "Pick the beaker, place it on the mixer, then flip the toggle switch with the other arm.",
    "Piper_carry_basket_0426": "Pick the bottle then place it to the basket, carry the basket with the other arm.",
    "Piper_click_bell_0403": "Click the bell",
    "Piper_depress_pipette_0421": "Pick the pipette and move it to the center-top of the beaker, use the other arm to depress the plunger.",
    "Piper_dock_tubes_0421": "Pick up two centrifuge tubes from the table and dock them horizontally.",
    "Piper_insert_test_tube_0430": "Pick up the test tube, and it to the other arm, and insert it to the rack.",
    "Piper_items_hand_over_place_0421": "Pick up the pen, hand it over to the other arm and then place it in to the pen holder",
    "Piper_open_drawer_0421": "Open the drawer, pick the tomato with the other arm then place it in the drawer.",
    "Piper_open_pan_0421": "Grab the knod on the pan lid, lift it to open the pan, then pick the carrot with the other arm, place it in the pan, then move the lid back to the pan to close it.",
    "Piper_pour_dual_0427": "Pick the cup and the bottle with the other arm, pour the water from bottle to cup",
    "Piper_rearr_0421": "Pick the fork and the spoon, place them next to the plate.",
}


class PiperPolicySpec(SlaiPolicySpec):
    pass


@dataclass(frozen=True)
class XVLATrainConfig:
    name: str
    action_mode: str
    space_id: str
    dataset_names: tuple[str, ...]
    prompt: str | None = None
    prompts: tuple[str, ...] = ()

    @property
    def distribution_name(self) -> str:
        return self.dataset_names[0] if len(self.dataset_names) == 1 else self.name

    @property
    def distribution_aliases(self) -> tuple[str, ...]:
        if len(self.dataset_names) != 1:
            return (self.name, "Piper_all_tasks", "all_tasks")
        aliases = [self.name]
        for dataset_name in self.dataset_names:
            aliases.append(dataset_name)
            aliases.append(dataset_name.removeprefix("Piper_"))
        return tuple(dict.fromkeys(alias for alias in aliases if alias))


@dataclass(frozen=True)
class DecodedArmAction:
    joint: np.ndarray | None
    gripper: float
    ee_pose: np.ndarray | None
    binary_gripper: bool = False


@dataclass(frozen=True)
class DecodedPiperAction:
    arms: dict[str, DecodedArmAction]
    control_mode: ControlMode


def task_config_name(dataset_name: str) -> str:
    stem = dataset_name.removeprefix("Piper_").rsplit("_", 1)[0]
    return f"slai_piper_{stem}_ee20_xvla_pt_bs256_400000"


def make_task_config(dataset_name: str) -> XVLATrainConfig:
    return XVLATrainConfig(
        name=task_config_name(dataset_name),
        action_mode="slai_piper",
        space_id=ACTION_MODE_IDS["slai_piper"],
        dataset_names=(dataset_name,),
        prompt=PIPER_TASKS[dataset_name],
    )


def dataset_names_from_metas(metas_path: Any) -> tuple[str, ...]:
    paths = metas_path if isinstance(metas_path, list) else [metas_path]
    dataset_names = []
    for value in paths:
        path = Path(str(value))
        if path.name == "info.json" and path.parent.name == "meta":
            dataset_names.append(path.parent.parent.name)
    return tuple(dataset_names)


def config_from_json(path: Path) -> XVLATrainConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "configs" in payload:
        raise ValueError(f"{path} contains multiple train configs; pass a concrete registered train-config name")
    action_mode = str(payload.get("action_mode", "slai_piper"))
    if action_mode not in ACTION_MODE_IDS:
        raise ValueError(f"Unsupported X-VLA action_mode {action_mode!r}")
    dataset_names = payload.get("dataset_name") or dataset_names_from_metas(payload.get("train_metas_path"))
    if isinstance(dataset_names, str):
        dataset_names = (dataset_names,)
    prompt = payload.get("prompt")
    prompts = tuple(str(item) for item in payload.get("prompts", ()) if item)
    return XVLATrainConfig(
        name=str(payload.get("name") or payload.get("exp_name") or path.stem),
        action_mode=action_mode,
        space_id=ACTION_MODE_IDS[action_mode],
        dataset_names=tuple(str(name) for name in dataset_names),
        prompt=str(prompt) if prompt else None,
        prompts=prompts,
    )


XVLA_TRAIN_CONFIGS = {task_config_name(name): make_task_config(name) for name in PIPER_TASKS}
XVLA_TRAIN_CONFIGS["slai_piper_all_tasks_ee20_xvla_pt_bs256_400000"] = XVLATrainConfig(
    name="slai_piper_all_tasks_ee20_xvla_pt_bs256_400000",
    action_mode="slai_piper",
    space_id=ACTION_MODE_IDS["slai_piper"],
    dataset_names=tuple(PIPER_TASKS),
    prompts=tuple(PIPER_TASKS.values()),
)


def load_xvla_train_config(name_or_path: str) -> XVLATrainConfig:
    path = Path(name_or_path)
    if path.is_file():
        return config_from_json(path)
    if name_or_path not in XVLA_TRAIN_CONFIGS:
        raise ValueError(f"Unknown X-VLA train config {name_or_path!r}. Available: {sorted(XVLA_TRAIN_CONFIGS)}")
    return XVLA_TRAIN_CONFIGS[name_or_path]


def rpy_to_rotation(rpy: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation_format = slai_piper_policy.resolve_rotation_format(rotation_format)
    rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
    if rotation_format == "rpy":
        return rpy
    rotation = Rotation.from_euler("xyz", rpy, degrees=False)
    if rotation_format == "quat":
        return rotation.as_quat().astype(np.float64)
    return rotation.as_matrix()[:, :2].reshape(-1).astype(np.float64)


def rotation_to_rpy(values: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation_format = slai_piper_policy.resolve_rotation_format(rotation_format)
    values = np.asarray(values, dtype=np.float64)
    if rotation_format == "rpy":
        return values.reshape(3)
    if rotation_format == "quat":
        return Rotation.from_quat(values.reshape(4)).as_euler("xyz", degrees=False)
    columns = values.reshape(3, 2)
    x_axis = columns[:, 0]
    if np.linalg.norm(x_axis) < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-9)
    y_axis = columns[:, 1] - np.dot(x_axis, columns[:, 1]) * x_axis
    if np.linalg.norm(y_axis) < 1e-9:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(x_axis, y_axis))) > 0.9:
            y_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    return Rotation.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=1)).as_euler("xyz", degrees=False)


def image_to_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_piper_policy_spec(train_config_name: str) -> PiperPolicySpec:
    train_config = load_xvla_train_config(train_config_name)
    ids = train_config.space_id
    state_space = slai_piper_policy.StateSpaceConfig(ids=ids)
    action_space = slai_piper_policy.ActionSpaceConfig(ids=ids)
    image_space = slai_piper_policy.ImageSpaceConfig(ids="all")
    return PiperPolicySpec(
        train_config_name=train_config.name,
        train_config=train_config,
        state_space=state_space,
        action_space=action_space,
        image_space=image_space,
        state_dim=int(slai_piper_policy.get_space_dim(state_space)),
        action_dim=int(slai_piper_policy.get_space_dim(action_space)),
        model_action_dim=None,
        action_horizon=None,
        image_ids=tuple(slai_piper_policy.get_image_ids(image_space)),
        image_key_map=slai_piper_policy.get_image_key_map(image_space),
    )


def build_full_piper_state(snapshot: Any, spec: PiperPolicySpec, old_gripper: bool = False) -> np.ndarray:
    del old_gripper
    left, right = snapshot.state.left, snapshot.state.right
    return np.concatenate(
        (
            left.qpos[:6],
            [left.qpos[6]],
            left.end_pose[:3],
            rpy_to_rotation(left.end_pose[3:6], spec.state_space.ee_rotation),
            right.qpos[:6],
            [right.qpos[6]],
            right.end_pose[:3],
            rpy_to_rotation(right.end_pose[3:6], spec.state_space.ee_rotation),
        )
    ).astype(np.float64)


def build_policy_payload(
    snapshot: Any,
    *,
    prompt: str | None,
    spec: PiperPolicySpec,
    old_gripper: bool = False,
    domain_id: int = 19,
    steps: int = 10,
) -> dict[str, Any]:
    if prompt is None:
        raise ValueError("X-VLA policy payload requires a prompt")
    payload: dict[str, Any] = {
        "observation.state": build_full_piper_state(snapshot, spec, old_gripper=old_gripper),
        "prompt": prompt,
        "action_mode": spec.train_config.action_mode,
        "domain_id": int(domain_id),
        "steps": int(steps),
    }
    for image_id, dataset_key in spec.image_key_map.items():
        if image_id not in snapshot.images:
            raise KeyError(f"Snapshot is missing required image {image_id}")
        payload[dataset_key] = image_to_rgb(snapshot.images[image_id])
    return payload


class XVLAPiperClient:
    def __init__(
        self,
        train_config_name: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        control_mode: ControlMode = "ee_pose",
        api_key: str | None = None,
        joint_speed_percent: int = 50,
        ee_speed_percent: int = 50,
        gripper_threshold: float | None = None,
        gripper_lower: float | None = None,
        gripper_upper: float | None = None,
        old_gripper: bool = False,
    ) -> None:
        self.spec = load_piper_policy_spec(train_config_name)
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port, api_key=api_key)
        self.control_mode = control_mode
        self.joint_speed_percent = joint_speed_percent
        self.ee_speed_percent = ee_speed_percent
        self.gripper_threshold = gripper_threshold
        self.gripper_lower = gripper_lower
        self.gripper_upper = gripper_upper
        self.old_gripper = old_gripper
        self.validate_control_mode()

    def validate_control_mode(self) -> None:
        fields = set(slai_piper_policy.fields_from_action_config(self.spec.action_space))
        if "gripper" not in fields:
            raise ValueError(f"{self.spec.train_config_name}: deploy requires action_space to include gripper")
        if self.control_mode == "joints":
            if "joint" not in fields:
                raise ValueError(f"{self.spec.train_config_name}: control_mode='joints' requires joint actions")
            return
        if self.control_mode == "ee_pose":
            missing = {"ee_pos", "ee_rot"} - fields
            if missing:
                raise ValueError(
                    f"{self.spec.train_config_name}: control_mode='ee_pose' requires fields {sorted(missing)}"
                )
            return
        raise ValueError(f"Unsupported control_mode: {self.control_mode}")

    def get_server_metadata(self) -> Any:
        return self.client.get_server_metadata()

    def build_payload(self, snapshot: Any, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        return build_policy_payload(
            snapshot,
            prompt=prompt,
            spec=self.spec,
            old_gripper=self.old_gripper,
            domain_id=int(kwargs.get("domain_id", 19)),
            steps=int(kwargs.get("steps", 10)),
        )

    def infer(self, snapshot: Any, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        return self.client.infer(self.build_payload(snapshot, prompt=prompt, **kwargs))

    def infer_actions(self, snapshot: Any, prompt: str | None = None, **kwargs: Any) -> np.ndarray:
        return np.asarray(self.infer(snapshot, prompt=prompt, **kwargs)["actions"], dtype=np.float64)

    def decode_gripper_for_piper(self, value: float) -> tuple[float, bool]:
        value = PIPER_FULL_OPEN if float(value) >= 0.5 else 0.0
        if self.gripper_threshold is not None:
            return (PIPER_FULL_OPEN if value >= self.gripper_threshold else 0.0), False
        if self.gripper_upper is not None and value > self.gripper_upper:
            return PIPER_FULL_OPEN, False
        if self.gripper_lower is not None and value < self.gripper_lower:
            return 0.0, False
        return value, False

    def decode_action(self, action: np.ndarray) -> DecodedPiperAction:
        action = np.asarray(action, dtype=np.float64)
        action_space = slai_piper_policy.space_from_action_config(self.spec.action_space)
        slices = slai_piper_policy.field_slices_from_space(action_space)
        fields = set(slai_piper_policy.fields_from_action_config(self.spec.action_space))
        decoded: dict[str, DecodedArmAction] = {}
        for arm in action_space["arms"]:
            gripper, binary_gripper = self.decode_gripper_for_piper(
                float(action[slices[f"{arm}_gripper"]][0])
            )
            joint = None
            ee_pose = None
            if "joint" in fields:
                joint = np.concatenate((action[slices[f"{arm}_joint"]], np.array([gripper])), axis=0)
            if {"ee_pos", "ee_rot"}.issubset(fields):
                ee_rpy = rotation_to_rpy(action[slices[f"{arm}_ee_rot"]], self.spec.action_space.ee_rotation)
                ee_pose = np.concatenate((action[slices[f"{arm}_ee_pos"]], ee_rpy, np.array([gripper])), axis=0)
            decoded[arm] = DecodedArmAction(
                joint=joint,
                gripper=gripper,
                ee_pose=ee_pose,
                binary_gripper=binary_gripper,
            )
        return DecodedPiperAction(arms=decoded, control_mode=self.control_mode)

    def command_action(self, robot: Any, action: np.ndarray) -> None:
        decoded = self.decode_action(action)
        for arm_name, arm_action in decoded.arms.items():
            arm = robot.left if arm_name == "left" else robot.right
            if decoded.control_mode == "joints":
                arm.command_joint_positions(arm_action.joint, speed_percent=self.joint_speed_percent)
            else:
                arm.command_end_pose(arm_action.ee_pose, speed_percent=self.ee_speed_percent)


def spec_summary(spec: PiperPolicySpec) -> dict[str, Any]:
    return slai_policy_spec_summary(
        spec,
        extra={
            "action_mode": spec.train_config.action_mode,
            "dataset_names": list(spec.train_config.dataset_names),
        },
    )
