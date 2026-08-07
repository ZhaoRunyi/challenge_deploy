from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

from hardware.constants import PIPER_GRIPPER_FULL_OPEN_METERS
from hardware.schemas import RobotSnapshot
from . import slai_piper_policy
from .base import (
    ActionGripperEncoding,
    ControlMode,
    PolicySessionCapability,
    PolicyResponseFormatError,
    SlaiPiperClient,
    StateGripperEncoding,
    build_full_piper_state as build_slai_full_piper_state,
    image_to_rgb,
    quiet_close_policy_transport_on_construction_error,
)
from .specs import slai_policy_spec_summary


FASTWAM_IMAGE_IDS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
FASTWAM_RAW_PROPRIO_DIM = 32
FASTWAM_ACTION_DIM = 14
FASTWAM_DEFAULT_TRAIN_CONFIG = "piper_realworld_unseen_adapter"
FASTWAM_DEFAULT_ACTION_HORIZON = 32

FASTWAM_TASK_PROMPTS: dict[str, str] = {
    "beaker_mixer": "Pick the beaker, place it on the mixer, then flip the toggle switch with the other arm.",
    "carry_basket": "Pick the bottle then place it to the basket, carry the basket with the other arm.",
    "click_bell": "Click the bell.",
    "depress_pipette": "Pick the pipette and move it to the center-top of the beaker, use the other arm to depress the plunger.",
    "dock_tubes": "Pick up two centrifuge tubes from the table and dock them horizontally.",
    "insert_test_tube": "Pick up the test tube and place it in the rack.",
    "items_handover_place": "Pick up the pen, hand it over to the other arm and then place it in to the pen holder.",
    "open_drawer": "Open the drawer, pick the tomato with the other arm then place it in the drawer.",
    "open_pan": "Grab the knob on the pan lid, lift it to open the pan, then pick the carrot with the other arm, place it in the pan, then move the lid back to the pan to close it.",
    "pour_dual": "Pick the cup and the bottle with the other arm, pour the water from bottle to cup.",
    "rearr": "Pick the fork and the spoon, place them next to the plate.",
    "clean_plate": "grasp the towel to clean the plate",
    "click_two_bells": "Use the left arm to click the bell on the left and use the right arm to click the bell on the right.",
    "take_tissue": "pick up the tissue with the right arm",
    "tomato_basket": "Pick the tomato then place it into the basket.",
    "pick_test_one_tube": "Pick up the test tube on the right from the rack with the right arm.",
}

FASTWAM_TASK_DISTRIBUTION_ALIASES: dict[str, tuple[str, ...]] = {
    "items_handover_place": ("items_hand_over_place",),
    "pour_dual": ("pour_water_dual",),
    "pick_test_one_tube": ("pick_test_tube", "test_tube"),
}


@dataclass(frozen=True)
class FastWAMPolicySpec:
    train_config_name: str
    train_config: dict[str, Any]
    state_space: Any
    action_space: Any
    image_space: Any
    state_dim: int
    action_dim: int
    model_action_dim: int | None
    action_horizon: int
    image_ids: tuple[str, ...]
    image_key_map: dict[str, str]
    prompt: str | None = None
    distribution_name: str | None = None
    distribution_aliases: tuple[str, ...] = ()


def normalized_fastwam_prompt(prompt: str) -> str:
    return " ".join(str(prompt).strip().casefold().rstrip(".").split())


def fastwam_distribution_name_for_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    normalized_prompt = normalized_fastwam_prompt(prompt)
    for distribution_name, task_prompt in FASTWAM_TASK_PROMPTS.items():
        if normalized_prompt == normalized_fastwam_prompt(task_prompt):
            return distribution_name
    return None


def fastwam_distribution_aliases(distribution_name: str) -> tuple[str, ...]:
    aliases = FASTWAM_TASK_DISTRIBUTION_ALIASES.get(distribution_name, ())
    return tuple(alias for index, alias in enumerate(aliases) if alias and alias not in aliases[:index])


class FastWAMHTTPPolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        endpoint: str = "/infer",
        timeout_s: float = 120.0,
    ) -> None:
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        self.url = f"http://{host}:{int(port)}{endpoint}"
        self.timeout_s = float(timeout_s)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._closed = False

    def get_server_metadata(self) -> dict[str, Any]:
        return {}

    def new_inference_session(self) -> "FastWAMHTTPPolicyClient":
        """Clone immutable endpoint settings with an independent HTTP opener."""

        session = object.__new__(type(self))
        session.url = self.url
        session.timeout_s = self.timeout_s
        session.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        session._closed = False
        return session

    def set_inference_timeout(self, timeout_s: float) -> None:
        if timeout_s <= 0.0:
            raise ValueError("inference timeout must be positive")
        self.timeout_s = float(timeout_s)

    def close(self) -> None:
        # urllib doesn't expose an in-flight connection before ``open``
        # returns.  Matching its socket timeout to the coordinator deadline
        # bounds retirement; this flag prevents the lane from being reused.
        self._closed = True

    def open_payload(self, payload: dict[str, Any], *, timeout_s: float | None = None) -> Any:
        if self._closed:
            raise RuntimeError("FastWAM inference session is closed")
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self.opener.open(request, timeout=self.timeout_s if timeout_s is None else float(timeout_s))

    def probe(self, *, timeout_s: float = 8.0) -> dict[str, Any]:
        try:
            with self.open_payload({}, timeout_s=timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and "images" in body:
                return {"ok": True, "status": exc.code, "url": self.url}
            raise RuntimeError(
                f"FastWAM server preflight failed at {self.url}: expected HTTP 400 mentioning images for empty payload, "
                f"got HTTP {exc.code}: {body}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"FastWAM server preflight failed at {self.url}: server did not return the expected FastWAM HTTP response. "
                f"Expected HTTP 400 JSON for empty payload; got {exc!r}"
            ) from exc
        raise RuntimeError(
            f"FastWAM server preflight failed at {self.url}: expected HTTP 400 for empty payload, "
            f"got success body {body[:200]!r}"
        )

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with self.open_payload(payload) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FastWAM server returned HTTP {exc.code}: {error_body}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Failed to reach FastWAM server at {self.url}: {exc!r}") from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PolicyResponseFormatError(
                f"FastWAM server returned malformed JSON: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise PolicyResponseFormatError(
                f"FastWAM server returned non-object JSON: {type(result).__name__}"
            )
        if "error" in result:
            raise RuntimeError(f"FastWAM server error: {result['error']}")
        video_b64 = result.get("predicted_video_b64")
        if isinstance(video_b64, str):
            result = dict(result)
            result["predicted_video_bytes"] = base64.b64decode(video_b64)
        return result


def load_fastwam_policy_spec(
    train_config_name: str = FASTWAM_DEFAULT_TRAIN_CONFIG,
    *,
    action_horizon: int = FASTWAM_DEFAULT_ACTION_HORIZON,
    prompt: str | None = None,
) -> FastWAMPolicySpec:
    gripper = slai_piper_policy.GripperConfig(
        type="raw",
        threshold=0.01,
        full_width=PIPER_GRIPPER_FULL_OPEN_METERS,
    )
    state_space = slai_piper_policy.StateSpaceConfig(
        ids="all",
        arms="dual",
        ee_rotation="rot6d",
        gripper=gripper,
    )
    action_space = slai_piper_policy.ActionSpaceConfig(
        ids="joint_gripper",
        arms="dual",
        ee_rotation="rot6d",
        gripper=gripper,
    )
    image_space = slai_piper_policy.ImageSpaceConfig(ids=list(FASTWAM_IMAGE_IDS))
    prompt_distribution_name = fastwam_distribution_name_for_prompt(prompt)
    if action_horizon <= 0:
        raise ValueError("FastWAM action_horizon must be positive")
    return FastWAMPolicySpec(
        train_config_name=train_config_name,
        train_config={
            "server_protocol": "fastwam_http_json",
            "raw_proprio_dim": FASTWAM_RAW_PROPRIO_DIM,
        },
        state_space=state_space,
        action_space=action_space,
        image_space=image_space,
        state_dim=int(slai_piper_policy.get_space_dim(state_space)),
        action_dim=int(slai_piper_policy.get_space_dim(action_space)),
        model_action_dim=FASTWAM_ACTION_DIM,
        action_horizon=int(action_horizon),
        image_ids=tuple(slai_piper_policy.get_image_ids(image_space)),
        image_key_map={image_id: image_id for image_id in slai_piper_policy.get_image_ids(image_space)},
        prompt=prompt,
        distribution_name=prompt_distribution_name,
        distribution_aliases=fastwam_distribution_aliases(prompt_distribution_name) if prompt_distribution_name else (),
    )


def encode_rgb_image_as_base64_png(image_rgb: np.ndarray) -> str:
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel RGB image, got shape {image_rgb.shape}")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("cv2 failed to encode FastWAM image as PNG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def build_fastwam_proprio(
    snapshot: RobotSnapshot,
    spec: FastWAMPolicySpec,
    *,
    state_gripper_encoding: StateGripperEncoding = "policy",
) -> np.ndarray:
    proprio = build_slai_full_piper_state(
        snapshot,
        spec,
        state_gripper_encoding=state_gripper_encoding,
    ).astype(np.float32)
    if proprio.shape != (FASTWAM_RAW_PROPRIO_DIM,):
        raise ValueError(f"FastWAM proprio must be 32D, got {proprio.shape}")
    return proprio


def build_fastwam_images(snapshot: RobotSnapshot, spec: FastWAMPolicySpec) -> dict[str, str]:
    images: dict[str, str] = {}
    for image_id in spec.image_ids:
        if image_id not in snapshot.images:
            raise KeyError(f"Snapshot is missing required image {image_id}")
        images[image_id] = encode_rgb_image_as_base64_png(image_to_rgb(snapshot.images[image_id]))
    return images


def build_policy_payload(
    snapshot: RobotSnapshot,
    *,
    prompt: str | None,
    spec: FastWAMPolicySpec,
    action_horizon: int | None = None,
    num_inference_steps: int | None = None,
    seed: int | None = None,
    state_gripper_encoding: StateGripperEncoding = "policy",
    session_id: str | None = None,
) -> dict[str, Any]:
    if prompt is None:
        raise ValueError("FastWAM policy payload requires an instruction prompt")
    payload: dict[str, Any] = {
        "images": build_fastwam_images(snapshot, spec),
        "proprio": build_fastwam_proprio(
            snapshot,
            spec,
            state_gripper_encoding=state_gripper_encoding,
        ).tolist(),
        "instruction": prompt,
        "action_horizon": int(action_horizon or spec.action_horizon),
    }
    if num_inference_steps is not None:
        payload["num_inference_steps"] = int(num_inference_steps)
    if seed is not None:
        payload["seed"] = int(seed)
    if session_id is not None:
        payload["session_id"] = session_id
    return payload


class FastWAMPiperClient(SlaiPiperClient):
    SESSION_CAPABILITY = PolicySessionCapability.SESSION_ID

    def __init__(
        self,
        train_config_name: str = FASTWAM_DEFAULT_TRAIN_CONFIG,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        endpoint: str = "/infer",
        request_timeout_s: float = 120.0,
        control_mode: ControlMode = "joints",
        joint_speed_percent: int = 50,
        ee_speed_percent: int = 50,
        gripper_effort: int | None = None,
        gripper_action_frames: int = 1,
        gripper_threshold: float | None = None,
        gripper_lower: float | None = None,
        gripper_upper: float | None = None,
        action_horizon: int = FASTWAM_DEFAULT_ACTION_HORIZON,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        state_gripper_encoding: StateGripperEncoding = "policy",
        action_gripper_encoding: ActionGripperEncoding = "policy",
    ) -> None:
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        spec = load_fastwam_policy_spec(
            train_config_name,
            action_horizon=self.action_horizon,
        )
        policy_client = FastWAMHTTPPolicyClient(
            host,
            port,
            endpoint=endpoint,
            timeout_s=request_timeout_s,
        )
        try:
            super().__init__(
                spec=spec,
                policy_client=policy_client,
                control_mode=control_mode,
                joint_speed_percent=joint_speed_percent,
                ee_speed_percent=ee_speed_percent,
                gripper_threshold=gripper_threshold,
                gripper_lower=gripper_lower,
                gripper_upper=gripper_upper,
                state_gripper_encoding=state_gripper_encoding,
                action_gripper_encoding=action_gripper_encoding,
                gripper_effort=gripper_effort,
                gripper_action_frames=gripper_action_frames,
            )
        except BaseException:
            quiet_close_policy_transport_on_construction_error(policy_client)
            raise

    def build_payload(
        self,
        snapshot: RobotSnapshot,
        prompt: str | None = None,
        *,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if session_id is None:
            session_id = self.default_session_id
        return build_policy_payload(
            snapshot,
            prompt=prompt,
            spec=self.spec,
            action_horizon=self.action_horizon,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            state_gripper_encoding=self.state_gripper_encoding,
            session_id=session_id,
        )

    def probe_server(self, *, timeout_s: float = 8.0) -> dict[str, Any]:
        return self.client.probe(timeout_s=timeout_s)

def spec_summary(spec: FastWAMPolicySpec) -> dict[str, Any]:
    return slai_policy_spec_summary(
        spec,
        extra={
            "server_protocol": spec.train_config["server_protocol"],
            "raw_proprio_dim": spec.train_config["raw_proprio_dim"],
            "prompt": spec.prompt,
            "distribution_name": spec.distribution_name,
            "distribution_aliases": list(spec.distribution_aliases),
        },
    )
