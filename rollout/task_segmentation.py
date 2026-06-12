from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SAM31_MULTIPLEX_WEIGHTS = Path(__file__).resolve().parents[1] / "artifacts" / "sam3.1_multiplex.pt"


@dataclass(frozen=True)
class TaskObjectSpec:
    label: str
    phrases: tuple[str, ...]
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    max_instances: int = 1
    min_area: int = 80
    max_area_ratio: float = 0.45
    score_threshold: float = 0.17
    allow_border: bool = False
    negative_phrases: tuple[str, ...] = ()
    foreground_weight: float = 1.0
    box_fallback: bool = False


@dataclass(frozen=True)
class SelectedMask:
    label: str
    score: float
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


TASK_OBJECT_SPECS: dict[str, tuple[TaskObjectSpec, ...]] = {
    "ZhaoRunyi/Piper_beaker_mixer_0421": (
        TaskObjectSpec(
            "mixer",
            (
                "magnetic stirrer hotplate",
                "magnetic stirrer",
                "laboratory hotplate stirrer",
                "lab mixer machine",
                "white red magnetic stirrer",
                "biochemistry lab mixer",
            ),
            boxes=((0.03, 0.02, 0.53, 0.50),),
            max_area_ratio=0.45,
            score_threshold=0.15,
            negative_phrases=("beaker",),
            foreground_weight=0.15,
        ),
        TaskObjectSpec(
            "beaker",
            (
                "clear glass beaker",
                "transparent glass beaker",
                "clear glass cup",
                "glass beaker",
                "clear glass container",
                "small transparent beaker",
            ),
            boxes=((0.48, 0.05, 0.62, 0.24), (0.55, 0.06, 0.74, 0.28)),
            max_instances=2,
            max_area_ratio=0.14,
            score_threshold=0.14,
            negative_phrases=("hot plate", "mixer"),
            foreground_weight=0.0,
            box_fallback=True,
        ),
    ),
    "ZhaoRunyi/Piper_carry_basket_0421": (
        TaskObjectSpec(
            "basket",
            ("basket", "wicker basket", "woven basket"),
            boxes=((0.63, 0.00, 1.00, 0.62),),
            max_area_ratio=0.55,
            score_threshold=0.13,
            negative_phrases=("bottle",),
            foreground_weight=1.0,
        ),
        TaskObjectSpec(
            "bottle",
            ("water bottle", "plastic bottle", "clear water bottle", "blue cap bottle", "bottle"),
            boxes=((0.48, 0.02, 0.66, 0.36),),
            max_area_ratio=0.18,
            score_threshold=0.15,
            negative_phrases=("basket",),
            foreground_weight=1.0,
        ),
    ),
    "ZhaoRunyi/Piper_click_bell_0403": (
        TaskObjectSpec(
            "bell",
            ("blue yellow desk bell", "service bell", "desk bell", "call bell", "bell"),
            boxes=((0.46, 0.16, 0.63, 0.42),),
            min_area=30,
            max_area_ratio=0.06,
            score_threshold=0.14,
            foreground_weight=1.0,
        ),
    ),
    "ZhaoRunyi/Piper_depress_pipette_0421": (
        TaskObjectSpec(
            "pipette",
            (
                "white blue micropipette",
                "adjustable micropipette",
                "micropipette",
                "lab pipette",
                "pipette gun",
                "transfer pipette",
            ),
            boxes=((0.82, 0.03, 1.00, 0.46),),
            max_area_ratio=0.12,
            score_threshold=0.15,
            negative_phrases=("beaker",),
            foreground_weight=0.35,
        ),
        TaskObjectSpec(
            "beaker",
            ("clear glass beaker", "transparent glass beaker", "glass beaker", "clear glass container"),
            boxes=((0.60, 0.03, 0.79, 0.30),),
            max_area_ratio=0.14,
            score_threshold=0.15,
            negative_phrases=("pipette",),
            foreground_weight=0.0,
            box_fallback=True,
        ),
    ),
    "ZhaoRunyi/Piper_dock_tubes_0421": (
        TaskObjectSpec(
            "tube",
            (
                "50ml conical tube",
                "falcon tube",
                "white capped centrifuge tube",
                "plastic centrifuge tube",
                "sample tube",
            ),
            boxes=((0.12, 0.03, 0.30, 0.60), (0.63, 0.03, 0.82, 0.60)),
            max_instances=2,
            max_area_ratio=0.12,
            score_threshold=0.13,
            foreground_weight=0.0,
        ),
    ),
    "ZhaoRunyi/Piper_items_hand_over_place_0421": (
        TaskObjectSpec(
            "pen_holder",
            ("black mesh cup", "pen holder", "black cup", "mesh pen holder"),
            boxes=((0.56, 0.02, 0.77, 0.34),),
            max_area_ratio=0.16,
            score_threshold=0.15,
            negative_phrases=("marker", "pen"),
            foreground_weight=1.0,
        ),
        TaskObjectSpec(
            "pen",
            ("black marker", "marker", "marker pen", "pen"),
            boxes=((0.84, 0.06, 0.99, 0.34),),
            max_area_ratio=0.08,
            score_threshold=0.13,
            negative_phrases=("cup", "pen holder"),
            foreground_weight=1.0,
        ),
    ),
    "ZhaoRunyi/Piper_insert_test_tube_0426": (
        TaskObjectSpec(
            "tube_rack",
            (
                "white plastic test tube rack",
                "laboratory tube rack",
                "plastic centrifuge tube rack",
                "white sample rack",
                "test tube rack",
            ),
            boxes=((0.28, 0.08, 0.72, 0.42),),
            max_area_ratio=0.28,
            score_threshold=0.14,
            negative_phrases=("test tube",),
            foreground_weight=0.10,
        ),
        TaskObjectSpec(
            "test_tube",
            (
                "clear glass test tube",
                "transparent test tube",
                "glass sample tube",
                "laboratory test tube",
                "test tube",
            ),
            boxes=((0.76, 0.14, 0.89, 0.58),),
            max_area_ratio=0.08,
            score_threshold=0.13,
            negative_phrases=("tube rack",),
            foreground_weight=0.0,
            box_fallback=True,
        ),
    ),
    "ZhaoRunyi/Piper_open_drawer_0421": (
        TaskObjectSpec(
            "drawer",
            (
                "white plastic drawer box",
                "white storage drawer",
                "white plastic drawer",
                "plastic drawer box",
                "white rectangular storage box",
            ),
            boxes=((0.48, 0.00, 0.82, 0.44),),
            max_area_ratio=0.45,
            score_threshold=0.13,
            negative_phrases=("tomato",),
            foreground_weight=0.05,
        ),
        TaskObjectSpec(
            "tomato",
            ("tomato", "red tomato"),
            boxes=((0.11, 0.07, 0.28, 0.31),),
            max_area_ratio=0.08,
            score_threshold=0.15,
            negative_phrases=("box", "drawer"),
            foreground_weight=1.0,
        ),
    ),
    "ZhaoRunyi/Piper_open_pan_0421": (
        TaskObjectSpec(
            "pan",
            ("black frying pan", "frying pan", "pan", "pot"),
            boxes=((0.45, 0.00, 0.82, 0.58),),
            max_area_ratio=0.32,
            score_threshold=0.15,
            negative_phrases=("carrot",),
            foreground_weight=1.0,
        ),
        TaskObjectSpec(
            "lid_knob",
            ("wooden lid knob", "round wooden knob", "pot lid handle", "wooden handle"),
            boxes=((0.54, 0.04, 0.71, 0.28),),
            max_area_ratio=0.05,
            score_threshold=0.15,
            foreground_weight=0.8,
        ),
        TaskObjectSpec(
            "carrot",
            ("carrot", "orange carrot"),
            boxes=((0.86, 0.05, 1.00, 0.55),),
            max_area_ratio=0.08,
            score_threshold=0.15,
            negative_phrases=("pan",),
            foreground_weight=1.0,
        ),
    ),
    "ZhaoRunyi/Piper_pour_dual_0421": (
        TaskObjectSpec(
            "bottle",
            ("clear water bottle", "water bottle", "plastic bottle", "blue cap bottle", "bottle"),
            boxes=((0.53, 0.06, 0.82, 0.46),),
            max_area_ratio=0.18,
            score_threshold=0.15,
            negative_phrases=("cup", "beaker"),
            foreground_weight=1.0,
        ),
        TaskObjectSpec(
            "cup",
            ("clear glass beaker", "transparent glass beaker", "clear glass cup", "glass cup", "beaker"),
            boxes=((0.03, 0.05, 0.24, 0.38),),
            max_area_ratio=0.14,
            score_threshold=0.13,
            negative_phrases=("bottle",),
            foreground_weight=0.0,
            box_fallback=True,
        ),
    ),
    "ZhaoRunyi/Piper_rearr_0421": (
        TaskObjectSpec(
            "plate",
            ("white paper plate", "disposable paper plate", "paper plate", "paper dish"),
            boxes=((0.58, 0.38, 0.89, 0.77),),
            max_area_ratio=0.2,
            score_threshold=0.15,
            foreground_weight=0.0,
            box_fallback=True,
        ),
        TaskObjectSpec("fork", ("metal fork", "fork"), boxes=((0.85, 0.03, 1.00, 0.42),), max_area_ratio=0.08, score_threshold=0.14, negative_phrases=("spoon",), foreground_weight=1.0),
        TaskObjectSpec("spoon", ("metal spoon", "spoon"), boxes=((0.43, 0.55, 0.70, 1.00),), max_area_ratio=0.08, score_threshold=0.14, negative_phrases=("fork",), foreground_weight=1.0),
    ),
    "ZhaoRunyi/Piper_traffic_light_water_0609": (
        TaskObjectSpec(
            "magnetic_stirrer",
            (
                "magnetic stirrer hotplate",
                "magnetic stirrer",
                "laboratory hotplate stirrer",
                "white magnetic stirrer",
                "lab stirrer machine with knob",
            ),
            boxes=((0.18, 0.02, 0.78, 0.58),),
            max_area_ratio=0.50,
            score_threshold=0.14,
            negative_phrases=("beaker", "glass"),
            foreground_weight=0.15,
        ),
        TaskObjectSpec(
            "beaker",
            (
                "clear glass beaker",
                "transparent glass beaker",
                "large glass beaker",
                "middle sized glass beaker",
                "small glass beaker",
                "glass beaker with water",
            ),
            boxes=((0.05, 0.02, 0.95, 0.60),),
            max_instances=3,
            min_area=50,
            max_area_ratio=0.18,
            score_threshold=0.13,
            negative_phrases=("stirrer", "hotplate"),
            foreground_weight=0.0,
        ),
        TaskObjectSpec(
            "stirrer_control",
            (
                "toggle switch",
                "power switch",
                "stirrer knob",
                "control knob",
                "black control knob on magnetic stirrer",
            ),
            boxes=((0.18, 0.02, 0.78, 0.58),),
            max_instances=2,
            min_area=20,
            max_area_ratio=0.06,
            score_threshold=0.12,
            negative_phrases=("beaker",),
            foreground_weight=0.20,
        ),
    ),
}

TASK_OBJECT_SPECS["ZhaoRunyi/Piper_carry_basket_0426"] = TASK_OBJECT_SPECS["ZhaoRunyi/Piper_carry_basket_0421"]
TASK_OBJECT_SPECS["ZhaoRunyi/Piper_insert_test_tube_0430"] = TASK_OBJECT_SPECS["ZhaoRunyi/Piper_insert_test_tube_0426"]
TASK_OBJECT_SPECS["ZhaoRunyi/Piper_pour_dual_0427"] = TASK_OBJECT_SPECS["ZhaoRunyi/Piper_pour_dual_0421"]


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    if inter <= 0.0:
        return 0.0
    union = float(np.logical_or(mask_a, mask_b).sum())
    return inter / max(union, 1.0)


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1.0)


def _pixel_box_from_normalized(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def _cxcywh_from_xyxy(box: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        x2 - x1,
        y2 - y1,
    ]


def _alignment_score(
    candidate_box: tuple[int, int, int, int], prompt_box: tuple[int, int, int, int]
) -> float:
    iou = _bbox_iou(candidate_box, prompt_box)
    cx = 0.5 * (candidate_box[0] + candidate_box[2])
    cy = 0.5 * (candidate_box[1] + candidate_box[3])
    px = 0.5 * (prompt_box[0] + prompt_box[2])
    py = 0.5 * (prompt_box[1] + prompt_box[3])
    prompt_diag = max(
        ((prompt_box[2] - prompt_box[0]) ** 2 + (prompt_box[3] - prompt_box[1]) ** 2) ** 0.5,
        1.0,
    )
    dist_penalty = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / prompt_diag
    return iou - 0.1 * dist_penalty


def _foreground_from_background(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(image, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 0


def _best_component(mask: np.ndarray, prompt_box: list[int]) -> np.ndarray:
    binary = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if num_labels <= 2:
        return mask
    px = 0.5 * (prompt_box[0] + prompt_box[2])
    py = 0.5 * (prompt_box[1] + prompt_box[3])
    best_index = 0
    best_score = -1e18
    for idx in range(1, num_labels):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < 20.0:
            continue
        cx, cy = centroids[idx]
        distance = float((cx - px) ** 2 + (cy - py) ** 2)
        score = area - 0.01 * distance
        if score > best_score:
            best_score = score
            best_index = idx
    if best_index == 0:
        return mask
    return labels == best_index


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num_labels <= 2:
        return mask
    best_index = 0
    best_area = -1.0
    for idx in range(1, num_labels):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_index = idx
    if best_index == 0:
        return mask
    return labels == best_index


def _box_mask(
    shape: tuple[int, int], box: tuple[float, float, float, float]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = shape
    x1, y1, x2, y2 = _pixel_box_from_normalized(box, width, height)
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = True
    return mask, (x1, y1, x2, y2)


def _prepare_pil_image(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _clone_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return copy.deepcopy(value)


class TaskSegmenter:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not SAM31_MULTIPLEX_WEIGHTS.exists():
            raise FileNotFoundError(
                f"SAM3.1 checkpoint not found at {SAM31_MULTIPLEX_WEIGHTS}. "
                "Download facebook/sam3.1 into modelscope_cache first."
            )
        self.model = build_sam3_image_model(
            checkpoint_path=str(SAM31_MULTIPLEX_WEIGHTS),
            load_from_HF=False,
            device=self.device,
        )
        self.processor = Sam3Processor(self.model, device=self.device, confidence_threshold=0.01)

    def _autocast_context(self) -> Any:
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def select_masks(self, image: np.ndarray, repo_id: str, background_image: np.ndarray | None = None) -> list[SelectedMask]:
        specs = TASK_OBJECT_SPECS.get(repo_id)
        if not specs:
            raise KeyError(f"No task object specs configured for repo_id={repo_id!r}")

        height, width = image.shape[:2]
        image_area = float(height * width)
        selected: list[SelectedMask] = []
        pil_image = _prepare_pil_image(image)
        foreground_mask = _foreground_from_background(image, background_image) if background_image is not None else None

        with self._autocast_context():
            base_state = self.processor.set_image(pil_image)
            for spec in specs:
                scored: list[tuple[float, float, np.ndarray, tuple[int, int, int, int]]] = []
                for prompt in spec.phrases:
                    state = _clone_state(base_state)
                    state = self.processor.set_text_prompt(prompt, state)
                    if len(state["scores"]) == 0:
                        continue

                    scores = state["scores"].detach().float().cpu().numpy()
                    masks = state["masks"].detach().cpu().numpy().astype(bool)[:, 0]
                    for index in range(len(scores)):
                        raw_mask = masks[index]
                        raw_area = float(raw_mask.sum())
                        if raw_area < max(20.0, 0.5 * spec.min_area):
                            continue
                        if raw_area > image_area * max(0.9, spec.max_area_ratio * 3.0):
                            continue

                        fg_ratio = 0.0
                        candidate_mask = raw_mask
                        if foreground_mask is not None:
                            overlap = np.logical_and(raw_mask, foreground_mask)
                            overlap_area = float(overlap.sum())
                            fg_ratio = overlap_area / max(raw_area, 1.0)
                            if spec.foreground_weight >= 0.5 and overlap_area >= max(20.0, 0.08 * raw_area):
                                component = _largest_component(overlap)
                                if float(component.sum()) >= spec.min_area:
                                    candidate_mask = component

                        area = float(candidate_mask.sum())
                        if area < spec.min_area or area > image_area * spec.max_area_ratio:
                            continue
                        bbox = _mask_bbox(candidate_mask)
                        bbox_area = max(float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 1.0)
                        compactness = area / bbox_area
                        touches_border = bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= width - 1 or bbox[3] >= height - 1
                        rank_score = 0.45 * float(scores[index]) + 0.25 * compactness + spec.foreground_weight * fg_ratio
                        if touches_border and not spec.allow_border:
                            rank_score -= 0.10
                        scored.append((rank_score, float(scores[index]), candidate_mask, bbox))

                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                chosen_for_spec = 0
                for score, _, mask, bbox in scored:
                    if any(_mask_iou(mask, chosen.mask) > 0.55 for chosen in selected):
                        continue
                    selected.append(SelectedMask(label=spec.label, score=score, mask=mask, bbox=bbox))
                    chosen_for_spec += 1
                    if chosen_for_spec >= spec.max_instances:
                        break
                if chosen_for_spec >= spec.max_instances:
                    continue
                if not spec.box_fallback or not spec.boxes:
                    continue
                for box in spec.boxes:
                    mask, bbox = _box_mask((height, width), box)
                    if any(_mask_iou(mask, chosen.mask) > 0.55 for chosen in selected):
                        continue
                    selected.append(SelectedMask(label=spec.label, score=0.0, mask=mask, bbox=bbox))
                    chosen_for_spec += 1
                    if chosen_for_spec >= spec.max_instances:
                        break

        return selected


@lru_cache(maxsize=1)
def get_task_segmenter() -> TaskSegmenter:
    return TaskSegmenter()


def select_relevant_task_masks(
    image: np.ndarray, repo_id: str, *, background_image: np.ndarray | None = None
) -> list[SelectedMask]:
    return get_task_segmenter().select_masks(image, repo_id, background_image=background_image)
