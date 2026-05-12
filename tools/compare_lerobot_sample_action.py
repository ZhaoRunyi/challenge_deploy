#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import imageio
import imageio.v3 as iio
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from challenge_deploy.config import load_config
from challenge_deploy.lerobot_assets import _info_json, _parquet_path_for_episode, _video_path_for_episode
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.realsense import RealSenseRig
from replay_piper_npz_trajectory import REPLAY_DIM_NAMES, sim_gripper01_to_piper_opening, wait_until_robot_ready

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/clear/cobotmagic_Sim_clear_click_bell")
SAMPLE_SPECS: str | list[str] = "0" # 000

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/clear/cobotmagic_Sim_clear_items_handover_place")
SAMPLE_SPECS: str | list[str] = "0" #111

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/clear/cobotmagic_Sim_clear_open_drawer")
SAMPLE_SPECS: str | list[str] = "0" # 000

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/sample_loading_dual/cobotmagic_Sim_sample_loading_dual_000")
SAMPLE_SPECS: str | list[str] = "0" # 000


# ====================== NEW ======================= #

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/items_handover_place/cobotmagic_Sim_items_handover_place_000")
SAMPLE_SPECS: str | list[str] = "0" # 000

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/open_drawer/cobotmagic_Sim_drawer_open_place_000")
SAMPLE_SPECS: str | list[str] = "0" # 000

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/beaker_mixer_dual/cobotmagic_Sim_beaker_mixer_dual_000")
SAMPLE_SPECS: str | list[str] = "0" # 000

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/pour_water_dual/cobotmagic_Sim_pour_water_dual_000")
SAMPLE_SPECS: str | list[str] = "0" # 000

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/manipulate_pipette/cobotmagic_Sim_manipulate_pipette_000")
SAMPLE_SPECS: str | list[str] = "0" # 000

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/sample_loading_dual/cobotmagic_Sim_sample_loading_dual_000")
SAMPLE_SPECS: str | list[str] = "0" # 000

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/open_pan/cobotmagic_Sim_open_pan_000")
SAMPLE_SPECS: str | list[str] = "0" # 000

# ====================== 2nd round ======================= #
DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/items_handover_place/cobotmagic_Sim_items_handover_place_003")
SAMPLE_SPECS: str | list[str] = "0" # 000

DATASET_DIR = Path("/home/edemlab/challenge_ws/embodichain_ws/Embodied_Challenge/lerobot_dataset/manipulate_pipette/cobotmagic_Sim_manipulate_pipette_001")
SAMPLE_SPECS: str | list[str] = "0" # 000

CONFIG_PATH = ROOT / "configs" / "dual_piper_example.yaml"
OUTPUT_DIR = ROOT / "artifacts" / "lerobot_sample_action_compare"
FPS = 10.0
VIDEO_FPS = 30.0
SECONDS = 0.0
SPEED_PERCENT = 100
SETTLE_SECONDS = 1.0
READY_TIMEOUT = 15.0
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
PLOT_WIDTH_SCALE = 1.6
ACTION_COLOR = (32, 32, 220)
STATE_COLOR = (220, 90, 30)
TEXT_COLOR = (40, 40, 40)
GRID_COLOR = (220, 220, 220)
BORDER_COLOR = (138, 138, 138)
BACKGROUND_COLOR = 248
PLOT_MARGIN_FRACTION = 0.15
PLOT_MIN_MARGIN_JOINT = 0.10
PLOT_MIN_MARGIN_GRIPPER = 0.05
VIDEO_QUEUE_SIZE = 8

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run with no flags by default. Sample specs accept episode ids like 0, 42, or /path/to/episode_000000.parquet.")
    parser.add_argument("samples", nargs="*", help="Optional sample specs. Empty means use SAMPLE_SPECS at the top of this file.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--fps", type=float, default=FPS, help="Real robot replay rate in Hz. Does not change source trajectory length.")
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--speed-percent", type=int, default=SPEED_PERCENT)
    parser.add_argument("--settle-seconds", type=float, default=SETTLE_SECONDS)
    parser.add_argument("--ready-timeout", type=float, default=READY_TIMEOUT)
    return parser

def expand_sample_spec_value(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list sample spec, got {value!r}")
        return [str(item) for item in parsed]
    return [text]

def normalize_sample_specs(cli_samples: list[str]) -> list[str]:
    if cli_samples:
        return [item for raw_value in cli_samples for item in expand_sample_spec_value(raw_value)]
    return expand_sample_spec_value(SAMPLE_SPECS)

def resolve_sample_spec(default_dataset_dir: Path, sample_spec: str) -> tuple[Path, int, str]:
    if sample_spec.endswith(".parquet"):
        parquet_path = Path(sample_spec).expanduser().resolve()
        match = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
        if match is None or parquet_path.parents[1].name != "data":
            raise ValueError(f"Unsupported parquet sample spec: {sample_spec}")
        return parquet_path.parents[2], int(match.group(1)), parquet_path.stem
    episode_index = int(sample_spec)
    return default_dataset_dir, episode_index, f"episode_{episode_index:06d}"

def plot_label(index: int) -> str:
    side = "L" if index < 7 else "R"
    arm_index = index % 7
    return f"{side} grip" if arm_index == 6 else f"{side} j{arm_index + 1}"

def plot_points(values: np.ndarray, times_s: np.ndarray, rect: tuple[int, int, int, int], y_min: float, y_max: float) -> np.ndarray:
    x0, y0, x1, y1 = rect
    if len(values) == 1 or times_s[-1] <= times_s[0] + 1e-9:
        x_values = np.linspace(x0, x1 - 1, len(values), dtype=np.float64)
    else:
        x_values = x0 + (times_s - times_s[0]) / (times_s[-1] - times_s[0]) * (x1 - x0 - 1)
    y_values = y1 - (values - y_min) / (y_max - y_min) * (y1 - y0)
    y_values = np.clip(y_values, y0, y1 - 1)
    return np.stack((x_values, y_values), axis=1).round().astype(np.int32)

def put_text(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.27) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_COLOR, 1, cv2.LINE_AA)

def plot_point(value: float, time_s: float, time_start_s: float, time_stop_s: float, rect: tuple[int, int, int, int], y_min: float, y_max: float) -> np.ndarray:
    x0, y0, x1, y1 = rect
    if time_stop_s <= time_start_s + 1e-9:
        x_value = float(x0)
    else:
        x_value = x0 + (time_s - time_start_s) / (time_stop_s - time_start_s) * (x1 - x0 - 1)
    y_value = y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)
    return np.array([int(round(x_value)), int(round(np.clip(y_value, y0, y1 - 1)))], dtype=np.int32)

def plot_margin(index: int, value_range: float) -> float:
    minimum = PLOT_MIN_MARGIN_GRIPPER if index % 7 == 6 else PLOT_MIN_MARGIN_JOINT
    return max(value_range * PLOT_MARGIN_FRACTION, minimum)

def build_plot_context(width: int, height: int, times_s: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray | float | tuple[int, int, int, int]]]]:
    base_plot = np.full((height, width, 3), BACKGROUND_COLOR, dtype=np.uint8)
    plot_specs: list[dict[str, np.ndarray | float | tuple[int, int, int, int]]] = []
    cell_width = width // 2
    cell_height = height // 7
    duration_s = float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0
    time_start_s = float(times_s[0])
    time_stop_s = float(times_s[-1])
    for index, name in enumerate(REPLAY_DIM_NAMES):
        column = 0 if index < 7 else 1
        row = index % 7
        cell_x = column * cell_width
        cell_y = row * cell_height
        rect = (cell_x + 36, cell_y + 14, cell_x + cell_width - 10, cell_y + cell_height - 20)
        x0, y0, x1, y1 = rect
        action_values = actions[:, index]
        y_min = float(np.nanmin(action_values))
        y_max = float(np.nanmax(action_values))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or abs(y_max - y_min) < 1e-9:
            center = 0.0 if not np.isfinite(y_min) else y_min
            y_min, y_max = center - 1.0, center + 1.0
        margin = plot_margin(index, y_max - y_min)
        y_min -= margin
        y_max += margin
        cv2.rectangle(base_plot, (cell_x, cell_y), (cell_x + cell_width - 1, cell_y + cell_height - 1), GRID_COLOR, 1)
        cv2.rectangle(base_plot, (x0, y0), (x1, y1), BORDER_COLOR, 1)
        put_text(base_plot, plot_label(index), cell_x + 4, cell_y + 11)
        if row == 6:
            put_text(base_plot, "0.00s", x0, cell_y + cell_height - 5, 0.25)
            put_text(base_plot, f"{duration_s:.2f}s", max(x0, x1 - 36), cell_y + cell_height - 5, 0.25)
        if y_min <= 0.0 <= y_max:
            zero_y = int(round(y1 - (0.0 - y_min) / (y_max - y_min) * (y1 - y0)))
            cv2.line(base_plot, (x0, zero_y), (x1, zero_y), GRID_COLOR, 1)
        plot_specs.append(
            {
                "rect": rect,
                "y_min": y_min,
                "y_max": y_max,
                "action_points": plot_points(action_values, times_s, rect, y_min, y_max),
                "time_start_s": time_start_s,
                "time_stop_s": time_stop_s,
            }
        )
    return base_plot, plot_specs

def reveal_plot_frame(base_plot: np.ndarray, final_plot: np.ndarray, plot_rects: list[tuple[int, int, int, int]], ratio: float) -> np.ndarray:
    frame = base_plot.copy()
    ratio = float(np.clip(ratio, 0.0, 1.0))
    for x0, y0, x1, y1 in plot_rects:
        reveal_x = x0 + int(round((x1 - x0) * ratio))
        if reveal_x > x0:
            frame[y0:y1, x0:reveal_x] = final_plot[y0:y1, x0:reveal_x]
    return frame

class AsyncVideoWriter:
    def __init__(self, output_dir: Path, robot_fps: float, times_s: list[float], actions: np.ndarray) -> None:
        self.output_dir = output_dir
        self.output_path = output_dir / "real_vs_sim_3cam.mp4"
        self.snapshot_path = output_dir / "real_vs_sim_3cam_snapshot.png"
        self.times_s = np.asarray(times_s, dtype=np.float64)
        self.actions = np.asarray(actions, dtype=np.float64)
        self.robot_fps = robot_fps
        self.step_count = len(self.actions)
        self.output_frames = max(1, int(round(self.step_count * VIDEO_FPS / robot_fps)))
        self.repeat_counts = np.bincount(
            [min(self.step_count - 1, int(video_index * robot_fps / VIDEO_FPS)) for video_index in range(self.output_frames)],
            minlength=self.step_count,
        )
        self.items: queue.Queue[tuple[int, np.ndarray, np.ndarray] | None] = queue.Queue(maxsize=VIDEO_QUEUE_SIZE)
        self.failure: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="compare_lerobot_video_writer", daemon=True)
        self.thread.start()

    def enqueue(self, step_index: int, top_frame: np.ndarray, state: np.ndarray) -> None:
        if self.failure is not None:
            raise RuntimeError("Async video writer failed") from self.failure
        self.items.put((step_index, top_frame, state.copy()))

    def close(self) -> Path:
        self.items.put(None)
        self.thread.join()
        if self.failure is not None:
            raise RuntimeError("Async video writer failed") from self.failure
        return self.output_path

    def _run(self) -> None:
        writer = imageio.get_writer(self.output_path, fps=VIDEO_FPS, codec="mpeg4")
        try:
            plot_image = None
            plot_specs = None
            previous_state_points: list[np.ndarray | None] = [None] * len(REPLAY_DIM_NAMES)
            first_step = True
            while True:
                item = self.items.get()
                if item is None:
                    break
                step_index, top_frame, state = item
                if plot_image is None:
                    column_width = top_frame.shape[1] // 2
                    plot_width = max(column_width, int(round(column_width * PLOT_WIDTH_SCALE)))
                    plot_image, plot_specs = build_plot_context(plot_width, top_frame.shape[0], self.times_s, self.actions)
                    iio.imwrite(self.snapshot_path, top_frame[..., ::-1])
                for dim_index, spec in enumerate(plot_specs):
                    action_points = spec["action_points"]
                    if first_step:
                        cv2.circle(plot_image, tuple(action_points[0]), 1, ACTION_COLOR, -1, cv2.LINE_AA)
                    elif step_index > 0:
                        cv2.line(plot_image, tuple(action_points[step_index - 1]), tuple(action_points[step_index]), ACTION_COLOR, 1, cv2.LINE_AA)
                    state_point = plot_point(
                        float(state[dim_index]),
                        self.times_s[step_index],
                        float(spec["time_start_s"]),
                        float(spec["time_stop_s"]),
                        spec["rect"],
                        float(spec["y_min"]),
                        float(spec["y_max"]),
                    )
                    if previous_state_points[dim_index] is None:
                        cv2.circle(plot_image, tuple(state_point), 1, STATE_COLOR, -1, cv2.LINE_AA)
                    else:
                        cv2.line(plot_image, tuple(previous_state_points[dim_index]), tuple(state_point), STATE_COLOR, 1, cv2.LINE_AA)
                    previous_state_points[dim_index] = state_point
                first_step = False
                column_width = top_frame.shape[1] // 2
                combined = np.concatenate((top_frame[:, :column_width], top_frame[:, column_width:], plot_image), axis=1)
                for _ in range(int(self.repeat_counts[step_index])):
                    writer.append_data(combined[..., ::-1])
        except BaseException as exc:
            self.failure = exc
        finally:
            writer.close()

def save_diagnostics(
    output_dir: Path,
    parquet_path: Path,
    times_s: list[float],
    actions: list[np.ndarray],
    states: list[np.ndarray],
) -> tuple[Path, Path]:
    action_array = np.stack(actions).astype(np.float64)
    state_array = np.stack(states).astype(np.float64)
    error_array = state_array - action_array
    npz_path = output_dir / "trajectory_diagnostics.npz"
    np.savez_compressed(
        npz_path,
        times_s=np.asarray(times_s, dtype=np.float64),
        action_trajectory=action_array,
        state_trajectory=state_array,
        error_trajectory=error_array,
        names=np.asarray(REPLAY_DIM_NAMES),
    )
    summaries = []
    for state_index, state_name in enumerate(REPLAY_DIM_NAMES):
        state_values = state_array[:, state_index]
        best_index = state_index
        best_mae = float("inf")
        best_corr = None
        for action_index, action_name in enumerate(REPLAY_DIM_NAMES):
            action_values = action_array[:, action_index]
            mae = float(np.mean(np.abs(state_values - action_values)))
            corr = None if np.std(state_values) < 1e-9 or np.std(action_values) < 1e-9 else float(np.corrcoef(state_values, action_values)[0, 1])
            if mae < best_mae:
                best_index = action_index
                best_mae = mae
                best_corr = corr
        self_values = action_array[:, state_index]
        self_mae = float(np.mean(np.abs(state_values - self_values)))
        self_corr = None if np.std(state_values) < 1e-9 or np.std(self_values) < 1e-9 else float(np.corrcoef(state_values, self_values)[0, 1])
        summaries.append(
            {
                "state_index": state_index,
                "state_name": state_name,
                "matched_action_index": best_index,
                "matched_action_name": REPLAY_DIM_NAMES[best_index],
                "self_mae": self_mae,
                "matched_mae": best_mae,
                "self_corr": self_corr,
                "matched_corr": best_corr,
                "looks_like_order_issue": best_index != state_index and best_mae + 1e-6 < self_mae,
            }
        )
    json_path = output_dir / "trajectory_diagnostics.json"
    json_path.write_text(json.dumps({"parquet_path": str(parquet_path), "dimension_matches": summaries}, indent=2), encoding="utf-8")
    return npz_path, json_path

def run_sample(args: argparse.Namespace, sample_spec: str) -> None:
    dataset_dir, episode_index, output_name = resolve_sample_spec(Path(args.dataset_dir).expanduser().resolve(), sample_spec)
    info = _info_json(dataset_dir)
    parquet_path = _parquet_path_for_episode(dataset_dir, info, episode_index).resolve()
    dataframe = pd.read_parquet(parquet_path)
    robot_fps = float(args.fps)
    if robot_fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.seconds > 0.0:
        start_time_s = float(dataframe.iloc[0]["timestamp"])
        dataframe = dataframe[dataframe["timestamp"] <= start_time_s + args.seconds + 1e-9].reset_index(drop=True)
    else:
        dataframe = dataframe.reset_index(drop=True)
    action_trajectory = np.asarray(dataframe["action"].tolist(), dtype=np.float64)
    action_trajectory[:, 6] = [sim_gripper01_to_piper_opening(value) for value in action_trajectory[:, 6]]
    action_trajectory[:, 13] = [sim_gripper01_to_piper_opening(value) for value in action_trajectory[:, 13]]
    frame_indices = [int(value) for value in dataframe["frame_index"].tolist()]
    replay_times_s = [step_index / robot_fps for step_index in range(len(action_trajectory))]
    output_dir = Path(args.output_dir) / f"{dataset_dir.name}_{output_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "sample_spec": sample_spec,
                "episode_index": episode_index,
                "parquet_path": str(parquet_path),
                "fps": robot_fps,
                "speed_percent": args.speed_percent,
                "video_fps": VIDEO_FPS,
                "step_count": len(action_trajectory),
                "source_frame_indices": frame_indices,
                "replay_times_s": replay_times_s,
                "action_trajectory_14d": action_trajectory.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config = load_config(args.config)
    robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=True,
        name="compare_lerobot_sample_action",
    )
    cameras = RealSenseRig(config["cameras"]["serials"], width=int(config["cameras"]["width"]), height=int(config["cameras"]["height"]), fps=int(config["cameras"]["fps"]), warmup_frames=int(config["cameras"]["warmup_frames"]))
    sim_readers = [imageio.get_reader(_video_path_for_episode(dataset_dir, info, episode_index, f"{camera_name}.color")) for camera_name in CAMERA_NAMES]
    saved_times_s: list[float] = []
    saved_actions: list[np.ndarray] = []
    saved_states: list[np.ndarray] = []
    video_writer = AsyncVideoWriter(output_dir, robot_fps, replay_times_s, action_trajectory)
    capture_error: BaseException | None = None
    robot.connect(read_only=False)
    try:
        cameras.start()
        wait_until_robot_ready(robot, args.ready_timeout)
        if not robot.enable():
            print("Warning: Piper arm enable check did not report success; continuing anyway.", flush=True)
        robot.move_to_joint_positions(action_trajectory[0], speed_percent=args.speed_percent)
        if args.settle_seconds > 0.0:
            time.sleep(args.settle_seconds)
        for step_index, (action, frame_index) in enumerate(zip(action_trajectory, frame_indices)):
            started_at = time.monotonic()
            sim_frames = [reader.get_data(frame_index) for reader in sim_readers]
            robot.set_joint_positions(action, speed_percent=args.speed_percent)
            real_images = cameras.capture()
            state = robot.read_state().qpos.astype(np.float64).copy()
            compare_grid = np.concatenate([np.concatenate((real_images[camera_name], sim_frame[..., ::-1]), axis=1) for camera_name, sim_frame in zip(CAMERA_NAMES, sim_frames)], axis=0)
            video_writer.enqueue(step_index, compare_grid, state)
            saved_times_s.append(replay_times_s[step_index])
            saved_actions.append(action.copy())
            saved_states.append(state)
            remaining = (1.0 / robot_fps) - (time.monotonic() - started_at)
            if remaining > 0.0:
                time.sleep(remaining)
    except BaseException as exc:
        capture_error = exc
    finally:
        try:
            cameras.stop()
        finally:
            for reader in sim_readers:
                reader.close()
            robot.disconnect()
    video_path = video_writer.close()
    if capture_error is not None:
        raise capture_error
    diagnostics_npz_path, diagnostics_json_path = save_diagnostics(output_dir, parquet_path, saved_times_s, saved_actions, saved_states)
    print(json.dumps({"sample_spec": sample_spec, "episode_index": episode_index, "steps": len(saved_actions), "frame_start_index": min(frame_indices), "frame_stop_index_exclusive": max(frame_indices) + 1, "fps": robot_fps, "speed_percent": args.speed_percent, "video_fps": VIDEO_FPS, "layout": "3_rows_x_3_cols(real_left,sim_middle,plots_right_left_right_split)", "camera_names": list(CAMERA_NAMES), "plot_width_scale": PLOT_WIDTH_SCALE, "settle_seconds": args.settle_seconds, "parquet_path": str(parquet_path), "frame_indices": frame_indices, "replay_times_s": replay_times_s, "action_trajectory_14d": action_trajectory.tolist(), "metadata_path": str(metadata_path), "diagnostics_npz_path": str(diagnostics_npz_path), "diagnostics_json_path": str(diagnostics_json_path), "video_path": str(video_path), "output_dir": str(output_dir)}, indent=2), flush=True)

def main() -> None:
    args = build_parser().parse_args()
    for sample_spec in normalize_sample_specs(args.samples):
        run_sample(args, sample_spec)

if __name__ == "__main__":
    main()
