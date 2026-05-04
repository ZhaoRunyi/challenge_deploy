from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.constants import KAI0_GRIPPER_UNIT_SCALE, LEGACY_PIPER_DATA_GRIPPER_UNIT_SCALE

if TYPE_CHECKING:
    from challenge_deploy.piper import DualPiperSystem


DEFAULT_TRAJECTORY_PATH = Path("/home/edemlab/challenge_ws/Piper_click_bell_0403_video_action_mean_trajectory.npz")
SIM_GRIPPER_FULL_OPEN_M = 0.10
REPLAY_DIM_ALIASES: tuple[tuple[str, ...], ...] = (
    ("left_joint_1", "left_joint_waist"),
    ("left_joint_2", "left_joint_shoulder"),
    ("left_joint_3", "left_joint_elbow"),
    ("left_joint_4", "left_joint_forearm_roll"),
    ("left_joint_5", "left_joint_wrist_angle"),
    ("left_joint_6", "left_joint_wrist_rotate"),
    ("left_gripper",),
    ("right_joint_1", "right_joint_waist"),
    ("right_joint_2", "right_joint_shoulder"),
    ("right_joint_3", "right_joint_elbow"),
    ("right_joint_4", "right_joint_forearm_roll"),
    ("right_joint_5", "right_joint_wrist_angle"),
    ("right_joint_6", "right_joint_wrist_rotate"),
    ("right_gripper",),
)
REPLAY_DIM_NAMES: tuple[str, ...] = (
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
)
EMBODICHAIN_32D_REPLAY_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20, 21, 22)


@dataclass(frozen=True)
class LoadedTrajectory:
    trajectory_path: Path
    trajectory_key: str
    source_shape: tuple[int, ...]
    source_action_dim: int
    selected_indices: list[int]
    selected_action_names: list[str]
    replay_actions_sim: np.ndarray
    replay_qpos: np.ndarray
    available_keys: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a Piper dual-arm trajectory from an NPZ file. The replay command space is "
            "[left joints 6, left gripper01, right joints 6, right gripper01]."
        )
    )
    parser.add_argument(
        "trajectory_npz",
        nargs="?",
        default=str(DEFAULT_TRAJECTORY_PATH),
        help="Path to an NPZ file that stores the trajectory to replay.",
    )
    parser.add_argument(
        "--trajectory-key",
        default="action_mean_trajectory",
        help="NPZ array key that contains the time-major trajectory matrix.",
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--fps", type=float, default=10.0, help="Replay command frequency in Hz. Use 0 to send as fast as possible.")
    parser.add_argument("--speed-percent", type=int, default=20, help="Piper joint mode speed percent.")
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=None,
        help="Optional final Piper gripper opening threshold in meters. Values below it are clipped to 0.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="First trajectory frame to replay.")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum frames to replay. 0 means replay to the end.")
    parser.add_argument("--move-hz", type=float, default=30.0, help="Interpolation frequency used when moving to the first frame.")
    parser.add_argument("--settle-seconds", type=float, default=0.5, help="Sleep time after moving to the first frame.")
    parser.add_argument("--skip-move-to-start", action="store_true", help="Do not interpolate from current robot state to the first frame.")
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    parser.add_argument("--print-every", type=int, default=10, help="Print progress every N replayed frames.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the parsed trajectory summary; do not connect to hardware.")
    return parser


def _apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.left_can:
        set_by_dotted_path(config, "robot.left.can_name", args.left_can)
    if args.right_can:
        set_by_dotted_path(config, "robot.right.can_name", args.right_can)
    return config


def _hardware_gripper_to_legacy_raw(value: float) -> float:
    return float(value) * (KAI0_GRIPPER_UNIT_SCALE / LEGACY_PIPER_DATA_GRIPPER_UNIT_SCALE)


def _legacy_raw_gripper_to_hardware(value: float) -> float:
    return max(0.0, float(value) * (LEGACY_PIPER_DATA_GRIPPER_UNIT_SCALE / KAI0_GRIPPER_UNIT_SCALE))


def sim_gripper01_to_piper_opening(value: float, threshold: float | None = None) -> float:
    sim_value = float(np.clip(float(value), 0.0, 1.0))
    raw_value = sim_value * _hardware_gripper_to_legacy_raw(SIM_GRIPPER_FULL_OPEN_M)
    hardware_value = _legacy_raw_gripper_to_hardware(raw_value)
    if threshold is not None and hardware_value < threshold:
        return 0.0
    return hardware_value


def _extract_replay_actions(actions: np.ndarray, action_names: list[str] | None) -> tuple[np.ndarray, list[int], list[str]]:
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2:
        raise ValueError(f"Expected a 2-D trajectory matrix, got shape {actions.shape}")

    if action_names:
        name_to_index = {name: idx for idx, name in enumerate(action_names)}
        selected_indices: list[int] = []
        selected_names: list[str] = []
        missing: list[str] = []
        for aliases in REPLAY_DIM_ALIASES:
            match = next((alias for alias in aliases if alias in name_to_index), None)
            if match is None:
                missing.append("/".join(aliases))
                continue
            selected_indices.append(name_to_index[match])
            selected_names.append(match)
        if missing:
            raise ValueError(
                "Trajectory action_names exist but do not cover the required 14 replay dimensions: "
                + ", ".join(missing)
            )
        return actions[:, selected_indices], selected_indices, selected_names

    if actions.shape[1] == len(REPLAY_DIM_NAMES):
        return actions.copy(), list(range(actions.shape[1])), list(REPLAY_DIM_NAMES)
    if actions.shape[1] > max(EMBODICHAIN_32D_REPLAY_INDICES):
        selected = list(EMBODICHAIN_32D_REPLAY_INDICES)
        return actions[:, selected], selected, list(REPLAY_DIM_NAMES)
    raise ValueError(
        f"Cannot infer the 14 replay dimensions from action shape {actions.shape}. "
        "Provide action_names or a 14-D trajectory."
    )


def _convert_sim_actions_to_piper_qpos(actions: np.ndarray, gripper_threshold: float | None) -> np.ndarray:
    qpos = np.asarray(actions, dtype=np.float64).copy()
    qpos[:, 6] = [sim_gripper01_to_piper_opening(value, gripper_threshold) for value in qpos[:, 6]]
    qpos[:, 13] = [sim_gripper01_to_piper_opening(value, gripper_threshold) for value in qpos[:, 13]]
    return qpos


def load_replay_trajectory(
    trajectory_path: str | Path,
    *,
    trajectory_key: str,
    gripper_threshold: float | None,
) -> LoadedTrajectory:
    npz_path = Path(trajectory_path).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if trajectory_key not in data:
            raise KeyError(f"Key {trajectory_key!r} not found in {npz_path}. Available keys: {list(data.files)}")
        source = np.asarray(data[trajectory_key], dtype=np.float64)
        action_names = [str(name) for name in data["action_names"].tolist()] if "action_names" in data else None
        replay_actions_sim, selected_indices, selected_action_names = _extract_replay_actions(source, action_names)
        replay_qpos = _convert_sim_actions_to_piper_qpos(replay_actions_sim, gripper_threshold)
        return LoadedTrajectory(
            trajectory_path=npz_path,
            trajectory_key=trajectory_key,
            source_shape=tuple(source.shape),
            source_action_dim=int(source.shape[-1]) if source.ndim >= 1 else 1,
            selected_indices=selected_indices,
            selected_action_names=selected_action_names,
            replay_actions_sim=replay_actions_sim,
            replay_qpos=replay_qpos,
            available_keys=list(data.files),
        )


def select_replay_window(actions: np.ndarray, start_index: int, max_steps: int) -> tuple[np.ndarray, int, int]:
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    if start_index >= len(actions):
        raise ValueError(f"--start-index {start_index} is out of range for trajectory length {len(actions)}")

    stop_index = len(actions) if max_steps == 0 else min(len(actions), start_index + max_steps)
    window = actions[start_index:stop_index]
    if len(window) == 0:
        raise ValueError("Selected replay window is empty")
    return window, start_index, stop_index


def wait_until_robot_ready(robot: "DualPiperSystem", timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            robot.read_state()
            return
        except Exception as exc:  # pragma: no cover - hardware dependent
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for dual Piper state after {timeout_s:.1f}s") from last_error


def sleep_until_next_action(start_s: float, fps: float) -> None:
    if fps <= 0.0:
        return
    remaining_s = (1.0 / fps) - (time.monotonic() - start_s)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def build_summary(
    args: argparse.Namespace,
    loaded: LoadedTrajectory,
    replay_window: np.ndarray,
    start_index: int,
    stop_index: int,
) -> dict[str, Any]:
    return {
        "trajectory_path": str(loaded.trajectory_path),
        "trajectory_key": loaded.trajectory_key,
        "available_keys": loaded.available_keys,
        "source_shape": list(loaded.source_shape),
        "source_action_dim": loaded.source_action_dim,
        "selected_indices": loaded.selected_indices,
        "selected_action_names": loaded.selected_action_names,
        "replay_action_names": list(REPLAY_DIM_NAMES),
        "replay_window": {
            "start_index": start_index,
            "stop_index_exclusive": stop_index,
            "steps": int(len(replay_window)),
        },
        "replay_fps": args.fps,
        "speed_percent": args.speed_percent,
        "gripper_threshold_m": args.gripper_threshold,
        "move_to_start": not args.skip_move_to_start,
        "dry_run": bool(args.dry_run),
        "first_action_sim": replay_window[0].tolist(),
        "first_action_piper": loaded.replay_qpos[start_index].tolist(),
        "last_action_piper": loaded.replay_qpos[stop_index - 1].tolist(),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.fps < 0.0:
        raise ValueError("--fps must be non-negative")
    if args.move_hz <= 0.0:
        raise ValueError("--move-hz must be positive")
    if not 0 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent must be in [0, 100]")
    if args.gripper_threshold is not None and args.gripper_threshold < 0.0:
        raise ValueError("--gripper-threshold must be non-negative")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")

    loaded = load_replay_trajectory(
        args.trajectory_npz,
        trajectory_key=args.trajectory_key,
        gripper_threshold=args.gripper_threshold,
    )
    replay_window, start_index, stop_index = select_replay_window(
        loaded.replay_qpos,
        args.start_index,
        args.max_steps,
    )
    summary = build_summary(
        args=args,
        loaded=loaded,
        replay_window=loaded.replay_actions_sim[start_index:stop_index],
        start_index=start_index,
        stop_index=stop_index,
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return

    from challenge_deploy.piper import DualPiperSystem

    runtime_config = _apply_runtime_overrides(load_config(args.config), args)
    robot = DualPiperSystem(
        left_can_name=runtime_config["robot"]["left"]["can_name"],
        right_can_name=runtime_config["robot"]["right"]["can_name"],
        commands_enabled=True,
        name="replay_piper_npz_trajectory",
    )

    robot.connect(read_only=False)
    try:
        wait_until_robot_ready(robot, args.ready_timeout)
        print('{"hardware_init": "enable_dual_piper"}', flush=True)
        if not robot.enable():
            print("Warning: Piper arm enable check did not report success; continuing anyway.", flush=True)
        wait_until_robot_ready(robot, args.ready_timeout)

        current_state = robot.read_state().qpos.astype(np.float64)
        print(json.dumps({"current_qpos": current_state.tolist()}, indent=2), flush=True)

        if not args.skip_move_to_start:
            print(json.dumps({"move_to_start_qpos": replay_window[0].tolist()}, indent=2), flush=True)
            robot.move_to_joint_positions(
                replay_window[0],
                hz=args.move_hz,
                step_sizes=runtime_config["runtime"]["arm_steps_length"],
                speed_percent=args.speed_percent,
            )
            if args.settle_seconds > 0.0:
                time.sleep(args.settle_seconds)

        replay_started_at = time.monotonic()
        for local_index, qpos in enumerate(replay_window):
            global_index = start_index + local_index
            action_start_s = time.monotonic()
            robot.set_joint_positions(qpos, speed_percent=args.speed_percent)
            if (
                local_index == 0
                or local_index == len(replay_window) - 1
                or (local_index + 1) % args.print_every == 0
            ):
                print(
                    json.dumps(
                        {
                            "step": {
                                "global_index": global_index,
                                "local_index": local_index,
                                "total_steps": len(replay_window),
                                "left_gripper_m": float(qpos[6]),
                                "right_gripper_m": float(qpos[13]),
                            }
                        }
                    ),
                    flush=True,
                )
            sleep_until_next_action(action_start_s, args.fps)

        print(
            json.dumps(
                {
                    "done": {
                        "executed_steps": len(replay_window),
                        "elapsed_s": float(time.monotonic() - replay_started_at),
                        "final_target_qpos": replay_window[-1].tolist(),
                    }
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
