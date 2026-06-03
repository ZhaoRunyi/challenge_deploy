from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json
import threading
import time
from typing import Any, Callable, Sequence

import cv2
import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from hardware.constants import CAMERA_NAMES, PIPER_GRIPPER_FULL_OPEN_METERS
from hardware.piper import SinglePiperArm
from hardware.piper import DualPiperSystem
from hardware.realsense import RealSenseRig
from hardware.schemas import DualPiperState, PiperArmState
from rollout.recording import RolloutVideoRecorder, RecordingSchema


@dataclass(slots=True)
class TimestampedValue:
    timestamp_s: float
    value: Any


@dataclass(slots=True)
class HDF5TeleopCaptureFrame:
    timestamp_s: float
    puppet_state: DualPiperState
    master_state: DualPiperState
    puppet_pose_state: DualPiperState
    images: dict[str, np.ndarray]
    depth_images: dict[str, np.ndarray]
    source_timestamps: dict[str, float]


@dataclass(slots=True)
class HDF5TeleopLoadedEpisode:
    path: Path
    compressed: bool
    camera_names: tuple[str, ...]
    language_instruction: str | None
    qpos: np.ndarray
    qvel: np.ndarray
    effort: np.ndarray
    action: np.ndarray
    base_action: np.ndarray
    eef_quaternion: np.ndarray
    eef_6d: np.ndarray
    eef_left_time: np.ndarray
    eef_right_time: np.ndarray
    images: dict[str, list[np.ndarray]]
    depth_images: dict[str, list[np.ndarray]]
    source_timestamps: dict[str, np.ndarray]


class AsyncSampleQueue:
    def __init__(self, *, maxlen: int = 2000) -> None:
        self.samples: deque[TimestampedValue] = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, sample: TimestampedValue) -> None:
        with self.lock:
            self.samples.append(sample)

    def latest_timestamp_s(self) -> float | None:
        with self.lock:
            if not self.samples:
                return None
            return float(self.samples[-1].timestamp_s)

    def has_sample_at_or_after(self, timestamp_s: float) -> bool:
        with self.lock:
            return bool(self.samples) and float(self.samples[-1].timestamp_s) >= timestamp_s

    def pop_at_or_after(self, timestamp_s: float) -> TimestampedValue | None:
        with self.lock:
            while self.samples and float(self.samples[0].timestamp_s) < timestamp_s:
                self.samples.popleft()
            if not self.samples:
                return None
            return self.samples.popleft()

    def __len__(self) -> int:
        with self.lock:
            return len(self.samples)


def rotation_matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return np.concatenate([matrix[0, :2], matrix[1, :2], matrix[2, :2]], axis=0)


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    rot_6d = np.asarray(rot_6d, dtype=np.float64)
    if rot_6d.shape[-1] != 6:
        raise ValueError(f"Expected 6D rotation input, got shape {rot_6d.shape}")
    a1 = rot_6d[..., 0:5:2]
    a2 = rot_6d[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def abs_6d_to_abs_euler(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(20)
    left_xyz = action[0:3]
    left_6d = action[3:9]
    left_gripper = action[9]
    right_xyz = action[10:13]
    right_6d = action[13:19]
    right_gripper = action[19]

    rotation_cls = Rotation
    left_euler = rotation_cls.from_matrix(rotation_6d_to_matrix(left_6d)).as_euler("xyz", degrees=False)
    right_euler = rotation_cls.from_matrix(rotation_6d_to_matrix(right_6d)).as_euler("xyz", degrees=False)
    return np.concatenate(
        [
            left_xyz,
            left_euler,
            np.array([left_gripper], dtype=np.float64),
            right_xyz,
            right_euler,
            np.array([right_gripper], dtype=np.float64),
        ],
        axis=0,
    )


def arm_eef_quaternion(position: np.ndarray, gripper: float) -> np.ndarray:
    rotation_cls = Rotation
    quat = rotation_cls.from_euler("xyz", position[3:6], degrees=False).as_quat().astype(np.float64)
    return np.concatenate((position[:3], quat, np.array([gripper], dtype=np.float64)), axis=0)


def arm_eef_6d(position: np.ndarray, gripper: float) -> np.ndarray:
    rotation_cls = Rotation
    matrix = rotation_cls.from_euler("xyz", position[3:6], degrees=False).as_matrix()
    rot6d = rotation_matrix_to_6d(matrix)
    return np.concatenate((position[:3], rot6d, np.array([gripper], dtype=np.float64)), axis=0)


def dual_eef_quaternion(pose_state: DualPiperState, gripper_state: DualPiperState | None = None) -> np.ndarray:
    grippers = pose_state if gripper_state is None else gripper_state
    return np.concatenate(
        (
            arm_eef_quaternion(np.asarray(pose_state.left.end_pose, dtype=np.float64), float(grippers.left.qpos[6])),
            arm_eef_quaternion(np.asarray(pose_state.right.end_pose, dtype=np.float64), float(grippers.right.qpos[6])),
        ),
        axis=0,
    )


def dual_eef_6d(pose_state: DualPiperState, gripper_state: DualPiperState | None = None) -> np.ndarray:
    grippers = pose_state if gripper_state is None else gripper_state
    return np.concatenate(
        (
            arm_eef_6d(np.asarray(pose_state.left.end_pose, dtype=np.float64), float(grippers.left.qpos[6])),
            arm_eef_6d(np.asarray(pose_state.right.end_pose, dtype=np.float64), float(grippers.right.qpos[6])),
        ),
        axis=0,
    )


def infer_language_instruction(task_name: str, explicit: str | None = None) -> str:
    if explicit is not None:
        prompt = explicit.strip()
        if prompt:
            return prompt
    return task_name.replace("_", " ").strip() or task_name


def path_episode_index(path: Path) -> int | None:
    name = path.stem if path.suffix == ".hdf5" else path.name
    prefix = "episode_"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def next_episode_index(dataset_root: str | Path) -> int:
    root = Path(dataset_root)
    indices = []
    for path in root.glob("episode_*"):
        episode_index = path_episode_index(path)
        if episode_index is not None and (path.is_dir() or path.suffix == ".hdf5"):
            indices.append(episode_index)
    return max(indices) + 1 if indices else 0


def episode_directory(dataset_root: str | Path, episode_idx: int) -> Path:
    return Path(dataset_root) / f"episode_{episode_idx}"


def episode_base_path(dataset_root: str | Path, episode_idx: int) -> Path:
    return episode_directory(dataset_root, episode_idx) / f"episode_{episode_idx}"


def running_sentinel_path(dataset_root: str | Path, episode_idx: int) -> Path:
    return episode_directory(dataset_root, episode_idx) / f"episode_{episode_idx}_running.txt"


def encode_color_image(image: np.ndarray, *, jpeg_quality: int) -> np.ndarray:
    image = np.asarray(image)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg', ...) failed")
    return encoded


def encode_depth_image(depth_image: np.ndarray) -> np.ndarray:
    depth_image = np.asarray(depth_image)
    ok, encoded = cv2.imencode(".png", depth_image)
    if not ok:
        raise RuntimeError("cv2.imencode('.png', ...) failed")
    return encoded


def decode_image_value(value: Any, *, flags: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim >= 2:
        return array.copy()
    decoded = cv2.imdecode(array.astype(np.uint8), flags)
    if decoded is None:
        raise RuntimeError("cv2.imdecode(...) failed")
    return decoded


def decode_color_image(value: Any) -> np.ndarray:
    return decode_image_value(value, flags=cv2.IMREAD_COLOR)


def decode_depth_image(value: Any) -> np.ndarray:
    return decode_image_value(value, flags=cv2.IMREAD_UNCHANGED)


def positive_timestamp(timestamp_s: float) -> float | None:
    timestamp_s = float(timestamp_s)
    return timestamp_s if timestamp_s > 0.0 else None


PIPER_EE_ROTATION_PERIOD = 2.0 * np.pi


def normalized_gripper_value(value: float) -> float:
    return float(np.clip(float(value) / PIPER_GRIPPER_FULL_OPEN_METERS, 0.0, 1.0))


def stable_eef_positions(states: Sequence[DualPiperState]) -> list[np.ndarray]:
    poses = [np.stack((state.left.end_pose[:6], state.right.end_pose[:6]), axis=0).astype(np.float64) for state in states]
    for index in range(1, len(poses)):
        poses[index][:, 3:6] += PIPER_EE_ROTATION_PERIOD * np.round((poses[index - 1][:, 3:6] - poses[index][:, 3:6]) / PIPER_EE_ROTATION_PERIOD)
    return poses


def arm_state_vector_16(arm_state: PiperArmState, stable_pose: np.ndarray) -> np.ndarray:
    matrix = Rotation.from_euler("xyz", stable_pose[3:6], degrees=False).as_matrix()
    return np.concatenate((
        np.asarray(arm_state.qpos[:6], dtype=np.float64),
        np.asarray(stable_pose[:3], dtype=np.float64),
        rotation_matrix_to_6d(matrix),
        np.array([normalized_gripper_value(arm_state.qpos[6])], dtype=np.float64),
    ))


def dual_state_vector_32(joint_state: DualPiperState, stable_pose: np.ndarray) -> np.ndarray:
    return np.concatenate((
        arm_state_vector_16(joint_state.left, stable_pose[0]),
        arm_state_vector_16(joint_state.right, stable_pose[1]),
    ))


HDF5_TELEOP_VECTOR_NAMES = tuple(
    f"{arm}_{field}"
    for arm in ("left", "right")
    for field in (
        "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
        "ee_pos_x", "ee_pos_y", "ee_pos_z",
        "ee_rot6d_0", "ee_rot6d_1", "ee_rot6d_2", "ee_rot6d_3", "ee_rot6d_4", "ee_rot6d_5",
        "gripper",
    )
)


def dual_piper_state_moved(current: DualPiperState, previous: DualPiperState, tolerance: float) -> bool:
    arm_pairs = ((current.left, previous.left), (current.right, previous.right))
    for current_arm, previous_arm in arm_pairs:
        if np.any(np.abs(current_arm.qpos - previous_arm.qpos) > tolerance):
            return True
        if np.any(np.abs(current_arm.end_pose - previous_arm.end_pose) > tolerance):
            return True
    return False


class HDF5TeleopCollectionSource:
    def __init__(
        self,
        *,
        master_robot: DualPiperSystem,
        puppet_robot: DualPiperSystem,
        cameras: RealSenseRig,
        arm_sample_hz: float = 200.0,
        queue_maxlen: int = 2000,
    ) -> None:
        self.master_robot = master_robot
        self.puppet_robot = puppet_robot
        self.cameras = cameras
        self.arm_sample_hz = arm_sample_hz
        self.camera_names = tuple(cameras.serials)
        self.color_queues = {camera_name: AsyncSampleQueue(maxlen=queue_maxlen) for camera_name in self.camera_names}
        self.depth_queues = {camera_name: AsyncSampleQueue(maxlen=queue_maxlen) for camera_name in self.camera_names}
        self.arm_joint_queues = {
            "master_left": AsyncSampleQueue(maxlen=queue_maxlen),
            "master_right": AsyncSampleQueue(maxlen=queue_maxlen),
            "puppet_left": AsyncSampleQueue(maxlen=queue_maxlen),
            "puppet_right": AsyncSampleQueue(maxlen=queue_maxlen),
        }
        self.puppet_pose_queues = {
            "puppet_left": AsyncSampleQueue(maxlen=queue_maxlen),
            "puppet_right": AsyncSampleQueue(maxlen=queue_maxlen),
        }
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.last_error: Exception | None = None
        self.trace_lock = threading.Lock()
        self.sample_history: dict[str, list[float]] = {}
        self.selected_history: list[dict[str, float]] = []
        self.stationary_intervals: list[dict[str, float]] = []
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        self.cameras.start()
        self.stop_event.clear()
        for camera_name in self.camera_names:
            thread = threading.Thread(
                target=self.run_camera_sampler,
                args=(camera_name,),
                name=f"hdf5_teleop_camera_{camera_name}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

        arm_specs: tuple[tuple[str, SinglePiperArm, bool], ...] = (
            ("master_left", self.master_robot.left, False),
            ("master_right", self.master_robot.right, False),
            ("puppet_left", self.puppet_robot.left, True),
            ("puppet_right", self.puppet_robot.right, True),
        )
        for arm_name, arm, sample_pose in arm_specs:
            thread = threading.Thread(
                target=self.run_arm_sampler,
                args=(arm_name, arm, sample_pose),
                name=f"hdf5_teleop_arm_{arm_name}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)
        self.started = True

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=1.0)
        self.threads.clear()
        self.started = False

    def append_sample(self, topic_name: str, queue: AsyncSampleQueue, sample: TimestampedValue) -> None:
        queue.append(sample)
        with self.trace_lock:
            self.sample_history.setdefault(topic_name, []).append(float(sample.timestamp_s))

    def alignment_trace(self) -> dict[str, Any]:
        with self.trace_lock:
            return {
                "stack_timestamps_s": {name: list(values) for name, values in self.sample_history.items()},
                "selected_timestamps_s": [dict(values) for values in self.selected_history],
                "stationary_intervals_s": [dict(values) for values in self.stationary_intervals],
            }

    def append_stationary_interval(self, start_s: float, end_s: float) -> None:
        with self.trace_lock:
            self.stationary_intervals.append({"start_s": float(start_s), "end_s": float(end_s)})

    def reset_trace(self) -> None:
        with self.trace_lock:
            self.sample_history.clear()
            self.selected_history.clear()
            self.stationary_intervals.clear()

    def run_camera_sampler(self, camera_name: str) -> None:
        while not self.stop_event.is_set():
            try:
                capture = self.cameras.capture_camera_frame(camera_name)
                self.append_sample(f"camera_{camera_name}", self.color_queues[camera_name], TimestampedValue(capture.timestamp_s, capture.color_image))
                if self.cameras.enable_depth and capture.depth_image is not None:
                    depth_timestamp_s = capture.depth_timestamp_s if capture.depth_timestamp_s is not None else capture.timestamp_s
                    self.append_sample(f"depth_{camera_name}", self.depth_queues[camera_name], TimestampedValue(depth_timestamp_s, capture.depth_image))
            except Exception as exc:
                self.last_error = exc
                time.sleep(0.01)

    def run_arm_sampler(self, arm_name: str, arm: SinglePiperArm, sample_pose: bool) -> None:
        last_qpos_timestamp_s = 0.0
        last_pose_timestamp_s = 0.0
        period_s = 1.0 / self.arm_sample_hz if self.arm_sample_hz > 0.0 else 0.0
        while not self.stop_event.is_set():
            loop_start_s = time.monotonic()
            try:
                state = arm.read_state()
                qpos_timestamp_s = positive_timestamp(state.qpos_timestamp_s) or positive_timestamp(state.timestamp_s)
                if qpos_timestamp_s is not None and qpos_timestamp_s > last_qpos_timestamp_s:
                    self.append_sample(f"{arm_name}_joint", self.arm_joint_queues[arm_name], TimestampedValue(qpos_timestamp_s, state))
                    last_qpos_timestamp_s = qpos_timestamp_s
                pose_timestamp_s = positive_timestamp(state.end_pose_timestamp_s)
                if sample_pose and pose_timestamp_s is not None and pose_timestamp_s > last_pose_timestamp_s:
                    self.append_sample(f"{arm_name}_pose", self.puppet_pose_queues[arm_name], TimestampedValue(pose_timestamp_s, state))
                    last_pose_timestamp_s = pose_timestamp_s
            except Exception as exc:
                self.last_error = exc
                time.sleep(0.01)
            if period_s > 0.0:
                remaining_s = period_s - (time.monotonic() - loop_start_s)
                if remaining_s > 0.0:
                    time.sleep(remaining_s)

    def aligned_frame_time(self) -> float | None:
        image_timestamps = []
        for camera_name in self.camera_names:
            timestamp_s = self.color_queues[camera_name].latest_timestamp_s()
            if timestamp_s is None:
                return None
            image_timestamps.append(timestamp_s)
        if self.cameras.enable_depth:
            for camera_name in self.camera_names:
                timestamp_s = self.depth_queues[camera_name].latest_timestamp_s()
                if timestamp_s is None:
                    return None
                image_timestamps.append(timestamp_s)
        frame_time = min(image_timestamps)

        for queue in self.color_queues.values():
            if not queue.has_sample_at_or_after(frame_time):
                return None
        if self.cameras.enable_depth:
            for queue in self.depth_queues.values():
                if not queue.has_sample_at_or_after(frame_time):
                    return None
        for queue in self.arm_joint_queues.values():
            if not queue.has_sample_at_or_after(frame_time):
                return None
        for queue in self.puppet_pose_queues.values():
            if not queue.has_sample_at_or_after(frame_time):
                return None
        return frame_time

    def wait_until_ready(self, timeout_s: float = 10.0) -> bool:
        start_s = time.time()
        while time.time() - start_s < timeout_s:
            if self.last_error is not None:
                pass
            if self.aligned_frame_time() is not None:
                return True
            time.sleep(0.02)
        return False

    def get_frame(self) -> HDF5TeleopCaptureFrame | None:
        frame_time = self.aligned_frame_time()
        if frame_time is None:
            return None

        color_samples: dict[str, TimestampedValue] = {}
        depth_samples: dict[str, TimestampedValue] = {}
        for camera_name in self.camera_names:
            sample = self.color_queues[camera_name].pop_at_or_after(frame_time)
            if sample is None:
                return None
            color_samples[camera_name] = sample
        if self.cameras.enable_depth:
            for camera_name in self.camera_names:
                sample = self.depth_queues[camera_name].pop_at_or_after(frame_time)
                if sample is None:
                    return None
                depth_samples[camera_name] = sample

        arm_samples: dict[str, TimestampedValue] = {}
        for arm_name, queue in self.arm_joint_queues.items():
            sample = queue.pop_at_or_after(frame_time)
            if sample is None:
                return None
            arm_samples[arm_name] = sample

        pose_samples: dict[str, TimestampedValue] = {}
        for arm_name, queue in self.puppet_pose_queues.items():
            sample = queue.pop_at_or_after(frame_time)
            if sample is None:
                return None
            pose_samples[arm_name] = sample

        puppet_state = DualPiperState(
            left=arm_samples["puppet_left"].value,
            right=arm_samples["puppet_right"].value,
        )
        master_state = DualPiperState(
            left=arm_samples["master_left"].value,
            right=arm_samples["master_right"].value,
        )
        puppet_pose_state = DualPiperState(
            left=pose_samples["puppet_left"].value,
            right=pose_samples["puppet_right"].value,
        )
        source_timestamps = {f"camera_{name}": sample.timestamp_s for name, sample in color_samples.items()}
        source_timestamps.update({f"depth_{name}": sample.timestamp_s for name, sample in depth_samples.items()})
        source_timestamps.update({f"{name}_joint": sample.timestamp_s for name, sample in arm_samples.items()})
        source_timestamps.update({f"{name}_pose": sample.timestamp_s for name, sample in pose_samples.items()})
        source_timestamps["frame_time"] = frame_time
        with self.trace_lock:
            self.selected_history.append(dict(source_timestamps))

        return HDF5TeleopCaptureFrame(
            timestamp_s=frame_time,
            puppet_state=puppet_state,
            master_state=master_state,
            puppet_pose_state=puppet_pose_state,
            images={name: sample.value for name, sample in color_samples.items()},
            depth_images={name: sample.value for name, sample in depth_samples.items()},
            source_timestamps=source_timestamps,
        )


def collect_hdf5_teleop_episode(
    *,
    source: HDF5TeleopCollectionSource,
    max_timesteps: int | None,
    fps: float,
    countdown_seconds: int = 5,
    ready_timeout_s: float = 15.0,
    running_sentinel: Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
    start_source: bool = True,
    skip_stationary: bool = True,
    stationary_tolerance: float = 0.0005,
) -> list[HDF5TeleopCaptureFrame]:
    if max_timesteps is not None and max_timesteps < 1:
        raise ValueError("max_timesteps must be positive when set")
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    if start_source:
        source.start()
    if not source.wait_until_ready(timeout_s=ready_timeout_s):
        detail = f": {source.last_error}" if source.last_error is not None else ""
        raise RuntimeError(f"Timed out waiting for master/puppet/camera async queues{detail}")

    if running_sentinel is not None:
        running_sentinel.parent.mkdir(parents=True, exist_ok=True)
        running_sentinel.write_text("Delete this file to stop the current episode early.\n", encoding="utf-8")

    for seconds_left in range(max(0, countdown_seconds), 0, -1):
        print(f"{'*' * 20} Time (Seconds) Left {seconds_left} {'*' * 20}", flush=True)
        time.sleep(1.0)

    frames: list[HDF5TeleopCaptureFrame] = []
    target_frame_count = None if max_timesteps is None else max_timesteps + 1
    last_master_state: DualPiperState | None = None
    skipped_stationary = 0
    stationary_start_s: float | None = None
    stationary_end_s: float | None = None
    print_sync_failure = True

    def close_stationary_interval() -> None:
        nonlocal stationary_start_s, stationary_end_s
        if stationary_start_s is not None and stationary_end_s is not None:
            source.append_stationary_interval(stationary_start_s, stationary_end_s)
        stationary_start_s = None
        stationary_end_s = None

    try:
        while target_frame_count is None or len(frames) < target_frame_count:
            if running_sentinel is not None and not running_sentinel.exists():
                close_stationary_interval()
                break
            if stop_requested is not None and stop_requested():
                close_stationary_interval()
                break
            frame = source.get_frame()
            if frame is None:
                if print_sync_failure:
                    print("syn fail", flush=True)
                    print_sync_failure = False
                time.sleep(1.0 / fps)
                continue
            print_sync_failure = True
            if skip_stationary and last_master_state is not None and not dual_piper_state_moved(frame.master_state, last_master_state, stationary_tolerance):
                skipped_stationary += 1
                if stationary_start_s is None:
                    stationary_start_s = frame.timestamp_s
                stationary_end_s = frame.timestamp_s
                if skipped_stationary == 1 or skipped_stationary % 30 == 0:
                    print(f"Master arms are stationary, skipped {skipped_stationary} frame(s).", flush=True)
                time.sleep(1.0 / fps)
                continue
            close_stationary_interval()
            skipped_stationary = 0
            last_master_state = frame.master_state
            frames.append(frame)
            print(f"Frame data: {len(frames)}", flush=True)
            time.sleep(1.0 / fps)
    finally:
        close_stationary_interval()
        if running_sentinel is not None and running_sentinel.exists():
            running_sentinel.unlink()
    return frames

def save_hdf5_teleop_episode(
    *,
    output_path: str | Path,
    frames: Sequence[HDF5TeleopCaptureFrame],
    camera_names: Sequence[str],
    language_instruction: str,
    include_depth_images: bool = False,
    jpeg_quality: int = 95,
    action_from_state: bool = False,
) -> Path:
    if len(frames) < 2:
        raise ValueError("At least 2 frames are required to build an HDF5 teleop episode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path = output_path.with_suffix(".hdf5")
    data_size = len(frames) - 1
    frame0_time = float(frames[0].timestamp_s)
    source_timestamp_names = tuple(sorted(frames[0].source_timestamps.keys()))
    puppet_stable_poses = stable_eef_positions([frame.puppet_pose_state for frame in frames])
    master_stable_poses = stable_eef_positions([frame.master_state for frame in frames])

    with h5py.File(episode_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = True

        observations = root.create_group("observations")
        images_group = observations.create_group("images")
        for camera_name in camera_names:
            images_group.create_dataset(
                camera_name,
                (data_size,),
                dtype=h5py.vlen_dtype(np.dtype("uint8")),
                chunks=(1,),
            )

        if include_depth_images:
            depth_group = observations.create_group("images_depth")
            for camera_name in camera_names:
                depth_group.create_dataset(
                    camera_name,
                    (data_size,),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                    chunks=(1,),
                )
        else:
            depth_group = None

        source_timestamps_group = observations.create_group("source_timestamps")
        for name in source_timestamp_names:
            source_timestamps_group.create_dataset(name, (data_size,), dtype=np.float64)

        qpos = observations.create_dataset("qpos", (data_size, 32), dtype=np.float64)
        qvel = observations.create_dataset("qvel", (data_size, 14), dtype=np.float64)
        effort = observations.create_dataset("effort", (data_size, 14), dtype=np.float64)
        eef_quaternion = observations.create_dataset("eef_quaternion", (data_size, 16), dtype=np.float64)
        eef_6d = observations.create_dataset("eef_6d", (data_size, 20), dtype=np.float64)
        eef_left_time = observations.create_dataset("eef_left_time", (data_size,), dtype=np.float64)
        eef_right_time = observations.create_dataset("eef_right_time", (data_size,), dtype=np.float64)

        action = root.create_dataset("action", (data_size, 32), dtype=np.float64)
        base_action = root.create_dataset("base_action", (data_size, 2), dtype=np.float64)
        language = root.create_dataset("language_instruction", (1,), dtype=h5py.special_dtype(vlen=str))
        language[0] = language_instruction

        for index in range(data_size):
            observation_frame = frames[index]
            action_frame = frames[index + 1]
            qpos[index] = dual_state_vector_32(observation_frame.puppet_state, puppet_stable_poses[index])
            qvel[index] = observation_frame.puppet_state.qvel
            effort[index] = observation_frame.puppet_state.effort
            eef_quaternion[index] = dual_eef_quaternion(observation_frame.puppet_pose_state, observation_frame.puppet_state)
            eef_6d[index] = dual_eef_6d(observation_frame.puppet_pose_state, observation_frame.puppet_state)
            eef_left_time[index] = float(observation_frame.timestamp_s - frame0_time)
            eef_right_time[index] = float(observation_frame.timestamp_s - frame0_time)
            action_state = action_frame.puppet_state if action_from_state else action_frame.master_state
            action_poses = puppet_stable_poses if action_from_state else master_stable_poses
            action[index] = dual_state_vector_32(action_state, action_poses[index + 1])
            base_action[index] = np.array([0.0, 0.0], dtype=np.float64)

            for name in source_timestamp_names:
                source_timestamps_group[name][index] = float(observation_frame.source_timestamps.get(name, np.nan))

            for camera_name in camera_names:
                images_group[camera_name][index] = encode_color_image(
                    observation_frame.images[camera_name],
                    jpeg_quality=jpeg_quality,
                )
                if include_depth_images and depth_group is not None:
                    depth_group[camera_name][index] = encode_depth_image(
                        observation_frame.depth_images[camera_name]
                    )
    return episode_path


def save_hdf5_teleop_record_video(
    *,
    frames: Sequence[HDF5TeleopCaptureFrame],
    output_dir: str | Path,
    fps: float,
    name_prefix: str,
    action_from_state: bool = False,
    output_path: str | Path | None = None,
) -> Path | None:
    if len(frames) < 2:
        return None

    camera_names = tuple(frames[0].images)
    schema = RecordingSchema(
        camera_names=camera_names,
        action_names=HDF5_TELEOP_VECTOR_NAMES,
        state_names=HDF5_TELEOP_VECTOR_NAMES,
        used_action_names=frozenset(HDF5_TELEOP_VECTOR_NAMES),
    )
    recorder = RolloutVideoRecorder(
        output_dir=output_dir,
        schema=schema,
        fps=fps,
        name_prefix=name_prefix,
        output_path=output_path,
        keep_frames_in_memory=True,
        video_codec="libx264",
        video_output_params=("-preset", "veryfast", "-crf", "18"),
    )
    puppet_stable_poses = stable_eef_positions([frame.puppet_pose_state for frame in frames])
    master_stable_poses = stable_eef_positions([frame.master_state for frame in frames])
    for index in range(len(frames) - 1):
        observation_frame = frames[index]
        action_frame = frames[index + 1]
        recorder.record(
            images=observation_frame.images,
            state=dual_state_vector_32(observation_frame.puppet_state, puppet_stable_poses[index]),
            action=dual_state_vector_32(
                action_frame.puppet_state if action_from_state else action_frame.master_state,
                (puppet_stable_poses if action_from_state else master_stable_poses)[index + 1],
            ),
            timestamp_s=observation_frame.timestamp_s,
        )
    return recorder.finalize()


def offset_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean_s": float("nan"), "mean_abs_s": float("nan"), "max_abs_s": float("nan")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_s": float(np.mean(array)),
        "mean_abs_s": float(np.mean(np.abs(array))),
        "max_abs_s": float(np.max(np.abs(array))),
    }


def alignment_metrics(selected_timestamps: list[dict[str, float]]) -> dict[str, Any]:
    offsets: dict[str, list[float]] = {}
    for item in selected_timestamps:
        frame_time = item.get("frame_time")
        if frame_time is None:
            continue
        for topic_name, timestamp_s in item.items():
            if topic_name == "frame_time":
                continue
            offsets.setdefault(topic_name, []).append(float(timestamp_s) - float(frame_time))
    camera_offsets = {name: offset_summary(values) for name, values in offsets.items() if name.startswith("camera_")}
    source_offsets = {name: offset_summary(values) for name, values in offsets.items() if not name.startswith("camera_")}
    return {"camera_offsets_from_min_camera_s": camera_offsets, "source_offsets_from_min_camera_s": source_offsets}


def alignment_plot_indices(total_frames: int, plot_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    count = min(total_frames, max(1, int(plot_frames)))
    indices = np.linspace(0, total_frames - 1, count, dtype=int).tolist()
    return sorted(set(int(index) for index in indices))


def draw_alignment_plot(
    path: Path,
    stack_timestamps: dict[str, list[float]],
    selected_timestamps: list[dict[str, float]],
    plot_frames: int,
    stationary_intervals: list[dict[str, float]] | None = None,
) -> None:
    topic_names = sorted(stack_timestamps)
    if not topic_names:
        return
    all_times = [timestamp for values in stack_timestamps.values() for timestamp in values]
    if not all_times:
        return

    plotted_indices = alignment_plot_indices(len(selected_timestamps), plot_frames)
    if selected_timestamps:
        start_s = selected_timestamps[0].get("frame_time", min(all_times))
    else:
        start_s = min(all_times)
    end_s = max(all_times)
    span_s = max(end_s - start_s, 1e-6)

    fig_height = max(4.0, len(topic_names) * 0.7 + 2.0)
    fig, ax = plt.subplots(figsize=(20, fig_height), dpi=150)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(topic_names))))
    topic_y = {name: index for index, name in enumerate(topic_names)}
    stationary_intervals = stationary_intervals or []
    selected_frame_times = [float(item["frame_time"]) for item in selected_timestamps if "frame_time" in item]
    if len(selected_frame_times) >= 2:
        default_interval_width_s = float(np.median(np.diff(np.asarray(selected_frame_times, dtype=np.float64))))
    else:
        default_interval_width_s = 1.0 / 30.0

    for interval in stationary_intervals:
        interval_start = max(0.0, float(interval["start_s"]) - start_s)
        interval_end = max(interval_start + default_interval_width_s, float(interval["end_s"]) - start_s)
        for y in range(len(topic_names)):
            ax.fill_between(
                [interval_start, interval_end],
                y - 0.36,
                y + 0.36,
                color="lightgray",
                alpha=0.35,
                linewidth=0,
                zorder=0,
            )

    for topic_index, name in enumerate(topic_names):
        timestamps = np.asarray(stack_timestamps.get(name, []), dtype=np.float64)
        if timestamps.size == 0:
            continue
        valid_timestamps = timestamps[timestamps >= start_s]
        relative_times = valid_timestamps - start_s
        y_values = np.full(relative_times.shape, topic_y[name], dtype=np.float64)
        ax.scatter(relative_times, y_values, marker=".", color=colors[topic_index], s=10, alpha=0.22, zorder=2)

    selected_by_topic = {name: [] for name in topic_names}
    for item in selected_timestamps:
        for name in topic_names:
            if name in item:
                selected_by_topic[name].append(float(item[name]) - start_s)

    for name in topic_names:
        y = topic_y[name]
        times = np.asarray(selected_by_topic[name], dtype=np.float64)
        if times.size == 0:
            continue
        y_values = np.full(times.shape, y, dtype=np.float64)
        ax.scatter(times, y_values, marker="|", color=colors[y], s=90, linewidths=2, zorder=3)

    for frame_index in plotted_indices:
        item = selected_timestamps[frame_index]
        points = []
        for name in topic_names:
            if name in item:
                points.append((float(item[name]) - start_s, topic_y[name]))
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, color="red", alpha=0.85, linewidth=1.4, marker="o", markersize=3, zorder=4)
        ax.text(sum(xs) / len(xs), -0.55, f"Frm{frame_index}", color="red", fontsize=8, ha="center")

    ax.set_yticks(range(len(topic_names)))
    ax.set_yticklabels(topic_names, fontsize=9)
    ax.set_xlabel("Time (seconds) relative to first selected frame", fontsize=11)
    ax.set_title(
        f"Alignment trace\nframes={len(selected_timestamps)}, plotted={len(plotted_indices)}, span={span_s:.3f}s",
        fontsize=13,
    )
    ax.set_xlim(left=0.0)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_alignment_diagnostics(episode_path: str | Path, frames: Sequence[HDF5TeleopCaptureFrame], trace: dict[str, Any], plot_frames: int = 16) -> tuple[Path, Path]:
    episode_path = Path(episode_path)
    data_frames = max(0, len(frames) - 1)
    plot_frames = max(1, int(plot_frames))
    stem = f"{episode_path.stem}_alignment_plot{plot_frames}_frames{data_frames}"
    json_path = episode_path.with_name(stem + ".json")
    image_path = episode_path.with_name(stem + ".png")
    selected = [dict(frame.source_timestamps) for frame in frames[:data_frames]]
    stack = {name: [float(value) for value in values] for name, values in trace.get("stack_timestamps_s", {}).items()}
    stationary_intervals = [dict(values) for values in trace.get("stationary_intervals_s", [])]
    plotted_indices = alignment_plot_indices(len(selected), plot_frames)
    payload = {
        "episode_path": str(episode_path),
        "data_frames": data_frames,
        "plot_frames": plot_frames,
        "plotted_frame_indices": plotted_indices,
        "metrics": alignment_metrics(selected),
        "stack_timestamps_s": stack,
        "selected_timestamps_s": selected,
        "stationary_intervals_s": stationary_intervals,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_alignment_plot(image_path, stack, selected, plot_frames, stationary_intervals=stationary_intervals)
    return json_path, image_path

def load_hdf5_teleop_episode(path: str | Path) -> HDF5TeleopLoadedEpisode:
    path = Path(path)
    with h5py.File(path, "r") as root:
        compressed = bool(root.attrs.get("compress", False))
        camera_names = tuple(root["/observations/images"].keys())
        language_raw = root["/language_instruction"][()] if "/language_instruction" in root else None
        if language_raw is None:
            language_instruction = None
        elif isinstance(language_raw, np.ndarray):
            language_instruction = language_raw[0].decode("utf-8") if len(language_raw) > 0 and isinstance(language_raw[0], bytes) else (str(language_raw[0]) if len(language_raw) > 0 else None)
        else:
            language_instruction = language_raw.decode("utf-8") if isinstance(language_raw, bytes) else str(language_raw)

        images = {
            camera_name: [decode_color_image(value) if compressed else np.asarray(value).copy() for value in root[f"/observations/images/{camera_name}"][()]]
            for camera_name in camera_names
        }
        depth_images: dict[str, list[np.ndarray]] = {}
        if "/observations/images_depth" in root:
            for camera_name in root["/observations/images_depth"].keys():
                depth_images[camera_name] = [
                    decode_depth_image(value) if compressed else np.asarray(value).copy()
                    for value in root[f"/observations/images_depth/{camera_name}"][()]
                ]

        source_timestamps = {}
        if "/observations/source_timestamps" in root:
            for name in root["/observations/source_timestamps"].keys():
                source_timestamps[name] = root[f"/observations/source_timestamps/{name}"][()]

        return HDF5TeleopLoadedEpisode(
            path=path,
            compressed=compressed,
            camera_names=camera_names,
            language_instruction=language_instruction,
            qpos=root["/observations/qpos"][()],
            qvel=root["/observations/qvel"][()],
            effort=root["/observations/effort"][()],
            action=root["/action"][()],
            base_action=root["/base_action"][()],
            eef_quaternion=root["/observations/eef_quaternion"][()],
            eef_6d=root["/observations/eef_6d"][()],
            eef_left_time=root["/observations/eef_left_time"][()],
            eef_right_time=root["/observations/eef_right_time"][()],
            images=images,
            depth_images=depth_images,
            source_timestamps=source_timestamps,
        )


def compose_episode_vis_frame(
    cam_high: np.ndarray,
    cam_left_wrist: np.ndarray,
    cam_right_wrist: np.ndarray,
) -> np.ndarray:
    wrist_stack = np.concatenate((cam_left_wrist, cam_right_wrist), axis=0)
    wrist_stack = cv2.resize(wrist_stack, (0, 0), fx=0.5, fy=0.5)
    return np.concatenate((cam_high, wrist_stack), axis=1)


def save_hdf5_teleop_episode_preview(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    fps: int = 30,
    overwrite: bool = False,
) -> Path:
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".mp4") if output_path is None else Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    with h5py.File(input_path, "r") as root:
        image_group = root["/observations/images"]
        camera_names = tuple(image_group.keys())
        if set(("cam_high", "cam_left_wrist", "cam_right_wrist")) - set(camera_names):
            raise ValueError(
                f"Expected cam_high/cam_left_wrist/cam_right_wrist in episode, got {camera_names}"
            )
        frame_count = int(image_group["cam_high"].shape[0])

        def write_video(codec: str, output_params: list[str]) -> None:
            writer = imageio.get_writer(
                tmp_output,
                fps=fps,
                codec=codec,
                macro_block_size=1,
                output_params=output_params,
                ffmpeg_log_level="error",
            )
            try:
                for index in range(frame_count):
                    frame = compose_episode_vis_frame(
                        decode_color_image(image_group["cam_high"][index]),
                        decode_color_image(image_group["cam_left_wrist"][index]),
                        decode_color_image(image_group["cam_right_wrist"][index]),
                    )
                    writer.append_data(frame[..., ::-1])
            finally:
                writer.close()

        tmp_output = output_path.with_suffix(".tmp.mp4")
        try:
            write_video("libx264", ["-preset", "slow", "-crf", "0", "-pix_fmt", "yuv444p"])
        except Exception:
            tmp_output.unlink(missing_ok=True)
            write_video("libx264", ["-preset", "slow", "-crf", "8", "-pix_fmt", "yuv420p"])

    tmp_output.replace(output_path)
    return output_path
