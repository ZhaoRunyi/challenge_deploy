from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import multiprocessing
from pathlib import Path
import queue
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

import h5py
import numpy as np

from hardware.schemas import DualPiperState
from teleop.hdf5_teleop import (
    add_arm_source_timestamps,
    atomic_hdf5_file,
    dual_arm_array,
    dual_eef_quaternion,
    dual_state_vector_32,
    encode_color_image,
    stable_eef_positions,
)


class RolloutHDF5SessionCollector:
    """Build one immutable HDF5 payload per interactive episode."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        language_instruction: str,
        camera_names: Iterable[str],
        control_mode: str,
        can_topology: str,
        first_episode_index: int,
        jpeg_quality: int = 95,
    ) -> None:
        if first_episode_index < 0:
            raise ValueError("first_episode_index must be non-negative")
        self.dataset_root = Path(dataset_root)
        self.language_instruction = str(language_instruction)
        self.camera_names = tuple(str(name) for name in camera_names)
        self.control_mode = str(control_mode)
        self.can_topology = str(can_topology)
        self.next_episode_index = int(first_episode_index)
        self.jpeg_quality = int(jpeg_quality)
        self.buffer: list[RolloutHDF5Step] = []
        self.active_episode_number: int | None = None

    def begin_episode(self, episode_number: int) -> None:
        if self.active_episode_number is not None:
            raise RuntimeError(
                f"episode {self.active_episode_number} is already active"
            )
        self.buffer.clear()
        self.active_episode_number = int(episode_number)

    def validate_snapshot(self, snapshot: Any) -> None:
        slave_state = getattr(snapshot, "state", None)
        images = getattr(snapshot, "images", None)
        if not isinstance(slave_state, DualPiperState):
            raise TypeError("stable rollout snapshot must contain DualPiperState")
        if not isinstance(images, dict):
            raise TypeError("stable rollout snapshot images must be a dictionary")
        missing_cameras = [name for name in self.camera_names if name not in images]
        if missing_cameras:
            raise RuntimeError(f"stable rollout snapshot is missing cameras {missing_cameras}")

    def record_stable_step(
        self,
        *,
        snapshot_before_command: Any,
        action_state: DualPiperState,
        is_intervention: bool,
    ) -> None:
        if self.active_episode_number is None:
            raise RuntimeError("begin_episode() must be called before recording stable steps")
        slave_state = getattr(snapshot_before_command, "state", None)
        images = getattr(snapshot_before_command, "images", None)
        if not isinstance(action_state, DualPiperState):
            raise TypeError("stable rollout action_state must be DualPiperState")
        timestamp_s = float(snapshot_before_command.timestamp_s)
        self.buffer.append(
            RolloutHDF5Step(
                timestamp_s=timestamp_s,
                slave_state=deepcopy(slave_state),
                action_state=deepcopy(action_state),
                images={name: np.asarray(images[name]).copy() for name in self.camera_names},
                source_timestamps=source_timestamps_for_step(
                    timestamp_s=timestamp_s,
                    slave_state=slave_state,
                    camera_names=self.camera_names,
                    snapshot_source_timestamps=getattr(
                        snapshot_before_command,
                        "source_timestamps",
                        None,
                    ),
                ),
                is_intervention=bool(is_intervention),
            )
        )

    def finish_episode(
        self,
        episode_number: int,
        stop_reason: str,
        partial: bool,
    ) -> RolloutHDF5Episode | None:
        if self.active_episode_number != int(episode_number):
            raise RuntimeError(
                f"finishing episode {episode_number}, active episode is {self.active_episode_number}"
            )
        steps = tuple(self.buffer)
        self.active_episode_number = None
        self.buffer.clear()
        if not steps:
            return None
        episode_index = self.next_episode_index
        self.next_episode_index += 1
        output_path = self.dataset_root / f"episode_{episode_index}"
        return RolloutHDF5Episode(
            output_path=output_path,
            camera_names=self.camera_names,
            language_instruction=self.language_instruction,
            steps=steps,
            control_mode=self.control_mode,
            can_topology=self.can_topology,
            stop_reason=str(stop_reason),
            partial=bool(partial),
            jpeg_quality=self.jpeg_quality,
        )


@dataclass(frozen=True, slots=True)
class RolloutHDF5Step:
    timestamp_s: float
    slave_state: DualPiperState
    action_state: DualPiperState
    images: dict[str, np.ndarray]
    source_timestamps: dict[str, float]
    is_intervention: bool


@dataclass(frozen=True, slots=True)
class RolloutHDF5Episode:
    output_path: Path
    camera_names: tuple[str, ...]
    language_instruction: str
    steps: tuple[RolloutHDF5Step, ...]
    control_mode: str
    can_topology: str
    stop_reason: str
    partial: bool = False
    jpeg_quality: int = 95
    temporary_path: Path | None = None


def save_rollout_hdf5_episode(episode: RolloutHDF5Episode) -> Path:
    if not episode.steps:
        raise ValueError("At least one stable rollout step is required")
    output_path = Path(episode.output_path).with_suffix(".hdf5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    step_count = len(episode.steps)
    frame0_time = float(episode.steps[0].timestamp_s)
    source_names = tuple(
        sorted({name for step in episode.steps for name in step.source_timestamps})
    )
    slave_poses = stable_eef_positions([step.slave_state for step in episode.steps])
    action_poses = stable_eef_positions([step.action_state for step in episode.steps])

    with atomic_hdf5_file(
        output_path,
        temporary_path=episode.temporary_path,
        rdcc_nbytes=1024**2 * 2,
    ) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = True
        root.attrs["schema_version"] = 2
        root.attrs["control_mode"] = episode.control_mode
        root.attrs["can_topology"] = episode.can_topology
        root.attrs["stop_reason"] = episode.stop_reason
        root.attrs["partial"] = bool(episode.partial)

        observations = root.create_group("observations")
        images_group = observations.create_group("images")
        for camera_name in episode.camera_names:
            images_group.create_dataset(
                camera_name,
                (step_count,),
                dtype=h5py.vlen_dtype(np.dtype("uint8")),
                chunks=(1,),
            )
        source_group = observations.create_group("source_timestamps")
        for name in source_names:
            source_group.create_dataset(name, (step_count,), dtype=np.float64)

        qpos = observations.create_dataset("qpos", (step_count, 14), dtype=np.float64)
        qpos_feedback = observations.create_dataset("qpos_feedback", (step_count, 14), dtype=np.float64)
        qpos_command = observations.create_dataset("qpos_command", (step_count, 14), dtype=np.float64)
        qvel = observations.create_dataset("qvel", (step_count, 14), dtype=np.float64)
        effort = observations.create_dataset("effort", (step_count, 14), dtype=np.float64)
        end_pose = observations.create_dataset("end_pose", (step_count, 14), dtype=np.float64)
        eef_quaternion = observations.create_dataset("eef_quaternion", (step_count, 16), dtype=np.float64)
        eef_left_time = observations.create_dataset("eef_left_time", (step_count,), dtype=np.float64)
        eef_right_time = observations.create_dataset("eef_right_time", (step_count,), dtype=np.float64)
        state = root.create_dataset("state", (step_count, 32), dtype=np.float64)
        action = root.create_dataset("action", (step_count, 32), dtype=np.float64)
        is_intervention = root.create_dataset("is_intervention", (step_count,), dtype=np.bool_)
        language = root.create_dataset("language_instruction", (1,), dtype=h5py.special_dtype(vlen=str))
        language[0] = episode.language_instruction

        for index, step in enumerate(episode.steps):
            slave_state = step.slave_state
            qpos[index] = slave_state.qpos
            qpos_feedback[index] = dual_arm_array(slave_state, "qpos_feedback")
            qpos_command[index] = dual_arm_array(slave_state, "qpos_command")
            qvel[index] = slave_state.qvel
            effort[index] = slave_state.effort
            end_pose[index] = dual_arm_array(slave_state, "end_pose")
            eef_quaternion[index] = dual_eef_quaternion(slave_state, slave_state)
            eef_left_time[index] = float(step.timestamp_s - frame0_time)
            eef_right_time[index] = float(step.timestamp_s - frame0_time)
            state[index] = dual_state_vector_32(slave_state, slave_poses[index])
            action[index] = dual_state_vector_32(step.action_state, action_poses[index])
            is_intervention[index] = bool(step.is_intervention)
            for name in source_names:
                source_group[name][index] = float(step.source_timestamps.get(name, np.nan))
            for camera_name in episode.camera_names:
                if camera_name not in step.images:
                    raise KeyError(f"Missing required camera {camera_name!r} at step {index}")
                images_group[camera_name][index] = encode_color_image(
                    step.images[camera_name],
                    jpeg_quality=episode.jpeg_quality,
                )
    return output_path


@dataclass(slots=True)
class _WriterHandle:
    job_id: str
    episode: RolloutHDF5Episode
    process: multiprocessing.Process
    result_queue: Any


def _writer_entry(episode: RolloutHDF5Episode, result_queue: Any) -> None:
    try:
        path = save_rollout_hdf5_episode(episode)
        result_queue.put({"ok": True, "path": str(path), "steps": len(episode.steps)})
    except BaseException as exc:
        result_queue.put({"ok": False, "error": repr(exc)})


class RolloutHDF5WriterPool:
    """A bounded, killable two-process HDF5 writer pool with no pending queue."""

    def __init__(
        self,
        *,
        max_writers: int = 2,
        writer_target: Callable[[RolloutHDF5Episode, Any], None] = _writer_entry,
    ) -> None:
        if max_writers <= 0:
            raise ValueError("max_writers must be positive")
        self.max_writers = int(max_writers)
        self.writer_target = writer_target
        self.context = multiprocessing.get_context("spawn")
        self.handles: list[_WriterHandle] = []
        self._pending_results: list[dict[str, Any]] = []

    @property
    def active_count(self) -> int:
        self._pending_results.extend(self._reap_finished())
        return len(self.handles)

    def can_start_episode(self) -> bool:
        return self.active_count < self.max_writers

    def submit(self, episode: RolloutHDF5Episode) -> str:
        self._pending_results.extend(self._reap_finished())
        if len(self.handles) >= self.max_writers:
            raise RuntimeError("All HDF5 writer slots are busy; no pending queue is allowed")
        job_id = uuid4().hex
        output_path = Path(episode.output_path).with_suffix(".hdf5")
        temporary_path = output_path.with_name(
            f".{output_path.name}.{job_id}.tmp"
        )
        writer_episode = replace(episode, temporary_path=temporary_path)
        result_queue = self.context.Queue(maxsize=1)
        process = self.context.Process(
            target=self.writer_target,
            args=(writer_episode, result_queue),
            daemon=False,
        )
        process.start()
        self.handles.append(
            _WriterHandle(
                job_id=job_id,
                episode=writer_episode,
                process=process,
                result_queue=result_queue,
            )
        )
        return job_id

    def _reap_finished(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        active: list[_WriterHandle] = []
        for handle in self.handles:
            if handle.process.is_alive():
                active.append(handle)
                continue
            handle.process.join()
            try:
                result = handle.result_queue.get(timeout=0.2)
            except queue.Empty:
                result = {"ok": False, "error": f"writer exited with code {handle.process.exitcode}"}
            result.update(
                {
                    "job_id": handle.job_id,
                    "output_path": str(
                        Path(handle.episode.output_path).with_suffix(".hdf5")
                    ),
                }
            )
            results.append(result)
            handle.result_queue.close()
        self.handles = active
        return results

    def poll(self) -> list[dict[str, Any]]:
        results = self._pending_results
        self._pending_results = []
        results.extend(self._reap_finished())
        return results

    def terminate_oldest(self, *, reason: str = "terminated_by_user") -> dict[str, Any] | None:
        self._pending_results.extend(self._reap_finished())
        if not self.handles:
            return None
        handle = self.handles[0]
        handle.process.terminate()
        handle.process.join(timeout=5.0)
        if handle.process.is_alive():
            handle.process.kill()
            handle.process.join(timeout=5.0)
        self.handles.remove(handle)
        output_path = Path(handle.episode.output_path).with_suffix(".hdf5")
        temporary_path = handle.episode.temporary_path
        if temporary_path is None:
            raise RuntimeError("writer handle is missing its temporary path")
        completed_result: dict[str, Any] | None = None
        try:
            queued_result = handle.result_queue.get_nowait()
            if bool(queued_result.get("ok")) and output_path.exists():
                completed_result = dict(queued_result)
        except queue.Empty:
            if output_path.exists() and not temporary_path.exists():
                completed_result = {
                    "ok": True,
                    "path": str(output_path),
                    "steps": len(handle.episode.steps),
                    "completed_during_termination": True,
                }
        handle.result_queue.close()
        if completed_result is not None:
            completed_result.update(
                {
                    "job_id": handle.job_id,
                    "output_path": str(output_path),
                }
            )
            return completed_result
        if temporary_path.exists():
            temporary_path.unlink()
        failure_path = output_path.with_name(f"{output_path.name}.save_failed.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "ok": False,
            "job_id": handle.job_id,
            "output_path": str(output_path),
            "reason": reason,
            "exitcode": handle.process.exitcode,
        }
        failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return failure

    def close(self, *, terminate: bool = False) -> list[dict[str, Any]]:
        if terminate:
            failures = []
            while self.handles:
                failure = self.terminate_oldest(reason="pool_shutdown")
                if failure is not None:
                    failures.append(failure)
            return self.poll() + failures
        for handle in tuple(self.handles):
            handle.process.join()
        return self.poll()


def extend_rollout_limit(initial_steps: int, current_limit: int) -> int:
    if initial_steps <= 0:
        raise ValueError("Unlimited rollouts do not have an extendable step limit")
    increment = (int(initial_steps) + 1) // 2
    return int(current_limit) + increment


def source_timestamps_for_step(
    *,
    timestamp_s: float,
    slave_state: DualPiperState,
    camera_names: Iterable[str],
    snapshot_source_timestamps: Mapping[str, float] | None = None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    add_arm_source_timestamps(values, "slave_left", slave_state.left)
    add_arm_source_timestamps(values, "slave_right", slave_state.right)
    values["frame_time"] = float(timestamp_s)
    captured = snapshot_source_timestamps or {}
    for camera_name in camera_names:
        legacy_name = f"camera_{camera_name}"
        color_name = f"camera_{camera_name}_color_time"
        camera_timestamp_s = float(
            captured.get(
                color_name,
                captured.get(legacy_name, timestamp_s),
            )
        )
        values[legacy_name] = camera_timestamp_s
        values[color_name] = camera_timestamp_s
    return values
