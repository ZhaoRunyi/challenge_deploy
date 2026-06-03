from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resample a teleop HDF5 episode to a new fps using saved physical timestamps."
    )
    parser.add_argument("--file", type=Path, required=True, help="Input .hdf5 episode file.")
    parser.add_argument("--fps", type=float, default=10.0, help="Target output fps. Default: 10.")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: input file directory.",
    )
    return parser


def fps_filename_part(fps: float) -> str:
    return f"{float(fps):g}"


def output_path_for(input_path: Path, output_dir: Path, fps: float) -> Path:
    suffix = input_path.suffix if input_path.suffix else ".hdf5"
    return output_dir / f"{input_path.stem}_fps{fps_filename_part(fps)}{suffix}"


def nearest_index(sorted_timestamps: np.ndarray, timestamp_s: float) -> int:
    if len(sorted_timestamps) == 1:
        return 0

    index = int(np.searchsorted(sorted_timestamps, timestamp_s))
    if index <= 0:
        return 0
    if index >= len(sorted_timestamps):
        return len(sorted_timestamps) - 1

    left_index = index - 1
    right_index = index
    left_error = abs(timestamp_s - float(sorted_timestamps[left_index]))
    right_error = abs(timestamp_s - float(sorted_timestamps[right_index]))
    return left_index if left_error <= right_error else right_index


def select_resampled_indices(timestamps_s: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0.0:
        raise ValueError("--fps must be positive")
    if timestamps_s.ndim != 1 or len(timestamps_s) < 2:
        raise ValueError("Need at least two frame timestamps to resample")
    if not np.all(np.isfinite(timestamps_s)):
        raise ValueError("Frame timestamps contain NaN or Inf")
    if np.any(np.diff(timestamps_s) < 0.0):
        raise ValueError("Frame timestamps must be monotonically nondecreasing")

    step_s = 1.0 / float(fps)
    target_time_s = float(timestamps_s[0])
    end_time_s = float(timestamps_s[-1])
    selected_indices: list[int] = []

    while target_time_s <= end_time_s + 1e-9:
        source_index = nearest_index(timestamps_s, target_time_s)
        if not selected_indices or source_index != selected_indices[-1]:
            selected_indices.append(source_index)
        target_time_s += step_s

    if len(selected_indices) < 2:
        raise ValueError("Target fps produced fewer than two selected frames")
    return np.asarray(selected_indices, dtype=np.int64)


def timestamp_dataset(source_file: h5py.File) -> tuple[np.ndarray, str]:
    timestamp_paths = (
        "observations/source_timestamps/frame_time",
        "observations/eef_left_time",
        "observations/eef_right_time",
    )
    for path in timestamp_paths:
        if path in source_file:
            return np.asarray(source_file[path], dtype=np.float64), path
    raise KeyError(
        "No usable timestamp dataset found. Expected observations/source_timestamps/frame_time "
        "or observations/eef_left_time."
    )


def copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def copy_indexed_dataset(
    target_group: h5py.Group,
    dataset_name: str,
    source_dataset: h5py.Dataset,
    source_indices: np.ndarray,
    *,
    rebase_time: bool = False,
) -> h5py.Dataset:
    output_shape = (len(source_indices),) + tuple(source_dataset.shape[1:])
    create_kwargs = {"shape": output_shape, "dtype": source_dataset.dtype}
    if h5py.check_dtype(vlen=source_dataset.dtype) is not None:
        create_kwargs["chunks"] = (1,) + tuple(source_dataset.shape[1:])

    target_dataset = target_group.create_dataset(dataset_name, **create_kwargs)
    copy_attrs(source_dataset.attrs, target_dataset.attrs)

    if len(source_indices) == 0:
        return target_dataset

    if rebase_time:
        first_value = float(source_dataset[int(source_indices[0])])
        for output_index, source_index in enumerate(source_indices):
            target_dataset[output_index] = float(source_dataset[int(source_index)]) - first_value
        return target_dataset

    if h5py.check_dtype(vlen=source_dataset.dtype) is not None:
        for output_index, source_index in enumerate(source_indices):
            target_dataset[output_index] = source_dataset[int(source_index)]
    else:
        target_dataset[...] = source_dataset[source_indices]
    return target_dataset


def copy_full_dataset(
    target_group: h5py.Group,
    dataset_name: str,
    source_dataset: h5py.Dataset,
) -> h5py.Dataset:
    target_dataset = target_group.create_dataset(
        dataset_name,
        shape=source_dataset.shape,
        dtype=source_dataset.dtype,
    )
    copy_attrs(source_dataset.attrs, target_dataset.attrs)

    if source_dataset.shape == ():
        target_dataset[()] = source_dataset[()]
    elif h5py.check_dtype(vlen=source_dataset.dtype) is not None:
        for index in range(source_dataset.shape[0]):
            target_dataset[index] = source_dataset[index]
    else:
        target_dataset[...] = source_dataset[...]
    return target_dataset


def copy_full_item(target_group: h5py.Group, item_name: str, source_item: h5py.Group | h5py.Dataset) -> None:
    if isinstance(source_item, h5py.Dataset):
        copy_full_dataset(target_group, item_name, source_item)
        return

    child_group = target_group.create_group(item_name)
    copy_attrs(source_item.attrs, child_group.attrs)
    for child_name, child_item in source_item.items():
        copy_full_item(child_group, child_name, child_item)


def copy_resampled_group(
    target_group: h5py.Group,
    source_group: h5py.Group,
    source_indices: np.ndarray,
    source_row_count: int,
) -> None:
    copy_attrs(source_group.attrs, target_group.attrs)
    for item_name, source_item in source_group.items():
        if isinstance(source_item, h5py.Group):
            child_group = target_group.create_group(item_name)
            copy_resampled_group(child_group, source_item, source_indices, source_row_count)
            continue

        if source_item.shape and source_item.shape[0] == source_row_count:
            rebase_time = item_name in {"eef_left_time", "eef_right_time"}
            copy_indexed_dataset(
                target_group,
                item_name,
                source_item,
                source_indices,
                rebase_time=rebase_time,
            )
        else:
            copy_full_dataset(target_group, item_name, source_item)


def write_resampled_hdf5(input_path: Path, output_path: Path, fps: float) -> dict[str, object]:
    with h5py.File(input_path, "r") as source_file:
        timestamps_s, timestamp_key = timestamp_dataset(source_file)
        source_row_count = int(len(timestamps_s))
        selected_indices = select_resampled_indices(timestamps_s, fps)

        observation_indices = selected_indices[:-1]
        # In the teleop schema action[row] is the target from frame row + 1.
        action_indices = selected_indices[1:] - 1
        output_frame_count = int(len(observation_indices))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        if temporary_path.exists():
            temporary_path.unlink()

        with h5py.File(temporary_path, "w") as target_file:
            copy_attrs(source_file.attrs, target_file.attrs)
            target_file.attrs["resampled_from"] = str(input_path)
            target_file.attrs["resample_timestamp_key"] = timestamp_key
            target_file.attrs["resample_target_fps"] = float(fps)
            target_file.attrs["resample_source_frame_count"] = source_row_count
            target_file.attrs["resample_output_frame_count"] = output_frame_count

            for item_name, source_item in source_file.items():
                if item_name == "observations" and isinstance(source_item, h5py.Group):
                    observations_group = target_file.create_group(item_name)
                    copy_resampled_group(observations_group, source_item, observation_indices, source_row_count)
                    continue

                if (
                    isinstance(source_item, h5py.Dataset)
                    and source_item.shape
                    and source_item.shape[0] == source_row_count
                ):
                    indices = action_indices if item_name in {"action", "base_action"} else observation_indices
                    copy_indexed_dataset(target_file, item_name, source_item, indices)
                    continue

                copy_full_item(target_file, item_name, source_item)

        temporary_path.replace(output_path)

    duration_s = float(timestamps_s[-1] - timestamps_s[0])
    source_fps_estimate = (source_row_count - 1) / duration_s if duration_s > 0.0 else 0.0
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "timestamp_key": timestamp_key,
        "target_fps": float(fps),
        "source_frame_count": source_row_count,
        "selected_frame_count": int(len(selected_indices)),
        "output_frame_count": output_frame_count,
        "source_fps_estimate": source_fps_estimate,
    }


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.file.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else input_path.parent
    output_path = output_path_for(input_path, output_dir, args.fps)
    result = write_resampled_hdf5(input_path, output_path, args.fps)
    print(json.dumps({"hdf5_resample_result": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
