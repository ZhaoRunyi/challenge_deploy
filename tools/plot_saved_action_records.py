from pathlib import Path
import imageio
import math
import re
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# MODEL_KIND = "openpi_sim"
# KEYWORD = "Pi0-Embodichain-ClickBell-30000-0507-randtable"

# MODEL_KIND = "openpi_sim"
# KEYWORD = "Pi0-Embodichain-ClickBell-align-abs-30000-0508_joints_chunk_sync_"

# MODEL_KIND = "openpi_sim"
# KEYWORD = "Pi0-Embodichain-ClickBell-align-H10-30000-0508_joints_chunk_sync_"

MODEL_KIND = "openpi_sim"
KEYWORD = "Pi0-Embodichain-ClickBell-30000-0505"

MODEL_KIND = "openpi_sim"
KEYWORD = "Pi0-Embodichain-Random-ClickBell-30000-0508"

MODEL_KIND = "openpi_sim"
KEYWORD = "Pi0-Embodichain-Random-Joint1Long-ClickBell-30000-0508"

MODEL_KIND = "openpi_sim"
KEYWORD = "Pi0-Embodichain-Random-Direct-ClickBell-30000-0508"

MODEL_KIND = "openpi_sim"
KEYWORD = "Pi0-Embodichain-Random-OpenDrawer-30000"

MODEL_KIND = "openpi_sim"
KEYWORD = "Pi0-Embodichain-Random-OpenPan-30000"

RECORD_ROOTS = {"openpi": ROOT / "artifacts" / "openpi_records", "openpi_sim": ROOT / "artifacts" / "openpi_sim_records", "motus": ROOT / "artifacts" / "motus_records"}
def safe_name(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())).strip("._-")
def legend_name(text: str, keyword: str) -> str:
    start = text.lower().find(keyword.lower())
    return text if start < 0 or keyword in {"", "*"} else f"{text[:start]}*{text[start + len(keyword):]}"
def crop_main_view(frame: np.ndarray) -> np.ndarray:
    top_height = int(frame.shape[0] * 0.6)
    panel_width = min(frame.shape[1], int(round(top_height * 4 / 3)))
    return frame[:top_height, :panel_width]
def pad_main_view(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: frame.shape[0], : frame.shape[1]] = frame
    return canvas
def main() -> None:
    record_root = RECORD_ROOTS[MODEL_KIND]
    keyword = KEYWORD.strip().lower()
    action_paths = sorted(path for path in record_root.rglob("*_actions.npz") if keyword in {"", "*"} or keyword in path.parent.name.lower())
    if not action_paths:
        raise FileNotFoundError(f"No action npz matched MODEL_KIND={MODEL_KIND!r} KEYWORD={KEYWORD!r} under {record_root}")
    loaded = []
    for action_path in action_paths:
        with np.load(action_path, allow_pickle=True) as data:
            action_names = tuple(str(name) for name in data["action_names"].tolist())
            actions = np.asarray(data["action_mean_trajectory"], dtype=np.float64)
        video_paths = sorted(action_path.parent.glob("*_videos.mp4"))
        loaded.append((legend_name(action_path.parent.name, KEYWORD), action_names, actions.reshape(1, -1) if actions.ndim == 1 else actions, video_paths[0] if video_paths else None))
    action_names = loaded[0][1]
    loaded = [item for item in loaded if item[1] == action_names]
    cols, rows = min(4, len(action_names)), math.ceil(len(action_names) / min(4, len(action_names)))
    figure, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 2.8), squeeze=False)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(2, len(loaded))))
    legend_lines = []
    for dim_index, action_name in enumerate(action_names):
        axis = axes[dim_index // cols][dim_index % cols]
        for record_index, (record_name, _, actions, _) in enumerate(loaded):
            line = axis.plot(actions[:, dim_index], color=colors[record_index], linewidth=1.1, alpha=0.9, label=record_name)[0]
            if dim_index == 0:
                legend_lines.append(line)
        axis.set_title(action_name)
        axis.grid(True, alpha=0.25)
    for dim_index in range(len(action_names), rows * cols):
        axes[dim_index // cols][dim_index % cols].axis("off")
    figure.subplots_adjust(left=0.04, right=0.82, top=0.97, bottom=0.05, wspace=0.25, hspace=0.35)
    figure.legend(handles=legend_lines, loc="center left", bbox_to_anchor=(0.83, 0.5), fontsize=7, frameon=False)
    output_path = record_root / f"{MODEL_KIND}_{safe_name(KEYWORD) or 'all'}_actions_overlay.png"
    figure.savefig(output_path, dpi=180)
    print(output_path)
    videos = []
    for record_name, _, _, video_path in loaded:
        if video_path is None:
            continue
        metadata = iio.immeta(video_path)
        if "duration" not in metadata or "fps" not in metadata:
            continue
        iterator = iter(iio.imiter(video_path))
        first_frame = crop_main_view(next(iterator))
        videos.append((record_name, iterator, first_frame, int(math.ceil(float(metadata["duration"]) * float(metadata["fps"])))))
    if not videos:
        return
    cols, rows = min(4, len(videos)), math.ceil(len(videos) / min(4, len(videos)))
    cell_height = max(first_frame.shape[0] for _, _, first_frame, _ in videos)
    cell_width = max(first_frame.shape[1] for _, _, first_frame, _ in videos)
    total_frames = max(frame_count for _, _, _, frame_count in videos)
    output_path = record_root / f"{MODEL_KIND}_{safe_name(KEYWORD) or 'all'}_main_views.mp4"
    writer = imageio.get_writer(output_path, fps=10)
    for frame_index in range(total_frames):
        tiles = []
        for _, iterator, last_frame, frame_count in videos:
            if frame_index > 0 and frame_index < frame_count:
                try:
                    last_frame[:] = crop_main_view(next(iterator))
                except StopIteration:
                    pass
            tiles.append(pad_main_view(last_frame, cell_height, cell_width))
        while len(tiles) < rows * cols:
            tiles.append(np.zeros((cell_height, cell_width, 3), dtype=np.uint8))
        writer.append_data(np.concatenate([np.concatenate(tiles[row * cols : (row + 1) * cols], axis=1) for row in range(rows)], axis=0))
    writer.close()
    print(output_path)
if __name__ == "__main__":
    main()
