#!/usr/bin/env python3
from pathlib import Path
import csv
import shutil

workspace_root = Path(__file__).resolve().parents[2]
source_root = workspace_root / "embodichain_ws/Embodied_Challenge/results/visualize_distribution"
target_root = workspace_root / "deploy/artifacts/train_distributions"
mapping_path = target_root / "embodichain_visualize_distribution_mapping.tsv"

task_specs = [
    ("Rearrangement", "rearr_H10", "rearr", "rearr", "rearr", "rearr"),
    ("beaker_mixer_duel", "beaker_mixer_H10", "beaker_mixer", "beaker_mixer_H10", "beaker_mixer", "beaker_mixer"),
    ("carry_basket", "carry_basket_H10", "carry_basket", "carry_basket", "carry_basket", "carry_basket"),
    ("click_button", "click_bell_H10", "click_bell", "click_bell_H10", "click_bell", "click_bell"),
    ("drawer_open_place", "open_drawer_H10", "open_drawer", "open_drawer_H10", "open_drawer", "open_drawer"),
    ("items_handover_place", "items_hand_over_place_H10", "items_handover_place", "items_handover_place_H10", "items_handover_place", "items_handover_place"),
    ("manipulate_pipette_one_beaker", "depress_pipette_H10", "depress_pipette", "depress_pipette_H10", "depress_pipette", "depress_pipette"),
    ("open_pan", "open_pan_H10", "open_pan", "open_pan", "open_pan", "open_pan"),
    ("pour_water_dual", "pour_dual_H10", "pour_dual", "pour_water_dual_H10", "pour_dual", "pour_water_dual"),
    ("sample_loading_dual", "insert_test_tube_H10", "insert_test_tube", "insert_test_tube", "insert_test_tube", "insert_test_tube"),
]

def overlay_name(repo_id: str) -> str:
    safe_text = repo_id.replace("/", "__").strip()
    safe_chars = "".join(char if char.isalnum() or char in "._-" else "_" for char in safe_text)
    while "__" in safe_chars:
        safe_chars = safe_chars.replace("__", "_")
    return safe_chars.strip("._-") + "_cam_high_first_frame_overlay.png"

target_root.mkdir(parents=True, exist_ok=True)
copied_paths: set[Path] = set()
with mapping_path.open("w", encoding="utf-8", newline="") as file_handle:
    writer = csv.writer(file_handle, delimiter="\t")
    writer.writerow(["source_env", "prompt_train_config", "openpi_sim_train_config", "repo_id", "target_png"])
    for source_env, prompt_suffix, pi0_sim_suffix, pi05_sim_suffix, pi0_repo_suffix, pi05_repo_suffix in task_specs:
        for model_name, sim_suffix, repo_suffix in [("pi0", pi0_sim_suffix, pi0_repo_suffix), ("pi05", pi05_sim_suffix, pi05_repo_suffix)]:
            prompt_name = f"{model_name}_embodichain_{prompt_suffix}"
            config_name = f"{model_name}_embodichain_{sim_suffix}"
            repo_id = f"embodichain_sim_data/cobotmagic_Sim_{repo_suffix}"
            source_path = source_root / source_env / "object_distribution_overlay.png"
            target_path = target_root / overlay_name(repo_id)
            if target_path not in copied_paths:
                shutil.copy2(source_path, target_path)
                copied_paths.add(target_path)
            writer.writerow([source_env, prompt_name, config_name, repo_id, str(target_path)])
