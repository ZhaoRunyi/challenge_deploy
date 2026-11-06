from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import termios
import time
import tty

import cv2
import numpy as np
import yaml

from .core import board_pose, pose_matrix, solve_hand_eye, write_json
from .detector import detect


def key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def capture(args):
    from hardware.piper import SinglePiperArm
    from hardware.realsense import RealSenseRig, require_realsense
    cfg = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    board = json.loads(Path(args.board).read_text()) if not args.clean else None
    if not args.clean and args.camera == "cam_high" and args.mode != "eye-to-hand":
        raise ValueError("cam_high is fixed: use eye-to-hand")
    if not args.clean and "wrist" in args.camera and (args.mode != "eye-in-hand" or args.arm not in args.camera):
        raise ValueError("Wrist camera must use eye-in-hand with its own arm")
    cameras = cfg["cameras"]
    arms = {name: SinglePiperArm(name=name, can_name=cfg["robot"][name]["can_name"], commands_enabled=False)
            for name in ("left", "right")}
    rig = RealSenseRig(cameras["serials"], width=cameras["width"], height=cameras["height"], fps=cameras["fps"])
    pending = None
    window = None
    try:
        for arm in arms.values():
            arm.connect(read_only=True)
        rig.start()
        time.sleep(1)
        if args.window:
            from .capture_window import CaptureWindow
            window = CaptureWindow(rig, args.camera, board)
        print("READ ONLY. l: stage; s: save staged sample; q: discard staged sample; x: exit", flush=True)
        while True:
            k = window.next_key() if window else key()
            if k == "x":
                break
            if k == "q":
                if pending:
                    shutil.rmtree(pending)
                    pending = None
                print("Discarded staged sample; saved samples untouched.", flush=True)
                if window:
                    window.message("Discarded. l: stage | x: exit")
            if k == "s":
                if pending:
                    dest = output / (("clean_" if args.clean else "sample_") + time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns()))
                    pending.rename(dest)
                    pending = None
                    print(f"Saved {dest}", flush=True)
                    if window:
                        window.message(f"Saved {dest.name} | l: next sample")
                else:
                    print("No staged sample. Press l first.", flush=True)
                    if window:
                        window.message("No staged sample. Press l first.")
            if k != "l":
                continue
            if pending:
                print("Save or discard the current sample first.", flush=True)
                continue
            pending = Path(tempfile.mkdtemp(prefix=".pending_", dir=output))
            if window:
                window.message("Capturing fresh images and robot feedback; hold still...")
            try:
                before = {n: a.read_state() for n, a in arms.items()}
                metadata = {"mode": args.mode, "camera": args.camera, "arm": args.arm,
                            "reference": args.reference, "timestamp": time.time(), "cameras": {}, "arms": {}}
                for name, pipeline in rig.pipelines.items():
                    # Discard queued images, then wait for a newly delivered frame.
                    pipeline.poll_for_frames()
                    f = rig.capture_camera_frame(name, timeout_ms=5000)
                    if abs(time.time() - f.timestamp_s) > 0.5:
                        raise ValueError("Stale camera frame")
                    intr = pipeline.get_active_profile().get_stream(require_realsense().stream.color).as_video_stream_profile().get_intrinsics()
                    if any(abs(c) > 1e-12 for c in intr.coeffs) and str(intr.model) not in ("distortion.none", "distortion.brown_conrady"):
                        raise ValueError(f"Unsupported distortion: {intr.model}")
                    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
                    rect = cv2.undistort(f.color_image, K, np.array(intr.coeffs))
                    for suffix, image in (("raw", f.color_image), ("rect", rect)):
                        if not cv2.imwrite(str(pending / f"{name}_{suffix}.png"), image):
                            raise OSError("Image write failed")
                    metadata["cameras"][name] = {"serial": cameras["serials"][name], "width": intr.width,
                        "height": intr.height, "camera_matrix": K.tolist(), "distortion": list(intr.coeffs),
                        "distortion_model": str(intr.model), "timestamp": f.timestamp_s,
                        "device_timestamp_ms": f.color_frame_timestamp_ms}
                for name, arm in arms.items():
                    state = arm.read_state()
                    old = before[name]
                    for s in (old, state):
                        if min(s.qpos_timestamp_s, s.end_pose_timestamp_s) <= 0 or time.time() - min(s.qpos_timestamp_s, s.end_pose_timestamp_s) > 1.0:
                            raise ValueError(f"Stale feedback from {name}")
                    if np.max(np.abs(state.qpos[:6] - old.qpos[:6])) > 0.005 or np.linalg.norm(state.end_pose[:3] - old.end_pose[:3]) > 0.002:
                        raise ValueError("Arm moved during capture; hold still")
                    refs = {"eef": pose_matrix(state.end_pose).tolist()}
                    for i, pose in enumerate(arm.interface.GetFK(mode="feedback")):
                        p = np.asarray(pose, dtype=float)[:6]
                        refs[f"link{i+1}"] = pose_matrix(np.r_[p[:3] / 1000, np.deg2rad(p[3:])]).tolist()
                    metadata["arms"][name] = {"qpos": state.qpos.tolist(), "end_pose": state.end_pose.tolist(),
                        "references": refs, "feedback_timestamp": state.end_pose_timestamp_s,
                        "qpos_before": old.qpos.tolist(), "end_pose_before": old.end_pose.tolist(),
                        "feedback_timestamp_before": old.end_pose_timestamp_s}
                if not args.clean:
                    points = detect(pending / f"{args.camera}_rect.png")
                    K = np.array(metadata["cameras"][args.camera]["camera_matrix"])
                    T, rms, ids = board_pose(points, board, K)
                    metadata.update(T_camera_board=T.tolist(), reprojection_rms_px=rms, detected_ids=ids,
                                    board=board, detections=points)
                    print(f"PnP: {len(ids)} markers, RMS {rms:.3f}px", flush=True)
                write_json(pending / "sample.json", metadata)
                print(f"Staged {pending}; inspect images, then s or q.", flush=True)
                if window:
                    window.message(f"Staged {pending.name} | s: save | q: discard (preview remains live)")
            except Exception as e:
                shutil.rmtree(pending)
                pending = None
                print(f"Rejected: {e}", flush=True)
                if window:
                    window.message(f"Rejected: {e}")
    finally:
        if window:
            window.close()
        if pending:
            shutil.rmtree(pending)
        rig.stop()
        for arm in arms.values():
            arm.disconnect()


def solve(args):
    samples = [json.loads(p.read_text()) for p in sorted(Path(args.session).glob("sample_*/sample.json"))]
    if not samples:
        raise ValueError("No saved samples")
    first = samples[0]
    for s in samples:
        for field in ("mode", "camera", "arm", "reference", "board"):
            if s[field] != first[field]:
                raise ValueError(f"Mixed session {field}")
        for field in ("serial", "camera_matrix", "width", "height"):
            if s["cameras"][first["camera"]][field] != first["cameras"][first["camera"]][field]:
                raise ValueError(f"Camera changed: {field}")
    A = [s["arms"][s["arm"]]["references"][s["reference"]] for s in samples]
    C = [s["T_camera_board"] for s in samples]
    X, quality = solve_hand_eye(A, C, first["mode"])
    result = {"mode": first["mode"], "camera_name": first["camera"], "arm_name": first["arm"],
              "reference": first["reference"], "convention": "T_A_B maps B coordinates into A; meters; OpenCV optical x-right y-down z-forward",
              "camera_intrinsics": first["cameras"][first["camera"]], "sample_count": len(samples), "quality": quality,
              "T_reference_camera" if first["mode"] == "eye-in-hand" else "T_base_camera": X.tolist()}
    write_json(Path(args.session) / "calibration_result.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("board")
    p.add_argument("--output", default="artifacts/calibration/apriltag")
    p = sub.add_parser("capture")
    p.add_argument("--config", default="configs/dual_piper_example.yaml")
    p.add_argument("--board", default="artifacts/calibration/apriltag/board.json")
    p.add_argument("--output", required=True)
    p.add_argument("--camera", choices=["cam_high", "cam_left_wrist", "cam_right_wrist"], required=True)
    p.add_argument("--arm", choices=["left", "right"], required=True)
    p.add_argument("--reference", choices=["eef"] + [f"link{i}" for i in range(1, 7)], default="eef")
    p.add_argument("--mode", choices=["eye-in-hand", "eye-to-hand"], required=True)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--window", action="store_true", help="Live camera preview with detected Tag and pose axes")
    p = sub.add_parser("solve")
    p.add_argument("session")
    args = parser.parse_args()
    if args.command == "board":
        from .board import generate
        generate(args.output)
    elif args.command == "capture":
        capture(args)
    else:
        solve(args)


if __name__ == "__main__":
    main()
