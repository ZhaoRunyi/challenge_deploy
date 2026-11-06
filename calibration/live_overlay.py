"""Large read-only real-camera overlay against a pinned historical frame."""
import argparse
import json
from pathlib import Path
import shutil
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import yaml

from .core import write_json
from hardware.realsense import RealSenseRig, require_realsense


def blend(reference, current, alpha):
    if reference.shape != current.shape:
        raise ValueError("Reference/live dimensions differ; refusing implicit registration")
    return cv2.addWeighted(reference, 1 - alpha, current, alpha, 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", default="configs/dual_piper_example.yaml")
    p.add_argument("--camera", default="cam_high", choices=["cam_high", "cam_left_wrist", "cam_right_wrist"])
    p.add_argument("--test-seconds", type=float)
    args = p.parse_args()
    reference = cv2.imread(str(args.reference / f"{args.camera}.png"))
    if reference is None:
        raise ValueError("Missing reference image")
    provenance = json.loads((args.reference / "reference.json").read_text())
    cfg = yaml.safe_load(Path(args.config).read_text())
    cam = cfg["cameras"]
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.reference, args.output / "reference")
    rig = RealSenseRig({args.camera: cam["serials"][args.camera]}, width=cam["width"], height=cam["height"], fps=cam["fps"])
    root = tk.Tk()
    root.title("RSC - Live / historical overlay - READ ONLY")
    root.geometry(f"{root.winfo_screenwidth()-80}x{root.winfo_screenheight()-100}+20+20")
    root.attributes("-zoomed", True)
    root.configure(bg="#171717")
    canvas = tk.Canvas(root, bg="#171717", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    bar = ttk.Frame(root, padding=10)
    bar.pack(fill="x")
    alpha = tk.DoubleVar(value=.5)
    ttk.Label(bar, text="Historical").pack(side="left")
    ttk.Scale(bar, from_=0, to=1, variable=alpha, length=450).pack(side="left", padx=10)
    ttk.Label(bar, text="Live").pack(side="left")
    status = tk.StringVar(value="Starting camera...")
    ttk.Label(root, textvariable=status, padding=8).pack(fill="x")
    live = None
    last_frame = 0.
    count = 0
    start = time.monotonic()
    item = canvas.create_image(0, 0, anchor="center")
    session = {"reference": provenance, "camera": args.camera, "serial": cam["serials"][args.camera],
               "motion_commands_enabled": False, "display": "raw images, same dimensions, no image registration"}

    def save():
        if live is None or time.monotonic() - last_frame > 1:
            status.set("No fresh live image; nothing saved")
            return
        dest = args.output / f"snapshot_{time.time_ns()}"
        dest.mkdir()
        for name, data in (("live_raw", live), ("reference", reference), ("overlay", blend(reference, live, alpha.get()))):
            if not cv2.imwrite(str(dest / f"{name}.png"), data):
                raise OSError("Snapshot write failed")
        write_json(dest / "snapshot.json", {**session, "alpha": alpha.get(), "saved_at": time.time(),
                                           "frame_age_s": time.monotonic() - last_frame})
        print(f"SAVED {dest}", flush=True)

    ttk.Button(bar, text="Save snapshot", command=save).pack(side="right", padx=8)
    ttk.Button(bar, text="Close", command=root.destroy).pack(side="right")
    root.bind("<Escape>", lambda _: root.destroy())
    root.bind("s", lambda _: save())
    root.bind("<F11>", lambda _: root.attributes("-fullscreen", not root.attributes("-fullscreen")))

    def tick():
        nonlocal live, last_frame, count
        try:
            frames = rig.pipelines[args.camera].poll_for_frames()
            if frames:
                frame = frames.get_color_frame()
                if frame:
                    live = np.asanyarray(frame.get_data()).copy()
                    last_frame = time.monotonic()
                    count += 1
            if live is not None:
                age = time.monotonic() - last_frame
                output = blend(reference, live, alpha.get())
                h, w = output.shape[:2]
                width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
                scale = min(width / w, height / h)
                img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)).resize((max(1, int(w*scale)), max(1, int(h*scale))))
                canvas.photo = ImageTk.PhotoImage(img)
                canvas.coords(item, width/2, height/2)
                canvas.itemconfigure(item, image=canvas.photo)
                status.set(f"{'LIVE' if age < 1 else 'STALE - CAMERA STOPPED'} | {args.camera} | episode {provenance['episode']}, frame {provenance['frame']} | live {alpha.get():.0%} | received {count} | age {age:.2f}s | no robot commands")
                if count % 30 == 0 or count == 1:
                    write_json(args.output / "live_status.json", {"received_frames": count, "frame_age_s": age, "updated_at": time.time()})
            if args.test_seconds and time.monotonic() - start >= args.test_seconds:
                save()
                root.destroy()
                return
        except Exception as exc:
            status.set(f"ERROR: {exc}")
            print(f"ERROR: {exc}", flush=True)
        root.after(60, tick)

    try:
        root.update()
        rig.start()
        intr = rig.pipelines[args.camera].get_active_profile().get_stream(require_realsense().stream.color).as_video_stream_profile().get_intrinsics()
        session["intrinsics"] = {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy, "coeffs": list(intr.coeffs), "model": str(intr.model)}
        write_json(args.output / "session.json", session)
        print(f"READY {args.output.resolve()}", flush=True)
        tick()
        root.mainloop()
    finally:
        rig.stop()


if __name__ == "__main__":
    main()
