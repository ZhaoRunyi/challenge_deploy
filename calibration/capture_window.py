"""Read-only capture preview; OpenCV processing with a Tk display."""
import select
import sys
import termios
import time
import tty

import cv2
import numpy as np

from .core import board_pose
from .detector import detect


def annotate(image, K, board):
    canvas = image.copy()
    if board is None:
        return canvas, "Clean capture"
    points = detect(image)
    for ident, corners in points.items():
        polygon = np.array(corners, dtype=np.int32)
        cv2.polylines(canvas, [polygon], True, (0, 220, 255), 2)
        cv2.putText(canvas, f"ID {ident}", tuple(polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2)
    try:
        T, rms, _ = board_pose(points, board, K)
    except ValueError as exc:
        return canvas, str(exc)
    length = float(np.linalg.norm(np.array(board["corners_m"])[0] - np.array(board["corners_m"])[1])) / 2
    cv2.drawFrameAxes(canvas, K, None, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], length, 2)
    return canvas, f"Tag pose OK | RMS {rms:.3f}px | X red / Y green / Z blue"


class CaptureWindow:
    def __init__(self, rig, camera, board):
        import tkinter as tk
        from PIL import Image, ImageTk
        from hardware.realsense import require_realsense
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.rig, self.camera, self.board = rig, camera, board
        self.root = tk.Tk()
        self.root.title(f"RSC calibration - {camera} - READ ONLY")
        self.root.geometry("1280x900")
        self.root.attributes("-zoomed", True)
        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill="both", expand=True)
        self.status = tk.StringVar(value="Waiting for camera")
        tk.Label(self.root, textvariable=self.status).pack()
        self.action = tk.StringVar(value="l: stage | s: save | q: discard | x / Esc: exit")
        tk.Label(self.root, textvariable=self.action).pack()
        self.pressed = None
        self.root.bind("<KeyPress>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.on_key(None))
        intr = rig.pipelines[camera].get_active_profile().get_stream(require_realsense().stream.color).as_video_stream_profile().get_intrinsics()
        if any(abs(c) > 1e-12 for c in intr.coeffs) and str(intr.model) not in ("distortion.none", "distortion.brown_conrady"):
            self.close()
            raise ValueError(f"Unsupported distortion: {intr.model}")
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
        self.dist = np.array(intr.coeffs)

    def on_key(self, event):
        k = "x" if event is None or event.keysym == "Escape" else event.char.lower()
        if k in ("l", "s", "q", "x"):
            self.pressed = k

    def next_key(self):
        fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        old = termios.tcgetattr(fd) if fd is not None else None
        try:
            if fd is not None:
                tty.setcbreak(fd)
            while True:
                self.root.update()
                if self.pressed:
                    k, self.pressed = self.pressed, None
                    return k
                if fd is not None and select.select([fd], [], [], 0)[0]:
                    return sys.stdin.read(1).lower()
                try:
                    frame = self.rig.capture_camera_frame(self.camera, timeout_ms=150)
                    if abs(time.time() - frame.timestamp_s) > .5:
                        raise ValueError("Stale camera frame")
                    rect = cv2.undistort(frame.color_image, self.K, self.dist)
                    canvas, status = annotate(rect, self.K, self.board)
                    image = self.Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                    w, h = max(1, self.label.winfo_width()), max(1, self.label.winfo_height())
                    scale = min(w / image.width, h / image.height)
                    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
                    self.photo = self.ImageTk.PhotoImage(image)
                    self.label.configure(image=self.photo)
                    self.status.set(status)
                except (RuntimeError, ValueError) as exc:
                    self.label.configure(image="")
                    self.status.set(f"Preview unavailable: {exc}")
                time.sleep(.02)
        finally:
            if old is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def message(self, text):
        self.action.set(text)
        self.root.update_idletasks()

    def close(self):
        self.root.destroy()
