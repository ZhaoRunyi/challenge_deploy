"""A4 AprilTag matching the previous 36h11 / ID 0 / 100mm calibration."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from .core import write_json


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 1600, borderBits=1)
    page = np.full((2970, 2100), 255, np.uint8)
    page[985:1985, 550:1550] = cv2.resize(marker, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output / "apriltag_a4.png"), page)
    mm = 72 / 25.4
    c = canvas.Canvas(str(output / "apriltag_a4.pdf"), pagesize=(210 * mm, 297 * mm))
    c.drawImage(ImageReader(Image.fromarray(marker)), 55 * mm, 98.5 * mm, width=100 * mm, height=100 * mm)
    c.showPage()
    c.save()
    write_json(output / "board.json", {"family": "DICT_APRILTAG_36h11", "marker_id": 0,
        "size_m": 0.1, "page_mm": [210, 297], "frame": "tag center; x right, y up, z out of page",
        "corners_m": [[-.05, .05, 0], [.05, .05, 0], [.05, -.05, 0], [-.05, -.05, 0]]})
