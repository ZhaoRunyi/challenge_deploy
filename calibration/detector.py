import cv2
import numpy as np


def detect(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image) if isinstance(image, np.ndarray) else cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read {image}")
    corners, ids, _ = detector.detectMarkers(gray)
    points = {}
    for ident, polygon in zip([] if ids is None else ids.ravel(), corners):
        if str(ident) in points:
            raise ValueError(f"Duplicate marker ID {ident}")
        points[str(ident)] = np.asarray(polygon).reshape(4, 2).tolist()
    return points
