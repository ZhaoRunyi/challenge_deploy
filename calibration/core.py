from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pose_matrix(pose):
    """SDK XYZ meters, extrinsic XYZ Euler radians (Rz @ Ry @ Rx)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:6]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def transform(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).reshape(3)
    if not np.isfinite(T).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1):
        raise ValueError("Invalid rigid transform")
    return T


def board_pose(detections, board, K, max_rms=1.5):
    ident = str(board["marker_id"])
    if board["family"] != "DICT_APRILTAG_36h11" or ident not in detections:
        raise ValueError("Expected AprilTag 36h11 ID 0 is not visible")
    xyz = np.asarray(board["corners_m"], dtype=np.float64)
    uv = np.asarray(detections[ident], dtype=np.float64)
    if xyz.shape != (4, 3) or uv.shape != (4, 2) or not np.isfinite(uv).all():
        raise ValueError("Invalid square corners")
    if min(np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1)) < 40:
        raise ValueError("Tag too small/foreshortened: need >=40px per edge")
    ok, rotations, translations, _ = cv2.solvePnPGeneric(xyz, uv, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    for r, t in zip(rotations, translations):
        T = transform(cv2.Rodrigues(r)[0], t)
        if np.any((xyz @ T[:3, :3].T + T[:3, 3])[:, 2] <= 0):
            continue
        projected = cv2.projectPoints(xyz, r, t, K, None)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((projected - uv) ** 2, axis=1))))
        candidates.append((rms, T))
    candidates.sort(key=lambda item: item[0])
    if not ok or not candidates or candidates[0][0] > max_rms:
        raise ValueError("Bad PnP or excessive reprojection error")
    if len(candidates) > 1:
        angle = Rotation.from_matrix(candidates[0][1][:3, :3].T @ candidates[1][1][:3, :3]).magnitude()
        if candidates[1][0] - candidates[0][0] < 0.15 and angle > np.deg2rad(5):
            raise ValueError("Ambiguous planar pose; change board viewing angle")
    rms, T = candidates[0]
    return T, rms, [ident]


def solve_hand_eye(base_reference, camera_board, mode):
    """Eye-in: B_G X C_T = constant. Eye-to: inv(B_G) X C_T = constant."""
    if mode not in ("eye-in-hand", "eye-to-hand"):
        raise ValueError("Unknown calibration mode")
    A = np.asarray(base_reference, dtype=float)
    C = np.asarray(camera_board, dtype=float)
    if len(A) != len(C) or len(A) < 10:
        raise ValueError("Need >=10 matched poses; recommend 15-25")
    if A.shape[1:] != (4, 4) or C.shape[1:] != (4, 4):
        raise ValueError("Expected 4x4 rigid transforms")
    for matrix in list(A) + list(C):
        transform(matrix[:3, :3], matrix[:3, 3])
        if not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous transform")
    relative = np.array([Rotation.from_matrix(a[:3, :3] @ A[0, :3, :3].T).as_rotvec() for a in A[1:]])
    singular = np.linalg.svd(relative, compute_uv=False)
    if singular[1] < 0.15:
        raise ValueError("Insufficient rotation diversity: rotate about at least two axes")
    if mode == "eye-to-hand":
        A = np.linalg.inv(A)
    R, t = cv2.calibrateHandEye(list(A[:, :3, :3]), list(A[:, :3, 3]),
                               list(C[:, :3, :3]), list(C[:, :3, 3]),
                               method=cv2.CALIB_HAND_EYE_PARK)
    X = transform(R, t)
    constants = A @ X @ C
    center_t = np.median(constants[:, :3, 3], axis=0)
    center_R = Rotation.from_matrix(constants[:, :3, :3]).mean()
    translation = np.linalg.norm(constants[:, :3, 3] - center_t, axis=1)
    angles = (center_R.inv() * Rotation.from_matrix(constants[:, :3, :3])).magnitude()
    report = {"translation_residual_m": translation.tolist(),
              "rotation_residual_deg": np.rad2deg(angles).tolist(),
              "rotation_diversity_singular_values": singular.tolist()}
    if translation.max() > 0.015 or np.rad2deg(angles).max() > 3:
        raise ValueError("Inconsistent calibration (>15mm or >3deg): " + json.dumps(report))
    return X, report
