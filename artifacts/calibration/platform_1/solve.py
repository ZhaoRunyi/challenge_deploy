"""Offline fixed-camera calibration; no robot, camera, repo or network dependency."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def pack(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def unpack(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    T[:3, 3] = p[3:6]
    return T


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Unsafe manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    return manifest


def load(root, session=False):
    paths = sorted((root if session else root / 'samples').glob('*/sample.json'))
    if (not session and len(paths) != 10) or (session and len(paths) < 10):
        raise ValueError('platform_1 requires the frozen 10 samples')
    samples = [json.loads(p.read_text()) for p in paths]
    first = samples[0]
    for s in samples:
        for field in ('mode', 'camera', 'arm', 'reference', 'board'):
            if s[field] != first[field]:
                raise ValueError(f'Mixed {field}')
        for field in ('serial', 'camera_matrix', 'width', 'height', 'distortion'):
            if s['cameras']['cam_high'][field] != first['cameras']['cam_high'][field]:
                raise ValueError(f'Mixed intrinsics: {field}')
    if (first['mode'], first['camera'], first['arm'], first['reference']) != ('eye-to-hand', 'cam_high', 'left', 'eef'):
        raise ValueError('This bundle is fixed-camera / left EEF only')
    A = np.array([s['arms']['left']['references']['eef'] for s in samples])
    C = np.array([s['T_camera_board'] for s in samples])
    for T in list(A) + list(C):
        if not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1]):
            raise ValueError('Invalid transform')
        if not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(T[:3,:3]), 1):
            raise ValueError('Invalid rotation')
    diversity = np.linalg.svd(np.array([Rotation.from_matrix(a[:3,:3] @ A[0,:3,:3].T).as_rotvec() for a in A[1:]]), compute_uv=False)
    if diversity[1] < .15:
        raise ValueError('Insufficient rotation diversity')
    return paths, samples, A, C, diversity


def solve(root, session=False):
    paths, samples, A, C, diversity = load(root, session=session)
    first = samples[0]
    K = np.array(first['cameras']['cam_high']['camera_matrix'])
    xyz = np.array(first['board']['corners_m'])
    uv = np.array([s['detections'][str(s['board']['marker_id'])] for s in samples])
    inverse = np.linalg.inv(A)

    def residual(p):
        X, Y = unpack(p[:6]), unpack(p[6:])
        errors = []
        for a, observed in zip(A, uv):
            T = np.linalg.inv(X) @ a @ Y
            points = xyz @ T[:3,:3].T + T[:3,3]
            pixels = points @ K.T
            errors.append(pixels[:,:2] / np.maximum(pixels[:,2:], .001) - observed)
        return np.array(errors)

    fits = []
    for method in ('PARK', 'DANIILIDIS'):
        R,t = cv2.calibrateHandEye(list(inverse[:,:3,:3]), list(inverse[:,:3,3]),
            list(C[:,:3,:3]), list(C[:,:3,3]), method=getattr(cv2,'CALIB_HAND_EYE_'+method))
        X = np.eye(4)
        X[:3,:3], X[:3,3] = R, t.ravel()
        constants = inverse @ X @ C
        Y = np.eye(4)
        Y[:3,:3] = Rotation.from_matrix(constants[:,:3,:3]).mean().as_matrix()
        Y[:3,3] = np.median(constants[:,:3,3], axis=0)
        fit = least_squares(lambda p: residual(p).ravel(), np.r_[pack(X),pack(Y)],
            loss='soft_l1', f_scale=1., x_scale='jac', max_nfev=3000,
            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if fit.success and np.isfinite(fit.x).all():
            fits.append((fit,method))
    if not fits:
        raise ValueError('No optimizer converged')
    fit,method = min(fits,key=lambda item:item[0].cost)
    X,Y = unpack(fit.x[:6]),unpack(fit.x[6:])
    for a in A:
        T = np.linalg.inv(X) @ a @ Y
        if np.min((xyz @ T[:3,:3].T + T[:3,3])[:,2]) <= 0:
            raise ValueError('Tag behind camera')
    singular = np.linalg.svd(fit.jac,compute_uv=False)
    if np.linalg.matrix_rank(fit.jac) < 12:
        raise ValueError('Rank-deficient optimization')
    e = residual(fit.x)
    return dict(mode=first['mode'],camera_name='cam_high',arm_name='left',reference='eef',
        camera_intrinsics=first['cameras']['cam_high'],sample_count=len(samples),
        T_base_camera=X.tolist(),T_reference_tag=Y.tolist(),
        convention='T_A_B maps B to A; meters; camera x-right y-down z-forward',
        algorithm=dict(name='joint_corner_reprojection',loss='soft_l1',f_scale_px=1.,
                       initializers=['PARK','DANIILIDIS'],selected_initializer=method,
                       fixed_intrinsics=True,fixed_tag_size_m=first['board']['size_m']),
        status='best_current_candidate_not_independently_validated',optimizer_success=True,
        training_corner_rms_px=float(np.sqrt(np.mean(np.sum(e**2,axis=2)))),
        per_sample_corner_rms_px=np.sqrt(np.mean(np.sum(e**2,axis=2),axis=1)).tolist(),
        training_samples=[p.parent.name for p in paths],
        rotation_diversity_singular_values=diversity.tolist(),
        jacobian_singular_values=singular.tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--session',type=Path,help='New cam_high / left EEF session; at least 10 samples')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    verify(root)
    result=solve(args.session or root, session=args.session is not None)
    output=args.output or root/'local_runs'/datetime.now().strftime('%Y%m%d_%H%M%S')/'result.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print(f"Training corner RMS: {result['training_corner_rms_px']:.6f}px")
    print(output)


if __name__=='__main__':
    main()
