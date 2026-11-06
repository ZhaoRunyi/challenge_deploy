import importlib.util
from pathlib import Path
import shutil

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1] / 'artifacts/calibration/platform_1'
spec = importlib.util.spec_from_file_location('platform1_solver', ROOT / 'solve.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def test_roundtrip():
    p = np.array([.2, -.4, .1, .3, .1, .7])
    np.testing.assert_allclose(solver.pack(solver.unpack(p)), p)


def test_portable_solve(tmp_path):
    import json
    bundle = tmp_path / 'platform_1'
    shutil.copytree(ROOT, bundle, ignore=shutil.ignore_patterns('__pycache__', 'local_runs'))
    solver.verify(bundle)
    result = solver.solve(bundle)
    frozen = json.loads((bundle / 'results/cam_high.json').read_text())
    np.testing.assert_allclose(result['T_base_camera'], frozen['T_base_camera'], atol=1e-6)
    assert result['training_corner_rms_px'] == pytest.approx(1.6890562, abs=1e-4)
    assert result['sample_count'] == 10
    assert result['optimizer_success']


def test_tampered_data_rejected(tmp_path):
    bundle = tmp_path / 'platform_1'
    shutil.copytree(ROOT, bundle, ignore=shutil.ignore_patterns('__pycache__', 'local_runs'))
    sample = next((bundle / 'samples').glob('*/sample.json'))
    sample.write_text('{}')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        solver.verify(bundle)


def test_missing_samples_rejected(tmp_path):
    with pytest.raises(ValueError, match='10 samples'):
        solver.load(tmp_path)


def test_new_session_solve(tmp_path):
    session = tmp_path / 'new_capture'
    shutil.copytree(ROOT / 'samples', session)
    result = solver.solve(session, session=True)
    assert result['sample_count'] == 10
    assert result['training_corner_rms_px'] == pytest.approx(1.6890562, abs=1e-4)


def test_portable_alignment_assets():
    import json
    from PIL import Image
    ref = json.loads((ROOT / 'coarse_reference/reference.json').read_text())
    assert ref['frame'] == 434
    assert len(ref['joint_angles']['left']) == 6
    for name in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
        with Image.open(ROOT / 'coarse_reference' / (name + '.png')) as image:
            assert image.size == (640, 480)
    board = json.loads((ROOT / 'board/board.json').read_text())
    assert board['size_m'] == .1
