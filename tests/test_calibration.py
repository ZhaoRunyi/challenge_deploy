import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.core import board_pose, pose_matrix, solve_hand_eye
from calibration.detector import detect
from calibration.board import generate


@pytest.mark.parametrize("mode", ["eye-in-hand", "eye-to-hand"])
def test_known_hand_eye(mode):
    rng = np.random.default_rng(7)
    X = pose_matrix([0.1, -0.05, 0.07, 0.3, 0.2, -0.1])
    Y = pose_matrix([0.3, 0.2, 0.6, -0.2, 0.1, 0.3])
    A = [pose_matrix(np.r_[rng.uniform(-0.2, 0.2, 3), rng.uniform(-0.7, 0.7, 3)]) for _ in range(20)]
    C = [np.linalg.inv(X) @ (np.linalg.inv(a) if mode == "eye-in-hand" else a) @ Y for a in A]
    solved, _ = solve_hand_eye(A, C, mode)
    np.testing.assert_allclose(solved, X, atol=1e-8)


def test_degenerate_motion_rejected():
    with pytest.raises(ValueError, match="rotation diversity"):
        solve_hand_eye([np.eye(4)] * 12, [np.eye(4)] * 12, "eye-in-hand")


def test_pnp(tmp_path):
    import json
    generate(tmp_path)
    board = json.loads((tmp_path / "board.json").read_text())
    xyz = np.array(board["corners_m"])
    K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    T = pose_matrix([-.1, -.1, .7, .3, -.2, .1])
    uv = cv2.projectPoints(xyz, Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3], K, None)[0].reshape(-1, 2)
    result, rms, ids = board_pose({"0": uv}, board, K)
    np.testing.assert_allclose(result, T, atol=1e-6)
    assert rms < 1e-5 and ids == ["0"]


def test_printed_board_roundtrip(tmp_path):
    generate(tmp_path)
    points = detect(tmp_path / "apriltag_a4.png")
    assert list(points) == ["0"]
    np.testing.assert_allclose(points["0"], [[550, 985], [1549, 985], [1549, 1984], [550, 1984]], atol=1)


def test_missing_tag_rejected(tmp_path):
    import json
    generate(tmp_path)
    with pytest.raises(ValueError, match="not visible"):
        board_pose({}, json.loads((tmp_path / "board.json").read_text()), np.eye(3))


def test_preview_detection_array(tmp_path):
    generate(tmp_path)
    path = tmp_path / "apriltag_a4.png"
    assert detect(cv2.imread(str(path))) == detect(path)


def test_preview_axes_and_rejection(tmp_path, monkeypatch):
    import json
    from calibration import capture_window
    generate(tmp_path)
    board = json.loads((tmp_path / "board.json").read_text())
    K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    T = pose_matrix([0, 0, .7, .3, -.2, .1])
    uv = cv2.projectPoints(np.array(board["corners_m"]),
                          cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)[0].reshape(-1, 2)
    image = np.full((480, 640, 3), 255, np.uint8)
    monkeypatch.setattr(capture_window, "detect", lambda image: {"0": uv})
    drawn, status = capture_window.annotate(image, K, board)
    assert "pose OK" in status
    assert np.any(np.all(drawn == (0, 0, 255), axis=2))
    assert np.all(image == 255)
    monkeypatch.setattr(capture_window, "detect", lambda image: {})
    drawn, status = capture_window.annotate(image, K, board)
    assert "not visible" in status
    np.testing.assert_array_equal(drawn, image)
    drawn, status = capture_window.annotate(image, K, None)
    assert status == "Clean capture"


def test_read_only_capture_source():
    from pathlib import Path
    source = (Path(__file__).parents[1] / "calibration/cli.py").read_text()
    assert "commands_enabled=False" in source and "connect(read_only=True)" in source
    for command in (".enable(", ".command_", ".move_to_", ".set_joint_mode("):
        assert command not in source


def test_inconsistent_poses_rejected():
    rng = np.random.default_rng(7)
    A = [pose_matrix(np.r_[rng.uniform(-.2, .2, 3), rng.uniform(-.7, .7, 3)]) for _ in range(20)]
    C = [np.linalg.inv(a) for a in A]
    C[-1][:3, 3] += .15
    with pytest.raises(ValueError, match="Inconsistent calibration"):
        solve_hand_eye(A, C, "eye-in-hand")


def test_invalid_rigid_transform_rejected():
    A = np.repeat(np.eye(4)[None], 12, axis=0)
    A[0, 0, 0] = 2
    with pytest.raises(ValueError, match="Invalid rigid transform"):
        solve_hand_eye(A, A, "eye-in-hand")


def test_named_history_joints():
    from calibration.history import extract_joints, JOINTS
    names = [f"{a}_joint_{j}" for a in ("right", "left") for j in JOINTS]
    parsed = extract_joints(list(range(12)), [names])
    assert parsed["left"] == list(range(6, 12))
    assert parsed["right"] == list(range(6))


def test_history_extraction_and_compare(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from calibration.history import CAMERAS, JOINTS, extract, compare
    from calibration.core import write_json
    source = tmp_path / "dataset"
    names = [f"{a}_joint_{j}" for a in ("left", "right") for j in JOINTS]
    info = {"fps": 10, "chunks_size": 1000, "data_path": "episode_{episode_index:06d}.parquet",
            "video_path": "{video_key}.mp4", "features": {"observation.state": {"names": [names]}}}
    write_json(source / "meta/info.json", info)
    rows = [{"frame_index": i, "episode_index": 0, "timestamp": i / 10, "observation.state": [i / 100] * 12} for i in range(3)]
    pq.write_table(pa.Table.from_pylist(rows), source / "episode_000000.parquet")
    for cam in CAMERAS:
        writer = cv2.VideoWriter(str(source / f"observation.images.{cam}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        assert writer.isOpened()
        for i in range(3):
            writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
        writer.release()
    out = tmp_path / "refs"
    assert extract(source, 0, out, frames=[1]) == [1]
    reference = out / "frame_000001"
    record = json.loads((reference / "reference.json").read_text())
    assert record["joint_angles"]["left"] == [.01] * 6
    sample = tmp_path / "sample"
    metadata = {"arms": {a: {"qpos": [.01] * 6 + [0]} for a in ("left", "right")}}
    write_json(sample / "sample.json", metadata)
    for cam in CAMERAS:
        cv2.imwrite(str(sample / f"{cam}_raw.png"), cv2.imread(str(reference / f"{cam}.png")))
    compare(reference, sample, tmp_path / "overlay")
    assert (tmp_path / "overlay/comparison.json").is_file()
    metadata["arms"]["left"]["qpos"][0] = 1
    write_json(sample / "sample.json", metadata)
    with pytest.raises(ValueError, match="Robot pose differs"):
        compare(reference, sample, tmp_path / "bad")


def test_download_checksum(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from calibration.hf_reference import checked_file
    path = tmp_path / "file"
    path.write_bytes(b"abc")
    metadata = SimpleNamespace(size=3, lfs=None, blob_id=hashlib.sha1(b"blob 3\0abc").hexdigest())
    assert checked_file(path, metadata)
    metadata.lfs = SimpleNamespace(sha256=hashlib.sha256(b"abc").hexdigest())
    assert checked_file(path, metadata)
    path.write_bytes(b"xyz")
    assert not checked_file(path, metadata)


def test_live_overlay_pixels_and_dimension_guard():
    from calibration.live_overlay import blend
    old = np.full((8, 10, 3), 20, dtype=np.uint8)
    new = np.full((8, 10, 3), 100, dtype=np.uint8)
    np.testing.assert_array_equal(blend(old, new, 0), old)
    np.testing.assert_array_equal(blend(old, new, 1), new)
    assert np.all(blend(old, new, .5) == 60)
    with pytest.raises(ValueError, match="dimensions differ"):
        blend(old, new[:4], .5)


def test_historical_gripper_scale():
    from calibration.set_reference_grippers import targets
    ref = {"state_names": [["right_gripper", "left_gripper"]], "observation_state_raw": [.476, .619]}
    assert targets(ref, .1) == pytest.approx({"left": .0619, "right": .0476})
    with pytest.raises(ValueError):
        targets(ref, 1)
