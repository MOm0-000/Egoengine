import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from video_to_spider.adapters.hand_stereo import (
    require_stereo_hand_gate,
    run,
    triangulate_rectified_joints,
)


def test_rectified_triangulation_recovers_metric_points():
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    points = np.array([[[0.04, -0.02, 0.50], [0.02, 0.01, 0.80]]])
    left = np.empty((1, 2, 2))
    left[..., 0] = K[0, 0] * points[..., 0] / points[..., 2] + K[0, 2]
    left[..., 1] = K[1, 1] * points[..., 1] / points[..., 2] + K[1, 2]
    right_points = points.copy()
    right_points[..., 0] -= 0.06
    right = np.empty_like(left)
    right[..., 0] = K[0, 0] * right_points[..., 0] / right_points[..., 2] + K[0, 2]
    right[..., 1] = K[1, 1] * right_points[..., 1] / right_points[..., 2] + K[1, 2]

    recovered, valid, diagnostics = triangulate_rectified_joints(
        left, right, K_left=K, K_right=K, baseline_m=0.06,
        valid_observation=np.ones((1, 2), dtype=bool),
    )

    assert valid.all()
    np.testing.assert_allclose(recovered, points, atol=1e-10)
    np.testing.assert_allclose(diagnostics["reprojection_error_px"], 0, atol=1e-10)


def test_rectified_triangulation_supports_distinct_camera_intrinsics():
    K_left = np.array([[500.0, 0, 320.0], [0, 510.0, 240.0], [0, 0, 1]])
    K_right = np.array([[490.0, 0, 315.0], [0, 505.0, 238.0], [0, 0, 1]])
    points = np.array([[[0.04, -0.02, 0.50], [0.02, 0.01, 0.80]]])
    left = np.einsum("ij,...j->...i", K_left, points)
    left = left[..., :2] / left[..., 2:3]
    right_points = points.copy()
    right_points[..., 0] -= 0.06
    right = np.einsum("ij,...j->...i", K_right, right_points)
    right = right[..., :2] / right[..., 2:3]

    recovered, valid, diagnostics = triangulate_rectified_joints(
        left, right, K_left=K_left, K_right=K_right, baseline_m=0.06,
        valid_observation=np.ones((1, 2), dtype=bool),
    )

    assert valid.all()
    np.testing.assert_allclose(recovered, points, atol=1e-10)
    np.testing.assert_allclose(diagnostics["reprojection_error_px"], 0, atol=1e-10)


def test_triangulation_rejects_vertical_mismatch_and_negative_disparity():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1]])
    left = np.array([[[60.0, 40.0], [60.0, 40.0]]])
    right = np.array([[[55.0, 45.0], [65.0, 40.0]]])

    points, valid, _ = triangulate_rectified_joints(
        left, right, K_left=K, K_right=K, baseline_m=0.05,
        valid_observation=np.ones((1, 2), dtype=bool),
    )

    assert not valid.any()
    assert np.isnan(points).all()


def _write_wilor(path: Path, K: np.ndarray, points: np.ndarray, baseline: float) -> None:
    count = points.shape[0]
    translated = points.copy()
    if baseline:
        translated[..., 0] -= baseline
    uv_depth = np.einsum("ij,thkj->thki", K, translated)
    uv = uv_depth[..., :2] / uv_depth[..., 2:3]
    # Any camera-space points projecting to these pixels are sufficient; the
    # adapter deliberately discards their monocular depth.
    rays = np.concatenate([uv, np.ones((*uv.shape[:-1], 1))], axis=-1)
    rays = np.einsum("ij,thkj->thki", np.linalg.inv(K), rays)
    identity = np.eye(3, dtype=np.float32)
    np.savez_compressed(
        path,
        frame_indices=np.arange(count), timestamps_s=np.arange(count) / 30,
        side=np.broadcast_to(np.array([0, 1], dtype=np.int8), (count, 2)),
        valid=np.ones((count, 2), bool), score=np.ones((count, 2), np.float32),
        mano_global_orient=np.broadcast_to(identity, (count, 2, 3, 3)),
        mano_hand_pose=np.broadcast_to(identity, (count, 2, 15, 3, 3)),
        mano_betas=np.zeros((count, 2, 10), np.float32),
        joints_camera_rootrel=rays.astype(np.float32),
        vertices_camera_rootrel=np.zeros((count, 2, 778, 3), np.float32),
        translation_camera=np.zeros((count, 2, 3), np.float32),
    )


def test_stereo_hand_adapter_writes_independent_metric_artifact(tmp_path: Path):
    count = 4
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    points = np.zeros((count, 2, 21, 3), dtype=np.float64)
    points[..., 2] = 0.6
    points[..., 0] = np.linspace(-0.04, 0.04, 21)
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _write_wilor(left, K, points, 0.0)
    _write_wilor(right, K, points, 0.06)
    (tmp_path / "calibration").mkdir()
    np.save(tmp_path / "calibration/intrinsics.npy", K)
    np.save(tmp_path / "calibration/intrinsics_right.npy", K)
    (tmp_path / "calibration/stereo.json").write_text(
        json.dumps({"accepted": True, "baseline_m": 0.06}), encoding="utf-8"
    )

    output = run(Namespace(
        run_dir=tmp_path, left_artifact=left, right_artifact=right,
        output=tmp_path / "stereo.npz", overwrite=False,
    ))

    with np.load(output) as artifact:
        assert artifact["valid"].all()
        np.testing.assert_allclose(artifact["joints_camera_metric"], points, atol=2e-6)
    metrics = json.loads((tmp_path / "wilor_stereo_metrics.json").read_text())
    assert metrics["accepted_any_hand"]
    assert metrics["object_or_contact_used"] is False
    assert require_stereo_hand_gate(output)["accepted_any_hand"]

    with np.load(left) as source:
        changed = {key: np.asarray(source[key]) for key in source.files}
    changed["score"][0, 0] = 0.25
    np.savez_compressed(left, **changed)
    import pytest
    with pytest.raises(RuntimeError, match="stereo_hand_gate_stale: left_input"):
        require_stereo_hand_gate(output)


def test_stereo_hand_gate_hashes_intrinsics(tmp_path: Path):
    count = 4
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    points = np.zeros((count, 2, 21, 3), dtype=np.float64)
    points[..., 2] = 0.6
    points[..., 0] = np.linspace(-0.04, 0.04, 21)
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _write_wilor(left, K, points, 0.0)
    _write_wilor(right, K, points, 0.06)
    (tmp_path / "calibration").mkdir()
    np.save(tmp_path / "calibration/intrinsics.npy", K)
    np.save(tmp_path / "calibration/intrinsics_right.npy", K)
    (tmp_path / "calibration/stereo.json").write_text(
        json.dumps({"accepted": True, "baseline_m": 0.06}), encoding="utf-8"
    )
    output = run(Namespace(
        run_dir=tmp_path, left_artifact=left, right_artifact=right,
        output=tmp_path / "stereo.npz", overwrite=False,
    ))

    K[0, 0] = 510.0
    np.save(tmp_path / "calibration/intrinsics_right.npy", K)
    import pytest
    with pytest.raises(RuntimeError, match="stereo_hand_gate_stale: right_intrinsics"):
        require_stereo_hand_gate(output)
