import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_to_spider.ingest.stereo import (
    _load_vector,
    audit_rectified_pairs,
    discover_stereo_pairs,
    ingest_rectified_stereo,
)


def test_sensor_nanosecond_timestamps_are_converted_to_seconds(tmp_path: Path):
    path = tmp_path / "timestamps.json"
    path.write_text(
        json.dumps({"timestamp_sensor_ns": [1_000_000_000, 1_033_000_000]}),
        encoding="utf-8",
    )

    values = _load_vector(path, 2, integer=False)

    np.testing.assert_allclose(values, [1.0, 1.033], atol=1e-12)


def _stereo_images(root: Path, *, vertical_shift: int = 0, count: int = 4) -> tuple[Path, Path]:
    left_dir = root / "left"
    right_dir = root / "right"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(180, 240), dtype=np.uint8)
    for index in range(count):
        left = np.roll(base, index, axis=0)
        right = np.zeros_like(left)
        shifted = np.roll(left, vertical_shift, axis=0)
        right[:, :-8] = shifted[:, 8:]
        assert cv2.imwrite(str(left_dir / f"{index:06d}.png"), left)
        assert cv2.imwrite(str(right_dir / f"{index:06d}.png"), right)
    return left_dir, right_dir


def test_rectification_audit_accepts_horizontal_disparity(tmp_path: Path) -> None:
    left, right = _stereo_images(tmp_path)
    result = audit_rectified_pairs(discover_stereo_pairs(left, right))
    assert result["accepted"]
    assert result["vertical_disparity_abs_p95_px"] <= 1.5
    assert result["horizontal_disparity_median_px"] > 0


def test_rectification_audit_rejects_vertical_misalignment(tmp_path: Path) -> None:
    left, right = _stereo_images(tmp_path, vertical_shift=5)
    result = audit_rectified_pairs(discover_stereo_pairs(left, right))
    assert not result["accepted"]
    assert not result["checks"]["vertical_p95_within_limit"]


def test_rectification_audit_uses_per_camera_normalized_coordinates(tmp_path: Path) -> None:
    left, right = _stereo_images(tmp_path)
    K_left = np.array([[180.0, 0, 120.0], [0, 180.0, 90.0], [0, 0, 1]])
    # Different reported K for the right camera should be interpreted in its
    # own normalized ray coordinates, not subtracted as raw pixels.
    K_right = np.array([[180.0, 0, 112.0], [0, 180.0, 90.0], [0, 0, 1]])
    raw = audit_rectified_pairs(discover_stereo_pairs(left, right))
    normalized = audit_rectified_pairs(
        discover_stereo_pairs(left, right), K_left=K_left, K_right=K_right,
    )

    assert raw["horizontal_disparity_median_px"] > 0
    assert not normalized["checks"]["left_right_order_positive_disparity"]
    assert "normalized rectified coordinates" in normalized["method"]


def test_ingest_writes_standard_left_reference_run(tmp_path: Path) -> None:
    left, right = _stereo_images(tmp_path / "source")
    K_path = tmp_path / "K.npy"
    np.save(K_path, np.array([[180.0, 0, 120.0], [0, 180.0, 90.0], [0, 0, 1]]))
    run = tmp_path / "run"
    manifest_path = ingest_rectified_stereo(
        left_dir=left,
        right_dir=right,
        intrinsics_path=K_path,
        baseline_m=0.064,
        output_dir=run,
        task="move object",
        episode_id="demo",
        instruction="Move the cup on the table.",
        fps=30.0,
        static_camera=True,
        full_image_common_valid=True,
    )
    assert manifest_path.is_file()
    source = json.loads((run / "input/source.json").read_text())
    rows = json.loads((run / "frames/frame_index.json").read_text())["frames"]
    stereo = json.loads((run / "calibration/stereo.json").read_text())
    assert source["source_type"] == "calibrated_rectified_stereo"
    assert all((run / row["rgb_path"]).is_file() for row in rows)
    assert all((run / row["right_rgb_path"]).is_file() for row in rows)
    assert stereo["accepted"] and stereo["baseline_m"] == pytest.approx(0.064)
    assert stereo["common_valid_pixel_ratio"] == pytest.approx(1.0)
    assert np.load(run / "calibration/stereo_common_valid.npy").all()
    np.testing.assert_allclose(
        np.load(run / "calibration/T_world_camera.npy"),
        np.repeat(np.eye(4)[None], len(rows), axis=0),
    )


def test_ingest_preserves_distinct_rectified_intrinsics(tmp_path: Path) -> None:
    left, right = _stereo_images(tmp_path / "source")
    K_left = np.array([[180.0, 0, 120.0], [0, 180.0, 90.0], [0, 0, 1]])
    K_right = np.array([[179.5, 0, 119.0], [0, 181.0, 91.0], [0, 0, 1]])
    left_path = tmp_path / "K_left.npy"
    right_path = tmp_path / "K_right.npy"
    np.save(left_path, K_left)
    np.save(right_path, K_right)

    ingest_rectified_stereo(
        left_dir=left, right_dir=right, intrinsics_path=left_path,
        right_intrinsics_path=right_path, baseline_m=0.064,
        output_dir=tmp_path / "run", task="move object", episode_id="demo",
        instruction="Move the cup.", fps=30.0, static_camera=True,
        full_image_common_valid=True,
    )

    np.testing.assert_allclose(
        np.load(tmp_path / "run/calibration/intrinsics.npy"), K_left,
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "run/calibration/intrinsics_right.npy"), K_right,
    )
    stereo = json.loads((tmp_path / "run/calibration/stereo.json").read_text())
    np.testing.assert_allclose(stereo["K_rect_left"], K_left)
    np.testing.assert_allclose(stereo["K_rect_right"], K_right)


def test_moving_ego_camera_requires_pose_trajectory(tmp_path: Path) -> None:
    left, right = _stereo_images(tmp_path / "source")
    K_path = tmp_path / "K.npy"
    np.save(K_path, np.eye(3))
    with pytest.raises(ValueError, match="requires --camera-poses"):
        ingest_rectified_stereo(
            left_dir=left,
            right_dir=right,
            intrinsics_path=K_path,
            baseline_m=0.064,
            output_dir=tmp_path / "run",
            task="task",
            episode_id="0",
            instruction="Move object",
            fps=30.0,
            full_image_common_valid=True,
        )


def test_ingest_requires_explicit_common_valid_domain(tmp_path: Path) -> None:
    left, right = _stereo_images(tmp_path / "source")
    K_path = tmp_path / "K.npy"
    np.save(K_path, np.array([[180.0, 0, 120.0], [0, 180.0, 90.0], [0, 0, 1]]))
    with pytest.raises(ValueError, match="exactly one"):
        ingest_rectified_stereo(
            left_dir=left, right_dir=right, intrinsics_path=K_path,
            baseline_m=0.064, output_dir=tmp_path / "run", task="task",
            episode_id="0", instruction="Move object", fps=30.0,
            static_camera=True,
        )


def test_ingest_preserves_explicit_common_valid_mask(tmp_path: Path) -> None:
    left, right = _stereo_images(tmp_path / "source")
    K_path = tmp_path / "K.npy"
    np.save(K_path, np.array([[180.0, 0, 120.0], [0, 180.0, 90.0], [0, 0, 1]]))
    common = np.ones((180, 240), dtype=bool)
    common[:, :16] = False
    common_path = tmp_path / "common.npy"
    np.save(common_path, common)

    ingest_rectified_stereo(
        left_dir=left, right_dir=right, intrinsics_path=K_path,
        common_valid_mask_path=common_path, baseline_m=0.064,
        output_dir=tmp_path / "run", task="task", episode_id="0",
        instruction="Move object", fps=30.0, static_camera=True,
    )

    np.testing.assert_array_equal(
        np.load(tmp_path / "run/calibration/stereo_common_valid.npy"), common,
    )
