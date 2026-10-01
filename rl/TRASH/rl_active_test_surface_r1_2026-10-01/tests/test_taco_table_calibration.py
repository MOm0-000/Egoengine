from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_taco_pour_table_calibration import depth_format_candidate, project, triangulate
from audit_taco_pour_raw_depth_table import fit_plane, temporal_plane_stability


def test_resized_video_is_not_treated_as_metric_depth():
    assert not depth_format_candidate(dict(codec_name="h264", pix_fmt="yuv420p"))
    assert not depth_format_candidate(dict(codec_name="ffv1", pix_fmt="gray"))
    assert depth_format_candidate(dict(codec_name="ffv1", pix_fmt="gray16le"))


def test_known_camera_triangulation_preserves_metric_scale_and_extrinsic_direction():
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.]])
    first, second = np.eye(4), np.eye(4)
    second[0, 3] = -.1
    points = np.array([[0, 0, 2], [.2, .1, 2.1], [-.1, .2, 1.8]])
    a, b = project(points, first, intrinsic), project(points, second, intrinsic)
    recovered, passed, errors, angles = triangulate(intrinsic, first, second, a, b)
    np.testing.assert_allclose(points, recovered, atol=1e-12)
    assert passed.all() and (errors < 1e-9).all() and (angles > 1).all()
    b[:, 1] += 8
    _, passed, _, _ = triangulate(intrinsic, first, second, a, b)
    assert not passed.any()


def test_tiny_baseline_does_not_become_a_precise_table_measurement():
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.]])
    first, second = np.eye(4), np.eye(4)
    second[0, 3] = -1e-4
    point = np.array([[0., 0, 2]])
    _, passed, _, _ = triangulate(intrinsic, first, second,
        project(point, first, intrinsic), project(point, second, intrinsic))
    assert not passed.any()


def test_temporal_plane_stability_uses_one_common_world_point():
    xy = np.stack(np.meshgrid(np.linspace(-.2, .2, 20),
                              np.linspace(-.2, .2, 20)), axis=-1).reshape(-1, 2)
    chunks = []
    for offset in (-.001, 0.0, .001):
        z = .02 * xy[:, 0] - .01 * xy[:, 1] + .55 + offset
        chunks.append(np.column_stack((xy, z)))
    result = temporal_plane_stability(chunks, np.array([0.0, 0.0]))
    assert result["frames"] == 3
    assert result["plane_z_p99_minus_p1_mm"] == pytest.approx(1.96)
    assert fit_plane(chunks)["tilt_deg"] > 0
