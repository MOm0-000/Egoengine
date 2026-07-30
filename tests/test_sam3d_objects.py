import cv2
import numpy as np
import trimesh

from video_to_spider.adapters.sam3d_objects import (
    mesh_integrity,
    repair_and_canonicalize,
    static_fit_metrics,
)


def test_repair_canonicalizes_largest_component():
    large = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    small = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    small.apply_translation([10, 0, 0])
    repaired, metadata = repair_and_canonicalize(trimesh.util.concatenate([large, small]))
    assert np.isclose(np.max(repaired.extents), 1.0)
    np.testing.assert_allclose(repaired.bounds.mean(axis=0), 0.0, atol=1e-8)
    assert metadata["raw_longest_extent"] == 2.0
    integrity = mesh_integrity(repaired)
    assert integrity["qualified"]
    assert integrity["component_count"] == 1


def test_static_fit_returns_finite_traceable_metrics():
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.2])
    height, width = 120, 160
    K = np.array([[120.0, 0.0, width / 2], [0.0, 120.0, height / 2], [0.0, 0.0, 1.0]])
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (65, 45), (95, 75), 1, -1)
    depth = np.ones((height, width), dtype=np.float32)
    valid = np.ones_like(mask, dtype=bool)
    metrics, transform, scale = static_fit_metrics(
        mesh, mask.astype(bool), depth, valid, K, rotation_wxyz=[1, 0, 0, 0],
        translation=[0, 0, 1], layout_scale=[0.25, 0.25, 0.25],
    )
    assert 0.003 <= scale <= 0.5
    assert np.isfinite(transform).all()
    assert 0.0 <= metrics["silhouette_iou"] <= 1.0
    assert metrics["point_count"] > 0
