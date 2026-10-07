from __future__ import annotations

import numpy as np

from egoengine_repro.scene.support_surface_estimation import (
    backproject_metric_depth,
    camera_points_to_world,
)


def test_every_selected_valid_depth_pixel_is_preserved_for_signed_distance() -> None:
    depth = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    selector = np.array([[True, True, True], [False, True, True]])
    intrinsic = np.eye(3)
    camera, pixels = backproject_metric_depth(depth, intrinsic, selector=selector, spatial_stride_px=1)
    world = camera_points_to_world(camera, np.eye(4))
    normal = np.array([0.0, 0.0, 1.0])
    signed = world @ normal - 1.5
    assert pixels.tolist() == [[1, 0], [2, 0], [1, 1]]
    assert camera[:, 2].tolist() == [1.0, 2.0, 4.0]
    assert signed.tolist() == [-0.5, 0.5, 2.5]
