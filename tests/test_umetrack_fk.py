import numpy as np

from video_to_spider.optimization.umetrack_fk import (
    DISTAL_LANDMARKS,
    TIP_LANDMARKS,
    apply_semantic_calibration,
    axis_contract,
    geometric_frame,
)


def test_geometric_frame_is_proper_and_uses_dip_tip_axis():
    frame = geometric_frame([1, 0, 0], [0, 0, 2])
    np.testing.assert_allclose(frame, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(frame), 1.0)


def test_calibrated_frames_pass_exact_axis_contract():
    landmarks = np.zeros((2, 20, 3), dtype=float)
    for finger, (tip, dip) in enumerate(zip(TIP_LANDMARKS, DISTAL_LANDMARKS)):
        landmarks[:, dip] = [finger, 0, 0]
        landmarks[:, tip] = [finger, 0, 1]
    frames = np.broadcast_to(np.eye(3), (2, 5, 3, 3)).copy()
    calibrated = apply_semantic_calibration(frames, np.broadcast_to(np.eye(3), (5, 3, 3)))
    report = axis_contract(calibrated, landmarks)
    assert report["passed"]
    assert report["dip_to_tip_error_deg"]["max"] == 0.0
