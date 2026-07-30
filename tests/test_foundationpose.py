import numpy as np

from video_to_spider.adapters.foundationpose import _resize_observation, summarize_tracking, tracking_score


def test_tracking_score_rewards_valid_consistent_fit():
    good = {
        "valid_rate": 1.0, "mean_mask_iou": 0.8, "median_relative_depth_residual": 0.05,
        "translation_jump_p95_m": 0.01, "rotation_jump_p95_rad": 0.05,
    }
    bad = {
        "valid_rate": 0.5, "mean_mask_iou": 0.1, "median_relative_depth_residual": 1.0,
        "translation_jump_p95_m": 0.3, "rotation_jump_p95_rad": 2.0,
    }
    assert tracking_score(good) > tracking_score(bad)


def test_tracking_summary_counts_registrations_and_jumps():
    transforms = np.repeat(np.eye(4)[None], 4, axis=0)
    transforms[:, 0, 3] = [0.0, 0.01, 0.02, 0.03]
    metrics = summarize_tracking(
        transforms, np.ones(4, bool), np.full(4, 0.5), np.full(4, 0.1),
        np.array([True, False, True, False]),
    )
    assert metrics["valid_rate"] == 1.0
    assert metrics["registration_count"] == 2
    assert np.isclose(metrics["translation_jump_p95_m"], 0.01)


def test_resize_observation_scales_intrinsics_and_preserves_metric_depth():
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    depth = np.full((1080, 1920), 0.42, dtype=np.float32)
    mask = np.zeros((1080, 1920), dtype=bool)
    mask[300:600, 900:1200] = True
    K = np.array([[1200.0, 0.0, 960.0], [0.0, 1180.0, 540.0], [0.0, 0.0, 1.0]])
    resized_rgb, resized_depth, resized_mask, resized_K = _resize_observation(
        rgb, depth, mask, K, 640,
    )
    assert resized_rgb.shape == (360, 640, 3)
    assert resized_depth.shape == resized_mask.shape == (360, 640)
    assert np.isclose(resized_depth[resized_mask].mean(), 0.42)
    assert np.allclose(resized_K[0], K[0] / 3.0)
    assert np.allclose(resized_K[1], K[1] / 3.0)
    assert np.allclose(resized_K[2], K[2])
