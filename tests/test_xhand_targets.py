from pathlib import Path

import numpy as np

from video_to_spider.export.xhand_targets import (
    isotropic_hand_size_scale,
    normalize_exported_arrays,
    normalize_exported_keypoints,
    normalize_palm_relative_positions,
)


def test_palm_normalization_scales_observed_pose_without_neutral_replacement():
    wrist = np.asarray([[0.2, -0.1, 0.4]])
    palm = np.eye(3)[None]
    observed = wrist[:, None] + np.arange(15).reshape(1, 5, 3) / 100.0
    result = normalize_palm_relative_positions(wrist, palm, observed, 1.2)
    np.testing.assert_allclose(result - wrist[:, None], 1.2 * (observed - wrist[:, None]))


def test_exported_normalization_is_static_and_idempotent(tmp_path: Path):
    count = 2
    wrist = np.zeros((count, 7)); wrist[:, 3] = 1.0
    fingers = np.zeros((count, 5, 7)); fingers[:, :, 3] = 1.0
    human_neutral = np.arange(1, 16, dtype=float).reshape(5, 3) / 100.0
    fingers[:, :, :3] = human_neutral
    arrays = {
        "qpos_wrist_right": wrist,
        "qpos_finger_right": fingers,
        "human_palm_orientation_right": np.repeat(np.eye(3)[None], count, axis=0),
        "human_neutral_fingertip_vectors_right": human_neutral,
    }
    normalized, report = normalize_exported_arrays(arrays)
    scale = isotropic_hand_size_scale(human_neutral)
    np.testing.assert_allclose(
        normalized["qpos_finger_right"][:, :, :3],
        np.broadcast_to(scale * human_neutral, (count, 5, 3)),
    )
    assert report["applied"]
    again, repeated = normalize_exported_arrays(normalized)
    assert repeated == {"applied": False, "status": "already_applied"}
    np.testing.assert_allclose(again["qpos_finger_right"], normalized["qpos_finger_right"])

    artifact = tmp_path / "trajectory_keypoints.npz"
    np.savez_compressed(artifact, **arrays)
    normalize_exported_keypoints(artifact)
    first = artifact.read_bytes()
    normalize_exported_keypoints(artifact)
    assert artifact.read_bytes() == first
