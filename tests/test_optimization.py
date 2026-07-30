import numpy as np
import trimesh

from video_to_spider.manifest import RunManifest
from video_to_spider.optimization.contact import infer_contact
from video_to_spider.optimization.sequence import (
    _hand_reprojection_metrics,
    _optimization_quality_control,
    _optimize_global_scale,
    calibrate_hand_depth_scale,
    classify_hand_roles,
    enforce_simulation_floor,
    hand_anchored_object_observation,
    object_minimum_z,
    optimize_run,
    reject_unphysical_passive_hands,
    xhand_wrist_frames_from_joints,
)
from video_to_spider.optimization.smoothing import smooth_rotations, smooth_second_difference
from video_to_spider.schemas import validate_npz


def test_second_difference_smoothing_reduces_jitter():
    values = np.zeros((20, 3))
    values[:, 0] = np.linspace(0, 1, 20)
    values[10, 0] += 0.5
    result = smooth_second_difference(values, np.ones(20), 20.0)
    assert np.linalg.norm(np.diff(result, n=2, axis=0)) < np.linalg.norm(np.diff(values, n=2, axis=0))


def test_second_difference_smoothing_handles_noncontiguous_fingertips():
    values = np.arange(20 * 2 * 21 * 3, dtype=np.float64).reshape(20, 2, 21, 3)
    fingertips = values[:, :, [4, 8, 12, 16, 20]]
    assert not fingertips.flags.c_contiguous
    result = smooth_second_difference(fingertips, np.ones(20), 5.0)
    assert result.shape == fingertips.shape
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, fingertips, atol=1e-8)


def test_rotation_smoothing_preserves_so3():
    rotations = np.repeat(np.eye(3)[None], 8, axis=0)
    result = smooth_rotations(rotations, np.ones(8), 5.0)
    np.testing.assert_allclose(
        np.swapaxes(result, 1, 2) @ result, np.repeat(np.eye(3)[None], 8, axis=0), atol=1e-7
    )


def test_object_observation_rejects_inconsistent_hand_depth():
    count = 6
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    raw = np.repeat([[0.0, 0.0, 1.2]], count, axis=0)
    centroids = np.repeat([[50.0, 40.0]], count, axis=0)
    joints = np.zeros((count, 1, 21, 3), dtype=np.float64)
    joints[..., 2] = 0.30

    observations, _, metrics = hand_anchored_object_observation(
        K, raw, centroids, joints, np.ones((count, 1), bool), np.ones((count, 1)),
        object_valid=np.ones(count, bool), object_confidence=np.ones(count),
        mask_valid=np.ones(count, bool), metric_mask_depth=np.full(count, 1.25),
    )

    np.testing.assert_allclose(observations[:, 2], 1.2125)
    assert metrics["metric_depth_accepted_frame_count"] == count
    assert metrics["hand_depth_accepted_frame_count"] == 0
    assert metrics["hand_depth_rejected_frame_count"] == count


def test_hand_depth_calibration_rejects_full_hand_reprojection_collapse():
    count = 5
    K = np.array([[700.0, 0.0, 500.0], [0.0, 700.0, 300.0], [0.0, 0.0, 1.0]])
    joints = np.zeros((count, 1, 21, 3), dtype=np.float64)
    joints[..., 2] = 0.30
    joints[:, 0, :, 0] = np.linspace(-0.08, 0.08, 21)
    joints[:, 0, :, 1] = np.linspace(-0.04, 0.04, 21)
    objects = np.repeat([[0.0, 0.0, 1.2]], count, axis=0)

    calibrated, metrics = calibrate_hand_depth_scale(
        K, joints, np.ones((count, 1), bool), np.ones((count, 1)), objects,
        np.repeat([[500.0, 300.0]], count, axis=0), np.ones(count, bool),
    )

    record = metrics["per_hand"]["left"]
    assert record["candidate_scale_ratio"] == 4.0
    assert record["applied_scale_ratio"] == 1.0
    assert record["accepted"] is False
    assert record["rejection_reason"] == "full_hand_reprojection_exceeds_trust_region"
    np.testing.assert_allclose(calibrated, joints)


def test_hand_depth_calibration_accepts_projection_safe_adjustment():
    count = 5
    K = np.array([[700.0, 0.0, 500.0], [0.0, 700.0, 300.0], [0.0, 0.0, 1.0]])
    joints = np.zeros((count, 1, 21, 3), dtype=np.float64)
    joints[..., 2] = 1.0
    joints[:, 0, :, 0] = np.linspace(-0.01, 0.01, 21)
    objects = np.repeat([[0.0, 0.0, 1.05]], count, axis=0)

    calibrated, metrics = calibrate_hand_depth_scale(
        K, joints, np.ones((count, 1), bool), np.ones((count, 1)), objects,
        np.repeat([[500.0, 300.0]], count, axis=0), np.ones(count, bool),
    )

    record = metrics["per_hand"]["left"]
    assert np.isclose(record["applied_scale_ratio"], 1.05)
    assert record["accepted"] is True
    np.testing.assert_allclose(calibrated[:, 0, 0, 2], 1.05)


def test_hand_reprojection_qc_detects_exported_fingertip_shift():
    K = np.array([[700.0, 0.0, 500.0], [0.0, 700.0, 300.0], [0.0, 0.0, 1.0]])
    raw = np.zeros((4, 1, 5, 3), dtype=np.float64)
    raw[..., 2] = 1.0
    exported = raw.copy()
    exported[..., 0] = 0.10

    hand_metrics = _hand_reprojection_metrics(
        K, raw, exported, np.ones((4, 1), bool), ["right"],
    )
    qc = _optimization_quality_control(
        np.repeat([[0.0, 0.0, 1.2]], 4, axis=0),
        np.repeat([[0.0, 0.0, 1.2]], 4, axis=0),
        {"scale_ratio": 1.0}, {}, hand_metrics,
    )

    assert hand_metrics["per_hand"]["right"]["median_px"] == 70.0
    assert hand_metrics["passed"] is False
    assert qc["checks"]["hand_image_alignment_preserved"] is False
    assert qc["export_ready"] is False


def test_global_scale_is_clipped_to_conservative_bounds():
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 2, 3] = 1.0
    masks = np.ones((3, 80, 100), dtype=bool)

    scale, metrics = _optimize_global_scale(
        mesh, 0.1, K, poses, masks, np.ones(3, bool),
    )

    assert np.isclose(scale, 0.15)
    assert metrics["scale_ratio"] == 1.5
    assert metrics["raw_scale_ratio"] > 1.5
    assert metrics["ratio_was_clipped"] is True


def test_optimization_qc_rejects_depth_collapse():
    raw = np.repeat([[0.0, 0.0, 1.2]], 5, axis=0)
    collapsed = np.repeat([[0.0, 0.0, 0.3]], 5, axis=0)
    qc = _optimization_quality_control(
        raw, collapsed, {"scale_ratio": 0.75},
        {"relative_depth_residual_raw": 0.02, "relative_depth_residual_aligned": 0.7},
    )

    assert qc["export_ready"] is False
    assert qc["checks"]["camera_depth_scale_preserved"] is False
    assert qc["checks"]["metric_depth_residual_preserved"] is False


def test_contact_hysteresis_and_local_positions():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.04)
    count = 8
    object_pose = np.repeat(np.eye(4)[None], count, axis=0)
    fingertips = np.zeros((count, 1, 5, 3))
    fingertips[..., 0] = 0.045
    timestamps = np.arange(count) / 30.0
    contact, positions, metrics = infer_contact(
        mesh, object_pose, fingertips, timestamps, np.ones((count, 1), bool),
        enter_distance_m=0.01, exit_distance_m=0.02, min_duration_frames=2,
    )
    assert contact.shape == (count, 1, 5)
    assert contact.all()
    assert positions.shape == (1, 5, 3)
    assert metrics["contact_rate"] == 1.0


def test_xhand_wrist_frames_are_landmark_derived_and_handed():
    joints = np.zeros((2, 21, 3), dtype=np.float64)
    joints[:, 9, 2] = 1.0
    joints[:, 5, 1] = 1.0
    joints[:, 13, 1] = -1.0
    right = xhand_wrist_frames_from_joints(joints, "right")
    left = xhand_wrist_frames_from_joints(joints, "left")
    np.testing.assert_allclose(right, np.repeat(np.eye(3)[None], 2, axis=0))
    np.testing.assert_allclose(left, np.repeat(np.diag([-1.0, -1.0, 1.0])[None], 2, axis=0))
    np.testing.assert_allclose(np.linalg.det(right), 1.0)
    np.testing.assert_allclose(np.linalg.det(left), 1.0)


def test_xhand_wrist_frames_interpolate_missing_mano_frames():
    joints = np.zeros((5, 21, 3), dtype=np.float64)
    joints[[0, 4], 9, 2] = 1.0
    joints[[0, 4], 5, 1] = 1.0
    joints[[0, 4], 13, 1] = -1.0
    valid = np.array([True, False, False, False, True])

    rotations = xhand_wrist_frames_from_joints(joints, "right", valid)

    np.testing.assert_allclose(rotations, np.repeat(np.eye(3)[None], 5, axis=0))


def test_xhand_wrist_frames_keep_fully_missing_hand_invalid_but_finite():
    joints = np.zeros((3, 21, 3), dtype=np.float64)

    rotations = xhand_wrist_frames_from_joints(joints, "left", np.zeros(3, dtype=bool))

    np.testing.assert_allclose(
        rotations, np.repeat(np.diag([-1.0, -1.0, 1.0])[None], 3, axis=0),
    )


def test_roles_keep_visible_noninteracting_hand_passive():
    count = 4
    objects = np.repeat(np.eye(4)[None], count, axis=0)
    fingertips = np.zeros((count, 2, 5, 3), dtype=np.float64)
    fingertips[:, 0, :, 0] = 0.03
    fingertips[:, 1, :, 0] = 0.30
    valid = np.ones((count, 2), dtype=bool)
    roles, _ = classify_hand_roles(objects, fingertips, valid, 0.04)
    assert roles == ["active", "passive"]
    valid[1:, 1] = False
    roles, _ = classify_hand_roles(objects, fingertips, valid, 0.04)
    assert roles == ["active", "invalid"]


def test_floor_constraint_preserves_active_group_and_passive_hand():
    count = 3
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    objects = np.repeat(np.eye(4)[None], count, axis=0)
    objects[:, 2, 3] = -0.02
    wrists = np.repeat(np.eye(4)[None, None], count * 2, axis=0).reshape(count, 2, 4, 4)
    wrists[:, 0, 2, 3] = -0.04
    wrists[:, 1, 2, 3] = -0.10
    fingertips = np.zeros((count, 2, 5, 3), dtype=np.float64)
    fingertips[:, 0, :, 2] = -0.03
    fingertips[:, 1, :, 2] = -0.08
    relative_before = wrists[:, 0, 2, 3] - objects[:, 2, 3]
    adjusted_object, adjusted_wrist, adjusted_tips, metrics = enforce_simulation_floor(
        mesh, objects, wrists, fingertips, ["active", "passive"],
    )
    assert object_minimum_z(mesh, adjusted_object).min() >= 0.002 - 1e-9
    assert adjusted_wrist[:, :, 2, 3].min() >= 0.060 - 1e-9
    assert adjusted_tips[..., 2].min() >= 0.060 - 1e-9
    np.testing.assert_allclose(
        adjusted_wrist[:, 0, 2, 3] - adjusted_object[:, 2, 3], relative_before,
    )
    assert metrics["passive_hand_constant_shift_m"]["right"] > 0


def test_unphysical_passive_hand_is_rejected_before_floor_shift():
    count = 3
    wrists = np.repeat(np.eye(4)[None, None], count * 2, axis=0).reshape(count, 2, 4, 4)
    fingertips = np.zeros((count, 2, 5, 3), dtype=np.float64)
    wrists[:, 1, 2, 3] = -0.10
    fingertips[:, 1, :, 2] = -0.08

    roles, metrics = reject_unphysical_passive_hands(
        wrists, fingertips, ["active", "passive"],
    )

    assert roles == ["active", "invalid"]
    assert metrics["right"]["rejected"] is True
    assert metrics["right"]["required_floor_correction_m"] > 0.05


def test_sequence_optimizer_writes_valid_artifacts(tmp_path):
    count, height, width = 8, 80, 100
    for relative in ("object_tracking", "hands", "segmentation", "calibration", "mesh_proposals/p0"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    (tmp_path / "frames").mkdir()
    RunManifest.create(
        tmp_path / "manifest.json", run_id="synthetic", source_episode="synthetic",
        config={}, frame_count=count, fps=30.0,
    )
    frame_indices = np.arange(count)
    timestamps = frame_indices / 30.0
    (tmp_path / "frames/frame_index.json").write_text(__import__("json").dumps({
        "frames": [
            {"frame_index": int(index), "source_frame_index": int(index), "rgb_path": "unused"}
            for index in frame_indices
        ]
    }))
    object_pose = np.repeat(np.eye(4)[None], count, axis=0)
    object_pose[:, :3, 3] = [0.0, 0.0, 1.2]
    np.savez_compressed(
        tmp_path / "object_tracking/foundationpose_raw.npz", frame_indices=frame_indices,
        timestamps_s=timestamps, T_camera_object=object_pose, valid=np.ones(count, bool),
        confidence=np.ones(count), registration_frame=np.array([True] + [False] * (count - 1)),
        depth_residual=np.ones(count), mask_iou=np.full(count, 0.5),
    )
    identity = np.eye(3)
    joints = np.zeros((count, 2, 21, 3))
    joints[..., 2] = 0.30
    joints[:, :, 9, 2] = 0.40
    joints[:, :, 5, 1] = 0.03
    joints[:, :, 13, 1] = -0.03
    joints[:, :, [4, 8, 12, 16, 20], 2] = 0.36
    np.savez_compressed(
        tmp_path / "hands/wilor_raw.npz", frame_indices=frame_indices, timestamps_s=timestamps,
        valid=np.ones((count, 2), bool), score=np.ones((count, 2)),
        joints_camera_rootrel=joints, translation_camera=np.zeros((count, 2, 3)),
        mano_global_orient=np.broadcast_to(identity, (count, 2, 3, 3)),
        mano_hand_pose=np.broadcast_to(identity, (count, 2, 15, 3, 3)),
        mano_betas=np.zeros((count, 2, 10)),
    )
    masks = np.zeros((count, height, width), dtype=np.uint8)
    masks[:, 35:45, 45:55] = 1
    np.savez_compressed(
        tmp_path / "segmentation/object_masks.npz", frame_indices=frame_indices,
        timestamps_s=timestamps, masks=masks, valid=np.ones(count, bool),
    )
    np.save(tmp_path / "calibration/T_world_camera.npy", np.repeat(np.eye(4)[None], count, axis=0))
    np.save(tmp_path / "calibration/intrinsics.npy", np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]]))
    trimesh.creation.icosphere(subdivisions=1, radius=0.5).export(tmp_path / "mesh_proposals/p0/visual.obj")
    (tmp_path / "object_tracking/selected_mesh.json").write_text(
        '{"canonical_visual_mesh":"mesh_proposals/p0/visual.obj","scale_to_m":0.08}'
    )
    aligned, contact = optimize_run(tmp_path)
    validate_npz(aligned, "aligned_trajectory")
    validate_npz(contact, "contact")

    with np.load(tmp_path / "hands/wilor_raw.npz") as artifact:
        hands = {key: np.asarray(artifact[key]) for key in artifact.files}
    hands["valid"][:, 0] = False
    hands["score"][:, 0] = 0.0
    hands["joints_camera_rootrel"][:, 0] = 0.0
    np.savez_compressed(tmp_path / "hands/wilor_raw.npz", **hands)

    aligned, contact = optimize_run(tmp_path, overwrite=True)

    with np.load(aligned) as artifact:
        assert artifact["T_sim_wrist"].shape[1] == 1
        assert artifact["valid_hand"].all()
    with np.load(contact) as artifact:
        assert artifact["contact"].shape[1] == 1
    metrics = __import__("json").loads(
        (tmp_path / "optimization/optimization_metrics.json").read_text()
    )
    assert metrics["hands"]["roles"]["left"] == "invalid"
    assert metrics["hands"]["artifact_hand_order"] == ["right"]
