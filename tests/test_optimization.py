import numpy as np
import trimesh

from video_to_spider.manifest import RunManifest
from video_to_spider.optimization.contact import infer_contact
from video_to_spider.optimization.sequence import optimize_run
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
    joints[:, :, :, 0] = np.linspace(-0.01, 0.01, 21)
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
