import json

import cv2
import numpy as np
import trimesh

from video_to_spider.visualization import (
    FOUNDATIONPOSE_VIDEO,
    MESH_VIDEO,
    OPTIMIZATION_VIDEO,
    render_run_visualizations,
)
from video_to_spider.eval.metrics import build_run_report


def _make_visualization_run(root):
    count, height, width = 4, 96, 128
    for relative in (
        "input", "frames/rgb", "calibration", "segmentation", "mesh_proposals/p0",
        "mesh_proposals/p1", "object_tracking", "optimization",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "input/source.json").write_text(json.dumps({
        "video": {"fps": 5.0, "width": width, "height": height},
    }))
    rows = []
    for index in range(count):
        image = np.full((height, width, 3), (35, 42, 50), dtype=np.uint8)
        cv2.rectangle(image, (42 + index, 31), (84 + index, 65), (85, 115, 145), -1)
        relative = f"frames/rgb/{index:06d}.jpg"
        assert cv2.imwrite(str(root / relative), image)
        rows.append({
            "frame_index": index, "source_frame_index": index,
            "timestamp_s": index / 5.0, "rgb_path": relative,
        })
    (root / "frames/frame_index.json").write_text(json.dumps({"frames": rows}))
    K = np.array([[110.0, 0.0, width / 2], [0.0, 110.0, height / 2], [0.0, 0.0, 1.0]])
    np.save(root / "calibration/intrinsics.npy", K)
    np.save(root / "calibration/T_world_camera.npy", np.repeat(np.eye(4)[None], count, axis=0))

    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    mesh.export(root / "mesh_proposals/p0/visual.obj")
    second = trimesh.creation.box(extents=(1.0, 0.3, 0.8))
    second.export(root / "mesh_proposals/p1/visual.obj")
    ranking = {
        "proposals": [
            {"proposal_id": "p0", "qualified": True, "rank": 1, "static_score": 0.8,
             "visual_mesh": "p0/visual.obj"},
            {"proposal_id": "p1", "qualified": True, "rank": 2, "static_score": 0.6,
             "visual_mesh": "p1/visual.obj"},
        ],
    }
    (root / "mesh_proposals/mesh_ranking.json").write_text(json.dumps(ranking))
    (root / "object_tracking/selected_mesh.json").write_text(json.dumps({
        "proposal_id": "p0", "canonical_visual_mesh": "mesh_proposals/p0/visual.obj",
        "scale_to_m": 0.12,
    }))

    frame_indices = np.arange(count, dtype=np.int64)
    timestamps = frame_indices / 5.0
    masks = np.zeros((count, height, width), dtype=np.uint8)
    masks[:, 33:64, 47:81] = 1
    np.savez_compressed(
        root / "segmentation/object_masks.npz", frame_indices=frame_indices,
        timestamps_s=timestamps, masks=masks, valid=np.ones(count, bool),
    )
    raw_pose = np.repeat(np.eye(4)[None], count, axis=0)
    raw_pose[:, 2, 3] = 1.0
    raw_pose[:, 0, 3] = np.array([-0.02, 0.03, -0.01, 0.01])
    np.savez_compressed(
        root / "object_tracking/foundationpose_raw.npz", frame_indices=frame_indices,
        timestamps_s=timestamps, T_camera_object=raw_pose,
        valid=np.array([True, True, False, True]), confidence=np.array([0.8, 0.7, 0.0, 0.75]),
        registration_frame=np.array([True, False, False, True]),
        depth_residual=np.full(count, 0.01), mask_iou=np.array([0.6, 0.5, 0.0, 0.55]),
        segment_id=np.array([0, 0, -1, 1]),
    )

    aligned_pose = raw_pose.copy()
    aligned_pose[:, 0, 3] = np.linspace(-0.01, 0.01, count)
    fingertips = np.zeros((count, 2, 5, 3), dtype=np.float32)
    fingertips[..., 2] = 1.0
    fingertips[:, 0, :, 0] = -0.03
    fingertips[:, 1, :, 0] = 0.03
    identity = np.eye(4)
    np.savez_compressed(
        root / "optimization/aligned_trajectory.npz", frame_indices=frame_indices,
        timestamps_s=timestamps, T_sim_object=aligned_pose[:, None],
        T_sim_wrist=np.broadcast_to(identity, (count, 2, 4, 4)),
        fingertips_sim=fingertips, object_scale_to_m=np.array([0.08]),
        valid_object=np.array([[True], [True], [False], [True]]),
        valid_hand=np.ones((count, 2), bool),
    )
    contacts = np.zeros((count, 2, 5), dtype=np.float32)
    contacts[1:3, 0, 1] = 1.0
    np.savez_compressed(
        root / "optimization/contact.npz", frame_indices=frame_indices,
        timestamps_s=timestamps, contact=contacts,
    )
    (root / "optimization/optimization_metrics.json").write_text(json.dumps({
        "T_sim_world": identity.tolist(),
        "raw_vs_aligned": {
            "object_translation_acceleration_jitter_raw_m_s2": 4.0,
            "object_translation_acceleration_jitter_aligned_m_s2": 0.5,
            "object_mask_centroid_reprojection_raw_px": 8.0,
            "object_mask_centroid_reprojection_aligned_px": 2.0,
        },
    }))


def test_run_visualizations_write_decodable_videos_and_manifest(tmp_path):
    _make_visualization_run(tmp_path)
    manifest_path = render_run_visualizations(
        tmp_path, max_side=160, mesh_duration_s=0.4,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["diagnostic_only"] is True
    assert manifest["ground_truth_consumed"] is False
    assert manifest["invalid_gaps_preserved"] is True
    assert set(manifest["outputs"]) == {
        f"visualization/{MESH_VIDEO}",
        f"visualization/{FOUNDATIONPOSE_VIDEO}",
        f"visualization/{OPTIMIZATION_VIDEO}",
    }
    report = json.loads(build_run_report(tmp_path).read_text())
    assert report["diagnostics"]["visualization"]["status"] == "available"
    for name in (MESH_VIDEO, FOUNDATIONPOSE_VIDEO, OPTIMIZATION_VIDEO):
        path = tmp_path / "visualization" / name
        assert path.stat().st_size > 1000
        capture = cv2.VideoCapture(str(path))
        success, frame = capture.read()
        capture.release()
        assert success
        assert frame.size > 0


def test_run_visualizations_support_single_right_hand_artifact(tmp_path):
    _make_visualization_run(tmp_path)
    aligned_path = tmp_path / "optimization/aligned_trajectory.npz"
    with np.load(aligned_path) as artifact:
        aligned = {key: np.asarray(artifact[key]) for key in artifact.files}
    for key in ("T_sim_wrist", "fingertips_sim", "valid_hand"):
        aligned[key] = aligned[key][:, 1:2]
    np.savez_compressed(aligned_path, **aligned)
    contact_path = tmp_path / "optimization/contact.npz"
    with np.load(contact_path) as artifact:
        contact = {key: np.asarray(artifact[key]) for key in artifact.files}
    contact["contact"] = contact["contact"][:, 1:2]
    np.savez_compressed(contact_path, **contact)
    metrics_path = tmp_path / "optimization/optimization_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["hands"] = {"artifact_hand_order": ["right"]}
    metrics_path.write_text(json.dumps(metrics))

    manifest_path = render_run_visualizations(
        tmp_path, max_side=160, mesh_duration_s=0.4,
    )

    assert (tmp_path / "visualization" / OPTIMIZATION_VIDEO).is_file()
    assert OPTIMIZATION_VIDEO in manifest_path.read_text()
