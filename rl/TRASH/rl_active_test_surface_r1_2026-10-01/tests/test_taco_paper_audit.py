"""Paper-first diagnostic classifications must not create physics pass claims."""

from pathlib import Path
import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from egoengine_repro.retarget.paper_audit import (
    alignment_invariants, artifact, input_status, project_world, support_clearance, transform_report,
)


def test_rigid_transform_audit_rejects_reflection_and_nonfinite():
    transform = np.eye(4)[None]
    assert transform_report(transform)["rigid_within_float32_tolerance"]
    transform[0, 0, 0] = -1
    assert not transform_report(transform)["rigid_within_float32_tolerance"]
    transform[0, 0, 0] = np.nan
    assert not transform_report(transform)["finite"]


def test_common_alignment_preserves_hand_object_and_camera_relations():
    rng = np.random.default_rng(7)
    objects = np.tile(np.eye(4), (3, 2, 1, 1))
    objects[..., :3, :3] = Rotation.random(6, random_state=rng).as_matrix().reshape(3, 2, 3, 3)
    objects[..., :3, 3] = rng.normal(size=(3, 2, 3))
    camera = np.tile(np.eye(4), (3, 1, 1))
    camera[:, :3, :3] = Rotation.random(3, random_state=rng).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [.1, .3, -.2]).as_matrix()
    transform[:3, 3] = [.6, .1, .72]
    joints = rng.normal(size=(3, 2, 21, 3))
    assert max(alignment_invariants(joints, objects, camera, transform).values()) < 1e-12


def test_points_behind_camera_are_not_accepted_from_in_image_coordinates():
    camera = np.eye(4)[None]
    points = np.array([[[0., 0., -1.], [0., 0., 1.], [0., 0., 0.]]])
    report = project_world(points, camera, np.eye(3), [100, 100])
    assert report["positive_depth_fraction"] == pytest.approx(1 / 3)
    assert report["front_and_in_image_fraction"] == pytest.approx(1 / 3)


def test_support_uses_every_vertex_and_pose():
    vertices = np.array([[0, 0, -.1], [0, 0, .2], [0, 1, 0.]])
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, 2, 3] = [.72, .82]
    np.testing.assert_allclose(support_clearance(vertices, poses, .72), [-.1, 0], atol=1e-12)


def test_media_mismatch_does_not_silently_truncate_or_reject_world_gt():
    counts = dict(hands=361, tool=361, target=361, camera=361, rgb=358, depth=356)
    status = input_status(counts, True, True, True)
    assert status["gt_structurally_usable"]
    assert not status["media_frame_counts_match"]
    assert status["physical_compatibility"] == "unvalidated"
    assert status["task_success"] == "not_evaluated"


def test_bad_camera_does_not_become_an_automatic_calibration_repair():
    counts = dict(hands=400, tool=400, target=400, camera=400, rgb=400)
    status = input_status(counts, True, True, False)
    assert status["gt_structurally_usable"]
    assert not status["released_camera_front_check_passed"]
    assert status["frame_count_match_is_not_temporal_correspondence_proof"]


def test_real_input_audit_preserves_four_episodes_and_reports_known_mismatches():
    path = ROOT / "runs/paper_first_audit_v1/inputs.json"
    report = json.loads(path.read_text())
    episodes = {e["task"]: e for e in report["episodes"]}
    assert set(episodes) == {"brush", "cut", "skim", "smear"}
    assert episodes["cut"]["counts"]["hands"] == 361
    assert episodes["cut"]["counts"]["rgb"] == 358
    assert episodes["cut"]["counts"]["depth_original"] == 356
    assert not episodes["skim"]["status"]["released_camera_front_check_passed"]
    assert all(e["status"]["gt_structurally_usable"] for e in episodes.values())
    assert not report["frames_trimmed"]
    assert not report["artifacts_modified"]


def test_initialization_report_does_not_accept_or_apply_a_reset():
    path = ROOT / "runs/paper_first_audit_v1/initialization.json"
    report = json.loads(path.read_text())
    assert report["topology"]["ik_runtime_pairs_equal"]
    assert report["topology"]["hand_shell_geometry_equal_pinned_source"]
    assert report["topology"]["omitted"]["omitted_intrahand_pairs"] == 102
    assert report["initial_distances"]["hand_floor"]["min_distance_m"] < -.013
    assert report["alignment"]["native_compiled_scale_check_max_error_m"] < 1e-7
    assert not report["strict_gate_passed"]
    assert not report["state_projection_applied"]
    assert report["simulation_steps_executed"] == 0
    for source in report["preserved_artifacts"]:
        assert artifact(Path(source["path"])) == source
