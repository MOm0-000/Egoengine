"""Failure-injection regressions for collision audit false-pass paths."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from audit_taco_thumb_assembly import kinematic_source_check
from audit_taco_initial_contacts import overlapping_bounds
from audit_taco_initial_controls import forward_case
from diagnose_taco_initial_hands import audit_saved_candidate
from egoengine_repro.retarget.collision_audit import distances
from egoengine_repro.retarget.initial_hand import state_summary
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
BASELINE = ROOT / "runs/taco_pour_bimanual_gt_v1"
PREVIOUS = ROOT / "runs/taco_pour_initial_hand_v3/initial_hand_candidate.npz"
LATEST = ROOT / "runs/taco_pour_initial_hand_v4"
URDF = ROOT / "runs/taco_pour_thumb_assembly_v1/xhand_left.urdf"


@pytest.fixture
def state():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    with np.load(LATEST / "initial_hand_candidate.npz", allow_pickle=False) as source:
        qpos = source["qpos"][0]
    return model, qpos


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_object_position_never_passes_feasibility(state, value):
    model, qpos = state
    qpos[36] = value
    with pytest.raises(ValueError, match="finite"):
        state_summary(model, qpos)


@pytest.mark.parametrize("scale", [0, .5, 2])
def test_invalid_object_quaternion_is_rejected_not_silently_normalized(state, scale):
    model, qpos = state
    qpos[39:43] *= scale
    with pytest.raises(ValueError, match="unit quaternion"):
        state_summary(model, qpos)


def test_nonfinite_distance_is_not_interpreted_as_zero_violations(state, monkeypatch):
    model, qpos = state
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *_: np.nan)
    with pytest.raises(ValueError, match="nonfinite.*distance"):
        distances(model, data, [(model.geom("floor").id, model.geom("collision_hand_right_palm_0").id)])


def test_joint_origin_mismatch_changes_source_conclusion(state):
    model, _ = state
    model.body_pos[model.body("left_hand_thumb_rota_link1").id, 0] += .01
    report = kinematic_source_check(model, URDF, "left")
    assert not report["matches_source_within_tolerance"]
    assert "mismatch" in report["interpretation"]


def test_joint_parent_mismatch_is_checked_even_with_identical_local_numbers(state):
    model, _ = state
    model.body_parentid[model.body("left_hand_thumb_rota_link1").id] = model.body("left_hand_link").id
    report = kinematic_source_check(model, URDF, "left")
    assert not report["matches_source_within_tolerance"]
    assert not report["joints"][1]["parent_matches"]


def test_stale_native_mesh_report_rejected_before_geometry_loading(tmp_path, monkeypatch):
    import audit_taco_initialization

    report = json.loads((LATEST / "report.json").read_text())
    native = json.loads((LATEST / "native_geometry_audit.json").read_text())
    native["source_assets"][0]["sha256"] = "0" * 64
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "native_geometry_audit.json").write_text(json.dumps(native))

    def must_not_load(*_):
        raise AssertionError("stale audit reached geometry loading")

    monkeypatch.setattr(audit_taco_initialization, "visual_meshes", must_not_load)
    with pytest.raises(ValueError, match="artifact changed"):
        audit_saved_candidate(SCENE, BASELINE, tmp_path, frozen_reference=PREVIOUS)
    assert not (tmp_path / "audit.json").exists()


def test_valid_candidate_and_equivalent_quaternion_sign_remain_accepted(state):
    model, qpos = state
    original = qpos.copy()
    assert state_summary(model, qpos)["declared_state_feasible"]
    np.testing.assert_array_equal(qpos, original)
    qpos[39:43] *= -1
    assert state_summary(model, qpos)["declared_state_feasible"]


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -1])
def test_invalid_feasibility_tolerance_cannot_disable_collision_checks(state, tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        state_summary(*state, tolerance=tolerance)


def test_nonfinite_aabb_is_unknown_not_clear():
    with pytest.raises(ValueError, match="finite"):
        overlapping_bounds(np.array([[np.nan, 0, 0]]), np.zeros((1, 3)))


def test_nonfinite_controls_rejected_before_forward_dynamics(state, monkeypatch):
    def must_not_evaluate(*_):
        raise AssertionError("invalid state reached forward dynamics")

    monkeypatch.setattr(mujoco, "mj_forward", must_not_evaluate)
    model, qpos = state
    with pytest.raises(ValueError, match="finite.*qvel"):
        forward_case(model, qpos, np.zeros(model.nv), np.full(model.nu, np.nan))


def test_mesh_snapshot_includes_external_collision_assets_and_detects_change(tmp_path):
    mesh = tmp_path / "collision.obj"
    mesh.write_bytes(b"initial mesh fixture")
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco><asset><mesh name="collision" file="collision.obj"/></asset></mujoco>')
    snapshot = scene_mesh_artifacts(scene)
    assert snapshot == [artifact(mesh)]
    mesh.write_bytes(b"modified mesh fixture")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_artifacts(snapshot)
    assert len(scene_mesh_artifacts(SCENE)) == 92


def test_native_inventory_exposes_bodies_outside_the_shell_pair_universe():
    report = json.loads((ROOT / "runs/taco_pour_collision_review_v1/report.json").read_text())
    inventory = report["inventory"]
    assert inventory["native_hand_mesh_count"] == 26 and inventory["hand_shell_count"] == 24
    assert inventory["native_open_mesh_count"] == 17
    assert inventory["native_meshes_without_same_body_shell"] == ["right_index_bend_visual", "left_index_bend_visual"]
    assert inventory["shell_intrahand_declared"] == 30
    assert inventory["shell_intrahand_omitted"] == 102
    assert inventory["omitted_nonadjacent_shell_pairs"] == 82
    assert inventory["native_body_pair_counts"]["intrahand"]["possible_native_body_pairs"] == 156
    assert not inventory["coverage_complete"]


def test_collision_review_preserves_history_and_is_not_a_model_fix():
    report = json.loads((ROOT / "runs/taco_pour_collision_review_v1/report.json").read_text())
    assert report["original_frames_validated"] == 198
    assert report["existing_v4_declared_state_results_unchanged"]
    assert not report["model_or_pairs_modified"] and not report["reference_or_candidate_modified"]
    assert not report["formal_pipeline_method_selected"] and not report["accepted_as_reset"]
    assert not report["old_native_audit_had_full_collision_mesh_snapshot"]
    assert report["simulation_steps_executed"] == 0 and not report["training_ready"]
    verify_artifacts(report["preserved_artifacts"])
    verify_artifacts(report["current_full_scene_mesh_snapshot"])
