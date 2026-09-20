"""Initial-hand search must preserve objects/GT and remain separate from a reset."""

import json
from pathlib import Path
import sys

import mink
import mujoco
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from audit_taco_initial_contacts import overlapping_bounds
from diagnose_taco_initial_hands import run
from egoengine_repro.retarget.initial_hand import (
    above_fixed_geometry_seed, solve_initial_hands, state_summary,
)
from egoengine_repro.retarget.paper_audit import artifact, verify_artifacts

BASELINE = ROOT / "runs/taco_pour_bimanual_gt_v1"
SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
LATEST = ROOT / "runs/taco_pour_initial_hand_v3"
ACTIVE = ROOT / "runs/taco_pour_bimanual_mano_fk_right_guard_v1"
VELOCITY = dict(base_translation=1.5, base_rotation=4.0, finger=8.0)


def baseline_qpos(run=BASELINE):
    with np.load(run / "robot_reference.npz", allow_pickle=False) as source:
        return source["qpos"][0]


def test_invalid_solver_inputs_fail_before_search():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    with pytest.raises(ValueError, match="one finite"):
        solve_initial_hands(model, np.zeros((1, 50)), VELOCITY)
    with pytest.raises(ValueError, match="positive source"):
        solve_initial_hands(model, baseline_qpos(), VELOCITY, source_dt=0)


def test_geometric_seed_changes_only_named_world_z_hand_joints():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    before = baseline_qpos()
    seed, report = above_fixed_geometry_seed(model, before)
    expected = sorted(int(model.joint(f"{prefix}_forearm_tz_link_joint").qposadr[0]) for prefix in ("R", "L"))
    assert np.flatnonzero(seed != before).tolist() == expected
    np.testing.assert_array_equal(seed[36:], before[36:])
    assert not report["executed_as_motion"]
    assert state_summary(model, seed)["declared_state_feasible"]


def test_fixed_object_violation_is_not_repaired_by_the_hand_solver():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    invalid = baseline_qpos()
    invalid[int(model.joint("left_object_joint").qposadr[0]) + 2] -= .05
    with pytest.raises(ValueError, match="fixed objects"):
        solve_initial_hands(model, invalid, VELOCITY)


def test_existing_diagnostic_outputs_are_never_overwritten():
    with pytest.raises(FileExistsError):
        run(SCENE, BASELINE, LATEST)


def test_failed_solver_exports_a_failure_not_a_reset(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise mink.NoSolutionFound("injected-test-failure")

    monkeypatch.setattr(mink, "solve_ik", fail)
    output = tmp_path / "failed"
    report = run(SCENE, ACTIVE, output, max_iterations=1)
    assert not report["candidate_declared_state_feasible"]
    assert not report["accepted_as_reset"]
    assert (output / "failed_initial_hand_candidate.npz").is_file()
    assert not (output / "initial_hand_candidate.npz").exists()
    assert report["solver"]["iterations"] == 0
    with np.load(output / "failed_initial_hand_candidate.npz", allow_pickle=False) as data:
        assert "qvel" not in data and "ctrl" not in data
        np.testing.assert_array_equal(data["qpos"][0], baseline_qpos(ACTIVE))


def test_saved_attempts_preserve_failures_and_candidate_scope():
    for version, feasible in ((1, False), (2, False), (3, True)):
        directory = ROOT / f"runs/taco_pour_initial_hand_v{version}"
        report = json.loads((directory / "report.json").read_text())
        assert report["candidate_declared_state_feasible"] is feasible
        assert report["baseline_frames"] == 198
        assert report["solver"]["constrained_hand_pair_count"] == 1734
        assert not report["accepted_as_reset"] and not report["qvel_selected"]
        assert report["simulation_steps_executed"] == 0
        verify_artifacts(report["preserved_artifacts"])
        with np.load(report["candidate"]["path"], allow_pickle=False) as candidate:
            assert candidate["qpos"].shape == (1, 50)
            np.testing.assert_array_equal(candidate["qpos"][0, 36:], baseline_qpos()[36:])
            assert "ctrl" not in candidate and "qvel" not in candidate
        with np.load(directory / "solver_iterations.npz", allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace["qpos"][:, 36:], np.tile(baseline_qpos()[36:], (len(trace["qpos"]), 1)))


def test_candidate_independently_passes_declared_not_complete_geometry():
    report = json.loads((LATEST / "audit.json").read_text())
    assert report["independent_state_check"]["declared_state_feasible"]
    assert report["native_hand_table_min_clearance_m"] > 0
    assert report["native_hand_object_sampled_inside_points"] == 0
    assert report["native_hand_object_aabb_pairs_checked"] == 52
    assert not report["complete_intrahand_geometry_coverage"]
    assert not report["accepted_as_reset"] and not report["training_ready"]
    assert not report["omitted_intrahand_coverage"]["decisions_applied"]


def test_native_aabb_audit_does_not_skip_checks_when_shells_are_clear():
    first = np.array([[0, 0, 0], [1, 1, 1]])
    assert overlapping_bounds(first, first + .5)
    assert not overlapping_bounds(first, first + 2)
    report = json.loads((LATEST / "native_geometry_audit.json").read_text())
    assert report["input_is_projection_diagnostic_candidate"]
    assert len(report["native_aabb_checks"]) == 52
    assert len(report["records"]) > 0
    assert "independent of collision-shell" in report["selection"]
    assert all(d.get("inside_samples_50um", 0) == 0 for r in report["records"] for d in r["directions"])


def test_next_frame_check_is_not_a_replay_success_or_reference_edit():
    report = json.loads((LATEST / "report.json").read_text())
    next_frame = report["next_reference"]
    assert next_frame["max_speed_limit_ratio"] < 1
    assert next_frame["interpolation_samples"][0]["declared_state_feasible"]
    assert not next_frame["interpolation_samples"][-1]["declared_state_feasible"]
    assert not next_frame["reference_row_1_modified"]
    assert not next_frame["physical_transition_simulated"]
    assert not report["source_reference_modified"]


def test_solver_retains_a_feasible_candidate_without_claiming_an_optimum():
    report = json.loads((LATEST / "report.json").read_text())
    solver = report["solver"]
    assert solver["retained_feasible_iteration"] is not None
    assert solver["retained_normalized_posture_cost"] > 0
    assert not solver["globally_minimum_correction_proven"]
    assert solver["termination_reason"].startswith("qp_failed")
    assert solver["distance_acceptance_tolerance_m"] == 1e-6
    assert solver["minimum_distance_m"] == 0
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    candidate = protocol["initialization"]["diagnostic_candidate"]
    assert candidate["declared_state_feasible"]
    assert not candidate["accepted_as_reset"] and not candidate["qvel_selected"]
    assert Path(protocol["inputs"]["historical_orientation_bug_baseline"]) == BASELINE
    assert Path(protocol["inputs"]["active_robot_reference"]) == (
        ROOT / "runs/taco_pour_bimanual_mano_fk_right_guard_v1/robot_reference.npz")
    assert candidate["baseline"] == "historical_orientation_bug_reference_not_current_mano_fk"
    assert not protocol["audit_results"]["new_reset_applied"] and not protocol["training_ready"]
