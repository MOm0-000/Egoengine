"""Separate injected audit faults from actually observed historical discrepancies."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from audit_taco_initialization import visual_meshes, world_vertices
from egoengine_repro.retarget.paper_audit import verify_artifacts
from revalidate_taco_pour_diagnostics import BASELINE, SCENE, metric_differences, run

OUTPUT = ROOT / "runs/taco_pour_bug_revalidation_v1"


@pytest.mark.parametrize("old,new", [
    (True, 1), (np.nan, np.nan), (1.0, np.inf),
    ({"distance": 0.0}, {}), ([0, 1], [0]), (3, 3.00000000001),
])
def test_comparison_cannot_hide_invalid_changed_or_missing_metrics(old, new):
    assert metric_differences(old, new)


def test_comparison_allows_only_numeric_roundoff_and_added_metadata():
    assert not metric_differences({"distance": .01}, {"distance": .01 + 1e-14, "new_metadata": False})
    assert metric_differences({"distance": .01}, {"distance": .02})


def test_old_outputs_are_never_overwritten_by_revalidation():
    with pytest.raises(FileExistsError):
        run(OUTPUT)


def test_full_revalidation_distinguishes_fault_injection_from_real_inputs():
    report = json.loads((OUTPUT / "report.json").read_text())
    assert report["discrepancy_count"] == 0
    assert report["original_reference_collision_rows_recomputed"] == 198
    assert all(c["outcome"] == "recomputed_equal" and not c["differences"] for c in report["comparisons"])
    assert report["bug_reproductions_were_injected_faults_not_observed_bad_dataset_rows"]
    assert not report["actual_pour_invalid_numeric_state_found"]
    assert not report["actual_recorded_asset_hash_mismatch_found"]
    assert sum(r["qpos_rows"] for r in report["numeric_archives"]) == 289
    assert len(report["historical_brush_numeric_screen_only"]) == 3
    assert [c["declared_state_feasible"] for c in report["candidate_results"]] == [False, False, True, True]


def test_revalidation_covers_the_previously_unrecomputed_claims():
    report = json.loads((OUTPUT / "report.json").read_text())
    checked = {c["check"] for c in report["comparisons"]}
    assert {"reference_all_198_collision_families", "reference_all_198_fk_velocity_command_and_joint_metrics",
            "reference_all_198_omitted_pair_topology", "thumb_all_saved_states_and_35_sweep_points",
            "v4_all_four_instantaneous_control_cases", "v4_full_native_thumb_intersections",
            "object_all_198_table_clearances_and_selected_native_samples"} <= checked
    assert report["scene_meshes_matched_to_historical_hashes"] == 92
    assert report["source_thumb_kinematics_match"]


def test_revalidation_preserves_inputs_and_does_not_promote_a_reset():
    report = json.loads((OUTPUT / "report.json").read_text())
    verify_artifacts(report["preserved_artifacts"])
    verify_artifacts(report["scene_mesh_dependencies"])
    assert not any(report[k] for k in ("model_changed", "reference_changed", "candidates_changed",
        "legacy_reports_overwritten", "rl_or_physical_rollout_run", "formal_initialization_selected",
        "collision_coverage_plan_progressed", "training_ready"))
    assert report["simulation_steps_executed"] == 0
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    audit = protocol["audit_results"]
    assert Path(audit["active_bug_impact_revalidation"]) == OUTPUT / "report.json"
    assert audit["revalidated_metric_groups"] == report["comparison_count"]
    assert audit["revalidated_metric_discrepancies"] == report["discrepancy_count"]
    assert not audit["injected_bug_counterexamples_are_actual_bad_pour_data"]
    assert not protocol["training_ready"]
    assert "corrected_actor_saturates_on_both_replay_and_ppo_paths_reference_timing_and_action_frame_decision_required" in protocol["blocking_checks"]


def test_original_native_support_measurements_are_recomputed_not_inferred(monkeypatch):
    def forbidden(*_):
        raise AssertionError("revalidation must not execute physics")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    with np.load(BASELINE / "robot_reference.npz", allow_pickle=False) as source:
        qpos = source["qpos"][0]
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    meshes, _ = visual_meshes(SCENE, model)
    old = json.loads((BASELINE / "initialization_audit.json").read_text())
    for record in old["native_visual_support"]:
        geom = model.geom(record["geom"]).id
        points = world_vertices(model, data, geom, meshes[geom])
        clearance = float(points[:, 2].min() - model.geom("floor").pos[2])
        assert clearance == pytest.approx(record["initial_table_clearance_m"], abs=1e-12)
        assert bool(meshes[geom].is_watertight) == record["watertight"]
    np.testing.assert_array_equal(data.qpos, qpos)
    assert data.time == 0
