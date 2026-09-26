import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_source57_active_lag_correction_gate_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_source57_active_lag_correction_gate_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_active_lag_gate_is_read_only_and_reproduces_the_parent_path():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert report["status"] == "completed_read_only_gate"
    assert report["paper_faithful"] is False
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert report["regression_gate"] == {
        "tail_selected_sources44_through56_exactly_reproduced": True,
        "formal_source57_PPO_anchor_exactly_reproduced": True,
        "translation_OFF_anchor_exactly_reproduced": True,
        "source57_actor_forward_calls": 1,
        "same_source57_snapshot_and_post_forward_hidden_for_all_candidates": True,
    }
    assert contract["runtime"]["training_allowed"] is False
    assert contract["runtime"]["chunk_commit_allowed"] is False


def test_unit_gain_correction_is_exact_and_only_changes_translation():
    report = _report()
    correction = report["active_correction"]
    np.testing.assert_allclose(
        correction["normalized_delta_before_projection"],
        [0.2912306785583496, -0.9234150499105453, 1.9603002071380615],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        correction["translation_before_projection"],
        [-0.107322096824646, -0.25023405253887177, 2.598892867565155],
        rtol=0.0,
        atol=0.0,
    )
    assert correction["translation_after_projection"] == [
        -0.107322096824646,
        -0.25023405253887177,
        1.0,
    ]
    assert correction["projection_changed_axes"] == ["z"]
    branches = {row["name"]: row for row in report["branches"]}
    formal = np.asarray(branches["formal_source57_PPO"]["normalized_action"])
    active = np.asarray(
        branches["PPO_plus_unit_tool_position_lag_correction"]["normalized_action"]
    )
    assert np.array_equal(active[3:], formal[3:])
    arrays = np.load(RUN / "active_lag_correction_branches.npz")
    assert arrays["action"].shape == (3, 36)


def test_active_correction_improves_score_but_does_not_pass_endpoint58():
    report = _report()
    branches = {row["name"]: row for row in report["branches"]}
    assert {
        name: row["outcome"]["objective_score"]
        for name, row in branches.items()
    } == {
        "formal_source57_PPO": 1.0227254629135132,
        "translation_OFF": 1.023302674293518,
        "PPO_plus_unit_tool_position_lag_correction": 1.017898440361023,
    }
    active = branches["PPO_plus_unit_tool_position_lag_correction"]
    assert active["outcome"]["position_error_m"] == 0.11264189332723618
    assert active["outcome"]["rotation_error_rad"] == 0.5905364155769348
    assert active["endpoint58_contact"]["sum_normal_force"] == 0.0
    assert active["passes_endpoint58"] is False
    assert active["continuation"] is None
    assert report["decision"]["best_successful_intervals"] == 37
    assert report["decision"]["forty_of_forty"] is False
    assert report["decision"]["gain_or_axis_sweep_authorized"] is False


def test_protocol_records_failed_candidate_and_keeps_all_training_blocked():
    report = _report()
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "gate_A_active_corrective_parameterization_decision_required"
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"]["source57_active_lag_correction_gate"]
    assert gate["active_candidate_passes_endpoint58"] is False
    assert gate["gain_or_axis_sweep_authorized"] is False
    assert gate["actor_LR_5e_minus_5_unblocked"] is False
    assert gate["learned_gate_training_authorized"] is False
    assert gate["PPO_retraining_authorized"] is False
    assert gate["chunk_acceptance_or_commit_authorized"] is False
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_gate_A_action_feasibility_classification"
    )
    assert half_lr["latest_state_entry_evidence"][
        "source57_active_lag_correction_report"
    ]["sha256"] == _sha256(REPORT)
    assert report["decision"]["PPO_retraining_authorized"] is False
    assert report["decision"]["chunk_acceptance_or_commit_authorized"] is False
