import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_source57_temporal_hold_gate_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_source57_temporal_hold_gate_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_temporal_hold_gate_is_read_only_and_reproduces_the_parent_path():
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
    assert contract["runtime"]["learned_gate_training_allowed"] is False
    assert contract["runtime"]["chunk_commit_allowed"] is False


def test_all_copied_residuals_are_feasible_and_no_silent_clamp_is_used():
    report = _report()
    branches = report["branches"]
    assert len(branches) == 6
    assert all(row["inside_source57_state_feasible_bounds"] for row in branches)
    assert all(row["executed"] for row in branches)
    assert report["measurement_boundaries"][
        "all_copied_actions_checked_against_source57_support"
    ] is True
    assert report["measurement_boundaries"]["no_silent_clamp"] is True
    assert report["measurement_boundaries"]["no_source57_axis_or_scale_sweep"] is True
    arrays = np.load(RUN / "temporal_hold_branches.npz")
    assert arrays["feasible"].tolist() == [True] * 6
    assert arrays["action"].shape == (6, 36)


def test_no_hold_branch_passes_and_strong_pinky_force_is_not_sufficient():
    report = _report()
    branches = {row["name"]: row for row in report["branches"]}
    assert {
        name: row["outcome"]["objective_score"]
        for name, row in branches.items()
    } == {
        "formal_source57_PPO": 1.0227254629135132,
        "translation_OFF": 1.023302674293518,
        "hold_source56_right_wrist_rotation": 1.0225476026535034,
        "hold_source56_right_fingers": 1.0230097770690918,
        "hold_source56_right_wrist_rotation_and_fingers": 1.0228244066238403,
        "hold_source56_entire_right_hand": 1.0232295989990234,
    }
    assert all(row["passes_endpoint58"] is False for row in branches.values())
    assert all(row["continuation"] is None for row in branches.values())
    assert branches["formal_source57_PPO"]["endpoint58_contact"][
        "sum_normal_force"
    ] == 0.17438175529241562
    assert branches["translation_OFF"]["endpoint58_contact"][
        "sum_normal_force"
    ] == 18.338611602783203
    assert branches["translation_OFF"]["passes_endpoint58"] is False
    assert report["decision"]["passing_candidates"] == []
    assert report["decision"]["any_temporal_hold_passes_endpoint58"] is False
    assert report["decision"]["best_successful_intervals"] == 37
    assert report["decision"]["forty_of_forty"] is False


def test_protocol_moves_to_active_lag_correction_and_keeps_training_blocked():
    report = _report()
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "source57_temporal_hold_insufficient_active_lag_correction_required"
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"]["source57_temporal_hold_gate"]
    assert gate["passing_candidates"] == []
    assert gate["temporal_hold_switch_sufficient"] is False
    assert gate["strong_pinky_force_sufficient_for_tracking"] is False
    assert gate["actor_LR_5e_minus_5_unblocked"] is False
    assert gate["learned_gate_training_authorized"] is False
    assert gate["reward_change_authorized"] is False
    assert gate["PPO_retraining_authorized"] is False
    assert gate["chunk_acceptance_or_commit_authorized"] is False
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_source57_temporal_hold_gate"
    )
    assert half_lr["latest_state_entry_evidence"][
        "source57_temporal_hold_report"
    ]["sha256"] == _sha256(REPORT)
    assert report["decision"]["PPO_retraining_authorized"] is False
    assert report["decision"]["chunk_acceptance_or_commit_authorized"] is False
