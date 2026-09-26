import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/taco_pour_single_pass_endpoint47_49_gate_v1.yaml"
REPORT = ROOT / "runs/taco_pour_single_pass_endpoint47_49_gate_v1/report.json"
CANDIDATE = (
    ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_read_only_gate_is_fail_closed_and_reproduces_formal_traces():
    contract = yaml.safe_load(CONTRACT.read_text())
    report = _report()
    assert contract["status"] == "authorized_read_only_gate"
    assert contract["runtime"]["training_allowed"] is False
    assert contract["runtime"]["optimizer_steps"] == 0
    assert contract["runtime"]["chunk_commit_allowed"] is False
    assert contract["frozen_next_candidate"] == {
        "sole_future_change": "actor_learning_rate",
        "baseline": 1e-4,
        "candidate": 5e-5,
        "actor_mini_epochs": 1,
        "approximate_KL_basis": {
            "observed_maximum_post_optimizer_fixed_probe_exact_KL_mean": (
                0.039736180923459874
            ),
            "local_quadratic_step_scaling": 0.25,
            "approximate_candidate_KL": 0.009934045230864968,
        },
        "paper_parameter_recovery_claimed": False,
        "fresh_training_authorized": False,
    }
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["formal_trace_bitwise_reproduced"] == {
        "Replay": True,
        "PPO": True,
    }
    assert report["formal_results"]["Replay"]["successful_intervals"] == 30
    assert report["formal_results"]["PPO"]["successful_intervals"] == 28


def test_both_zero_residual_interventions_fail_the_predeclared_gate():
    report = _report()
    rows = {row["source_endpoint"]: row for row in report["counterfactuals"]}
    assert rows[47]["endpoint_49_score_delta_vs_formal_PPO"] < 0
    assert rows[47]["first_failure_endpoint_or_60"] == 49
    assert rows[47]["survives_endpoint_49"] is False
    assert rows[48]["endpoint_49_score_delta_vs_formal_PPO"] == 0
    assert rows[48]["first_failure_endpoint_or_60"] == 49
    assert rows[48]["survives_endpoint_49"] is False
    gate = report["candidate_evidence_gate"]
    assert gate["per_source_requirements_passed"] is False
    assert gate["joint_survival_requirement_passed"] is False
    assert gate["passed"] is False
    assert gate["candidate_may_await_explicit_training_authorization"] is False
    assert gate["fresh_training_authorized_by_this_gate"] is False


def test_source48_action_changes_hand_but_not_the_decoupled_tool():
    row = {
        item["source_endpoint"]: item for item in _report()["counterfactuals"]
    }[48]
    intervention = row["intervention_step"]
    assert intervention["formal_normalized_action_l2"] > 3
    assert intervention["replacement_normalized_action_l2"] == 0
    assert max(abs(value) for value in intervention[
        "executed_effective_residual_after_ctrlrange"
    ]) < 2e-8
    delta = row["endpoint_49_state_delta_vs_formal_PPO"]
    assert delta["controlled_hand_qpos_l2"] > 0.02
    assert delta["tool_freejoint_qpos_l2"] < 1e-7
    source = _report()["aligned_source46_48_attribution"]["48"]["PPO"]
    assert not any(
        label.startswith("right-tool-")
        for label in source["state_at_source"]["contact"]["active"]
    )


def test_half_lr_candidate_and_protocol_remain_blocked():
    candidate = yaml.safe_load(CANDIDATE.read_text())
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    assert candidate["status"] == (
        "frozen_blocked_not_selected_after_last_OFF_reversal_gate"
    )
    assert candidate["single_change"] == {
        "field": "PPO_actor_learning_rate",
        "baseline": 1e-4,
        "candidate": 5e-5,
    }
    assert candidate["pretraining_read_only_gate"]["passed"] is False
    assert candidate["execution_gate"]["fresh_training_authorized"] is False
    assert candidate["execution_gate"]["fallback_learning_rate_sweep_allowed"] is False
    evidence = candidate["pretraining_read_only_gate"]
    assert _sha256(Path(evidence["contract"]["path"])) == evidence["contract"]["sha256"]
    assert _sha256(Path(evidence["report"]["path"])) == evidence["report"]["sha256"]
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == (
        "final_OFF_reversal_does_not_restore_endpoint57_source56_semantic_authority_attribution_required"
    )
    assert protocol["blocking_checks"] == [
        "final_OFF_reversal_does_not_restore_endpoint57_source56_semantic_authority_attribution_required"
    ]
