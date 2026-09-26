import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_postfix_policy_extremization_audit_v1"


def test_postfix_policy_extremization_audit_is_read_only_and_complete():
    report = json.loads((RUN / "report.json").read_text())
    assert report["status"] == "completed_read_only_no_algorithm_change"
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["checkpoint_resume_for_training"] is False
    assert report["chunk_commit_written"] is False
    assert report["source_run"]["formal_trace_bitwise_reproduced"] == {
        "Replay": True,
        "PPO": True,
    }
    assert report["decision_boundary"]["second_fresh_PPO_authorized"] is False


def test_failure_and_patch_chain_match_the_frozen_evidence():
    report = json.loads((RUN / "report.json").read_text())
    failure = report["failure_attribution"]
    endpoint = failure["per_mode"]["ppo"]["40"]
    assert endpoint["terminated"] is True
    assert endpoint["objective_score"] > 1.0
    assert endpoint["position_error_m"] < 0.12
    assert endpoint["rotation_error_rad"] < 1.5
    assert endpoint["policy"]["unit_bound_count"] == 7
    assert endpoint["residual"]["lost_max_abs"] < 1e-6
    assert failure["first_outcome_endpoint_with_worse_PPO_score_in_window"] == 35

    evolution = report["fixed_probe_evolution"]
    assert evolution["final_patch_chain_exact"] is True
    assert evolution["stages"] == 41
    assert len(evolution["optimizer_transitions"]) == 32
    assert len(evolution["RMS_commit_transitions"]) == 8
    focus = evolution["failure_focus"]["by_transition_type"]
    assert focus["optimizer"]["net_signed_mu_change_toward_final_saturation"] > 0
    assert focus["RMS_commit"]["net_signed_mu_change_toward_final_saturation"] < 0
    assert focus["optimizer"]["transitions_increasing_focus_saturation_count"] == 17
    assert focus["optimizer"]["transitions_decreasing_focus_saturation_count"] == 0


def test_update_and_stochastic_audits_report_all_frozen_rows():
    report = json.loads((RUN / "report.json").read_text())
    updates = report["update_dynamics"]["updates"]
    assert len(updates) == 32
    assert updates[0]["ratio"]["min"] == 1.0
    assert updates[0]["ratio"]["max"] == 1.0
    assert max(
        row["ratio_outside_0_8_1_2_fraction"] for row in updates
    ) == 0.8375
    assert max(
        row["exact_distribution_KL_old_to_current"]["mean"] for row in updates
    ) > 0.2

    stochastic = report["deterministic_vs_stochastic"]["stochastic"]
    assert stochastic["all_rollouts_reported"] is True
    assert stochastic["selection_or_best_of_allowed"] is False
    assert len(stochastic["seeds"]) == 32
    assert len(stochastic["rows"]) == 32
    assert stochastic["reached_endpoint_40_count"] == 29
    assert stochastic["passed_endpoint_40_count"] == 28
    assert stochastic["full_40_step_pass_count"] == 0


def test_single_actor_pass_candidate_is_frozen_but_not_authorized():
    candidate_path = ROOT / "configs/taco_pour_postfix_single_actor_pass_candidate_v1.yaml"
    candidate = yaml.safe_load(candidate_path.read_text())
    assert candidate["status"] == "frozen_candidate_not_authorized_to_run"
    assert candidate["single_change"] == {
        "field": "PPO_actor_mini_epochs",
        "baseline": 4,
        "candidate": 1,
        "actor_updates_per_training_epoch": 1,
        "total_actor_updates_over_8_epochs": 8,
    }
    assert candidate["unchanged"]["learning_rate"] == 0.0001
    assert candidate["unchanged"]["asymmetric_critic_mini_epochs"] == 4
    assert candidate["execution_gate"]["separately_authorized_fresh_training_required"] is True
    evidence = Path(candidate["evidence"]["report"]["path"])
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == candidate["evidence"]["report"]["sha256"]
