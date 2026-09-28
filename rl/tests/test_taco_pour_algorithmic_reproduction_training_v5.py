import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from video_to_spider.rl.algorithmic_training_v5 import truncated_support_health


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/taco_pour_algorithmic_reproduction_training_v5.yaml"
RUNNER = ROOT / "scripts/run_taco_pour_algorithmic_candidate_C_v5.py"
AUDIT = ROOT / "src/video_to_spider/rl/algorithmic_training_v5.py"
RUN_ROOT = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v5"
PROTOCOL = ROOT / "configs/replay_rl_protocol.yaml"
SUMMARIZER = ROOT / "scripts/summarize_taco_pour_algorithmic_candidate_C_v5.py"
COMPARISON = RUN_ROOT / "comparison.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v5_changes_only_actor_mean_regularization_and_is_fresh():
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["schema"] == (
        "taco_pour_algorithmic_reproduction_training_benchmark_v5"
    )
    assert contract["status"] == "authorized_candidate_C_fresh_400k"
    assert contract["paper_faithful"] is False
    candidate = contract["candidate_C"]
    assert candidate["seeds"] == [0, 1, 2]
    assert candidate["initialization"] == "replay_preserving"
    assert candidate["bounds_loss_coef"] == 0.005
    assert candidate["bound_loss_type"] == "regularisation"
    assert candidate["actor_mini_epochs"] == 1
    assert candidate["actor_learning_rate"] == 1.0e-4
    assert candidate["linear_LR_allowed"] is False
    assert candidate["warm_start_allowed"] is False
    assert candidate["chunk_commit_allowed"] is False
    assert [row["epoch"] for row in contract["fixed_milestones"]] == [
        0, 62, 125, 188, 250
    ]


def test_v5_binds_public_H2S2R_provenance_and_new_implementation_only():
    contract = yaml.safe_load(CONTRACT.read_text())
    upstream = Path(contract["inputs"]["H2S2R_LSTM_config"]["path"])
    assert contract["inputs"]["H2S2R_LSTM_config"]["sha256"] == _sha256(upstream)
    public = yaml.safe_load(upstream.read_text())
    assert public["ppo"]["bounds_loss_coef"] == 0.005
    assert public["ppo"]["bound_loss_type"] == "regularisation"
    assert contract["implementation_contract"]["training_runner_sha256"] == _sha256(
        RUNNER
    )
    assert contract["implementation_contract"][
        "v5_algorithmic_training_sha256"
    ] == _sha256(AUDIT)
    base = Path(contract["inputs"]["base_v4_contract"]["path"])
    assert contract["inputs"]["base_v4_contract"]["sha256"] == _sha256(base)


def test_support_health_has_declared_per_coordinate_semantics():
    mu = torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float32)
    sigma = torch.tensor([[0.5, 1.0, 0.25]], dtype=torch.float32)
    low = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float32)
    high = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    report = truncated_support_health(mu=mu, sigma=sigma, low=low, high=high)
    assert report["raw_mu_outside_support_fraction"] == pytest.approx(2 / 3)
    assert report["maximum_support_violation_in_sigma"] == pytest.approx(4.0)
    assert report["p95_support_violation_in_sigma"] > 3.0
    assert 0.0 < report["minimum_truncated_normalization_mass"] < 1.0
    assert report["minimum_truncated_normalization_mass"] <= report[
        "p01_truncated_normalization_mass"
    ]
    assert report["p01_truncated_normalization_mass"] <= report[
        "median_truncated_normalization_mass"
    ]


def test_v5_runner_freezes_loss_gate_and_forbids_follow_on_changes():
    source = RUNNER.read_text()
    assert "bounds_loss_coef=0.005" in source
    assert 'bound_loss_type="regularisation"' in source
    assert "lr_schedule=None" in source
    assert "replay_preserving_initialization" in source
    assert source.index("save_milestone(0)") < source.index(
        "while int(agent.epoch_num) < 250"
    )
    assert "command_bitwise_equals_Replay" in source
    assert "contact_flags_bitwise_equal_Replay" in source
    assert '"chunk_commit_written": False' in source


def test_v5_summarizer_preserves_fixed_milestones_and_classification():
    source = SUMMARIZER.read_text()
    assert 'MILESTONES = (0, 62, 125, 188, 250)' in source
    assert '"C1_mean_regularization_fixes_structural_instability"' in source
    assert '"C2_numerical_stability_fixed_but_performance_drifts"' in source
    assert '"C3_mass_floor_still_occurs"' in source
    assert '"C4_strict_task_success"' in source
    assert '"raw_mu_cross_candidate_values_invented": False' in source
    assert '"chunk_commit_written": False' in source


def test_materialized_v5_runs_are_fixed_budget_or_fail_closed():
    for seed in (0, 1, 2):
        root = RUN_ROOT / "training" / "candidate_C" / f"seed_{seed}"
        report_path = root / "report.json"
        failure_path = root / "failure_report.json"
        if not report_path.is_file() and not failure_path.is_file():
            continue
        assert report_path.is_file() ^ failure_path.is_file()
        report = json.loads((report_path if report_path.is_file() else failure_path).read_text())
        assert report["candidate"] == "C"
        assert report["seed"] == seed
        assert report["fresh_actor_critic_optimizers_RMS"] is True
        assert report["warm_start_used"] is False
        assert report["pretraining_validation"]["all_step0_checks_passed"] is True
        assert report["training"]["bounds_loss_coef"] == 0.005
        assert report["training"]["lr_schedule"] is None
        assert report["linear_LR_executed"] is False
        assert report["intermediate_checkpoint_selected"] is False
        assert report["chunk_commit_written"] is False
        if report_path.is_file():
            assert report["status"] == "completed_fixed_400k_no_chunk_commit"
            assert report["training"]["completed_actor_updates"] == 250
            assert set(report["validations"]) == {"0", "62", "125", "188", "250"}
        else:
            assert report["status"] == "failed_closed_before_fixed_400k_budget"
            assert report["eligible_as_400k_outcome"] is False
            assert report["retry_executed"] is False
            assert report["threshold_relaxed"] is False


def test_materialized_v5_comparison_is_consistent_when_present():
    if not COMPARISON.is_file():
        return
    report = json.loads(COMPARISON.read_text())
    assert report["schema"] == "taco_pour_algorithmic_candidate_C_comparison_v5"
    assert report["paper_faithful"] is False
    assert report["only_algorithm_change_from_B"] == {
        "bounds_loss_coef": 0.005,
        "bound_loss_type": "regularisation",
        "linear_LR_enabled": False,
    }
    assert report["chunk_commit_written"] is False
    assert report["forbidden_actions_executed"] == []
    assert report["v4_comparison"]["raw_mu_cross_candidate_values_invented"] is False
    assert set(report["seeds"]) == {"0", "1", "2"}
    assert report["prefrozen_classification"][
        "C3_mass_floor_still_occurs"
    ]["triggered"] is True
    assert report["aggregate"]["mass_floor_failure_seeds"] == [2]
    assert report["aggregate"][
        "global_minimum_truncated_normalization_mass"
    ] == pytest.approx(8.46822612e-13)


def test_v5_is_active_evidence_but_does_not_authorize_follow_on():
    protocol = yaml.safe_load(PROTOCOL.read_text())
    row = protocol["evaluation"][
        "algorithmic_reproduction_training_benchmark_v5"
    ]
    assert row["active_algorithmic_evidence"] is True
    assert row["prefrozen_classification"] == {
        "C1_mean_regularization_fixes_structural_instability": False,
        "C2_numerical_stability_fixed_but_performance_drifts": {
            "evaluated_at_fixed_400k_budget": False,
            "diagnostic_performance_drift_visible_by_300k": True,
        },
        "C3_mass_floor_still_occurs": True,
        "C4_strict_task_success": False,
    }
    assert row["regularization_coefficient_sweep_allowed"] is False
    assert row["linear_LR_follow_on_automatically_authorized"] is False
    assert row["chunk_commit_written"] is False
