import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/taco_pour_algorithmic_B_500k_extension_v4.yaml"
RUNNER = ROOT / "scripts/run_taco_pour_algorithmic_B_500k_extension_v4.py"
SUMMARIZER = (
    ROOT
    / "scripts/summarize_taco_pour_algorithmic_B_500k_extension_v4.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_B_500k_extension_is_one_fixed_budget_continuation():
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["schema"] == (
        "taco_pour_algorithmic_candidate_B_500k_extension_v4"
    )
    assert contract["status"] == "authorized_candidate_B_500k_continuation"
    extension = contract["candidate_B_500k_extension"]
    assert extension == {
        "candidate": "B",
        "seeds": [0, 1, 2],
        "source_epoch": 62,
        "source_physics_steps": 99_200,
        "source_control_intervals": 9_920,
        "target_epoch": 313,
        "target_physics_steps": 500_800,
        "target_control_intervals": 50_080,
        "additional_actor_updates_per_seed": 251,
        "additional_physics_steps_per_seed": 401_600,
        "candidate_A_continuation_allowed": False,
        "fresh_restart_allowed": False,
        "cross_run_warm_start_allowed": False,
        "seed_specific_settings_allowed": False,
        "intermediate_checkpoint_selection_allowed": False,
        "fallback_sweep_allowed": False,
        "automatic_1m_allowed": False,
        "chunk_commit_allowed": False,
    }
    assert [row["epoch"] for row in contract["fixed_milestones"]] == [
        62,
        125,
        188,
        250,
        313,
    ]
    frozen = contract["frozen_training_settings"]
    assert frozen["actor_mini_epochs"] == 1
    assert frozen["critic_mini_epochs"] == 4
    assert frozen["actor_learning_rate"] == 1.0e-4
    assert frozen["residual_scale"] == 0.05
    assert frozen["observation_dimension"] == 236


def test_B_500k_extension_binds_all_three_verified_sources_and_runner():
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["implementation_contract"][
        "continuation_runner_sha256"
    ] == _sha256(RUNNER)
    expected = {0: (36, 57), 1: (29, 50), 2: (37, 58)}
    for seed, (intervals, failure) in expected.items():
        source = contract["source_100k"][f"seed{seed}"]
        assert source["expected_successful_intervals"] == intervals
        assert source["expected_first_failure_endpoint"] == failure
        for name in ("report", "checkpoint", "validation"):
            path = Path(source[name]["path"])
            assert path.is_file()
            assert source[name]["sha256"] == _sha256(path)


def test_B_500k_runner_restores_and_validates_before_new_update():
    source = RUNNER.read_text()
    restore = source.index('agent.model.load_state_dict(source_payload["actor"]')
    recapture = source.index("recaptured = build_checkpoint_payload(")
    validation = source.index("prevalidation, pre_arrays = validate_payload(")
    update = source.index("while int(agent.epoch_num) < 313:")
    assert restore < recapture < validation < update
    assert '"source_validation_all_arrays_exact"' in source
    assert '"candidate_A_continued": False' in source
    assert '"intermediate_checkpoint_selection_executed": False' in source
    assert '"automatic_1m_executed": False' in source
    assert '"chunk_commit_written": False' in source


def test_materialized_B_500k_runs_are_fail_closed():
    contract = yaml.safe_load(CONTRACT.read_text())
    run_root = Path(contract["output_directory"])
    for seed in (0, 1, 2):
        report_path = run_root / f"seed_{seed}" / "report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text())
        assert report["status"] == "completed_fixed_500k_no_chunk_commit"
        assert report["candidate"] == "B"
        assert report["seed"] == seed
        assert report["resume_preflight"]["status"] == (
            "passed_before_new_optimizer_update"
        )
        assert report["resume_preflight"][
            "source_validation_all_arrays_exact"
        ] is True
        assert all(report["resume_preflight"]["resume_field_equality"].values())
        assert report["training"]["additional_physics_steps"] == 401_600
        assert report["training"]["total_physics_steps"] == 500_800
        assert report["training"]["additional_actor_updates"] == 251
        assert report["training"]["total_actor_updates"] == 313
        assert set(report["validations"]) == {"62", "125", "188", "250", "313"}
        assert report["candidate_A_continued"] is False
        assert report["intermediate_checkpoint_selection_executed"] is False
        assert report["hyperparameter_change_executed"] is False
        assert report["automatic_1m_executed"] is False
        assert report["chunk_commit_written"] is False


def test_fixed_budget_summarizer_preserves_prefrozen_decision_rules():
    source = SUMMARIZER.read_text()
    assert 'all_above_replay and median_500k >= 35' in source
    assert 'len(beating_replay) >= 2' in source
    assert 'and len(beating_replay) < 3' in source
    assert 'fixed_budget_evaluable and median_500k <= 36' in source
    assert 'fixed_budget_evaluable and strict_seeds' in source
    assert '"failed_closed_before_500k_seeds": failed_seeds' in source
    assert '"automatic_1m_authorized": False' in source
    assert '"intermediate_checkpoint_selected": False' in source
    assert '"chunk_commit_authorized": False' in source


def test_materialized_B_500k_comparison_is_fail_closed():
    comparison_path = (
        ROOT
        / "runs/taco_pour_algorithmic_reproduction_training_v4"
        / "B_budget_extension_500k/comparison.json"
    )
    if not comparison_path.is_file():
        return
    comparison = json.loads(comparison_path.read_text())
    assert comparison["status"] in {
        "completed_fixed_budget_comparison_no_automatic_follow_on",
        "fixed_budget_extension_failed_closed_no_automatic_follow_on",
    }
    assert set(comparison["seeds"]) == {"0", "1", "2"}
    assert comparison["Candidate_A_continued"] is False
    assert comparison["automatic_1m_authorized"] is False
    assert comparison["automatic_1m_executed"] is False
    assert comparison["intermediate_checkpoint_selected"] is False
    assert comparison["chunk_commit_authorized"] is False
    assert comparison["chunk_commit_written"] is False
    if comparison["status"].startswith("fixed_budget_extension_failed"):
        assert comparison["aggregate"][
            "fixed_budget_three_seed_classification_evaluable"
        ] is False
        assert comparison["aggregate"][
            "failed_closed_before_500k_seeds"
        ] == [0, 1, 2]
        assert all(
            not row["evaluable"]
            for row in comparison["prefrozen_classifications"].values()
        )


def test_materialized_seed_failures_are_not_500k_scores_or_fallbacks():
    expected = {
        0: (161, 162, 160),
        1: (194, 195, 160),
        2: (191, 192, 148),
    }
    for seed, (last_update, failed_epoch, samples) in expected.items():
        failure_path = (
            ROOT
            / "runs/taco_pour_algorithmic_reproduction_training_v4"
            / f"B_budget_extension_500k/seed_{seed}/failure_report.json"
        )
        if not failure_path.is_file():
            continue
        failure = json.loads(failure_path.read_text())
        assert failure["status"] == "failed_closed_before_fixed_500k_budget"
        assert failure["last_completed_actor_update_epoch"] == last_update
        assert failure["failed_rollout_epoch"] == failed_epoch
        assert failure["failed_rollout_completed_samples"] == samples
        if seed == 1:
            assert failure["exception"]["rollout_ratio_error"] > failure[
                "exception"
            ]["frozen_ratio_tolerance"]
            assert failure["exception"]["canonical_ratio_error"] == 0
        else:
            assert failure["exception"]["normalization_mass"] < failure[
                "exception"
            ]["frozen_failclosed_threshold"]
        assert failure["retry_executed"] is False
        assert failure["threshold_relaxed"] is False
        assert failure["intermediate_checkpoint_selected"] is False
        assert failure["eligible_as_500k_outcome"] is False
        assert failure["chunk_commit_written"] is False
