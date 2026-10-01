import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/taco_pour_algorithmic_reproduction_training_v4.yaml"
EXTENSION = ROOT / "configs/taco_pour_algorithmic_seed_robustness_v4.yaml"
RUNNER = ROOT / "scripts/run_taco_pour_algorithmic_seed_robustness_v4.py"
COMPARISON = (
    ROOT
    / "runs/taco_pour_algorithmic_reproduction_training_v4"
    / "seed_robustness_100k"
    / "comparison.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extension_changes_only_authorization_and_seed_scope():
    base = yaml.safe_load(BASE.read_text())
    extension = yaml.safe_load(EXTENSION.read_text())
    assert extension["schema"] == "taco_pour_algorithmic_seed_robustness_extension_v4"
    assert extension["status"] == "authorized_seed_robustness_100k"
    assert Path(extension["base_v4_contract"]["path"]).name == BASE.name
    assert extension["base_v4_contract"]["sha256"] == _sha256(BASE)
    assert extension["seed_robustness_extension"] == {
        "stage": "seed_robustness_100k",
        "candidates": ["A", "B"],
        "new_seeds": [1, 2],
        "physics_steps_per_candidate_seed": 99_200,
        "actor_updates_per_candidate_seed": 62,
        "seed0_rerun_allowed": False,
        "warm_start_allowed": False,
        "checkpoint_selection_allowed": False,
        "fallback_sweep_allowed": False,
        "chunk_commit_allowed": False,
    }
    for key in (
        "frozen_task",
        "candidates",
        "validation",
        "quality_gates",
        "inputs",
    ):
        assert extension[key] == base[key]
    assert extension["training_budget"]["seed0_milestones"] == (
        base["training_budget"]["seed0_milestones"]
    )
    assert extension["training_budget"]["milestone_500k_authorized"] is False
    assert extension["training_budget"]["milestone_1m_authorized"] is False
    assert extension["runtime"]["chunk_commit_allowed"] is False
    assert extension["runtime"]["old_checkpoint_warm_start_allowed"] is False


def test_extension_runner_is_hash_bound_and_preflights_before_training():
    extension = yaml.safe_load(EXTENSION.read_text())
    implementation = extension["implementation_contract"]
    assert implementation["training_runner_sha256"] == _sha256(RUNNER)
    assert implementation["base_v4_training_runner_sha256"] == _sha256(
        ROOT / "scripts/run_taco_pour_algorithmic_training_v3.py"
    )
    source = RUNNER.read_text()
    assert source.index("pretraining_validation = {") < source.index(
        "while int(agent.epoch_num) < final_epoch"
    )
    assert "command_bitwise_equals_Replay" in source
    assert "qpos_bitwise_equals_Replay" in source
    assert "qvel_bitwise_equals_Replay" in source
    assert "score_bitwise_equals_Replay" in source
    assert "args.seed not in (1, 2)" in source
    assert 'args.target_milestone != "100k"' in source


def test_materialized_extension_runs_are_fail_closed():
    extension = yaml.safe_load(EXTENSION.read_text())
    run_root = Path(extension["output_directory"])
    for candidate in ("A", "B"):
        for seed in (1, 2):
            report_path = (
                run_root
                / "training"
                / f"candidate_{candidate}"
                / f"seed_{seed}"
                / "100k"
                / "report.json"
            )
            if not report_path.is_file():
                continue
            import json

            report = json.loads(report_path.read_text())
            assert report["status"] == (
                "completed_seed_robustness_100k_no_chunk_commit"
            )
            assert report["seed"] == seed
            assert report["candidate"] == candidate
            assert report["training"]["simulation_physics_steps"] == 99_200
            assert report["training"]["actor_updates"] == 62
            assert report["fresh_actor_critic_optimizers_RMS"] is True
            assert report["warm_start_used"] is False
            assert report["pretraining_validation"]["all_checks_passed"] is True
            assert report["chunk_commit_written"] is False
            if candidate == "B":
                summary = report["pretraining_validation"]["summary"]
                assert summary["successful_intervals"] == 30
                assert summary["first_failure_endpoint"] == 51
                assert all(
                    report["pretraining_validation"]["checks"].values()
                )


def test_completed_seed_robustness_uses_prefrozen_case_2_decision():
    if not COMPARISON.is_file():
        return
    import json

    comparison = json.loads(COMPARISON.read_text())
    assert comparison["status"] == (
        "seed_robustness_100k_completed_no_chunk_commit"
    )
    assert comparison["seed0_rerun"] is False
    assert comparison["new_runs"] == ["B1", "B2", "A1", "A2"]
    assert comparison["prefrozen_decision"] == {
        "case": "case_2_B_partial_seed_improvement",
        "next_authorization": "none_pending_seed_sensitivity_review",
        "automatic_500k_execution_allowed": False,
    }
    expected = {
        "B": [(30, 36, 6), (30, 29, -1), (30, 37, 7)],
        "A": [(36, 29, -7), (30, 23, -7), (29, 36, 7)],
    }
    for candidate, triples in expected.items():
        rows = comparison["candidates"][candidate]
        assert [
            (
                row["step0"]["successful_intervals"],
                row["at_100k"]["successful_intervals"],
                row["delta_validated_intervals"],
            )
            for row in rows
        ] == triples
        assert all(row["physics_steps"] == 99_200 for row in rows)
        assert all(row["actor_updates"] == 62 for row in rows)
        assert all(row["checkpoint_roundtrip_exact"] for row in rows)
        assert all(row["all_canonical_identity_gates_passed"] for row in rows)
    assert comparison["aggregates"]["B"] == {
        "seeds_beating_Replay_at_100k": [0, 2],
        "seeds_improving_from_own_step0": [0, 2],
        "median_validated_intervals_at_0": 30.0,
        "median_validated_intervals_at_100k": 36.0,
        "median_delta": 6.0,
        "at_least_35_at_100k_count": 2,
        "all_seeds_improve_from_own_step0": False,
    }
    assert comparison["aggregates"]["A"] == {
        "seeds_beating_Replay_at_100k": [2],
        "seeds_improving_from_own_step0": [2],
        "median_validated_intervals_at_0": 30.0,
        "median_validated_intervals_at_100k": 29.0,
        "median_delta": -7.0,
        "at_least_35_at_100k_count": 1,
        "all_seeds_improve_from_own_step0": False,
    }
    assert comparison["significance_test_executed"] is False
    assert comparison["hyperparameter_change_executed"] is False
    assert comparison["intermediate_checkpoint_selection_executed"] is False
    assert comparison["chunk_commit_written"] is False
