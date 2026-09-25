"""Frozen 3+1 curriculum sampler and no-training gate contracts."""

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/taco_pour_tail_curriculum_3plus1_v1.yaml"
REPORT = ROOT / "runs/taco_pour_tail_curriculum_sampler_gate_v1/report.json"
RUN = ROOT / "runs/taco_pour_tail_curriculum_3plus1_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_3plus1_contract_changes_only_the_start_distribution():
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["schema"] == "taco_pour_tail_curriculum_3plus1_v1"
    assert contract["training_budget"] == {
        "worlds": 4,
        "epochs": 8,
        "horizon_per_world_per_epoch": 40,
        "samples_per_epoch": 160,
        "total_samples": 1280,
        "seed": 0,
    }
    assignment = contract["fixed_world_assignment"]
    assert [assignment[f"world_{index}"]["reset_endpoint"] for index in range(4)] == [20, 20, 20, 46]
    assert assignment["assignment_changes_between_epochs"] is False
    assert assignment["random_tail_endpoint_selection"] is False
    frozen = contract["frozen_controls"]
    for name in (
        "objective_unchanged",
        "contact_and_lift_coefficients_unchanged",
        "observation_unchanged",
        "residual_action_mapping_unchanged",
        "PPO_network_optimizer_learning_rate_unchanged",
    ):
        assert frozen[name] is True
    for name in (
        "epoch_sweep_allowed",
        "seed_sweep_allowed",
        "world_ratio_sweep_allowed",
        "tail_endpoint_sweep_allowed",
        "training_budget_sweep_allowed",
    ):
        assert frozen[name] is False
    assert contract["acceptance"]["start_endpoint"] == 20
    assert contract["acceptance"]["required_consecutive_intervals"] == 40
    assert contract["acceptance"]["curriculum_rollout_can_be_committed_directly"] is False


def test_real_sampler_gate_passes_without_training_or_normalization_mutation():
    report = json.loads(REPORT.read_text())
    assert report["schema"] == "taco_pour_tail_curriculum_sampler_gate_v1"
    assert report["status"] == "passed"
    assert report["PPO_training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["frozen_world_start_endpoints"] == [20, 20, 20, 46]
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert report["normalization_immutability"]["passed"] is True
    assert report["episode_reset"]["passed"] is True
    assert report["actor_update_refresh"]["stale_hidden_rejected_after_actor_change"] is True
    assert report["actor_update_refresh"]["passed"] is True
    assert report["semantic_limit"]["tail_start_is_off_policy_curriculum"] is True
    assert report["semantic_limit"]["CPU_acceptance_start_endpoint"] == 20


def test_single_frozen_experiment_changed_coverage_but_failed_40_of_40():
    comparison = json.loads((RUN / "comparison.json").read_text())
    assert comparison["status"] == "CPU_40_of_40_failed"
    assert comparison["frozen_budget"]["total_samples"] == 1280
    tail = comparison["tail_visitation"]
    assert tail["endpoint_46_to_50_total_visits"] == 252
    assert tail["fraction_of_1280"] == 0.196875
    assert tail["epochs_with_endpoint_50_visit"] == 8
    assert comparison["terminations"]["three_endpoint20_worlds"]["any"] == 28
    assert comparison["terminations"]["one_endpoint46_world"]["any"] == 43
    cpu = comparison["deterministic_CPU_validation"]
    assert cpu["replay"]["validated_steps"] == 29
    assert cpu["rl"]["validated_steps"] == 32
    assert cpu["rl"]["first_failure"]["endpoint"] == 53
    assert comparison["runtime_contract"]["world_start_endpoints"] == [20, 20, 20, 46]
    assert comparison["interpretation_limits"]["tail_start_is_off_policy"] is True
    assert comparison["interpretation_limits"]["result_proves_tail_sparsity_is_root_cause"] is False
