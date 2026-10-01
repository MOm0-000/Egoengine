import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from video_to_spider.rl.algorithmic_benchmark import (
    CHECKPOINT_SCHEMA,
    build_checkpoint_payload,
    replay_preserving_initialization,
    validate_budget_plan,
    validate_checkpoint_payload,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/taco_pour_algorithmic_reproduction_training_v1.yaml"
RUN = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _ActorNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.other = torch.nn.Linear(3, 3)
        self.mu = torch.nn.Linear(3, 2)
        self.sigma = torch.nn.Parameter(torch.zeros(2))
        self.fixed_sigma = True


class _Critic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Linear(3, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-3)


def test_contract_freezes_staged_cpu_only_benchmark_and_budget():
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["schema"] == (
        "taco_pour_algorithmic_reproduction_training_benchmark_v1"
    )
    assert contract["paper_faithful"] is False
    assert contract["runtime"]["CPU_MuJoCo_Warp_is_only_training_backend"] is True
    assert contract["runtime"]["CPU_MuJoCo_Warp_is_only_validation_backend"] is True
    assert contract["runtime"]["chunk_commit_allowed"] is False
    assert contract["runtime"]["old_checkpoint_warm_start_allowed"] is False
    assert contract["runtime"]["automatic_candidate_C_allowed"] is False
    assert contract["candidates"]["B"]["step0_gate"]["failure_action"] == (
        "stop_without_training_A_or_B"
    )
    assert contract["frozen_task"]["actor_mini_epochs"] == 1
    assert contract["frozen_task"]["actor_learning_rate"] == 1.0e-4
    assert contract["frozen_task"]["actor_learning_rate_5e_minus_5_allowed"] is False

    milestones = validate_budget_plan(contract["training_budget"]["seed0_milestones"])
    assert [row["actual_physics_steps"] for row in milestones] == [
        99_200, 500_800, 1_000_000,
    ]
    assert [row["actual_control_intervals"] for row in milestones] == [
        9_920, 50_080, 100_000,
    ]


def test_candidate_B_initialization_changes_only_declared_mean_and_sigma():
    torch.manual_seed(4)
    network = _ActorNetwork()
    before = {key: value.detach().clone() for key, value in network.state_dict().items()}
    agent = SimpleNamespace(model=SimpleNamespace(a2c_network=network))
    audit = replay_preserving_initialization(agent, sigma_multiplier=0.25)

    assert set(audit["changed_network_state_keys"]) == {"mu.weight", "mu.bias", "sigma"}
    assert torch.count_nonzero(network.mu.weight) == 0
    assert torch.count_nonzero(network.mu.bias) == 0
    assert torch.equal(network.other.weight, before["other.weight"])
    assert torch.equal(network.other.bias, before["other.bias"])
    assert torch.equal(torch.exp(network.sigma), torch.full((2,), 0.25))
    assert network.sigma.requires_grad


def test_checkpoint_envelope_is_complete_resumable_and_no_commit():
    actor = torch.nn.Linear(3, 2)
    critic = _Critic()
    agent = SimpleNamespace(
        model=actor,
        asymmetric_critic_net=critic,
        optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-4),
        _observation_normalization_version=7,
        epoch_num=62,
        frame=9_920,
        rnn_states=[torch.ones(1, 1, 4)],
        obs={"obs": torch.zeros(1, 3)},
        dones=torch.zeros(1, dtype=torch.bool),
        env=SimpleNamespace(get_env_state=lambda: {"qpos": np.ones((1, 2))}),
    )
    payload = build_checkpoint_payload(
        agent,
        candidate="B",
        seed=0,
        simulation_physics_steps=99_200,
        simulation_control_intervals=9_920,
        config_hashes={"contract": "deadbeef"},
    )
    validate_checkpoint_payload(payload)
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["chunk_commit_allowed"] is False
    assert payload["observation_normalization_version"] == 7
    assert payload["simulation_physics_steps"] == 99_200
    assert payload["simulation_control_intervals"] == 9_920
    assert {"python", "numpy", "torch_cpu"} <= payload["rng_states"].keys()

    broken = dict(payload, simulation_physics_steps=99_201)
    with pytest.raises(ValueError, match="counters disagree"):
        validate_checkpoint_payload(broken)


def test_formal_B0_gate_fails_before_any_training_or_commit():
    report = json.loads((RUN / "candidate_B/seed_0/report.json").read_text())
    comparison = json.loads((RUN / "comparison.json").read_text())
    candidate_a = json.loads((RUN / "candidate_A/seed_0/report.json").read_text())

    assert report["status"] == "failed_training_forbidden"
    assert report["backend"] == "CPU_MuJoCo_Warp"
    assert report["optimizer_updates"] == 0
    assert report["simulation_physics_steps_for_training"] == 0
    assert report["training_A_or_B_started"] is False
    assert report["chunk_commit_written"] is False
    assert report["replay"] == {
        "successful_intervals": 30,
        "first_failure_endpoint": 51,
        "forty_of_forty": False,
        "endpoint40_score": 0.6679563522338867,
        "endpoint50_score": 0.9944888353347778,
        "endpoint60_score": 1.2191834449768066,
    }
    assert report["candidate_B_step0"]["successful_intervals"] == 21
    assert report["candidate_B_step0"]["first_failure_endpoint"] == 42
    assert report["maximum_abs_actor_mu"] == 0.0
    assert report["maximum_abs_deterministic_action"] == 0.0
    assert report["maximum_abs_command_difference"] == 1.1920928955078125e-7
    assert report["command_rows_differing"] == 32
    assert report["first_differing_command_endpoint"] == 21
    assert report["maximum_abs_qpos_difference"] == 0.1835172176361084
    assert report["maximum_abs_qvel_difference"] == 5.342435359954834
    assert report["checks"] == {
        "mean_residual_within_tolerance": True,
        "deterministic_action_within_tolerance": True,
        "deterministic_command_bitwise_equals_formal_Replay": False,
        "formal_Replay_is_30_of_40_fail51": True,
        "candidate_B_step0_is_30_of_40_fail51": False,
    }

    assert comparison["status"] == "B0_failed_no_training"
    assert comparison["training_started"] is False
    assert comparison["A_at_100k"] is None
    assert comparison["B_at_100k"] is None
    assert comparison["structured_architecture_experiment_authorized"] is False
    assert candidate_a["status"] == "not_started_due_candidate_B_step0_gate_failure"
    assert candidate_a["optimizer_updates"] == 0
    assert not (RUN / "checkpoints").exists()
    assert not list(RUN.rglob("*.pth"))
    assert not list(RUN.rglob("*.pt.gz"))


def test_B0_arrays_and_artifact_hashes_are_reconstructible():
    report_path = RUN / "candidate_B/seed_0/report.json"
    report = json.loads(report_path.read_text())
    comparison = json.loads((RUN / "comparison.json").read_text())
    trajectory_path = Path(report["trajectory"]["path"])
    assert trajectory_path == RUN / "candidate_B/seed_0/step0_trajectory.npz"
    assert report["trajectory"]["sha256"] == _sha256(trajectory_path)
    assert comparison["candidate_B_step0_report"]["sha256"] == _sha256(report_path)

    arrays = np.load(trajectory_path)
    assert arrays["endpoint"].tolist() == list(range(21, 61))
    assert arrays["actor_mu"].shape == (40, 36)
    assert arrays["deterministic_action"].shape == (40, 36)
    assert not np.any(arrays["actor_mu"])
    assert not np.any(arrays["deterministic_action"])
    assert np.count_nonzero(
        np.any(arrays["candidate_B_ctrl"] != arrays["replay_ctrl"], axis=1)
    ) == 32


def test_protocol_records_local_benchmark_exception_and_B0_stop():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    gate = protocol["evaluation"]["algorithmic_reproduction_training_benchmark_v1"]
    assert gate["status"] == "B0_failed_training_forbidden"
    assert gate["classification"] == "local_engineering_not_exact_reproduction"
    assert gate["Gate_Q_exact_reproduction_blocker_superseded"] is False
    assert gate["local_benchmark_exception_authorized"] is True
    assert gate["Replay"] == {"successful_intervals": 30, "first_failure_endpoint": 51}
    assert gate["candidate_B_step0"] == {
        "successful_intervals": 21,
        "first_failure_endpoint": 42,
        "maximum_abs_actor_mu": 0.0,
        "maximum_abs_deterministic_action": 0.0,
        "maximum_abs_command_difference": 1.1920928955078125e-7,
    }
    assert gate["training_started"] is False
    assert gate["optimizer_updates"] == 0
    assert gate["simulation_physics_steps_for_training"] == 0
    assert gate["chunk_commit_written"] is False
    assert gate["next_action"] == "stop_without_training_A_or_B"
    assert protocol["training_ready"] is False
    assert protocol["audit_results"]["algorithmic_reproduction_training_B0"] == str(
        RUN / "candidate_B/seed_0/report.json"
    )
