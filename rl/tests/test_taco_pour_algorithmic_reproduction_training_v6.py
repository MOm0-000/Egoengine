"""Candidate-D bounded-mean math, provenance, and materialized-evidence tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import yaml

from video_to_spider.rl.algorithmic_training_v6 import (
    load_support_anchored_profile,
    support_anchored_bounded_mean,
)
from video_to_spider.rl.state_feasible_truncated_gaussian import (
    normalized_action_bounds_numpy,
    sample_truncated_normal,
    truncated_normal_entropy,
    truncated_normal_log_prob,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/taco_pour_support_anchored_bounded_mean_truncated_gaussian_v1.yaml"
CONTRACT = ROOT / "configs/taco_pour_algorithmic_reproduction_training_v6.yaml"
RUNNER = ROOT / "scripts/run_taco_pour_algorithmic_candidate_D_v6.py"
AUDIT = ROOT / "src/video_to_spider/rl/algorithmic_training_v6.py"
PROTOCOL = ROOT / "configs/replay_rl_protocol.yaml"
RUN_ROOT = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v6"
COMPARISON = RUN_ROOT / "comparison.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("low", "high"),
    [(-1.0, 1.0), (-0.2, 1.0), (-1.0, 0.2), (0.0, 1.0), (-1.0, 0.0), (-0.01, 0.7)],
)
def test_support_anchored_transform_is_finite_monotone_and_zero_anchored(low, high):
    values = torch.tensor(
        [-100.0, -10.0, -1.0, -1.0e-6, 0.0, 1.0e-6, 1.0, 10.0, 100.0],
        dtype=torch.float32,
    )
    lows = torch.full_like(values, low)
    highs = torch.full_like(values, high)
    bounded = support_anchored_bounded_mean(values, lows, highs)
    assert torch.isfinite(bounded).all()
    assert torch.all(bounded >= lows)
    assert torch.all(bounded <= highs)
    assert bounded[4].item() == 0.0
    assert torch.equal(bounded[4], torch.zeros_like(bounded[4]))
    assert torch.all(bounded[1:] >= bounded[:-1])

    eta = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    mu = support_anchored_bounded_mean(
        eta, torch.tensor([low]), torch.tensor([high])
    )
    mu.sum().backward()
    expected = high if high > 0.0 else -low
    assert torch.isfinite(eta.grad).all()
    assert eta.grad.item() == pytest.approx(expected)
    assert eta.grad.item() > 0.0


def test_bounded_mean_distribution_sampling_likelihood_entropy_and_gradient():
    torch.manual_seed(17)
    eta = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]], requires_grad=True)
    low = torch.tensor([[0.0, -1.0, -0.2, -1.0, -0.01]])
    high = torch.tensor([[1.0, 0.0, 1.0, 0.2, 0.7]])
    sigma = torch.full_like(eta, 0.25, requires_grad=True)
    mu = support_anchored_bounded_mean(eta, low, high)
    action = sample_truncated_normal(
        mu, sigma, low, high, minimum_mass=1.0e-12
    ).detach()
    assert torch.all(action >= low)
    assert torch.all(action <= high)
    first = truncated_normal_log_prob(
        action, mu, sigma, low, high, minimum_mass=1.0e-12
    ).sum(dim=-1)
    second = truncated_normal_log_prob(
        action, mu, sigma, low, high, minimum_mass=1.0e-12
    ).sum(dim=-1)
    assert torch.equal(torch.exp(first - second), torch.ones_like(first))
    entropy = truncated_normal_entropy(
        mu, sigma, low, high, minimum_mass=1.0e-12
    ).sum()
    loss = -(first.mean() + 0.01 * entropy)
    loss.backward()
    assert torch.isfinite(first).all()
    assert torch.isfinite(entropy)
    assert torch.isfinite(eta.grad).all()
    assert torch.isfinite(sigma.grad).all()
    assert torch.count_nonzero(eta.grad) > 0


def test_real_36d_reference_bounds_keep_every_bounded_mean_inside_support():
    reference = np.load(
        ROOT / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz"
    )["ctrl"].astype(np.float32)
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "runs/taco_pour_floor_contact_v1/candidate.xml")
    )
    spec, _ = load_support_anchored_profile(PROFILE)
    low, high, _ = normalized_action_bounds_numpy(
        reference,
        ctrllimited=np.asarray(model.actuator_ctrllimited, dtype=bool),
        ctrlrange=np.asarray(model.actuator_ctrlrange, dtype=np.float64),
        residual_scale=spec.residual_scale,
        reference_snap_tolerance=spec.reference_snap_tolerance,
    )
    assert low.shape == high.shape == (198, 36)
    for value in (-100.0, -10.0, -1.0, -1.0e-6, 0.0, 1.0e-6, 1.0, 10.0, 100.0):
        eta = torch.full(low.shape, value, dtype=torch.float32)
        bounded = support_anchored_bounded_mean(
            eta, torch.from_numpy(low).float(), torch.from_numpy(high).float()
        )
        assert torch.isfinite(bounded).all()
        assert torch.all(bounded >= torch.from_numpy(low).float())
        assert torch.all(bounded <= torch.from_numpy(high).float())
        if value == 0.0:
            assert torch.equal(bounded, torch.zeros_like(bounded))


def test_v6_profile_and_contract_are_single_variable_candidate_D():
    spec, report = load_support_anchored_profile(PROFILE)
    assert spec.minimum_normalization_mass == 1.0e-12
    assert spec.optimizer_training_authorized is True
    assert report["distribution"]["ppo_mus_semantics"] == "bounded_distribution_mean"
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["schema"] == "taco_pour_algorithmic_reproduction_training_benchmark_v6"
    assert contract["candidate_D"]["bounds_loss_coef"] == 0.0
    assert contract["candidate_D"]["actor_learning_rate"] == 1.0e-4
    assert contract["candidate_D"]["linear_LR_allowed"] is False
    assert contract["candidate_D"]["warm_start_allowed"] is False
    assert contract["candidate_D"]["chunk_commit_allowed"] is False
    assert contract["implementation_contract"]["training_runner_sha256"] == _sha256(RUNNER)
    assert contract["implementation_contract"]["v6_algorithmic_training_sha256"] == _sha256(AUDIT)
    assert contract["inputs"]["bounded_mean_distribution_profile"]["sha256"] == _sha256(PROFILE)


def test_materialized_v6_runs_are_fixed_budget_or_fail_closed():
    for seed in (0, 1, 2):
        root = RUN_ROOT / "training" / "candidate_D" / f"seed_{seed}"
        report_path = root / "report.json"
        failure_path = root / "failure_report.json"
        if not report_path.is_file() and not failure_path.is_file():
            continue
        assert report_path.is_file() ^ failure_path.is_file()
        report = json.loads((report_path if report_path.is_file() else failure_path).read_text())
        assert report["candidate"] == "D"
        assert report["seed"] == seed
        assert report["fresh_actor_critic_optimizers_RMS"] is True
        assert report["warm_start_used"] is False
        assert report["pretraining_validation"]["all_step0_checks_passed"] is True
        assert report["training"]["bounds_loss_coef"] == 0.0
        assert report["training"]["lr_schedule"] is None
        assert report["intermediate_checkpoint_selected"] is False
        assert report["chunk_commit_written"] is False
        for row in report["learning_curve"].values():
            support = row.get("cumulative", {})
            if support.get("actor_updates", 0):
                assert support["maximum_bounded_mu_outside_support_count"] == 0


def test_materialized_v6_comparison_and_protocol_when_present():
    if not COMPARISON.is_file():
        return
    comparison = json.loads(COMPARISON.read_text())
    assert comparison["schema"] == "taco_pour_algorithmic_candidate_D_comparison_v6"
    assert comparison["only_algorithm_change_from_B"] == (
        "support_anchored_bounded_distribution_mean"
    )
    assert comparison["forbidden_actions_executed"] == []
    assert comparison["chunk_commit_written"] is False
    assert comparison["aggregate"]["completed_seeds"] == [0, 1, 2]
    assert comparison["aggregate"]["failed_seeds"] == []
    assert comparison["aggregate"]["mass_floor_failure_seeds"] == []
    assert comparison["aggregate"]["median_validated_intervals"] == {
        "0": 30.0,
        "100k": 35.0,
        "200k": 31.0,
        "300k": 35.0,
        "400k": 33.0,
    }
    assert comparison["aggregate"]["seeds_beating_Replay_at_400k"] == [0, 1, 2]
    assert comparison["aggregate"]["maximum_bounded_mu_outside_support_count"] == 0
    assert comparison["aggregate"]["global_minimum_truncated_normalization_mass"] == pytest.approx(
        0.499965943376135
    )
    frozen = comparison["prefrozen_classification"]
    assert frozen["D1_raw_mean_support_collapse_mechanism_solved"]["triggered"] is True
    assert frozen["D2_structurally_stable_and_task_learning_useful"]["triggered"] is True
    assert frozen["D3_structurally_stable_but_performance_drifts"]["triggered"] is False
    assert frozen["D4_bounded_mean_but_mass_floor_occurs"]["triggered"] is False
    assert frozen["D5_strict_task_success"] == {
        "triggered": True,
        "events": [{"seed": 2, "epoch": 125}],
        "seed_robust": False,
        "automatic_chunk_commit_allowed": False,
    }
    protocol = yaml.safe_load(PROTOCOL.read_text())
    row = protocol["evaluation"]["algorithmic_reproduction_training_benchmark_v6"]
    assert row["active_algorithmic_evidence"] is True
    assert row["active_local_baseline"] is True
    assert row["prefrozen_classification"][
        "D1_raw_mean_support_collapse_mechanism_solved"
    ] is True
    assert row["prefrozen_classification"][
        "D2_structurally_stable_and_task_learning_useful"
    ] is True
    assert row["prefrozen_classification"]["D5_strict_task_success"][
        "seed_robust"
    ] is False
    assert row["intermediate_checkpoint_selected"] is False
    assert row["chunk_commit_written"] is False
