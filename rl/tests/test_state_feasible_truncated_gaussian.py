"""Pure-math and fail-closed tests for the local truncated-Gaussian candidate."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "external/human2sim2robot"),
]

from video_to_spider.rl.state_feasible_truncated_gaussian import (
    deterministic_truncated_action,
    load_truncated_gaussian_profile,
    normalized_action_bounds_numpy,
    sample_truncated_normal,
    truncated_normal_entropy,
    truncated_normal_log_prob,
)


PROFILE = ROOT / "configs/taco_pour_state_feasible_truncated_gaussian_v1.yaml"


def test_profile_is_gate_only_and_keeps_scale():
    spec, report = load_truncated_gaussian_profile(PROFILE)
    assert spec.residual_scale == 0.05
    assert spec.reference_snap_tolerance == 2e-7
    assert spec.optimizer_training_authorized is False
    assert report["paper_faithful"] is False
    assert report["distribution"]["official_minus1_plus1_clamp_enabled"] is False


def test_bounds_snap_only_numeric_reference_error_and_fail_closed():
    reference = np.array([[0.2, -0.1700000018]], np.float64)
    low, high, audit = normalized_action_bounds_numpy(
        reference,
        ctrllimited=np.array([True, True]),
        ctrlrange=np.array([[0.0, 1.0], [-0.17, 1.83]]),
        residual_scale=0.05,
        reference_snap_tolerance=2e-7,
    )
    np.testing.assert_allclose(low, [[-1.0, 0.0]], atol=0.0)
    np.testing.assert_allclose(high, [[1.0, 1.0]], atol=0.0)
    assert audit["snapped_component_count"] == 1
    assert audit["maximum_reference_violation"] < 2e-7
    with pytest.raises(ValueError, match="above the"):
        normalized_action_bounds_numpy(
            np.array([[0.2, -0.170001]]),
            ctrllimited=np.array([True, True]),
            ctrlrange=np.array([[0.0, 1.0], [-0.17, 1.83]]),
            residual_scale=0.05,
            reference_snap_tolerance=2e-7,
        )


def test_sampling_logprob_entropy_gradient_and_ratio_are_consistent():
    torch.manual_seed(7)
    mu = torch.tensor([[-3.0, -0.2, 0.5, 2.0]], requires_grad=True)
    sigma = torch.tensor([[1.0, 0.8, 1.1, 0.6]], requires_grad=True)
    low = torch.tensor([[0.0, -1.0, -0.25, -1.0]])
    high = torch.tensor([[1.0, 0.0, 0.75, 0.2]])
    action = sample_truncated_normal(
        mu, sigma, low, high, minimum_mass=1e-12
    ).detach()
    assert torch.all(action >= low)
    assert torch.all(action <= high)
    assert torch.equal(torch.clamp(action, -1.0, 1.0), action)
    first = truncated_normal_log_prob(
        action, mu, sigma, low, high, minimum_mass=1e-12
    ).sum(dim=-1)
    second = truncated_normal_log_prob(
        action, mu, sigma, low, high, minimum_mass=1e-12
    ).sum(dim=-1)
    assert torch.equal(first, second)
    ratio = torch.exp(first - second)
    assert torch.equal(ratio, torch.ones_like(ratio))
    entropy = truncated_normal_entropy(
        mu, sigma, low, high, minimum_mass=1e-12
    ).sum()
    loss = -(first.mean() + 0.01 * entropy)
    loss.backward()
    assert torch.isfinite(first).all()
    assert torch.isfinite(entropy)
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(sigma.grad).all()


def test_deterministic_mode_preserves_zero_at_one_sided_boundary():
    mu = torch.tensor([[-2.0, 2.0, 0.25]])
    low = torch.tensor([[0.0, -1.0, -0.5]])
    high = torch.tensor([[1.0, 0.0, 0.5]])
    action = deterministic_truncated_action(mu, low, high)
    torch.testing.assert_close(action, torch.tensor([[0.0, 0.0, 0.25]]))


def test_offline_evidence_passes_without_authorizing_training():
    report = json.loads((
        ROOT / "runs/taco_pour_truncated_gaussian_offline_gate_v1/report.json"
    ).read_text())
    assert report["status"] == "passed"
    assert report["sampled_components"] == 368640
    assert report["sampled_outside_state_bounds"] == 0
    assert report["official_clamp_changed_components"] == 0
    assert report["theoretical_ctrlrange_lost_components"] == 0
    assert all(report["checks"].values())
    assert report["decision"]["optimizer_training_authorized"] is False


def test_integration_evidence_is_zero_optimizer_and_exact():
    run = ROOT / "runs/taco_pour_truncated_gaussian_integration_gate_v1"
    report = json.loads((run / "report.json").read_text())
    assert report["status"] == "passed"
    assert report["PPO_optimizer_steps"] == 0
    assert report["task_level_training_executed"] is False
    assert report["chunk_commit_written"] is False
    assert report["frozen_rollout"]["samples"] == 160
    assert report["environment_action_semantics"][
        "actual_ctrlrange_nonzero_lost_components"
    ] == 0
    assert report["environment_action_semantics"][
        "maximum_abs_residual_lost_to_ctrlrange"
    ] == 0.0
    assert all(report["checks"].values())
    assert report["decision"]["optimizer_training_authorized"] is False
    assert not list(run.rglob("*.pth"))
    assert not list(run.rglob("*committed_boundary*"))
