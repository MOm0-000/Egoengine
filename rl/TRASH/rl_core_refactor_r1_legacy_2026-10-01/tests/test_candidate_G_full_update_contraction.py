from __future__ import annotations

import numpy as np
import pytest
import torch

from video_to_spider.rl.full_update_contraction import (
    BudgetLedger,
    ContractionContractError,
    decide_matrix,
    interpolate_actor_state,
    prefix_summary,
)


def states():
    base = {
        "a2c_network.mu.weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "a2c_network.sigma": torch.tensor([0.0], dtype=torch.float32),
        "a2c_network.running_mean_std.running_mean": torch.tensor([3.0]),
    }
    full = {
        "a2c_network.mu.weight": torch.tensor([3.0, 6.0], dtype=torch.float32),
        "a2c_network.sigma": torch.tensor([0.2], dtype=torch.float32),
        "a2c_network.running_mean_std.running_mean": torch.tensor([3.0]),
    }
    return base, full


def test_endpoints_clone_and_interior_uses_declared_float64_recipe():
    base, full = states()
    names = ["a2c_network.mu.weight", "a2c_network.sigma"]
    zero, za = interpolate_actor_state(base, full, names, 0.0)
    one, oa = interpolate_actor_state(base, full, names, 1.0)
    half, ha = interpolate_actor_state(base, full, names, 0.5)
    assert all(torch.equal(zero[k], base[k]) for k in base)
    assert all(torch.equal(one[k], full[k]) for k in full)
    expected = (base["a2c_network.mu.weight"].double() + 0.5 *
                (full["a2c_network.mu.weight"].double() -
                 base["a2c_network.mu.weight"].double())).float()
    assert torch.equal(half["a2c_network.mu.weight"], expected)
    assert za["endpoint_direct_clone"] and oa["endpoint_direct_clone"]
    assert ha["effective_displacement_ratio"] == pytest.approx(0.5)
    assert half["a2c_network.mu.weight"].data_ptr() != base["a2c_network.mu.weight"].data_ptr()
    assert torch.equal(base["a2c_network.mu.weight"], torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("mutation", ["key", "shape", "dtype", "buffer", "nonfinite"])
def test_interpolation_rejects_invalid_actor_pairs(mutation):
    base, full = states()
    if mutation == "key":
        full.pop("a2c_network.sigma")
    elif mutation == "shape":
        full["a2c_network.sigma"] = torch.ones(2)
    elif mutation == "dtype":
        full["a2c_network.sigma"] = full["a2c_network.sigma"].double()
    elif mutation == "buffer":
        full["a2c_network.running_mean_std.running_mean"] = torch.tensor([4.0])
    else:
        full["a2c_network.sigma"] = torch.tensor([float("nan")])
    with pytest.raises(ContractionContractError):
        interpolate_actor_state(
            base, full,
            ["a2c_network.mu.weight", "a2c_network.sigma"], 0.5,
        )


def test_log_sigma_is_a_parameter_and_buffer_is_not_interpolated():
    base, full = states()
    result, audit = interpolate_actor_state(
        base, full,
        ["a2c_network.mu.weight", "a2c_network.sigma"], 0.25,
    )
    assert result["a2c_network.sigma"].item() == pytest.approx(0.05)
    assert torch.equal(
        result["a2c_network.running_mean_std.running_mean"],
        base["a2c_network.running_mean_std.running_mean"],
    )
    assert audit["buffer_count"] == 1
    assert audit["eligible_for_training_resume"] is False


def test_prefix_never_recovers_after_first_failure_and_timeout_is_not_failure():
    endpoints = np.arange(41, 81)
    scores = np.full(40, 0.9)
    scores[3] = 1.01
    scores[4:] = 0.1
    row = prefix_summary(endpoints, np.zeros(40, bool), scores, np.ones(40))
    assert row["N"] == 3
    assert row["first_failure_endpoint"] == 44
    normal_timeout = np.zeros(40, bool); normal_timeout[-1] = True
    clean = prefix_summary(endpoints, normal_timeout, np.full(40, 0.9), np.ones(40))
    assert clean["N"] == 40 and clean["first_failure_endpoint"] is None


def test_common_alpha_decision_does_not_pick_per_seed_best():
    rows = []
    for seed in range(3):
        for alpha, n in ((0.0, 20), (1.0, 14), (0.5, 20 if seed != 2 else 19),
                         (0.25, 21), (0.125, 20)):
            rows.append({"seed": seed, "alpha": alpha, "N": n})
    decision = decide_matrix(rows)
    assert decision["common_preservation_set_C"] == [0.25, 0.125]
    assert decision["common_extension_set_E"] == [0.25]
    rows.pop()
    with pytest.raises(ContractionContractError):
        decide_matrix(rows)


def test_budget_ledger_fails_closed():
    ledger = BudgetLedger()
    for _ in range(15):
        ledger.add_batch(160)
        ledger.add_rollout(40, 400)
    assert ledger.as_dict()["total_actor_sample_rows"] == 3300
    with pytest.raises(ContractionContractError):
        ledger.add_rollout(1, 10)
