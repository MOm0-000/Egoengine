from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "external/human2sim2robot"),
    str(ROOT / "external/spider_compat"),
]
sys.path.append("/data_all/zzx/egoengine_new/external/dexplore_python_deps")

from video_to_spider.rl.state_feasible_truncated_gaussian import (
    truncated_normal_entropy,
    truncated_normal_log_prob,
)
from video_to_spider.rl.value_loss_isolation import (
    BudgetCounter,
    IsolationContractError,
    exact_loss_bundle,
    gradient_decomposition,
    single_shadow_step,
    valid_prefix_summary,
)


class TinyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigma = torch.nn.Parameter(torch.full((2,), -0.7))
        self.actor_mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Tanh()
        )
        self.mu = torch.nn.Linear(4, 2)
        self.value = torch.nn.Linear(4, 1)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a2c_network = TinyNetwork()


class TinyAgent:
    def __init__(self) -> None:
        self.model = TinyModel()
        self.cfg = SimpleNamespace(
            e_clip=0.2,
            critic_coef=4.0,
            bounds_loss_coef=0.0,
            grad_norm=1.0,
        )
        self.current_entropy_coef = 0.0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)

    def evaluate_ppo_distribution(self, data):
        network = self.model.a2c_network
        hidden = network.actor_mlp(data["obs"])
        raw = network.mu(hidden)
        # Keep the synthetic support transform simple and differentiable.
        mu = data["action_lows"] + (
            data["action_highs"] - data["action_lows"]
        ) * (torch.tanh(raw) + 1.0) / 2.0
        sigma = torch.exp(network.sigma).expand_as(mu)
        logp = truncated_normal_log_prob(
            data["actions"], mu, sigma,
            data["action_lows"], data["action_highs"],
            minimum_mass=1e-12,
        ).sum(dim=-1)
        entropy = truncated_normal_entropy(
            mu, sigma, data["action_lows"], data["action_highs"],
            minimum_mass=1e-12,
        ).sum(dim=-1)
        return {
            "neglogp": -logp,
            "entropy": entropy,
            "mu": mu,
            "raw_location": raw,
            "sigma": sigma,
            "values": network.value(hidden),
        }


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(4)
    obs = torch.randn(8, 3)
    low = torch.full((8, 2), -1.0)
    high = torch.full((8, 2), 1.0)
    actions = torch.zeros(8, 2)
    return {
        "obs": obs,
        "actions": actions,
        "action_lows": low,
        "action_highs": high,
        "advantages": torch.tensor([1.0, -0.5, 0.2, -1.2, 0.8, -0.7, 0.3, 0.1]),
        "old_values": torch.zeros(8, 1),
        "returns": torch.linspace(-1.0, 1.0, 8).reshape(-1, 1),
    }


def test_exact_losses_weight_clip_and_canonical_denominator_detach() -> None:
    agent = TinyAgent()
    data = batch()
    bundle = exact_loss_bundle(agent, data)
    assert bundle.ratio.detach().equal(torch.ones_like(bundle.ratio))
    assert bundle.canonical_old["neglogp"].requires_grad is False
    assert torch.equal(
        bundle.internal_value_weighted.detach(),
        (2.0 * bundle.internal_value_unweighted).detach(),
    )
    assert torch.equal(
        bundle.full.detach(),
        (bundle.policy + bundle.internal_value_weighted).detach(),
    )
    bundle.policy.backward()
    assert agent.model.a2c_network.mu.weight.grad is not None
    assert torch.count_nonzero(agent.model.a2c_network.mu.weight.grad) > 0


def test_gradient_linearity_and_none_paths() -> None:
    agent = TinyAgent()
    report, rows = gradient_decomposition(agent, batch())
    assert report["linearity"]["passed"]
    assert report["linearity"]["relative_L2_error"] <= 1e-5
    by_name = {row["parameter"]: row for row in rows}
    assert by_name["a2c_network.value.weight"]["policy_is_none"]
    assert by_name["a2c_network.mu.weight"]["value_is_none"]
    assert by_name["a2c_network.sigma"]["value_is_none"]
    assert not by_name["a2c_network.actor_mlp.0.weight"]["policy_is_none"]
    assert not by_name["a2c_network.actor_mlp.0.weight"]["value_is_none"]


def test_independent_shadow_steps_and_value_only_shared_policy_effect() -> None:
    agent = TinyAgent()
    data = batch()
    canonical = exact_loss_bundle(agent, data).canonical_old
    actor = deepcopy(agent.model.state_dict())
    optimizer = deepcopy(agent.optimizer.state_dict())
    full, _ = single_shadow_step(
        agent, data, branch="FULL", pre_actor_state=actor,
        pre_optimizer_state=optimizer, canonical_old=canonical,
    )
    full_optimizer_hash = full["state_hashes"]["optimizer"]
    policy, _ = single_shadow_step(
        agent, data, branch="POLICY_ONLY", pre_actor_state=actor,
        pre_optimizer_state=optimizer, canonical_old=canonical,
    )
    value, _ = single_shadow_step(
        agent, data, branch="VALUE_ONLY", pre_actor_state=actor,
        pre_optimizer_state=optimizer, canonical_old=canonical,
    )
    assert full["optimizer_steps"] == policy["optimizer_steps"] == value["optimizer_steps"] == 1
    assert full_optimizer_hash != policy["state_hashes"]["optimizer"]
    assert value["bounded_mu_change"]["maximum_absolute"] > 0
    assert value["policy_direct_parameter_delta_l2"] == 0
    assert policy["internal_value_head_delta_l2"] == 0


def test_budget_and_prefix_fail_closed() -> None:
    budget = BudgetCounter()
    budget.add_collection(480)
    budget.add_closed_loop(480)
    budget.add_critic_steps(12)
    for _ in range(9):
        budget.add_actor_step()
    assert budget.report()["within_all_limits"]
    with pytest.raises(IsolationContractError):
        budget.add_actor_step()

    summary = valid_prefix_summary(
        np.arange(41, 81),
        np.asarray([False] * 20 + [True] + [False] * 19),
        np.asarray([0.9] * 20 + [1.01] + [0.4] * 19),
        np.ones(40),
    )
    assert summary["valid_prefix_intervals"] == 20
    assert summary["first_failure_endpoint"] == 61
    assert summary["valid_prefix_tracking_reward_sum"] == 20.0


def test_endpoint80_timeout_is_not_tracking_failure() -> None:
    summary = valid_prefix_summary(
        np.arange(41, 81), np.zeros(40, dtype=bool),
        np.full(40, 0.9), np.ones(40),
    )
    assert summary["forty_of_forty"]
    assert summary["first_failure_endpoint"] is None
