"""Explicit fixed-boundary rollout collection and layout transforms."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .policy import DistributionOutput, PolicyBundle, policy_step


@dataclass
class RolloutBatch:
    actions: torch.Tensor
    neglogp: torch.Tensor
    values: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    action_low: torch.Tensor
    action_high: torch.Tensor
    raw_location: torch.Tensor
    observations: torch.Tensor
    critic_observations: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    terminated: torch.Tensor
    timeout: torch.Tensor
    returns: torch.Tensor
    block_start_states: tuple[torch.Tensor, ...]
    reset_states: tuple[torch.Tensor, ...]
    source_endpoint: torch.Tensor
    outcome_endpoint: torch.Tensor
    command_reference_endpoint: torch.Tensor
    reward_reference_endpoint: torch.Tensor
    next_goal_reference_endpoint: torch.Tensor
    ppo_flat_index: torch.Tensor

    def validate(self, *, worlds: int, horizon: int) -> None:
        batch = worlds * horizon
        vector_names = {
            "actions": 36,
            "mu": 36,
            "sigma": 36,
            "action_low": 36,
            "action_high": 36,
            "raw_location": 36,
            "observations": 236,
            "critic_observations": 108,
        }
        for name, width in vector_names.items():
            if getattr(self, name).shape != (batch, width):
                raise ValueError(f"{name} has the wrong world-major shape")
        if self.values.shape != (batch, 1) or self.returns.shape != (batch, 1):
            raise ValueError("value tensors have the wrong shape")
        for name in (
            "neglogp", "rewards", "dones", "terminated", "timeout",
            "source_endpoint", "outcome_endpoint", "command_reference_endpoint",
            "reward_reference_endpoint", "next_goal_reference_endpoint", "ppo_flat_index",
        ):
            if getattr(self, name).shape != (batch,):
                raise ValueError(f"{name} has the wrong shape")
        if not torch.equal(
            self.outcome_endpoint, self.source_endpoint + 1
        ) or not torch.equal(self.command_reference_endpoint, self.outcome_endpoint):
            raise ValueError("source/command/outcome temporal contract changed")
        if not torch.equal(self.reward_reference_endpoint, self.outcome_endpoint):
            raise ValueError("reward is not aligned with physical outcome")
        if not torch.equal(self.next_goal_reference_endpoint, self.outcome_endpoint + 1):
            raise ValueError("next observation goal is not one endpoint ahead")


def world_major(time_major: torch.Tensor) -> torch.Tensor:
    if time_major.ndim < 2:
        raise ValueError("time-major tensor must contain time and world axes")
    return time_major.transpose(0, 1).reshape(
        time_major.shape[0] * time_major.shape[1], *time_major.shape[2:]
    )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float = 0.998,
    tau: float = 0.95,
) -> torch.Tensor:
    """Compute GAE on `[T,W,1]`; visible right truncation is bootstrapped."""
    if rewards.shape != values.shape or rewards.ndim != 3 or rewards.shape[-1] != 1:
        raise ValueError("GAE rewards/values must share [T,W,1]")
    if dones.shape != rewards.shape[:2] or last_values.shape != rewards.shape[1:]:
        raise ValueError("GAE done/bootstrap shapes disagree")
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(last_values)
    for time_index in reversed(range(rewards.shape[0])):
        if time_index == rewards.shape[0] - 1:
            next_value = last_values
            next_nonterminal = 1.0 - dones[time_index].float().unsqueeze(1)
        else:
            next_value = values[time_index + 1]
            next_nonterminal = 1.0 - dones[time_index + 1].float().unsqueeze(1)
        delta = rewards[time_index] + gamma * next_value * next_nonterminal - values[time_index]
        last = delta + gamma * tau * next_nonterminal * last
        advantages[time_index] = last
    return advantages + values


def valid_prefix(terminated: Sequence[bool], timeout: Sequence[bool]) -> int:
    """Count only the consecutive feasible prefix; never reconnect after failure."""
    if len(terminated) != len(timeout):
        raise ValueError("termination and timeout arrays differ in length")
    for index, (failed, timed_out) in enumerate(zip(terminated, timeout, strict=True)):
        if bool(failed):
            return index
        if bool(timed_out):
            return index + 1
    return len(terminated)


class FixedBoundaryCollector:
    """Collect the current four-world, 40-interval window with explicit state."""

    def __init__(
        self,
        *,
        environment: Any,
        policy: PolicyBundle,
        boundary_state: dict[str, Any],
        reset_states: Sequence[torch.Tensor],
        horizon: int = 40,
        start_endpoint: int = 40,
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.boundary_state = boundary_state
        self.reset_states = tuple(state.clone() for state in reset_states)
        self.horizon = int(horizon)
        self.start_endpoint = int(start_endpoint)

    def _critic_value(self, states: torch.Tensor) -> torch.Tensor:
        critic = self.policy.critic
        if critic is None:
            return torch.zeros((states.shape[0], 1), dtype=torch.float32)
        critic.model.eval()
        with torch.no_grad():
            normalized = critic.model.norm_obs(states, update_stats=False)
            result = critic.model({"obs": normalized, "is_train": True})
        return result["values"]

    def collect(self) -> RolloutBatch:
        worlds = len(self.environment.worlds)
        self.environment.restore_all(self.boundary_state)
        rnn_states = tuple(state.clone() for state in self.reset_states)
        observation_raw = self.environment.current_observation()
        if not isinstance(observation_raw, dict):
            raise ValueError("R1 collector requires asymmetric actor/critic observations")

        rows: dict[str, list[torch.Tensor]] = {name: [] for name in (
            "actions", "neglogp", "values", "mu", "sigma", "action_low",
            "action_high", "raw_location", "observations", "critic_observations",
            "rewards", "dones", "terminated", "timeout", "source_endpoint",
            "outcome_endpoint", "command_reference_endpoint", "reward_reference_endpoint",
            "next_goal_reference_endpoint",
        )}
        block_states: list[list[torch.Tensor]] = [[] for _ in rnn_states]
        for offset in range(self.horizon):
            if offset % 4 == 0:
                for index, state in enumerate(rnn_states):
                    block_states[index].append(state.clone())
            observation = torch.as_tensor(observation_raw["obs"], dtype=torch.float32)
            critic_observation = torch.as_tensor(observation_raw["states"], dtype=torch.float32)
            low, high = self.environment.normalized_action_bounds()
            action, output = policy_step(
                self.policy.actor,
                observation,
                rnn_states,
                low,
                high,
                stochastic=True,
                distribution=self.policy.distribution,
            )
            values = self._critic_value(critic_observation)
            next_raw, reward, done, info = self.environment.step(action, auto_reset=True)
            terminated = torch.as_tensor(info["terminated"], dtype=torch.bool)
            timeout = torch.as_tensor(info.get("timeout", np.zeros(worlds)), dtype=torch.bool)
            source = torch.full((worlds,), self.start_endpoint + offset, dtype=torch.int64)
            outcome = source + 1
            values_by_name = {
                "actions": action,
                "neglogp": output.neglogp,
                "values": values,
                "mu": output.mu,
                "sigma": output.sigma,
                "action_low": low,
                "action_high": high,
                "raw_location": output.raw_location,
                "observations": observation,
                "critic_observations": critic_observation,
                "rewards": torch.as_tensor(reward, dtype=torch.float32),
                "dones": torch.as_tensor(done, dtype=torch.bool),
                "terminated": terminated,
                "timeout": timeout,
                "source_endpoint": source,
                "outcome_endpoint": outcome,
                "command_reference_endpoint": outcome,
                "reward_reference_endpoint": outcome,
                "next_goal_reference_endpoint": outcome + 1,
            }
            for name, value in values_by_name.items():
                rows[name].append(value.clone())
            rnn_states = output.rnn_states
            if bool(torch.as_tensor(done).any()):
                done_mask = torch.as_tensor(done, dtype=torch.bool).reshape(1, worlds, 1)
                rnn_states = tuple(
                    torch.where(done_mask, reset, state)
                    for state, reset in zip(rnn_states, self.reset_states, strict=True)
                )
            observation_raw = next_raw

        next_states = torch.as_tensor(observation_raw["states"], dtype=torch.float32)
        last_values = self._critic_value(next_states)
        rewards_tm = torch.stack(rows["rewards"]).unsqueeze(-1)
        values_tm = torch.stack(rows["values"])
        dones_tm = torch.stack(rows["dones"])
        returns_tm = compute_gae(rewards_tm, values_tm, dones_tm, last_values)
        flat: dict[str, torch.Tensor] = {
            name: world_major(torch.stack(value)) for name, value in rows.items()
        }
        flat["returns"] = world_major(returns_tm)
        flat["block_start_states"] = tuple(
            torch.cat(states, dim=1) for states in block_states
        )
        flat["reset_states"] = self.reset_states
        flat["ppo_flat_index"] = torch.arange(worlds * self.horizon, dtype=torch.int64)
        batch = RolloutBatch(**flat)
        batch.validate(worlds=worlds, horizon=self.horizon)
        return batch
