"""Explicit fixed-boundary rollout collection and layout transforms."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .policy import PolicyBundle, burn_in_prefix, policy_step


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
    tracking_reward: torch.Tensor
    contact_bonus: torch.Tensor
    lift_reward: torch.Tensor
    tracking_score: torch.Tensor
    position_error: torch.Tensor
    rotation_error: torch.Tensor
    episode_start: torch.Tensor
    done_after: torch.Tensor
    terminated: torch.Tensor
    timeout: torch.Tensor
    returns: torch.Tensor
    last_values: torch.Tensor
    block_start_states: tuple[torch.Tensor, ...]
    reset_states: tuple[torch.Tensor, ...]
    source_endpoint: torch.Tensor
    outcome_endpoint: torch.Tensor
    command_reference_endpoint: torch.Tensor
    reward_reference_endpoint: torch.Tensor
    next_goal_reference_endpoint: torch.Tensor
    rollout_step: torch.Tensor
    world_index: torch.Tensor
    episode_serial: torch.Tensor
    ppo_flat_index: torch.Tensor
    normalization_version: int

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
        if self.last_values.shape != (worlds, 1):
            raise ValueError("last_values has the wrong shape")
        for name in (
            "neglogp", "rewards", "tracking_reward", "contact_bonus", "lift_reward",
            "tracking_score", "position_error", "rotation_error",
            "episode_start", "done_after", "terminated", "timeout",
            "source_endpoint", "outcome_endpoint", "command_reference_endpoint",
            "reward_reference_endpoint", "next_goal_reference_endpoint", "rollout_step",
            "world_index", "episode_serial", "ppo_flat_index",
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
        if not torch.equal(self.done_after, self.terminated | self.timeout):
            raise ValueError("done_after must equal terminated | timeout")
        for world in range(worlds):
            offset = world * horizon
            if not bool(self.episode_start[offset]):
                raise ValueError("the first row of every world must start an episode")
            for step in range(1, horizon):
                row = offset + step
                previous = row - 1
                if bool(self.episode_start[row]) != bool(self.done_after[previous]):
                    raise ValueError("episode_start is not the preceding post-action done")
                expected_source = (
                    self.source_endpoint[offset]
                    if bool(self.done_after[previous])
                    else self.outcome_endpoint[previous]
                )
                if self.source_endpoint[row] != expected_source:
                    raise ValueError("reference cursor disagrees with reset/transition history")
        finite_names = (
            "actions", "neglogp", "values", "mu", "sigma", "action_low", "action_high",
            "raw_location", "observations", "critic_observations", "rewards",
            "tracking_reward", "contact_bonus", "lift_reward", "tracking_score",
            "position_error", "rotation_error", "returns",
            "last_values",
        )
        if any(not bool(torch.isfinite(getattr(self, name)).all()) for name in finite_names):
            raise ValueError("rollout contains non-finite values")


def world_major(time_major: torch.Tensor) -> torch.Tensor:
    if time_major.ndim < 2:
        raise ValueError("time-major tensor must contain time and world axes")
    return time_major.transpose(0, 1).reshape(
        time_major.shape[0] * time_major.shape[1], *time_major.shape[2:]
    )


def pack_block_start_states(
    block_states: Sequence[torch.Tensor], *, worlds: int
) -> torch.Tensor:
    """Pack `[block][layers,world,hidden]` as world-major BPTT sequences."""
    if not block_states:
        raise ValueError("at least one recurrent block state is required")
    layers, observed_worlds, hidden = block_states[0].shape
    if observed_worlds != worlds or any(
        state.shape != (layers, worlds, hidden) for state in block_states
    ):
        raise ValueError("recurrent block states have inconsistent shapes")
    return torch.stack(tuple(block_states), dim=0).permute(1, 2, 0, 3).reshape(
        layers, worlds * len(block_states), hidden
    )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    done_after: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float = 0.998,
    tau: float = 0.95,
) -> torch.Tensor:
    """Compute GAE on `[T,W,1]`; visible right truncation is bootstrapped."""
    if rewards.shape != values.shape or rewards.ndim != 3 or rewards.shape[-1] != 1:
        raise ValueError("GAE rewards/values must share [T,W,1]")
    if done_after.shape != rewards.shape[:2] or last_values.shape != rewards.shape[1:]:
        raise ValueError("GAE done/bootstrap shapes disagree")
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(last_values)
    for time_index in reversed(range(rewards.shape[0])):
        next_value = last_values if time_index == rewards.shape[0] - 1 else values[time_index + 1]
        not_done = (~done_after[time_index]).to(values.dtype).unsqueeze(-1)
        delta = rewards[time_index] + gamma * not_done * next_value - values[time_index]
        last = delta + gamma * tau * not_done * last
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
        observation_prefix: Sequence[torch.Tensor | Any],
        horizon: int = 40,
        start_endpoint: int = 40,
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.boundary_state = boundary_state
        self.observation_prefix = tuple(observation_prefix)
        if len(self.observation_prefix) != 20:
            raise ValueError("fixed-boundary collector requires sources20--39 observations")
        self.reset_states: tuple[torch.Tensor, ...] = ()
        self.horizon = int(horizon)
        self.start_endpoint = int(start_endpoint)

    def _critic_value(self, states: torch.Tensor) -> torch.Tensor:
        critic = self.policy.critic
        if critic is None:
            return torch.zeros((states.shape[0], 1), dtype=torch.float32)
        critic.model.eval()
        with torch.no_grad():
            result = critic.model({
                "obs": states,
                "is_train": False,
                "update_obs_stats": False,
            })
        return result["values"]

    def collect(self) -> RolloutBatch:
        worlds = len(self.environment.worlds)
        self.environment.restore_all(self.boundary_state)
        self.reset_states = burn_in_prefix(
            self.policy.actor, self.observation_prefix, worlds=worlds
        )
        rnn_states = tuple(state.clone() for state in self.reset_states)
        observation_raw = self.environment.current_observation()
        if not isinstance(observation_raw, dict):
            raise ValueError("R1 collector requires asymmetric actor/critic observations")

        rows: dict[str, list[torch.Tensor]] = {name: [] for name in (
            "actions", "neglogp", "values", "mu", "sigma", "action_low",
            "action_high", "raw_location", "observations", "critic_observations",
            "rewards", "tracking_reward", "contact_bonus", "lift_reward", "tracking_score",
            "position_error", "rotation_error", "episode_start", "done_after", "terminated", "timeout", "source_endpoint",
            "outcome_endpoint", "command_reference_endpoint", "reward_reference_endpoint",
            "next_goal_reference_endpoint", "rollout_step", "world_index", "episode_serial",
        )}
        block_states: list[list[torch.Tensor]] = [[] for _ in rnn_states]
        episode_start = torch.ones(worlds, dtype=torch.bool)
        episode_serial = torch.zeros(worlds, dtype=torch.int64)
        required_info = (
            "source_reference_endpoint", "outcome_reference_endpoint",
            "command_reference_endpoint", "reward_reference_endpoint",
            "next_observation_goal_reference_endpoint", "time_outs", "terminated",
            "aggregate_tracking_reward", "aggregate_contact_bonus", "lift_reward",
            "object_tracking_error", "object_position_error", "object_rotation_error",
        )
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
            missing = [name for name in required_info if name not in info]
            if missing:
                raise KeyError(f"environment info is missing required fields: {missing}")
            terminated = torch.as_tensor(info["terminated"], dtype=torch.bool).clone()
            timeout = torch.as_tensor(info["time_outs"], dtype=torch.bool).clone()
            done_after = terminated | timeout
            returned_done = torch.as_tensor(done, dtype=torch.bool)
            if not torch.equal(returned_done, done_after):
                raise RuntimeError("environment done disagrees with terminated | time_outs")
            endpoints = {
                "source_endpoint": torch.as_tensor(info["source_reference_endpoint"], dtype=torch.int64),
                "outcome_endpoint": torch.as_tensor(info["outcome_reference_endpoint"], dtype=torch.int64),
                "command_reference_endpoint": torch.as_tensor(info["command_reference_endpoint"], dtype=torch.int64),
                "reward_reference_endpoint": torch.as_tensor(info["reward_reference_endpoint"], dtype=torch.int64),
                "next_goal_reference_endpoint": torch.as_tensor(
                    info["next_observation_goal_reference_endpoint"], dtype=torch.int64
                ),
            }
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
                "tracking_reward": torch.as_tensor(info["aggregate_tracking_reward"], dtype=torch.float32),
                "contact_bonus": torch.as_tensor(info["aggregate_contact_bonus"], dtype=torch.float32),
                "lift_reward": torch.as_tensor(info["lift_reward"], dtype=torch.float32),
                "tracking_score": torch.as_tensor(info["object_tracking_error"], dtype=torch.float32),
                "position_error": torch.as_tensor(info["object_position_error"], dtype=torch.float32).reshape(worlds, -1)[:, 0],
                "rotation_error": torch.as_tensor(info["object_rotation_error"], dtype=torch.float32).reshape(worlds, -1)[:, 0],
                "episode_start": episode_start,
                "done_after": done_after,
                "terminated": terminated,
                "timeout": timeout,
                **endpoints,
                "rollout_step": torch.full((worlds,), offset, dtype=torch.int64),
                "world_index": torch.arange(worlds, dtype=torch.int64),
                "episode_serial": episode_serial,
            }
            for name, value in values_by_name.items():
                rows[name].append(value.clone())
            rnn_states = output.rnn_states
            if bool(done_after.any()):
                done_mask = done_after.reshape(1, worlds, 1)
                rnn_states = tuple(
                    torch.where(done_mask, reset, state)
                    for state, reset in zip(rnn_states, self.reset_states, strict=True)
                )
            episode_serial = episode_serial + done_after.to(torch.int64)
            episode_start = done_after
            observation_raw = next_raw

        next_states = torch.as_tensor(observation_raw["states"], dtype=torch.float32)
        last_values = self._critic_value(next_states)
        rewards_tm = torch.stack(rows["rewards"]).unsqueeze(-1)
        values_tm = torch.stack(rows["values"])
        done_after_tm = torch.stack(rows["done_after"])
        returns_tm = compute_gae(rewards_tm, values_tm, done_after_tm, last_values)
        flat: dict[str, torch.Tensor] = {
            name: world_major(torch.stack(value)) for name, value in rows.items()
        }
        flat["returns"] = world_major(returns_tm)
        flat["last_values"] = last_values
        flat["block_start_states"] = tuple(
            pack_block_start_states(states, worlds=worlds)
            for states in block_states
        )
        flat["reset_states"] = self.reset_states
        flat["ppo_flat_index"] = torch.arange(worlds * self.horizon, dtype=torch.int64)
        flat["normalization_version"] = int(self.policy.normalization_version)
        batch = RolloutBatch(**flat)
        batch.validate(worlds=worlds, horizon=self.horizon)
        return batch
