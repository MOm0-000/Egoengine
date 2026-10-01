"""Concrete actor/critic construction and the sole recurrent evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from human2sim2robot.ppo.utils import models
from human2sim2robot.ppo.utils.asymmetric_critic import (
    AsymmetricCritic,
    AsymmetricCriticConfig,
)
from human2sim2robot.ppo.utils.network import MlpConfig, NetworkConfig, RnnConfig

from .distribution import (
    DistributionSpec,
    deterministic,
    entropy,
    log_prob,
    sample,
    support_anchored_bounded_mean,
)


ACTOR_OBSERVATION_DIM = 236
CRITIC_OBSERVATION_DIM = 108
ACTION_DIM = 36


def actor_network_config() -> NetworkConfig:
    return NetworkConfig(
        mlp=MlpConfig(units=[512, 512]),
        rnn=RnnConfig(
            units=1024,
            layers=1,
            name="lstm",
            layer_norm=True,
            before_mlp=False,
            concat_input=False,
            concat_output=True,
        ),
        separate_value_mlp=False,
        asymmetric_critic=False,
    )


def critic_network_config() -> NetworkConfig:
    return NetworkConfig(
        mlp=MlpConfig(units=[1024, 512]),
        rnn=None,
        separate_value_mlp=False,
        asymmetric_critic=True,
    )


def make_actor(*, worlds: int) -> models.ModelA2CContinuousLogStd:
    if worlds < 1:
        raise ValueError("world count must be positive")
    return models.ModelA2CContinuousLogStd(
        network_config=actor_network_config(),
        actions_num=ACTION_DIM,
        input_shape=(ACTOR_OBSERVATION_DIM,),
        normalize_value=True,
        normalize_input=True,
        value_size=1,
        num_seqs=worlds,
    )


def make_external_critic(*, worlds: int, horizon: int = 40) -> AsymmetricCritic:
    batch_size = worlds * horizon
    config = AsymmetricCriticConfig(
        name="xhand_asymmetric_critic_mlp",
        network=critic_network_config(),
        normalize_input=True,
        learning_rate=5.0e-5,
        mini_epochs=4,
        truncate_grads=True,
        minibatch_size=batch_size,
        grad_norm=1.0,
        e_clip=0.2,
    )
    return AsymmetricCritic(
        state_shape=(CRITIC_OBSERVATION_DIM,),
        value_size=1,
        ppo_device="cpu",
        num_agents=1,
        horizon_length=horizon,
        num_actors=worlds,
        num_actions=ACTION_DIM,
        seq_length=4,
        normalize_value=True,
        config=config,
        writer=None,
        max_epochs=1,
        multi_gpu=False,
        zero_rnn_on_done=True,
    )


@dataclass
class DistributionOutput:
    raw_location: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    values: torch.Tensor
    neglogp: torch.Tensor
    entropy: torch.Tensor
    rnn_states: tuple[torch.Tensor, ...]


@dataclass
class PolicyBundle:
    actor: models.ModelA2CContinuousLogStd
    actor_optimizer: torch.optim.Optimizer
    critic: AsymmetricCritic | None
    distribution: DistributionSpec
    normalization_version: int = 0

    @classmethod
    def create(
        cls,
        *,
        worlds: int,
        actor_learning_rate: float = 1.0e-4,
        with_critic: bool = True,
    ) -> "PolicyBundle":
        actor = make_actor(worlds=worlds)
        optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=actor_learning_rate,
            eps=1.0e-8,
            weight_decay=0.0,
        )
        return cls(
            actor=actor,
            actor_optimizer=optimizer,
            critic=make_external_critic(worlds=worlds) if with_critic else None,
            distribution=DistributionSpec(),
        )

    def load_actor_only(self, state: dict[str, torch.Tensor], *, version: int) -> None:
        incompatible = self.actor.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict actor import returned incompatible keys")
        self.normalization_version = int(version)


def recurrent_evaluate(
    actor: models.ModelA2CContinuousLogStd,
    *,
    observations: torch.Tensor,
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    episode_start: torch.Tensor,
    block_start_states: Sequence[torch.Tensor],
    reset_states: Sequence[torch.Tensor],
    worlds: int,
    horizon: int = 40,
    sequence: int = 4,
    distribution: DistributionSpec | None = None,
) -> DistributionOutput:
    """Evaluate the exact world-major BPTT layout used by collection.

    The caller supplies both block-start memory and the nonzero boundary reset
    memory. No environment or Candidate identity is consulted here.
    """
    distribution = distribution or DistributionSpec()
    distribution.validate()
    if observations.shape != (worlds * horizon, ACTOR_OBSERVATION_DIM):
        raise ValueError("unexpected actor observation layout")
    if actions.shape != (worlds * horizon, ACTION_DIM):
        raise ValueError("unexpected action layout")
    if horizon % sequence:
        raise ValueError("horizon must be divisible by BPTT sequence length")
    if len(block_start_states) != len(reset_states):
        raise ValueError("rollout and reset recurrent-state structures differ")
    blocks = horizon // sequence
    expected_sequences = worlds * blocks
    if any(state.shape[1] != expected_sequences for state in block_start_states):
        raise ValueError("block-start recurrent states do not match rollout layout")
    if any(state.shape[1] != worlds for state in reset_states):
        raise ValueError("reset recurrent states do not match world count")

    was_training = actor.training
    actor.eval()
    normalized = actor.norm_obs(observations, update_stats=False)
    by_world = normalized.reshape(worlds, horizon, ACTOR_OBSERVATION_DIM)
    episode_start_by_world = episode_start.reshape(worlds, horizon).bool()
    raw_rows: list[list[torch.Tensor | None]] = [[None] * horizon for _ in range(worlds)]
    logstd_rows: list[list[torch.Tensor | None]] = [[None] * horizon for _ in range(worlds)]
    value_rows: list[list[torch.Tensor | None]] = [[None] * horizon for _ in range(worlds)]
    last_states: list[torch.Tensor] | None = None
    for block in range(blocks):
        indices = torch.as_tensor(
            [world * blocks + block for world in range(worlds)], dtype=torch.long
        )
        states = [state.index_select(1, indices) for state in block_start_states]
        for local_step in range(sequence):
            time_index = block * sequence + local_step
            starts = episode_start_by_world[:, time_index]
            if bool(starts.any().item()):
                states = [
                    torch.where(starts.reshape(1, worlds, 1), reset, state)
                    for state, reset in zip(states, reset_states, strict=True)
                ]
            raw, logstd, value, states = actor.a2c_network(
                {"obs": by_world[:, time_index], "rnn_states": states}
            )
            for world in range(worlds):
                raw_rows[world][time_index] = raw[world]
                logstd_rows[world][time_index] = logstd[world]
                value_rows[world][time_index] = value[world]
        last_states = states

    def flatten(rows: list[list[torch.Tensor | None]]) -> torch.Tensor:
        if any(value is None for row in rows for value in row):
            raise RuntimeError("recurrent evaluator left an output unfilled")
        return torch.stack([value for row in rows for value in row])  # type: ignore[arg-type]

    raw_location = flatten(raw_rows)
    sigma = torch.exp(flatten(logstd_rows))
    values = flatten(value_rows)
    mu = support_anchored_bounded_mean(raw_location, low, high)
    neglogp = -log_prob(
        actions,
        mu,
        sigma,
        low,
        high,
        minimum_mass=distribution.minimum_normalization_mass,
    ).sum(dim=-1)
    ent = entropy(
        mu,
        sigma,
        low,
        high,
        minimum_mass=distribution.minimum_normalization_mass,
    ).sum(dim=-1)
    if was_training:
        actor.train()
    if last_states is None:
        raise RuntimeError("recurrent evaluator did not execute")
    return DistributionOutput(
        raw_location=raw_location,
        mu=mu,
        sigma=sigma,
        values=values,
        neglogp=neglogp,
        entropy=ent,
        rnn_states=tuple(last_states),
    )


@torch.no_grad()
def policy_step(
    actor: models.ModelA2CContinuousLogStd,
    observation: torch.Tensor,
    rnn_states: Sequence[torch.Tensor],
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    stochastic: bool,
    distribution: DistributionSpec | None = None,
) -> tuple[torch.Tensor, DistributionOutput]:
    distribution = distribution or DistributionSpec()
    actor.eval()
    normalized = actor.norm_obs(observation, update_stats=False)
    raw, logstd, values, next_states = actor.a2c_network(
        {"obs": normalized, "rnn_states": list(rnn_states)}
    )
    sigma = torch.exp(logstd)
    mu = support_anchored_bounded_mean(raw, low, high)
    action = (
        sample(mu, sigma, low, high, minimum_mass=distribution.minimum_normalization_mass)
        if stochastic
        else deterministic(mu, low, high)
    )
    neglogp = -log_prob(
        action, mu, sigma, low, high,
        minimum_mass=distribution.minimum_normalization_mass,
    ).sum(dim=-1)
    ent = entropy(
        mu, sigma, low, high,
        minimum_mass=distribution.minimum_normalization_mass,
    ).sum(dim=-1)
    return action, DistributionOutput(
        raw_location=raw,
        mu=mu,
        sigma=sigma,
        values=values,
        neglogp=neglogp,
        entropy=ent,
        rnn_states=tuple(next_states),
    )


@torch.no_grad()
def burn_in_prefix(
    actor: models.ModelA2CContinuousLogStd,
    observations: Sequence[torch.Tensor | Any],
    *,
    worlds: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Encode the fixed source20--39 prefix without sampling or RMS mutation."""
    if len(observations) != 20:
        raise ValueError("R1 boundary context requires exactly 20 prefix observations")
    actor.eval()
    states = tuple(
        state[:, :1].repeat(1, worlds, 1).zero_()
        for state in actor.get_default_rnn_state()
    )
    for row in observations:
        observation = torch.as_tensor(row, dtype=torch.float32)
        if observation.shape == (1, ACTOR_OBSERVATION_DIM) and worlds > 1:
            observation = observation.repeat(worlds, 1)
        if observation.shape != (worlds, ACTOR_OBSERVATION_DIM):
            raise ValueError("prefix observation has the wrong shape")
        normalized = actor.norm_obs(observation, update_stats=False)
        _, _, _, next_states = actor.a2c_network(
            {"obs": normalized, "rnn_states": list(states)}
        )
        states = tuple(next_states)
    return states
