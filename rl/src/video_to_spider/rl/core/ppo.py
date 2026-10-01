"""Direct, non-inherited PPO execution for the one supported Pour contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .audit import detached_snapshot, distribution, module_sha256
from .distribution import exact_kl
from .policy import DistributionOutput, PolicyBundle, recurrent_evaluate
from .rollout import FixedBoundaryCollector, RolloutBatch


@dataclass(frozen=True)
class PPOConfig:
    worlds: int = 4
    horizon: int = 40
    sequence: int = 4
    gamma: float = 0.998
    gae_tau: float = 0.95
    clip: float = 0.2
    actor_learning_rate: float = 1.0e-4
    critic_learning_rate: float = 5.0e-5
    actor_updates: int = 1
    critic_updates: int = 4
    critic_coefficient: float = 4.0
    entropy_coefficient: float = 0.0
    bounds_loss_coefficient: float = 0.0
    grad_norm: float = 1.0

    def validate(self) -> None:
        expected = {
            "worlds": 4, "horizon": 40, "sequence": 4,
            "actor_updates": 1, "critic_updates": 4,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"R1 does not support {name}={getattr(self, name)}")
        if self.actor_learning_rate != 1.0e-4 or self.critic_learning_rate != 5.0e-5:
            raise ValueError("R1 learning rates are frozen")
        if self.entropy_coefficient != 0.0 or self.bounds_loss_coefficient != 0.0:
            raise ValueError("R1 entropy/bounds losses are frozen off")


@dataclass
class PreparedBatch:
    actor_input: dict[str, Any]
    critic_input: dict[str, torch.Tensor]
    raw_advantage: torch.Tensor
    normalized_advantage: torch.Tensor


def prepare_batch(policy: PolicyBundle, batch: RolloutBatch) -> PreparedBatch:
    """Build actor/critic inputs using one consistent value-RMS snapshot.

    B1 remains closed: old values and returns are normalized as one pair after
    a single statistics update, so both losses observe the same transform.
    The actor-internal head is an auxiliary regressor for the same normalized
    GAE return.  It is never used as the rollout baseline.
    """
    raw_advantage = (batch.returns - batch.values).sum(dim=1)
    advantage = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1.0e-8)
    if policy.critic is None:
        old_values, returns = batch.values, batch.returns
    else:
        old_values, returns = policy.critic.model.value_mean_std.normalize_pair(
            batch.values, batch.returns, update_stats=True
        )
        policy.critic.model.value_mean_std.eval()
    actor_input: dict[str, Any] = {
        "old_values": old_values,
        "old_logp_actions": batch.neglogp,
        "advantages": advantage,
        "returns": returns,
        "actions": batch.actions,
        "obs": batch.observations,
        "episode_start": batch.episode_start,
        "rnn_states": batch.block_start_states,
        "mu": batch.mu,
        "sigma": batch.sigma,
        "raw_location": batch.raw_location,
        "action_lows": batch.action_low,
        "action_highs": batch.action_high,
        "normalization_version": batch.normalization_version,
    }
    critic_input = {
        "old_values": old_values,
        "returns": returns,
        "actions": batch.actions,
        "obs": batch.critic_observations,
        "done_after": batch.done_after,
    }
    return PreparedBatch(actor_input, critic_input, raw_advantage, advantage)


def update_external_critic(
    policy: PolicyBundle, prepared: PreparedBatch, config: PPOConfig
) -> list[float]:
    critic = policy.critic
    if critic is None:
        return []
    if bool(config.critic_updates) and bool(critic.cfg.freeze_critic):
        return []
    losses: list[float] = []
    for _ in range(config.critic_updates):
        critic.model.train()
        critic.model.running_mean_std.eval()
        result = critic.model(
            {
                "obs": prepared.critic_input["obs"],
                "actions": prepared.critic_input["actions"],
                "is_train": True,
                "update_obs_stats": False,
            }
        )
        values = result["values"]
        old_values = prepared.critic_input["old_values"]
        returns = prepared.critic_input["returns"]
        clipped = old_values + (values - old_values).clamp(-config.clip, config.clip)
        loss = torch.max((values - returns).square(), (clipped - returns).square()).mean()
        for parameter in critic.model.parameters():
            parameter.grad = None
        loss.backward()
        nn.utils.clip_grad_norm_(critic.model.parameters(), config.grad_norm)
        critic.optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def update_actor(
    policy: PolicyBundle,
    actor_input: dict[str, Any],
    reset_states: tuple[torch.Tensor, ...],
    config: PPOConfig,
) -> dict[str, Any]:
    """Apply the one FULL actor update with a differentiable canonical old policy."""
    config.validate()
    actor = policy.actor
    actor.eval()
    before_hash = module_sha256(actor)
    current = recurrent_evaluate(
        actor,
        observations=actor_input["obs"],
        actions=actor_input["actions"],
        low=actor_input["action_lows"],
        high=actor_input["action_highs"],
        episode_start=actor_input["episode_start"],
        block_start_states=actor_input["rnn_states"],
        reset_states=reset_states,
        worlds=config.worlds,
        horizon=config.horizon,
        sequence=config.sequence,
        distribution=policy.distribution,
    )
    if int(actor_input["normalization_version"]) != int(policy.normalization_version):
        raise RuntimeError("rollout and recomputation use different observation RMS versions")
    rollout_ratio = torch.exp(actor_input["old_logp_actions"] - current.neglogp)
    if not bool(torch.isfinite(rollout_ratio).all()):
        raise RuntimeError("live rollout likelihood ratio is non-finite")
    rollout_identity_error = float((rollout_ratio - 1.0).abs().max().detach())
    if rollout_identity_error > 1.0e-4:
        raise RuntimeError(
            f"live rollout/recomputation likelihood mismatch: {rollout_identity_error}"
        )
    float32_envelope = 128.0 * torch.finfo(torch.float32).eps
    live_errors = {
        "mu": float((actor_input["mu"] - current.mu).abs().max().detach()),
        "sigma": float((actor_input["sigma"] - current.sigma).abs().max().detach()),
        "raw_location": float(
            (actor_input["raw_location"] - current.raw_location).abs().max().detach()
        ),
        "neglogp": float(
            (actor_input["old_logp_actions"] - current.neglogp).abs().max().detach()
        ),
    }
    if any(not torch.isfinite(torch.tensor(value)) for value in live_errors.values()):
        raise RuntimeError("live rollout/recomputation comparison is non-finite")
    if any(value > float32_envelope for name, value in live_errors.items() if name != "neglogp"):
        raise RuntimeError(f"live rollout distribution mismatch: {live_errors}")
    canonical_old = detached_snapshot(
        {"neglogp": current.neglogp, "mu": current.mu, "sigma": current.sigma}
    )
    ratio = torch.exp(canonical_old["neglogp"] - current.neglogp)
    identity_error = float((ratio - 1.0).abs().max().detach())
    if identity_error != 0.0:
        raise RuntimeError(f"canonical likelihood identity failed: {identity_error}")
    advantage = actor_input["advantages"]
    clipped_ratio = ratio.clamp(1.0 - config.clip, 1.0 + config.clip)
    actor_loss = torch.max(-advantage * ratio, -advantage * clipped_ratio).mean()
    returns = actor_input["returns"]
    internal_value_loss = (current.values - returns).square().mean()
    loss = actor_loss + 2.0 * internal_value_loss
    for parameter in actor.parameters():
        parameter.grad = None
    loss.backward()
    norm = nn.utils.clip_grad_norm_(actor.parameters(), config.grad_norm)
    policy.actor_optimizer.step()
    with torch.no_grad():
        post = recurrent_evaluate(
            actor,
            observations=actor_input["obs"],
            actions=actor_input["actions"],
            low=actor_input["action_lows"],
            high=actor_input["action_highs"],
            episode_start=actor_input["episode_start"],
            block_start_states=actor_input["rnn_states"],
            reset_states=reset_states,
            worlds=config.worlds,
            horizon=config.horizon,
            sequence=config.sequence,
            distribution=policy.distribution,
        )
        post_ratio = torch.exp(canonical_old["neglogp"] - post.neglogp)
        kl = exact_kl(
            canonical_old["mu"], canonical_old["sigma"], post.mu, post.sigma,
            actor_input["action_lows"], actor_input["action_highs"],
        )
    return {
        "actor_loss": float(actor_loss.detach()),
        "internal_value_loss": float(internal_value_loss.detach()),
        "total_loss": float(loss.detach()),
        "gradient_norm_before_clip": float(norm.detach()),
        "canonical_ratio_max_abs_error": identity_error,
        "live_rollout_ratio_max_abs_error": rollout_identity_error,
        "live_recomputation_max_abs_errors": live_errors,
        "post_ratio": distribution(post_ratio),
        "exact_kl": distribution(kl),
        "actor_sha256_before": before_hash,
        "actor_sha256_after": module_sha256(actor),
    }


class PPOTrainer:
    """One concrete trainer with the complete epoch sequence visible here."""

    def __init__(
        self,
        *,
        policy: PolicyBundle,
        collector: FixedBoundaryCollector,
        config: PPOConfig | None = None,
    ) -> None:
        self.policy = policy
        self.collector = collector
        self.config = config or PPOConfig()
        self.config.validate()

    def run_epoch(self) -> dict[str, Any]:
        # prepare_epoch_boundary_and_context is performed explicitly by collect().
        actor_rms_before = {
            name: value.detach().clone()
            for name, value in self.policy.actor.running_mean_std.state_dict().items()
        }
        critic_rms_before = (
            {
                name: value.detach().clone()
                for name, value in self.policy.critic.model.running_mean_std.state_dict().items()
            }
            if self.policy.critic is not None
            else None
        )
        batch = self.collector.collect()
        prepared = prepare_batch(self.policy, batch)
        critic_losses = update_external_critic(self.policy, prepared, self.config)
        actor_report = update_actor(
            self.policy, prepared.actor_input, batch.reset_states, self.config
        )
        # Observation statistics are committed only after every optimizer pass.
        if any(
            not torch.equal(value, actor_rms_before[name])
            for name, value in self.policy.actor.running_mean_std.state_dict().items()
        ):
            raise RuntimeError("actor observation RMS mutated during rollout/update")
        if critic_rms_before is not None and any(
            not torch.equal(value, critic_rms_before[name])
            for name, value in self.policy.critic.model.running_mean_std.state_dict().items()
        ):
            raise RuntimeError("critic observation RMS mutated during rollout/update")
        self.policy.actor.update_obs_stats(batch.observations)
        if self.policy.critic is not None:
            self.policy.critic.model.update_obs_stats(batch.critic_observations)
        self.policy.normalization_version += 1
        return {
            "samples": self.config.worlds * self.config.horizon,
            "critic_losses": critic_losses,
            "actor": actor_report,
            "normalization_version_after_commit": self.policy.normalization_version,
            "training_enabled": True,
            "chunk_commit_enabled": False,
            "batch": batch,
            "raw_advantage": prepared.raw_advantage,
            "normalized_advantage": prepared.normalized_advantage,
        }
