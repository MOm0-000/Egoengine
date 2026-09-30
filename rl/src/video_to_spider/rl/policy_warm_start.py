"""Audited actor-only warm start for the Candidate-G cross-window experiment."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch

from video_to_spider.rl.algorithmic_training_v6 import (
    SupportAnchoredBoundedMeanPpoAgent,
    support_anchored_bounded_mean,
)
from video_to_spider.rl.replay_rl import _model_state_sha256
from video_to_spider.rl.state_feasible_truncated_gaussian import (
    _tensor_tree_sha256,
    truncated_normal_entropy,
)


SCHEMA = "taco_pour_candidate_G_actor_only_warm_start_v1"


def _optimizer_is_fresh(optimizer: torch.optim.Optimizer) -> bool:
    state = optimizer.state_dict()
    return state.get("state") == {} and all(
        int(group.get("step", 0)) == 0 for group in state.get("param_groups", ())
    )


def inherit_actor_only(
    agent: Any,
    donor_payload: dict[str, Any],
) -> dict[str, Any]:
    """Strictly import the complete donor actor while preserving fresh training state."""
    if donor_payload.get("schema") != "taco_pour_algorithmic_benchmark_checkpoint_v1":
        raise ValueError("Candidate G donor uses an unsupported checkpoint schema")
    if donor_payload.get("candidate") != "D" or int(donor_payload.get("seed", -1)) != 2:
        raise ValueError("Candidate G donor must be Candidate D seed 2")
    if int(donor_payload.get("agent_epoch", -1)) != 125:
        raise ValueError("Candidate G donor must be the epoch-125 milestone")
    if not _optimizer_is_fresh(agent.optimizer):
        raise RuntimeError("actor optimizer is not fresh before donor import")
    critic_hash_before = _model_state_sha256(
        agent.asymmetric_critic_net.model.state_dict()
    )
    critic_optimizer_fresh_before = _optimizer_is_fresh(
        agent.asymmetric_critic_net.optimizer
    )
    if not critic_optimizer_fresh_before:
        raise RuntimeError("critic optimizer is not fresh before donor import")

    incompatible = agent.model.load_state_dict(donor_payload["actor"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict donor actor import returned incompatible keys")
    agent._observation_normalization_version = int(
        donor_payload["observation_normalization_version"]
    )

    actor_hash = _model_state_sha256(agent.model.state_dict())
    donor_hash = _model_state_sha256(donor_payload["actor"])
    critic_hash_after = _model_state_sha256(
        agent.asymmetric_critic_net.model.state_dict()
    )
    checks = {
        "actor_state_strictly_equal_donor": actor_hash == donor_hash,
        "critic_unchanged_by_actor_import": critic_hash_before == critic_hash_after,
        "actor_optimizer_fresh": _optimizer_is_fresh(agent.optimizer),
        "critic_optimizer_fresh": _optimizer_is_fresh(
            agent.asymmetric_critic_net.optimizer
        ),
        "normalization_version_inherited": (
            int(agent._observation_normalization_version)
            == int(donor_payload["observation_normalization_version"])
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Candidate G inheritance gate failed: {checks}")
    return {
        "schema": SCHEMA,
        "donor_candidate": "D",
        "donor_seed": 2,
        "donor_epoch": 125,
        "actor_state_sha256": actor_hash,
        "inherited_state_keys": sorted(donor_payload["actor"]),
        "inherited_state_key_count": len(donor_payload["actor"]),
        "actor_input_normalization_keys": sorted(
            key for key in donor_payload["actor"] if "running_mean_std" in key
        ),
        "observation_normalization_version": int(
            agent._observation_normalization_version
        ),
        "critic_state_sha256": critic_hash_after,
        "checks": checks,
        "donor_critic_loaded": False,
        "donor_actor_optimizer_loaded": False,
        "donor_critic_optimizer_loaded": False,
        "donor_RNN_or_environment_loaded": False,
        "mean_head_zeroed_after_import": False,
        "sigma_rescaled_after_import": False,
    }


class CandidateGWarmStartPpoAgent(SupportAnchoredBoundedMeanPpoAgent):
    """Candidate D PPO plus lossless visitation/credit provenance for Candidate G."""

    def __init__(self, *args, run_id: str, training_seed: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if int(self.cfg.mini_epochs) != 1:
            raise ValueError("Candidate G is frozen to one actor mini-epoch")
        self.candidate_g_run_id = str(run_id)
        self.candidate_g_training_seed = int(training_seed)
        self._candidate_g_credit: dict[str, np.ndarray] | None = None
        self._candidate_g_credit_attached_epoch: int | None = None

    @staticmethod
    def _world_major(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def evaluate_ppo_distribution(self, input_dict) -> dict[str, torch.Tensor]:
        """Recompute likelihood with Candidate G's nonzero boundary reset memory.

        Candidate D's immutable evaluator knows only zero-reset and tail-curriculum
        rollouts.  Candidate G has the same distribution math but restores the
        current-actor memory paired with the fixed endpoint-40 physics boundary.
        Keeping this adapter here preserves the historical Candidate-D source hash.
        """

        actions = input_dict["actions"]
        low = input_dict["action_lows"]
        high = input_dict["action_highs"]
        obs = self._preproc_obs(input_dict["obs"])
        model_was_training = self.model.training
        self.model.eval()
        if not self.is_rnn:
            raise ValueError("Candidate G requires the frozen RNN actor")
        worlds = int(self.cfg.num_actors)
        horizon = int(self.cfg.horizon_length)
        sequence = int(self.cfg.seq_length)
        if (
            actions.shape[0] != worlds * horizon
            or horizon % sequence != 0
            or self.minibatch_size != self.batch_size
        ):
            raise ValueError("Candidate G requires the frozen full-world rollout batch")
        blocks = horizon // sequence
        normalized_obs = self.model.norm_obs(obs, update_stats=False)
        obs_by_world = normalized_obs.reshape(worlds, horizon, *normalized_obs.shape[1:])
        dones_by_world = input_dict["dones"].reshape(worlds, horizon).bool()
        initial_states = input_dict["rnn_states"]
        reset_states = self.env.rollout_reset_rnn_states()
        if reset_states is None:
            raise RuntimeError("Candidate G likelihood lacks its boundary reset memory")
        if len(initial_states) != len(reset_states):
            raise ValueError("rollout and reset RNN state structures differ")

        raw_rows: list[list[torch.Tensor | None]] = [
            [None] * horizon for _ in range(worlds)
        ]
        logstd_rows: list[list[torch.Tensor | None]] = [
            [None] * horizon for _ in range(worlds)
        ]
        value_rows: list[list[torch.Tensor | None]] = [
            [None] * horizon for _ in range(worlds)
        ]
        last_states = None
        for block in range(blocks):
            indices = torch.as_tensor(
                [world * blocks + block for world in range(worlds)],
                dtype=torch.long,
                device=self.device,
            )
            states = [state.index_select(1, indices) for state in initial_states]
            for local_step in range(sequence):
                time_index = block * sequence + local_step
                done = dones_by_world[:, time_index]
                if bool(done.any().item()):
                    states = [
                        torch.where(
                            done.reshape(1, worlds, 1),
                            reset.to(state.device),
                            state,
                        )
                        for state, reset in zip(states, reset_states, strict=True)
                    ]
                raw_step, logstd_step, value_step, states = self.model.a2c_network({
                    "obs": obs_by_world[:, time_index],
                    "rnn_states": states,
                })
                for world in range(worlds):
                    raw_rows[world][time_index] = raw_step[world]
                    logstd_rows[world][time_index] = logstd_step[world]
                    value_rows[world][time_index] = value_step[world]
            last_states = states

        def flatten(rows):
            if any(value is None for row in rows for value in row):
                raise RuntimeError("Candidate G evaluator left an output unfilled")
            return torch.stack([value for row in rows for value in row])

        raw_location = flatten(raw_rows)
        logstd = flatten(logstd_rows)
        values = flatten(value_rows)
        sigma = torch.exp(logstd)
        bounded_mu = support_anchored_bounded_mean(raw_location, low, high)
        neglogp = self.recompute_truncated_neglogp(
            actions, bounded_mu, sigma, low, high
        )
        entropy = truncated_normal_entropy(
            bounded_mu,
            sigma,
            low,
            high,
            minimum_mass=self.distribution_spec.minimum_normalization_mass,
        ).sum(dim=-1)
        result = {
            "neglogp": neglogp,
            "entropy": entropy,
            "mu": bounded_mu,
            "raw_location": raw_location,
            "sigma": sigma,
            "values": values,
            "rnn_states": last_states,
        }
        if model_was_training:
            self.model.train()
        self._latest_distribution_evaluation = {
            name: value.detach().clone()
            for name, value in result.items()
            if torch.is_tensor(value)
        }
        return result

    def prepare_dataset(self, batch_dict) -> None:
        raw_advantage = (batch_dict["returns"] - batch_dict["values"]).sum(dim=1)
        shaped = self.experience_buffer.tensor_dict["rewards"]
        shaped_world_major = shaped.transpose(0, 1).reshape(
            self.batch_size, *shaped.shape[2:]
        )
        super().prepare_dataset(batch_dict)
        self._candidate_g_credit = {
            "rollout_value_before_update": self._world_major(batch_dict["values"]),
            "gae_return": self._world_major(batch_dict["returns"]),
            "raw_advantage": self._world_major(raw_advantage),
            "normalized_advantage": self._world_major(
                self.dataset.values_dict["advantages"]
            ),
            "shaped_training_reward": self._world_major(shaped_world_major),
        }
        self._candidate_g_credit_attached_epoch = None

    def train_actor_critic(self, input_dict):
        result = super().train_actor_critic(input_dict)
        if self._candidate_g_credit is None:
            raise RuntimeError("Candidate G credit was not prepared")
        if self._candidate_g_credit_attached_epoch == int(self.epoch_num):
            raise RuntimeError("Candidate G credit was attached more than once")
        canonical = self._canonical_old_policy_for_latest_update
        if canonical is None:
            raise RuntimeError("Candidate G requires the canonical old policy")
        reset = self.env.rollout_reset_rnn_states()
        if reset is None:
            raise RuntimeError("Candidate G rollout lacks its nonzero reset context")
        normalizer = self._rollout_actor_normalizer
        if normalizer is None or self._rollout_actor_parameter_hash is None:
            raise RuntimeError("Candidate G rollout provenance was not captured")
        rms_hash = hashlib.sha256(
            json.dumps(normalizer, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.env.attach_training_credit(
            **self._candidate_g_credit,
            canonical_old_logprob=(-canonical["neglogp"]).detach().cpu().numpy(),
            actor_hash=self._rollout_actor_parameter_hash,
            rms_hash=rms_hash,
            normalization_version=int(self._rollout_normalization_version),
            reset_context_hash=_tensor_tree_sha256(reset),
        )
        self._candidate_g_credit_attached_epoch = int(self.epoch_num)
        return result

    def train_epoch(self):
        result = super().train_epoch()
        if self._candidate_g_credit_attached_epoch != int(self.epoch_num):
            raise RuntimeError("Candidate G epoch ended without joined credit evidence")
        self._candidate_g_credit = None
        return result
