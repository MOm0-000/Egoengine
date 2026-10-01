"""Candidate-D support-anchored bounded-mean PPO implementation.

The v4/v5 policy and evidence remain immutable.  Candidate D changes only the
parameterization of the location passed to the existing state-truncated Normal:
the actor emits an unconstrained latent ``eta`` and this module maps it into the
current state-dependent support before sampling or likelihood evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .algorithmic_training import AlgorithmicBenchmarkPpoAgent
from .algorithmic_training_v5 import MeanRegularizedAlgorithmicPpoAgent
from .state_feasible_truncated_gaussian import (
    TruncatedGaussianActionSpec,
    _distribution_terms,
    deterministic_truncated_action,
    truncated_normal_entropy,
)


SCHEMA = "taco_pour_algorithmic_training_update_audit_v6"
PROFILE_SCHEMA = "taco_support_anchored_bounded_mean_truncated_gaussian_v1"


def support_anchored_bounded_mean(
    raw_location: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Map an unconstrained location into a zero-anchored asymmetric support.

    At exactly zero the positive branch is selected when it has nonzero width;
    otherwise the negative branch is selected.  The value is always bitwise
    zero, while the available-direction derivative is kept nonzero for
    one-sided supports such as ``[0, 1]`` and ``[-1, 0]``.
    """

    if not (raw_location.shape == low.shape == high.shape):
        raise ValueError("raw location and support tensors must have identical shapes")
    if not all(torch.isfinite(value).all() for value in (raw_location, low, high)):
        raise ValueError("bounded-mean inputs must be finite")
    if bool((low >= high).any().item()):
        raise ValueError("bounded-mean support must have positive width")
    if bool(((low > 0.0) | (high < 0.0)).any().item()):
        raise ValueError("bounded-mean support must contain zero")

    positive_width = high
    negative_width = -low
    use_positive = (raw_location > 0.0) | (
        (raw_location == 0.0) & (positive_width > 0.0)
    )
    scale = torch.where(use_positive, positive_width, negative_width)
    bounded = scale * torch.tanh(raw_location)
    # Fail closed on any future change that violates the mathematical contract.
    if not bool(torch.isfinite(bounded).all()):
        raise RuntimeError("support-anchored bounded mean became nonfinite")
    tolerance = 2.0 * torch.finfo(bounded.dtype).eps
    if bool(((bounded < low - tolerance) | (bounded > high + tolerance)).any()):
        raise RuntimeError("support-anchored bounded mean escaped its support")
    return bounded


def load_support_anchored_profile(
    path: str | Path,
) -> tuple[TruncatedGaussianActionSpec, dict[str, Any]]:
    path = Path(path)
    raw = path.read_bytes()
    profile = yaml.safe_load(raw)
    distribution = profile.get("distribution", {})
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError("unsupported bounded-mean profile schema")
    if profile.get("status") != "authorized_local_algorithmic_candidate":
        raise ValueError("bounded-mean profile is not authorized")
    if profile.get("paper_faithful") is not False:
        raise ValueError("bounded-mean candidate must not be paper-faithful")
    expected = {
        "kind": "independent_state_truncated_normal",
        "location_parameterization": "support_anchored_piecewise_tanh",
        "network_output_semantics": "raw_location_latent_eta",
        "ppo_mus_semantics": "bounded_distribution_mean",
        "deterministic_action": "bounded_distribution_mean",
        "official_minus1_plus1_clamp_enabled": False,
    }
    for key, value in expected.items():
        if distribution.get(key) != value:
            raise ValueError(f"bounded-mean profile changed {key}")
    scale = float(distribution.get("residual_scale", math.nan))
    tolerance = float(distribution.get("reference_snap_tolerance", math.nan))
    minimum_mass = float(distribution.get("minimum_normalization_mass", math.nan))
    if not math.isclose(scale, 0.05) or not math.isclose(tolerance, 2.0e-7):
        raise ValueError("bounded-mean profile changed residual execution semantics")
    if not math.isclose(minimum_mass, 1.0e-12):
        raise ValueError("bounded-mean profile changed the normalization-mass floor")
    if profile.get("promotion", {}).get("optimizer_training_authorized") is not True:
        raise ValueError("bounded-mean profile did not authorize optimizer training")
    spec = TruncatedGaussianActionSpec(
        profile_id=str(profile["profile_id"]),
        residual_scale=scale,
        reference_snap_tolerance=tolerance,
        minimum_normalization_mass=minimum_mass,
        optimizer_training_authorized=True,
    )
    return spec, {
        "profile_id": spec.profile_id,
        "profile_path": str(path.resolve()),
        "profile_sha256": hashlib.sha256(raw).hexdigest(),
        "status": profile["status"],
        "paper_faithful": False,
        "distribution": distribution,
        "promotion": profile["promotion"],
    }


def bounded_mean_support_health(
    *,
    raw_location: torch.Tensor,
    bounded_mu: torch.Tensor,
    sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> dict[str, Any]:
    if not (
        raw_location.shape
        == bounded_mu.shape
        == sigma.shape
        == low.shape
        == high.shape
    ):
        raise ValueError("bounded-mean support-health tensors must share a shape")
    terms = _distribution_terms(
        bounded_mu, sigma, low, high, minimum_mass=1.0e-12
    )
    *_, mass = terms
    raw64 = raw_location.detach().double().reshape(-1)
    mu64 = bounded_mu.detach().double()
    sigma64 = sigma.detach().double()
    low64 = low.detach().double()
    high64 = high.detach().double()
    outside = (mu64 < low64) | (mu64 > high64)
    normalized_position = ((mu64 - low64) / (high64 - low64)).reshape(-1)
    distance_sigma = (
        torch.minimum(mu64 - low64, high64 - mu64) / sigma64
    ).reshape(-1)
    tanh_abs = torch.tanh(raw64).abs()
    at_boundary = torch.isclose(mu64, low64, rtol=0.0, atol=1.0e-7) | torch.isclose(
        mu64, high64, rtol=0.0, atol=1.0e-7
    )
    return {
        "minimum_truncated_normalization_mass": float(mass.min().detach().cpu()),
        "p01_truncated_normalization_mass": float(
            torch.quantile(mass.detach().double().reshape(-1), 0.01).cpu()
        ),
        "median_truncated_normalization_mass": float(
            torch.quantile(mass.detach().double().reshape(-1), 0.50).cpu()
        ),
        "raw_location_abs_mean": float(raw64.abs().mean().cpu()),
        "raw_location_abs_p95": float(torch.quantile(raw64.abs(), 0.95).cpu()),
        "raw_location_abs_max": float(raw64.abs().max().cpu()),
        "tanh_saturation_fraction": float((tanh_abs >= 0.99).double().mean().cpu()),
        "bounded_mu_outside_support_count": int(outside.sum().cpu()),
        "bounded_mu_at_support_fraction": float(at_boundary.double().mean().cpu()),
        "bounded_mu_normalized_support_position": {
            "minimum": float(normalized_position.min().cpu()),
            "p05": float(torch.quantile(normalized_position, 0.05).cpu()),
            "median": float(torch.quantile(normalized_position, 0.50).cpu()),
            "p95": float(torch.quantile(normalized_position, 0.95).cpu()),
            "maximum": float(normalized_position.max().cpu()),
        },
        "minimum_distance_to_support_boundary_in_sigma": float(
            distance_sigma.min().cpu()
        ),
    }


class SupportAnchoredBoundedMeanPpoAgent(AlgorithmicBenchmarkPpoAgent):
    """Candidate D with bounded means on every rollout and optimizer path."""

    # Reuse the already-audited v5 semantic/canonical identity contract without
    # inheriting v5's mean-regularization loss or report semantics.
    _validate_canonical_first_update_identity = (
        MeanRegularizedAlgorithmicPpoAgent._validate_canonical_first_update_identity
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if float(self.cfg.bounds_loss_coef or 0.0) != 0.0:
            raise ValueError("Candidate D requires bounds_loss_coef=0")
        if self.cfg.lr_schedule is not None:
            raise ValueError("Candidate D forbids an LR schedule")
        self._latest_support_arrays: dict[str, np.ndarray] | None = None

    def init_tensors(self) -> None:
        super().init_tensors()
        actions = self.experience_buffer.tensor_dict["actions"]
        self.experience_buffer.tensor_dict["raw_locations"] = torch.empty_like(actions)
        self.update_list.append("raw_locations")
        self.tensor_list = self.update_list + ["obses", "states", "dones"]

    def _actor_outputs(self, obs, *, advance_only: bool = False):
        result = super()._actor_outputs(obs, advance_only=advance_only)
        if advance_only:
            return result
        low, high = self.env.current_normalized_action_bounds()
        low = low.to(self.device)
        high = high.to(self.device)
        raw_location = result["mus"]
        if not (low.shape == high.shape == raw_location.shape):
            raise ValueError("actor batch and bounded-mean support differ")
        result["raw_locations"] = raw_location
        result["mus"] = support_anchored_bounded_mean(raw_location, low, high)
        return result

    def prepare_dataset(self, batch_dict) -> None:
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["raw_locations"] = batch_dict["raw_locations"]

    def evaluate_ppo_distribution(self, input_dict) -> dict[str, torch.Tensor]:
        """Exact recurrent evaluator with the support transform in-graph."""

        actions = input_dict["actions"]
        low = input_dict["action_lows"]
        high = input_dict["action_highs"]
        obs = self._preproc_obs(input_dict["obs"])
        model_was_training = self.model.training
        self.model.eval()
        if not self.is_rnn:
            raise ValueError("Candidate D requires the frozen RNN actor")
        worlds = int(self.cfg.num_actors)
        horizon = int(self.cfg.horizon_length)
        sequence = int(self.cfg.seq_length)
        if (
            actions.shape[0] != worlds * horizon
            or horizon % sequence != 0
            or self.minibatch_size != self.batch_size
        ):
            raise ValueError("Candidate D requires the frozen full-world rollout batch")
        blocks = horizon // sequence
        normalized_obs = self.model.norm_obs(obs, update_stats=False)
        obs_by_world = normalized_obs.reshape(worlds, horizon, *normalized_obs.shape[1:])
        dones_by_world = input_dict["dones"].reshape(worlds, horizon).bool()
        initial_states = input_dict["rnn_states"]
        curriculum = self.env.tail_curriculum_audit()
        if curriculum is None:
            reset_states = tuple(
                torch.zeros(
                    (state.shape[0], worlds, state.shape[2]),
                    dtype=state.dtype,
                    device=state.device,
                )
                for state in initial_states
            )
        else:
            reset_states = self.env.curriculum_rnn_reset_states()
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
                raise RuntimeError("Candidate D evaluator left an output unfilled")
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

    def train_actor_critic(self, input_dict):
        result = super().train_actor_critic(input_dict)
        post = self._latest_distribution_evaluation
        if post is None or "raw_location" not in post:
            raise RuntimeError("Candidate D did not retain its bounded distribution")
        health = bounded_mean_support_health(
            raw_location=post["raw_location"],
            bounded_mu=post["mu"],
            sigma=post["sigma"],
            low=input_dict["action_lows"],
            high=input_dict["action_highs"],
        )
        if health["bounded_mu_outside_support_count"] != 0:
            raise RuntimeError("Candidate D bounded mean escaped support")
        row = self.update_reports[-1]
        row.update(health)
        row["location_parameterization"] = {
            "network_output": "raw_location_latent_eta",
            "ppo_mu": "support_anchored_bounded_distribution_mean",
            "bounds_loss_coef": 0.0,
        }
        self._latest_support_arrays = {
            "raw_location": post["raw_location"].detach().cpu().numpy(),
            "bounded_mu": post["mu"].detach().cpu().numpy(),
            "sigma": post["sigma"].detach().cpu().numpy(),
            "low": input_dict["action_lows"].detach().cpu().numpy(),
            "high": input_dict["action_highs"].detach().cpu().numpy(),
        }
        return result

    def train_epoch(self):
        result = super().train_epoch()
        if self._latest_support_arrays is None:
            raise RuntimeError("Candidate D support arrays were not captured")
        path = self.audit_dir / f"epoch_{self.epoch_num:04d}_support.npz"
        np.savez_compressed(path, **self._latest_support_arrays)
        artifact = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        row = self.epoch_reports[-1]
        row["support_distribution"] = artifact
        summary_path = self.audit_dir / f"epoch_{self.epoch_num:04d}_summary.json"
        summary_path.write_text(json.dumps(row, indent=2) + "\n")
        self._latest_support_arrays = None
        return result

    def finalize_algorithmic_audit(self) -> dict[str, Any]:
        result = super().finalize_algorithmic_audit()
        path = Path(result["path"])
        manifest = json.loads(path.read_text())
        manifest["schema"] = SCHEMA
        manifest["location_parameterization"] = (
            "support_anchored_piecewise_tanh"
        )
        manifest["bounds_loss_coef"] = 0.0
        manifest["rollout_canonical_likelihood_role"] = (
            "diagnostic_below_semantic_hard_fail_1e-4"
        )
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        result.update(
            schema=SCHEMA,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            location_parameterization=manifest["location_parameterization"],
            bounds_loss_coef=0.0,
        )
        return result
