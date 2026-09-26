"""Local state-feasible truncated-Gaussian action distribution for xHand PPO."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from human2sim2robot.ppo.ppo_agent import PpoAgent as OfficialPpoAgent


@dataclass(frozen=True)
class TruncatedGaussianActionSpec:
    profile_id: str
    residual_scale: float
    reference_snap_tolerance: float
    minimum_normalization_mass: float
    optimizer_training_authorized: bool


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def observation_normalizer_report(model: Any, *, version: int) -> dict[str, Any]:
    """Return a compact identity for the observation transform in use."""
    if not getattr(model, "normalize_input", False):
        return {"enabled": False, "version": int(version)}
    normalizer = model.running_mean_std
    state = normalizer.state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return {
        "enabled": True,
        "version": int(version),
        "state_sha256": digest.hexdigest(),
        "count": float(normalizer.count.detach().cpu()),
        "mean_sha256": _tensor_sha256(normalizer.running_mean),
        "variance_sha256": _tensor_sha256(normalizer.running_var),
    }


def load_truncated_gaussian_profile(
    path: str | Path,
) -> tuple[TruncatedGaussianActionSpec, dict[str, Any]]:
    path = Path(path)
    raw = path.read_bytes()
    profile = yaml.safe_load(raw)
    if profile.get("schema") != "taco_state_feasible_action_distribution_v1":
        raise ValueError("unsupported action-distribution profile schema")
    if profile.get("status") != "engineering_candidate_gate_only":
        raise ValueError("truncated-Gaussian profile is not a gate-only candidate")
    if profile.get("paper_faithful") is not False:
        raise ValueError("local action distribution must not be marked paper-faithful")
    distribution = profile.get("distribution", {})
    if distribution.get("kind") != "independent_state_truncated_normal":
        raise ValueError("unsupported candidate distribution")
    if distribution.get("deterministic_action") != "clip_mu_to_state_bounds":
        raise ValueError("deterministic candidate must clip mu to state bounds")
    if distribution.get("official_minus1_plus1_clamp_enabled") is not False:
        raise ValueError("official action clamp must be disabled for this candidate")
    scale = float(distribution.get("residual_scale", math.nan))
    tolerance = float(distribution.get("reference_snap_tolerance", math.nan))
    minimum_mass = float(distribution.get("minimum_normalization_mass", math.nan))
    if not math.isclose(scale, 0.05):
        raise ValueError("candidate is frozen to residual_scale=0.05")
    if not math.isclose(tolerance, 2e-7):
        raise ValueError("candidate is frozen to reference_snap_tolerance=2e-7")
    if not math.isfinite(minimum_mass) or not 0.0 < minimum_mass < 1.0:
        raise ValueError("minimum normalization mass must lie in (0,1)")
    if profile.get("promotion", {}).get("optimizer_training_authorized") is not False:
        raise ValueError("gate-only candidate must fail closed on optimizer training")
    spec = TruncatedGaussianActionSpec(
        profile_id=str(profile["profile_id"]),
        residual_scale=scale,
        reference_snap_tolerance=tolerance,
        minimum_normalization_mass=minimum_mass,
        optimizer_training_authorized=False,
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


def normalized_action_bounds_numpy(
    reference_ctrl: np.ndarray,
    *,
    ctrllimited: np.ndarray,
    ctrlrange: np.ndarray,
    residual_scale: float,
    reference_snap_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive normalized bounds, snapping only tolerance-level reference errors."""
    reference = np.asarray(reference_ctrl, np.float64)
    limited = np.asarray(ctrllimited, bool)
    ranges = np.asarray(ctrlrange, np.float64)
    if reference.ndim != 2 or reference.shape[1] != len(limited):
        raise ValueError("reference control and ctrlrange dimensions differ")
    if ranges.shape != (len(limited), 2):
        raise ValueError("ctrlrange must have shape (actions,2)")
    below = np.where(limited, np.maximum(ranges[:, 0] - reference, 0.0), 0.0)
    above = np.where(limited, np.maximum(reference - ranges[:, 1], 0.0), 0.0)
    violation = np.maximum(below, above)
    maximum = float(violation.max(initial=0.0))
    if maximum > reference_snap_tolerance:
        raise ValueError(
            f"reference exceeds ctrlrange by {maximum:.9g}, above the "
            f"{reference_snap_tolerance:.9g} tolerance"
        )
    lower_f32 = ranges[:, 0].astype(np.float32)
    upper_f32 = ranges[:, 1].astype(np.float32)
    lower_f32 = np.where(
        lower_f32.astype(np.float64) < ranges[:, 0],
        np.nextafter(lower_f32, np.float32(np.inf)),
        lower_f32,
    )
    upper_f32 = np.where(
        upper_f32.astype(np.float64) > ranges[:, 1],
        np.nextafter(upper_f32, np.float32(-np.inf)),
        upper_f32,
    )
    snapped = reference.astype(np.float32)
    snapped[:, limited] = np.clip(
        snapped[:, limited], lower_f32[limited], upper_f32[limited]
    )
    scale_f32 = np.float32(residual_scale)
    low = np.full(reference.shape, -1.0, np.float32)
    high = np.full(reference.shape, 1.0, np.float32)
    low[:, limited] = np.maximum(
        np.float32(-1.0),
        (lower_f32[limited] - snapped[:, limited]) / scale_f32,
    )
    high[:, limited] = np.minimum(
        np.float32(1.0),
        (upper_f32[limited] - snapped[:, limited]) / scale_f32,
    )
    inward_adjustment_count = 0
    for _ in range(4):
        requested_low = (snapped + scale_f32 * low).astype(np.float64)
        requested_high = (snapped + scale_f32 * high).astype(np.float64)
        low_outside = limited & (requested_low < ranges[:, 0])
        high_outside = limited & (requested_high > ranges[:, 1])
        if not (low_outside.any() or high_outside.any()):
            break
        inward_adjustment_count += int(low_outside.sum() + high_outside.sum())
        low = np.where(
            low_outside,
            np.nextafter(low, np.float32(np.inf)),
            low,
        )
        high = np.where(
            high_outside,
            np.nextafter(high, np.float32(-np.inf)),
            high,
        )
    else:
        raise RuntimeError("could not construct float32-safe action bounds")
    if np.any(low > high):
        raise ValueError("normalized action interval is empty")
    return low.astype(np.float64), high.astype(np.float64), {
        "reference_snap_tolerance": reference_snap_tolerance,
        "snapped_component_count": int((violation > 0.0).sum()),
        "maximum_reference_violation": maximum,
        "float32_inward_bound_adjustment_count": inward_adjustment_count,
        "empty_interval_count": int((low > high).sum()),
        "minimum_interval_width": float((high - low).min()),
    }


def _standard_normal_pdf(value: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)


def _distribution_terms(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    minimum_mass: float,
) -> tuple[torch.Tensor, ...]:
    if mu.shape != sigma.shape or mu.shape != low.shape or mu.shape != high.shape:
        raise ValueError("mu, sigma and bounds must have identical shapes")
    if not all(torch.isfinite(value).all() for value in (mu, sigma, low, high)):
        raise ValueError("truncated-Gaussian inputs must be finite")
    if bool((sigma <= 0.0).any().item()):
        raise ValueError("sigma must be strictly positive")
    if bool((low >= high).any().item()):
        raise ValueError("truncated-Gaussian intervals must have positive width")
    mu64 = mu.to(torch.float64)
    sigma64 = sigma.to(torch.float64)
    low64 = low.to(torch.float64)
    high64 = high.to(torch.float64)
    alpha = (low64 - mu64) / sigma64
    beta = (high64 - mu64) / sigma64
    cdf_low = torch.special.ndtr(alpha)
    cdf_high = torch.special.ndtr(beta)
    mass = cdf_high - cdf_low
    if bool((mass < minimum_mass).any().item()):
        minimum = float(mass.min().item())
        raise ValueError(
            f"truncated-Gaussian normalization mass {minimum:.9g} is below "
            f"the {minimum_mass:.9g} fail-closed threshold"
        )
    return mu64, sigma64, low64, high64, alpha, beta, cdf_low, mass


def truncated_normal_log_prob(
    action: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    minimum_mass: float,
) -> torch.Tensor:
    """Per-coordinate log probability under an independent truncated Normal."""
    terms = _distribution_terms(
        mu, sigma, low, high, minimum_mass=minimum_mass
    )
    mu64, sigma64, low64, high64, _, _, _, mass = terms
    action64 = action.to(torch.float64)
    tolerance = 8.0 * torch.finfo(action.dtype).eps
    if bool(((action64 < low64 - tolerance) | (action64 > high64 + tolerance)).any().item()):
        raise ValueError("action lies outside its truncated-Gaussian support")
    z = (action64 - mu64) / sigma64
    result = (
        -0.5 * z.square()
        - torch.log(sigma64)
        - 0.5 * math.log(2.0 * math.pi)
        - torch.log(mass)
    )
    return result.to(mu.dtype)


def truncated_normal_entropy(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    minimum_mass: float,
) -> torch.Tensor:
    """Per-coordinate differential entropy of a truncated Normal."""
    terms = _distribution_terms(
        mu, sigma, low, high, minimum_mass=minimum_mass
    )
    _, sigma64, _, _, alpha, beta, _, mass = terms
    correction = (
        alpha * _standard_normal_pdf(alpha)
        - beta * _standard_normal_pdf(beta)
    ) / (2.0 * mass)
    entropy = (
        torch.log(sigma64)
        + 0.5 * math.log(2.0 * math.pi * math.e)
        + torch.log(mass)
        + correction
    )
    return entropy.to(mu.dtype)


def sample_truncated_normal(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    minimum_mass: float,
) -> torch.Tensor:
    """Inverse-CDF sample with no rejection and exact state-dependent support."""
    terms = _distribution_terms(
        mu, sigma, low, high, minimum_mass=minimum_mass
    )
    mu64, sigma64, low64, high64, _, _, cdf_low, mass = terms
    uniform = torch.rand(mu.shape, dtype=torch.float64, device=mu.device)
    probability = cdf_low + uniform * mass
    epsilon = torch.finfo(torch.float64).eps
    probability = torch.clamp(probability, epsilon, 1.0 - epsilon)
    sample = mu64 + sigma64 * torch.special.ndtri(probability)
    sample = torch.maximum(torch.minimum(sample, high64), low64)
    return sample.to(mu.dtype)


def deterministic_truncated_action(
    mu: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    if mu.shape != low.shape or mu.shape != high.shape:
        raise ValueError("mu and bounds must have identical shapes")
    return torch.maximum(torch.minimum(mu, high), low)


class StateFeasibleTruncatedGaussianPpoAgent(OfficialPpoAgent):
    """PPO candidate with bounds stored in rollouts and correct truncated likelihood."""

    def __init__(
        self,
        *args,
        distribution_spec: TruncatedGaussianActionSpec,
        commit_observation_stats_after_epoch: bool = True,
        **kwargs,
    ):
        self.distribution_spec = distribution_spec
        super().__init__(*args, **kwargs)
        if self.cfg.clip_actions:
            raise ValueError("official [-1,1] action clamp must be disabled")
        if not hasattr(self.env, "current_normalized_action_bounds"):
            raise ValueError("environment does not provide state-dependent action bounds")
        self.env.enable_state_feasible_action_contract(
            reference_snap_tolerance=distribution_spec.reference_snap_tolerance
        )
        self._training_trace_distribution = None
        self._commit_observation_stats_after_epoch = bool(
            commit_observation_stats_after_epoch
        )
        if (
            not self._commit_observation_stats_after_epoch
            and float(self.cfg.learning_rate) != 0.0
        ):
            raise ValueError(
                "observation-stat commit may be disabled only for an lr=0 gate"
            )
        self._observation_normalization_version = 0
        self._observation_normalization_history: list[dict[str, Any]] = []
        self._likelihood_identity_checks: list[dict[str, Any]] = []
        self._critic_value_identity_checks: list[dict[str, Any]] = []
        self._actor_updates_in_epoch = 0
        self._rollout_actor_observations_for_rms = None
        self._rollout_critic_observations_for_rms = None

    def _normalization_pair(self) -> dict[str, Any]:
        result = {
            "actor": observation_normalizer_report(
                self.model, version=self._observation_normalization_version
            )
        }
        if self.has_asymmetric_critic:
            result["critic"] = observation_normalizer_report(
                self.asymmetric_critic_net.model,
                version=self._observation_normalization_version,
            )
        return result

    def observation_normalization_audit(self) -> dict[str, Any]:
        return {
            "schema": "frozen_rollout_observation_normalization_v1",
            "semantics": (
                "one frozen actor/critic observation transform is used for rollout "
                "collection and every PPO likelihood/value recomputation; raw rollout "
                "observations update statistics only after all optimizer passes"
            ),
            "current_version": self._observation_normalization_version,
            "current": self._normalization_pair(),
            "epochs": list(self._observation_normalization_history),
            "pre_optimizer_likelihood_identity_checks": list(
                self._likelihood_identity_checks
            ),
            "pre_optimizer_critic_value_identity_checks": list(
                self._critic_value_identity_checks
            ),
        }

    def _validate_first_update_ratio(self, ratio: torch.Tensor) -> None:
        if self._actor_updates_in_epoch != 0:
            self._actor_updates_in_epoch += 1
            return
        maximum_error = float((ratio.detach() - 1.0).abs().max().cpu())
        row = {
            "epoch": int(self.epoch_num),
            "samples": int(ratio.numel()),
            "maximum_abs_error_from_one": maximum_error,
            "tolerance": 5.0e-6,
        }
        self._likelihood_identity_checks.append(row)
        self._actor_updates_in_epoch += 1
        if maximum_error > row["tolerance"]:
            raise RuntimeError(
                "old/new likelihood ratio differs before the first optimizer "
                f"update: max |ratio-1|={maximum_error:.9g}"
            )

    def init_tensors(self) -> None:
        super().init_tensors()
        actions = self.experience_buffer.tensor_dict["actions"]
        self.experience_buffer.tensor_dict["action_lows"] = torch.empty_like(actions)
        self.experience_buffer.tensor_dict["action_highs"] = torch.empty_like(actions)
        self.update_list.extend(("action_lows", "action_highs"))
        self.tensor_list = self.update_list + ["obses", "states", "dones"]

    def _actor_outputs(self, obs, *, advance_only: bool = False) -> dict[str, torch.Tensor]:
        processed = self._preproc_obs(obs["obs"])
        self.model.eval()
        rnn_input = self.rnn_states
        input_dict = {
            "obs": self.model.norm_obs(processed, update_stats=False),
            "rnn_states": rnn_input,
        }
        with torch.no_grad():
            mu, logstd, value, states = self.model.a2c_network(input_dict)
            result = {"rnn_states": states}
            if advance_only:
                return result
            sigma = torch.exp(logstd)
            if self.has_asymmetric_critic:
                value = self.get_asymmetric_critic_value({
                    "is_train": False,
                    "states": obs["states"],
                })
            else:
                value = self.model.denorm_value(value)
            result.update(mus=mu, sigmas=sigma, values=value)
            observer = getattr(self, "_observe_actor_forward", None)
            if observer is not None:
                observer(
                    raw_observation=obs,
                    processed_observation=processed,
                    normalized_observation=input_dict["obs"],
                    rnn_input=rnn_input,
                    rnn_output=states,
                    result=result,
                )
            return result

    def advance_rnn_from_observation(self, obs) -> dict[str, torch.Tensor]:
        """Replay observation history without sampling or querying physical bounds."""
        return self._actor_outputs(obs, advance_only=True)

    def get_action_values(self, obs) -> dict:
        result = self._actor_outputs(obs)
        low, high = self.env.current_normalized_action_bounds()
        low = low.to(self.device)
        high = high.to(self.device)
        if low.shape != result["mus"].shape or high.shape != result["mus"].shape:
            raise ValueError("actor batch and state-dependent bounds differ")
        action = sample_truncated_normal(
            result["mus"], result["sigmas"], low, high,
            minimum_mass=self.distribution_spec.minimum_normalization_mass,
        )
        log_prob = truncated_normal_log_prob(
            action, result["mus"], result["sigmas"], low, high,
            minimum_mass=self.distribution_spec.minimum_normalization_mass,
        ).sum(dim=-1)
        result.update(
            actions=action,
            neglogpacs=-log_prob,
            action_lows=low,
            action_highs=high,
            deterministic_actions=deterministic_truncated_action(
                result["mus"], low, high
            ),
        )
        self._training_trace_distribution = (
            action, result["mus"], result["sigmas"]
        )
        return result

    def get_deterministic_action_values(self, obs) -> dict:
        """Evaluate the truncated-distribution mode without sampling.

        CPU acceptance uses this path so the judge neither consumes exploration
        RNG nor accidentally executes an unclipped Gaussian mean.
        """
        result = self._actor_outputs(obs)
        low, high = self.env.current_normalized_action_bounds()
        low = low.to(self.device)
        high = high.to(self.device)
        if low.shape != result["mus"].shape or high.shape != result["mus"].shape:
            raise ValueError("actor batch and state-dependent bounds differ")
        result.update(
            action_lows=low,
            action_highs=high,
            deterministic_actions=deterministic_truncated_action(
                result["mus"], low, high
            ),
        )
        return result

    def env_step(self, actions: torch.Tensor) -> tuple:
        recorder = getattr(self.env, "record_training_policy_distribution", None)
        if recorder is not None:
            distribution = self._training_trace_distribution
            if distribution is None or distribution[0] is not actions:
                raise RuntimeError("candidate env_step did not receive its latest sample")
            recorder(*distribution)
        result = super().env_step(actions)
        self._training_trace_distribution = None
        return result

    def prepare_dataset(self, batch_dict) -> None:
        actor_observations = self._preproc_obs(batch_dict["obses"])
        if not torch.is_tensor(actor_observations):
            raise TypeError("frozen normalization contract requires tensor observations")
        self._rollout_actor_observations_for_rms = (
            actor_observations.detach().clone()
        )
        if self.has_asymmetric_critic:
            critic_observations = self.asymmetric_critic_net._preproc_obs(
                batch_dict["states"]
            )
            if not torch.is_tensor(critic_observations):
                raise TypeError("critic normalization requires tensor states")
            self._rollout_critic_observations_for_rms = (
                critic_observations.detach().clone()
            )
            critic = self.asymmetric_critic_net
            if critic.is_rnn:
                raise ValueError(
                    "critic value identity gate currently requires the formal MLP critic"
                )
            critic.eval()
            with torch.no_grad():
                recomputed = critic.model({
                    "obs": critic_observations,
                    "actions": batch_dict["actions"],
                    "rnn_states": None,
                    "is_train": False,
                    "update_obs_stats": False,
                })["values"]
            stored = batch_dict["values"]
            maximum_error = float((recomputed - stored).abs().max().cpu())
            check = {
                "epoch": int(self.epoch_num),
                "samples": int(stored.shape[0]),
                "maximum_abs_error": maximum_error,
                "tolerance": 2.0e-5,
            }
            self._critic_value_identity_checks.append(check)
            if maximum_error > check["tolerance"]:
                raise RuntimeError(
                    "critic value changed before its first optimizer update: "
                    f"max error={maximum_error:.9g}"
                )
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["action_lows"] = batch_dict["action_lows"]
        self.dataset.values_dict["action_highs"] = batch_dict["action_highs"]

    def train_asymmetric_critic(self) -> float:
        """Train critic weights while keeping its input transform immutable."""
        critic = self.asymmetric_critic_net
        loss = 0.0
        for _ in range(critic.cfg.mini_epochs):
            if self.cfg.freeze_critic:
                break
            for index in range(len(critic.dataset)):
                critic.train()
                if critic.cfg.normalize_input:
                    critic.model.running_mean_std.eval()
                loss += critic.calc_gradients(critic.dataset[index]).item()
        average = loss / (critic.cfg.mini_epochs * critic.num_minibatches)
        critic.epoch_num += 1
        critic.lr, _ = critic.scheduler.update(
            critic.lr, 0, critic.epoch_num, 0, 0
        )
        critic.update_lr(critic.lr)
        critic.frame += critic.batch_size
        if critic.writer is not None:
            critic.writer.add_scalar("losses/cval_loss", average, critic.frame)
            critic.writer.add_scalar("info/cval_lr", critic.lr, critic.frame)
        return average

    def train_epoch(self):
        """Use one immutable observation transform for a whole PPO epoch."""
        self._actor_updates_in_epoch = 0
        before = self._normalization_pair()
        result = super().train_epoch()
        before_commit = self._normalization_pair()
        if before_commit != before:
            raise RuntimeError(
                "observation normalizer changed during rollout or PPO update"
            )
        observer = getattr(
            self, "_observe_observation_normalization_update", None
        )
        before_actor_state = None
        if observer is not None:
            before_actor_state = {
                name: value.detach().cpu().contiguous().clone()
                for name, value in self.model.state_dict().items()
            }
        if self._commit_observation_stats_after_epoch:
            if self._rollout_actor_observations_for_rms is None:
                raise RuntimeError("actor rollout observations were not retained")
            self.model.update_obs_stats(self._rollout_actor_observations_for_rms)
            if self.has_asymmetric_critic:
                if self._rollout_critic_observations_for_rms is None:
                    raise RuntimeError("critic rollout states were not retained")
                self.asymmetric_critic_net.model.update_obs_stats(
                    self._rollout_critic_observations_for_rms
                )
            self._observation_normalization_version += 1
        after = self._normalization_pair()
        row = {
            "epoch": int(self.epoch_num),
            "version_used_for_rollout_and_updates": int(
                self._observation_normalization_version
                - int(self._commit_observation_stats_after_epoch)
            ),
            "statistics_committed_after_updates": (
                self._commit_observation_stats_after_epoch
            ),
            "before": before,
            "after": after,
        }
        self._observation_normalization_history.append(row)
        if observer is not None:
            after_actor_state = {
                name: value.detach().cpu().contiguous().clone()
                for name, value in self.model.state_dict().items()
            }
            observer(
                report=row,
                before_actor=before_actor_state,
                after_actor=after_actor_state,
            )
        self._rollout_actor_observations_for_rms = None
        self._rollout_critic_observations_for_rms = None
        return result

    def recompute_truncated_neglogp(
        self,
        actions: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        return -truncated_normal_log_prob(
            actions, mu, sigma, low, high,
            minimum_mass=self.distribution_spec.minimum_normalization_mass,
        ).sum(dim=-1)

    def evaluate_ppo_distribution(self, input_dict) -> dict[str, torch.Tensor]:
        """Replay the frozen rollout layout and evaluate its exact likelihood.

        The curriculum restores nonzero recurrent memory after an episode reset.
        Replaying a conventional batched RNN with only ``dones`` would instead
        zero that memory.  Process the four worlds in their original temporal
        order, injecting the same reset memory, while retaining four-step BPTT
        boundaries through the stored block-start states.
        """
        actions = input_dict["actions"]
        low = input_dict["action_lows"]
        high = input_dict["action_highs"]
        obs = self._preproc_obs(input_dict["obs"])
        if not self.is_rnn:
            raise ValueError("the frozen truncated-Gaussian candidate requires the RNN actor")
        worlds = int(self.cfg.num_actors)
        horizon = int(self.cfg.horizon_length)
        sequence = int(self.cfg.seq_length)
        if (
            actions.shape[0] != worlds * horizon
            or horizon % sequence != 0
            or self.minibatch_size != self.batch_size
        ):
            raise ValueError(
                "exact rollout likelihood requires the frozen full 4-world batch"
            )
        blocks = horizon // sequence
        # Use the exact transform frozen before rollout collection. Statistics
        # are committed explicitly only after every PPO mini-epoch completes.
        # Normalizing one time step at a time under mutable statistics would
        # silently evaluate a different policy.
        normalized_obs = self.model.norm_obs(obs, update_stats=False)
        obs_by_world = normalized_obs.reshape(
            worlds, horizon, *normalized_obs.shape[1:]
        )
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
            raise ValueError("rollout and curriculum RNN state structures differ")

        mu_rows: list[list[torch.Tensor | None]] = [
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
            state_indices = torch.as_tensor(
                [world * blocks + block for world in range(worlds)],
                dtype=torch.long,
                device=self.device,
            )
            states = [
                state.index_select(1, state_indices) for state in initial_states
            ]
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
                mu_step, logstd_step, value_step, states = self.model.a2c_network({
                    "obs": obs_by_world[:, time_index],
                    "rnn_states": states,
                })
                for world in range(worlds):
                    mu_rows[world][time_index] = mu_step[world]
                    logstd_rows[world][time_index] = logstd_step[world]
                    value_rows[world][time_index] = value_step[world]
            last_states = states

        def flatten(rows):
            if any(value is None for row in rows for value in row):
                raise RuntimeError("exact rollout evaluator left an output unfilled")
            return torch.stack([
                value for row in rows for value in row  # type: ignore[misc]
            ])

        mu = flatten(mu_rows)
        logstd = flatten(logstd_rows)
        values = flatten(value_rows)
        sigma = torch.exp(logstd)
        neglogp = self.recompute_truncated_neglogp(
            actions, mu, sigma, low, high
        )
        entropy = truncated_normal_entropy(
            mu, sigma, low, high,
            minimum_mass=self.distribution_spec.minimum_normalization_mass,
        ).sum(dim=-1)
        return {
            "neglogp": neglogp,
            "entropy": entropy,
            "mu": mu,
            "sigma": sigma,
            "values": values,
            "rnn_states": last_states,
        }

    def train_actor_critic(self, input_dict):
        if not self.distribution_spec.optimizer_training_authorized:
            raise RuntimeError(
                "optimizer training is forbidden by the gate-only truncated-Gaussian profile"
            )
        value_preds = input_dict["old_values"]
        old_neglogp = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        returns = input_dict["returns"]
        current = self.evaluate_ppo_distribution(input_dict)
        ratio = torch.exp(old_neglogp - current["neglogp"])
        self._validate_first_update_ratio(ratio)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.cfg.e_clip, 1.0 + self.cfg.e_clip
        )
        actor_loss = torch.max(-advantage * ratio, -advantage * clipped_ratio).mean()
        value_clipped = value_preds + (current["values"] - value_preds).clamp(
            -self.cfg.e_clip, self.cfg.e_clip
        )
        critic_loss = torch.max(
            (current["values"] - returns).square(),
            (value_clipped - returns).square(),
        ).squeeze(dim=1).mean()
        if self.cfg.bounds_loss_coef is None:
            bounds_loss = torch.zeros((), device=self.device)
        elif self.cfg.bound_loss_type == "regularisation":
            bounds_loss = current["mu"].square().sum(dim=-1).mean()
        elif self.cfg.bound_loss_type == "bound":
            soft_bound = 1.1
            lower = torch.clamp_max(current["mu"] + soft_bound, 0.0).square()
            upper = torch.clamp_min(current["mu"] - soft_bound, 0.0).square()
            bounds_loss = (lower + upper).sum(dim=-1).mean()
        else:
            raise ValueError(f"unknown bound loss type {self.cfg.bound_loss_type}")
        entropy = current["entropy"].mean()
        loss = (
            actor_loss
            + 0.5 * critic_loss * self.cfg.critic_coef
            - entropy * self.current_entropy_coef
            + bounds_loss * (self.cfg.bounds_loss_coef or 0.0)
        )
        if self.cfg.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for parameter in self.model.parameters():
                parameter.grad = None
        self.scaler.scale(loss).backward()
        self.truncate_gradients_and_step()
        with torch.no_grad():
            # The exact likelihood ratio is used above. This sample estimate is
            # only the scheduler/logging KL diagnostic for the same actions and
            # state bounds; no untruncated Normal KL is mixed into this profile.
            kl = torch.clamp((current["neglogp"] - old_neglogp).mean(), min=0.0)
        return (
            actor_loss,
            critic_loss,
            entropy,
            kl,
            self.current_lr,
            1.0,
            current["mu"].detach(),
            current["sigma"].detach(),
            bounds_loss,
        )
