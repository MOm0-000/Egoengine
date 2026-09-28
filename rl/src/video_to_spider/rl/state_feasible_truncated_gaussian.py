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


@dataclass(frozen=True)
class LikelihoodIdentityGateSpec:
    """Numerical envelope for one frozen-policy likelihood recomputation."""

    mu_atol: float
    sigma_atol: float
    ratio_atol: float
    semantic_identity_hard_fail: float

    @classmethod
    def float32_ulp_aware_v1(cls) -> "LikelihoodIdentityGateSpec":
        epsilon = float(torch.finfo(torch.float32).eps)
        return cls(
            mu_atol=8.0 * epsilon,
            sigma_atol=8.0 * epsilon,
            ratio_atol=128.0 * epsilon,
            semantic_identity_hard_fail=1.0e-4,
        )

    def validate(self) -> None:
        epsilon = float(torch.finfo(torch.float32).eps)
        expected = (8.0 * epsilon, 8.0 * epsilon, 128.0 * epsilon, 1.0e-4)
        observed = (
            float(self.mu_atol),
            float(self.sigma_atol),
            float(self.ratio_atol),
            float(self.semantic_identity_hard_fail),
        )
        if observed != expected:
            raise ValueError(
                "likelihood identity gate must be frozen to 8/8/128 float32 "
                f"eps and semantic hard fail 1e-4, got {observed}"
            )
        if not self.ratio_atol < self.semantic_identity_hard_fail < 0.2:
            raise ValueError("likelihood identity tolerances are not strictly nested")


@dataclass(frozen=True)
class CanonicalOldPolicyGateSpec:
    """Identity contract for the differentiable canonical old policy.

    The rollout/autograd actor outputs are retained as numerical diagnostics,
    but only their likelihood ratio is a numerical gate.  The PPO denominator
    is detached from the *same* differentiable evaluation used for the first
    optimizer loss, so its ratio identity has a much stricter contract.
    """

    rollout_to_canonical_ratio_atol: float
    canonical_ratio_atol: float
    semantic_identity_hard_fail: float

    @classmethod
    def v1(cls) -> "CanonicalOldPolicyGateSpec":
        epsilon = float(torch.finfo(torch.float32).eps)
        return cls(
            rollout_to_canonical_ratio_atol=128.0 * epsilon,
            canonical_ratio_atol=epsilon,
            semantic_identity_hard_fail=1.0e-4,
        )

    def validate(self) -> None:
        epsilon = float(torch.finfo(torch.float32).eps)
        expected = (128.0 * epsilon, epsilon, 1.0e-4)
        observed = (
            float(self.rollout_to_canonical_ratio_atol),
            float(self.canonical_ratio_atol),
            float(self.semantic_identity_hard_fail),
        )
        if observed != expected:
            raise ValueError(
                "canonical old-policy gate must be frozen to 128 float32 eps "
                f"for rollout equivalence, one eps for canonical identity, and "
                f"semantic hard fail 1e-4; got {observed}"
            )
        if not (
            self.canonical_ratio_atol
            < self.rollout_to_canonical_ratio_atol
            < self.semantic_identity_hard_fail
            < 0.2
        ):
            raise ValueError("canonical old-policy tolerances are not strictly nested")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_tree_sha256(value: Any) -> str:
    """Hash tensor provenance without changing dtype, shape, or ordering."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            digest.update(np.asarray([len(item)], dtype=np.int64).tobytes())
            for child in item:
                update(child)
            return
        raise TypeError(f"unsupported tensor provenance value {type(item)!r}")

    update(value)
    return digest.hexdigest()


def _module_parameter_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def numerical_likelihood_identity_report(
    *,
    old_mu: torch.Tensor,
    current_mu: torch.Tensor,
    old_sigma: torch.Tensor,
    current_sigma: torch.Tensor,
    old_neglogp: torch.Tensor,
    current_neglogp: torch.Tensor,
    ratio: torch.Tensor,
    gate: LikelihoodIdentityGateSpec,
) -> dict[str, Any]:
    """Classify float32 rollout/autograd output differences without mutation."""

    gate.validate()
    pairs = (
        ("mu", old_mu, current_mu),
        ("sigma", old_sigma, current_sigma),
        ("logprob", old_neglogp, current_neglogp),
    )
    for name, old, current in pairs:
        if old.shape != current.shape:
            raise ValueError(f"old/current {name} shapes differ")
        if not bool(torch.isfinite(old).all() and torch.isfinite(current).all()):
            raise ValueError(f"old/current {name} contains nonfinite values")
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("likelihood ratio contains nonfinite values")
    mu_difference = (current_mu.detach() - old_mu.detach()).abs()
    sigma_difference = (current_sigma.detach() - old_sigma.detach()).abs()
    logprob_difference = (
        current_neglogp.detach() - old_neglogp.detach()
    ).abs()
    ratio_detached = ratio.detach()
    ratio_error = (ratio_detached - 1.0).abs()
    max_mu = float(mu_difference.max().cpu())
    max_sigma = float(sigma_difference.max().cpu())
    max_logprob = float(logprob_difference.max().cpu())
    max_ratio = float(ratio_error.max().cpu())
    numerical = (
        max_mu <= gate.mu_atol
        and max_sigma <= gate.sigma_atol
        and max_ratio <= gate.ratio_atol
    )
    return {
        "max_abs_mu_difference": max_mu,
        "max_abs_sigma_difference": max_sigma,
        "max_abs_logprob_difference": max_logprob,
        "number_of_different_mu_elements": int(
            torch.count_nonzero(current_mu.detach() != old_mu.detach()).cpu()
        ),
        "number_of_different_sigma_elements": int(
            torch.count_nonzero(current_sigma.detach() != old_sigma.detach()).cpu()
        ),
        "samples_with_nonzero_logprob_difference": int(
            torch.count_nonzero(
                current_neglogp.detach() != old_neglogp.detach()
            ).cpu()
        ),
        "ratio_min": float(ratio_detached.min().cpu()),
        "ratio_max": float(ratio_detached.max().cpu()),
        "ratio_max_abs_error_from_one": max_ratio,
        "maximum_abs_error_from_one": max_ratio,
        "mu_tolerance": float(gate.mu_atol),
        "sigma_tolerance": float(gate.sigma_atol),
        "ratio_tolerance": float(gate.ratio_atol),
        "semantic_identity_hard_fail_threshold": float(
            gate.semantic_identity_hard_fail
        ),
        "policy_output_numerical_equivalence_passed": bool(numerical),
        "semantic_identity_hard_fail": bool(
            max_ratio > gate.semantic_identity_hard_fail
        ),
    }


def canonical_old_policy_identity_report(
    *,
    rollout_mu: torch.Tensor,
    canonical_mu: torch.Tensor,
    rollout_sigma: torch.Tensor,
    canonical_sigma: torch.Tensor,
    rollout_neglogp: torch.Tensor,
    canonical_neglogp: torch.Tensor,
    rollout_to_canonical_ratio: torch.Tensor,
    canonical_optimizer_ratio: torch.Tensor,
    semantic_identity: dict[str, bool],
    gate: CanonicalOldPolicyGateSpec,
) -> dict[str, Any]:
    """Audit rollout policy equivalence and canonical PPO identity.

    Mu and sigma differences intentionally remain diagnostic only.  The
    rollout/autograd boundary is authorized by semantic provenance plus the
    action likelihood, while the PPO denominator itself is produced by
    detaching the exact differentiable evaluation used by the loss.
    """

    gate.validate()
    pairs = (
        ("mu", rollout_mu, canonical_mu),
        ("sigma", rollout_sigma, canonical_sigma),
        ("neglogp", rollout_neglogp, canonical_neglogp),
    )
    for name, rollout, canonical in pairs:
        if rollout.shape != canonical.shape:
            raise ValueError(f"rollout/canonical {name} shapes differ")
        if not bool(torch.isfinite(rollout).all() and torch.isfinite(canonical).all()):
            raise ValueError(f"rollout/canonical {name} contains nonfinite values")
    for name, ratio in (
        ("rollout_to_canonical", rollout_to_canonical_ratio),
        ("canonical_optimizer", canonical_optimizer_ratio),
    ):
        if not bool(torch.isfinite(ratio).all()):
            raise ValueError(f"{name} ratio contains nonfinite values")

    mu_difference = (canonical_mu.detach() - rollout_mu.detach()).abs()
    sigma_difference = (canonical_sigma.detach() - rollout_sigma.detach()).abs()
    logprob_difference = (
        canonical_neglogp.detach() - rollout_neglogp.detach()
    ).abs()
    external_ratio = rollout_to_canonical_ratio.detach()
    canonical_ratio = canonical_optimizer_ratio.detach()
    external_error = (external_ratio - 1.0).abs()
    canonical_error = (canonical_ratio - 1.0).abs()
    max_external_error = float(external_error.max().cpu())
    max_canonical_error = float(canonical_error.max().cpu())
    semantic_passed = bool(semantic_identity) and all(semantic_identity.values())
    rollout_likelihood_passed = (
        max_external_error <= gate.rollout_to_canonical_ratio_atol
    )
    canonical_ratio_exact = bool(
        torch.equal(canonical_ratio, torch.ones_like(canonical_ratio))
    )
    canonical_ratio_passed = (
        max_canonical_error <= gate.canonical_ratio_atol
    )
    semantic_hard_fail = (
        max_external_error > gate.semantic_identity_hard_fail
    )
    return {
        "semantic_state_identity": dict(semantic_identity),
        "semantic_state_identity_passed": semantic_passed,
        "rollout_vs_canonical": {
            "max_abs_mu_difference": float(mu_difference.max().cpu()),
            "max_abs_sigma_difference": float(sigma_difference.max().cpu()),
            "max_abs_logprob_difference": float(logprob_difference.max().cpu()),
            "mu_element_difference_count": int(
                torch.count_nonzero(
                    canonical_mu.detach() != rollout_mu.detach()
                ).cpu()
            ),
            "sigma_element_difference_count": int(
                torch.count_nonzero(
                    canonical_sigma.detach() != rollout_sigma.detach()
                ).cpu()
            ),
            "ratio_min": float(external_ratio.min().cpu()),
            "ratio_max": float(external_ratio.max().cpu()),
            "max_abs_ratio_minus_one": max_external_error,
            "ratio_tolerance": float(
                gate.rollout_to_canonical_ratio_atol
            ),
            "likelihood_equivalence_passed": rollout_likelihood_passed,
            "mu_sigma_are_diagnostic_only": True,
        },
        "canonical_optimizer_identity": {
            "ratio_min": float(canonical_ratio.min().cpu()),
            "ratio_max": float(canonical_ratio.max().cpu()),
            "max_abs_ratio_minus_one": max_canonical_error,
            "ratio_tolerance": float(gate.canonical_ratio_atol),
            "ratio_bitwise_equal_one": canonical_ratio_exact,
            "passed": canonical_ratio_passed,
        },
        "semantic_identity_hard_fail_threshold": float(
            gate.semantic_identity_hard_fail
        ),
        "semantic_identity_hard_fail": semantic_hard_fail,
        "passed": bool(
            semantic_passed
            and rollout_likelihood_passed
            and canonical_ratio_passed
        ),
    }


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
    """Derive bounds that preserve, but never worsen, a baseline violation."""
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
    formal = reference.astype(np.float32)
    formal64 = formal.astype(np.float64)
    below_f32 = np.where(limited, np.maximum(ranges[:, 0] - formal64, 0.0), 0.0)
    above_f32 = np.where(limited, np.maximum(formal64 - ranges[:, 1], 0.0), 0.0)
    allowed_lower = np.where(below_f32 > 0.0, formal64, ranges[:, 0])
    allowed_upper = np.where(above_f32 > 0.0, formal64, ranges[:, 1])
    scale_f32 = np.float32(residual_scale)
    low = np.full(reference.shape, -1.0, np.float32)
    high = np.full(reference.shape, 1.0, np.float32)
    low[:, limited] = np.maximum(
        np.float32(-1.0),
        ((allowed_lower[:, limited] - formal64[:, limited]) / residual_scale).astype(
            np.float32
        ),
    )
    high[:, limited] = np.minimum(
        np.float32(1.0),
        ((allowed_upper[:, limited] - formal64[:, limited]) / residual_scale).astype(
            np.float32
        ),
    )
    inward_adjustment_count = 0
    for _ in range(64):
        requested_low = (formal + scale_f32 * low).astype(np.float64)
        requested_high = (formal + scale_f32 * high).astype(np.float64)
        low_outside = limited & (requested_low < allowed_lower)
        high_outside = limited & (requested_high > allowed_upper)
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
    if np.any((low > 0.0) | (high < 0.0)):
        raise RuntimeError("zero residual is outside baseline-preserving support")
    return low.astype(np.float64), high.astype(np.float64), {
        "reference_snap_tolerance": reference_snap_tolerance,
        "execution_reference": "formal_reference_ctrl_unmodified",
        "bounds_semantics": "do_not_worsen_baseline_ctrlrange_violation",
        "baseline_violating_component_count": int((violation > 0.0).sum()),
        "snapped_component_count": 0,
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
        likelihood_identity_gate: LikelihoodIdentityGateSpec | None = None,
        canonical_old_policy_gate: CanonicalOldPolicyGateSpec | None = None,
        commit_observation_stats_after_epoch: bool = True,
        **kwargs,
    ):
        self.distribution_spec = distribution_spec
        self.likelihood_identity_gate = (
            LikelihoodIdentityGateSpec.float32_ulp_aware_v1()
            if likelihood_identity_gate is None
            else likelihood_identity_gate
        )
        self.likelihood_identity_gate.validate()
        self.canonical_old_policy_gate = canonical_old_policy_gate
        if self.canonical_old_policy_gate is not None:
            self.canonical_old_policy_gate.validate()
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
        self._rollout_actor_parameter_hash: str | None = None
        self._rollout_actor_normalizer: dict[str, Any] | None = None
        self._rollout_normalization_version: int | None = None
        self._rollout_semantic_hashes: dict[str, str] | None = None
        self._canonical_old_policy_for_epoch: dict[str, torch.Tensor] | None = None
        self._canonical_old_policy_for_latest_update: (
            dict[str, torch.Tensor] | None
        ) = None

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
            "schema": (
                "canonical_old_policy_observation_normalization_v1"
                if self.canonical_old_policy_gate is not None
                else "frozen_rollout_observation_normalization_v1"
            ),
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

    def _validate_first_update_identity(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        current: dict[str, torch.Tensor],
        ratio: torch.Tensor,
        pre_recompute_actor_parameter_hash: str,
        pre_recompute_actor_normalizer: dict[str, Any],
        pre_recompute_normalization_version: int,
    ) -> None:
        if self._actor_updates_in_epoch != 0:
            self._actor_updates_in_epoch += 1
            return
        if (
            self._rollout_actor_parameter_hash is None
            or self._rollout_actor_normalizer is None
            or self._rollout_normalization_version is None
        ):
            raise RuntimeError("rollout identity was not captured before PPO recompute")
        actor_parameter_hash_equal = (
            pre_recompute_actor_parameter_hash
            == self._rollout_actor_parameter_hash
        )
        actor_rms_hash_equal = (
            pre_recompute_actor_normalizer == self._rollout_actor_normalizer
        )
        normalization_version_equal = (
            int(pre_recompute_normalization_version)
            == int(self._rollout_normalization_version)
        )
        numerical = numerical_likelihood_identity_report(
            old_mu=input_dict["mu"],
            current_mu=current["mu"],
            old_sigma=input_dict["sigma"],
            current_sigma=current["sigma"],
            old_neglogp=input_dict["old_logp_actions"],
            current_neglogp=current["neglogp"],
            ratio=ratio,
            gate=self.likelihood_identity_gate,
        )
        passed = (
            actor_parameter_hash_equal
            and actor_rms_hash_equal
            and normalization_version_equal
            and numerical["policy_output_numerical_equivalence_passed"]
        )
        if numerical["semantic_identity_hard_fail"]:
            classification = "semantic_implementation_identity_defect"
        elif not (
            actor_parameter_hash_equal
            and actor_rms_hash_equal
            and normalization_version_equal
        ):
            classification = "frozen_policy_state_identity_defect"
        elif not numerical["policy_output_numerical_equivalence_passed"]:
            classification = "float32_numerical_envelope_exceeded"
        else:
            classification = "float32_numerical_equivalence_pass"
        row = {
            "epoch": int(self.epoch_num),
            "samples": int(ratio.numel()),
            "actor_parameter_hash_equal": actor_parameter_hash_equal,
            "actor_RMS_hash_equal": actor_rms_hash_equal,
            "normalization_version_equal": normalization_version_equal,
            **numerical,
            "tolerance": float(self.likelihood_identity_gate.ratio_atol),
            "numerical_identity_passed": bool(passed),
            "classification": classification,
        }
        self._likelihood_identity_checks.append(row)
        self._actor_updates_in_epoch += 1
        if not passed:
            raise RuntimeError(
                "pre-optimizer likelihood identity gate failed: "
                f"classification={classification}, "
                "actor_hash_equal="
                f"{actor_parameter_hash_equal}, RMS_hash_equal={actor_rms_hash_equal}, "
                f"normalization_version_equal={normalization_version_equal}, "
                "max_mu="
                f"{numerical['max_abs_mu_difference']:.9g}, max_sigma="
                f"{numerical['max_abs_sigma_difference']:.9g}, max_ratio="
                f"{numerical['ratio_max_abs_error_from_one']:.9g}"
            )

    def _validate_canonical_first_update_identity(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        canonical: dict[str, torch.Tensor],
        rollout_to_canonical_ratio: torch.Tensor,
        canonical_optimizer_ratio: torch.Tensor,
        pre_recompute_actor_parameter_hash: str,
        pre_recompute_actor_normalizer: dict[str, Any],
        pre_recompute_normalization_version: int,
    ) -> None:
        if self.canonical_old_policy_gate is None:
            raise RuntimeError("canonical old-policy gate was not configured")
        if self._actor_updates_in_epoch != 0:
            self._actor_updates_in_epoch += 1
            return
        if (
            self._rollout_actor_parameter_hash is None
            or self._rollout_actor_normalizer is None
            or self._rollout_normalization_version is None
            or self._rollout_semantic_hashes is None
        ):
            raise RuntimeError(
                "rollout identity was not captured before canonicalization"
            )
        semantic = {
            "actor_parameter_hash_equal": (
                pre_recompute_actor_parameter_hash
                == self._rollout_actor_parameter_hash
            ),
            "actor_RMS_hash_equal": (
                pre_recompute_actor_normalizer
                == self._rollout_actor_normalizer
            ),
            "normalization_version_equal": (
                int(pre_recompute_normalization_version)
                == int(self._rollout_normalization_version)
            ),
            "observations_equal": (
                _tensor_tree_sha256(input_dict["obs"])
                == self._rollout_semantic_hashes["obs"]
            ),
            "dones_equal": (
                _tensor_tree_sha256(input_dict["dones"])
                == self._rollout_semantic_hashes["dones"]
            ),
            "actions_equal": (
                _tensor_tree_sha256(input_dict["actions"])
                == self._rollout_semantic_hashes["actions"]
            ),
            "action_low_equal": (
                _tensor_tree_sha256(input_dict["action_lows"])
                == self._rollout_semantic_hashes["action_lows"]
            ),
            "action_high_equal": (
                _tensor_tree_sha256(input_dict["action_highs"])
                == self._rollout_semantic_hashes["action_highs"]
            ),
            "RNN_start_states_equal": (
                _tensor_tree_sha256(input_dict["rnn_states"])
                == self._rollout_semantic_hashes["rnn_states"]
            ),
        }
        report = canonical_old_policy_identity_report(
            rollout_mu=input_dict["mu"],
            canonical_mu=canonical["mu"],
            rollout_sigma=input_dict["sigma"],
            canonical_sigma=canonical["sigma"],
            rollout_neglogp=input_dict["old_logp_actions"],
            canonical_neglogp=canonical["neglogp"],
            rollout_to_canonical_ratio=rollout_to_canonical_ratio,
            canonical_optimizer_ratio=canonical_optimizer_ratio,
            semantic_identity=semantic,
            gate=self.canonical_old_policy_gate,
        )
        if report["semantic_identity_hard_fail"]:
            classification = "semantic_implementation_identity_defect"
        elif not report["semantic_state_identity_passed"]:
            classification = "frozen_policy_state_identity_defect"
        elif not report["rollout_vs_canonical"][
            "likelihood_equivalence_passed"
        ]:
            classification = "rollout_canonical_likelihood_mismatch"
        elif not report["canonical_optimizer_identity"]["passed"]:
            classification = "canonical_ratio_implementation_defect"
        else:
            classification = "canonical_old_policy_identity_pass"
        row = {
            "epoch": int(self.epoch_num),
            "samples": int(canonical_optimizer_ratio.numel()),
            **report,
            "classification": classification,
            # Compatibility fields used by bounded-size benchmark summaries.
            "maximum_abs_error_from_one": report[
                "canonical_optimizer_identity"
            ]["max_abs_ratio_minus_one"],
            "ratio_max_abs_error_from_one": report[
                "rollout_vs_canonical"
            ]["max_abs_ratio_minus_one"],
            "numerical_identity_passed": bool(report["passed"]),
        }
        self._likelihood_identity_checks.append(row)
        self._actor_updates_in_epoch += 1
        if not report["passed"]:
            raise RuntimeError(
                "canonical old-policy gate failed: "
                f"classification={classification}, semantic={semantic}, "
                "rollout_ratio_error="
                f"{report['rollout_vs_canonical']['max_abs_ratio_minus_one']:.9g}, "
                "canonical_ratio_error="
                f"{report['canonical_optimizer_identity']['max_abs_ratio_minus_one']:.9g}"
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
        self._rollout_actor_parameter_hash = _module_parameter_sha256(self.model)
        self._rollout_actor_normalizer = observation_normalizer_report(
            self.model, version=self._observation_normalization_version
        )
        self._rollout_normalization_version = int(
            self._observation_normalization_version
        )
        if self.canonical_old_policy_gate is not None:
            self._rollout_semantic_hashes = {
                "obs": _tensor_tree_sha256(batch_dict["obses"]),
                "dones": _tensor_tree_sha256(batch_dict["dones"]),
                "actions": _tensor_tree_sha256(batch_dict["actions"]),
                "action_lows": _tensor_tree_sha256(batch_dict["action_lows"]),
                "action_highs": _tensor_tree_sha256(batch_dict["action_highs"]),
                "rnn_states": _tensor_tree_sha256(batch_dict["rnn_states"]),
            }
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
        self._rollout_actor_parameter_hash = None
        self._rollout_actor_normalizer = None
        self._rollout_normalization_version = None
        self._rollout_semantic_hashes = None
        self._canonical_old_policy_for_epoch = None
        self._canonical_old_policy_for_latest_update = None
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
            "frozen_before_commit": before_commit,
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
        self._rollout_actor_parameter_hash = None
        self._rollout_actor_normalizer = None
        self._rollout_normalization_version = None
        self._rollout_semantic_hashes = None
        self._canonical_old_policy_for_epoch = None
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
        # Rollout collection uses eval mode, so likelihood recomputation must
        # use the same module mode. Eval mode does not disable autograd; the PPO
        # surrogate still differentiates through this forward. Restore the
        # caller's mode before returning.
        model_was_training = self.model.training
        self.model.eval()
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
        result = {
            "neglogp": neglogp,
            "entropy": entropy,
            "mu": mu,
            "sigma": sigma,
            "values": values,
            "rnn_states": last_states,
        }
        if model_was_training:
            self.model.train()
        return result

    def train_actor_critic(self, input_dict):
        if not self.distribution_spec.optimizer_training_authorized:
            raise RuntimeError(
                "optimizer training is forbidden by the gate-only truncated-Gaussian profile"
            )
        value_preds = input_dict["old_values"]
        rollout_old_neglogp = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        returns = input_dict["returns"]
        pre_recompute_actor_parameter_hash = _module_parameter_sha256(self.model)
        pre_recompute_actor_normalizer = observation_normalizer_report(
            self.model, version=self._observation_normalization_version
        )
        pre_recompute_normalization_version = int(
            self._observation_normalization_version
        )
        current = self.evaluate_ppo_distribution(input_dict)
        if self.canonical_old_policy_gate is None:
            old_neglogp = rollout_old_neglogp
            ratio = torch.exp(old_neglogp - current["neglogp"])
            self._validate_first_update_identity(
                input_dict=input_dict,
                current=current,
                ratio=ratio,
                pre_recompute_actor_parameter_hash=pre_recompute_actor_parameter_hash,
                pre_recompute_actor_normalizer=pre_recompute_actor_normalizer,
                pre_recompute_normalization_version=pre_recompute_normalization_version,
            )
        else:
            if self._actor_updates_in_epoch == 0:
                # This one differentiable evaluation is both the canonical old
                # policy and the current side of the first PPO loss.  Detaching
                # the denominator preserves an exactly identical value while
                # retaining the policy gradient through current["neglogp"].
                self._canonical_old_policy_for_epoch = {
                    name: current[name].detach().clone()
                    for name in ("neglogp", "mu", "sigma")
                }
            if self._canonical_old_policy_for_epoch is None:
                raise RuntimeError("canonical old policy was not initialized")
            canonical_old = self._canonical_old_policy_for_epoch
            old_neglogp = canonical_old["neglogp"]
            rollout_to_canonical_ratio = torch.exp(
                rollout_old_neglogp - current["neglogp"]
            )
            ratio = torch.exp(old_neglogp - current["neglogp"])
            if self._actor_updates_in_epoch == 0:
                self._validate_canonical_first_update_identity(
                    input_dict=input_dict,
                    canonical=current,
                    rollout_to_canonical_ratio=rollout_to_canonical_ratio,
                    canonical_optimizer_ratio=ratio,
                    pre_recompute_actor_parameter_hash=(
                        pre_recompute_actor_parameter_hash
                    ),
                    pre_recompute_actor_normalizer=(
                        pre_recompute_actor_normalizer
                    ),
                    pre_recompute_normalization_version=(
                        pre_recompute_normalization_version
                    ),
                )
            else:
                self._actor_updates_in_epoch += 1
            self._canonical_old_policy_for_latest_update = {
                name: value.detach().clone()
                for name, value in canonical_old.items()
            }
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
