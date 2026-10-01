"""The one active bounded, state-truncated policy distribution.

This module is deliberately independent of every historical Candidate agent.
Rollout sampling, deterministic validation and PPO likelihood recomputation all
call the same functions below.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class DistributionSpec:
    residual_scale: float = 0.05
    reference_snap_tolerance: float = 2.0e-7
    minimum_normalization_mass: float = 1.0e-12

    def validate(self) -> None:
        if self.residual_scale != 0.05:
            raise ValueError("R1 supports only the frozen 0.05 residual scale")
        if self.reference_snap_tolerance != 2.0e-7:
            raise ValueError("R1 supports only the audited reference snap tolerance")
        if self.minimum_normalization_mass <= 0.0:
            raise ValueError("normalization-mass floor must be positive")


def support_anchored_bounded_mean(
    raw_location: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Map latent locations into support while keeping zero exactly anchored."""
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
    bounded = torch.where(use_positive, positive_width, negative_width) * torch.tanh(
        raw_location
    )
    tolerance = 2.0 * torch.finfo(bounded.dtype).eps
    if not bool(torch.isfinite(bounded).all()):
        raise RuntimeError("support-anchored bounded mean became nonfinite")
    if bool(((bounded < low - tolerance) | (bounded > high + tolerance)).any()):
        raise RuntimeError("support-anchored bounded mean escaped support")
    return bounded


def _standard_normal_pdf(value: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)


def _terms(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    minimum_mass: float,
) -> tuple[torch.Tensor, ...]:
    if not (mu.shape == sigma.shape == low.shape == high.shape):
        raise ValueError("mu, sigma and bounds must have identical shapes")
    if not all(torch.isfinite(value).all() for value in (mu, sigma, low, high)):
        raise ValueError("truncated-Gaussian inputs must be finite")
    if bool((sigma <= 0.0).any().item()) or bool((low >= high).any().item()):
        raise ValueError("sigma must be positive and every interval nonempty")
    mu64, sigma64, low64, high64 = (
        value.to(torch.float64) for value in (mu, sigma, low, high)
    )
    alpha = (low64 - mu64) / sigma64
    beta = (high64 - mu64) / sigma64
    cdf_low = torch.special.ndtr(alpha)
    mass = torch.special.ndtr(beta) - cdf_low
    if bool((mass < minimum_mass).any().item()):
        raise ValueError(
            f"truncated-Gaussian normalization mass {float(mass.min()):.9g} "
            f"is below {minimum_mass:.9g}"
        )
    return mu64, sigma64, low64, high64, alpha, beta, cdf_low, mass


def log_prob(
    action: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    minimum_mass: float = 1.0e-12,
) -> torch.Tensor:
    """Per-coordinate log probability for the independent truncated Normal."""
    mu64, sigma64, low64, high64, _, _, _, mass = _terms(
        mu, sigma, low, high, minimum_mass=minimum_mass
    )
    action64 = action.to(torch.float64)
    tolerance = 8.0 * torch.finfo(action.dtype).eps
    if bool(((action64 < low64 - tolerance) | (action64 > high64 + tolerance)).any()):
        raise ValueError("action lies outside its truncated-Gaussian support")
    z = (action64 - mu64) / sigma64
    value = (
        -0.5 * z.square()
        - torch.log(sigma64)
        - 0.5 * math.log(2.0 * math.pi)
        - torch.log(mass)
    )
    return value.to(mu.dtype)


def entropy(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    minimum_mass: float = 1.0e-12,
) -> torch.Tensor:
    _, sigma64, _, _, alpha, beta, _, mass = _terms(
        mu, sigma, low, high, minimum_mass=minimum_mass
    )
    correction = (
        alpha * _standard_normal_pdf(alpha) - beta * _standard_normal_pdf(beta)
    ) / (2.0 * mass)
    return (
        torch.log(sigma64)
        + 0.5 * math.log(2.0 * math.pi * math.e)
        + torch.log(mass)
        + correction
    ).to(mu.dtype)


def sample(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    minimum_mass: float = 1.0e-12,
) -> torch.Tensor:
    mu64, sigma64, low64, high64, _, _, cdf_low, mass = _terms(
        mu, sigma, low, high, minimum_mass=minimum_mass
    )
    uniform = torch.rand(mu.shape, dtype=torch.float64, device=mu.device)
    probability = cdf_low + uniform * mass
    epsilon = torch.finfo(torch.float64).eps
    result = mu64 + sigma64 * torch.special.ndtri(
        probability.clamp(epsilon, 1.0 - epsilon)
    )
    return torch.maximum(torch.minimum(result, high64), low64).to(mu.dtype)


def deterministic(mu: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    if not (mu.shape == low.shape == high.shape):
        raise ValueError("mu and bounds must have identical shapes")
    return torch.maximum(torch.minimum(mu, high), low)


def exact_kl(
    old_mu: torch.Tensor,
    old_sigma: torch.Tensor,
    new_mu: torch.Tensor,
    new_sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Exact factorized KL(old || new), reduced over action dimensions."""
    mu0, sigma0, mu1, sigma1, low64, high64 = [
        value.detach().double()
        for value in (old_mu, old_sigma, new_mu, new_sigma, low, high)
    ]
    a0, b0 = (low64 - mu0) / sigma0, (high64 - mu0) / sigma0
    a1, b1 = (low64 - mu1) / sigma1, (high64 - mu1) / sigma1
    z0 = (torch.special.ndtr(b0) - torch.special.ndtr(a0)).clamp_min(1.0e-15)
    z1 = (torch.special.ndtr(b1) - torch.special.ndtr(a1)).clamp_min(1.0e-15)
    phi_a, phi_b = _standard_normal_pdf(a0), _standard_normal_pdf(b0)
    mean_z = (phi_a - phi_b) / z0
    second_z = 1.0 + (a0 * phi_a - b0 * phi_b) / z0
    mean_x = mu0 + sigma0 * mean_z
    second_x = mu0.square() + 2.0 * mu0 * sigma0 * mean_z + sigma0.square() * second_z
    new_quadratic = (second_x - 2.0 * mu1 * mean_x + mu1.square()) / sigma1.square()
    return (
        torch.log(sigma1 * z1)
        - torch.log(sigma0 * z0)
        + 0.5 * (new_quadratic - second_z)
    ).sum(dim=-1)


def normalized_action_bounds(
    reference: np.ndarray,
    ctrl_limited: np.ndarray,
    ctrl_range: np.ndarray,
    spec: DistributionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return state-dependent normalized residual support, failing closed."""
    spec.validate()
    ref = np.asarray(reference, dtype=np.float64)
    limited = np.asarray(ctrl_limited, dtype=bool)
    ranges = np.asarray(ctrl_range, dtype=np.float64)
    if ranges.shape != (ref.size, 2) or limited.shape != (ref.size,):
        raise ValueError("actuator metadata shape mismatch")
    low = np.full(ref.shape, -1.0, dtype=np.float64)
    high = np.full(ref.shape, 1.0, dtype=np.float64)
    for index in np.flatnonzero(limited):
        lower, upper = ranges[index]
        value = ref[index]
        if value < lower and lower - value <= spec.reference_snap_tolerance:
            value = lower
        if value > upper and value - upper <= spec.reference_snap_tolerance:
            value = upper
        if value < lower - spec.reference_snap_tolerance or value > upper + spec.reference_snap_tolerance:
            raise ValueError("reference command lies outside ctrlrange beyond snap tolerance")
        low[index] = max(-1.0, (lower - value) / spec.residual_scale)
        high[index] = min(1.0, (upper - value) / spec.residual_scale)
    low = low.astype(np.float32).astype(np.float64)
    high = high.astype(np.float32).astype(np.float64)
    if np.any(low >= high):
        raise ValueError("state-feasible action interval is empty")
    return low, high
