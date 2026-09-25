#!/usr/bin/env python3
"""Offline 1,280-state gate for the state-feasible truncated Gaussian."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/human2sim2robot")]

from video_to_spider.rl.state_feasible_truncated_gaussian import (
    deterministic_truncated_action,
    load_truncated_gaussian_profile,
    normalized_action_bounds_numpy,
    sample_truncated_normal,
    truncated_normal_entropy,
    truncated_normal_log_prob,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v4-run", type=Path,
        default=ROOT / "runs/taco_pour_policy_distribution_attribution_v1/report.json",
    )
    parser.add_argument(
        "--profile", type=Path,
        default=ROOT / "configs/taco_pour_state_feasible_truncated_gaussian_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_truncated_gaussian_offline_gate_v1/report.json",
    )
    parser.add_argument("--samples-per-state", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.samples_per_state != 8:
        raise ValueError("the frozen offline gate requires eight samples per state")
    args.output.parent.mkdir(parents=True)

    spec, profile = load_truncated_gaussian_profile(args.profile)
    run = json.loads(args.v4_run.read_text())
    traces = run["diagnostic"]["training_runs"]
    if len(traces) != 1:
        raise ValueError("v4 attribution run must contain one training run")
    trace = traces[0]["training_visitation"]
    if trace.get("schema") != "taco_ppo_training_visitation_v4":
        raise ValueError("offline gate requires the v4 trace")
    fields = (
        "actor_mu", "actor_sigma", "reference_ctrl",
        "source_endpoint", "outcome_endpoint",
    )
    parts = {name: [] for name in fields}
    trace_dir = args.v4_run.parent / "ppo_diagnostic_chunk_20/training_visitation"
    artifact_hashes = []
    for epoch in trace["epochs"]:
        path = trace_dir / Path(epoch["visits"]["path"]).name
        digest = _sha256(path)
        if digest != epoch["visits"]["sha256"]:
            raise ValueError("v4 trace artifact hash mismatch")
        artifact_hashes.append(digest)
        with np.load(path, allow_pickle=False) as raw:
            for name in fields:
                parts[name].append(np.asarray(raw[name]))
    data = {name: np.concatenate(rows, axis=0) for name, rows in parts.items()}
    if data["actor_mu"].shape != (1280, 36):
        raise ValueError("offline gate requires exactly 1,280 36-D states")

    contract = trace["action_contract"]["ctrlrange"]
    limited = np.asarray(contract["ctrllimited"], bool)
    ranges = np.asarray(contract["ctrlrange"], np.float64)
    low, high, bounds_audit = normalized_action_bounds_numpy(
        data["reference_ctrl"],
        ctrllimited=limited,
        ctrlrange=ranges,
        residual_scale=spec.residual_scale,
        reference_snap_tolerance=spec.reference_snap_tolerance,
    )
    lower_f32 = ranges[:, 0].astype(np.float32)
    upper_f32 = ranges[:, 1].astype(np.float32)
    lower_f32 = np.where(
        lower_f32.astype(np.float64) < ranges[:, 0],
        np.nextafter(lower_f32, np.float32(np.inf)), lower_f32,
    )
    upper_f32 = np.where(
        upper_f32.astype(np.float64) > ranges[:, 1],
        np.nextafter(upper_f32, np.float32(-np.inf)), upper_f32,
    )
    snapped_reference = np.asarray(data["reference_ctrl"], np.float32).copy()
    snapped_reference[:, limited] = np.clip(
        snapped_reference[:, limited], lower_f32[limited], upper_f32[limited]
    )
    mu = torch.tensor(data["actor_mu"], dtype=torch.float32)
    sigma = torch.tensor(data["actor_sigma"], dtype=torch.float32)
    low_t = torch.tensor(low, dtype=torch.float32)
    high_t = torch.tensor(high, dtype=torch.float32)

    sampled_outside = 0
    official_clamp_changes = 0
    theoretical_ctrlrange_losses = 0
    finite_logprob = True
    sampled_action_min = np.inf
    sampled_action_max = -np.inf
    for draw in range(args.samples_per_state):
        torch.manual_seed(draw)
        action = sample_truncated_normal(
            mu, sigma, low_t, high_t,
            minimum_mass=spec.minimum_normalization_mass,
        )
        sampled_outside += int(((action < low_t) | (action > high_t)).sum().item())
        official_clamp_changes += int((torch.clamp(action, -1.0, 1.0) != action).sum().item())
        logprob = truncated_normal_log_prob(
            action, mu, sigma, low_t, high_t,
            minimum_mass=spec.minimum_normalization_mass,
        )
        finite_logprob = finite_logprob and bool(torch.isfinite(logprob).all().item())
        action_np = action.numpy().astype(np.float32)
        requested = (
            snapped_reference + np.float32(spec.residual_scale) * action_np
        ).astype(np.float32)
        after = requested.astype(np.float64)
        after[:, limited] = np.clip(
            after[:, limited], ranges[limited, 0], ranges[limited, 1]
        )
        theoretical_ctrlrange_losses += int(
            (after != requested.astype(np.float64)).sum()
        )
        sampled_action_min = min(sampled_action_min, float(action.min().item()))
        sampled_action_max = max(sampled_action_max, float(action.max().item()))

    tail_rows = (
        (data["outcome_endpoint"] >= 46) & (data["outcome_endpoint"] <= 53)
    )
    tail_mu = mu[tail_rows].detach().clone().requires_grad_(True)
    tail_sigma = sigma[tail_rows].detach().clone().requires_grad_(True)
    tail_low = low_t[tail_rows]
    tail_high = high_t[tail_rows]
    torch.manual_seed(101)
    tail_action = sample_truncated_normal(
        tail_mu.detach(), tail_sigma.detach(), tail_low, tail_high,
        minimum_mass=spec.minimum_normalization_mass,
    )
    old_logprob = truncated_normal_log_prob(
        tail_action, tail_mu, tail_sigma, tail_low, tail_high,
        minimum_mass=spec.minimum_normalization_mass,
    ).sum(dim=-1)
    recomputed_logprob = truncated_normal_log_prob(
        tail_action, tail_mu, tail_sigma, tail_low, tail_high,
        minimum_mass=spec.minimum_normalization_mass,
    ).sum(dim=-1)
    ratio = torch.exp(recomputed_logprob - old_logprob)
    entropy = truncated_normal_entropy(
        tail_mu, tail_sigma, tail_low, tail_high,
        minimum_mass=spec.minimum_normalization_mass,
    ).sum(dim=-1)
    loss = -(recomputed_logprob.mean() + 0.01 * entropy.mean())
    loss.backward()

    alpha = (low - data["actor_mu"]) / data["actor_sigma"]
    beta = (high - data["actor_mu"]) / data["actor_sigma"]
    mass = ndtr(beta) - ndtr(alpha)
    deterministic = deterministic_truncated_action(mu, low_t, high_t)
    one_sided_zero = (
        ((low_t == 0.0) & (mu < 0.0) & (deterministic == 0.0))
        | ((high_t == 0.0) & (mu > 0.0) & (deterministic == 0.0))
    )
    one_sided_candidates = (
        ((low_t == 0.0) & (mu < 0.0))
        | ((high_t == 0.0) & (mu > 0.0))
    )

    checks = {
        "all_1280_intervals_nonempty": bool(np.all(low < high)),
        "normalization_mass_above_fail_closed_minimum": bool(
            np.all(mass >= spec.minimum_normalization_mass)
        ),
        "all_eight_samples_inside_state_bounds": sampled_outside == 0,
        "official_minus1_plus1_clamp_noop": official_clamp_changes == 0,
        "theoretical_ctrlrange_loss_exactly_zero": theoretical_ctrlrange_losses == 0,
        "tail_logprob_finite": finite_logprob and bool(torch.isfinite(old_logprob).all()),
        "tail_entropy_finite": bool(torch.isfinite(entropy).all()),
        "tail_mu_gradient_finite": bool(torch.isfinite(tail_mu.grad).all()),
        "tail_sigma_gradient_finite": bool(torch.isfinite(tail_sigma.grad).all()),
        "same_policy_logprob_bitwise_equal": bool(torch.equal(old_logprob, recomputed_logprob)),
        "same_policy_ratio_bitwise_one": bool(torch.equal(ratio, torch.ones_like(ratio))),
        "one_sided_deterministic_zero_preserved": bool(
            one_sided_candidates.any() and torch.equal(one_sided_zero, one_sided_candidates)
        ),
    }
    passed = all(checks.values())
    report = {
        "schema": "taco_pour_truncated_gaussian_offline_gate_v1",
        "status": "passed" if passed else "failed",
        "optimizer_steps": 0,
        "task_level_training_executed": False,
        "profile": profile,
        "inputs": {
            "v4_run": {"path": str(args.v4_run.resolve()), "sha256": _sha256(args.v4_run)},
            "trace_artifact_sha256": artifact_hashes,
            "script": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        },
        "states": 1280,
        "actions_per_state": 36,
        "samples_per_state": args.samples_per_state,
        "sampled_components": 1280 * 36 * args.samples_per_state,
        "bounds": bounds_audit,
        "minimum_normalization_mass_observed": float(mass.min()),
        "sampled_action_range": [sampled_action_min, sampled_action_max],
        "sampled_outside_state_bounds": sampled_outside,
        "official_clamp_changed_components": official_clamp_changes,
        "theoretical_ctrlrange_lost_components": theoretical_ctrlrange_losses,
        "tail_state_count": int(tail_rows.sum()),
        "same_policy_ratio_max_abs_error": float((ratio - 1.0).abs().max().item()),
        "one_sided_zero_candidate_count": int(one_sided_candidates.sum().item()),
        "checks": checks,
        "decision": {
            "offline_gate_passed": passed,
            "four_world_zero_optimizer_gate_required_next": passed,
            "optimizer_training_authorized": False,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "bounds": bounds_audit,
        "minimum_normalization_mass_observed": report["minimum_normalization_mass_observed"],
        "sampled_components": report["sampled_components"],
        "checks": checks,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
