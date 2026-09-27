"""Compact, passive audit hooks for the fixed Pour A/B training benchmark.

The long benchmark has one actor update per 160-sample rollout.  The earlier
lossless credit recorder is intentionally too large for 625 epochs, so this
module records the declared per-update statistics and the exact rollout credit
arrays without retaining a model patch after every optimizer step.  It does
not alter the PPO objective, sampling, optimizer, normalization, or physics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .state_feasible_truncated_gaussian import (
    StateFeasibleTruncatedGaussianPpoAgent,
    deterministic_truncated_action,
)


SCHEMA = "taco_pour_algorithmic_training_update_audit_v1"


def _distribution(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    array = (
        values.detach().cpu().double().numpy()
        if torch.is_tensor(values)
        else np.asarray(values, dtype=np.float64)
    ).reshape(-1)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _parameter_l2_delta(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> float:
    total = 0.0
    for name in before:
        delta = after[name].detach().cpu().double() - before[name].double()
        total += float(delta.square().sum())
    return float(np.sqrt(total))


def _normal_pdf(value: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * value.square()) / np.sqrt(2.0 * np.pi)


def exact_truncated_normal_kl(
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
    a0 = (low64 - mu0) / sigma0
    b0 = (high64 - mu0) / sigma0
    a1 = (low64 - mu1) / sigma1
    b1 = (high64 - mu1) / sigma1
    z0 = (torch.special.ndtr(b0) - torch.special.ndtr(a0)).clamp_min(1.0e-15)
    z1 = (torch.special.ndtr(b1) - torch.special.ndtr(a1)).clamp_min(1.0e-15)
    phi_a = _normal_pdf(a0)
    phi_b = _normal_pdf(b0)
    mean_z = (phi_a - phi_b) / z0
    second_z = 1.0 + (a0 * phi_a - b0 * phi_b) / z0
    mean_x = mu0 + sigma0 * mean_z
    second_x = (
        mu0.square()
        + 2.0 * mu0 * sigma0 * mean_z
        + sigma0.square() * second_z
    )
    new_quadratic = (
        second_x - 2.0 * mu1 * mean_x + mu1.square()
    ) / sigma1.square()
    return (
        torch.log(sigma1 * z1)
        - torch.log(sigma0 * z0)
        + 0.5 * (new_quadratic - second_z)
    ).sum(dim=-1)


class AlgorithmicBenchmarkPpoAgent(StateFeasibleTruncatedGaussianPpoAgent):
    """The frozen agent plus bounded-size, non-mutating update evidence."""

    def __init__(self, *args, audit_dir: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if int(self.minibatch_size) != int(self.batch_size):
            raise ValueError("algorithmic audit requires one full-rollout minibatch")
        if int(self.cfg.mini_epochs) != 1:
            raise ValueError("algorithmic benchmark is frozen to one actor pass")
        self.audit_dir = Path(audit_dir)
        if self.audit_dir.exists():
            raise FileExistsError(self.audit_dir)
        self.audit_dir.mkdir(parents=True)
        self.update_reports: list[dict[str, Any]] = []
        self.epoch_reports: list[dict[str, Any]] = []
        self._credit_arrays: dict[str, np.ndarray] | None = None
        self._gradient_norm_before_clip = float("nan")
        self._gradient_norm_after_clip = float("nan")
        self._latest_distribution_evaluation: dict[str, torch.Tensor] | None = None
        self._training_reward_mean = float("nan")

    def evaluate_ppo_distribution(self, input_dict):
        result = super().evaluate_ppo_distribution(input_dict)
        self._latest_distribution_evaluation = {
            name: value.detach().clone()
            for name, value in result.items()
            if torch.is_tensor(value)
        }
        return result

    def prepare_dataset(self, batch_dict) -> None:
        raw_advantage = (batch_dict["returns"] - batch_dict["values"]).sum(dim=1)
        values = batch_dict["values"].detach().cpu().numpy()
        returns = batch_dict["returns"].detach().cpu().numpy()
        # The official recurrent rollout return intentionally omits rewards
        # from ``batch_dict`` after GAE construction. Read the already-filled
        # experience-buffer tensor without changing or reinserting it into the
        # optimizer dataset.
        rollout_rewards = self.experience_buffer.tensor_dict["rewards"]
        self._training_reward_mean = float(
            rollout_rewards.detach().float().mean().cpu()
        )
        super().prepare_dataset(batch_dict)
        normalized = self.dataset.values_dict["advantages"].detach().cpu().numpy()
        self._credit_arrays = {
            "raw_advantage": raw_advantage.detach().cpu().numpy(),
            "normalized_advantage": normalized,
            "return": returns,
            "value": values,
        }

    def truncate_gradients_and_step(self) -> None:
        if self.cfg.multi_gpu:
            raise ValueError("algorithmic benchmark is single-process CPU PPO")
        if self.cfg.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm)
            self._gradient_norm_before_clip = float(norm.detach().cpu())
            self._gradient_norm_after_clip = min(
                self._gradient_norm_before_clip, float(self.cfg.grad_norm)
            )
        else:
            squares = [
                parameter.grad.detach().float().square().sum()
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ]
            norm = float(torch.stack(squares).sum().sqrt().cpu())
            self._gradient_norm_before_clip = norm
            self._gradient_norm_after_clip = norm
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train_actor_critic(self, input_dict):
        before = {
            name: value.detach().cpu().clone()
            for name, value in self.model.named_parameters()
        }
        old_mu = input_dict["mu"].detach().clone()
        old_sigma = input_dict["sigma"].detach().clone()
        old_neglogp = input_dict["old_logp_actions"].detach().clone()
        low = input_dict["action_lows"].detach().clone()
        high = input_dict["action_highs"].detach().clone()
        try:
            result = super().train_actor_critic(input_dict)
        except RuntimeError as error:
            if "pre-optimizer likelihood identity gate failed" not in str(error):
                raise
            failed = self._latest_distribution_evaluation
            if failed is None:
                raise RuntimeError("identity failure did not retain its distribution") from error
            with torch.no_grad():
                recomputed = self.evaluate_ppo_distribution(input_dict)
                ratio = torch.exp(old_neglogp - recomputed["neglogp"])
            failure_path = self.audit_dir / f"epoch_{self.epoch_num:04d}_identity_failure.npz"
            np.savez_compressed(
                failure_path,
                old_mu=old_mu.detach().cpu().numpy(),
                failed_mu=failed["mu"].detach().cpu().numpy(),
                recomputed_mu=recomputed["mu"].detach().cpu().numpy(),
                old_sigma=old_sigma.detach().cpu().numpy(),
                failed_sigma=failed["sigma"].detach().cpu().numpy(),
                recomputed_sigma=recomputed["sigma"].detach().cpu().numpy(),
                old_neglogp=old_neglogp.detach().cpu().numpy(),
                failed_neglogp=failed["neglogp"].detach().cpu().numpy(),
                recomputed_neglogp=recomputed["neglogp"].detach().cpu().numpy(),
                failed_ratio=torch.exp(
                    old_neglogp - failed["neglogp"]
                ).detach().cpu().numpy(),
                ratio=ratio.detach().cpu().numpy(),
                action=input_dict["actions"].detach().cpu().numpy(),
                action_low=low.detach().cpu().numpy(),
                action_high=high.detach().cpu().numpy(),
            )
            raise
        after = {
            name: value.detach().cpu().clone()
            for name, value in self.model.named_parameters()
        }
        with torch.no_grad():
            post = self.evaluate_ppo_distribution(input_dict)
            ratio = torch.exp(old_neglogp - post["neglogp"])
            exact_kl = exact_truncated_normal_kl(
                old_mu, old_sigma, post["mu"], post["sigma"], low, high
            )
            advantage = input_dict["advantages"]
            unclipped = advantage * ratio
            clipped = advantage * torch.clamp(
                ratio, 1.0 - self.cfg.e_clip, 1.0 + self.cfg.e_clip
            )
            deterministic = deterministic_truncated_action(post["mu"], low, high)
            at_low = torch.isclose(deterministic, low, rtol=0.0, atol=1.0e-7)
            at_high = torch.isclose(deterministic, high, rtol=0.0, atol=1.0e-7)
            residual = deterministic * float(self.distribution_spec.residual_scale)
        row = {
            "epoch": int(self.epoch_num),
            "global_actor_update": len(self.update_reports) + 1,
            "pre_optimizer_identity": dict(
                self._likelihood_identity_checks[-1]
            ),
            "pre_optimizer_ratio_identity_max_abs_error": float(
                self._likelihood_identity_checks[-1]["maximum_abs_error_from_one"]
            ),
            "post_optimizer_ratio": _distribution(ratio),
            "post_optimizer_ratio_outside_0p8_1p2_fraction": float(
                ((ratio < 0.8) | (ratio > 1.2)).float().mean().cpu()
            ),
            "surrogate_changed_by_clip_fraction": float(
                (torch.abs(unclipped - clipped) > 1.0e-8).float().mean().cpu()
            ),
            "exact_old_to_new_truncated_policy_KL": _distribution(exact_kl),
            "entropy_pre_optimizer": float(result[2].detach().cpu()),
            "sigma_post_optimizer": _distribution(post["sigma"]),
            "gradient_norm_before_clip": self._gradient_norm_before_clip,
            "gradient_norm_after_clip": self._gradient_norm_after_clip,
            "parameter_delta_l2": _parameter_l2_delta(before, after),
            "deterministic_action_bound_fraction": float(
                (at_low | at_high).float().mean().cpu()
            ),
            "deterministic_effective_residual_RMS": float(
                residual.square().mean().sqrt().cpu()
            ),
            "training_reward_mean": self._training_reward_mean,
        }
        if not np.isfinite(np.asarray([
            row["gradient_norm_before_clip"], row["parameter_delta_l2"],
            row["exact_old_to_new_truncated_policy_KL"]["mean"],
        ])).all():
            raise RuntimeError("nonfinite algorithmic benchmark update evidence")
        self.update_reports.append(row)
        return result

    def train_epoch(self):
        version_before = int(self._observation_normalization_version)
        result = super().train_epoch()
        if self._credit_arrays is None:
            raise RuntimeError("algorithmic benchmark credit arrays were not captured")
        credit_path = self.audit_dir / f"epoch_{self.epoch_num:04d}_credit.npz"
        np.savez_compressed(credit_path, **self._credit_arrays)
        update = self.update_reports[-1]
        row = {
            "epoch": int(self.epoch_num),
            "samples": int(self.batch_size),
            "actor_updates": 1,
            "observation_normalization_version_used": version_before,
            "observation_normalization_version_after_commit": int(
                self._observation_normalization_version
            ),
            "credit": {
                "path": str(credit_path.resolve()),
                "sha256": hashlib.sha256(credit_path.read_bytes()).hexdigest(),
            },
            "update": update,
        }
        path = self.audit_dir / f"epoch_{self.epoch_num:04d}_summary.json"
        path.write_text(json.dumps(row, indent=2) + "\n")
        self.epoch_reports.append(row)
        self._credit_arrays = None
        return result

    def finalize_algorithmic_audit(self) -> dict[str, Any]:
        manifest = {
            "schema": SCHEMA,
            "status": "completed",
            "epochs": len(self.epoch_reports),
            "actor_updates": len(self.update_reports),
            "one_actor_update_per_epoch": (
                len(self.epoch_reports) == len(self.update_reports)
            ),
            "epoch_reports": self.epoch_reports,
        }
        path = self.audit_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema": SCHEMA,
            "epochs": len(self.epoch_reports),
            "actor_updates": len(self.update_reports),
        }
