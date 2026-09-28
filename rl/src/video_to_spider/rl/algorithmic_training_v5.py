"""Candidate-C audit policy for the v5 mean-regularization ablation.

This module deliberately extends, rather than edits, the immutable v4 audit
implementation.  The PPO loss itself remains the official H2S2R loss already
implemented by :mod:`state_feasible_truncated_gaussian`; v5 only changes the
configured mean-regularization coefficient and records passive support-health
metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .algorithmic_training import AlgorithmicBenchmarkPpoAgent
from .state_feasible_truncated_gaussian import (
    _module_parameter_sha256,
    _tensor_tree_sha256,
    canonical_old_policy_identity_report,
    observation_normalizer_report,
)


SCHEMA = "taco_pour_algorithmic_training_update_audit_v5"


def _quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values.detach().double().reshape(-1), probability).cpu())


def truncated_support_health(
    *,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> dict[str, float]:
    """Return passive per-coordinate support diagnostics in float64."""

    if not (mu.shape == sigma.shape == low.shape == high.shape):
        raise ValueError("support-health tensors must have identical shapes")
    if not all(bool(torch.isfinite(value).all()) for value in (mu, sigma, low, high)):
        raise ValueError("support-health tensors must be finite")
    if bool((sigma <= 0).any()) or bool((low >= high).any()):
        raise ValueError("support-health distribution is invalid")

    mu64, sigma64, low64, high64 = (
        value.detach().double() for value in (mu, sigma, low, high)
    )
    alpha = (low64 - mu64) / sigma64
    beta = (high64 - mu64) / sigma64
    mass = torch.special.ndtr(beta) - torch.special.ndtr(alpha)
    violation = torch.where(
        mu64 < low64,
        (low64 - mu64) / sigma64,
        torch.where(mu64 > high64, (mu64 - high64) / sigma64, 0.0),
    )
    outside = (mu64 < low64) | (mu64 > high64)
    return {
        "minimum_truncated_normalization_mass": float(mass.min().cpu()),
        "p01_truncated_normalization_mass": _quantile(mass, 0.01),
        "median_truncated_normalization_mass": _quantile(mass, 0.50),
        "raw_mu_outside_support_fraction": float(outside.double().mean().cpu()),
        "maximum_support_violation_in_sigma": float(violation.max().cpu()),
        "p95_support_violation_in_sigma": _quantile(violation, 0.95),
    }


class MeanRegularizedAlgorithmicPpoAgent(AlgorithmicBenchmarkPpoAgent):
    """v5 agent with diagnostic rollout/canonical equivalence semantics.

    Semantic state identity and the canonical optimizer ratio remain strict.
    The separate no-grad rollout likelihood is diagnostic until its deviation
    exceeds the frozen 1e-4 semantic fail-closed threshold.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.cfg.bound_loss_type != "regularisation":
            raise ValueError("Candidate C requires regularisation bound loss")
        if float(self.cfg.bounds_loss_coef or 0.0) != 0.005:
            raise ValueError("Candidate C requires bounds_loss_coef=0.005")
        if self.cfg.lr_schedule is not None:
            raise ValueError("Candidate C forbids an LR schedule")

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
                pre_recompute_actor_normalizer == self._rollout_actor_normalizer
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
        rollout_error = float(
            report["rollout_vs_canonical"]["max_abs_ratio_minus_one"]
        )
        semantic_passed = bool(report["semantic_state_identity_passed"])
        canonical_exact = bool(
            report["canonical_optimizer_identity"]["ratio_bitwise_equal_one"]
        )
        rollout_hard_fail = rollout_error > float(
            self.canonical_old_policy_gate.semantic_identity_hard_fail
        )
        passed = semantic_passed and canonical_exact and not rollout_hard_fail
        report["rollout_vs_canonical"]["likelihood_equivalence_is_diagnostic_only"] = True
        report["rollout_vs_canonical"]["semantic_hard_fail_passed"] = (
            not rollout_hard_fail
        )
        report["passed"] = passed

        if not semantic_passed:
            classification = "frozen_policy_state_identity_defect"
        elif not canonical_exact:
            classification = "canonical_ratio_implementation_defect"
        elif rollout_hard_fail:
            classification = "rollout_canonical_semantic_hard_fail"
        elif not report["rollout_vs_canonical"]["likelihood_equivalence_passed"]:
            classification = "rollout_canonical_numerical_diagnostic"
        else:
            classification = "canonical_old_policy_identity_pass"
        row = {
            "epoch": int(self.epoch_num),
            "samples": int(canonical_optimizer_ratio.numel()),
            **report,
            "classification": classification,
            "maximum_abs_error_from_one": report[
                "canonical_optimizer_identity"
            ]["max_abs_ratio_minus_one"],
            "ratio_max_abs_error_from_one": rollout_error,
            "numerical_identity_passed": passed,
            "v5_gate_semantics": (
                "semantic fields exact; canonical optimizer ratio bitwise one; "
                "rollout/canonical likelihood diagnostic unless above 1e-4"
            ),
        }
        self._likelihood_identity_checks.append(row)
        self._actor_updates_in_epoch += 1
        if not passed:
            raise RuntimeError(
                "v5 canonical old-policy gate failed: "
                f"classification={classification}, semantic={semantic}, "
                f"rollout_ratio_error={rollout_error:.9g}, "
                "canonical_ratio_error="
                f"{report['canonical_optimizer_identity']['max_abs_ratio_minus_one']:.9g}"
            )

    def train_actor_critic(self, input_dict):
        result = super().train_actor_critic(input_dict)
        post = self._latest_distribution_evaluation
        if post is None:
            raise RuntimeError("v5 audit did not retain the post-update distribution")
        health = truncated_support_health(
            mu=post["mu"],
            sigma=post["sigma"],
            low=input_dict["action_lows"],
            high=input_dict["action_highs"],
        )
        bounds_loss = float(result[8].detach().cpu())
        row = self.update_reports[-1]
        row.update(health)
        row["actor_mean_regularization"] = {
            "type": "regularisation",
            "coefficient": 0.005,
            "unscaled_mean_sum_squared_mu": bounds_loss,
            "scaled_loss_contribution": 0.005 * bounds_loss,
        }
        if not all(math.isfinite(float(value)) for value in health.values()):
            raise RuntimeError("nonfinite v5 support-health instrumentation")
        return result

    def finalize_algorithmic_audit(self) -> dict[str, Any]:
        result = super().finalize_algorithmic_audit()
        path = Path(result["path"])
        manifest = json.loads(path.read_text())
        manifest["schema"] = SCHEMA
        manifest["mean_regularization"] = {
            "type": "regularisation",
            "coefficient": 0.005,
        }
        manifest["rollout_canonical_likelihood_role"] = (
            "diagnostic_below_semantic_hard_fail_1e-4"
        )
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        result["schema"] = SCHEMA
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["mean_regularization"] = manifest["mean_regularization"]
        result["rollout_canonical_likelihood_role"] = manifest[
            "rollout_canonical_likelihood_role"
        ]
        return result
