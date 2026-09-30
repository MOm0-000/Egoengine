"""Bounded helpers for Candidate-G first-update value-loss isolation.

The functions in this module are deliberately physics-free.  They evaluate the
already frozen Candidate-G PPO batch, decompose its two actor-optimizer loss
paths, and apply at most one independent shadow Adam step.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from video_to_spider.rl.algorithmic_training import exact_truncated_normal_kl
from video_to_spider.rl.state_feasible_truncated_gaussian import (
    deterministic_truncated_action,
)


BRANCHES = ("BASE", "FULL", "POLICY_ONLY", "VALUE_ONLY")
PARAMETER_GROUPS = (
    "all_parameters",
    "shared_MLP_LSTM_layer_norm",
    "policy_mean_head",
    "log_sigma",
    "internal_value_head",
)


class IsolationContractError(RuntimeError):
    """Raised when a bounded diagnostic invariant is violated."""


@dataclass(frozen=True)
class LossBundle:
    policy: torch.Tensor
    internal_value_unweighted: torch.Tensor
    internal_value_weighted: torch.Tensor
    full: torch.Tensor
    entropy: torch.Tensor
    ratio: torch.Tensor
    current: Mapping[str, torch.Tensor]
    canonical_old: Mapping[str, torch.Tensor]


def tensor_tree_sha256(value: Any) -> str:
    """Hash a tensor/tree without relying on pickle byte stability."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, np.int64).tobytes())
            digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray")
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, np.int64).tobytes())
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def parameter_group(name: str) -> str:
    if name.endswith("a2c_network.sigma") or name == "a2c_network.sigma":
        return "log_sigma"
    if ".mu." in name:
        return "policy_mean_head"
    if ".value." in name:
        return "internal_value_head"
    if any(token in name for token in (".actor_mlp.", ".rnn.", ".layer_norm.")):
        return "shared_MLP_LSTM_layer_norm"
    raise IsolationContractError(f"unclassified actor parameter: {name}")


def exact_loss_bundle(
    agent: Any,
    input_dict: Mapping[str, Any],
    *,
    canonical_old: Mapping[str, torch.Tensor] | None = None,
) -> LossBundle:
    """Build the exact frozen Candidate-G actor loss without updating state."""

    current = agent.evaluate_ppo_distribution(input_dict)
    if canonical_old is None:
        canonical_old = {
            name: current[name].detach().clone()
            for name in ("neglogp", "mu", "sigma")
        }
    old_neglogp = canonical_old["neglogp"].detach()
    ratio = torch.exp(old_neglogp - current["neglogp"])
    advantage = input_dict["advantages"]
    clipped_ratio = torch.clamp(
        ratio, 1.0 - agent.cfg.e_clip, 1.0 + agent.cfg.e_clip
    )
    policy = torch.max(
        -advantage * ratio, -advantage * clipped_ratio
    ).mean()
    value_preds = input_dict["old_values"]
    returns = input_dict["returns"]
    value_clipped = value_preds + (current["values"] - value_preds).clamp(
        -agent.cfg.e_clip, agent.cfg.e_clip
    )
    internal_value = torch.max(
        (current["values"] - returns).square(),
        (value_clipped - returns).square(),
    ).squeeze(dim=1).mean()
    value_weight = 0.5 * float(agent.cfg.critic_coef)
    weighted = internal_value * value_weight
    if not math.isclose(value_weight, 2.0, rel_tol=0.0, abs_tol=0.0):
        raise IsolationContractError(
            f"Candidate-G value-loss weight changed: {value_weight}"
        )
    if (agent.cfg.bounds_loss_coef or 0.0) != 0.0:
        raise IsolationContractError("bounded diagnostic requires zero bounds loss")
    if float(agent.current_entropy_coef) != 0.0:
        raise IsolationContractError("bounded diagnostic requires zero entropy loss")
    return LossBundle(
        policy=policy,
        internal_value_unweighted=internal_value,
        internal_value_weighted=weighted,
        full=policy + weighted,
        entropy=current["entropy"].mean(),
        ratio=ratio,
        current=current,
        canonical_old=canonical_old,
    )


def _flatten_gradient(
    names: list[str],
    gradients: tuple[torch.Tensor | None, ...],
    selected: Iterable[str],
) -> torch.Tensor:
    chosen = set(selected)
    rows = []
    for name, gradient in zip(names, gradients, strict=True):
        if name not in chosen:
            continue
        if gradient is None:
            # Use the true parameter shape, supplied by a sentinel later.
            continue
        rows.append(gradient.detach().double().reshape(-1).cpu())
    return torch.cat(rows) if rows else torch.zeros(0, dtype=torch.float64)


def gradient_decomposition(
    agent: Any,
    input_dict: Mapping[str, Any],
    *,
    canonical_old: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract policy/value/full gradients from one unchanged graph."""

    bundle = exact_loss_bundle(agent, input_dict, canonical_old=canonical_old)
    named = list(agent.model.named_parameters())
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    gradients = {
        "policy": torch.autograd.grad(
            bundle.policy, parameters, retain_graph=True, allow_unused=True
        ),
        "value": torch.autograd.grad(
            bundle.internal_value_weighted,
            parameters,
            retain_graph=True,
            allow_unused=True,
        ),
        "full": torch.autograd.grad(
            bundle.full, parameters, retain_graph=False, allow_unused=True
        ),
    }

    parameter_rows: list[dict[str, Any]] = []
    maximum_abs_linearity_error = 0.0
    numerator_sq = 0.0
    full_sq = 0.0
    sum_sq = 0.0
    for index, (name, parameter) in enumerate(named):
        values: dict[str, Any] = {}
        p = gradients["policy"][index]
        v = gradients["value"][index]
        f = gradients["full"][index]
        zero = torch.zeros_like(parameter)
        pd = zero if p is None else p
        vd = zero if v is None else v
        fd = zero if f is None else f
        error = fd - (pd + vd)
        maximum_abs_linearity_error = max(
            maximum_abs_linearity_error,
            float(error.detach().abs().max().cpu()),
        )
        numerator_sq += float(error.detach().double().square().sum().cpu())
        full_sq += float(fd.detach().double().square().sum().cpu())
        sum_sq += float((pd + vd).detach().double().square().sum().cpu())
        for label, gradient in (("policy", p), ("value", v), ("full", f)):
            values[f"{label}_is_none"] = gradient is None
            values[f"{label}_is_exact_zero"] = (
                False if gradient is None else bool(torch.count_nonzero(gradient) == 0)
            )
            values[f"{label}_l2"] = (
                None
                if gradient is None
                else float(gradient.detach().double().norm().cpu())
            )
        parameter_rows.append({
            "parameter": name,
            "group": parameter_group(name),
            "elements": int(parameter.numel()),
            **values,
        })

    relative = math.sqrt(numerator_sq) / max(
        math.sqrt(full_sq), math.sqrt(sum_sq), 1.0e-12
    )
    groups: dict[str, Any] = {}
    group_members = {
        "all_parameters": names,
        **{
            group: [name for name in names if parameter_group(name) == group]
            for group in PARAMETER_GROUPS[1:]
        },
    }
    # For group vector arithmetic, absent gradients are explicit zeros.
    parameter_by_name = dict(named)
    for group, member_names in group_members.items():
        vectors: dict[str, torch.Tensor] = {}
        for label in ("policy", "value", "full"):
            rows = []
            for name in member_names:
                index = names.index(name)
                gradient = gradients[label][index]
                if gradient is None:
                    gradient = torch.zeros_like(parameter_by_name[name])
                rows.append(gradient.detach().double().reshape(-1).cpu())
            vectors[label] = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.float64)
        p, v, f = vectors["policy"], vectors["value"], vectors["full"]
        pnorm, vnorm, fnorm = (float(x.norm()) for x in (p, v, f))
        cosine = None if pnorm <= 1.0e-12 or vnorm <= 1.0e-12 else float(torch.dot(p, v) / (pnorm * vnorm))
        denom = fnorm * fnorm
        groups[group] = {
            "parameter_count": len(member_names),
            "element_count": int(f.numel()),
            "policy_l2": pnorm,
            "weighted_value_l2": vnorm,
            "full_l2": fnorm,
            "weighted_value_to_policy_norm_ratio": None if pnorm <= 1.0e-12 else vnorm / pnorm,
            "policy_value_cosine": cosine,
            "policy_projection_on_full": None if denom <= 1.0e-24 else float(torch.dot(p, f) / denom),
            "value_projection_on_full": None if denom <= 1.0e-24 else float(torch.dot(v, f) / denom),
            "policy_none_count": sum(gradients["policy"][names.index(name)] is None for name in member_names),
            "value_none_count": sum(gradients["value"][names.index(name)] is None for name in member_names),
            "full_none_count": sum(gradients["full"][names.index(name)] is None for name in member_names),
        }
    report = {
        "losses": {
            "L_policy": float(bundle.policy.detach().cpu()),
            "L_internal_value_unweighted": float(bundle.internal_value_unweighted.detach().cpu()),
            "two_times_L_internal_value": float(bundle.internal_value_weighted.detach().cpu()),
            "L_full": float(bundle.full.detach().cpu()),
        },
        "canonical_ratio": {
            "minimum": float(bundle.ratio.detach().min().cpu()),
            "maximum": float(bundle.ratio.detach().max().cpu()),
            "max_abs_minus_one": float((bundle.ratio.detach() - 1.0).abs().max().cpu()),
        },
        "groups": groups,
        "linearity": {
            "relative_L2_error": relative,
            "maximum_absolute_element_error": maximum_abs_linearity_error,
            "tolerance": 1.0e-5,
            "passed": relative <= 1.0e-5,
        },
    }
    return report, parameter_rows


def _distribution(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().double().reshape(-1).cpu().numpy()
    return {
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def single_shadow_step(
    agent: Any,
    input_dict: Mapping[str, Any],
    *,
    branch: str,
    pre_actor_state: Mapping[str, Any],
    pre_optimizer_state: Mapping[str, Any],
    canonical_old: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Apply exactly one independent Adam step, or no step for BASE."""

    if branch not in BRANCHES:
        raise IsolationContractError(f"unknown branch {branch}")
    agent.model.load_state_dict(deepcopy(pre_actor_state), strict=True)
    agent.optimizer.load_state_dict(deepcopy(pre_optimizer_state))
    before_parameters = {
        name: value.detach().cpu().clone()
        for name, value in agent.model.named_parameters()
    }
    bundle = exact_loss_bundle(agent, input_dict, canonical_old=canonical_old)
    selected = {
        "FULL": bundle.full,
        "POLICY_ONLY": bundle.policy,
        "VALUE_ONLY": bundle.internal_value_weighted,
    }.get(branch)
    gradient_before_clip = None
    manual_gradient_before_clip = None
    gradient_after_clip = None
    optimizer_steps = 0
    if selected is not None:
        for parameter in agent.model.parameters():
            parameter.grad = None
        agent.scaler.scale(selected).backward()
        squares = [
            parameter.grad.detach().double().square().sum()
            for parameter in agent.model.parameters()
            if parameter.grad is not None
        ]
        gradient_before_clip = float(torch.stack(squares).sum().sqrt().cpu())
        agent.scaler.unscale_(agent.optimizer)
        clipped = torch.nn.utils.clip_grad_norm_(
            agent.model.parameters(), agent.cfg.grad_norm
        )
        # The official path records clip_grad_norm_'s native-dtype reduction.
        # The float64 sum above is retained only as an audit cross-check; it is
        # not a second semantic gate because different reduction dtypes can
        # legitimately differ by several float32 ulps at large norms.
        native_gradient_before_clip = float(clipped.detach().cpu())
        manual_gradient_before_clip = gradient_before_clip
        gradient_before_clip = native_gradient_before_clip
        squares = [
            parameter.grad.detach().double().square().sum()
            for parameter in agent.model.parameters()
            if parameter.grad is not None
        ]
        gradient_after_clip = float(torch.stack(squares).sum().sqrt().cpu())
        agent.scaler.step(agent.optimizer)
        agent.scaler.update()
        optimizer_steps = 1
    post_state = deepcopy(agent.model.state_dict())
    after_parameters = dict(agent.model.named_parameters())
    with torch.no_grad():
        post = agent.evaluate_ppo_distribution(input_dict)
        ratio = torch.exp(canonical_old["neglogp"] - post["neglogp"])
        exact_kl = exact_truncated_normal_kl(
            canonical_old["mu"], canonical_old["sigma"],
            post["mu"], post["sigma"],
            input_dict["action_lows"], input_dict["action_highs"],
        )
        deterministic = deterministic_truncated_action(
            post["mu"], input_dict["action_lows"], input_dict["action_highs"]
        )
        delta_sq = 0.0
        shared_sq = 0.0
        direct_value_head_sq = 0.0
        direct_mean_head_sq = 0.0
        dot = 0.0
        # Compute a fresh policy gradient at the common pre-state only for the
        # first-order diagnostic; the caller already counts this backward.
        for name, parameter in after_parameters.items():
            delta = parameter.detach().cpu().double() - before_parameters[name].double()
            value = float(delta.square().sum())
            delta_sq += value
            group = parameter_group(name)
            if group == "shared_MLP_LSTM_layer_norm":
                shared_sq += value
            elif group == "internal_value_head":
                direct_value_head_sq += value
            elif group in ("policy_mean_head", "log_sigma"):
                direct_mean_head_sq += value
        mu_change = (post["mu"] - canonical_old["mu"]).abs()
        sigma_change = (post["sigma"] - canonical_old["sigma"]).abs()
    # Evaluate post losses without changing RMS or optimizer state.
    post_bundle = exact_loss_bundle(agent, input_dict, canonical_old=canonical_old)
    report = {
        "branch": branch,
        "optimizer_steps": optimizer_steps,
        "selected_loss": None if selected is None else float(selected.detach().cpu()),
        "loss_before": {
            "policy": float(bundle.policy.detach().cpu()),
            "internal_value_unweighted": float(bundle.internal_value_unweighted.detach().cpu()),
            "internal_value_weighted": float(bundle.internal_value_weighted.detach().cpu()),
            "full": float(bundle.full.detach().cpu()),
        },
        "loss_after": {
            "policy": float(post_bundle.policy.detach().cpu()),
            "internal_value_unweighted": float(post_bundle.internal_value_unweighted.detach().cpu()),
            "internal_value_weighted": float(post_bundle.internal_value_weighted.detach().cpu()),
            "full": float(post_bundle.full.detach().cpu()),
        },
        "gradient_norm_before_clip": gradient_before_clip,
        "manual_float64_gradient_norm_before_clip": manual_gradient_before_clip,
        "manual_minus_native_gradient_norm": (
            manual_gradient_before_clip - gradient_before_clip
            if manual_gradient_before_clip is not None else None
        ),
        "gradient_norm_after_clip": gradient_after_clip,
        "parameter_delta_l2": math.sqrt(delta_sq),
        "shared_parameter_delta_l2": math.sqrt(shared_sq),
        "internal_value_head_delta_l2": math.sqrt(direct_value_head_sq),
        "policy_direct_parameter_delta_l2": math.sqrt(direct_mean_head_sq),
        "bounded_mu_change": {
            "maximum_absolute": float(mu_change.max().cpu()),
            "RMS": float(mu_change.square().mean().sqrt().cpu()),
        },
        "sigma_change": {
            "maximum_absolute": float(sigma_change.max().cpu()),
            "RMS": float(sigma_change.square().mean().sqrt().cpu()),
        },
        "exact_old_to_new_truncated_policy_KL": _distribution(exact_kl),
        "joint_ratio": _distribution(ratio),
        "true_clip_active_fraction": float(
            (
                ((input_dict["advantages"] > 0) & (ratio > 1.0 + agent.cfg.e_clip))
                | ((input_dict["advantages"] < 0) & (ratio < 1.0 - agent.cfg.e_clip))
            ).float().mean().cpu()
        ),
        "deterministic_action_bound_fraction": float(
            (
                torch.isclose(deterministic, input_dict["action_lows"], rtol=0.0, atol=1e-7)
                | torch.isclose(deterministic, input_dict["action_highs"], rtol=0.0, atol=1e-7)
            ).float().mean().cpu()
        ),
        "post_distribution": {
            "raw_location": post.get("raw_location", post["mu"]).detach().cpu(),
            "bounded_mu": post["mu"].detach().cpu(),
            "sigma": post["sigma"].detach().cpu(),
            "neglogp": post["neglogp"].detach().cpu(),
            "ratio": ratio.detach().cpu(),
            "exact_kl": exact_kl.detach().cpu(),
        },
        "state_hashes": {
            "actor": tensor_tree_sha256(post_state),
            "optimizer": tensor_tree_sha256(agent.optimizer.state_dict()),
        },
    }
    return report, post_state


def valid_prefix_summary(
    endpoints: np.ndarray,
    terminated: np.ndarray,
    scores: np.ndarray,
    tracking_rewards: np.ndarray,
) -> dict[str, Any]:
    """Count only the prefix before the first tracking/nonfinite failure."""

    endpoints = np.asarray(endpoints, dtype=np.int64)
    terminated = np.asarray(terminated, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    rewards = np.asarray(tracking_rewards, dtype=np.float64)
    invalid = terminated | ~np.isfinite(scores) | (scores > 1.0)
    first_index = int(np.argmax(invalid)) if invalid.any() else None
    valid = len(endpoints) if first_index is None else first_index
    failure = None if first_index is None else int(endpoints[first_index])
    return {
        "valid_prefix_intervals": int(valid),
        "first_failure_endpoint": failure,
        "forty_of_forty": first_index is None,
        "valid_prefix_tracking_reward_sum": float(rewards[:valid].sum()),
        "postfailure_rows_recorded_but_not_counted": int(len(endpoints) - valid - (0 if first_index is None else 1)),
        "endpoint80_timeout_is_tracking_failure": False,
    }


class BudgetCounter:
    """Fail-closed accounting for the bounded real experiment."""

    LIMITS = {
        "collection_control_intervals": 480,
        "closed_loop_control_intervals": 480,
        "total_control_intervals": 960,
        "physics_steps": 9600,
        "actor_optimizer_steps": 9,
        "critic_optimizer_steps": 12,
        "all_optimizer_steps": 21,
    }

    def __init__(self) -> None:
        self.values = {name: 0 for name in self.LIMITS}

    def add(self, name: str, amount: int) -> None:
        if name not in self.values:
            raise KeyError(name)
        self.values[name] += int(amount)
        if self.values[name] > self.LIMITS[name]:
            raise IsolationContractError(
                f"budget exceeded for {name}: {self.values[name]} > {self.LIMITS[name]}"
            )

    def add_collection(self, intervals: int) -> None:
        self.add("collection_control_intervals", intervals)
        self.add("total_control_intervals", intervals)
        self.add("physics_steps", intervals * 10)

    def add_closed_loop(self, intervals: int) -> None:
        self.add("closed_loop_control_intervals", intervals)
        self.add("total_control_intervals", intervals)
        self.add("physics_steps", intervals * 10)

    def add_actor_step(self) -> None:
        self.add("actor_optimizer_steps", 1)
        self.add("all_optimizer_steps", 1)

    def add_critic_steps(self, count: int) -> None:
        self.add("critic_optimizer_steps", count)
        self.add("all_optimizer_steps", count)

    def report(self) -> dict[str, Any]:
        return {
            "observed": dict(self.values),
            "limits": dict(self.LIMITS),
            "within_all_limits": all(
                self.values[name] <= limit for name, limit in self.LIMITS.items()
            ),
        }
