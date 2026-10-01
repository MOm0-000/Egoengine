"""Pure helpers for the bounded Candidate-G FULL-update contraction audit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from video_to_spider.rl.value_loss_isolation import parameter_group


ALPHAS = (0.0, 1.0, 0.5, 0.25, 0.125)
NEW_ALPHAS = (0.5, 0.25, 0.125)
LABELS = {
    0.0: "BASE_ANCHOR",
    1.0: "FULL_ANCHOR",
    0.5: "HALF",
    0.25: "QUARTER",
    0.125: "EIGHTH",
}


class ContractionContractError(RuntimeError):
    """Raised when the bounded diagnostic contract is violated."""


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.detach().cpu().contiguous().numpy().tobytes()
        == right.detach().cpu().contiguous().numpy().tobytes()
    )


def interpolate_actor_state(
    base: Mapping[str, torch.Tensor],
    full: Mapping[str, torch.Tensor],
    parameter_names: Sequence[str],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Interpolate named parameters in float64 and clone BASE buffers."""

    if float(alpha) not in ALPHAS:
        raise ContractionContractError(f"undeclared alpha: {alpha}")
    if set(base) != set(full):
        raise ContractionContractError("BASE/FULL actor keys differ")
    parameters = set(parameter_names)
    if not parameters or not parameters.issubset(base):
        raise ContractionContractError("parameter key set is empty or invalid")
    output: dict[str, torch.Tensor] = {}
    total_sq = full_sq = dot = 0.0
    unchanged = total = 0
    group_sq: dict[str, float] = {}
    for name in base:
        left, right = base[name], full[name]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise ContractionContractError(f"non-tensor actor state: {name}")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ContractionContractError(f"shape/dtype mismatch: {name}")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ContractionContractError(f"nonfinite actor state: {name}")
        if name not in parameters:
            if not _same_tensor(left, right):
                raise ContractionContractError(f"nonparameter buffer changed: {name}")
            output[name] = left.detach().cpu().clone()
            continue
        if alpha == 0.0:
            value = left.detach().cpu().clone()
        elif alpha == 1.0:
            value = right.detach().cpu().clone()
        else:
            value = (
                left.detach().cpu().double()
                + float(alpha)
                * (right.detach().cpu().double() - left.detach().cpu().double())
            ).to(left.dtype)
        output[name] = value.clone()
        displacement = value.double() - left.detach().cpu().double()
        full_displacement = right.detach().cpu().double() - left.detach().cpu().double()
        total_sq += float(displacement.square().sum())
        full_sq += float(full_displacement.square().sum())
        dot += float((displacement * full_displacement).sum())
        unchanged += int(torch.count_nonzero(value == left.detach().cpu()))
        total += int(value.numel())
        group = parameter_group(name)
        group_sq[group] = group_sq.get(group, 0.0) + float(
            displacement.square().sum()
        )
    if any(output[name].untyped_storage().data_ptr() == base[name].untyped_storage().data_ptr()
           for name in output):
        raise ContractionContractError("interpolated state shares BASE storage")
    displacement_l2 = math.sqrt(total_sq)
    full_l2 = math.sqrt(full_sq)
    cosine = None
    if displacement_l2 > 0.0 and full_l2 > 0.0:
        cosine = dot / (displacement_l2 * full_l2)
    audit = {
        "alpha": float(alpha),
        "label": LABELS[float(alpha)],
        "parameter_count": len(parameters),
        "buffer_count": len(base) - len(parameters),
        "displacement_l2": displacement_l2,
        "full_displacement_l2": full_l2,
        "effective_displacement_ratio": (
            None if full_l2 == 0.0 else displacement_l2 / full_l2
        ),
        "cosine_with_full_displacement": cosine,
        "quantized_unchanged_elements": unchanged,
        "parameter_elements": total,
        "group_displacement_l2": {
            name: math.sqrt(value) for name, value in group_sq.items()
        },
        "buffers_bitwise_equal": True,
        "endpoint_direct_clone": alpha in (0.0, 1.0),
        "eligible_for_training_resume": False,
        "eligible_for_chunk_commit": False,
    }
    return output, audit


def prefix_summary(
    endpoints: np.ndarray,
    terminated: np.ndarray,
    scores: np.ndarray,
    tracking_rewards: np.ndarray,
) -> dict[str, Any]:
    """Count a finite score<=1 prefix; a later recovery never changes N."""

    endpoints = np.asarray(endpoints, dtype=np.int64)
    terminated = np.asarray(terminated, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    rewards = np.asarray(tracking_rewards, dtype=np.float64)
    if not (len(endpoints) == len(terminated) == len(scores) == len(rewards)):
        raise ContractionContractError("trajectory arrays differ in length")
    # A finite, feasible endpoint-80 row may carry the normal window timeout;
    # timeout is not a tracking failure under this diagnostic contract.
    tracking_terminated = terminated & ~(
        (endpoints == 80) & np.isfinite(scores) & (scores <= 1.0)
    )
    invalid = tracking_terminated | ~np.isfinite(scores) | (scores > 1.0)
    first = int(np.flatnonzero(invalid)[0]) if invalid.any() else None
    n = len(scores) if first is None else first
    first_failure = None if first is None else int(endpoints[first])
    first20 = min(n, 20)
    return {
        "N": int(n),
        "successful_intervals": int(n),
        "valid_prefix_intervals": int(n),
        "first_failure_endpoint": first_failure,
        "forty_of_forty": first is None and len(scores) == 40,
        "retained_base_intervals": min(n, 20),
        "lost_base_intervals": max(20 - n, 0),
        "added_intervals_beyond_base": max(n - 20, 0),
        "valid_prefix_tracking_reward_sum": float(rewards[:n].sum()),
        "first20_or_valid_tracking_reward_sum": float(rewards[:first20].sum()),
        "postfailure_rows_recorded_but_not_counted": len(scores) - n - (0 if first is None else 1),
    }


def decide_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared common-alpha C/E rule to a complete matrix."""

    lookup = {(int(row["seed"]), float(row["alpha"])): int(row["N"]) for row in rows}
    expected = {(seed, alpha) for seed in (0, 1, 2) for alpha in ALPHAS}
    missing = sorted(expected - set(lookup))
    if missing:
        raise ContractionContractError(f"incomplete condition matrix: {missing}")
    common = [
        alpha for alpha in NEW_ALPHAS
        if all(lookup[(seed, alpha)] >= 20 for seed in (0, 1, 2))
    ]
    extension = [
        alpha for alpha in common
        if sum(lookup[(seed, alpha)] > 20 for seed in (0, 1, 2)) >= 2
    ]
    if common:
        classification = "COMMON_SCALE_RESTORES_BASE_PREFIX"
    elif any(lookup[(seed, alpha)] >= 20 for seed in (0, 1, 2) for alpha in NEW_ALPHAS):
        classification = "MIXED_SCALE_RESPONSE"
    else:
        classification = "NO_RESTORATION_ON_PREDECLARED_SCALES"
    return {
        "classification": classification,
        "common_preservation_set_C": common,
        "common_extension_set_E": extension,
    }


@dataclass
class BudgetLedger:
    """Fail-closed accounting for this no-training diagnostic."""

    batch_forwards: int = 0
    batch_rows: int = 0
    rollouts: int = 0
    control_intervals: int = 0
    physics_steps: int = 0
    burnin_rows: int = 0
    closed_loop_rows: int = 0

    def add_batch(self, rows: int) -> None:
        self.batch_forwards += 1
        self.batch_rows += int(rows)
        self._check()

    def add_rollout(self, intervals: int, physics_steps: int, burnin: int = 20) -> None:
        self.rollouts += 1
        self.control_intervals += int(intervals)
        self.physics_steps += int(physics_steps)
        self.burnin_rows += int(burnin)
        self.closed_loop_rows += int(intervals)
        self._check()

    def _check(self) -> None:
        limits = {
            "batch_forwards": 15,
            "batch_rows": 2400,
            "rollouts": 15,
            "control_intervals": 600,
            "physics_steps": 6000,
            "burnin_rows": 300,
            "closed_loop_rows": 600,
        }
        for name, limit in limits.items():
            if int(getattr(self, name)) > limit:
                raise ContractionContractError(f"budget exceeded: {name}")

    def as_dict(self) -> dict[str, int]:
        return {
            "fixed_batch_forwards": self.batch_forwards,
            "fixed_batch_actor_sample_rows": self.batch_rows,
            "rollouts": self.rollouts,
            "control_intervals": self.control_intervals,
            "physics_steps": self.physics_steps,
            "burnin_actor_sample_rows": self.burnin_rows,
            "closed_loop_actor_sample_rows": self.closed_loop_rows,
            "total_actor_sample_rows": self.batch_rows + self.burnin_rows + self.closed_loop_rows,
            "new_training_samples": 0,
            "new_training_control_intervals": 0,
            "backward_calls": 0,
            "actor_optimizer_steps": 0,
            "critic_optimizer_steps": 0,
        }
