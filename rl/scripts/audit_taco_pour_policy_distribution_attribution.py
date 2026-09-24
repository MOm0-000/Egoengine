#!/usr/bin/env python3
"""Attribute PPO action infeasibility to policy centre and exploration spread."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _normal_interval_probability(
    mu: np.ndarray, sigma: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    return ndtr((high - mu) / sigma) - ndtr((low - mu) / sigma)


def _metrics(data: dict[str, np.ndarray], rows: np.ndarray, components: np.ndarray) -> dict:
    ix = np.ix_(rows, components)
    pre = data["sampled_action_preclamp"][ix]
    clamped = data["sampled_action_clamped"][ix]
    mu = data["actor_mu"][ix]
    sigma = data["actor_sigma"][ix]
    low = data["feasible_low"][ix]
    high = data["feasible_high"][ix]
    requested = data["requested_residual"][ix]
    effective = data["effective_residual_after_ctrlrange"][ix]
    lost = data["residual_lost_to_ctrlrange"][ix]

    tolerance = 2e-7
    mu_low = mu < low
    mu_high = mu > high
    mu_out = mu_low | mu_high
    mu_out_policy = (mu < -1.0) | (mu > 1.0)
    clamped_mu = np.clip(mu, -1.0, 1.0)
    clamped_mu_out_actuator = (clamped_mu < low) | (clamped_mu > high)
    sampled_official_clamped = pre != clamped
    ctrl_lost = np.abs(lost) > tolerance
    fully_blocked = (np.abs(requested) > tolerance) & (np.abs(effective) <= tolerance)
    pre_within_policy = (pre >= -1.0) & (pre <= 1.0)

    p_policy_out = ndtr((-1.0 - mu) / sigma) + 1.0 - ndtr((1.0 - mu) / sigma)
    lower_additional = np.where(
        low > -1.0,
        ndtr((low - mu) / sigma) - ndtr((-1.0 - mu) / sigma),
        0.0,
    )
    upper_additional = np.where(
        high < 1.0,
        ndtr((1.0 - mu) / sigma) - ndtr((high - mu) / sigma),
        0.0,
    )
    p_additional_ctrlrange = np.maximum(lower_additional, 0.0) + np.maximum(
        upper_additional, 0.0
    )
    p_final_ctrlrange_out = np.where(
        low > -1.0, ndtr((low - mu) / sigma), 0.0
    ) + np.where(high < 1.0, 1.0 - ndtr((high - mu) / sigma), 0.0)
    p_feasible_preclamp = _normal_interval_probability(mu, sigma, low, high)

    component_count = int(mu.size)
    lost_count = int(ctrl_lost.sum())
    lost_mu_out = int((ctrl_lost & mu_out).sum())
    lost_mu_in = int((ctrl_lost & ~mu_out).sum())
    lost_clamped_mu_out = int((ctrl_lost & clamped_mu_out_actuator).sum())
    lost_clamped_mu_in = int((ctrl_lost & ~clamped_mu_out_actuator).sum())
    return {
        "world_step_count": int(len(rows)),
        "component_count": component_count,
        "mu_outside_feasible": {
            "count": int(mu_out.sum()),
            "fraction": _ratio(mu_out.sum(), component_count),
            "below_count": int(mu_low.sum()),
            "above_count": int(mu_high.sum()),
        },
        "mu_outside_official_minus1_plus1": {
            "count": int(mu_out_policy.sum()),
            "fraction": _ratio(mu_out_policy.sum(), component_count),
        },
        "officially_clamped_mu_outside_actuator_feasible": {
            "count": int(clamped_mu_out_actuator.sum()),
            "fraction": _ratio(clamped_mu_out_actuator.sum(), component_count),
        },
        "Gaussian_probability_mass": {
            "outside_official_minus1_plus1": _summary(p_policy_out),
            "additional_inside_minus1_plus1_but_outside_ctrlrange_feasible": _summary(
                p_additional_ctrlrange
            ),
            "outside_ctrlrange_feasible_after_official_clamp": _summary(
                p_final_ctrlrange_out
            ),
            "inside_ctrlrange_feasible_before_official_clamp": _summary(
                p_feasible_preclamp
            ),
            "expected_outside_official_component_count": float(p_policy_out.sum()),
            "expected_additional_ctrlrange_component_count": float(
                p_additional_ctrlrange.sum()
            ),
            "expected_final_ctrlrange_lost_component_count": float(
                p_final_ctrlrange_out.sum()
            ),
        },
        "observed_layers": {
            "official_clamp_changed_component_count": int(sampled_official_clamped.sum()),
            "official_clamp_changed_component_fraction": _ratio(
                sampled_official_clamped.sum(), component_count
            ),
            "ctrlrange_lost_component_count": lost_count,
            "ctrlrange_lost_component_fraction": _ratio(lost_count, component_count),
            "fully_blocked_component_count": int(fully_blocked.sum()),
            "ctrlrange_lost_with_preclamp_inside_minus1_plus1": int(
                (ctrl_lost & pre_within_policy).sum()
            ),
            "ctrlrange_lost_with_preclamp_outside_minus1_plus1": int(
                (ctrl_lost & ~pre_within_policy).sum()
            ),
            "ctrlrange_lost_while_mu_outside_feasible": lost_mu_out,
            "ctrlrange_lost_while_mu_inside_feasible": lost_mu_in,
            "ctrlrange_lost_mu_outside_fraction": _ratio(lost_mu_out, lost_count),
            "ctrlrange_lost_mu_inside_fraction": _ratio(lost_mu_in, lost_count),
            "ctrlrange_lost_while_officially_clamped_mu_outside_actuator_feasible": (
                lost_clamped_mu_out
            ),
            "ctrlrange_lost_while_officially_clamped_mu_inside_actuator_feasible": (
                lost_clamped_mu_in
            ),
            "ctrlrange_lost_clamped_mu_outside_fraction": _ratio(
                lost_clamped_mu_out, lost_count
            ),
            "ctrlrange_lost_clamped_mu_inside_fraction": _ratio(
                lost_clamped_mu_in, lost_count
            ),
            "sum_abs_lost_over_sum_abs_requested": _ratio(
                np.abs(lost).sum(), np.abs(requested).sum()
            ),
        },
        "actor_mu": _summary(mu),
        "actor_sigma": _summary(sigma),
        "feasible_interval": {
            "low": _summary(low),
            "high": _summary(high),
            "width": _summary(high - low),
            "zero_width_component_count": int(np.isclose(high, low).sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-report", type=Path,
        default=ROOT / "runs/taco_pour_policy_distribution_attribution_v1/report.json",
    )
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_policy_distribution_attribution_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / (
            "runs/taco_pour_policy_distribution_attribution_v1/"
            "policy_distribution_attribution_audit.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    report = json.loads(args.run_report.read_text())
    contract = yaml.safe_load(args.contract.read_text())
    if report.get("schema") != "taco_pour_policy_distribution_attribution_run_v1":
        raise ValueError("run report has the wrong schema")
    if report.get("status") != "diagnostic_complete_no_commit":
        raise ValueError("diagnostic did not complete")
    if report.get("incoming_boundary_restored_after_diagnostic") is not True:
        raise ValueError("incoming boundary was not restored")
    if report.get("promotion", {}).get("allowed") is not False:
        raise ValueError("diagnostic unexpectedly allowed promotion")
    if report.get("diagnostic_no_commit", {}).get("sha256") != _sha256(args.contract):
        raise ValueError("run report is not bound to the attribution contract")

    runs = report.get("diagnostic", {}).get("training_runs", [])
    if len(runs) != 1:
        raise ValueError("expected exactly one diagnostic training run")
    trace = runs[0].get("training_visitation", {})
    required = tuple(contract["required_trace"]["required_fields"])
    if trace.get("schema") != "taco_ppo_training_visitation_v4":
        raise ValueError("training trace is not v4")
    if trace.get("status") != "complete" or len(trace.get("epochs", [])) != 8:
        raise ValueError("training trace is incomplete")
    if trace.get("action_contract", {}).get("dimensions") != 36:
        raise ValueError("training trace is not 36-D")
    if trace["action_contract"].get("residual_scale") != 0.05:
        raise ValueError("attribution formula is frozen to residual_scale=0.05")
    if trace["action_contract"].get("residual_clip") != 0.05:
        raise ValueError("attribution formula is frozen to residual_clip=0.05")

    parts: dict[str, list[np.ndarray]] = {name: [] for name in required}
    for epoch in trace["epochs"]:
        path = args.run_report.parent / "ppo_diagnostic_chunk_20" / "training_visitation" / Path(
            epoch["visits"]["path"]
        ).name
        if _sha256(path) != epoch["visits"]["sha256"]:
            raise ValueError(f"trace artifact hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as raw:
            missing = [name for name in required if name not in raw.files]
            if missing:
                raise ValueError(f"trace artifact lacks {missing}")
            for name in required:
                value = np.asarray(raw[name])
                if value.shape != (160, 36):
                    raise ValueError(f"{name} must have shape (160,36), got {value.shape}")
                parts[name].append(value)
            for name in ("source_endpoint", "outcome_endpoint"):
                parts.setdefault(name, []).append(np.asarray(raw[name], np.int32))
    data = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    if len(data["actor_mu"]) != 1280:
        raise ValueError("frozen diagnostic must contain exactly 1280 samples")
    if np.any(data["actor_sigma"] <= 0.0):
        raise ValueError("actor sigma must be strictly positive")
    if not np.array_equal(
        data["sampled_action_clamped"],
        np.clip(data["sampled_action_preclamp"], -1.0, 1.0),
    ):
        raise ValueError("official action clamp does not match the trace")
    decomposition_error = float(np.max(np.abs(
        data["requested_residual"]
        - data["effective_residual_after_ctrlrange"]
        - data["residual_lost_to_ctrlrange"]
    )))
    if decomposition_error > np.finfo(np.float64).eps:
        raise ValueError("residual decomposition identity failed")

    action_contract = trace["action_contract"]
    range_contract = action_contract["ctrlrange"]
    limited = np.asarray(range_contract["ctrllimited"], bool)
    ranges = np.asarray(range_contract["ctrlrange"], np.float64)
    reference = np.asarray(data["reference_ctrl"], np.float64)
    reference_below = np.where(
        limited, np.maximum(ranges[:, 0] - reference, 0.0), 0.0
    )
    reference_above = np.where(
        limited, np.maximum(reference - ranges[:, 1], 0.0), 0.0
    )
    reference_violation = np.maximum(reference_below, reference_above)
    reference_tolerance = 2e-7
    if np.any(reference_violation > reference_tolerance):
        raise ValueError("reference control exceeds actuator ctrlrange beyond tolerance")
    # Float32 reference serialization can put a value a few 1e-8 beyond a
    # decimal ctrlrange endpoint (for example -0.1700000018 vs -0.17).  Snap
    # only those tolerance-level discrepancies before deriving the normalized
    # feasible interval; larger violations remain fail-closed above.
    reference_for_interval = reference.copy()
    reference_for_interval[:, limited] = np.clip(
        reference_for_interval[:, limited],
        ranges[limited, 0],
        ranges[limited, 1],
    )
    low = np.full(reference.shape, -1.0, np.float64)
    high = np.full(reference.shape, 1.0, np.float64)
    low[:, limited] = np.maximum(
        -1.0,
        (ranges[limited, 0] - reference_for_interval[:, limited]) / 0.05,
    )
    high[:, limited] = np.minimum(
        1.0,
        (ranges[limited, 1] - reference_for_interval[:, limited]) / 0.05,
    )
    if np.any(low > high):
        raise ValueError("derived feasible action interval is empty")
    data["feasible_low"] = low
    data["feasible_high"] = high

    actuator_names = tuple(action_contract["actuator_names"])
    groups = {
        "all": np.arange(36),
        "right_wrist_translation": np.arange(0, 3),
        "right_wrist_rotation": np.arange(3, 6),
        "right_fingers": np.arange(6, 18),
        "left_wrist_translation": np.arange(18, 21),
        "left_wrist_rotation": np.arange(21, 24),
        "left_fingers": np.arange(24, 36),
        "both_fingers": np.asarray([*range(6, 18), *range(24, 36)]),
    }
    all_rows = np.arange(1280)
    row_groups = {
        "all": all_rows,
        "anchor_worlds_0_1_2": all_rows[all_rows % 4 < 3],
        "tail_world_3": all_rows[all_rows % 4 == 3],
        "outcome_endpoint_46_to_53": all_rows[
            (data["outcome_endpoint"] >= 46) & (data["outcome_endpoint"] <= 53)
        ],
    }
    grouped = {
        row_name: {
            group_name: _metrics(data, rows, components)
            for group_name, components in groups.items()
        }
        for row_name, rows in row_groups.items()
    }
    per_actuator = {}
    for index, name in enumerate(actuator_names):
        metrics = _metrics(data, all_rows, np.asarray([index]))
        lost = np.abs(data["residual_lost_to_ctrlrange"][:, index]) > 2e-7
        signed = data["residual_lost_to_ctrlrange"][:, index]
        metrics["observed_layers"].update({
            "negative_lost_count": int((lost & (signed < 0.0)).sum()),
            "positive_lost_count": int((lost & (signed > 0.0)).sum()),
        })
        per_actuator[name] = metrics

    finger = grouped["all"]["both_fingers"]
    lost_count = finger["observed_layers"]["ctrlrange_lost_component_count"]
    mu_out_lost = finger["observed_layers"][
        "ctrlrange_lost_while_mu_outside_feasible"
    ]
    mu_in_lost = finger["observed_layers"][
        "ctrlrange_lost_while_mu_inside_feasible"
    ]
    clamped_mu_out_lost = finger["observed_layers"][
        "ctrlrange_lost_while_officially_clamped_mu_outside_actuator_feasible"
    ]
    clamped_mu_in_lost = finger["observed_layers"][
        "ctrlrange_lost_while_officially_clamped_mu_inside_actuator_feasible"
    ]
    output = {
        "schema": "taco_pour_policy_distribution_attribution_audit_v1",
        "status": "completed_descriptive_no_promotion",
        "inputs": {
            "run_report": {"path": str(args.run_report.resolve()), "sha256": _sha256(args.run_report)},
            "contract": {"path": str(args.contract.resolve()), "sha256": _sha256(args.contract)},
            "audit_script": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        },
        "scope": {
            "training_worlds": 4,
            "epochs": 8,
            "samples": 1280,
            "seed": 0,
            "fixed_reset_endpoints": [20, 20, 20, 46],
            "GPU_training_is_nondeterministic": True,
            "performance_comparison_allowed": False,
            "chunk_commit_allowed": False,
            "new_action_parameterization_selected": False,
            "new_scale_selected": False,
        },
        "contracts": {
            "actor_sigma_semantics": "standard deviation = exp(logstd), not variance",
            "feasible_interval_formula": {
                "low": "max(-1, (ctrl_low-reference_ctrl)/0.05)",
                "high": "min(+1, (ctrl_high-reference_ctrl)/0.05)",
            },
            "reference_ctrlrange_numeric_contract": {
                "tolerance": reference_tolerance,
                "outside_exact_component_count": int((reference_violation > 0.0).sum()),
                "outside_tolerance_component_count": int(
                    (reference_violation > reference_tolerance).sum()
                ),
                "maximum_violation": float(reference_violation.max()),
                "tolerance_level_values_snapped_for_interval_derivation": int(
                    (reference_violation > 0.0).sum()
                ),
            },
            "sampled_action_clamp_exact": True,
            "requested_equals_effective_plus_lost": True,
            "decomposition_max_abs_error": decomposition_error,
        },
        "grouped_attribution": grouped,
        "per_actuator_attribution": per_actuator,
        "descriptive_attribution": {
            "finger_ctrlrange_lost_components": lost_count,
            "lost_while_actor_mu_outside_feasible": mu_out_lost,
            "lost_while_actor_mu_inside_feasible": mu_in_lost,
            "lost_while_actor_mu_outside_fraction": _ratio(mu_out_lost, lost_count),
            "lost_while_actor_mu_inside_fraction": _ratio(mu_in_lost, lost_count),
            "lost_while_officially_clamped_mu_outside_actuator_feasible": (
                clamped_mu_out_lost
            ),
            "lost_while_officially_clamped_mu_inside_actuator_feasible": (
                clamped_mu_in_lost
            ),
            "lost_while_officially_clamped_mu_outside_fraction": _ratio(
                clamped_mu_out_lost, lost_count
            ),
            "lost_while_officially_clamped_mu_inside_fraction": _ratio(
                clamped_mu_in_lost, lost_count
            ),
            "interpretation_limit": (
                "Raw mu outside the intersected interval can be repaired by the official "
                "[-1,1] clamp. Officially-clamped mu outside the actuator-feasible interval "
                "identifies a centre that still requests an infeasible control target; a lost "
                "sample with officially-clamped mu inside identifies exploration departure "
                "from an actuator-feasible centre. These are distribution attributions, not "
                "causal task-performance claims."
            ),
        },
        "decision": {
            "diagnostic_only": True,
            "result_may_not_promote_or_commit_a_chunk": True,
            "CPU_validation_score_may_not_be_used_as_performance_evidence": True,
            "action_parameterization_decision_requires_this_attribution": True,
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "status": output["status"],
        "finger_mu": finger["mu_outside_feasible"],
        "finger_observed_layers": finger["observed_layers"],
        "finger_Gaussian_probability_mass": finger["Gaussian_probability_mass"],
        "descriptive_attribution": output["descriptive_attribution"],
    }, indent=2))


if __name__ == "__main__":
    main()
