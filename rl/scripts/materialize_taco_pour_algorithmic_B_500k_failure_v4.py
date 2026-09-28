#!/usr/bin/env python3
"""Materialize a fail-closed Candidate-B continuation record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "runs/taco_pour_algorithmic_reproduction_training_v4"
    / "B_budget_extension_500k"
)
FAILURES = {
    0: {
        "classification": (
            "state_feasible_truncated_gaussian_normalization_mass_below_"
            "frozen_safety_threshold"
        ),
        "last_update": 161,
        "last_update_physics": 257_600,
        "failed_epoch": 162,
        "failed_samples": 160,
        "physics_at_exception": 259_200,
        "milestones": (62, 125),
        "exception_type": "ValueError",
        "exception_message": (
            "truncated-Gaussian normalization mass 6.48980869e-13 "
            "is below the 1e-12 fail-closed threshold"
        ),
        "normalization_mass": 6.48980869e-13,
        "frozen_failclosed_threshold": 1.0e-12,
        "site": "state_feasible_truncated_gaussian._distribution_terms",
    },
    1: {
        "classification": "rollout_canonical_likelihood_numeric_gate_failure",
        "last_update": 194,
        "last_update_physics": 310_400,
        "failed_epoch": 195,
        "failed_samples": 160,
        "physics_at_exception": 312_000,
        "milestones": (62, 125, 188),
        "exception_type": "RuntimeError",
        "exception_message": (
            "canonical old-policy gate failed: "
            "classification=rollout_canonical_likelihood_mismatch, "
            "semantic identity all true, rollout_ratio_error=1.62124634e-05, "
            "canonical_ratio_error=0"
        ),
        "rollout_ratio_error": 1.62124634e-05,
        "frozen_ratio_tolerance": 1.52587890625e-05,
        "canonical_ratio_error": 0.0,
        "site": "_validate_canonical_first_update_identity",
    },
    2: {
        "classification": (
            "state_feasible_truncated_gaussian_normalization_mass_below_"
            "frozen_safety_threshold"
        ),
        "last_update": 191,
        "last_update_physics": 305_600,
        "failed_epoch": 192,
        "failed_samples": 148,
        "physics_at_exception": 307_080,
        "milestones": (62, 125, 188),
        "exception_type": "ValueError",
        "exception_message": (
            "truncated-Gaussian normalization mass 6.6346928e-13 "
            "is below the 1e-12 fail-closed threshold"
        ),
        "normalization_mass": 6.6346928e-13,
        "frozen_failclosed_threshold": 1.0e-12,
        "site": "state_feasible_truncated_gaussian._distribution_terms",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def validation_summary(path: Path) -> dict[str, object]:
    with np.load(path) as arrays:
        endpoints = arrays["endpoint"].astype(int).tolist()
        terminated = arrays["terminated"].astype(bool).tolist()
        scores = arrays["tracking_score"].astype(float).tolist()
        position = arrays["position_error"].astype(float).tolist()
        rotation = arrays["rotation_error"].astype(float).tolist()
        sigma = arrays["actor_sigma"].astype(np.float64)
        actions = arrays["deterministic_action"].astype(np.float64)
    failure = next(
        (endpoint for endpoint, done in zip(endpoints, terminated, strict=True) if done),
        None,
    )
    successful = 40 if failure is None else failure - 21

    def endpoint_row(endpoint: int) -> dict[str, float]:
        index = endpoints.index(endpoint)
        p = position[index]
        r = rotation[index]
        return {
            "score": scores[index],
            "position_error_m": p,
            "rotation_error_rad": r,
            "position_squared_contribution": (p / 0.12) ** 2,
            "rotation_squared_contribution": (r / 1.5) ** 2,
        }

    result = {
        "successful_intervals": successful,
        "first_failure_endpoint": failure,
        "forty_of_forty": failure is None,
        "beats_Replay": successful > 30,
        "meaningful_refinement": successful >= 35,
        "endpoint40": endpoint_row(40),
        "endpoint50": endpoint_row(50),
        "endpoint60": endpoint_row(60),
        "final_sigma": {
            "minimum": float(sigma.min()),
            "mean": float(sigma.mean()),
            "maximum": float(sigma.max()),
        },
        "validation_residual_RMS": float(
            np.sqrt(np.mean(np.square(actions * 0.05)))
        ),
        "validation_action_bound_fraction": float(
            np.mean(np.isclose(np.abs(actions), 1.0))
        ),
    }
    if failure is not None:
        result["first_failure"] = endpoint_row(failure)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=FAILURES)
    args = parser.parse_args()
    seed = int(args.seed)
    expected = FAILURES[seed]
    seed_root = RUN_ROOT / f"seed_{seed}"
    output = seed_root / "failure_report.json"
    preflight_path = seed_root / "precontinuation_validation/report.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight["status"] != "passed_before_new_optimizer_update":
        raise RuntimeError(f"seed-{seed} resume preflight did not pass")
    if not preflight["source_validation_all_arrays_exact"]:
        raise RuntimeError(f"seed-{seed} source validation was not exact")

    manifest_path = seed_root / "training_visitation/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "training_failed_after_logged_rollout":
        raise RuntimeError("training visitation is not fail-closed")
    epochs = manifest["epochs"]
    if (
        int(epochs[-1]["epoch"]) != int(expected["failed_epoch"])
        or int(epochs[-1]["sample_count"]) != int(expected["failed_samples"])
    ):
        raise RuntimeError("unexpected partial failure location")
    update_summaries = sorted(
        (seed_root / "update_audit").glob("epoch_*_summary.json")
    )
    update_credits = sorted(
        (seed_root / "update_audit").glob("epoch_*_credit.npz")
    )
    expected_updates = int(expected["last_update"]) - 62
    if (
        len(update_summaries) != expected_updates
        or len(update_credits) != expected_updates
    ):
        raise RuntimeError("completed continuation update count differs")
    if update_summaries[-1].name != (
        f"epoch_{int(expected['last_update']):04d}_summary.json"
    ):
        raise RuntimeError("last complete update differs")

    milestone_rows: dict[str, dict[str, object]] = {
        "62": {
            "epoch": 62,
            "physics_steps": 99_200,
            "summary": preflight["summary"],
            "trajectory": preflight["trajectory"],
        }
    }
    milestone_physics = {125: 200_000, 188: 300_800}
    for epoch in expected["milestones"]:
        if epoch == 62:
            continue
        path = seed_root / "validations" / f"epoch_{epoch:04d}.npz"
        milestone_rows[str(epoch)] = {
            "epoch": epoch,
            "physics_steps": milestone_physics[epoch],
            "summary": validation_summary(path),
            "trajectory": artifact(path),
        }
    last_milestone = int(expected["milestones"][-1])
    last_validation = milestone_rows[str(last_milestone)]
    checkpoint_path = (
        seed_root / "checkpoints" / f"epoch_{last_milestone:04d}.pt.gz"
    )
    exception = {
        "type": expected["exception_type"],
        "message": expected["exception_message"],
        "site": expected["site"],
    }
    for name in (
        "normalization_mass",
        "frozen_failclosed_threshold",
        "rollout_ratio_error",
        "frozen_ratio_tolerance",
        "canonical_ratio_error",
    ):
        if name in expected:
            exception[name] = expected[name]
    report = {
        "schema": "taco_pour_algorithmic_candidate_B_500k_failclosed_v4",
        "status": "failed_closed_before_fixed_500k_budget",
        "classification": expected["classification"],
        "paper_faithful": False,
        "candidate": "B",
        "seed": seed,
        "source_epoch": 62,
        "target_epoch": 313,
        "last_completed_actor_update_epoch": expected["last_update"],
        "last_completed_physics_steps": expected["last_update_physics"],
        "failed_rollout_epoch": expected["failed_epoch"],
        "failed_rollout_completed_samples": expected["failed_samples"],
        "failed_rollout_expected_samples": 160,
        "physics_steps_at_exception": expected["physics_at_exception"],
        "exception": exception,
        "resume_preflight": artifact(preflight_path),
        "source_validation_all_arrays_exact": True,
        "last_fixed_milestone": {
            "epoch": last_milestone,
            "physics_steps": last_validation["physics_steps"],
            "successful_intervals": last_validation["summary"][
                "successful_intervals"
            ],
            "first_failure_endpoint": last_validation["summary"][
                "first_failure_endpoint"
            ],
            "checkpoint": artifact(checkpoint_path),
            "validation": last_validation["trajectory"],
        },
        "fixed_milestone_validations": milestone_rows,
        "partial_training_visitation": {
            **artifact(manifest_path),
            "status": manifest["status"],
            "logged_epochs": len(epochs),
            "first_epoch": int(epochs[0]["epoch"]),
            "last_epoch": int(epochs[-1]["epoch"]),
        },
        "completed_update_evidence": {
            "count": len(update_summaries),
            "first_epoch": 63,
            "last_epoch": expected["last_update"],
            "summary_sha256": {
                path.name: sha256(path) for path in update_summaries
            },
            "credit_sha256": {
                path.name: sha256(path) for path in update_credits
            },
        },
        "retry_executed": False,
        "threshold_relaxed": False,
        "intermediate_checkpoint_selected": False,
        "automatic_1m_executed": False,
        "chunk_commit_written": False,
        "eligible_as_500k_outcome": False,
        "interpretation": (
            "The fixed Candidate-B action-distribution contract became "
            "numerically invalid for this seed before 500k. This is a "
            "fail-closed seed-instability result, not a 500k performance "
            "score. The last fixed milestone may not be selected as a fallback."
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(artifact(output), indent=2))


if __name__ == "__main__":
    main()
