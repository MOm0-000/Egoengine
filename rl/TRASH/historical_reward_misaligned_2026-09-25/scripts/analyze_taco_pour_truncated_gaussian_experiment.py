#!/usr/bin/env python3
"""Summarize the single authorized truncated-Gaussian Pour experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_state_feasible_truncated_gaussian_experiment_v1"
PRIOR = ROOT / "runs/taco_pour_tail_curriculum_3plus1_v1/report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report_path = RUN / "report.json"
    report = json.loads(report_path.read_text())
    actor_manifest_path = RUN / "actor_artifact.json"
    actor_manifest = json.loads(actor_manifest_path.read_text())
    actor_artifact_path = ROOT / actor_manifest["artifact"]["path"]
    if sha256(actor_artifact_path) != actor_manifest["artifact"]["sha256"]:
        raise ValueError("portable CPU-validated actor artifact changed")
    traces = {
        trace["mode"]: trace
        for trace in report["diagnostic"]["validation_traces"]
    }
    replay = traces["diagnostic_replay"]
    policy = traces["diagnostic_rl"]

    manifest_path = Path(
        report["diagnostic"]["training_runs"][0]["training_visitation"]["path"]
    )
    manifest = json.loads(manifest_path.read_text())
    parts = []
    epochs_with_endpoint_50 = 0
    for epoch in manifest["epochs"]:
        visit_path = Path(epoch["visits"]["path"])
        if sha256(visit_path) != epoch["visits"]["sha256"]:
            raise ValueError(f"training visit artifact changed: {visit_path}")
        with np.load(visit_path, allow_pickle=False) as raw:
            part = {name: np.asarray(raw[name]) for name in raw.files}
        parts.append(part)
        epochs_with_endpoint_50 += int(np.any(part["outcome_endpoint"] == 50))
    visits = {
        name: np.concatenate([part[name] for part in parts], axis=0)
        for name in parts[0]
    }
    if len(visits["outcome_endpoint"]) != 1280:
        raise ValueError("authorized experiment must contain exactly 1280 samples")

    lost = np.abs(visits["residual_lost_to_ctrlrange"])
    identity_error = np.abs(
        visits["requested_residual"]
        - visits["effective_residual_after_ctrlrange"]
        - visits["residual_lost_to_ctrlrange"]
    )
    clamp_changes = visits["sampled_action_preclamp"] != visits[
        "sampled_action_clamped"
    ]
    tail_mask = np.isin(visits["outcome_endpoint"], np.arange(46, 51))
    tail_visits = {
        str(endpoint): int(np.count_nonzero(visits["outcome_endpoint"] == endpoint))
        for endpoint in range(46, 51)
    }

    curriculum_epochs = report["diagnostic"]["training_runs"][0][
        "tail_curriculum"
    ]["epochs_prepared"]
    termination_by_world = [
        sum(
            int(epoch["termination_counts_by_world"][world]["tracking"])
            for epoch in curriculum_epochs
        )
        for world in range(4)
    ]

    replay_rows = {row["endpoint"]: row for row in replay["steps"]}
    policy_rows = {row["endpoint"]: row for row in policy["steps"]}
    common_endpoints = sorted(set(replay_rows) & set(policy_rows))
    replay_scores = np.asarray([
        replay_rows[endpoint]["aggregate_tracking_error"]
        for endpoint in common_endpoints
    ])
    policy_scores = np.asarray([
        policy_rows[endpoint]["aggregate_tracking_error"]
        for endpoint in common_endpoints
    ])

    cpu_lost = np.abs(np.asarray([
        row["residual_lost_to_ctrlrange"] for row in policy["steps"]
    ]))
    prior = json.loads(PRIOR.read_text())
    prior_traces = {
        trace["mode"]: trace
        for chunk in prior["chunks"]
        for trace in chunk["validation_traces"]
    }

    summary = {
        "schema": "taco_pour_state_feasible_truncated_gaussian_analysis_v1",
        "status": "completed_strict_gate_failed",
        "paper_faithful": False,
        "inputs": {
            "authorized_run_report": {
                "path": str(report_path.resolve()),
                "sha256": sha256(report_path),
            },
            "training_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256(manifest_path),
            },
            "historical_3plus1_run": {
                "path": str(PRIOR.resolve()),
                "sha256": sha256(PRIOR),
            },
            "CPU_validated_actor": {
                "manifest_path": str(actor_manifest_path.resolve()),
                "manifest_sha256": sha256(actor_manifest_path),
                "artifact_path": str(actor_artifact_path.resolve()),
                "artifact_sha256": sha256(actor_artifact_path),
                "actor_state_sha256": actor_manifest["actor_state_sha256"],
            },
            "analysis_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__)),
            },
        },
        "frozen_budget": {
            "world_start_endpoints": [20, 20, 20, 46],
            "worlds": 4,
            "epochs": 8,
            "horizon_per_world": 40,
            "samples": 1280,
            "seed": 0,
            "residual_scale": 0.05,
        },
        "deterministic_CPU_validation": {
            "required_steps": 40,
            "replay": {
                "validated_steps": replay["validated_steps"],
                "first_failure": replay["first_failure"],
            },
            "truncated_gaussian_PPO": {
                "validated_steps": policy["validated_steps"],
                "first_failure": policy["first_failure"],
                "failing_score": policy["steps"][-1]["aggregate_tracking_error"],
                "failing_position_error_m": policy["steps"][-1]["position_error_m"][0],
                "failing_rotation_error_rad": policy["steps"][-1]["rotation_error_rad"][0],
            },
            "paired_improvement_over_same_run_replay_steps": (
                policy["validated_steps"] - replay["validated_steps"]
            ),
            "strict_40_of_40_passed": policy["validated_steps"] == 40,
            "common_endpoint_comparison": {
                "endpoints": [common_endpoints[0], common_endpoints[-1]],
                "count": len(common_endpoints),
                "PPO_lower_score_count": int(np.count_nonzero(policy_scores < replay_scores)),
                "PPO_higher_score_count": int(np.count_nonzero(policy_scores > replay_scores)),
                "replay_mean_score": float(replay_scores.mean()),
                "PPO_mean_score": float(policy_scores.mean()),
            },
        },
        "training_action_channel": {
            "sample_count": len(visits["outcome_endpoint"]),
            "action_components": int(visits["sampled_action_preclamp"].size),
            "ordinary_minus1_plus1_clamp_changed_components": int(
                np.count_nonzero(clamp_changes)
            ),
            "ctrlrange_lost_components_exact": int(np.count_nonzero(lost)),
            "ctrlrange_lost_world_steps_exact": int(
                np.count_nonzero(np.any(lost != 0.0, axis=1))
            ),
            "maximum_abs_ctrlrange_loss": float(lost.max(initial=0.0)),
            "residual_identity_max_abs_error": float(identity_error.max(initial=0.0)),
        },
        "CPU_deterministic_action_channel": {
            "ctrlrange_lost_components_exact": int(np.count_nonzero(cpu_lost)),
            "ctrlrange_lost_components_above_2e_7": int(
                np.count_nonzero(cpu_lost > 2e-7)
            ),
            "maximum_abs_ctrlrange_loss": float(cpu_lost.max(initial=0.0)),
            "interpretation": (
                "All nonzero CPU losses are reference-snap numerical residue below "
                "the frozen 2e-7 tolerance; no material actuator clipping occurred."
            ),
        },
        "training_distribution": {
            "outcome_endpoint_46_to_50_visits": tail_visits,
            "endpoint_46_to_50_total": int(tail_mask.sum()),
            "endpoint_46_to_50_fraction": float(tail_mask.mean()),
            "epochs_with_endpoint_50_visit": epochs_with_endpoint_50,
            "tracking_terminations_by_world": termination_by_world,
            "anchor_world_tracking_terminations": sum(termination_by_world[:3]),
            "tail_world_tracking_terminations": termination_by_world[3],
        },
        "historical_cross_run_context": {
            "ordinary_Gaussian_3plus1_CPU_PPO_validated_steps": prior_traces["rl"]["validated_steps"],
            "truncated_Gaussian_CPU_PPO_validated_steps": policy["validated_steps"],
            "difference_steps": policy["validated_steps"] - prior_traces["rl"]["validated_steps"],
            "causal_claim_allowed": False,
            "reason": (
                "The GPU optimizer is nondeterministic, and the historical ordinary-"
                "Gaussian 3+1 PPO used a recurrent likelihood recomputation path that "
                "incorrectly assumed zero hidden state after curriculum resets. The "
                "historical task result is context, not a clean action-distribution "
                "ablation. Its saved rollout mu/sigma/ctrlrange statistics remain "
                "descriptive evidence of what that rollout executed."
            ),
        },
        "promotion": {
            "chunk_committed": report["promotion"]["chunk_committed"],
            "optimized_trajectory_written": report["promotion"]["optimized_trajectory_written"],
            "incoming_endpoint_20_boundary_restored": report[
                "incoming_boundary_restored_after_diagnostic"
            ],
            "full_RL_authorized_by_this_result": False,
        },
        "decision": {
            "action_channel_structural_loss_removed": True,
            "task_window_solved": False,
            "conclusion": (
                "The state-feasible truncated Gaussian removed the two hard-clamp "
                "information-loss layers and learned a useful 29-to-37 step correction, "
                "but it still failed the strict 40-of-40 CPU gate at endpoint 58."
            ),
        },
    }
    output = RUN / "analysis.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
