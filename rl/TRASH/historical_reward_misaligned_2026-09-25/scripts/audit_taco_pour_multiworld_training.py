#!/usr/bin/env python3
"""Compare the frozen one-world and four-world Pour coverage experiments."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(range(46, 51))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def local_path(recorded: str, parent: Path) -> Path:
    candidate = parent / Path(recorded).name
    return candidate if candidate.exists() else Path(recorded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "runs/taco_pour_multiworld_training_v1",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "runs/taco_pour_training_coverage_v1/coverage_analysis.json",
    )
    parser.add_argument(
        "--gate", type=Path,
        default=ROOT / "runs/taco_pour_multiworld_state_gate_v1/report.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "comparison.json"
    if output.exists():
        raise FileExistsError(output)

    report_path = args.run_dir / "report.json"
    report = json.loads(report_path.read_text())
    baseline = json.loads(args.baseline.read_text())
    gate = json.loads(args.gate.read_text())
    if report["local_settings"]["worlds"] != 4 or report["local_settings"]["ppo_epochs"] != 8:
        raise ValueError("comparison requires the frozen four-world/eight-epoch run")
    if gate.get("status") != "passed" or gate.get("worlds") != 4:
        raise ValueError("multi-world engineering gate did not pass")
    chunks = report.get("chunks", [])
    if len(chunks) != 1 or len(chunks[0].get("training_runs", [])) != 1:
        raise ValueError("comparison requires exactly one PPO fallback")
    trace = chunks[0]["training_runs"][0]["training_visitation"]
    manifest_path = args.run_dir / "ppo_chunk_20/training_visitation/manifest.json"
    if sha256(manifest_path) != trace["sha256"]:
        raise ValueError("visitation manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "taco_ppo_training_visitation_v2" or len(manifest["epochs"]) != 8:
        raise ValueError("unexpected visitation trace")

    worlds = 4
    per_world_rows = [[] for _ in range(worlds)]
    per_epoch = []
    all_rows = []
    for artifact in manifest["epochs"]:
        visit_path = local_path(artifact["visits"]["path"], manifest_path.parent)
        summary_path = local_path(artifact["summary"]["path"], manifest_path.parent)
        if sha256(visit_path) != artifact["visits"]["sha256"]:
            raise ValueError(f"visit hash mismatch: {visit_path}")
        if sha256(summary_path) != artifact["summary"]["sha256"]:
            raise ValueError(f"summary hash mismatch: {summary_path}")
        with np.load(visit_path, allow_pickle=False) as source:
            data = {name: source[name] for name in source.files}
        if len(data["source_endpoint"]) != 160:
            raise ValueError("each epoch must contain 40 steps x 4 worlds")
        reshaped = {
            name: value.reshape(40, worlds, *value.shape[1:])
            for name, value in data.items()
        }
        counts = {
            str(endpoint): int((data["outcome_endpoint"] == endpoint).sum())
            for endpoint in TARGETS
        }
        per_epoch.append({
            "epoch": int(artifact["epoch"]),
            "target_endpoint_visits": counts,
            "endpoint50_visited": counts["50"] > 0,
            "termination_endpoints": list(map(
                int, data["outcome_endpoint"][data["tracking_terminated"]]
            )),
        })
        all_rows.append(data)
        for world in range(worlds):
            for step in range(40):
                per_world_rows[world].append({
                    name: value[step, world] for name, value in reshaped.items()
                })

    data = {
        name: np.concatenate([epoch[name] for epoch in all_rows], axis=0)
        for name in all_rows[0]
    }
    episodes = []
    for world, rows in enumerate(per_world_rows):
        current = None
        for sample, row in enumerate(rows):
            source = int(row["source_endpoint"])
            outcome = int(row["outcome_endpoint"])
            if current is None:
                if source != 20:
                    raise ValueError("each independent episode must restart at endpoint 20")
                current = {"world": world, "first_sample": sample, "max_endpoint": outcome}
            current["last_sample"] = sample
            current["outcome_endpoint"] = outcome
            current["max_endpoint"] = max(current["max_endpoint"], outcome)
            current["sample_count"] = sample - current["first_sample"] + 1
            if bool(row["tracking_terminated"]) or bool(row["time_out"]):
                current["tracking_terminated"] = bool(row["tracking_terminated"])
                current["time_out"] = bool(row["time_out"])
                episodes.append(current)
                current = None
        if current is not None:
            current["incomplete_at_training_end"] = True
            episodes.append(current)

    visits = {}
    for endpoint in TARGETS:
        selected = data["outcome_endpoint"] == endpoint
        visits[str(endpoint)] = {
            "count": int(selected.sum()),
            "epochs_with_visit": sum(
                row["target_endpoint_visits"][str(endpoint)] > 0 for row in per_epoch
            ),
            "tool_position_error_m": distribution(data["tool_position_error_m"][selected]),
            "tool_rotation_error_rad": distribution(data["tool_rotation_error_rad"][selected]),
            "normalized_ellipse_score": distribution(data["tool_objective_score"][selected]),
        }

    validation = {
        row["mode"]: {
            "validated_steps": row["validated_steps"],
            "first_failure": row["first_failure"],
        }
        for row in chunks[0]["validation_traces"]
    }
    baseline_samples = baseline["training_samples"]
    baseline_visits = {
        endpoint: row["visit_count"]
        for endpoint, row in baseline_samples["per_target_endpoint"].items()
    }
    total_target_visits = sum(row["count"] for row in visits.values())
    completed = [episode for episode in episodes if "incomplete_at_training_end" not in episode]
    termination_counts = Counter(map(
        int, data["outcome_endpoint"][data["tracking_terminated"]]
    ))
    comparison = {
        "schema": "taco_pour_multiworld_training_comparison_v1",
        "status": "absolute_tail_coverage_and_CPU_validation_improved_but_40_of_40_failed",
        "artifacts": {
            "formal_report": {"path": "runs/taco_pour_multiworld_training_v1/report.json", "sha256": sha256(report_path)},
            "visitation_manifest": {"path": "runs/taco_pour_multiworld_training_v1/ppo_chunk_20/training_visitation/manifest.json", "sha256": sha256(manifest_path)},
            "engineering_gate": {"path": "runs/taco_pour_multiworld_state_gate_v1/report.json", "sha256": sha256(args.gate)},
            "one_world_baseline": {"path": "runs/taco_pour_training_coverage_v1/coverage_analysis.json", "sha256": sha256(args.baseline)},
        },
        "formal_outcome": {
            "status": report["status"],
            "committed_reference_index": report["committed_reference_index"],
            "task_success": report["task_success"],
            "CPU_validation": validation,
        },
        "four_world_training": {
            "epochs": 8,
            "worlds": worlds,
            "samples": len(data["source_endpoint"]),
            "episode_segments": len(episodes),
            "completed_episode_segments": len(completed),
            "incomplete_segments_at_training_end": len(episodes) - len(completed),
            "target_endpoint_total_visits": total_target_visits,
            "target_endpoint_visit_fraction": total_target_visits / len(data["source_endpoint"]),
            "per_target_endpoint": visits,
            "episode_segments_reaching_endpoint": {
                str(endpoint): sum(episode["max_endpoint"] >= endpoint for episode in episodes)
                for endpoint in TARGETS
            },
            "termination_endpoint_counts": {
                str(endpoint): count for endpoint, count in sorted(termination_counts.items())
            },
            "per_epoch": per_epoch,
            "per_world_target_visits": {
                str(world): {
                    str(endpoint): sum(
                        int(row["outcome_endpoint"]) == endpoint for row in rows
                    ) for endpoint in TARGETS
                } for world, rows in enumerate(per_world_rows)
            },
        },
        "one_world_to_four_world": {
            "samples": [baseline_samples["total_samples"], len(data["source_endpoint"])],
            "target_endpoint_total_visits": [
                baseline_samples["target_endpoint_total_visits"], total_target_visits,
            ],
            "target_endpoint_visit_fraction": [
                baseline_samples["target_endpoint_visit_fraction_of_all_samples"],
                total_target_visits / len(data["source_endpoint"]),
            ],
            "visits_by_endpoint": {
                endpoint: [baseline_visits[endpoint], visits[endpoint]["count"]]
                for endpoint in map(str, TARGETS)
            },
            "epochs_with_endpoint50_visit": [
                baseline_samples["per_target_endpoint"]["50"]["epoch_visit_count"],
                visits["50"]["epochs_with_visit"],
            ],
            "deterministic_CPU_PPO_validated_steps": [
                baseline["run_outcome"]["cpu_validation"]["rl"]["validated_steps"],
                validation["rl"]["validated_steps"],
            ],
            "same_run_CPU_Replay_validated_steps": [
                baseline["run_outcome"]["cpu_validation"]["replay"]["validated_steps"],
                validation["replay"]["validated_steps"],
            ],
        },
        "decision": {
            "absolute_tail_samples_increased": True,
            "tail_sample_fraction_increased": (
                total_target_visits / len(data["source_endpoint"])
                > baseline_samples["target_endpoint_visit_fraction_of_all_samples"]
            ),
            "deterministic_CPU_policy_improved_over_one_world_run": (
                validation["rl"]["validated_steps"]
                > baseline["run_outcome"]["cpu_validation"]["rl"]["validated_steps"]
            ),
            "deterministic_CPU_policy_improved_over_same_run_Replay": (
                validation["rl"]["validated_steps"]
                > validation["replay"]["validated_steps"]
            ),
            "passed_40_of_40": validation["rl"]["validated_steps"] == 40,
            "interpretation": (
                "Four worlds increased absolute endpoint 46--50 samples from 24 to 96 "
                "and CPU validation from 28 to 31 steps, but the target-band share stayed "
                "at 7.5% and the policy still failed 40/40. This supports a modest benefit "
                "from more independent rollouts, not a sufficient solution or proof that "
                "coverage sparsity was the sole cause. GPU training nondeterminism prevents "
                "a bitwise causal comparison across the two runs."
            ),
        },
    }
    output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({
        "status": comparison["status"],
        "visits": {endpoint: row["count"] for endpoint, row in visits.items()},
        "target_fraction": comparison["four_world_training"]["target_endpoint_visit_fraction"],
        "CPU_validation": validation,
    }, indent=2))


if __name__ == "__main__":
    main()
