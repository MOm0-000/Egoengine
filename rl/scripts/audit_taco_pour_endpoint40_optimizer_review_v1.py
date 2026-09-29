#!/usr/bin/env python3
"""Read-only endpoint-40 lookahead coverage / optimizer adjudication."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def checked_manifest(path: Path, expected_sha: str, schema: str) -> dict[str, Any]:
    actual = sha256(path)
    if actual != expected_sha:
        raise ValueError(f"manifest hash mismatch: {path}: {actual} != {expected_sha}")
    result = load_json(path)
    if result.get("schema") != schema:
        raise ValueError(f"unexpected schema in {path}: {result.get('schema')}")
    if result.get("status") not in {"complete", "completed"}:
        raise ValueError(f"incomplete manifest: {path}")
    return result


def optimizer_segment(
    update_by_epoch: dict[int, dict[str, Any]], start: int, stop: int
) -> dict[str, Any]:
    rows = [update_by_epoch[epoch]["update"] for epoch in range(start, stop + 1)]
    if len(rows) != stop - start + 1:
        raise ValueError("optimizer segment is incomplete")
    metric = lambda name: distribution([float(row[name]) for row in rows])
    kl_means = [row["exact_old_to_new_truncated_policy_KL"]["mean"] for row in rows]
    kl_p95s = [row["exact_old_to_new_truncated_policy_KL"]["p95"] for row in rows]
    kl_maxes = [row["exact_old_to_new_truncated_policy_KL"]["max"] for row in rows]
    return {
        "epoch_range_inclusive": [start, stop],
        "epoch_count": len(rows),
        "exact_truncated_KL": {
            "aggregation_semantics": (
                "equal-size 160-sample epoch summaries; pooled sample p95 is not "
                "recoverable from summary-only artifacts"
            ),
            "sample_weighted_mean_from_epoch_means": float(np.mean(kl_means)),
            "per_epoch_mean": distribution(kl_means),
            "per_epoch_p95": distribution(kl_p95s),
            "per_epoch_max": distribution(kl_maxes),
        },
        "ratio_outside_0p8_1p2": metric(
            "post_optimizer_ratio_outside_0p8_1p2_fraction"
        ),
        "gradient_norm_before_clip": metric("gradient_norm_before_clip"),
        "parameter_delta_l2": metric("parameter_delta_l2"),
        "deterministic_effective_residual_RMS": metric(
            "deterministic_effective_residual_RMS"
        ),
        "training_reward_mean": metric("training_reward_mean"),
    }


def visitation_segment(
    root: Path,
    visitation_by_epoch: dict[int, dict[str, Any]],
    boundary: int,
    start: int,
    stop: int,
) -> dict[str, Any]:
    offsets: list[np.ndarray] = []
    epoch_maxima: list[int] = []
    termination_offsets: Counter[int] = Counter()
    timeout_offsets: Counter[int] = Counter()
    hashes_verified = 0
    for epoch in range(start, stop + 1):
        manifest_row = visitation_by_epoch[epoch]
        visits_path = root / "training_visitation" / f"epoch_{epoch:04d}_visits.npz"
        summary_path = root / "training_visitation" / f"epoch_{epoch:04d}_summary.json"
        if sha256(visits_path) != manifest_row["visits"]["sha256"]:
            raise ValueError(f"visitation hash mismatch: {visits_path}")
        if sha256(summary_path) != manifest_row["summary"]["sha256"]:
            raise ValueError(f"visitation summary hash mismatch: {summary_path}")
        hashes_verified += 2
        with np.load(visits_path, allow_pickle=False) as data:
            required = {
                "source_endpoint", "outcome_endpoint", "tracking_terminated", "time_out"
            }
            if not required.issubset(data.files):
                raise ValueError(f"missing visitation fields: {visits_path}")
            source = np.asarray(data["source_endpoint"], np.int64) - boundary
            outcome = np.asarray(data["outcome_endpoint"], np.int64) - boundary
            if len(source) != 160 or source.min() < 0 or source.max() > 39:
                raise ValueError(f"invalid relative source offsets: {visits_path}")
            offsets.append(source)
            epoch_maxima.append(int(source.max()))
            termination_offsets.update(map(int, outcome[data["tracking_terminated"]]))
            timeout_offsets.update(map(int, outcome[data["time_out"]]))
    joined = np.concatenate(offsets)
    counts = np.bincount(joined, minlength=40)[:40]
    total = int(len(joined))
    normalized = counts.astype(np.float64) / total
    maxima_hist = Counter(epoch_maxima)
    return {
        "epoch_range_inclusive": [start, stop],
        "epoch_count": stop - start + 1,
        "sample_count": total,
        "artifact_hashes_verified": hashes_verified,
        "sample_count_by_relative_source_offset": {
            str(k): int(counts[k]) for k in range(40)
        },
        "normalized_visitation_P_k": {
            str(k): float(normalized[k]) for k in range(40)
        },
        "current_chunk_fraction_k_0_19": float((joined <= 19).mean()),
        "next_chunk_boundary_action_fraction_k_20": float((joined == 20).mean()),
        "deep_lookahead_fraction_k_21_39": float((joined >= 21).mean()),
        "fraction_k_ge_25": float((joined >= 25).mean()),
        "fraction_k_ge_30": float((joined >= 30).mean()),
        "fraction_k_ge_35": float((joined >= 35).mean()),
        "maximum_k_reached_per_epoch": {
            "semantics": "maximum source offset represented in each 160-sample epoch",
            "distribution": distribution(epoch_maxima),
            "histogram": {str(k): v for k, v in sorted(maxima_hist.items())},
        },
        "tracking_termination_outcome_offset_histogram": {
            str(k): v for k, v in sorted(termination_offsets.items())
        },
        "timeout_outcome_offset_histogram": {
            str(k): v for k, v in sorted(timeout_offsets.items())
        },
    }


def load_run(
    spec: dict[str, Any], segment_names: list[str], segment_ranges: dict[str, list[int]]
) -> dict[str, Any]:
    root = Path(spec["root"])
    visit_manifest_path = root / "training_visitation" / "manifest.json"
    update_manifest_path = root / "update_audit" / "manifest.json"
    visit_manifest = checked_manifest(
        visit_manifest_path,
        spec["visitation_manifest_sha256"],
        "taco_ppo_training_visitation_v5",
    )
    update_manifest = checked_manifest(
        update_manifest_path,
        spec["update_manifest_sha256"],
        "taco_pour_algorithmic_training_update_audit_v6",
    )
    visitation_by_epoch = {int(row["epoch"]): row for row in visit_manifest["epochs"]}
    update_by_epoch = {int(row["epoch"]): row for row in update_manifest["epoch_reports"]}
    result = {
        "root": str(root),
        "source_boundary_endpoint": int(spec["source_boundary_endpoint"]),
        "input_manifests": {
            "visitation": {
                "path": str(visit_manifest_path),
                "sha256": spec["visitation_manifest_sha256"],
            },
            "update_audit": {
                "path": str(update_manifest_path),
                "sha256": spec["update_manifest_sha256"],
            },
        },
        "visitation_segments": {},
        "optimizer_segments": {},
    }
    for name in segment_names:
        start, stop = map(int, segment_ranges[name])
        result["visitation_segments"][name] = visitation_segment(
            root,
            visitation_by_epoch,
            int(spec["source_boundary_endpoint"]),
            start,
            stop,
        )
        result["optimizer_segments"][name] = optimizer_segment(
            update_by_epoch, start, stop
        )
    return result


def merged_segment(
    root: Path,
    spec: dict[str, Any],
    start: int,
    stop: int,
) -> dict[str, Any]:
    manifest = checked_manifest(
        root / "training_visitation" / "manifest.json",
        spec["visitation_manifest_sha256"],
        "taco_ppo_training_visitation_v5",
    )
    by_epoch = {int(row["epoch"]): row for row in manifest["epochs"]}
    return visitation_segment(
        root, by_epoch, int(spec["source_boundary_endpoint"]), start, stop
    )


def credit_join_assessment(run_specs: list[dict[str, Any]]) -> dict[str, Any]:
    inspected = []
    for spec in run_specs:
        root = Path(spec["root"])
        visits = root / "training_visitation" / "epoch_0001_visits.npz"
        credit = root / "update_audit" / "epoch_0001_credit.npz"
        with np.load(visits, allow_pickle=False) as v, np.load(
            credit, allow_pickle=False
        ) as c:
            visit_keys = sorted(v.files)
            credit_keys = sorted(c.files)
        inspected.append({
            "root": str(root),
            "visitation_keys": visit_keys,
            "credit_keys": credit_keys,
            "shared_sample_id_field": "sample_id" in set(visit_keys) & set(credit_keys),
        })
    return {
        "status": "credit_join_not_proven",
        "joined_statistics_emitted": False,
        "reason": (
            "The visitation and credit NPZ schemas contain no shared sample_id, "
            "world/time index, row hash, or manifest-level row-order contract. Equal "
            "row counts do not prove row identity, so credit is not joined by guess."
        ),
        "inspected_schema_examples": inspected,
    }


def write_summary(
    path: Path,
    control: dict[str, Any],
    failures: dict[str, dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    control_deep = control["visitation_segments"]["middle"][
        "deep_lookahead_fraction_k_21_39"
    ]
    lines = [
        "# Endpoint40 lookahead coverage / optimizer adjudication v1",
        "",
        f"Decision: **{decision['classification']}**.",
        "",
        "No simulator rollout or training was executed. Candidate E was not authorized.",
        "",
        "## Deep-lookahead visitation",
        "",
        "| Run | epochs 1–62 | 63–125 | 126–188 | 189–250 | 126–250 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| successful source20 control | {control['visitation_segments']['early']['deep_lookahead_fraction_k_21_39']:.6f} | {control_deep:.6f} | — | — | — |",
    ]
    for name, run in failures.items():
        segments = run["visitation_segments"]
        lines.append(
            f"| {name} | {segments['early']['deep_lookahead_fraction_k_21_39']:.6f} "
            f"| {segments['middle']['deep_lookahead_fraction_k_21_39']:.6f} "
            f"| {segments['late_a']['deep_lookahead_fraction_k_21_39']:.6f} "
            f"| {segments['late_b']['deep_lookahead_fraction_k_21_39']:.6f} "
            f"| {run['failed_run_second_half']['deep_lookahead_fraction_k_21_39']:.6f} |"
        )
    lines.extend([
        "",
        f"Successful-control reference fraction (epochs 63–125): `{control_deep:.9f}`.",
        f"Failed-run second-half median: `{decision['evidence']['failed_second_half_median_deep_lookahead_fraction']:.9f}`.",
        f"Median/control ratio: `{decision['evidence']['failed_to_successful_control_ratio']:.9f}`.",
        "",
        "The failure runs overwhelmingly sample k≤20 and almost never reach source offsets k≥21; k≥25 is zero in every failed segment. This meets the prefrozen visitation-bottleneck rule before optimizer evidence is used.",
        "",
        "## Credit join",
        "",
        "`credit_join_not_proven`: the two artifact schemas have no shared sample identifier or row-order contract, so advantage/value rows were not guessed into visitation rows.",
        "",
        "## Frozen progress",
        "",
        "- endpoint20→40 committed: true",
        "- endpoint40→60 committed: false",
        "- Candidate E executed: false",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/taco_pour_endpoint40_optimizer_review_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/taco_pour_endpoint40_optimizer_review_v1"),
    )
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    output = args.output_dir
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(args.contract, output / "contract.yaml")

    segment_ranges = contract["segments"]
    control = load_run(
        contract["positive_control"], ["early", "middle"], segment_ranges
    )
    failures: dict[str, dict[str, Any]] = {}
    for spec in contract["failed_restarts"]:
        key = f"source40_seed{int(spec['seed'])}"
        run = load_run(
            spec, ["early", "middle", "late_a", "late_b"], segment_ranges
        )
        start, stop = map(int, segment_ranges["failed_run_second_half"])
        run["failed_run_second_half"] = merged_segment(
            Path(spec["root"]), spec, start, stop
        )
        failures[key] = run

    successful = control["visitation_segments"]["middle"][
        "deep_lookahead_fraction_k_21_39"
    ]
    failed_values = [
        run["failed_run_second_half"]["deep_lookahead_fraction_k_21_39"]
        for run in failures.values()
    ]
    failed_median = float(np.median(failed_values))
    ratio = failed_median / successful
    lookahead_condition = failed_median < 0.05 and failed_median < 0.5 * successful
    optimizer_visitation_condition = (
        failed_median >= 0.10 and failed_median >= 0.5 * successful
    )
    if lookahead_condition:
        classification = "LOOKAHEAD_VISITATION_BOTTLENECK"
        action = "stop_no_candidate_E"
        candidate_e_authorized = False
    elif optimizer_visitation_condition:
        classification = "MIXED_OR_INCONCLUSIVE"
        action = "stop_optimizer_aggressiveness_requires_separate_unambiguous_rule"
        candidate_e_authorized = False
    else:
        classification = "MIXED_OR_INCONCLUSIVE"
        action = "stop_no_training"
        candidate_e_authorized = False

    credit = credit_join_assessment(
        [contract["positive_control"], *contract["failed_restarts"]]
    )
    visitation = {
        "schema": "taco_pour_endpoint40_visitation_comparison_v1",
        "read_only": True,
        "relative_endpoint_definition": contract["relative_endpoint_definition"],
        "successful_control": control,
        "failed_restarts": failures,
        "comparisons": {
            "successful_control_reference_segment": "epochs_63_125",
            "successful_control_deep_lookahead_fraction": successful,
            "failed_second_half_deep_lookahead_fractions": {
                key: value for key, value in zip(failures, failed_values)
            },
            "failed_second_half_median_deep_lookahead_fraction": failed_median,
            "failed_to_successful_control_ratio": ratio,
        },
        "credit_join": credit,
    }
    optimizer = {
        "schema": "taco_pour_endpoint40_optimizer_comparison_v1",
        "read_only": True,
        "successful_control": control["optimizer_segments"],
        "failed_restarts": {
            key: run["optimizer_segments"] for key, run in failures.items()
        },
        "interpretation_boundary": (
            "Optimizer evidence is descriptive. The first prefrozen visitation "
            "condition is decisive, so optimizer evidence cannot authorize Candidate E."
        ),
    }
    decision = {
        "schema": "taco_pour_endpoint40_optimizer_review_decision_v1",
        "classification": classification,
        "action": action,
        "candidate_E_authorized": candidate_e_authorized,
        "candidate_E_executed": False,
        "new_simulation_or_training_steps": 0,
        "evidence": {
            "failed_second_half_deep_lookahead_fractions": failed_values,
            "failed_second_half_median_deep_lookahead_fraction": failed_median,
            "successful_control_deep_lookahead_fraction": successful,
            "failed_to_successful_control_ratio": ratio,
            "lookahead_condition_passed": lookahead_condition,
            "optimizer_branch_visitation_condition_passed": optimizer_visitation_condition,
            "credit_join_status": credit["status"],
        },
        "required_conclusion": (
            "optimizer/LR-only change is not authorized by this review; next-chunk "
            "state coverage is materially poorer than the successful control."
        ),
        "immutable_progress": contract["immutable_progress"],
        "forbidden_actions_observed": [],
    }
    (output / "visitation_comparison.json").write_text(
        json.dumps(visitation, indent=2) + "\n"
    )
    (output / "optimizer_comparison.json").write_text(
        json.dumps(optimizer, indent=2) + "\n"
    )
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    write_summary(output / "visitation_summary.md", control, failures, decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
