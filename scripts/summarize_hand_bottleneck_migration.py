#!/usr/bin/env python3
"""Summarize the revised Hand bottleneck migration check on the fixed ADT rows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
OUTPUT_ROOT = RUNS / "hot3d_hand_diagnosis"


def _rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("rows", []))
    if isinstance(data, list):
        return list(data)
    raise TypeError(f"unexpected summary shape: {path}")


def _accepted_from_dict_summary(path: Path) -> dict[str, bool | None]:
    result: dict[str, bool | None] = {}
    for row in _rows(path):
        key = Path(row.get("run_dir", "")).name
        record = row.get("record") or {}
        if record.get("status") != "evaluated":
            result[key] = None
            continue
        if "accepted_any_hand" in record:
            value = bool(record.get("accepted_any_hand"))
        else:
            metrics = record.get("metrics") or {}
            value = bool(metrics.get("accepted_any_hand", False))
        result[key] = value
    return result


def _accepted_from_list_summary(path: Path) -> dict[str, bool | None]:
    result: dict[str, bool | None] = {}
    for row in _rows(path):
        key = Path(row.get("run_dir", "")).name
        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict) or "accepted_any_hand" not in metrics:
            result[key] = None
            continue
        result[key] = bool(metrics.get("accepted_any_hand", False))
    return result


def _method_row(
    method: str,
    description: str,
    accepted: dict[str, bool | None],
    all_keys: list[str],
) -> dict[str, Any]:
    evaluated = [key for key in all_keys if key in accepted and accepted[key] is not None]
    accepted_count = sum(1 for key in evaluated if accepted[key] is True)
    return {
        "method": method,
        "description": description,
        "evaluated_runs": len(evaluated),
        "accepted_any_hand": accepted_count,
        "accepted_rate": (accepted_count / len(evaluated)) if evaluated else None,
        "per_run": {
            key: {
                "accepted_any_hand": accepted.get(key),
            }
            for key in all_keys
        },
    }


def _audit_side_rows() -> list[dict[str, Any]]:
    data = json.loads(
        (RUNS / "adt_hand_observation_audit_summary.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    for entry in data:
        run_name = Path(entry["run_dir"]).name
        for model, payload in entry.get("models", {}).items():
            if payload.get("status") != "evaluated":
                rows.append(
                    {
                        "run": run_name,
                        "model": model,
                        "side": None,
                        "left_detection_rate": None,
                        "right_detection_rate": None,
                        "both_detection_rate": None,
                        "status": payload.get("status"),
                    }
                )
                continue
            for side, metrics in payload.get("per_side", {}).items():
                rows.append(
                    {
                        "run": run_name,
                        "model": model,
                        "side": side,
                        "left_detection_rate": metrics.get("left_detection_rate"),
                        "right_detection_rate": metrics.get("right_detection_rate"),
                        "both_detection_rate": metrics.get("both_detection_rate"),
                        "left_required_rate": metrics.get("left_required_rate"),
                        "right_required_rate": metrics.get("right_required_rate"),
                        "vertical_disparity_p95_px": metrics.get(
                            "vertical_disparity_p95_px"
                        ),
                        "status": "evaluated",
                    }
                )
    return rows


def _sequence_rows() -> list[dict[str, Any]]:
    data = json.loads(
        (RUNS / "adt_sequence_benchmark_all_summary.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    for record in data.get("rows", []):
        optimization = record.get("sequence_optimization") or {}
        rows.append(
            {
                "run": Path(record.get("source_run", "")).name,
                "sequence_name": record.get("sequence_name"),
                "prototype": record.get("prototype"),
                "status": optimization.get("status"),
                "reason": optimization.get("reason"),
                "aligned_valid_rate": record.get("aligned_valid_rate"),
                "aligned_scale": record.get("aligned_scale"),
                "raw_scale": record.get("raw_scale"),
                "scale_delta": record.get("scale_delta"),
                "violations": record.get("violations", []),
            }
        )
    return rows


def main() -> int:
    audit_rows = _audit_side_rows()
    sequence_rows = _sequence_rows()
    audit_run_names = sorted({row["run"] for row in audit_rows})

    method_specs = [
        (
            "H0",
            "WiLoR + stereo triangulation",
            _accepted_from_dict_summary(RUNS / "adt_stereo_hand_benchmark_summary.json"),
        ),
        (
            "H1",
            "HaMeR + stereo fusion",
            _accepted_from_dict_summary(RUNS / "adt_hamer_hand_benchmark_summary.json"),
        ),
        (
            "H2",
            "HaMeR + calibrated stereo BA",
            _accepted_from_dict_summary(RUNS / "adt_hamer_hand_ba_benchmark_summary.json"),
        ),
        (
            "H3",
            "UmeTrack",
            _accepted_from_list_summary(RUNS / "adt_umetrack_hand_benchmark_summary.json"),
        ),
        (
            "H4",
            "POEM / POEM-v2",
            _accepted_from_list_summary(RUNS / "adt_poem_hand_benchmark_summary.json"),
        ),
        (
            "H5",
            "Best proposal + stereo MANO/depth/bone BA",
            _accepted_from_dict_summary(RUNS / "adt_hamer_hand_h5_benchmark_summary.json"),
        ),
    ]

    method_rows = [
        _method_row(method, description, accepted, audit_run_names)
        for method, description, accepted in method_specs
    ]
    migration = {
        "schema_version": "1.0",
        "config": {
            "object_perception": "Phase 0 final mask branch",
            "stereo_depth": "FoundationStereo",
            "object_reconstruction": "O5 hybrid candidate pool",
            "sequence": "strict validate_only",
            "fixed_adt_rows": "runs/adt_depth_benchmark_summary.json",
        },
        "hand_methods": method_rows,
        "observation_audit": audit_rows,
        "sequence_regression": sequence_rows,
        "migration_gate": {
            "accepted_any_hand_improved": any(
                row["accepted_any_hand"] > 1 for row in method_rows
            ),
            "work_seq107_shared_metric_hand_depth_verified": any(
                row["run"].startswith("adt_Apartment_release_work_seq107_M1292_BookDeepLearning")
                and row["status"] == "export_qc_passed"
                for row in sequence_rows
            ),
            "metric_scale_rewritten_count": sum(
                1
                for row in sequence_rows
                for violation in row["violations"]
                if str(violation).startswith("metric_scale_rewritten_")
            ),
        },
    }
    summary_path = OUTPUT_ROOT / "adt_migration_summary.json"
    summary_path.write_text(
        json.dumps(migration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    matrix_path = OUTPUT_ROOT / "adt_migration_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "run",
                "model",
                "side",
                "left_detection_rate",
                "right_detection_rate",
                "both_detection_rate",
                "left_required_rate",
                "right_required_rate",
                "vertical_disparity_p95_px",
                "sequence_status",
                "sequence_reason",
                "scale_delta",
            ]
        )
        sequence_by_run = {row["run"]: row for row in sequence_rows}
        for row in audit_rows:
            sequence = sequence_by_run.get(row["run"], {})
            writer.writerow(
                [
                    row["run"],
                    row["model"],
                    row["side"],
                    row.get("left_detection_rate"),
                    row.get("right_detection_rate"),
                    row.get("both_detection_rate"),
                    row.get("left_required_rate"),
                    row.get("right_required_rate"),
                    row.get("vertical_disparity_p95_px"),
                    sequence.get("status"),
                    sequence.get("reason"),
                    sequence.get("scale_delta"),
                ]
            )
    print(f"summary -> {summary_path}")
    print(f"matrix -> {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
