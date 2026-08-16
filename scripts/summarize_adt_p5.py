#!/usr/bin/env python3
"""Summarize the ADT P5 GT-replacement matrix.

This is a benchmark-only aggregator. It reads existing base runs and P5
variant runs, normalizes both old and new ``p5_metrics.json`` formats, computes
rescue gains, and writes the three files the ADT upstream plan asks for:

``runs/adt_p5_gt_replacement_summary.json``
``runs/gt_replacement_matrix.csv``
``runs/bottleneck_ranking.csv``
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
CALIBRATION_SUMMARY_PATH = RUNS_ROOT / "adt_calibration_rescue_summary.json"
CALIBRATION_MATRIX_PATH = RUNS_ROOT / "calibration_rescue_matrix.csv"
VARIANTS = ("depth_gt", "mask_gt", "mesh_gt")
METRIC_KEYS = (
    "centroid_translation_error_m_median",
    "centroid_translation_error_m_p95",
    "add_s_m_median",
    "add_s_m_p95",
    "rotation_error_deg_median",
    "symmetry_aware_rotation_error_deg_median",
    "tracking_score",
)


def _number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return math.nan


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("metrics"), dict):
        return payload["metrics"]
    return payload


def _find_base_runs() -> list[Path]:
    runs = []
    for path in sorted(RUNS_ROOT.glob("adt_*_rgbobject*")):
        if "_p5_" in path.name:
            continue
        if (
            path.is_dir()
            and (path / "p5_metrics.json").is_file()
            and (path / "object_tracking/foundationpose_raw.npz").is_file()
        ):
            runs.append(path)
    return runs


def _load_variant(base: Path, variant: str | None) -> dict[str, Any] | None:
    run_dir = base if variant is None else base.parent / f"{base.name}_p5_{variant}"
    path = run_dir / "p5_metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = _normalize_payload(payload)
    metrics["run_dir"] = str(run_dir)
    if variant is not None:
        gate_path = run_dir / "object_tracking/tracking_gate.json"
        if gate_path.is_file():
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            metrics["gate_accepted"] = bool(gate.get("accepted", False))
        else:
            metrics["gate_accepted"] = None
    return metrics


def _rescue(base_metrics: dict[str, Any], variant_metrics: dict[str, Any]) -> dict[str, float]:
    return {
        f"{key.replace('_median', '_rescue')}": _number(base_metrics.get(key))
        - _number(variant_metrics.get(key))
        for key in (
            "centroid_translation_error_m_median",
            "add_s_m_median",
            "symmetry_aware_rotation_error_deg_median",
        )
    }


def _calibration_module_row() -> dict[str, Any] | None:
    if not CALIBRATION_SUMMARY_PATH.is_file():
        return None
    payload = json.loads(CALIBRATION_SUMMARY_PATH.read_text(encoding="utf-8"))
    records = payload.get("rows", [])
    if not records:
        return None
    rescues: list[float] = []
    failed = 0
    for record in records:
        raw = record.get("raw") or {}
        aligned = record.get("aligned") or {}
        raw_median = _number((raw.get("centroid_translation_error_m") or {}).get("median"))
        aligned_median = _number((aligned.get("centroid_translation_error_m") or {}).get("median"))
        if math.isfinite(raw_median) and math.isfinite(aligned_median):
            rescues.append(raw_median - aligned_median)
        else:
            failed += 1
    median_rescue = float(sorted(rescues)[len(rescues) // 2]) if rescues else math.nan
    positive = float(sum(value > 0.001 for value in rescues) / len(rescues)) if rescues else 0.0
    return {
        "module": "calibration_gt",
        "available_samples": len(records),
        "failure_count": failed,
        "median_centroid_rescue_mm": 1000.0 * median_rescue,
        "median_add_s_rescue_mm": math.nan,
        "median_symmetry_rotation_rescue_deg": math.nan,
        "positive_centroid_rescue_fraction": positive,
        "priority_centroid": 1000.0 * median_rescue * positive,
        "note": "full-upstream stereo branch; raw FP vs GT-calibration+validate_only sequence",
    }


def main() -> int:
    rows: list[dict[str, Any]] = []
    for base in _find_base_runs():
        base_metrics = _load_variant(base, None)
        if base_metrics is None:
            continue
        row: dict[str, Any] = {
            "run": base.name,
            "run_dir": str(base),
            "f0": base_metrics,
            "variants": {},
            "rescue_gain": {},
            "failed_variants": [],
        }
        for variant in VARIANTS:
            metrics = _load_variant(base, variant)
            if metrics is None:
                continue
            row["variants"][variant] = metrics
            if metrics.get("status") == "failed":
                row["failed_variants"].append(
                    {"variant": variant, "error": metrics.get("error")}
                )
                continue
            row["rescue_gain"][variant] = _rescue(base_metrics, metrics)
        rows.append(row)

    summary = {
        "schema_version": "1.0",
        "stage": "P5 GT Replacement",
        "base_run_count": len(rows),
        "variants": list(VARIANTS),
        "rows": rows,
    }
    summary_path = RUNS_ROOT / "adt_p5_gt_replacement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    matrix_path = RUNS_ROOT / "gt_replacement_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        header = ["run", "variant", "status"]
        header.extend(METRIC_KEYS)
        header.extend(["gate_accepted", "error"])
        writer.writerow(header)
        for row in rows:
            for variant, metrics in [("f0", row["f0"]), *row["variants"].items()]:
                writer.writerow(
                    [row["run"], variant, metrics.get("status", "ok")]
                    + [_number(metrics.get(key)) for key in METRIC_KEYS]
                    + [metrics.get("gate_accepted", None), metrics.get("error", "")]
                )

    module_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        values = []
        failures = 0
        for row in rows:
            if variant in row["rescue_gain"]:
                gain = row["rescue_gain"][variant]
                values.append(
                    {
                        "run": row["run"],
                        "centroid_translation_error_m_rescue": gain.get(
                            "centroid_translation_error_m_rescue", math.nan
                        ),
                        "add_s_m_rescue": gain.get("add_s_m_rescue", math.nan),
                        "symmetry_aware_rotation_error_deg_rescue": gain.get(
                            "symmetry_aware_rotation_error_deg_rescue", math.nan
                        ),
                    }
                )
            if any(item.get("variant") == variant for item in row["failed_variants"]):
                failures += 1
        if not values and failures == 0:
            continue
        def median_gain(key: str) -> float:
            vals = [value[key] for value in values if math.isfinite(value[key])]
            if not vals:
                return math.nan
            return float(sorted(vals)[len(vals) // 2])

        centroid_median = median_gain("centroid_translation_error_m_rescue")
        add_s_median = median_gain("add_s_m_rescue")
        rotation_median = median_gain("symmetry_aware_rotation_error_deg_rescue")
        positive_centroid = float(
            sum(value["centroid_translation_error_m_rescue"] > 0.001 for value in values)
            / len(values)
        ) if values else 0.0
        module_rows.append(
            {
                "module": variant,
                "available_samples": len(values) + failures,
                "failure_count": failures,
                "median_centroid_rescue_mm": 1000.0 * centroid_median,
                "median_add_s_rescue_mm": 1000.0 * add_s_median,
                "median_symmetry_rotation_rescue_deg": rotation_median,
                "positive_centroid_rescue_fraction": positive_centroid,
                "priority_centroid": 1000.0 * centroid_median * positive_centroid,
                "note": "FoundationPose RGB object branch",
            }
        )
    calibration_row = _calibration_module_row()
    if calibration_row is not None:
        module_rows.append(calibration_row)
    module_rows.sort(key=lambda row: _number(row.get("priority_centroid")), reverse=True)
    ranking_path = RUNS_ROOT / "bottleneck_ranking.csv"
    fieldnames = []
    for item in module_rows:
        for key in item:
            if key not in fieldnames:
                fieldnames.append(key)
    with ranking_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if module_rows:
            writer.writeheader()
            writer.writerows(module_rows)

    print(f"summary -> {summary_path}")
    print(f"matrix   -> {matrix_path}")
    print(f"ranking  -> {ranking_path}")
    print(json.dumps(module_rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
