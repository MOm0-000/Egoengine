#!/usr/bin/env python3
"""Strict sequence attempt over all fixed ADT rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_adt_sequence_benchmark as seq  # noqa: E402

STATE_PATH = REPO_ROOT / "runs/adt_sequence_benchmark_all_state.json"
OUTPUT_PATH = REPO_ROOT / "runs/adt_sequence_benchmark_all_summary.json"
MATRIX_PATH = REPO_ROOT / "runs/sequence_benchmark_all_matrix.csv"


def _best_source(row: dict[str, Any]) -> Path:
    source = Path(row["run_dir"])
    if not source.exists():
        raise FileNotFoundError(source)
    return source


def _record(row: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    source = _best_source(row)
    clone = seq.RUNS_ROOT / f"{source.name}_p6_sequence_all"
    seq._clone_run(source, clone)
    returncode, log = seq._run_sequence(clone)
    sequence = seq._sequence_record(clone, returncode, Path(log))
    record: dict[str, Any] = {
        "sequence_name": row["sequence"],
        "prototype": row["prototype"],
        "object_uid": row["object_uid"],
        "window": row["window"],
        "source_run": str(source),
        "clone_run": str(clone),
        "sequence_returncode": returncode,
        "sequence_optimization": sequence,
        "scale_delta": None,
        "raw_scale": None,
        "aligned_scale": None,
        "violations": [],
    }
    if (source / "object_tracking/selected_mesh.json").is_file():
        selected = json.loads(
            (source / "object_tracking/selected_mesh.json").read_text(encoding="utf-8")
        )
        record["raw_scale"] = float(selected.get("scale_to_m", math.nan))
    if sequence.get("has_aligned"):
        try:
            seq_dir = seq.ADT_ROOT / row["sequence"]
            prepared_dir = seq._prepared_dir(row)
            _, T_world_camera, _ = seq._gt_camera_and_object(seq_dir, prepared_dir, str(row["object_uid"]))
            aligned_transforms, aligned_valid, aligned_scale = seq._aligned_camera_transforms(clone, T_world_camera)
            record["aligned_valid_rate"] = float(aligned_valid.mean()) if aligned_valid.size else 0.0
            record["aligned_scale"] = float(aligned_scale)
            raw_scale = record.get("raw_scale")
            if math.isfinite(raw_scale) and raw_scale > 0:
                record["scale_delta"] = (float(aligned_scale) - raw_scale) / raw_scale
        except Exception as exc:
            record["aligned_evaluation_error"] = f"{type(exc).__name__}: {exc}"
    if sequence.get("status") == "failed_before_export":
        record["violations"].append("sequence_failed_before_export")
    if sequence.get("status") == "export_qc_failed":
        record["violations"].append("sequence_export_qc_failed")
    if sequence.get("has_aligned") and record.get("scale_delta") is not None and abs(record["scale_delta"]) > 1e-6:
        record["violations"].append(f"metric_scale_rewritten_{record['scale_delta']:+.6g}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = json.loads(seq.SUMMARY_PATH.read_text(encoding="utf-8"))
    state = {"schema_version": "1.0", "rows": {}}
    if STATE_PATH.is_file() and not args.force:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    records = []
    for row in rows:
        key = f"{row['sequence']}__{row['prototype']}__f{row['window'][0]}_{row['window'][1]}"
        if key in state.get("rows", {}) and not args.force:
            record = state["rows"][key]
        else:
            try:
                record = _record(row, state)
            except Exception as exc:
                record = {
                    "sequence_name": row["sequence"],
                    "prototype": row["prototype"],
                    "object_uid": row["object_uid"],
                    "window": row["window"],
                    "source_run": row["run_dir"],
                    "sequence_optimization": {"status": "driver_error", "reason": f"{type(exc).__name__}: {exc}"},
                    "violations": ["driver_error"],
                }
            state["rows"][key] = record
        records.append(record)
        STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUTPUT_PATH.write_text(json.dumps({"schema_version": "1.0", "candidate_count": len(records), "rows": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with MATRIX_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "status", "reason", "aligned_valid_rate", "aligned_scale", "violations"])
        for record in records:
            optimization = record.get("sequence_optimization") or {}
            writer.writerow([
                Path(record.get("source_run", "")).name,
                optimization.get("status"),
                optimization.get("reason"),
                record.get("aligned_valid_rate", math.nan),
                record.get("aligned_scale", math.nan),
                "|".join(record.get("violations", [])),
            ])
    print(json.dumps(records, indent=2, ensure_ascii=False)[:20000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
