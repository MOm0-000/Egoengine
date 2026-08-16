#!/usr/bin/env python3
"""Final 10-sample Phase 0 matrix: S1 primary + S2 multianchor fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_adt_phase0_sam3_ablation as phase0  # noqa: E402
import run_adt_phase0_sam3_s2 as s2  # noqa: E402

OUTPUT_JSON = RUNS_ROOT / "adt_phase0_final_10_summary.json"
OUTPUT_CSV = RUNS_ROOT / "adt_phase0_final_10_matrix.csv"


def _read_s2() -> dict[tuple[str, str, int, int], dict[str, Any]]:
    path = RUNS_ROOT / "adt_phase0_sam3_s2_multianchor_summary.json"
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (row["prototype"], row["sequence"], row["window_start"], row["window_end"]): row
        for row in rows
    }


def _main() -> int:
    rows = phase0._rows()
    s2_by_key = _read_s2()
    final_rows: list[dict[str, Any]] = []
    for row in rows:
        run_dir = phase0._object_run(row)
        if run_dir is None:
            continue
        key = (row["prototype"], row["sequence"], row["window"][0], row["window"][1])
        s0 = phase0._mask_metrics(run_dir)
        s1 = phase0._mask_metrics_s1(run_dir)
        s2_row = s2_by_key.get(key)
        s2_metrics = s2_row.get("s2_multianchor") if s2_row else None
        if s2_metrics is None:
            merged_dir = run_dir / "segmentation_s2_multianchor_merged"
            if (merged_dir / "object_masks.npz").is_file():
                s2_metrics = s2._mask_metrics_for(run_dir, "segmentation_s2_multianchor_merged")
        use_s2 = bool(s2_metrics and s2_metrics.get("valid_rate") is not None and float(s2_metrics.get("valid_rate") or 0.0) > float(s1.get("valid_rate") or 0.0))
        final = s2_metrics if use_s2 else s1
        final_rows.append({
            "prototype": row["prototype"],
            "sequence": row["sequence"],
            "window_start": row["window"][0],
            "window_end": row["window"][1],
            "run_dir": str(run_dir),
            "s0_valid_rate": s0.get("valid_rate"),
            "s0_mean_iou": s0.get("mean_iou"),
            "s1_valid_rate": s1.get("valid_rate"),
            "s1_mean_iou": s1.get("mean_iou"),
            "s2_valid_rate": s2_metrics.get("valid_rate") if s2_metrics else None,
            "s2_mean_iou": s2_metrics.get("mean_iou") if s2_metrics else None,
            "final_pipeline": "s2_multianchor" if use_s2 else "s1",
            "final_valid_rate": final.get("valid_rate"),
            "final_mean_iou": final.get("mean_iou"),
            "final_median_iou": final.get("median_iou"),
            "final_frames_with_mask": final.get("frames_with_mask"),
        })
    valid_rates = np.asarray([float(row["final_valid_rate"]) for row in final_rows], dtype=float)
    ious = np.asarray([float(row["final_mean_iou"]) for row in final_rows], dtype=float)
    summary = {
        "rows": final_rows,
        "aggregate": {
            "sample_count": int(len(final_rows)),
            "macro_valid_rate": float(valid_rates.mean()) if valid_rates.size else float("nan"),
            "macro_mean_iou": float(ious.mean()) if ious.size else float("nan"),
            "frames_with_mask": int(sum(int(row["final_frames_with_mask"]) for row in final_rows)),
            "frame_count": int(30 * len(final_rows)),
            "overall_mask_coverage": float(valid_rates.mean()) if valid_rates.size else float("nan"),
            "failed_rescue": {
                row["prototype"]: {
                    "s0_valid_rate": row["s0_valid_rate"],
                    "s1_valid_rate": row["s1_valid_rate"],
                    "s2_valid_rate": row["s2_valid_rate"],
                    "final_valid_rate": row["final_valid_rate"],
                    "final_mean_iou": row["final_mean_iou"],
                }
                for row in final_rows if row["prototype"] in {"BlackCeramicMug", "Flask", "StepStool"}
            },
        },
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    headers = [
        "prototype", "sequence", "window_start", "window_end",
        "s0_valid_rate", "s0_mean_iou", "s1_valid_rate", "s1_mean_iou",
        "s2_valid_rate", "s2_mean_iou", "final_pipeline", "final_valid_rate", "final_mean_iou", "final_median_iou",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in final_rows:
            handle.write(",".join(str(row.get(header, "")) for header in headers) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
