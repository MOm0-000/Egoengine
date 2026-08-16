#!/usr/bin/env python3
"""Aggregate Phase B/C ADT ablation artifacts into one summary JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(value: Any, digits: int = 3) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def main() -> int:
    rows = []
    for run_dir in sorted(RUNS_ROOT.glob("*_rgbobject")):
        name = run_dir.name
        sam3d = _load_json(run_dir / "mesh_proposals/mesh_ranking.json")
        omvg = _load_json(run_dir / "mesh_proposals/omvg_mesh_ranking.json")
        depth = _load_json(run_dir / "depth_roi_refined/depth_metrics.json")
        fp_final = _load_json(run_dir / "object_tracking/tracking_metrics.json")
        fp_gate = _load_json(run_dir / "object_tracking/tracking_gate.json")
        fp_omvg_candidate = _load_json(run_dir / "object_tracking/foundationpose_candidates/omvg_00001/tracking_metrics.json")
        fp_sam3d_candidates = sorted(
            (run_dir / "object_tracking/foundationpose_candidates").glob("proposal_*/tracking_metrics.json")
        ) if (run_dir / "object_tracking/foundationpose_candidates").exists() else []

        row: dict[str, Any] = {"run": name}
        row["sam3d_qualified_count"] = sam3d.get("qualified_count") if sam3d else None
        row["sam3d_success"] = sam3d.get("success") if sam3d else None
        row["omvg_qualified_count"] = omvg.get("qualified_count") if omvg else None
        row["omvg_static_iou"] = _fmt(omvg["proposals"][0]["fit"]["silhouette_iou"]) if omvg else None
        row["omvg_static_relative_depth"] = _fmt(omvg["proposals"][0]["fit"]["relative_depth_residual"]) if omvg else None

        # Current final tracking file may be from O0 or O1; keep it labelled.
        row["fp_final_valid_rate"] = _fmt(fp_final["metrics"]["valid_rate"]) if fp_final else None
        row["fp_final_iou"] = _fmt(fp_final["metrics"]["mean_mask_iou"]) if fp_final else None
        row["fp_final_rel_depth"] = _fmt(fp_final["metrics"]["median_relative_depth_residual"]) if fp_final else None
        row["fp_final_tracking_score"] = _fmt(fp_final["metrics"]["tracking_score"]) if fp_final else None
        row["fp_final_accepted"] = fp_gate.get("accepted") if fp_gate else None

        # SAM3D candidate best metrics (if any were generated).
        sam3d_metrics = []
        for path in fp_sam3d_candidates:
            data = _load_json(path)
            if data and "metrics" in data:
                sam3d_metrics.append(data["metrics"])
        if sam3d_metrics:
            best_sam3d = max(sam3d_metrics, key=lambda m: float(m.get("tracking_score", -1)))
            row["sam3d_best_tracking_score"] = _fmt(best_sam3d.get("tracking_score"))
            row["sam3d_best_iou"] = _fmt(best_sam3d.get("mean_mask_iou"))
            row["sam3d_best_rel_depth"] = _fmt(best_sam3d.get("median_relative_depth_residual"))
        else:
            row["sam3d_best_tracking_score"] = None
            row["sam3d_best_iou"] = None
            row["sam3d_best_rel_depth"] = None

        if fp_omvg_candidate and "metrics" in fp_omvg_candidate:
            m = fp_omvg_candidate["metrics"]
            row["omvg_tracking_valid_rate"] = _fmt(m.get("valid_rate"))
            row["omvg_tracking_iou"] = _fmt(m.get("mean_mask_iou"))
            row["omvg_tracking_rel_depth"] = _fmt(m.get("median_relative_depth_residual"))
            row["omvg_tracking_score"] = _fmt(m.get("tracking_score"))
        else:
            row["omvg_tracking_valid_rate"] = None
            row["omvg_tracking_iou"] = None
            row["omvg_tracking_rel_depth"] = None
            row["omvg_tracking_score"] = None

        if depth:
            row["depth"] = {variant: {k: _fmt(v) for k, v in metrics.items()} for variant, metrics in depth["variants"].items()}
        else:
            row["depth"] = None
        rows.append(row)

    output = RUNS_ROOT / "adt_phase_bc_ablation_summary.json"
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
