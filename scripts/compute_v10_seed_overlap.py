#!/usr/bin/env python3
"""Compute WiLoR/EgoForce seed overlap from v9 outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
V9 = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v9"
OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v10"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_rows = list(csv.DictReader((V9 / "seed_recall_38.csv").open(newline="")))
    view_rows = list(csv.DictReader((V9 / "detection_2d_38.csv").open(newline="")))
    view_by_key = {
        (r["clip"], int(r["frame"]), r["camera"]): r for r in view_rows
    }

    rows = []
    counts = Counter()
    by_type = {
        "edge_hand": Counter(),
        "very_small_hand": Counter(),
        "center_hand": Counter(),
        "other": Counter(),
    }
    for r in seed_rows:
        w = r["b0_at_least_one"] == "True"
        e = r["at_least_one"] == "True"
        w_both = r["b0_both"] == "True"
        e_both = r["both"] == "True"
        union_at_least = w or e
        union_both = w_both or e_both
        # Frame-level geometric type from the two views.
        types = []
        for cam in ("left", "right"):
            vr = view_by_key[(r["clip"], int(r["frame"]), cam)]
            diag = float(vr["gt_bbox_diag_px"])
            dist = float(vr["gt_wrist_distance_to_image_boundary_px"])
            if dist < 60:
                types.append("edge_hand")
            elif diag < 60:
                types.append("very_small_hand")
            else:
                types.append("center_hand")
        ftype = types[0] if types[0] == types[1] else "edge_hand"
        rows.append(
            {
                "clip": r["clip"],
                "frame": r["frame"],
                "wiLor_only": w and not e,
                "egoforce_only": e and not w,
                "both_success": w and e,
                "neither_success": not w and not e,
                "union_at_least_one": union_at_least,
                "wiLor_both": w_both,
                "egoforce_both": e_both,
                "union_both": union_both,
                "failure_type": ftype,
            }
        )
        counts["wiLor_only"] += w and not e
        counts["egoforce_only"] += e and not w
        counts["both_success"] += w and e
        counts["neither_success"] += not w and not e
        counts["union_at_least_one"] += union_at_least
        counts["union_both"] += union_both
        by_type[ftype]["wiLor"] += w
        by_type[ftype]["egoforce"] += e
        by_type[ftype]["union_at_least"] += union_at_least

    summary = {
        "n_frames": len(rows),
        "wiLor_at_least_one": int(counts["wiLor_only"] + counts["both_success"]),
        "egoforce_at_least_one": int(counts["egoforce_only"] + counts["both_success"]),
        "wiLor_only": int(counts["wiLor_only"]),
        "egoforce_only": int(counts["egoforce_only"]),
        "both_success": int(counts["both_success"]),
        "neither_success": int(counts["neither_success"]),
        "union_at_least_one": int(counts["union_at_least_one"]),
        "union_both": int(counts["union_both"]),
        "by_failure_type": {k: dict(v) for k, v in by_type.items()},
    }

    with (OUT / "seed_overlap_38.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip",
                "frame",
                "failure_type",
                "wiLor_only",
                "egoforce_only",
                "both_success",
                "neither_success",
                "union_at_least_one",
                "wiLor_both",
                "egoforce_both",
                "union_both",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "seed_overlap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
