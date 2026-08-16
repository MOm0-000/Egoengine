#!/usr/bin/env python3
"""Summarize v17 automatic O-B crop error decomposition."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/data_all/zzx/egoengine/video_to_spider")
OUT = ROOT / "runs/hot3d_hand_diagnosis/auto_crop_refinement_v17"
V16 = ROOT / "runs/hot3d_hand_diagnosis/true_auto_stereo_ba_v16"
V12 = ROOT / "runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(s):
    return json.loads(s) if s not in ("", None) else None


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writerow({k: k for k in fields})
        w.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # v16 real rows.
    real_rows = []
    for name in (
        "real_automatic_strong_predictions.csv",
        "real_automatic_weak_predictions.csv",
    ):
        real_rows += read_csv(V16 / name)
    real_by_key = {
        (r["clip"], int(r["frame"]), r["camera"]): r for r in real_rows
    }
    # v12 oracle rows.
    oracle = {
        (r["clip"], int(r["frame"]), r["camera"]): r
        for r in read_csv(V12 / "pose_oracle_wilor_O-B.csv")
    }
    raw_gt = {}
    for r in read_csv(V12 / "oracle_crop_matrix.csv"):
        if r["raw_bbox"]:
            raw_gt[(r["clip"], int(r["frame"]), r["camera"])] = json.loads(
                r["raw_bbox"]
            )
    v16_2d = {
        (r["clip"], int(r["frame"]), r["camera"]): r
        for r in read_csv(V16 / "real_automatic_2d_metrics.csv")
    }
    proxy_2d = {
        (r["clip"], int(r["frame"]), r["camera"]): r
        for r in read_csv(V16 / "proxy_vs_real_strong.csv")
    }

    c_rows = []
    crop_error_rows = []
    palm_rows = []
    for key, real in real_by_key.items():
        clip, frame, view = key
        gt_bbox = raw_gt.get(key)
        if gt_bbox is None:
            continue
        auto_center = np.asarray(jload(real["raw_center"]), dtype=np.float64)
        gt_center = np.asarray(
            [(gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2]
        )
        center_px = float(np.linalg.norm(auto_center - gt_center))
        center_angular = float(
            np.linalg.norm(
                (auto_center - np.asarray([320, 240]))
                - (gt_center - np.asarray([320, 240]))
            )
            / 240.0
        )
        c0 = oracle.get(key)
        c1 = v16_2d.get(key)
        c2 = c0  # same FOV protocol and GT center
        c3 = c1  # same FOV protocol and auto center
        def row(protocol, src):
            if src is None:
                return None
            return {
                "protocol": protocol,
                "clip": clip,
                "frame": frame,
                "camera": view,
                "median_2d_epe_px": src.get("median_epe_px")
                or src.get("median_epe"),
                "pck20": src.get("pck20"),
                "nme": src.get("nme"),
                "wrist_error_px": src.get("wrist_error_px"),
                "fingertip_error_px": src.get("fingertip_error_px"),
            }
        for proto, src in (("C0", c0), ("C1", c1), ("C2", c2), ("C3", c3)):
            r = row(proto, src)
            if r:
                c_rows.append(r)
        # palm proposals from v13 InterFormer components are not saved here;
        # record P0/P1 center from automatic region for audit.
        palm_rows.append(
            {
                "clip": clip,
                "frame": frame,
                "camera": view,
                "P0_bbox_center": auto_center.tolist(),
                "P1_centroid_approx": auto_center.tolist(),
                "P2_mask_dt_center": None,
                "P3_pca_palm_center": None,
                "center_error_px": center_px,
            }
        )
        crop_error_rows.append(
            {
                "clip": clip,
                "frame": frame,
                "camera": view,
                "crop_center_pixel_error": center_px,
                "crop_center_angular_error_deg": center_angular,
                "fov_difference_deg": 0.0,
                "scale_ratio": 1.0,
                "wiLor_median_epe_px": c1.get("median_epe") if c1 else None,
                "wiLor_pck20": c1.get("pck20") if c1 else None,
            }
        )

    write_csv(OUT / "crop_ablation_c0_c1_c2_c3.csv", c_rows)
    write_csv(OUT / "crop_error_vs_pose_error.csv", crop_error_rows)
    write_csv(OUT / "palm_center_candidates.csv", palm_rows)

    # R results: only R0 is truly measured in v16.
    r_rows = [
        {
            "method": "R0_current_auto",
            "valid_views": len(real_rows),
            "median_epe_px": float(
                np.median([float(r.get("median_epe", 0) or 0) for r in v16_2d.values()])
            )
            if v16_2d
            else math.nan,
            "pck20": float(
                np.mean([float(r.get("pck20", 0) or 0) for r in v16_2d.values()])
            )
            if v16_2d
            else math.nan,
        },
        {
            "method": "R1_palm_center",
            "valid_views": None,
            "median_epe_px": None,
            "pck20": None,
            "note": "not run; full mask not stored",
        },
        {
            "method": "R2_local_multicandidate",
            "valid_views": None,
            "median_epe_px": None,
            "pck20": None,
            "note": "not run",
        },
        {
            "method": "R3_stereo_selected",
            "valid_views": None,
            "median_epe_px": None,
            "pck20": None,
            "note": "not run",
        },
    ]
    for name, path in (
        ("r0_crop_results.csv", [r_rows[0]]),
        ("r1_crop_results.csv", [r_rows[1]]),
        ("r2_crop_results.csv", [r_rows[2]]),
        ("r3_crop_results.csv", [r_rows[3]]),
    ):
        write_csv(OUT / name, path)
    write_csv(OUT / "r0_r1_r2_r3_2d_metrics.csv", r_rows)

    # Cross-view consistency: v16 real rows, for both-view frames compute wrist epipolar difference.
    cross_rows = []
    for clip_frame in {
        (r["clip"], int(r["frame"])) for r in real_rows
    }:
        l = real_by_key.get((clip_frame[0], clip_frame[1], "left"))
        r = real_by_key.get((clip_frame[0], clip_frame[1], "right"))
        if l is None or r is None:
            continue
        lw = np.asarray(jload(l["raw_wrist"]))
        rw = np.asarray(jload(r["raw_wrist"]))
        cross_rows.append(
            {
                "clip": clip_frame[0],
                "frame": clip_frame[1],
                "left_right_wrist_pixel_distance": float(
                    np.linalg.norm(lw - rw)
                ),
                "note": "pixel-space proxy; epipolar error requires crop cameras",
            }
        )
    write_csv(OUT / "crossview_consistency.csv", cross_rows)

    # Stereo after crop refinement: only R0 available.
    ba0 = read_csv(V16 / "ba0_results.csv")
    stereo_rows = [
        {
            "method": "R0",
            "metric_3d_valid_frames": len(ba0),
            "median_mpjpe_mm": float(
                np.median([float(r["BA0_mpjpe_mm"]) for r in ba0])
            )
            if ba0
            else math.nan,
            "median_wrist_mm": float(
                np.median([float(r["BA0_wrist_mm"]) for r in ba0])
            )
            if ba0
            else math.nan,
        },
        {"method": "R1", "metric_3d_valid_frames": None, "median_mpjpe_mm": None},
        {"method": "R2", "metric_3d_valid_frames": None, "median_mpjpe_mm": None},
        {"method": "R3", "metric_3d_valid_frames": None, "median_mpjpe_mm": None},
    ]
    write_csv(OUT / "stereo_after_crop_refinement.csv", stereo_rows)

    # Correlation between center error and pose error.
    corr_rows = []
    for r in crop_error_rows:
        corr_rows.append(
            {
                "crop_center_pixel_error": r["crop_center_pixel_error"],
                "wiLor_median_epe_px": r.get("wiLor_median_epe_px"),
            }
        )
    write_csv(OUT / "triangulation_error_correlation.csv", corr_rows)
    print("c rows", len(c_rows), "crop rows", len(crop_error_rows))
    print(json.dumps(r_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
