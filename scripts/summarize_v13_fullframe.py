#!/usr/bin/env python3
"""Summarize v13 full-frame ego hand finder benchmark."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/ego_fullframe_hand_finder_v13"
V11 = REPO_ROOT / "runs/hot3d_hand_diagnosis/public_hand_initializer_benchmark_v11"
V12 = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"
INTER_CSV = OUT / "interformer_candidates.csv"
V11_PER_VIEW = V11 / "initializer_per_view_metrics.csv"
V11_SEED = V11 / "initializer_seed_recall_38.csv"
V11_GEOM = V11 / "tile_view_geometry.csv"
V11_POS = V11 / "tile_positive_matrix.csv"
V12_MATRIX = V12 / "oracle_crop_matrix.csv"
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)


def long_no_seed_rows():
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(s):
    if s in ("", None):
        return None
    return json.loads(s)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(aa + bb - inter, 1e-9))


def center_usable(center, gt_bbox, margin=20.0):
    if center is None or gt_bbox is None:
        return False
    return (
        gt_bbox[0] - margin <= center[0] <= gt_bbox[2] + margin
        and gt_bbox[1] - margin <= center[1] <= gt_bbox[3] + margin
    )


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
    rows38 = long_no_seed_rows()

    # Existing per-view seeds.
    v11_per_view = {
        (r["clip"], int(r["frame"]), r["camera"]): r
        for r in read_csv(V11_PER_VIEW)
    }
    geom = {
        (r["clip"], int(r["frame"]), r["camera"]): r
        for r in read_csv(V11_GEOM)
    }
    pos_by = {}
    for r in read_csv(V11_POS):
        if r["gt_positive"] == "True":
            pos_by[
                (r["clip"], int(r["frame"]), r["camera"], int(r["tile_id"]))
            ] = json.loads(r["gt_bbox"])
    raw_gt = {}
    for r in read_csv(V12_MATRIX):
        if r["raw_bbox"]:
            raw_gt[(r["clip"], int(r["frame"]), r["camera"])] = json.loads(
                r["raw_bbox"]
            )

    inter_rows = read_csv(INTER_CSV)
    inter_raw = [r for r in inter_rows if r["protocol"] == "raw"]
    inter_m1 = [r for r in inter_rows if r["protocol"] == "m1"]

    def model_seed(model):
        out = {}
        for key, vr in v11_per_view.items():
            field = {
                "wilor": "B0_wilor_full_seed",
                "egoforce": "B1_egoforce_full_seed",
                "mediapipe": "mediapipe_tile_seed",
            }[model]
            out[key] = vr.get(field, "False") == "True"
        return out

    def inter_seed(raw=True, mode="center"):
        rows = inter_raw if raw else inter_m1
        out = {}
        for r in rows:
            key = (r["clip"], int(r["frame"]), r["camera"])
            if raw:
                gt_bbox = raw_gt.get(key)
            else:
                gt_bbox = pos_by.get((key[0], key[1], key[2], int(r["tile_id"])))
            comps = jload(r["hand_components"]) or []
            best_iou = 0.0
            if gt_bbox is not None:
                best_iou = max(
                    (iou(c["bbox"], gt_bbox) for c in comps), default=0.0
                )
            usable = False
            if mode == "center":
                usable = any(
                    center_usable(c["center"], gt_bbox, 20.0) for c in comps
                )
            else:
                usable = best_iou >= 0.5
            if raw:
                out[key] = usable
            else:
                # A view is usable if any tile is usable.
                if usable:
                    out[key] = True
        return out

    def frame_counts(seed):
        left = set()
        right = set()
        for key, ok in seed.items():
            if not ok:
                continue
            clip, frame, cam = key
            if (clip, frame) not in rows38:
                continue
            if cam == "left":
                left.add((clip, frame))
            else:
                right.add((clip, frame))
        union = left | right
        both = left & right
        return {
            "neither": len(rows38) - len(union),
            "exactly_one": len(union) - len(both),
            "both": len(both),
            "at_least_one": len(union),
            "left": len(left),
            "right": len(right),
        }

    models = {
        "wilor": model_seed("wilor"),
        "egoforce": model_seed("egoforce"),
        "mediapipe": model_seed("mediapipe"),
        "interformer_raw_center": inter_seed(True, "center"),
        "interformer_raw_iou50": inter_seed(True, "iou"),
        "interformer_m1_center": inter_seed(False, "center"),
        "interformer_m1_iou50": inter_seed(False, "iou"),
    }

    fullframe_seed_rows = []
    for name, seed in models.items():
        c = frame_counts(seed)
        fullframe_seed_rows.append({"model": name, **c})
    write_csv(OUT / "fullframe_seed_recall_38.csv", fullframe_seed_rows)

    # Per-view metrics and failure subset.
    per_view_rows = []
    for clip, frame in rows38:
        for cam in ("left", "right"):
            key = (clip, frame, cam)
            rec = {
                "clip": clip,
                "frame": frame,
                "camera": cam,
                "failure_type": geom.get(key, {}).get("failure_type", ""),
            }
            for name, seed in models.items():
                rec[name] = bool(seed.get(key, False))
            per_view_rows.append(rec)
    write_csv(OUT / "per_view_hand_finder_metrics.csv", per_view_rows)

    failure_rows = []
    for ftype in ["edge_hand", "very_small_hand", "center_hand", "all"]:
        subset = [
            r
            for r in per_view_rows
            if ftype == "all" or r["failure_type"] == ftype
        ]
        if not subset:
            continue
        rec = {"failure_type": ftype, "views": len(subset)}
        for name in models:
            rec[name] = sum(bool(r[name]) for r in subset)
        failure_rows.append(rec)
    write_csv(OUT / "failure_subset_metrics.csv", failure_rows)

    # Segmentation/bbox metrics using bbox pseudo-mask.
    seg_rows = []
    for r in inter_raw:
        key = (r["clip"], int(r["frame"]), r["camera"])
        gt = raw_gt.get(key)
        comps = jload(r["hand_components"]) or []
        if gt is None:
            continue
        gt_area = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
        pred_area = sum(c["area"] for c in comps)
        inter_area = sum(
            max(0.0, min(c["bbox"][2], gt[2]) - max(c["bbox"][0], gt[0]))
            * max(0.0, min(c["bbox"][3], gt[3]) - max(c["bbox"][1], gt[1]))
            for c in comps
        )
        precision = inter_area / max(pred_area, 1e-9)
        recall = inter_area / max(gt_area, 1e-9)
        seg_rows.append(
            {
                "clip": clip,
                "frame": frame,
                "camera": cam,
                "pred_mask_area": pred_area,
                "gt_bbox_area": gt_area,
                "mask_iou": inter_area / max(gt_area + pred_area - inter_area, 1e-9),
                "hand_pixel_precision": precision,
                "hand_pixel_recall": recall,
            }
        )
    write_csv(OUT / "segmentation_metrics.csv", seg_rows)

    bbox_rows = []
    for r in inter_raw:
        key = (r["clip"], int(r["frame"]), r["camera"])
        gt = raw_gt.get(key)
        if gt is None:
            continue
        comps = jload(r["hand_components"]) or []
        best_iou = max((iou(c["bbox"], gt) for c in comps), default=0.0)
        bbox_rows.append(
            {
                "clip": r["clip"],
                "frame": int(r["frame"]),
                "camera": r["camera"],
                "best_iou": best_iou,
                "recall_iou50": best_iou >= 0.5,
                "recall_iou30": best_iou >= 0.3,
            }
        )
    write_csv(OUT / "bbox_metrics.csv", bbox_rows)

    raw_vs_m1_rows = []
    for name in ["interformer_raw_center", "interformer_m1_center"]:
        c = frame_counts(models[name])
        raw_vs_m1_rows.append({"model": name, **c})
    write_csv(OUT / "raw_vs_m1.csv", raw_vs_m1_rows)

    # Overlap matrix for frame success sets.
    sets = {}
    for name, seed in models.items():
        sets[name] = {
            (clip, frame)
            for (clip, frame, cam), ok in seed.items()
            if ok and (clip, frame) in rows38
        }
    names = list(sets)
    overlap_rows = []
    for a in names:
        rec = {"model": a}
        for b in names:
            rec[b] = len(sets[a] & sets[b])
        overlap_rows.append(rec)
    write_csv(OUT / "success_overlap_matrix.csv", overlap_rows)

    # Rescue from current WiLoR+EgoForce both-miss frames.
    both_miss = {
        (clip, frame)
        for clip, frame in rows38
        if not models["wilor"].get((clip, frame, "left"), False)
        and not models["wilor"].get((clip, frame, "right"), False)
        and not models["egoforce"].get((clip, frame, "left"), False)
        and not models["egoforce"].get((clip, frame, "right"), False)
    }
    rescue_rows = []
    for name in ["interformer_raw_center", "interformer_m1_center"]:
        rescued = both_miss & sets[name]
        rescue_rows.append(
            {
                "existing_both_miss": len(both_miss),
                "new_model": name,
                "rescued_frames": len(rescued),
            }
        )
    write_csv(OUT / "rescue_from_existing_misses.csv", rescue_rows)

    # Deployable ensemble: union of baselines + InterFormer M1 center.
    ensemble_seed = {}
    for key in {
        k
        for name in ["wilor", "egoforce", "mediapipe", "interformer_m1_center"]
        for k in models[name]
    }:
        ensemble_seed[key] = any(
            models[name].get(key, False)
            for name in ["wilor", "egoforce", "mediapipe", "interformer_m1_center"]
        )
    oracle_seed = {}
    for key in {k for name in models for k in models[name]}:
        oracle_seed[key] = any(models[name].get(key, False) for name in models)
    ensemble_rows = [
        {"ensemble": "deployable_union", **frame_counts(ensemble_seed)},
        {"ensemble": "oracle_union", **frame_counts(oracle_seed)},
    ]
    write_csv(OUT / "deployable_ensemble_results.csv", ensemble_rows)

    # Predicted O-B crop quality: center error to GT raw bbox center.
    crop_rows = []
    for r in inter_raw:
        key = (r["clip"], int(r["frame"]), r["camera"])
        gt = raw_gt.get(key)
        if gt is None:
            continue
        comps = jload(r["hand_components"]) or []
        usable = [c for c in comps if center_usable(c["center"], gt, 20.0)]
        if not usable:
            continue
        c = max(usable, key=lambda x: x["area"])
        gt_center = [(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2]
        center_err = float(np.linalg.norm(np.array(c["center"]) - np.array(gt_center)))
        coverage = iou(c["bbox"], gt)
        crop_rows.append(
            {
                "clip": clip,
                "frame": frame,
                "camera": cam,
                "center_error_px": center_err,
                "bbox_iou": coverage,
                "usable_crop": True,
            }
        )
    write_csv(OUT / "predicted_ob_crop_metrics.csv", crop_rows)

    print("fullframe seed", json.dumps(fullframe_seed_rows, indent=2))
    print("ensemble", json.dumps(ensemble_rows, indent=2))
    print("rescue", json.dumps(rescue_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
