#!/usr/bin/env python3
"""Summarize v11 public hand initializer benchmark."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/public_hand_initializer_benchmark_v11"
)
V8 = REPO_ROOT / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v8_multitile_full"
V9 = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v9"
V10 = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v10"
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
MANIFEST_PATH = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"

TILE_PROTOCOL_PATH = OUT / "tile_protocol.json"
POSITIVE_CSV = OUT / "tile_positive_matrix.csv"
VIEW_GEOMETRY_CSV = OUT / "tile_view_geometry.csv"
MEDIAPIPE_CSV = OUT / "initializer_candidates_mediapipe_tile.csv"
WILOR_CSV = OUT / "initializer_candidates_wilor_tile.csv"
EGOFORCE_TILE_CSV = OUT / "initializer_candidates_egoforce_tile.csv"
COMMON_POSE_CSV = OUT / "common_pose_head_results.csv"

REQUIRED = [0, 4, 8, 12, 16, 20]
FINGERTIPS = [4, 8, 12, 16, 20]


def long_no_seed_rows() -> list[tuple[str, int]]:
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(s: str) -> Any:
    if s is None or s == "":
        return None
    return json.loads(s)


def iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(aa + bb - inter, 1e-9))


def load_positive_by_tile() -> dict[tuple[Any, ...], dict[str, str]]:
    out = {}
    for r in read_csv(POSITIVE_CSV):
        key = (r["clip"], int(r["frame"]), r["camera"], int(r["tile_id"]))
        if r["gt_positive"] == "True":
            out[key] = r
    return out


def load_geometry_by_view() -> dict[tuple[str, int, str], dict[str, str]]:
    out = {}
    for r in read_csv(VIEW_GEOMETRY_CSV):
        out[(r["clip"], int(r["frame"]), r["camera"])] = r
    return out


def model_candidates(path: Path) -> list[dict[str, str]]:
    return [
        r
        for r in read_csv(path)
        if r.get("proposal_valid") == "True"
    ]


def top_candidate_per_tile(
    cands: list[dict[str, str]],
) -> dict[tuple[Any, ...], dict[str, str]]:
    best = {}
    for r in cands:
        key = (
            r["clip"],
            int(r["frame"]),
            r["camera"],
            int(r["tile_id"]),
        )
        score = float(r["confidence"])
        if key not in best or score > float(best[key]["confidence"]):
            best[key] = r
    return best


def compute_ap(
    cands: list[dict[str, str]],
    pos_by: dict[tuple[Any, ...], dict[str, str]],
    threshold: float,
) -> dict[str, float]:
    num_gt = len(pos_by)
    if num_gt == 0:
        return {
            "ap": math.nan,
            "recall_top1": math.nan,
            "precision_top1": math.nan,
            "num_gt": 0,
        }
    # Each positive tile has one GT. Keep only the highest-scoring TP per tile.
    tile_best_score = {}
    tile_highest_iou = {}
    for cand in cands:
        key = (
            cand["clip"],
            int(cand["frame"]),
            cand["camera"],
            int(cand["tile_id"]),
        )
        if key not in pos_by:
            continue
        bbox = jload(cand["bbox_tile"])
        if bbox is None:
            continue
        i = iou(bbox, jload(pos_by[key]["gt_bbox"]))
        score = float(cand["confidence"])
        tile_highest_iou[key] = max(tile_highest_iou.get(key, 0.0), i)
        if i >= threshold:
            if key not in tile_best_score or score > tile_best_score[key]:
                tile_best_score[key] = score

    tp_tiles = len(tile_best_score)
    recall_top1 = tp_tiles / num_gt

    # Precision@top1: for each positive tile choose highest-scoring candidate
    # among all candidates; if that candidate matches threshold, it is TP.
    tile_top = {}
    for cand in cands:
        key = (
            cand["clip"],
            int(cand["frame"]),
            cand["camera"],
            int(cand["tile_id"]),
        )
        if key not in pos_by:
            continue
        bbox = jload(cand["bbox_tile"])
        if bbox is None:
            continue
        score = float(cand["confidence"])
        if key not in tile_top or score > float(tile_top[key]["confidence"]):
            tile_top[key] = cand
    tp_precision = sum(
        1
        for key, cand in tile_top.items()
        if iou(jload(cand["bbox_tile"]), jload(pos_by[key]["gt_bbox"]))
        >= threshold
    )
    precision_top1 = tp_precision / max(1, len(tile_top))

    # VOC-style AP over all candidates, with one GT per positive tile.
    # Only candidates on positive tiles are considered.
    scored = []
    used = set()
    for cand in cands:
        key = (
            cand["clip"],
            int(cand["frame"]),
            cand["camera"],
            int(cand["tile_id"]),
        )
        if key not in pos_by:
            continue
        bbox = jload(cand["bbox_tile"])
        if bbox is None:
            continue
        i = iou(bbox, jload(pos_by[key]["gt_bbox"]))
        tp = i >= threshold
        # If duplicate TP on same tile, only highest-scoring one counts.
        if tp and key in used:
            tp = False
        elif tp:
            used.add(key)
        scored.append((float(cand["confidence"]), tp))
    if not scored:
        ap = 0.0
    else:
        scored.sort(key=lambda x: x[0], reverse=True)
        tp = np.cumsum([int(x[1]) for x in scored], dtype=np.float64)
        fp = np.cumsum([int(not x[1]) for x in scored], dtype=np.float64)
        recall = tp / num_gt
        precision = tp / np.maximum(tp + fp, 1e-9)
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[0.0], precision, [0.0]])
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        inds = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[inds + 1] - mrec[inds]) * mpre[inds + 1]))

    return {
        "ap": ap,
        "recall_top1": float(recall_top1),
        "precision_top1": float(precision_top1),
        "num_gt": num_gt,
    }


def tile_detector_view_hits(
    top_by_tile: dict[tuple[Any, ...], dict[str, str]],
    pos_by: dict[tuple[Any, ...], dict[str, str]],
    threshold: float = 0.5,
) -> dict[tuple[str, int, str], dict[str, str]]:
    hits = {}
    for key, cand in top_by_tile.items():
        if key not in pos_by:
            continue
        bbox = jload(cand["bbox_tile"])
        if bbox is None:
            continue
        if iou(bbox, jload(pos_by[key]["gt_bbox"])) >= threshold:
            hits[(cand["clip"], int(cand["frame"]), cand["camera"])] = cand
    return hits


def frame_seed_from_views(
    view_hits: dict[tuple[str, int, str], Any],
    rows: list[tuple[str, int]],
) -> tuple[int, int, int, int]:
    left = set()
    right = set()
    for clip, frame, cam in view_hits:
        if (clip, frame) not in rows:
            continue
        if cam == "left":
            left.add((clip, frame))
        else:
            right.add((clip, frame))
    union = left | right
    both = left & right
    return len(union), len(both), len(left), len(right)


def load_full_model_controls() -> tuple[
    dict[tuple[str, int], dict[str, bool]],
    dict[tuple[str, int], dict[str, bool]],
]:
    rows = read_csv(V9 / "detection_2d_38.csv")
    b0 = {}
    b1 = {}
    for r in rows:
        key = (r["clip"], int(r["frame"]))
        b0.setdefault(key, {})[r["camera"]] = r["b0_seed"] == "True"
        b1.setdefault(key, {})[r["camera"]] = r["egoforce_seed"] == "True"
    return b0, b1


def full_frame_counts(
    seeds: dict[tuple[str, int], dict[str, bool]],
    rows: list[tuple[str, int]],
) -> tuple[int, int, int, int]:
    union = 0
    both = 0
    left = 0
    right = 0
    for key in rows:
        s = seeds.get(key, {})
        l = s.get("left", False)
        r = s.get("right", False)
        left += int(l)
        right += int(r)
        union += int(l or r)
        both += int(l and r)
    return union, both, left, right


def native_pose_metrics_for_model(
    cands: list[dict[str, str]],
    pos_by: dict[tuple[Any, ...], dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    top = top_candidate_per_tile(cands)
    for key, cand in top.items():
        if key not in pos_by:
            continue
        bbox = jload(cand["bbox_tile"])
        if bbox is None or iou(bbox, jload(pos_by[key]["gt_bbox"])) < 0.5:
            continue
        kps = jload(cand["keypoints_tile"])
        if kps is None or len(kps) < 21:
            continue
        gt = jload(pos_by[key]["gt_joints_uv"])
        vis = jload(pos_by[key]["gt_joints_visible"])
        pred = np.asarray(kps[:21], dtype=np.float64)
        gt = np.asarray(gt, dtype=np.float64)
        vis = np.asarray(vis, dtype=bool)
        mask = vis
        if mask.sum() < 3:
            continue
        e = np.linalg.norm(pred[mask] - gt[mask], axis=-1)
        req_mask = np.asarray([j in REQUIRED for j in range(21)], dtype=bool) & mask
        wrist_e = float(np.linalg.norm(pred[0] - gt[0])) if mask[0] else math.nan
        tip_mask = np.asarray([j in FINGERTIPS for j in range(21)], dtype=bool) & mask
        tip_e = (
            float(np.mean(np.linalg.norm(pred[tip_mask] - gt[tip_mask], axis=-1)))
            if tip_mask.any()
            else math.nan
        )
        bbox_wh = gt.max(axis=0) - gt.min(axis=0)
        bbox_diag = float(np.linalg.norm(bbox_wh))
        rows.append(
            {
                "clip": key[0],
                "frame": key[1],
                "camera": key[2],
                "tile_id": key[3],
                "model": cand["model"],
                "valid_joints": int(mask.sum()),
                "mean_epe_px": float(np.mean(e)),
                "median_epe_px": float(np.median(e)),
                "pck5": float(np.mean(e <= 5)),
                "pck10": float(np.mean(e <= 10)),
                "pck20": float(np.mean(e <= 20)),
                "nme": float(np.median(e / bbox_diag)) if bbox_diag > 0 else math.nan,
                "wrist_error_px": wrist_e,
                "fingertip_error_px": tip_e,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = long_no_seed_rows()
    pos_by = load_positive_by_tile()
    geom_by_view = load_geometry_by_view()
    protocol = json.loads(TILE_PROTOCOL_PATH.read_text(encoding="utf-8"))

    candidates = {
        "wilor_detector_tile": model_candidates(WILOR_CSV),
        "egoforce_detector_tile": model_candidates(EGOFORCE_TILE_CSV),
        "mediapipe_tile": model_candidates(MEDIAPIPE_CSV),
    }
    top = {k: top_candidate_per_tile(v) for k, v in candidates.items()}
    view_hits = {
        k: tile_detector_view_hits(top[k], pos_by) for k in candidates
    }
    tile_metrics = {}
    for name, cands in candidates.items():
        tile_metrics[name] = {
            "ap50": compute_ap(cands, pos_by, 0.5),
            "ap75": compute_ap(cands, pos_by, 0.75),
        }

    b0_full, b1_full = load_full_model_controls()
    full_counts = {
        "B0_wilor_full_m1": full_frame_counts(b0_full, rows),
        "B1_egoforce_full": full_frame_counts(b1_full, rows),
    }
    tile_counts = {
        name: frame_seed_from_views(view_hits[name], rows)
        for name in candidates
    }

    # Union seed: WiLoR full, EgoForce full, and MediaPipe tile are all
    # automatic initializers. This is the inference-compatible union upper
    # bound for this benchmark (winner selection is still oracle for
    # diagnosis only).
    union_views: set[tuple[str, int, str]] = set()
    for key in rows:
        if b0_full.get(key, {}).get("left", False):
            union_views.add((key[0], key[1], "left"))
        if b0_full.get(key, {}).get("right", False):
            union_views.add((key[0], key[1], "right"))
        if b1_full.get(key, {}).get("left", False):
            union_views.add((key[0], key[1], "left"))
        if b1_full.get(key, {}).get("right", False):
            union_views.add((key[0], key[1], "right"))
    for k in view_hits["mediapipe_tile"]:
        union_views.add(k)
    union_counts = frame_seed_from_views(union_views, rows)

    seed_rows = []
    for name, counts in full_counts.items():
        seed_rows.append(
            {
                "model": name,
                "input": "RAW/M1 full model control",
                "at_least_one_view": counts[0],
                "both_view": counts[1],
                "left_view": counts[2],
                "right_view": counts[3],
            }
        )
    for name, counts in tile_counts.items():
        seed_rows.append(
            {
                "model": name,
                "input": "M1 fixed perspective tiles",
                "at_least_one_view": counts[0],
                "both_view": counts[1],
                "left_view": counts[2],
                "right_view": counts[3],
            }
        )
    seed_rows.append(
        {
            "model": "B2_100doh_100K+ego",
            "input": "unavailable_official_google_drive_weights_404",
            "at_least_one_view": math.nan,
            "both_view": math.nan,
            "left_view": math.nan,
            "right_view": math.nan,
        }
    )
    seed_rows.append(
        {
            "model": "public_initializer_union",
            "input": "B0_full + B1_full + mediapipe_tile",
            "at_least_one_view": union_counts[0],
            "both_view": union_counts[1],
            "left_view": union_counts[2],
            "right_view": union_counts[3],
        }
    )
    write_csv(OUT / "initializer_seed_recall_38.csv", seed_rows)

    # Per-view records.
    view_records = []
    for clip, frame in rows:
        for cam in ("left", "right"):
            key = (clip, frame, cam)
            geom = geom_by_view.get(key, {})
            rec = {
                "clip": clip,
                "frame": frame,
                "camera": cam,
                "failure_type": geom.get("failure_type", ""),
                "gt_visible_raw": geom.get("gt_visible_raw", ""),
                "gt_bbox_diag_px": geom.get("gt_bbox_diag_px", ""),
                "gt_wrist_distance_to_image_boundary_px": geom.get(
                    "gt_wrist_distance_to_image_boundary_px", ""
                ),
                "B0_wilor_full_seed": b0_full.get((clip, frame), {}).get(cam, False),
                "B1_egoforce_full_seed": b1_full.get((clip, frame), {}).get(cam, False),
                "wilor_detector_tile_seed": key in view_hits["wilor_detector_tile"],
                "egoforce_detector_tile_seed": key
                in view_hits["egoforce_detector_tile"],
                "mediapipe_tile_seed": key in view_hits["mediapipe_tile"],
                "union_seed": key in union_views,
            }
            view_records.append(rec)
    write_csv(OUT / "initializer_per_view_metrics.csv", view_records)

    # Tile metrics.
    tile_metric_rows = []
    for name, metrics in tile_metrics.items():
        tile_metric_rows.append(
            {
                "model": name,
                "num_gt_positive_tiles": metrics["ap50"]["num_gt"],
                "ap50": metrics["ap50"]["ap"],
                "ap75": metrics["ap75"]["ap"],
                "recall_at_iou50": metrics["ap50"]["recall_top1"],
                "precision_at_iou50": metrics["ap50"]["precision_top1"],
                "recall_at_iou75": metrics["ap75"]["recall_top1"],
                "precision_at_iou75": metrics["ap75"]["precision_top1"],
                "candidate_count": len(candidates[name]),
            }
        )
    write_csv(OUT / "initializer_tile_metrics.csv", tile_metric_rows)

    # Failure subset metrics.
    failure_rows = []
    type_order = ["center_hand", "edge_hand", "very_small_hand", "invisible"]
    for ftype in type_order:
        subset = [r for r in view_records if r["failure_type"] == ftype]
        if not subset:
            continue
        row = {
            "failure_type": ftype,
            "gt_views": len(subset),
            "B0_wilor_full_seed": sum(r["B0_wilor_full_seed"] for r in subset),
            "B1_egoforce_full_seed": sum(r["B1_egoforce_full_seed"] for r in subset),
            "mediapipe_tile_seed": sum(r["mediapipe_tile_seed"] for r in subset),
            "union_seed": sum(r["union_seed"] for r in subset),
        }
        failure_rows.append(row)
    write_csv(OUT / "failure_subset_metrics.csv", failure_rows)

    # Proposal overlap matrix for automatic frame-level seed sets.
    sets = {
        "B0_wilor_full": {
            (r["clip"], r["frame"])
            for r in view_records
            if r["B0_wilor_full_seed"]
        },
        "B1_egoforce_full": {
            (r["clip"], r["frame"])
            for r in view_records
            if r["B1_egoforce_full_seed"]
        },
        "mediapipe_tile": {
            (r["clip"], r["frame"])
            for r in view_records
            if r["mediapipe_tile_seed"]
        },
    }
    names = list(sets)
    overlap_rows = []
    for a in names:
        row = {"model": a}
        for b in names:
            row[b] = len(sets[a] & sets[b])
        overlap_rows.append(row)
    write_csv(OUT / "proposal_overlap_matrix.csv", overlap_rows)

    union_summary = {
        "n_frames": len(rows),
        "model_seed_at_least_one": {k: v[0] for k, v in full_counts.items()},
        "tile_seed_at_least_one": {k: v[0] for k, v in tile_counts.items()},
        "public_initializer_union_at_least_one": union_counts[0],
        "public_initializer_union_both": union_counts[1],
        "pairwise_overlap": {a: {b: len(sets[a] & sets[b]) for b in names} for a in names},
        "100doh_status": "official_weights_unavailable",
        "wildhands_status": "requires_vitpose_body_detector_and_heavy_detectron2_path_not_run",
        "umetrack_status": "crop_oracle_not_automatic",
    }
    (OUT / "proposal_union_summary.json").write_text(
        json.dumps(union_summary, indent=2), encoding="utf-8"
    )

    ensemble_rows = [
        {
            "ensemble": "union_B0_full_B1_full_mediapipe_tile",
            "at_least_one_view": union_counts[0],
            "both_view": union_counts[1],
            "left_view": union_counts[2],
            "right_view": union_counts[3],
            "selection_policy": "oracle_union_upper_bound_no_gt_ranking_in_final",
        }
    ]
    write_csv(OUT / "ensemble_results.csv", ensemble_rows)

    # Native pose metrics from each tile detector's own keypoints.
    native_rows = []
    for name, cands in candidates.items():
        native_rows.extend(native_pose_metrics_for_model(cands, pos_by))
    write_csv(OUT / "native_pose_results.csv", native_rows)

    # Common pose metrics (WiLoR pose head on MediaPipe bboxes).
    common_rows = read_csv(COMMON_POSE_CSV)
    common_metric_rows = []
    for cand in common_rows:
        key = (
            cand["clip"],
            int(cand["frame"]),
            cand["camera"],
            int(cand["tile_id"]),
        )
        pos = pos_by.get(key)
        if pos is None:
            continue
        pred = np.asarray(jload(cand["keypoints_2d_tile"]), dtype=np.float64)
        gt = np.asarray(jload(pos["gt_joints_uv"]), dtype=np.float64)
        vis = np.asarray(jload(pos["gt_joints_visible"]), dtype=bool)
        mask = vis
        if mask.sum() < 3:
            continue
        e = np.linalg.norm(pred[mask] - gt[mask], axis=-1)
        wh = gt.max(axis=0) - gt.min(axis=0)
        bbox_diag = float(np.linalg.norm(wh))
        tip_mask = np.asarray([j in FINGERTIPS for j in range(21)], dtype=bool) & mask
        common_metric_rows.append(
            {
                "clip": cand["clip"],
                "frame": int(cand["frame"]),
                "camera": cand["camera"],
                "tile_id": int(cand["tile_id"]),
                "proposal_source": cand["proposal_source"],
                "pose_head": cand["pose_head"],
                "valid_joints": int(mask.sum()),
                "median_epe_px": float(np.median(e)),
                "mean_epe_px": float(np.mean(e)),
                "pck5": float(np.mean(e <= 5)),
                "pck10": float(np.mean(e <= 10)),
                "pck20": float(np.mean(e <= 20)),
                "nme": float(np.median(e / bbox_diag)) if bbox_diag > 0 else math.nan,
                "wrist_error_px": float(np.linalg.norm(pred[0] - gt[0])) if mask[0] else math.nan,
                "fingertip_error_px": (
                    float(np.mean(np.linalg.norm(pred[tip_mask] - gt[tip_mask], axis=-1)))
                    if tip_mask.any()
                    else math.nan
                ),
            }
        )
    write_csv(OUT / "common_pose_head_metrics.csv", common_metric_rows)

    # No public initializer materially raised one-view seed, so no new stereo
    # downstream triangulation was triggered.
    write_csv(
        OUT / "stereo_downstream_results.csv",
        [
            {
                "stage": "stereo_downstream",
                "status": "not_triggered",
                "reason": "no_public_initializer_beat_existing_hybrid_12_38",
                "final_both_view_coverage": union_counts[1],
                "mpjpe_mm": "",
                "root_aligned_mpjpe_mm": "",
                "pa_mpjpe_mm": "",
                "wrist_error_mm": "",
                "fingertip_error_mm": "",
                "reprojection_error_px": "",
            }
        ],
    )

    print(json.dumps(union_summary, indent=2))
    print(f"tile metrics: {json.dumps(tile_metrics, indent=2)}")
    print(f"seed rows: {json.dumps(seed_rows, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
