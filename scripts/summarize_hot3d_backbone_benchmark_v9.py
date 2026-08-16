#!/usr/bin/env python3
"""Summarize the v9 hand-backbone benchmark on the 38 no-seed subset.

Run with the v2s-wilor environment (HOT3D toolkit and numpy/scipy are
available there):

    conda run -n v2s-wilor python \
        scripts/summarize_hot3d_backbone_benchmark_v9.py
"""

from __future__ import annotations

import ast
import csv
import json
import math
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HOT3D_TOOLKIT_REPO = Path(
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo"
)
if str(HOT3D_TOOLKIT_REPO) not in sys.path:
    sys.path.insert(0, str(HOT3D_TOOLKIT_REPO))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_tracking_toolkit import camera
from scripts.evaluate_hot3d_fov_ablation import (
    CLIPS_ROOT,
    MANO_DIR,
    REQUIRED,
    FINGERTIPS,
    _build_gt_world,
    _transform_points,
)
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel


OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v9"
SUBSET_MANIFEST = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
EGOFORCE_SELECTED = OUT / "egoforce_selected_predictions.json"
V8_SEED_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v8_multitile_full/seed_availability_o4_m1_m2.csv"
)
V8_TILE_CANDIDATES = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v8_multitile_full/tile_inference_candidates.csv"
)
PCK_THRESHOLDS = (5.0, 10.0, 20.0)


def long_no_seed_rows() -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def parse_selected_key(key: str) -> tuple[str, int, str]:
    clip, frame, view = ast.literal_eval(key)
    return str(clip), int(frame), str(view)


def _arr(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _valid_required(pred_j2d: np.ndarray, pred_j3d: np.ndarray) -> bool:
    if pred_j2d.shape[0] < 21 or pred_j3d.shape[0] < 21:
        return False
    req = pred_j2d[REQUIRED]
    z = pred_j3d[REQUIRED, 2]
    return bool(
        np.isfinite(req).all()
        and np.isfinite(z).all()
        and (z > 0).all()
        and (req[:, 0] >= 0).all()
        and (req[:, 0] < 640).all()
        and (req[:, 1] >= 0).all()
        and (req[:, 1] < 480).all()
    )


def _gt_bbox_diag(gt_uv: np.ndarray) -> float:
    uv = np.asarray(gt_uv, dtype=np.float64)
    if uv.shape[0] == 0 or not np.isfinite(uv).all():
        return math.nan
    wh = uv.max(axis=0) - uv.min(axis=0)
    return float(np.linalg.norm(wh))


def _procrustes_pa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a0 = a.mean(axis=0)
    b0 = b.mean(axis=0)
    a_c = a - a0
    b_c = b - b0
    u, _, vt = np.linalg.svd(a_c.T @ b_c)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return float(np.mean(np.linalg.norm(b_c @ r.T - a_c, axis=-1)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    clip_entry = {item["clip"]: item for item in manifest["items"]}
    rows = long_no_seed_rows()

    # Pre-build GT world per clip. The original subset uses a single selected
    # hand identity per clip.
    gt_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    mano_model = MANOHandModel(str(MANO_DIR))
    for clip in sorted({clip for clip, _ in rows}):
        entry = clip_entry[clip]
        run_dir = Path(entry["run_dir"]).resolve()
        frame_numbers, world = _build_gt_world(run_dir, mano_model)
        frame_to_world = {
            int(f): world[i] for i, f in enumerate(frame_numbers)
        }
        gt_cache[clip] = (frame_numbers, world, frame_to_world)

    egoforce = json.loads(EGOFORCE_SELECTED.read_text(encoding="utf-8"))
    egoforce_by_key = {
        parse_selected_key(key): value for key, value in egoforce.items()
    }

    v8_seed_rows = {
        (r["clip"], int(r["frame"])): r
        for r in csv.DictReader(V8_SEED_CSV.open(newline="", encoding="utf-8"))
    }

    frame_records: list[dict[str, Any]] = []
    view_records: list[dict[str, Any]] = []
    matched_views: list[dict[str, Any]] = []
    both_matched: list[dict[str, Any]] = []

    for clip, frame in rows:
        entry = clip_entry[clip]
        run_dir = Path(entry["run_dir"]).resolve()
        selected_side = entry["selected_side"]
        frame_numbers, world, frame_to_world = gt_cache[clip]
        if frame not in frame_to_world:
            print(f"missing GT world for {clip}/{frame}", file=sys.stderr)
            continue
        gt_world = frame_to_world[frame]
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
            raw_images = {}
            for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
                raw_images[view] = tar.extractfile(
                    f"{frame:06d}.image_{stream_id}.jpg"
                ).read()

        view_seed = {}
        view_gt_uv = {}
        view_gt_eye = {}
        view_gt_visible = {}
        view_gt_bbox_diag = {}
        view_gt_wrist_boundary = {}
        view_pred = {}

        for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
            raw_cam = camera.from_json(cams[stream_id])
            eye = raw_cam.world_to_eye(gt_world)
            uv = raw_cam.world_to_window(gt_world)
            visible = bool(
                (eye[REQUIRED, 2] > 0).any()
                and raw_cam.w_visible(uv[REQUIRED]).any()
            )
            view_gt_uv[view] = np.asarray(uv, dtype=np.float64)
            view_gt_eye[view] = np.asarray(eye, dtype=np.float64)
            view_gt_visible[view] = visible
            view_gt_bbox_diag[view] = _gt_bbox_diag(view_gt_uv[view])
            wrist_uv = view_gt_uv[view][0]
            view_gt_wrist_boundary[view] = float(
                min(
                    wrist_uv[0],
                    639.0 - wrist_uv[0],
                    wrist_uv[1],
                    479.0 - wrist_uv[1],
                )
            )

            pred = egoforce_by_key.get((clip, frame, view))
            view_pred[view] = pred
            if pred is None:
                view_seed[view] = False
                continue
            pred_j2d = _arr(pred["pred_j2d_raw"])
            pred_j3d = _arr(pred["pred_j3d_cam"])
            view_seed[view] = _valid_required(pred_j2d, pred_j3d)

        left_seed = view_seed.get("left", False)
        right_seed = view_seed.get("right", False)
        b0_row = v8_seed_rows.get((clip, frame), {})
        b0_left = b0_row.get("left_seed", "False") == "True"
        b0_right = b0_row.get("right_seed", "False") == "True"
        frame_records.append(
            {
                "clip": clip,
                "frame": frame,
                "selected_side": selected_side,
                "left_seed": left_seed,
                "right_seed": right_seed,
                "at_least_one": left_seed or right_seed,
                "both": left_seed and right_seed,
                "b0_left_seed": b0_left,
                "b0_right_seed": b0_right,
                "b0_at_least_one": b0_left or b0_right,
                "b0_both": b0_left and b0_right,
            }
        )

        for view in ("left", "right"):
            pred = view_pred[view]
            gt_visible = view_gt_visible[view]
            record = {
                "clip": clip,
                "frame": frame,
                "camera": view,
                "selected_side": selected_side,
                "gt_visible_raw": gt_visible,
                "gt_bbox_diag_px": view_gt_bbox_diag[view],
                "gt_wrist_distance_to_image_boundary_px": view_gt_wrist_boundary[
                    view
                ],
                "b0_seed": (
                    v8_seed_rows.get((clip, frame), {})
                    .get(f"{view}_seed", "False")
                    == "True"
                ),
                "egoforce_seed": bool(view_seed.get(view, False)),
                "egoforce_pred_exists": pred is not None,
            }
            if pred is not None:
                pred_j2d = _arr(pred["pred_j2d_raw"])
                pred_j3d = _arr(pred["pred_j3d_cam"])
                gt_uv = view_gt_uv[view]
                wrist_err = float(np.linalg.norm(pred_j2d[0] - gt_uv[0]))
                matched = bool(
                    gt_visible
                    and record["egoforce_seed"]
                    and wrist_err <= 100.0
                )
                record["wrist_error_px"] = wrist_err
                record["matched"] = matched
                if matched:
                    matched_views.append(
                        {
                            "clip": clip,
                            "frame": frame,
                            "camera": view,
                            "pred_j2d": pred_j2d,
                            "pred_j3d": pred_j3d,
                            "gt_uv": view_gt_uv[view],
                            "gt_eye": view_gt_eye[view],
                            "gt_bbox_diag": view_gt_bbox_diag[view],
                        }
                    )
            else:
                record["wrist_error_px"] = math.nan
                record["matched"] = False
            view_records.append(record)

        if (
            view_seed.get("left", False)
            and view_seed.get("right", False)
            and view_gt_visible["left"]
            and view_gt_visible["right"]
        ):
            both_matched.append((clip, frame))

    # Seed availability summary.
    n = len(frame_records)
    seed_summary = {
        "subset": "38 long/no-seed frames",
        "B0_M1_at_least_one_seed": int(
            sum(
                1
                for r in frame_records
                if v8_seed_rows.get((r["clip"], r["frame"]), {})
                .get("at_least_one", "False")
                == "True"
            )
        ),
        "B0_M1_both_seed": int(
            sum(
                1
                for r in frame_records
                if v8_seed_rows.get((r["clip"], r["frame"]), {})
                .get("both", "False")
                == "True"
            )
        ),
        "B1_EgoForce_at_least_one_seed": int(
            sum(r["at_least_one"] for r in frame_records)
        ),
        "B1_EgoForce_both_seed": int(
            sum(r["both"] for r in frame_records)
        ),
        "B1_EgoForce_left_seed": int(
            sum(r["left_seed"] for r in frame_records)
        ),
        "B1_EgoForce_right_seed": int(
            sum(r["right_seed"] for r in frame_records)
        ),
        "n_frames": n,
    }

    # 2D localization on matched EgoForce observations.
    def _pixel_metrics(matched: list[dict[str, Any]]) -> dict[str, Any]:
        if not matched:
            return {
                "n": 0,
                "median_epe_px": math.nan,
                "mean_epe_px": math.nan,
                "pck5": math.nan,
                "pck10": math.nan,
                "pck20": math.nan,
                "nme": math.nan,
                "median_wrist_px": math.nan,
                "median_fingertip_px": math.nan,
            }
        all_e = []
        wrist_e = []
        tip_e = []
        nme_vals = []
        for m in matched:
            e = np.linalg.norm(m["pred_j2d"] - m["gt_uv"], axis=-1)
            all_e.append(e)
            wrist_e.append(e[0])
            tip_e.append(e[FINGERTIPS])
            if np.isfinite(m["gt_bbox_diag"]) and m["gt_bbox_diag"] > 1:
                nme_vals.append(e / m["gt_bbox_diag"])
        all_e = np.concatenate(all_e)
        wrist_e = np.asarray(wrist_e)
        tip_e = np.concatenate(tip_e)
        nme_vals = np.concatenate(nme_vals) if nme_vals else np.asarray([math.nan])
        return {
            "n": len(matched),
            "median_epe_px": float(np.median(all_e)),
            "mean_epe_px": float(np.mean(all_e)),
            "pck5": float(np.mean(all_e <= 5)),
            "pck10": float(np.mean(all_e <= 10)),
            "pck20": float(np.mean(all_e <= 20)),
            "nme": float(np.median(nme_vals)) if nme_vals.size else math.nan,
            "median_wrist_px": float(np.median(wrist_e)),
            "median_fingertip_px": float(np.median(tip_e)),
        }

    pixel_metrics = _pixel_metrics(matched_views)

    # Absolute 3D metrics.
    def _pose_metrics(matched: list[dict[str, Any]]) -> dict[str, Any]:
        if not matched:
            return {
                "n": 0,
                "median_mpjpe_mm": math.nan,
                "mean_mpjpe_mm": math.nan,
                "median_root_mpjpe_mm": math.nan,
                "median_pa_mpjpe_mm": math.nan,
                "median_wrist_mm": math.nan,
                "median_wrist_depth_mm": math.nan,
                "median_fingertip_mm": math.nan,
            }
        mpjpe = []
        root_mpjpe = []
        pa_mpjpe = []
        wrist = []
        wrist_depth = []
        tip = []
        for m in matched:
            gt = m["gt_eye"]
            pred = m["pred_j3d"]
            e = np.linalg.norm(pred - gt, axis=-1)
            mpjpe.append(np.mean(e))
            root_mpjpe.append(np.mean(np.linalg.norm((pred - pred[0]) - (gt - gt[0]), axis=-1)))
            pa_mpjpe.append(_procrustes_pa(gt, pred))
            wrist.append(np.linalg.norm(pred[0] - gt[0]) * 1000.0)
            wrist_depth.append(abs(pred[0, 2] - gt[0, 2]) * 1000.0)
            tip.append(np.mean(np.linalg.norm(pred[FINGERTIPS] - gt[FINGERTIPS], axis=-1)) * 1000.0)
        return {
            "n": len(matched),
            "median_mpjpe_mm": float(np.median(mpjpe) * 1000.0),
            "mean_mpjpe_mm": float(np.mean(mpjpe) * 1000.0),
            "median_root_mpjpe_mm": float(np.median(root_mpjpe) * 1000.0),
            "median_pa_mpjpe_mm": float(np.median(pa_mpjpe) * 1000.0),
            "median_wrist_mm": float(np.median(wrist)),
            "median_wrist_depth_mm": float(np.median(wrist_depth)),
            "median_fingertip_mm": float(np.median(tip)),
        }

    pose_metrics = _pose_metrics(matched_views)

    # Stereo consistency.
    def _stereo_consistency(rows_both: list[tuple[str, int]]) -> dict[str, Any]:
        wrist_dis = []
        joint_dis = []
        n = 0
        for clip, frame in rows_both:
            entry = clip_entry[clip]
            frame_numbers, world, frame_to_world = gt_cache[clip]
            gt_world = frame_to_world[frame]
            with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
                cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
            pred_world = {}
            for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
                pred = egoforce_by_key.get((clip, frame, view))
                if pred is None:
                    continue
                pred_cam = _arr(pred["pred_j3d_cam"])
                raw_cam = camera.from_json(cams[stream_id])
                pred_world[view] = _transform_points(
                    np.asarray(raw_cam.T_world_from_eye, dtype=np.float64),
                    pred_cam,
                )
            if len(pred_world) < 2:
                continue
            n += 1
            wrist_dis.append(
                float(np.linalg.norm(pred_world["left"][0] - pred_world["right"][0]))
                * 1000.0
            )
            joint_dis.append(
                float(
                    np.mean(
                        np.linalg.norm(
                            pred_world["left"] - pred_world["right"],
                            axis=-1,
                        )
                    )
                    * 1000.0
                )
            )
        return {
            "n_both_frames": n,
            "median_wrist_disagreement_mm": (
                float(np.median(wrist_dis)) if wrist_dis else math.nan
            ),
            "median_joint_disagreement_mm": (
                float(np.median(joint_dis)) if joint_dis else math.nan
            ),
        }

    stereo = _stereo_consistency(both_matched)

    # Failure subset: simple geometric taxonomy for GT-visible views.
    failure_counts: dict[str, dict[str, int]] = {
        "center_hand": {"gt_views": 0, "b0_seed": 0, "egoforce_seed": 0, "matched": 0},
        "edge_hand": {"gt_views": 0, "b0_seed": 0, "egoforce_seed": 0, "matched": 0},
        "very_small_hand": {"gt_views": 0, "b0_seed": 0, "egoforce_seed": 0, "matched": 0},
        "other": {"gt_views": 0, "b0_seed": 0, "egoforce_seed": 0, "matched": 0},
    }
    for r in view_records:
        if not r["gt_visible_raw"]:
            continue
        diag = r["gt_bbox_diag_px"]
        boundary_dist = r.get("gt_wrist_distance_to_image_boundary_px", 1e9)
        if boundary_dist < 60.0:
            ftype = "edge_hand"
        elif diag < 60:
            ftype = "very_small_hand"
        else:
            ftype = "center_hand"
        failure_counts[ftype]["gt_views"] += 1
        if r["b0_seed"]:
            failure_counts[ftype]["b0_seed"] += 1
        if r["egoforce_seed"]:
            failure_counts[ftype]["egoforce_seed"] += 1
        if r["matched"]:
            failure_counts[ftype]["matched"] += 1

    # Write outputs.
    seed_path = OUT / "seed_recall_38.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip",
                "frame",
                "selected_side",
                "left_seed",
                "right_seed",
                "at_least_one",
                "both",
                "b0_left_seed",
                "b0_right_seed",
                "b0_at_least_one",
                "b0_both",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_records)

    detection_path = OUT / "detection_2d_38.csv"
    with detection_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip",
                "frame",
                "camera",
                "selected_side",
                "gt_visible_raw",
                "gt_bbox_diag_px",
                "gt_wrist_distance_to_image_boundary_px",
                "b0_seed",
                "egoforce_seed",
                "egoforce_pred_exists",
                "wrist_error_px",
                "matched",
            ],
        )
        writer.writeheader()
        writer.writerows(view_records)

    absolute_path = OUT / "absolute_3d_38.csv"
    with absolute_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip",
                "frame",
                "camera",
                "mpjpe_mm",
                "root_mpjpe_mm",
                "pa_mpjpe_mm",
                "wrist_error_mm",
                "wrist_depth_error_mm",
                "fingertip_error_mm",
            ],
        )
        writer.writeheader()
        for m in matched_views:
            gt = m["gt_eye"]
            pred = m["pred_j3d"]
            e = np.linalg.norm(pred - gt, axis=-1)
            root_e = np.linalg.norm((pred - pred[0]) - (gt - gt[0]), axis=-1)
            writer.writerow(
                {
                    "clip": m["clip"],
                    "frame": m["frame"],
                    "camera": m["camera"],
                    "mpjpe_mm": float(np.mean(e) * 1000.0),
                    "root_mpjpe_mm": float(np.mean(root_e) * 1000.0),
                    "pa_mpjpe_mm": float(_procrustes_pa(gt, pred) * 1000.0),
                    "wrist_error_mm": float(
                        np.linalg.norm(pred[0] - gt[0]) * 1000.0
                    ),
                    "wrist_depth_error_mm": float(
                        abs(pred[0, 2] - gt[0, 2]) * 1000.0
                    ),
                    "fingertip_error_mm": float(
                        np.mean(
                            np.linalg.norm(
                                pred[FINGERTIPS] - gt[FINGERTIPS], axis=-1
                            )
                        )
                        * 1000.0
                    ),
                }
            )

    stereo_path = OUT / "stereo_consistency_38.csv"
    with stereo_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in stereo.items():
            writer.writerow([k, v])

    failure_path = OUT / "failure_subset_comparison.csv"
    with failure_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["failure_type", "gt_views", "b0_seed", "egoforce_seed", "egoforce_matched"]
        )
        for ftype, counts in failure_counts.items():
            writer.writerow(
                [ftype, counts["gt_views"], counts["b0_seed"], counts["egoforce_seed"], counts["matched"]]
            )

    (OUT / "seed_recall_38.json").write_text(
        json.dumps(seed_summary, indent=2), encoding="utf-8"
    )
    (OUT / "detection_2d_38.json").write_text(
        json.dumps(pixel_metrics, indent=2), encoding="utf-8"
    )
    (OUT / "absolute_3d_38.json").write_text(
        json.dumps(pose_metrics, indent=2), encoding="utf-8"
    )
    (OUT / "stereo_consistency_38.json").write_text(
        json.dumps(stereo, indent=2), encoding="utf-8"
    )

    print(json.dumps(
        {
            "seed_summary": seed_summary,
            "pixel_metrics": pixel_metrics,
            "pose_metrics": pose_metrics,
            "stereo": stereo,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
