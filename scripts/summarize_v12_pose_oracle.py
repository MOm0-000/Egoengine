#!/usr/bin/env python3
"""Summarize v12 hand pose oracle benchmark."""

from __future__ import annotations

import csv
import json
import math
import sys
import tarfile
from collections import defaultdict
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
EGOFORCE_ROOT = REPO_ROOT / "third_party/EgoForce"
if str(EGOFORCE_ROOT) not in sys.path:
    sys.path.insert(0, str(EGOFORCE_ROOT))

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
from camera_models.fisheye624 import OVR624Distortion
from scripts.evaluate_hot3d_fov_ablation import (
    CLIPS_ROOT,
    MANO_DIR,
    FINGERTIPS,
    REQUIRED,
    _build_gt_world,
    _transform_points,
)


OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"
MATRIX_CSV = OUT / "oracle_crop_matrix.csv"
V11_GEOMETRY = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/public_hand_initializer_benchmark_v11/tile_view_geometry.csv"
)
SUBSET_MANIFEST = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
PCK_THRESHOLDS = (5.0, 10.0, 20.0)


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


def jload(s):
    if s in (None, ""):
        return None
    return json.loads(s)


def build_gt_cache(rows):
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    clip_entry = {item["clip"]: item for item in manifest["items"]}
    mano = MANOHandModel(str(MANO_DIR))
    cache = {}
    for clip in sorted({c for c, _ in rows}):
        run_dir = Path(clip_entry[clip]["run_dir"]).resolve()
        frames, world = _build_gt_world(run_dir, mano)
        cache[clip] = {int(f): world[i] for i, f in enumerate(frames)}
    return cache


def raw_cameras_for_rows(rows):
    cams = {}
    for clip, frame in rows:
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            js = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
        cams[(clip, frame)] = {
            "left": camera.from_json(js["1201-1"]),
            "right": camera.from_json(js["1201-2"]),
        }
    return cams


def load_geometry():
    out = {}
    for r in read_csv(V11_GEOMETRY):
        out[(r["clip"], int(r["frame"]), r["camera"])] = r
    return out


def pinhole_from_row(row, protocol):
    f = float(row[f"{protocol}_f"])
    T = np.asarray(jload(row[f"{protocol}_T_world_from_eye"]), dtype=np.float64)
    return camera.PinholePlaneCameraModel(
        width=256,
        height=256,
        f=(f, f),
        c=(127.5, 127.5),
        distort_coeffs=[],
        T_world_from_eye=T,
    )


def transform_points(T, pts):
    return _transform_points(T, pts)


def procrustes_pa(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    pc = pred - pred.mean(0)
    gc = gt - gt.mean(0)
    u, _, vt = np.linalg.svd(gc.T @ pc)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return float(np.mean(np.linalg.norm(pc @ r.T - gc, axis=-1)))


def bone_length_error(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    pd = np.linalg.norm(pred[:, None, :] - pred[None, :, :], axis=-1)
    gd = np.linalg.norm(gt[:, None, :] - gt[None, :, :], axis=-1)
    mask = np.triu(np.ones_like(gd, dtype=bool), 1)
    return float(np.mean(np.abs(pd[mask] - gd[mask])))


def _pixel_metrics(pred_uv, gt_uv, gt_vis, gt_diag):
    pred_uv = np.asarray(pred_uv, dtype=np.float64)
    gt_uv = np.asarray(gt_uv, dtype=np.float64)
    gt_vis = np.asarray(gt_vis, dtype=bool)
    if pred_uv.shape[0] < 21 or gt_uv.shape[0] < 21:
        return None
    pred = pred_uv[:21]
    gt = gt_uv[:21]
    mask = gt_vis[:21] & np.isfinite(pred).all(axis=-1)
    if mask.sum() < 3:
        return None
    e = np.linalg.norm(pred[mask] - gt[mask], axis=-1)
    diag = gt_diag if gt_diag and gt_diag > 0 else float(np.linalg.norm(gt.max(0) - gt.min(0)))
    tip_mask = np.asarray([j in FINGERTIPS for j in range(21)], dtype=bool) & mask
    return {
        "n_joints": int(mask.sum()),
        "mean_epe_px": float(np.mean(e)),
        "median_epe_px": float(np.median(e)),
        "pck5": float(np.mean(e <= 5)),
        "pck10": float(np.mean(e <= 10)),
        "pck20": float(np.mean(e <= 20)),
        "nme": float(np.median(e / diag)),
        "wrist_error_px": float(np.linalg.norm(pred[0] - gt[0])) if mask[0] else math.nan,
        "fingertip_error_px": (
            float(np.mean(np.linalg.norm(pred[tip_mask] - gt[tip_mask], axis=-1)))
            if tip_mask.any()
            else math.nan
        ),
    }


def _pose_metrics(pred_world, gt_world):
    pred = np.asarray(pred_world, dtype=np.float64)
    gt = np.asarray(gt_world, dtype=np.float64)
    if pred.shape != gt.shape or pred.shape[0] < 21:
        return None
    pred = pred[:21]
    gt = gt[:21]
    abs_e = np.linalg.norm(pred - gt, axis=-1)
    pred_root = pred - pred[0]
    gt_root = gt - gt[0]
    root_e = np.linalg.norm(pred_root - gt_root, axis=-1)
    pa = procrustes_pa(pred, gt)
    tips = np.asarray(FINGERTIPS)
    return {
        "n_joints": 21,
        "mpjpe_mm": float(np.mean(abs_e) * 1000),
        "root_aligned_mpjpe_mm": float(np.mean(root_e) * 1000),
        "pa_mpjpe_mm": pa * 1000,
        "wrist_error_mm": float(abs_e[0] * 1000),
        "fingertip_error_mm": float(np.mean(abs_e[tips] * 1000)),
        "bone_length_error_mm": bone_length_error(pred, gt) * 1000,
    }


def triangulate_rays(ray1, ray2):
    c1, d1 = ray1
    c2, d2 = ray2
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    a = float(d1 @ d1)
    b = float(d1 @ d2)
    c = float(d2 @ d2)
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        return None
    w = c1 - c2
    e = float(d1 @ w)
    f = float(d2 @ w)
    s = (b * f - c * e) / denom
    t = (a * f - b * e) / denom
    if s <= 0 or t <= 0:
        return None
    p1 = c1 + s * d1
    p2 = c2 + t * d2
    return 0.5 * (p1 + p2)


def crop_ray(cam, uv):
    p = cam.window_to_eye(uv)
    world = cam.eye_to_world(p)
    direction = world - cam.pos()
    n = np.linalg.norm(direction)
    if n < 1e-12:
        return None
    return cam.pos(), direction / n


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = long_no_seed_rows()
    gt_cache = build_gt_cache(rows)
    raw_cams = raw_cameras_for_rows(rows)
    geometry = load_geometry()
    matrix = {
        (r["clip"], int(r["frame"]), r["camera"]): r
        for r in read_csv(MATRIX_CSV)
    }

    # Model/protocol files.
    pred_files = {}
    for model in ("wilor", "hamer"):
        for protocol in ("O-A", "O-B"):
            pred_files[(model, protocol)] = read_csv(
                OUT / f"pose_oracle_{model}_{protocol}.csv"
            )
    pred_files[("umetrack", "O-A")] = [
        r for r in read_csv(OUT / "pose_oracle_umetrack.csv") if r["protocol"] == "O-A"
    ]
    pred_files[("umetrack", "O-B")] = [
        r for r in read_csv(OUT / "pose_oracle_umetrack.csv") if r["protocol"] == "O-B"
    ]
    egoforce_rows = read_csv(OUT / "pose_oracle_egoforce.csv")

    def keyed_preds(rows):
        out = {}
        for r in rows:
            out[(r["clip"], int(r["frame"]), r["camera"])] = r
        return out

    pred_keyed = {k: keyed_preds(v) for k, v in pred_files.items()}

    pose_2d_rows = []
    root_rows = []
    absolute_rows = []
    crop_sensitivity_rows = []
    stereo_rows = []

    for clip, frame in rows:
        gt_world = gt_cache[clip][frame]
        for view in ("left", "right"):
            mrow = matrix.get((clip, frame, view))
            if mrow is None:
                continue
            ftype = geometry.get((clip, frame, view), {}).get("failure_type", "unknown")
            for (model, protocol), preds in pred_keyed.items():
                pred = preds.get((clip, frame, view))
                if pred is None:
                    continue
                gt_uv = np.asarray(jload(mrow[f"{protocol}_gt_joints_uv"]), dtype=np.float64)
                gt_vis = np.asarray(jload(mrow[f"{protocol}_gt_joints_visible"]), dtype=bool)
                gt_bbox = jload(mrow[f"{protocol}_gt_bbox"])
                gt_diag = (
                    float(np.linalg.norm(np.asarray(gt_bbox[2:]) - np.asarray(gt_bbox[:2])))
                    if gt_bbox
                    else 0.0
                )
                if model in ("wilor", "hamer"):
                    pred_uv = np.asarray(jload(pred["joints_2d_crop"]), dtype=np.float64)
                    p2 = _pixel_metrics(pred_uv, gt_uv, gt_vis, gt_diag)
                    pred_cam = np.asarray(jload(pred["joints_cam"]), dtype=np.float64)
                    T = np.asarray(jload(mrow[f"{protocol}_T_world_from_eye"]), dtype=np.float64)
                    pred_world = transform_points(T, pred_cam)
                elif model == "umetrack":
                    pred_uv = np.asarray(jload(pred["joints_2d_crop"]), dtype=np.float64)
                    p2 = _pixel_metrics(pred_uv, gt_uv, gt_vis, gt_diag)
                    pred_world = np.asarray(jload(pred["landmarks_world_mm"]), dtype=np.float64) / 1000.0
                else:
                    continue
                if p2 is not None:
                    p2.update(
                        {
                            "clip": clip,
                            "frame": frame,
                            "camera": view,
                            "model": model,
                            "protocol": protocol,
                            "failure_type": ftype,
                            "gt_diag_px": gt_diag,
                        }
                    )
                    pose_2d_rows.append(p2)
                    crop_sensitivity_rows.append(p2)

                pose = _pose_metrics(pred_world, gt_world)
                if pose is not None:
                    pose.update(
                        {
                            "clip": clip,
                            "frame": frame,
                            "camera": view,
                            "model": model,
                            "protocol": protocol,
                            "failure_type": ftype,
                        }
                    )
                    root_rows.append(pose)
                    absolute_rows.append(pose)

        # EgoForce raw predictions.
        for view in ("left", "right"):
            pred = next(
                (
                    r
                    for r in egoforce_rows
                    if r["clip"] == clip
                    and int(r["frame"]) == frame
                    and r["camera"] == view
                ),
                None,
            )
            if pred is None or not pred["pred_j2d_raw"]:
                continue
            raw_cam = raw_cams[(clip, frame)][view]
            gt_uv = np.asarray(raw_cam.world_to_window(gt_world), dtype=np.float64)
            gt_eye = np.asarray(raw_cam.world_to_eye(gt_world), dtype=np.float64)
            gt_vis = (
                np.isfinite(gt_uv).all(axis=-1)
                & (gt_eye[:, 2] > 0)
                & raw_cam.w_visible(gt_uv)
            )
            gt_bbox_pts = gt_uv[gt_vis]
            gt_diag = float(np.linalg.norm(gt_bbox_pts.max(0) - gt_bbox_pts.min(0))) if gt_bbox_pts.size else 0
            pred_uv = np.asarray(jload(pred["pred_j2d_raw"]), dtype=np.float64)
            p2 = _pixel_metrics(pred_uv, gt_uv, gt_vis, gt_diag)
            if p2 is not None:
                p2.update(
                    {
                        "clip": clip,
                        "frame": frame,
                        "camera": view,
                        "model": "egoforce",
                        "protocol": "raw_gt_bbox",
                        "failure_type": geometry.get(
                            (clip, frame, view), {}
                        ).get("failure_type", "unknown"),
                        "gt_diag_px": gt_diag,
                    }
                )
                pose_2d_rows.append(p2)
                crop_sensitivity_rows.append(p2)
            pred_cam = np.asarray(jload(pred["pred_j3d_cam"]), dtype=np.float64)
            pred_world = raw_cam.eye_to_world(pred_cam)
            pose = _pose_metrics(pred_world, gt_world)
            if pose is not None:
                pred_eye = np.asarray(pred_cam, dtype=np.float64)
                pose.update(
                    {
                        "clip": clip,
                        "frame": frame,
                        "camera": view,
                        "model": "egoforce",
                        "protocol": "raw_gt_bbox",
                        "failure_type": geometry.get(
                            (clip, frame, view), {}
                        ).get("failure_type", "unknown"),
                        "wrist_depth_error_mm": float(
                            abs(pred_eye[0, 2] - gt_eye[0, 2]) * 1000
                        ),
                    }
                )
                root_rows.append(pose)
                absolute_rows.append(pose)

    # Stereo fusion for crop-based models where both views have predictions.
    for (model, protocol), preds in pred_keyed.items():
        for clip, frame in rows:
            left = preds.get((clip, frame, "left"))
            right = preds.get((clip, frame, "right"))
            if left is None or right is None:
                continue
            mleft = matrix.get((clip, frame, "left"))
            mright = matrix.get((clip, frame, "right"))
            if mleft is None or mright is None:
                continue
            cl = pinhole_from_row(mleft, protocol)
            cr = pinhole_from_row(mright, protocol)
            gt_world = gt_cache[clip][frame]
            if model in ("wilor", "hamer"):
                ul = np.asarray(jload(left["joints_2d_crop"]), dtype=np.float64)
                ur = np.asarray(jload(right["joints_2d_crop"]), dtype=np.float64)
            else:
                ul = np.asarray(jload(left["joints_2d_crop"]), dtype=np.float64)
                ur = np.asarray(jload(right["joints_2d_crop"]), dtype=np.float64)
            pts = []
            reproj = []
            for j in range(min(len(ul), len(ur), 21)):
                rl = crop_ray(cl, ul[j])
                rr = crop_ray(cr, ur[j])
                if rl is None or rr is None:
                    pts.append(np.full(3, np.nan))
                    reproj.append(np.nan)
                    continue
                p = triangulate_rays(rl, rr)
                if p is None:
                    pts.append(np.full(3, np.nan))
                    reproj.append(np.nan)
                    continue
                pts.append(p)
                projl = cl.world_to_window(p)
                projr = cr.world_to_window(p)
                reproj.append(
                    max(
                        float(np.linalg.norm(projl - ul[j])),
                        float(np.linalg.norm(projr - ur[j])),
                    )
                )
            pts = np.asarray(pts, dtype=np.float64)
            valid = np.isfinite(pts).all(axis=-1)
            if valid.sum() < 3:
                continue
            pose = _pose_metrics(pts, gt_world)
            if pose is None:
                continue
            pose.update(
                {
                    "clip": clip,
                    "frame": frame,
                    "model": f"{model}_stereo",
                    "protocol": protocol,
                    "reprojection_error_px": float(np.nanmedian(reproj)),
                }
            )
            stereo_rows.append(pose)

    # GT 2D oracle triangulation from raw cameras for both-view rows.
    for clip, frame in rows:
        cl = raw_cams[(clip, frame)]["left"]
        cr = raw_cams[(clip, frame)]["right"]
        gt_world = gt_cache[clip][frame]
        ul = np.asarray(cl.world_to_window(gt_world), dtype=np.float64)
        ur = np.asarray(cr.world_to_window(gt_world), dtype=np.float64)
        pts = []
        reproj = []
        def raw_ray(cam, uv):
            q = (np.asarray(uv, dtype=np.float64) - np.asarray(cam.c)) / np.asarray(cam.f)
            dist = OVR624Distortion(np.asarray(cam.distort, dtype=np.float32))
            q = dist.inverse_evaluate(q)
            d_cam = cam.unproject(q)
            d_cam = d_cam / np.linalg.norm(d_cam)
            return cam.pos(), d_cam

        for j in range(21):
            p = triangulate_rays(raw_ray(cl, ul[j]), raw_ray(cr, ur[j]))
            if p is None:
                pts.append(np.full(3, np.nan))
                reproj.append(np.nan)
                continue
            pts.append(p)
            reproj.append(
                max(
                    float(np.linalg.norm(cl.world_to_window(p) - ul[j])),
                    float(np.linalg.norm(cr.world_to_window(p) - ur[j])),
                )
            )
        pts = np.asarray(pts, dtype=np.float64)
        valid = np.isfinite(pts).all(axis=-1)
        if valid.sum() >= 3:
            pose = _pose_metrics(pts, gt_world)
            pose.update(
                {
                    "clip": clip,
                    "frame": frame,
                    "model": "gt2d_stereo_oracle",
                    "protocol": "raw",
                    "reprojection_error_px": float(np.nanmedian(reproj)),
                }
            )
            stereo_rows.append(pose)

    def write_rows(path, rows):
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writerow({field: field for field in fields})
            w.writerows(rows)

    write_rows(OUT / "pose_2d_matrix.csv", pose_2d_rows)
    write_rows(OUT / "root_relative_3d_matrix.csv", root_rows)
    write_rows(OUT / "absolute_3d_matrix.csv", absolute_rows)
    write_rows(OUT / "stereo_oracle_fusion.csv", stereo_rows)
    write_rows(OUT / "crop_sensitivity.csv", crop_sensitivity_rows)

    def aggregate(rows, metric_fields):
        groups = defaultdict(list)
        for r in rows:
            key = (r.get("model"), r.get("protocol"), r.get("failure_type", "all"))
            groups[key].append(r)
            key_all = (r.get("model"), r.get("protocol"), "all")
            groups[key_all].append(r)
        out = []
        for key, vals in groups.items():
            rec = {"model": key[0], "protocol": key[1], "failure_type": key[2], "n": len(vals)}
            for field in metric_fields:
                arr = [float(v[field]) for v in vals if v.get(field) is not None and math.isfinite(float(v[field]))]
                rec[f"median_{field}"] = float(np.median(arr)) if arr else math.nan
            out.append(rec)
        return out

    pose_2d_summary = aggregate(
        pose_2d_rows,
        [
            "mean_epe_px",
            "median_epe_px",
            "pck5",
            "pck10",
            "pck20",
            "nme",
            "wrist_error_px",
            "fingertip_error_px",
        ],
    )
    write_rows(OUT / "pose_2d_summary.csv", pose_2d_summary)

    root_summary = aggregate(
        root_rows,
        [
            "root_aligned_mpjpe_mm",
            "pa_mpjpe_mm",
            "bone_length_error_mm",
            "fingertip_error_mm",
        ],
    )
    write_rows(OUT / "root_relative_3d_summary.csv", root_summary)
    abs_summary = aggregate(
        absolute_rows,
        [
            "mpjpe_mm",
            "wrist_error_mm",
            "fingertip_error_mm",
        ],
    )
    write_rows(OUT / "absolute_3d_summary.csv", abs_summary)
    write_rows(OUT / "failure_subset_pose_metrics.csv", pose_2d_summary)

    # Automatic seed recall vs oracle-pose ceiling.
    automatic_vs_oracle = []
    auto_seed = {
        "wilor": 11,
        "hamer": 0,
        "egoforce": 10,
        "umetrack": 0,
    }
    for model in ("wilor", "hamer", "egoforce", "umetrack"):
        for protocol in ("O-A", "O-B"):
            if model == "egoforce" and protocol != "O-A":
                continue
            p2 = [
                r
                for r in pose_2d_rows
                if r["model"] == model
                and r["protocol"] == (protocol if model != "egoforce" else "raw_gt_bbox")
            ]
            rt = [
                r
                for r in root_rows
                if r["model"] == model
                and r["protocol"] == (protocol if model != "egoforce" else "raw_gt_bbox")
            ]
            absr = [
                r
                for r in absolute_rows
                if r["model"] == model
                and r["protocol"] == (protocol if model != "egoforce" else "raw_gt_bbox")
            ]
            automatic_vs_oracle.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "automatic_at_least_one_seed_38": auto_seed.get(model, ""),
                    "oracle_views": len(p2),
                    "oracle_median_epe_px": (
                        float(np.median([float(r["median_epe_px"]) for r in p2]))
                        if p2
                        else math.nan
                    ),
                    "oracle_pck20": (
                        float(np.mean([float(r["pck20"]) for r in p2]))
                        if p2
                        else math.nan
                    ),
                    "oracle_root_aligned_mpjpe_mm": (
                        float(np.median([float(r["root_aligned_mpjpe_mm"]) for r in rt]))
                        if rt
                        else math.nan
                    ),
                    "oracle_absolute_mpjpe_mm": (
                        float(np.median([float(r["mpjpe_mm"]) for r in absr]))
                        if absr
                        else math.nan
                    ),
                }
            )
    write_rows(OUT / "automatic_vs_oracle.csv", automatic_vs_oracle)

    print("pose 2d rows", len(pose_2d_rows))
    print("root rows", len(root_rows))
    print("stereo rows", len(stereo_rows))
    print("pose summary", json.dumps(pose_2d_summary, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
