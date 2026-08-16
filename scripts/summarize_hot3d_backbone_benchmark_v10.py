#!/usr/bin/env python3
"""Summarize v10 Full-EgoForce and WiLoR+EgoForce Hybrid."""

from __future__ import annotations

import ast
import csv
import json
import math
import sys
import tarfile
from collections import Counter
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
from scripts.evaluate_hot3d_fov_ablation import (
    CLIPS_ROOT,
    MANO_DIR,
    REQUIRED,
    FINGERTIPS,
    _build_gt_world,
    _transform_points,
)
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
from camera_models.fisheye624 import OVR624Distortion


OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v10"
V9 = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v9"
SUBSET_MANIFEST = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
V9_SEED_CSV = V9 / "seed_recall_38.csv"
E0_SELECTED = V9 / "egoforce_selected_predictions.json"
E1_SELECTED = OUT / "egoforce_full_selected_predictions.json"
E1_MATRIX = OUT / "egoforce_full_matrix.csv"


def long_no_seed_rows() -> list[tuple[str, int]]:
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def parse_key(key: str):
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


def _bbox_diag(uv: np.ndarray) -> float:
    uv = np.asarray(uv, dtype=np.float64)
    wh = uv.max(axis=0) - uv.min(axis=0)
    return float(np.linalg.norm(wh))


def _procrustes_pa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a_c.T @ b_c)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return float(np.mean(np.linalg.norm(b_c @ r.T - a_c, axis=-1)))


def _ray_triangulate(
    raw_cams: dict[str, Any],
    uv_left: np.ndarray,
    uv_right: np.ndarray,
):
    """Triangulate raw-fisheye rays in world coordinates."""
    out = np.zeros((len(uv_left), 3), dtype=np.float64)
    valid = np.zeros(len(uv_left), dtype=bool)
    reproj = np.full(len(uv_left), np.inf, dtype=np.float64)

    def ray(cam, uv):
        q = (np.asarray(uv, dtype=np.float64) - np.asarray(cam.c)) / np.asarray(
            cam.f
        )
        dist = OVR624Distortion(np.asarray(cam.distort, dtype=np.float32))
        q = dist.inverse_evaluate(q)
        d_cam = cam.unproject(q)
        d_cam = d_cam / np.linalg.norm(d_cam, axis=-1, keepdims=True)
        d_world = d_cam @ cam.orient().T
        return cam.pos(), d_world

    c1, d1 = ray(raw_cams["left"], uv_left)
    c2, d2 = ray(raw_cams["right"], uv_right)
    for i in range(len(uv_left)):
        if not (np.isfinite(d1[i]).all() and np.isfinite(d2[i]).all()):
            continue
        a = float(d1[i] @ d1[i])
        b = float(d1[i] @ d2[i])
        c = float(d2[i] @ d2[i])
        denom = a * c - b * b
        if abs(denom) < 1e-12:
            continue
        w = c1 - c2
        e = float(d1[i] @ w)
        f = float(d2[i] @ w)
        s = (b * f - c * e) / denom
        t = (a * f - b * e) / denom
        if s <= 0 or t <= 0:
            continue
        p1 = c1 + s * d1[i]
        p2 = c2 + t * d2[i]
        point = 0.5 * (p1 + p2)
        out[i] = point
        valid[i] = True
        uvl = raw_cams["left"].eye_to_window(raw_cams["left"].world_to_eye(point))
        uvr = raw_cams["right"].eye_to_window(raw_cams["right"].world_to_eye(point))
        reproj[i] = max(
            float(np.linalg.norm(uvl - uv_left[i])),
            float(np.linalg.norm(uvr - uv_right[i])),
        )
    return out, valid, reproj


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    clip_entry = {item["clip"]: item for item in manifest["items"]}
    rows = long_no_seed_rows()

    mano = MANOHandModel(str(MANO_DIR))
    gt_cache = {}
    for clip in sorted({clip for clip, _ in rows}):
        run_dir = Path(clip_entry[clip]["run_dir"]).resolve()
        frame_numbers, world = _build_gt_world(run_dir, mano)
        gt_cache[clip] = {
            int(f): world[i] for i, f in enumerate(frame_numbers)
        }

    v9_seed = {
        (r["clip"], int(r["frame"])): r
        for r in csv.DictReader(V9_SEED_CSV.open(newline="", encoding="utf-8"))
    }

    def read_selected(path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        return {parse_key(k): v for k, v in data.items()}

    e0 = read_selected(E0_SELECTED)
    e1 = read_selected(E1_SELECTED)

    e1_matrix_rows = list(csv.DictReader(E1_MATRIX.open(newline="", encoding="utf-8")))
    e1_matrix_by_view = {}
    for r in e1_matrix_rows:
        key = (r["clip"], int(r["frame"]), r["camera"])
        e1_matrix_by_view.setdefault(key, []).append(r)

    frame_rows = []
    view_rows = []
    matched_e0 = []
    matched_e1 = []
    both_e0 = []
    both_e1 = []

    for clip, frame in rows:
        selected_side = clip_entry[clip]["selected_side"]
        gt_world = gt_cache[clip][frame]
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
        raw_cams = {
            "left": camera.from_json(cams["1201-1"]),
            "right": camera.from_json(cams["1201-2"]),
        }

        view_gt_uv = {}
        view_gt_eye = {}
        view_gt_vis = {}
        view_gt_diag = {}
        for view, cam in raw_cams.items():
            eye = cam.world_to_eye(gt_world)
            uv = cam.world_to_window(gt_world)
            view_gt_uv[view] = np.asarray(uv, dtype=np.float64)
            view_gt_eye[view] = np.asarray(eye, dtype=np.float64)
            view_gt_vis[view] = bool(
                (eye[REQUIRED, 2] > 0).any()
                and cam.w_visible(uv[REQUIRED]).any()
            )
            view_gt_diag[view] = _bbox_diag(view_gt_uv[view])

        w_row = v9_seed.get((clip, frame), {})
        w_seed = {
            v: w_row.get(f"b0_{v}_seed", "False") == "True"
            for v in ("left", "right")
        }
        e0_seed = {}
        e1_seed = {}
        for view in ("left", "right"):
            for name, selected in (("e0", e0), ("e1", e1)):
                pred = selected.get((clip, frame, view))
                if pred is None:
                    ok = False
                else:
                    ok = _valid_required(
                        _arr(pred["pred_j2d_raw"]),
                        _arr(pred["pred_j3d_cam"]),
                    )
                if name == "e0":
                    e0_seed[view] = ok
                else:
                    e1_seed[view] = ok

        union_at_least = any(
            [w_seed["left"], w_seed["right"], e1_seed["left"], e1_seed["right"]]
        )
        union_both = (w_seed["left"] and w_seed["right"]) or (
            e1_seed["left"] and e1_seed["right"]
        )
        frame_rows.append(
            {
                "clip": clip,
                "frame": frame,
                "wiLor_at_least": any(w_seed.values()),
                "egoforce_e1_at_least": any(e1_seed.values()),
                "hybrid_at_least": union_at_least,
                "wiLor_both": all(w_seed.values()),
                "egoforce_e1_both": all(e1_seed.values()),
                "hybrid_both": union_both,
            }
        )

        for view in ("left", "right"):
            gt_vis = view_gt_vis[view]
            pred_e0 = e0.get((clip, frame, view))
            pred_e1 = e1.get((clip, frame, view))
            row = {
                "clip": clip,
                "frame": frame,
                "camera": view,
                "gt_visible_raw": gt_vis,
                "wiLor_seed": w_seed[view],
                "egoforce_e0_seed": e0_seed[view],
                "egoforce_e1_seed": e1_seed[view],
                "hybrid_seed": w_seed[view] or e1_seed[view],
            }
            for name, pred, seed, matched_list in (
                ("e0", pred_e0, e0_seed[view], matched_e0),
                ("e1", pred_e1, e1_seed[view], matched_e1),
            ):
                if pred is None:
                    row[f"{name}_matched"] = False
                    row[f"{name}_wrist_error_px"] = math.nan
                    continue
                p2d = _arr(pred["pred_j2d_raw"])
                p3d = _arr(pred["pred_j3d_cam"])
                wrist_err = float(np.linalg.norm(p2d[0] - view_gt_uv[view][0]))
                matched = bool(gt_vis and seed and wrist_err <= 100.0)
                row[f"{name}_matched"] = matched
                row[f"{name}_wrist_error_px"] = wrist_err
                if matched:
                    matched_list.append(
                        {
                            "clip": clip,
                            "frame": frame,
                            "camera": view,
                            "pred_j2d": p2d,
                            "pred_j3d": p3d,
                            "gt_uv": view_gt_uv[view],
                            "gt_eye": view_gt_eye[view],
                            "gt_diag": view_gt_diag[view],
                        }
                    )
            view_rows.append(row)

        if e0_seed["left"] and e0_seed["right"] and view_gt_vis["left"] and view_gt_vis["right"]:
            both_e0.append((clip, frame))
        if e1_seed["left"] and e1_seed["right"] and view_gt_vis["left"] and view_gt_vis["right"]:
            both_e1.append((clip, frame))

    # Frame-level summaries.
    def count(rows, field):
        return int(sum(bool(r[field]) for r in rows))

    e1_frame_summary = {
        "at_least_one": count(frame_rows, "egoforce_e1_at_least"),
        "both": count(frame_rows, "egoforce_e1_both"),
    }
    w_at_least = count(frame_rows, "wiLor_at_least")
    e_at_least = count(frame_rows, "egoforce_e1_at_least")
    hybrid_summary = {
        "at_least_one": count(frame_rows, "hybrid_at_least"),
        "both": count(frame_rows, "hybrid_both"),
        "wiLor_only": int(
            sum(
                r["wiLor_at_least"] and not r["egoforce_e1_at_least"]
                for r in frame_rows
            )
        ),
        "egoforce_only": int(
            sum(
                r["egoforce_e1_at_least"] and not r["wiLor_at_least"]
                for r in frame_rows
            )
        ),
        "both_models": int(
            sum(
                r["wiLor_at_least"] and r["egoforce_e1_at_least"]
                for r in frame_rows
            )
        ),
        "neither": int(
            sum(
                not r["wiLor_at_least"] and not r["egoforce_e1_at_least"]
                for r in frame_rows
            )
        ),
    }

    # Failure stage decomposition for E1 selected top candidate per view.
    def top_candidate_for_view(key):
        cands = e1_matrix_by_view.get(key, [])
        if not cands:
            return None
        return max(
            cands,
            key=lambda r: (
                float(r["mean_required_kpt_w"]),
                float(r["candidate_score"]),
            ),
        )

    stage_counts = Counter()
    stage_rows = []
    for clip, frame in rows:
        for view in ("left", "right"):
            key = (clip, frame, view)
            cand = top_candidate_for_view(key)
            if cand is None:
                stage = "A_no_hand_proposal"
            elif cand["hand_crop_valid"] != "True":
                stage = "C_hand_crop_invalid"
            elif cand["forearm_present"] != "True":
                stage = "B_no_forearm_proposal"
            elif cand["transformer_valid"] != "True":
                stage = "D_transformer_failure"
            elif cand["absolute_solver_valid"] != "True":
                stage = "F_ray_space_solve_poor"
            elif cand["final_observation_valid"] != "True":
                stage = "G_final_quality_gate_failure"
            else:
                stage = "success"
            stage_counts[stage] += 1
            stage_rows.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "camera": view,
                    "failure_stage": stage,
                    "hand_proposal_exists": cand is not None,
                    "hand_score": cand["candidate_score"] if cand else "",
                    "forearm_present": cand["forearm_present"] if cand else "",
                    "forearm_score": cand["forearm_score"] if cand else "",
                    "hand_crop_valid": cand["hand_crop_valid"] if cand else "",
                    "forearm_crop_valid": cand["forearm_crop_valid"] if cand else "",
                    "transformer_valid": cand["transformer_valid"] if cand else "",
                    "absolute_solver_valid": cand["absolute_solver_valid"] if cand else "",
                    "final_observation_valid": cand["final_observation_valid"] if cand else "",
                }
            )

    # Metrics.
    def pixel_metrics(matched):
        if not matched:
            return {"n": 0}
        all_e = []
        wrist_e = []
        tip_e = []
        nme = []
        for m in matched:
            e = np.linalg.norm(m["pred_j2d"] - m["gt_uv"], axis=-1)
            all_e.append(e)
            wrist_e.append(e[0])
            tip_e.append(e[FINGERTIPS])
            if np.isfinite(m["gt_diag"]) and m["gt_diag"] > 1:
                nme.append(e / m["gt_diag"])
        all_e = np.concatenate(all_e)
        wrist_e = np.asarray(wrist_e)
        tip_e = np.concatenate(tip_e)
        nme = np.concatenate(nme) if nme else np.asarray([math.nan])
        return {
            "n": len(matched),
            "median_epe_px": float(np.median(all_e)),
            "mean_epe_px": float(np.mean(all_e)),
            "pck5": float(np.mean(all_e <= 5)),
            "pck10": float(np.mean(all_e <= 10)),
            "pck20": float(np.mean(all_e <= 20)),
            "nme": float(np.median(nme)) if nme.size else math.nan,
            "median_wrist_px": float(np.median(wrist_e)),
            "median_fingertip_px": float(np.median(tip_e)),
        }

    def pose_metrics(matched):
        if not matched:
            return {"n": 0}
        mpjpe = []
        root = []
        pa = []
        wrist = []
        depth = []
        tip = []
        for m in matched:
            e = np.linalg.norm(m["pred_j3d"] - m["gt_eye"], axis=-1)
            mpjpe.append(np.mean(e))
            root.append(np.mean(np.linalg.norm((m["pred_j3d"] - m["pred_j3d"][0]) - (m["gt_eye"] - m["gt_eye"][0]), axis=-1)))
            pa.append(_procrustes_pa(m["gt_eye"], m["pred_j3d"]))
            wrist.append(np.linalg.norm(m["pred_j3d"][0] - m["gt_eye"][0]) * 1000)
            depth.append(abs(m["pred_j3d"][0, 2] - m["gt_eye"][0, 2]) * 1000)
            tip.append(np.mean(np.linalg.norm(m["pred_j3d"][FINGERTIPS] - m["gt_eye"][FINGERTIPS], axis=-1)) * 1000)
        return {
            "n": len(matched),
            "median_mpjpe_mm": float(np.median(mpjpe) * 1000),
            "median_root_mpjpe_mm": float(np.median(root) * 1000),
            "median_pa_mpjpe_mm": float(np.median(pa) * 1000),
            "median_wrist_mm": float(np.median(wrist)),
            "median_wrist_depth_mm": float(np.median(depth)),
            "median_fingertip_mm": float(np.median(tip)),
        }

    e0_pix = pixel_metrics(matched_e0)
    e1_pix = pixel_metrics(matched_e1)
    e0_pose = pose_metrics(matched_e0)
    e1_pose = pose_metrics(matched_e1)

    # Stereo triangulation for E1 both frames.
    tri_rows = []
    for clip, frame in both_e1:
        gt_world = gt_cache[clip][frame]
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
        raw_cams = {
            "left": camera.from_json(cams["1201-1"]),
            "right": camera.from_json(cams["1201-2"]),
        }
        gt_eye = raw_cams["left"].world_to_eye(gt_world)
        preds = {}
        for view in ("left", "right"):
            p = e1.get((clip, frame, view))
            if p is not None:
                preds[view] = _arr(p["pred_j2d_raw"])
        if set(preds) != {"left", "right"}:
            continue
        points, valid, reproj = _ray_triangulate(
            raw_cams, preds["left"], preds["right"]
        )
        if not valid[0]:
            continue
        mask = valid
        e = np.linalg.norm(points[mask] - gt_eye[mask], axis=-1)
        root_e = np.linalg.norm(
            (points[mask] - points[0]) - (gt_eye[mask] - gt_eye[0]),
            axis=-1,
        )
        pa = _procrustes_pa(gt_eye[mask], points[mask])
        tip_mask = valid & np.isin(np.arange(21), FINGERTIPS)
        fingertip_points = points[tip_mask]
        fingertip_gt = gt_eye[tip_mask]
        fingertip_e = (
            np.mean(np.linalg.norm(fingertip_points - fingertip_gt, axis=-1))
            if len(fingertip_points)
            else math.nan
        )
        tri_rows.append(
            {
                "clip": clip,
                "frame": frame,
                "mpjpe_mm": float(np.mean(e) * 1000),
                "root_mpjpe_mm": float(np.mean(root_e) * 1000),
                "pa_mpjpe_mm": float(pa * 1000),
                "wrist_error_mm": float(np.linalg.norm(points[0] - gt_eye[0]) * 1000),
                "fingertip_error_mm": float(fingertip_e * 1000),
                "reprojection_error_px": float(np.median(reproj[valid])),
            }
        )
    tri_summary = {"n": len(tri_rows)}
    if tri_rows:
        for key in ("mpjpe_mm", "root_mpjpe_mm", "pa_mpjpe_mm", "wrist_error_mm", "fingertip_error_mm", "reprojection_error_px"):
            tri_summary[f"median_{key}"] = float(np.median([r[key] for r in tri_rows]))

    def crossview_consistency(both_frames, selected):
        wrist_dis = []
        joint_dis = []
        root_dis = []
        n = 0
        for clip, frame in both_frames:
            with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
                cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
            pred_world = {}
            for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
                pred = selected.get((clip, frame, view))
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
            left = pred_world["left"]
            right = pred_world["right"]
            wrist_dis.append(float(np.linalg.norm(left[0] - right[0])) * 1000)
            joint_dis.append(
                float(np.mean(np.linalg.norm(left - right, axis=-1))) * 1000
            )
            root_dis.append(
                float(
                    np.mean(
                        np.linalg.norm(
                            (left - left[0]) - (right - right[0]),
                            axis=-1,
                        )
                    )
                    * 1000
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
            "median_root_relative_disagreement_mm": (
                float(np.median(root_dis)) if root_dis else math.nan
            ),
        }

    e0_cross = crossview_consistency(both_e0, e0)
    e1_cross = crossview_consistency(both_e1, e1)

    # Write.
    with (OUT / "egoforce_full_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "E1_seed": e1_frame_summary,
                "E1_pixel": e1_pix,
                "E1_pose": e1_pose,
                "Hybrid_seed": hybrid_summary,
                "Stereo_triangulation": tri_summary,
                "Failure_stage_counts": dict(stage_counts),
                "E0_crossview": e0_cross,
                "E1_crossview": e1_cross,
            },
            f,
            indent=2,
        )

    with (OUT / "egoforce_e0_e1_2d.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "E0", "E1"])
        for key in e0_pix.keys() | e1_pix.keys():
            writer.writerow([key, e0_pix.get(key), e1_pix.get(key)])

    with (OUT / "egoforce_e0_e1_3d.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "E0", "E1"])
        for key in e0_pose.keys() | e1_pose.keys():
            writer.writerow([key, e0_pose.get(key), e1_pose.get(key)])

    with (OUT / "egoforce_failure_stage_matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip",
                "frame",
                "camera",
                "failure_stage",
                "hand_proposal_exists",
                "hand_score",
                "forearm_present",
                "forearm_score",
                "hand_crop_valid",
                "forearm_crop_valid",
                "transformer_valid",
                "absolute_solver_valid",
                "final_observation_valid",
            ],
        )
        writer.writeheader()
        writer.writerows(stage_rows)

    (OUT / "egoforce_failure_stage_summary.json").write_text(
        json.dumps(dict(stage_counts), indent=2), encoding="utf-8"
    )

    with (OUT / "egoforce_crossview_consistency.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "E0", "E1"])
        for key in sorted(e0_cross.keys() | e1_cross.keys()):
            writer.writerow([key, e0_cross.get(key), e1_cross.get(key)])

    with (OUT / "hybrid_seed_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protocol", "at_least_one", "both"])
        writer.writerow(["WiLoR", 11, 2])
        writer.writerow(["EgoForce_E1", e1_frame_summary["at_least_one"], e1_frame_summary["both"]])
        writer.writerow(["Hybrid", hybrid_summary["at_least_one"], hybrid_summary["both"]])
        writer.writerow(
            [
                "Hybrid_wiLor_only",
                hybrid_summary["wiLor_only"],
                "",
            ]
        )
        writer.writerow(
            [
                "Hybrid_egoforce_only",
                hybrid_summary["egoforce_only"],
                "",
            ]
        )

    with (OUT / "hybrid_candidate_matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip",
                "frame",
                "camera",
                "gt_visible_raw",
                "wiLor_seed",
                "egoforce_e0_seed",
                "egoforce_e1_seed",
                "hybrid_seed",
                "e0_matched",
                "e0_wrist_error_px",
                "e1_matched",
                "e1_wrist_error_px",
            ],
        )
        writer.writeheader()
        writer.writerows(view_rows)

    with (OUT / "hybrid_stereo_3d.csv").open("w", newline="") as f:
        if tri_rows:
            writer = csv.DictWriter(f, fieldnames=list(tri_rows[0].keys()))
            writer.writeheader()
            writer.writerows(tri_rows)
        else:
            f.write("clip,frame\n")

    print(json.dumps({
        "E1_seed": e1_frame_summary,
        "Hybrid_seed": hybrid_summary,
        "Failure_stage_counts": dict(stage_counts),
        "E0_pixel": e0_pix,
        "E1_pixel": e1_pix,
        "E0_pose": e0_pose,
        "E1_pose": e1_pose,
        "Stereo_triangulation": tri_summary,
        "E0_crossview": e0_cross,
        "E1_crossview": e1_cross,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
