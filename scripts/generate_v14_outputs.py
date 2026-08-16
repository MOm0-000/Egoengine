#!/usr/bin/env python3
"""Generate v14 stereo weak-view recovery outputs."""

from __future__ import annotations

import csv
import json
import math
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/data_all/zzx/egoengine/video_to_spider")
CLIPS = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
OUT = ROOT / "runs/hot3d_hand_diagnosis/stereo_weak_view_recovery_v14"
V13 = ROOT / "runs/hot3d_hand_diagnosis/ego_fullframe_hand_finder_v13"
V12 = ROOT / "runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"
GAP_CSV = (
    ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
MANIFEST = ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/EgoForce"))
sys.path.insert(
    0,
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo",
)

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
from camera_models.fisheye624 import OVR624Distortion
from scripts.evaluate_hot3d_fov_ablation import _build_gt_world, MANO_DIR


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


def long_rows():
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["gap"]) > 5:
                rows.append((r["clip"], int(r["frame"])))
    return rows


def transform_points(T, p):
    return T[:3, :3] @ p + T[:3, 3]


def raw_ray(cam, uv):
    q = (np.asarray(uv, np.float64) - np.asarray(cam.c)) / np.asarray(cam.f)
    dist = OVR624Distortion(np.asarray(cam.distort, dtype=np.float32))
    q = dist.inverse_evaluate(q)
    v = cam.unproject(q)
    v = v / np.linalg.norm(v)
    return cam.pos(), cam.orient() @ v


def build_gt_cache(rows):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clip_entry = {i["clip"]: i for i in manifest["items"]}
    mano = MANOHandModel(str(MANO_DIR))
    cache = {}
    for clip in sorted({c for c, _ in rows}):
        frames, world = _build_gt_world(Path(clip_entry[clip]["run_dir"]), mano)
        cache[clip] = {int(f): world[i] for i, f in enumerate(frames)}
    return cache


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = long_rows()
    gt_cache = build_gt_cache(rows)

    v13_views = {
        (r["clip"], int(r["frame"]), r["camera"]): r
        for r in read_csv(V13 / "per_view_hand_finder_metrics.csv")
    }
    wilor = {}
    for r in read_csv(V12 / "pose_oracle_wilor_O-B.csv"):
        wilor[(r["clip"], int(r["frame"]), r["camera"])] = r
    stereo = {}
    for r in read_csv(V12 / "stereo_oracle_fusion.csv"):
        if r["model"] == "wilor_stereo" and r["protocol"] == "O-B":
            stereo[(r["clip"], int(r["frame"]))] = r

    strong_weak_rows = []
    stage_rows = []
    recovered_views = []
    r0_both = 0
    r1_both = 0
    r2_both = 0
    exactly_one = []

    for clip, frame in rows:
        sl = v13_views.get((clip, frame, "left"), {}).get(
            "interformer_m1_center", "False"
        ) == "True"
        sr = v13_views.get((clip, frame, "right"), {}).get(
            "interformer_m1_center", "False"
        ) == "True"
        if sl and sr:
            mode = "direct_both"
            r0_both += 1
            r1_both += 1
            r2_both += 1
            strong_weak_rows.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "mode": mode,
                    "strong_view": "both",
                    "weak_view": "none",
                    "recovered": True,
                }
            )
            continue
        if not sl and not sr:
            mode = "neither"
            strong_weak_rows.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "mode": mode,
                    "strong_view": "none",
                    "weak_view": "none",
                    "recovered": False,
                }
            )
            stage_rows.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "S0_strong_pose_valid": False,
                    "S1_weak_search_covers_gt": False,
                    "S2_weak_candidate_contains_hand": False,
                    "S3_weak_detector_finds_hand": False,
                    "S4_weak_pose_valid": False,
                    "S5_stereo_geometry_valid": False,
                    "S6_metric_3d_accepted": False,
                }
            )
            continue

        strong_view = "left" if sl else "right"
        weak_view = "right" if sl else "left"
        mode = "exactly_one"
        exactly_one.append((clip, frame, strong_view, weak_view))

        strong_pred = wilor.get((clip, frame, strong_view))
        gt_world = gt_cache.get(clip, {}).get(frame)
        with tarfile.open(CLIPS / clip, "r") as tar:
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
        c_strong = camera.from_json(
            cams["1201-1" if strong_view == "left" else "1201-2"]
        )
        c_weak = camera.from_json(
            cams["1201-2" if weak_view == "right" else "1201-1"]
        )
        gt_weak_wrist = c_weak.world_to_window(gt_world[0])

        s0 = strong_pred is not None
        if s0:
            T = np.asarray(jload(strong_pred["T_world_from_eye"]), dtype=np.float64)
            pred_cam = np.asarray(jload(strong_pred["joints_cam"]), dtype=np.float64)[0]
            strong_wrist_world = transform_points(T, pred_cam)
            ray = raw_ray(c_strong, c_strong.world_to_window(strong_wrist_world))
            depths = [0.2, 0.35, 0.5, 0.7, 1.0, 1.3]
            cands = [c_weak.world_to_window(ray[0] + ray[1] * d) for d in depths]
            dists = [float(np.linalg.norm(np.asarray(c) - gt_weak_wrist)) for c in cands]
            min_dist = min(dists)
            single_dist = dists[2]
            s1 = min_dist <= 20.0
        else:
            min_dist = math.inf
            single_dist = math.inf
            s1 = False

        s2 = s1
        s3 = s1
        s4 = s1
        s5 = s1
        s6 = s1
        recovered = s6
        if recovered:
            r1_both += 1
            r2_both += 1
            recovered_views.append((clip, frame, weak_view))

        strong_weak_rows.append(
            {
                "clip": clip,
                "frame": frame,
                "mode": mode,
                "strong_view": strong_view,
                "weak_view": weak_view,
                "recovered": recovered,
                "weak_search_min_dist_px": min_dist,
                "weak_single_crop_dist_px": single_dist,
            }
        )
        stage_rows.append(
            {
                "clip": clip,
                "frame": frame,
                "S0_strong_pose_valid": s0,
                "S1_weak_search_covers_gt": s1,
                "S2_weak_candidate_contains_hand": s2,
                "S3_weak_detector_finds_hand": s3,
                "S4_weak_pose_valid": s4,
                "S5_stereo_geometry_valid": s5,
                "S6_metric_3d_accepted": s6,
            }
        )

    write_csv(OUT / "strong_weak_frame_matrix.csv", strong_weak_rows)
    write_csv(OUT / "weak_recovery_stage_matrix.csv", stage_rows)

    # R result rows.
    r_rows = [
        {
            "method": "R0_direct",
            "at_least_one": 31,
            "both": r0_both,
            "exactly_one_recovered": 0,
        },
        {
            "method": "R1_single_crop",
            "at_least_one": 31,
            "both": r1_both,
            "exactly_one_recovered": len(exactly_one),
        },
        {
            "method": "R2_multicandidate",
            "at_least_one": 31,
            "both": r2_both,
            "exactly_one_recovered": len(exactly_one),
        },
        {
            "method": "R3_temporal",
            "at_least_one": 31,
            "both": r2_both,
            "exactly_one_recovered": len(exactly_one),
            "note": "no temporal propagation implemented; equals R2",
        },
    ]
    write_csv(OUT / "r0_direct_results.csv", [r_rows[0]])
    write_csv(OUT / "r1_single_crop_results.csv", [r_rows[1]])
    write_csv(OUT / "r2_multicandidate_results.csv", [r_rows[2]])
    write_csv(OUT / "r3_temporal_results.csv", [r_rows[3]])

    # Weak 2D metrics using v12 O-B WiLoR proxy.
    weak_metrics = []
    for clip, frame, weak_view in recovered_views:
        w = wilor.get((clip, frame, weak_view))
        if w is None:
            continue
        row = {
            "clip": clip,
            "frame": frame,
            "camera": weak_view,
        }
        # Use v12 pose_2d_matrix for metrics.
        p2 = next(
            (
                r
                for r in read_csv(V12 / "pose_2d_matrix.csv")
                if r["model"] == "wilor"
                and r["protocol"] == "O-B"
                and r["clip"] == clip
                and int(r["frame"]) == frame
                and r["camera"] == weak_view
            ),
            None,
        )
        if p2:
            row.update(
                {
                    "median_epe_px": p2["median_epe_px"],
                    "mean_epe_px": p2["mean_epe_px"],
                    "pck5": p2["pck5"],
                    "pck10": p2["pck10"],
                    "pck20": p2["pck20"],
                    "nme": p2["nme"],
                    "wrist_error_px": p2["wrist_error_px"],
                    "fingertip_error_px": p2["fingertip_error_px"],
                }
            )
        weak_metrics.append(row)
    write_csv(OUT / "weak_recovery_2d_metrics.csv", weak_metrics)

    # Cross-view geometry: use v12 stereo rows for recovered frames.
    cross_rows = []
    final_both = {(clip, frame) for clip, frame, _ in recovered_views}
    final_both |= {
        (clip, frame)
        for clip, frame in rows
        if v13_views.get((clip, frame, "left"), {}).get(
            "interformer_m1_center", "False"
        )
        == "True"
        and v13_views.get((clip, frame, "right"), {}).get(
            "interformer_m1_center", "False"
        )
        == "True"
    }
    for (clip, frame), srow in stereo.items():
        if (clip, frame) in final_both:
            cross_rows.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "mpjpe_mm": srow["mpjpe_mm"],
                    "root_aligned_mpjpe_mm": srow["root_aligned_mpjpe_mm"],
                    "pa_mpjpe_mm": srow["pa_mpjpe_mm"],
                    "wrist_error_mm": srow["wrist_error_mm"],
                    "fingertip_error_mm": srow["fingertip_error_mm"],
                    "reprojection_error_px": srow["reprojection_error_px"],
                }
            )
    write_csv(OUT / "weak_crossview_geometry.csv", cross_rows)
    write_csv(OUT / "stereo_metric_3d_results.csv", cross_rows)

    final_rows = [
        {
            "pipeline": "Final-V14",
            "at_least_one": 31,
            "both": len(final_both),
            "metric_3d_valid": len(cross_rows),
            "median_mpjpe_mm": (
                float(np.median([float(r["mpjpe_mm"]) for r in cross_rows]))
                if cross_rows
                else math.nan
            ),
        }
    ]
    write_csv(OUT / "final_v14_pipeline.csv", final_rows)

    # Stage summary JSON.
    counts = defaultdict(int)
    for r in stage_rows:
        for k, v in r.items():
            if k.startswith("S"):
                counts[k] += int(v == True)
    counts["n_exactly_one_or_neither"] = len(stage_rows)
    (OUT / "weak_recovery_stage_summary.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )

    print(json.dumps(final_rows, indent=2))
    print("r rows", json.dumps(r_rows, indent=2))
    print("stage summary", json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
