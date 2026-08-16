#!/usr/bin/env python3
"""H1 stereo-hand ablation: WiLoR -> HaMeR with the same stereo fusion.

The script runs HaMeR independently on left/right rectified ADT frames, then
uses the existing calibrated-stereo triangulation and fixed QC gates without
rewriting a single threshold. It stores candidate artifacts under
`hands_hamer` / `hands_right_hamer` and writes a paired H0/H1 summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "runs/adt_depth_benchmark_summary.json"
OUTPUT_PATH = REPO_ROOT / "runs/adt_hamer_hand_benchmark_summary.json"
MATRIX_PATH = REPO_ROOT / "runs/hamer_hand_benchmark_matrix.csv"

REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
HAND_ORDER = ("left", "right")


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return float("nan")
    return float(np.percentile(values, q))


def _load(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = np.einsum("ij,...j->...i", K, points)
    return projected[..., :2] / projected[..., 2:3]


def _summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0, "median": -1.0, "p95": -1.0}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
    }


def _failure_reason_counts(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    observation_valid: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    joints: np.ndarray,
) -> dict[str, Any]:
    from video_to_spider.adapters.hand_stereo import (
        MAX_JOINT_DEPTH_M,
        MAX_REPROJECTION_ERROR_PX,
        MIN_DISPARITY_PX,
        MIN_JOINT_DEPTH_M,
    )

    left = np.asarray(left_uv, dtype=np.float64)
    right = np.asarray(right_uv, dtype=np.float64)
    observed = np.asarray(observation_valid, dtype=bool)
    disparity = np.asarray(diagnostics["disparity_px"], dtype=np.float64)
    vertical = np.asarray(diagnostics["vertical_disparity_abs_px"], dtype=np.float64)
    reprojection = np.asarray(diagnostics["reprojection_error_px"], dtype=np.float64)
    depth = np.asarray(joints[..., 2], dtype=np.float64)
    nonfinite = ~(
        np.isfinite(left).all(axis=-1)
        & np.isfinite(right).all(axis=-1)
        & np.isfinite(depth)
    )
    per_hand: dict[str, Any] = {}
    for hand, side in enumerate(HAND_ORDER):
        reasons: dict[str, int] = {}
        for name, mask in {
            "input_invalid": ~observed[:, hand],
            "nonfinite_projection": nonfinite[:, hand],
            "disparity_below_min_px": disparity[:, hand] < MIN_DISPARITY_PX,
            "vertical_above_max_px": vertical[:, hand] > MAX_REPROJECTION_ERROR_PX,
            "reprojection_above_max_px": reprojection[:, hand] > MAX_REPROJECTION_ERROR_PX,
            "depth_out_of_range_m": (
                (depth[:, hand] < MIN_JOINT_DEPTH_M)
                | (depth[:, hand] > MAX_JOINT_DEPTH_M)
            ),
        }.items():
            reasons[name] = [int(x) for x in np.asarray(mask, dtype=bool).sum(axis=1)]
        per_hand[side] = reasons
    return {"per_hand": per_hand, "limits": {k: v for k, v in {
        "min_disparity_px": MIN_DISPARITY_PX,
        "max_vertical_or_reprojection_error_px": MAX_REPROJECTION_ERROR_PX,
        "joint_depth_m": [MIN_JOINT_DEPTH_M, MAX_JOINT_DEPTH_M],
    }.items()}}


def evaluate_hamer_stereo(run_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    from video_to_spider.adapters.hand_stereo import (
        MAX_REPROJECTION_ERROR_PX,
        MIN_DISPARITY_PX,
        MIN_JOINT_DEPTH_M,
        MAX_JOINT_DEPTH_M,
        MIN_REQUIRED_FRAME_RATE,
        MIN_REQUIRED_JOINT_RATE,
        triangulate_rectified_joints,
    )
    from video_to_spider.schemas import validate_wilor_raw

    root = run_dir.resolve()
    left_path = root / "hands_hamer/hamer_raw.npz"
    right_path = root / "hands_right_hamer/hamer_raw.npz"
    output_path = root / "hands_hamer/hamer_stereo_raw.npz"
    metrics_path = output_path.with_name("hamer_stereo_metrics.json")
    if (output_path.exists() or metrics_path.exists()) and not overwrite:
        raise FileExistsError(f"HaMeR stereo output exists: {output_path}")

    stereo = json.loads((root / "calibration/stereo.json").read_text(encoding="utf-8"))
    if not stereo.get("accepted", False):
        raise RuntimeError("stereo rectification gate is not accepted")
    left = _load(left_path)
    right = _load(right_path)
    if left is None or right is None:
        return {"status": "input_artifact_missing", "left_exists": left_path.is_file(), "right_exists": right_path.is_file()}
    if not np.array_equal(left["frame_indices"], right["frame_indices"]):
        raise ValueError("left/right HaMeR timelines differ")
    K_left = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    K_right = np.load(root / "calibration/intrinsics_right.npy").astype(np.float64)
    left_joints = left["joints_camera_rootrel"] + left["translation_camera"][:, :, None]
    right_joints = right["joints_camera_rootrel"] + right["translation_camera"][:, :, None]
    left_uv = _project(K_left, left_joints)
    right_uv = _project(K_right, right_joints)
    observation_valid = left["valid"][:, :, None] & right["valid"][:, :, None]
    observation_valid = np.broadcast_to(observation_valid, left_uv.shape[:-1])
    joints, joint_valid, diagnostics = triangulate_rectified_joints(
        left_uv,
        right_uv,
        K_left=K_left,
        K_right=K_right,
        baseline_m=float(stereo["baseline_m"]),
        valid_observation=observation_valid,
    )
    frame_valid = joint_valid[:, :, REQUIRED].all(axis=-1)
    joint_rate = joint_valid.mean(axis=(0, 2))
    frame_rate = frame_valid.mean(axis=0)
    accepted_hand = (joint_rate >= MIN_REQUIRED_JOINT_RATE) & (frame_rate >= MIN_REQUIRED_FRAME_RATE)
    valid = frame_valid & accepted_hand[None]
    failure_analysis = _failure_reason_counts(left_uv, right_uv, observation_valid, diagnostics, joints)
    wrist = np.where(valid[..., None], joints[:, :, 0], 0.0)
    score = np.minimum(left["score"], right["score"]).astype(np.float32)
    score[~valid] = 0.0
    payload = {
        key: np.asarray(left[key])
        for key in (
            "frame_indices", "timestamps_s", "side", "mano_global_orient",
            "mano_hand_pose", "mano_betas", "joints_camera_rootrel",
            "vertices_camera_rootrel",
        )
        if key in left
    }
    payload.update(
        {
            "valid": valid,
            "score": score,
            "translation_camera": wrist.astype(np.float32),
            "joints_camera_metric": np.nan_to_num(joints, nan=0.0).astype(np.float32),
            "joint_metric_valid": joint_valid,
        }
    )
    validate_wilor_raw(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    per_hand: dict[str, Any] = {}
    for hand, side in enumerate(HAND_ORDER):
        mask = joint_valid[:, hand]
        per_hand[side] = {
            "accepted": bool(accepted_hand[hand]),
            "joint_valid_rate": float(joint_rate[hand]),
            "required_landmarks_frame_valid_rate": float(frame_rate[hand]),
            "disparity_px": _summary(diagnostics["disparity_px"][:, hand][mask]),
            "vertical_disparity_abs_px": _summary(diagnostics["vertical_disparity_abs_px"][:, hand][mask]),
            "reprojection_error_px": _summary(diagnostics["reprojection_error_px"][:, hand][mask]),
        }
    metrics = {
        "schema_version": "1.0",
        "method": "independent left/right HaMeR 2D rays triangulated by calibrated rectified stereo",
        "ground_truth_used": False,
        "object_or_contact_used": False,
        "scale_or_shift_alignment_applied": False,
        "baseline_m": float(stereo["baseline_m"]),
        "limits": {
            "min_disparity_px": MIN_DISPARITY_PX,
            "max_vertical_or_reprojection_error_px": MAX_REPROJECTION_ERROR_PX,
            "joint_depth_m": [MIN_JOINT_DEPTH_M, MAX_JOINT_DEPTH_M],
            "min_joint_valid_rate": MIN_REQUIRED_JOINT_RATE,
            "min_required_landmarks_frame_valid_rate": MIN_REQUIRED_FRAME_RATE,
        },
        "per_hand": per_hand,
        "accepted_any_hand": bool(accepted_hand.any()),
        "artifacts": {
            "output": str(output_path),
            "left_input": str(left_path),
            "right_input": str(right_path),
            "stereo_calibration": str(root / "calibration/stereo.json"),
        },
        "failure_analysis": failure_analysis,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"status": "evaluated", "metrics": metrics}


def run_hamer_for_view(
    run_dir: Path, camera_view: str, checkpoint: Path, detector_checkpoint: Path,
    python_bin: Path, gpu: int, overwrite: bool,
) -> None:
    view_dir = run_dir / ("hands_hamer" if camera_view == "left" else "hands_right_hamer")
    metadata = view_dir / "metadata.json"
    if metadata.exists() and not overwrite:
        return
    command = [
        str(python_bin), "-m", "video_to_spider.adapters.hamer",
        "--run-dir", str(run_dir),
        "--checkpoint", str(checkpoint),
        "--detector-checkpoint", str(detector_checkpoint),
        "--camera-view", camera_view,
        "--device", "cuda",
        "--overwrite",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    log_path = view_dir / "infer.log"
    view_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--python-bin", type=Path, default=Path("/home/zzx/miniconda3/envs/v2s-hamer/bin/python"))
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "third_party/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
    parser.add_argument("--detector-checkpoint", type=Path, default=REPO_ROOT / "third_party/WiLoR/pretrained_models/detector.pt")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--infer", action="store_true", help="run HaMeR per-view inference before evaluation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--only-run", action="append", default=[], help="run_dir path substring; repeatable")
    args = parser.parse_args()

    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    selected_count = 0
    for row in rows:
        run_dir = Path(row["run_dir"])
        if not (run_dir / "hands/wilor_stereo_raw.npz").is_file():
            continue
        if args.only_run and not any(token in str(run_dir) for token in args.only_run):
            continue
        if args.max_runs is not None and selected_count >= args.max_runs:
            break
        selected_count += 1
        if args.infer:
            run_hamer_for_view(run_dir, "left", args.checkpoint, args.detector_checkpoint, args.python_bin, args.gpu, args.overwrite)
            run_hamer_for_view(run_dir, "right", args.checkpoint, args.detector_checkpoint, args.python_bin, args.gpu, args.overwrite)
        try:
            record = evaluate_hamer_stereo(run_dir, overwrite=args.overwrite)
        except FileExistsError:
            metrics_path = run_dir / "hands_hamer/hamer_stereo_metrics.json"
            record = {"status": "already_evaluated", "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}}
        results.append({"sequence": row["sequence"], "prototype": row["prototype"], "run_dir": str(run_dir), "record": record})

    summary = {"schema_version": "1.0", "stage": "H1 HaMeR stereo-hand ablation", "rows": results}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.matrix.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "status", "hand", "accepted", "joint_valid_rate", "required_frame_rate", "vertical_p95_px", "reprojection_p95_px", "positive_triangulation_ratio"])
        for row in results:
            record = row.get("record") or {}
            metrics = record.get("metrics", {})
            per_hand = metrics.get("per_hand", {})
            if not per_hand:
                writer.writerow([Path(row["run_dir"]).name, record.get("status", "missing"), "", "", "", "", "", "", ""])
                continue
            for side, m in per_hand.items():
                writer.writerow([
                    Path(row["run_dir"]).name,
                    record.get("status"),
                    side,
                    m.get("accepted"),
                    m.get("joint_valid_rate"),
                    m.get("required_landmarks_frame_valid_rate"),
                    m.get("vertical_disparity_abs_px", {}).get("p95"),
                    m.get("reprojection_error_px", {}).get("p95"),
                    "",
                ])
    print(f"summary -> {args.output}")
    print(f"matrix   -> {args.matrix}")
    print(json.dumps(results, indent=2, ensure_ascii=False)[:16000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
