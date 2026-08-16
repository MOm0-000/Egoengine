#!/usr/bin/env python3
"""H5: best available proposal + stereo MANO/depth/bone bundle adjustment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "runs/adt_depth_benchmark_summary.json"
OUTPUT_PATH = REPO_ROOT / "runs/adt_hamer_hand_h5_benchmark_summary.json"
MATRIX_PATH = REPO_ROOT / "runs/hamer_hand_h5_benchmark_matrix.csv"

sys_path = str(REPO_ROOT / "scripts")
import sys

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
import run_adt_hamer_hand_ba_benchmark as h2  # noqa: E402

REQUIRED = h2.REQUIRED
FINGERTIPS = h2.FINGERTIPS
HAND_ORDER = h2.HAND_ORDER
MAX_REPROJECTION_ERROR_PX = h2.MAX_REPROJECTION_ERROR_PX
MIN_JOINT_DEPTH_M = h2.MIN_JOINT_DEPTH_M
MAX_JOINT_DEPTH_M = h2.MAX_JOINT_DEPTH_M
MIN_REQUIRED_JOINT_RATE = h2.MIN_REQUIRED_JOINT_RATE
MIN_REQUIRED_FRAME_RATE = h2.MIN_REQUIRED_FRAME_RATE


def _sample_depth(depth_m: np.ndarray, valid: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv, dtype=np.float64)
    rows = np.clip(np.rint(uv[..., 1]).astype(np.int64), 0, depth_m.shape[0] - 1)
    cols = np.clip(np.rint(uv[..., 0]).astype(np.int64), 0, depth_m.shape[1] - 1)
    frame_indices = np.arange(uv.shape[0], dtype=np.int64)[:, None]
    depths = depth_m[frame_indices, rows, cols]
    valid_depth = valid[frame_indices, rows, cols] & np.isfinite(depths) & (depths > 0.05) & (depths < 3.0)
    return depths, valid_depth


def _optimize_hand_h5(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    left_prior: np.ndarray,
    right_prior: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    baseline_m: float,
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, J, _ = left_uv.shape
    observation_valid = left_valid[:, None] & right_valid[:, None]
    finite_obs = np.isfinite(left_uv).all(axis=-1) & np.isfinite(right_uv).all(axis=-1)
    observation_valid = observation_valid & finite_obs

    prior = h2._interpolate_prior(left_prior, left_valid)
    right_miss = ~left_valid & right_valid
    prior[right_miss] = right_prior[right_miss]
    prior = h2._interpolate_prior(prior, left_valid | right_valid)
    initial = prior.copy()
    initial[~np.isfinite(initial)] = 0.0
    baseline = np.asarray([baseline_m, 0.0, 0.0], dtype=np.float64)

    # MANO-shaped hand prior: wrist-to-fingertip lengths should be temporally stable.
    def bone_lengths(joints: np.ndarray) -> np.ndarray:
        wrist = joints[..., 0:1, :]
        tips = joints[..., FINGERTIPS, :]
        return np.linalg.norm(tips - wrist, axis=-1)

    prior_bones = bone_lengths(prior)
    prior_bone_median = np.median(prior_bones[np.isfinite(prior_bones)])

    def residual(x: np.ndarray) -> np.ndarray:
        joints = x.reshape(T, J, 3)
        projected_left = h2._project(K_left, joints)
        projected_right = h2._project(K_right, joints - baseline)
        terms: list[np.ndarray] = []
        if observation_valid.any():
            terms.append((projected_left[observation_valid] - left_uv[observation_valid]) / 3.0)
            terms.append((projected_right[observation_valid] - right_uv[observation_valid]) / 3.0)
            terms.append((projected_left[observation_valid, 1] - projected_right[observation_valid, 1]) / 3.0)
        # MANO prior from HaMeR.
        terms.append(((joints - prior) * 0.12).ravel())
        # Shared hand shape: bone lengths near their prior median.
        bones = bone_lengths(joints)
        finite_bones = np.isfinite(bones)
        terms.append(((bones[finite_bones] - prior_bone_median) * 0.15).ravel())
        # FoundationStereo depth anchor on required landmarks.
        required_uv = projected_left[:, REQUIRED]
        sampled, sampled_valid = _sample_depth(depth_m, depth_valid, required_uv)
        depth_mask = observation_valid[:, REQUIRED] & sampled_valid
        if depth_mask.any():
            terms.append(((joints[:, REQUIRED, 2][depth_mask] - sampled[depth_mask]) * 0.25).ravel())
        # Temporal second difference.
        if T >= 3:
            terms.append(((joints[2:] - 2.0 * joints[1:-1] + joints[:-2]) * 0.20).ravel())
        terms.append(np.log1p(np.exp(-(joints[..., 2] - MIN_JOINT_DEPTH_M))) * 0.02)
        return np.concatenate([np.asarray(term, dtype=np.float64).ravel() for term in terms])

    result = least_squares(residual, initial.ravel(), method="trf", max_nfev=120, xtol=1e-5, ftol=1e-5)
    joints = result.x.reshape(T, J, 3)
    projected_left = h2._project(K_left, joints)
    projected_right = h2._project(K_right, joints - baseline)
    left_error = np.linalg.norm(projected_left - left_uv, axis=-1)
    right_error = np.linalg.norm(projected_right - right_uv, axis=-1)
    vertical = np.abs(projected_left[..., 1] - projected_right[..., 1])
    joint_valid = (
        observation_valid
        & (joints[..., 2] >= MIN_JOINT_DEPTH_M)
        & (joints[..., 2] <= MAX_JOINT_DEPTH_M)
        & (left_error <= MAX_REPROJECTION_ERROR_PX)
        & (right_error <= MAX_REPROJECTION_ERROR_PX)
        & (vertical <= MAX_REPROJECTION_ERROR_PX)
    )
    return joints, joint_valid, left_error, right_error


def _evaluate_run(run_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    from video_to_spider.schemas import validate_wilor_raw

    root = run_dir.resolve()
    left_path = root / "hands_hamer/hamer_raw.npz"
    right_path = root / "hands_right_hamer/hamer_raw.npz"
    output_path = root / "hands_hamer_h5/hamer_stereo_h5_raw.npz"
    metrics_path = output_path.with_name("hamer_stereo_h5_metrics.json")
    if (output_path.exists() or metrics_path.exists()) and not overwrite:
        raise FileExistsError(f"H5 output exists: {output_path}")

    left = h2._load(left_path)
    right = h2._load(right_path)
    if left is None or right is None:
        return {"status": "input_artifact_missing", "left_exists": left_path.is_file(), "right_exists": right_path.is_file()}

    stereo = json.loads((root / "calibration/stereo.json").read_text(encoding="utf-8"))
    if not stereo.get("accepted", False):
        raise RuntimeError("stereo rectification gate is not accepted")
    baseline_m = float(stereo["baseline_m"])
    K_left = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    K_right = np.load(root / "calibration/intrinsics_right.npy").astype(np.float64)
    depth_path = root / "depth/metric_depth.zarr"
    if not depth_path.is_dir():
        raise RuntimeError(f"missing depth zarr {depth_path}")
    depth_group = zarr.open(str(depth_path), mode="r")
    depth_lookup = {int(frame): index for index, frame in enumerate(depth_group["frame_indices"])}
    left_abs = left["joints_camera_rootrel"] + left["translation_camera"][:, :, None]
    right_abs = right["joints_camera_rootrel"] + right["translation_camera"][:, :, None]
    left_uv = h2._project(K_left, left_abs)
    right_uv = h2._project(K_right, right_abs)
    fps = h2._fps(root)

    optimized_joints = np.zeros((len(left_abs), 2, 21, 3), dtype=np.float64)
    joint_valid = np.zeros((len(left_abs), 2, 21), dtype=bool)
    left_error_all = np.full((len(left_abs), 2, 21), np.nan, dtype=np.float64)
    right_error_all = np.full((len(left_abs), 2, 21), np.nan, dtype=np.float64)
    for hand, side in enumerate(HAND_ORDER):
        depth_here = np.zeros((len(left_abs), depth_group["depth_m"].shape[1], depth_group["depth_m"].shape[2]), dtype=np.float32)
        valid_here = np.zeros_like(depth_here, dtype=bool)
        for frame_index in range(len(left_abs)):
            frame = int(left["frame_indices"][frame_index])
            depth_at = depth_lookup.get(frame)
            if depth_at is None:
                continue
            depth_here[frame_index] = np.asarray(depth_group["depth_m"][depth_at], dtype=np.float32)
            valid_here[frame_index] = np.asarray(depth_group["valid"][depth_at], dtype=bool)
        joints, jvalid, left_err, right_err = _optimize_hand_h5(
            left_uv[:, hand],
            right_uv[:, hand],
            left["valid"][:, hand],
            right["valid"][:, hand],
            left_abs[:, hand],
            right_abs[:, hand],
            K_left,
            K_right,
            baseline_m,
            depth_here,
            valid_here,
        )
        optimized_joints[:, hand] = joints
        joint_valid[:, hand] = jvalid
        left_error_all[:, hand] = left_err
        right_error_all[:, hand] = right_err

    frame_valid = joint_valid[:, :, REQUIRED].all(axis=-1)
    joint_rate = joint_valid.mean(axis=(0, 2))
    frame_rate = frame_valid.mean(axis=0)
    accepted_hand = (joint_rate >= MIN_REQUIRED_JOINT_RATE) & (frame_rate >= MIN_REQUIRED_FRAME_RATE)
    valid = frame_valid & accepted_hand[None]
    wrist = np.where(valid[..., None], optimized_joints[:, :, 0], 0.0)
    score = np.minimum(left["score"], right["score"]).astype(np.float32)
    score[~valid] = 0.0
    joints_rootrel = optimized_joints - wrist[:, :, None, :]
    payload = {
        key: np.asarray(left[key])
        for key in ("frame_indices", "timestamps_s", "side", "mano_global_orient", "mano_hand_pose", "mano_betas", "vertices_camera_rootrel")
        if key in left
    }
    payload.update({
        "valid": valid,
        "score": score,
        "joints_camera_rootrel": joints_rootrel.astype(np.float32),
        "translation_camera": wrist.astype(np.float32),
        "joints_camera_metric": np.nan_to_num(optimized_joints, nan=0.0).astype(np.float32),
        "joint_metric_valid": joint_valid,
    })
    validate_wilor_raw(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    per_hand = {}
    for hand, side in enumerate(HAND_ORDER):
        mask = joint_valid[:, hand]
        per_hand[side] = {
            "accepted": bool(accepted_hand[hand]),
            "joint_valid_rate": float(joint_rate[hand]),
            "required_landmarks_frame_valid_rate": float(frame_rate[hand]),
            "left_reprojection_error_px": {"median": h2._percentile(left_error_all[:, hand][mask], 50), "p95": h2._percentile(left_error_all[:, hand][mask], 95)},
            "right_reprojection_error_px": {"median": h2._percentile(right_error_all[:, hand][mask], 50), "p95": h2._percentile(right_error_all[:, hand][mask], 95)},
            "positive_triangulation_ratio": float(((optimized_joints[:, hand, :, 2] > 0.05) & (optimized_joints[:, hand, :, 2] < 3.0)).mean()),
            "temporal_jitter": {
                "wrist": h2._finite_difference_jitter(optimized_joints[:, hand, 0][joint_valid[:, hand, 0]], fps),
                "fingertips": h2._finite_difference_jitter(np.concatenate([optimized_joints[:, hand, finger][joint_valid[:, hand, finger]] for finger in FINGERTIPS], axis=0), fps),
            },
            "bone_length": h2._bone_length_cv(optimized_joints[:, hand], joint_valid[:, hand]),
        }
    metrics = {
        "schema_version": "1.0",
        "method": "Best available proposal (HaMeR) + stereo MANO/depth/bone bundle adjustment",
        "ground_truth_used": False,
        "object_or_contact_used": False,
        "scale_or_shift_alignment_applied": False,
        "per_hand": per_hand,
        "accepted_any_hand": bool(accepted_hand.any()),
        "artifacts": {"output": str(output_path), "left_input": str(left_path), "right_input": str(right_path)},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"status": "evaluated", "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--only-run", action="append", default=[])
    args = parser.parse_args()
    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    results = []
    selected = 0
    for row in rows:
        run_dir = Path(row["run_dir"])
        if not (run_dir / "hands_hamer/hamer_raw.npz").is_file():
            continue
        if args.only_run and not any(token in str(run_dir) for token in args.only_run):
            continue
        if args.max_runs is not None and selected >= args.max_runs:
            break
        selected += 1
        try:
            record = _evaluate_run(run_dir, overwrite=args.overwrite)
        except FileExistsError:
            metrics_path = run_dir / "hands_hamer_h5/hamer_stereo_h5_metrics.json"
            record = {"status": "already_evaluated", "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}}
        results.append({"sequence": row["sequence"], "prototype": row["prototype"], "run_dir": str(run_dir), "record": record})
    OUTPUT_PATH.write_text(json.dumps({"schema_version": "1.0", "stage": "H5 best proposal + stereo MANO BA", "rows": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with MATRIX_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "status", "hand", "accepted", "joint_valid_rate", "required_frame_rate", "left_reproj_p95_px", "right_reproj_p95_px", "positive_triangulation_ratio", "wrist_velocity_p95_m_s", "bone_cv_median"])
        for row in results:
            record = row.get("record") or {}
            metrics = record.get("metrics", {})
            per_hand = metrics.get("per_hand", {})
            if not per_hand:
                writer.writerow([Path(row["run_dir"]).name, record.get("status", "missing"), "", "", "", "", "", "", "", "", ""])
                continue
            for side, m in per_hand.items():
                writer.writerow([Path(row["run_dir"]).name, record.get("status"), side, m.get("accepted"), m.get("joint_valid_rate"), m.get("required_landmarks_frame_valid_rate"), m.get("left_reprojection_error_px", {}).get("p95"), m.get("right_reprojection_error_px", {}).get("p95"), m.get("positive_triangulation_ratio"), m.get("temporal_jitter", {}).get("wrist", {}).get("velocity_p95_m_s"), m.get("bone_length", {}).get("bone_length_cv_median")])
    print(json.dumps(results, indent=2, ensure_ascii=False)[:16000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
