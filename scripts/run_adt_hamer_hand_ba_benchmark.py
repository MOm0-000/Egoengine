#!/usr/bin/env python3
"""H2 stereo-hand ablation: HaMeR + lightweight calibrated stereo bundle adjustment.

For each detected hand and each frame this solves a small non-linear least
squares problem over shared 3D MANO joints in the left rectified camera. The
residuals keep the H1 HaMeR MANO prediction as a pose prior while enforcing
left/right reprojection, stereo epipolar consistency, positive depth, and
temporal second-difference smoothness. Fixed P7 QC gates are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "runs/adt_depth_benchmark_summary.json"
OUTPUT_PATH = REPO_ROOT / "runs/adt_hamer_hand_ba_benchmark_summary.json"
MATRIX_PATH = REPO_ROOT / "runs/hamer_hand_ba_benchmark_matrix.csv"

REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
HAND_ORDER = ("left", "right")
MAX_REPROJECTION_ERROR_PX = 3.0
MIN_JOINT_DEPTH_M = 0.05
MAX_JOINT_DEPTH_M = 3.0
MIN_REQUIRED_JOINT_RATE = 0.70
MIN_REQUIRED_FRAME_RATE = 0.60


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return math.nan
    return float(np.percentile(values, q))


def _load(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    projected = np.einsum("ij,...j->...i", K, points)
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[..., :2] / projected[..., 2:3]


def _fps(run_dir: Path) -> float:
    frame_index = run_dir / "frames/frame_index.json"
    if frame_index.is_file():
        payload = json.loads(frame_index.read_text(encoding="utf-8"))
        return float(payload.get("fps", 30.0))
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return float(payload.get("fps", 30.0))
    return 30.0


def _finite_difference_jitter(values: np.ndarray, fps: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        return {"velocity_p95_m_s": math.nan, "acceleration_p95_m_s2": math.nan, "jerk_p95_m_s3": math.nan}
    velocity = np.diff(values, axis=0) * fps
    finite_velocity = velocity[np.isfinite(velocity).all(axis=1)]
    acceleration = np.diff(finite_velocity, axis=0) * fps if len(finite_velocity) >= 2 else np.empty((0, values.shape[1]))
    acceleration = acceleration[np.isfinite(acceleration).all(axis=1)]
    jerk = np.diff(acceleration, axis=0) * fps if len(acceleration) >= 2 else np.empty((0, values.shape[1]))
    jerk = jerk[np.isfinite(jerk).all(axis=1)]
    return {
        "velocity_p95_m_s": _percentile(np.linalg.norm(finite_velocity, axis=1), 95),
        "acceleration_p95_m_s2": _percentile(np.linalg.norm(acceleration, axis=1), 95),
        "jerk_p95_m_s3": _percentile(np.linalg.norm(jerk, axis=1), 95),
    }


def _bone_length_cv(joints: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    wrist = joints[:, 0]
    fingertips = joints[:, FINGERTIPS]
    frame_valid = valid[:, 0] & valid[:, FINGERTIPS].all(axis=1)
    lengths = np.linalg.norm(fingertips - wrist[:, None, :], axis=-1)
    usable = lengths[frame_valid]
    cvs = []
    for bone in range(5):
        values = usable[:, bone] if usable.size else np.empty((0,))
        if len(values) < 3 or float(values.mean()) < 1e-6:
            cvs.append(math.nan)
        else:
            cvs.append(float(values.std() / values.mean()))
    return {
        "bone_length_cv_median": _percentile(np.asarray(cvs, dtype=np.float64), 50),
        "bone_length_cv_p95": _percentile(np.asarray(cvs, dtype=np.float64), 95),
        "usable_bone_frames": int(frame_valid.sum()),
    }


def _interpolate_prior(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    result = values.copy()
    timeline = np.arange(len(values))
    for joint in range(values.shape[1]):
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            continue
        for axis in range(3):
            result[:, joint, axis] = np.interp(
                timeline, indices, values[indices, joint, axis],
            )
    return result


def _optimize_hand(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    left_prior: np.ndarray,
    right_prior: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    baseline_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimize one hand across all frames and return joints and QC masks."""
    T, J, _ = left_uv.shape
    observation_valid = left_valid[:, None] & right_valid[:, None]
    finite_obs = np.isfinite(left_uv).all(axis=-1) & np.isfinite(right_uv).all(axis=-1)
    observation_valid = observation_valid & finite_obs

    # Prefer the left HaMeR prior. Fill missed left frames with the right prior.
    prior = _interpolate_prior(left_prior, left_valid)
    right_miss = ~left_valid & right_valid
    prior[right_miss] = right_prior[right_miss]
    prior = _interpolate_prior(prior, left_valid | right_valid)

    initial = prior.copy()
    initial[~np.isfinite(initial)] = 0.0
    baseline = np.asarray([baseline_m, 0.0, 0.0], dtype=np.float64)

    def residual(x: np.ndarray) -> np.ndarray:
        joints = x.reshape(T, J, 3)
        projected_left = _project(K_left, joints)
        projected_right = _project(K_right, joints - baseline)
        terms: list[np.ndarray] = []
        if observation_valid.any():
            terms.append((projected_left[observation_valid] - left_uv[observation_valid]) / 3.0)
            terms.append((projected_right[observation_valid] - right_uv[observation_valid]) / 3.0)
            terms.append((projected_left[observation_valid, 1] - projected_right[observation_valid, 1]) / 3.0)
        terms.append(((joints - prior) * 0.10).ravel())
        if T >= 3:
            terms.append(((joints[2:] - 2.0 * joints[1:-1] + joints[:-2]) * 0.20).ravel())
        terms.append(np.log1p(np.exp(-(joints[..., 2] - MIN_JOINT_DEPTH_M))) * 0.02)
        return np.concatenate([np.asarray(term, dtype=np.float64).ravel() for term in terms])

    result = least_squares(
        residual,
        initial.ravel(),
        method="trf",
        max_nfev=120,
        xtol=1e-5,
        ftol=1e-5,
    )
    joints = result.x.reshape(T, J, 3)
    projected_left = _project(K_left, joints)
    projected_right = _project(K_right, joints - baseline)
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
    return joints, joint_valid, left_error, right_error, vertical


def _evaluate_run(run_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    from video_to_spider.schemas import validate_wilor_raw

    root = run_dir.resolve()
    left_path = root / "hands_hamer/hamer_raw.npz"
    right_path = root / "hands_right_hamer/hamer_raw.npz"
    output_path = root / "hands_hamer_ba/hamer_stereo_ba_raw.npz"
    metrics_path = output_path.with_name("hamer_stereo_ba_metrics.json")
    if (output_path.exists() or metrics_path.exists()) and not overwrite:
        raise FileExistsError(f"H2 output exists: {output_path}")

    left = _load(left_path)
    right = _load(right_path)
    if left is None or right is None:
        return {"status": "input_artifact_missing", "left_exists": left_path.is_file(), "right_exists": right_path.is_file()}
    if not np.array_equal(left["frame_indices"], right["frame_indices"]):
        raise ValueError("left/right HaMeR timelines differ")

    stereo = json.loads((root / "calibration/stereo.json").read_text(encoding="utf-8"))
    if not stereo.get("accepted", False):
        raise RuntimeError("stereo rectification gate is not accepted")
    baseline_m = float(stereo["baseline_m"])
    K_left = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    K_right = np.load(root / "calibration/intrinsics_right.npy").astype(np.float64)
    left_abs = left["joints_camera_rootrel"] + left["translation_camera"][:, :, None]
    right_abs = right["joints_camera_rootrel"] + right["translation_camera"][:, :, None]
    left_uv = _project(K_left, left_abs)
    right_uv = _project(K_right, right_abs)
    fps = _fps(root)

    optimized_joints = np.zeros((len(left_abs), 2, 21, 3), dtype=np.float64)
    joint_valid = np.zeros((len(left_abs), 2, 21), dtype=bool)
    left_error_all = np.full((len(left_abs), 2, 21), np.nan, dtype=np.float64)
    right_error_all = np.full((len(left_abs), 2, 21), np.nan, dtype=np.float64)
    vertical_all = np.full((len(left_abs), 2, 21), np.nan, dtype=np.float64)
    failure_counts: dict[str, Any] = {}
    for hand, side in enumerate(HAND_ORDER):
        joints, jvalid, left_err, right_err, vertical = _optimize_hand(
            left_uv[:, hand],
            right_uv[:, hand],
            left["valid"][:, hand],
            right["valid"][:, hand],
            left_abs[:, hand],
            right_abs[:, hand],
            K_left,
            K_right,
            baseline_m,
        )
        optimized_joints[:, hand] = joints
        joint_valid[:, hand] = jvalid
        left_error_all[:, hand] = left_err
        right_error_all[:, hand] = right_err
        vertical_all[:, hand] = vertical
        obs = left["valid"][:, hand, None] & right["valid"][:, hand, None]
        failure_counts[side] = {
            "input_invalid": int((~obs.any(axis=1)).sum()),
            "nonfinite_projection": int((~np.isfinite(left_uv[:, hand]).all(axis=-1) | ~np.isfinite(right_uv[:, hand]).all(axis=-1)).sum()),
            "reprojection_above_max_px": int((left_err > MAX_REPROJECTION_ERROR_PX).sum() + (right_err > MAX_REPROJECTION_ERROR_PX).sum()),
            "vertical_above_max_px": int((vertical > MAX_REPROJECTION_ERROR_PX).sum()),
            "depth_out_of_range_m": int(((joints[..., 2] < MIN_JOINT_DEPTH_M) | (joints[..., 2] > MAX_JOINT_DEPTH_M)).sum()),
        }

    frame_valid = joint_valid[:, :, REQUIRED].all(axis=-1)
    joint_rate = joint_valid.mean(axis=(0, 2))
    frame_rate = frame_valid.mean(axis=0)
    accepted_hand = (joint_rate >= MIN_REQUIRED_JOINT_RATE) & (frame_rate >= MIN_REQUIRED_FRAME_RATE)
    valid = frame_valid & accepted_hand[None]
    wrist = np.where(valid[..., None], optimized_joints[:, :, 0], 0.0)
    score = np.minimum(left["score"], right["score"]).astype(np.float32)
    score[~valid] = 0.0

    payload = {
        key: np.asarray(left[key])
        for key in (
            "frame_indices", "timestamps_s", "side", "mano_global_orient",
            "mano_hand_pose", "mano_betas", "vertices_camera_rootrel",
        )
        if key in left
    }
    joints_rootrel = optimized_joints - wrist[:, :, None, :]
    payload.update(
        {
            "valid": valid,
            "score": score,
            "joints_camera_rootrel": joints_rootrel.astype(np.float32),
            "translation_camera": wrist.astype(np.float32),
            "joints_camera_metric": np.nan_to_num(optimized_joints, nan=0.0).astype(np.float32),
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
            "left_reprojection_error_px": {
                "median": _percentile(left_error_all[:, hand][mask], 50),
                "p95": _percentile(left_error_all[:, hand][mask], 95),
            },
            "right_reprojection_error_px": {
                "median": _percentile(right_error_all[:, hand][mask], 50),
                "p95": _percentile(right_error_all[:, hand][mask], 95),
            },
            "vertical_correspondence_error_px": {
                "median": _percentile(vertical_all[:, hand][mask], 50),
                "p95": _percentile(vertical_all[:, hand][mask], 95),
            },
            "positive_triangulation_ratio": float(
                ((optimized_joints[:, hand, :, 2] > 0.05) & (optimized_joints[:, hand, :, 2] < 3.0)).mean()
            ),
            "failure_reason_counts": failure_counts[side],
            "temporal_jitter": {
                "wrist": _finite_difference_jitter(optimized_joints[:, hand, 0][joint_valid[:, hand, 0]], fps),
                "fingertips": _finite_difference_jitter(
                    np.concatenate([optimized_joints[:, hand, finger][joint_valid[:, hand, finger]] for finger in FINGERTIPS], axis=0),
                    fps,
                ),
            },
            "bone_length": _bone_length_cv(optimized_joints[:, hand], joint_valid[:, hand]),
        }
    metrics = {
        "schema_version": "1.0",
        "method": "HaMeR + calibrated stereo bundle adjustment of shared MANO joints",
        "ground_truth_used": False,
        "object_or_contact_used": False,
        "scale_or_shift_alignment_applied": False,
        "baseline_m": baseline_m,
        "limits": {
            "max_reprojection_error_px": MAX_REPROJECTION_ERROR_PX,
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
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"status": "evaluated", "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--only-run", action="append", default=[])
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
        try:
            record = _evaluate_run(run_dir, overwrite=args.overwrite)
        except FileExistsError:
            metrics_path = run_dir / "hands_hamer_ba/hamer_stereo_ba_metrics.json"
            record = {"status": "already_evaluated", "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}}
        results.append({"sequence": row["sequence"], "prototype": row["prototype"], "run_dir": str(run_dir), "record": record})

    summary = {"schema_version": "1.0", "stage": "H2 HaMeR stereo bundle adjustment", "rows": results}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.matrix.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "status", "hand", "accepted", "joint_valid_rate", "required_frame_rate", "left_reproj_p95_px", "right_reproj_p95_px", "vertical_p95_px", "positive_triangulation_ratio", "wrist_velocity_p95_m_s", "bone_cv_median"])
        for row in results:
            record = row.get("record") or {}
            metrics = record.get("metrics", {})
            per_hand = metrics.get("per_hand", {})
            if not per_hand:
                writer.writerow([Path(row["run_dir"]).name, record.get("status", "missing"), "", "", "", "", "", "", "", "", "", ""])
                continue
            for side, m in per_hand.items():
                writer.writerow([
                    Path(row["run_dir"]).name,
                    record.get("status"),
                    side,
                    m.get("accepted"),
                    m.get("joint_valid_rate"),
                    m.get("required_landmarks_frame_valid_rate"),
                    m.get("left_reprojection_error_px", {}).get("p95"),
                    m.get("right_reprojection_error_px", {}).get("p95"),
                    m.get("vertical_correspondence_error_px", {}).get("p95"),
                    m.get("positive_triangulation_ratio"),
                    m.get("temporal_jitter", {}).get("wrist", {}).get("velocity_p95_m_s"),
                    m.get("bone_length", {}).get("bone_length_cv_median"),
                ])
    print(f"summary -> {args.output}")
    print(f"matrix   -> {args.matrix}")
    print(json.dumps(results, indent=2, ensure_ascii=False)[:16000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
