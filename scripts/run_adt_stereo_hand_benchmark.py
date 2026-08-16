#!/usr/bin/env python3
"""P7 ADT stereo-hand auxiliary benchmark.

This benchmark consumes existing WiLoR stereo artifacts and reports the
geometry checks the ADT upstream plan asks for. It does not rerun WiLoR,
FoundationPose, or any third-party model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
OUTPUT_PATH = RUNS_ROOT / "adt_stereo_hand_benchmark_summary.json"
MATRIX_PATH = RUNS_ROOT / "stereo_hand_benchmark_matrix.csv"

REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
HAND_ORDER = ("left", "right")


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
    projected = np.einsum("ij,...j->...i", K, points)
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


def _bone_length_cv(joints: np.ndarray, valid: np.ndarray, hand: int) -> dict[str, float]:
    wrist = joints[:, hand, 0]
    fingertips = joints[:, hand, FINGERTIPS]
    frame_valid = valid[:, hand, 0] & valid[:, hand, FINGERTIPS].all(axis=1)
    lengths = np.linalg.norm(fingertips - wrist[:, None, :], axis=-1)
    usable = lengths[frame_valid]
    cv_values: list[float] = []
    for bone in range(5):
        values = usable[:, bone] if usable.size else np.empty((0,))
        if len(values) < 3 or float(values.mean()) < 1e-6:
            cv_values.append(math.nan)
            continue
        cv_values.append(float(values.std() / values.mean()))
    return {
        "bone_length_cv_median": _percentile(np.asarray(cv_values, dtype=np.float64), 50),
        "bone_length_cv_p95": _percentile(np.asarray(cv_values, dtype=np.float64), 95),
        "bone_length_cv_values": cv_values,
        "usable_bone_frames": int(frame_valid.sum()),
    }


def _failure_analysis_record(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "hands/hand_stereo_failure_analysis.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("per_hand", {})


def _hand_record(run_dir: Path) -> dict[str, Any] | None:
    stereo_path = run_dir / "hands/wilor_stereo_raw.npz"
    left_path = run_dir / "hands/wilor_raw.npz"
    right_path = run_dir / "hands_right/wilor_raw.npz"
    stereo_metrics_path = run_dir / "hands/wilor_stereo_metrics.json"
    if not stereo_path.is_file():
        return {"status": "stereo_artifact_missing", "reason": str(stereo_path)}
    stereo = _load(stereo_path)
    left = _load(left_path)
    right = _load(right_path)
    if stereo is None or left is None or right is None:
        return {
            "status": "input_artifact_missing",
            "reason": {
                "stereo": stereo is None,
                "left": left is None,
                "right": right is None,
            },
        }

    K_left = np.load(run_dir / "calibration/intrinsics.npy")
    K_right = np.load(run_dir / "calibration/intrinsics_right.npy")
    joints = stereo["joints_camera_metric"].astype(np.float64)
    joint_valid = stereo["joint_metric_valid"].astype(bool)
    fps = _fps(run_dir)
    left_abs = left["joints_camera_rootrel"].astype(np.float64) + left["translation_camera"].astype(np.float64)[:, :, None, :]
    right_abs = right["joints_camera_rootrel"].astype(np.float64) + right["translation_camera"].astype(np.float64)[:, :, None, :]
    left_proj = _project(K_left, joints)
    right_proj = _project(K_right, joints)
    left_abs_proj = _project(K_left, left_abs)
    right_abs_proj = _project(K_right, right_abs)

    records: dict[str, Any] = {}
    accepted_any = False
    for hand, side in enumerate(HAND_ORDER):
        mask = joint_valid[:, hand]
        left_obs_valid = np.broadcast_to(left["valid"][:, hand, None], mask.shape)
        right_obs_valid = np.broadcast_to(right["valid"][:, hand, None], mask.shape)
        left_mask = mask & left_obs_valid
        right_mask = mask & right_obs_valid
        both_mask = mask & left_obs_valid & right_obs_valid
        left_reproj = np.linalg.norm(left_proj[:, hand] - left_abs_proj[:, hand], axis=-1)[left_mask]
        right_reproj = np.linalg.norm(right_proj[:, hand] - right_abs_proj[:, hand], axis=-1)[right_mask]
        vertical = np.abs(left_abs_proj[:, hand, :, 1] - right_abs_proj[:, hand, :, 1])[both_mask]
        depths = joints[:, hand, :, 2][mask]
        positive_ratio = float(((depths > 0.05) & (depths < 3.0)).mean()) if depths.size else 0.0

        jitter: dict[str, Any] = {"wrist": {}, "fingertips": {}}
        wrist_values = joints[:, hand, 0][joint_valid[:, hand, 0]]
        jitter["wrist"] = _finite_difference_jitter(wrist_values, fps)
        fingertip_values = []
        for finger in FINGERTIPS:
            finger_valid = joint_valid[:, hand, finger]
            values = joints[:, hand, finger][finger_valid]
            fingertip_values.append(values)
        fingertip_series = np.concatenate(fingertip_values, axis=0) if fingertip_values and fingertip_values[0].size else np.empty((0, 3))
        jitter["fingertips"] = _finite_difference_jitter(fingertip_series, fps)

        metrics_payload = json.loads(stereo_metrics_path.read_text(encoding="utf-8")) if stereo_metrics_path.is_file() else {}
        gate = metrics_payload.get("per_hand", {}).get(side, {})
        accepted = bool(gate.get("accepted", False))
        accepted_any = accepted_any or accepted
        failure_analysis = _failure_analysis_record(run_dir)
        records[side] = {
            "accepted": accepted,
            "joint_valid_rate": float(joint_valid[:, hand].mean()),
            "required_landmarks_frame_valid_rate": float(joint_valid[:, hand][:, REQUIRED].all(axis=1).mean()),
            "failure_reason_counts": failure_analysis.get(side) if failure_analysis else None,
            "left_reprojection_error_px": {
                "median": _percentile(left_reproj, 50),
                "p95": _percentile(left_reproj, 95),
            },
            "right_reprojection_error_px": {
                "median": _percentile(right_reproj, 50),
                "p95": _percentile(right_reproj, 95),
            },
            "vertical_correspondence_error_px": {
                "median": _percentile(vertical, 50),
                "p95": _percentile(vertical, 95),
            },
            "positive_triangulation_ratio": positive_ratio,
            "temporal_jitter": jitter,
            "bone_length": _bone_length_cv(joints, joint_valid, hand),
            "gate_from_artifact": gate,
        }
    return {
        "status": "evaluated",
        "accepted_any_hand": accepted_any,
        "per_hand": records,
        "artifacts": {
            "stereo": str(stereo_path),
            "left": str(left_path),
            "right": str(right_path),
            "metrics": str(stereo_metrics_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    args = parser.parse_args()
    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        record = _hand_record(run_dir)
        results.append(
            {
                "sequence": row["sequence"],
                "prototype": row["prototype"],
                "run_dir": str(run_dir),
                "record": record,
            }
        )
    summary = {"schema_version": "1.0", "stage": "P7 Stereo Hand Auxiliary", "rows": results}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.matrix.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "run",
                "status",
                "hand",
                "accepted",
                "joint_valid_rate",
                "required_landmarks_frame_valid_rate",
                "left_reproj_p95_px",
                "right_reproj_p95_px",
                "vertical_p95_px",
                "positive_triangulation_ratio",
                "wrist_velocity_p95_m_s",
                "wrist_acceleration_p95_m_s2",
                "bone_cv_median",
            ]
        )
        for row in results:
            record = row["record"] or {}
            per_hand = record.get("per_hand", {})
            if not per_hand:
                writer.writerow([Path(row["run_dir"]).name, record.get("status", "missing"), "", "", "", "", "", "", "", "", "", "", ""])
                continue
            for side, metrics in per_hand.items():
                writer.writerow(
                    [
                        Path(row["run_dir"]).name,
                        record.get("status"),
                        side,
                        metrics.get("accepted"),
                        metrics.get("joint_valid_rate"),
                        metrics.get("required_landmarks_frame_valid_rate"),
                        metrics.get("left_reprojection_error_px", {}).get("p95"),
                        metrics.get("right_reprojection_error_px", {}).get("p95"),
                        metrics.get("vertical_correspondence_error_px", {}).get("p95"),
                        metrics.get("positive_triangulation_ratio"),
                        metrics.get("temporal_jitter", {}).get("wrist", {}).get("velocity_p95_m_s"),
                        metrics.get("temporal_jitter", {}).get("wrist", {}).get("acceleration_p95_m_s2"),
                        metrics.get("bone_length", {}).get("bone_length_cv_median"),
                    ]
                )
    print(f"summary -> {args.output}")
    print(f"matrix   -> {args.matrix}")
    print(json.dumps(results, indent=2, ensure_ascii=False)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
