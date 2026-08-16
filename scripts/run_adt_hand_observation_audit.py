#!/usr/bin/env python3
"""Lightweight ADT stereo-hand 2D observation audit.

This script reuses existing WiLoR and HaMeR artifacts only. It does not rerun
any model and does not consume ADT hand GT (which is unavailable).
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
SUMMARY_PATH = REPO_ROOT / "runs/adt_depth_benchmark_summary.json"
OUTPUT_PATH = REPO_ROOT / "runs/adt_hand_observation_audit_summary.json"
MATRIX_PATH = REPO_ROOT / "runs/adt_hand_observation_audit_matrix.csv"
HAND_ORDER = ("left", "right")
REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)


def _load(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = np.einsum("ij,...j->...i", K, points)
    return projected[..., :2] / projected[..., 2:3]


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else math.nan


def _model_side_stats(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    side_index: int,
    K_left: np.ndarray,
    K_right: np.ndarray,
) -> dict[str, Any]:
    side = HAND_ORDER[side_index]
    left_valid = left["valid"][:, side_index].astype(bool)
    right_valid = right["valid"][:, side_index].astype(bool)
    left_required = left_valid.copy()
    right_required = right_valid.copy()
    left_joints = left["joints_camera_rootrel"][:, side_index]
    right_joints = right["joints_camera_rootrel"][:, side_index]
    left_required &= np.isfinite(left_joints[:, REQUIRED]).all(axis=(1, 2))
    right_required &= np.isfinite(right_joints[:, REQUIRED]).all(axis=(1, 2))
    both = left_valid & right_valid
    epipolar = math.nan
    if both.any():
        left_abs = left_joints + left["translation_camera"][:, side_index, None]
        right_abs = right_joints + right["translation_camera"][:, side_index, None]
        left_uv = _project(K_left, left_abs)
        right_uv = _project(K_right, right_abs)
        vertical = np.abs(left_uv[:, :, 1] - right_uv[:, :, 1])
        epipolar = _percentile(vertical[both], 95)
    return {
        "side": side,
        "frames": int(len(left_valid)),
        "left_detected_frames": int(left_valid.sum()),
        "right_detected_frames": int(right_valid.sum()),
        "left_detection_rate": float(left_valid.mean()),
        "right_detection_rate": float(right_valid.mean()),
        "left_required_frames": int(left_required.sum()),
        "right_required_frames": int(right_required.sum()),
        "left_required_rate": float(left_required.mean()),
        "right_required_rate": float(right_required.mean()),
        "both_detected_frames": int(both.sum()),
        "both_detection_rate": float(both.mean()),
        "left_median_confidence": _percentile(left["score"][left_valid, side_index], 50),
        "right_median_confidence": _percentile(right["score"][right_valid, side_index], 50),
        "vertical_disparity_p95_px": epipolar,
    }


def _audit_run(run_dir: Path) -> dict[str, Any]:
    K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
    K_right = np.load(run_dir / "calibration/intrinsics_right.npy").astype(np.float64)
    result: dict[str, Any] = {}
    for model, paths in (
        (
            "wilor",
            {
                "left": run_dir / "hands/wilor_raw.npz",
                "right": run_dir / "hands_right/wilor_raw.npz",
            },
        ),
        (
            "hamer",
            {
                "left": run_dir / "hands_hamer/hamer_raw.npz",
                "right": run_dir / "hands_right_hamer/hamer_raw.npz",
            },
        ),
    ):
        left = _load(paths["left"])
        right = _load(paths["right"])
        if left is None or right is None:
            result[model] = {
                "status": "artifacts_missing",
                "left_exists": paths["left"].is_file(),
                "right_exists": paths["right"].is_file(),
            }
            continue
        result[model] = {
            "status": "evaluated",
            "per_side": {
                side: _model_side_stats(left, right, side_index, K_left, K_right)
                for side_index, side in enumerate(HAND_ORDER)
            },
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    args = parser.parse_args()
    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        results.append(
            {
                "sequence": row["sequence"],
                "prototype": row["prototype"],
                "window": row["window"],
                "run_dir": str(run_dir),
                "models": _audit_run(run_dir),
            }
        )
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.matrix.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "run",
                "model",
                "side",
                "left_detection_rate",
                "right_detection_rate",
                "both_detection_rate",
                "left_required_rate",
                "right_required_rate",
                "left_median_confidence",
                "right_median_confidence",
                "vertical_disparity_p95_px",
            ]
        )
        for row in results:
            run_name = Path(row["run_dir"]).name
            for model, payload in row["models"].items():
                if payload.get("status") != "evaluated":
                    writer.writerow([run_name, model, "", payload.get("status"), "", "", "", "", "", "", ""])
                    continue
                for side, metrics in payload["per_side"].items():
                    writer.writerow(
                        [
                            run_name,
                            model,
                            side,
                            metrics["left_detection_rate"],
                            metrics["right_detection_rate"],
                            metrics["both_detection_rate"],
                            metrics["left_required_rate"],
                            metrics["right_required_rate"],
                            metrics["left_median_confidence"],
                            metrics["right_median_confidence"],
                            metrics["vertical_disparity_p95_px"],
                        ]
                    )
    print(f"summary -> {args.output}")
    print(f"matrix -> {args.matrix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
