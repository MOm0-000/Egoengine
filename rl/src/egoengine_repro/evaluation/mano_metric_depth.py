"""Evaluation-only validation for WiLoR/MANO metric-depth evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import artifact_record
from .metrics import summary


DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_gt_positive_frame_rate": 0.95,
    "min_paired_frame_rate": 0.90,
    "max_affine_median_error_m": 0.10,
    "max_affine_p95_error_m": 0.20,
}


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _verified_artifact(manifest: dict[str, Any], name: str, manifest_path: Path) -> Path:
    record = manifest.get("artifacts", {}).get(name)
    if not isinstance(record, dict) or "path" not in record:
        raise ValueError(f"ground-truth manifest has no {name} artifact")
    path = Path(record["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    current = artifact_record(path)
    if record.get("sha256") != current["sha256"]:
        raise ValueError(f"ground-truth {name} artifact changed: {path}")
    return path


def _world_to_camera(T_world_camera: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    transforms = np.asarray(T_world_camera, dtype=np.float64)
    points = np.asarray(points_world, dtype=np.float64)
    rotation = np.swapaxes(transforms[:, :3, :3], 1, 2)
    return np.einsum(
        "tij,thkj->thki", rotation,
        points - transforms[:, None, None, :3, 3],
    )


def _depth_comparison(predicted: np.ndarray, target: np.ndarray, paired: np.ndarray) -> dict[str, Any]:
    values = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    use = np.asarray(paired, dtype=bool) & np.isfinite(values) & (values > 0)
    error = values[use] - truth[use]
    ratio = values[use] / truth[use]
    return {
        "paired_count": int(np.sum(use)),
        "signed_error_m": summary(error),
        "absolute_error_m": summary(np.abs(error)),
        "depth_ratio": summary(ratio),
    }


def evaluate_mano_metric_depth_against_gt(
    evidence_metadata_path: str | Path,
    ground_truth_manifest_path: str | Path,
    output_path: str | Path,
    *, thresholds: dict[str, float] | None = None,
) -> Path:
    """Score an automatic MANO depth candidate without exposing GT to inference."""
    evidence_path = Path(evidence_metadata_path).resolve()
    manifest_path = Path(ground_truth_manifest_path).resolve()
    destination = Path(output_path).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("uses_ground_truth") is not True or manifest.get("scope") != "evaluation_only":
        raise ValueError("MANO metric-depth validation requires evaluation-only ground truth")
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    hand_path = _verified_artifact(manifest, "hand", manifest_path)
    camera_path = _verified_artifact(manifest, "camera", manifest_path)
    observations_path = evidence_path.parent / "observations.npz"
    observations = _npz(observations_path)
    hand = _npz(hand_path)
    camera = _npz(camera_path)
    frames = np.asarray(observations["frame_indices"], dtype=np.int64)
    if not (
        np.array_equal(frames, hand["frame_indices"])
        and np.array_equal(frames, camera["frame_indices"])
    ):
        raise ValueError("MANO evidence and evaluation GT timelines differ")
    hand_order = [str(value) for value in hand["hand_order"].tolist()]
    if hand_order != ["left", "right"]:
        raise ValueError(f"unexpected TACO hand order: {hand_order}")
    points_world = np.asarray(hand["T_world_joint"], dtype=np.float64)[..., :3, 3]
    points_camera = _world_to_camera(camera["T_world_camera"], points_world)
    gt_depth = np.median(points_camera[..., 2], axis=-1)
    usable = np.asarray(observations["usable"], dtype=bool)
    raw_depth = np.asarray(observations["raw_depth_median"], dtype=np.float64)
    mano_depth = np.asarray(observations["mano_depth_median"], dtype=np.float64)
    quality = manifest.get("quality_labels", {}).get("camera", "unlabeled")
    proxy = manifest.get("camera_calibration_proxy")
    independent_camera_gt = bool(
        quality == "dataset_calibration_ground_truth"
        and not proxy
    )
    candidates = evidence.get("candidate_fit_diagnostics_by_side", {})
    eligible_sides = set(
        (evidence.get("candidate_scale_shift_evidence") or {}).get("eligible_sides", []),
    )
    per_hand: dict[str, Any] = {}
    for hand_index, side in enumerate(hand_order):
        candidate = candidates.get(side, {})
        gt_positive = np.isfinite(gt_depth[:, hand_index]) & (gt_depth[:, hand_index] > 0)
        paired = usable[:, hand_index] & gt_positive
        affine_available = bool(
            candidate.get("status") == "candidate"
            and np.isfinite(candidate.get("scale", np.nan))
            and float(candidate.get("scale", -1.0)) > 0
        )
        scale_only_available = bool(
            np.isfinite(candidate.get("scale_only", np.nan))
            and float(candidate.get("scale_only", -1.0)) > 0
        )
        comparisons = {
            "raw_depth": _depth_comparison(raw_depth[:, hand_index], gt_depth[:, hand_index], paired),
            "wilor_mano_depth": _depth_comparison(mano_depth[:, hand_index], gt_depth[:, hand_index], paired),
        }
        if scale_only_available:
            comparisons["scale_only_calibrated_depth"] = _depth_comparison(
                float(candidate["scale_only"]) * raw_depth[:, hand_index],
                gt_depth[:, hand_index], paired,
            )
        if affine_available:
            comparisons["affine_calibrated_depth"] = _depth_comparison(
                float(candidate["scale"]) * raw_depth[:, hand_index] + float(candidate["shift_m"]),
                gt_depth[:, hand_index], paired,
            )
        affine_error = comparisons.get("affine_calibrated_depth", {}).get("absolute_error_m", {})
        raw_error = comparisons["raw_depth"]["absolute_error_m"]
        checks = {
            "independent_camera_gt": independent_camera_gt,
            "gt_positive_frame_rate": float(np.mean(gt_positive)) >= limits["min_gt_positive_frame_rate"],
            "paired_frame_rate": float(np.mean(paired)) >= limits["min_paired_frame_rate"],
            "inference_qc_eligible_side": side in eligible_sides,
            "positive_affine_candidate": affine_available,
            "affine_median_error_m": (
                affine_error.get("median") is not None
                and float(affine_error["median"]) < limits["max_affine_median_error_m"]
            ),
            "affine_p95_error_m": (
                affine_error.get("p95") is not None
                and float(affine_error["p95"]) < limits["max_affine_p95_error_m"]
            ),
            "affine_improves_raw_median": (
                affine_error.get("median") is not None and raw_error.get("median") is not None
                and float(affine_error["median"]) < float(raw_error["median"])
            ),
            "affine_improves_raw_p95": (
                affine_error.get("p95") is not None and raw_error.get("p95") is not None
                and float(affine_error["p95"]) < float(raw_error["p95"])
            ),
        }
        per_hand[side] = {
            "usable_frame_rate": float(np.mean(usable[:, hand_index])),
            "gt_positive_frame_rate": float(np.mean(gt_positive)),
            "paired_frame_rate": float(np.mean(paired)),
            "inference_qc_eligible_side": side in eligible_sides,
            "candidate": candidate,
            "comparisons": comparisons,
            "checks": checks,
            "independent_metric_validation_passed": bool(all(checks.values())),
        }
    passed_sides = [
        side for side, result in per_hand.items()
        if result["independent_metric_validation_passed"]
    ]
    report = {
        "schema_version": "1.0",
        "profile": "mano_metric_depth_gt_validation_v1",
        "evaluation_uses_ground_truth": True,
        "ground_truth_scope": "evaluation_only",
        "ground_truth_may_enter_inference": False,
        "source_evidence": artifact_record(evidence_path),
        "source_observations": artifact_record(observations_path),
        "ground_truth_manifest": artifact_record(manifest_path),
        "camera_quality_label": quality,
        "independent_camera_ground_truth": independent_camera_gt,
        "camera_proxy": proxy,
        "thresholds": limits,
        "per_hand": per_hand,
        "passed_sides": passed_sides,
        "independent_metric_validation_passed": bool(passed_sides),
        "fusion_allowed": False,
        "fusion_status": (
            "blocked_pending_controlled_depth_and_pose_improvement"
            if passed_sides else "blocked_independent_metric_validation_failed"
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination
