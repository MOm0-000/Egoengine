"""Triangulate independent left/right hand observations in calibrated stereo.

Both WiLoR passes are used only for image-space joint rays and MANO rotations.
Metric joint positions come from rectified stereo geometry, never from the
object, contact labels, or ground truth.  A fixed reprojection/geometry gate
rejects unreliable joints instead of fitting an episode-specific scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..schemas import validate_wilor_raw


HAND_ORDER = ("left", "right")
MIN_DISPARITY_PX = 1.0
MAX_REPROJECTION_ERROR_PX = 3.0
MIN_JOINT_DEPTH_M = 0.05
MAX_JOINT_DEPTH_M = 3.0
MIN_REQUIRED_JOINT_RATE = 0.70
MIN_REQUIRED_FRAME_RATE = 0.60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def require_stereo_hand_gate(artifact_path: str | Path) -> dict[str, Any]:
    artifact = Path(artifact_path).resolve()
    metrics_path = artifact.with_name("wilor_stereo_metrics.json")
    if not artifact.is_file() or not metrics_path.is_file():
        raise RuntimeError("stereo_hand_gate_missing")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    recorded = metrics.get("artifacts", {})
    for role in (
        "output", "left_input", "right_input", "stereo_calibration",
        "left_intrinsics", "right_intrinsics",
    ):
        record = recorded.get(role, {})
        path = Path(record.get("path", ""))
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise RuntimeError(f"stereo_hand_gate_stale: {role}")
    if Path(recorded["output"]["path"]).resolve() != artifact:
        raise RuntimeError("stereo_hand_gate_stale: output path differs")
    if not metrics.get("accepted_any_hand", False):
        raise RuntimeError("stereo_hand_gate_rejected")
    return metrics


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def project_joints(K: np.ndarray, joints_camera: np.ndarray) -> np.ndarray:
    points = np.asarray(joints_camera, dtype=np.float64)
    projected = np.einsum("ij,...j->...i", K, points)
    return projected[..., :2] / projected[..., 2:3]


def triangulate_rectified_joints(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    *,
    K_left: np.ndarray,
    K_right: np.ndarray,
    baseline_m: float,
    valid_observation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Triangulate in the left camera using the original rectified pixels."""
    left = np.asarray(left_uv, dtype=np.float64)
    right = np.asarray(right_uv, dtype=np.float64)
    observed = np.asarray(valid_observation, dtype=bool)
    if left.shape != right.shape or left.shape[-1] != 2:
        raise ValueError("left/right joint pixels must have a shared (..., 2) shape")
    if observed.shape != left.shape[:-1]:
        raise ValueError("valid_observation does not match joint pixels")
    if not np.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError("baseline_m must be positive and finite")
    fx_left, fy_left = float(K_left[0, 0]), float(K_left[1, 1])
    fx_right, fy_right = float(K_right[0, 0]), float(K_right[1, 1])
    cx_left, cy_left = float(K_left[0, 2]), float(K_left[1, 2])
    cx_right, cy_right = float(K_right[0, 2]), float(K_right[1, 2])
    for name, value in (
        ("fx_left", fx_left), ("fy_left", fy_left),
        ("fx_right", fx_right), ("fy_right", fy_right),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")

    # Rectified cameras may still be exported with slightly different focal
    # lengths or principal points. Work in each camera's normalized image
    # plane rather than assuming the raw-pixel shortcut fx*B/(u_l-u_r).
    x_left = (left[..., 0] - cx_left) / fx_left
    x_right = (right[..., 0] - cx_right) / fx_right
    y_left = (left[..., 1] - cy_left) / fy_left
    y_right = (right[..., 1] - cy_right) / fy_right
    focal_x_equivalent = 0.5 * (fx_left + fx_right)
    focal_y_equivalent = 0.5 * (fy_left + fy_right)
    normalized_disparity = x_left - x_right
    disparity = normalized_disparity * focal_x_equivalent
    vertical = np.abs(y_left - y_right) * focal_y_equivalent
    depth = float(baseline_m) / np.maximum(normalized_disparity, 1e-12)
    joints = np.empty((*left.shape[:-1], 3), dtype=np.float64)
    joints[..., 2] = depth
    joints[..., 0] = x_left * depth
    joints[..., 1] = 0.5 * (y_left + y_right) * depth
    finite = np.isfinite(left).all(axis=-1) & np.isfinite(right).all(axis=-1)
    geometric = (
        observed
        & finite
        & (disparity >= MIN_DISPARITY_PX)
        & (vertical <= MAX_REPROJECTION_ERROR_PX)
        & (depth >= MIN_JOINT_DEPTH_M)
        & (depth <= MAX_JOINT_DEPTH_M)
    )
    projected_left = project_joints(K_left, joints)
    # For a rectified pair, map the left-camera point into the right camera by
    # subtracting the physical baseline from X.
    right_points = joints.copy()
    right_points[..., 0] -= float(baseline_m)
    projected_right = project_joints(K_right, right_points)
    left_error = np.linalg.norm(projected_left - left, axis=-1)
    right_error = np.linalg.norm(projected_right - right, axis=-1)
    reprojection = np.maximum(left_error, right_error)
    valid = geometric & (reprojection <= MAX_REPROJECTION_ERROR_PX)
    joints[~valid] = np.nan
    return joints, valid, {
        "disparity_px": disparity,
        "vertical_disparity_abs_px": vertical,
        "reprojection_error_px": reprojection,
    }


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


def run(args: argparse.Namespace) -> Path:
    root = args.run_dir.resolve()
    left_path = (args.left_artifact or root / "hands/wilor_raw.npz").resolve()
    right_path = (args.right_artifact or root / "hands_right/wilor_raw.npz").resolve()
    output_path = (args.output or root / "hands/wilor_stereo_raw.npz").resolve()
    left_intrinsics_path = (root / "calibration/intrinsics.npy").resolve()
    right_intrinsics_path = (root / "calibration/intrinsics_right.npy").resolve()
    stereo_path = (root / "calibration/stereo.json").resolve()
    metrics_path = output_path.with_name("wilor_stereo_metrics.json")
    if (output_path.exists() or metrics_path.exists()) and not args.overwrite:
        raise FileExistsError(f"stereo hand output exists: {output_path}")
    stereo = json.loads(stereo_path.read_text(encoding="utf-8"))
    if not stereo.get("accepted", False):
        raise RuntimeError("stereo rectification gate is not accepted")
    left = _load_artifact(left_path)
    right = _load_artifact(right_path)
    if not np.array_equal(left["frame_indices"], right["frame_indices"]):
        raise ValueError("left/right WiLoR timelines differ")
    K_left = np.load(left_intrinsics_path).astype(np.float64)
    K_right = np.load(right_intrinsics_path).astype(np.float64)
    left_joints = left["joints_camera_rootrel"] + left["translation_camera"][:, :, None]
    right_joints = right["joints_camera_rootrel"] + right["translation_camera"][:, :, None]
    left_uv = project_joints(K_left, left_joints)
    right_uv = project_joints(K_right, right_joints)
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
    required = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
    frame_valid = joint_valid[:, :, required].all(axis=-1)
    joint_rate = joint_valid.mean(axis=(0, 2))
    frame_rate = frame_valid.mean(axis=0)
    accepted_hand = (joint_rate >= MIN_REQUIRED_JOINT_RATE) & (
        frame_rate >= MIN_REQUIRED_FRAME_RATE
    )
    valid = frame_valid & accepted_hand[None]
    # Use the independently triangulated wrist as translation and keep the
    # official left-view MANO-relative geometry/rotations only where stereo is
    # valid. Full triangulated joints are exported explicitly for optimization.
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
            "vertical_disparity_abs_px": _summary(
                diagnostics["vertical_disparity_abs_px"][:, hand][mask]
            ),
            "reprojection_error_px": _summary(
                diagnostics["reprojection_error_px"][:, hand][mask]
            ),
        }
    metrics = {
        "schema_version": "1.0",
        "method": "independent left/right WiLoR 2D rays triangulated by calibrated rectified stereo",
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
            "output": {"path": str(output_path), "sha256": _sha256(output_path)},
            "left_input": {"path": str(left_path), "sha256": _sha256(left_path)},
            "right_input": {"path": str(right_path), "sha256": _sha256(right_path)},
            "stereo_calibration": {
                "path": str(stereo_path),
                "sha256": _sha256(stereo_path),
            },
            "left_intrinsics": {
                "path": str(left_intrinsics_path),
                "sha256": _sha256(left_intrinsics_path),
            },
            "right_intrinsics": {
                "path": str(right_intrinsics_path),
                "sha256": _sha256(right_intrinsics_path),
            },
        },
        "outputs": [str(output_path), str(metrics_path)],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if not accepted_hand.any():
        raise RuntimeError("stereo_hand_gate_rejected: no hand passes fixed triangulation QC")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--left-artifact", type=Path)
    parser.add_argument("--right-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    print(run(_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
