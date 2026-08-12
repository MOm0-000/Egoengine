"""Metric-depth gates for automatic monocular and calibrated-stereo runs.

DA3 is the primary metric-depth source.  UniDepth is deliberately kept as an
independent checker: this module never rescales or blends either prediction.
FoundationStereo is checked against the ingested calibration, geometry labels,
global coverage, and automatic object-mask coverage. No path fits scale/shift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LIMITS = {
    "min_valid_pixel_ratio": 0.98,
    "min_object_usable_frame_ratio": 0.75,
    "secondary_to_primary_ratio_min": 0.5,
    "secondary_to_primary_ratio_max": 2.0,
    "max_object_log_ratio_robust_mad": 0.12,
    "max_background_log_ratio_robust_mad": 0.10,
    "background_stride_px": 16,
}

DEFAULT_STEREO_LIMITS = {
    "min_valid_pixel_ratio": 0.75,
    "min_object_valid_pixel_ratio": 0.75,
    "min_object_usable_frame_ratio": 0.75,
    "min_depth_m": 0.05,
    "max_depth_m": 20.0,
    "max_invalid_depth_label_ratio": 0.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_fingerprint(path: Path) -> str:
    """Hash a file or directory tree without relying on mtimes or inode order."""
    source = path.resolve()
    if source.is_file():
        return _sha256(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = file_path.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_hash = bytes.fromhex(_sha256(file_path))
        digest.update(file_hash)
    return digest.hexdigest()


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "median": -1.0, "p05": -1.0, "p95": -1.0}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def _robust_mad(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return math.inf
    center = np.median(array)
    return float(1.4826 * np.median(np.abs(array - center)))


def evaluate_depth_consistency(
    run_dir: str | Path,
    *,
    primary_path: str | Path | None = None,
    secondary_path: str | Path | None = None,
    limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate two already-inferred metric-depth artifacts without fitting them."""
    import zarr

    root = Path(run_dir).resolve()
    primary_file = Path(primary_path or root / "depth/metric_depth.zarr").resolve()
    secondary_file = Path(
        secondary_path or root / "depth_crosscheck/unidepth_depth.zarr"
    ).resolve()
    primary_metadata = primary_file.parent / "metadata.json"
    secondary_metadata = secondary_file.parent / "metadata.json"
    object_masks_path = root / "segmentation/object_masks.npz"
    hand_masks_path = root / "segmentation/hand_masks.npz"
    for required in (
        primary_file,
        secondary_file,
        primary_metadata,
        secondary_metadata,
        object_masks_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    thresholds = {**DEFAULT_LIMITS, **(limits or {})}
    primary = zarr.open_group(str(primary_file), mode="r")
    secondary = zarr.open_group(str(secondary_file), mode="r")
    primary_indices = np.asarray(primary["frame_indices"], dtype=np.int64)
    secondary_indices = np.asarray(secondary["frame_indices"], dtype=np.int64)
    timeline_equal = bool(np.array_equal(primary_indices, secondary_indices))
    if not timeline_equal:
        raise ValueError("primary and secondary depth timelines differ")
    if primary["depth_m"].shape != secondary["depth_m"].shape:
        raise ValueError("primary and secondary depth shapes differ")

    with np.load(object_masks_path, allow_pickle=False) as artifact:
        mask_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
        object_masks = np.asarray(artifact["masks"], dtype=bool)
        object_valid = np.asarray(artifact["valid"], dtype=bool)
    mask_lookup = {int(frame): index for index, frame in enumerate(mask_indices)}
    if not all(int(frame) in mask_lookup for frame in primary_indices):
        raise ValueError("object-mask timeline does not cover the depth timeline")
    selected = np.asarray([mask_lookup[int(frame)] for frame in primary_indices], dtype=np.int64)
    object_masks = object_masks[selected]
    object_valid = object_valid[selected]

    hand_masks = np.zeros_like(object_masks)
    if hand_masks_path.exists():
        with np.load(hand_masks_path, allow_pickle=False) as artifact:
            hand_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
            hand_lookup = {int(frame): index for index, frame in enumerate(hand_indices)}
            if all(int(frame) in hand_lookup for frame in primary_indices):
                hand_selected = np.asarray(
                    [hand_lookup[int(frame)] for frame in primary_indices], dtype=np.int64
                )
                hand_masks = np.asarray(artifact["masks"], dtype=bool)[hand_selected]

    primary_valid_count = 0
    secondary_valid_count = 0
    pixel_count = 0
    object_ratios: list[float] = []
    background_log_ratios: list[float] = []
    usable_object_frames = 0
    stride = int(thresholds["background_stride_px"])
    for index in range(len(primary_indices)):
        primary_depth = np.asarray(primary["depth_m"][index], dtype=np.float32)
        secondary_depth = np.asarray(secondary["depth_m"][index], dtype=np.float32)
        primary_valid = np.isfinite(primary_depth) & (primary_depth > 0)
        secondary_valid = np.isfinite(secondary_depth) & (secondary_depth > 0)
        primary_valid_count += int(primary_valid.sum())
        secondary_valid_count += int(secondary_valid.sum())
        pixel_count += int(primary_depth.size)

        object_usable = object_masks[index] & primary_valid & secondary_valid
        coverage = float(object_usable.sum() / max(1, object_masks[index].sum()))
        if object_valid[index] and coverage >= 0.75:
            primary_median = float(np.median(primary_depth[object_usable]))
            secondary_median = float(np.median(secondary_depth[object_usable]))
            object_ratios.append(secondary_median / primary_median)
            usable_object_frames += 1

        background = ~(object_masks[index] | hand_masks[index])
        background = background[::stride, ::stride]
        both = (
            primary_valid[::stride, ::stride]
            & secondary_valid[::stride, ::stride]
            & background
        )
        if both.any():
            log_ratio = np.log(
                secondary_depth[::stride, ::stride][both]
                / primary_depth[::stride, ::stride][both]
            )
            background_log_ratios.append(float(np.median(log_ratio)))

    primary_valid_ratio = primary_valid_count / max(1, pixel_count)
    secondary_valid_ratio = secondary_valid_count / max(1, pixel_count)
    object_usable_ratio = usable_object_frames / max(1, int(object_valid.sum()))
    object_log_ratios = np.log(np.maximum(np.asarray(object_ratios), 1e-9))
    median_ratio = float(np.median(object_ratios)) if object_ratios else -1.0
    checks = {
        "timeline_equal": timeline_equal,
        "primary_valid_pixel_ratio": bool(
            primary_valid_ratio >= thresholds["min_valid_pixel_ratio"]
        ),
        "secondary_valid_pixel_ratio": bool(
            secondary_valid_ratio >= thresholds["min_valid_pixel_ratio"]
        ),
        "object_usable_frame_ratio": bool(
            object_usable_ratio >= thresholds["min_object_usable_frame_ratio"]
        ),
        "median_scale_ratio_in_range": bool(
            thresholds["secondary_to_primary_ratio_min"]
            <= median_ratio
            <= thresholds["secondary_to_primary_ratio_max"]
        ),
        "object_scale_ratio_temporally_stable": bool(
            _robust_mad(object_log_ratios)
            <= thresholds["max_object_log_ratio_robust_mad"]
        ),
        "background_scale_ratio_temporally_stable": bool(
            _robust_mad(background_log_ratios)
            <= thresholds["max_background_log_ratio_robust_mad"]
        ),
    }
    return {
        "schema_version": "1.0",
        "policy": "DA3 primary; UniDepth reject-only cross-check; no rescaling or blending",
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "limits": thresholds,
        "primary": {
            "depth_path": str(primary_file),
            "depth_fingerprint_sha256": _artifact_fingerprint(primary_file),
            "metadata_path": str(primary_metadata),
            "metadata_sha256": _sha256(primary_metadata),
            "valid_pixel_ratio": float(primary_valid_ratio),
        },
        "secondary": {
            "depth_path": str(secondary_file),
            "depth_fingerprint_sha256": _artifact_fingerprint(secondary_file),
            "metadata_path": str(secondary_metadata),
            "metadata_sha256": _sha256(secondary_metadata),
            "valid_pixel_ratio": float(secondary_valid_ratio),
        },
        "object_usable_frame_ratio": float(object_usable_ratio),
        "secondary_to_primary_object_depth_ratio": _summary(object_ratios),
        "object_log_ratio_robust_mad": _robust_mad(object_log_ratios),
        "secondary_to_primary_background_depth_ratio": {
            **_summary(np.exp(background_log_ratios)),
            "log_ratio_robust_mad": _robust_mad(background_log_ratios),
        },
        "ground_truth_used": False,
    }


def evaluate_stereo_depth(
    run_dir: str | Path,
    *,
    primary_path: str | Path | None = None,
    limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Gate raw calibrated-stereo metric depth without a learned scale cross-check."""
    import zarr

    root = Path(run_dir).resolve()
    primary_file = Path(primary_path or root / "depth/metric_depth.zarr").resolve()
    primary_metadata = primary_file.parent / "metadata.json"
    stereo_path = root / "calibration/stereo.json"
    frame_index_path = root / "frames/frame_index.json"
    intrinsics_path = root / "calibration/intrinsics.npy"
    right_intrinsics_path = root / "calibration/intrinsics_right.npy"
    common_valid_path = root / "calibration/stereo_common_valid.npy"
    object_masks_path = root / "segmentation/object_masks.npz"
    for required in (
        primary_file,
        primary_metadata,
        stereo_path,
        frame_index_path,
        intrinsics_path,
        right_intrinsics_path,
        common_valid_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    thresholds = {**DEFAULT_STEREO_LIMITS, **(limits or {})}
    metadata = json.loads(primary_metadata.read_text(encoding="utf-8"))
    stereo = json.loads(stereo_path.read_text(encoding="utf-8"))
    rows = json.loads(frame_index_path.read_text(encoding="utf-8"))["frames"]
    expected_indices = np.asarray(
        [row["source_frame_index"] for row in rows], dtype=np.int64
    )
    expected_shape = (
        len(rows),
        int(json.loads((root / "input/source.json").read_text())["video"]["height"]),
        int(json.loads((root / "input/source.json").read_text())["video"]["width"]),
    )
    group = zarr.open_group(str(primary_file), mode="r")
    if "depth_m" not in group or "valid" not in group or "frame_indices" not in group:
        raise ValueError("stereo metric depth requires depth_m, valid, and frame_indices")
    if tuple(group["depth_m"].shape) != expected_shape or tuple(group["valid"].shape) != expected_shape:
        raise ValueError(
            f"stereo depth shape mismatch: {group['depth_m'].shape}, expected {expected_shape}"
        )
    timeline_equal = bool(
        np.array_equal(np.asarray(group["frame_indices"], dtype=np.int64), expected_indices)
    )
    K = np.load(intrinsics_path).astype(np.float64)
    K_right = np.load(right_intrinsics_path).astype(np.float64)
    common_valid = np.asarray(np.load(common_valid_path), dtype=bool)
    metadata_K = np.asarray(metadata.get("K_rect", np.full((3, 3), np.nan)), dtype=np.float64)
    stereo_K = np.asarray(stereo.get("K_rect_left", np.full((3, 3), np.nan)), dtype=np.float64)
    stereo_K_right = np.asarray(
        stereo.get("K_rect_right", np.full((3, 3), np.nan)), dtype=np.float64
    )
    K_equal = bool(
        metadata_K.shape == (3, 3)
        and stereo_K.shape == (3, 3)
        and np.allclose(metadata_K, K, atol=1e-5, rtol=1e-6)
        and np.allclose(stereo_K, K, atol=1e-5, rtol=1e-6)
    )
    shared_rectified_K = bool(
        K_right.shape == (3, 3)
        and stereo_K_right.shape == (3, 3)
        and np.allclose(stereo_K_right, K_right, atol=1e-5, rtol=1e-6)
        and np.allclose(K, K_right, atol=1e-5, rtol=1e-6)
    )
    common_valid_metadata_equal = bool(
        common_valid.shape == expected_shape[1:]
        and metadata.get("common_valid_mask") == "calibration/stereo_common_valid.npy"
        and np.isclose(
            float(common_valid.mean()),
            float(metadata.get("common_valid_pixel_ratio", float("nan"))),
            atol=1e-9,
        )
    )
    metadata_baseline = float(metadata.get("baseline_m", float("nan")))
    stereo_baseline = float(stereo.get("baseline_m", float("nan")))
    baseline_equal = bool(
        np.isfinite(metadata_baseline)
        and np.isclose(metadata_baseline, stereo_baseline, atol=1e-9, rtol=1e-6)
    )

    valid_count = 0
    pixel_count = int(expected_shape[0] * common_valid.sum())
    invalid_label_count = 0
    valid_outside_common_count = 0
    valid_depth_values: list[np.ndarray] = []
    object_masks: np.ndarray | None = None
    object_valid: np.ndarray | None = None
    object_coverage_values: list[float] = []
    usable_object_frames = 0
    valid_object_frame_count = 0
    object_outside_common_count = 0
    object_total_pixel_count = 0
    if object_masks_path.exists():
        with np.load(object_masks_path, allow_pickle=False) as artifact:
            mask_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
            mask_lookup = {int(frame): index for index, frame in enumerate(mask_indices)}
            if not all(int(frame) in mask_lookup for frame in expected_indices):
                raise ValueError("object-mask timeline does not cover the stereo depth timeline")
            selected = np.asarray(
                [mask_lookup[int(frame)] for frame in expected_indices], dtype=np.int64
            )
            object_masks = np.asarray(artifact["masks"], dtype=bool)[selected]
            object_valid = np.asarray(artifact["valid"], dtype=bool)[selected]
            if object_masks.shape != expected_shape or object_valid.shape != (expected_shape[0],):
                raise ValueError("object-mask artifact shape does not match stereo depth")
            valid_masks = object_masks & object_valid[:, None, None]
            object_total_pixel_count = int(valid_masks.sum())
            object_outside_common_count = int(
                np.count_nonzero(valid_masks & ~common_valid[None])
            )
    else:
        raise FileNotFoundError(
            "stereo production depth gate requires automatic object masks; "
            "run segmentation before gating"
        )
    for index in range(expected_shape[0]):
        depth = np.asarray(group["depth_m"][index], dtype=np.float32)
        stored_valid = np.asarray(group["valid"][index], dtype=bool)
        labeled_physical = np.isfinite(depth) & (depth > 0)
        operational_valid = (
            np.isfinite(depth)
            & (depth >= thresholds["min_depth_m"])
            & (depth <= thresholds["max_depth_m"])
        )
        invalid_label_count += int(np.count_nonzero(stored_valid & ~labeled_physical))
        valid_outside_common_count += int(np.count_nonzero(stored_valid & ~common_valid))
        accepted = stored_valid & operational_valid & common_valid
        valid_count += int(accepted.sum())
        if accepted.any():
            valid_depth_values.append(depth[accepted])
        if object_masks is not None and object_valid is not None and object_valid[index]:
            valid_object_frame_count += 1
            mask_count = int(object_masks[index].sum())
            coverage = float((object_masks[index] & accepted).sum() / max(1, mask_count))
            object_coverage_values.append(coverage)
            if coverage >= thresholds["min_object_valid_pixel_ratio"]:
                usable_object_frames += 1
    valid_ratio = valid_count / max(1, pixel_count)
    invalid_label_ratio = invalid_label_count / max(1, pixel_count)
    values = np.concatenate(valid_depth_values) if valid_depth_values else np.empty(0)
    model = str(metadata.get("model", ""))
    object_usable_ratio = (
        usable_object_frames / max(1, valid_object_frame_count)
        if object_masks is not None else None
    )
    object_common_coverage = (
        1.0 - object_outside_common_count / max(1, object_total_pixel_count)
        if object_masks is not None else None
    )
    checks = {
        "rectification_gate_accepted": bool(stereo.get("accepted", False)),
        "foundationstereo_model": "foundationstereo" in model.lower(),
        "raw_metric_without_alignment": bool(
            metadata.get("ground_truth_used", False) is False
            and metadata.get("scale_or_shift_alignment_applied", False) is False
        ),
        "timeline_equal": timeline_equal,
        "calibration_intrinsics_equal": K_equal,
        "shared_left_right_rectified_intrinsics": shared_rectified_K,
        "common_valid_domain_equal": bool(
            common_valid_metadata_equal and valid_outside_common_count == 0
        ),
        "calibration_baseline_equal": baseline_equal,
        "valid_pixel_ratio": bool(valid_ratio >= thresholds["min_valid_pixel_ratio"]),
        "object_valid_coverage": bool(
            object_usable_ratio >= thresholds["min_object_usable_frame_ratio"]
        ),
        "object_within_stereo_common_domain": bool(
            object_common_coverage >= thresholds["min_object_valid_pixel_ratio"]
        ),
        "valid_labels_are_physical": bool(
            invalid_label_ratio <= thresholds["max_invalid_depth_label_ratio"]
        ),
    }
    return {
        "schema_version": "1.0",
        "gate_kind": "calibrated_stereo_native_metric",
        "policy": (
            "FoundationStereo raw fx*baseline/disparity; calibrated stereo and artifact "
            "integrity gate; no GT scale, rescaling, blending, or per-video compensation"
        ),
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "limits": thresholds,
        "primary": {
            "depth_path": str(primary_file),
            "depth_fingerprint_sha256": _artifact_fingerprint(primary_file),
            "metadata_path": str(primary_metadata),
            "metadata_sha256": _sha256(primary_metadata),
            "valid_pixel_ratio": float(valid_ratio),
            "valid_depth_m": _summary(values),
        },
        "calibration": {
            "stereo_path": str(stereo_path),
            "stereo_sha256": _sha256(stereo_path),
            "intrinsics_path": str(intrinsics_path),
            "intrinsics_sha256": _sha256(intrinsics_path),
            "right_intrinsics_path": str(right_intrinsics_path),
            "right_intrinsics_sha256": _sha256(right_intrinsics_path),
            "common_valid_path": str(common_valid_path),
            "common_valid_sha256": _sha256(common_valid_path),
            "baseline_m": stereo_baseline,
        },
        "invalid_depth_label_ratio": float(invalid_label_ratio),
        "valid_pixels_outside_common_domain": int(valid_outside_common_count),
        "object_depth_coverage": {
            "mask_artifact_available": object_masks is not None,
            "mask_artifact_path": str(object_masks_path) if object_masks is not None else None,
            "mask_artifact_sha256": _sha256(object_masks_path) if object_masks is not None else None,
            "valid_object_frame_count": int(valid_object_frame_count),
            "usable_object_frame_count": int(usable_object_frames),
            "usable_object_frame_ratio": object_usable_ratio,
            "per_frame_valid_pixel_ratio": _summary(object_coverage_values),
            "object_pixel_ratio_within_common_domain": object_common_coverage,
            "object_pixels_outside_common_domain": int(object_outside_common_count),
        },
        "ground_truth_used": False,
    }
def write_depth_gate(
    run_dir: str | Path,
    *,
    primary_path: str | Path | None = None,
    secondary_path: str | Path | None = None,
    output_path: str | Path | None = None,
    limits: dict[str, float] | None = None,
) -> Path:
    root = Path(run_dir).resolve()
    destination = Path(output_path or root / "depth/depth_gate.json").resolve()
    metadata_path = Path(primary_path).parent / "metadata.json" if primary_path else root / "depth/metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else {}
    )
    is_foundationstereo = "foundationstereo" in str(metadata.get("model", "")).lower()
    payload = (
        evaluate_stereo_depth(root, primary_path=primary_path, limits=limits)
        if is_foundationstereo
        else evaluate_depth_consistency(
            root, primary_path=primary_path, secondary_path=secondary_path, limits=limits
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def require_depth_gate(run_dir: str | Path) -> dict[str, Any] | None:
    """Require a fresh accepted gate for DA3/stereo, allowing legacy DAv2 runs."""
    root = Path(run_dir).resolve()
    metadata_path = root / "depth/metadata.json"
    gate_path = root / "depth/depth_gate.json"
    model = ""
    if metadata_path.exists():
        model = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("model", ""))
    is_da3 = "da3" in model.lower() or "depth anything 3" in model.lower()
    is_foundationstereo = "foundationstereo" in model.lower()
    if not gate_path.exists():
        if is_da3:
            raise RuntimeError("depth_gate_missing: DA3 depth requires an independent UniDepth check")
        if is_foundationstereo:
            raise RuntimeError(
                "depth_gate_missing: FoundationStereo requires a calibrated-stereo integrity gate"
            )
        return None
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if payload.get("gate_kind") == "calibrated_stereo_native_metric":
        recorded = payload["primary"]
        depth_artifact = Path(recorded["depth_path"])
        if (
            not depth_artifact.exists()
            or _artifact_fingerprint(depth_artifact)
            != recorded.get("depth_fingerprint_sha256")
        ):
            raise RuntimeError("depth_gate_stale: primary depth artifact changed after gating")
        artifact_metadata = Path(recorded["metadata_path"])
        if not artifact_metadata.exists() or _sha256(artifact_metadata) != recorded["metadata_sha256"]:
            raise RuntimeError("depth_gate_stale: primary metadata changed after gating")
        calibration = payload["calibration"]
        for label, path_key, hash_key in (
            ("stereo", "stereo_path", "stereo_sha256"),
            ("intrinsics", "intrinsics_path", "intrinsics_sha256"),
            (
                "right intrinsics", "right_intrinsics_path",
                "right_intrinsics_sha256",
            ),
            ("common valid", "common_valid_path", "common_valid_sha256"),
        ):
            artifact = Path(calibration[path_key])
            if not artifact.exists() or _sha256(artifact) != calibration[hash_key]:
                raise RuntimeError(f"depth_gate_stale: {label} calibration changed after gating")
        object_coverage = payload.get("object_depth_coverage", {})
        if object_coverage.get("mask_artifact_available", False):
            mask_path = Path(object_coverage["mask_artifact_path"])
            if (
                not mask_path.exists()
                or _sha256(mask_path) != object_coverage.get("mask_artifact_sha256")
            ):
                raise RuntimeError("depth_gate_stale: object masks changed after gating")
    else:
        for role in ("primary", "secondary"):
            recorded = payload[role]
            depth_artifact = Path(recorded["depth_path"])
            if (
                not depth_artifact.exists()
                or _artifact_fingerprint(depth_artifact)
                != recorded.get("depth_fingerprint_sha256")
            ):
                raise RuntimeError(
                    f"depth_gate_stale: {role} depth artifact changed after gating"
                )
            artifact_metadata = Path(recorded["metadata_path"])
            if not artifact_metadata.exists() or _sha256(artifact_metadata) != recorded["metadata_sha256"]:
                raise RuntimeError(f"depth_gate_stale: {role} metadata changed after gating")
    if not payload.get("accepted", False):
        failed = [name for name, passed in payload.get("checks", {}).items() if not passed]
        raise RuntimeError(f"depth_gate_rejected: {', '.join(failed)}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--primary-path", type=Path)
    parser.add_argument("--secondary-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = write_depth_gate(
        args.run_dir,
        primary_path=args.primary_path,
        secondary_path=args.secondary_path,
        output_path=args.output_path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(path)
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
