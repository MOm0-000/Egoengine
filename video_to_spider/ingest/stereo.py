"""Ingest already-rectified stereo without modifying any upstream algorithm.

Device-specific fisheye/rolling-shutter calibration stays in the official
camera toolkit.  This integration boundary accepts only synchronized pinhole
pairs, audits their epipolar geometry, and writes the same left-reference run
layout consumed by the existing RGB stages.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..manifest import RunManifest, stage_cache_key
from ..schemas import SCHEMA_VERSION, validate_transforms
from .egodex import keyword_candidates


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RECTIFIED_VERTICAL_P95_MAX_PX = 1.5
RECTIFIED_VERTICAL_MEDIAN_MAX_PX = 1.0
MIN_RECTIFIED_INLIER_MATCHES = 40
MIN_POSITIVE_DISPARITY_RATIO = 0.60


def discover_stereo_pairs(left_dir: str | Path, right_dir: str | Path) -> list[tuple[Path, Path, str]]:
    left_root = Path(left_dir).resolve()
    right_root = Path(right_dir).resolve()
    if not left_root.is_dir() or not right_root.is_dir():
        raise FileNotFoundError("left and right stereo inputs must both be directories")
    left = {
        path.relative_to(left_root).as_posix(): path
        for path in left_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    right = {
        path.relative_to(right_root).as_posix(): path
        for path in right_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not left:
        raise ValueError(f"no supported images below {left_root}")
    if left.keys() != right.keys():
        raise ValueError(
            "left/right relative filenames differ; "
            f"missing_right={sorted(left.keys() - right.keys())[:5]}, "
            f"missing_left={sorted(right.keys() - left.keys())[:5]}"
        )
    return [(left[name], right[name], name) for name in sorted(left)]


def load_intrinsics(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        K = np.asarray(np.load(source), dtype=np.float64)
    elif source.suffix.lower() == ".json":
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("K_rect", "K", "intrinsics"):
                if key in payload:
                    payload = payload[key]
                    break
        K = np.asarray(payload, dtype=np.float64)
    else:
        values = np.fromstring(source.read_text(encoding="utf-8"), sep=" ")
        if values.size < 9:
            raise ValueError(f"intrinsics must contain at least nine numbers: {source}")
        K = values[:9].reshape(3, 3)
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
        or not np.isclose(K[2, 2], 1.0)
    ):
        raise ValueError(f"invalid rectified pinhole K: {K}")
    return K


def _load_vector(path: str | Path | None, count: int, *, integer: bool) -> np.ndarray | None:
    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() == ".npy":
        values = np.load(source)
    else:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
        selected_key: str | None = None
        if isinstance(payload, dict):
            for key in ("frame_indices", "frame_ids", "timestamps_s", "timestamp_sensor_ns"):
                if key in payload:
                    payload = payload[key]
                    selected_key = key
                    break
        values = np.asarray(payload)
    values = np.asarray(values, dtype=np.int64 if integer else np.float64)
    if not integer and source.suffix.lower() != ".npy" and selected_key == "timestamp_sensor_ns":
        values = values * 1e-9
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f"expected vector of length {count}, got {values.shape} from {source}")
    if len(np.unique(values)) != count or (count > 1 and np.any(np.diff(values) <= 0)):
        raise ValueError(f"timeline values must be unique and strictly increasing: {source}")
    return values


def _sample_offsets(count: int, maximum: int = 12) -> np.ndarray:
    return np.unique(np.rint(np.linspace(0, count - 1, min(count, maximum))).astype(np.int64))


def audit_rectified_pairs(
    pairs: list[tuple[Path, Path, str]],
    *,
    K_left: np.ndarray | None = None,
    K_right: np.ndarray | None = None,
    common_valid_mask: np.ndarray | None = None,
    vertical_p95_max_px: float = RECTIFIED_VERTICAL_P95_MAX_PX,
    vertical_median_max_px: float = RECTIFIED_VERTICAL_MEDIAN_MAX_PX,
    min_inlier_matches: int = MIN_RECTIFIED_INLIER_MATCHES,
    min_positive_disparity_ratio: float = MIN_POSITIVE_DISPARITY_RATIO,
) -> dict[str, Any]:
    """Feature-only epipolar audit; no predicted depth or task labels are used."""
    if (K_left is None) != (K_right is None):
        raise ValueError("K_left and K_right must be supplied together")
    if K_left is not None and K_right is not None:
        K_left = np.asarray(K_left, dtype=np.float64)
        K_right = np.asarray(K_right, dtype=np.float64)
        if K_left.shape != (3, 3) or K_right.shape != (3, 3):
            raise ValueError("rectified intrinsics must both be 3x3")
        fx_equivalent = 0.5 * (float(K_left[0, 0]) + float(K_right[0, 0]))
        fy_equivalent = 0.5 * (float(K_left[1, 1]) + float(K_right[1, 1]))
    else:
        fx_equivalent = fy_equivalent = 1.0
    feature_mask: np.ndarray | None = None
    if common_valid_mask is not None:
        feature_mask = np.asarray(common_valid_mask, dtype=bool)
        if feature_mask.ndim != 2 or not feature_mask.any():
            raise ValueError("common_valid_mask must be a non-empty 2-D mask")
        feature_mask = feature_mask.astype(np.uint8) * 255
    detector = cv2.ORB_create(nfeatures=3000, fastThreshold=8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    vertical: list[float] = []
    disparity: list[float] = []
    frame_records: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None
    for offset in _sample_offsets(len(pairs)):
        left_path, right_path, name = pairs[int(offset)]
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f"unreadable or unequal stereo pair at {name}")
        if feature_mask is not None and feature_mask.shape != left.shape:
            raise ValueError(
                f"common-valid mask shape {feature_mask.shape} differs from images {left.shape}"
            )
        if expected_shape is None:
            expected_shape = left.shape
        elif left.shape != expected_shape:
            raise ValueError(f"stereo resolution changes at {name}: {left.shape} vs {expected_shape}")
        key_left, desc_left = detector.detectAndCompute(left, feature_mask)
        key_right, desc_right = detector.detectAndCompute(right, feature_mask)
        accepted: list[cv2.DMatch] = []
        if desc_left is not None and desc_right is not None and len(desc_right) >= 2:
            for match_pair in matcher.knnMatch(desc_left, desc_right, k=2):
                if len(match_pair) == 2 and match_pair[0].distance < 0.75 * match_pair[1].distance:
                    accepted.append(match_pair[0])
        left_points = np.asarray(
            [key_left[item.queryIdx].pt for item in accepted], dtype=np.float64
        ).reshape(-1, 2)
        right_points = np.asarray(
            [key_right[item.trainIdx].pt for item in accepted], dtype=np.float64
        ).reshape(-1, 2)
        if K_left is not None and K_right is not None:
            left_x = (left_points[:, 0] - K_left[0, 2]) / K_left[0, 0]
            right_x = (right_points[:, 0] - K_right[0, 2]) / K_right[0, 0]
            left_y = (left_points[:, 1] - K_left[1, 2]) / K_left[1, 1]
            right_y = (right_points[:, 1] - K_right[1, 2]) / K_right[1, 1]
            dx = (left_x - right_x) * fx_equivalent
            dy = (left_y - right_y) * fy_equivalent
        else:
            dx = left_points[:, 0] - right_points[:, 0]
            dy = left_points[:, 1] - right_points[:, 1]
        if dy.size:
            center = float(np.median(dy))
            mad = float(1.4826 * np.median(np.abs(dy - center)))
            tolerance = max(2.0, 4.0 * mad)
            inlier = np.abs(dy - center) <= tolerance
            vertical.extend(np.abs(dy[inlier]).tolist())
            disparity.extend(dx[inlier].tolist())
            inlier_count = int(inlier.sum())
        else:
            inlier_count = 0
        frame_records.append(
            {
                "pair_offset": int(offset),
                "relative_name": name,
                "ratio_test_matches": len(accepted),
                "vertical_robust_inlier_matches": inlier_count,
            }
        )

    vertical_array = np.asarray(vertical, dtype=np.float64)
    disparity_array = np.asarray(disparity, dtype=np.float64)
    vertical_median = float(np.median(vertical_array)) if vertical_array.size else float("inf")
    vertical_p95 = float(np.percentile(vertical_array, 95)) if vertical_array.size else float("inf")
    positive_ratio = float(np.mean(disparity_array > 0)) if disparity_array.size else 0.0
    median_disparity = float(np.median(disparity_array)) if disparity_array.size else float("nan")
    checks = {
        "sufficient_feature_correspondences": bool(vertical_array.size >= min_inlier_matches),
        "vertical_median_within_limit": bool(vertical_median <= vertical_median_max_px),
        "vertical_p95_within_limit": bool(vertical_p95 <= vertical_p95_max_px),
        "left_right_order_positive_disparity": bool(
            positive_ratio >= min_positive_disparity_ratio and median_disparity > 0
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": (
            "ORB ratio-test matches in per-camera normalized rectified coordinates, "
            "reported in equivalent pixels with robust vertical-disparity outlier rejection"
            if K_left is not None else
            "ORB ratio-test matches with robust vertical-disparity outlier rejection"
        ),
        "uses_depth_prediction": False,
        "sampled_pair_count": len(frame_records),
        "sampled_pairs": frame_records,
        "robust_inlier_match_count": int(vertical_array.size),
        "vertical_disparity_abs_median_px": vertical_median,
        "vertical_disparity_abs_p95_px": vertical_p95,
        "horizontal_disparity_median_px": median_disparity,
        "positive_horizontal_disparity_ratio": positive_ratio,
        "limits": {
            "vertical_disparity_abs_median_px_max": float(vertical_median_max_px),
            "vertical_disparity_abs_p95_px_max": float(vertical_p95_max_px),
            "minimum_robust_inlier_matches": int(min_inlier_matches),
            "minimum_positive_horizontal_disparity_ratio": float(min_positive_disparity_ratio),
        },
        "checks": checks,
        "accepted": bool(all(checks.values())),
        "image_height": int(expected_shape[0]) if expected_shape else 0,
        "image_width": int(expected_shape[1]) if expected_shape else 0,
    }


def ingest_rectified_stereo(
    *,
    left_dir: str | Path,
    right_dir: str | Path,
    intrinsics_path: str | Path,
    right_intrinsics_path: str | Path | None = None,
    common_valid_mask_path: str | Path | None = None,
    full_image_common_valid: bool = False,
    baseline_m: float,
    output_dir: str | Path,
    task: str,
    episode_id: str,
    instruction: str,
    fps: float,
    camera_poses_path: str | Path | None = None,
    timestamps_path: str | Path | None = None,
    frame_indices_path: str | Path | None = None,
    static_camera: bool = False,
    geometric_audit: dict[str, Any] | None = None,
) -> Path:
    pairs = discover_stereo_pairs(left_dir, right_dir)
    if not np.isfinite(baseline_m) or not 0.01 <= baseline_m <= 0.30:
        raise ValueError("baseline_m must be finite and within [0.01, 0.30] m")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if camera_poses_path is None and not static_camera:
        raise ValueError(
            "moving egocentric stereo requires --camera-poses; use --static-camera only for a rigid camera"
        )
    if camera_poses_path is not None and static_camera:
        raise ValueError("camera_poses_path and static_camera are mutually exclusive")
    if (common_valid_mask_path is None) == (not full_image_common_valid):
        raise ValueError(
            "provide exactly one of common_valid_mask_path or full_image_common_valid"
        )
    K = load_intrinsics(intrinsics_path)
    K_right = (
        load_intrinsics(right_intrinsics_path)
        if right_intrinsics_path is not None
        else K.copy()
    )
    first_image = cv2.imread(str(pairs[0][0]), cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise RuntimeError(f"cannot read first rectified image: {pairs[0][0]}")
    if common_valid_mask_path is not None:
        mask_source = Path(common_valid_mask_path).resolve()
        if mask_source.suffix.lower() == ".npy":
            common_valid = np.asarray(np.load(mask_source), dtype=bool)
        else:
            loaded_mask = cv2.imread(str(mask_source), cv2.IMREAD_GRAYSCALE)
            if loaded_mask is None:
                raise RuntimeError(f"cannot read common-valid mask: {mask_source}")
            common_valid = loaded_mask > 0
        common_valid_source = str(mask_source)
    else:
        common_valid = np.ones(first_image.shape, dtype=bool)
        common_valid_source = "explicit_full_image_common_valid_declaration"
    if common_valid.shape != first_image.shape or not common_valid.any():
        raise ValueError(
            f"common-valid mask must be non-empty and match {first_image.shape}, "
            f"got {common_valid.shape}"
        )
    feature_audit = audit_rectified_pairs(
        pairs, K_left=K, K_right=K_right, common_valid_mask=common_valid,
    )
    if geometric_audit is not None:
        required = {"accepted", "checks", "schema_version"}
        missing = sorted(required.difference(geometric_audit.keys()))
        if missing:
            raise ValueError(f"geometric_audit is missing required fields: {', '.join(missing)}")
        if not geometric_audit["accepted"]:
            failed = [name for name, passed in geometric_audit["checks"].items() if not passed]
            raise RuntimeError(f"stereo_geometry_rejected: {', '.join(failed)}")
        audit = dict(geometric_audit)
        audit["feature_audit"] = feature_audit
    else:
        audit = feature_audit
    if not audit["accepted"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError(f"stereo_rectification_rejected: {', '.join(failed)}")

    count = len(pairs)
    source_indices = _load_vector(frame_indices_path, count, integer=True)
    if source_indices is None:
        source_indices = np.arange(count, dtype=np.int64)
    source_timestamps = _load_vector(timestamps_path, count, integer=False)
    if source_timestamps is None:
        source_timestamps = source_indices.astype(np.float64) / float(fps)
    relative_timestamps = source_timestamps - source_timestamps[0]
    if count > 1 and np.any(np.diff(relative_timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if camera_poses_path is not None:
        T_world_camera = np.asarray(np.load(camera_poses_path), dtype=np.float64)
        if T_world_camera.shape != (count, 4, 4):
            raise ValueError(f"camera poses must be {(count, 4, 4)}, got {T_world_camera.shape}")
        pose_source = str(Path(camera_poses_path).resolve())
    else:
        T_world_camera = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
        pose_source = "explicit_static_camera_identity"
    validate_transforms("T_world_camera", T_world_camera)

    run_dir = Path(output_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    for relative in ("input", "frames/rgb", "frames/right", "calibration"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    config = {
        "source_type": "calibrated_rectified_stereo",
        "baseline_m": float(baseline_m),
        "fps": float(fps),
        "frame_count": count,
        "static_camera": bool(static_camera),
        "common_valid_domain": (
            "explicit_mask" if common_valid_mask_path is not None else "declared_full_image"
        ),
        "common_valid_pixel_ratio": float(common_valid.mean()),
    }
    manifest = RunManifest.create(
        run_dir / "manifest.json",
        run_id=run_dir.name,
        source_episode=f"stereo:{task}/{episode_id}",
        config=config,
        frame_count=count,
        fps=float(fps),
    )
    inputs: list[Path] = [Path(intrinsics_path), *[item for pair in pairs for item in pair[:2]]]
    for optional in (
        right_intrinsics_path, common_valid_mask_path, camera_poses_path,
        timestamps_path, frame_indices_path,
    ):
        if optional is not None:
            inputs.append(Path(optional))
    cache_key = stage_cache_key("ingest_rectified_stereo", config, inputs)
    manifest.start_stage(
        "ingest",
        cache_key=cache_key,
        command=["ingest-stereo", str(Path(left_dir).resolve()), str(Path(right_dir).resolve())],
        environment="v2s-core",
    )

    rows: list[dict[str, Any]] = []
    for offset, (left, right, name) in enumerate(pairs):
        suffix = left.suffix.lower()
        left_relative = Path("frames/rgb") / f"{offset:06d}{suffix}"
        right_relative = Path("frames/right") / f"{offset:06d}{right.suffix.lower()}"
        shutil.copy2(left, run_dir / left_relative)
        shutil.copy2(right, run_dir / right_relative)
        rows.append(
            {
                "frame_index": offset,
                "source_frame_index": int(source_indices[offset]),
                "timestamp_s": float(relative_timestamps[offset]),
                "source_timestamp_s": float(source_timestamps[offset]),
                "rgb_path": left_relative.as_posix(),
                "right_rgb_path": right_relative.as_posix(),
                "source_relative_name": name,
            }
        )
    height, width = first_image.shape[:2]
    source = {
        "schema_version": SCHEMA_VERSION,
        "source_type": "calibrated_rectified_stereo",
        "task_directory": task,
        "episode_id": str(episode_id),
        "instruction_text": instruction,
        "instruction_source": "explicit_user_or_dataset_text",
        "object_keyword_candidates": keyword_candidates(instruction, task),
        "video": {
            "frame_count": count,
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
        },
        "selected_frame_interval": [0, count],
        "reference_camera": "rectified_left",
        "right_camera": "rectified_right",
        "camera_pose_source": pose_source,
        "depth_route_policy": "FoundationStereo when stereo gate passes; DA3 explicit fallback",
    }
    (run_dir / "input/source.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "frames/frame_index.json").write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "fps": float(fps), "frames": rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(run_dir / "calibration/intrinsics.npy", K.astype(np.float32))
    np.save(run_dir / "calibration/intrinsics_right.npy", K_right.astype(np.float32))
    np.save(run_dir / "calibration/T_world_camera.npy", T_world_camera.astype(np.float32))
    T_left_right = np.eye(4, dtype=np.float32)
    T_left_right[0, 3] = float(baseline_m)
    np.save(run_dir / "calibration/T_left_right.npy", T_left_right)
    np.save(run_dir / "calibration/stereo_common_valid.npy", common_valid)
    stereo = {
        "schema_version": SCHEMA_VERSION,
        "source_type": "calibrated_rectified_stereo",
        "reference_camera": "left",
        "left_right_order": "physical left then physical right; positive disparity expected",
        "K_rect_left": K.tolist(),
        "K_rect_right": K_right.tolist(),
        "baseline_m": float(baseline_m),
        "T_left_right_semantics": (
            "column-vector transform from rectified right camera coordinates to "
            "rectified left camera coordinates; +baseline on left-camera X"
        ),
        "rectification_audit": audit,
        "common_valid_mask_source": common_valid_source,
        "common_valid_pixel_ratio": float(common_valid.mean()),
        "accepted": bool(audit["accepted"]),
        "depth_scale_or_shift_fit_allowed": False,
        "device_specific_rectification": "performed upstream with an official camera model",
    }
    (run_dir / "calibration/stereo.json").write_text(
        json.dumps(stereo, indent=2) + "\n", encoding="utf-8"
    )
    manifest.data["allowed_inputs"] = [
        "rectified_left_rgb",
        "rectified_right_rgb",
        "rectified_intrinsics",
        "stereo_baseline",
        "rectified_common_valid_domain",
        "camera_extrinsics",
        "sensor_timestamps",
        "instruction_text",
        "object_keyword_candidates",
    ]
    manifest.finish_stage(
        "ingest",
        success=True,
        outputs=[
            "input/source.json",
            "frames/frame_index.json",
            "calibration/intrinsics.npy",
            "calibration/intrinsics_right.npy",
            "calibration/T_world_camera.npy",
            "calibration/T_left_right.npy",
            "calibration/stereo_common_valid.npy",
            "calibration/stereo.json",
        ],
        quality_metrics={
            "decoded_stereo_pairs": count,
            "rectification_accepted": bool(audit["accepted"]),
            "vertical_disparity_abs_p95_px": audit["vertical_disparity_abs_p95_px"],
            "positive_horizontal_disparity_ratio": audit["positive_horizontal_disparity_ratio"],
            "baseline_m": float(baseline_m),
        },
    )
    return run_dir / "manifest.json"
