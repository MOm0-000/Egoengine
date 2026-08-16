"""FoundationPose proposal screening, bidirectional tracking, and QC artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import trimesh

from ..manifest import RunManifest, stage_cache_key
from ..schemas import SCHEMA_VERSION, validate_foundationpose_raw
from .depth_gate import require_depth_gate
from .sam3d_objects import _projected_mesh_mask_depth
from .tracking_gate import DEFAULT_LIMITS, write_tracking_gate


def tracking_score(metrics: dict[str, float]) -> float:
    """Fixed V1 score used for proposal screening and final selection."""
    continuity = math.exp(-min(max(metrics["translation_jump_p95_m"], 0.0) / 0.10, 20.0))
    rotation = math.exp(-min(max(metrics["rotation_jump_p95_rad"], 0.0) / 0.75, 20.0))
    depth = math.exp(-min(max(metrics["median_relative_depth_residual"], 0.0), 20.0))
    return float(
        0.30 * metrics["valid_rate"] + 0.30 * metrics["mean_mask_iou"]
        + 0.15 * depth + 0.15 * continuity + 0.10 * rotation
    )


def _rotation_angle(relative: np.ndarray) -> float:
    cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def summarize_tracking(
    transforms: np.ndarray, valid: np.ndarray, mask_iou: np.ndarray,
    relative_depth_residual: np.ndarray, registration_frame: np.ndarray,
) -> dict[str, float | int]:
    valid = np.asarray(valid, dtype=bool)
    valid_pairs = valid[:-1] & valid[1:]
    translation_jumps = np.linalg.norm(np.diff(transforms[:, :3, 3], axis=0), axis=1)[valid_pairs]
    rotation_jumps = np.asarray([
        _rotation_angle(np.linalg.inv(transforms[index]) @ transforms[index + 1])
        for index in range(len(transforms) - 1) if valid_pairs[index]
    ])
    usable_depth = np.isfinite(relative_depth_residual) & valid
    metrics: dict[str, float | int] = {
        "valid_rate": float(valid.mean()) if valid.size else 0.0,
        "mean_mask_iou": float(mask_iou[valid].mean()) if valid.any() else 0.0,
        "median_relative_depth_residual": (
            float(np.median(relative_depth_residual[usable_depth])) if usable_depth.any() else 20.0
        ),
        "translation_jump_p95_m": (
            float(np.percentile(translation_jumps, 95)) if translation_jumps.size else 0.0
        ),
        "rotation_jump_p95_rad": (
            float(np.percentile(rotation_jumps, 95)) if rotation_jumps.size else 0.0
        ),
        "registration_count": int(np.count_nonzero(registration_frame)),
    }
    metrics["tracking_score"] = tracking_score(metrics)  # type: ignore[arg-type]
    return metrics


def _read_frames(run_dir: Path) -> dict[int, Path]:
    payload = json.loads((run_dir / "frames/frame_index.json").read_text(encoding="utf-8"))
    lookup: dict[int, Path] = {}
    for item in payload["frames"]:
        path = run_dir / item["rgb_path"]
        # EgoDex-style runs use local frame_index == source_frame_index. ADT and
        # other sliced clips have arbitrary source frame numbers, while mask and
        # depth artifacts carry source_frame_index. Accept both keys.
        lookup[int(item["frame_index"])] = path
        if "source_frame_index" in item:
            lookup[int(item["source_frame_index"])] = path
    return lookup


def _load_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def pose_quality(
    mesh_m: trimesh.Trimesh, pose: np.ndarray, mask: np.ndarray, depth_m: np.ndarray,
    valid_depth: np.ndarray, K: np.ndarray,
) -> tuple[float, float, float]:
    rendered_mask, rendered_depth = _projected_mesh_mask_depth(mesh_m, K, pose, mask.shape, 1.0)
    union = np.logical_or(rendered_mask, mask).sum()
    iou = float(np.logical_and(rendered_mask, mask).sum() / union) if union else 0.0
    overlap = rendered_mask & mask & valid_depth & np.isfinite(rendered_depth) & np.isfinite(depth_m)
    if not overlap.any():
        return iou, float("inf"), float("inf")
    residual = float(np.median(np.abs(rendered_depth[overlap] - depth_m[overlap])))
    relative = residual / max(float(np.median(depth_m[overlap])), 1e-3)
    return iou, residual, relative


def _resize_observation(
    rgb: np.ndarray, depth_m: np.ndarray, mask: np.ndarray, K: np.ndarray,
    max_input_side: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resize model inputs and scale the camera intrinsics consistently."""
    height, width = depth_m.shape
    if max_input_side is None or max_input_side <= 0 or max(height, width) <= max_input_side:
        return rgb, depth_m, mask, K
    scale = float(max_input_side) / float(max(height, width))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    scale_x = resized_width / float(width)
    scale_y = resized_height / float(height)
    resized_K = np.asarray(K, dtype=np.float64).copy()
    resized_K[0, :] *= scale_x
    resized_K[1, :] *= scale_y
    return (
        cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA),
        cv2.resize(depth_m, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST),
        cv2.resize(
            mask.astype(np.uint8), (resized_width, resized_height), interpolation=cv2.INTER_NEAREST,
        ).astype(bool),
        resized_K,
    )


def _make_estimator(repository: Path, mesh_m: trimesh.Trimesh, shared: tuple[Any, Any, Any] | None = None):
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr

    if shared is None:
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
    else:
        scorer, refiner, glctx = shared
    estimator = FoundationPose(
        model_pts=mesh_m.vertices, model_normals=mesh_m.vertex_normals, mesh=mesh_m,
        scorer=scorer, refiner=refiner, glctx=glctx, debug=0,
        debug_dir=f"/tmp/video-to-spider-foundationpose-{os.getpid()}",
    )
    return estimator, (scorer, refiner, glctx)


def _track_subset(
    repository: Path, mesh_m: trimesh.Trimesh, frame_indices: np.ndarray, timestamps: np.ndarray,
    masks: np.ndarray, mask_valid: np.ndarray, frames: dict[int, Path], depth_group: Any,
    depth_lookup: dict[int, int], K: np.ndarray, anchor_at: int, *, register_iter: int,
    track_iter: int, iou_reregister: float, relative_depth_reregister: float,
    max_input_side: int | None,
    shared: tuple[Any, Any, Any] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[Any, Any, Any]]:
    count = len(frame_indices)
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    valid = np.zeros(count, dtype=bool)
    confidence = np.zeros(count, dtype=np.float32)
    registration = np.zeros(count, dtype=bool)
    depth_residual = np.zeros(count, dtype=np.float32)
    relative_depth = np.zeros(count, dtype=np.float32)
    quality_observed = np.zeros(count, dtype=bool)
    mask_iou = np.zeros(count, dtype=np.float32)
    segment_id = np.full(count, -1, dtype=np.int32)
    estimator, shared = _make_estimator(repository, mesh_m, shared)
    current_segment = -1

    def observe(index: int, force_registration: bool) -> bool:
        nonlocal estimator, current_segment
        if not bool(mask_valid[index]) or not masks[index].any():
            return False
        frame_index = int(frame_indices[index])
        rgb_bgr = cv2.imread(str(frames[frame_index]), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"cannot read frame {frames[frame_index]}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth_at = depth_lookup[frame_index]
        depth = np.asarray(depth_group["depth_m"][depth_at], dtype=np.float32)
        depth_valid_at = np.asarray(depth_group["valid"][depth_at]).astype(bool)
        depth = np.where(depth_valid_at & np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
        model_rgb, model_depth, model_mask, model_K = _resize_observation(
            rgb, depth, masks[index], K, max_input_side,
        )
        if force_registration:
            pose = estimator.register(
                K=model_K, rgb=model_rgb, depth=model_depth, ob_mask=model_mask,
                iteration=register_iter,
            )
            current_segment += 1
            registration[index] = True
        else:
            pose = estimator.track_one(rgb=model_rgb, depth=model_depth, K=model_K, iteration=track_iter)
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(pose).all() or np.linalg.det(pose[:3, :3]) <= 0:
            return False
        iou, residual, relative = pose_quality(mesh_m, pose, masks[index], depth, depth_valid_at, K)
        if not force_registration and (iou < iou_reregister or relative > relative_depth_reregister):
            pose = np.asarray(
                estimator.register(
                    K=model_K, rgb=model_rgb, depth=model_depth, ob_mask=model_mask,
                    iteration=register_iter,
                ),
                dtype=np.float64,
            ).reshape(4, 4)
            current_segment += 1
            registration[index] = True
            iou, residual, relative = pose_quality(mesh_m, pose, masks[index], depth, depth_valid_at, K)
        transforms[index] = pose.astype(np.float32)
        mask_iou[index] = iou
        depth_residual[index] = residual
        relative_depth[index] = relative
        quality_observed[index] = bool(np.isfinite(residual) and np.isfinite(relative))
        valid[index] = bool(np.isfinite(pose).all() and pose[2, 3] > 0 and iou > 0.005)
        confidence[index] = float(np.clip(0.55 * iou + 0.45 * math.exp(-min(relative, 20.0)), 0, 1))
        segment_id[index] = current_segment
        return valid[index]

    observe(anchor_at, True)
    for direction in (range(anchor_at + 1, count), range(anchor_at - 1, -1, -1)):
        estimator, _ = _make_estimator(repository, mesh_m, shared)
        current_segment += 1
        if bool(mask_valid[anchor_at]) and masks[anchor_at].any():
            observe(anchor_at, True)
        need_registration = False
        for index in direction:
            success = observe(index, need_registration)
            need_registration = not success
    arrays = {
        "frame_indices": frame_indices.astype(np.int64), "timestamps_s": timestamps.astype(np.float64),
        "T_camera_object": transforms, "valid": valid, "confidence": confidence,
        "registration_frame": registration, "depth_residual": depth_residual,
        "relative_depth_residual": relative_depth, "quality_observed": quality_observed,
        "mask_iou": mask_iou, "segment_id": segment_id,
    }
    return arrays, shared


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _candidate_passes_fixed_tracking_gate(metrics: dict[str, float]) -> bool:
    """Use the same fixed thresholds as the final full-trajectory gate during proposal screening."""
    return bool(
        metrics["valid_rate"] >= DEFAULT_LIMITS["min_valid_rate"]
        and metrics["mean_mask_iou"] >= DEFAULT_LIMITS["min_mean_mask_iou"]
        and metrics["median_relative_depth_residual"]
        <= DEFAULT_LIMITS["max_median_relative_depth_residual"]
        and metrics["translation_jump_p95_m"]
        <= DEFAULT_LIMITS["max_translation_jump_p95_m"]
        and metrics["rotation_jump_p95_rad"]
        <= DEFAULT_LIMITS["max_rotation_jump_p95_rad"]
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _run_impl(
    run_dir: str | Path, *, foundationpose_root: str | Path,
    ranking_path: str | Path | None = None,
    max_candidates: int = 3, screening_radius: int = 5, register_iter: int = 5,
    track_iter: int = 2, iou_reregister: float = 0.05,
    relative_depth_reregister: float = 0.75, max_input_side: int | None = 960,
    overwrite: bool = False,
) -> Path:
    root = Path(run_dir).resolve()
    require_depth_gate(root)
    repository = Path(foundationpose_root).resolve()
    output_dir = root / "object_tracking"
    output_path = output_dir / "foundationpose_raw.npz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output_path}; pass --overwrite")
    ranking_path = Path(ranking_path).resolve() if ranking_path else root / "mesh_proposals/mesh_ranking.json"
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    candidates = [item for item in ranking["proposals"] if item.get("qualified")][:max_candidates]
    if not candidates:
        raise RuntimeError("object_tracking_failed: no qualified mesh proposal")
    masks_artifact = _load_masks(root / "segmentation/object_masks.npz")
    frame_indices = masks_artifact["frame_indices"]
    timestamps = masks_artifact["timestamps_s"]
    masks = masks_artifact["masks"].astype(bool)
    mask_valid = masks_artifact["valid"].astype(bool)
    frames = _read_frames(root)
    K = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    import zarr

    depth_group = zarr.open(str(root / "depth/metric_depth.zarr"), mode="r")
    depth_lookup = {int(frame): index for index, frame in enumerate(np.asarray(depth_group["frame_indices"]))}
    keyframe_scores = {int(item["frame_index"]): float(item["score"]) for item in ranking["keyframes"]}
    anchor_at = max(
        [index for index in range(len(frame_indices)) if mask_valid[index]],
        key=lambda index: keyframe_scores.get(int(frame_indices[index]), 0.0),
    )
    screening_indices = np.arange(
        max(0, anchor_at - screening_radius), min(len(frame_indices), anchor_at + screening_radius + 1)
    )
    candidate_records = []
    shared = None
    started = time.monotonic()
    for candidate in candidates:
        candidate_id = candidate["proposal_id"]
        mesh_path = root / "mesh_proposals" / candidate["visual_mesh"]
        mesh_m = trimesh.load_mesh(mesh_path, process=False)
        mesh_m.apply_scale(float(candidate["selected_scale_m"]))
        arrays, shared = _track_subset(
            repository, mesh_m, frame_indices[screening_indices], timestamps[screening_indices],
            masks[screening_indices], mask_valid[screening_indices], frames, depth_group, depth_lookup,
            K, int(np.where(screening_indices == anchor_at)[0][0]), register_iter=register_iter,
            track_iter=track_iter, iou_reregister=iou_reregister,
            relative_depth_reregister=relative_depth_reregister,
            max_input_side=max_input_side, shared=shared,
        )
        metrics = summarize_tracking(
            arrays["T_camera_object"], arrays["valid"], arrays["mask_iou"],
            arrays["relative_depth_residual"], arrays["registration_frame"],
        )
        safe_id = output_dir / "foundationpose_candidates" / candidate_id
        _write_json(safe_id / "tracking_metrics.json", {
            "schema_version": SCHEMA_VERSION, "proposal_id": candidate_id,
            "screening_frame_indices": arrays["frame_indices"].tolist(), "metrics": metrics,
        })
        candidate_records.append({"proposal": candidate, "metrics": metrics})
    safe_candidates = [
        item for item in candidate_records
        if _candidate_passes_fixed_tracking_gate(item["metrics"])
    ]
    pool = safe_candidates or candidate_records
    selected = max(pool, key=lambda item: float(item["metrics"]["tracking_score"]))
    if not safe_candidates:
        raise RuntimeError("object_tracking_failed: all proposals missed minimum tracking safety thresholds")
    selected_proposal = selected["proposal"]
    selected_mesh_path = root / "mesh_proposals" / selected_proposal["visual_mesh"]
    selected_mesh_m = trimesh.load_mesh(selected_mesh_path, process=False)
    selected_mesh_m.apply_scale(float(selected_proposal["selected_scale_m"]))
    arrays, shared = _track_subset(
        repository, selected_mesh_m, frame_indices, timestamps, masks, mask_valid, frames,
        depth_group, depth_lookup, K, anchor_at, register_iter=register_iter, track_iter=track_iter,
        iou_reregister=iou_reregister, relative_depth_reregister=relative_depth_reregister,
        max_input_side=max_input_side, shared=shared,
    )
    validate_foundationpose_raw(arrays)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    final_metrics = summarize_tracking(
        arrays["T_camera_object"], arrays["valid"], arrays["mask_iou"],
        arrays["relative_depth_residual"], arrays["registration_frame"],
    )
    selected_path = output_dir / "selected_mesh.json"
    _write_json(selected_path, {
        "schema_version": SCHEMA_VERSION, "proposal_id": selected_proposal["proposal_id"],
        "mesh_ranking": (
            str(ranking_path.relative_to(root))
            if ranking_path.is_relative_to(root) else str(ranking_path)
        ),
        "canonical_visual_mesh": str(selected_mesh_path.relative_to(root)),
        "scale_to_m": float(selected_proposal["selected_scale_m"]),
        "canonical": selected_proposal["canonical"],
        "static_rank": selected_proposal["rank"], "static_score": selected_proposal["static_score"],
        "screening_metrics": selected["metrics"], "selection_policy": "highest fixed tracking_score passing safety thresholds",
        "all_candidates": [
            {"proposal_id": item["proposal"]["proposal_id"], "metrics": item["metrics"],
             "accepted": item is selected, "rejection_reason": None if item is selected else "lower_tracking_score"}
            for item in candidate_records
        ],
    })
    tracking_metrics_path = output_dir / "tracking_metrics.json"
    _write_json(tracking_metrics_path, {
        "schema_version": SCHEMA_VERSION, "anchor_frame_index": int(frame_indices[anchor_at]),
        "metrics": final_metrics, "runtime_s": float(time.monotonic() - started),
        "iou_reregister": iou_reregister, "relative_depth_reregister": relative_depth_reregister,
        "max_input_side": max_input_side,
    })
    tracking_gate_path = write_tracking_gate(root, metrics_path=tracking_metrics_path)
    tracking_gate = json.loads(tracking_gate_path.read_text(encoding="utf-8"))
    from ..visualization import render_foundationpose, render_mesh_proposals

    visualization_outputs: list[str] = []
    visualization_warnings: list[str] = []
    for renderer in (render_mesh_proposals, render_foundationpose):
        try:
            visualization_path = renderer(root, overwrite=True)
            visualization_outputs.append(str(visualization_path.relative_to(root)))
        except Exception as error:
            visualization_warnings.append(
                f"{renderer.__name__} failed without invalidating tracking artifacts: "
                f"{type(error).__name__}: {error}"
            )
    if not tracking_gate["accepted"]:
        failed = [
            name for name, passed in tracking_gate.get("checks", {}).items() if not passed
        ]
        raise RuntimeError(f"object_tracking_gate_rejected: {', '.join(failed)}")
    manifest = RunManifest.load(root / "manifest.json")
    depth_metadata = json.loads(
        (root / "depth/metadata.json").read_text(encoding="utf-8")
    )
    warnings = [
        f"{depth_metadata.get('model', 'metric depth')} scale is retained as the raw "
        "FoundationPose observation; any later WP7 adjustment is explicitly audited"
    ]
    if bool(selected_proposal.get("canonical", {}).get("debug_only")):
        warnings.append("Selected mesh is marked debug_only and must not be promoted as a real WP5 result")
    warnings.extend(visualization_warnings)
    manifest.finish_stage(
        "foundationpose", success=True,
        outputs=[
            str(output_path.relative_to(root)), str(selected_path.relative_to(root)),
            str(tracking_metrics_path.relative_to(root)),
            str(tracking_gate_path.relative_to(root)),
            *visualization_outputs,
        ],
        quality_metrics=final_metrics,
        warnings=warnings,
    )
    return output_path


def run(
    run_dir: str | Path, *, foundationpose_root: str | Path,
    ranking_path: str | Path | None = None,
    max_candidates: int = 3, screening_radius: int = 5, register_iter: int = 5,
    track_iter: int = 2, iou_reregister: float = 0.05,
    relative_depth_reregister: float = 0.75, max_input_side: int | None = 960,
    overwrite: bool = False,
) -> Path:
    root = Path(run_dir).resolve()
    ranking_path = Path(ranking_path).resolve() if ranking_path else root / "mesh_proposals/mesh_ranking.json"
    inputs = [
        ranking_path, root / "segmentation/object_masks.npz",
        root / "calibration/intrinsics.npy",
    ]
    config = {
        "max_candidates": max_candidates, "screening_radius": screening_radius,
        "register_iter": register_iter, "track_iter": track_iter,
        "iou_reregister": iou_reregister,
        "relative_depth_reregister": relative_depth_reregister,
        "max_input_side": max_input_side,
        "ranking_path": str(ranking_path),
    }
    manifest = RunManifest.load(root / "manifest.json")
    manifest.start_stage(
        "foundationpose", cache_key=stage_cache_key("foundationpose", config, inputs),
        command=sys.argv, environment="v2s-foundationpose",
    )
    try:
        return _run_impl(
            run_dir, foundationpose_root=foundationpose_root,
            ranking_path=ranking_path,
            max_candidates=max_candidates, screening_radius=screening_radius,
            register_iter=register_iter, track_iter=track_iter,
            iou_reregister=iou_reregister,
            relative_depth_reregister=relative_depth_reregister,
            max_input_side=max_input_side, overwrite=overwrite,
        )
    except Exception as exc:
        candidate_metrics = sorted(
            str(path.relative_to(root))
            for path in (root / "object_tracking/foundationpose_candidates").glob("*/tracking_metrics.json")
        )
        diagnostic_outputs = [
            str(path.relative_to(root))
            for path in (
                root / "object_tracking/foundationpose_raw.npz",
                root / "object_tracking/selected_mesh.json",
                root / "object_tracking/tracking_metrics.json",
                root / "object_tracking/tracking_gate.json",
                root / "visualization/05_mesh_proposals.mp4",
                root / "visualization/06_foundationpose.mp4",
            )
            if path.exists()
        ]
        manifest = RunManifest.load(root / "manifest.json")
        manifest.finish_stage(
            "foundationpose", success=False,
            outputs=[*candidate_metrics, *diagnostic_outputs],
            quality_metrics={"error_type": type(exc).__name__, "error": str(exc)},
            warnings=["FoundationPose did not produce a valid complete trajectory"],
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--foundationpose-root", type=Path, required=True)
    parser.add_argument(
        "--ranking-path", type=Path,
        help="Optional refitted mesh ranking; defaults to mesh_proposals/mesh_ranking.json",
    )
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--screening-radius", type=int, default=5)
    parser.add_argument("--register-iter", type=int, default=5)
    parser.add_argument("--track-iter", type=int, default=2)
    parser.add_argument("--iou-reregister", type=float, default=0.05)
    parser.add_argument("--relative-depth-reregister", type=float, default=0.75)
    parser.add_argument(
        "--max-input-side", type=int, default=960,
        help="Resize model inputs and scale K consistently; 0 disables resizing",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(run(
        args.run_dir, foundationpose_root=args.foundationpose_root,
        ranking_path=args.ranking_path,
        max_candidates=args.max_candidates, screening_radius=args.screening_radius,
        register_iter=args.register_iter, track_iter=args.track_iter,
        iou_reregister=args.iou_reregister,
        relative_depth_reregister=args.relative_depth_reregister,
        max_input_side=args.max_input_side, overwrite=args.overwrite,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
