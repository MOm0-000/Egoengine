"""O1 -- masked multi-frame metric object reconstruction.

This is the strong geometric baseline in the upstream ablation plan: it keeps
the SAM3 object mask fixed, consumes calibrated RGB pinhole / rectified stereo
depth and per-frame camera extrinsics, and fuses object-visible metric depth
into a single object point cloud and mesh.  Metric scale is never fitted or
rewritten; the resulting mesh is directly in the reference camera frame with
``selected_scale_m = 1.0``.

The generated proposal follows the same ``mesh_proposals`` layout consumed by
``foundationpose.py`` so downstream tracking can be evaluated without changing
any third-party pose tracker.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from skimage.filters import threshold_otsu

from ..manifest import RunManifest, stage_cache_key
from ..schemas import SCHEMA_VERSION
from .depth_gate import require_depth_gate
from .sam3d_objects import _load_masks, _projected_mesh_mask_depth, mesh_integrity


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_finite_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, tuple):
        return [_finite_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_frame_entries(run_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((run_dir / "frames/frame_index.json").read_text(encoding="utf-8"))
    return list(payload["frames"])


def _source_to_local(frame_entries: list[dict[str, Any]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for position, item in enumerate(frame_entries):
        mapping[int(item["source_frame_index"])] = position
    return mapping


def _open_depth(run_dir: Path):
    import zarr

    group = zarr.open(str(run_dir / "depth/metric_depth.zarr"), mode="r")
    lookup = {int(frame): index for index, frame in enumerate(np.asarray(group["frame_indices"]))}
    return group, lookup


def _extract_foreground_points(
    mask: np.ndarray, depth_m: np.ndarray, valid: np.ndarray, K: np.ndarray, *,
    erode_radius: int = 2, min_points: int = 64, max_points: int = 20000,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Extract object-foreground metric points for one frame.

    Boundary pixels are first removed with a light morphological erosion.  The
    remaining valid depth values are split with Otsu's threshold and only the
    nearer cluster is retained.  This is a generic foreground-vs-background
    separation step: a hand-held rigid object is expected to occupy a compact
    depth range, while leaked/transparent mask pixels usually lie much farther.
    It does not use object pose or ground truth and is not tuned per sample.
    """
    if erode_radius > 0:
        kernel = np.ones((2 * erode_radius + 1, 2 * erode_radius + 1), dtype=np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    usable = mask.astype(bool) & valid.astype(bool) & np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.where(usable)
    if xs.size < min_points:
        return np.empty((0, 3), dtype=np.float64), {"point_count": int(xs.size)}
    values = depth_m[usable].astype(np.float64)
    try:
        threshold = float(threshold_otsu(values))
    except (ValueError, TypeError):
        threshold = float(np.percentile(values, 80))
    near = usable & (depth_m <= threshold)
    ys, xs = np.where(near)
    if xs.size < min_points:
        return np.empty((0, 3), dtype=np.float64), {
            "point_count": int(xs.size), "depth_otsu_m": threshold,
        }
    stride = max(1, int(math.ceil(xs.size / max_points)))
    ys = ys[::stride].astype(np.float64)
    xs = xs[::stride].astype(np.float64)
    z = depth_m[ys.astype(np.int64), xs.astype(np.int64)].astype(np.float64)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=1)
    p05, p95 = np.percentile(points, [5, 95], axis=0)
    return points, {
        "point_count": int(points.shape[0]), "depth_otsu_m": threshold,
        "depth_median_m": float(np.median(z)), "depth_p05_m": float(p05[2]),
        "depth_p95_m": float(p95[2]),
    }


def _estimate_voxel_size(points: np.ndarray, target_points: int = 4500) -> float:
    """Pick a voxel size from local point spacing without assuming object size."""
    if len(points) < 32:
        return 0.005
    sample = points if len(points) <= 2500 else points[np.random.default_rng(0).choice(len(points), 2500, replace=False)]
    tree = cKDTree(sample)
    distances, _ = tree.query(sample, k=min(8, len(sample)))
    if distances.ndim == 1:
        distances = distances[:, None]
    spacing = float(np.median(distances[:, 1:]) if distances.shape[1] > 1 else np.median(distances))
    spacing = float(np.clip(spacing, 1e-4, 0.05))
    # Add a little slack so noisy multi-frame points are still regularized.
    return max(spacing * 1.4, 0.002)


def _voxel_downsample(points: np.ndarray, voxel_size: float, max_points: int = 5000) -> np.ndarray:
    quantized = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(quantized, axis=0, return_index=True)
    points = points[indices]
    if points.shape[0] > max_points:
        points = points[np.random.default_rng(0).choice(points.shape[0], max_points, replace=False)]
    return points


def _convex_hull_mesh(points: np.ndarray) -> trimesh.Trimesh:
    hull = ConvexHull(points)
    return trimesh.Trimesh(vertices=hull.points, faces=hull.simplices, process=False)


def _tetra_circumradius(vertices: np.ndarray, simplex: np.ndarray) -> float:
    pts = vertices[simplex]
    a = pts[1] - pts[0]
    b = pts[2] - pts[0]
    c = pts[3] - pts[0]
    matrix = np.stack([a, b, c], axis=0)
    rhs = np.asarray([np.dot(a, a), np.dot(b, b), np.dot(c, c)], dtype=np.float64) / 2.0
    try:
        bary = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return float("inf")
    center = pts[0] + bary
    return float(np.linalg.norm(center - pts[0]))


def _alpha_shape_mesh(points: np.ndarray, alpha: float) -> trimesh.Trimesh:
    tri = Delaunay(points)
    radii = np.asarray([_tetra_circumradius(points, simplex) for simplex in tri.simplices])
    kept = radii <= alpha
    if kept.sum() < 4:
        raise RuntimeError("alpha shape produced no interior tetrahedra")
    face_counts: dict[tuple[int, int, int], int] = {}
    for simplex in tri.simplices[kept]:
        for face in (
            (simplex[0], simplex[1], simplex[2]), (simplex[0], simplex[1], simplex[3]),
            (simplex[0], simplex[2], simplex[3]), (simplex[1], simplex[2], simplex[3]),
        ):
            key = tuple(sorted(face))
            face_counts[key] = face_counts.get(key, 0) + 1
    faces = [face for face, count in face_counts.items() if count == 1]
    if len(faces) < 4:
        raise RuntimeError("alpha shape produced too few boundary faces")
    return trimesh.Trimesh(vertices=points, faces=np.asarray(faces, dtype=np.int64), process=False)


def _repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.unique_faces())
    if np.any(~mesh.nondegenerate_faces()):
        mesh.update_faces(mesh.faces[mesh.nondegenerate_faces()])
    mesh.remove_infinite_values()
    trimesh.repair.fix_normals(mesh, multibody=False)
    return mesh


def _projected_static_metrics(
    mesh: trimesh.Trimesh, mask: np.ndarray, depth_m: np.ndarray, valid_depth: np.ndarray, K: np.ndarray,
) -> dict[str, Any]:
    """Static fit against the reference frame; mesh is already metric in that camera."""
    rendered_mask, rendered_depth = _projected_mesh_mask_depth(
        mesh, K, np.eye(4, dtype=np.float64), mask.shape, 1.0
    )
    union = np.logical_or(rendered_mask, mask).sum()
    iou = float(np.logical_and(rendered_mask, mask).sum() / union) if union else 0.0
    overlap = rendered_mask & mask & valid_depth & np.isfinite(rendered_depth) & np.isfinite(depth_m)
    if overlap.any():
        residual = float(np.median(np.abs(rendered_depth[overlap] - depth_m[overlap])))
        relative = residual / max(float(np.median(depth_m[overlap])), 1e-3)
    else:
        residual, relative = float("inf"), float("inf")
    depth_score = float(math.exp(-min(relative / 0.10, 20.0))) if math.isfinite(relative) else 0.0
    return {
        "silhouette_iou": iou, "depth_residual_m": residual,
        "relative_depth_residual": relative, "depth_score": depth_score,
        "static_score": float(0.75 * iou + 0.25 * depth_score),
    }


def _build_mesh_proposal(
    points: np.ndarray, *, mesh_method: str, alpha: float, proposal_id: str,
    output_dir: Path, reference_frame_index: int,
) -> tuple[Path, dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    voxel_size = _estimate_voxel_size(points)
    points = _voxel_downsample(points, voxel_size)
    if mesh_method == "convex":
        mesh = _convex_hull_mesh(points)
    elif mesh_method == "alpha":
        alpha_value = alpha if alpha > 0 else 2.5 * voxel_size
        mesh = _alpha_shape_mesh(points, alpha_value)
    else:
        raise ValueError(f"unknown mesh method: {mesh_method}")
    mesh = _repair_mesh(mesh)
    proposal_dir = output_dir / proposal_id
    proposal_dir.mkdir(parents=True, exist_ok=True)
    visual_path = proposal_dir / "visual.obj"
    collision_path = proposal_dir / "collision_source.obj"
    mesh.export(visual_path)
    mesh.export(collision_path)
    point_cloud = trimesh.PointCloud(points)
    point_cloud_path = proposal_dir / "fused_points.ply"
    point_cloud.export(point_cloud_path)
    integrity = mesh_integrity(mesh)
    return visual_path, {
        "proposal_id": proposal_id, "reference_frame_index": reference_frame_index,
        "mesh_method": mesh_method, "voxel_size_m": voxel_size,
        "alpha_m": (alpha if alpha > 0 else 2.5 * voxel_size),
        "visual_mesh": str(visual_path.relative_to(output_dir)),
        "collision_source_mesh": str(collision_path.relative_to(output_dir)),
        "point_cloud": str(point_cloud_path.relative_to(output_dir)),
        "selected_scale_m": 1.0, "metric_scale_rewritten": False,
        "integrity": integrity,
        "canonical": {
            "origin": "reference camera frame", "scale": 1.0,
            "debug_only": False, "T_raw_canonical": np.eye(4).tolist(),
        },
    }


def _run_impl(
    run_dir: str | Path, *, mesh_method: str = "alpha", alpha: float = 0.0,
    erode_radius: int = 2, overwrite: bool = False,
) -> Path:
    root = Path(run_dir).resolve()
    require_depth_gate(root)
    ranking_path = root / "mesh_proposals/omvg_mesh_ranking.json"
    if ranking_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {ranking_path}; pass --overwrite")

    frame_entries = _read_frame_entries(root)
    source_to_local = _source_to_local(frame_entries)
    masks = _load_masks(root / "segmentation/object_masks.npz")
    depth_group, depth_lookup = _open_depth(root)
    K = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    T_world_camera = np.load(root / "calibration/T_world_camera.npy").astype(np.float64)

    frame_indices = masks["frame_indices"]
    mask_arrays = masks["masks"].astype(bool)
    mask_valid = masks["valid"].astype(bool)

    per_frame_records: list[dict[str, Any]] = []
    world_points_by_frame: list[tuple[int, np.ndarray]] = []
    best_count = 0
    reference_index = -1
    reference_world = np.eye(4, dtype=np.float64)

    for mask_index, source_frame in enumerate(frame_indices):
        source_frame = int(source_frame)
        if not bool(mask_valid[mask_index]) or not mask_arrays[mask_index].any():
            per_frame_records.append({
                "frame_index": source_frame, "valid_mask": False, "point_count": 0,
            })
            continue
        local_index = source_to_local.get(source_frame)
        if local_index is None or local_index >= T_world_camera.shape[0]:
            raise RuntimeError(f"no camera extrinsics for frame {source_frame}")
        depth_index = depth_lookup.get(source_frame)
        if depth_index is None:
            per_frame_records.append({
                "frame_index": source_frame, "valid_mask": True, "point_count": 0,
                "error": "missing depth frame",
            })
            continue
        depth_m = np.asarray(depth_group["depth_m"][depth_index], dtype=np.float64)
        depth_valid = np.asarray(depth_group["valid"][depth_index]).astype(bool)
        camera_points, point_metrics = _extract_foreground_points(
            mask_arrays[mask_index], depth_m, depth_valid, K, erode_radius=erode_radius,
        )
        record = {"frame_index": source_frame, "valid_mask": True, **point_metrics}
        if camera_points.shape[0] < 8:
            per_frame_records.append(record)
            continue
        T_world_cam = T_world_camera[local_index]
        ones = np.ones((camera_points.shape[0], 1), dtype=np.float64)
        world_points = (T_world_cam[:3, :3] @ camera_points.T).T + T_world_cam[:3, 3]
        world_points_by_frame.append((source_frame, world_points))
        record["world_point_count"] = int(world_points.shape[0])
        per_frame_records.append(record)
        if world_points.shape[0] > best_count:
            best_count = world_points.shape[0]
            reference_index = local_index
            reference_world = T_world_cam

    if not world_points_by_frame:
        raise RuntimeError("omvg_failed: no frame with usable object depth points")

    T_ref_world = np.linalg.inv(reference_world)
    fused_points: list[np.ndarray] = []
    for _, world_points in world_points_by_frame:
        camera_points_ref = (T_ref_world[:3, :3] @ world_points.T).T + T_ref_world[:3, 3]
        fused_points.append(camera_points_ref)
    fused = np.concatenate(fused_points, axis=0)
    reference_frame_index = int(frame_entries[reference_index]["source_frame_index"])

    proposal_id = "omvg_00001"
    output_dir = root / "mesh_proposals"
    visual_path, record = _build_mesh_proposal(
        fused, mesh_method=mesh_method, alpha=alpha, proposal_id=proposal_id,
        output_dir=output_dir, reference_frame_index=reference_frame_index,
    )
    mesh_m = trimesh.load_mesh(visual_path, process=False)
    reference_depth_index = depth_lookup[reference_frame_index]
    reference_mask = mask_arrays[np.flatnonzero(frame_indices == reference_frame_index)[0]]
    reference_depth = np.asarray(depth_group["depth_m"][reference_depth_index], dtype=np.float64)
    reference_valid = np.asarray(depth_group["valid"][reference_depth_index]).astype(bool)
    static = _projected_static_metrics(mesh_m, reference_mask, reference_depth, reference_valid, K)
    qualified = bool(
        record["integrity"]["qualified"] and static["silhouette_iou"] >= 0.005
        and math.isfinite(static["relative_depth_residual"])
    )
    record.update({"qualified": qualified, "fit": static, "rank": 1})
    record["static_score"] = float(static["static_score"])
    _write_json(output_dir / proposal_id / "metadata.json", record)

    fused_pc = trimesh.PointCloud(fused)
    fused_pc.export(output_dir / "omvg_fused_points.ply")
    metrics = {
        "schema_version": SCHEMA_VERSION, "stage": "object_mvg_mesh_proposals",
        "mesh_method": mesh_method, "alpha_m": record["alpha_m"],
        "voxel_size_m": record["voxel_size_m"], "metric_scale_rewritten": False,
        "reference_frame_index": reference_frame_index,
        "frame_count": len(per_frame_records),
        "valid_mask_frame_count": int(sum(bool(item.get("valid_mask")) for item in per_frame_records)),
        "usable_depth_frame_count": len(world_points_by_frame),
        "fused_point_count": int(fused.shape[0]),
        "fused_centroid_m": np.median(fused, axis=0).tolist(),
        "fused_extent_m": (np.max(fused, axis=0) - np.min(fused, axis=0)).tolist(),
        "per_frame": per_frame_records,
        "static_fit": static, "qualified_count": int(qualified),
        "proposals": [record],
        "success": bool(qualified),
        "failure_reason": None if qualified else "omvg_no_qualified_proposal",
    }
    _write_json(output_dir / "omvg_object_mvg_metrics.json", metrics)
    _write_json(ranking_path, {
        "schema_version": SCHEMA_VERSION, "stage": "object_mvg_mesh_proposals",
        "ranking_policy": (
            "qualified = mesh_integrity and reference-frame silhouette/depth fit; "
            "selected_scale_m is fixed to 1.0 (metric scale is not rewritten)"
        ),
        "foundationpose_used": False,
        "keyframes": [
            {"frame_index": reference_frame_index, "score": float(static["static_score"])}
        ],
        "proposals": [record], "qualified_count": int(qualified),
        "success": bool(qualified),
        "failure_reason": None if qualified else "omvg_no_qualified_proposal",
    })

    manifest = RunManifest.load(root / "manifest.json")
    manifest.finish_stage(
        "object_mvg", success=bool(qualified),
        outputs=[
            str(ranking_path.relative_to(root)),
            str((output_dir / "omvg_object_mvg_metrics.json").relative_to(root)),
            str((output_dir / "omvg_fused_points.ply").relative_to(root)),
            str((output_dir / proposal_id / "visual.obj").relative_to(root)),
        ],
        quality_metrics={
            "qualified_count": int(qualified), "fused_point_count": int(fused.shape[0]),
            "reference_silhouette_iou": static["silhouette_iou"],
            "reference_relative_depth_residual": static["relative_depth_residual"],
            "metric_scale_rewritten": False,
        },
        warnings=[
            "O1 uses SAM3 masks unchanged; mask/object-depth errors are not corrected by geometric fusion",
            "metric scale is retained from calibrated depth and is never fitted",
        ],
    )
    if not qualified:
        raise RuntimeError("omvg_failed: no qualified O1 mesh proposal")
    return ranking_path


def run(
    run_dir: str | Path, *, mesh_method: str = "alpha", alpha: float = 0.0,
    erode_radius: int = 2, overwrite: bool = False,
) -> Path:
    root = Path(run_dir).resolve()
    ranking_path = root / "mesh_proposals/omvg_mesh_ranking.json"
    if ranking_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {ranking_path}; pass --overwrite")
    inputs = [
        root / "segmentation/object_masks.npz",
        root / "calibration/intrinsics.npy",
        root / "calibration/T_world_camera.npy",
    ]
    config = {"mesh_method": mesh_method, "alpha": alpha, "erode_radius": erode_radius}
    manifest = RunManifest.load(root / "manifest.json")
    manifest.start_stage(
        "object_mvg", cache_key=stage_cache_key("object_mvg", config, inputs),
        command=sys.argv, environment="v2s-core",
    )
    try:
        return _run_impl(
            run_dir, mesh_method=mesh_method, alpha=alpha, erode_radius=erode_radius,
            overwrite=overwrite,
        )
    except Exception as exc:
        manifest = RunManifest.load(root / "manifest.json")
        diagnostics = [
            str(path.relative_to(root))
            for path in (
                root / "mesh_proposals/omvg_mesh_ranking.json",
                root / "mesh_proposals/omvg_object_mvg_metrics.json",
                root / "mesh_proposals/omvg_fused_points.ply",
            )
            if path.exists()
        ]
        manifest.finish_stage(
            "object_mvg", success=False, outputs=diagnostics,
            quality_metrics={"error_type": type(exc).__name__, "error": str(exc)},
            warnings=["masked multi-frame metric reconstruction failed"],
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mesh-method", choices=["alpha", "convex"], default="alpha")
    parser.add_argument("--alpha", type=float, default=0.0, help="Explicit alpha in meters; 0 auto-selects")
    parser.add_argument("--erode-radius", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(run(
        args.run_dir, mesh_method=args.mesh_method, alpha=args.alpha,
        erode_radius=args.erode_radius, overwrite=args.overwrite,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
