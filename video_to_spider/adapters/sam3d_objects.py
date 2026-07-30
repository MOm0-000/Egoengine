"""SAM 3D Objects proposal generation and deterministic static ranking.

The model import is delayed until :func:`run` so proposal artifacts can be
inspected and ranked in the lightweight core environment. WP5 never imports or
calls FoundationPose.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import trimesh

from ..manifest import RunManifest, stage_cache_key
from ..schemas import SCHEMA_VERSION


@dataclass(frozen=True)
class KeyframeCandidate:
    frame_index: int
    mask_index: int
    score: float
    area_ratio: float
    boundary_safety: float
    hand_occlusion: float
    sharpness: float
    depth_valid_rate: float


def _read_frame_index(run_dir: Path) -> dict[int, Path]:
    payload = json.loads((run_dir / "frames/frame_index.json").read_text(encoding="utf-8"))
    return {int(item["frame_index"]): run_dir / item["rgb_path"] for item in payload["frames"]}


def _load_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    low, high = float(np.min(values)), float(np.max(values))
    if high - low < 1e-12:
        return np.ones_like(values)
    return (values - low) / (high - low)


def rank_keyframes(
    run_dir: str | Path, *, object_masks_path: str | Path | None = None,
    hand_masks_path: str | Path | None = None, depth_path: str | Path | None = None,
) -> list[KeyframeCandidate]:
    """Rank valid masked frames without consulting object/hand ground truth."""
    root = Path(run_dir)
    objects = _load_masks(Path(object_masks_path or root / "segmentation/object_masks.npz"))
    hands = _load_masks(Path(hand_masks_path or root / "segmentation/hand_masks.npz"))
    if not np.array_equal(objects["frame_indices"], hands["frame_indices"]):
        raise ValueError("object and hand mask timelines differ")
    frame_paths = _read_frame_index(root)
    depth_group = None
    depth_file = Path(depth_path or root / "depth/metric_depth.zarr")
    if depth_file.exists():
        import zarr

        depth_group = zarr.open(str(depth_file), mode="r")
        depth_lookup = {
            int(frame): index for index, frame in enumerate(np.asarray(depth_group["frame_indices"]))
        }
    else:
        depth_lookup = {}

    raw: list[dict[str, float | int]] = []
    for mask_index, frame_index_raw in enumerate(objects["frame_indices"]):
        frame_index = int(frame_index_raw)
        mask = objects["masks"][mask_index].astype(bool)
        hand = hands["masks"][mask_index].astype(bool)
        if not bool(objects["valid"][mask_index]) or not mask.any():
            continue
        image = cv2.imread(str(frame_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot read frame {frame_paths[frame_index]}")
        area = float(mask.mean())
        border = np.zeros_like(mask)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        boundary_safety = float(1.0 - np.logical_and(mask, border).sum() / max(mask.sum(), 1))
        hand_occlusion = float(np.logical_and(mask, hand).sum() / max(mask.sum(), 1))
        ys, xs = np.where(mask)
        y0, y1 = max(0, int(ys.min()) - 8), min(mask.shape[0], int(ys.max()) + 9)
        x0, x1 = max(0, int(xs.min()) - 8), min(mask.shape[1], int(xs.max()) + 9)
        sharpness = float(cv2.Laplacian(image[y0:y1, x0:x1], cv2.CV_64F).var())
        depth_valid_rate = 0.0
        if frame_index in depth_lookup:
            depth_at = depth_lookup[frame_index]
            valid_depth = np.asarray(depth_group["valid"][depth_at]).astype(bool)
            depth_m = np.asarray(depth_group["depth_m"][depth_at])
            depth_valid_rate = float((valid_depth & np.isfinite(depth_m) & (depth_m > 0))[mask].mean())
        raw.append({
            "frame_index": frame_index, "mask_index": mask_index, "area_ratio": area,
            "boundary_safety": boundary_safety, "hand_occlusion": hand_occlusion,
            "sharpness": sharpness, "depth_valid_rate": depth_valid_rate,
        })
    if not raw:
        return []
    sharpness = _normalize01(np.asarray([item["sharpness"] for item in raw]))
    area_score = np.minimum(
        np.asarray([item["area_ratio"] for item in raw], dtype=np.float64) / 0.01, 1.0
    )
    result = []
    for index, item in enumerate(raw):
        score = (
            0.25 * area_score[index] + 0.20 * float(item["boundary_safety"])
            + 0.20 * (1.0 - float(item["hand_occlusion"])) + 0.20 * sharpness[index]
            + 0.15 * float(item["depth_valid_rate"])
        )
        result.append(KeyframeCandidate(score=float(score), **item))
    return sorted(result, key=lambda item: (-item.score, item.frame_index))


def _as_trimesh(mesh_or_scene: Any) -> trimesh.Trimesh:
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene.copy()
    if isinstance(mesh_or_scene, trimesh.Scene):
        geometries = [geometry for geometry in mesh_or_scene.geometry.values() if len(geometry.faces)]
        if not geometries:
            raise ValueError("SAM 3D output scene contains no triangle geometry")
        return trimesh.util.concatenate(geometries)
    raise TypeError(f"unsupported SAM 3D mesh output: {type(mesh_or_scene)!r}")


def repair_and_canonicalize(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Keep the largest component, repair basic defects, center and normalize it."""
    source = mesh.copy()
    source.remove_infinite_values()
    source.remove_unreferenced_vertices()
    components = source.split(only_watertight=False)
    if components:
        source = max(components, key=lambda component: float(component.area))
    source.update_faces(source.nondegenerate_faces())
    source.update_faces(source.unique_faces())
    source.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(source, multibody=False)
    trimesh.repair.fill_holes(source)
    bounds = np.asarray(source.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    longest = float(np.max(extents))
    if not np.isfinite(longest) or longest <= 1e-8:
        raise ValueError("mesh has zero or invalid extent")
    canonical = source.copy()
    canonical.vertices = (np.asarray(canonical.vertices) - center) / longest
    canonical.remove_unreferenced_vertices()
    T_raw_canonical = np.eye(4, dtype=np.float64)
    T_raw_canonical[:3, :3] *= longest
    T_raw_canonical[:3, 3] = center
    return canonical, {
        "T_raw_canonical": T_raw_canonical.tolist(), "raw_center": center.tolist(),
        "raw_longest_extent": longest,
        "canonical_origin": "axis-aligned bounding-box center",
        "canonical_axes": "SAM 3D postprocessed mesh axes",
    }


def mesh_integrity(mesh: trimesh.Trimesh) -> dict[str, Any]:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    finite = bool(np.isfinite(mesh.vertices).all() and np.isfinite(face_areas).all())
    positive_axes = bool(np.all(extents > 1e-6))
    nondegenerate_rate = float(np.mean(face_areas > 1e-12)) if face_areas.size else 0.0
    score = (
        0.35 * float(finite) + 0.25 * float(positive_axes) + 0.20 * nondegenerate_rate
        + 0.10 * float(mesh.is_winding_consistent) + 0.10 * float(mesh.is_watertight)
    )
    return {
        "finite": finite, "vertex_count": int(len(mesh.vertices)), "face_count": int(len(mesh.faces)),
        "component_count": int(len(mesh.split(only_watertight=False))), "extents_canonical": extents.tolist(),
        "nondegenerate_face_rate": nondegenerate_rate, "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent), "score": float(score),
        "qualified": bool(finite and positive_axes and len(mesh.faces) >= 4 and nondegenerate_rate > 0.99),
    }


def _quaternion_wxyz_matrix(quaternion: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(quaternion), dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _projected_mesh_mask_depth(
    mesh: trimesh.Trimesh, K: np.ndarray, T_camera_object: np.ndarray,
    image_shape: tuple[int, int], scale_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale_m)
    camera = vertices @ T_camera_object[:3, :3].T + T_camera_object[:3, 3]
    z = camera[:, 2]
    image_mask = np.zeros(image_shape, dtype=np.uint8)
    image_depth = np.full(image_shape, np.inf, dtype=np.float32)
    if np.count_nonzero(z > 1e-5) < 3:
        return image_mask.astype(bool), image_depth
    uvw = camera @ K.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    height, width = image_shape
    for face in np.asarray(mesh.faces):
        if np.any(z[face] <= 1e-5):
            continue
        polygon = np.rint(uv[face]).astype(np.int32)
        if polygon[:, 0].max() < 0 or polygon[:, 0].min() >= width:
            continue
        if polygon[:, 1].max() < 0 or polygon[:, 1].min() >= height:
            continue
        cv2.fillConvexPoly(image_mask, polygon, 1)
    # A full CPU z-buffer at 1080p is unnecessarily expensive for static WP5
    # ranking. Use the robust visible-vertex depth as the deterministic depth
    # proxy; FoundationPose supplies the learned render/depth score in WP6.
    if image_mask.any():
        image_depth[image_mask > 0] = float(np.median(z[z > 1e-5]))
    return image_mask.astype(bool), image_depth


def estimate_mask_pointcloud(
    mask: np.ndarray, depth_m: np.ndarray, valid: np.ndarray, K: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    usable = mask.astype(bool) & valid.astype(bool) & np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.where(usable)
    if xs.size < 8:
        return np.empty((0, 3), dtype=np.float64), {"point_count": int(xs.size)}
    stride = max(1, int(math.ceil(xs.size / 20000)))
    xs, ys = xs[::stride].astype(np.float64), ys[::stride].astype(np.float64)
    z = depth_m[ys.astype(np.int64), xs.astype(np.int64)].astype(np.float64)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=1)
    p05, p95 = np.percentile(points, [5, 95], axis=0)
    return points, {
        "point_count": int(points.shape[0]), "depth_median_m": float(np.median(z)),
        "extent_x_m": float(p95[0] - p05[0]), "extent_y_m": float(p95[1] - p05[1]),
        "depth_p05_m": float(p05[2]), "depth_p95_m": float(p95[2]),
    }


def static_fit_metrics(
    mesh: trimesh.Trimesh, mask: np.ndarray, depth_m: np.ndarray, valid_depth: np.ndarray,
    K: np.ndarray, *, rotation_wxyz: Iterable[float], translation: Iterable[float],
    layout_scale: Iterable[float],
) -> tuple[dict[str, Any], np.ndarray, float]:
    points, point_metrics = estimate_mask_pointcloud(mask, depth_m, valid_depth, K)
    mesh_extent = max(float(np.max(mesh.extents)), 1e-8)
    if points.shape[0] >= 8:
        p05, p95 = np.percentile(points, [5, 95], axis=0)
        point_extent = max(float(p95[0] - p05[0]), float(p95[1] - p05[1]), 0.005)
        depth_scale_m = point_extent / mesh_extent
        centroid = np.median(points, axis=0)
    else:
        depth_scale_m = 0.05 / mesh_extent
        centroid = np.asarray(translation, dtype=np.float64)
    layout_scale_value = float(np.median(np.abs(np.asarray(layout_scale, dtype=np.float64))))
    if not np.isfinite(layout_scale_value) or layout_scale_value <= 0:
        layout_scale_value = depth_scale_m
    scale_m = float(np.clip(0.75 * layout_scale_value + 0.25 * depth_scale_m, 0.003, 0.5))
    T_camera_object = np.eye(4, dtype=np.float64)
    T_camera_object[:3, :3] = _quaternion_wxyz_matrix(rotation_wxyz)
    predicted_translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if not np.isfinite(predicted_translation).all() or predicted_translation[2] <= 0:
        predicted_translation = centroid
    T_camera_object[:3, 3] = predicted_translation
    rendered_mask, rendered_depth = _projected_mesh_mask_depth(
        mesh, K, T_camera_object, mask.shape, scale_m
    )
    union = np.logical_or(rendered_mask, mask).sum()
    intersection = np.logical_and(rendered_mask, mask).sum()
    iou = float(intersection / union) if union else 0.0
    overlap = rendered_mask & mask & valid_depth & np.isfinite(rendered_depth) & np.isfinite(depth_m)
    if overlap.any():
        absolute_depth_residual = float(np.median(np.abs(rendered_depth[overlap] - depth_m[overlap])))
        relative_depth_residual = absolute_depth_residual / max(float(np.median(depth_m[overlap])), 1e-3)
    else:
        absolute_depth_residual = float("inf")
        relative_depth_residual = float("inf")
    depth_score = float(math.exp(-min(relative_depth_residual, 20.0))) if np.isfinite(relative_depth_residual) else 0.0
    metrics = {
        **point_metrics, "layout_scale_raw": layout_scale_value,
        "depth_initialized_scale_m": float(depth_scale_m), "selected_scale_m": scale_m,
        "silhouette_iou": iou, "silhouette_residual": float(1.0 - iou),
        "depth_residual_m": absolute_depth_residual,
        "relative_depth_residual": relative_depth_residual, "depth_score": depth_score,
        "T_camera_object_initial": T_camera_object.tolist(),
    }
    return metrics, T_camera_object, scale_m


def _tensor_list(value: Any, length: int) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < length:
        raise ValueError(f"model output has {array.size} values, expected {length}")
    return array[:length].tolist()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _move_module(module: Any, device: str) -> None:
    if module is not None:
        module.to(device)


def _release_cuda(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _cached_condition_embedder(
    pipeline: Any, key: str, input_dict: dict[str, Any], input_mapping: list[str],
) -> tuple[Any, Any]:
    """Compute condition tokens on CUDA, then return a lightweight cached callable."""
    import torch

    embedder = pipeline.condition_embedders[key]
    _move_module(embedder, "cuda")
    try:
        condition_args = pipeline.map_input_keys(input_dict, input_mapping)
        condition_kwargs = {name: value for name, value in input_dict.items() if name not in input_mapping}
        embedded, passthrough_args, passthrough_kwargs = pipeline.embed_condition(
            embedder, *condition_args, **condition_kwargs,
        )
    finally:
        _move_module(embedder, "cpu")
        _release_cuda(torch)
    if embedded is None:
        def cached(*args: Any, **kwargs: Any) -> Any:
            return embedder(*passthrough_args, **passthrough_kwargs)
    else:
        def cached(*args: Any, **kwargs: Any) -> Any:
            return embedded
    return embedder, cached


def _load_inference(config_file: Path, *, compile_model: bool, low_vram: bool) -> Any:
    repository = config_file.parents[2]
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "notebook"))
    from inference import (
        BLACKLIST_FILTERS, WHITELIST_FILTERS, Inference, check_hydra_safety,
    )

    if not low_vram:
        return Inference(str(config_file), compile=compile_model)

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_file)
    config.rendering_engine = "pytorch3d"
    config.compile_model = compile_model
    config.workspace_dir = str(config_file.parent)
    config.device = "cpu"
    config.depth_model.device = "cpu"
    check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
    pipeline = instantiate(config)
    pipeline.device = torch.device("cuda")
    return pipeline


def _run_low_vram_pipeline(
    pipeline: Any, image: np.ndarray, *, seed: int, moge_resolution_level: int,
) -> dict[str, Any]:
    """Run the official pipeline with one model family resident on CUDA at a time."""
    import torch

    device = "cuda"
    image = pipeline.merge_image_and_mask(image, None)
    depth_model = pipeline.depth_model
    original_infer = depth_model.model.infer

    def infer_at_bounded_resolution(input_image: Any, **kwargs: Any) -> Any:
        kwargs["resolution_level"] = moge_resolution_level
        return original_infer(input_image, **kwargs)

    depth_model.model.infer = infer_at_bounded_resolution
    _move_module(depth_model.model, device)
    depth_model.device = torch.device(device)
    try:
        pointmap_dict = pipeline.compute_pointmap(image, None)
    finally:
        _move_module(depth_model.model, "cpu")
        depth_model.device = torch.device("cpu")
        depth_model.model.infer = original_infer
        _release_cuda(torch)

    pointmap = pointmap_dict["pointmap"]
    points = type(pipeline)._down_sample_img(pointmap)
    point_colors = type(pipeline)._down_sample_img(pointmap_dict["pts_color"])
    ss_input = pipeline.preprocess_image(image, pipeline.ss_preprocessor, pointmap=pointmap)
    ss_embedder, cached_ss_embedder = _cached_condition_embedder(
        pipeline, "ss_condition_embedder", ss_input, pipeline.ss_condition_input_mapping,
    )
    pipeline.condition_embedders["ss_condition_embedder"] = cached_ss_embedder
    ss_modules = [pipeline.models["ss_generator"], pipeline.models["ss_decoder"]]
    for module in ss_modules:
        _move_module(module, device)
    try:
        torch.manual_seed(seed)
        ss_result = pipeline.sample_sparse_structure(ss_input)
    finally:
        for module in ss_modules:
            _move_module(module, "cpu")
        pipeline.condition_embedders["ss_condition_embedder"] = ss_embedder
        _release_cuda(torch)

    pointmap_scale = ss_input.get("pointmap_scale", None)
    pointmap_shift = ss_input.get("pointmap_shift", None)
    ss_result.update(
        pipeline.pose_decoder(
            ss_result, scene_scale=pointmap_scale, scene_shift=pointmap_shift,
        )
    )
    ss_result["scale"] = ss_result["scale"] * ss_result["downsample_factor"]
    coords = ss_result["coords"]
    slat_input = pipeline.preprocess_image(image, pipeline.slat_preprocessor)
    slat_embedder, cached_slat_embedder = _cached_condition_embedder(
        pipeline, "slat_condition_embedder", slat_input,
        pipeline.slat_condition_input_mapping,
    )
    pipeline.condition_embedders["slat_condition_embedder"] = cached_slat_embedder
    slat_modules = [pipeline.models["slat_generator"]]
    for module in slat_modules:
        _move_module(module, device)
    try:
        slat = pipeline.sample_slat(slat_input, coords)
    finally:
        for module in slat_modules:
            _move_module(module, "cpu")
        pipeline.condition_embedders["slat_condition_embedder"] = slat_embedder
        _release_cuda(torch)

    outputs: dict[str, Any] = {}
    for output_name, model_name in (
        ("gaussian", "slat_decoder_gs"), ("mesh", "slat_decoder_mesh"),
    ):
        decoder = pipeline.models[model_name]
        _move_module(decoder, device)
        try:
            with torch.no_grad():
                outputs[output_name] = decoder(slat)
        finally:
            _move_module(decoder, "cpu")
            _release_cuda(torch)
    outputs = pipeline.postprocess_slat_output(
        outputs, with_mesh_postprocess=True, with_texture_baking=False,
        use_vertex_color=True,
    )
    gaussian = outputs.get("gaussian")
    try:
        if gaussian is not None and pipeline.layout_post_optimization_method_GS is not None:
            ss_result.update(pipeline.run_post_optimization_GS(
                deepcopy(gaussian[0]), pointmap_dict["intrinsics"], ss_result,
                ss_input, backend="gsplat",
            ))
    except Exception as error:
        ss_result["layout_postprocess_error"] = f"{type(error).__name__}: {error}"
    return {
        **ss_result, **outputs,
        "pointmap": points.cpu().permute((1, 2, 0)),
        "pointmap_colors": point_colors.cpu().permute((1, 2, 0)),
    }


def _run_impl(
    run_dir: str | Path, *, config_path: str | Path, seeds: Iterable[int] = (42, 43, 44),
    max_keyframes: int = 2, max_proposals: int = 5, compile_model: bool = False,
    low_vram: bool = False, moge_resolution_level: int = 6,
    overwrite: bool = False, dry_run: bool = False,
) -> Path:
    root = Path(run_dir).resolve()
    output_dir = root / "mesh_proposals"
    ranking_path = output_dir / "mesh_ranking.json"
    if ranking_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {ranking_path}; pass --overwrite")
    ranking = rank_keyframes(root)
    if not ranking:
        raise RuntimeError("mesh_failed: no valid SAM object keyframe")
    keyframe_payload = [candidate.__dict__ for candidate in ranking]
    seed_list = list(seeds)
    if dry_run:
        _write_json(output_dir / "dry_run.json", {
            "schema_version": SCHEMA_VERSION, "keyframes": keyframe_payload,
            "config_path": str(Path(config_path).resolve()), "seeds": seed_list,
        })
        return output_dir / "dry_run.json"

    config_file = Path(config_path).resolve()
    inference = _load_inference(config_file, compile_model=compile_model, low_vram=low_vram)
    frames = _read_frame_index(root)
    objects = _load_masks(root / "segmentation/object_masks.npz")
    import zarr

    depth_group = zarr.open(str(root / "depth/metric_depth.zarr"), mode="r")
    depth_lookup = {int(frame): index for index, frame in enumerate(np.asarray(depth_group["frame_indices"]))}
    K = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    jobs = [(keyframe, seed) for keyframe in ranking[:max_keyframes] for seed in seed_list][:max_proposals]
    if len(jobs) < 3:
        jobs = (jobs * 3)[:3]
    proposals: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for proposal_index, (keyframe, seed) in enumerate(jobs):
        proposal_id = f"proposal_{proposal_index:02d}_f{keyframe.frame_index:06d}_s{seed}"
        proposal_dir = output_dir / proposal_id
        proposal_dir.mkdir(parents=True, exist_ok=True)
        prior_failure = proposal_dir / "failure.json"
        if overwrite and prior_failure.exists():
            history_dir = output_dir / "failure_history"
            history_dir.mkdir(parents=True, exist_ok=True)
            prior_failure.replace(history_dir / f"{proposal_id}_{time.time_ns()}.json")
        image_bgr = cv2.imread(str(frames[keyframe.frame_index]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"cannot read keyframe {frames[keyframe.frame_index]}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = objects["masks"][keyframe.mask_index].astype(bool)
        started = time.monotonic()
        record: dict[str, Any] = {
            "proposal_id": proposal_id, "frame_index": keyframe.frame_index,
            "mask_index": keyframe.mask_index, "seed": int(seed), "qualified": False,
        }
        try:
            rgba = np.concatenate([image_rgb, (mask.astype(np.uint8) * 255)[..., None]], axis=-1)
            if low_vram:
                output = _run_low_vram_pipeline(
                    inference, rgba, seed=seed,
                    moge_resolution_level=moge_resolution_level,
                )
            else:
                output = inference._pipeline.run(
                    rgba, None, seed=seed, stage1_only=False, with_mesh_postprocess=True,
                    with_texture_baking=False, with_layout_postprocess=True, use_vertex_color=True,
                )
            raw = _as_trimesh(output["glb"])
            raw_path = proposal_dir / "raw.glb"
            raw.export(raw_path)
            canonical, canonical_metadata = repair_and_canonicalize(raw)
            visual_path = proposal_dir / "visual.obj"
            collision_path = proposal_dir / "collision_source.obj"
            canonical.export(visual_path)
            canonical.export(collision_path)
            rotation = _tensor_list(output["rotation"], 4)
            translation = _tensor_list(output["translation"], 3)
            layout_scale = _tensor_list(output["scale"], 3)
            depth_at = depth_lookup[keyframe.frame_index]
            fit, T_camera_object, selected_scale = static_fit_metrics(
                canonical, mask, np.asarray(depth_group["depth_m"][depth_at]),
                np.asarray(depth_group["valid"][depth_at]).astype(bool), K,
                rotation_wxyz=rotation, translation=translation, layout_scale=layout_scale,
            )
            integrity = mesh_integrity(canonical)
            static_score = 0.35 * integrity["score"] + 0.40 * fit["silhouette_iou"] + 0.25 * fit["depth_score"]
            record.update({
                "qualified": bool(integrity["qualified"]), "static_score": float(static_score),
                "visual_mesh": str(visual_path.relative_to(output_dir)),
                "collision_source_mesh": str(collision_path.relative_to(output_dir)),
                "raw_mesh": str(raw_path.relative_to(output_dir)), "integrity": integrity,
                "fit": fit, "canonical": canonical_metadata,
                "model_layout": {"rotation_wxyz": rotation, "translation": translation, "scale": layout_scale},
                "selected_scale_m": selected_scale,
                "T_camera_object_initial": T_camera_object.tolist(),
            })
            _write_json(proposal_dir / "metadata.json", _finite_json(record))
        except Exception as error:
            record.update({
                "error_type": type(error).__name__, "error": str(error),
                "static_score": -1.0, "runtime_s": float(time.monotonic() - started),
            })
            _write_json(proposal_dir / "failure.json", record)
            if low_vram:
                import torch
                _release_cuda(torch)
        record["runtime_s"] = float(time.monotonic() - started)
        proposals.append(record)
    ordered = sorted(proposals, key=lambda item: (not item.get("qualified", False), -float(item.get("static_score", -1))))
    for rank, proposal in enumerate(ordered, start=1):
        proposal["rank"] = rank
    qualified_count = sum(bool(item.get("qualified")) for item in ordered)
    payload = {
        "schema_version": SCHEMA_VERSION, "stage": "sam3d_objects_mesh_proposals",
        "ranking_policy": "0.35 mesh_integrity + 0.40 silhouette_iou + 0.25 exp(-relative_depth_residual)",
        "foundationpose_used": False, "keyframes": keyframe_payload,
        "proposals": [_finite_json(item) for item in ordered], "qualified_count": qualified_count,
        "success": bool(qualified_count > 0),
        "failure_reason": None if qualified_count else "mesh_failed_no_qualified_proposal",
    }
    _write_json(ranking_path, payload)
    manifest = RunManifest.load(root / "manifest.json")
    manifest.finish_stage(
        "sam3d_objects", success=bool(qualified_count), outputs=[str(ranking_path.relative_to(root))],
        quality_metrics={"proposal_count": len(ordered), "qualified_count": qualified_count},
        warnings=["metric depth is a low-weight scale prior due to the recorded scene-scale conflict"],
    )
    if not qualified_count:
        raise RuntimeError("mesh_failed: no qualified SAM 3D proposal")
    return ranking_path


def run(
    run_dir: str | Path, *, config_path: str | Path, seeds: Iterable[int] = (42, 43, 44),
    max_keyframes: int = 2, max_proposals: int = 5, compile_model: bool = False,
    low_vram: bool = False, moge_resolution_level: int = 6,
    overwrite: bool = False, dry_run: bool = False,
) -> Path:
    root = Path(run_dir).resolve()
    ranking_path = root / "mesh_proposals/mesh_ranking.json"
    if ranking_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {ranking_path}; pass --overwrite")
    if dry_run:
        return _run_impl(
            run_dir, config_path=config_path, seeds=seeds, max_keyframes=max_keyframes,
            max_proposals=max_proposals, compile_model=compile_model,
            low_vram=low_vram, moge_resolution_level=moge_resolution_level,
            overwrite=overwrite, dry_run=True,
        )
    config_file = Path(config_path).resolve()
    seed_list = list(seeds)
    config = {
        "seeds": seed_list, "max_keyframes": max_keyframes,
        "max_proposals": max_proposals, "compile_model": compile_model,
        "low_vram": low_vram, "moge_resolution_level": moge_resolution_level,
    }
    inputs = [
        root / "segmentation/object_masks.npz", root / "segmentation/hand_masks.npz",
        root / "calibration/intrinsics.npy", config_file,
    ]
    manifest = RunManifest.load(root / "manifest.json")
    manifest.start_stage(
        "sam3d_objects", cache_key=stage_cache_key("sam3d_objects", config, inputs),
        command=sys.argv, environment="v2s-sam3d",
    )
    try:
        return _run_impl(
            run_dir, config_path=config_path, seeds=seed_list,
            max_keyframes=max_keyframes, max_proposals=max_proposals,
            compile_model=compile_model, low_vram=low_vram,
            moge_resolution_level=moge_resolution_level,
            overwrite=overwrite, dry_run=False,
        )
    except Exception as exc:
        failure_outputs: list[str] = []
        failure_metrics: dict[str, Any] = {"error_type": type(exc).__name__, "error": str(exc)}
        if ranking_path.exists():
            ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
            for proposal in ranking.get("proposals", []):
                failure_path = root / "mesh_proposals" / proposal["proposal_id"] / "failure.json"
                if failure_path.exists():
                    failure_outputs.append(str(failure_path.relative_to(root)))
            failure_metrics.update({
                "proposal_count": len(ranking.get("proposals", [])),
                "qualified_count": int(ranking.get("qualified_count", 0)),
                "failure_reason": ranking.get("failure_reason"),
            })
        manifest = RunManifest.load(root / "manifest.json")
        manifest.finish_stage(
            "sam3d_objects", success=False, outputs=failure_outputs,
            quality_metrics=failure_metrics,
            warnings=["SAM 3D Objects did not produce a qualified real mesh proposal"],
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-keyframes", type=int, default=2)
    parser.add_argument("--max-proposals", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--low-vram", action="store_true",
        help="Construct on CPU and move one official model stage to CUDA at a time",
    )
    parser.add_argument(
        "--moge-resolution-level", type=int, choices=range(10), default=6,
        help="MoGe token resolution level used by low-VRAM mode (0-9; output size is unchanged)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(run(
        args.run_dir, config_path=args.config_path, seeds=args.seeds,
        max_keyframes=args.max_keyframes, max_proposals=args.max_proposals,
        compile_model=args.compile, low_vram=args.low_vram,
        moge_resolution_level=args.moge_resolution_level,
        overwrite=args.overwrite, dry_run=args.dry_run,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
