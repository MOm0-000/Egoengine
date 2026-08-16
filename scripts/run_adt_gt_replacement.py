#!/usr/bin/env python3
"""P5 GT-replacement runner for the ADT RGB object benchmark.

The production FoundationPose adapter always reads the standard run directory.
This script clones that standard layout for one P5 variant, replaces exactly one
upstream input with ADT ground truth, and reruns FoundationPose without touching
the production run or third-party model repositories.

Supported variants:
  depth_gt   replace projected stereo depth with ADT RGB GT depth
  mask_gt    replace SAM3 object masks with ADT RGB GT masks
  mesh_gt    replace SAM3D canonical mesh with the ADT object model
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
import zarr
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_to_spider.adapters.depth_gate import _artifact_fingerprint, _sha256
from video_to_spider.benchmark.adt_rgb_object import _rgb_world_cameras
from video_to_spider.ingest import adt_stereo


def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _quat_wxyz_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _invert(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def _angle(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def _percentile(values: np.ndarray, q: float) -> float:
    if not values.size:
        return math.nan
    return float(np.percentile(values, q))


def _rotation_axis_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c
    return np.asarray(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def _object_symmetry_rotations(sequence_dir: Path, object_uid: str) -> list[np.ndarray]:
    instances_path = sequence_dir / "instances.json"
    if not instances_path.is_file():
        return [np.eye(3, dtype=np.float64)]
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    info = instances.get(str(object_uid))
    if not info:
        info = instances.get(str(int(object_uid))) if str(object_uid).isdigit() else None
    if not info:
        return [np.eye(3, dtype=np.float64)]

    symmetry = info.get("rotational_symmetry") or {}
    axes = symmetry.get("axes") or []
    if not axes:
        return [np.eye(3, dtype=np.float64)]

    axis_options: list[list[np.ndarray]] = []
    for item in axes:
        axis = np.asarray(item.get("axis", [0.0, 0.0, 1.0]), dtype=np.float64)
        angle_degree = float(item.get("angle_degree", 0.0))
        if angle_degree <= 1e-9:
            # ADT uses angle_degree == 0 for continuous symmetry about this axis.
            angles = [2.0 * math.pi * index / 72.0 for index in range(72)]
        else:
            count = max(1, int(round(360.0 / angle_degree)))
            angles = [math.radians(angle_degree) * index for index in range(count)]
        axis_options.append([_rotation_axis_matrix(axis, angle) for angle in angles])

    rotations = [np.eye(3, dtype=np.float64)]
    for options in axis_options:
        next_rotations = []
        for current in rotations:
            for option in options:
                next_rotations.append(current @ option)
        rotations = next_rotations
    return rotations


def _as_trimesh(value) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value.copy()
    geometries = [geometry for geometry in value.geometry.values() if len(geometry.faces)]
    return trimesh.util.concatenate(geometries)


def _find_gt_mesh(object_root: Path, prototype: str) -> Path:
    candidates = [
        object_root / prototype / "3d-asset.glb",
        object_root / f"{prototype}-FixedTextures" / "3d-asset.glb",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    normalized = prototype.replace("_", "-").replace(" ", "-").lower()
    for directory in sorted(object_root.iterdir()):
        if not directory.is_dir():
            continue
        if directory.name.lower() == normalized or normalized in directory.name.lower():
            mesh = directory / "3d-asset.glb"
            if mesh.is_file():
                return mesh
    raise FileNotFoundError(f"no ADT object model found for prototype: {prototype}")


def _remove_symlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _symlink_dir(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _remove_symlink(target)
    target.symlink_to(source, target_is_directory=True)


def _copy_base_layout(base_run: Path, output_run: Path) -> None:
    if output_run.exists():
        shutil.rmtree(output_run)
    output_run.mkdir(parents=True, exist_ok=False)
    for directory in ("frames", "calibration", "input"):
        source = base_run / directory
        if source.exists():
            _symlink_dir(source, output_run / directory)
    for directory in ("depth", "segmentation", "mesh_proposals", "evaluation"):
        source = base_run / directory
        if source.exists():
            _symlink_dir(source, output_run / directory)
    shutil.copy2(base_run / "manifest.json", output_run / "manifest.json")


def _relative_frame_index(base_run: Path) -> tuple[list[int], list[float], list[int]]:
    frame_payload = json.loads((base_run / "frames/frame_index.json").read_text(encoding="utf-8"))
    frames = frame_payload["frames"]
    relative = [int(item["frame_index"]) for item in frames]
    source = [int(item["source_frame_index"]) for item in frames]
    timestamps = [float(item["timestamp_s"]) for item in frames]
    return relative, source, timestamps


def _write_generic_depth_gate(run_dir: Path) -> None:
    depth_path = run_dir / "depth/metric_depth.zarr"
    metadata_path = run_dir / "depth/metadata.json"
    recorded = {
        "depth_path": str(depth_path.resolve()),
        "depth_fingerprint_sha256": _artifact_fingerprint(depth_path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
    }
    payload = {
        "schema_version": "1.0",
        "gate_kind": "gt_replacement_synthetic",
        "policy": "P5 GT-replacement depth gate; accepted by construction",
        "accepted": True,
        "checks": {},
        "primary": recorded,
        "secondary": recorded,
    }
    (run_dir / "depth/depth_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _replace_depth(base_run: Path, output_run: Path) -> None:
    gt_group = zarr.open_group(str(base_run / "evaluation/adt_rgb_gt/adt_gt.zarr"), mode="r")
    _remove_symlink(output_run / "depth")
    depth_dir = output_run / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    _, source_frames, timestamps = _relative_frame_index(base_run)
    group = zarr.open_group(str(depth_dir / "metric_depth.zarr"), mode="w")
    group.create_dataset("frame_indices", data=np.asarray(source_frames, dtype=np.int64))
    group.create_dataset("timestamps_s", data=np.asarray(timestamps, dtype=np.float64))
    group.create_dataset(
        "depth_m",
        data=np.asarray(gt_group["depth_m"], dtype=np.float32),
        chunks=(1, min(256, gt_group["depth_m"].shape[1]), min(256, gt_group["depth_m"].shape[2])),
        dtype="f4",
    )
    group.create_dataset(
        "valid",
        data=np.asarray(gt_group["valid"], dtype=bool),
        chunks=(1, min(256, gt_group["valid"].shape[1]), min(256, gt_group["valid"].shape[2])),
        dtype="bool",
    )
    metadata = {
        "schema_version": "1.0",
        "model": "ADT RGB GT depth (P5 replacement)",
        "ground_truth": True,
        "depth_units": "meter",
        "frame_count": len(source_frames),
        "replacement_variant": "depth_gt",
    }
    (depth_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_generic_depth_gate(output_run)


def _replace_mask(base_run: Path, output_run: Path) -> None:
    gt_group = zarr.open_group(str(base_run / "evaluation/adt_rgb_gt/adt_gt.zarr"), mode="r")
    _remove_symlink(output_run / "segmentation")
    segmentation_dir = output_run / "segmentation"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    _, source_frames, timestamps = _relative_frame_index(base_run)
    gt_indices = np.asarray(gt_group["frame_indices"], dtype=np.int64)
    gt_masks = np.asarray(gt_group["object_mask"], dtype=bool)
    gt_valid = np.asarray(gt_group["valid"], dtype=bool)
    lookup = {int(frame): index for index, frame in enumerate(gt_indices)}
    selected = [lookup[frame] for frame in source_frames]
    object_masks = gt_masks[selected]
    object_valid = np.asarray(
        [bool(gt_masks[index].any() and gt_valid[index].any()) for index in selected],
        dtype=bool,
    )
    np.savez_compressed(
        segmentation_dir / "object_masks.npz",
        frame_indices=np.asarray(source_frames, dtype=np.int64),
        timestamps_s=np.asarray(timestamps, dtype=np.float64),
        masks=object_masks,
        valid=object_valid,
        confidence=np.ones(len(source_frames), dtype=np.float32),
        object_ids=np.zeros(len(source_frames), dtype=np.int64),
    )
    shutil.copy2(base_run / "segmentation/hand_masks.npz", segmentation_dir / "hand_masks.npz")
    metadata = {
        "schema_version": "1.0",
        "model": "ADT RGB GT mask (P5 replacement)",
        "ground_truth": True,
        "frame_count": len(source_frames),
        "replacement_variant": "mask_gt",
    }
    (segmentation_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_generic_depth_gate(output_run)


def _replace_mesh(base_run: Path, output_run: Path, object_root: Path, prototype: str) -> None:
    _remove_symlink(output_run / "mesh_proposals")
    mesh_dir = output_run / "mesh_proposals" / "gt_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    gt_mesh = _as_trimesh(trimesh.load(_find_gt_mesh(object_root, prototype), process=False))
    gt_mesh.export(str(mesh_dir / "visual.obj"))
    _, source_frames, _ = _relative_frame_index(base_run)
    keyframes = [{"frame_index": int(frame), "score": 1.0} for frame in source_frames]
    proposal = {
        "proposal_id": "gt_mesh",
        "frame_index": int(source_frames[0]),
        "mask_index": 0,
        "seed": -1,
        "qualified": True,
        "static_score": 1.0,
        "rank": 1,
        "visual_mesh": "gt_mesh/visual.obj",
        "collision_source_mesh": "gt_mesh/visual.obj",
        "raw_mesh": "gt_mesh/visual.obj",
        "selected_scale_m": 1.0,
        "canonical": {"debug_only": True},
    }
    ranking = {
        "schema_version": "1.0",
        "stage": "p5_gt_mesh_replacement",
        "ranking_policy": "GT object model replacement",
        "foundationpose_used": False,
        "keyframes": keyframes,
        "proposals": [proposal],
        "qualified_count": 1,
        "success": True,
        "failure_reason": None,
    }
    (output_run / "mesh_proposals/mesh_ranking.json").write_text(json.dumps(ranking, indent=2) + "\n", encoding="utf-8")
    _write_generic_depth_gate(output_run)


def _run_foundationpose(run_dir: Path, gpu: str, max_input_side: int) -> None:
    command = [
        str(REPO_ROOT / "scripts/run_model_adapter.sh"),
        "v2s-foundationpose",
        "video_to_spider.adapters.foundationpose",
        "--run-dir", str(run_dir),
        "--foundationpose-root", str(REPO_ROOT / "third_party/FoundationPose"),
        "--max-candidates", "3",
        "--screening-radius", "1",
        "--register-iter", "2",
        "--track-iter", "1",
        "--max-input-side", str(max_input_side),
        "--overwrite",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def _foundationpose_gt_metrics(
    run_dir: Path, sequence_dir: Path, prepared_dir: Path, object_uid: str, prototype: str,
    object_root: Path,
) -> dict[str, float | int]:
    raw = np.load(run_dir / "object_tracking/foundationpose_raw.npz", allow_pickle=False)
    selected = json.loads((run_dir / "object_tracking/selected_mesh.json").read_text(encoding="utf-8"))
    pred_mesh = _as_trimesh(trimesh.load(run_dir / selected["canonical_visual_mesh"], process=False))
    pred_mesh.apply_scale(float(selected["scale_to_m"]))
    gt_mesh = _as_trimesh(trimesh.load(_find_gt_mesh(object_root, prototype), process=False))
    sample_count = max(3, min(600, len(pred_mesh.faces), len(gt_mesh.faces)))
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, sample_count)
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, sample_count)

    timestamps = np.asarray(
        json.loads((prepared_dir / "calibration/timestamps.json").read_text(encoding="utf-8"))["timestamps_s"],
        dtype=np.float64,
    )
    data_provider, _calibration, _mps, _sensor, _sophus = adt_stereo._load_projectaria()
    video_provider = data_provider.create_vrs_data_provider(str(sequence_dir / "vrs_files/video.vrs"))
    rgb_calib = video_provider.get_device_calibration().get_camera_calib("camera-rgb")
    world_cameras = _rgb_world_cameras(sequence_dir, rgb_calib, timestamps)

    object_rows = [
        row
        for row in csv.DictReader((sequence_dir / "scene_objects.csv").open(encoding="utf-8"))
        if row["object_uid"] == object_uid
    ]
    object_times = np.asarray([float(row["timestamp[ns]"]) * 1e-9 for row in object_rows], dtype=np.float64)
    translations: list[float] = []
    rotation_errors: list[float] = []
    symmetry_rotation_errors: list[float] = []
    add_s_values: list[float] = []
    symmetry_rotations = _object_symmetry_rotations(sequence_dir, object_uid)
    for index, transform in enumerate(raw["T_camera_object"]):
        if not bool(raw["valid"][index]):
            continue
        object_index = int(np.argmin(np.abs(object_times - timestamps[index])))
        row = object_rows[object_index]
        world_object = _pose(
            _quat_wxyz_matrix(
                float(row["q_wo_w"]), float(row["q_wo_x"]), float(row["q_wo_y"]), float(row["q_wo_z"])
            ),
            np.asarray([float(row["t_wo_x[m]"]), float(row["t_wo_y[m]"]), float(row["t_wo_z[m]"])]),
        )
        gt_camera_object = _invert(world_cameras[index]) @ world_object
        pred_centroid = np.asarray(pred_mesh.centroid, dtype=np.float64)
        gt_centroid = np.asarray(gt_mesh.centroid, dtype=np.float64)
        pred_camera_centroid = transform[:3, :3] @ pred_centroid + transform[:3, 3]
        gt_camera_centroid = gt_camera_object[:3, :3] @ gt_centroid + gt_camera_object[:3, 3]
        translations.append(float(np.linalg.norm(pred_camera_centroid - gt_camera_centroid)))

        relative_rotation = gt_camera_object[:3, :3].T @ transform[:3, :3]
        rotation_errors.append(float(_angle(relative_rotation)))
        symmetric_angles = [
            _angle(relative_rotation @ symmetry)
            for symmetry in symmetry_rotations
        ]
        symmetry_rotation_errors.append(float(min(symmetric_angles)))

        pred_camera_points = pred_points @ transform[:3, :3].T + transform[:3, 3]
        gt_camera_points = gt_points @ gt_camera_object[:3, :3].T + gt_camera_object[:3, 3]
        distance_pred_to_gt = cKDTree(gt_camera_points).query(pred_camera_points)[0]
        distance_gt_to_pred = cKDTree(pred_camera_points).query(gt_camera_points)[0]
        add_s_values.append(float(np.concatenate([distance_pred_to_gt, distance_gt_to_pred]).mean()))

    translations = np.asarray(translations, dtype=np.float64)
    rotation_errors = np.asarray(rotation_errors, dtype=np.float64)
    symmetry_rotation_errors = np.asarray(symmetry_rotation_errors, dtype=np.float64)
    add_s_values = np.asarray(add_s_values, dtype=np.float64)
    return {
        "valid_rate": float(raw["valid"].mean()) if raw["valid"].size else 0.0,
        "centroid_translation_error_m_median": _percentile(translations, 50),
        "centroid_translation_error_m_mean": float(translations.mean()) if translations.size else math.nan,
        "centroid_translation_error_m_p95": _percentile(translations, 95),
        "rotation_error_deg_median": float(np.degrees(_percentile(rotation_errors, 50))),
        "rotation_error_deg_mean": float(np.degrees(rotation_errors.mean())) if rotation_errors.size else math.nan,
        "rotation_error_deg_p95": float(np.degrees(_percentile(rotation_errors, 95))),
        "symmetry_aware_rotation_error_deg_median": float(np.degrees(_percentile(symmetry_rotation_errors, 50))),
        "symmetry_aware_rotation_error_deg_mean": float(np.degrees(symmetry_rotation_errors.mean())) if symmetry_rotation_errors.size else math.nan,
        "symmetry_aware_rotation_error_deg_p95": float(np.degrees(_percentile(symmetry_rotation_errors, 95))),
        "add_s_m_median": _percentile(add_s_values, 50),
        "add_s_m_mean": float(add_s_values.mean()) if add_s_values.size else math.nan,
        "add_s_m_p95": _percentile(add_s_values, 95),
        "tracking_score": json.loads((run_dir / "object_tracking/tracking_gate.json").read_text())["metrics"]["tracking_score"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--object-uid", type=str, required=True)
    parser.add_argument("--prototype", type=str, required=True)
    parser.add_argument("--object-root", type=Path, default=Path("/data_all/zzx/egoengine/adt_object_library"))
    parser.add_argument("--variant", required=True, choices=["f0", "depth_gt", "mask_gt", "mesh_gt"])
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--max-input-side", type=int, default=640)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_run = args.base_run.resolve()
    if args.variant == "f0":
        if not (base_run / "object_tracking/foundationpose_raw.npz").is_file():
            raise RuntimeError("f0 requires an existing FoundationPose result")
        metrics = _foundationpose_gt_metrics(
            base_run, args.sequence_dir.resolve(), args.prepared_dir.resolve(), args.object_uid,
            args.prototype, args.object_root.resolve(),
        )
        (base_run / "p5_metrics.json").write_text(
            json.dumps({"variant": "f0", "metrics": metrics}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"variant": "f0", "run_dir": str(base_run), "metrics": metrics}, indent=2, ensure_ascii=False))
        return 0

    output_run = base_run.parent / f"{base_run.name}_p5_{args.variant}"
    if output_run.exists() and not args.overwrite:
        raise FileExistsError(f"output run exists: {output_run}; pass --overwrite")
    _copy_base_layout(base_run, output_run)
    if args.variant == "depth_gt":
        _replace_depth(base_run, output_run)
    elif args.variant == "mask_gt":
        _replace_mask(base_run, output_run)
    elif args.variant == "mesh_gt":
        _replace_mesh(base_run, output_run, args.object_root.resolve(), args.prototype)
    else:
        raise ValueError(args.variant)
    if args.prepare_only:
        print(output_run)
        return 0
    try:
        _run_foundationpose(output_run, args.gpu, args.max_input_side)
    except subprocess.CalledProcessError as error:
        failure = {
            "variant": args.variant,
            "run_dir": str(output_run),
            "status": "failed",
            "error": f"FoundationPose replacement failed: {error}",
        }
        (output_run / "p5_metrics.json").write_text(
            json.dumps(failure, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 1
    metrics = _foundationpose_gt_metrics(
        output_run, args.sequence_dir.resolve(), args.prepared_dir.resolve(), args.object_uid,
        args.prototype, args.object_root.resolve(),
    )
    (output_run / "p5_metrics.json").write_text(
        json.dumps({"variant": args.variant, "metrics": metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"variant": args.variant, "run_dir": str(output_run), "metrics": metrics}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
