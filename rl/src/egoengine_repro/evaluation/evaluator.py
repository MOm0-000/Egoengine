"""Unified 3.1 evaluator with strict ground-truth isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import zarr
from scipy.spatial import cKDTree

from .. import SCHEMA_VERSION
from ..artifacts import artifact_record
from ..config import ReproConfig
from .metrics import (
    _sample_mesh,
    contact_metrics,
    contact_slip,
    depth_metrics,
    depth_warp_residual,
    mask_metrics,
    mesh_metrics,
    object_trajectory_metrics,
    project_points,
    rotation_geodesic,
    summary,
)


FINGERTIP_INDICES = np.asarray([4, 8, 12, 16, 20])


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _artifact_path(manifest: dict[str, Any], name: str) -> Path | None:
    record = manifest.get("artifacts", {}).get(name)
    if record is None:
        return None
    raw = record.get("path") if isinstance(record, dict) else record
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path(manifest["_manifest_dir"]) / path
    path = path.resolve()
    current = artifact_record(path)
    if isinstance(record, dict) and record.get("sha256") != current["sha256"]:
        raise ValueError(f"ground-truth artifact changed after registration: {path}")
    return path


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _available(metrics: dict[str, Any]) -> dict[str, Any]:
    return {"status": "available", "metrics": metrics}


def _world_to_camera(T_world_camera: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    rotation = np.swapaxes(T_world_camera[:, :3, :3], 1, 2)
    return np.einsum(
        "tij,thkj->thki", rotation,
        points_world - T_world_camera[:, None, None, :3, 3],
    )


def _anatomical_palm_rotation(joints: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(joints, dtype=np.float64)
    z_axis = values[:, 9] - values[:, 0]
    lateral = values[:, 5] - values[:, 13]
    z_norm = np.linalg.norm(z_axis, axis=1)
    z_unit = z_axis / np.maximum(z_norm[:, None], 1e-12)
    x_axis = np.cross(lateral, z_unit)
    x_norm = np.linalg.norm(x_axis, axis=1)
    x_unit = x_axis / np.maximum(x_norm[:, None], 1e-12)
    y_axis = np.cross(z_unit, x_unit)
    y_norm = np.linalg.norm(y_axis, axis=1)
    y_unit = y_axis / np.maximum(y_norm[:, None], 1e-12)
    if side == "left":
        x_unit *= -1.0
        y_unit *= -1.0
    rotation = np.stack([x_unit, y_unit, z_unit], axis=-1)
    valid = (
        np.isfinite(rotation).all(axis=(1, 2)) & (z_norm > 1e-8)
        & (x_norm > 1e-8) & (y_norm > 1e-8)
    )
    return rotation, valid


def _hand_metrics(run_dir: Path, gt_hand_path: Path, gt_camera_path: Path) -> dict[str, Any]:
    pred = _npz(run_dir / "hands/wilor_raw.npz")
    gt = _npz(gt_hand_path)
    camera = _npz(gt_camera_path)
    if not np.array_equal(pred["frame_indices"], gt["frame_indices"]):
        raise ValueError("WiLoR and hand GT frame indices differ")
    if not np.array_equal(pred["frame_indices"], camera["frame_indices"]):
        raise ValueError("WiLoR and camera GT frame indices differ")
    gt_world = gt["T_world_joint"][..., :3, 3]
    gt_camera = _world_to_camera(camera["T_world_camera"], gt_world)
    pred_camera = pred["joints_camera_rootrel"] + pred["translation_camera"][:, :, None]
    valid = pred["valid"].astype(bool)[:, :, None] & (gt["confidence"] > 0)
    absolute = np.linalg.norm(pred_camera - gt_camera, axis=-1)
    rootrel = np.linalg.norm(
        (pred_camera - pred_camera[:, :, [0]]) - (gt_camera - gt_camera[:, :, [0]]), axis=-1,
    )
    K = camera["intrinsics"]
    if K.shape == (3, 3):
        K = np.repeat(K[None], len(pred_camera), axis=0)
    pred_uv, pred_projectable = project_points(K[:, None], pred_camera)
    gt_uv, gt_projectable = project_points(K[:, None], gt_camera)
    reprojection = np.linalg.norm(pred_uv - gt_uv, axis=-1)
    reprojection_valid = valid & pred_projectable & gt_projectable
    result: dict[str, Any] = {"hand_order": gt["hand_order"].tolist(), "per_hand": {}}
    for hand, side in enumerate(gt["hand_order"].tolist()):
        hand_valid = valid[:, hand]
        frame_valid = hand_valid[:, 0]
        pred_palm, pred_palm_valid = _anatomical_palm_rotation(pred_camera[:, hand], str(side))
        gt_palm, gt_palm_valid = _anatomical_palm_rotation(gt_camera[:, hand], str(side))
        rotation_valid = frame_valid & pred_palm_valid & gt_palm_valid
        wrist_rotation = rotation_geodesic(pred_palm, gt_palm)
        result["per_hand"][str(side)] = {
            "valid_rate": float(pred["valid"][:, hand].mean()),
            "wrist_translation_error_m": summary(absolute[:, hand, 0][frame_valid]),
            "fingertip_mpjpe_m": summary(absolute[:, hand, FINGERTIP_INDICES][hand_valid[:, FINGERTIP_INDICES]]),
            "all_joint_mpjpe_m": summary(absolute[:, hand][hand_valid]),
            "root_relative_joint_mpjpe_m": summary(rootrel[:, hand, 1:][hand_valid[:, 1:]]),
            "wrist_rotation_geodesic_rad": summary(wrist_rotation[rotation_valid]),
            "wrist_rotation_definition": "geodesic between anatomical palm frames built from 21 joints",
            "reprojection_error_px": summary(reprojection[:, hand][reprojection_valid[:, hand]]),
        }
    identity_cost = np.linalg.norm(pred_camera[:, :, 0] - gt_camera[:, :, 0], axis=-1).sum(axis=1)
    swapped_cost = np.linalg.norm(pred_camera[:, :, 0] - gt_camera[:, ::-1, 0], axis=-1).sum(axis=1)
    assignment = swapped_cost < identity_cost
    both_valid = pred["valid"].astype(bool).all(axis=1)
    usable_assignment = assignment[both_valid]
    result["identity"] = {
        "swapped_assignment_frame_rate": float(usable_assignment.mean()) if usable_assignment.size else None,
        "id_switch_count": int(np.count_nonzero(np.diff(usable_assignment.astype(np.int8)))) if usable_assignment.size > 1 else 0,
        "definition": "changes in minimum-cost left/right wrist assignment",
    }
    return result


def _camera_metrics(run_dir: Path, gt_camera_path: Path) -> dict[str, Any]:
    gt = _npz(gt_camera_path)
    pred = np.load(run_dir / "calibration/T_world_camera.npy").astype(np.float64)
    if len(pred) != len(gt["T_world_camera"]):
        raise ValueError("predicted and GT camera timelines differ")
    translation = np.linalg.norm(pred[:, :3, 3] - gt["T_world_camera"][:, :3, 3], axis=1)
    rotation = rotation_geodesic(pred[:, :3, :3], gt["T_world_camera"][:, :3, :3])
    return {
        "translation_error_m": summary(translation), "rotation_geodesic_rad": summary(rotation),
        "interpretation": "calibration passthrough check; not an independently inferred camera trajectory",
    }


def _predicted_mesh(run_dir: Path) -> trimesh.Trimesh:
    selected = json.loads((run_dir / "object_tracking/selected_mesh.json").read_text(encoding="utf-8"))
    mesh = trimesh.load_mesh(run_dir / selected["canonical_visual_mesh"], process=False)
    aligned_path = run_dir / "optimization/aligned_trajectory.npz"
    scale = float(selected["scale_to_m"])
    if aligned_path.exists():
        aligned = _npz(aligned_path)
        scale = float(aligned["object_scale_to_m"][0])
    mesh.apply_scale(scale)
    return mesh


def _segmentation_metrics(run_dir: Path, gt_path: Path) -> dict[str, Any]:
    pred = _npz(run_dir / "segmentation/object_masks.npz")
    gt = _npz(gt_path)
    if not np.array_equal(pred["frame_indices"], gt["frame_indices"]):
        raise ValueError("predicted and GT mask frame indices differ")
    return mask_metrics(pred["masks"], gt["masks"], pred.get("valid"), gt.get("valid"))


def _depth_metrics(run_dir: Path, gt_path: Path, config: ReproConfig, camera_path: Path) -> dict[str, Any]:
    root = zarr.open_group(str(run_dir / "depth/metric_depth.zarr"), mode="r")
    pred = np.asarray(root["depth_m"])
    gt = _npz(gt_path)
    result = depth_metrics(pred, gt["depth_m"], gt.get("valid"))
    camera = _npz(camera_path)
    result["cross_frame_warp"] = depth_warp_residual(
        pred, camera["intrinsics"], camera["T_world_camera"],
        stride=int(config.data["evaluation"]["depth_warp_stride"]),
    )
    return result


def _object_prediction(
    run_dir: Path, preferred_frame: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    raw_path = run_dir / "object_tracking/foundationpose_raw.npz"
    if preferred_frame == "camera" and raw_path.is_file():
        raw = _npz(raw_path)
        return raw["frame_indices"], raw["T_camera_object"], raw["valid"].astype(bool), "camera"
    aligned_path = run_dir / "optimization/aligned_trajectory.npz"
    if aligned_path.exists():
        artifact = _npz(aligned_path)
        return (
            artifact["frame_indices"], artifact["T_sim_object"][:, 0],
            artifact["valid_object"][:, 0].astype(bool), "sim",
        )
    raw = _npz(raw_path)
    return raw["frame_indices"], raw["T_camera_object"], raw["valid"].astype(bool), "camera"


def _trajectory_metrics(
    run_dir: Path, gt_path: Path, gt_mesh: trimesh.Trimesh | None, config: ReproConfig,
) -> dict[str, Any]:
    gt = _npz(gt_path)
    preferred_frame = "camera" if "T_camera_object" in gt else None
    frames, pred_pose, pred_valid, frame_name = _object_prediction(run_dir, preferred_frame)
    gt_pose = gt.get(f"T_{frame_name}_object", gt.get("T_object"))
    if gt_pose is None:
        raise ValueError(f"object GT does not provide T_{frame_name}_object or T_object")
    if not np.array_equal(frames, gt["frame_indices"]):
        raise ValueError("predicted and GT object frame indices differ")
    valid = pred_valid & gt.get("valid", np.ones(len(frames), dtype=bool)).astype(bool)
    mesh_points = None
    if gt_mesh is not None:
        mesh_points = _sample_mesh(
            gt_mesh, int(config.data["evaluation"]["object_add_sample_count"]), seed=31,
        )
    result = object_trajectory_metrics(pred_pose, gt_pose, valid, mesh_points=mesh_points)
    result["coordinate_frame"] = frame_name
    return result


def _fingertip_object_distance(
    fingertips: np.ndarray, object_poses: np.ndarray, mesh: trimesh.Trimesh,
) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    distances = []
    for frame in range(len(fingertips)):
        world_vertices = vertices @ object_poses[frame, :3, :3].T + object_poses[frame, :3, 3]
        distances.append(cKDTree(world_vertices).query(fingertips[frame].reshape(-1, 3), workers=-1)[0])
    return {"distance_m": summary(np.concatenate(distances))}


def _joint_metrics(run_dir: Path, gt_contact_path: Path | None, gt_mesh: trimesh.Trimesh | None) -> dict[str, Any]:
    aligned_path = run_dir / "optimization/aligned_trajectory.npz"
    contact_path = run_dir / "optimization/contact.npz"
    if not aligned_path.exists() or not contact_path.exists():
        raise FileNotFoundError("aligned trajectory and contact artifacts are required")
    aligned, predicted_contact = _npz(aligned_path), _npz(contact_path)
    result: dict[str, Any] = {
        "predicted_contact_slip": contact_slip(
            aligned["fingertips_sim"], aligned["T_sim_object"][:, 0],
            predicted_contact["contact"], aligned["timestamps_s"],
        )
    }
    if gt_mesh is not None:
        result["predicted_fingertip_object"] = _fingertip_object_distance(
            aligned["fingertips_sim"], aligned["T_sim_object"][:, 0], gt_mesh,
        )
    if gt_contact_path is not None:
        gt_contact = _npz(gt_contact_path)
        result["contact_classification"] = contact_metrics(
            predicted_contact["contact"], gt_contact["contact"], gt_contact.get("valid"),
        )
    return result


def evaluate_run(
    run_dir: str | Path, ground_truth_manifest: str | Path, output_dir: str | Path,
    config: ReproConfig,
) -> Path:
    """Evaluate available 3.1 modalities without exposing GT to inference code."""
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    if run == output or run in output.parents:
        raise ValueError("evaluation output must be outside the source run")
    output.mkdir(parents=True, exist_ok=True)
    gt_manifest_path = Path(ground_truth_manifest).resolve()
    gt_manifest = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
    if gt_manifest.get("uses_ground_truth") is not True or gt_manifest.get("scope") != "evaluation_only":
        raise ValueError("ground-truth manifest must be explicitly evaluation-only")
    gt_manifest["_manifest_dir"] = str(gt_manifest_path.parent)
    hand_path = _artifact_path(gt_manifest, "hand")
    camera_path = _artifact_path(gt_manifest, "camera")
    mask_path = _artifact_path(gt_manifest, "segmentation")
    depth_path = _artifact_path(gt_manifest, "depth")
    mesh_path = _artifact_path(gt_manifest, "mesh")
    trajectory_path = _artifact_path(gt_manifest, "object_trajectory")
    contact_path = _artifact_path(gt_manifest, "contact")
    gt_mesh = trimesh.load_mesh(mesh_path, process=False) if mesh_path else None
    modalities: dict[str, Any] = {}
    hand_prediction = run / "hands/wilor_raw.npz"
    segmentation_prediction = run / "segmentation/object_masks.npz"
    depth_prediction = run / "depth/metric_depth.zarr"
    mesh_prediction = run / "object_tracking/selected_mesh.json"
    trajectory_predictions = (
        run / "object_tracking/foundationpose_raw.npz",
        run / "optimization/aligned_trajectory.npz",
    )
    modalities["hand"] = (
        _available(_hand_metrics(run, hand_path, camera_path))
        if hand_path and camera_path and hand_prediction.is_file()
        else _unavailable("hand/camera GT and WiLoR prediction are required")
    )
    modalities["camera"] = (
        _available(_camera_metrics(run, camera_path)) if camera_path else _unavailable("camera GT not supplied")
    )
    modalities["segmentation"] = (
        _available(_segmentation_metrics(run, mask_path))
        if mask_path and segmentation_prediction.is_file()
        else _unavailable("segmentation GT and predicted object masks are required")
    )
    modalities["depth"] = (
        _available(_depth_metrics(run, depth_path, config, camera_path))
        if depth_path and camera_path and depth_prediction.is_dir()
        else _unavailable("depth/camera GT and predicted metric depth are required")
    )
    modalities["mesh"] = (
        _available(mesh_metrics(
            _predicted_mesh(run), gt_mesh,
            sample_count=int(config.data["evaluation"]["mesh_sample_count"]),
            fscore_thresholds_m=config.data["evaluation"]["fscore_thresholds_m"],
        )) if gt_mesh is not None and mesh_prediction.is_file()
        else _unavailable("mesh GT and selected predicted mesh are required")
    )
    modalities["object_trajectory"] = (
        _available(_trajectory_metrics(run, trajectory_path, gt_mesh, config))
        if trajectory_path and any(path.is_file() for path in trajectory_predictions)
        else _unavailable("object trajectory GT and predicted object trajectory are required")
    )
    joint_inputs = (
        run / "optimization/aligned_trajectory.npz",
        run / "optimization/contact.npz",
    )
    modalities["joint_relation"] = (
        _available(_joint_metrics(run, contact_path, gt_mesh))
        if all(path.is_file() for path in joint_inputs)
        else _unavailable("aligned trajectory and predicted contact artifacts are required")
    )
    report = {
        "schema_version": SCHEMA_VERSION, "profile": config.name,
        "profile_uses_ground_truth": config.uses_ground_truth,
        "evaluation_uses_ground_truth": True,
        "ground_truth_scope": "evaluation_only", "source_run": str(run),
        "ground_truth_manifest": artifact_record(gt_manifest_path),
        "modalities": modalities,
        "available_modalities": [name for name, record in modalities.items() if record["status"] == "available"],
        "unavailable_modalities": [name for name, record in modalities.items() if record["status"] != "available"],
    }
    report_path = output / "offline_3_1_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path
