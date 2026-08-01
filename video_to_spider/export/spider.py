"""Convert aligned V1 artifacts into SPIDER's actual keypoint dataset layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import trimesh

from ..coordinates import matrix_to_quaternion_wxyz, resample_transforms
from ..schemas import validate_aligned_trajectory, validate_contact

DATASET_NAME = "video_to_spider_egodex"
HAND_ROLES = {"invalid", "passive", "active"}


def _pose7(transforms: np.ndarray) -> np.ndarray:
    result = np.empty((transforms.shape[0], 7), dtype=np.float32)
    result[:, :3] = transforms[:, :3, 3]
    result[:, 3:] = matrix_to_quaternion_wxyz(transforms[:, :3, :3])
    return result


def _identity_pose(count: int) -> np.ndarray:
    result = np.zeros((count, 7), dtype=np.float32)
    result[:, 3] = 1.0
    return result


def _resample_points(source_t: np.ndarray, points: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    flat = points.reshape(points.shape[0], -1)
    result = np.empty((target_t.size, flat.shape[1]), dtype=np.float64)
    for index in range(flat.shape[1]):
        result[:, index] = np.interp(target_t, source_t, flat[:, index])
    return result.reshape((target_t.size,) + points.shape[1:])


def _resample_binary(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(source_t, target_t, side="right") - 1
    return values[np.clip(indices, 0, source_t.size - 1)]


def _copy_robot_assets(spider_package_root: Path, dataset_root: Path, robot_type: str) -> None:
    destination_root = dataset_root / "processed" / DATASET_NAME / "assets" / "robots"
    for name in {"mano", robot_type}:
        source = spider_package_root / "assets" / "robots" / name
        if not source.is_dir():
            raise FileNotFoundError(f"SPIDER robot assets missing: {source}")
        shutil.copytree(source, destination_root / name, dirs_exist_ok=True)


def _object_minimum_z(mesh: trimesh.Trimesh, transforms: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return np.asarray([
        float((vertices @ transform[:3, :3].T + transform[:3, 3]).min(axis=0)[2])
        for transform in transforms
    ])


def _align_initial_support(
    aligned: Mapping[str, np.ndarray], contact: Mapping[str, np.ndarray],
    visual_mesh: trimesh.Trimesh, *, object_clearance_m: float = 0.002,
    floor_tolerance_m: float = 1e-4,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Place the pre-contact object support surface on the simulation floor."""
    adjusted = {key: np.asarray(value).copy() for key, value in aligned.items()}
    object_poses = adjusted["T_sim_object"][:, 0]
    contact_frames = np.asarray(contact["contact"]).astype(bool).any(axis=(1, 2))
    first_contact = (
        int(np.flatnonzero(contact_frames)[0]) if contact_frames.any() else len(object_poses)
    )
    support_frame_count = max(1, min(first_contact, 10))
    minimum_before = _object_minimum_z(visual_mesh, object_poses)
    support_minimum = float(np.median(minimum_before[:support_frame_count]))
    if support_minimum < -floor_tolerance_m:
        raise ValueError(
            "SPIDER export blocked: object trajectory penetrates the floor; "
            f"minimum z={support_minimum:.6f} m"
        )
    shift = object_clearance_m - support_minimum
    adjusted["T_sim_object"][:, :, 2, 3] += shift
    adjusted["T_sim_wrist"][:, :, 2, 3] += shift
    adjusted["fingertips_sim"][:, :, :, 2] += shift
    minimum_after = _object_minimum_z(visual_mesh, adjusted["T_sim_object"][:, 0])
    if float(minimum_after.min()) < -1e-4:
        correction = object_clearance_m - float(minimum_after.min())
        shift += correction
        adjusted["T_sim_object"][:, :, 2, 3] += correction
        adjusted["T_sim_wrist"][:, :, 2, 3] += correction
        adjusted["fingertips_sim"][:, :, :, 2] += correction
        minimum_after = _object_minimum_z(visual_mesh, adjusted["T_sim_object"][:, 0])
    return adjusted, {
        "support_frame_count": support_frame_count,
        "first_contact_frame": first_contact,
        "object_min_z_support_before_m": support_minimum,
        "global_z_shift_m": float(shift),
        "object_min_z_after_m": float(minimum_after.min()),
        "object_clearance_m": object_clearance_m,
    }


def _simulation_preflight(
    aligned: Mapping[str, np.ndarray], visual_mesh: trimesh.Trimesh,
    hand_sides: Sequence[str], hand_roles: Mapping[str, str], *, floor_tolerance_m: float = 1e-4,
    hand_target_clearance_m: float = 0.002,
) -> dict[str, object]:
    roles = {side: hand_roles.get(side, "active") for side in hand_sides}
    unsupported = {side: role for side, role in roles.items() if role not in HAND_ROLES}
    if unsupported:
        raise ValueError(f"unsupported hand roles: {unsupported}")
    invalid = [side for side, role in roles.items() if role == "invalid"]
    if invalid:
        raise ValueError(
            f"SPIDER export blocked: requested hands failed reconstruction quality: {invalid}"
        )

    object_min_z = _object_minimum_z(visual_mesh, aligned["T_sim_object"][:, 0])
    if float(object_min_z.min()) < -floor_tolerance_m:
        raise ValueError(
            "SPIDER export blocked: object trajectory penetrates the floor; "
            f"minimum z={float(object_min_z.min()):.6f} m"
        )
    hand_metrics: dict[str, dict[str, float | str]] = {}
    for side in hand_sides:
        hand = hand_sides.index(side)
        wrist = aligned["T_sim_wrist"][:, hand, :3, 3]
        fingertips = aligned["fingertips_sim"][:, hand]
        target_min_z = min(float(wrist[:, 2].min()), float(fingertips[..., 2].min()))
        if target_min_z < hand_target_clearance_m - floor_tolerance_m:
            raise ValueError(
                f"SPIDER export blocked: {side} hand targets lack xHand floor clearance; "
                f"minimum z={target_min_z:.6f} m, required={hand_target_clearance_m:.6f} m"
            )
        wrist_tip_distance = np.linalg.norm(fingertips - wrist[:, None], axis=-1)
        if float(np.max(wrist_tip_distance)) > 0.35:
            raise ValueError(
                f"SPIDER export blocked: {side} wrist/fingertip geometry is inconsistent; "
                f"maximum distance={float(np.max(wrist_tip_distance)):.4f} m"
            )
        hand_metrics[side] = {
            "role": roles[side], "target_min_z_m": target_min_z,
            "wrist_tip_distance_median_m": float(np.median(wrist_tip_distance)),
            "wrist_tip_distance_max_m": float(np.max(wrist_tip_distance)),
        }
    return {
        "passed": True, "floor_tolerance_m": floor_tolerance_m,
        "hand_target_clearance_m": hand_target_clearance_m,
        "object_min_z_m": float(object_min_z.min()), "hands": hand_metrics,
    }


def export_spider_dataset(
    *, aligned_path: str | Path, contact_path: str | Path, visual_mesh_path: str | Path,
    dataset_root: str | Path, task: str, data_id: int, source_run_id: str,
    hand_sides: Sequence[str], spider_package_root: str | Path,
    hand_roles: Mapping[str, str] | None = None,
    embodiment_type: str = "right", robot_type: str = "xhand", ref_dt: float = 0.02,
) -> dict[str, Path]:
    """Export one trial. A single real object always occupies right_object."""
    if embodiment_type not in {"right", "left", "bimanual"}:
        raise ValueError("embodiment_type must be right, left, or bimanual")
    if len(set(hand_sides)) != len(hand_sides) or any(side not in {"left", "right"} for side in hand_sides):
        raise ValueError("hand_sides must contain unique left/right values")
    expected_sides = {
        "right": {"right"}, "left": {"left"}, "bimanual": {"left", "right"},
    }[embodiment_type]
    if set(hand_sides) != expected_sides:
        raise ValueError(
            f"{embodiment_type} embodiment requires hand_sides={sorted(expected_sides)}, "
            f"got {sorted(hand_sides)}"
        )
    aligned_file, contact_file = Path(aligned_path), Path(contact_path)
    with np.load(aligned_file, allow_pickle=False) as aligned_npz:
        aligned = {key: np.asarray(aligned_npz[key]) for key in aligned_npz.files}
    with np.load(contact_file, allow_pickle=False) as contact_npz:
        contact = {key: np.asarray(contact_npz[key]) for key in contact_npz.files}
    validate_aligned_trajectory(aligned)
    validate_contact(contact)
    source_t = aligned["timestamps_s"].astype(np.float64)
    if not np.array_equal(aligned["frame_indices"], contact["frame_indices"]) or not np.allclose(source_t, contact["timestamps_s"]):
        raise ValueError("aligned and contact artifacts have different timelines")
    if aligned["T_sim_wrist"].shape[1] != len(hand_sides) or contact["contact"].shape[1] != len(hand_sides):
        raise ValueError("hand_sides length does not match hand dimensions")
    if aligned["T_sim_object"].shape[1] != 1:
        raise ValueError("V1 SPIDER export requires exactly one object")
    visual_mesh = trimesh.load_mesh(visual_mesh_path, process=False)
    visual_mesh.apply_scale(float(aligned["object_scale_to_m"][0]))
    aligned, support_alignment = _align_initial_support(
        aligned, contact, visual_mesh,
    )
    resolved_roles = dict(hand_roles or {side: "active" for side in hand_sides})
    preflight = _simulation_preflight(aligned, visual_mesh, hand_sides, resolved_roles)
    target_t = np.arange(source_t[0], source_t[-1] + ref_dt * 0.25, ref_dt, dtype=np.float64)
    if target_t[-1] > source_t[-1] + 1e-9:
        target_t = target_t[:-1]
    object_transform = resample_transforms(source_t, aligned["T_sim_object"][:, 0], target_t)
    count = target_t.size
    output: dict[str, np.ndarray] = {
        "qpos_obj_right": _pose7(object_transform),
        "qpos_obj_left": _identity_pose(count),
    }
    for side in ("right", "left"):
        if side in hand_sides:
            hand_index = hand_sides.index(side)
            wrist = resample_transforms(source_t, aligned["T_sim_wrist"][:, hand_index], target_t)
            fingertips = _resample_points(source_t, aligned["fingertips_sim"][:, hand_index], target_t)
            qpos_finger = np.zeros((count, 5, 7), dtype=np.float32)
            qpos_finger[:, :, :3] = fingertips
            qpos_finger[:, :, 3] = 1.0
            output[f"qpos_wrist_{side}"] = _pose7(wrist)
            output[f"qpos_finger_{side}"] = qpos_finger
            output[f"contact_{side}"] = _resample_binary(source_t, contact["contact"][:, hand_index], target_t).astype(np.float32)
            output[f"contact_pos_{side}"] = contact["contact_pos_object_local"][hand_index].astype(np.float32)
        else:
            output[f"qpos_wrist_{side}"] = _identity_pose(count)
            output[f"qpos_finger_{side}"] = np.repeat(_identity_pose(count)[:, None], 5, axis=1)
            output[f"contact_{side}"] = np.zeros((count, 5), dtype=np.float32)
            output[f"contact_pos_{side}"] = np.zeros((5, 3), dtype=np.float32)
    root = Path(dataset_root).resolve()
    mano_dir = root / "processed" / DATASET_NAME / "mano" / embodiment_type / task / str(data_id)
    mano_dir.mkdir(parents=True, exist_ok=True)
    keypoints_path = mano_dir / "trajectory_keypoints.npz"
    np.savez_compressed(keypoints_path, **output)
    object_dir = root / "processed" / DATASET_NAME / "assets" / "objects" / source_run_id
    object_dir.mkdir(parents=True, exist_ok=True)
    # Aligned trajectories express object poses in meters while WP5 preserves a
    # normalized canonical mesh. Materialize the single episode-level scale in
    # the exported visual mesh so SPIDER never sees a unit-sized object.
    visual_mesh.export(object_dir / "visual.obj")
    _copy_robot_assets(Path(spider_package_root).resolve(), root, robot_type)
    task_info = {
        "task": task, "dataset_name": DATASET_NAME, "robot_type": "mano",
        "embodiment_type": embodiment_type, "data_id": int(data_id), "ref_dt": float(ref_dt),
        "right_object_mesh_dir": str(object_dir.relative_to(root)), "left_object_mesh_dir": None,
        "right_object_convex_dir": None, "left_object_convex_dir": None,
        "source_run_id": source_run_id, "num_frames": int(count),
        "hand_sides": list(hand_sides), "inactive_side_encoding": "zero_xyz_identity_wxyz",
        "hand_roles": resolved_roles, "simulation_preflight": preflight,
        "support_alignment": support_alignment,
    }
    task_info_path = mano_dir.parent / "task_info.json"
    task_info_path.write_text(json.dumps(task_info, indent=2) + "\n", encoding="utf-8")
    export_manifest = {
        "schema_version": "1.0", "dataset_root": str(root), "dataset_name": DATASET_NAME,
        "task": task, "data_id": int(data_id), "embodiment_type": embodiment_type,
        "robot_type": robot_type, "keypoints": str(keypoints_path), "task_info": str(task_info_path),
        "source_aligned": str(aligned_file.resolve()), "source_contact": str(contact_file.resolve()),
        "source_visual_mesh": str(Path(visual_mesh_path).resolve()),
        "object_scale_to_m_materialized": float(aligned["object_scale_to_m"][0]),
        "hand_roles": resolved_roles, "simulation_preflight": preflight,
        "support_alignment": support_alignment,
    }
    export_manifest_path = mano_dir / "export_manifest.json"
    export_manifest_path.write_text(json.dumps(export_manifest, indent=2) + "\n", encoding="utf-8")
    return {"keypoints": keypoints_path, "task_info": task_info_path, "export_manifest": export_manifest_path, "object_dir": object_dir}
