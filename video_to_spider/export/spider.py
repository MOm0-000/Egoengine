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


def _longest_true_window(values: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, value in enumerate(np.r_[np.asarray(values, dtype=bool), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = (start, index)
            if best is None or candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
            start = None
    return best


def _crop_to_longest_valid_window(
    aligned: Mapping[str, np.ndarray], contact: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Remove invalid upstream tails before interpolation into MINK targets."""
    count = len(aligned["timestamps_s"])
    valid = np.asarray(aligned["valid_object"], dtype=bool).all(axis=1)
    valid &= np.asarray(aligned["valid_hand"], dtype=bool).all(axis=1)
    bounds = _longest_true_window(valid)
    if bounds is None or bounds[1] - bounds[0] < 2:
        raise ValueError(
            "SPIDER export requires at least two consecutive valid object/hand frames"
        )
    start, stop = bounds
    aligned_timeline_keys = {
        "frame_indices", "timestamps_s", "T_sim_object", "T_sim_wrist",
        "fingertips_sim", "fingertip_orientation_sim", "mano_pose",
        "valid_object", "valid_hand", "confidence_object", "confidence_hand",
    }
    contact_timeline_keys = {"frame_indices", "timestamps_s", "contact"}
    cropped_aligned = {
        key: (np.asarray(value)[start:stop].copy() if key in aligned_timeline_keys else np.asarray(value).copy())
        for key, value in aligned.items()
    }
    cropped_contact = {
        key: (np.asarray(value)[start:stop].copy() if key in contact_timeline_keys else np.asarray(value).copy())
        for key, value in contact.items()
    }
    return cropped_aligned, cropped_contact, {
        "source_frame_count": int(count),
        "selected_start_row": int(start),
        "selected_stop_row_exclusive": int(stop),
        "selected_frame_count": int(stop - start),
        "trimmed_frame_count": int(count - (stop - start)),
        "selected_first_frame_index": int(aligned["frame_indices"][start]),
        "selected_last_frame_index": int(aligned["frame_indices"][stop - 1]),
    }


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


def _resample_rotations(
    source_t: np.ndarray, rotations: np.ndarray, target_t: np.ndarray,
) -> np.ndarray:
    """SLERP an arbitrary stack of proper rotation matrices along time."""
    from scipy.spatial.transform import Rotation, Slerp

    values = np.asarray(rotations, dtype=np.float64)
    if values.shape[0] != len(source_t) or values.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape (T, ..., 3, 3)")
    flat = values.reshape(len(source_t), -1, 3, 3)
    result = np.empty((len(target_t), flat.shape[1], 3, 3), dtype=np.float64)
    if len(source_t) == 1:
        result[:] = flat[0]
    else:
        for joint in range(flat.shape[1]):
            result[:, joint] = Slerp(
                source_t, Rotation.from_matrix(flat[:, joint]),
            )(target_t).as_matrix()
    return result.reshape((len(target_t),) + values.shape[1:])


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
    floor_tolerance_m: float = 1e-4, hand_target_clearance_m: float = 0.002,
    max_frame_correction_m: float = 0.05,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Place the observed interaction group on a valid simulation support.

    Every active hand/object frame receives the same z correction.  This is a
    coordinate-frame projection, not a grasp initializer: it cannot alter a
    hand--object displacement or synthesize contact.
    """
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
    hand_minimum = np.minimum(
        adjusted["T_sim_wrist"][:, :, 2, 3].min(axis=1),
        adjusted["fingertips_sim"][:, :, :, 2].min(axis=(1, 2)),
    )
    # A single global support shift can expose a later hand target below the
    # floor. Preserve hand-object geometry with a per-frame shared correction.
    frame_correction = np.maximum.reduce([
        np.zeros(len(minimum_after)),
        object_clearance_m - minimum_after,
        hand_target_clearance_m - hand_minimum,
    ])
    if float(frame_correction.max()) > max_frame_correction_m:
        raise ValueError(
            "SPIDER export blocked: hand targets lack xHand floor clearance; "
            f"required shared correction={float(frame_correction.max()):.6f} m, "
            f"maximum={max_frame_correction_m:.6f} m"
        )
    adjusted["T_sim_object"][:, :, 2, 3] += frame_correction[:, None]
    adjusted["T_sim_wrist"][:, :, 2, 3] += frame_correction[:, None]
    adjusted["fingertips_sim"][:, :, :, 2] += frame_correction[:, None, None]
    minimum_after = _object_minimum_z(visual_mesh, adjusted["T_sim_object"][:, 0])
    relative_wrist_error = np.max(np.abs(
        (
            adjusted["T_sim_wrist"][:, :, :3, 3]
            - adjusted["T_sim_object"][:, :, :3, 3]
        )
        - (
            aligned["T_sim_wrist"][:, :, :3, 3]
            - aligned["T_sim_object"][:, :, :3, 3]
        )
    ))
    relative_tip_error = np.max(np.abs(
        (
            adjusted["fingertips_sim"]
            - adjusted["T_sim_object"][:, :, None, :3, 3]
        )
        - (
            aligned["fingertips_sim"]
            - aligned["T_sim_object"][:, :, None, :3, 3]
        )
    ))
    relative_error = float(max(relative_wrist_error, relative_tip_error))
    if relative_error > 1e-9:
        raise RuntimeError(
            "support alignment changed an observed hand-object displacement: "
            f"maximum error={relative_error:.3e} m"
        )
    return adjusted, {
        "support_frame_count": support_frame_count,
        "first_contact_frame": first_contact,
        "object_min_z_support_before_m": support_minimum,
        "global_z_shift_m": float(shift),
        "per_frame_floor_correction_max_m": float(frame_correction.max()),
        "per_frame_floor_correction_frame_count": int(
            np.count_nonzero(frame_correction > floor_tolerance_m)
        ),
        "object_min_z_after_m": float(minimum_after.min()),
        "object_clearance_m": object_clearance_m,
        "hand_target_clearance_m": hand_target_clearance_m,
        "max_frame_correction_m": max_frame_correction_m,
        "hand_object_relative_displacement_max_error_m": relative_error,
        "hand_object_interaction_preserved": True,
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
    normalize_xhand_morphology: bool = False,
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
    aligned, contact, valid_window = _crop_to_longest_valid_window(aligned, contact)
    source_t = aligned["timestamps_s"].astype(np.float64)
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
    # Prevent a sub-ULP arange overshoot from being interpreted as SLERP
    # extrapolation at an otherwise exact endpoint.
    target_t = np.clip(target_t, source_t[0], source_t[-1])
    object_transform = resample_transforms(source_t, aligned["T_sim_object"][:, 0], target_t)
    count = target_t.size
    output: dict[str, np.ndarray] = {
        "qpos_obj_right": _pose7(object_transform),
        "qpos_obj_left": _identity_pose(count),
        "fingertip_orientation_source": np.asarray(
            str(aligned.get("fingertip_orientation_source", "unknown"))
        ),
    }
    if "human_neutral_fingertip_vectors" in aligned:
        neutral = np.asarray(aligned["human_neutral_fingertip_vectors"])
        if neutral.shape != (len(hand_sides), 5, 3):
            raise ValueError(
                "human_neutral_fingertip_vectors must have shape (H, 5, 3)"
            )
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
            output[f"human_palm_orientation_{side}"] = wrist[:, :3, :3].astype(
                np.float32
            )
            if "human_neutral_fingertip_vectors" in aligned:
                output[f"human_neutral_fingertip_vectors_{side}"] = neutral[
                    hand_index
                ].astype(np.float32)
            output[f"mano_pose_{side}"] = _resample_rotations(
                source_t, aligned["mano_pose"][:, hand_index], target_t,
            ).astype(np.float32)
            if "fingertip_orientation_sim" not in aligned:
                raise ValueError(
                    "MINK export requires fingertip_orientation_sim; rerun sequence "
                    "optimization with the landmark-derived distal-frame implementation"
                )
            output[f"fingertip_orientation_{side}"] = _resample_rotations(
                source_t, aligned["fingertip_orientation_sim"][:, hand_index], target_t,
            ).astype(np.float32)
            output[f"contact_{side}"] = _resample_binary(source_t, contact["contact"][:, hand_index], target_t).astype(np.float32)
            output[f"contact_pos_{side}"] = contact["contact_pos_object_local"][hand_index].astype(np.float32)
        else:
            output[f"qpos_wrist_{side}"] = _identity_pose(count)
            output[f"qpos_finger_{side}"] = np.repeat(_identity_pose(count)[:, None], 5, axis=1)
            output[f"human_palm_orientation_{side}"] = np.repeat(
                np.eye(3, dtype=np.float32)[None], count, axis=0,
            )
            output[f"mano_pose_{side}"] = np.repeat(
                np.eye(3, dtype=np.float32)[None, None], count * 15, axis=0,
            ).reshape(count, 15, 3, 3)
            output[f"fingertip_orientation_{side}"] = np.repeat(
                np.eye(3, dtype=np.float32)[None, None], count * 5, axis=0,
            ).reshape(count, 5, 3, 3)
            output[f"contact_{side}"] = np.zeros((count, 5), dtype=np.float32)
            output[f"contact_pos_{side}"] = np.zeros((5, 3), dtype=np.float32)
    root = Path(dataset_root).resolve()
    mano_dir = root / "processed" / DATASET_NAME / "mano" / embodiment_type / task / str(data_id)
    mano_dir.mkdir(parents=True, exist_ok=True)
    keypoints_path = mano_dir / "trajectory_keypoints.npz"
    np.savez_compressed(keypoints_path, **output)
    morphology_normalization: dict[str, object] = {
        "applied": False, "status": "disabled",
    }
    if normalize_xhand_morphology:
        if robot_type != "xhand" or embodiment_type != "right":
            raise ValueError(
                "palm morphology normalization currently supports xhand/right only"
            )
        if "human_palm_orientation_right" not in output:
            raise ValueError(
                "palm morphology normalization requires calibrated source palm "
                "orientation and neutral fingertip vectors; the current aligned "
                "artifact does not provide them"
            )
        from .xhand_targets import normalize_exported_keypoints
        morphology_normalization = normalize_exported_keypoints(keypoints_path)
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
        "fingertip_orientation_source": (
            str(aligned.get("fingertip_orientation_source", "unknown"))
        ),
        "hand_roles": resolved_roles, "simulation_preflight": preflight,
        "support_alignment": support_alignment,
        "morphology_normalization": morphology_normalization,
        "source_valid_window": valid_window,
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
        "mano_pose_exported_for_mink": True,
        "hand_roles": resolved_roles, "simulation_preflight": preflight,
        "support_alignment": support_alignment,
        "morphology_normalization": morphology_normalization,
        "source_valid_window": valid_window,
    }
    export_manifest_path = mano_dir / "export_manifest.json"
    export_manifest_path.write_text(json.dumps(export_manifest, indent=2) + "\n", encoding="utf-8")
    return {"keypoints": keypoints_path, "task_info": task_info_path, "export_manifest": export_manifest_path, "object_dir": object_dir}
