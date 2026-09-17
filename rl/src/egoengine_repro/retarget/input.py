"""Adapters from frozen pipeline artifacts to the MINK input protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .schema import validate_human_reference


FINGERTIP_JOINT_INDICES = (4, 8, 12, 16, 20)
DISTAL_JOINT_INDICES = (3, 7, 11, 15, 19)


def _pose7_to_transform(pose: np.ndarray) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 7:
        raise ValueError("SPIDER poses must have shape (...,7)")
    result = np.broadcast_to(np.eye(4), values.shape[:-1] + (4, 4)).copy()
    result[..., :3, 3] = values[..., :3]
    quaternions = values[..., [4, 5, 6, 3]].reshape(-1, 4)
    result[..., :3, :3] = Rotation.from_quat(quaternions).as_matrix().reshape(
        values.shape[:-1] + (3, 3),
    )
    return result


def _apply_finger_direction_frames(
    tips: np.ndarray, joint_positions: np.ndarray, wrist_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tips = np.asarray(tips, dtype=np.float64).copy()
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    wrist_rotation = np.asarray(wrist_rotation, dtype=np.float64)
    if joint_positions.shape != tips.shape[:2] + (21, 3):
        raise ValueError("MANO joint positions must have shape (T,H,21,3)")
    if wrist_rotation.shape != tips.shape[:2] + (3, 3):
        raise ValueError("wrist rotations do not match the fingertip targets")
    direction_valid = np.zeros(tips.shape[:3], dtype=bool)
    directions = np.zeros(tips.shape[:3] + (3,), dtype=np.float64)
    for frame in range(tips.shape[0]):
        for hand in range(tips.shape[1]):
            for finger, (distal_index, tip_index) in enumerate(zip(
                DISTAL_JOINT_INDICES, FINGERTIP_JOINT_INDICES,
            )):
                direction = (
                    joint_positions[frame, hand, tip_index]
                    - joint_positions[frame, hand, distal_index]
                )
                norm = float(np.linalg.norm(direction))
                if not np.isfinite(norm) or norm < 1e-8:
                    tips[frame, hand, finger, :3, :3] = wrist_rotation[frame, hand]
                    continue
                direction /= norm
                directions[frame, hand, finger] = direction
                if finger == 0:
                    x_axis = direction
                    guide = wrist_rotation[frame, hand, :, 2]
                    z_axis = guide - x_axis * float(np.dot(guide, x_axis))
                    if np.linalg.norm(z_axis) < 1e-8:
                        guide = wrist_rotation[frame, hand, :, 1]
                        z_axis = guide - x_axis * float(np.dot(guide, x_axis))
                    z_axis /= np.linalg.norm(z_axis)
                else:
                    z_axis = -direction
                    guide = wrist_rotation[frame, hand, :, 0]
                    x_axis = guide - z_axis * float(np.dot(guide, z_axis))
                    if np.linalg.norm(x_axis) < 1e-8:
                        guide = wrist_rotation[frame, hand, :, 1]
                        x_axis = guide - z_axis * float(np.dot(guide, z_axis))
                    x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                tips[frame, hand, finger, :3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
                direction_valid[frame, hand, finger] = True
    return tips, direction_valid, directions


def _finger_direction_frames(sim_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tips = sim_joints[..., list(FINGERTIP_JOINT_INDICES), :, :].copy()
    tips, direction_valid, _ = _apply_finger_direction_frames(
        tips, sim_joints[..., :3, 3], sim_joints[..., 0, :3, :3],
    )
    return tips, direction_valid


def _load_mano21_joints(
    path: str | Path, *, frame_indices: np.ndarray, count: int,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        if "joint_positions_sim" not in artifact:
            raise ValueError("MANO21 artifact misses joint_positions_sim")
        joints = np.asarray(artifact["joint_positions_sim"], dtype=np.float64)
        source_frames = np.asarray(artifact["frame_indices"], dtype=np.int64)
    if joints.shape == (count, 21, 3):
        joints = joints[:, None]
    if joints.shape != (count, 1, 21, 3):
        raise ValueError("MANO21 joint_positions_sim must have shape (T,21,3) or (T,1,21,3)")
    if not np.array_equal(source_frames, frame_indices):
        raise ValueError("MANO21 and SPIDER keypoint frame indices do not match")
    if not np.isfinite(joints).all():
        raise ValueError("MANO21 joint positions contain non-finite values")
    return joints


def _load_transform(path: str | Path) -> np.ndarray:
    source = Path(path).resolve()
    if source.suffix == ".json":
        value = np.asarray(json.loads(source.read_text(encoding="utf-8"))["T_sim_world"], dtype=np.float64)
    elif source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as artifact:
            value = np.asarray(artifact["T_sim_world"], dtype=np.float64)
    else:
        value = np.asarray(np.load(source, allow_pickle=False), dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("T_sim_world must be a finite (4,4) transform")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("T_sim_world has an invalid homogeneous row")
    return value


def _aligned_object_reference(path: str | Path, frame_indices: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        object_key = "T_sim_object_reference" if "T_sim_object_reference" in artifact else "T_sim_object"
        objects = np.asarray(artifact[object_key], dtype=np.float64)
        object_frames = np.asarray(artifact["frame_indices"], dtype=np.int64)
    lookup = {int(frame): index for index, frame in enumerate(object_frames)}
    missing = [int(frame) for frame in frame_indices if int(frame) not in lookup]
    if missing:
        raise ValueError(f"object reference misses oracle frames: {missing[:5]}")
    return objects[[lookup[int(frame)] for frame in frame_indices]]


def human_reference_from_aligned(run_dir: str | Path, output_path: str | Path) -> Path:
    """Convert current aligned artifacts without altering or overwriting them.

    Current artifacts contain fingertip positions but no fingertip orientations.
    The adapter records identity orientation with a false orientation-valid mask,
    ensuring MINK applies position constraints only for those targets.
    """
    run = Path(run_dir).resolve()
    output = Path(output_path).resolve()
    if run == output.parent or run in output.parents:
        raise ValueError("retarget input must be written outside the source run")
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(run / "optimization/aligned_trajectory.npz", allow_pickle=False) as artifact:
        aligned = {key: np.asarray(artifact[key]) for key in artifact.files}
    metrics = json.loads((run / "optimization/optimization_metrics.json").read_text(encoding="utf-8"))
    hand_order = np.asarray(metrics["hands"]["artifact_hand_order"])
    t, h = aligned["fingertips_sim"].shape[:2]
    fingertip_target = np.repeat(np.eye(4)[None, None, None], t * h * 5, axis=0).reshape(t, h, 5, 4, 4)
    fingertip_target[..., :3, 3] = aligned["fingertips_sim"]
    arrays = {
        "frame_indices": aligned["frame_indices"], "timestamps_s": aligned["timestamps_s"],
        "hand_order": hand_order,
        "T_sim_wrist_target": aligned["T_sim_wrist"],
        "T_sim_fingertip_target": fingertip_target,
        "valid_hand": aligned["valid_hand"].astype(bool),
        "valid_fingertip_orientation": np.zeros((t, h, 5), dtype=bool),
        "confidence_hand": aligned["confidence_hand"],
        "confidence_fingertip": np.repeat(aligned["confidence_hand"][..., None], 5, axis=2),
        "T_sim_object_reference": aligned["T_sim_object"],
        "object_side": np.asarray("right"),
        "orientation_source": np.asarray("unavailable_in_auto_current"),
    }
    validate_human_reference(arrays)
    np.savez_compressed(output, **arrays)
    return output


def human_reference_from_spider_keypoints(
    keypoints_path: str | Path, output_path: str | Path, *, side: str = "right",
    ref_dt: float = 0.02, mano21_path: str | Path | None = None,
) -> Path:
    """Convert frozen SPIDER MANO keypoints to the common MINK input protocol."""
    if side not in {"left", "right", "bimanual"}:
        raise ValueError("side must be left, right, or bimanual")
    if ref_dt <= 0:
        raise ValueError("ref_dt must be positive")
    source = Path(keypoints_path).resolve()
    with np.load(source, allow_pickle=False) as artifact:
        keypoints = {key: np.asarray(artifact[key]) for key in artifact.files}
    hand_order = ("left", "right") if side == "bimanual" else (side,)
    def _active_object_side() -> str:
        sides = [side for side in ("left", "right") if f"qpos_obj_{side}" in keypoints]
        for candidate in sides:
            pose = np.asarray(keypoints[f"qpos_obj_{candidate}"], dtype=np.float64)
            if pose.size == 0:
                continue
            position_zero = np.allclose(pose[:, :3], 0.0, atol=1e-12)
            quaternion = pose[:, 3:]
            quaternion_degenerate = np.allclose(
                np.linalg.norm(quaternion, axis=-1), 0.0, atol=1e-12,
            )
            if not (position_zero and quaternion_degenerate):
                return candidate
        if "right" in sides:
            return "right"
        if "left" in sides:
            return "left"
        return hand_order[-1]

    object_side = _active_object_side()
    required = {
        *(f"qpos_wrist_{hand}" for hand in hand_order),
        *(f"qpos_finger_{hand}" for hand in hand_order),
        f"qpos_obj_{object_side}",
    }
    missing = required - set(keypoints)
    if missing:
        raise ValueError(f"SPIDER keypoints miss arrays: {sorted(missing)}")
    wrist = np.stack([
        _pose7_to_transform(keypoints[f"qpos_wrist_{hand}"]) for hand in hand_order
    ], axis=1)
    fingers = np.stack([
        _pose7_to_transform(keypoints[f"qpos_finger_{hand}"]) for hand in hand_order
    ], axis=1)
    objects = _pose7_to_transform(keypoints[f"qpos_obj_{object_side}"])
    count, hand_count = wrist.shape[:2]
    if wrist.shape != (count, hand_count, 4, 4) or fingers.shape != (count, hand_count, 5, 4, 4):
        raise ValueError("SPIDER wrist/finger poses have unexpected shapes")
    if objects.shape != (count, 4, 4):
        raise ValueError("SPIDER object poses have an unexpected shape")
    frame_indices = np.arange(count, dtype=np.int64)
    mano_joints = None
    fingertip_directions = None
    if mano21_path is not None:
        if hand_count != 1:
            raise ValueError("MANO21 sidecar input currently supports one hand per artifact")
        mano_joints = _load_mano21_joints(
            mano21_path, frame_indices=frame_indices, count=count,
        )
        fingers_with_direction, orientation_valid, fingertip_directions = (
            _apply_finger_direction_frames(
                fingers, mano_joints, wrist[..., :3, :3],
            )
        )
        fingers = fingers_with_direction
        orientation_source = "mano_21_joint_distal_bone_direction"
    else:
        finger_quaternions = np.stack([
            np.asarray(keypoints[f"qpos_finger_{hand}"][..., 3:], dtype=np.float64)
            for hand in hand_order
        ], axis=1)
        identity_quaternion = np.zeros_like(finger_quaternions)
        identity_quaternion[..., 0] = 1.0
        orientation_valid = ~np.all(
            np.isclose(finger_quaternions, identity_quaternion, atol=1e-8), axis=-1,
        )
        orientation_source = "spider_keypoint_pose_or_unavailable_identity_quaternion"
    arrays = {
        "frame_indices": frame_indices,
        "timestamps_s": np.arange(count, dtype=np.float64) * float(ref_dt),
        "hand_order": np.asarray(hand_order),
        "T_sim_wrist_target": wrist,
        "T_sim_fingertip_target": fingers,
        "valid_hand": np.ones((count, hand_count), dtype=bool),
        "valid_fingertip_orientation": orientation_valid,
        "confidence_hand": np.ones((count, hand_count), dtype=np.float64),
        "confidence_fingertip": np.ones((count, hand_count, 5), dtype=np.float64),
        "T_sim_object_reference": objects[:, None],
        "object_side": np.asarray(object_side),
        "hand_source": np.asarray("spider_mano_keypoints"),
        "orientation_source": np.asarray(orientation_source),
        "source_keypoints": np.asarray(str(source)),
    }
    if mano_joints is not None and fingertip_directions is not None:
        arrays["mano_joint_positions_sim"] = mano_joints
        arrays["fingertip_direction_sim"] = fingertip_directions
        arrays["source_mano21"] = np.asarray(str(Path(mano21_path).resolve()))
        # Palm target: centroid of the five MCP joints (Thumb/Index/Middle/Ring/Little
        # MCP), oriented with the wrist frame.  The palm is a different point from the
        # wrist (~5-6 cm distal along the hand) and is otherwise unconstrained.
        mcp_indices = (1, 5, 9, 13, 17)
        palm_position = mano_joints[:, 0, mcp_indices, :].mean(axis=1)  # (T, 3)
        palm_targets = np.repeat(
            np.eye(4)[None, None], count * hand_count, axis=0,
        ).reshape(count, hand_count, 4, 4)
        palm_targets[:, 0, :3, 3] = palm_position
        palm_targets[:, 0, :3, :3] = wrist[:, 0, :3, :3]
        arrays["T_sim_palm_target"] = palm_targets
    validate_human_reference(arrays)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return output


def human_reference_from_oracle_hand(
    hand_ground_truth_path: str | Path, T_sim_world_path: str | Path, output_path: str | Path,
    *, hand_order: Sequence[str] | None = None, object_reference_path: str | Path | None = None,
    object_side: str = "right", confidence_threshold: float = 0.0,
) -> Path:
    """Build an isolated paper-profile MINK input from oracle hand poses."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(hand_ground_truth_path, allow_pickle=False) as artifact:
        gt = {key: np.asarray(artifact[key]) for key in artifact.files}
    required = {"frame_indices", "timestamps_s", "hand_order", "T_world_joint", "confidence"}
    missing = required - set(gt)
    if missing:
        raise ValueError(f"oracle hand artifact misses arrays: {sorted(missing)}")
    source_order = [str(side) for side in gt["hand_order"]]
    selected_order = list(hand_order) if hand_order is not None else source_order
    if not selected_order or set(selected_order) - set(source_order):
        raise ValueError("requested hand_order is empty or unavailable in oracle artifact")
    selected_indices = [source_order.index(side) for side in selected_order]
    world_joints = np.asarray(gt["T_world_joint"], dtype=np.float64)[:, selected_indices]
    confidence = np.asarray(gt["confidence"], dtype=np.float64)[:, selected_indices]
    if world_joints.shape[2:] != (21, 4, 4) or confidence.shape != world_joints.shape[:3]:
        raise ValueError("oracle hand transforms/confidence must have shape (T,H,21,...)")
    T_sim_world = _load_transform(T_sim_world_path)
    sim_joints = np.einsum("ij,thnjk->thnik", T_sim_world, world_joints)
    fingertip_targets, direction_valid = _finger_direction_frames(sim_joints)
    wrist_confidence = confidence[..., 0]
    fingertip_confidence = confidence[..., list(FINGERTIP_JOINT_INDICES)]
    valid_hand = np.isfinite(sim_joints[..., 0, :, :]).all(axis=(-1, -2)) & (
        wrist_confidence > float(confidence_threshold)
    )
    valid_fingertip = np.isfinite(
        sim_joints[..., list(FINGERTIP_JOINT_INDICES), :, :]
    ).all(axis=(-1, -2)) & (fingertip_confidence > float(confidence_threshold))
    arrays = {
        "frame_indices": np.asarray(gt["frame_indices"], dtype=np.int64),
        "timestamps_s": np.asarray(gt["timestamps_s"], dtype=np.float64),
        "hand_order": np.asarray(selected_order),
        "T_sim_wrist_target": sim_joints[..., 0, :, :],
        "T_sim_fingertip_target": fingertip_targets,
        "valid_hand": valid_hand,
        "valid_fingertip_orientation": valid_fingertip & valid_hand[..., None] & direction_valid,
        "confidence_hand": wrist_confidence,
        "confidence_fingertip": fingertip_confidence,
        "object_side": np.asarray(object_side),
        "hand_source": np.asarray("oracle_hand"),
        "orientation_source": np.asarray("egodex_oracle_distal_bone_direction"),
        "T_sim_world": T_sim_world,
        "coordinate_alignment_source": np.asarray(str(Path(T_sim_world_path).resolve())),
    }
    if object_reference_path is not None:
        arrays["T_sim_object_reference"] = _aligned_object_reference(
            object_reference_path, arrays["frame_indices"],
        )
    validate_human_reference(arrays)
    np.savez_compressed(output, **arrays)
    return output
