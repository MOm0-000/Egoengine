"""Recover complete OakInk-v2 MANO joints as immutable sidecar inputs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..artifacts import artifact_record
from .input import DISTAL_JOINT_INDICES, FINGERTIP_JOINT_INDICES


MANO_TO_OPENPOSE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)


def _raw_mano_window(annotation: dict[str, Any], task_info: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch

    object_trajectory = annotation["obj_transf"][task_info["oakink_obj_id"]]
    poses, betas, translations = [], [], []
    start, end = (int(value) for value in task_info["oakink_window"])
    for frame in range(start, end):
        if frame not in annotation["raw_mano"] or frame not in object_trajectory:
            continue
        mano = annotation["raw_mano"][frame]
        poses.append(mano["rh__pose_coeffs"])
        betas.append(mano["rh__betas"])
        translations.append(mano["rh__tsl"])
    if not poses:
        raise ValueError("OakInk-v2 window contains no complete MANO frames")
    count = int(task_info["n_frames"])
    source_count = sum(value.shape[0] for value in poses)
    indices = np.linspace(0, source_count - 1, num=min(count, source_count)).round().astype(int)
    return (
        torch.cat(poses, dim=0)[indices],
        torch.cat(betas, dim=0)[indices],
        torch.cat(translations, dim=0)[indices],
    )


def _mano_openpose_joints(pose_coeffs: Any, betas: Any, translations: Any, model_dir: Path) -> np.ndarray:
    import torch
    from smplx import MANOLayer
    from smplx.vertex_ids import vertex_ids

    quaternions = np.asarray(pose_coeffs.cpu(), dtype=np.float64)
    xyzw = np.concatenate([quaternions[..., 1:], quaternions[..., :1]], axis=-1)
    rotations = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(
        len(quaternions), 16, 3, 3,
    )
    layer = MANOLayer(
        model_path=str(model_dir), is_rhand=True, use_pca=False, flat_hand_mean=True,
    )
    with torch.no_grad():
        output = layer(
            global_orient=torch.as_tensor(rotations[:, :1], dtype=torch.float32),
            hand_pose=torch.as_tensor(rotations[:, 1:], dtype=torch.float32),
            betas=betas.float(), pose2rot=False,
        )
    fingertip_vertices = torch.index_select(
        output.vertices, 1,
        torch.as_tensor(list(vertex_ids["mano"].values()), dtype=torch.long),
    )
    joints = torch.cat([output.joints, fingertip_vertices], dim=1)
    joints = joints[:, torch.as_tensor(MANO_TO_OPENPOSE, dtype=torch.long)]
    joints = joints - joints[:, :1] + translations[:, None]
    return joints.cpu().numpy().astype(np.float64)


def recover_oakinkv2_mano21(
    task_info_path: str | Path, annotation_path: str | Path,
    keypoints_path: str | Path, mano_model_dir: str | Path, output_path: str | Path,
) -> Path:
    """Run MANO FK and align all 21 joints to the frozen SPIDER coordinates."""
    task_info_source = Path(task_info_path).resolve()
    annotation_source = Path(annotation_path).resolve()
    keypoints_source = Path(keypoints_path).resolve()
    model_dir = Path(mano_model_dir).resolve()
    if not (model_dir / "MANO_RIGHT.pkl").is_file():
        raise FileNotFoundError(model_dir / "MANO_RIGHT.pkl")
    task_info = json.loads(task_info_source.read_text(encoding="utf-8"))
    if annotation_source.stem != str(task_info["oakink_seq_token"]):
        raise ValueError("OakInk-v2 annotation does not match task_info sequence token")
    with annotation_source.open("rb") as stream:
        annotation = pickle.load(stream)
    pose_coeffs, betas, translations = _raw_mano_window(annotation, task_info)
    joints_world = _mano_openpose_joints(pose_coeffs, betas, translations, model_dir)
    world_to_sim = Rotation.from_euler("xyz", [np.pi / 2, 0.0, 0.0])
    joints_sim = world_to_sim.apply(joints_world.reshape(-1, 3)).reshape(joints_world.shape)
    joints_sim[..., 2] -= float(task_info["trajectory_z_offset"])
    with np.load(keypoints_source, allow_pickle=False) as artifact:
        wrist = np.asarray(artifact["qpos_wrist_right"][..., :3], dtype=np.float64)
        fingertips = np.asarray(artifact["qpos_finger_right"][..., :3], dtype=np.float64)
    if joints_sim.shape != (len(wrist), 21, 3):
        raise ValueError("recovered MANO timeline does not match frozen keypoints")
    joints_sim += (wrist - joints_sim[:, 0])[:, None]
    recovered_tips = joints_sim[:, list(FINGERTIP_JOINT_INDICES)]
    tip_error = np.linalg.norm(recovered_tips - fingertips, axis=-1)
    directions = recovered_tips - joints_sim[:, list(DISTAL_JOINT_INDICES)]
    lengths = np.linalg.norm(directions, axis=-1)
    valid = np.isfinite(directions).all(axis=-1) & (lengths > 1e-8)
    directions[valid] /= lengths[valid][:, None]
    directions[~valid] = 0.0
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        frame_indices=np.arange(len(joints_sim), dtype=np.int64),
        timestamps_s=np.arange(len(joints_sim), dtype=np.float64) * float(task_info["ref_dt"]),
        joint_positions_sim=joints_sim,
        fingertip_positions_sim=recovered_tips,
        distal_directions_sim=directions,
        valid_distal_direction=valid,
        fingertip_indices=np.asarray(FINGERTIP_JOINT_INDICES, dtype=np.int64),
        distal_joint_indices=np.asarray(DISTAL_JOINT_INDICES, dtype=np.int64),
        source_coordinate_system=np.asarray("spider_oakinkv2_sim"),
    )
    report = {
        "schema_version": "1.0", "frame_count": len(joints_sim),
        "valid_direction_rate": float(valid.mean()),
        "fingertip_alignment_error_mean_m": float(tip_error.mean()),
        "fingertip_alignment_error_max_m": float(tip_error.max()),
        "distal_bone_length_mean_m": float(lengths[valid].mean()),
        "inputs": {
            "task_info": artifact_record(task_info_source),
            "annotation": artifact_record(annotation_source),
            "keypoints": artifact_record(keypoints_source),
            "mano_model": artifact_record(model_dir / "MANO_RIGHT.pkl"),
        },
        "output": artifact_record(destination),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return destination
