#!/usr/bin/env python3
"""Build a finite bimanual reference artifact for scene/interface validation.

The right-hand trajectory is the existing TACO xHand kinematic trajectory.
The left hand is a translated copy, not left-hand GT or a kinematic mirror.
This legacy fixture is superseded by retarget_taco_bimanual_gt.py. Object poses are TACO GT
poses transformed by the rigid alignment implied by the existing right-hand
artifact's first frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RIGHT = ROOT / "models/taco_xhand/xhand/right/taco_brush_brush_bowl_20230927_027/0/trajectory_kinematic.npz"
DEFAULT_RIGHT_GT = ROOT / "data/taco_v1/dev4/object_poses/Object_Poses/(brush, brush, bowl)/20230927_027/tool_071.npy"
DEFAULT_LEFT_GT = ROOT / "data/taco_v1/dev4/object_poses/Object_Poses/(brush, brush, bowl)/20230927_027/target_146.npy"
DEFAULT_OUT = ROOT / "models/taco_xhand/xhand/bimanual/taco_brush_brush_bowl_20230927_027/0/trajectory_kinematic_provisional.npz"
DEFAULT_META = ROOT / "models/taco_xhand/xhand/bimanual/taco_brush_brush_bowl_20230927_027/0/reference_metadata.json"


def _sim_alignment(right_qpos: np.ndarray, right_gt: np.ndarray) -> np.ndarray:
    """Return T_sim<-gt from the existing unilateral right-object pose."""
    q = right_qpos[-7:]
    sim = np.eye(4)
    sim[:3, :3] = Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix()
    sim[:3, 3] = q[:3]
    return sim @ np.linalg.inv(right_gt[0])


def _to_axis_angle(poses: np.ndarray) -> np.ndarray:
    result = np.zeros((len(poses), 6), dtype=np.float32)
    result[:, :3] = poses[:, :3, 3]
    result[:, 3:] = Rotation.from_matrix(poses[:, :3, :3]).as_rotvec().astype(np.float32)
    return result


def _to_pose7(poses: np.ndarray) -> np.ndarray:
    result = np.zeros((len(poses), 7), dtype=np.float32)
    result[:, :3] = poses[:, :3, 3]
    quat_xyzw = Rotation.from_matrix(poses[:, :3, :3]).as_quat().astype(np.float32)
    result[:, 3:] = quat_xyzw[:, [3, 0, 1, 2]]
    return result


def build(right_path: Path, right_gt_path: Path, left_gt_path: Path, out: Path, meta: Path) -> None:
    right = np.load(right_path)
    qpos_right = np.asarray(right["qpos"], dtype=np.float32)
    qvel_right = np.asarray(right["qvel"], dtype=np.float32)
    if qpos_right.shape[1] != 25 or qvel_right.shape[1] != 24:
        raise ValueError(f"expected unilateral (25,24) state, got {qpos_right.shape}, {qvel_right.shape}")
    tool_gt = np.asarray(np.load(right_gt_path), dtype=np.float32)
    target_gt = np.asarray(np.load(left_gt_path), dtype=np.float32)
    n = min(len(qpos_right), len(tool_gt), len(target_gt))
    qpos_right, qvel_right, tool_gt, target_gt = qpos_right[:n], qvel_right[:n], tool_gt[:n], target_gt[:n]

    alignment = _sim_alignment(qpos_right[0], tool_gt)
    tool_sim = np.einsum("ij,njk->nik", alignment, tool_gt)
    target_sim = np.einsum("ij,njk->nik", alignment, target_gt)
    tool7 = _to_pose7(tool_sim)
    target7 = _to_pose7(target_sim)

    # qpos layout: right hand (18), left hand (18), right free object (7), left free object (7).
    qpos = np.zeros((n, 50), dtype=np.float32)
    qpos[:, :18] = qpos_right[:, :18]
    qpos[:, 18:36] = qpos_right[:, :18]
    qpos[:, 18] -= 0.25  # separate the provisional left wrist from the right wrist
    qpos[:, 36:43] = tool7
    qpos[:, 43:50] = target7

    qvel = np.zeros((n, 48), dtype=np.float32)
    qvel[:, :18] = qvel_right[:, :18]
    qvel[:, 18:36] = qvel_right[:, :18]
    dt = 1.0 / float(right.get("frequency", 30.0))
    scene = ROOT / "models/taco_xhand/xhand/bimanual/taco_brush_brush_bowl_20230927_027/scene_act.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    for index in range(n):
        first, last = max(0, index - 1), min(n - 1, index + 1)
        velocity = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, velocity, (last - first) * dt,
                                 qpos[first].astype(np.float64), qpos[last].astype(np.float64))
        qvel[index] = velocity

    # Only the two hands are actuated; objects are passive free bodies.
    ctrl = np.zeros((n, 36), dtype=np.float32)
    ctrl[:, :36] = qpos[:, :36]
    np.savez_compressed(out, qpos=qpos, qvel=qvel, ctrl=ctrl, frequency=np.asarray(30.0))
    payload = {
        "schema_version": "0.1-provisional-bimanual",
        "status": "interface_smoke_only",
        "scene": "taco_brush_brush_bowl_20230927_027",
        "right_hand_source": str(right_path),
        "right_object_gt": str(right_gt_path),
        "left_object_gt": str(left_gt_path),
        "left_hand_source": "translated right-hand placeholder; not retargeted left-hand GT",
        "source_frame_alignment_verified": False,
        "eligible_for_rl_validation": False,
        "coordinate_alignment": alignment.tolist(),
        "layout": {"right_hand": [0, 18], "left_hand": [18, 36], "right_object_free_qpos": [36, 43], "left_object_free_qpos": [43, 50]},
        "finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(ctrl).all()),
        "frames": int(n),
        "dt_s": dt,
    }
    meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-fixture", action="store_true")
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--right-gt", type=Path, default=DEFAULT_RIGHT_GT)
    parser.add_argument("--left-gt", type=Path, default=DEFAULT_LEFT_GT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    args = parser.parse_args()
    if not args.legacy_fixture:
        parser.error("legacy placeholder generator disabled; run retarget_taco_bimanual_gt.py for real two-hand GT")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    build(args.right, args.right_gt, args.left_gt, args.out, args.meta)
    print(args.out)


if __name__ == "__main__":
    main()
