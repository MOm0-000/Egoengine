#!/usr/bin/env python3
"""Generate BODex XHand grasp candidates outside the formal control pipeline.

This program deliberately has no dependency on the formal renderer/controller.
It copies one object mesh into a disposable BODex asset package, asks BODex for
right-hand poses using the recorded SAPIEN object pose, and writes a
self-describing diagnostic NPZ in the source-object frame.
The result is a proposal set only: it is not a simulated grasp success and it
is never eligible for the formal renderer/3.3 chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "xhand_bodex_grasp_candidates_v2_diagnostic_only"
STAGES = ("pregrasp", "grasp", "squeeze")
CANONICAL_JOINTS = (
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
)
SAPIEN_JOINTS = (
    "right_hand_thumb_bend_joint",
    "right_hand_index_bend_joint",
    "right_hand_mid_joint1",
    "right_hand_ring_joint1",
    "right_hand_pinky_joint1",
    "right_hand_thumb_rota_joint1",
    "right_hand_index_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint2",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_joint2",
)
SAPIEN_TO_CANONICAL = np.asarray(
    [SAPIEN_JOINTS.index(name) for name in CANONICAL_JOINTS], dtype=np.int64
)


def apply_deximit_rollout_contract(
    wrist_pose: np.ndarray, qpos_sapien: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply DexImit's right-hand preprocessing before physical rollout.

    ``GraspSynthesizer`` returns geometric seeds.  DexImit's executor does not
    send those arrays directly to the robot: it moves pregrasp 10 cm away from
    grasp, copies grasp orientation to pregrasp, and relaxes Sapien joints 2:7
    at pregrasp and grasp.  Keeping this transformation in the candidate
    artifact prevents downstream evaluators from silently executing raw seeds.
    """
    poses = np.asarray(wrist_pose, dtype=np.float64).copy()
    joints = np.asarray(qpos_sapien, dtype=np.float64).copy()
    if poses.ndim != 3 or poses.shape[1:] != (3, 7):
        raise ValueError("BODex wrist seeds must have shape (N, 3, 7)")
    if joints.shape != (len(poses), 3, len(SAPIEN_JOINTS)):
        raise ValueError("BODex Sapien-order joints do not match wrist seeds")
    direction_hand = -np.asarray((-0.25, 0.25, 1.0), dtype=np.float64)
    direction_hand /= np.linalg.norm(direction_hand)
    grasp_rotation = Rotation.from_quat(
        poses[:, 1, 3:], scalar_first=True,
    )
    # A single vector is deliberately used here.  ``np.broadcast_to`` returns
    # a read-only view which newer SciPy releases reject inside Rotation.apply.
    direction_world = grasp_rotation.apply(direction_hand)
    poses[:, 0, :3] = poses[:, 1, :3] - 0.1 * direction_world
    poses[:, 0, 3:] = poses[:, 1, 3:]
    joints[:, 0:2, 2:7] -= 0.2
    return poses.astype(np.float32), joints.astype(np.float32)


def world_poses_to_object_frame(
    wrist_pose_world: np.ndarray, object_pose_wxyz: np.ndarray,
) -> np.ndarray:
    """Convert BODex world-frame wrist poses to the source-object frame.

    DexImit's upstream generator solves BODex with the SAPIEN object pose and
    returns world-frame wrist poses. The diagnostic artifact stores a local
    pose so the screening runner can relocate the complete hand/object pair to
    its settled pose without applying a second, implicit frame convention.
    """
    poses = np.asarray(wrist_pose_world, dtype=np.float64)
    object_pose = np.asarray(object_pose_wxyz, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (3, 7):
        raise ValueError("BODex wrist poses must have shape (N, 3, 7)")
    if object_pose.shape != (7,) or not np.isfinite(object_pose).all():
        raise ValueError("object pose must contain seven finite values")
    quaternion_norm = np.linalg.norm(object_pose[3:])
    if not 0.5 < quaternion_norm < 1.5:
        raise ValueError("object pose quaternion has an invalid norm")

    object_rotation = Rotation.from_quat(
        object_pose[3:] / quaternion_norm, scalar_first=True,
    )
    world_rotation = Rotation.from_quat(
        poses.reshape(-1, 7)[:, 3:], scalar_first=True,
    )
    local_rotation = object_rotation.inv() * world_rotation
    local = np.empty_like(poses)
    local[..., :3] = object_rotation.inv().apply(
        poses[..., :3].reshape(-1, 3) - object_pose[None, :3]
    ).reshape(poses.shape[0], poses.shape[1], 3)
    local[..., 3:] = local_rotation.as_quat(
        scalar_first=True,
    ).reshape(poses.shape[0], poses.shape[1], 4)
    return local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deximit-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True, help="Meter-scale object mesh.")
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic .npz file.")
    parser.add_argument("--work-dir", type=Path, required=True, help="New disposable BODex asset directory.")
    parser.add_argument("--episode-id", required=True)
    pose_group = parser.add_mutually_exclusive_group(required=True)
    pose_group.add_argument(
        "--object-pose-wxyz", type=float, nargs=7,
        metavar=("X", "Y", "Z", "W", "QX", "QY", "QZ"),
        help=(
            "SAPIEN object pose used by the upstream generator after warmup; "
            "BODex is solved in this pose and the result is stored locally."
        ),
    )
    pose_group.add_argument(
        "--legacy-identity-pose", action="store_true",
        help=(
            "Reproduce the old identity-pose candidate protocol explicitly; "
            "this is for legacy artifacts and is not the upstream-equivalent path."
        ),
    )
    parser.add_argument("--fingers", type=int, choices=(2, 3, 4, 5), default=3)
    parser.add_argument("--grasp-depth", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--num-grasps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def resolved_existing(path: Path, label: str) -> Path:
    value = path.expanduser().resolve(strict=True)
    if not value.is_file() and label == "mesh":
        raise ValueError(f"{label} must be a file: {value}")
    if not value.is_dir() and label == "DexImit root":
        raise ValueError(f"{label} must be a directory: {value}")
    return value


def resolved_new(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if value.exists():
        raise FileExistsError(f"Refusing to overwrite existing {label}: {value}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_worktree_state(root: Path) -> dict[str, object]:
    """Record the source revision and tracked working-tree changes used here."""
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        check=True, capture_output=True,
    ).stdout
    state_hash = hashlib.sha256(status.encode("utf-8") + b"\0" + diff).hexdigest()
    return {
        "deximit_commit": commit,
        "deximit_worktree_dirty": bool(status),
        "deximit_worktree_status": status.splitlines(),
        "deximit_worktree_state_sha256": state_hash,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_object_package(mesh_file: Path, work_dir: Path) -> dict[str, Any]:
    """Make BODex's small object-package contract without modifying the source mesh."""
    import trimesh

    mesh = trimesh.load(mesh_file, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The input must contain exactly one triangular surface mesh.")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("The input mesh is empty.")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("The input mesh has non-finite vertex coordinates.")
    if mesh.faces.shape[1] != 3:
        raise ValueError("The input mesh is not triangular.")

    package = work_dir / "object"
    mesh_dir = package / "mesh"
    info_dir = package / "info"
    urdf_dir = package / "urdf"
    urdf_mesh_dir = urdf_dir / "meshes"
    for directory in (mesh_dir, info_dir, urdf_mesh_dir):
        directory.mkdir(parents=True, exist_ok=False)

    package_mesh = mesh_dir / "simplified.obj"
    mesh.export(package_mesh, file_type="obj")
    shutil.copyfile(package_mesh, urdf_mesh_dir / "simplified.obj")

    # Match DexImit's pipeline/gen_traj.py contract exactly.  BODex optimizes
    # force closure around this declared gravity center, so replacing it with
    # an integrated center of mass changes the generated grasp distribution.
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    mass_center = 0.5 * (bounds[0] + bounds[1])
    obb = bounds[1] - bounds[0]
    if (
        not np.isfinite(mass_center).all() or not np.isfinite(obb).all()
        or np.any(obb <= 0)
    ):
        raise ValueError("mesh bounds cannot define DexImit's BODex object contract")
    info = {
        "gravity_center": np.asarray(mass_center, dtype=float).tolist(),
        "obb": np.asarray(obb, dtype=float).tolist(),
    }
    (info_dir / "simplified.json").write_text(json.dumps(info, indent=2) + "\n")
    robot_name = re.sub(r"[^A-Za-z0-9_]+", "_", mesh_file.stem).strip("_")
    if not robot_name:
        robot_name = "diagnostic_object"
    urdf = f"""<robot name=\"{robot_name}\">
  <link name=\"simplified\">
    <inertial><origin xyz=\"0 0 0\" rpy=\"0 0 0\"/></inertial>
    <visual><origin xyz=\"0 0 0\" rpy=\"0 0 0\"/><geometry>
      <mesh filename=\"meshes/simplified.obj\" scale=\"1 1 1\"/>
    </geometry></visual>
    <collision><origin xyz=\"0 0 0\" rpy=\"0 0 0\"/><geometry>
      <mesh filename=\"meshes/simplified.obj\" scale=\"1 1 1\"/>
    </geometry></collision>
  </link>
</robot>
"""
    (urdf_dir / "coacd.urdf").write_text(urdf)
    return {
        "package": package,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "bounds_m": np.asarray(mesh.bounds, dtype=float),
        "extent_m": np.asarray(mesh.extents, dtype=float),
        "gravity_center_m": np.asarray(mass_center, dtype=float),
        "bbox_extent_m": np.asarray(obb, dtype=float),
        "urdf_robot_name": robot_name,
        "packaged_obj_sha256": sha256(package_mesh),
    }


def install_bodex_import_path(deximit_root: Path) -> None:
    package_root = deximit_root / "third_party" / "any2dex" / "any2dex"
    bodex_root = package_root / "third_party" / "BODex_api"
    if not (bodex_root / "src" / "bodex").is_dir():
        raise FileNotFoundError(f"BODex source is incomplete: {bodex_root}")
    sys.path.insert(0, str(package_root))
    sys.path.insert(0, str(bodex_root / "src"))


def main() -> None:
    args = parse_args()
    if args.num_grasps <= 0:
        raise ValueError("--num-grasps must be positive")
    deximit_root = resolved_existing(args.deximit_root, "DexImit root")
    mesh_file = resolved_existing(args.mesh, "mesh")
    output = resolved_new(args.output, "output")
    work_dir = resolved_new(args.work_dir, "work directory")
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    if output == work_dir or output.is_relative_to(work_dir):
        raise ValueError("Output must be outside the disposable work directory.")
    object_pose_wxyz = np.asarray(
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        if args.legacy_identity_pose else args.object_pose_wxyz,
        dtype=np.float64,
    )
    if object_pose_wxyz.shape != (7,) or not np.isfinite(object_pose_wxyz).all():
        raise ValueError("--object-pose-wxyz must contain seven finite values")
    object_quaternion_norm = np.linalg.norm(object_pose_wxyz[3:])
    if not 0.5 < object_quaternion_norm < 1.5:
        raise ValueError("--object-pose-wxyz quaternion has an invalid norm")
    object_pose_wxyz[3:] /= object_quaternion_norm

    work_dir.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    object_info = write_object_package(mesh_file, work_dir)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("BODex needs CUDA, but this isolated environment cannot see a GPU.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    install_bodex_import_path(deximit_root)
    from util.bodex_util import GraspSynthesizer

    started = datetime.now(timezone.utc)
    generator = GraspSynthesizer(
        hand=1,
        hand_type="xhand",
        dof=12,
        num_grasp=args.num_grasps,
        fingers=args.fingers,
        grasp_depth=args.grasp_depth,
    )
    # Upstream DexImit passes the settled SAPIEN pose into BODex. BODex returns
    # world-frame poses; convert them back to the artifact's object frame below.
    raw = generator.synthesize_grasp(
        str(object_info["package"]), object_pose_wxyz.tolist(), 1.0
    )
    finished = datetime.now(timezone.utc)
    if raw.ndim != 4 or raw.shape[1:] != (1, 3, 19):
        raise RuntimeError(f"Unexpected BODex output shape: {raw.shape}")
    returned = raw[:, 0]
    wrist_pose_world = returned[:, :, :7]
    wrist_pose = world_poses_to_object_frame(wrist_pose_world, object_pose_wxyz)
    qpos_sapien_raw = returned[:, :, 7:]
    qpos_canonical_raw = qpos_sapien_raw[:, :, SAPIEN_TO_CANONICAL]
    wrist_pose_rollout, qpos_sapien_rollout = apply_deximit_rollout_contract(
        wrist_pose, qpos_sapien_raw,
    )
    qpos_canonical_rollout = qpos_sapien_rollout[:, :, SAPIEN_TO_CANONICAL]
    quat_norm = np.linalg.norm(wrist_pose[:, :, 3:], axis=-1)
    candidate_valid = (
        np.isfinite(returned).all(axis=(1, 2))
        & np.isfinite(quat_norm).all(axis=1)
        & np.all((quat_norm > 0.5) & (quat_norm < 1.5), axis=1)
    )
    if not np.any(candidate_valid):
        raise RuntimeError("BODex returned no finite candidate pose.")

    provenance = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "episode_id": args.episode_id,
        "generator": "BODex GraspSynthesizer (right XHand)",
        **git_worktree_state(deximit_root),
        "deximit_root": deximit_root,
        "source_mesh": mesh_file,
        "source_mesh_sha256": sha256(mesh_file),
        "object_frame": (
            "source mesh coordinates; meters; BODex solved with the recorded "
            "SAPIEN object pose and returned poses transformed back to this frame"
            if not args.legacy_identity_pose else
            "source mesh coordinates; meters; legacy BODex solve at identity pose"
        ),
        "generation_pose_protocol": (
            "upstream_sapien_pose" if not args.legacy_identity_pose
            else "legacy_identity_pose"
        ),
        "generation_object_pose_wxyz": object_pose_wxyz,
        "seed": args.seed,
        "fingers": args.fingers,
        "grasp_depth": args.grasp_depth,
        "requested_candidates": args.num_grasps,
        "returned_candidates": int(len(returned)),
        "finite_candidates": int(candidate_valid.sum()),
        "stages": list(STAGES),
        "qpos_sapien_joint_order": list(SAPIEN_JOINTS),
        "qpos_canonical_joint_order": list(CANONICAL_JOINTS),
        "deximit_rollout_contract": {
            "right_hand": True,
            "pregrasp_distance_m": 0.1,
            "pregrasp_orientation": "copy grasp orientation",
            "relaxed_sapien_joint_slice": [2, 7],
            "relaxed_stage_slice": [0, 2],
            "relaxation_rad": -0.2,
            "grasp_offset_m": 0.0,
            "squeeze_angle_offset_rad": 0.0,
        },
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "object_geometry": object_info,
        "limitations": [
            "This is a DexImit-preprocessed proposal set, not a current-MuJoCo success result.",
            "No candidate is fed to the formal renderer/controller by this program.",
            "A later independent audit must check reachability, penetration, contact force, and lift stability.",
        ],
    }
    tmp_output = output.with_name(f".{output.stem}.tmp.npz")
    with tmp_output.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(SCHEMA),
            diagnostic_only=np.asarray(True),
            formal_renderer_3_3_eligible=np.asarray(False),
            episode_id=np.asarray(args.episode_id),
            stages=np.asarray(STAGES),
            hand_pose_object_wxyz=wrist_pose_rollout,
            qpos_sapien_order=qpos_sapien_rollout,
            qpos_canonical_order=qpos_canonical_rollout,
            raw_hand_pose_object_wxyz=wrist_pose.astype(np.float32),
            raw_qpos_sapien_order=qpos_sapien_raw.astype(np.float32),
            raw_qpos_canonical_order=qpos_canonical_raw.astype(np.float32),
            generation_object_pose_wxyz=object_pose_wxyz.astype(np.float32),
            candidate_valid=candidate_valid,
            qpos_sapien_joint_order=np.asarray(SAPIEN_JOINTS),
            qpos_canonical_joint_order=np.asarray(CANONICAL_JOINTS),
            source_mesh_sha256=np.asarray(provenance["source_mesh_sha256"]),
            provenance_json=np.asarray(json.dumps(provenance, default=json_ready, sort_keys=True)),
        )
    os.replace(tmp_output, output)
    provenance["output_npz"] = output
    provenance["output_npz_sha256"] = sha256(output)
    (work_dir / "provenance.json").write_text(
        json.dumps(provenance, default=json_ready, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "output": str(output),
        "work_dir": str(work_dir),
        "returned_candidates": int(len(returned)),
        "finite_candidates": int(candidate_valid.sum()),
        "diagnostic_only": True,
    }, ensure_ascii=False))
    # BODex's bundled native collision extension can corrupt the heap while
    # Python tears down its process-global mesh cache.  At this point every
    # artifact has already been atomically renamed and hashed.  Flush the only
    # user-facing stream, then skip the broken third-party finalizers.  Errors
    # anywhere above still raise normally and keep a nonzero exit status.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
