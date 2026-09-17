#!/usr/bin/env python3
"""Run and render one isolated BODex ``真人 | 参考 | 仿真`` diagnostic.

This is deliberately outside the formal renderer/3.3 chain.  BODex provides
three object-local geometric hand targets, not a source-timed demonstration.
The script maps one selected target to the current XHand, records a normal
MuJoCo free-object rollout, then renders only the stored states.  The middle
panel is the project's formal source-timed robot reference; the right panel is
the independent BODex rollout.  Neither render is fed back to the simulator.

The original RGB is aligned by annotated contact phases solely to make a
side-by-side review.  It is visibly labelled as non-synchronous and cannot be
interpreted as a motion-imitation or formal ``real | ref | sim`` result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import trimesh

from egoengine_repro.action.contracts import XHAND_SELF_FLOOR_TOLERANCE_M
from egoengine_repro.action.geometry import (
    explicit_collision_pairs,
    minimum_pair_distance,
)
from egoengine_repro.action.replay import (
    MujocoReplayBackend,
    direct_object_actuator_names,
    forbidden_object_constraint_names,
    validate_physics_trace,
)
from generate_bodex_candidates import (
    CANONICAL_JOINTS,
    SAPIEN_JOINTS,
    SAPIEN_TO_CANONICAL,
    apply_deximit_rollout_contract,
)


SCHEMA = "xhand_bodex_free_object_diagnostic_trace_v5_runtime_model_bound"
CANDIDATE_SCHEMA = "xhand_bodex_grasp_candidates_v2_diagnostic_only"
STAGE_NAMES = ("预抓取", "接近", "收紧", "抬起", "保持")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
WIDTH, HEIGHT = 720, 480
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
# BODex diagnostics must use the scene's fixed third-person view.  A moving or
# hand/object-following camera is intentionally unsupported: it can make a
# failed grasp look more stable than it is and prevents fair comparison across
# candidates.
THIRD_PERSON_CAMERA = "front"
DEFAULT_MAX_OBJECT_PENETRATION_M = 0.002
ROOT_JOINTS = (
    "R_forearm_tx_link_joint",
    "R_forearm_ty_link_joint",
    "R_forearm_tz_link_joint",
    "R_forearm_roll_link_joint",
    "R_forearm_pitch_link_joint",
    "R_forearm_yaw_link_joint",
)

# BODex root poses the fixed URDF ``base`` link.  Current MuJoCo uses
# ``right_hand_link`` as its six-DoF wrist body, so this is the fixed
# base-to-hand transform from BODex's published XHand URDF.
BASE_TO_HAND_TRANSLATION_M = np.asarray((-0.1, -0.03, -0.01), dtype=np.float64)
BASE_TO_HAND_ROTATION = Rotation.from_euler(
    "xyz", (-np.pi / 2.0, 0.0, np.pi / 2.0),
).as_matrix()
# The published BODex/DexImit XHand URDF uses axis ``0 -1 0`` for index
# abduction, while the current MuJoCo XHand uses ``0 1 0``.  A name-only copy
# silently mirrors that joint.  All other named finger axes have equal signs.
BODEX_TO_MUJOCO_JOINT_SIGN = np.asarray(
    (1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    dtype=np.float64,
)


@dataclass(frozen=True)
class Target:
    """One mapped BODex target in the current MuJoCo qpos convention."""

    qpos: np.ndarray
    hand_position_world_m: np.ndarray
    hand_rotation_world: np.ndarray


def scalar_joint_qpos_addresses(
    model: mujoco.MjModel, names: tuple[str, ...],
) -> np.ndarray:
    """Resolve named one-DoF joints without assuming global qpos layout."""
    addresses: list[int] = []
    scalar_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or int(model.jnt_type[joint]) not in scalar_types:
            raise ValueError(f"diagnostic scene lacks scalar joint {name!r}")
        addresses.append(int(model.jnt_qposadr[joint]))
    if len(set(addresses)) != len(addresses):
        raise ValueError("named diagnostic joints do not have unique qpos addresses")
    return np.asarray(addresses, dtype=np.int64)


def robot_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    """Return Cartesian-root then canonical-finger qpos addresses."""
    return scalar_joint_qpos_addresses(model, ROOT_JOINTS + CANONICAL_JOINTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument(
        "--object-reference", type=Path, required=True,
        help="Reference NPZ used only for its first one-time object free-joint pose.",
    )
    parser.add_argument("--real-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--physics-profile",
        choices=("current", "deximit_drive", "deximit_approx", "deximit_full"),
        default="current",
        help=(
            "current uses bounded XHand effort; deximit_drive reproduces DexImit's "
            "stiff finger position drives; deximit_approx also uses its default object mass; "
            "deximit_full additionally requires the isolated single-hull scene and applies "
            "DexImit's object damping plus stiff drives to all 18 robot joints."
        ),
    )
    parser.add_argument(
        "--deximit-object-mass", type=float, default=0.05885494234682545,
        help="Object mass for deximit_approx; smear-eraser default is mesh volume * 300.",
    )
    parser.add_argument("--finger-kp", type=float, default=2.0)
    parser.add_argument("--finger-kv", type=float, default=0.002)
    parser.add_argument("--finger-cap", type=float, default=1.1)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--approach-s", type=float, default=1.0)
    parser.add_argument("--squeeze-s", type=float, default=0.75)
    parser.add_argument("--lift-s", type=float, default=1.0)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--lift-m", type=float, default=0.05)
    parser.add_argument(
        "--motion-mode", choices=("vertical_lift", "source"), default="vertical_lift",
        help="Use a diagnostic vertical lift or the human-reviewed object motion segment.",
    )
    parser.add_argument("--human-reference", type=Path)
    parser.add_argument("--manual-label", type=Path)
    parser.add_argument(
        "--object-mesh", type=Path,
        help="Source object mesh used to reproduce DexImit's final mean-vertex error gate.",
    )
    parser.add_argument(
        "--max-object-penetration-m", type=float,
        default=DEFAULT_MAX_OBJECT_PENETRATION_M,
        help="Maximum sampled hand/object collision overlap allowed by the strict gate.",
    )
    parser.add_argument(
        "--closure-stage", choices=("grasp", "squeeze"), default="squeeze",
        help="BODex hand target held while lifting; default preserves the original squeeze test.",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--real-contact-start-row", type=int, required=True,
        help="Zero-based source row where the compared human hand starts approach/contact.",
    )
    parser.add_argument(
        "--real-stable-grasp-start-row", type=int, required=True,
        help="Zero-based source row where stable opposed human contact begins.",
    )
    parser.add_argument(
        "--real-stable-grasp-end-row", type=int, required=True,
        help="Zero-based final source row to show while the human grasp is still stable.",
    )
    parser.add_argument(
        "--render-failed-diagnostic", action="store_true",
        help=(
            "Explicitly allow a video for a candidate that failed the diagnostic "
            "grasp gate.  The default is to record its trace and audit only."
        ),
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    value = path.expanduser().resolve(strict=True)
    if not value.is_file():
        raise ValueError(f"{label} must be a file: {value}")
    return value




def load_candidate(path: Path, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        if schema != CANDIDATE_SCHEMA:
            raise ValueError("candidate is not a BODex diagnostic candidate artifact")
        if not bool(np.asarray(data["diagnostic_only"]).item()):
            raise ValueError("candidate must explicitly be diagnostic-only")
        if bool(np.asarray(data["formal_renderer_3_3_eligible"]).item()):
            raise ValueError("BODex diagnostic candidate must not be formal-renderer eligible")
        poses = np.asarray(data["hand_pose_object_wxyz"], dtype=np.float64)
        joints = np.asarray(data["qpos_canonical_order"], dtype=np.float64)
        joints_sapien = np.asarray(data["qpos_sapien_order"], dtype=np.float64)
        raw_poses = np.asarray(data["raw_hand_pose_object_wxyz"], dtype=np.float64)
        raw_joints_sapien = np.asarray(data["raw_qpos_sapien_order"], dtype=np.float64)
        raw_joints = np.asarray(data["raw_qpos_canonical_order"], dtype=np.float64)
        valid = np.asarray(data["candidate_valid"], dtype=bool)
        stages = tuple(str(value) for value in np.asarray(data["stages"]).tolist())
        order = tuple(str(value) for value in np.asarray(
            data["qpos_canonical_joint_order"],
        ).tolist())
        sapien_order = tuple(str(value) for value in np.asarray(
            data["qpos_sapien_joint_order"],
        ).tolist())
        required_v2 = {
            "qpos_sapien_order", "raw_hand_pose_object_wxyz",
            "raw_qpos_sapien_order", "raw_qpos_canonical_order", "provenance_json",
        }
        if required_v2 - set(data.files):
            raise ValueError("candidate lacks the DexImit rollout-contract arrays")
        provenance = json.loads(str(np.asarray(data["provenance_json"]).item()))
        contract = provenance.get("deximit_rollout_contract")
        if (
            provenance.get("schema") != CANDIDATE_SCHEMA
            or not isinstance(contract, dict)
            or contract.get("pregrasp_distance_m") != 0.1
            or contract.get("relaxation_rad") != -0.2
        ):
            raise ValueError("candidate provenance does not match DexImit's rollout contract")
        metadata = {
            "schema": schema,
            "episode_id": str(np.asarray(data["episode_id"]).item()),
            "source_mesh_sha256": str(np.asarray(data["source_mesh_sha256"]).item()),
            "stages": stages,
            "joint_order": order,
            "sapien_joint_order": sapien_order,
            "deximit_rollout_contract": contract,
        }
    expected_order = (
        "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2", "right_hand_index_bend_joint",
        "right_hand_index_joint1", "right_hand_index_joint2",
        "right_hand_mid_joint1", "right_hand_mid_joint2",
        "right_hand_ring_joint1", "right_hand_ring_joint2",
        "right_hand_pinky_joint1", "right_hand_pinky_joint2",
    )
    if (
        poses.ndim != 3 or poses.shape[1:] != (3, 7)
        or joints.shape != (len(poses), 3, 12)
        or joints_sapien.shape != joints.shape
        or raw_poses.shape != poses.shape
        or raw_joints_sapien.shape != joints.shape
        or raw_joints.shape != joints.shape
        or valid.shape != (len(poses),) or stages != ("pregrasp", "grasp", "squeeze")
        or order != expected_order
        or sapien_order != SAPIEN_JOINTS
    ):
        raise ValueError("BODex candidate arrays or declared joint order are malformed")
    expected_poses, expected_joints_sapien = apply_deximit_rollout_contract(
        raw_poses, raw_joints_sapien,
    )
    if not (
        np.allclose(poses, expected_poses, atol=1e-7, rtol=0.0)
        and np.allclose(joints_sapien, expected_joints_sapien, atol=1e-7, rtol=0.0)
        and np.allclose(joints, joints_sapien[:, :, SAPIEN_TO_CANONICAL], atol=0.0, rtol=0.0)
        and np.allclose(raw_joints, raw_joints_sapien[:, :, SAPIEN_TO_CANONICAL], atol=0.0, rtol=0.0)
    ):
        raise ValueError("BODex candidate arrays do not implement the declared rollout contract")
    if not 0 <= index < len(poses) or not valid[index]:
        raise ValueError("candidate index is outside the finite BODex proposal set")
    pose, joint = poses[index].copy(), joints[index].copy()
    norm = np.linalg.norm(pose[:, 3:], axis=1)
    if (
        not np.isfinite(pose).all() or not np.isfinite(joint).all()
        or not np.allclose(norm, 1.0, atol=2e-5)
    ):
        raise ValueError("selected BODex pose/joint target is non-finite or unnormalised")
    return pose, joint, metadata


def load_object_pose(path: Path, *, expected_nq: int, object_address: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != expected_nq or not len(qpos):
        raise ValueError("object reference qpos does not match the diagnostic scene")
    result = qpos[0, object_address : object_address + 7].copy()
    if (
        result.shape != (7,) or not np.isfinite(result).all()
        or not np.isclose(np.linalg.norm(result[3:]), 1.0, atol=1e-6)
    ):
        raise ValueError("object reference does not contain a finite free-joint pose")
    return result


def pose_matrix(pose_wxyz: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("pose must be a finite xyz+wxyz vector")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def source_motion_contract(
    human_reference: Path, manual_label: Path, object_qpos: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    label = json.loads(manual_label.read_text(encoding="utf-8"))
    if (
        label.get("schema") != "deximit_manual_subactions_v1_diagnostic_only"
        or label.get("diagnostic_only") is not True
        or label.get("formal_renderer_3_3_eligible") is not False
        or label.get("hand") != "right"
    ):
        raise ValueError("manual label violates the isolated diagnostic contract")
    rows = label.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("manual label lacks action rows")
    grasp, motion = int(rows["grasp"]), int(rows["motion"])
    if not 0 <= int(rows["pregrasp"]) < grasp < motion:
        raise ValueError("manual pregrasp/grasp/motion rows are not ordered")
    with np.load(human_reference, allow_pickle=False) as values:
        reference = np.asarray(values["T_sim_object_reference"], dtype=np.float64)
    if reference.ndim == 4 and reference.shape[1] == 1:
        reference = reference[:, 0]
    if reference.ndim != 3 or reference.shape[1:] != (4, 4) or motion >= len(reference):
        raise ValueError("human reference lacks the labelled object poses")
    source_relative = reference[motion] @ np.linalg.inv(reference[grasp])
    initial = pose_matrix(object_qpos)
    relocate = initial @ np.linalg.inv(reference[grasp])
    relative_world = relocate @ source_relative @ np.linalg.inv(relocate)
    return relative_world, {
        "label_source": "human_reviewed_video_and_geometry",
        "pregrasp_row": int(rows["pregrasp"]),
        "grasp_row": grasp,
        "motion_row": motion,
        "translation_m": relative_world[:3, 3].tolist(),
        "translation_norm_m": float(np.linalg.norm(relative_world[:3, 3])),
        "rotation_deg": float(np.degrees(
            Rotation.from_matrix(relative_world[:3, :3]).magnitude(),
        )),
    }


def deximit_endpoint_metric(
    mesh_path: Path, initial_object_qpos: np.ndarray, final_object_qpos: np.ndarray,
    motion_transform: np.ndarray,
) -> dict[str, object]:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("DexImit endpoint metric requires one triangle mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("object mesh has no finite 3D vertices")
    initial = pose_matrix(initial_object_qpos)
    actual = pose_matrix(final_object_qpos)
    target = motion_transform @ initial

    def world(transform: np.ndarray) -> np.ndarray:
        return vertices @ transform[:3, :3].T + transform[:3, 3]

    errors = np.linalg.norm(world(actual) - world(target), axis=1)
    mean_error = float(errors.mean())
    return {
        "definition": "DexImit final mean object-vertex error",
        "threshold_m": 0.02,
        "mean_vertex_error_m": mean_error,
        "maximum_vertex_error_m": float(errors.max()),
        "center_error_m": float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
        "rotation_error_deg": float(np.degrees(Rotation.from_matrix(
            actual[:3, :3].T @ target[:3, :3],
        ).magnitude())),
        "passed": bool(mean_error <= 0.02),
    }


def map_target(
    model: mujoco.MjModel, object_qpos: np.ndarray, pose_object_wxyz: np.ndarray,
    joints: np.ndarray, object_address: int,
) -> Target:
    """Map a BODex object-local base pose to the current XHand root joints."""
    object_rotation = Rotation.from_quat(
        object_qpos[3:], scalar_first=True,
    ).as_matrix()
    base_rotation = Rotation.from_quat(
        pose_object_wxyz[3:], scalar_first=True,
    ).as_matrix()
    hand_rotation = object_rotation @ base_rotation @ BASE_TO_HAND_ROTATION
    hand_position = object_qpos[:3] + object_rotation @ (
        pose_object_wxyz[:3] + base_rotation @ BASE_TO_HAND_TRANSLATION_M
    )
    # The MuJoCo root hierarchy is Rz(roll) Rx(pitch) Ry(-yaw).  Uppercase
    # scipy notation is intrinsic and therefore matches that body chain.
    # At the Euler singularity SciPy chooses one of several equivalent wrist
    # triples.  ``checked_target`` immediately verifies the reconstructed body
    # rotation, so suppress only this expected representation warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        roll, pitch, negative_yaw = Rotation.from_matrix(hand_rotation).as_euler("ZXY")
    root_addresses = scalar_joint_qpos_addresses(model, ROOT_JOINTS)
    finger_addresses = scalar_joint_qpos_addresses(model, CANONICAL_JOINTS)
    qpos = model.qpos0.copy()
    qpos[root_addresses[:3]] = hand_position
    qpos[root_addresses[3:]] = (roll, pitch, -negative_yaw)
    qpos[finger_addresses] = joints * BODEX_TO_MUJOCO_JOINT_SIGN
    qpos[object_address : object_address + 7] = object_qpos
    return Target(qpos, hand_position, hand_rotation)


def project_hand_joint_limits(
    model: mujoco.MjModel, qpos: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project a BODex target into the current model's attainable joint box.

    DexImit sometimes sends a relaxed target just beyond a URDF limit and lets
    PhysX stop the joint at that limit.  MuJoCo ``mj_forward`` does not project
    an initialized qpos, so static checks and initialization must do so
    explicitly rather than auditing an impossible hand shape.
    """
    result = np.asarray(qpos, dtype=np.float64).copy()
    if result.shape != (model.nq,) or not np.isfinite(result).all():
        raise ValueError("hand target qpos is malformed")
    violations: dict[str, float] = {}
    for name in CANONICAL_JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or not bool(model.jnt_limited[joint]):
            raise ValueError(f"diagnostic scene lacks a limited XHand joint {name!r}")
        address = int(model.jnt_qposadr[joint])
        lower, upper = (float(value) for value in model.jnt_range[joint])
        raw = float(result[address])
        clipped = float(np.clip(raw, lower, upper))
        result[address] = clipped
        violations[name] = abs(raw - clipped)
    return result, violations


def transform_hand_target(
    backend: MujocoReplayBackend, target: Target, relative_world: np.ndarray,
    object_qpos: np.ndarray,
) -> Target:
    rotation = relative_world[:3, :3] @ target.hand_rotation_world
    position = relative_world[:3, :3] @ target.hand_position_world_m + relative_world[:3, 3]
    roll, pitch, negative_yaw = Rotation.from_matrix(rotation).as_euler("ZXY")
    root_addresses = scalar_joint_qpos_addresses(backend.model, ROOT_JOINTS)
    qpos = target.qpos.copy()
    qpos[root_addresses[:3]] = position
    qpos[root_addresses[3:]] = (roll, pitch, -negative_yaw)
    moved = Target(qpos, position, rotation)
    checked_target(backend, moved, object_qpos)
    return moved


def dynamic_rollout_targets(
    backend: MujocoReplayBackend, targets: list[Target],
    projected_pregrasp_qpos: np.ndarray, *, closure_stage: str,
    motion_transform: np.ndarray | None, object_qpos: np.ndarray,
    vertical_lift_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the one shared dynamic target sequence used by screen and rerun.

    Only the directly initialized pregrasp is projected into joint limits.
    Later BODex commands remain raw, matching PhysX's behavior of accepting a
    drive target beyond a joint limit and letting the joint constraint stop
    the actual configuration.  Mixing projected screening targets with raw
    selected-candidate targets silently changes the grasp being evaluated.
    """
    if len(targets) != 3 or closure_stage not in {"grasp", "squeeze"}:
        raise ValueError("BODex dynamic rollout target contract is malformed")
    pregrasp = np.asarray(projected_pregrasp_qpos, dtype=np.float64).copy()
    if pregrasp.shape != (backend.model.nq,) or not np.isfinite(pregrasp).all():
        raise ValueError("projected BODex pregrasp target is malformed")
    grasp = targets[1].qpos.copy()
    closure_target = targets[{"grasp": 1, "squeeze": 2}[closure_stage]]
    closure = closure_target.qpos.copy()
    if motion_transform is None:
        move = closure.copy()
        root_addresses = scalar_joint_qpos_addresses(backend.model, ROOT_JOINTS)
        move[root_addresses[2]] += float(vertical_lift_m)
    else:
        move = transform_hand_target(
            backend, closure_target, motion_transform, object_qpos,
        ).qpos
    return pregrasp, grasp, closure, move


def strict_grasp_gate(
    *, hard_legal: bool, opposed_contact: bool, hold_opposed_contact: bool,
    held_lift: bool, stable_hold: bool, minimum_object_gap_m: float,
    maximum_object_penetration_m: float,
) -> bool:
    """One success definition shared by screening and full-trace reruns."""
    gap = float(minimum_object_gap_m)
    limit = float(maximum_object_penetration_m)
    if not np.isfinite((gap, limit)).all() or limit <= 0.0:
        raise ValueError("hand/object penetration gate values are invalid")
    return bool(
        hard_legal and opposed_contact and hold_opposed_contact
        and held_lift and stable_hold and gap >= -limit
    )


def checked_target(
    backend: MujocoReplayBackend, target: Target, object_qpos: np.ndarray,
) -> None:
    """Reject a mapping that does not reproduce BODex's intended hand root."""
    data = backend.data
    data.qpos[:] = target.qpos
    data.qvel[:] = 0.0
    backend.mujoco.mj_forward(backend.model, data)
    hand = backend.mujoco.mj_name2id(
        backend.model, backend.mujoco.mjtObj.mjOBJ_BODY, "right_hand_link",
    )
    if hand < 0:
        raise ValueError("diagnostic scene has no right_hand_link body")
    position_error = float(np.linalg.norm(data.xpos[hand] - target.hand_position_world_m))
    rotation_error = float(Rotation.from_matrix(
        target.hand_rotation_world.T @ data.xmat[hand].reshape(3, 3),
    ).magnitude())
    if position_error > 1.0e-9 or rotation_error > 1.0e-9:
        raise RuntimeError(
            "BODex-to-XHand root mapping is not exact: "
            f"position={position_error:g}, rotation={rotation_error:g}"
        )
    if not np.allclose(
        data.qpos[backend.object_qpos_address : backend.object_qpos_address + 7],
        object_qpos, atol=0.0, rtol=0.0,
    ):
        raise RuntimeError("target mapping unexpectedly altered the free-object initial pose")


def target_gaps(
    backend: MujocoReplayBackend, target: Target,
    pairs: dict[str, tuple[tuple[int, int], ...]],
) -> dict[str, float]:
    data = backend.data
    data.qpos[:] = target.qpos
    data.qvel[:] = 0.0
    backend.mujoco.mj_forward(backend.model, data)
    return {
        family: minimum_pair_distance(backend.model, data, backend.mujoco, values)
        for family, values in pairs.items()
    }


def apply_physics_profile(
    backend: MujocoReplayBackend, profile: str, deximit_object_mass: float,
) -> dict[str, object]:
    """Apply one shared, diagnostic-only controller/physics contract.

    Screening and selected-candidate reruns use this same helper so a candidate
    cannot silently be replayed with a different actuator interpretation.
    DexImit profiles remain in-memory MuJoCo approximations and never modify the
    source scene XML.
    """
    if backend.physics_substeps != 0 or not np.isclose(
        float(backend.data.time), 0.0, atol=0.0, rtol=0.0,
    ):
        raise RuntimeError("diagnostic physics profile must be applied before simulation starts")
    if profile == "current":
        return {
            "name": profile,
            "finger_control": "bounded XHand effort from URDF",
            "object_mass_kg": float(backend.model.body_mass[backend.object_body_id]),
        }
    model = backend.model
    model.opt.integrator = int(backend.mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    driven_actuators: list[str] = []
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        joint_name = (
            backend.mujoco.mj_id2name(
                model, backend.mujoco.mjtObj.mjOBJ_JOINT, joint,
            ) or ""
        )
        is_finger = joint_name.startswith("right_hand_")
        if not is_finger and profile != "deximit_full":
            continue
        # MuJoCo affine equivalent of the original SAPIEN position drive:
        # torque = 1000 * target - 1000 * q - 100 * qvel.
        model.actuator_gaintype[actuator] = int(backend.mujoco.mjtGain.mjGAIN_FIXED)
        model.actuator_gainprm[actuator, :] = 0.0
        model.actuator_gainprm[actuator, 0] = 1000.0
        model.actuator_biastype[actuator] = int(backend.mujoco.mjtBias.mjBIAS_AFFINE)
        model.actuator_biasprm[actuator, :] = 0.0
        model.actuator_biasprm[actuator, 1] = -1000.0
        model.actuator_biasprm[actuator, 2] = -100.0
        model.actuator_ctrllimited[actuator] = 0
        # DexImit's released SAPIEN drive uses force_limit=1e10.  Disabling
        # only the control-target range is not equivalent: several MuJoCo
        # wrist actuators retain explicit +/-100 N or +/-30 Nm force limits.
        # Both limiter flags must be off for the diagnostic "no force cap"
        # claim to be true.
        model.actuator_forcelimited[actuator] = 0
        driven_actuators.append(joint_name)
    expected_actuators = 18 if profile == "deximit_full" else 12
    if len(driven_actuators) != expected_actuators:
        raise ValueError(
            f"DexImit drive profile resolved {len(driven_actuators)} actuators, "
            f"expected {expected_actuators}",
        )
    driven_ids = [
        actuator for actuator in range(model.nu)
        if (
            backend.mujoco.mj_id2name(
                model, backend.mujoco.mjtObj.mjOBJ_JOINT,
                int(model.actuator_trnid[actuator, 0]),
            ) or ""
        ) in set(driven_actuators)
    ]
    if any(
        bool(model.actuator_ctrllimited[actuator])
        or bool(model.actuator_forcelimited[actuator])
        for actuator in driven_ids
    ):
        raise RuntimeError("DexImit diagnostic drive still has an active control or force limit")
    original_mass = float(model.body_mass[backend.object_body_id])
    original_inertia = model.body_inertia[backend.object_body_id].copy()
    applied_mass = original_mass
    if profile in {"deximit_approx", "deximit_full"}:
        target_mass = float(deximit_object_mass)
        if not np.isfinite(target_mass) or target_mass <= 0.0:
            raise ValueError("DexImit approximate object mass must be finite and positive")
        ratio = target_mass / original_mass
        model.body_mass[backend.object_body_id] = target_mass
        model.body_inertia[backend.object_body_id] *= ratio
        applied_mass = target_mass
    object_damping = None
    if profile == "deximit_full":
        object_geom_names = sorted(
            backend.mujoco.mj_id2name(
                model, backend.mujoco.mjtObj.mjOBJ_GEOM, geom,
            ) or ""
            for geom in backend.object_geom_ids
        )
        if object_geom_names != ["right_object_single"]:
            raise ValueError(
                "deximit_full requires the isolated one-convex-hull object scene; "
                f"found {object_geom_names}",
            )
        dof = int(model.jnt_dofadr[backend.object_joint_id])
        model.dof_damping[dof : dof + 6] = 20.0
        object_damping = model.dof_damping[dof : dof + 6].tolist()
        model.opt.timestep = 1.0 / 240.0
    # body_subtreemass, qpos0-dependent constants and model statistics are
    # compiler-derived.  Directly editing body_mass/body_inertia without this
    # refresh leaves a model whose public mass fields disagree internally.
    backend.mujoco.mj_setConst(model, backend.data)
    backend.mujoco.mj_resetData(model, backend.data)
    return {
        "name": profile,
        "finger_control": "DexImit-compatible position drive kp=1000, kv=100, no force cap",
        "driven_actuators": driven_actuators,
        "driven_actuator_count": len(driven_ids),
        "driven_control_limits_disabled": True,
        "driven_force_limits_disabled": True,
        "object_mass_kg": applied_mass,
        "original_scene_object_mass_kg": original_mass,
        "object_inertia_kg_m2": model.body_inertia[backend.object_body_id].tolist(),
        "original_scene_object_inertia_kg_m2": original_inertia.tolist(),
        "object_subtree_mass_kg": float(model.body_subtreemass[backend.object_body_id]),
        "model_constants_refreshed_after_runtime_edits": True,
        "object_free_joint_damping": object_damping,
        "physics_timestep_s": float(model.opt.timestep),
        "limitations": (
            "MuJoCo diagnostic approximation only; identical numeric settings do not make "
            "the MuJoCo and PhysX contact solvers equivalent."
        ),
    }


def save_runtime_model(
    model: mujoco.MjModel, path: Path,
) -> tuple[mujoco.MjModel, str]:
    """Save and reload the exact compiled model used by a diagnostic trace.

    Runtime mass, inertia, damping, actuator, and contact edits are not present
    in the source XML.  Binding a trace only to that XML therefore makes an
    apparently valid artifact impossible to reproduce in a fresh process.
    """
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime model: {path}")
    mujoco.mj_saveModel(model, str(path), None)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("MuJoCo did not create the runtime model artifact")
    reloaded = mujoco.MjModel.from_binary_path(str(path))
    signature = (model.nq, model.nv, model.nu, model.nbody, model.ngeom, model.npair)
    reloaded_signature = (
        reloaded.nq, reloaded.nv, reloaded.nu,
        reloaded.nbody, reloaded.ngeom, reloaded.npair,
    )
    if reloaded_signature != signature:
        raise RuntimeError(
            f"saved MuJoCo model signature changed: {signature} -> {reloaded_signature}",
        )
    return reloaded, sha256(path)


def compensated_target(
    backend: MujocoReplayBackend, inverse_data: mujoco.MjData,
    inverse_force: np.ndarray, desired: np.ndarray, velocity: np.ndarray,
    acceleration: np.ndarray, finger_kp: float, finger_kv: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an independent bounded inverse-dynamics hand command.

    The object portion is replaced by the live state before the inverse-dynamics
    call and no object actuator exists.  It is therefore impossible for this
    helper to pose-servo the object.
    """
    model, mujoco = backend.model, backend.mujoco
    robot_qpos = backend.actuator_qpos_addresses
    robot_qvel = backend.actuator_qvel_addresses
    if velocity.shape != (len(robot_qvel),) or acceleration.shape != velocity.shape:
        raise ValueError("BODex diagnostic target derivatives have invalid dimensions")
    live_desired = desired.copy()
    object_address = backend.object_qpos_address
    live_desired[object_address : object_address + 7] = backend.data.qpos[
        object_address : object_address + 7
    ]
    inverse_data.qpos[:] = live_desired
    inverse_data.qvel[:] = 0.0
    inverse_data.qvel[robot_qvel] = velocity
    mujoco.mj_forward(model, inverse_data)
    inverse_data.qacc[:] = 0.0
    inverse_data.qacc[robot_qvel] = acceleration
    mujoco.mj_rne(model, inverse_data, 1, inverse_force)
    command = live_desired.copy()
    finger_ids = set(backend.finger_torque_ids.tolist())
    for actuator, qpos_address in enumerate(robot_qpos.tolist()):
        joint = int(model.actuator_trnid[actuator, 0])
        dof = int(model.jnt_dofadr[joint])
        gear = float(model.actuator_gear[actuator, 0])
        required = float(inverse_force[dof] / gear)
        if actuator in finger_ids:
            kp, kv = finger_kp, finger_kv
        else:
            kp = float(model.actuator_gainprm[actuator, 0])
            kv = -float(model.actuator_biasprm[actuator, 2])
        if abs(gear) <= 1e-12 or kp <= 0.0:
            raise ValueError(f"invalid actuator gain or gear at index {actuator}")
        command[qpos_address] = (
            live_desired[qpos_address]
            + (required + kv * inverse_data.qvel[dof]) / kp
        )
    return live_desired, command


def smooth_target(
    first: np.ndarray, second: np.ndarray, phase: float, duration_s: float,
    qpos_addresses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C1 smoothstep target and its derivatives for the 18 robot joints."""
    alpha = float(phase)
    duration = float(duration_s)
    if not 0.0 < alpha <= 1.0 or duration <= 0.0:
        raise ValueError("diagnostic phase or duration is invalid")
    addresses = np.asarray(qpos_addresses, dtype=np.int64)
    if addresses.shape != (18,) or len(np.unique(addresses)) != len(addresses):
        raise ValueError("diagnostic robot qpos addresses are malformed")
    delta = second[addresses] - first[addresses]
    blend = 3.0 * alpha**2 - 2.0 * alpha**3
    velocity = (6.0 * alpha - 6.0 * alpha**2) / duration * delta
    acceleration = (6.0 - 12.0 * alpha) / duration**2 * delta
    result = first.copy()
    result[addresses] = first[addresses] + blend * delta
    return result, velocity, acceleration


def contact_row(backend: MujocoReplayBackend) -> tuple[np.ndarray, np.ndarray]:
    """Measure real digit-object contacts and normal force at one physics state."""
    contact = np.zeros(len(FINGERS), dtype=bool)
    normal = np.zeros(len(FINGERS), dtype=np.float64)
    force = np.zeros(6, dtype=np.float64)
    for index in range(backend.data.ncon):
        item = backend.data.contact[index]
        # Positive ``gap`` values add inactive early detections to
        # mjData.contact.  They are useful for matching PhysX's broad contact
        # generation distance but are not load-bearing contact and must not be
        # counted as a grasp.
        if int(item.efc_address) < 0:
            continue
        first, second = int(item.geom1), int(item.geom2)
        for hand, obj in ((first, second), (second, first)):
            if obj not in backend.object_geom_ids or hand not in backend.hand_geom_names:
                continue
            name = backend.hand_geom_names[hand]
            matching = [number for number, finger in enumerate(FINGERS) if finger in name]
            if len(matching) != 1:
                continue
            backend.mujoco.mj_contactForce(backend.model, backend.data, index, force)
            if abs(float(force[0])) <= 1.0e-8:
                continue
            digit = matching[0]
            contact[digit] = True
            normal[digit] += abs(float(force[0]))
    return contact, normal


def run_phase(
    backend: MujocoReplayBackend, inverse_data: mujoco.MjData,
    inverse_force: np.ndarray, trace: dict[str, list[object]], *, name: str,
    first: np.ndarray, second: np.ndarray, duration_s: float, finger_kp: float,
    finger_kv: float, pairs: dict[str, tuple[tuple[int, int], ...]],
) -> dict[str, object]:
    model = backend.model
    count = max(1, int(round(float(duration_s) / float(model.opt.timestep))))
    object_z: list[float] = []
    self_gap: list[float] = []
    floor_gap: list[float] = []
    object_gap: list[float] = []
    contacts: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    target_error: list[float] = []
    robot_addresses = backend.actuator_qpos_addresses
    for step in range(1, count + 1):
        desired, velocity, acceleration = smooth_target(
            first, second, step / count, duration_s, robot_addresses,
        )
        desired, command = compensated_target(
            backend, inverse_data, inverse_force, desired, velocity, acceleration,
            finger_kp, finger_kv,
        )
        backend.step(backend.reference_action(command), float(model.opt.timestep))
        if (
            not np.isfinite(backend.data.qpos).all()
            or not np.isfinite(backend.data.qvel).all()
            or not np.isfinite(backend.data.qacc).all()
            or any(int(item.number) > 0 for item in backend.data.warning)
        ):
            raise FloatingPointError(
                f"numerically unstable rollout at phase {name}, step {step}/{count}",
            )
        touched, normal = contact_row(backend)
        gaps = {
            family: minimum_pair_distance(model, backend.data, backend.mujoco, values)
            for family, values in pairs.items()
        }
        trace["qpos"].append(backend.data.qpos.copy())
        trace["qvel"].append(backend.data.qvel.copy())
        trace["ctrl"].append(backend.data.ctrl.copy())
        trace["desired_qpos"].append(desired)
        trace["control_target_qpos"].append(command)
        trace["phase"].append(name)
        trace["time_s"].append(float(backend.data.time))
        trace["finger_contact"].append(touched)
        trace["finger_normal_force_n"].append(normal)
        trace["hand_self_gap_m"].append(gaps["self"])
        trace["hand_floor_gap_m"].append(gaps["floor"])
        trace["hand_object_gap_m"].append(gaps["object"])
        object_z.append(float(backend.data.xpos[backend.object_body_id, 2]))
        self_gap.append(gaps["self"])
        floor_gap.append(gaps["floor"])
        object_gap.append(gaps["object"])
        contacts.append(touched)
        normals.append(normal)
        target_error.append(float(np.linalg.norm(
            backend.data.qpos[robot_addresses] - desired[robot_addresses],
        )))
    contact_array = np.asarray(contacts, dtype=bool)
    normal_array = np.asarray(normals, dtype=np.float64)
    return {
        "phase": name,
        "duration_s": float(count * model.opt.timestep),
        "physics_samples": count,
        "object_z_start_m": object_z[0],
        "object_z_end_m": object_z[-1],
        "object_z_min_m": min(object_z),
        "object_z_peak_m": max(object_z),
        "minimum_hand_self_gap_m": min(self_gap),
        "minimum_hand_floor_gap_m": min(floor_gap),
        "minimum_hand_object_gap_m": min(object_gap),
        "contacted_digits": [
            finger for number, finger in enumerate(FINGERS)
            if bool(contact_array[:, number].any())
        ],
        "simultaneous_thumb_and_other_observed": bool(
            (contact_array[:, 0] & contact_array[:, 1:].any(axis=1)).any()
        ),
        "peak_normal_force_n": {
            finger: float(normal_array[:, number].max())
            for number, finger in enumerate(FINGERS)
        },
        "maximum_target_error_l2": max(target_error),
        "final_target_error_l2": target_error[-1],
    }


def require_scene(backend: MujocoReplayBackend) -> None:
    model, mujoco = backend.model, backend.mujoco
    if model.nq != 25 or model.nv != 24 or model.nu != 18:
        raise ValueError("diagnostic currently supports the one-right-XHand free-object scene")
    direct = direct_object_actuator_names(
        model, mujoco, object_body_id=backend.object_body_id,
    )
    if direct:
        raise ValueError(f"diagnostic scene has forbidden object actuators: {direct}")
    forbidden = forbidden_object_constraint_names(
        model, mujoco, object_joint_id=backend.object_joint_id,
        object_body_id=backend.object_body_id,
    )
    if forbidden:
        raise ValueError(f"diagnostic scene has forbidden object constraints: {forbidden}")


def read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode original RGB video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("original RGB video has no frames")
    return frames


def fit(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(WIDTH / width, HEIGHT / height)
    resized = cv2.resize(
        frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA,
    )
    result = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    offset_x = (WIDTH - resized.shape[1]) // 2
    offset_y = (HEIGHT - resized.shape[0]) // 2
    result[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    return result


def label(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 57), fill=(0, 0, 0))
    font_title = ImageFont.truetype(str(FONT), 22)
    font_subtitle = ImageFont.truetype(str(FONT), 14)
    draw.text((12, 6), title, font=font_title, fill=(235, 235, 235))
    draw.text((12, 34), subtitle, font=font_subtitle, fill=(185, 185, 185))
    return np.asarray(image)


def phase_aligned_real_rows(
    phases: np.ndarray, *, contact_start: int, stable_start: int, stable_end: int,
) -> np.ndarray:
    """Map static BODex phases onto the relevant human contact interval."""
    values = np.asarray(phases)
    if values.ndim != 1 or not len(values):
        raise ValueError("diagnostic phases are malformed")
    if not 0 <= contact_start <= stable_start <= stable_end:
        raise ValueError("human contact rows must be monotonic and non-negative")
    result = np.full(len(values), contact_start, dtype=np.int64)
    approach = np.flatnonzero(values == STAGE_NAMES[1])
    if len(approach):
        result[approach] = np.rint(np.linspace(
            contact_start, stable_start, len(approach),
        )).astype(np.int64)
    stable = np.flatnonzero(np.isin(values, STAGE_NAMES[2:]))
    if len(stable):
        result[stable] = np.rint(np.linspace(
            stable_start, stable_end, len(stable),
        )).astype(np.int64)
    return result


def require_fixed_camera(model: mujoco.MjModel, name: str) -> int:
    """Return a compiled camera id only when it is genuinely non-following."""
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if camera < 0:
        raise ValueError(f"diagnostic scene lacks third-person camera {name!r}")
    if int(model.cam_mode[camera]) != 0:
        raise ValueError(
            f"camera {name!r} is not fixed (compiled mode={int(model.cam_mode[camera])}); "
            "following cameras are forbidden",
        )
    return camera


def render(
    scene: Path, real_video: Path, formal_reference: Path,
    trace: dict[str, np.ndarray], output: Path,
    *, fps: int, outcome: str, real_contact_rows: tuple[int, int, int],
) -> dict[str, object]:
    real = read_video(real_video)
    with np.load(formal_reference, allow_pickle=False) as reference_artifact:
        if "qpos" not in reference_artifact.files:
            raise ValueError("formal reference lacks qpos")
        reference_qpos = np.asarray(reference_artifact["qpos"], dtype=np.float64)
    model = mujoco.MjModel.from_xml_path(str(scene))
    if model.nq != trace["qpos"].shape[1]:
        raise ValueError("recorded BODex trace does not match rendering scene")
    if (
        reference_qpos.shape != (len(real), model.nq)
        or not np.isfinite(reference_qpos).all()
    ):
        raise ValueError("formal reference qpos does not align with the original video")
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    sim_data = mujoco.MjData(model)
    ref_data = mujoco.MjData(model)
    camera_id = require_fixed_camera(model, THIRD_PERSON_CAMERA)
    step_dt = float(model.opt.timestep)
    stride = max(1, int(round(1.0 / (fps * step_dt))))
    indices = np.arange(0, len(trace["qpos"]), stride, dtype=np.int64)
    if indices[-1] != len(trace["qpos"]) - 1:
        indices = np.append(indices, len(trace["qpos"]) - 1)
    real_rows = phase_aligned_real_rows(
        trace["phase"], contact_start=real_contact_rows[0],
        stable_start=real_contact_rows[1], stable_end=real_contact_rows[2],
    )
    if int(real_rows.max()) >= len(real):
        raise ValueError(
            f"phase-aligned source row {int(real_rows.max())} exceeds {len(real)} video frames"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(output, fps=fps) as writer:
            for display_index, index in enumerate(indices.tolist()):
                real_index = int(real_rows[index])
                sim_data.qpos[:] = trace["qpos"][index]
                sim_data.qvel[:] = 0.0
                mujoco.mj_forward(model, sim_data)
                ref_data.qpos[:] = reference_qpos[real_index]
                ref_data.qvel[:] = 0.0
                mujoco.mj_forward(model, ref_data)
                renderer.update_scene(ref_data, camera=camera_id)
                reference = renderer.render().copy()
                renderer.update_scene(sim_data, camera=camera_id)
                simulation = renderer.render().copy()
                phase = str(trace["phase"][index])
                subtitle = f"阶段：{phase}；物理时刻 {trace['time_s'][index]:.2f} 秒"
                panels = (
                    label(
                        fit(real[real_index]), "真人：原始 TACO 视频",
                        f"接触阶段对齐到源行 {real_index}；不是逐帧同步",
                    ),
                    label(
                        reference, "参考：真人序列重定向结果",
                        f"源行 {real_index}；与左栏同一时刻；不是 BODex 目标",
                    ),
                    label(simulation, f"仿真：自由物体（{outcome}）", f"{subtitle}；无物体控制；仅诊断"),
                )
                writer.append_data(np.concatenate(panels, axis=1))
    finally:
        renderer.close()
    return {
        "frames": int(len(indices)),
        "fps": int(fps),
        "duration_s": float(len(indices) / fps),
        "render_stride_physics_steps": int(stride),
        "real_alignment": {
            "kind": "contact-phase-aligned; not frame-synchronous",
            "contact_start_row": int(real_contact_rows[0]),
            "stable_grasp_start_row": int(real_contact_rows[1]),
            "stable_grasp_end_row": int(real_contact_rows[2]),
            "minimum_displayed_row": int(real_rows[indices].min()),
            "maximum_displayed_row": int(real_rows[indices].max()),
        },
        "formal_reference": str(formal_reference),
        "formal_reference_sha256": sha256(formal_reference),
    }


def main() -> int:
    args = parse_args()
    if (
        min(
            args.finger_kp, args.finger_kv, args.finger_cap, args.settle_s,
            args.approach_s, args.squeeze_s, args.lift_s, args.hold_s, args.lift_m,
            args.max_object_penetration_m,
        ) <= 0.0 or args.fps <= 0
        or not np.isfinite(args.deximit_object_mass)
        or args.deximit_object_mass <= 0.0
        or not (
            0 <= args.real_contact_start_row
            <= args.real_stable_grasp_start_row
            <= args.real_stable_grasp_end_row
        )
    ):
        raise ValueError("controller, duration, lift and fps values must be positive")
    candidate_path = require_file(args.candidate, "candidate")
    scene_path = require_file(args.scene, "scene")
    object_reference = require_file(args.object_reference, "object reference")
    real_video = require_file(args.real_video, "real video")
    if real_video.name.lower() != "color.mp4":
        raise ValueError("real input must be the original TACO color.mp4")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output_dir}")

    pose, joints, candidate_meta = load_candidate(candidate_path, args.candidate_index)
    impedance = (
        (args.finger_kp, args.finger_kv, args.finger_cap)
        if args.physics_profile == "current" else None
    )
    backend = MujocoReplayBackend(
        scene_path, object_joint_name="right_object_joint", hand_order=("right",),
        seed=args.seed, finger_impedance=impedance,
    )
    require_scene(backend)
    physics_profile = apply_physics_profile(
        backend, args.physics_profile, args.deximit_object_mass,
    )
    object_qpos = load_object_pose(
        object_reference, expected_nq=backend.model.nq,
        object_address=backend.object_qpos_address,
    )
    human_reference = None
    manual_label = None
    object_mesh = None
    motion_transform = None
    motion_contract: dict[str, object] = {
        "kind": "diagnostic_vertical_lift",
        "vertical_lift_m": float(args.lift_m),
    }
    if args.motion_mode == "source":
        if args.human_reference is None or args.manual_label is None:
            raise ValueError("source motion requires --human-reference and --manual-label")
        human_reference = require_file(args.human_reference, "human reference")
        manual_label = require_file(args.manual_label, "manual label")
        if args.object_mesh is None:
            raise ValueError("source motion requires --object-mesh for the DexImit endpoint gate")
        object_mesh = require_file(args.object_mesh, "object mesh")
        if sha256(object_mesh) != str(candidate_meta["source_mesh_sha256"]):
            raise ValueError("object mesh does not match the candidate source mesh hash")
        motion_transform, motion_contract = source_motion_contract(
            human_reference, manual_label, object_qpos,
        )
    elif any(value is not None for value in (
        args.human_reference, args.manual_label, args.object_mesh,
    )):
        raise ValueError("human reference, manual label and object mesh are source-motion only")
    targets = [
        map_target(backend.model, object_qpos, pose[stage], joints[stage], backend.object_qpos_address)
        for stage in range(3)
    ]
    pairs = explicit_collision_pairs(
        backend.model, backend.mujoco, object_geom_ids=backend.object_geom_ids,
        hand_sides=("right",),
    )
    for target in targets:
        checked_target(backend, target, object_qpos)
    projected_targets = []
    target_limit_projection = []
    for target in targets:
        qpos, violations = project_hand_joint_limits(backend.model, target.qpos)
        projected_targets.append(Target(
            qpos, target.hand_position_world_m, target.hand_rotation_world,
        ))
        target_limit_projection.append(violations)
    static_gaps = [target_gaps(backend, target, pairs) for target in projected_targets]

    pregrasp, grasp, closure, lift = dynamic_rollout_targets(
        backend, targets, projected_targets[0].qpos,
        closure_stage=args.closure_stage, motion_transform=motion_transform,
        object_qpos=object_qpos, vertical_lift_m=args.lift_m,
    )
    backend.initialize(pregrasp, np.zeros(backend.model.nv, dtype=np.float64))
    initial_contact, initial_normal = contact_row(backend)
    initial_gaps = {
        family: minimum_pair_distance(backend.model, backend.data, backend.mujoco, values)
        for family, values in pairs.items()
    }
    trace: dict[str, list[object]] = {
        "qpos": [backend.data.qpos.copy()],
        "qvel": [backend.data.qvel.copy()],
        "ctrl": [backend.data.ctrl.copy()],
        "desired_qpos": [pregrasp.copy()],
        "control_target_qpos": [pregrasp.copy()],
        "phase": ["初始化"],
        "time_s": [float(backend.data.time)],
        "finger_contact": [initial_contact],
        "finger_normal_force_n": [initial_normal],
        "hand_self_gap_m": [initial_gaps["self"]],
        "hand_floor_gap_m": [initial_gaps["floor"]],
        "hand_object_gap_m": [initial_gaps["object"]],
    }
    inverse_data = mujoco.MjData(backend.model)
    inverse_force = np.zeros(backend.model.nv, dtype=np.float64)
    phases = []
    for name, first, second, duration in (
        (STAGE_NAMES[0], pregrasp, pregrasp, args.settle_s),
        (STAGE_NAMES[1], pregrasp, grasp, args.approach_s),
        (STAGE_NAMES[2], grasp, closure, args.squeeze_s),
        (STAGE_NAMES[3], closure, lift, args.lift_s),
        (STAGE_NAMES[4], lift, lift, args.hold_s),
    ):
        phases.append(run_phase(
            backend, inverse_data, inverse_force, trace, name=name, first=first,
            second=second, duration_s=duration, finger_kp=args.finger_kp,
            finger_kv=args.finger_kv, pairs=pairs,
        ))

    arrays = {
        name: np.asarray(values)
        for name, values in trace.items()
    }
    in_memory_replay_error = validate_physics_trace(
        backend.model, backend.mujoco, qpos=arrays["qpos"], qvel=arrays["qvel"],
        ctrl=arrays["ctrl"], time_s=arrays["time_s"],
    )
    rest_z = float(np.median(np.asarray(
        [phase["object_z_end_m"] for phase in phases[:1]], dtype=np.float64,
    )))
    hold = phases[-1]
    opposed_steps = (
        arrays["finger_contact"][:, 0]
        & arrays["finger_contact"][:, 1:].any(axis=1)
    )
    closure_or_later = np.isin(arrays["phase"], STAGE_NAMES[2:])
    hold_steps = arrays["phase"] == STAGE_NAMES[-1]
    simultaneous_opposed_contact = bool((opposed_steps & closure_or_later).any())
    hold_opposed_contact = bool((opposed_steps & hold_steps).any())
    held_lift = bool(
        float(hold["object_z_end_m"]) >= rest_z + 0.02
        and float(hold["object_z_min_m"]) >= rest_z + 0.019
    )
    hard_legal = bool(
        arrays["hand_self_gap_m"].min() >= -XHAND_SELF_FLOOR_TOLERANCE_M
        and arrays["hand_floor_gap_m"].min() >= -XHAND_SELF_FLOOR_TOLERANCE_M
    )
    hold_vertical_span_m = float(hold["object_z_peak_m"]) - float(hold["object_z_min_m"])
    stable_hold = hold_vertical_span_m <= 0.01
    minimum_object_gap_m = float(arrays["hand_object_gap_m"].min())
    object_penetration_safe = bool(minimum_object_gap_m >= -args.max_object_penetration_m)
    deximit_gate = (
        deximit_endpoint_metric(
            object_mesh,
            object_qpos,
            arrays["qpos"][-1, backend.object_qpos_address : backend.object_qpos_address + 7],
            motion_transform,
        )
        if object_mesh is not None and motion_transform is not None else None
    )
    # This intentionally demanding gate avoids treating a transient upward
    # push as a grasp.  A diagnostic pass is still never a formal success.
    diagnostic_pass = strict_grasp_gate(
        hard_legal=hard_legal,
        opposed_contact=simultaneous_opposed_contact,
        hold_opposed_contact=hold_opposed_contact,
        held_lift=held_lift,
        stable_hold=stable_hold,
        minimum_object_gap_m=minimum_object_gap_m,
        maximum_object_penetration_m=args.max_object_penetration_m,
    )
    outcome = "诊断通过" if diagnostic_pass else "诊断失败"
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_model_path = output_dir / "runtime_model.mjb"
    runtime_model, runtime_model_hash = save_runtime_model(
        backend.model, runtime_model_path,
    )
    replay_error = validate_physics_trace(
        runtime_model, backend.mujoco, qpos=arrays["qpos"], qvel=arrays["qvel"],
        ctrl=arrays["ctrl"], time_s=arrays["time_s"],
    )
    if any(
        not np.isclose(replay_error[key], in_memory_replay_error[key], atol=1.0e-15, rtol=0.0)
        for key in replay_error
    ):
        raise RuntimeError("saved runtime model replay differs from the in-memory replay")
    trace_path = output_dir / "trace.npz"
    with trace_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(SCHEMA),
            diagnostic_only=np.asarray(True),
            formal_renderer_3_3_eligible=np.asarray(False),
            candidate_index=np.asarray(args.candidate_index),
            candidate_sha256=np.asarray(sha256(candidate_path)),
            source_scene_sha256=np.asarray(sha256(scene_path)),
            runtime_model_sha256=np.asarray(runtime_model_hash),
            object_reference_sha256=np.asarray(sha256(object_reference)),
            object_pose_written_at_initialize_only=np.asarray(True),
            object_pose_writes_after_initialize=np.asarray(0, dtype=np.int64),
            direct_object_actuator_count=np.asarray(0, dtype=np.int64),
            physics_profile_json=np.asarray(json.dumps(physics_profile, sort_keys=True)),
            **arrays,
        )
    video_path: Path | None = None
    if diagnostic_pass or args.render_failed_diagnostic:
        video_path = output_dir / (
            "real_ref_sim_passed_diagnostic.mp4"
            if diagnostic_pass else "real_ref_sim_failed_diagnostic.mp4"
        )
        render_report: dict[str, object] = {
            "rendered": True,
            **render(
                scene_path, real_video, object_reference, arrays, video_path, fps=args.fps,
                outcome=outcome,
                real_contact_rows=(
                    args.real_contact_start_row,
                    args.real_stable_grasp_start_row,
                    args.real_stable_grasp_end_row,
                ),
            ),
        }
    else:
        render_report = {
            "rendered": False,
            "reason": "candidate did not pass the diagnostic grasp gate",
        }
    audit = {
        "schema": "xhand_bodex_real_ref_sim_diagnostic_v6_runtime_model_bound",
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "outcome": outcome,
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
            "index": args.candidate_index,
            **candidate_meta,
        },
        "scene": str(scene_path),
        "scene_sha256": sha256(scene_path),
        "runtime_model": {
            "path": str(runtime_model_path),
            "sha256": runtime_model_hash,
            "format": "MuJoCo compiled MJB containing all diagnostic runtime edits",
            "independent_replay_max_abs_error": replay_error,
        },
        "object_initialization": {
            "reference": str(object_reference),
            "reference_sha256": sha256(object_reference),
            "one_time_free_joint_qpos_wxyz": object_qpos.tolist(),
            "pose_writes_after_initialize": 0,
            "direct_object_actuators": 0,
        },
        "coordinate_contract": {
            "bodex_pose": "object-local BODex URDF base pose, metres, wxyz",
            "xhand_pose": "current MuJoCo right_hand_link root",
            "mapping_root_position_error_m_max": 1.0e-9,
            "mapping_root_rotation_error_rad_max": 1.0e-9,
            "joint_order_checked": list(candidate_meta["joint_order"]),
            "joint_axis_sign_mapping": {
                name: float(sign) for name, sign in zip(
                    CANONICAL_JOINTS, BODEX_TO_MUJOCO_JOINT_SIGN,
                )
            },
            "reason_for_index_sign_flip": (
                "BODex/DexImit URDF index abduction axis is 0 -1 0; "
                "current MuJoCo axis is 0 1 0"
            ),
            "target_joint_limit_projection_rad": dict(zip(
                ("pregrasp", "grasp", "squeeze"), target_limit_projection,
            )),
        },
        "projected_bodex_static_target_gaps_m": static_gaps,
        "controller": {
            "kind": (
                "independent C1 inverse-dynamics hand feedforward with bounded XHand effort"
                if args.physics_profile == "current"
                else "independent C1 inverse-dynamics targets with DexImit-style position drives"
            ),
            "bounded_finger_impedance": (
                [args.finger_kp, args.finger_kv, args.finger_cap]
                if args.physics_profile == "current" else None
            ),
            "object_control": "none",
            "closure_stage": args.closure_stage,
            "maximum_hand_object_penetration_m": args.max_object_penetration_m,
            "physics_profile": physics_profile,
            "motion_mode": args.motion_mode,
            "motion_contract": motion_contract,
        },
        "human_reference": {
            "path": str(human_reference) if human_reference else None,
            "sha256": sha256(human_reference) if human_reference else None,
        },
        "manual_label": {
            "path": str(manual_label) if manual_label else None,
            "sha256": sha256(manual_label) if manual_label else None,
        },
        "object_mesh": {
            "path": str(object_mesh) if object_mesh else None,
            "sha256": sha256(object_mesh) if object_mesh else None,
        },
        "deximit_endpoint_gate": deximit_gate,
        "phases": phases,
        "physics": {
            "trace": str(trace_path),
            "trace_sha256": sha256(trace_path),
            "physics_dt_s": float(backend.model.opt.timestep),
            "replay_max_abs_error": replay_error,
            "minimum_hand_self_gap_m": float(arrays["hand_self_gap_m"].min()),
            "minimum_hand_floor_gap_m": float(arrays["hand_floor_gap_m"].min()),
            "minimum_hand_object_gap_m": minimum_object_gap_m,
            "object_penetration_gate": object_penetration_safe,
            "rest_object_z_m": rest_z,
            "held_lift_gate": held_lift,
            "simultaneous_thumb_and_other_observed": simultaneous_opposed_contact,
            "hold_simultaneous_thumb_and_other_observed": hold_opposed_contact,
            "hold_vertical_span_m": hold_vertical_span_m,
            "stable_hold_gate": stable_hold,
            "hard_legal": hard_legal,
        },
        "video": {
            "path": str(video_path) if video_path is not None else None,
            "camera": (
                "场景固定第三人称相机 front；不允许手、物体或接触点跟随取景"
            ),
            "meaning": {
                "真人": "原始 TACO color.mp4；按已审计接触阶段取行，不与物理轨迹逐帧同步",
                "参考": "正式 ref.npz 在真人栏同一源行的机器人参考；不是 BODex 目标",
                "仿真": "已记录的 MuJoCo 自由物体状态；物体只在初始化写入一次，随后只受接触和重力影响",
            },
            **render_report,
        },
        "limitations": [
            "BODex 输出是几何抓姿候选，不是当前项目的 source-timed 参考轨迹。",
            "该视频不宣称真人动作和仿真动作逐帧一致，且不能进入正式 renderer/3.3。",
            "BODex 抓紧目标与当前 MuJoCo 碰撞壳存在差异，原始静态间隙已原样写入本审计。",
        ],
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir), "trace": str(trace_path),
        "video": str(video_path) if video_path is not None else None, "audit": str(audit_path),
        "outcome": outcome, "replay_error": replay_error,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
