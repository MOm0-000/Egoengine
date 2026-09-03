#!/usr/bin/env python3
"""Replay DexImit's recorded SAPIEN hand trajectory in isolated MuJoCo.

Only the hand pose and finger joints cross the engine boundary.  The MuJoCo
object is initialized once and then remains free, so the SAPIEN object trace is
used exclusively for comparison.  A video is produced only after the strict
MuJoCo grasp gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egoengine_repro.action.contracts import XHAND_SELF_FLOOR_TOLERANCE_M
from egoengine_repro.action.geometry import explicit_collision_pairs, minimum_pair_distance
from egoengine_repro.action.replay import MujocoReplayBackend, validate_physics_trace
from bodex_triptych import (
    BODEX_TO_MUJOCO_JOINT_SIGN,
    CANONICAL_JOINTS,
    DEFAULT_MAX_OBJECT_PENETRATION_M,
    FINGERS,
    ROOT_JOINTS,
    contact_row,
    deximit_endpoint_metric,
    load_object_pose,
    pose_matrix,
    render,
    require_file,
    require_scene,
    save_runtime_model,
    scalar_joint_qpos_addresses,
    sha256,
    source_motion_contract,
    strict_grasp_gate,
)
from generate_bodex_candidates import SAPIEN_TO_CANONICAL
from sapien_equiv import apply_profile, read_body


TRACE_SCHEMA_V1 = "deximit_sapien_hand_trace_v1_diagnostic_only"
TRACE_SCHEMA_V2 = "deximit_sapien_hand_trace_v2_post_step_diagnostic_only"
SAPIEN_SUMMARY_SCHEMA = (
    "deximit_original_sapien_screen_v4_physical_contact_metrics_diagnostic_only"
)
OUT_SCHEMA = "deximit_hand_trace_mujoco_replay_v4_sapien_equivalent_diagnostic_only"
PHASE_MAP = {
    "pregrasp": "预抓取",
    "grasp": "接近",
    "squeeze": "收紧",
    "demonstrated_object_motion": "抬起",
    "hold": "保持",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sapien-trace", type=Path, required=True)
    parser.add_argument("--sapien-summary", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--object-reference", type=Path, required=True)
    parser.add_argument("--human-reference", type=Path, required=True)
    parser.add_argument("--manual-label", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--real-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--max-object-penetration-m", type=float,
        default=DEFAULT_MAX_OBJECT_PENETRATION_M,
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--contact-scale", type=float, default=1.0,
        help=(
            "Multiplier on the MuJoCo acceleration-level stiffness derived from "
            "SAPIEN's measured peak impulse and penetration."
        ),
    )
    parser.add_argument(
        "--target-source", choices=("drive", "observed"), default="drive",
        help=(
            "drive tests converted controller equivalence; observed replays SAPIEN's "
            "actual hand state to isolate contact-solver equivalence."
        ),
    )
    return parser.parse_args()


def apply_contact_equivalence(
    backend: MujocoReplayBackend, metrics: dict[str, object], *, scale: float,
) -> dict[str, object]:
    """Convert recorded PhysX contact behavior into MuJoCo pair parameters."""
    body = read_body(metrics)
    model, mj = backend.model, backend.mujoco
    rest_distance = 2.0 * body.rest_offset
    detection_distance = 2.0 * body.contact_offset
    if detection_distance < rest_distance:
        raise ValueError("SAPIEN contact distance is smaller than its rest distance")
    version = tuple(int(part) for part in mj.__version__.split(".")[:2])
    if version >= (3, 9):
        pair_margin, pair_gap = rest_distance, detection_distance - rest_distance
        margin_semantics = "MuJoCo 3.9+: force at margin, detect at margin+gap"
    else:
        pair_margin, pair_gap = detection_distance, detection_distance - rest_distance
        margin_semantics = "MuJoCo <=3.8: detect at margin, force at margin-gap"
    # This value was selected before any candidate replay, using the isolated
    # eraser drop and slide probes.  ``scale`` changes stiffness in the same
    # direction as the old option without reusing candidate peak impulses.
    time_constant = 0.0075 / float(np.sqrt(scale))
    pairs: list[str] = []
    for pair in range(model.npair):
        geom1, geom2 = int(model.pair_geom1[pair]), int(model.pair_geom2[pair])
        if geom1 not in backend.object_geom_ids and geom2 not in backend.object_geom_ids:
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_PAIR, pair) or f"pair_{pair}"
        model.pair_margin[pair] = pair_margin
        model.pair_gap[pair] = pair_gap
        is_table = name.startswith(("floor_", "table_"))
        # PhysX's default material combination is the arithmetic mean.  The
        # table uses 1.0/1.0 and the object 0.7/0.5, while robot and object both
        # use 0.7/0.5.
        dynamic_friction = 0.75 if is_table else 0.5
        model.pair_friction[pair, 0:2] = dynamic_friction
        model.pair_solref[pair, :] = (time_constant, 1.0)
        pairs.append(name)
    if not pairs:
        raise RuntimeError("no explicit object contact pair accepted SAPIEN conversion")
    return {
        "changed_pairs": pairs,
        "changed_pair_count": len(pairs),
        "mujoco_pair_margin_m": pair_margin,
        "mujoco_pair_gap_m": pair_gap,
        "mujoco_margin_gap_semantics": margin_semantics,
        "physx_pair_detection_distance_m": detection_distance,
        "physx_pair_rest_distance_m": rest_distance,
        "mujoco_contact_time_constant_s": time_constant,
        "contact_calibration_source": "candidate-free eraser drop and slide micro probe",
        "mujoco_hand_object_dynamic_friction": 0.5,
        "mujoco_table_object_dynamic_friction": 0.75,
        "physx_static_friction": body.static_friction,
        "physx_dynamic_friction": body.dynamic_friction,
        "friction_policy": "arithmetic material combination; kinetic coefficient in XML",
        "stiffness_semantics": "MuJoCo positive solref time constant and damping ratio",
    }


def settle_object(
    backend: MujocoReplayBackend, initial_qpos: np.ndarray, *, duration_s: float = 5.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the free object to a stable table pose before the measured rollout.

    DexImit performs an equivalent warm-up before candidate execution.  This
    uses a separate ``MjData`` so the recorded rollout still begins at time
    zero and contains no object-pose writes after initialization.
    """
    model, mj = backend.model, backend.mujoco
    qpos = np.asarray(initial_qpos, dtype=np.float64)
    if qpos.shape != (model.nq,) or duration_s <= 0.0:
        raise ValueError("object warm-up state or duration is invalid")
    data = mj.MjData(model)
    warm_qpos = qpos.copy()
    root = scalar_joint_qpos_addresses(model, ROOT_JOINTS)
    object_xyz = qpos[
        backend.object_qpos_address : backend.object_qpos_address + 3
    ]
    warm_qpos[root[:3]] = object_xyz + np.asarray((1.0, 0.0, 1.0))
    data.qpos[:] = warm_qpos
    data.qvel[:] = 0.0
    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    object_floor_pairs = tuple(
        (int(model.pair_geom1[pair]), int(model.pair_geom2[pair]))
        for pair in range(model.npair)
        if {
            int(model.pair_geom1[pair]), int(model.pair_geom2[pair]),
        } == {floor_id, *backend.object_geom_ids}
    )
    if len(object_floor_pairs) != 1:
        raise RuntimeError(
            f"expected one explicit object-floor pair, found {object_floor_pairs}",
        )
    floor_geoms = frozenset(object_floor_pairs[0])

    def has_active_floor_contact(lift: float) -> bool:
        data.qpos[:] = warm_qpos
        data.qpos[backend.object_qpos_address + 2] += float(lift)
        data.qvel[:] = 0.0
        mj.mj_forward(model, data)
        return any(
            frozenset((int(data.contact[index].geom1), int(data.contact[index].geom2)))
            == floor_geoms
            and int(data.contact[index].efc_address) >= 0
            for index in range(data.ncon)
        )

    clearance_lift = 0.0
    if has_active_floor_contact(0.0):
        low, high = 0.0, 0.001
        while has_active_floor_contact(high):
            high *= 2.0
            if high > 0.2:
                raise RuntimeError("could not place object above the MuJoCo floor")
        for _ in range(48):
            middle = 0.5 * (low + high)
            if has_active_floor_contact(middle):
                low = middle
            else:
                high = middle
        clearance_lift = high + 5.0e-5
    warm_qpos[backend.object_qpos_address + 2] += clearance_lift
    data.qpos[:] = warm_qpos
    data.qvel[:] = 0.0
    target = backend.reference_action(warm_qpos)
    data.ctrl[:] = target
    mj.mj_forward(model, data)
    initial_floor_gap = minimum_pair_distance(
        model, data, mj, object_floor_pairs,
    )
    count = int(np.ceil(duration_s / float(model.opt.timestep)))
    tail_positions: list[np.ndarray] = []
    for step in range(count):
        data.ctrl[:] = target
        mj.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("MuJoCo object warm-up became non-finite")
        if step >= count - 60:
            tail_positions.append(
                data.qpos[
                    backend.object_qpos_address : backend.object_qpos_address + 3
                ].copy()
            )
    mj.mj_forward(model, data)
    if any(int(item.number) > 0 for item in data.warning):
        raise FloatingPointError("MuJoCo object warm-up raised a numerical warning")
    address = backend.object_qpos_address
    joint_dof = int(model.jnt_dofadr[backend.object_joint_id])
    settled = data.qpos[address : address + 7].copy()
    velocity = data.qvel[joint_dof : joint_dof + 6].copy()
    tail = np.asarray(tail_positions)
    translation = float(np.linalg.norm(settled[:3] - qpos[address : address + 3]))
    initial_rotation = Rotation.from_quat(qpos[address + 3 : address + 7], scalar_first=True)
    rotation = float((
        Rotation.from_quat(settled[3:], scalar_first=True) * initial_rotation.inv()
    ).magnitude())
    final_floor_gap = minimum_pair_distance(
        model, data, mj, object_floor_pairs,
    )
    position_span = float(np.linalg.norm(tail.max(axis=0) - tail.min(axis=0)))
    stable = bool(
        np.linalg.norm(velocity[:3]) <= 1.0e-3
        and np.linalg.norm(velocity[3:]) <= 0.05
        and position_span <= 5.0e-4
    )
    if not stable:
        raise RuntimeError(
            "MuJoCo object did not settle before replay: "
            f"clearance_lift={clearance_lift:g}, initial_floor_gap={initial_floor_gap:g}, "
            f"final_floor_gap={final_floor_gap:g}, velocity={velocity.tolist()}, "
            f"position_span={position_span:g}",
        )
    return settled, {
        "kind": "free-object pre-roll equivalent to DexImit warm-up",
        "duration_s": duration_s,
        "physics_steps": count,
        "hand_isolated_during_warmup": True,
        "initial_nonpenetration_lift_m": clearance_lift,
        "initial_floor_gap_m": float(initial_floor_gap),
        "final_floor_gap_m": float(final_floor_gap),
        "position_change_m": translation,
        "rotation_change_rad": rotation,
        "final_free_joint_velocity": velocity.tolist(),
        "tail_position_span_m": position_span,
        "stable": stable,
        "recorded_rollout_clock_includes_warmup": False,
        "object_pose_writes_after_recorded_initialize": 0,
    }


def calibrate_floor_contact(
    backend: MujocoReplayBackend, initial_qpos: np.ndarray,
    contact: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Validate the candidate-free table-contact profile with one warm-up."""
    model, mj = backend.model, backend.mujoco
    floor_pairs = [
        pair for pair in range(model.npair)
        if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_PAIR, pair) or "").startswith("floor_")
        and (
            int(model.pair_geom1[pair]) in backend.object_geom_ids
            or int(model.pair_geom2[pair]) in backend.object_geom_ids
        )
    ]
    if len(floor_pairs) != 1:
        raise RuntimeError(f"expected one object-floor pair, found {floor_pairs}")
    pair = floor_pairs[0]
    settled, warmup = settle_object(backend, initial_qpos)
    return settled, {
        "method": "single validation of candidate-free drop/slide calibration",
        "selected_time_constant_s": float(model.pair_solref[pair, 0]),
        "selected_damping_ratio": float(model.pair_solref[pair, 1]),
        "warmup": warmup,
    }


def trace_contract(path: Path) -> dict[str, np.ndarray | float | int]:
    with np.load(path, allow_pickle=False) as values:
        schema = str(np.asarray(values["schema"]).item())
        if (
            schema not in {TRACE_SCHEMA_V1, TRACE_SCHEMA_V2}
            or not bool(np.asarray(values["diagnostic_only"]).item())
            or bool(np.asarray(values["formal_renderer_3_3_eligible"]).item())
        ):
            raise ValueError("input is not an isolated DexImit SAPIEN hand trace")
        result: dict[str, np.ndarray | float | int] = {
            "schema": schema,
            "candidate_index": int(np.asarray(values["candidate_index"]).item()),
            "pool_depth": int(np.asarray(values["pool_depth"]).item()),
            "physics_dt_s": float(np.asarray(values["physics_dt_s"]).item()),
            "frame_skip": int(np.asarray(values["control_frame_skip"]).item()),
            "phase": np.asarray(values["phase"]).astype("U32"),
            "object_pose": np.asarray(values["object_pose_sapien_wxyz"], dtype=np.float64),
            "hand_pose": np.asarray(
                values["right_hand_link_pose_sapien_wxyz"], dtype=np.float64,
            ),
            "robot_qpos": np.asarray(values["right_robot_qpos_sapien"], dtype=np.float64),
            "drive_target": np.asarray(
                values["right_drive_target_sapien"], dtype=np.float64,
            ),
            "hand_drive_pose": np.asarray(
                values["right_hand_link_drive_pose_sapien_wxyz"], dtype=np.float64,
            ),
        }
        if schema == TRACE_SCHEMA_V2:
            if str(np.asarray(values["sample_semantics"]).item()) != (
                "post-step state under same-index drive target"
            ):
                raise ValueError("SAPIEN v2 trace has unknown sample semantics")
            result.update({
                "sample_time": np.asarray(values["physics_sample_time_s"], dtype=np.float64),
                "robot_qvel": np.asarray(values["right_robot_qvel_sapien"], dtype=np.float64),
                "hand_linear_velocity": np.asarray(
                    values["right_hand_link_linear_velocity_sapien"], dtype=np.float64,
                ),
                "hand_angular_velocity": np.asarray(
                    values["right_hand_link_angular_velocity_sapien"], dtype=np.float64,
                ),
                "object_linear_velocity": np.asarray(
                    values["object_linear_velocity_sapien"], dtype=np.float64,
                ),
                "object_angular_velocity": np.asarray(
                    values["object_angular_velocity_sapien"], dtype=np.float64,
                ),
            })
    count = len(result["phase"])
    dt = float(result["physics_dt_s"])
    if schema == TRACE_SCHEMA_V1:
        result["sample_time"] = (np.arange(count, dtype=np.float64) + 1.0) * dt
    sample_time = np.asarray(result["sample_time"], dtype=np.float64)
    if (
        count < 2
        or result["object_pose"].shape != (count, 7)
        or result["hand_pose"].shape != (count, 7)
        or result["robot_qpos"].shape != (count, 18)
        or result["drive_target"].shape != (count, 18)
        or result["hand_drive_pose"].shape != (count, 7)
        or not np.isfinite(result["object_pose"]).all()
        or not np.isfinite(result["hand_pose"]).all()
        or not np.isfinite(result["robot_qpos"]).all()
        or not np.isfinite(result["drive_target"]).all()
        or not np.isfinite(result["hand_drive_pose"]).all()
        or set(np.unique(result["phase"]).tolist()) != set(PHASE_MAP)
        or not np.isclose(result["physics_dt_s"], 1.0 / 240.0, atol=1.0e-12)
        or result["frame_skip"] != 12
        or sample_time.shape != (count,)
        or not np.isfinite(sample_time).all()
        or not np.allclose(
            sample_time, (np.arange(count, dtype=np.float64) + 1.0) * dt,
            atol=1.0e-10, rtol=0.0,
        )
    ):
        raise ValueError("SAPIEN hand trace arrays violate the released timing contract")
    if schema == TRACE_SCHEMA_V2 and (
        np.asarray(result["robot_qvel"]).shape != (count, 18)
        or np.asarray(result["hand_linear_velocity"]).shape != (count, 3)
        or np.asarray(result["hand_angular_velocity"]).shape != (count, 3)
        or np.asarray(result["object_linear_velocity"]).shape != (count, 3)
        or np.asarray(result["object_angular_velocity"]).shape != (count, 3)
        or not all(np.isfinite(np.asarray(result[name])).all() for name in (
            "robot_qvel", "hand_linear_velocity", "hand_angular_velocity",
            "object_linear_velocity", "object_angular_velocity",
        ))
    ):
        raise ValueError("SAPIEN v2 velocity arrays are malformed")
    return result


def summary_contract(path: Path, trace_path: Path, trace: dict[str, object]) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != SAPIEN_SUMMARY_SCHEMA
        or summary.get("diagnostic_only") is not True
        or summary.get("formal_renderer_3_3_eligible") is not False
        or int(summary.get("original_sapien_pass_count", 0)) < 1
    ):
        raise ValueError("SAPIEN summary does not contain an original diagnostic pass")
    matches = [
        row for row in summary.get("original_sapien_passes", [])
        if int(row.get("source_candidate_index", -1)) == int(trace["candidate_index"])
        and int(row.get("depth", -1)) == int(trace["pool_depth"])
    ]
    if len(matches) != 1:
        raise ValueError("SAPIEN summary does not uniquely identify the exported trace")
    row = matches[0]
    metrics = row.get("metrics", {})
    if (
        Path(row.get("exported_trajectory", "")).resolve() != trace_path
        or row.get("exported_trajectory_sha256") != sha256(trace_path)
        or not str(metrics.get("contact_step_definition", "")).startswith(
            "load-bearing object contact"
        )
    ):
        raise ValueError("SAPIEN summary trace or physical-contact provenance does not match")
    initial = np.asarray(row["metrics"]["object_initial_pose"], dtype=np.float64)
    if initial.shape != (4, 4) or not np.isfinite(initial).all():
        raise ValueError("SAPIEN summary lacks a finite initial object transform")
    return {
        "row": row,
        "object_initial": initial,
        "sapien_endpoint_error_m": float(row["metrics"]["mean_target_vertex_error_m"]),
    }


def mapped_hand_targets(
    model: mujoco.MjModel, hand_pose: np.ndarray, robot_qpos: np.ndarray,
    align: np.ndarray, object_qpos: np.ndarray, object_address: int,
) -> tuple[np.ndarray, dict[str, float]]:
    count = len(hand_pose)
    result = np.repeat(model.qpos0[None], count, axis=0)
    result[:, object_address : object_address + 7] = object_qpos
    root_addresses = scalar_joint_qpos_addresses(model, ROOT_JOINTS)
    finger_addresses = scalar_joint_qpos_addresses(model, CANONICAL_JOINTS)
    canonical = robot_qpos[:, 6:][:, SAPIEN_TO_CANONICAL]
    result[:, finger_addresses] = canonical * BODEX_TO_MUJOCO_JOINT_SIGN
    for index, pose in enumerate(hand_pose):
        target = align @ pose_matrix(pose)
        roll, pitch, negative_yaw = Rotation.from_matrix(target[:3, :3]).as_euler("ZXY")
        result[index, root_addresses[:3]] = target[:3, 3]
        result[index, root_addresses[3:]] = (roll, pitch, -negative_yaw)

    maximum_limit_violation = 0.0
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        if not name.startswith("right_hand_") or not bool(model.jnt_limited[joint]):
            continue
        address = int(model.jnt_qposadr[joint])
        low, high = (float(value) for value in model.jnt_range[joint])
        maximum_limit_violation = max(
            maximum_limit_violation,
            float(np.maximum(low - result[:, address], 0.0).max()),
            float(np.maximum(result[:, address] - high, 0.0).max()),
        )
    if maximum_limit_violation > 1.0e-6:
        raise ValueError(
            f"recorded SAPIEN hand exceeds MuJoCo joint limits by {maximum_limit_violation:g} rad",
        )

    data = mujoco.MjData(model)
    hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link")
    position_error = rotation_error = 0.0
    for index in np.linspace(0, count - 1, min(count, 65), dtype=np.int64):
        data.qpos[:] = result[index]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        expected = align @ pose_matrix(hand_pose[index])
        position_error = max(
            position_error,
            float(np.linalg.norm(data.xpos[hand_body] - expected[:3, 3])),
        )
        rotation_error = max(
            rotation_error,
            float(Rotation.from_matrix(
                expected[:3, :3].T @ data.xmat[hand_body].reshape(3, 3),
            ).magnitude()),
        )
    if position_error > 1.0e-9 or rotation_error > 1.0e-9:
        raise RuntimeError("SAPIEN hand-link trajectory does not map exactly to MuJoCo")
    return result, {
        "sampled_root_position_error_m_max": position_error,
        "sampled_root_rotation_error_rad_max": rotation_error,
        "finger_joint_limit_violation_rad_max": maximum_limit_violation,
    }


def mapped_initial_velocity(
    backend: MujocoReplayBackend, source: dict[str, object], align: np.ndarray,
    initial_qpos: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Map a v2 post-step SAPIEN state into MuJoCo generalized velocity."""
    velocity = np.zeros(backend.model.nv, dtype=np.float64)
    if source["schema"] == TRACE_SCHEMA_V1:
        return velocity, {
            "available": False,
            "reason": "legacy v1 SAPIEN trace did not record physical velocities",
        }
    model, data, mj = backend.model, backend.data, backend.mujoco
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)
    world_rotation = align[:3, :3]
    hand_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_hand_link")
    root_joints = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in ROOT_JOINTS
    ]
    root_dofs = np.asarray([int(model.jnt_dofadr[joint]) for joint in root_joints])
    jac_position = np.zeros((3, model.nv), dtype=np.float64)
    jac_rotation = np.zeros((3, model.nv), dtype=np.float64)
    mj.mj_jacBody(model, data, jac_position, jac_rotation, hand_body)
    hand_jacobian = np.vstack((jac_position, jac_rotation))[:, root_dofs]
    hand_twist = np.concatenate((
        world_rotation @ np.asarray(source["hand_linear_velocity"])[0],
        world_rotation @ np.asarray(source["hand_angular_velocity"])[0],
    ))
    def solve_twist(jacobian: np.ndarray, twist: np.ndarray, label: str) -> tuple[np.ndarray, float]:
        condition = float(np.linalg.cond(jacobian))
        if not np.isfinite(condition) or condition > 1.0e10:
            raise ValueError(
                f"{label} velocity mapping is singular or ill-conditioned: {condition:g}",
            )
        solution = np.linalg.solve(jacobian, twist)
        residual = float(np.linalg.norm(jacobian @ solution - twist))
        if not np.isfinite(solution).all() or residual > 1.0e-9:
            raise ValueError(f"{label} velocity mapping residual is too large: {residual:g}")
        return solution, condition

    velocity[root_dofs], hand_condition = solve_twist(
        hand_jacobian, hand_twist, "hand-root",
    )

    finger_joints = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        for name in CANONICAL_JOINTS
    ]
    finger_dofs = np.asarray([int(model.jnt_dofadr[joint]) for joint in finger_joints])
    source_finger_velocity = np.asarray(source["robot_qvel"])[0, 6:][SAPIEN_TO_CANONICAL]
    velocity[finger_dofs] = source_finger_velocity * BODEX_TO_MUJOCO_JOINT_SIGN

    object_dof = int(model.jnt_dofadr[backend.object_joint_id])
    object_dofs = np.arange(object_dof, object_dof + 6, dtype=np.int64)
    jac_position.fill(0.0)
    jac_rotation.fill(0.0)
    mj.mj_jacBody(
        model, data, jac_position, jac_rotation, backend.object_body_id,
    )
    object_jacobian = np.vstack((jac_position, jac_rotation))[:, object_dofs]
    object_twist = np.concatenate((
        world_rotation @ np.asarray(source["object_linear_velocity"])[0],
        world_rotation @ np.asarray(source["object_angular_velocity"])[0],
    ))
    velocity[object_dofs], object_condition = solve_twist(
        object_jacobian, object_twist, "object-free-joint",
    )
    return velocity, {
        "available": True,
        "hand_jacobian_condition_number": hand_condition,
        "object_jacobian_condition_number": object_condition,
        "hand_twist_reconstruction_error": float(np.linalg.norm(
            hand_jacobian @ velocity[root_dofs] - hand_twist,
        )),
        "object_twist_reconstruction_error": float(np.linalg.norm(
            object_jacobian @ velocity[object_dofs] - object_twist,
        )),
    }


def main() -> int:
    args = parse_args()
    if (
        args.max_object_penetration_m <= 0.0 or args.fps <= 0
        or not np.isfinite(args.contact_scale) or args.contact_scale <= 0.0
    ):
        raise ValueError("penetration threshold, fps and contact scale must be positive")
    paths = {
        name: require_file(value, name.replace("_", " "))
        for name, value in {
            "sapien_trace": args.sapien_trace,
            "sapien_summary": args.sapien_summary,
            "scene": args.scene,
            "object_reference": args.object_reference,
            "human_reference": args.human_reference,
            "manual_label": args.manual_label,
            "object_mesh": args.object_mesh,
            "real_video": args.real_video,
        }.items()
    }
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")

    source = trace_contract(paths["sapien_trace"])
    source_summary = summary_contract(paths["sapien_summary"], paths["sapien_trace"], source)
    backend = MujocoReplayBackend(
        paths["scene"], object_joint_name="right_object_joint", hand_order=("right",),
        seed=args.seed,
    )
    require_scene(backend)
    object_qpos = load_object_pose(
        paths["object_reference"], expected_nq=backend.model.nq,
        object_address=backend.object_qpos_address,
    )
    profile_qpos = backend.model.qpos0.copy()
    profile_qpos[
        backend.object_qpos_address : backend.object_qpos_address + 7
    ] = object_qpos
    sapien_metrics = source_summary["row"]["metrics"]
    physics_profile = apply_profile(backend, sapien_metrics, profile_qpos)
    contact_override = apply_contact_equivalence(
        backend, sapien_metrics, scale=args.contact_scale,
    )
    object_qpos, floor_calibration = calibrate_floor_contact(
        backend, profile_qpos, contact_override,
    )
    contact_override["floor_calibration"] = floor_calibration
    physics_profile["object_initialization"] = floor_calibration["warmup"]
    dt = float(source["physics_dt_s"])
    if not np.isclose(backend.model.opt.timestep, dt, atol=1.0e-12):
        raise ValueError("MuJoCo and exported SAPIEN physics timesteps differ")
    sapien_initial = np.asarray(source_summary["object_initial"], dtype=np.float64)
    mujoco_initial = pose_matrix(object_qpos)
    align = mujoco_initial @ np.linalg.inv(sapien_initial)
    drive_targets, target_mapping = mapped_hand_targets(
        backend.model,
        np.asarray(source["hand_drive_pose"]),
        np.asarray(source["drive_target"]),
        align,
        object_qpos,
        backend.object_qpos_address,
    )
    observed, observed_mapping = mapped_hand_targets(
        backend.model,
        np.asarray(source["hand_pose"]),
        np.asarray(source["robot_qpos"]),
        align,
        object_qpos,
        backend.object_qpos_address,
    )
    desired = drive_targets if args.target_source == "drive" else observed.copy()
    phases = np.asarray([PHASE_MAP[str(value)] for value in source["phase"]])
    pairs = explicit_collision_pairs(
        backend.model, backend.mujoco, object_geom_ids=backend.object_geom_ids,
        hand_sides=("right",),
    )
    initial_velocity, velocity_mapping = mapped_initial_velocity(
        backend, source, align, observed[0],
    )
    backend.initialize(observed[0], initial_velocity)
    backend.data.time = float(np.asarray(source["sample_time"])[0])
    backend.data.ctrl[:] = backend.reference_action(desired[0])
    backend.mujoco.mj_forward(backend.model, backend.data)
    trace: dict[str, list[object]] = {
        "qpos": [], "qvel": [], "ctrl": [], "desired_qpos": [],
        "phase": [], "time_s": [], "finger_contact": [],
        "finger_normal_force_n": [], "hand_self_gap_m": [],
        "hand_floor_gap_m": [], "hand_object_gap_m": [],
    }

    def capture(target: np.ndarray, phase: str) -> None:
        touched, normal = contact_row(backend)
        gaps = {
            family: minimum_pair_distance(
                backend.model, backend.data, backend.mujoco, values,
            )
            for family, values in pairs.items()
        }
        trace["qpos"].append(backend.data.qpos.copy())
        trace["qvel"].append(backend.data.qvel.copy())
        trace["ctrl"].append(backend.data.ctrl.copy())
        trace["desired_qpos"].append(target.copy())
        trace["phase"].append(phase)
        trace["time_s"].append(float(backend.data.time))
        trace["finger_contact"].append(touched)
        trace["finger_normal_force_n"].append(normal)
        trace["hand_self_gap_m"].append(gaps["self"])
        trace["hand_floor_gap_m"].append(gaps["floor"])
        trace["hand_object_gap_m"].append(gaps["object"])

    capture(desired[0], str(phases[0]))
    for index in range(1, len(desired)):
        backend.step(backend.reference_action(desired[index]), dt)
        if (
            not np.isfinite(backend.data.qpos).all()
            or not np.isfinite(backend.data.qvel).all()
            or any(int(item.number) > 0 for item in backend.data.warning)
        ):
            raise FloatingPointError(f"MuJoCo replay became unstable at source step {index}")
        capture(desired[index], str(phases[index]))

    arrays = {name: np.asarray(values) for name, values in trace.items()}
    in_memory_replay_error = validate_physics_trace(
        backend.model, backend.mujoco,
        qpos=arrays["qpos"], qvel=arrays["qvel"], ctrl=arrays["ctrl"],
        time_s=arrays["time_s"],
    )
    pregrasp = np.flatnonzero(arrays["phase"] == "预抓取")
    hold = np.flatnonzero(arrays["phase"] == "保持")
    if not len(pregrasp) or not len(hold):
        raise ValueError("cross-engine trace lacks pregrasp or hold samples")
    rest_window = pregrasp[-min(30, len(pregrasp)) :]
    object_z = arrays["qpos"][:, backend.object_qpos_address + 2]
    rest_z = float(np.median(object_z[rest_window]))
    opposed = arrays["finger_contact"][:, 0] & arrays["finger_contact"][:, 1:].any(axis=1)
    closure_or_later = np.isin(arrays["phase"], ("收紧", "抬起", "保持"))
    opposed_contact = bool((opposed & closure_or_later).any())
    hold_opposed = bool(opposed[hold].any())
    hold_min_lift = float(object_z[hold].min() - rest_z)
    hold_end_lift = float(object_z[hold[-1]] - rest_z)
    held_lift = bool(hold_min_lift >= 0.019 and hold_end_lift >= 0.02)
    hold_span = float(object_z[hold].max() - object_z[hold].min())
    stable_hold = hold_span <= 0.01
    minimum_object_gap = float(arrays["hand_object_gap_m"].min())
    hard_legal = bool(
        arrays["hand_self_gap_m"].min() >= -XHAND_SELF_FLOOR_TOLERANCE_M
        and arrays["hand_floor_gap_m"].min() >= -XHAND_SELF_FLOOR_TOLERANCE_M
    )
    passed = strict_grasp_gate(
        hard_legal=hard_legal,
        opposed_contact=opposed_contact,
        hold_opposed_contact=hold_opposed,
        held_lift=held_lift,
        stable_hold=stable_hold,
        minimum_object_gap_m=minimum_object_gap,
        maximum_object_penetration_m=args.max_object_penetration_m,
    )
    motion_transform, motion_contract = source_motion_contract(
        paths["human_reference"], paths["manual_label"], object_qpos,
    )
    endpoint = deximit_endpoint_metric(
        paths["object_mesh"], object_qpos,
        arrays["qpos"][-1, backend.object_qpos_address : backend.object_qpos_address + 7],
        motion_transform,
    )
    aligned_sapien_object = np.asarray([
        pose_matrix(pose) for pose in np.asarray(source["object_pose"])
    ])
    aligned_sapien_object = align[None] @ aligned_sapien_object
    mujoco_object_position = arrays["qpos"][:, backend.object_qpos_address : backend.object_qpos_address + 3]
    position_difference = np.linalg.norm(
        mujoco_object_position - aligned_sapien_object[:, :3, 3], axis=1,
    )
    robot_addresses = backend.actuator_qpos_addresses
    tracking_error = np.linalg.norm(
        arrays["qpos"][:, robot_addresses] - desired[:, robot_addresses], axis=1,
    )
    observed_error = np.linalg.norm(
        arrays["qpos"][:, robot_addresses] - observed[:, robot_addresses], axis=1,
    )

    output.mkdir(parents=True, exist_ok=False)
    runtime_model_path = output / "runtime_model.mjb"
    runtime_model, runtime_model_hash = save_runtime_model(
        backend.model, runtime_model_path,
    )
    replay_error = validate_physics_trace(
        runtime_model, backend.mujoco,
        qpos=arrays["qpos"], qvel=arrays["qvel"], ctrl=arrays["ctrl"],
        time_s=arrays["time_s"],
    )
    if any(
        not np.isclose(replay_error[key], in_memory_replay_error[key], atol=1.0e-15, rtol=0.0)
        for key in replay_error
    ):
        raise RuntimeError("saved runtime model replay differs from the in-memory replay")
    trace_path = output / "trace.npz"
    with trace_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(OUT_SCHEMA),
            diagnostic_only=np.asarray(True),
            formal_renderer_3_3_eligible=np.asarray(False),
            source_trace_sha256=np.asarray(sha256(paths["sapien_trace"])),
            source_scene_sha256=np.asarray(sha256(paths["scene"])),
            runtime_model_sha256=np.asarray(runtime_model_hash),
            physics_profile_json=np.asarray(json.dumps(physics_profile, sort_keys=True)),
            contact_override_json=np.asarray(json.dumps(contact_override, sort_keys=True)),
            object_pose_written_at_initialize_only=np.asarray(True),
            object_pose_writes_after_initialize=np.asarray(0, dtype=np.int64),
            direct_object_actuator_count=np.asarray(0, dtype=np.int64),
            **arrays,
        )
    video_path = None
    video_report: dict[str, object] = {
        "rendered": False,
        "reason": "strict MuJoCo grasp gate did not pass",
    }
    if passed:
        video_path = output / "real_ref_sim_passed_diagnostic.mp4"
        video_report = {
            "rendered": True,
            **render(
                paths["scene"], paths["real_video"], paths["object_reference"],
                arrays, video_path, fps=args.fps, outcome="诊断通过",
                real_contact_rows=(18, 30, 39),
            ),
        }
    audit = {
        "schema": OUT_SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "outcome": "诊断通过" if passed else "诊断失败",
        "input": {
            "kind": (
                "converted original SAPIEN drive targets"
                if args.target_source == "drive"
                else "SAPIEN observed hand-state replay for contact isolation"
            ),
            "target_source": args.target_source,
            "sapien_trace": str(paths["sapien_trace"]),
            "sapien_trace_sha256": sha256(paths["sapien_trace"]),
            "sapien_summary": str(paths["sapien_summary"]),
            "candidate_index": int(source["candidate_index"]),
            "pool_depth": int(source["pool_depth"]),
            "physics_samples": len(desired),
            "physics_dt_s": dt,
            "control_frame_skip": int(source["frame_skip"]),
            "drive_target_mapping": target_mapping,
            "observed_initial_mapping": observed_mapping,
            "observed_initial_velocity_mapping": velocity_mapping,
            "source_sample_semantics": (
                "post-step state under same-index drive target"
                if source["schema"] == TRACE_SCHEMA_V2
                else "legacy v1 post-step timing inferred; source velocities unavailable"
            ),
            "world_alignment_sapien_to_mujoco": align.tolist(),
        },
        "physics_profile": physics_profile,
        "contact_override": contact_override,
        "runtime_model": {
            "path": str(runtime_model_path),
            "sha256": runtime_model_hash,
            "format": "MuJoCo compiled MJB containing all diagnostic runtime edits",
            "independent_replay_max_abs_error": replay_error,
        },
        "motion_contract": motion_contract,
        "physics": {
            "trace": str(trace_path),
            "trace_sha256": sha256(trace_path),
            "replay_max_abs_error": replay_error,
            "object_control": "none",
            "object_pose_writes_after_initialize": 0,
            "minimum_hand_self_gap_m": float(arrays["hand_self_gap_m"].min()),
            "minimum_hand_floor_gap_m": float(arrays["hand_floor_gap_m"].min()),
            "minimum_hand_object_gap_m": minimum_object_gap,
            "maximum_allowed_object_penetration_m": args.max_object_penetration_m,
            "hard_legal": hard_legal,
            "opposed_contact_after_closure": opposed_contact,
            "opposed_contact_during_hold": hold_opposed,
            "rest_object_z_m": rest_z,
            "hold_end_lift_m": hold_end_lift,
            "hold_min_lift_m": hold_min_lift,
            "held_lift_gate": held_lift,
            "hold_vertical_span_m": hold_span,
            "stable_hold_gate": stable_hold,
            "peak_normal_force_n": float(arrays["finger_normal_force_n"].max()),
            "hand_target_l2_error_max": float(tracking_error.max()),
            "hand_target_l2_error_mean": float(tracking_error.mean()),
            "mujoco_vs_sapien_observed_hand_l2_error_max": float(observed_error.max()),
            "mujoco_vs_sapien_observed_hand_l2_error_mean": float(observed_error.mean()),
            "mujoco_vs_aligned_sapien_object_center_error_m_final": float(position_difference[-1]),
            "mujoco_vs_aligned_sapien_object_center_error_m_max": float(position_difference.max()),
            "strict_grasp_gate_passed": passed,
        },
        "endpoint_comparison": {
            "original_sapien_mean_vertex_error_m": source_summary["sapien_endpoint_error_m"],
            "mujoco": endpoint,
        },
        "video": {
            "path": str(video_path) if video_path else None,
            "camera": "固定第三人称 front；禁止跟随",
            **video_report,
        },
    }
    audit_path = output / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit": str(audit_path),
        "trace": str(trace_path),
        "video": str(video_path) if video_path else None,
        "outcome": audit["outcome"],
        "strict_gate": passed,
        "mujoco_endpoint_error_m": endpoint["mean_vertex_error_m"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
