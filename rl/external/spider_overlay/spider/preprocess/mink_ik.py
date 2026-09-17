"""Paper-style MINK retargeting for xHand references.

This is the production counterpart of EgoEngine Eq. (1): five fingertip poses
and the wrist orientation are tracked under joint-limit and self-collision
constraints.  Fingertip SO(3) targets are landmark-derived frames whose distal
axis comes from DIP-to-tip and whose twist reference comes from the palm normal.
The xHand site's implementation-specific axes are calibrated once in the neutral
configuration rather than being assumed to match those human frames. The
default profile follows Eq. (1) with joint limits and self-collision only;
scene-geometry constraints remain an explicit non-paper ablation. It does not
synthesize a grasp type or alter the observed contact labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mink
import mujoco
import numpy as np
import trimesh
import tyro

from spider import ROOT
from spider.io import get_processed_data_dir


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_SITES = tuple(f"right_{finger}_tip" for finger in FINGERS)
JOINT_LIMIT_SAFETY_MARGIN = 1e-5


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def _matrix_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def _wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quaternion


def interpolate_pose7(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate xyz linearly and unit wxyz on the shortest quaternion arc."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    start = np.asarray(first, dtype=np.float64)
    end = np.asarray(second, dtype=np.float64)
    if start.shape != (7,) or end.shape != (7,):
        raise ValueError("pose inputs must have shape (7,)")
    result = np.empty(7, dtype=np.float64)
    result[:3] = (1.0 - alpha) * start[:3] + alpha * end[:3]
    qa = start[3:] / max(np.linalg.norm(start[3:]), 1e-12)
    qb = end[3:] / max(np.linalg.norm(end[3:]), 1e-12)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        quaternion = (1.0 - alpha) * qa + alpha * qb
    else:
        angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
        quaternion = (
            math.sin((1.0 - alpha) * angle) * qa
            + math.sin(alpha * angle) * qb
        ) / math.sin(angle)
    result[3:] = quaternion / max(np.linalg.norm(quaternion), 1e-12)
    return result


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _geometric_frame(normal: np.ndarray, direction: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(direction, dtype=np.float64)
    z_axis /= max(np.linalg.norm(z_axis), 1e-12)
    x_axis = np.asarray(normal, dtype=np.float64)
    x_axis -= float(np.dot(x_axis, z_axis)) * z_axis
    x_axis /= max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=-1)


def _site_local_fingertip_frames(
    model: mujoco.MjModel, side: str = "right",
) -> np.ndarray:
    """Map each MuJoCo tip site frame to a morphology-neutral distal frame.

    MuJoCo site axes are XML implementation details.  The returned proper
    rotations express the same geometric convention used by the human targets
    (palm-normal x axis and DIP-to-tip z axis) in each neutral tip-site frame.
    A human geometric frame ``H`` therefore targets the site rotation
    ``H @ local_frame.T``.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    palm_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm"
    )
    palm_normal = data.site_xmat[palm_id].reshape(3, 3)[:, 0]
    frames = []
    for finger in FINGERS:
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_tip"
        )
        body_id = int(model.site_bodyid[site_id])
        direction = data.site_xpos[site_id] - data.xpos[body_id]
        geometric = _geometric_frame(palm_normal, direction)
        site_rotation = data.site_xmat[site_id].reshape(3, 3)
        frames.append(site_rotation.T @ geometric)
    return np.asarray(frames)


def _mink_residual_scale(coefficient: float, name: str) -> float:
    """Convert an L2 objective coefficient into MINK's residual multiplier.

    MINK squares ``Task.cost`` while assembling its QP.  Passing an equation
    coefficient directly would therefore apply its square.  Taking the square
    root makes a configured coefficient, in particular EgoEngine's
    :math:`lambda_w`, multiply the quadratic loss exactly once.
    """
    value = float(coefficient)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative coefficient")
    return math.sqrt(value)


@dataclass(frozen=True)
class GeometryConstraint:
    """One exact MuJoCo signed-distance lower bound."""

    label: str
    geom1_id: int
    geom2_id: int
    minimum_distance_m: float


def _signed_geom_distance_and_jacobian(
    configuration: mink.Configuration, geom1_id: int, geom2_id: int,
) -> tuple[float, np.ndarray]:
    """Return exact signed distance and its tangent-space Jacobian.

    MuJoCo reverses the direction of ``fromto`` once two geoms penetrate.  The
    sign correction below makes the returned row the derivative of the signed
    distance on both sides of contact.
    """
    model = configuration.model
    data = configuration.data
    fromto = np.empty(6, dtype=np.float64)
    distance = float(mujoco.mj_geomDistance(
        model, data, geom1_id, geom2_id, 1.0, fromto,
    ))
    normal = fromto[3:] - fromto[:3]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        return distance, np.zeros(model.nv, dtype=np.float64)
    normal /= normal_norm
    jac1 = np.empty((3, model.nv), dtype=np.float64)
    jac2 = np.empty((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(
        model, data, jac2, None, fromto[3:], int(model.geom_bodyid[geom2_id]),
    )
    mujoco.mj_jac(
        model, data, jac1, None, fromto[:3], int(model.geom_bodyid[geom1_id]),
    )
    row = normal @ (jac2 - jac1)
    if distance < 0.0:
        row = -row
    return distance, row


class SignedDistanceRecoveryTask(mink.Task):
    """Push one already-violating geom pair back into its feasible set.

    ``CollisionAvoidanceLimit`` is a velocity damper: once penetration exists,
    it permits separating velocity but supplies no objective that creates it.
    This task supplies that missing recovery objective.  It is instantiated only
    for a currently violating pair and is discarded after each projection step.
    """

    def __init__(
        self,
        geom1_id: int,
        geom2_id: int,
        minimum_distance_m: float,
        cost: float,
        *,
        gain: float = 0.5,
    ) -> None:
        super().__init__(
            cost=np.asarray([float(cost)], dtype=np.float64),
            gain=gain,
            lm_damping=0.0,
        )
        self.geom1_id = int(geom1_id)
        self.geom2_id = int(geom2_id)
        self.minimum_distance_m = float(minimum_distance_m)

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        distance, _ = _signed_geom_distance_and_jacobian(
            configuration, self.geom1_id, self.geom2_id,
        )
        return np.asarray(
            [self.minimum_distance_m - distance], dtype=np.float64,
        )

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        _, distance_jacobian = _signed_geom_distance_and_jacobian(
            configuration, self.geom1_id, self.geom2_id,
        )
        return -distance_jacobian[None, :]


class ExactSignedDistanceLimit(mink.Limit):
    """Linearized hard bounds for every explicitly audited geometry pair.

    Unlike MINK's generic collision helper, this limit does not discard
    parent-child pairs.  That matters for the xHand XML, whose explicit
    palm-thumb collision pair is kinematically adjacent but still part of the
    simulator's collision model and q_ref gate.
    """

    def __init__(
        self,
        geometry_constraints: list[GeometryConstraint],
        *,
        gain: float = 0.2,
        activation_distance_m: float = 0.02,
    ) -> None:
        if not 0.0 < gain <= 1.0:
            raise ValueError("exact signed-distance gain must be in (0, 1]")
        if activation_distance_m <= 0.0:
            raise ValueError("activation distance must be positive")
        self.geometry_constraints = geometry_constraints
        self.gain = float(gain)
        self.activation_distance_m = float(activation_distance_m)

    def compute_qp_inequalities(
        self, configuration: mink.Configuration, dt: float,
    ) -> mink.Constraint:
        del dt  # Bounds are expressed in tangent displacement, not velocity.
        rows = []
        upper_bounds = []
        for constraint in self.geometry_constraints:
            distance, jacobian = _signed_geom_distance_and_jacobian(
                configuration, constraint.geom1_id, constraint.geom2_id,
            )
            if distance > (
                constraint.minimum_distance_m + self.activation_distance_m
            ):
                continue
            # d(q + dq) ~= d(q) + J dq >= d_min.
            rows.append(-jacobian)
            # A pre-existing violation may not have a simultaneously feasible
            # fixed-rate separating step for every active pair.  In that case
            # forbid worsening it and let SignedDistanceRecoveryTask choose a
            # compatible separating direction.  Feasible pairs retain a
            # velocity-damper buffer and cannot be crossed freely.
            upper_bounds.append(
                self.gain * (distance - constraint.minimum_distance_m)
                if distance > constraint.minimum_distance_m else 0.0
            )
        if not rows:
            return mink.Constraint()
        return mink.Constraint(
            G=np.asarray(rows, dtype=np.float64),
            h=np.asarray(upper_bounds, dtype=np.float64),
        )


def _exact_geometry_violations(
    configuration: mink.Configuration,
    constraints: list[GeometryConstraint],
    *,
    tolerance_m: float,
) -> list[dict[str, Any]]:
    """Evaluate the discrete configuration, not only its velocity bounds."""
    violations = []
    for constraint in constraints:
        distance = float(mujoco.mj_geomDistance(
            configuration.model,
            configuration.data,
            constraint.geom1_id,
            constraint.geom2_id,
            1.0,
            None,
        ))
        deficit = constraint.minimum_distance_m - distance
        if deficit > tolerance_m:
            violations.append({
                "label": constraint.label,
                "geom1_id": constraint.geom1_id,
                "geom2_id": constraint.geom2_id,
                "geom1": _geom_name(configuration.model, constraint.geom1_id),
                "geom2": _geom_name(configuration.model, constraint.geom2_id),
                "minimum_distance_m": constraint.minimum_distance_m,
                "signed_distance_m": distance,
                "deficit_m": deficit,
                "constraint": constraint,
            })
    return sorted(violations, key=lambda item: item["deficit_m"], reverse=True)


def _exact_joint_limit_violations(
    model: mujoco.MjModel, qpos: np.ndarray, *, tolerance: float = 1e-8,
) -> list[dict[str, Any]]:
    violations = []
    for joint_id in range(model.njnt):
        if (
            model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            or not model.jnt_limited[joint_id]
        ):
            continue
        address = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        value = float(qpos[address])
        deficit = max(float(lower - value), float(value - upper), 0.0)
        if deficit > tolerance:
            violations.append({
                "joint_id": joint_id,
                "joint": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_id,
                ) or "",
                "value": value,
                "lower": float(lower),
                "upper": float(upper),
                "deficit": deficit,
            })
    return sorted(violations, key=lambda item: item["deficit"], reverse=True)


def project_configuration_collisions(
    configuration: mink.Configuration,
    geometry_constraints: list[GeometryConstraint],
    damping_task: mink.DampingTask,
    limits: list[mink.Limit],
    qp_constraints: list[Any],
    *,
    sim_dt: float,
    solver: str,
    damping: float,
    maximum_iterations: int,
    tolerance_m: float,
    safety_margin_m: float,
    recovery_cost: float,
    recovery_gain: float,
) -> dict[str, Any]:
    """Project a committed target onto exact discrete collision feasibility.

    The projection has a fixed iteration budget.  Failure is returned to the
    caller and must reject the trajectory; it is never silently relaxed.
    """
    initial = _exact_geometry_violations(
        configuration, geometry_constraints, tolerance_m=tolerance_m,
    )
    initial_violation_count = len(initial)
    initial_label_counts = {
        label: sum(item["label"] == label for item in initial)
        for label in sorted({item["label"] for item in initial})
    }
    serializable_initial = [
        {key: value for key, value in item.items() if key != "constraint"}
        for item in initial[:20]
    ]
    initial_maximum_deficit = max(
        (item["deficit_m"] for item in initial), default=0.0,
    )
    iterations = 0
    qp_failures = 0
    while initial and iterations < maximum_iterations:
        recovery_tasks = [
            SignedDistanceRecoveryTask(
                item["geom1_id"],
                item["geom2_id"],
                item["minimum_distance_m"] + safety_margin_m,
                recovery_cost,
                gain=recovery_gain,
            )
            for item in initial
        ]
        try:
            velocity = mink.solve_ik(
                configuration,
                [damping_task, *recovery_tasks],
                sim_dt,
                solver,
                damping=damping,
                limits=limits,
                constraints=qp_constraints,
            )
        except mink.NoSolutionFound:
            qp_failures += 1
            break
        configuration.integrate_inplace(velocity, sim_dt)
        iterations += 1
        initial = _exact_geometry_violations(
            configuration, geometry_constraints, tolerance_m=tolerance_m,
        )

    final_violations = initial
    serializable_final = [
        {key: value for key, value in item.items() if key != "constraint"}
        for item in final_violations
    ]
    return {
        "accepted": not final_violations,
        "iterations": iterations,
        "qp_failures": qp_failures,
        "initial_violation_count": initial_violation_count,
        "initial_violation_label_counts": initial_label_counts,
        "initial_violations_top20": serializable_initial,
        "initial_maximum_deficit_m": initial_maximum_deficit,
        "final_violation_count": len(final_violations),
        "final_violations": serializable_final,
    }


def _set_target(
    task: mink.FrameTask, position: np.ndarray, rotation: np.ndarray,
) -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    task.set_target(mink.SE3.from_matrix(transform))


def _enable_distance_queries(model: mujoco.MjModel, names: list[str]) -> None:
    for name in names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise RuntimeError(f"missing collision geom {name}")
        model.geom_contype[geom_id] = 1
        model.geom_conaffinity[geom_id] = 1


def make_limits(
    model: mujoco.MjModel,
    *,
    floor_clearance_m: float,
    non_distal_object_clearance_m: float,
    distal_object_max_penetration_m: float,
    paper_only_constraints: bool = True,
) -> tuple[list[mink.Limit], dict[str, Any], list[GeometryConstraint]]:
    hand_geoms = [
        _geom_name(model, geom_id) for geom_id in range(model.ngeom)
        if _geom_name(model, geom_id).startswith("collision_hand_right_")
    ]
    object_geoms = [
        _geom_name(model, geom_id) for geom_id in range(model.ngeom)
        if _geom_name(model, geom_id).startswith("right_object_")
        and _geom_name(model, geom_id).rsplit("_", 1)[-1].isdigit()
    ]
    floor_geoms = [
        _geom_name(model, geom_id) for geom_id in range(model.ngeom)
        if _geom_name(model, geom_id) == "floor"
    ]
    distal_geoms = [
        name for name in hand_geoms
        if name.rsplit("_", 1)[-1] == "0" and "palm" not in name
    ]
    non_distal_geoms = [name for name in hand_geoms if name not in distal_geoms]
    if not hand_geoms or len(distal_geoms) != 5:
        raise RuntimeError(
            "MINK requires named xHand and five distal collision geoms"
        )
    if not paper_only_constraints and (not object_geoms or not floor_geoms):
        raise RuntimeError(
            "digital-twin MINK constraints require named object and floor geoms"
        )
    query_geoms = [*hand_geoms]
    if not paper_only_constraints:
        query_geoms.extend([*object_geoms, *floor_geoms])
    _enable_distance_queries(model, query_geoms)

    hand_set = set(hand_geoms)
    self_pairs: list[tuple[list[str], list[str]]] = []
    self_pair_ids: list[tuple[int, int]] = []
    for pair_id in range(model.npair):
        first_id = int(model.pair_geom1[pair_id])
        second_id = int(model.pair_geom2[pair_id])
        first = _geom_name(model, first_id)
        second = _geom_name(model, second_id)
        if first in hand_set and second in hand_set:
            self_pairs.append(([first], [second]))
            self_pair_ids.append((first_id, second_id))
    if not self_pairs:
        raise RuntimeError("scene.xml has no explicit xHand self-collision pairs")

    limits: list[mink.Limit] = [
        mink.ConfigurationLimit(
            model, min_distance_from_limits=JOINT_LIMIT_SAFETY_MARGIN,
        ),
        mink.CollisionAvoidanceLimit(
            model,
            self_pairs,
            gain=0.2,
            minimum_distance_from_collisions=0.0,
            collision_detection_distance=0.01,
            bound_relaxation=0.0,
        ),
    ]
    geom_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in [*hand_geoms, *object_geoms, *floor_geoms]
    }
    geometry_constraints = [
        GeometryConstraint("self_collision", first, second, 0.0)
        for first, second in self_pair_ids
    ]
    if not paper_only_constraints:
        limits.extend([
            mink.CollisionAvoidanceLimit(
                model,
                [(hand_geoms, floor_geoms)],
                gain=0.2,
                minimum_distance_from_collisions=floor_clearance_m,
                collision_detection_distance=max(0.02, floor_clearance_m + 0.01),
                bound_relaxation=0.0,
            ),
            mink.CollisionAvoidanceLimit(
                model,
                [(non_distal_geoms, object_geoms)],
                gain=0.2,
                minimum_distance_from_collisions=non_distal_object_clearance_m,
                collision_detection_distance=max(
                    0.02, non_distal_object_clearance_m + 0.01,
                ),
                bound_relaxation=0.0,
            ),
            mink.CollisionAvoidanceLimit(
                model,
                [(distal_geoms, object_geoms)],
                gain=0.2,
                minimum_distance_from_collisions=-distal_object_max_penetration_m,
                collision_detection_distance=0.01,
                bound_relaxation=0.0,
            ),
        ])
        geometry_constraints.extend(
            GeometryConstraint(
                "hand_floor", geom_ids[first], geom_ids[second], floor_clearance_m,
            )
            for first in hand_geoms for second in floor_geoms
        )
        geometry_constraints.extend(
            GeometryConstraint(
                "non_distal_object",
                geom_ids[first],
                geom_ids[second],
                non_distal_object_clearance_m,
            )
            for first in non_distal_geoms for second in object_geoms
        )
        geometry_constraints.extend(
            GeometryConstraint(
                "distal_object",
                geom_ids[first],
                geom_ids[second],
                -distal_object_max_penetration_m,
            )
            for first in distal_geoms for second in object_geoms
        )
    limits.append(ExactSignedDistanceLimit(
        geometry_constraints, gain=0.2, activation_distance_m=0.02,
    ))
    return limits, {
        "constraint_profile": (
            "EgoEngine_Eq1_joint_limits_and_self_collision_only"
            if paper_only_constraints else "digital_twin_scene_geometry"
        ),
        "paper_only_constraints": paper_only_constraints,
        "joint_limits": True,
        "joint_limit_safety_margin_m_or_rad": JOINT_LIMIT_SAFETY_MARGIN,
        "self_collision_pair_count": len(self_pairs),
        "floor_geom_count": len(floor_geoms),
        "non_distal_hand_geom_count": len(non_distal_geoms),
        "distal_hand_geom_count": len(distal_geoms),
        "object_geom_count": len(object_geoms),
        "floor_clearance_m": None if paper_only_constraints else floor_clearance_m,
        "non_distal_object_clearance_m": (
            None if paper_only_constraints else non_distal_object_clearance_m
        ),
        "distal_object_max_penetration_m": (
            None if paper_only_constraints else distal_object_max_penetration_m
        ),
        "collision_avoidance_gain": 0.2,
        "exact_signed_distance_limit_gain": 0.2,
        "exact_signed_distance_activation_m": 0.02,
        "exact_discrete_geometry_constraint_count": len(geometry_constraints),
    }, geometry_constraints


def _initialize_floor_clearance(
    model: mujoco.MjModel, data: mujoco.MjData, clearance_m: float,
) -> float:
    """Lift the floating hand out of floor penetration before velocity limits.

    MINK collision limits are velocity dampers and assume a feasible initial
    configuration.  Aligning the palm site while xHand is in its zero joint pose
    can point the open fingers through the floor, so repair only the free-joint
    initialization before any task iteration begins.
    """
    hand = [
        geom_id for geom_id in range(model.ngeom)
        if _geom_name(model, geom_id).startswith("collision_hand_right_")
    ]
    floor = [
        geom_id for geom_id in range(model.ngeom)
        if _geom_name(model, geom_id) == "floor"
    ]
    if not hand or not floor:
        raise RuntimeError("scene.xml lacks hand or floor collision geometry")
    mujoco.mj_forward(model, data)
    minimum = min(
        mujoco.mj_geomDistance(model, data, first, second, 1.0, None)
        for first in hand for second in floor
    )
    lift = max(0.0, float(clearance_m - minimum + 1e-4))
    data.qpos[2] += lift
    mujoco.mj_forward(model, data)
    return lift


def moving_average_configurations(
    model: mujoco.MjModel, configurations: np.ndarray, window: int,
) -> np.ndarray:
    values = np.asarray(configurations, dtype=np.float64)
    if window < 1 or window > len(values):
        raise ValueError("invalid moving-average window")
    filtered = np.stack([
        values[index : index + window].mean(axis=0)
        for index in range(len(values) - window + 1)
    ])
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        quaternion = filtered[:, qpos_address + 3 : qpos_address + 7]
        norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
        quaternion /= np.maximum(norm, 1e-12)
    return filtered


def _trajectory_metrics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    wrist_targets: np.ndarray,
    finger_positions: np.ndarray,
    fingertip_frame_targets: np.ndarray,
    site_local_fingertip_frames: np.ndarray,
    observed_contact_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    tip_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in FINGER_SITES
    ]
    position_errors = []
    orientation_errors = []
    direction_errors = []
    wrist_errors = []
    for frame, configuration in enumerate(qpos):
        data.qpos[:] = configuration
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        position_errors.append(
            np.linalg.norm(data.site_xpos[tip_ids] - finger_positions[frame], axis=1)
        )
        actual_frames = np.asarray([
            data.site_xmat[site_id].reshape(3, 3)
            @ site_local_fingertip_frames[index]
            for index, site_id in enumerate(tip_ids)
        ])
        relative = np.swapaxes(actual_frames, -1, -2) @ fingertip_frame_targets[frame]
        cosine = np.clip(
            (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
        orientation_errors.append(np.arccos(cosine))
        dot = np.sum(
            actual_frames[:, :, 2] * fingertip_frame_targets[frame, :, :, 2],
            axis=1,
        )
        direction_errors.append(np.arccos(np.clip(dot, -1.0, 1.0)))
        palm_quaternion = _wxyz_from_matrix(data.site_xmat[palm_id].reshape(3, 3))
        wrist_dot = abs(float(np.dot(palm_quaternion, wrist_targets[frame, 3:])))
        wrist_errors.append(2.0 * math.acos(float(np.clip(wrist_dot, -1.0, 1.0))))
    position = np.asarray(position_errors)
    orientation = np.asarray(orientation_errors)
    direction = np.asarray(direction_errors)
    wrist = np.asarray(wrist_errors)

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    result: dict[str, Any] = {
        "fingertip_position_error_m": {
            "mean": float(position.mean()),
            "p95": float(np.percentile(position, 95)),
            "max": float(position.max()),
        },
        "fingertip_orientation_error_rad": {
            "mean": float(orientation.mean()),
            "p95": float(np.percentile(orientation, 95)),
            "max": float(orientation.max()),
        },
        "fingertip_direction_error_rad": {
            "mean": float(direction.mean()),
            "p95": float(np.percentile(direction, 95)),
            "max": float(direction.max()),
        },
        "wrist_orientation_error_rad": {
            "mean": float(wrist.mean()),
            "p95": float(np.percentile(wrist, 95)),
            "max": float(wrist.max()),
        },
        "per_finger": {
            finger: {
                "position_error_m": summary(position[:, index]),
                "orientation_error_rad": summary(orientation[:, index]),
                "direction_error_rad": summary(direction[:, index]),
            }
            for index, finger in enumerate(FINGERS)
        },
    }
    if observed_contact_mask is not None:
        mask = np.asarray(observed_contact_mask, dtype=bool)
        if mask.shape != position.shape:
            raise ValueError("observed_contact_mask must have shape (T, 5)")
        by_contact = {}
        for label, selection in (
            ("contact", mask), ("non_contact", ~mask),
        ):
            if not np.any(selection):
                by_contact[label] = {"sample_count": 0}
                continue
            by_contact[label] = {
                "sample_count": int(np.count_nonzero(selection)),
                "position_error_m": summary(position[selection]),
                "orientation_error_rad": summary(orientation[selection]),
                "direction_error_rad": summary(direction[selection]),
            }
        contact_frames = np.any(mask, axis=1)
        by_contact["wrist_during_contact_frames"] = (
            {"frame_count": int(np.count_nonzero(contact_frames)),
             "orientation_error_rad": summary(wrist[contact_frames])}
            if np.any(contact_frames) else {"frame_count": 0}
        )
        by_contact["wrist_outside_contact_frames"] = (
            {"frame_count": int(np.count_nonzero(~contact_frames)),
             "orientation_error_rad": summary(wrist[~contact_frames])}
            if np.any(~contact_frames) else {"frame_count": 0}
        )
        result["by_observed_contact"] = by_contact

    worst_position_flat = np.argsort(position, axis=None)[-10:][::-1]
    result["worst_fingertip_position_samples"] = [
        {
            "output_frame": int(np.unravel_index(flat, position.shape)[0]),
            "finger": FINGERS[np.unravel_index(flat, position.shape)[1]],
            "error_m": float(position[np.unravel_index(flat, position.shape)]),
            "observed_contact": bool(
                observed_contact_mask[np.unravel_index(flat, position.shape)]
            ) if observed_contact_mask is not None else None,
        }
        for flat in worst_position_flat
    ]
    worst_orientation_flat = np.argsort(orientation, axis=None)[-10:][::-1]
    result["worst_fingertip_orientation_samples"] = [
        {
            "output_frame": int(np.unravel_index(flat, orientation.shape)[0]),
            "finger": FINGERS[np.unravel_index(flat, orientation.shape)[1]],
            "error_rad": float(orientation[np.unravel_index(flat, orientation.shape)]),
            "observed_contact": bool(
                observed_contact_mask[np.unravel_index(flat, orientation.shape)]
            ) if observed_contact_mask is not None else None,
        }
        for flat in worst_orientation_flat
    ]
    worst_wrist = np.argsort(wrist)[-10:][::-1]
    result["worst_wrist_orientation_frames"] = [
        {"output_frame": int(frame), "error_rad": float(wrist[frame])}
        for frame in worst_wrist
    ]
    return result


def _geometry_metrics(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, Any]:
    def name(geom_id: int) -> str:
        return _geom_name(model, geom_id)

    hand = [
        geom_id for geom_id in range(model.ngeom)
        if name(geom_id).startswith("collision_hand_right_")
    ]
    distal = [
        geom_id for geom_id in hand
        if name(geom_id).rsplit("_", 1)[-1] == "0"
        and "palm" not in name(geom_id)
    ]
    non_distal = [geom_id for geom_id in hand if geom_id not in distal]
    objects = [
        geom_id for geom_id in range(model.ngeom)
        if name(geom_id).startswith("right_object_")
        and name(geom_id).rsplit("_", 1)[-1].isdigit()
    ]
    floor = [geom_id for geom_id in range(model.ngeom) if name(geom_id) == "floor"]
    hand_set = set(hand)
    self_pairs = [
        (int(model.pair_geom1[pair]), int(model.pair_geom2[pair]))
        for pair in range(model.npair)
        if int(model.pair_geom1[pair]) in hand_set
        and int(model.pair_geom2[pair]) in hand_set
    ]
    groups = {
        "self_collision": self_pairs,
        "hand_floor": [(first, second) for first in hand for second in floor],
        "non_distal_object": [
            (first, second) for first in non_distal for second in objects
        ],
        "distal_object": [(first, second) for first in distal for second in objects],
    }
    data = mujoco.MjData(model)
    result: dict[str, Any] = {}
    for label, pairs in groups.items():
        per_frame = []
        for configuration in qpos:
            data.qpos[:] = configuration
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            per_frame.append(min(
                mujoco.mj_geomDistance(model, data, first, second, 1.0, None)
                for first, second in pairs
            ))
        values = np.asarray(per_frame)
        minimum_at = int(np.argmin(values))
        result[label] = {
            "minimum_signed_distance_m": float(values[minimum_at]),
            "minimum_at_output_frame": minimum_at,
        }

    violations = np.zeros((len(qpos), 0), dtype=np.float64)
    columns = []
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        address = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        columns.append(np.maximum(lower - qpos[:, address], qpos[:, address] - upper))
    if columns:
        violations = np.maximum(np.stack(columns, axis=1), 0.0)
    result["joint_limits"] = {
        "violation_frame_count": int(
            np.count_nonzero(np.any(violations > 1e-8, axis=1))
        ) if violations.size else 0,
        "maximum_violation": float(violations.max()) if violations.size else 0.0,
    }
    return result


def evaluate_qref_gate(
    trajectory_metrics: dict[str, Any], geometry_metrics: dict[str, Any],
    *,
    non_distal_object_clearance_m: float,
    distal_object_max_penetration_m: float,
    floor_clearance_m: float,
    paper_only_constraints: bool = True,
) -> dict[str, Any]:
    """Apply fixed fidelity and feasibility limits before Replay or MPC."""
    limits = {
        "max_fingertip_position_p95_m": 0.015,
        "max_fingertip_orientation_p95_rad": math.radians(30.0),
        "max_wrist_orientation_p95_rad": math.radians(10.0),
        "geometry_tolerance_m": 1e-4,
        "minimum_self_distance_m": 0.0,
        "minimum_floor_distance_m": floor_clearance_m,
        "minimum_non_distal_object_distance_m": non_distal_object_clearance_m,
        "minimum_distal_object_distance_m": -distal_object_max_penetration_m,
    }
    tolerance = limits["geometry_tolerance_m"]
    checks = {
        "fingertip_position_fidelity": bool(
            trajectory_metrics["fingertip_position_error_m"]["p95"]
            <= limits["max_fingertip_position_p95_m"]
        ),
        "fingertip_orientation_fidelity": bool(
            trajectory_metrics["fingertip_orientation_error_rad"]["p95"]
            <= limits["max_fingertip_orientation_p95_rad"]
        ),
        "wrist_orientation_fidelity": bool(
            trajectory_metrics["wrist_orientation_error_rad"]["p95"]
            <= limits["max_wrist_orientation_p95_rad"]
        ),
        "joint_limits": geometry_metrics["joint_limits"]["violation_frame_count"] == 0,
        "self_collision": bool(
            geometry_metrics["self_collision"]["minimum_signed_distance_m"]
            >= limits["minimum_self_distance_m"] - tolerance
        ),
    }
    scene_diagnostics = {
        "floor_collision": bool(
            geometry_metrics["hand_floor"]["minimum_signed_distance_m"]
            >= limits["minimum_floor_distance_m"] - tolerance
        ),
        "non_distal_object_collision": bool(
            geometry_metrics["non_distal_object"]["minimum_signed_distance_m"]
            >= limits["minimum_non_distal_object_distance_m"] - tolerance
        ),
        "distal_object_penetration": bool(
            geometry_metrics["distal_object"]["minimum_signed_distance_m"]
            >= limits["minimum_distal_object_distance_m"] - tolerance
        ),
    }
    if not paper_only_constraints:
        checks.update(scene_diagnostics)
    return {
        "schema_version": "1.0",
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "limits": limits,
        "trajectory_metrics": trajectory_metrics,
        "geometry_metrics": geometry_metrics,
        "scene_geometry_diagnostics": scene_diagnostics,
        "policy": (
            "paper Eq.(1): hard fingertip/wrist target fidelity, joint limits, "
            "and self-collision; scene geometry is diagnostic"
            if paper_only_constraints else
            "extended digital-twin q_ref fidelity and scene-feasibility gate"
        ),
        "paper_only_constraints": paper_only_constraints,
    }


def qref_refinement_eligibility(gate: dict[str, Any]) -> dict[str, Any]:
    """Separate hard configuration feasibility from q_ref fidelity diagnostics.

    EgoEngine Eq. (1) defines joint limits and self-collision through the
    feasible configuration set, but does not specify a numerical fingertip
    fidelity rejection threshold before Replay/MPC.  The additional scene
    collision checks are hard constraints in this implementation.  Position
    and orientation thresholds remain visible diagnostics and may only be
    bypassed when the caller explicitly requests refinement continuation.
    """
    feasibility_names = ["joint_limits", "self_collision"]
    if not gate.get("paper_only_constraints", False):
        feasibility_names.extend([
            "floor_collision",
            "non_distal_object_collision",
            "distal_object_penetration",
        ])
    fidelity_names = (
        "fingertip_position_fidelity",
        "fingertip_orientation_fidelity",
        "wrist_orientation_fidelity",
    )
    checks = gate["checks"]
    hard_failures = [name for name in feasibility_names if not checks[name]]
    fidelity_warnings = [name for name in fidelity_names if not checks[name]]
    return {
        "eligible": not hard_failures,
        "hard_feasibility_checks": list(feasibility_names),
        "hard_failures": hard_failures,
        "diagnostic_fidelity_checks": list(fidelity_names),
        "fidelity_warnings": fidelity_warnings,
    }


def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "xhand",
    embodiment_type: str = "right",
    task: str = "pick_spoon_bowl",
    data_id: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    sim_dt: float = 0.005,
    ref_dt: float = 0.02,
    finger_position_cost: float = 1.0,
    # Position (metres) and SO(3) (radians) residuals require a unit-aware
    # balance. This fixed coefficient is the rounded squared ratio of the
    # global 15 mm and 30 degree fidelity tolerances, not a per-video fit.
    finger_orientation_cost: float = 1e-3,
    # Same global tolerance normalization as the fingertip orientation term:
    # round((15 mm / 10 degrees)^2). This is fixed across videos.
    lambda_w: float = 7.5e-3,
    wrist_orientation_cost: float | None = None,
    posture_cost: float = 1e-4,
    velocity_damping_cost: float = 1e-3,
    wrist_init_steps: int = 300,
    finger_init_steps: int = 500,
    collision_init_steps: int = 100,
    tracking_iterations_per_substep: int = 4,
    average_frame_size: int = 1,
    solver: str = "daqp",
    damping: float = 1e-5,
    floor_clearance_m: float = 0.0,
    non_distal_object_clearance_m: float = 0.001,
    distal_object_max_penetration_m: float = 0.0025,
    paper_only_constraints: bool = True,
    collision_projection_max_iterations: int = 30,
    collision_projection_tolerance_m: float = 1e-6,
    collision_projection_safety_margin_m: float = 5e-5,
    collision_projection_recovery_cost: float = 10.0,
    collision_projection_recovery_gain: float = 0.5,
    use_object_contact_position_targets: bool = False,
    controller_contact_target_policy: str = "object_surface",
    allow_fidelity_rejected_qref_for_refinement: bool = False,
    save_video: bool = True,
    seed: int = 0,
) -> None:
    if robot_type != "xhand" or embodiment_type != "right":
        raise ValueError("production MINK currently supports xhand/right only")
    if controller_contact_target_policy not in {
        "object_surface", "fingertip_collision_center", "qref_fingertip_site",
    }:
        raise ValueError(
            "controller_contact_target_policy must be object_surface or "
            "fingertip_collision_center or qref_fingertip_site"
        )
    if wrist_orientation_cost is not None:
        if lambda_w != 1.0:
            raise ValueError(
                "use either --lambda-w or deprecated --wrist-orientation-cost, not both"
            )
        lambda_w = wrist_orientation_cost
    finger_position_scale = _mink_residual_scale(
        finger_position_cost, "finger_position_cost",
    )
    finger_orientation_scale = _mink_residual_scale(
        finger_orientation_cost, "finger_orientation_cost",
    )
    wrist_orientation_scale = _mink_residual_scale(lambda_w, "lambda_w")
    if average_frame_size != 1:
        raise ValueError(
            "post-IK qpos averaging can invalidate collision constraints; "
            "production MINK requires average_frame_size=1"
        )
    if tracking_iterations_per_substep < 1:
        raise ValueError("tracking_iterations_per_substep must be positive")
    if collision_projection_max_iterations < 1:
        raise ValueError("collision_projection_max_iterations must be positive")
    if collision_projection_tolerance_m < 0.0:
        raise ValueError("collision_projection_tolerance_m must be nonnegative")
    if collision_projection_safety_margin_m <= collision_projection_tolerance_m:
        raise ValueError(
            "collision_projection_safety_margin_m must exceed the tolerance"
        )
    if collision_projection_recovery_cost <= 0.0:
        raise ValueError("collision_projection_recovery_cost must be positive")
    if not 0.0 < collision_projection_recovery_gain <= 1.0:
        raise ValueError("collision_projection_recovery_gain must be in (0, 1]")
    np.random.seed(seed)
    dataset = Path(dataset_dir).resolve()
    robot_dir = Path(get_processed_data_dir(
        dataset_dir=str(dataset), dataset_name=dataset_name,
        robot_type=robot_type, embodiment_type=embodiment_type,
        task=task, data_id=data_id,
    ))
    mano_dir = Path(get_processed_data_dir(
        dataset_dir=str(dataset), dataset_name=dataset_name,
        robot_type="mano", embodiment_type=embodiment_type,
        task=task, data_id=data_id,
    ))
    model_path = robot_dir.parent / "scene.xml"
    targets_path = mano_dir / "trajectory_keypoints.npz"
    if not model_path.exists() or not targets_path.exists():
        raise FileNotFoundError(f"missing MINK input: {model_path} or {targets_path}")

    with np.load(targets_path, allow_pickle=False) as source:
        required = {
            "qpos_wrist_right", "qpos_finger_right", "qpos_obj_right",
            "contact_right", "contact_pos_right", "fingertip_orientation_right",
        }
        missing = sorted(required - set(source.files))
        if missing:
            raise RuntimeError(
                "MINK requires exported fingertip poses and contact metadata; missing "
                + ", ".join(missing)
            )
        wrist = np.asarray(source["qpos_wrist_right"][start_idx:end_idx], dtype=float)
        fingers = np.asarray(source["qpos_finger_right"][start_idx:end_idx], dtype=float)
        objects = np.asarray(source["qpos_obj_right"][start_idx:end_idx], dtype=float)
        contacts = np.asarray(source["contact_right"][start_idx:end_idx], dtype=float)
        contact_positions_local = np.asarray(
            source["contact_pos_right"], dtype=float,
        )
        fingertip_frames = np.asarray(
            source["fingertip_orientation_right"][start_idx:end_idx], dtype=float
        )
        fingertip_orientation_source = str(
            source["fingertip_orientation_source"]
            if "fingertip_orientation_source" in source.files
            else "unknown_legacy_source"
        )
    if not len(wrist) or not (
        len(wrist) == len(fingers) == len(objects) == len(contacts)
        == len(fingertip_frames)
    ):
        raise RuntimeError("MINK input arrays have inconsistent or empty timelines")
    if contact_positions_local.shape != (5, 3):
        raise RuntimeError("contact_pos_right must have shape (5, 3)")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = sim_dt
    configuration = mink.Configuration(model)
    data = configuration.data
    site_local_frames = _site_local_fingertip_frames(model)
    orthogonality_error = np.max(np.linalg.norm(
        np.swapaxes(fingertip_frames, -1, -2) @ fingertip_frames - np.eye(3),
        axis=(-2, -1),
    ))
    minimum_determinant = float(np.min(np.linalg.det(fingertip_frames)))
    if orthogonality_error > 1e-4 or minimum_determinant < 0.9999:
        raise RuntimeError(
            "fingertip_orientation_right must contain proper SO(3) matrices; "
            f"max orthogonality error={orthogonality_error:.3e}, "
            f"minimum determinant={minimum_determinant:.6f}"
        )
    fingertip_site_rotations = np.matmul(
        fingertip_frames, np.swapaxes(site_local_frames, -1, -2)[None],
    )

    task_info_path = mano_dir.parent / "task_info.json"
    task_info = json.loads(task_info_path.read_text(encoding="utf-8"))
    visual_mesh_path = dataset / task_info["right_object_mesh_dir"] / "visual.obj"
    visual_mesh = trimesh.load_mesh(visual_mesh_path, process=False)
    vertices = np.asarray(visual_mesh.vertices, dtype=np.float64)
    vertex_normals = np.asarray(visual_mesh.vertex_normals, dtype=np.float64)
    nearest_vertices = np.argmin(
        np.linalg.norm(
            contact_positions_local[:, None, :] - vertices[None, :, :], axis=-1,
        ),
        axis=1,
    )
    contact_normals_local = vertex_normals[nearest_vertices]
    outward = contact_positions_local - np.asarray(visual_mesh.centroid)
    flip = np.sum(contact_normals_local * outward, axis=1) < 0.0
    contact_normals_local[flip] *= -1.0
    contact_normals_local /= np.maximum(
        np.linalg.norm(contact_normals_local, axis=1, keepdims=True), 1e-12,
    )
    tip_radii = np.asarray([
        model.geom_size[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM,
            f"collision_hand_right_{finger}_0",
        ), 0]
        for finger in FINGERS
    ])
    contact_site_offsets = np.maximum(
        tip_radii - distal_object_max_penetration_m, 0.0,
    )
    contact_site_targets_local = (
        contact_positions_local
        + contact_site_offsets[:, None] * contact_normals_local
    )
    object_rotations = np.asarray([
        _matrix_from_wxyz(pose[3:]) for pose in objects
    ])
    contact_positions_world = np.einsum(
        "tij,fj->tfi", object_rotations, contact_site_targets_local,
    ) + objects[:, None, :3]
    contact_mask = contacts >= 0.5
    proposed_contact_targets = np.where(
        contact_mask[..., None], contact_positions_world, fingers[..., :3],
    )
    finger_position_targets = (
        proposed_contact_targets if use_object_contact_position_targets
        else fingers[..., :3].copy()
    )
    proposed_contact_correction = np.linalg.norm(
        proposed_contact_targets - fingers[..., :3], axis=-1,
    )
    contact_correction = np.linalg.norm(
        finger_position_targets - fingers[..., :3], axis=-1,
    )

    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    configuration.update()
    data.qpos[:3] += wrist[0, :3] - data.site_xpos[palm_id]
    data.qpos[-7:] = objects[0]
    configuration.update()

    wrist_task = mink.FrameTask(
        frame_name="right_palm", frame_type="site",
        position_cost=0.0, orientation_cost=wrist_orientation_scale,
        lm_damping=1.0,
    )
    finger_tasks = [
        mink.FrameTask(
            frame_name=site, frame_type="site",
            position_cost=finger_position_scale,
            orientation_cost=finger_orientation_scale,
            lm_damping=1.0,
        )
        for site in FINGER_SITES
    ]
    posture_task = mink.PostureTask(model, cost=posture_cost)
    posture_task.set_target(configuration.q.copy())
    damping_task = mink.DampingTask(model, cost=velocity_damping_cost)
    object_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint"
    )
    object_dof_start = int(model.jnt_dofadr[object_joint_id])
    freeze_object = mink.DofFreezingTask(
        model, list(range(object_dof_start, object_dof_start + 6)),
    )
    limits, limit_report, geometry_constraints = make_limits(
        model,
        floor_clearance_m=floor_clearance_m,
        non_distal_object_clearance_m=non_distal_object_clearance_m,
        distal_object_max_penetration_m=distal_object_max_penetration_m,
        paper_only_constraints=paper_only_constraints,
    )
    projection_audit: dict[str, Any] = {
        "schema_version": "1.0",
        "policy": (
            "exact MuJoCo signed-distance validation after every interpolated "
            "target; bounded recovery objective; reject on failed projection"
        ),
        "parameters": {
            "maximum_iterations": collision_projection_max_iterations,
            "tolerance_m": collision_projection_tolerance_m,
            "safety_margin_m": collision_projection_safety_margin_m,
            "recovery_cost": collision_projection_recovery_cost,
            "recovery_gain": collision_projection_recovery_gain,
            "joint_limit_safety_margin_m_or_rad": JOINT_LIMIT_SAFETY_MARGIN,
            "joint_limit_validation_tolerance": 1e-8,
        },
        "checkpoint_count": 0,
        "recovery_checkpoint_count": 0,
        "total_recovery_iterations": 0,
        "maximum_recovery_iterations": 0,
        "maximum_initial_deficit_m": 0.0,
        "initial_violation_label_counts": {},
        "failure": None,
    }

    def projection_checkpoint(context: dict[str, Any]) -> None:
        result = project_configuration_collisions(
            configuration,
            geometry_constraints,
            damping_task,
            limits,
            [freeze_object],
            sim_dt=sim_dt,
            solver=solver,
            damping=damping,
            maximum_iterations=collision_projection_max_iterations,
            tolerance_m=collision_projection_tolerance_m,
            safety_margin_m=collision_projection_safety_margin_m,
            recovery_cost=collision_projection_recovery_cost,
            recovery_gain=collision_projection_recovery_gain,
        )
        joint_violations = _exact_joint_limit_violations(
            model, configuration.q, tolerance=1e-8,
        )
        result["joint_limit_violations"] = joint_violations
        result["accepted"] = bool(result["accepted"] and not joint_violations)
        projection_audit["checkpoint_count"] += 1
        if result["initial_violation_count"]:
            projection_audit["recovery_checkpoint_count"] += 1
        projection_audit["total_recovery_iterations"] += result["iterations"]
        projection_audit["maximum_recovery_iterations"] = max(
            projection_audit["maximum_recovery_iterations"], result["iterations"],
        )
        projection_audit["maximum_initial_deficit_m"] = max(
            projection_audit["maximum_initial_deficit_m"],
            result["initial_maximum_deficit_m"],
        )
        label_counts = projection_audit["initial_violation_label_counts"]
        for label, count in result["initial_violation_label_counts"].items():
            label_counts[label] = label_counts.get(label, 0) + count
        if result["accepted"]:
            return
        failure = {"context": context, "projection": result}
        projection_audit["failure"] = failure
        robot_dir.mkdir(parents=True, exist_ok=True)
        (robot_dir / "mink_projection_audit.json").write_text(
            json.dumps(projection_audit, indent=2) + "\n", encoding="utf-8",
        )
        np.savez_compressed(
            robot_dir / "trajectory_mink_projection_rejected.npz",
            qpos=configuration.q.copy(),
            source_frame=np.asarray(context.get("source_frame", -1)),
            substep=np.asarray(context.get("substep", -1)),
        )
        raise RuntimeError(
            "MINK exact configuration feasibility projection failed at "
            + json.dumps(context, sort_keys=True)
        )

    initial_floor_lift_m = (
        0.0 if paper_only_constraints
        else _initialize_floor_clearance(model, data, floor_clearance_m)
    )
    configuration.update()
    projection_checkpoint({"stage": "initial_configuration", "source_frame": 0})
    posture_task.set_target(configuration.q.copy())
    tasks = [
        posture_task, damping_task, wrist_task, *finger_tasks,
    ]

    def set_targets(frame: int, alpha: float = 1.0) -> None:
        previous = max(0, frame - 1)
        wrist_pose = interpolate_pose7(wrist[previous], wrist[frame], alpha)
        wrist_rotation = _matrix_from_wxyz(wrist_pose[3:])
        _set_target(wrist_task, wrist_pose[:3], wrist_rotation)
        for finger, task_instance in enumerate(finger_tasks):
            previous_pose = np.concatenate((
                finger_position_targets[previous, finger],
                _wxyz_from_matrix(fingertip_site_rotations[previous, finger]),
            ))
            current_pose = np.concatenate((
                finger_position_targets[frame, finger],
                _wxyz_from_matrix(fingertip_site_rotations[frame, finger]),
            ))
            fingertip_pose = interpolate_pose7(
                previous_pose, current_pose, alpha,
            )
            _set_target(
                task_instance,
                fingertip_pose[:3],
                _matrix_from_wxyz(fingertip_pose[3:]),
            )

    def solve_steps(
        frame: int, active_tasks: list[Any], active_limits: list[mink.Limit],
        count: int, *, stage: str, iterations_per_target: int = 1,
    ) -> None:
        previous_object = objects[max(0, frame - 1)]
        for substep in range(count):
            alpha = (substep + 1) / count
            set_targets(frame, alpha)
            object_pose = interpolate_pose7(previous_object, objects[frame], alpha)
            for iteration in range(iterations_per_target):
                data.qpos[-7:] = object_pose
                configuration.update()
                try:
                    velocity = mink.solve_ik(
                        configuration, active_tasks, sim_dt, solver,
                        damping=damping, limits=active_limits,
                        constraints=[freeze_object],
                    )
                except mink.NoSolutionFound as error:
                    raise RuntimeError(
                        f"MINK QP infeasible at source frame {frame}, "
                        f"substep {substep + 1}/{count}, iteration "
                        f"{iteration + 1}/{iterations_per_target}"
                    ) from error
                configuration.integrate_inplace(velocity, sim_dt)
                data.qpos[-7:] = object_pose
            configuration.update()
            projection_checkpoint({
                "stage": stage,
                "source_frame": frame,
                "substep": substep + 1,
                "substep_count": count,
            })

    solve_steps(
        0, [posture_task, damping_task, wrist_task], limits, wrist_init_steps,
        stage="wrist_initialization",
    )
    solve_steps(
        0, tasks, limits, finger_init_steps, stage="finger_initialization",
    )
    solve_steps(
        0, tasks, limits, collision_init_steps, stage="collision_initialization",
    )

    raw = []
    minimum_substeps = max(1, int(round(ref_dt / sim_dt)))
    for frame in range(len(wrist)):
        previous = max(0, frame - 1)
        object_translation = float(
            np.linalg.norm(objects[frame, :3] - objects[previous, :3])
        )
        fingertip_translation = float(np.max(np.linalg.norm(
            finger_position_targets[frame] - finger_position_targets[previous], axis=1,
        )))
        wrist_rotation_delta = _rotation_angle(
            _matrix_from_wxyz(wrist[previous, 3:]),
            _matrix_from_wxyz(wrist[frame, 3:]),
        )
        finger_rotation_delta = max(
            _rotation_angle(
                fingertip_site_rotations[previous, finger],
                fingertip_site_rotations[frame, finger],
            )
            for finger in range(5)
        )
        substeps = max(
            minimum_substeps,
            int(math.ceil(object_translation / 0.001)),
            int(math.ceil(fingertip_translation / 0.0015)),
            int(math.ceil(max(wrist_rotation_delta, finger_rotation_delta) / math.radians(5.0))),
        )
        if substeps > 40:
            raise RuntimeError(
                f"MINK input discontinuity at source frame {frame}: "
                f"requires {substeps} interpolation substeps (maximum 40)"
            )
        solve_steps(
            frame, tasks, limits, substeps,
            stage="sequence_tracking",
            iterations_per_target=tracking_iterations_per_substep,
        )
        raw.append(configuration.q.copy())
    raw_qpos = np.asarray(raw)
    filtered = raw_qpos
    qvel = np.zeros((len(filtered) - 1, model.nv), dtype=np.float64)
    for frame in range(1, len(filtered)):
        mujoco.mj_differentiatePos(
            model, qvel[frame - 1], ref_dt, filtered[frame - 1], filtered[frame]
        )
    qpos = filtered[1:]
    first_center = 1
    reference_indices = np.arange(first_center, first_center + len(qpos))
    contact = contacts[reference_indices]

    evaluation_data = mujoco.MjData(model)
    object_site_ids = [
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE,
            f"track_object_right_{finger}_tip",
        )
        for finger in FINGERS
    ]
    hand_tip_site_ids = [
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, site,
        )
        for site in FINGER_SITES
    ]
    if any(site_id < 0 for site_id in object_site_ids):
        raise RuntimeError("scene.xml lacks object-side contact tracking sites")
    if any(site_id < 0 for site_id in hand_tip_site_ids):
        raise RuntimeError("scene.xml lacks xHand fingertip sites")
    contact_pos_surface = np.empty((len(qpos), 5, 3), dtype=np.float32)
    contact_pos_qref_fingertip_site = np.empty(
        (len(qpos), 5, 3), dtype=np.float32,
    )
    for frame, value in enumerate(qpos):
        evaluation_data.qpos[:] = value
        evaluation_data.qvel[:] = 0.0
        mujoco.mj_forward(model, evaluation_data)
        contact_pos_surface[frame] = evaluation_data.site_xpos[object_site_ids]
        contact_pos_qref_fingertip_site[frame] = evaluation_data.site_xpos[
            hand_tip_site_ids
        ]
    contact_pos_fingertip_center = np.asarray(
        contact_positions_world[reference_indices], dtype=np.float32,
    )
    contact_pos_by_policy = {
        "object_surface": contact_pos_surface,
        "fingertip_collision_center": contact_pos_fingertip_center,
        "qref_fingertip_site": contact_pos_qref_fingertip_site,
    }
    contact_pos = contact_pos_by_policy[controller_contact_target_policy]

    robot_dir.mkdir(parents=True, exist_ok=True)
    targets_at_output = reference_indices
    metrics = _trajectory_metrics(
        model, qpos, wrist[targets_at_output], finger_position_targets[targets_at_output],
        fingertip_frames[targets_at_output], site_local_frames,
        observed_contact_mask=contact_mask[targets_at_output],
    )
    geometry_metrics = _geometry_metrics(model, qpos)
    objective_metadata = {
        "equation": "L_tip + lambda_w * L_wrist",
        "quadratic_loss_coefficients": {
            "finger_position": finger_position_cost,
            "finger_full_orientation": finger_orientation_cost,
            "lambda_w": lambda_w,
        },
        "mink_residual_scales": {
            "finger_position": finger_position_scale,
            "finger_full_orientation": finger_orientation_scale,
            "wrist_orientation": wrist_orientation_scale,
        },
        "coefficient_mapping": (
            "MINK squares Task.cost; each equation coefficient is passed "
            "as its square root"
        ),
        "extra_regularizer_residual_scales": {
            "posture": posture_cost,
            "velocity_damping": velocity_damping_cost,
        },
    }
    qref_gate = evaluate_qref_gate(
        metrics, geometry_metrics,
        non_distal_object_clearance_m=non_distal_object_clearance_m,
        distal_object_max_penetration_m=distal_object_max_penetration_m,
        floor_clearance_m=floor_clearance_m,
        paper_only_constraints=paper_only_constraints,
    )
    qref_gate["objective"] = objective_metadata
    refinement_eligibility = qref_refinement_eligibility(qref_gate)
    qref_gate["refinement_eligibility"] = refinement_eligibility
    qref_gate["continuation"] = {
        "explicitly_requested": allow_fidelity_rejected_qref_for_refinement,
        "continued_as_prior": bool(
            allow_fidelity_rejected_qref_for_refinement
            and refinement_eligibility["eligible"]
            and not qref_gate["accepted"]
        ),
        "paper_alignment": (
            "EgoEngine does not define fixed numerical q_ref fidelity rejection "
            "thresholds; feasibility remains hard and final Replay/MPC MuJoCo "
            "acceptance remains mandatory"
        ),
    }
    qref_gate["target_policy"] = {
        "finger_position": (
            "observed fingertip outside contact; object-local visual-mesh surface "
            "point plus robot tip radius during observed contact"
            if use_object_contact_position_targets
            else "observed human fingertip positions at every frame"
        ),
        "finger_orientation": (
            f"full SO(3) from {fingertip_orientation_source}, mapped through "
            "each xHand tip site's neutral axes"
        ),
        "finger_direction": "DIP-to-tip axis (reported as a diagnostic subset)",
        "finger_axial_twist": f"provided by {fingertip_orientation_source}",
        "wrist_orientation": f"provided by {fingertip_orientation_source}",
        "grasp_mode_classifier": False,
        "contact_aware_preshape": False,
        "object_contact_position_targets_enabled": (
            use_object_contact_position_targets
        ),
    }
    qref_gate["initial_floor_clearance_lift_m"] = initial_floor_lift_m
    qref_gate["collision_projection"] = projection_audit
    qref_gate["contact_target_correction_m"] = {
        "enabled": use_object_contact_position_targets,
        "active_sample_count": int(np.count_nonzero(contact_mask)),
        "mean": float(contact_correction[contact_mask].mean())
        if contact_mask.any() else 0.0,
        "p95": float(np.percentile(contact_correction[contact_mask], 95))
        if contact_mask.any() else 0.0,
        "max": float(contact_correction.max()),
        "tip_center_surface_offsets_m": contact_site_offsets.tolist(),
        "proposed_if_enabled_p95": float(np.percentile(
            proposed_contact_correction[contact_mask], 95,
        )) if contact_mask.any() else 0.0,
        "proposed_if_enabled_max": float(proposed_contact_correction.max()),
        "per_finger": {
            finger: {
                "active_sample_count": int(np.count_nonzero(contact_mask[:, index])),
                "mean": float(contact_correction[contact_mask[:, index], index].mean())
                if np.any(contact_mask[:, index]) else 0.0,
                "p95": float(np.percentile(
                    contact_correction[contact_mask[:, index], index], 95,
                )) if np.any(contact_mask[:, index]) else 0.0,
                "max": float(contact_correction[:, index].max()),
            }
            for index, finger in enumerate(FINGERS)
        },
    }
    gate_path = robot_dir / "mink_qref_gate.json"
    (robot_dir / "mink_projection_audit.json").write_text(
        json.dumps(projection_audit, indent=2) + "\n", encoding="utf-8",
    )
    gate_path.write_text(json.dumps(qref_gate, indent=2) + "\n", encoding="utf-8")
    trajectory_arrays = {
        "qpos": qpos, "qvel": qvel, "contact": contact.astype(np.float32),
        "contact_pos": contact_pos,
        "contact_pos_object_surface": contact_pos_surface,
        "contact_pos_fingertip_collision_center": contact_pos_fingertip_center,
        "contact_pos_qref_fingertip_site": contact_pos_qref_fingertip_site,
        "controller_contact_target_policy": np.asarray(
            controller_contact_target_policy,
        ),
        "frequency": np.asarray(1.0 / ref_dt),
        "reference_indices": reference_indices, "raw_qpos": raw_qpos,
    }
    if not qref_gate["accepted"]:
        np.savez_compressed(
            robot_dir / "trajectory_mink_rejected.npz", **trajectory_arrays,
        )
        if not (
            allow_fidelity_rejected_qref_for_refinement
            and refinement_eligibility["eligible"]
        ):
            raise RuntimeError(
                "MINK q_ref rejected: " + ", ".join(qref_gate["failed_checks"])
            )
    trajectory_path = robot_dir / "trajectory_kinematic.npz"
    np.savez_compressed(trajectory_path, **trajectory_arrays)

    replay_data = mujoco.MjData(model)
    replay_data.qpos[:] = qpos[0]
    replay_data.qvel[:] = qvel[0]
    mujoco.mj_forward(model, replay_data)
    rollout = np.empty_like(qpos)
    rollout[0] = qpos[0]
    replay_substeps = minimum_substeps
    for frame in range(1, len(qpos)):
        replay_data.ctrl[:] = qpos[frame, : model.nu]
        for _ in range(replay_substeps):
            mujoco.mj_step(model, replay_data)
        rollout[frame] = replay_data.qpos
    np.savez_compressed(robot_dir / "trajectory_ikrollout.npz", qpos=rollout)

    video_path = robot_dir / "visualization_ik.mp4"
    if save_video:
        model.vis.global_.offwidth = 720
        model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(model, height=480, width=720)
        frames = []
        for value in qpos:
            evaluation_data.qpos[:] = value
            evaluation_data.qvel[:] = 0.0
            mujoco.mj_forward(model, evaluation_data)
            renderer.update_scene(evaluation_data, camera="front")
            frames.append(renderer.render())
        imageio.mimsave(video_path, frames, fps=round(1.0 / ref_dt))
        renderer.close()

    metadata = {
        "schema_version": "1.0",
        "method": "EgoEngine Eq.(1) MINK with digital-twin collision safety",
        "model": str(model_path),
        "targets": str(targets_path),
        "source_frame_count": len(wrist),
        "output_frame_count": len(qpos),
        "reference_indices": [int(reference_indices[0]), int(reference_indices[-1])],
        "objective": objective_metadata,
        "orientation": {
            "source": fingertip_orientation_source,
            "mapping": (
                "full SO(3) target calibrated against each neutral xHand "
                "fingertip-site frame"
            ),
            "axial_fingertip_twist": (
                "from calibrated MANO distal-link FK when source=mano_fk; "
                "legacy landmark proxy only in explicit ablations"
            ),
            "identity_placeholder_used": False,
            "mano_internal_axis_convention_used": bool(
                fingertip_orientation_source == "mano_fk"
            ),
        },
        "constraints": limit_report,
        "collision_projection": projection_audit,
        "initial_floor_clearance_lift_m": initial_floor_lift_m,
        "geometry_metrics": geometry_metrics,
        "qref_gate": qref_gate,
        "contact_policy": (
            "observed per-finger labels preserved; "
            + (
                "contact=true fingertip positions use the tracked object-local "
                "surface point (non-paper diagnostic override); "
                if use_object_contact_position_targets
                else "all fingertip positions remain the observed human targets "
                "(paper-aligned default); "
            )
            + "no grasp-mode classifier or preshape synthesis"
        ),
        "controller_contact_target_policy": {
            "selected": controller_contact_target_policy,
            "object_surface_saved": True,
            "fingertip_collision_center_saved": True,
            "qref_fingertip_site_saved": True,
            "collision_center_offset_source": (
                "xHand distal collision radius minus allowed shallow "
                "penetration, along the visual-mesh outward normal"
            ),
            "paper_status": (
                "EgoEngine mentions an auxiliary contact objective but does "
                "not specify surface-point versus collision-center semantics"
            ),
        },
        "contact_target_correction_m": {
            "enabled": use_object_contact_position_targets,
            "active_sample_count": int(np.count_nonzero(contact_mask)),
            "mean": float(contact_correction[contact_mask].mean())
            if contact_mask.any() else 0.0,
            "p95": float(np.percentile(contact_correction[contact_mask], 95))
            if contact_mask.any() else 0.0,
            "max": float(contact_correction.max()),
            "tip_center_surface_offsets_m": contact_site_offsets.tolist(),
            "proposed_if_enabled_p95": float(np.percentile(
                proposed_contact_correction[contact_mask], 95,
            )) if contact_mask.any() else 0.0,
            "proposed_if_enabled_max": float(proposed_contact_correction.max()),
            "surface_normal_source": "nearest visual-mesh vertex normal in object frame",
        },
        "temporal_policy": {
            "previous_frame_warm_start": True,
            "velocity_damping_cost": velocity_damping_cost,
            "moving_average_frames": average_frame_size,
            "tracking_iterations_per_interpolated_target": (
                tracking_iterations_per_substep
            ),
            "adaptive_target_interpolation": {
                "object_translation_step_m": 0.001,
                "fingertip_translation_step_m": 0.0015,
                "rotation_step_deg": 5.0,
                "maximum_substeps": 40,
            },
            "exact_collision_projection_after_each_interpolated_target": True,
        },
        "metrics": metrics,
        "seed": seed,
        "video": str(video_path) if save_video else None,
    }
    (robot_dir / "mink_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    tyro.cli(main)
