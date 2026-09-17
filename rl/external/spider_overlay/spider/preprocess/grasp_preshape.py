"""Build a contact-aware xHand pre-grasp/grasp trajectory with MINK.

The native SPIDER IK tracks human fingertips independently.  A physically valid
opposed contact can therefore appear only briefly and disappear before MJWP sees
enough of it to transport the object.  This module samples antipodal contacts
from the current object geometry, solves reachable candidates for every xHand
finger, approaches the best verified grasp without forbidden collision, and
retargets it along the annotated object trajectory.

Only the selected distal fingertip geoms may touch the object.  MINK enforces
joint limits, hand self-collision, non-contact hand-object separation, and floor
separation.  The saved trajectory is accepted only after a second MuJoCo pass
verifies those constraints and sustained physical thumb opposition.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


class PreshapeInfeasibleError(RuntimeError):
    """The generic search completed without a physically acceptable grasp."""


@dataclass(frozen=True)
class ContactState:
    """Physical hand-object contact state for one configuration."""

    fingers: frozenset[str]
    penetration_by_finger: dict[str, float]
    away_normal_by_finger: dict[str, np.ndarray]


@dataclass
class GraspCandidate:
    """One geometry-sampled opposed-contact IK candidate."""

    qpos: np.ndarray
    secondary_finger: str
    surface_points: dict[str, np.ndarray]
    surface_normals: dict[str, np.ndarray]
    direction_index: int
    assignment_swapped: bool
    solver_failed: bool
    target_error_m: float
    contact_state: ContactState
    forbidden_collisions: list[dict[str, Any]]
    grasp_floor_penetration_m: float
    trajectory_floor_penetration_m: float


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(index)) or ""


def _transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation).reshape(3, 3)
    result[:3, 3] = position
    return result


def _body_transform(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str
) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"body not found in model: {body_name}")
    return _transform(data.xpos[body_id], data.xmat[body_id])


def _site_transform(
    model: mujoco.MjModel, data: mujoco.MjData, site_name: str
) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"site not found in model: {site_name}")
    return _transform(data.site_xpos[site_id], data.site_xmat[site_id])


def _contact_state(
    model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> ContactState:
    penetrations: dict[str, list[float]] = {finger: [] for finger in FINGER_ORDER}
    away_normals: dict[str, list[np.ndarray]] = {
        finger: [] for finger in FINGER_ORDER
    }
    object_token = f"{side}_object"
    for contact in data.contact[: data.ncon]:
        first = _name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        second = _name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if object_token not in first and object_token not in second:
            continue
        for finger in FINGER_ORDER:
            finger_token = f"{side}_{finger}"
            if finger_token in first and object_token in second:
                penetrations[finger].append(max(0.0, -float(contact.dist)))
                # MuJoCo's contact x-axis points from geom1 towards geom2.
                away_normals[finger].append(-np.asarray(contact.frame[:3]).copy())
            elif object_token in first and finger_token in second:
                penetrations[finger].append(max(0.0, -float(contact.dist)))
                away_normals[finger].append(np.asarray(contact.frame[:3]).copy())
    active = frozenset(finger for finger, values in penetrations.items() if values)
    max_penetration = {
        finger: max(values) for finger, values in penetrations.items() if values
    }
    mean_normals: dict[str, np.ndarray] = {}
    for finger, values in away_normals.items():
        if not values:
            continue
        normal = np.mean(values, axis=0)
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            mean_normals[finger] = normal / norm
    return ContactState(active, max_penetration, mean_normals)


def _collision_geometry(
    model: mujoco.MjModel, side: str, secondary_finger: str
) -> tuple[list[str], list[str], list[str]]:
    """Enable collision bits for MINK and return relevant geom groups."""
    hand_geoms: list[str] = []
    object_geoms: list[str] = []
    for geom_id in range(model.ngeom):
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        is_hand = geom_name.startswith(f"collision_hand_{side}_")
        suffix = geom_name.rsplit("_", maxsplit=1)[-1]
        is_object = geom_name.startswith(f"{side}_object_") and suffix.isdigit()
        if is_hand:
            hand_geoms.append(geom_name)
        if is_object:
            object_geoms.append(geom_name)
        if is_hand or is_object or geom_name == "floor":
            # Generated SPIDER scenes use explicit <contact><pair> entries and
            # leave these masks at zero.  MINK checks the masks before calling
            # mj_geomDistance, so enable them only in this in-memory IK model.
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 1
    # Contact is legal on any link of the two selected fingers.  Restricting
    # this to the terminal capsule rejects valid xHand grasps in which the
    # thumb's penultimate capsule shares the load.
    selected_geoms = [
        name
        for name in hand_geoms
        if f"_{side}_thumb_" in name or f"_{side}_{secondary_finger}_" in name
    ]
    if not selected_geoms:
        raise ValueError("selected xHand finger collision geoms are missing")
    if not object_geoms:
        raise ValueError(f"no convex {side} object collision geoms found")
    return hand_geoms, object_geoms, selected_geoms


def _limits(
    model: mujoco.MjModel,
    hand_geoms: list[str],
    object_geoms: list[str],
    selected_geoms: list[str],
    *,
    allow_selected_contact: bool,
    collision_margin_m: float,
    contact_penetration_m: float,
) -> list[mink.Limit]:
    nonselected = [name for name in hand_geoms if name not in selected_geoms]
    limits: list[mink.Limit] = [
        mink.ConfigurationLimit(model),
        mink.CollisionAvoidanceLimit(
            model,
            [(hand_geoms, hand_geoms)],
            minimum_distance_from_collisions=collision_margin_m,
            collision_detection_distance=0.02,
            bound_relaxation=1e-4,
        ),
        mink.CollisionAvoidanceLimit(
            model,
            [(nonselected, object_geoms)],
            minimum_distance_from_collisions=collision_margin_m,
            collision_detection_distance=0.02,
            bound_relaxation=1e-4,
        ),
        mink.CollisionAvoidanceLimit(
            model,
            [(hand_geoms, ["floor"])],
            minimum_distance_from_collisions=collision_margin_m,
            collision_detection_distance=0.02,
            bound_relaxation=1e-4,
        ),
    ]
    limits.append(
        mink.CollisionAvoidanceLimit(
            model,
            [(selected_geoms, object_geoms)],
            minimum_distance_from_collisions=(
                -contact_penetration_m
                if allow_selected_contact
                else collision_margin_m
            ),
            collision_detection_distance=0.02,
            bound_relaxation=2e-4 if allow_selected_contact else 1e-4,
        )
    )
    return limits


def _object_dofs(model: mujoco.MjModel, side: str) -> list[int]:
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_object_joint"
    )
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{side}_object_joint must be a free joint")
    first = int(model.jnt_dofadr[joint_id])
    return list(range(first, first + 6))


def _solve(
    configuration: mink.Configuration,
    tasks: list[mink.Task],
    limits: list[mink.Limit],
    freeze_object: mink.DofFreezingTask,
    *,
    iterations: int,
    dt: float,
) -> None:
    for _ in range(iterations):
        velocity = mink.solve_ik(
            configuration,
            tasks,
            dt=dt,
            solver="daqp",
            damping=1e-4,
            limits=limits,
            constraints=[freeze_object],
        )
        configuration.integrate_inplace(velocity, dt)


def _make_tasks(
    model: mujoco.MjModel,
    side: str,
    secondary_finger: str,
    palm_target: np.ndarray,
    fingertip_targets: dict[str, np.ndarray],
    posture_target: np.ndarray,
) -> list[mink.Task]:
    palm = mink.FrameTask(
        f"{side}_palm",
        "site",
        position_cost=20.0,
        orientation_cost=8.0,
        lm_damping=1e-4,
    )
    palm.set_target(mink.SE3.from_matrix(palm_target))
    posture = mink.PostureTask(model, cost=0.05, lm_damping=1e-4)
    posture.set_target(posture_target)
    tasks: list[mink.Task] = [palm, posture]
    for finger in ("thumb", secondary_finger):
        task = mink.FrameTask(
            f"{side}_{finger}_tip",
            "site",
            position_cost=120.0,
            orientation_cost=0.0,
            lm_damping=1e-4,
        )
        task.set_target(mink.SE3.from_translation(fingertip_targets[finger]))
        tasks.append(task)
    return tasks


def _object_geom_ids(model: mujoco.MjModel, side: str) -> list[int]:
    result = []
    for geom_id in range(model.ngeom):
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        suffix = geom_name.rsplit("_", maxsplit=1)[-1]
        if geom_name.startswith(f"{side}_object_") and suffix.isdigit():
            result.append(geom_id)
    if not result:
        raise ValueError(f"no convex {side} object mesh geoms found")
    return result


def _floor_height(model: mujoco.MjModel, data: mujoco.MjData) -> float | None:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0 or model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE:
        return None
    return float(data.geom_xpos[floor_id, 2])


def _ray_hit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: list[int],
    origin: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for geom_id in geom_ids:
        normal = np.zeros(3, dtype=np.float64)
        distance = mujoco.mj_rayMesh(
            model, data, geom_id, origin, direction, normal
        )
        if distance < 0.0 or (best is not None and distance >= best[0]):
            continue
        best = (float(distance), origin + distance * direction, normal.copy())
    if best is None:
        return None
    _, point, normal = best
    normal /= max(np.linalg.norm(normal), 1e-9)
    return point, normal


def _sample_contact_pairs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    *,
    angular_samples: int = 12,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Sample antipodal surface pairs from the current convex object meshes.

    Horizontal rays cover tabletop grasps.  Three tilted rings make the same
    procedure usable when the object is airborne or when a side face is not
    reachable from the horizontal mid-plane.  Contact heights scale with the
    object's collision radius rather than with a particular video.
    """
    geom_ids = _object_geom_ids(model, side)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object")
    center = data.xpos[body_id].copy()
    radius = max(
        np.linalg.norm(data.geom_xpos[geom_id] - center)
        + float(model.geom_rbound[geom_id])
        for geom_id in geom_ids
    )
    ray_length = max(0.05, 2.5 * radius)
    small_offset = 0.12 * radius
    large_offset = 0.25 * radius
    height_patterns = [
        (0.0, 0.0),
        (small_offset, -small_offset),
        (-small_offset, small_offset),
        (large_offset, 0.0),
        (0.0, large_offset),
    ]
    # Sample several gravity-relative cross-sections.  This is necessary for
    # tall objects whose collision-frame origin lies near the support plane;
    # a fixed few-millimetre band around the origin would reject every legal
    # side contact.  Fractions of the measured collision radius transfer across
    # object classes and scene scales.
    height_patterns.extend(
        (fraction * radius, fraction * radius)
        for fraction in (-0.50, -0.25, 0.25, 0.50)
    )
    floor_z = _floor_height(model, data)
    clearance = min(0.008, 0.25 * radius)
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    directions: list[np.ndarray] = []
    for elevation in (0.0, np.deg2rad(25.0), np.deg2rad(-25.0)):
        for sample in range(angular_samples):
            azimuth = 2.0 * np.pi * sample / angular_samples
            directions.append(
                np.array(
                    [
                        np.cos(elevation) * np.cos(azimuth),
                        np.cos(elevation) * np.sin(azimuth),
                        np.sin(elevation),
                    ]
                )
            )
    for direction in directions:
        for plus_z, minus_z in height_patterns:
            plus_origin = center + ray_length * direction
            minus_origin = center - ray_length * direction
            # A world-z cross-section is deliberate: the feasibility filter is
            # relative to gravity/floor, independent of object orientation.
            plus_origin[2] += plus_z
            minus_origin[2] += minus_z
            plus = _ray_hit(model, data, geom_ids, plus_origin, -direction)
            minus = _ray_hit(model, data, geom_ids, minus_origin, direction)
            if plus is None or minus is None:
                continue
            plus_point, plus_normal = plus
            minus_point, minus_normal = minus
            if floor_z is not None and min(plus_point[2], minus_point[2]) < (
                floor_z + clearance
            ):
                continue
            if np.linalg.norm(plus_point - minus_point) < 2.0 * clearance:
                continue
            pairs.append(
                (plus_point, plus_normal, minus_point, minus_normal)
            )
    # Ray intersections can repeat across neighbouring convex pieces/angles.
    unique: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    for pair in pairs:
        key = tuple(np.round(np.concatenate([pair[0], pair[2]]) * 1000).astype(int))
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


def _finger_priority(contact: np.ndarray, grasp_start: int) -> list[str]:
    """Prioritize observed fingers, while always retaining all xHand options."""
    stop = min(len(contact), grasp_start + 10)
    counts = contact[grasp_start:stop, 1:5].sum(axis=0)
    ranked = sorted(
        enumerate(FINGER_ORDER[1:]),
        key=lambda item: (-float(counts[item[0]]), item[0]),
    )
    return [finger for _, finger in ranked]


def _physical_contact_labels(
    model_path: Path, qpos: np.ndarray, side: str
) -> np.ndarray:
    """Recover legacy/missing contact labels from MuJoCo collision geometry."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    # This call enables collision masks for every hand and object geom.  The
    # selected finger affects only the returned list, which is unused here.
    _collision_geometry(model, side, "index")
    data = mujoco.MjData(model)
    labels = np.zeros((len(qpos), len(FINGER_ORDER)), dtype=bool)
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        contacts = _contact_state(model, data, side)
        for channel, finger in enumerate(FINGER_ORDER):
            labels[frame, channel] = finger in contacts.fingers
    return labels


def _candidate_score(candidate: GraspCandidate) -> tuple[float, ...]:
    selected = {"thumb", candidate.secondary_finger}
    opposed = selected.issubset(candidate.contact_state.fingers)
    penetration = max(
        (
            candidate.contact_state.penetration_by_finger.get(finger, 1.0)
            for finger in selected
        ),
        default=1.0,
    )
    return (
        0.0 if opposed else 1.0,
        float(len(candidate.forbidden_collisions)),
        1.0 if candidate.solver_failed else 0.0,
        penetration,
        candidate.grasp_floor_penetration_m,
        candidate.trajectory_floor_penetration_m,
        candidate.target_error_m,
    )


def _trajectory_floor_lift_profile(
    model: mujoco.MjModel,
    grasp_qpos: np.ndarray,
    object_trajectory: np.ndarray,
    side: str,
) -> np.ndarray:
    """Minimum shared hand-object z correction for every trajectory frame."""
    data = mujoco.MjData(model)
    data.qpos[:] = grasp_qpos
    mujoco.mj_forward(model, data)
    grasp_object = _body_transform(model, data, f"{side}_object")
    object_inverse = np.linalg.inv(grasp_object)
    hand_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).startswith(
            f"collision_hand_{side}_"
        )
    ]
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_object_joint"
    )
    object_address = int(model.jnt_qposadr[object_joint])
    local_geoms = {
        geom_id: object_inverse
        @ _transform(data.geom_xpos[geom_id], data.geom_xmat[geom_id])
        for geom_id in hand_geom_ids
    }
    profile = np.zeros(len(object_trajectory), dtype=np.float64)
    for frame, state in enumerate(object_trajectory):
        data.qpos[:] = grasp_qpos
        data.qpos[object_address : object_address + 7] = state[
            object_address : object_address + 7
        ]
        mujoco.mj_forward(model, data)
        floor_z = _floor_height(model, data)
        if floor_z is None:
            return profile
        object_world = _body_transform(model, data, f"{side}_object")
        frame_worst = 0.0
        for geom_id, local_transform in local_geoms.items():
            world = object_world @ local_transform
            rotation = world[:3, :3]
            position = world[:3, 3]
            geom_type = model.geom_type[geom_id]
            size = model.geom_size[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent_z = float(size[0])
            elif geom_type in {
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
            }:
                extent_z = float(size[0] + abs(rotation[2, 2]) * size[1])
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                extent_z = float(np.abs(rotation[2]) @ size[:3])
            else:
                extent_z = float(model.geom_rbound[geom_id])
            frame_worst = max(
                frame_worst, floor_z - (float(position[2]) - extent_z)
            )
        profile[frame] = max(0.0, frame_worst)
    return profile


def _search_grasp_candidate(
    model_path: Path,
    qpos: np.ndarray,
    contact: np.ndarray,
    grasp_start: int,
    side: str,
    *,
    collision_margin_m: float,
    contact_penetration_m: float,
    solver_dt: float,
) -> tuple[list[GraspCandidate], dict[str, Any]]:
    """Search object geometry and xHand fingers for a feasible opposed grasp."""
    sampling_model = mujoco.MjModel.from_xml_path(str(model_path))
    sampling_data = mujoco.MjData(sampling_model)
    sampling_data.qpos[:] = qpos[grasp_start]
    mujoco.mj_forward(sampling_model, sampling_data)
    pairs = _sample_contact_pairs(sampling_model, sampling_data, side)
    if not pairs:
        raise PreshapeInfeasibleError(
            "object mesh ray sampling produced no legal antipodal pairs"
        )
    # Keep runtime bounded for complex convex decompositions while preserving
    # angular, height, and elevation diversity.
    if len(pairs) > 72:
        indices = np.linspace(0, len(pairs) - 1, 72).astype(int)
        pairs = [pairs[index] for index in indices]

    candidates: list[GraspCandidate] = []
    for secondary_finger in _finger_priority(contact, grasp_start):
        model = mujoco.MjModel.from_xml_path(str(model_path))
        hand_geoms, object_geoms, selected_geoms = _collision_geometry(
            model, side, secondary_finger
        )
        limits = _limits(
            model,
            hand_geoms,
            object_geoms,
            selected_geoms,
            allow_selected_contact=True,
            collision_margin_m=collision_margin_m,
            contact_penetration_m=contact_penetration_m,
        )
        freeze_object = mink.DofFreezingTask(model, _object_dofs(model, side))
        initial = mink.Configuration(model, q=qpos[grasp_start].copy())
        palm_target = _site_transform(model, initial.data, f"{side}_palm")
        # Median distal capsule radius is a morphology-derived conversion from
        # surface contact to fingertip-site target (8 mm for the current xHand).
        distal_radii = []
        for finger in ("thumb", secondary_finger):
            geom_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"collision_hand_{side}_{finger}_0",
            )
            if geom_id >= 0:
                distal_radii.append(float(model.geom_size[geom_id, 0]))
        site_offset = 0.8 * float(np.median(distal_radii or [0.01]))
        for direction_index, pair in enumerate(pairs):
            plus_point, plus_normal, minus_point, minus_normal = pair
            for swapped in (False, True):
                surface_points = {
                    "thumb": minus_point if swapped else plus_point,
                    secondary_finger: plus_point if swapped else minus_point,
                }
                surface_normals = {
                    "thumb": minus_normal if swapped else plus_normal,
                    secondary_finger: plus_normal if swapped else minus_normal,
                }
                targets = {
                    finger: surface_points[finger]
                    + site_offset * surface_normals[finger]
                    for finger in ("thumb", secondary_finger)
                }
                configuration = mink.Configuration(
                    model, q=qpos[grasp_start].copy()
                )
                tasks = _make_tasks(
                    model,
                    side,
                    secondary_finger,
                    palm_target,
                    targets,
                    qpos[grasp_start],
                )
                solver_failed = False
                try:
                    _solve(
                        configuration,
                        tasks,
                        limits,
                        freeze_object,
                        iterations=60,
                        dt=solver_dt,
                    )
                except Exception:
                    # DAQP can report infeasibility after reaching a useful
                    # iterate; retain it but rank it below a completed solve.
                    solver_failed = True
                mujoco.mj_forward(model, configuration.data)
                contact_state = _contact_state(model, configuration.data, side)
                forbidden = _forbidden_collisions(
                    model,
                    configuration.data,
                    side,
                    set(selected_geoms),
                    tolerance_m=5e-4,
                )
                # Support-plane overlap is corrected later by a measured shared
                # hand-object lift.  Keep uncorrectable self/object collisions
                # in candidate feasibility, but do not conflate the two.
                forbidden = [
                    item
                    for item in forbidden
                    if "floor" not in {item["geom1"], item["geom2"]}
                ]
                target_error = sum(
                    np.linalg.norm(
                        _site_transform(
                            model, configuration.data, f"{side}_{finger}_tip"
                        )[:3, 3]
                        - targets[finger]
                    )
                    for finger in ("thumb", secondary_finger)
                )
                floor_profile = _trajectory_floor_lift_profile(
                    model,
                    configuration.q,
                    qpos,
                    side,
                )
                candidates.append(
                    GraspCandidate(
                        qpos=configuration.q.copy(),
                        secondary_finger=secondary_finger,
                        surface_points={
                            key: value.copy() for key, value in surface_points.items()
                        },
                        surface_normals={
                            key: value.copy() for key, value in surface_normals.items()
                        },
                        direction_index=direction_index,
                        assignment_swapped=swapped,
                        solver_failed=solver_failed,
                        target_error_m=float(target_error),
                        contact_state=contact_state,
                        forbidden_collisions=forbidden,
                        grasp_floor_penetration_m=float(
                            floor_profile[grasp_start]
                        ),
                        trajectory_floor_penetration_m=float(
                            floor_profile.max(initial=0.0)
                        ),
                    )
                )
    opposed_candidates = [
        candidate
        for candidate in candidates
        if {"thumb", candidate.secondary_finger}.issubset(
            candidate.contact_state.fingers
        )
        and not candidate.forbidden_collisions
    ]
    if not opposed_candidates:
        raise PreshapeInfeasibleError(
            "no geometry-sampled MINK candidate achieved thumb opposition"
        )
    by_floor = sorted(
        opposed_candidates,
        key=lambda item: (
            item.solver_failed,
            item.trajectory_floor_penetration_m,
            item.grasp_floor_penetration_m,
            _candidate_score(item),
        ),
    )
    by_penetration = sorted(opposed_candidates, key=_candidate_score)
    shortlist: list[GraspCandidate] = []
    seen: set[tuple[str, int, bool]] = set()
    for candidate in by_floor[:4] + by_penetration[:4]:
        key = (
            candidate.secondary_finger,
            candidate.direction_index,
            candidate.assignment_swapped,
        )
        if key not in seen:
            seen.add(key)
            shortlist.append(candidate)
    return shortlist, {
        "surface_pair_count": len(pairs),
        "candidate_count": len(candidates),
        "opposed_collision_free_candidate_count": len(opposed_candidates),
        "refinement_shortlist_count": len(shortlist),
        "secondary_finger_order": _finger_priority(contact, grasp_start),
        "coarse_lowest_floor_score": list(_candidate_score(by_floor[0])),
        "coarse_lowest_penetration_score": list(
            _candidate_score(by_penetration[0])
        ),
    }


def _refine_candidate(
    model_path: Path,
    candidate: GraspCandidate,
    object_trajectory: np.ndarray,
    side: str,
    *,
    collision_margin_m: float,
    contact_penetration_m: float,
    solver_iterations: int,
    solver_dt: float,
    grasp_start: int,
) -> GraspCandidate:
    """Independently back off each contact along its measured normal."""
    secondary_finger = candidate.secondary_finger
    model = mujoco.MjModel.from_xml_path(str(model_path))
    hand_geoms, object_geoms, selected_geoms = _collision_geometry(
        model, side, secondary_finger
    )
    limits = _limits(
        model,
        hand_geoms,
        object_geoms,
        selected_geoms,
        allow_selected_contact=True,
        collision_margin_m=collision_margin_m,
        contact_penetration_m=min(contact_penetration_m, 0.001),
    )
    freeze_object = mink.DofFreezingTask(model, _object_dofs(model, side))
    initial = mink.Configuration(model, q=candidate.qpos.copy())
    mujoco.mj_forward(model, initial.data)
    initial_contacts = _contact_state(model, initial.data, side)
    fingers = ("thumb", secondary_finger)
    if any(
        finger not in initial_contacts.away_normal_by_finger for finger in fingers
    ):
        return candidate
    palm_target = _site_transform(model, initial.data, f"{side}_palm")
    base_targets = {
        finger: _site_transform(model, initial.data, f"{side}_{finger}_tip")[:3, 3]
        for finger in fingers
    }
    desired_penetration = min(5e-4, 0.5 * contact_penetration_m)
    nominal_offsets = {
        finger: max(
            0.0,
            initial_contacts.penetration_by_finger[finger] - desired_penetration,
        )
        for finger in fingers
    }
    refined: list[GraspCandidate] = [candidate]
    deltas = (-5e-4, 0.0, 5e-4)
    for thumb_delta in deltas:
        for secondary_delta in deltas:
            offsets = {
                "thumb": max(0.0, nominal_offsets["thumb"] + thumb_delta),
                secondary_finger: max(
                    0.0, nominal_offsets[secondary_finger] + secondary_delta
                ),
            }
            targets = {
                finger: base_targets[finger]
                + offsets[finger]
                * initial_contacts.away_normal_by_finger[finger]
                for finger in fingers
            }
            configuration = mink.Configuration(model, q=candidate.qpos.copy())
            tasks = _make_tasks(
                model,
                side,
                secondary_finger,
                palm_target,
                targets,
                candidate.qpos,
            )
            solver_failed = False
            try:
                _solve(
                    configuration,
                    tasks,
                    limits,
                    freeze_object,
                    iterations=solver_iterations,
                    dt=solver_dt,
                )
            except Exception:
                solver_failed = True
            mujoco.mj_forward(model, configuration.data)
            contacts = _contact_state(model, configuration.data, side)
            forbidden = _forbidden_collisions(
                model,
                configuration.data,
                side,
                set(selected_geoms),
                tolerance_m=5e-4,
            )
            forbidden = [
                item
                for item in forbidden
                if "floor" not in {item["geom1"], item["geom2"]}
            ]
            selected = {"thumb", secondary_finger}
            if not selected.issubset(contacts.fingers):
                continue
            target_error = sum(
                np.linalg.norm(
                    _site_transform(
                        model, configuration.data, f"{side}_{finger}_tip"
                    )[:3, 3]
                    - targets[finger]
                )
                for finger in fingers
            )
            floor_profile = _trajectory_floor_lift_profile(
                model,
                configuration.q,
                object_trajectory,
                side,
            )
            refined.append(
                GraspCandidate(
                    qpos=configuration.q.copy(),
                    secondary_finger=secondary_finger,
                    surface_points=candidate.surface_points,
                    surface_normals=candidate.surface_normals,
                    direction_index=candidate.direction_index,
                    assignment_swapped=candidate.assignment_swapped,
                    solver_failed=solver_failed,
                    target_error_m=float(target_error),
                    contact_state=contacts,
                    forbidden_collisions=forbidden,
                    grasp_floor_penetration_m=float(
                        floor_profile[grasp_start]
                    ),
                    trajectory_floor_penetration_m=float(
                        floor_profile.max(initial=0.0)
                    ),
                )
            )
    if not refined:
        return candidate
    def refinement_score(item: GraspCandidate) -> tuple[float, ...]:
        selected = {"thumb", item.secondary_finger}
        opposed = selected.issubset(item.contact_state.fingers)
        penetration = max(
            item.contact_state.penetration_by_finger.get(finger, 1.0)
            for finger in selected
        )
        return (
            0.0 if opposed else 1.0,
            float(len(item.forbidden_collisions)),
            penetration,
            1.0 if item.solver_failed else 0.0,
            item.grasp_floor_penetration_m,
            item.trajectory_floor_penetration_m,
            item.target_error_m,
        )

    refined.sort(key=refinement_score)
    return refined[0]


def _forbidden_collisions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    selected_geoms: set[str],
    tolerance_m: float,
) -> list[dict[str, Any]]:
    """Return penetrating contacts other than the two allowed fingertip contacts."""
    violations: list[dict[str, Any]] = []
    object_token = f"{side}_object"
    hand_token = f"collision_hand_{side}_"
    for contact in data.contact[: data.ncon]:
        if float(contact.dist) >= -tolerance_m:
            continue
        first = _name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        second = _name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        names = {first, second}
        hand_names = {name for name in names if name.startswith(hand_token)}
        object_contact = any(name.startswith(object_token) for name in names)
        allowed = bool(object_contact and hand_names and hand_names <= selected_geoms)
        relevant = bool(
            hand_names
            and (
                object_contact
                or "floor" in names
                or len(hand_names) == 2
            )
        )
        if relevant and not allowed:
            violations.append(
                {
                    "geom1": first,
                    "geom2": second,
                    "penetration_m": -float(contact.dist),
                }
            )
    return violations


def _joint_limit_violations(model: mujoco.MjModel, qpos: np.ndarray) -> int:
    limited = np.flatnonzero(
        model.jnt_limited
        & (model.jnt_type != mujoco.mjtJoint.mjJNT_FREE)
    )
    count = 0
    for joint_id in limited:
        address = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        count += int(np.count_nonzero(qpos[:, address] < lower - 1e-6))
        count += int(np.count_nonzero(qpos[:, address] > upper + 1e-6))
    return count


def _qvel(model: mujoco.MjModel, qpos: np.ndarray, frequency: float) -> np.ndarray:
    result = np.zeros((len(qpos), model.nv), dtype=np.float64)
    if len(qpos) < 2:
        return result
    dt = 1.0 / frequency
    for frame in range(1, len(qpos)):
        mujoco.mj_differentiatePos(
            model, result[frame], dt, qpos[frame - 1], qpos[frame]
        )
    result[0] = result[1]
    return result


def _mpc_contact_handoff(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    grasp_start: int,
    selected_fingers: tuple[str, str],
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the synthesized grasp, rather than stale human contacts, for MPC."""
    contact = np.zeros((len(qpos), len(FINGER_ORDER)), dtype=np.float32)
    selected_channels = [FINGER_ORDER.index(finger) for finger in selected_fingers]
    contact[grasp_start:, selected_channels] = 1.0
    contact_pos = np.zeros(
        (len(qpos), len(FINGER_ORDER), 3), dtype=np.float32
    )
    site_ids = []
    for finger in FINGER_ORDER:
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_tip"
        )
        if site_id < 0:
            raise ValueError(f"missing xHand fingertip site: {side}_{finger}_tip")
        site_ids.append(site_id)
    data = mujoco.MjData(model)
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        contact_pos[frame] = data.site_xpos[site_ids]
    return contact, contact_pos


def _set_xhand_wrist_pose(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    palm_target: np.ndarray,
    previous_qpos: np.ndarray,
    side: str,
) -> np.ndarray:
    """Set xHand's xyz + intrinsic Z-X-(-Y) wrist coordinates exactly."""
    if side != "right":
        raise NotImplementedError(
            "analytic wrist transport currently supports right xHand"
        )
    result = qpos.copy()
    translation_joints = [
        "R_forearm_tx_link_joint",
        "R_forearm_ty_link_joint",
        "R_forearm_tz_link_joint",
    ]
    rotation_joints = [
        "R_forearm_roll_link_joint",
        "R_forearm_pitch_link_joint",
        "R_forearm_yaw_link_joint",
    ]
    for axis, joint_name in enumerate(translation_joints):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        result[int(model.jnt_qposadr[joint_id])] = palm_target[axis, 3]
    palm_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm"
    )
    if np.linalg.norm(model.site_pos[palm_site]) > 1e-9:
        raise ValueError(
            "analytic xHand wrist transport requires palm site at body origin"
        )
    palm_local_rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(palm_local_rotation, model.site_quat[palm_site])
    hand_rotation = palm_target[:3, :3] @ palm_local_rotation.reshape(3, 3).T
    zxy = Rotation.from_matrix(hand_rotation).as_euler("ZXY")
    desired = np.array([zxy[0], zxy[1], -zxy[2]])
    for angle, joint_name in zip(desired, rotation_joints, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        address = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        equivalents = angle + 2.0 * np.pi * np.arange(-2, 3)
        feasible = equivalents[(equivalents >= lower) & (equivalents <= upper)]
        if not len(feasible):
            raise RuntimeError(f"no in-limit Euler branch for {joint_name}")
        result[address] = feasible[np.argmin(np.abs(feasible - previous_qpos[address]))]
    return result


def _search_approach(
    model: mujoco.MjModel,
    object_trajectory: np.ndarray,
    grasp_qpos: np.ndarray,
    object_to_palm: np.ndarray,
    side: str,
    selected_geoms: list[str],
    grasp_start: int,
    ramp_start: int,
    base_clearance_m: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Choose a collision-free straight-line pre-grasp approach."""
    grasp_data = mujoco.MjData(model)
    grasp_data.qpos[:] = grasp_qpos
    mujoco.mj_forward(model, grasp_data)
    grasp_object = _body_transform(model, grasp_data, f"{side}_object")
    palm_world = _site_transform(model, grasp_data, f"{side}_palm")[:3, 3]
    radial = palm_world - grasp_object[:3, 3]
    radial /= max(np.linalg.norm(radial), 1e-9)
    directions = [
        radial,
        np.array([0.0, 0.0, 1.0]),
        radial + np.array([0.0, 0.0, 1.0]),
    ]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for sample in range(30):
        z = 1.0 - 2.0 * (sample + 0.5) / 30.0
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        azimuth = sample * golden_angle
        directions.append(
            np.array([radius * np.cos(azimuth), radius * np.sin(azimuth), z])
        )
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_object_joint"
    )
    object_address = int(model.jnt_qposadr[object_joint])
    evaluations: list[tuple[int, float, float, np.ndarray]] = []
    data = mujoco.MjData(model)
    for clearance in (
        base_clearance_m,
        1.5 * base_clearance_m,
        2.0 * base_clearance_m,
    ):
        for world_direction in directions:
            world_direction = world_direction / max(
                np.linalg.norm(world_direction), 1e-9
            )
            local_direction = grasp_object[:3, :3].T @ world_direction
            previous = grasp_qpos.copy()
            collision_frames = 0
            worst_penetration = 0.0
            for frame in range(grasp_start):
                data.qpos[:] = object_trajectory[frame]
                mujoco.mj_forward(model, data)
                object_world = _body_transform(model, data, f"{side}_object")
                alpha = min(
                    1.0,
                    max(
                        0.0,
                        (frame - ramp_start) / max(grasp_start - ramp_start, 1),
                    ),
                )
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                target = object_world @ object_to_palm
                target[:3, 3] += (
                    object_world[:3, :3] @ local_direction
                ) * ((1.0 - alpha) * clearance)
                current = previous.copy()
                current[object_address : object_address + 7] = object_trajectory[
                    frame, object_address : object_address + 7
                ]
                current = _set_xhand_wrist_pose(
                    model, current, target, previous, side
                )
                data.qpos[:] = current
                mujoco.mj_forward(model, data)
                violations = _forbidden_collisions(
                    model,
                    data,
                    side,
                    set(),
                    tolerance_m=5e-4,
                )
                if violations:
                    collision_frames += 1
                    worst_penetration = max(
                        worst_penetration,
                        max(item["penetration_m"] for item in violations),
                    )
                previous = current
            evaluations.append(
                (collision_frames, worst_penetration, clearance, local_direction)
            )
    collision_frames, penetration, clearance, direction = min(
        evaluations, key=lambda item: item[:3]
    )
    return direction, clearance, {
        "candidate_count": len(evaluations),
        "collision_frame_count": collision_frames,
        "maximum_forbidden_penetration_m": penetration,
        "selected_clearance_m": clearance,
        "selected_direction_object_frame": direction.tolist(),
        "selected_contact_geoms": selected_geoms,
    }


def build_grasp_preshape(
    model_path: str | Path,
    trajectory_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    embodiment_type: str = "right",
    approach_frames: int = 12,
    pregrasp_clearance_m: float = 0.03,
    collision_margin_m: float = 0.001,
    contact_penetration_m: float = 0.0025,
    solver_iterations: int = 180,
    solver_dt: float = 0.01,
) -> dict[str, Any]:
    """Create and validate a contact-aware pre-grasp/grasp trajectory."""
    if embodiment_type != "right":
        raise NotImplementedError(
            "contact-aware preshape currently supports the project xHand/right path"
        )
    if approach_frames < 2:
        raise ValueError("approach_frames must be at least 2")
    if pregrasp_clearance_m <= 0.0 or collision_margin_m < 0.0:
        raise ValueError("pregrasp clearance must be positive and margin non-negative")
    if contact_penetration_m <= 0.0:
        raise ValueError("contact_penetration_m must be positive")

    model_path = Path(model_path).resolve()
    trajectory_path = Path(trajectory_path).resolve()
    output_path = Path(output_path).resolve()
    report_path = Path(report_path).resolve()
    physical_model = mujoco.MjModel.from_xml_path(str(model_path))
    with np.load(trajectory_path, allow_pickle=False) as artifact:
        arrays = {key: np.asarray(artifact[key]) for key in artifact.files}
    qpos_original = np.asarray(arrays["qpos"], dtype=np.float64)
    if qpos_original.ndim != 2 or qpos_original.shape[1] != physical_model.nq:
        raise ValueError(
            f"qpos shape {qpos_original.shape} incompatible with nq={physical_model.nq}"
        )
    frequency = float(np.asarray(arrays.get("frequency", 50.0)))

    raw_contact = arrays.get("contact")
    if raw_contact is None:
        reference_contact = np.zeros(
            (len(qpos_original), len(FINGER_ORDER)), dtype=bool
        )
    else:
        raw_contact = np.asarray(raw_contact)
        if raw_contact.ndim != 2 or raw_contact.shape[0] != len(qpos_original):
            raise ValueError("contact labels must have one row per qpos frame")
        if raw_contact.shape[1] < len(FINGER_ORDER):
            raise ValueError("right-hand contact labels require five finger channels")
        # Legacy SPIDER right-hand artifacts can retain bimanual ten-channel
        # layout.  The right hand occupies the final five channels.
        reference_contact = (
            raw_contact[:, -len(FINGER_ORDER) :].astype(np.float64) >= 0.5
        )
    active = reference_contact.any(axis=1)
    persistent_active = np.convolve(active.astype(int), np.ones(3, dtype=int), "same")
    active_frames = np.flatnonzero(persistent_active >= 2)
    grasp_timing_source = "input_contact_labels"
    if not len(active_frames):
        physical_contact = _physical_contact_labels(
            model_path, qpos_original, embodiment_type
        )
        physical_active = physical_contact.any(axis=1)
        persistent_physical = np.convolve(
            physical_active.astype(int), np.ones(3, dtype=int), "same"
        )
        active_frames = np.flatnonzero(persistent_physical >= 2)
        if len(active_frames):
            reference_contact = physical_contact
            grasp_timing_source = "mujoco_physical_contact_fallback"
    if len(active_frames):
        grasp_start = int(active_frames[0])
    else:
        object_joint = mujoco.mj_name2id(
            physical_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{embodiment_type}_object_joint",
        )
        object_address = int(physical_model.jnt_qposadr[object_joint])
        object_position = qpos_original[:, object_address : object_address + 3]
        displacement = np.linalg.norm(object_position - object_position[0], axis=1)
        motion_threshold = max(0.005, 0.1 * float(displacement.max()))
        moving = np.flatnonzero(displacement >= motion_threshold)
        grasp_start = max(0, int(moving[0]) - approach_frames) if len(moving) else 0
        grasp_timing_source = "object_motion_fallback"
    ramp_start = max(0, grasp_start - approach_frames)

    coarse_candidates, search_report = _search_grasp_candidate(
        model_path,
        qpos_original,
        reference_contact,
        grasp_start,
        embodiment_type,
        collision_margin_m=min(collision_margin_m, 5e-4),
        contact_penetration_m=contact_penetration_m,
        solver_dt=solver_dt,
    )
    refined_pairs = [
        (
            coarse_candidate,
            _refine_candidate(
                model_path,
                coarse_candidate,
                qpos_original,
                embodiment_type,
                collision_margin_m=collision_margin_m,
                contact_penetration_m=contact_penetration_m,
                solver_iterations=solver_iterations,
                solver_dt=solver_dt,
                grasp_start=grasp_start,
            ),
        )
        for coarse_candidate in coarse_candidates
    ]

    def final_candidate_score(
        pair: tuple[GraspCandidate, GraspCandidate],
    ) -> tuple[float, ...]:
        candidate = pair[1]
        selected = {"thumb", candidate.secondary_finger}
        opposed = selected.issubset(candidate.contact_state.fingers)
        penetration = max(
            candidate.contact_state.penetration_by_finger.get(finger, 1.0)
            for finger in selected
        )
        return (
            0.0 if opposed else 1.0,
            float(len(candidate.forbidden_collisions)),
            0.0 if penetration <= contact_penetration_m else 1.0,
            candidate.trajectory_floor_penetration_m,
            candidate.grasp_floor_penetration_m,
            penetration,
            1.0 if candidate.solver_failed else 0.0,
        )

    coarse_candidate, grasp_candidate = min(
        refined_pairs, key=final_candidate_score
    )
    search_report["selected_final_score"] = list(
        final_candidate_score((coarse_candidate, grasp_candidate))
    )
    secondary_finger = grasp_candidate.secondary_finger

    clearance_model = mujoco.MjModel.from_xml_path(str(model_path))
    _collision_geometry(clearance_model, embodiment_type, secondary_finger)
    raw_floor_lift = _trajectory_floor_lift_profile(
        clearance_model,
        grasp_candidate.qpos,
        qpos_original,
        embodiment_type,
    )
    # A short max filter anticipates fast object rotations; unlike an average,
    # it cannot reduce a required collision clearance.
    padded_lift = np.pad(raw_floor_lift, (2, 2), mode="edge")
    floor_lift = np.array(
        [padded_lift[index : index + 5].max() for index in range(len(raw_floor_lift))]
    )
    floor_lift = np.where(
        floor_lift > 0.0, floor_lift + collision_margin_m, 0.0
    )
    # Before physical grasp closure the object must remain supported by its
    # native trajectory; lifting it to clear a not-yet-attached hand makes the
    # MPC initial state float and fail the stability check.
    floor_lift[: max(grasp_start - 1, 0)] = 0.0
    object_joint_for_lift = mujoco.mj_name2id(
        physical_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"{embodiment_type}_object_joint",
    )
    object_address_for_lift = int(
        physical_model.jnt_qposadr[object_joint_for_lift]
    )
    qpos_original[:, object_address_for_lift + 2] += floor_lift
    if "contact_pos" in arrays and len(arrays["contact_pos"]) == len(floor_lift):
        arrays["contact_pos"] = np.asarray(arrays["contact_pos"]).copy()
        arrays["contact_pos"][:, :, 2] += floor_lift[:, None]

    grasp_qpos = grasp_candidate.qpos.copy()
    wrist_z_joint = mujoco.mj_name2id(
        physical_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "R_forearm_tz_link_joint",
    )
    if wrist_z_joint < 0:
        raise ValueError("xHand wrist z slide joint is missing")
    wrist_z_address = int(physical_model.jnt_qposadr[wrist_z_joint])
    grasp_qpos[wrist_z_address] += floor_lift[grasp_start]
    grasp_qpos[
        object_address_for_lift : object_address_for_lift + 7
    ] = qpos_original[
        grasp_start, object_address_for_lift : object_address_for_lift + 7
    ]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    _, _, selected_geoms = _collision_geometry(
        model, embodiment_type, secondary_finger
    )

    grasp_data = mujoco.MjData(model)
    grasp_data.qpos[:] = grasp_qpos
    mujoco.mj_forward(model, grasp_data)
    grasp_object = _body_transform(model, grasp_data, f"{embodiment_type}_object")
    object_to_palm = np.linalg.inv(grasp_object) @ _site_transform(
        model, grasp_data, f"{embodiment_type}_palm"
    )
    fingers = ("thumb", secondary_finger)
    approach_direction, selected_clearance, approach_report = _search_approach(
        model,
        qpos_original,
        grasp_qpos,
        object_to_palm,
        embodiment_type,
        selected_geoms,
        grasp_start,
        ramp_start,
        pregrasp_clearance_m,
    )

    qpos = qpos_original.copy()
    object_data = mujoco.MjData(model)
    previous = grasp_qpos.copy()
    solver_failures: list[dict[str, Any]] = []
    object_joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"{embodiment_type}_object_joint",
    )
    object_qpos_address = int(model.jnt_qposadr[object_joint])
    for frame in range(len(qpos)):
        object_data.qpos[:] = qpos_original[frame]
        mujoco.mj_forward(model, object_data)
        object_world = _body_transform(
            model, object_data, f"{embodiment_type}_object"
        )
        alpha = min(
            1.0,
            max(0.0, (frame - ramp_start) / max(grasp_start - ramp_start, 1)),
        )
        # Smoothstep avoids a velocity discontinuity at pre-grasp and contact.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        clearance = (1.0 - alpha) * selected_clearance
        palm_target = object_world @ object_to_palm
        palm_target[:3, 3] += (
            object_world[:3, :3] @ approach_direction
        ) * clearance
        current = previous.copy()
        # Preserve the annotated object exactly; MINK freezes its six velocity DOFs.
        current[object_qpos_address : object_qpos_address + 7] = qpos_original[
            frame, object_qpos_address : object_qpos_address + 7
        ]
        try:
            current = _set_xhand_wrist_pose(
                model,
                current,
                palm_target,
                previous,
                embodiment_type,
            )
        except Exception as error:
            solver_failures.append(
                {"frame": frame, "error": f"{type(error).__name__}: {error}"}
            )
        qpos[frame] = current
        previous = current.copy()

    # Validate with masks enabled so self/environment collisions omitted from
    # the generated explicit contact-pair list are still visible.
    validation_data = mujoco.MjData(model)
    forbidden_frames: list[dict[str, Any]] = []
    opposed_frames: list[int] = []
    selected_penetrations: list[float] = []
    for frame in range(len(qpos)):
        validation_data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, validation_data)
        contacts = _contact_state(model, validation_data, embodiment_type)
        if set(fingers).issubset(contacts.fingers):
            opposed_frames.append(frame)
            selected_penetrations.extend(
                contacts.penetration_by_finger[finger] for finger in fingers
            )
        violations = _forbidden_collisions(
            model,
            validation_data,
            embodiment_type,
            set(selected_geoms),
            tolerance_m=5e-4,
        )
        if violations:
            forbidden_frames.append(
                {"frame": frame, "contacts": violations[:10]}
            )

    grasp_frame_count = len(qpos) - grasp_start
    opposed_grasp_frames = [frame for frame in opposed_frames if frame >= grasp_start]
    joint_limit_violations = _joint_limit_violations(model, qpos)
    max_contact_penetration = max(selected_penetrations, default=0.0)
    accepted = bool(
        not forbidden_frames
        and joint_limit_violations == 0
        and len(opposed_grasp_frames) >= 3
        and len(opposed_grasp_frames) / max(grasp_frame_count, 1) >= 0.9
        and max_contact_penetration <= contact_penetration_m + 5e-4
    )

    mpc_contact, mpc_contact_pos = _mpc_contact_handoff(
        model,
        qpos,
        grasp_start,
        fingers,
        embodiment_type,
    )
    arrays["qpos"] = qpos
    arrays["qvel"] = _qvel(model, qpos, frequency)
    arrays["contact"] = mpc_contact
    arrays["contact_pos"] = mpc_contact_pos
    arrays.pop("qpos_rollout", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "accepted": accepted,
        "method": "MINK object-relative contact-constrained xHand retargeting",
        "model_path": str(model_path),
        "input_trajectory": str(trajectory_path),
        "output_trajectory": str(output_path),
        "embodiment_type": embodiment_type,
        "selected_fingers": ["thumb", secondary_finger],
        "grasp_search": {
            **search_report,
            "grasp_frame": grasp_start,
            "selected_direction_index": grasp_candidate.direction_index,
            "assignment_swapped": grasp_candidate.assignment_swapped,
            "coarse_penetration_m": (
                coarse_candidate.contact_state.penetration_by_finger
            ),
            "refined_penetration_m": (
                grasp_candidate.contact_state.penetration_by_finger
            ),
            "refined_forbidden_collision_count": len(
                grasp_candidate.forbidden_collisions
            ),
            "rigid_transport_floor_penetration_m": (
                grasp_candidate.trajectory_floor_penetration_m
            ),
            "grasp_frame_floor_penetration_m": (
                grasp_candidate.grasp_floor_penetration_m
            ),
            "shared_floor_correction_m": {
                "maximum": float(floor_lift.max(initial=0.0)),
                "mean": float(floor_lift.mean()),
                "active_frame_count": int(np.count_nonzero(floor_lift > 0.0)),
                "policy": (
                    "minimum hand-object shared z lift plus collision margin; "
                    "five-frame conservative max filter; disabled until the "
                    "final approach frame"
                ),
            },
        },
        "timeline": {
            "pregrasp_start_frame": 0,
            "approach_ramp_start_frame": ramp_start,
            "grasp_start_frame": grasp_start,
            "grasp_timing_source": grasp_timing_source,
            "grasp_end_frame_exclusive": len(qpos),
            "approach_frames": grasp_start - ramp_start,
            "approach_search": approach_report,
        },
        "constraints": {
            "joint_limits": True,
            "self_collision_margin_m": collision_margin_m,
            "noncontact_object_margin_m": collision_margin_m,
            "floor_margin_m": collision_margin_m,
            "allowed_contact_penetration_m": contact_penetration_m,
            "object_dofs_frozen_during_each_solve": True,
        },
        "validation": {
            "solver_failure_count": len(solver_failures),
            "solver_failures": solver_failures,
            "joint_limit_violation_count": joint_limit_violations,
            "forbidden_collision_frame_count": len(forbidden_frames),
            "forbidden_collision_frames": forbidden_frames[:20],
            "opposed_grasp_frame_count": len(opposed_grasp_frames),
            "grasp_frame_count": grasp_frame_count,
            "opposed_grasp_frame_rate": len(opposed_grasp_frames)
            / max(grasp_frame_count, 1),
            "max_selected_contact_penetration_m": max_contact_penetration,
        },
        "mpc_handoff": {
            "contact_channels": list(FINGER_ORDER),
            "active_fingers_from_grasp": list(fingers),
            "contact_activation_frame": grasp_start,
            "contact_position_policy": (
                "synthesized xHand fingertip-site reference positions"
            ),
            "reused_input_contact_labels": False,
        },
        "handoff": "MPC when accepted; native trajectory fallback otherwise",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run contact-aware xHand preshape generation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trajectory-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument(
        "--embodiment-type", choices=["right", "left", "bimanual"], default="right"
    )
    parser.add_argument("--approach-frames", type=int, default=12)
    parser.add_argument("--pregrasp-clearance-m", type=float, default=0.03)
    parser.add_argument("--collision-margin-m", type=float, default=0.001)
    parser.add_argument("--contact-penetration-m", type=float, default=0.0025)
    parser.add_argument("--solver-iterations", type=int, default=180)
    parser.add_argument("--solver-dt", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:
        report = build_grasp_preshape(
            args.model_path,
            args.trajectory_path,
            args.output_path,
            args.report_path,
            embodiment_type=args.embodiment_type,
            approach_frames=args.approach_frames,
            pregrasp_clearance_m=args.pregrasp_clearance_m,
            collision_margin_m=args.collision_margin_m,
            contact_penetration_m=args.contact_penetration_m,
            solver_iterations=args.solver_iterations,
            solver_dt=args.solver_dt,
        )
    except PreshapeInfeasibleError as error:
        report = {
            "schema_version": "1.0",
            "accepted": False,
            "status": "infeasible",
            "method": "MINK object-relative contact-constrained xHand retargeting",
            "model_path": str(args.model_path.resolve()),
            "input_trajectory": str(args.trajectory_path.resolve()),
            "output_trajectory": None,
            "embodiment_type": args.embodiment_type,
            "reason": str(error),
            "handoff": "native trajectory fallback",
        }
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
