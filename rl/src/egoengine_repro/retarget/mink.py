"""Confidence-weighted MINK retargeter with feasibility diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..action.contracts import (
    HUMAN_CONTACT_POSITION_SOURCE,
    XHAND_FINGER_ORDER,
    contact_reference_signature,
    human_mano_signature,
    mujoco_model_signature,
    precontact_clearance_mask,
    reference_trajectory_signature,
)
from ..action.geometry import physical_object_geom_ids
from ..config import ReproConfig
from .schema import validate_human_reference, validate_robot_reference
from .collision_audit import audit_intrahand_trajectory


FINGER_NAMES = tuple(f"{finger}_tip" for finger in XHAND_FINGER_ORDER)
FINGER_DIRECTION_AXES = ((0, 1.0), (2, -1.0), (2, -1.0), (2, -1.0), (2, -1.0))


class _VelocityLock:
    """Freeze a free object's tangent velocity during kinematic retargeting.

    The object pose is GT input for geometry queries, not an IK degree of
    freedom.  Without this lock an object-collision constraint can be satisfied
    by moving the object's free joint in the QP, only for that movement to be
    overwritten immediately by the next GT-pose update.
    """

    def __init__(self, model: Any, dof_address: int, constraint_type: Any):
        self.indices = np.arange(int(dof_address), int(dof_address) + 6, dtype=np.int64)
        self.model = model
        self.constraint_type = constraint_type

    def compute_qp_inequalities(self, _configuration: Any, _dt: float) -> Any:
        matrix = np.zeros((12, self.model.nv), dtype=np.float64)
        matrix[:6, self.indices] = np.eye(6)
        matrix[6:, self.indices] = -np.eye(6)
        return self.constraint_type(matrix, np.zeros(12, dtype=np.float64))


class _FrameDisplacementLimit:
    """Enforce one physical velocity envelope across all IK sub-iterations.

    MINK's stock velocity limit applies to each solver call independently.
    Contact/collision cleanup adds calls, so the aggregate frame-to-frame
    displacement can silently exceed the declared joint speed.  This limit is
    anchored to the previous accepted frame and cannot be reset by extra QPs.
    """

    def __init__(
        self, model: Any, mujoco: Any, velocity_map: dict[str, float],
        constraint_type: Any,
    ) -> None:
        dofs: list[int] = []
        qpos_addresses: list[int] = []
        speeds: list[float] = []
        for joint in range(int(model.njnt)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
            if name not in velocity_map:
                continue
            if int(model.jnt_type[joint]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError(f"frame displacement limit requires a 1-DoF joint: {name}")
            dofs.append(int(model.jnt_dofadr[joint]))
            qpos_addresses.append(int(model.jnt_qposadr[joint]))
            speeds.append(float(velocity_map[name]))
        self.model = model
        self.constraint_type = constraint_type
        self.dofs = np.asarray(dofs, dtype=np.int64)
        self.qpos_addresses = np.asarray(qpos_addresses, dtype=np.int64)
        self.speeds = np.asarray(speeds, dtype=np.float64)
        self.lower: np.ndarray | None = None
        self.upper: np.ndarray | None = None

    def set_previous(self, qpos: np.ndarray | None, dt: float | None = None) -> None:
        if qpos is None:
            self.lower = self.upper = None
            return
        if dt is None or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("frame displacement limit requires a positive frame dt")
        center = np.asarray(qpos, dtype=np.float64)[self.qpos_addresses]
        radius = self.speeds * float(dt)
        # QP feasibility tolerances can otherwise put the integrated endpoint
        # a few 1e-7 beyond an exact bound.  Reserve at most one micrometre/
        # microradian per source interval (and at most 1% of tiny intervals),
        # so the independently recomputed physical velocity remains legal.
        radius -= np.minimum(1.0e-6, 0.01 * radius)
        self.lower = center - radius
        self.upper = center + radius

    def compute_qp_inequalities(self, configuration: Any, _dt: float) -> Any:
        if self.lower is None or self.upper is None or not len(self.dofs):
            return self.constraint_type()
        projection = np.eye(self.model.nv, dtype=np.float64)[self.dofs]
        current = np.asarray(configuration.q, dtype=np.float64)[self.qpos_addresses]
        matrix = np.vstack([projection, -projection])
        bounds = np.concatenate([self.upper - current, current - self.lower])
        return self.constraint_type(matrix, bounds)


class _StrictCollisionLimit:
    """Turn MINK's one-sided collision brake into a feasibility constraint.

    Upstream MINK stops *additional* motion when a pair already penetrates, but
    it does not require the pair to leave penetration.  That behavior is fine
    for a collision-free seed and wrong for per-frame object GT updates.  This
    adapter keeps MINK's Jacobian and pair filtering, removes its ``/dt``
    relaxation, and requires a bounded separating displacement while invalid.
    """

    def __init__(
        self, inner: Any, mujoco: Any, *, minimum_distance: float,
        depenetration_step: float, gain: float = 0.85,
    ) -> None:
        self.inner = inner
        self.mujoco = mujoco
        self.minimum_distance = float(minimum_distance)
        self.depenetration_step = float(depenetration_step)
        self.gain = float(gain)
        self.geom_id_pairs = inner.geom_id_pairs
        # Dynamic contact-timing limits disable this collision family on GT
        # contact frames.  Static self/object/floor limits remain active.
        self.active = True
        self.enabled = False

    def compute_qp_inequalities(self, configuration: Any, dt: float) -> Any:
        constraint = self.inner.compute_qp_inequalities(configuration, dt)
        if not self.active:
            return type(constraint)()
        if not self.enabled:
            return constraint
        if constraint.inactive or constraint.G is None or constraint.h is None:
            return constraint
        h = np.asarray(constraint.h, dtype=np.float64).copy()
        fromto = np.empty(6, dtype=np.float64)
        detection = float(self.inner.collision_detection_distance)
        invalid: list[tuple[float, int]] = []
        for index, (geom_a, geom_b) in enumerate(self.geom_id_pairs):
            distance = float(self.mujoco.mj_geomDistance(
                configuration.model, configuration.data,
                geom_a, geom_b, detection, fromto,
            ))
            if abs(distance - detection) < 1e-12:
                continue
            residual = distance - self.minimum_distance
            if residual >= 0.0:
                h[index] = self.gain * residual
            else:
                # Keep all currently invalid pairs approximately stationary,
                # then actively resolve only the deepest one.  Demanding a
                # finite separating step from many opposing contact normals in
                # the same linearized QP is commonly infeasible.
                h[index] = 1e-5
                invalid.append((residual, index))
        if invalid:
            residual, index = min(invalid)
            h[index] = -min(
                self.depenetration_step,
                self.gain * (-residual),
            )
        return type(constraint)(constraint.G, h)


def _wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return xyzw[[3, 0, 1, 2]]


def _pose_to_se3(mink: Any, transform: np.ndarray) -> Any:
    return mink.SE3(wxyz_xyz=np.concatenate([_wxyz(transform[:3, :3]), transform[:3, 3]]))


def _joint_velocity_limits(model: Any, mujoco: Any, settings: dict[str, float]) -> dict[str, float]:
    try:
        category_limits = {
            key: float(settings[key])
            for key in ("base_translation", "base_rotation", "finger")
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "velocity_limits must define numeric base_translation, base_rotation, and finger values"
        ) from error
    if any(not np.isfinite(value) or value <= 0.0 for value in category_limits.values()):
        raise ValueError("all joint velocity limits must be finite and positive")
    limits: dict[str, float] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None or "object" in name or int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        if any(token in name for token in ("_tx_", "_ty_", "_tz_")):
            limits[name] = category_limits["base_translation"]
        elif any(token in name for token in ("roll", "pitch", "yaw")):
            limits[name] = category_limits["base_rotation"]
        else:
            limits[name] = category_limits["finger"]
    return limits


def _enable_planning_collision_masks(
    model: Any, hand_geom_ids: list[int], object_geom_ids: list[int],
) -> None:
    """Make specified geoms visible to MINK's collision-pair enumerator.

    SPIDER's scene XML uses explicit MuJoCo ``<pair>`` elements for physical
    hand--object contact and consequently leaves the default geom collision
    masks at ``0/0``.  MuJoCo honors those explicit pairs at execution, but
    MINK's *planning* collision limit deliberately filters on the default
    masks, producing an empty constraint set.  This modifies only the private
    MjModel used while solving IK; it neither changes XML nor rollout physics.
    """
    for geom_id in set(hand_geom_ids) | set(object_geom_ids):
        model.geom_contype[geom_id] |= 1
        model.geom_conaffinity[geom_id] |= 1


def _explicit_collision_groups(
    model: Any, mujoco: Any, *, hand_geom_ids: set[int],
    object_geom_ids: set[int] | None = None,
) -> list[tuple[list[str], list[str]]]:
    """Return one MINK group per collision pair present in the runtime XML.

    A broad ``hand x hand`` Cartesian product is not equivalent to SPIDER's
    explicit runtime contact model: it includes deliberately overlapping
    adjacent link shells and fixed bases.  Reusing ``model.pair_geom*`` keeps
    planning and execution collision semantics identical.
    """
    groups: list[tuple[list[str], list[str]]] = []
    object_ids = None if object_geom_ids is None else set(object_geom_ids)
    for pair_id in range(int(model.npair)):
        geom_a = int(model.pair_geom1[pair_id])
        geom_b = int(model.pair_geom2[pair_id])
        if object_ids is None:
            selected = geom_a in hand_geom_ids and geom_b in hand_geom_ids
        else:
            selected = (
                (geom_a in hand_geom_ids and geom_b in object_ids)
                or (geom_b in hand_geom_ids and geom_a in object_ids)
            )
        if not selected:
            continue
        name_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
        name_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)
        if name_a is None or name_b is None:
            raise ValueError("explicit collision pair contains an unnamed geom")
        groups.append(([name_a], [name_b]))
    return groups


def _shrink_finger_joint_ranges(
    model: Any, mujoco: Any, margin_rad: float,
) -> np.ndarray:
    """Keep IK solutions away from hard XHand finger limits.

    The returned ranges are the immutable physical ranges used by diagnostics.
    Only the private MINK planning model is narrowed.  Runtime MuJoCo models
    and their physical joint limits are never modified.
    """
    margin = float(margin_rad)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("finger joint-limit margin must be finite and non-negative")
    physical_ranges = np.asarray(model.jnt_range, dtype=np.float64).copy()
    if margin == 0.0:
        return physical_ranges
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        if not bool(model.jnt_limited[joint]) or "_hand_" not in name:
            continue
        lower, upper = physical_ranges[joint]
        if upper - lower <= 2.0 * margin:
            raise ValueError(
                f"finger joint {name} is narrower than twice the requested limit margin"
            )
        model.jnt_range[joint] = [lower + margin, upper - margin]
    return physical_ranges


def _project_configuration_into_joint_ranges(
    configuration: Any, model: Any, mujoco: Any,
) -> None:
    """Make the initial MINK state legal after planning ranges are narrowed."""
    q = np.asarray(configuration.q, dtype=np.float64).copy()
    changed = False
    for joint in range(model.njnt):
        if (
            not bool(model.jnt_limited[joint])
            or int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ):
            continue
        address = int(model.jnt_qposadr[joint])
        lower, upper = np.asarray(model.jnt_range[joint], dtype=np.float64)
        projected = float(np.clip(q[address], lower, upper))
        if projected != float(q[address]):
            q[address] = projected
            changed = True
    if changed:
        configuration.update(q=q)


def _set_object_pose(model: Any, data: Any, mujoco: Any, side: str, transform: np.ndarray) -> None:
    """Place the *kinematic planning geometry* at the GT object pose.

    This is deliberately restricted to a free joint.  Retargeting may use the
    GT pose to query hand--object geometry, but an act-scene's six object
    joints are a direct object-control mechanism and must never be supported.
    The generated reference is subsequently replayed in a fresh free-object
    simulation where no such pose writes occur.
    """
    name = f"{side}_object_joint"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(
            "retargeting requires a free-object scene.xml; act-scene object "
            "controls are forbidden"
        )
    address = int(model.jnt_qposadr[joint_id])
    data.qpos[address : address + 3] = transform[:3, 3]
    data.qpos[address + 3 : address + 7] = _wxyz(transform[:3, :3])


def _load_object_hulls(model_path: Path) -> list[Any]:
    """Load the object convex hull meshes with outward-oriented face normals."""
    import re

    import trimesh

    text = Path(model_path).read_text(encoding="utf-8")
    meshdir = Path(model_path).parent
    compiler = re.search(r'<compiler[^>]*meshdir="([^"]+)"', text)
    if compiler:
        meshdir = (Path(model_path).parent / compiler.group(1)).resolve()
    hulls: list[Any] = []
    for match in re.finditer(r'<mesh\s+name="([^"]+)"\s+file="([^"]+)"', text):
        name, relative = match.groups()
        if "/convex/" not in relative:
            continue
        path = (meshdir / relative).resolve()
        mesh = trimesh.load(str(path), force="mesh", process=True)
        normals = np.asarray(mesh.face_normals, dtype=np.float64).copy()
        if mesh.is_watertight:
            probes = mesh.triangles_center + 1e-4 * normals
            inside = mesh.contains(probes)
            normals[inside] = -normals[inside]
        hulls.append((mesh, normals))
    if not hulls:
        raise RuntimeError("no convex hull meshes found for surface projection")
    return hulls


def _finger_capsule_geoms(
    model: Any, data: Any, mujoco: Any, side: str, finger_names: tuple[str, ...],
) -> list[list[tuple[np.ndarray, np.ndarray, float, float]]]:
    """Per finger: list of ALL collision capsules as
    (center offset in tip-site frame, axis dir in tip-site frame, half-length, radius)."""
    result: list[list[tuple[np.ndarray, np.ndarray, float, float]]] = []
    for finger in finger_names:
        base = finger.removesuffix("_tip") if finger.endswith("_tip") else finger
        ids = [
            geom for geom in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
            .startswith(f"collision_hand_{side}_{base}")
        ]
        if not ids:
            result.append([(np.zeros(3), np.array([0.0, 0.0, -1.0]), 0.005, 0.005)])
            continue
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}")
        mujoco.mj_forward(model, data)
        site_position = data.site_xpos[site_id].copy()
        site_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
        capsules = []
        for geom in ids:
            geom_position = data.geom_xpos[geom].copy()
            geom_rotation = data.geom_xmat[geom].reshape(3, 3).copy()
            offset = site_rotation.T @ (geom_position - site_position)
            axis_local = site_rotation.T @ (geom_rotation @ np.array([0.0, 0.0, 1.0]))
            radius = float(model.geom_size[geom][0])
            half_length = float(model.geom_size[geom][1])
            capsules.append((offset, axis_local, half_length, radius))
        result.append(capsules)
    return result


def _capsule_hull_contact(
    site_pose: np.ndarray,
    capsules: list[tuple[np.ndarray, np.ndarray, float, float]],
    hulls: list[Any],
    transform_object: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    """Signed distance from the finger capsule surface to the object hulls,
    measured at the given tip-site pose.  Returns (deepest, outward normal in
    world frame) or None when no hull data is available.  deepest < 0 means the
    capsule penetrates the object surface."""
    import trimesh

    rotation, translation = transform_object[:3, :3], transform_object[:3, 3]
    site_rotation, site = site_pose[:3, :3], site_pose[:3, 3]
    deepest = np.inf
    normal = np.zeros(3)
    deepest_sample: np.ndarray | None = None
    deepest_hull: Any | None = None
    for offset, axis_local, half_length, radius in capsules:
        center = site + site_rotation @ offset
        axis = site_rotation @ axis_local
        samples = np.stack(
            [center + step * half_length * axis for step in np.linspace(-1.0, 1.0, 7)],
            axis=0,
        )
        local = (rotation.T @ (samples - translation).T).T
        for hull, face_normals in hulls:
            surface, unsigned, tri_idx = trimesh.proximity.closest_point(hull, local)
            inside = np.zeros(len(samples), dtype=bool)
            try:
                inside = hull.contains(local)
            except Exception:
                pass
            inside &= unsigned > 1e-3
            signed = np.where(inside, -unsigned, unsigned)
            best = int(np.argmin(signed))
            if signed[best] - radius < deepest:
                deepest = float(signed[best] - radius)
                normal = np.asarray(face_normals[tri_idx[best]], dtype=np.float64)
                # The orientation check must probe the exact capsule sample
                # and hull that produced ``deepest``.  Reusing the final loop
                # variables silently paired the deepest normal with a
                # different capsule/hull for multi-link fingers.
                deepest_sample = samples[best].copy()
                deepest_hull = hull
    if deepest >= np.inf or deepest_sample is None or deepest_hull is None:
        return None
    world_normal = rotation @ normal
    # Verify the normal orientation numerically: moving the deepest sample a
    # little along the normal must increase the capsule-surface distance
    # (i.e. the normal really points OUT of the object).  Face normals of
    # non-watertight hull pieces are otherwise unreliable.
    local = rotation.T @ (deepest_sample - translation)
    before = float(trimesh.proximity.closest_point(deepest_hull, [local])[1][0])
    after = float(trimesh.proximity.closest_point(
        deepest_hull, [local + rotation.T @ (world_normal * 2e-3)],
    )[1][0])
    if after < before:
        world_normal = -world_normal
    return deepest, world_normal


def _contact_correction_clearance(
    *, deepest_m: float, gt_contact: bool, contact_aware: bool,
    object_clearance_correction: bool, contact_reach_m: float,
    contact_attraction_requires_reach: bool, contact_clearance_m: float,
    object_clearance_m: float,
) -> float | None:
    """Choose a surface target without inventing an unobserved contact.

    Contact attraction and collision clearance have opposite meanings.  A
    demonstrated contact may be pulled toward ``contact_clearance_m``.  A
    non-contacting digit may only be pushed *out* when it violates the object
    clearance; merely being near the object is not permission to attract it.
    """
    deepest = float(deepest_m)
    if not np.isfinite(deepest):
        raise ValueError("finger-object distance must be finite")
    if contact_aware and gt_contact:
        if contact_attraction_requires_reach and deepest > contact_reach_m:
            return None
        return float(contact_clearance_m)
    if object_clearance_correction and deepest < object_clearance_m:
        return float(object_clearance_m)
    return None


def _opposition_targets(
    thumb_target_pos: np.ndarray,
    fingertip_targets: np.ndarray,
    hulls: list[Any],
    transform_object: np.ndarray,
    spread_m: float = 0.012,
    beyond_m: float = 0.002,
) -> np.ndarray | None:
    """Synthesize antipodal opposition targets for the four fingers.

    Finds the handle piece nearest the thumb target, computes its principal
    (long) axis and the outward normal at the thumb contact, then places the
    four fingertip targets on the FAR side of the handle, spread along the
    handle axis so they oppose the thumb.  Returns the new [5,4,4] targets."""
    import trimesh

    rotation, translation = transform_object[:3, :3], transform_object[:3, 3]
    thumb_local = rotation.T @ (thumb_target_pos - translation)
    best_piece = None
    best_distance = np.inf
    best_face = -1
    for piece, normals in hulls:
        surface, unsigned, tri_idx = trimesh.proximity.closest_point(piece, [thumb_local])
        if unsigned[0] < best_distance:
            best_distance = float(unsigned[0])
            best_piece = piece
            best_face = int(tri_idx[0])
    if best_piece is None or best_distance > 0.03:
        return None
    vertices = np.asarray(best_piece.vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    covariance = np.cov((vertices - center).T)
    _, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]  # long axis (object frame)
    normal_local = np.asarray(best_piece.face_normals[best_face], dtype=np.float64)
    # verify outward orientation
    probe = thumb_local
    before = float(trimesh.proximity.closest_point(best_piece, [probe])[1][0])
    after = float(trimesh.proximity.closest_point(best_piece, [probe + normal_local * 2e-3])[1][0])
    if after < before:
        normal_local = -normal_local
    offsets = (vertices - center) @ normal_local
    half_thickness = float(offsets.max() - offsets.min()) / 2.0 + 1e-3
    targets = fingertip_targets.copy()
    for finger in range(1, 5):
        spread = (finger - 2.5) * spread_m
        local_pos = center + axis * spread - normal_local * (half_thickness + 0.015 + beyond_m)
        targets[finger, :3, 3] = rotation @ local_pos + translation
        targets[finger, :3, :3] = fingertip_targets[finger, :3, :3]
    return targets


def _wrap_shift_for_finger(
    site_pose: np.ndarray,
    capsules: list[tuple[np.ndarray, np.ndarray, float, float]],
    hulls: list[Any],
    transform_object: np.ndarray,
    beyond_m: float = 0.002,
) -> float | None:
    """Wrap synthesis for the four fingers: move the fingertip target so the
    finger capsule passes THROUGH the handle and its inner surface sits
    `beyond_m` past the far side (the handle ends up between the fingers and
    the thumb).  Returns the target shift along the outward normal (negative =
    deeper through the object) or None when no crossing geometry is found."""
    import trimesh

    rotation, translation = transform_object[:3, :3], transform_object[:3, 3]
    site_rotation, site = site_pose[:3, :3], site_pose[:3, 3]
    best_sample = None
    best_signed = np.inf
    best_normal = None
    best_hull = None
    for offset, axis_local, half_length, radius in capsules:
        center = site + site_rotation @ offset
        axis = site_rotation @ axis_local
        samples = np.stack(
            [center + step * half_length * axis for step in np.linspace(-1.0, 1.0, 7)],
            axis=0,
        )
        local = (rotation.T @ (samples - translation).T).T
        for hull, face_normals in hulls:
            surface, unsigned, tri_idx = trimesh.proximity.closest_point(hull, local)
            inside = np.zeros(len(samples), dtype=bool)
            try:
                inside = hull.contains(local)
            except Exception:
                pass
            inside &= unsigned > 1e-3
            signed = np.where(inside, -unsigned, unsigned)
            best = int(np.argmin(signed))
            if signed[best] < best_signed:
                best_signed = float(signed[best])
                best_sample = samples[best]
                best_normal = np.asarray(face_normals[tri_idx[best]], dtype=np.float64)
                best_hull = hull
    if best_sample is None or best_hull is None:
        return None
    world_normal = rotation @ best_normal
    # verify orientation
    probe = best_sample
    local_probe = rotation.T @ (probe - translation)
    before = float(trimesh.proximity.closest_point(best_hull, [local_probe])[1][0])
    after = float(trimesh.proximity.closest_point(
        best_hull, [local_probe + rotation.T @ (world_normal * 2e-3)])[1][0])
    if after < before:
        world_normal = -world_normal
    # capsule surface point facing the object, then ray-cast through the hull
    radius = max(float(c[3]) for c in capsules)
    surface_point = best_sample - world_normal * radius
    local_origin = rotation.T @ (surface_point - translation)
    local_direction = rotation.T @ (-world_normal)
    hits, _, _ = best_hull.ray.intersects_location(
        [local_origin], [local_direction],
    )
    if len(hits) == 0:
        return None
    depths = (hits - local_origin) @ local_direction
    exit_distance = float(np.max(depths))
    if exit_distance <= 0:
        return None
    return float(-(exit_distance + beyond_m))


def _box_smooth(values: np.ndarray, half: int) -> np.ndarray:
    """Causal-free box filter over the leading axis (edges clamped)."""
    if half <= 0 or len(values) <= 1:
        return values
    smoothed = values.copy()
    count = len(values)
    for index in range(count):
        low, high = max(0, index - half), min(count, index + half + 1)
        smoothed[index] = values[low:high].mean(axis=0)
    return smoothed


def _project_fingertip_targets(
    hulls: list[Any], human: dict[str, np.ndarray], frame: int,
    hand_count: int, margin: float,
    capsule_geoms: list[list[list[tuple[np.ndarray, np.ndarray, float, float]]]],
    reach: float = 0.008,
) -> tuple[np.ndarray, np.ndarray]:
    """Push fingertip targets relative to the object hulls along the surface normal.

    Generic surface/grasp correction driven by the reference fingertip collision
    capsules (axis segments + radius) against the object convex hulls.  Whenever
    a capsule touches or intersects a hull (distance < radius), the fingertip
    target is shifted along the outward normal to a target clearance of
    (radius + margin):

    - margin > 0: anti-penetration projection (capsule tangent + margin off);
    - margin < 0: grasp squeeze -- the target penetrates the surface by
      |margin|, so the soft position controller presses into the object and
      builds opposition grip force.

    No per-episode parameters.

    Returns (corrected targets [H, 5, 4, 4], shifts [H, 5, 3]).
    """
    import trimesh

    transform_object = human["T_sim_object_reference"][frame, 0]
    rotation, translation = transform_object[:3, :3], transform_object[:3, 3]
    corrected = human["T_sim_fingertip_target"][frame].copy()
    shifts = np.zeros((hand_count, 5, 3), dtype=np.float64)
    for hand in range(hand_count):
        for finger in range(5):
            target_rotation = corrected[hand, finger, :3, :3]
            site = corrected[hand, finger, :3, 3]
            deepest = np.inf
            normal = np.zeros(3)
            touching_radius = np.inf
            for offset, axis_local, half_length, radius in capsule_geoms[hand][finger]:
                center = site + target_rotation @ offset
                axis = target_rotation @ axis_local
                samples = np.stack(
                    [center + t * half_length * axis for t in np.linspace(-1.0, 1.0, 7)],
                    axis=0,
                )
                local = (rotation.T @ (samples - translation).T).T
                for hull, face_normals in hulls:
                    surface, unsigned, tri_idx = trimesh.proximity.closest_point(hull, local)
                    inside = np.zeros(len(samples), dtype=bool)
                    try:
                        inside = hull.contains(local)
                    except Exception:
                        pass
                    inside &= unsigned > 1e-3
                    signed = np.where(inside, -unsigned, unsigned)
                    best = int(np.argmin(signed))
                    # capsule surface distance = axis distance - radius
                    if signed[best] - radius < deepest:
                        deepest = float(signed[best] - radius)
                        normal = np.asarray(face_normals[tri_idx[best]], dtype=np.float64)
                        touching_radius = radius
            # Activate when the capsule is at/near the surface:
            #  - margin >= 0 (anti-penetration): shift whenever penetrating
            #    deeper than the clearance;
            #  - margin <  0 (grasp squeeze): shift whenever the capsule surface
            #    is within `reach` of the hull, pressing it to |margin| depth so
            #    the soft position controller builds opposition grip force.
            # shift_amount is along the outward normal (negative = inward).
            if margin >= 0.0:
                trigger = deepest < margin
            else:
                trigger = deepest < reach
            if getattr(_project_fingertip_targets, "_debug", False):
                print(f"[proj-debug] frame={frame} hand={hand} finger={finger} "
                      f"deepest={deepest:.4f} trigger={trigger} "
                      f"touching_radius={touching_radius}", flush=True)
            if trigger and touching_radius < np.inf:
                shift_amount = -deepest + margin
                shift_world = rotation @ (normal * shift_amount)
                shifts[hand, finger] = shift_world
                corrected[hand, finger, :3, 3] += shift_world
    return corrected, shifts


def _joint_limit_diagnostic(
    model: Any, data: Any, mujoco: Any, *, physical_ranges: np.ndarray | None = None,
) -> tuple[float, bool]:
    margins = []
    violation = False
    ranges = model.jnt_range if physical_ranges is None else np.asarray(physical_ranges)
    if ranges.shape != model.jnt_range.shape:
        raise ValueError("physical joint ranges do not match the planning model")
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id] or int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        address = int(model.jnt_qposadr[joint_id])
        value = float(data.qpos[address])
        lower, upper = ranges[joint_id]
        margins.append(min(value - lower, upper - value))
        violation |= value < lower - 1e-6 or value > upper + 1e-6
    return (float(min(margins)) if margins else 1e6), bool(violation)


def _collision_diagnostic(
    model: Any, data: Any, mujoco: Any, pairs: list[tuple[int, int]],
    minimum: float, *, tolerance: float = 1e-6,
) -> tuple[float, bool]:
    distances = []
    fromto = np.empty(6, dtype=np.float64)
    for geom_a, geom_b in pairs:
        distances.append(float(mujoco.mj_geomDistance(model, data, geom_a, geom_b, 1.0, fromto)))
    value = min(distances) if distances else 1e6
    return value, bool(value < minimum - tolerance)


def retarget_with_mink(
    human_reference_path: str | Path, model_path: str | Path, output_path: str | Path,
    config: ReproConfig, *, solver: str = "daqp",
    contact_reference_path: str | Path | None = None,
) -> Path:
    """Solve Eq. 1 frame-by-frame and write the unified robot reference artifact."""
    try:
        import mink
        import mujoco
    except ImportError as error:
        raise RuntimeError("enhanced retargeting must run in the SPIDER environment with mink and mujoco") from error

    with np.load(human_reference_path, allow_pickle=False) as artifact:
        human = {key: np.asarray(artifact[key]) for key in artifact.files}
    validate_human_reference(human)
    settings = config.data["retarget"]
    smoothing_window = int(settings.get("smoothing_window_frames", 0))
    if smoothing_window > 0:
        half = smoothing_window // 2
        human["T_sim_fingertip_target"][..., :3, 3] = _box_smooth(
            human["T_sim_fingertip_target"][..., :3, 3], half,
        )
        human["T_sim_wrist_target"][..., :3, 3] = _box_smooth(
            human["T_sim_wrist_target"][..., :3, 3], half,
        )
        if "T_sim_object_reference" in human:
            human["T_sim_object_reference"][..., :3, 3] = _box_smooth(
                human["T_sim_object_reference"][..., :3, 3], half,
            )
        matrix_keys = ["T_sim_fingertip_target", "T_sim_wrist_target"]
        if "T_sim_object_reference" in human:
            matrix_keys.append("T_sim_object_reference")
        for matrix_key in matrix_keys:
            matrices = human[matrix_key][..., :3, :3]
            smoothed = _box_smooth(matrices, half)
            for index in np.ndindex(smoothed.shape[:-2]):
                u, _, vh = np.linalg.svd(smoothed[index])
                projected = u @ vh
                if np.linalg.det(projected) < 0.0:
                    u[:, -1] *= -1.0
                    projected = u @ vh
                smoothed[index] = projected
            human[matrix_key][..., :3, :3] = smoothed
    contact_aware = bool(settings.get("contact_aware_retarget", False))
    contact_timing_weighting = bool(settings.get("contact_timing_weighting", False))
    contact_timing_cost_multiplier = float(
        settings.get("contact_timing_cost_multiplier", 1.0)
    )
    if (
        not np.isfinite(contact_timing_cost_multiplier)
        or contact_timing_cost_multiplier <= 0.0
    ):
        raise ValueError("contact_timing_cost_multiplier must be finite and positive")
    contact_clearance = float(settings.get("contact_clearance_m", 0.002))
    contact_clearance_thumb = float(
        settings.get("contact_clearance_thumb_m", contact_clearance)
    )
    precontact_clearance = float(
        settings.get("precontact_finger_clearance_m", 0.0)
    )
    contact_reach = float(settings.get("contact_reach_m", 0.015))
    contact_rounds = int(settings.get("contact_correction_rounds", 3))
    contact_attraction_max_shift = float(
        settings.get("contact_attraction_max_shift_m", 0.03)
    )
    object_depenetration_max_shift = float(
        settings.get("object_depenetration_max_shift_m", 0.03)
    )
    contact_attraction_requires_reach = bool(
        settings.get("contact_attraction_requires_reach", False)
    )
    contact_aware_requires_gt_mask = bool(
        settings.get("contact_aware_requires_gt_mask", False)
    )
    if (
        not np.isfinite(contact_clearance)
        or contact_clearance < 0.0
        or not np.isfinite(contact_clearance_thumb)
        or contact_clearance_thumb < 0.0
        or not np.isfinite(contact_attraction_max_shift)
        or contact_attraction_max_shift < 0.0
        or not np.isfinite(object_depenetration_max_shift)
        or object_depenetration_max_shift <= 0.0
        or contact_rounds < 1
        or not np.isfinite(contact_reach)
        or contact_reach < 0.0
        or not np.isfinite(precontact_clearance)
        or precontact_clearance < 0.0
    ):
        raise ValueError("contact attraction limits/reach/rounds are invalid")
    contact_wrap_synthesis = bool(settings.get("contact_wrap_synthesis", False))
    object_clearance_correction = bool(
        settings.get("object_clearance_correction", False)
    )
    object_collision_avoidance = bool(settings.get("object_collision_avoidance", False))
    object_collision_clearance = float(settings.get("object_collision_clearance_m", 0.002))
    object_collision_detection = float(settings.get("object_collision_detection_m", 0.03))
    floor_collision_avoidance = bool(settings.get("floor_collision_avoidance", False))
    floor_collision_clearance = float(settings.get("floor_collision_clearance_m", 0.0))
    floor_collision_detection = float(settings.get("floor_collision_detection_m", 0.05))
    if (
        not np.isfinite(object_collision_clearance)
        or object_collision_clearance < 0.0
        or not np.isfinite(object_collision_detection)
        or object_collision_detection <= object_collision_clearance
    ):
        raise ValueError("object collision clearance/detection settings are invalid")
    if (
        not np.isfinite(floor_collision_clearance)
        or floor_collision_clearance < 0.0
        or not np.isfinite(floor_collision_detection)
        or floor_collision_detection <= floor_collision_clearance
    ):
        raise ValueError("floor collision clearance/detection settings are invalid")
    strict_collision_resolution = bool(
        settings.get("strict_collision_resolution", False)
    )
    strict_collision_iterations = int(
        settings.get("strict_collision_resolution_iterations", 128)
    )
    depenetration_step = float(
        settings.get("collision_depenetration_step_m", 0.002)
    )
    collision_validation_tolerance = float(
        settings.get("collision_validation_tolerance_m", 1e-6)
    )
    if (
        not np.isfinite(depenetration_step)
        or depenetration_step <= 0.0
        or not np.isfinite(collision_validation_tolerance)
        or collision_validation_tolerance < 0.0
        or strict_collision_iterations < 1
    ):
        raise ValueError("collision depenetration step must be finite and positive")
    contact_mask: np.ndarray | None = None
    contact_signature = ""
    contact_position_source = ""
    episode_id = ""
    object_side = str(human.get("object_side", np.asarray("right")).item())
    if (contact_aware or contact_timing_weighting) and contact_reference_path is not None:
        with np.load(contact_reference_path, allow_pickle=False) as artifact:
            contact_mask = np.asarray(artifact["contact"], dtype=bool)
            finger_order = tuple(str(value) for value in np.asarray(
                artifact["finger_order"],
            ).tolist()) if "finger_order" in artifact.files else ()
            contact_positions = np.asarray(
                artifact["contact_pos_ref"], dtype=np.float64,
            ) if "contact_pos_ref" in artifact.files else np.empty((0, 0, 0))
            contact_frames = np.asarray(
                artifact["frame_indices"], dtype=np.int64,
            ) if "frame_indices" in artifact.files else np.empty(0, dtype=np.int64)
            contact_times = np.asarray(
                artifact["timestamps_s"], dtype=np.float64,
            ) if "timestamps_s" in artifact.files else np.empty(0)
            contact_side = str(np.asarray(
                artifact["side"],
            ).item()) if "side" in artifact.files else ""
            episode_id = str(np.asarray(
                artifact["episode_id"],
            ).item()) if "episode_id" in artifact.files else ""
            contact_distance = np.asarray(
                artifact["contact_distance_m"], dtype=np.float64,
            ) if "contact_distance_m" in artifact.files else np.empty((0, 0))
            contact_distance_valid = np.asarray(
                artifact["contact_distance_valid"], dtype=bool,
            ) if "contact_distance_valid" in artifact.files else np.empty((0, 0), dtype=bool)
            contact_threshold = (
                float(np.asarray(artifact["contact_threshold_m"]).item())
                if "contact_threshold_m" in artifact.files else float("nan")
            )
            contact_definition = str(np.asarray(
                artifact["contact_definition"],
            ).item()) if "contact_definition" in artifact.files else ""
            contact_state = np.asarray(
                artifact["contact_state"], dtype=np.int8,
            ) if "contact_state" in artifact.files else None
            contact_score = np.asarray(
                artifact["contact_score"], dtype=np.float64,
            ) if "contact_score" in artifact.files else None
            contact_patch = np.asarray(
                artifact["contact_patch_pos_obj_m"], dtype=np.float64,
            ) if "contact_patch_pos_obj_m" in artifact.files else None
            contact_normal = np.asarray(
                artifact["contact_patch_normal_obj"], dtype=np.float64,
            ) if "contact_patch_normal_obj" in artifact.files else None
            contact_schema = str(np.asarray(
                artifact["contact_schema"],
            ).item()) if "contact_schema" in artifact.files else "xhand_gt_contact_reference_v4"
            contact_position_source = str(np.asarray(
                artifact["contact_position_source"],
            ).item()) if "contact_position_source" in artifact.files else ""
            stored_contact_signature = str(np.asarray(
                artifact["contact_reference_signature"],
            ).item()) if "contact_reference_signature" in artifact.files else ""
            stored_mano_signature = str(np.asarray(
                artifact["human_mano_signature"],
            ).item()) if "human_mano_signature" in artifact.files else ""
        human_order = tuple(str(value) for value in human["hand_order"].tolist())
        if (
            finger_order != XHAND_FINGER_ORDER
            or human_order != (contact_side,)
            or contact_side != object_side
            or not episode_id
            or not np.array_equal(contact_frames, human["frame_indices"])
            or not np.array_equal(contact_times, human["timestamps_s"])
            or "mano_joint_positions_sim" not in human
        ):
            raise ValueError(
                "contact reference sample/side/timeline does not exactly match the "
                "single-hand human reference"
            )
        expected_mano_signature = human_mano_signature(
            human["frame_indices"], human["timestamps_s"], contact_side,
            human["mano_joint_positions_sim"][:, 0],
        )
        contact_signature = contact_reference_signature(
            contact_frames, contact_times, contact_side, contact_mask,
            contact_positions,
            finger_order=finger_order,
            episode_id=episode_id,
            contact_distance_m=contact_distance,
            contact_distance_valid=contact_distance_valid,
            contact_threshold_m=contact_threshold,
            contact_definition=contact_definition,
            contact_position_source=contact_position_source,
            human_mano_signature_value=stored_mano_signature,
            contact_state=contact_state,
            contact_score=contact_score,
            contact_patch_pos_obj_m=contact_patch,
            contact_patch_normal_obj=contact_normal,
            contact_schema=contact_schema,
        )
        if (
            stored_contact_signature != contact_signature
            or stored_mano_signature != expected_mano_signature
            or contact_position_source != HUMAN_CONTACT_POSITION_SOURCE
        ):
            raise ValueError(
                "contact reference identity is stale or does not come from the "
                "current upstream human reference"
            )
    if contact_aware and contact_aware_requires_gt_mask and contact_mask is None:
        raise ValueError("contact-aware nominal retargeting requires an explicit GT contact mask")
    if precontact_clearance > 0.0 and contact_mask is None:
        raise ValueError("pre-contact finger clearance requires an explicit contact mask")
    if precontact_clearance > 0.0 and not strict_collision_resolution:
        raise ValueError("pre-contact finger clearance requires strict collision resolution")
    if contact_aware and "T_sim_object_reference" not in human:
        raise ValueError("contact-aware retargeting requires an aligned object reference")
    model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
    # Capture the unmodified runtime scene identity before private planning
    # masks and narrowed IK joint ranges mutate this MjModel.
    runtime_model_signature = mujoco_model_signature(model, mujoco)
    physical_joint_ranges = _shrink_finger_joint_ranges(
        model, mujoco, float(settings.get("finger_joint_limit_margin_rad", 0.0)),
    )
    configuration = mink.Configuration(model)
    _project_configuration_into_joint_ranges(configuration, model, mujoco)
    hand_order = human["hand_order"].tolist()
    surface_projection = bool(settings.get("surface_projection", False))
    object_hulls: list[Any] | None = None
    capsule_geoms: dict[str, list[list[tuple[np.ndarray, np.ndarray, float, float]]]] = {}
    projection_margin = float(settings.get("surface_projection_margin", 0.002))
    projection_reach = float(settings.get("surface_projection_reach", 0.008))
    if (
        not np.isfinite(projection_margin)
        or projection_margin < 0.0
        or not np.isfinite(projection_reach)
        or projection_reach <= 0.0
    ):
        raise ValueError("surface projection margin/reach settings are invalid")
    if surface_projection or contact_aware:
        object_hulls = _load_object_hulls(Path(model_path).resolve())
        for side in hand_order:
            capsule_geoms[side] = _finger_capsule_geoms(
                model, configuration.data, mujoco, side, FINGER_NAMES,
            )
    wrist_tasks, finger_tasks = {}, {}
    for hand, side in enumerate(hand_order):
        wrist_tasks[hand] = mink.FrameTask(
            f"{side}_palm", "site", settings["wrist_position_cost"],
            settings["wrist_orientation_cost"], lm_damping=1.0,
        )
        for finger, name in enumerate(FINGER_NAMES):
            finger_tasks[hand, finger] = mink.FrameTask(
                f"{side}_{name}", "site", settings["fingertip_position_cost"],
                settings["fingertip_orientation_cost"], lm_damping=1.0,
            )
    posture = mink.PostureTask(model, cost=float(settings["posture_cost"]))
    posture.set_target(configuration.q.copy())
    tasks = [posture, *wrist_tasks.values(), *finger_tasks.values()]
    limits: list[Any] = []
    if settings.get("joint_position_limits", True):
        limits.append(mink.ConfigurationLimit(model))
    velocity_map = _joint_velocity_limits(model, mujoco, settings["velocity_limits"])
    velocity_limit = None
    frame_displacement_limit = None
    if settings.get("joint_velocity_limits", True):
        velocity_limit = mink.VelocityLimit(model, velocity_map)
        limits.append(velocity_limit)
        if settings.get("frame_displacement_limits", False):
            frame_displacement_limit = _FrameDisplacementLimit(
                model, mujoco, velocity_map, mink.limits.Constraint,
            )
            limits.append(frame_displacement_limit)
    collision_limit = None
    object_collision_limit = None
    precontact_limits: dict[int, _StrictCollisionLimit] = {}
    floor_collision_limit = None
    strict_collision_limits: list[_StrictCollisionLimit] = []
    object_collision_pairs: list[tuple[int, int]] = []
    floor_collision_pairs: list[tuple[int, int]] = []
    collision_geoms = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("collision_hand_")
    ]
    collision_geoms = [name for name in collision_geoms if name is not None]
    collision_minimum = float(settings.get("self_collision_clearance_m", 0.002))
    if not np.isfinite(collision_minimum) or collision_minimum < 0.0:
        raise ValueError("self-collision clearance must be finite and non-negative")
    hand_geom_ids = [model.geom(name).id for name in collision_geoms]
    # SPIDER runtime uses explicit XML pairs and therefore leaves default masks
    # at 0/0.  MINK enumerates planning pairs from those masks, so they must be
    # enabled *before* either collision limit is constructed.  The old order
    # silently created an empty self-collision constraint.
    if settings.get("self_collision", True) and hand_geom_ids:
        _enable_planning_collision_masks(model, hand_geom_ids, [])
    if settings.get("self_collision", True):
        if collision_geoms:
            self_collision_groups = _explicit_collision_groups(
                model, mujoco, hand_geom_ids=set(hand_geom_ids),
            )
            if not self_collision_groups:
                raise ValueError(
                    "self-collision was requested but the runtime XML has no explicit hand pairs"
                )
            collision_limit = mink.CollisionAvoidanceLimit(
                model, self_collision_groups,
                minimum_distance_from_collisions=collision_minimum,
                collision_detection_distance=0.02,
                include_explicit_pairs=True,
            )
            if not collision_limit.geom_id_pairs:
                raise ValueError(
                    "self-collision was requested but MINK enumerated zero hand pairs"
                )
            if strict_collision_resolution:
                collision_limit = _StrictCollisionLimit(
                    collision_limit, mujoco,
                    minimum_distance=collision_minimum,
                    depenetration_step=depenetration_step,
                )
                strict_collision_limits.append(collision_limit)
            limits.append(collision_limit)
    if object_collision_avoidance:
        object_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_side}_object_joint",
        )
        if object_joint_id < 0 or int(model.jnt_type[object_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(
                "object_collision_avoidance requires a free-object scene.xml; "
                "act-scene object controls are forbidden"
            )
        object_body_id = int(model.jnt_bodyid[object_joint_id])
        object_collision_geoms = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            for geom in physical_object_geom_ids(model, mujoco, object_body_id)
        ]
        if not collision_geoms or not object_collision_geoms:
            raise ValueError("object_collision_avoidance requires hand and object collision geoms")
        _enable_planning_collision_masks(model, hand_geom_ids, [
            model.geom(name).id for name in object_collision_geoms
        ])
        object_geom_id_set = {
            model.geom(name).id for name in object_collision_geoms
        }
        object_collision_groups = _explicit_collision_groups(
            model, mujoco, hand_geom_ids=set(hand_geom_ids),
            object_geom_ids=object_geom_id_set,
        )
        if not object_collision_groups:
            raise ValueError("runtime XML has no explicit hand-object collision pairs")
        object_collision_limit = mink.CollisionAvoidanceLimit(
            model,
            object_collision_groups,
            minimum_distance_from_collisions=object_collision_clearance,
            collision_detection_distance=object_collision_detection,
        )
        object_collision_pairs = list(object_collision_limit.geom_id_pairs)
        if not object_collision_pairs:
            raise ValueError("no hand-object collision pairs passed MuJoCo filtering")
        if strict_collision_resolution:
            object_collision_limit = _StrictCollisionLimit(
                object_collision_limit, mujoco,
                minimum_distance=object_collision_clearance,
                depenetration_step=depenetration_step,
            )
            strict_collision_limits.append(object_collision_limit)
        limits.append(object_collision_limit)
        limits.append(_VelocityLock(
            model, int(model.jnt_dofadr[object_joint_id]), mink.limits.Constraint,
        ))
        if precontact_clearance > 0.0:
            for finger, finger_name in enumerate(FINGER_NAMES):
                base_name = finger_name.removesuffix("_tip")
                finger_geom_ids = {
                    geom_id for geom_id in hand_geom_ids
                    if f"_{base_name}_" in (
                        mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id,
                        ) or ""
                    )
                }
                groups = _explicit_collision_groups(
                    model, mujoco, hand_geom_ids=finger_geom_ids,
                    object_geom_ids=object_geom_id_set,
                )
                if not groups:
                    raise ValueError(
                        f"pre-contact clearance has no runtime pairs for {finger_name}"
                    )
                inner = mink.CollisionAvoidanceLimit(
                    model, groups,
                    minimum_distance_from_collisions=precontact_clearance,
                    collision_detection_distance=object_collision_detection,
                )
                strict = _StrictCollisionLimit(
                    inner, mujoco,
                    minimum_distance=precontact_clearance,
                    depenetration_step=depenetration_step,
                )
                precontact_limits[finger] = strict
                strict_collision_limits.append(strict)
                limits.append(strict)
    if floor_collision_avoidance:
        floor_geom_ids = {
            geom for geom in range(model.ngeom)
            if (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom,
            ) or "") == "floor"
        }
        if not floor_geom_ids:
            raise ValueError("floor collision avoidance requested but no floor geom exists")
        _enable_planning_collision_masks(model, hand_geom_ids, list(floor_geom_ids))
        floor_collision_groups = _explicit_collision_groups(
            model, mujoco, hand_geom_ids=set(hand_geom_ids),
            object_geom_ids=floor_geom_ids,
        )
        if not floor_collision_groups:
            # ``scene_torque.xml`` adds every hand--floor pair, while the
            # kinematic ``scene.xml`` used by MINK historically omitted them.
            # With one floor geom, this Cartesian product is exactly the
            # torque-runtime contact set (not the unsafe hand--hand product).
            floor_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
                for geom in sorted(floor_geom_ids)
            ]
            floor_collision_groups = [(collision_geoms, floor_names)]
        floor_collision_limit = mink.CollisionAvoidanceLimit(
            model, floor_collision_groups,
            minimum_distance_from_collisions=floor_collision_clearance,
            collision_detection_distance=floor_collision_detection,
        )
        floor_collision_pairs = list(floor_collision_limit.geom_id_pairs)
        if not floor_collision_pairs:
            raise ValueError("no hand-floor collision pairs passed MuJoCo filtering")
        if strict_collision_resolution:
            floor_collision_limit = _StrictCollisionLimit(
                floor_collision_limit, mujoco,
                minimum_distance=floor_collision_clearance,
                depenetration_step=depenetration_step,
            )
            strict_collision_limits.append(floor_collision_limit)
        limits.append(floor_collision_limit)
    count, hand_count = len(human["frame_indices"]), len(hand_order)
    if contact_mask is not None and contact_mask.shape != (count, 5):
        raise ValueError(
            f"contact mask must have shape ({count},5), got {contact_mask.shape}"
        )
    palm_tasks: dict[int, Any] = {}
    palm_constraint = float(settings.get("palm_position_cost", 0.0)) > 0.0
    if palm_constraint and "T_sim_palm_target" in human:
        for hand, side in enumerate(hand_order):
            palm_tasks[hand] = mink.FrameTask(
                f"{side}_palm", "site", settings.get("palm_position_cost", 0.0),
                settings.get("palm_orientation_cost", 0.0), lm_damping=1.0,
            )
            tasks.append(palm_tasks[hand])
    qpos = np.zeros((count, model.nq), dtype=np.float64)
    wrist_actual = np.repeat(np.eye(4)[None, None], count * hand_count, axis=0).reshape(count, hand_count, 4, 4)
    tips_actual = np.repeat(np.eye(4)[None, None, None], count * hand_count * 5, axis=0).reshape(count, hand_count, 5, 4, 4)
    fingertip_position_error = np.zeros(count)
    fingertip_orientation_error = np.zeros(count)
    fingertip_position_error_projected = np.zeros(count)
    projection_shift = np.zeros((count, hand_count, 5), dtype=np.float64)
    projected_targets = np.repeat(
        np.eye(4)[None, None, None], count * hand_count * 5, axis=0,
    ).reshape(count, hand_count, 5, 4, 4)
    wrist_position_error = np.zeros(count)
    wrist_orientation_error = np.zeros(count)
    retargeting_loss = np.zeros(count)
    joint_margin = np.zeros(count)
    joint_violation = np.zeros(count, dtype=bool)
    collision_distance = np.zeros(count)
    collision_violation = np.zeros(count, dtype=bool)
    object_collision_distance = np.full(count, np.nan)
    object_collision_violation = np.zeros(count, dtype=bool)
    precontact_distance = np.full((count, 5), np.nan, dtype=np.float64)
    precontact_violation = np.zeros((count, 5), dtype=bool)
    precontact_required = (
        precontact_clearance_mask(contact_mask, contact_state)
        if precontact_limits else np.zeros((count, 5), dtype=bool)
    )
    floor_collision_distance = np.full(count, np.nan)
    floor_collision_violation = np.zeros(count, dtype=bool)
    sim_dt = float(settings["sim_dt"])
    max_iterations = int(settings["max_iterations_per_frame"])
    if not np.isfinite(sim_dt) or sim_dt <= 0.0 or max_iterations < 1:
        raise ValueError("sim_dt and max_iterations_per_frame must be positive")
    initial_iterations = max(100, max_iterations)
    for frame in range(count):
        for finger, limit in precontact_limits.items():
            limit.active = bool(precontact_required[frame, finger])
        for strict_limit in strict_collision_limits:
            strict_limit.enabled = False
        if "T_sim_object_reference" in human:
            _set_object_pose(model, configuration.data, mujoco, object_side, human["T_sim_object_reference"][frame, 0])
        configuration.update()
        fingertip_targets = human["T_sim_fingertip_target"][frame]
        if surface_projection and object_hulls is not None:
            corrected, shifts = _project_fingertip_targets(
                object_hulls, human, frame, hand_count, projection_margin,
                [capsule_geoms[side] for side in hand_order],
                projection_reach,
            )
            fingertip_targets = corrected
            projected_targets[frame] = corrected
            projection_shift[frame] = np.linalg.norm(shifts, axis=-1)
        opposition_enabled = bool(settings.get("opposition_synthesis", False))
        if (contact_aware and opposition_enabled and contact_mask is not None
                and contact_mask[frame, 0] and object_hulls is not None):
            # ramp the synthesized opposition targets in over `opposition_ramp`
            # frames so the IK can track the transition smoothly
            ramp = int(settings.get("opposition_ramp_frames", 10))
            onset = 0
            for scan in range(frame, -1, -1):
                if not contact_mask[scan, 0]:
                    onset = scan + 1
                    break
            alpha = min(1.0, (frame - onset + 1) / max(ramp, 1))
            for hand in range(hand_count):
                opposition = _opposition_targets(
                    human["T_sim_fingertip_target"][frame, hand, 0, :3, 3],
                    fingertip_targets[hand],
                    object_hulls,
                    human["T_sim_object_reference"][frame, 0],
                )
                if opposition is not None:
                    original = human["T_sim_fingertip_target"][frame, hand]
                    blended = fingertip_targets[hand].copy()
                    blended[1:, :3, 3] = (
                        (1.0 - alpha) * original[1:, :3, 3] + alpha * opposition[1:, :3, 3]
                    )
                    fingertip_targets[hand] = blended
        for hand in range(hand_count):
            hand_confidence = float(human["confidence_hand"][frame, hand]) if human["valid_hand"][frame, hand] else 0.0
            if not settings.get("confidence_weighting", True):
                hand_confidence = float(hand_confidence > 0)
            wrist_task = wrist_tasks[hand]
            wrist_task.set_position_cost(float(settings["wrist_position_cost"]) * hand_confidence)
            wrist_task.set_orientation_cost(float(settings["wrist_orientation_cost"]) * hand_confidence)
            wrist_task.set_target(_pose_to_se3(mink, human["T_sim_wrist_target"][frame, hand]))
            if hand in palm_tasks:
                palm_tasks[hand].set_target(
                    _pose_to_se3(mink, human["T_sim_palm_target"][frame, hand])
                )
            for finger in range(5):
                confidence = float(human["confidence_fingertip"][frame, hand, finger]) * float(hand_confidence > 0)
                if (contact_timing_weighting and contact_mask is not None
                        and bool(contact_mask[frame, finger])):
                    # GT contact timing is an input-quality ablation: it only
                    # prioritizes tracking the already observed fingertip
                    # target.  It does not synthesize a grasp target, change
                    # object geometry, or alter any MuJoCo control.
                    confidence *= contact_timing_cost_multiplier
                if not settings.get("confidence_weighting", True):
                    confidence = float(confidence > 0)
                task = finger_tasks[hand, finger]
                task.set_position_cost(float(settings["fingertip_position_cost"]) * confidence)
                orientation_weight = confidence * float(human["valid_fingertip_orientation"][frame, hand, finger])
                axis, _ = FINGER_DIRECTION_AXES[finger]
                orientation_components = np.ones(3, dtype=np.float64)
                orientation_components[axis] = 0.0
                task.set_orientation_cost(
                    float(settings["fingertip_orientation_cost"]) * orientation_weight
                    * orientation_components
                )
                task.set_target(_pose_to_se3(mink, fingertip_targets[hand, finger]))
        iteration_count = initial_iterations if frame == 0 else max_iterations
        frame_dt = (
            sim_dt if frame == 0 else
            float(human["timestamps_s"][frame] - human["timestamps_s"][frame - 1])
        )
        if frame_displacement_limit is not None:
            frame_displacement_limit.set_previous(
                None if frame == 0 else qpos[frame - 1],
                None if frame == 0 else frame_dt,
            )
        integration_dt = (
            sim_dt if frame == 0 else
            min(sim_dt, frame_dt / iteration_count)
        )
        for iteration in range(iteration_count):
            try:
                velocity = mink.solve_ik(
                    configuration, tasks, integration_dt, solver, damping=1e-5,
                    limits=limits, safety_break=True,
                )
            except Exception as error:
                raise RuntimeError(
                    f"MINK solve failed at frame {frame}, main iteration {iteration}"
                ) from error
            configuration.integrate_inplace(velocity, integration_dt)
            if "T_sim_object_reference" in human:
                _set_object_pose(model, configuration.data, mujoco, object_side, human["T_sim_object_reference"][frame, 0])
                configuration.update()
        if (contact_aware or object_clearance_correction) and object_hulls is not None:
            for hand, side in enumerate(hand_order):
                transform_object = human["T_sim_object_reference"][frame, 0]
                for _round in range(contact_rounds):
                    shifted = False
                    for finger, name in enumerate(FINGER_NAMES):
                        site_pose = configuration.get_transform_frame_to_world(
                            f"{side}_{name}", "site",
                        ).as_matrix()
                        measured = _capsule_hull_contact(
                            site_pose, capsule_geoms[side][finger], object_hulls,
                            transform_object,
                        )
                        if measured is None:
                            continue
                        deepest, normal = measured
                        gt_contact = bool(
                            contact_mask is not None and contact_mask[frame, finger]
                        )
                        import os
                        if os.environ.get("CONTACT_DEBUG"):
                            print(f"[contact-debug] frame={frame} round={_round} "
                                  f"{name} deepest={deepest:.4f} mask={gt_contact}",
                                  flush=True)
                        clearance = _contact_correction_clearance(
                            deepest_m=deepest,
                            gt_contact=gt_contact,
                            contact_aware=contact_aware,
                            object_clearance_correction=object_clearance_correction,
                            contact_reach_m=contact_reach,
                            contact_attraction_requires_reach=(
                                contact_attraction_requires_reach
                            ),
                            contact_clearance_m=(
                                contact_clearance_thumb if finger == 0
                                else contact_clearance
                            ),
                            object_clearance_m=object_collision_clearance,
                        )
                        if clearance is None:
                            continue
                        desired_shift = float(-deepest + clearance)
                        if finger == 0:
                            shift = float(np.clip(
                                desired_shift,
                                -contact_attraction_max_shift,
                                object_depenetration_max_shift,
                            ))
                        else:
                            wrap_shift = None
                            if contact_wrap_synthesis:
                                wrap_shift = _wrap_shift_for_finger(
                                    site_pose, capsule_geoms[side][finger], object_hulls,
                                    transform_object, beyond_m=0.002,
                                )
                            if wrap_shift is None:
                                shift = float(np.clip(
                                    desired_shift,
                                    -contact_attraction_max_shift,
                                    object_depenetration_max_shift,
                                ))
                            else:
                                shift = float(np.clip(wrap_shift, -0.05, 0.05))
                        if abs(shift) < 5e-4:
                            continue
                        task = finger_tasks[hand, finger]
                        target = task.transform_target_to_world
                        target_translation = np.asarray(target.translation()) + normal * shift
                        task.set_target(
                            mink.SE3.from_rotation_and_translation(
                                target.rotation(), target_translation,
                            )
                        )
                        shifted = True
                    if not shifted:
                        break
                    for correction_iteration in range(max(1, max_iterations)):
                        try:
                            velocity = mink.solve_ik(
                                configuration, tasks, integration_dt, solver, damping=1e-5,
                                limits=limits, safety_break=True,
                            )
                        except Exception as error:
                            raise RuntimeError(
                                f"MINK solve failed at frame {frame}, contact round "
                                f"{_round}, iteration {correction_iteration}"
                            ) from error
                        configuration.integrate_inplace(velocity, integration_dt)
                        if "T_sim_object_reference" in human:
                            _set_object_pose(model, configuration.data, mujoco, object_side, human["T_sim_object_reference"][frame, 0])
                            configuration.update()
        # A GT object pose can appear already intersecting the current robot
        # state.  MINK's stock velocity damper cannot leave such a state.  Fit
        # the human targets first, then resolve each runtime collision family
        # with bounded separating QPs before accepting the nominal frame.
        active_strict_limits = [
            limit for limit in strict_collision_limits if limit.active
        ]
        if active_strict_limits:
            for active_index, active_limit in enumerate(active_strict_limits):
                for strict_limit in strict_collision_limits:
                    strict_limit.enabled = False
                for strict_limit in active_strict_limits[: active_index + 1]:
                    strict_limit.enabled = True
                for resolution_iteration in range(strict_collision_iterations):
                    family_rows = [
                        (
                            _collision_diagnostic(
                            model, configuration.data, mujoco,
                            strict_limit.geom_id_pairs,
                            strict_limit.minimum_distance,
                            )[0],
                            strict_limit.minimum_distance,
                        )
                        for strict_limit in active_strict_limits[: active_index + 1]
                    ]
                    minimum, required = min(
                        family_rows, key=lambda row: row[0] - row[1]
                    )
                    if all(
                        value >= threshold - collision_validation_tolerance
                        for value, threshold in family_rows
                    ):
                        break
                    try:
                        # This is a feasibility projection, not another human-
                        # tracking iteration.  Reusing ``tasks`` here made the
                        # Cartesian targets pull directly against the active
                        # separating constraint, creating a false equilibrium
                        # just below the requested clearance no matter how many
                        # iterations were allowed.  With no task objective,
                        # solve_ik uses damping as a minimum-displacement QP
                        # while retaining every joint/velocity/collision limit.
                        velocity = mink.solve_ik(
                            configuration, (), integration_dt, solver,
                            damping=1e-5, limits=limits, safety_break=True,
                        )
                    except Exception as error:
                        raise RuntimeError(
                            f"MINK collision resolution failed at frame {frame}, "
                            f"family {active_index}, iteration {resolution_iteration}"
                        ) from error
                    configuration.integrate_inplace(velocity, integration_dt)
                    if "T_sim_object_reference" in human:
                        _set_object_pose(
                            model, configuration.data, mujoco, object_side,
                            human["T_sim_object_reference"][frame, 0],
                        )
                        configuration.update()
                else:
                    raise RuntimeError(
                        f"MINK collision resolution did not converge at frame {frame}, "
                        f"family {active_index}: minimum={minimum:.9f} m, "
                        f"required={required:.9f} m"
                    )
        qpos[frame] = configuration.q.copy()
        for hand, side in enumerate(hand_order):
            wrist_actual[frame, hand] = configuration.get_transform_frame_to_world(f"{side}_palm", "site").as_matrix()
            for finger, name in enumerate(FINGER_NAMES):
                tips_actual[frame, hand, finger] = configuration.get_transform_frame_to_world(f"{side}_{name}", "site").as_matrix()
        valid_hand = human["valid_hand"][frame].astype(bool)
        valid_tip = np.repeat(valid_hand[:, None], 5, axis=1)
        tip_position_values = np.linalg.norm(
            tips_actual[frame, ..., :3, 3] - human["T_sim_fingertip_target"][frame, ..., :3, 3], axis=-1,
        )
        tip_position_projected = np.linalg.norm(
            tips_actual[frame, ..., :3, 3] - fingertip_targets[..., :3, 3], axis=-1,
        )
        tip_orientation_values = np.zeros((hand_count, 5), dtype=np.float64)
        for finger, (axis, sign) in enumerate(FINGER_DIRECTION_AXES):
            actual_direction = sign * tips_actual[frame, :, finger, :3, axis]
            target_direction = sign * human["T_sim_fingertip_target"][frame, :, finger, :3, axis]
            cosine = np.einsum("hi,hi->h", actual_direction, target_direction)
            tip_orientation_values[:, finger] = np.arccos(np.clip(cosine, -1.0, 1.0))
        wrist_position_values = np.linalg.norm(
            wrist_actual[frame, ..., :3, 3] - human["T_sim_wrist_target"][frame, ..., :3, 3], axis=-1,
        )
        wrist_orientation_values = Rotation.from_matrix(
            np.swapaxes(wrist_actual[frame, ..., :3, :3], -1, -2)
            @ human["T_sim_wrist_target"][frame, ..., :3, :3]
        ).magnitude()
        fingertip_position_error[frame] = float(tip_position_values[valid_tip].mean()) if valid_tip.any() else 0.0
        fingertip_position_error_projected[frame] = (
            float(tip_position_projected[valid_tip].mean()) if valid_tip.any() else 0.0
        )
        orientation_valid = valid_tip & human["valid_fingertip_orientation"][frame].astype(bool)
        fingertip_orientation_error[frame] = float(tip_orientation_values[orientation_valid].mean()) if orientation_valid.any() else 0.0
        wrist_position_error[frame] = float(wrist_position_values[valid_hand].mean()) if valid_hand.any() else 0.0
        wrist_orientation_error[frame] = float(wrist_orientation_values[valid_hand].mean()) if valid_hand.any() else 0.0
        retargeting_loss[frame] = (
            float(settings["fingertip_position_cost"]) * fingertip_position_error[frame] ** 2
            + float(settings["fingertip_orientation_cost"]) * fingertip_orientation_error[frame] ** 2
            + float(settings["wrist_position_cost"]) * wrist_position_error[frame] ** 2
            + float(settings["wrist_orientation_cost"]) * wrist_orientation_error[frame] ** 2
        )
        joint_margin[frame], joint_violation[frame] = _joint_limit_diagnostic(
            model, configuration.data, mujoco, physical_ranges=physical_joint_ranges,
        )
        pairs = collision_limit.geom_id_pairs if collision_limit is not None else []
        collision_distance[frame], collision_violation[frame] = _collision_diagnostic(
            model, configuration.data, mujoco, pairs, collision_minimum,
            tolerance=collision_validation_tolerance,
        )
        if object_collision_limit is not None:
            (
                object_collision_distance[frame],
                object_collision_violation[frame],
            ) = _collision_diagnostic(
                model, configuration.data, mujoco, object_collision_pairs,
                object_collision_clearance,
                tolerance=collision_validation_tolerance,
            )
        if floor_collision_limit is not None:
            (
                floor_collision_distance[frame],
                floor_collision_violation[frame],
            ) = _collision_diagnostic(
                model, configuration.data, mujoco, floor_collision_pairs,
                floor_collision_clearance,
                tolerance=collision_validation_tolerance,
            )
        for finger, limit in precontact_limits.items():
            precontact_distance[frame, finger], violation = _collision_diagnostic(
                model, configuration.data, mujoco, limit.geom_id_pairs,
                precontact_clearance, tolerance=collision_validation_tolerance,
            )
            precontact_violation[frame, finger] = bool(
                limit.active and violation
            )
    qvel = np.zeros((count, model.nv), dtype=np.float64)
    for frame in range(1, count):
        dt = float(human["timestamps_s"][frame] - human["timestamps_s"][frame - 1])
        mujoco.mj_differentiatePos(model, qvel[frame], dt, qpos[frame - 1], qpos[frame])
    velocity_violation = np.zeros(count, dtype=bool)
    if velocity_limit is not None and velocity_limit.indices.size:
        velocity_violation = (np.abs(qvel[:, velocity_limit.indices]) > velocity_limit.limit[None] + 1e-6).any(axis=1)
    if strict_collision_resolution:
        failures = {
            "joint": int(joint_violation.sum()),
            "velocity": int(velocity_violation.sum()),
            "self": int(collision_violation.sum()),
            "object": int(object_collision_violation.sum()),
            "floor": int(floor_collision_violation.sum()),
            "precontact": int(precontact_violation.sum()),
        }
        if any(failures.values()):
            raise RuntimeError(
                "strict MINK produced an illegal reference and will not write it: "
                f"{failures}"
            )
    output = {
        "frame_indices": human["frame_indices"], "timestamps_s": human["timestamps_s"],
        "hand_order": human["hand_order"], "qpos": qpos, "qvel": qvel,
        "mujoco_model_signature": np.asarray(runtime_model_signature),
        "reference_trajectory_signature": np.asarray(
            reference_trajectory_signature(
                human["frame_indices"], human["timestamps_s"], qpos,
            )
        ),
        "T_sim_wrist": wrist_actual, "T_sim_fingertip": tips_actual,
        "T_sim_wrist_target": human["T_sim_wrist_target"],
        "T_sim_fingertip_target": human["T_sim_fingertip_target"],
        "valid_hand": human["valid_hand"], "confidence_hand": human["confidence_hand"],
        "retargeting_loss": retargeting_loss,
        "fingertip_position_error_m": fingertip_position_error,
        "fingertip_position_error_vs_projected_m": fingertip_position_error_projected,
        "fingertip_projection_shift_mean_m": projection_shift.mean(axis=(1, 2)) if surface_projection else np.zeros(count),
        "fingertip_orientation_error_rad": fingertip_orientation_error,
        "fingertip_direction_error_rad": fingertip_orientation_error,
        "wrist_position_error_m": wrist_position_error,
        "wrist_orientation_error_rad": wrist_orientation_error,
        "joint_limit_min_margin": joint_margin,
        "joint_limit_violation": joint_violation,
        "velocity_limit_violation": velocity_violation,
        "self_collision_min_distance_m": collision_distance,
        "self_collision_violation": collision_violation,
        "object_collision_min_distance_m": object_collision_distance,
        "object_collision_violation": object_collision_violation,
        "floor_collision_min_distance_m": floor_collision_distance,
        "floor_collision_violation": floor_collision_violation,
        "frequency": np.asarray(1.0 / np.median(np.diff(human["timestamps_s"]))),
    }
    if "T_sim_object_reference" in human:
        output["T_sim_object_reference"] = human["T_sim_object_reference"]
    if contact_aware:
        output["contact_aware_retarget"] = np.asarray(True)
        output["contact_clearance_m"] = np.asarray(contact_clearance)
        output["contact_wrap_synthesis"] = np.asarray(contact_wrap_synthesis)
        output["contact_attraction_max_shift_m"] = np.asarray(
            contact_attraction_max_shift
        )
        output["contact_attraction_requires_reach"] = np.asarray(
            contact_attraction_requires_reach
        )
        output["object_depenetration_max_shift_m"] = np.asarray(
            object_depenetration_max_shift
        )
    if contact_signature:
        output["contact_reference_signature"] = np.asarray(contact_signature)
        output["contact_position_source"] = np.asarray(contact_position_source)
        output["episode_id"] = np.asarray(episode_id)
    if precontact_limits:
        output["precontact_finger_clearance_m"] = np.asarray(
            precontact_clearance
        )
        output["precontact_clearance_required"] = precontact_required
        output["precontact_finger_object_min_distance_m"] = precontact_distance
        output["precontact_finger_clearance_violation"] = precontact_violation
    output["finger_joint_limit_margin_rad"] = np.asarray(
        float(settings.get("finger_joint_limit_margin_rad", 0.0))
    )
    output["strict_collision_resolution"] = np.asarray(
        strict_collision_resolution
    )
    if contact_timing_weighting:
        output["contact_timing_weighting"] = np.asarray(True)
        output["contact_timing_cost_multiplier"] = np.asarray(contact_timing_cost_multiplier)
    if object_clearance_correction:
        output["object_clearance_correction"] = np.asarray(True)
    if object_collision_avoidance:
        output["object_collision_avoidance"] = np.asarray(True)
        output["object_collision_clearance_m"] = np.asarray(object_collision_clearance)
    if floor_collision_avoidance:
        output["floor_collision_avoidance"] = np.asarray(True)
        output["floor_collision_clearance_m"] = np.asarray(floor_collision_clearance)
    if surface_projection:
        output["T_sim_fingertip_target_projected"] = projected_targets
        output["fingertip_projection_shift_m"] = projection_shift
        output["surface_projection_margin"] = np.asarray(projection_margin)
    validate_robot_reference(output)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    intrahand_audit = audit_intrahand_trajectory(
        model, qpos, tolerance=collision_validation_tolerance,
    )
    intrahand_path = destination.parent / "intrahand_collision_audit.json"
    intrahand_artifact = {
        "scene": str(Path(model_path).resolve()),
        "human_reference": str(Path(human_reference_path).resolve()),
        **intrahand_audit,
    }
    intrahand_path.write_text(json.dumps(intrahand_artifact, indent=2) + "\n")
    np.savez_compressed(destination, **output)
    report = {
        "schema_version": "1.0", "profile": config.name, "solver": solver,
        "human_reference": str(Path(human_reference_path).resolve()),
        "scene": str(Path(model_path).resolve()),
        "contact_reference": (
            None if contact_reference_path is None
            else str(Path(contact_reference_path).resolve())
        ),
        "mujoco_model_signature": runtime_model_signature,
        "reference_trajectory_signature": str(
            output["reference_trajectory_signature"].item()
        ),
        "contact_reference_signature": contact_signature or None,
        "episode_id": episode_id or None,
        "frame_count": count, "hand_order": hand_order,
        "joint_limit_violation_count": int(joint_violation.sum()),
        "velocity_limit_violation_count": int(velocity_violation.sum()),
        "self_collision_violation_count": int(collision_violation.sum()),
        "self_collision_pair_count": 0 if collision_limit is None else len(collision_limit.geom_id_pairs),
        "self_collision_min_distance_m": (
            None if collision_limit is None else float(collision_distance.min())
        ),
        "self_collision_clearance_m": collision_minimum,
        "finger_joint_limit_margin_rad": float(
            settings.get("finger_joint_limit_margin_rad", 0.0)
        ),
        "strict_collision_resolution": strict_collision_resolution,
        "collision_depenetration_step_m": depenetration_step,
        "collision_validation_tolerance_m": collision_validation_tolerance,
        "object_collision_violation_count": int(object_collision_violation.sum()),
        "object_collision_pair_count": len(object_collision_pairs),
        "object_collision_min_distance_m": (
            None if not object_collision_avoidance else float(object_collision_distance.min())
        ),
        "object_collision_avoidance": object_collision_avoidance,
        "floor_collision_violation_count": int(floor_collision_violation.sum()),
        "floor_collision_pair_count": len(floor_collision_pairs),
        "floor_collision_min_distance_m": (
            None if not floor_collision_avoidance
            else float(floor_collision_distance.min())
        ),
        "floor_collision_avoidance": floor_collision_avoidance,
        "object_clearance_correction": object_clearance_correction,
        "contact_timing_weighting": contact_timing_weighting,
        "contact_timing_cost_multiplier": (
            contact_timing_cost_multiplier if contact_timing_weighting else None
        ),
        "precontact_finger_clearance_m": (
            precontact_clearance if precontact_limits else None
        ),
        "precontact_finger_clearance_violation_count": (
            int(precontact_violation.sum()) if precontact_limits else None
        ),
        "precontact_clearance_semantics": (
            "before_each_digit_first_human_contact_evidence; never-observed digits remain guarded"
            if precontact_limits else None
        ),
        "contact_attraction_max_shift_m": (
            contact_attraction_max_shift if contact_aware else None
        ),
        "contact_attraction_requires_reach": (
            contact_attraction_requires_reach if contact_aware else None
        ),
        "object_depenetration_max_shift_m": (
            object_depenetration_max_shift
            if (contact_aware or object_clearance_correction) else None
        ),
        "retargeting_loss_mean": float(retargeting_loss.mean()),
        "fingertip_position_error_mean_m": float(fingertip_position_error.mean()),
        "fingertip_position_error_vs_projected_mean_m": float(fingertip_position_error_projected.mean()),
        "fingertip_projection_shift_mean_m": float(
            projection_shift.mean() if surface_projection else 0.0
        ),
        "surface_projection": surface_projection,
        "fingertip_direction_error_mean_rad": float(fingertip_orientation_error.mean()),
        "wrist_orientation_error_mean_rad": float(wrist_orientation_error.mean()),
        "intrahand_collision_audit_path": str(intrahand_path),
        "intrahand_collision_summary": {
            "pair_count": intrahand_audit["pair_count"],
            "counts": intrahand_audit["counts"],
            "by_classification": intrahand_audit["by_classification"],
            "min_distance_m": intrahand_audit["min_distance_m"],
            "penetrating_frames": intrahand_audit["penetrating_frames"],
            "tolerance_m": intrahand_audit["tolerance_m"],
        },
        "output": str(destination),
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination
