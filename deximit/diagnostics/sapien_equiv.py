"""Convert recorded PhysX behavior into an equivalent MuJoCo profile.

The conversion is deliberately based on observable behavior instead of copying
engine-specific numbers.  In particular, PhysX rigid-body damping specifies a
per-second velocity decay rate, while MuJoCo free-joint damping is a viscous
generalized force.  The latter is calibrated in the actual compiled model so a
single no-contact step has the same velocity multiplier as PhysX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SOURCE_TIMESTEP_S = 1.0 / 240.0
PHYSX_TGS_POSITION_ITERATIONS = 25
MUJOCO_TGS_SUBSTEP_S = SOURCE_TIMESTEP_S / PHYSX_TGS_POSITION_ITERATIONS


@dataclass(frozen=True)
class SapienBody:
    mass: float
    inertia: np.ndarray
    com_pos: np.ndarray
    com_quat: np.ndarray
    linear_decay_rate: float
    angular_decay_rate: float
    contact_offset: float
    rest_offset: float
    static_friction: float
    dynamic_friction: float
    min_separation: float
    max_impulse: float


class PhysxTgsForceDrive:
    """Apply PhysX's scalar implicit force-drive update in MuJoCo.

    PhysX solves one drive coordinate at a time using that coordinate's
    articulated unit impulse response.  More importantly, TGS keeps the
    accumulated drive impulse and joint displacement across all position
    iterations of one simulation frame.  Each iteration applies only the
    change from the previously accumulated impulse.  A regular MuJoCo
    position actuator has neither of those semantics.

    This diagnostic converter follows ``computeImplicitDriveParamsForceDrive``
    and ``computeDriveImpulse`` in NVIDIA PhysX's
    ``DyCpuGpuArticulation.h``.  The released DexImit controller uses zero
    velocity targets and a practically unlimited 1e10 force limit.
    """

    def __init__(
        self, model: Any, mujoco_module: Any, joint_names: list[str], *,
        stiffness: float = 1000.0, damping: float = 100.0,
    ) -> None:
        self.model = model
        self.mujoco = mujoco_module
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.dt = float(model.opt.timestep)
        if (
            len(joint_names) != 18 or len(set(joint_names)) != 18
            or not np.isfinite((self.stiffness, self.damping, self.dt)).all()
            or self.stiffness < 0.0 or self.damping < 0.0
            or not np.isclose(self.dt, MUJOCO_TGS_SUBSTEP_S, atol=1.0e-15)
        ):
            raise ValueError("PhysX TGS drive contract is invalid")
        self.qpos_addresses: list[int] = []
        self.dof_addresses: list[int] = []
        self.lower_limits = np.full(18, -np.inf, dtype=np.float64)
        self.upper_limits = np.full(18, np.inf, dtype=np.float64)
        for name in joint_names:
            joint = mujoco_module.mj_name2id(
                model, mujoco_module.mjtObj.mjOBJ_JOINT, name,
            )
            if joint < 0 or int(model.jnt_type[joint]) not in (
                int(mujoco_module.mjtJoint.mjJNT_HINGE),
                int(mujoco_module.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError(f"PhysX TGS drive lacks scalar joint {name!r}")
            self.qpos_addresses.append(int(model.jnt_qposadr[joint]))
            self.dof_addresses.append(int(model.jnt_dofadr[joint]))
            if bool(model.jnt_limited[joint]):
                self.lower_limits[len(self.qpos_addresses) - 1] = float(
                    model.jnt_range[joint, 0]
                )
                self.upper_limits[len(self.qpos_addresses) - 1] = float(
                    model.jnt_range[joint, 1]
                )
        self.target = np.zeros(18, dtype=np.float64)
        self.full_mass = np.empty((model.nv, model.nv), dtype=np.float64)
        self._frame_qpos = np.empty(18, dtype=np.float64)
        self._frame_target = np.empty(18, dtype=np.float64)
        self._inverse_mass = np.empty((18, 18), dtype=np.float64)
        self._drive_bias = np.empty(18, dtype=np.float64)
        self._drive_velocity_multiplier = np.empty(18, dtype=np.float64)
        self._drive_position_coefficient = np.empty(18, dtype=np.float64)
        self._accumulated_drive_impulse = np.zeros(18, dtype=np.float64)
        self._frame_iteration = PHYSX_TGS_POSITION_ITERATIONS

    def set_target(self, target: np.ndarray) -> None:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != (18,) or not np.isfinite(value).all():
            raise ValueError("PhysX TGS drive target must be one finite 18-vector")
        if self._frame_iteration < PHYSX_TGS_POSITION_ITERATIONS:
            raise RuntimeError("cannot change a PhysX drive target inside one TGS frame")
        self.target[:] = value

    def begin_frame(self, data: Any) -> None:
        """Initialize the state shared by one frame's 25 TGS iterations.

        PhysX builds the articulation drive rows once per full simulation
        frame.  Therefore the unit response and the initial position error do
        not get silently rebuilt at every TGS substep.
        """
        if self._frame_iteration < PHYSX_TGS_POSITION_ITERATIONS:
            raise RuntimeError("previous PhysX TGS frame is incomplete")
        self.mujoco.mj_forward(self.model, data)
        self.mujoco.mj_fullM(self.model, self.full_mass, data.qM)
        robot_mass = self.full_mass[
            np.ix_(self.dof_addresses, self.dof_addresses)
        ]
        self._inverse_mass[:] = np.linalg.inv(robot_mass)
        self._frame_qpos[:] = data.qpos[self.qpos_addresses]
        self._frame_target[:] = self.target
        initial_error = self._frame_target - self._frame_qpos

        # PhysX calls computeImplicitDriveParamsForceDrive with the TGS
        # substep as dt and the full 1/240 s frame as simDt.  Target velocity
        # is zero in DexImit, so its two target-velocity terms vanish.
        implicit_term = self.dt * (
            self.dt * self.stiffness + self.damping
        )
        unit_response = np.diag(self._inverse_mass)
        response_scale = 1.0 / (1.0 + implicit_term * unit_response)
        self._drive_position_coefficient[:] = (
            self.stiffness * response_scale * self.dt
        )
        self._drive_velocity_multiplier[:] = -response_scale * implicit_term
        self._drive_bias[:] = self._drive_position_coefficient * initial_error
        self._accumulated_drive_impulse[:] = 0.0
        self._frame_iteration = 0

    def apply_impulses(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        """Apply drive/limit rows and return their generalized impulses.

        The returned vectors use the same 18-axis order as ``joint_names``.
        Keeping drive and hard-limit impulses separate makes it possible to
        audit loaded contact without pretending that the custom velocity
        update was solved as a native MuJoCo actuator force.
        """
        if self._frame_iteration >= PHYSX_TGS_POSITION_ITERATIONS:
            raise RuntimeError("begin_frame must precede each PhysX TGS frame")
        if not np.array_equal(self.target, self._frame_target):
            raise RuntimeError("PhysX drive target changed inside one TGS frame")
        joint_position_delta = (
            data.qpos[self.qpos_addresses] - self._frame_qpos
        )
        drive_impulses = np.zeros(18, dtype=np.float64)
        limit_impulses = np.zeros(18, dtype=np.float64)
        for axis, dof in enumerate(self.dof_addresses):
            previous_impulse = self._accumulated_drive_impulse[axis]
            accumulated_impulse = (
                previous_impulse
                + self._drive_velocity_multiplier[axis] * float(data.qvel[dof])
                + self._drive_bias[axis]
                - self._drive_position_coefficient[axis]
                * joint_position_delta[axis]
            )
            impulse_delta = accumulated_impulse - previous_impulse
            self._accumulated_drive_impulse[axis] = accumulated_impulse
            data.qvel[self.dof_addresses] += (
                self._inverse_mass[:, axis] * impulse_delta
            )
            drive_impulses[axis] += impulse_delta
            # PhysX revisits articulation limits while scalar drive rows are
            # solved.  Waiting until every drive row has run lets one blocked
            # joint inject a spurious coupled velocity into the other joints.
            # Enforce every active unilateral limit after each scalar row.
            predicted = (
                data.qpos[self.qpos_addresses]
                + data.qvel[self.dof_addresses] * self.dt
            )
            for limit_axis, limit_dof in enumerate(self.dof_addresses):
                boundary = None
                if predicted[limit_axis] < self.lower_limits[limit_axis]:
                    boundary = self.lower_limits[limit_axis]
                elif predicted[limit_axis] > self.upper_limits[limit_axis]:
                    boundary = self.upper_limits[limit_axis]
                if boundary is None:
                    continue
                desired_velocity = (
                    boundary - data.qpos[self.qpos_addresses[limit_axis]]
                ) / self.dt
                limit_impulse = (
                    desired_velocity - data.qvel[limit_dof]
                ) / float(self._inverse_mass[limit_axis, limit_axis])
                data.qvel[self.dof_addresses] += (
                    self._inverse_mass[:, limit_axis] * limit_impulse
                )
                limit_impulses[limit_axis] += limit_impulse
        self._frame_iteration += 1
        return drive_impulses, limit_impulses

    def project_hard_limits(self, data: Any) -> None:
        """Match PhysX's measured reduced-coordinate hard-limit behavior."""
        qpos = data.qpos[self.qpos_addresses]
        qvel = data.qvel[self.dof_addresses]
        below = qpos < self.lower_limits
        above = qpos > self.upper_limits
        outside = below | above
        if not np.any(outside):
            return
        data.qpos[self.qpos_addresses] = np.clip(
            qpos, self.lower_limits, self.upper_limits,
        )
        outward = (below & (qvel < 0.0)) | (above & (qvel > 0.0))
        outward_dofs = np.asarray(self.dof_addresses, dtype=int)[outward]
        data.qvel[outward_dofs] = 0.0


@dataclass
class _FrictionAnchor:
    hand_body: int
    object_local: np.ndarray
    hand_local: np.ndarray


class PhysxStrongFriction:
    """Retain PhysX-style tangential friction error between source frames.

    PhysX patch friction keeps up to two correlation anchors and reuses their
    position error on later frames.  MuJoCo's ordinary contact friction is
    velocity-only, so merely copying the static and dynamic coefficients lets
    a grasp creep even when the Coulomb limit is ample.  This converter keeps
    one anchor for each convex fingertip/object pair, applies a critically
    damped tangential correction, and clamps it to the recorded static
    Coulomb limit.  One anchor is deliberate here: the source convex fingertip
    patches need translational holding but provide no evidence for inventing a
    second torsional lock.

    Call ``zero_native_friction`` and rebuild constraints before ``apply``.
    The caller owns ``data.qfrc_applied`` and must clear the previous
    microstep's applied anchor force before calling ``apply`` again.
    """

    def __init__(
        self, model: Any, mujoco_module: Any, *, object_body: int,
        object_geom: int, pair_hand_bodies: dict[int, int],
        static_friction: float, time_constant: float = SOURCE_TIMESTEP_S,
        correlation_distance: float = 0.025,
    ) -> None:
        self.model = model
        self.mujoco = mujoco_module
        self.object_body = int(object_body)
        self.object_geom = int(object_geom)
        self.pair_hand_bodies = {
            int(pair): int(body) for pair, body in pair_hand_bodies.items()
        }
        self.static_friction = float(static_friction)
        self.time_constant = float(time_constant)
        self.correlation_distance = float(correlation_distance)
        if (
            self.object_body <= 0 or self.object_body >= model.nbody
            or self.object_geom < 0 or self.object_geom >= model.ngeom
            or not self.pair_hand_bodies
            or any(pair < 0 or pair >= model.npair for pair in self.pair_hand_bodies)
            or any(body <= 0 or body >= model.nbody for body in self.pair_hand_bodies.values())
            or not np.isfinite((
                self.static_friction, self.time_constant,
                self.correlation_distance,
            )).all()
            or self.static_friction < 0.0 or self.time_constant <= 0.0
            or self.correlation_distance <= 0.0
        ):
            raise ValueError("PhysX strong-friction contract is invalid")
        self.pair_by_geoms: dict[frozenset[int], int] = {}
        for pair in self.pair_hand_bodies:
            geoms = frozenset((
                int(model.pair_geom1[pair]), int(model.pair_geom2[pair]),
            ))
            if self.object_geom not in geoms or geoms in self.pair_by_geoms:
                raise ValueError("strong-friction pair mapping is ambiguous")
            hand_geom = next(iter(geoms - {self.object_geom}))
            if int(model.geom_bodyid[hand_geom]) != self.pair_hand_bodies[pair]:
                raise ValueError("strong-friction pair/body mapping is inconsistent")
            self.pair_by_geoms[geoms] = pair
        mass = float(model.body_mass[self.object_body])
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("strong-friction object mass must be positive")
        self.stiffness = mass / self.time_constant**2
        self.damping = 2.0 * mass / self.time_constant
        self._anchors: dict[int, _FrictionAnchor] = {}

    @property
    def anchor_count(self) -> int:
        return len(self._anchors)

    def reset(self) -> None:
        self._anchors.clear()

    def zero_native_friction(self) -> None:
        for pair in self.pair_hand_bodies:
            self.model.pair_friction[pair, :2] = 0.0

    def _point_velocity(
        self, data: Any, body: int, point: np.ndarray,
    ) -> np.ndarray:
        spatial = np.zeros(6, dtype=np.float64)
        self.mujoco.mj_objectVelocity(
            self.model, data, self.mujoco.mjtObj.mjOBJ_BODY,
            int(body), spatial, 0,
        )
        return (
            spatial[3:]
            + np.cross(spatial[:3], point - data.xpos[int(body)])
        )

    def apply(self, data: Any) -> dict[int, np.ndarray]:
        """Add clamped anchor forces and return world force on the object."""
        positions: dict[int, list[np.ndarray]] = {}
        normals: dict[int, list[np.ndarray]] = {}
        normal_force: dict[int, float] = {}
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = self.pair_by_geoms.get(frozenset((
                int(contact.geom1), int(contact.geom2),
            )))
            if pair is None or int(contact.efc_address) < 0:
                continue
            force = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(
                self.model, data, contact_index, force,
            )
            if float(force[0]) <= 1.0e-9:
                continue
            position = np.asarray(contact.pos, dtype=np.float64).copy()
            normal = np.asarray(contact.frame[:3], dtype=np.float64).copy()
            toward_object = data.xpos[self.object_body] - position
            if float(normal @ toward_object) < 0.0:
                normal = -normal
            positions.setdefault(pair, []).append(position)
            normals.setdefault(pair, []).append(normal)
            normal_force[pair] = (
                normal_force.get(pair, 0.0) + max(0.0, float(force[0]))
            )
        active_pairs = set(positions)
        for pair in tuple(self._anchors):
            if pair not in active_pairs:
                del self._anchors[pair]

        applied: dict[int, np.ndarray] = {}
        for pair in active_pairs:
            hand_body = self.pair_hand_bodies[pair]
            point = np.mean(positions[pair], axis=0)
            normal = np.mean(normals[pair], axis=0)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= np.finfo(np.float64).eps:
                continue
            normal /= normal_norm
            if pair not in self._anchors:
                object_rotation = data.xmat[self.object_body].reshape(3, 3)
                hand_rotation = data.xmat[hand_body].reshape(3, 3)
                self._anchors[pair] = _FrictionAnchor(
                    hand_body=hand_body,
                    object_local=(
                        object_rotation.T
                        @ (point - data.xpos[self.object_body])
                    ),
                    hand_local=(
                        hand_rotation.T @ (point - data.xpos[hand_body])
                    ),
                )
            anchor = self._anchors[pair]
            object_point = (
                data.xpos[self.object_body]
                + data.xmat[self.object_body].reshape(3, 3)
                @ anchor.object_local
            )
            hand_point = (
                data.xpos[anchor.hand_body]
                + data.xmat[anchor.hand_body].reshape(3, 3)
                @ anchor.hand_local
            )
            error = object_point - hand_point
            error -= float(error @ normal) * normal
            if float(np.linalg.norm(error)) > self.correlation_distance:
                del self._anchors[pair]
                continue
            relative_velocity = (
                self._point_velocity(data, self.object_body, object_point)
                - self._point_velocity(data, anchor.hand_body, hand_point)
            )
            tangent_velocity = (
                relative_velocity
                - float(relative_velocity @ normal) * normal
            )
            desired = (
                -self.stiffness * error - self.damping * tangent_velocity
            )
            capacity = self.static_friction * normal_force[pair]
            desired_norm = float(np.linalg.norm(desired))
            if desired_norm > capacity > 0.0:
                desired *= capacity / desired_norm
            elif capacity <= 0.0:
                desired[:] = 0.0
            self.mujoco.mj_applyFT(
                self.model, data, desired, np.zeros(3), object_point,
                self.object_body, data.qfrc_applied,
            )
            self.mujoco.mj_applyFT(
                self.model, data, -desired, np.zeros(3), hand_point,
                anchor.hand_body, data.qfrc_applied,
            )
            applied[pair] = desired.copy()
        return applied


def advance_physx_tgs_microstep(
    model: Any, data: Any, mujoco_module: Any, drive: PhysxTgsForceDrive,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one MuJoCo contact step with the PhysX drive impulse ordering."""
    if not int(model.opt.disableflags) & int(mujoco_module.mjtDisableBit.mjDSBL_ACTUATION):
        raise RuntimeError("native MuJoCo actuation must be disabled for PhysX TGS drive")
    if not int(model.opt.disableflags) & int(mujoco_module.mjtDisableBit.mjDSBL_LIMIT):
        raise RuntimeError("native MuJoCo limits must be disabled for PhysX hard limits")
    mujoco_module.mj_forward(model, data)
    data.qvel[:] += data.qacc * drive.dt
    drive_impulses, limit_impulses = drive.apply_impulses(data)
    mujoco_module.mj_integratePos(model, data.qpos, data.qvel, drive.dt)
    drive.project_hard_limits(data)
    mujoco_module.mj_normalizeQuat(model, data.qpos)
    data.time += drive.dt
    mujoco_module.mj_forward(model, data)
    return drive_impulses, limit_impulses


def incoming_joint_wrenches_child_frame(
    model: Any, data: Any, mujoco_module: Any, body_ids: list[int],
    joint_ids: list[int],
) -> np.ndarray:
    """Measure parent-to-child joint wrenches in each child-link frame.

    ``mjData.cfrc_int`` is a world-oriented spatial wrench referenced at the
    root subtree center of mass, in torque:force order.  SAPIEN/PhysX reports
    the incoming-joint wrench at the child joint frame in force:torque order.
    This applies the same ``mju_transformSpatial`` conversion used internally
    by MuJoCo's force and torque sensors, then changes only the component order.
    """
    if len(body_ids) != len(joint_ids) or not body_ids:
        raise ValueError("incoming-joint wrench ids must be non-empty and paired")
    bodies = [int(value) for value in body_ids]
    joints = [int(value) for value in joint_ids]
    if (
        len(set(bodies)) != len(bodies)
        or len(set(joints)) != len(joints)
        or any(body <= 0 or body >= model.nbody for body in bodies)
        or any(joint < 0 or joint >= model.njnt for joint in joints)
    ):
        raise ValueError("incoming-joint wrench ids are invalid")
    for body, joint in zip(bodies, joints):
        if int(model.jnt_bodyid[joint]) != body:
            raise ValueError("incoming joint does not belong to the child body")

    mujoco_module.mj_rnePostConstraint(model, data)
    result = np.empty((len(bodies), 6), dtype=np.float64)
    for index, (body, joint) in enumerate(zip(bodies, joints)):
        root = int(model.body_rootid[body])
        transformed = np.zeros(6, dtype=np.float64)
        mujoco_module.mju_transformSpatial(
            transformed,
            np.asarray(data.cfrc_int[body], dtype=np.float64),
            1,
            np.asarray(data.xanchor[joint], dtype=np.float64),
            np.asarray(data.subtree_com[root], dtype=np.float64),
            np.asarray(data.xmat[body], dtype=np.float64),
        )
        result[index, :3] = transformed[3:]
        result[index, 3:] = transformed[:3]
    return result


def incoming_joint_wrenches_at_loader_frames(
    model: Any, data: Any, mujoco_module: Any, body_ids: list[int],
    joint_ids: list[int], pose_in_child_p_wxyz: np.ndarray,
) -> np.ndarray:
    """Measure joint wrenches in loader-defined incoming-joint frames.

    SAPIEN's PhysX loader can rotate an incoming joint relative to the URDF
    child-link frame so the articulation uses PhysX's canonical axes.  The
    supplied poses map each incoming-joint frame into its child-link frame.
    This function includes both that rotation and any small origin offset.
    """
    poses = np.asarray(pose_in_child_p_wxyz, dtype=np.float64)
    if poses.shape != (len(body_ids), 7) or not np.isfinite(poses).all():
        raise ValueError("loader joint-frame poses must have shape (joint, 7)")
    if not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=2.0e-5):
        raise ValueError("loader joint-frame quaternions are not unit length")
    bodies = [int(value) for value in body_ids]
    joints = [int(value) for value in joint_ids]
    if len(bodies) != len(joints) or not bodies:
        raise ValueError("loader joint-frame ids must be non-empty and paired")
    for body, joint in zip(bodies, joints):
        if (
            body <= 0 or body >= model.nbody
            or joint < 0 or joint >= model.njnt
            or int(model.jnt_bodyid[joint]) != body
        ):
            raise ValueError("loader joint-frame mapping is invalid")

    mujoco_module.mj_rnePostConstraint(model, data)
    result = np.empty((len(bodies), 6), dtype=np.float64)
    for index, (body, pose) in enumerate(zip(bodies, poses)):
        root = int(model.body_rootid[body])
        body_rotation = np.asarray(
            data.xmat[body], dtype=np.float64,
        ).reshape(3, 3)
        joint_rotation_child = np.empty(9, dtype=np.float64)
        mujoco_module.mju_quat2Mat(joint_rotation_child, pose[3:])
        joint_rotation_world = (
            body_rotation @ joint_rotation_child.reshape(3, 3)
        )
        joint_position_world = (
            np.asarray(data.xpos[body], dtype=np.float64)
            + body_rotation @ pose[:3]
        )
        transformed = np.zeros(6, dtype=np.float64)
        mujoco_module.mju_transformSpatial(
            transformed,
            np.asarray(data.cfrc_int[body], dtype=np.float64),
            1,
            joint_position_world,
            np.asarray(data.subtree_com[root], dtype=np.float64),
            joint_rotation_world.reshape(-1),
        )
        result[index, :3] = transformed[3:]
        result[index, 3:] = transformed[:3]
    return result


def read_body(metrics: dict[str, object]) -> SapienBody:
    """Validate and extract the runtime body values recorded from SAPIEN."""
    inertia = np.asarray(metrics["object_inertia_kg_m2"], dtype=np.float64)
    com = np.asarray(metrics["object_cmass_local_pose_wxyz"], dtype=np.float64)
    shapes = metrics.get("object_collision_shapes")
    if not isinstance(shapes, list) or len(shapes) != 1 or not isinstance(shapes[0], dict):
        raise ValueError("SAPIEN equivalence requires exactly one recorded collision shape")
    shape = shapes[0]
    body = SapienBody(
        mass=float(metrics["object_mass_kg"]),
        inertia=inertia,
        com_pos=com[:3],
        com_quat=com[3:],
        linear_decay_rate=float(metrics["object_linear_damping"]),
        angular_decay_rate=float(metrics["object_angular_damping"]),
        contact_offset=float(shape["contact_offset_m"]),
        rest_offset=float(shape["rest_offset_m"]),
        static_friction=float(shape["static_friction"]),
        dynamic_friction=float(shape["dynamic_friction"]),
        min_separation=float(metrics["minimum_right_hand_contact_separation_m"]),
        max_impulse=float(metrics["maximum_right_hand_contact_impulse_ns"]),
    )
    scalars = np.asarray([
        body.mass, body.linear_decay_rate, body.angular_decay_rate,
        body.contact_offset, body.rest_offset, body.static_friction,
        body.dynamic_friction, body.min_separation, body.max_impulse,
    ])
    if (
        inertia.shape != (3,) or com.shape != (7,)
        or not np.isfinite(inertia).all() or not np.isfinite(com).all()
        or not np.isfinite(scalars).all() or body.mass <= 0.0
        or (inertia <= 0.0).any() or body.linear_decay_rate < 0.0
        or body.angular_decay_rate < 0.0 or body.contact_offset < body.rest_offset
        or body.static_friction < 0.0 or body.dynamic_friction < 0.0
        or not np.isclose(np.linalg.norm(body.com_quat), 1.0, atol=1.0e-5)
    ):
        raise ValueError("recorded SAPIEN body properties are malformed")
    return body


def physx_decay(rate: float, dt: float) -> float:
    """One-step PhysX velocity multiplier measured by an isolated probe."""
    rate, dt = float(rate), float(dt)
    if not np.isfinite((rate, dt)).all() or rate < 0.0 or dt <= 0.0:
        raise ValueError("decay rate and timestep must be finite and non-negative")
    return max(0.0, 1.0 - rate * dt)


def scalar_viscous_damping(mass_like: float, rate: float, dt: float) -> float:
    """Closed-form MuJoCo damping for the same scalar implicit-Euler decay."""
    target = physx_decay(rate, dt)
    mass_like = float(mass_like)
    if mass_like <= 0.0 or target <= 0.0:
        raise ValueError("finite viscous conversion requires positive inertia and decay")
    return mass_like * (1.0 / target - 1.0) / dt


def implicit_drive_kd(kp: float, kd: float, dt: float) -> float:
    """Match a PhysX drive evaluated at the end of a step in MuJoCo.

    MuJoCo's implicit integrator already treats velocity damping implicitly.
    Adding ``dt * kp`` to its velocity gain accounts for PhysX also evaluating
    the position error at the end of the step.  This is the scalar one-step
    equivalent; the report keeps that limitation explicit for coupled joints.
    """
    kp, kd, dt = float(kp), float(kd), float(dt)
    if not np.isfinite((kp, kd, dt)).all() or kp < 0.0 or kd < 0.0 or dt <= 0.0:
        raise ValueError("drive gains and timestep are invalid")
    return kd + dt * kp


def apply_physx_velocity_decay(
    qvel: np.ndarray, first_dof: int, *, linear_rate: float,
    angular_rate: float, source_dt: float = SOURCE_TIMESTEP_S,
) -> dict[str, float]:
    """Apply PhysX rigid-body damping once per source-sized physics step.

    The isolated object probe shows that PhysX multiplies linear and angular
    velocity by ``1 - rate*dt`` once at the beginning of each 1/240 s step.
    Applying it once here, rather than distributing it over MuJoCo's 25 TGS
    substeps, also preserves the source displacement.
    """
    velocity = np.asarray(qvel)
    if first_dof < 0 or first_dof + 6 > len(velocity):
        raise ValueError("free-joint velocity slice is outside qvel")
    linear = physx_decay(linear_rate, source_dt)
    angular = physx_decay(angular_rate, source_dt)
    velocity[first_dof : first_dof + 3] *= linear
    velocity[first_dof + 3 : first_dof + 6] *= angular
    return {
        "linear_velocity_factor": linear,
        "angular_velocity_factor": angular,
        "source_timestep_s": float(source_dt),
    }


def apply_physx_frame_start_gravity(
    model: Any, data: Any, mujoco_module: Any, gravity: np.ndarray, *,
    source_dt: float = SOURCE_TIMESTEP_S,
) -> dict[str, object]:
    """Apply PhysX TGS's default once-per-frame gravity impulse.

    PhysX TGS uses one temporal substep per position iteration, but its default
    scene flag applies gravity/external forces once at the beginning of the
    full simulation frame.  MuJoCo normally applies gravity during every small
    step.  Isolate only gravity's generalized acceleration, apply the full
    source-frame impulse, and leave model gravity disabled for the caller's TGS
    substeps.  Contacts are disabled in both probe evaluations, so an existing
    contact constraint cannot leak into the gravity impulse.
    """
    gravity_value = np.asarray(gravity, dtype=np.float64)
    source_dt = float(source_dt)
    if (
        gravity_value.shape != (3,)
        or not np.isfinite(gravity_value).all()
        or not np.isfinite(source_dt)
        or source_dt <= 0.0
    ):
        raise ValueError("frame-start gravity contract is invalid")
    flags_before = int(model.opt.disableflags)
    velocity_before = data.qvel.copy()
    try:
        model.opt.disableflags = (
            flags_before
            | int(mujoco_module.mjtDisableBit.mjDSBL_CONTACT)
        )
        model.opt.gravity[:] = gravity_value
        mujoco_module.mj_forward(model, data)
        acceleration_with_gravity = data.qacc.copy()
        model.opt.gravity[:] = 0.0
        mujoco_module.mj_forward(model, data)
        gravity_acceleration = acceleration_with_gravity - data.qacc
        data.qvel[:] += gravity_acceleration * source_dt
    except Exception:
        data.qvel[:] = velocity_before
        model.opt.gravity[:] = gravity_value
        raise
    finally:
        model.opt.disableflags = flags_before
    return {
        "method": "generalized gravity difference with contacts disabled",
        "source_timestep_s": source_dt,
        "gravity_m_s2": gravity_value.tolist(),
        "gravity_disabled_during_tgs_substeps": True,
        "generalized_velocity_delta": (
            data.qvel - velocity_before
        ).tolist(),
    }


def _one_step_factor(
    backend: Any, qpos: np.ndarray, dof: int, damping: float,
) -> tuple[float, float]:
    model, mj = backend.model, backend.mujoco
    old = float(model.dof_damping[dof])
    old_gravity = model.opt.gravity.copy()
    old_flags = int(model.opt.disableflags)
    try:
        model.dof_damping[dof] = float(damping)
        model.opt.gravity[:] = 0.0
        model.opt.disableflags = (
            old_flags
            | int(mj.mjtDisableBit.mjDSBL_CONTACT)
            | int(mj.mjtDisableBit.mjDSBL_ACTUATION)
        )
        data = mj.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.qvel[dof] = 1.0
        mj.mj_forward(model, data)
        mj.mj_step(model, data)
        factor = float(data.qvel[dof])
        base = int(model.jnt_dofadr[backend.object_joint_id])
        object_velocity = data.qvel[base : base + 6].copy()
        cross = float(np.linalg.norm(np.delete(object_velocity, dof - base)))
    finally:
        model.dof_damping[dof] = old
        model.opt.gravity[:] = old_gravity
        model.opt.disableflags = old_flags
    return factor, cross


def calibrate_free_damping(
    backend: Any, qpos: np.ndarray, *, linear_rate: float, angular_rate: float,
) -> dict[str, object]:
    """Fit six MuJoCo damping values to the two PhysX decay rates."""
    model = backend.model
    state = np.asarray(qpos, dtype=np.float64)
    if state.shape != (model.nq,) or not np.isfinite(state).all():
        raise ValueError("damping calibration qpos does not match the MuJoCo model")
    dt = float(model.opt.timestep)
    joint_dof = int(model.jnt_dofadr[backend.object_joint_id])
    targets = [physx_decay(linear_rate, dt)] * 3 + [physx_decay(angular_rate, dt)] * 3
    if min(targets) <= 0.0:
        raise ValueError("SAPIEN damping reaches zero in one step and has no finite equivalent")

    # Use the generalized mass diagonal only to bracket the numerical fit.  The
    # final values come from the actual MuJoCo step, including its quaternion
    # and implicit-integrator conventions.
    data = backend.mujoco.MjData(model)
    data.qpos[:] = state
    backend.mujoco.mj_forward(model, data)
    dense = np.zeros((model.nv, model.nv), dtype=np.float64)
    backend.mujoco.mj_fullM(model, dense, data.qM)

    fitted: list[float] = []
    measured: list[float] = []
    cross_terms: list[float] = []
    for axis, target in enumerate(targets):
        dof = joint_dof + axis
        rate = linear_rate if axis < 3 else angular_rate
        guess = scalar_viscous_damping(float(dense[dof, dof]), rate, dt)
        low, high = 0.0, max(guess, np.finfo(np.float64).eps)
        factor, _ = _one_step_factor(backend, state, dof, high)
        for _ in range(32):
            if factor <= target:
                break
            high *= 2.0
            factor, _ = _one_step_factor(backend, state, dof, high)
        else:
            raise RuntimeError(f"could not bracket object damping for free-joint axis {axis}")
        for _ in range(52):
            middle = 0.5 * (low + high)
            factor, _ = _one_step_factor(backend, state, dof, middle)
            if factor > target:
                low = middle
            else:
                high = middle
        value = 0.5 * (low + high)
        factor, cross = _one_step_factor(backend, state, dof, value)
        fitted.append(value)
        measured.append(factor)
        cross_terms.append(cross)

    model.dof_damping[joint_dof : joint_dof + 6] = np.asarray(fitted)
    return {
        "method": "numerical one-step free-decay match in compiled MuJoCo model",
        "physics_dt_s": dt,
        "physx_linear_rate_s_inv": float(linear_rate),
        "physx_angular_rate_s_inv": float(angular_rate),
        "target_velocity_factors": targets,
        "mujoco_free_joint_damping": fitted,
        "measured_velocity_factors": measured,
        "max_abs_factor_error": float(np.max(np.abs(np.asarray(measured) - targets))),
        "max_cross_axis_velocity": float(max(cross_terms)),
    }


def apply_profile(
    backend: Any, metrics: dict[str, object], initial_qpos: np.ndarray,
) -> dict[str, object]:
    """Apply the SAPIEN-equivalent object and implicit drive profile in memory."""
    if backend.physics_substeps != 0 or not np.isclose(backend.data.time, 0.0):
        raise RuntimeError("SAPIEN equivalence must be applied before simulation")
    body = read_body(metrics)
    model, mj = backend.model, backend.mujoco
    model.opt.integrator = int(mj.mjtIntegrator.mjINT_IMPLICITFAST)
    model.opt.timestep = 1.0 / 240.0

    body_id = backend.object_body_id
    model.body_mass[body_id] = body.mass
    model.body_inertia[body_id] = body.inertia
    model.body_ipos[body_id] = body.com_pos
    model.body_iquat[body_id] = body.com_quat / np.linalg.norm(body.com_quat)
    model.body_sameframe[body_id] = int(mj.mjtSameFrame.mjSAMEFRAME_NONE)
    mj.mj_setConst(model, backend.data)
    mj.mj_resetData(model, backend.data)

    kp = 1000.0
    source_kd = 100.0
    uses_tgs_substeps = np.isclose(
        float(model.opt.timestep), MUJOCO_TGS_SUBSTEP_S, atol=1.0e-15,
    )
    target_kd = (
        source_kd if uses_tgs_substeps
        else implicit_drive_kd(kp, source_kd, float(model.opt.timestep))
    )
    driven: list[str] = []
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint) or ""
        if not (name.startswith("right_hand_") or "forearm" in name):
            continue
        model.actuator_gaintype[actuator] = int(mj.mjtGain.mjGAIN_FIXED)
        model.actuator_gainprm[actuator, :] = 0.0
        model.actuator_gainprm[actuator, 0] = kp
        model.actuator_biastype[actuator] = int(mj.mjtBias.mjBIAS_AFFINE)
        model.actuator_biasprm[actuator, :] = 0.0
        model.actuator_biasprm[actuator, 1] = -kp
        model.actuator_biasprm[actuator, 2] = -target_kd
        model.actuator_ctrllimited[actuator] = 0
        model.actuator_forcelimited[actuator] = 0
        driven.append(name)
    if len(driven) != 18:
        raise ValueError(f"SAPIEN-equivalent drive expected 18 actuators, found {len(driven)}")

    if uses_tgs_substeps:
        first_dof = int(model.jnt_dofadr[backend.object_joint_id])
        model.dof_damping[first_dof : first_dof + 6] = 0.0
        damping = {
            "method": "measured PhysX full-step velocity multiplier",
            "application": "once before every 1/240 s source step",
            "mujoco_substeps_per_source_step": PHYSX_TGS_POSITION_ITERATIONS,
            "linear_velocity_factor": physx_decay(
                body.linear_decay_rate, SOURCE_TIMESTEP_S,
            ),
            "angular_velocity_factor": physx_decay(
                body.angular_decay_rate, SOURCE_TIMESTEP_S,
            ),
            "mujoco_free_joint_damping": [0.0] * 6,
        }
    else:
        damping = calibrate_free_damping(
            backend, initial_qpos,
            linear_rate=body.linear_decay_rate,
            angular_rate=body.angular_decay_rate,
        )
    mj.mj_resetData(model, backend.data)
    return {
        "name": "sapien_equivalent_v1",
        "object": {
            "mass_kg": body.mass,
            "inertia_kg_m2": body.inertia.tolist(),
            "com_pos_m": body.com_pos.tolist(),
            "com_quat_wxyz": body.com_quat.tolist(),
            "source": "recorded SAPIEN runtime values; no density-based rescaling",
        },
        "damping": damping,
        "drive": {
            "source_physx_kp": kp,
            "source_physx_kd": source_kd,
            "mujoco_kp": kp,
            "mujoco_kd": target_kd,
            "conversion": (
                "same force gains over 25 temporal substeps matching PhysX TGS"
                if uses_tgs_substeps
                else "kd_mujoco = kd_physx + dt * kp_physx"
            ),
            "driven_actuators": driven,
            "control_limits_disabled": True,
            "force_limits_disabled": True,
            "scope": "scalar one-step implicit-drive equivalence",
        },
        "model_constants_refreshed_after_runtime_edits": True,
    }


def contact_profile(body: SapienBody, dt: float, scale: float = 1.0) -> dict[str, float]:
    """Derive a MuJoCo acceleration-level contact profile from SAPIEN evidence."""
    dt, scale = float(dt), float(scale)
    depth = max(abs(body.min_separation), 5.0e-5)
    peak_force = body.max_impulse / dt
    stiffness = scale * peak_force / (body.mass * depth)
    damping = 2.0 * np.sqrt(stiffness)
    if not np.isfinite((stiffness, damping)).all() or stiffness <= 0.0 or scale <= 0.0:
        raise ValueError("derived contact profile is not finite and positive")
    return {
        "accel_stiffness_s_inv2": float(stiffness),
        "accel_damping_s_inv": float(damping),
        "stiffness_scale": scale,
        "source_peak_impulse_ns": body.max_impulse,
        "source_peak_force_n": peak_force,
        "source_min_separation_m": body.min_separation,
        "effective_mass_approximation_kg": body.mass,
    }
