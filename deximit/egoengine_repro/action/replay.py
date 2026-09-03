"""Deterministic bounded-effort MuJoCo execution backend."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .geometry import body_subtree_ids, physical_object_geom_ids
from .types import BackendObservation, SimulationState


def _xhand_joint_effort_limit(joint_name: str) -> float | None:
    """Return the published XHand 6-DoF URDF effort for one finger joint.

    The SPIDER torque scene carries these same values in motor ``ctrlrange``;
    retaining this named contract here makes a stale scene with a uniform
    multi-Nm torque range fail before it can generate a false grasp result.
    """
    suffixes = {
        "hand_thumb_bend_joint": 1.1,
        "hand_thumb_rota_joint1": 1.1,
        "hand_thumb_rota_joint2": 0.4,
        "hand_index_bend_joint": 0.4,
        "hand_index_joint1": 1.1,
        "hand_index_joint2": 0.4,
        "hand_mid_joint1": 1.1,
        "hand_mid_joint2": 0.4,
        "hand_ring_joint1": 1.1,
        "hand_ring_joint2": 0.4,
        "hand_pinky_joint1": 1.1,
        "hand_pinky_joint2": 0.4,
    }
    return next((limit for suffix, limit in suffixes.items() if joint_name.endswith(suffix)), None)


def direct_object_actuator_names(
    model: object, mujoco: object, *, object_body_id: int | None = None,
) -> tuple[str, ...]:
    """Return actuators that directly control a simulated object.

    Such controls invalidate a physical-grasp experiment because an object can
    follow its GT pose without being carried by the hand.
    """
    result: list[str] = []
    object_bodies = (
        body_subtree_ids(model, int(object_body_id))
        if object_body_id is not None else frozenset()
    )
    for actuator in range(int(model.nu)):
        actuator_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) or ""
        )
        transmission = (
            int(np.asarray(model.actuator_trntype)[actuator])
            if hasattr(model, "actuator_trntype") else None
        )
        joint_types = {
            int(value) for value in (
                getattr(getattr(mujoco, "mjtTrn", object()), "mjTRN_JOINT", 0),
                getattr(getattr(mujoco, "mjtTrn", object()), "mjTRN_JOINTINPARENT", 1),
            )
        }
        joint_backed = transmission is None or transmission in joint_types
        joint = (
            int(np.asarray(model.actuator_trnid)[actuator, 0])
            if joint_backed and hasattr(model, "actuator_trnid") else -1
        )
        joint_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
            if joint >= 0 and hasattr(mujoco.mjtObj, "mjOBJ_JOINT") else ""
        )
        joint_body = (
            int(model.jnt_bodyid[joint])
            if joint >= 0 and hasattr(model, "jnt_bodyid") else -1
        )
        exact_object_body = joint_body in object_bodies
        if exact_object_body:
            result.append(actuator_name or f"actuator#{actuator}")
            continue
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, joint_body) or ""
            if joint_body >= 0 and hasattr(mujoco.mjtObj, "mjOBJ_BODY") else ""
        )
        named_object_target = any(
            "object" in value.lower()
            for value in (actuator_name, joint_name, body_name)
        )
        if named_object_target:
            result.append(actuator_name or f"actuator#{actuator}")
    return tuple(result)


def require_free_object_joint(
    model: object, mujoco: object, joint_name: str,
) -> tuple[int, int]:
    """Resolve the one allowed object representation: a named free joint."""
    joint_id = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name),
    ))
    if joint_id < 0:
        raise ValueError(f"free object joint not found: {joint_name}")
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"object joint is not free: {joint_name}")
    return joint_id, int(model.jnt_bodyid[joint_id])


def forbidden_object_constraint_names(
    model: object, mujoco: object, *, object_joint_id: int, object_body_id: int,
) -> tuple[str, ...]:
    """Return equality constraints which can externally hold the free object."""
    result: list[str] = []
    equalities = int(getattr(model, "neq", 0))
    if not equalities:
        return ()
    connect = int(getattr(mujoco.mjtEq, "mjEQ_CONNECT", -1001))
    weld = int(getattr(mujoco.mjtEq, "mjEQ_WELD", -1002))
    joint = int(getattr(mujoco.mjtEq, "mjEQ_JOINT", -1003))
    object_bodies = body_subtree_ids(model, object_body_id)
    object_joints = {
        joint_id for joint_id in range(int(getattr(model, "njnt", 0)))
        if int(model.jnt_bodyid[joint_id]) in object_bodies
    } if hasattr(model, "jnt_bodyid") else {int(object_joint_id)}
    object_joints.add(int(object_joint_id))
    # Any geom rigidly attached below the object root can constrain the free
    # body through a distance equality, including a renderer-only visual geom.
    # Physics gap metrics still use physical_object_geom_ids separately.
    object_geoms = {
        geom for geom in range(int(getattr(model, "ngeom", 0)))
        if int(model.geom_bodyid[geom]) in object_bodies
    } if hasattr(model, "geom_bodyid") else set()
    distance = int(getattr(mujoco.mjtEq, "mjEQ_DISTANCE", -1004))
    for equality in range(equalities):
        kind = int(model.eq_type[equality])
        first, second = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
        holds_object = (
            (kind in {connect, weld} and bool(object_bodies & {first, second}))
            or (kind == joint and bool(object_joints & {first, second}))
            or (kind == distance and bool(object_geoms & {first, second}))
        )
        if holds_object:
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, equality,
            ) or f"equality#{equality}"
            result.append(name)
    return tuple(result)


def validate_physics_trace(
    model: object, mujoco: object, *, qpos: np.ndarray, qvel: np.ndarray,
    ctrl: np.ndarray, time_s: np.ndarray, tolerance: float = 1.0e-10,
) -> dict[str, float]:
    """Replay every saved control and reject a state sequence it cannot produce.

    Trace control at state ``i`` is the control that advanced state ``i-1`` to
    ``i``.  This is the convention used by both active physics producers.  A
    content-consistent replay closes the loophole where qpos (including the
    object pose) and self-reported provenance flags could be edited together.
    """
    positions = np.asarray(qpos, dtype=np.float64)
    velocities = np.asarray(qvel, dtype=np.float64)
    controls = np.asarray(ctrl, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    limit = float(tolerance)
    count = len(positions)
    if (
        positions.shape != (count, int(model.nq)) or count < 1
        or velocities.shape != (count, int(model.nv))
        or controls.shape != (count, int(model.nu)) or times.shape != (count,)
        or not np.isfinite(positions).all() or not np.isfinite(velocities).all()
        or not np.isfinite(controls).all() or not np.isfinite(times).all()
        or not np.isfinite(limit) or limit <= 0.0
    ):
        raise ValueError("physics trace arrays or replay tolerance are malformed")
    data = mujoco.MjData(model)
    data.qpos[:] = positions[0]
    data.qvel[:] = velocities[0]
    data.ctrl[:] = controls[0]
    data.time = float(times[0])
    mujoco.mj_forward(model, data)
    qpos_error = qvel_error = time_error = 0.0
    for state in range(1, count):
        data.ctrl[:] = controls[state]
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        qpos_error = max(qpos_error, float(np.max(np.abs(data.qpos - positions[state]))))
        qvel_error = max(qvel_error, float(np.max(np.abs(data.qvel - velocities[state]))))
        time_error = max(time_error, abs(float(data.time) - float(times[state])))
    errors = {
        "qpos": qpos_error,
        "qvel": qvel_error,
        "time_s": time_error,
    }
    if max(errors.values()) > limit:
        raise ValueError(
            "saved physics trace is not reproduced by its controls: "
            f"max_abs_error={errors}"
        )
    return errors


class MujocoReplayBackend:
    """Execute reference controls in one MuJoCo world without optimization."""

    def __init__(
        self, model_path: str | Path, *, object_joint_name: str = "right_object_joint",
        hand_order: Sequence[str] = ("right",), fingertip_geom_prefix: str = "collision_hand_",
        seed: int = 0, object_actuator_gain: float | None = None,
        finger_impedance: tuple[float, float, float] | None = None,
        arm_impedance: tuple[float, float, float] | None = None,
    ):
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("MujocoReplayBackend requires mujoco") from error
        self.mujoco = mujoco
        self.seed = int(seed)
        np.random.seed(self.seed)
        self.hand_order = tuple(str(side) for side in hand_order)
        unsupported = set(self.hand_order) - {"left", "right"}
        if unsupported or not self.hand_order or len(set(self.hand_order)) != len(self.hand_order):
            raise ValueError(f"unsupported or duplicate hand sides: {self.hand_order}")
        hand_prefixes = tuple(f"{side}_hand_" for side in self.hand_order)
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        joint_transmissions = {
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        }
        if any(
            int(value) not in joint_transmissions
            for value in np.asarray(self.model.actuator_trntype).tolist()
        ):
            raise ValueError("MujocoReplayBackend supports only joint-backed actuators")
        self.object_joint_id, self.object_body_id = require_free_object_joint(
            self.model, mujoco, object_joint_name,
        )
        self.object_qpos_address = int(
            self.model.jnt_qposadr[self.object_joint_id],
        )
        object_constraints = forbidden_object_constraint_names(
            self.model, mujoco, object_joint_id=self.object_joint_id,
            object_body_id=self.object_body_id,
        )
        if object_constraints:
            raise ValueError(
                "free object is externally constrained by a forbidden equality: "
                f"{object_constraints}"
            )
        if object_actuator_gain is not None and float(object_actuator_gain) != 0.0:
            raise ValueError(
                "direct object target-pose actuation is forbidden in this project: "
                "GT may be used only as an observation, reference, or evaluation target"
            )
        direct_object_actuators = direct_object_actuator_names(
            self.model, mujoco, object_body_id=self.object_body_id,
        )
        if direct_object_actuators:
            raise ValueError(
                "model has direct object actuators and is forbidden for physical-grasp "
                f"evaluation: {direct_object_actuators}. Use the free-object scene.xml."
            )
        self.finger_torque_ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self.arm_torque_ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self._finger_torque_bounds: dict[int, tuple[float, float]] = {}
        self.finger_impedance = finger_impedance
        self.arm_impedance = arm_impedance
        if finger_impedance is not None:
            kp, kv, limit = (float(v) for v in finger_impedance)
            if not np.isfinite((kp, kv, limit)).all() or min(kp, kv, limit) <= 0.0:
                raise ValueError("finger impedance values must be finite and positive")
            self._finger_kp, self._finger_kv, self._finger_limit = kp, kv, limit
            finger_ids = []
            for aid in range(self.model.nu):
                joint = int(self.model.actuator_trnid[aid, 0])
                joint_name = (
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
                )
                if joint_name.startswith(hand_prefixes):
                    if (
                        int(self.model.actuator_gaintype[aid])
                        != int(mujoco.mjtGain.mjGAIN_FIXED)
                        or int(self.model.actuator_biastype[aid])
                        != int(mujoco.mjtBias.mjBIAS_NONE)
                        or int(self.model.actuator_dyntype[aid])
                        != int(mujoco.mjtDyn.mjDYN_NONE)
                    ):
                        raise ValueError(
                            f"XHand impedance requires a direct torque motor: {joint_name}"
                        )
                    expected_effort = _xhand_joint_effort_limit(joint_name)
                    if expected_effort is None:
                        raise ValueError(f"unknown XHand finger effort contract: {joint_name}")
                    if not bool(self.model.actuator_ctrllimited[aid]):
                        raise ValueError(
                            f"XHand torque actuator {joint_name} lacks a bounded ctrlrange; "
                            "regenerate scene_torque.xml from the URDF effort limits"
                        )
                    low, high = (float(value) for value in self.model.actuator_ctrlrange[aid])
                    if low >= 0.0 or high <= 0.0:
                        raise ValueError(
                            f"XHand torque actuator {joint_name} has invalid ctrlrange [{low:g}, {high:g}]; "
                            "regenerate scene_torque.xml from the URDF effort limits"
                        )
                    if low < -expected_effort - 1e-6 or high > expected_effort + 1e-6:
                        raise ValueError(
                            f"XHand torque actuator {joint_name} exceeds URDF effort ±{expected_effort:g} "
                            f"with ctrlrange [{low:g}, {high:g}]; do not use a uniform torque limit"
                        )
                    self._finger_torque_bounds[aid] = (low, high)
                    finger_ids.append(aid)
            self.finger_torque_ids = np.asarray(finger_ids, dtype=np.int64)
            if len(self.finger_torque_ids) != 12 * len(self.hand_order):
                raise ValueError(
                    "XHand impedance requires exactly 12 named torque motors per selected hand"
                )
            if len(self.finger_torque_ids):
                print(
                    f"[replay] impedance-controlled torque fingers: "
                    f"{len(self.finger_torque_ids)} actuators, kp={kp:g} kv={kv:g} "
                    f"safety_cap={limit:g}; motor ctrlrange is the per-joint physical cap",
                    flush=True,
                )
        if arm_impedance is not None:
            kp, kv, limit = (float(v) for v in arm_impedance)
            if not np.isfinite((kp, kv, limit)).all() or min(kp, kv, limit) <= 0.0:
                raise ValueError("arm impedance values must be finite and positive")
            self._arm_kp, self._arm_kv, self._arm_limit = kp, kv, limit
            arm_ids = []
            for aid in range(self.model.nu):
                joint = int(self.model.actuator_trnid[aid, 0])
                joint_name = (
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
                )
                if "forearm" in joint_name and int(self.model.actuator_gaintype[aid]) == 0:
                    arm_ids.append(aid)
            self.arm_torque_ids = np.asarray(arm_ids, dtype=np.int64)
            if len(self.arm_torque_ids):
                print(
                    f"[replay] impedance-controlled torque forearm: "
                    f"{len(self.arm_torque_ids)} actuators, kp={kp:g} kv={kv:g} "
                    f"limit={limit:g}",
                    flush=True,
                )
        self.palm_site_ids = []
        for side in self.hand_order:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm")
            if site_id < 0:
                raise ValueError(f"palm site not found: {side}_palm")
            self.palm_site_ids.append(int(site_id))
        self.hand_joint_qpos_addresses = np.asarray([
            int(self.model.jnt_qposadr[joint])
            for joint in range(self.model.njnt)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint) or "").startswith(hand_prefixes)
        ], dtype=np.int64)
        actuator_joint_ids = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int64)
        if (actuator_joint_ids < 0).any():
            raise ValueError("MujocoReplayBackend supports only joint-backed actuators")
        self.actuator_qpos_addresses = np.asarray(
            [int(self.model.jnt_qposadr[joint]) for joint in actuator_joint_ids], dtype=np.int64,
        )
        self.actuator_qvel_addresses = np.asarray(
            [int(self.model.jnt_dofadr[joint]) for joint in actuator_joint_ids], dtype=np.int64,
        )
        self.object_geom_ids = physical_object_geom_ids(
            self.model, mujoco, self.object_body_id,
        )
        collision_prefixes = tuple(
            f"{fingertip_geom_prefix}{side}_" for side in self.hand_order
        )
        self.hand_geom_names = {}
        for geom in range(self.model.ngeom):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom,
            ) or ""
            if name.startswith(collision_prefixes):
                self.hand_geom_names[geom] = name
        self._simulation_cost = 0
        self._physics_substeps = 0

    @property
    def action_dimension(self) -> int:
        return int(self.model.nu)

    @property
    def simulation_cost(self) -> int:
        return self._simulation_cost

    @property
    def physics_substeps(self) -> int:
        return self._physics_substeps

    @property
    def object_height(self) -> float:
        return float(self.data.xpos[self.object_body_id, 2])

    def initialize(self, qpos: np.ndarray, qvel: np.ndarray) -> SimulationState:
        positions = np.asarray(qpos, dtype=np.float64)
        velocities = np.asarray(qvel, dtype=np.float64)
        if (
            positions.shape != (self.model.nq,) or velocities.shape != (self.model.nv,)
            or not np.isfinite(positions).all() or not np.isfinite(velocities).all()
        ):
            raise ValueError("initial qpos/qvel shape does not match MuJoCo model")
        self.data.qpos[:] = positions
        self.data.qvel[:] = velocities
        self._apply_control(self.reference_action(positions))
        self.data.time = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return self.snapshot()

    def restore(self, state: SimulationState) -> None:
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        if state.ctrl is not None:
            self.data.ctrl[:] = state.ctrl
        self.data.time = state.time
        self.mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> SimulationState:
        return SimulationState(
            qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(), time=float(self.data.time),
            ctrl=self.data.ctrl.copy(),
        )

    def reference_action(self, qpos: np.ndarray) -> np.ndarray:
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (self.model.nq,):
            raise ValueError(f"reference qpos expected shape ({self.model.nq},), got {values.shape}")
        return values[self.actuator_qpos_addresses].copy()

    def _apply_control(
        self, control: np.ndarray, *, finger_torque_scale: np.ndarray | None = None,
    ) -> None:
        """Write position targets, converting torque fingers to bounded impedance.

        Each XHand torque is clamped by both the caller's optional safety cap
        and its named URDF effort range encoded in the motor ctrlrange.  A
        source-timed contact-maintenance controller may additionally tighten
        the torque envelope of *already touching* finger motors.  It cannot
        increase any motor's declared effort, affect a wrist motor, or create
        an actuator for the free object.
        """
        local_scale = None
        if finger_torque_scale is not None:
            local_scale = np.asarray(finger_torque_scale, dtype=np.float64)
            if (
                local_scale.shape != (self.model.nu,)
                or not np.isfinite(local_scale).all()
                or (local_scale < 0.0).any() or (local_scale > 1.0).any()
            ):
                raise ValueError("finger torque scale must be finite in [0, 1] per actuator")
            nonfinger = np.ones(self.model.nu, dtype=bool)
            nonfinger[self.finger_torque_ids] = False
            if not np.allclose(local_scale[nonfinger], 1.0, atol=0.0, rtol=0.0):
                raise ValueError("finger torque scale must leave non-finger actuators unchanged")
        self.data.ctrl[:] = control
        for aid in self.finger_torque_ids.tolist():
            joint = int(self.model.actuator_trnid[aid, 0])
            qpos_address = int(self.model.jnt_qposadr[joint])
            dof_address = int(self.model.jnt_dofadr[joint])
            target = float(control[aid])
            current = float(self.data.qpos[qpos_address])
            velocity = float(self.data.qvel[dof_address])
            torque = self._finger_kp * (target - current) - self._finger_kv * velocity
            lower, upper = self._finger_torque_bounds.get(
                aid, (-self._finger_limit, self._finger_limit),
            )
            lower = max(lower, -self._finger_limit)
            upper = min(upper, self._finger_limit)
            if local_scale is not None:
                scale = float(local_scale[aid])
                lower *= scale
                upper *= scale
            self.data.ctrl[aid] = float(np.clip(torque, lower, upper))
        for aid in self.arm_torque_ids.tolist():
            joint = int(self.model.actuator_trnid[aid, 0])
            qpos_address = int(self.model.jnt_qposadr[joint])
            dof_address = int(self.model.jnt_dofadr[joint])
            target = float(control[aid])
            current = float(self.data.qpos[qpos_address])
            velocity = float(self.data.qvel[dof_address])
            torque = self._arm_kp * (target - current) - self._arm_kv * velocity
            self.data.ctrl[aid] = float(np.clip(torque, -self._arm_limit, self._arm_limit))

    def reference_hand_joints(self, qpos: np.ndarray) -> np.ndarray:
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (self.model.nq,):
            raise ValueError(f"reference qpos expected shape ({self.model.nq},), got {values.shape}")
        return values[self.hand_joint_qpos_addresses].copy()

    def _contacts(self) -> tuple[bool, bool]:
        thumb, other = False, False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            # MuJoCo may retain broad-phase contacts in ``mjData.contact``
            # while excluding them from the constraint solver (for example
            # when a positive gap is used).  Such an entry has no physical
            # contact force and must not release a contact-dependent policy.
            if int(contact.efc_address) < 0:
                continue
            pair = ((int(contact.geom1), int(contact.geom2)), (int(contact.geom2), int(contact.geom1)))
            for hand_geom, object_geom in pair:
                if object_geom not in self.object_geom_ids or hand_geom not in self.hand_geom_names:
                    continue
                name = self.hand_geom_names[hand_geom]
                thumb |= "thumb" in name
                other |= any(finger in name for finger in ("index", "middle", "ring", "pinky"))
        return thumb, other

    def step(
        self, action: np.ndarray, duration_s: float,
        *, finger_torque_scale: np.ndarray | None = None,
    ) -> BackendObservation:
        control = np.asarray(action, dtype=np.float64)
        duration = float(duration_s)
        if control.shape != (self.model.nu,) or not np.isfinite(control).all():
            raise ValueError(f"action expected shape ({self.model.nu},), got {control.shape}")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("step duration must be finite and positive")
        if finger_torque_scale is not None:
            # Validate once before the physics loop.  _apply_control repeats
            # the check defensively because it is also used by initialize.
            scale = np.asarray(finger_torque_scale, dtype=np.float64)
            if scale.shape != (self.model.nu,):
                raise ValueError("finger torque scale shape does not match the actuator layout")
        count = max(1, int(round(duration / float(self.model.opt.timestep))))
        for _ in range(count):
            # ``control`` carries desired joint positions, while XHand fingers
            # are torque motors driven by a PD impedance law.  Re-evaluate that
            # law every MuJoCo substep from the *live* qpos/qvel; computing it
            # only once per 0.1--0.2 s MPC segment silently turns a position
            # target into an unrealistically stale constant torque.
            self._apply_control(control, finger_torque_scale=finger_torque_scale)
            self.mujoco.mj_step(self.model, self.data)
        # ``mj_step`` integrates qpos/qvel after its position-dependent
        # computations.  Without a final forward pass, xpos/site_xpos/contact
        # describe the pre-integration state while the returned qpos describes
        # the post-integration state: a silent one-physics-step contact skew.
        # mj_forward refreshes derived observations without writing state.
        self.mujoco.mj_forward(self.model, self.data)
        self._simulation_cost += 1
        self._physics_substeps += count
        object_pose = np.eye(4, dtype=np.float64)
        object_pose[:3, 3] = self.data.xpos[self.object_body_id]
        object_pose[:3, :3] = self.data.xmat[self.object_body_id].reshape(3, 3)
        wrist_poses = np.repeat(np.eye(4)[None], len(self.palm_site_ids), axis=0)
        for index, site_id in enumerate(self.palm_site_ids):
            wrist_poses[index, :3, 3] = self.data.site_xpos[site_id]
            wrist_poses[index, :3, :3] = self.data.site_xmat[site_id].reshape(3, 3)
        thumb, other = self._contacts()
        return BackendObservation(
            object_pose=object_pose, wrist_poses=wrist_poses,
            hand_joint_positions=self.reference_hand_joints(self.data.qpos),
            thumb_contact=thumb, other_finger_contact=other,
        )
