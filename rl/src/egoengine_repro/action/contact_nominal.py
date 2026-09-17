"""GT-contact-guided, robot-only nominal correction.

This module deliberately builds a *proposal* for later free-object sampling.
It never controls an object.  GT contact positions are expressed in the GT
object frame and softly guide matching XHand tip sites plus named collision
geometry; collision, joint-range and temporal penalties keep the proposal
close to the incoming dynamics-legal nominal.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .contracts import (
    XHAND_FINGER_ORDER,
    XHAND_PRECONTACT_CLEARANCE_M,
    precontact_clearance_mask,
)
from .geometry import explicit_collision_pairs, minimum_pair_distance
from .replay import MujocoReplayBackend


FINGERS = XHAND_FINGER_ORDER
# MuJoCo site-frame columns used as the distal approach axis.  The thumb site
# uses +Y; the other XHand tip sites use -Z in their native local frame.
FINGER_DIRECTION_AXES = ((0, 1.0), (2, -1.0), (2, -1.0), (2, -1.0), (2, -1.0))


@dataclass(frozen=True)
class ContactNominalConfig:
    """Bounded soft-contact correction settings in physical units."""

    lookahead_frames: int = 6
    iterations_per_frame: int = 8
    wrist_translation_cap_m: float = 0.012
    wrist_rotation_cap_rad: float = 0.10
    finger_cap_rad: float = 0.20
    finger_joint_limit_margin_rad: float = 0.005
    # Match the upstream strict-reference collision validation tolerance.
    # This nominal is a sampler seed, so it must approach rather than tunnel
    # through the free object.
    contact_penetration_cap_m: float = 0.00005
    self_floor_penetration_cap_m: float = 0.00005
    # Human fingertip coordinates are not a robot-mesh contact point.  Keep
    # that Cartesian signal weak and let the GT per-digit event additionally
    # reduce the real XHand-geometry/object gap toward a small positive
    # clearance.  This is still only a proposal, never a force-closure claim.
    tip_target_weight: float = 0.04
    # Align the distal approach axis with the projected object normal only
    # while contact evidence is active.  This is a fixed diagnostic setting,
    # not a tuned production weight.
    normal_contact_weight: float = 0.20
    geometry_contact_weight: float = 3.0
    geometry_target_clearance_m: float = 0.0001
    precontact_finger_clearance_m: float = XHAND_PRECONTACT_CLEARANCE_M
    geometry_finite_difference: float = 1.0e-4
    baseline_weight: float = 0.08
    temporal_weight: float = 0.12
    # Diagnostic-only mode for references that already contain penetration.
    # It allows a candidate to inherit an existing violation, but never to
    # make a collision family worse than the incoming reference.  Keep this
    # disabled for any nominal intended for downstream physics use.
    preserve_baseline_violations: bool = False
    baseline_preservation_tolerance_m: float = 1.0e-6

    def validate(self) -> None:
        values = np.asarray((
            self.wrist_translation_cap_m, self.wrist_rotation_cap_rad,
            self.finger_cap_rad, self.finger_joint_limit_margin_rad,
            self.contact_penetration_cap_m, self.self_floor_penetration_cap_m,
            self.tip_target_weight, self.geometry_contact_weight,
            self.normal_contact_weight,
            self.geometry_target_clearance_m, self.precontact_finger_clearance_m,
            self.geometry_finite_difference, self.baseline_weight,
            self.temporal_weight, self.baseline_preservation_tolerance_m,
        ), dtype=np.float64)
        if (
            self.lookahead_frames < 1
            or self.iterations_per_frame < 1
            or not np.isfinite(values).all()
            or min(
                self.wrist_translation_cap_m, self.wrist_rotation_cap_rad,
                self.finger_cap_rad, self.finger_joint_limit_margin_rad,
                self.contact_penetration_cap_m,
                self.self_floor_penetration_cap_m,
                self.tip_target_weight, self.geometry_contact_weight,
                self.normal_contact_weight,
                self.geometry_target_clearance_m,
                self.precontact_finger_clearance_m,
                self.geometry_finite_difference,
            ) <= 0.0
            or min(self.baseline_weight, self.temporal_weight) < 0.0
            or self.baseline_preservation_tolerance_m < 0.0
        ):
            raise ValueError("contact nominal configuration is invalid")


@dataclass(frozen=True)
class ContactNominalResult:
    qpos: np.ndarray
    target_world_m: np.ndarray
    target_object_local_m: np.ndarray
    target_weight: np.ndarray
    target_source_frame: np.ndarray
    tip_error_before_m: np.ndarray
    tip_error_after_m: np.ndarray
    geom_gap_before_m: np.ndarray
    geom_gap_after_m: np.ndarray
    correction_qpos: np.ndarray


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def contact_target_schedule(
    contact: np.ndarray, contact_positions_world_m: np.ndarray,
    object_transforms_world: np.ndarray, *, lookahead_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build smooth object-local fingertip targets from GT contact events.

    The pre-contact lookahead only ramps a *hand* target in.  It does not move
    the object and it does not assert that the resulting XHand pose is already
    a physical grasp.
    """
    mask = np.asarray(contact, dtype=bool)
    points = np.asarray(contact_positions_world_m, dtype=np.float64)
    transforms = np.asarray(object_transforms_world, dtype=np.float64)
    if (
        mask.ndim != 2 or mask.shape[1] != len(FINGERS)
        or points.shape != (*mask.shape, 3)
        or transforms.shape != (len(mask), 4, 4)
        or not np.isfinite(points).all()
        or not np.isfinite(transforms).all()
        or lookahead_frames < 1
    ):
        raise ValueError("GT contact positions, mask and object transforms are misaligned")
    count = len(mask)
    local = np.full((count, len(FINGERS), 3), np.nan, dtype=np.float64)
    world = np.full_like(local, np.nan)
    weight = np.zeros((count, len(FINGERS)), dtype=np.float64)
    source = np.full((count, len(FINGERS)), -1, dtype=np.int64)
    for finger in range(len(FINGERS)):
        active = np.flatnonzero(mask[:, finger])
        for frame in range(count):
            future = active[active >= frame]
            if not len(future) or future[0] - frame > lookahead_frames:
                continue
            anchor = int(future[0])
            # Full target weight at a marked GT contact frame; ramp only the
            # lead-in.  A later sampler remains responsible for real closure.
            alpha = 1.0 if anchor == frame else _smoothstep(
                1.0 - (anchor - frame) / float(lookahead_frames),
            )
            if alpha <= 0.0:
                continue
            homogeneous = np.append(points[anchor, finger], 1.0)
            anchor_local = np.linalg.inv(transforms[anchor]) @ homogeneous
            local[frame, finger] = anchor_local[:3]
            world[frame, finger] = (
                transforms[frame] @ np.append(anchor_local[:3], 1.0)
            )[:3]
            weight[frame, finger] = alpha
            source[frame, finger] = anchor
    return local, world, weight, source


def _actuated_robot_layout(
    backend: MujocoReplayBackend, config: ContactNominalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return robot qpos/dof addresses, physical scales and joint bounds."""
    model, mj = backend.model, backend.mujoco
    # The MuJoCo model contains both hands' actuators even when this helper is
    # solving one side.  Select by the named hand prefix so a left-hand solve
    # cannot accidentally move the right hand (or vice versa).
    prefixes = tuple(
        prefix
        for side in backend.hand_order
        for prefix in (
            f"{side}_hand_",
            "R_forearm_" if side == "right" else "L_forearm_",
        )
    )
    actuator_ids = np.asarray([
        actuator for actuator in range(model.nu)
        if (mj.mj_id2name(
            model, mj.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[actuator, 0]),
        ) or "").startswith(prefixes)
    ], dtype=np.int64)
    qpos_addresses = backend.actuator_qpos_addresses[actuator_ids].astype(np.int64)
    dof_addresses = backend.actuator_qvel_addresses[actuator_ids].astype(np.int64)
    if len(qpos_addresses) != 18 or len(np.unique(qpos_addresses)) != 18:
        raise ValueError(
            "contact nominal requires exactly 18 unique actuators for the selected hand"
        )
    if backend.object_qpos_address in set(qpos_addresses.tolist()):
        raise ValueError("contact nominal must not expose an object actuator")
    scales, lower, upper = [], [], []
    for actuator, address in zip(actuator_ids.tolist(), qpos_addresses.tolist()):
        joint = int(model.actuator_trnid[actuator, 0])
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint) or ""
        if "forearm" in name:
            scales.append(
                config.wrist_translation_cap_m
                if any(token in name for token in ("_tx_", "_ty_", "_tz_"))
                else config.wrist_rotation_cap_rad,
            )
        else:
            scales.append(config.finger_cap_rad)
        if bool(model.jnt_limited[joint]):
            low, high = (float(value) for value in model.jnt_range[joint])
            if "forearm" not in name:
                low += config.finger_joint_limit_margin_rad
                high -= config.finger_joint_limit_margin_rad
                if low >= high:
                    raise ValueError(f"finger joint range has no usable interior: {name}")
        else:
            low, high = -np.inf, np.inf
        lower.append(low)
        upper.append(high)
        if int(model.jnt_qposadr[joint]) != address:
            raise ValueError("actuator qpos address does not match its joint")
    return (
        qpos_addresses,
        dof_addresses,
        np.asarray(scales, dtype=np.float64),
        np.asarray([lower, upper], dtype=np.float64),
    )


def _finger_geometry_gaps(backend: MujocoReplayBackend) -> np.ndarray:
    model, data, mj = backend.model, backend.data, backend.mujoco
    fromto = np.empty(6, dtype=np.float64)
    result = np.full(len(FINGERS), np.inf, dtype=np.float64)
    object_geoms = sorted(backend.object_geom_ids)
    for index, finger in enumerate(FINGERS):
        hand_geoms = [
            geom for geom, name in backend.hand_geom_names.items()
            if f"_{finger}_" in name
        ]
        result[index] = min(float(mj.mj_geomDistance(
            model, data, hand, obj, 1.0, fromto,
        )) for hand in hand_geoms for obj in object_geoms)
    return result


def _tip_positions(backend: MujocoReplayBackend, tip_sites: np.ndarray) -> np.ndarray:
    return np.asarray(backend.data.site_xpos[tip_sites], dtype=np.float64).copy()


def _tip_axis(
    backend: MujocoReplayBackend, tip_sites: np.ndarray, finger: int,
) -> np.ndarray:
    column, sign = FINGER_DIRECTION_AXES[int(finger)]
    rotation = backend.data.site_xmat[int(tip_sites[int(finger)])].reshape(3, 3)
    axis = rotation[:, int(column)] * float(sign)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis).all() or norm <= 1.0e-12:
        raise ValueError("XHand fingertip axis is invalid")
    return axis / norm


def _geometry_gap_jacobian(
    backend: MujocoReplayBackend, candidate_qpos: np.ndarray,
    active_fingers: np.ndarray, qpos_addresses: np.ndarray, scales: np.ndarray,
    *, finite_difference: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference the measured collision-geometry gap.

    A fingertip site is only a kinematic proxy and may not sit on the capsule
    which reaches the object first.  Differentiating the actual named
    hand-object distances therefore gives the soft nominal a useful, but
    deliberately local, geometry relation.  The returned Jacobian is with
    respect to normalized bounded correction coordinates.
    """
    backend.data.qpos[:] = candidate_qpos
    backend.data.qvel[:] = 0.0
    backend.mujoco.mj_forward(backend.model, backend.data)
    baseline = _finger_geometry_gaps(backend)[active_fingers]
    jacobian = np.zeros((len(active_fingers), len(qpos_addresses)), dtype=np.float64)
    for column, address in enumerate(qpos_addresses.tolist()):
        step = min(float(finite_difference), 0.05 * float(scales[column]))
        if step <= 0.0:
            continue
        trial = candidate_qpos.copy()
        trial[address] += step
        backend.data.qpos[:] = trial
        backend.data.qvel[:] = 0.0
        backend.mujoco.mj_forward(backend.model, backend.data)
        jacobian[:, column] = (
            _finger_geometry_gaps(backend)[active_fingers] - baseline
        ) / step * scales[column]
    backend.data.qpos[:] = candidate_qpos
    backend.data.qvel[:] = 0.0
    backend.mujoco.mj_forward(backend.model, backend.data)
    return baseline, jacobian


def _candidate_is_safe(
    backend: MujocoReplayBackend, candidate_qpos: np.ndarray,
    pairs: dict[str, tuple[tuple[int, int], ...]], config: ContactNominalConfig,
    *, noncontact_fingers: np.ndarray | None = None,
    baseline_minima: dict[str, float] | None = None,
    baseline_finger_gaps: np.ndarray | None = None,
) -> bool:
    backend.data.qpos[:] = candidate_qpos
    backend.data.qvel[:] = 0.0
    backend.mujoco.mj_forward(backend.model, backend.data)
    minima = {
        "self": minimum_pair_distance(
            backend.model, backend.data, backend.mujoco, pairs["self"],
        ),
        "floor": minimum_pair_distance(
            backend.model, backend.data, backend.mujoco, pairs["floor"],
        ),
        "object": minimum_pair_distance(
            backend.model, backend.data, backend.mujoco, pairs["object"],
        ),
    }
    if config.preserve_baseline_violations:
        if baseline_minima is None or set(baseline_minima) != set(minima):
            raise ValueError(
                "baseline collision minima are required when preserving violations"
            )
        tolerance = config.baseline_preservation_tolerance_m
        legal = all(
            minima[name] >= float(baseline_minima[name]) - tolerance
            for name in minima
        )
    else:
        legal = bool(
            minima["self"] >= -config.self_floor_penetration_cap_m
            and minima["floor"] >= -config.self_floor_penetration_cap_m
            and minima["object"] >= -config.contact_penetration_cap_m
        )
    if not legal or noncontact_fingers is None or not len(noncontact_fingers):
        return legal
    gaps = _finger_geometry_gaps(backend)[noncontact_fingers]
    if config.preserve_baseline_violations:
        if baseline_finger_gaps is None:
            raise ValueError(
                "baseline finger gaps are required when preserving violations"
            )
        return bool(np.all(
            gaps >= baseline_finger_gaps[noncontact_fingers]
            - config.baseline_preservation_tolerance_m
        ))
    return bool(np.all(
        gaps >= (
            config.precontact_finger_clearance_m
            - config.contact_penetration_cap_m
        )
    ))


def _candidate_objective(
    backend: MujocoReplayBackend, candidate_qpos: np.ndarray,
    *, tip_sites: np.ndarray, scheduled_fingers: np.ndarray,
    contact_fingers: np.ndarray, target_world_m: np.ndarray,
    target_weight: np.ndarray, base_robot_qpos: np.ndarray,
    previous_delta: np.ndarray, qpos_addresses: np.ndarray,
    scales: np.ndarray, config: ContactNominalConfig,
    target_normals_world_m: np.ndarray | None = None,
) -> float:
    """Evaluate exactly the nonlinear least-squares objective for line search."""
    backend.data.qpos[:] = candidate_qpos
    backend.data.qvel[:] = 0.0
    backend.mujoco.mj_forward(backend.model, backend.data)
    cost = 0.0
    if len(scheduled_fingers):
        errors = (
            target_world_m[scheduled_fingers]
            - _tip_positions(backend, tip_sites)[scheduled_fingers]
        )
        cost += float(np.sum(
            target_weight[scheduled_fingers, None]
            * config.tip_target_weight * errors * errors
        ))
        if target_normals_world_m is not None and config.normal_contact_weight:
            axis_errors = np.asarray([
                -target_normals_world_m[finger] - _tip_axis(backend, tip_sites, int(finger))
                for finger in scheduled_fingers.tolist()
            ], dtype=np.float64)
            cost += float(np.sum(
                target_weight[scheduled_fingers, None]
                * config.normal_contact_weight * axis_errors * axis_errors
            ))
    if len(contact_fingers):
        gaps = _finger_geometry_gaps(backend)[contact_fingers]
        errors = np.minimum(
            config.geometry_target_clearance_m - gaps, 0.0,
        )
        cost += float(config.geometry_contact_weight * np.dot(errors, errors))
    current = (
        candidate_qpos[qpos_addresses] - base_robot_qpos
    ) / scales
    if config.baseline_weight:
        cost += float(config.baseline_weight * np.dot(current, current))
    if config.temporal_weight:
        temporal = current - previous_delta / scales
        cost += float(config.temporal_weight * np.dot(temporal, temporal))
    return cost


def improve_nominal_with_gt_contact(
    backend: MujocoReplayBackend, qpos: np.ndarray, *, contact: np.ndarray,
    contact_positions_world_m: np.ndarray, object_transforms_world: np.ndarray,
    contact_normals_world_m: np.ndarray | None = None,
    contact_state: np.ndarray | None = None,
    config: ContactNominalConfig = ContactNominalConfig(),
) -> ContactNominalResult:
    """Softly correct an existing nominal without touching its object qpos.

    The only optimization variables are 6 robot wrist joints and 12 XHand
    finger joints.  Contact targets are intentionally soft; the caller must
    still run dynamics screening and later free-object sampling.
    """
    config.validate()
    base = np.asarray(qpos, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != backend.model.nq or not np.isfinite(base).all():
        raise ValueError("contact nominal requires finite qpos with the runtime scene width")
    contact_mask = np.asarray(contact, dtype=bool)
    normals = None if contact_normals_world_m is None else np.asarray(
        contact_normals_world_m, dtype=np.float64,
    )
    if normals is not None and (
        normals.shape != (*contact_mask.shape, 3)
        or not np.isfinite(normals).all()
    ):
        raise ValueError("contact normals must align with the contact mask")
    if normals is not None:
        norms = np.linalg.norm(normals, axis=-1)
        if np.any(norms <= 1.0e-12) or not np.allclose(norms, 1.0, atol=1.0e-5):
            raise ValueError("contact normals must be finite unit vectors")
    local, world, weight, source = contact_target_schedule(
        contact_mask, contact_positions_world_m, object_transforms_world,
        lookahead_frames=config.lookahead_frames,
    )
    if len(base) != len(weight):
        raise ValueError("contact targets do not align with nominal qpos")
    qpos_addresses, dof_addresses, scales, limits = _actuated_robot_layout(
        backend, config,
    )
    lower, upper = limits
    pairs = explicit_collision_pairs(
        backend.model, backend.mujoco, object_geom_ids=backend.object_geom_ids,
        hand_sides=backend.hand_order,
    )
    side = backend.hand_order[0]
    tip_sites = np.asarray([
        backend.mujoco.mj_name2id(
            backend.model, backend.mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_tip",
        ) for finger in FINGERS
    ], dtype=np.int64)
    if (tip_sites < 0).any():
        raise ValueError("contact nominal scene lacks an XHand fingertip site")
    corrected = base.copy()
    target_error_before = np.full((len(base), len(FINGERS)), np.nan, dtype=np.float64)
    target_error_after = np.full_like(target_error_before, np.nan)
    gap_before = np.empty_like(target_error_before)
    gap_after = np.empty_like(target_error_before)
    previous_delta = np.zeros(len(qpos_addresses), dtype=np.float64)
    precontact_required = precontact_clearance_mask(contact_mask, contact_state)

    for frame in range(len(base)):
        active = np.flatnonzero(weight[frame] > 0.0)
        physical_active = np.flatnonzero(contact_mask[frame])
        noncontact = np.flatnonzero(precontact_required[frame])
        # Capture the incoming reference once.  In diagnostic mode this is
        # the floor against which every trial at this frame is compared.
        backend.data.qpos[:] = base[frame]
        backend.data.qvel[:] = 0.0
        backend.mujoco.mj_forward(backend.model, backend.data)
        baseline_minima = {
            name: minimum_pair_distance(
                backend.model, backend.data, backend.mujoco, pairs[name],
            ) for name in ("self", "floor", "object")
        }
        baseline_finger_gaps = _finger_geometry_gaps(backend)
        # Carry a residual through nominal motion while slowly returning to the
        # original trajectory when no GT contact signal remains.
        seed_delta = previous_delta * (0.96 if len(active) else 0.70)
        candidate = base[frame].copy()
        candidate[qpos_addresses] = np.clip(
            candidate[qpos_addresses] + seed_delta,
            np.maximum(lower, base[frame, qpos_addresses] - scales),
            np.minimum(upper, base[frame, qpos_addresses] + scales),
        )
        if not _candidate_is_safe(
            backend, candidate, pairs, config,
            noncontact_fingers=noncontact,
            baseline_minima=baseline_minima,
            baseline_finger_gaps=baseline_finger_gaps,
        ):
            # A correction must not make the incoming reference worse merely
            # because the preceding frame carried a residual.
            candidate = base[frame].copy()
        backend.data.qpos[:] = base[frame]
        backend.data.qvel[:] = 0.0
        backend.mujoco.mj_forward(backend.model, backend.data)
        initial_tip = _tip_positions(backend, tip_sites)
        gap_before[frame] = _finger_geometry_gaps(backend)
        if len(active):
            target_error_before[frame, active] = np.linalg.norm(
                initial_tip[active] - world[frame, active], axis=1,
            )

        for _ in range(config.iterations_per_frame if len(active) else 1):
            backend.data.qpos[:] = candidate
            backend.data.qvel[:] = 0.0
            backend.mujoco.mj_forward(backend.model, backend.data)
            if not len(active):
                break
            tip = _tip_positions(backend, tip_sites)
            residual_rows, jacobian_rows = [], []
            for finger in active.tolist():
                jacobian_position = np.zeros((3, backend.model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, backend.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(
                    backend.model, backend.data, jacobian_position, jacobian_rotation,
                    int(tip_sites[finger]),
                )
                row_weight = float(np.sqrt(
                    weight[frame, finger] * config.tip_target_weight,
                ))
                residual_rows.append(row_weight * (world[frame, finger] - tip[finger]))
                jacobian_rows.append(
                    row_weight * jacobian_position[:, dof_addresses] * scales[None],
                )
                if normals is not None and config.normal_contact_weight:
                    axis = _tip_axis(backend, tip_sites, finger)
                    target_axis = -normals[frame, finger]
                    axis_jacobian = np.cross(
                        jacobian_rotation[:, dof_addresses].T,
                        axis[None, :],
                    ).T
                    normal_weight = float(np.sqrt(
                        weight[frame, finger] * config.normal_contact_weight,
                    ))
                    residual_rows.append(normal_weight * (target_axis - axis))
                    jacobian_rows.append(
                        normal_weight * axis_jacobian * scales[None],
                    )
            gaps, gap_jacobian = _geometry_gap_jacobian(
                backend, candidate, physical_active, qpos_addresses, scales,
                finite_difference=config.geometry_finite_difference,
            )
            for index, finger in enumerate(physical_active.tolist()):
                gap_residual = (
                    config.geometry_target_clearance_m - gaps[index]
                )
                if gap_residual >= 0.0:
                    continue
                geometry_weight = float(np.sqrt(
                    config.geometry_contact_weight,
                ))
                residual_rows.append(geometry_weight * np.asarray([gap_residual]))
                jacobian_rows.append(
                    geometry_weight * gap_jacobian[index : index + 1]
                )
            current_normalized = (
                candidate[qpos_addresses] - base[frame, qpos_addresses]
            ) / scales
            previous_normalized = previous_delta / scales
            if config.baseline_weight:
                jacobian_rows.append(np.sqrt(config.baseline_weight) * np.eye(len(scales)))
                residual_rows.append(-np.sqrt(config.baseline_weight) * current_normalized)
            if config.temporal_weight:
                jacobian_rows.append(np.sqrt(config.temporal_weight) * np.eye(len(scales)))
                residual_rows.append(np.sqrt(config.temporal_weight) * (
                    previous_normalized - current_normalized
                ))
            matrix = np.vstack(jacobian_rows)
            vector = np.concatenate(residual_rows)
            normalized_step, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
            physical_step = np.clip(normalized_step * scales, -0.35 * scales, 0.35 * scales)
            if float(np.linalg.norm(physical_step)) < 1.0e-7:
                break
            current_objective = _candidate_objective(
                backend, candidate, tip_sites=tip_sites,
                scheduled_fingers=active, contact_fingers=physical_active,
                target_world_m=world[frame], target_weight=weight[frame],
                base_robot_qpos=base[frame, qpos_addresses],
                previous_delta=previous_delta,
                qpos_addresses=qpos_addresses, scales=scales, config=config,
                target_normals_world_m=None if normals is None else normals[frame],
            )
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = candidate.copy()
                trial[qpos_addresses] = np.clip(
                    trial[qpos_addresses] + scale * physical_step,
                    np.maximum(lower, base[frame, qpos_addresses] - scales),
                    np.minimum(upper, base[frame, qpos_addresses] + scales),
                )
                # The object block is copied from the source nominal every
                # time: this assertion catches accidental object correction.
                object = backend.object_qpos_address
                trial[object : object + 7] = base[frame, object : object + 7]
                if (
                    _candidate_is_safe(
                        backend, trial, pairs, config,
                        noncontact_fingers=noncontact,
                        baseline_minima=baseline_minima,
                        baseline_finger_gaps=baseline_finger_gaps,
                    )
                    and _candidate_objective(
                        backend, trial, tip_sites=tip_sites,
                        scheduled_fingers=active,
                        contact_fingers=physical_active,
                        target_world_m=world[frame],
                        target_weight=weight[frame],
                        base_robot_qpos=base[frame, qpos_addresses],
                        previous_delta=previous_delta,
                        qpos_addresses=qpos_addresses, scales=scales,
                        config=config,
                        target_normals_world_m=None if normals is None else normals[frame],
                    ) < current_objective - 1.0e-15
                ):
                    candidate = trial
                    accepted = True
                    break
            if not accepted:
                break
        corrected[frame] = candidate
        object = backend.object_qpos_address
        if not np.array_equal(
            corrected[frame, object : object + 7], base[frame, object : object + 7],
        ):
            raise AssertionError("contact nominal attempted to modify an object qpos")
        backend.data.qpos[:] = candidate
        backend.data.qvel[:] = 0.0
        backend.mujoco.mj_forward(backend.model, backend.data)
        final_tip = _tip_positions(backend, tip_sites)
        gap_after[frame] = _finger_geometry_gaps(backend)
        if len(active):
            target_error_after[frame, active] = np.linalg.norm(
                final_tip[active] - world[frame, active], axis=1,
            )
        previous_delta = candidate[qpos_addresses] - base[frame, qpos_addresses]

    return ContactNominalResult(
        qpos=corrected,
        target_world_m=world,
        target_object_local_m=local,
        target_weight=weight,
        target_source_frame=source,
        tip_error_before_m=target_error_before,
        tip_error_after_m=target_error_after,
        geom_gap_before_m=gap_before,
        geom_gap_after_m=gap_after,
        correction_qpos=corrected[:, qpos_addresses] - base[:, qpos_addresses],
    )
