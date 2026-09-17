"""Small, shared contracts for the active source-timed XHand pipeline.

This module deliberately contains no optimizer or renderer.  It is the one
place where the active pipeline defines controller defaults and the mapping
between a physics trace and source-video rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation


SOURCE_TIMED_TRACE_SCHEMA = "xhand_source_timed_free_object_trace_v4"
# Version the *controller semantics* separately from the trace container.  A
# qpos trace can satisfy the same row/time schema while having been generated
# by a materially different controller.  V2 means a per-interval C1 Hermite
# robot guide whose inverse-dynamics feedforward includes M(q) qacc.
SOURCE_TIMED_CONTROL_SCHEMA = "xhand_c1_inverse_dynamics_acceleration_v2"
CONTACT_REGION_SCHEMA = "xhand_contact_regions_v4"
RESIDUAL_SEARCH_SCHEMA = "xhand_residual_search_v9"
# The values may come from oracle GT in the present 3.2 upper-bound test or
# from a future 3.1 estimator, but they must remain an upstream *human-hand*
# reference.  A retargeted XHand projection is a downstream result and is not
# interchangeable with this evidence.
HUMAN_CONTACT_POSITION_SOURCE = "upstream_human_reference_T_sim_fingertip_target_v1"
CONTACT_REFERENCE_SCHEMA_V4 = "xhand_gt_contact_reference_v4"
CONTACT_REFERENCE_SURFACE_SCHEMA_V5 = "xhand_surface_contact_reference_v5"
CONTACT_STATE_UNKNOWN = np.int8(-1)
CONTACT_STATE_FREE = np.int8(0)
CONTACT_STATE_CONTACT = np.int8(1)
XHAND_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
XHAND_NOMINAL_EXECUTION_TIME_SCALE = 3.0
# A 1 mm kinematic endpoint clearance was smaller than the observed finite-
# effort tracking/path error and allowed hidden inter-frame impacts.  Five mm
# is the measured minimum that removed pre-evidence contacts on smear071; it is
# a controller safety margin, not a claimed human-skin contact threshold.
XHAND_PRECONTACT_CLEARANCE_M = 0.005
# Self-collision and hand--floor contact are never part of the demonstrated
# grasp.  Keep only a tiny numerical tolerance for distance queries; unlike a
# hand--object contact, neither family may use MuJoCo's millimetre-scale soft
# contact compression as an accepted state.
XHAND_SELF_FLOOR_TOLERANCE_M = 5.0e-5


def discrete_evidence_row(previous_row: int, current_row: int, phase: float) -> int:
    """Do not grant a discrete source-row observation before its endpoint.

    Robot targets may interpolate continuously between source rows.  A binary
    upstream observation such as contact, however, belongs to the previous row
    until the exact endpoint of the interval.  Centralising this rule prevents
    audits and residual scoring from acquiring opposite one-frame offsets.
    """
    previous, current = int(previous_row), int(current_row)
    value = float(phase)
    if previous < 0 or current < previous or not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("source evidence row/phase is invalid")
    return current if value >= 1.0 - 1.0e-12 else previous


def _validated_contact_state(contact: np.ndarray, state: np.ndarray | None) -> np.ndarray | None:
    """Validate the optional three-state surface evidence alongside its mask."""
    if state is None:
        return None
    evidence = np.asarray(contact, dtype=bool)
    values = np.asarray(state, dtype=np.int8)
    if (
        values.shape != evidence.shape
        or not np.isin(values, (CONTACT_STATE_UNKNOWN, CONTACT_STATE_FREE, CONTACT_STATE_CONTACT)).all()
        or not np.array_equal(evidence, values == CONTACT_STATE_CONTACT)
    ):
        raise ValueError("contact state must be UNKNOWN/FREE/CONTACT and agree with its mask")
    return values


def precontact_clearance_mask(
    contact: np.ndarray, contact_state: np.ndarray | None = None,
) -> np.ndarray:
    """Require clearance only before each digit's first contact evidence.

    The input is human fingertip proximity, not tactile truth.  Treating every
    later false sample as a hard robot release command creates a brittle phase
    machine and can destroy a valid morphology-specific support contact.
    """
    evidence = np.asarray(contact, dtype=bool)
    if evidence.ndim != 2 or not len(evidence):
        raise ValueError("contact evidence must be a nonempty row-by-digit mask")
    state = _validated_contact_state(evidence, contact_state)
    result = np.ones_like(evidence, dtype=bool) if state is None else np.zeros_like(evidence, dtype=bool)
    for digit in range(evidence.shape[1]):
        rows = np.flatnonzero(evidence[:, digit])
        if state is None and len(rows):
            result[int(rows[0]) :, digit] = False
        elif state is not None:
            # UNKNOWN is deliberately *not* evidence that the whole finger must
            # be 10 mm away.  Only an explicitly observed FREE patch before
            # the first surface-contact patch earns the clearance constraint.
            stop = int(rows[0]) if len(rows) else len(evidence)
            result[:stop, digit] = state[:stop, digit] == CONTACT_STATE_FREE
    return result


def precontact_clearance_required_at_phase(
    contact: np.ndarray, previous_row: int, current_row: int, phase: float,
    contact_state: np.ndarray | None = None,
) -> np.ndarray:
    """Hard-clear digits until the interval containing their first evidence.

    A 30 Hz binary proximity label cannot locate contact onset inside the last
    negative-to-first-positive interval.  Contact is therefore allowed in that
    one interval, while scoring still uses :func:`discrete_evidence_row` and
    cannot consume the future positive label early.  Never-observed digits
    remain guarded for the whole sequence.
    """
    mask = np.asarray(contact, dtype=bool)
    previous, current, value = int(previous_row), int(current_row), float(phase)
    if mask.ndim != 2 or not len(mask):
        raise ValueError("contact evidence must be a nonempty row-by-digit mask")
    if (
        previous < 0 or current < previous or current >= len(mask)
        or not np.isfinite(value) or not 0.0 < value <= 1.0
    ):
        raise ValueError("source-interval phase/rows are invalid")
    state = _validated_contact_state(mask, contact_state)
    source_float = (1.0 - value) * previous + value * current
    evidence_row = discrete_evidence_row(previous, current, value)
    required = np.ones(mask.shape[1], dtype=bool) if state is None else np.zeros(mask.shape[1], dtype=bool)
    for digit in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, digit])
        if len(rows):
            before_first = source_float <= float(rows[0] - 1) + 1.0e-12
        else:
            before_first = True
        if state is None:
            required[digit] = before_first
        else:
            required[digit] = bool(
                before_first and state[evidence_row, digit] == CONTACT_STATE_FREE,
            )
    return required


@dataclass(frozen=True)
class FingerImpedance:
    kp: float
    kv: float
    safety_cap: float

    def as_tuple(self) -> tuple[float, float, float]:
        values = (float(self.kp), float(self.kv), float(self.safety_cap))
        if not all(np.isfinite(values)) or min(values) <= 0.0:
            raise ValueError("finger impedance values must be finite and positive")
        return values


# Both profiles use the corrected zero-armature scene, but their targets and
# contact regimes differ.  They are named instead of hidden in runners, and
# every use remains conditional on the constant-target stability gate.
XHAND_NOMINAL_TRACKING_IMPEDANCE = FingerImpedance(2.0, 0.002, 1.1)
XHAND_DYNAMIC_CAGE_IMPEDANCE = FingerImpedance(5.0, 0.002, 1.1)


def mujoco_model_signature(model: object, mujoco: object) -> str:
    """Hash the compiled physics and rendering layout used by an active trace.

    Matching only ``nq`` or joint names is unsafe: geom transforms, mesh data,
    sites, cameras and solver options can change contact or make two panels
    visually incomparable while preserving every array shape.
    """
    digest = hashlib.sha256(b"xhand_compiled_mujoco_model_v2\n")
    digest.update(f"mujoco={getattr(mujoco, '__version__', 'unknown')}\n".encode())

    def hash_value(name: str, value: object) -> None:
        values = np.ascontiguousarray(np.asarray(value))
        if values.dtype.hasobject:
            raise TypeError(f"model signature field {name} has object dtype")
        digest.update(f"{name}:{values.dtype}:{values.shape}\n".encode())
        digest.update(values.tobytes())

    def hash_public_fields(prefix: str, owner: object, *, skip: frozenset[str] = frozenset()) -> None:
        for name in sorted(value for value in dir(owner) if not value.startswith("_")):
            if name in skip:
                continue
            value = getattr(owner, name)
            if not callable(value):
                hash_value(f"{prefix}.{name}", value)

    # Hash every public compiled scalar/array rather than maintaining a fragile
    # hand-written allowlist.  This automatically covers tendons, flexes,
    # plugins, sensors, keyframes, names, paths, and fields added by a newer
    # MuJoCo release.  Nested option/visual/stat structs are handled below.
    hash_public_fields("model", model, skip=frozenset({"opt", "vis", "stat"}))
    hash_public_fields("opt", model.opt)
    for group_name in ("global_", "headlight", "map", "quality", "rgba", "scale"):
        hash_public_fields(f"vis.{group_name}", getattr(model.vis, group_name))
    hash_public_fields("stat", model.stat)
    return digest.hexdigest()


def reference_trajectory_signature(
    frame_indices: np.ndarray, timestamps_s: np.ndarray, qpos: np.ndarray,
) -> str:
    """Hash the source-row identity and complete kinematic reference state.

    A model signature cannot distinguish two trajectories produced for the
    same scene.  Contact priors, physics traces and renderers must additionally
    bind to this value so a same-shaped artifact from another candidate or
    sample cannot be mixed in silently.
    """
    frames = np.asarray(frame_indices)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    positions = np.asarray(qpos, dtype=np.float64)
    if (
        frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer)
        or timestamps.shape != frames.shape
        or positions.ndim != 2 or positions.shape[0] != len(frames)
        or not len(frames) or not np.isfinite(timestamps).all()
        or not np.isfinite(positions).all()
        or (len(frames) > 1 and ((np.diff(frames) <= 0).any()
                                or (np.diff(timestamps) <= 0.0).any()))
    ):
        raise ValueError("reference trajectory arrays are malformed")
    digest = hashlib.sha256(b"xhand_reference_trajectory_v1\n")
    for name, values in (
        ("frame_indices", np.asarray(frames, dtype=np.int64)),
        ("timestamps_s", timestamps),
        ("qpos", positions),
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(f"{name}:{contiguous.dtype}:{contiguous.shape}\n".encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def contact_reference_signature(
    frame_indices: np.ndarray, timestamps_s: np.ndarray, side: str, contact: np.ndarray,
    contact_positions_ref_m: np.ndarray, *, finger_order: tuple[str, ...],
    episode_id: str,
    contact_distance_m: np.ndarray, contact_distance_valid: np.ndarray,
    contact_threshold_m: float, contact_definition: str,
    contact_position_source: str, human_mano_signature_value: str,
    contact_state: np.ndarray | None = None,
    contact_score: np.ndarray | None = None,
    contact_patch_pos_obj_m: np.ndarray | None = None,
    contact_patch_normal_obj: np.ndarray | None = None,
    contact_schema: str = CONTACT_REFERENCE_SCHEMA_V4,
) -> str:
    """Hash every GT contact input that changes a contact-surface prior.

    A trajectory signature only identifies the robot/object reference.  Two
    contact extractors can produce different masks or positions for that same
    trajectory while retaining identical array shapes.  Patch builders and
    consumers therefore bind the contact evidence independently.
    """
    frames = np.asarray(frame_indices)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    mask = np.asarray(contact, dtype=bool)
    positions = np.asarray(contact_positions_ref_m, dtype=np.float64)
    distance = np.asarray(contact_distance_m, dtype=np.float64)
    valid = np.asarray(contact_distance_valid, dtype=bool)
    threshold = float(contact_threshold_m)
    order = tuple(str(value) for value in finger_order)
    definition = str(contact_definition)
    position_source = str(contact_position_source)
    mano_signature = str(human_mano_signature_value)
    hand_side = str(side)
    episode = str(episode_id)
    schema = str(contact_schema)
    state = _validated_contact_state(mask, contact_state)
    surface = state is not None
    score = None if contact_score is None else np.asarray(contact_score, dtype=np.float64)
    patch = None if contact_patch_pos_obj_m is None else np.asarray(contact_patch_pos_obj_m, dtype=np.float64)
    normal = None if contact_patch_normal_obj is None else np.asarray(contact_patch_normal_obj, dtype=np.float64)
    if surface:
        if (
            schema != CONTACT_REFERENCE_SURFACE_SCHEMA_V5
            or score is None or score.shape != mask.shape or not np.isfinite(score).all()
            or (score < 0.0).any() or (score > 1.0).any()
            or patch is None or normal is None
            or patch.shape != (*mask.shape, 3) or normal.shape != (*mask.shape, 3)
            or not np.isfinite(patch[mask]).all() or not np.isfinite(normal[mask]).all()
            or not np.isnan(patch[~mask]).all() or not np.isnan(normal[~mask]).all()
            or (np.linalg.norm(normal[mask], axis=-1) <= 1.0e-10).any()
        ):
            raise ValueError("surface contact reference arrays are malformed")
    elif (
        schema != CONTACT_REFERENCE_SCHEMA_V4
        or score is not None or patch is not None or normal is not None
    ):
        raise ValueError("legacy contact reference has unexpected surface fields")
    if (
        frames.ndim != 1 or not len(frames)
        or not np.issubdtype(frames.dtype, np.integer)
        or timestamps.shape != frames.shape or not np.isfinite(timestamps).all()
        or (len(frames) > 1 and (
            (np.diff(frames) <= 0).any() or (np.diff(timestamps) <= 0.0).any()
        ))
        or mask.ndim != 2 or mask.shape[0] != len(frames)
        or positions.shape != (*mask.shape, 3)
        or distance.shape != mask.shape or valid.shape != mask.shape
        or not np.isfinite(positions).all()
        or not np.isfinite(distance[valid]).all()
        or not np.isfinite(threshold) or threshold <= 0.0
        or mask.shape[1] != len(order) or not all(order)
        or not definition or not position_source or not mano_signature
        or (not surface and not np.array_equal(mask, (distance <= threshold) & valid))
        or hand_side not in {"left", "right"} or not episode
    ):
        raise ValueError("GT contact reference arrays are malformed")
    digest = hashlib.sha256(f"{schema}\n".encode())
    digest.update(f"episode_id:{episode}\n".encode())
    digest.update(f"side:{hand_side}\n".encode())
    digest.update(f"finger_order:{','.join(order)}\n".encode())
    digest.update(f"contact_threshold_m:{threshold:.17g}\n".encode())
    digest.update(f"contact_definition:{definition}\n".encode())
    digest.update(f"contact_position_source:{position_source}\n".encode())
    digest.update(f"human_mano_signature:{mano_signature}\n".encode())
    for name, values in (
        ("frame_indices", np.asarray(frames, dtype=np.int64)),
        ("timestamps_s", timestamps),
        ("contact", mask),
        ("contact_positions_ref_m", positions),
        ("contact_distance_m", np.where(valid, distance, 0.0)),
        ("contact_distance_valid", valid),
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(f"{name}:{contiguous.dtype}:{contiguous.shape}\n".encode())
        digest.update(contiguous.tobytes())
    if surface:
        assert state is not None and score is not None and patch is not None and normal is not None
        for name, values in (
            ("contact_state", state), ("contact_score", score),
            ("contact_patch_pos_obj_m", np.where(mask[..., None], patch, 0.0)),
            ("contact_patch_normal_obj", np.where(mask[..., None], normal, 0.0)),
        ):
            contiguous = np.ascontiguousarray(values)
            digest.update(f"{name}:{contiguous.dtype}:{contiguous.shape}\n".encode())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def contact_region_signature(
    labels: np.ndarray, start_rows: np.ndarray, end_rows: np.ndarray,
    centers_obj_local_m: np.ndarray, normals_obj_local: np.ndarray,
    radii_m: np.ndarray, support_counts: np.ndarray, activation: np.ndarray,
) -> str:
    """Hash one complete contact-region representation."""
    names = np.asarray(labels).astype(str)
    starts = np.asarray(start_rows, dtype=np.int64)
    ends = np.asarray(end_rows, dtype=np.int64)
    centers = np.asarray(centers_obj_local_m, dtype=np.float64)
    normals = np.asarray(normals_obj_local, dtype=np.float64)
    radii = np.asarray(radii_m, dtype=np.float64)
    counts = np.asarray(support_counts, dtype=np.int64)
    weights = np.asarray(activation, dtype=np.float64)
    if not (
        names.ndim == 1 and len(names) >= 2 and len(set(names.tolist())) == len(names)
        and starts.shape == ends.shape == radii.shape == counts.shape == names.shape
        and centers.shape == normals.shape == (len(names), 3)
        and weights.ndim == 2 and weights.shape[0] == len(names)
        and (starts >= 0).all() and (ends >= starts).all()
        and (radii > 0.0).all() and (counts > 0).all()
        and np.isfinite(centers).all() and np.isfinite(normals).all()
        and np.isfinite(radii).all() and np.isfinite(weights).all()
        and (np.linalg.norm(normals, axis=1) > 1.0e-10).all()
        and (weights >= 0.0).all() and (weights <= 1.0).all()
    ):
        raise ValueError("contact-region arrays are malformed or labels are not unique")
    digest = hashlib.sha256(b"xhand_contact_regions_v1\n")
    for name in names.tolist():
        digest.update(str(name).encode())
        digest.update(b"\0")
    for name, values in (
        ("start_rows", starts), ("end_rows", ends), ("centers_obj_local_m", centers),
        ("normals_obj_local", normals), ("radii_m", radii),
        ("support_counts", counts), ("activation", weights),
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(f"{name}:{contiguous.dtype}:{contiguous.shape}\n".encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def residual_basis_signature(
    basis: np.ndarray, support_rows: np.ndarray, *,
    minimum_activation: float, max_normal_dot: float,
) -> str:
    """Bind a residual seed to its exact temporal/geometric support.

    A residual block is only meaningful together with the rows on which its
    blending basis is nonzero.  Matching model/contact-region signatures alone
    does not prevent a seed from a release window being reused in an opposed
    contact window.
    """
    weights = np.asarray(basis, dtype=np.float64)
    support = np.asarray(support_rows, dtype=bool)
    min_activation = float(minimum_activation)
    normal_dot = float(max_normal_dot)
    if (
        weights.ndim != 2 or not len(weights) or not weights.shape[1]
        or support.shape != (len(weights),) or not support.any()
        or not np.isfinite(weights).all() or (weights < 0.0).any()
        or (weights.sum(axis=1) > 1.0 + 1.0e-12).any()
        or not np.isfinite((min_activation, normal_dot)).all()
        or not 0.0 < min_activation <= 1.0 or not -1.0 <= normal_dot <= 1.0
    ):
        raise ValueError("residual basis/support arrays or thresholds are malformed")
    digest = hashlib.sha256(b"xhand_residual_basis_v1\n")
    for name, values in (
        ("basis", weights), ("support_rows", support),
        ("minimum_activation", np.asarray(min_activation, dtype=np.float64)),
        ("max_normal_dot", np.asarray(normal_dot, dtype=np.float64)),
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(f"{name}:{contiguous.dtype}:{contiguous.shape}\n".encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def human_mano_signature(
    frame_indices: np.ndarray, timestamps_s: np.ndarray, side: str,
    joint_positions_sim: np.ndarray,
) -> str:
    """Hash complete upstream MANO21 evidence for dense contact regions."""
    frames = np.asarray(frame_indices)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    joints = np.asarray(joint_positions_sim, dtype=np.float64)
    hand_side = str(side)
    if (
        frames.ndim != 1 or not len(frames)
        or not np.issubdtype(frames.dtype, np.integer)
        or timestamps.shape != frames.shape or not np.isfinite(timestamps).all()
        or joints.shape != (len(frames), 21, 3) or not np.isfinite(joints).all()
        or hand_side not in {"left", "right"}
        or (len(frames) > 1 and (
            (np.diff(frames) <= 0).any() or (np.diff(timestamps) <= 0.0).any()
        ))
    ):
        raise ValueError("MANO21 evidence arrays are malformed")
    digest = hashlib.sha256(b"xhand_human_mano21_v1\n")
    digest.update(f"side:{hand_side}\n".encode())
    for name, values in (
        ("frame_indices", np.asarray(frames, dtype=np.int64)),
        ("timestamps_s", timestamps),
        ("joint_positions_sim", joints),
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(f"{name}:{contiguous.dtype}:{contiguous.shape}\n".encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def rotation_matrix_to_mujoco_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to MuJoCo's scalar-first quaternion order."""
    xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    return xyzw[[3, 0, 1, 2]]


def validate_reference_timeline(
    frame_indices: np.ndarray, timestamps_s: np.ndarray, *, video_frame_count: int,
) -> None:
    """Validate the only supported real/ref timeline representation.

    ``frame_indices[row]`` is an actual decoded-video frame id.  Physics traces
    refer to ``row`` through ``reference_row_index``; the two must never be
    conflated.
    """
    frames = np.asarray(frame_indices)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if frames.ndim != 1 or timestamps.shape != frames.shape or not len(frames):
        raise ValueError("reference frame indices and timestamps must be aligned 1-D arrays")
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("reference frame indices must be integers")
    if (
        not np.isfinite(timestamps).all()
        or (np.diff(timestamps) <= 0.0).any()
        or (np.diff(frames) <= 0).any()
    ):
        raise ValueError("reference video frames and timestamps must be strictly increasing")
    if frames[0] < 0 or frames[-1] >= int(video_frame_count):
        raise ValueError("reference frame indices are outside the decoded real video")


def uniform_source_fps(timestamps_s: np.ndarray) -> int:
    """Return an integer FPS only when a constant-rate video can represent time."""
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("at least two source timestamps are required")
    cadence = np.diff(timestamps)
    nominal = float(np.median(cadence))
    tolerance = max(1.0e-9, nominal * 1.0e-5)
    if (
        not np.isfinite(cadence).all()
        or nominal <= 0.0
        or np.max(np.abs(cadence - nominal)) > tolerance
    ):
        raise ValueError("constant-FPS renderer rejects a nonuniform source timeline")
    fps_float = 1.0 / nominal
    fps = int(round(fps_float))
    if fps <= 0 or abs(fps_float - fps) > 1.0e-3:
        raise ValueError("source cadence does not correspond to an integer video FPS")
    return fps


def terminal_trace_rows(
    reference_row_index: np.ndarray, *, reference_count: int,
) -> np.ndarray:
    """Return the final physics state for every source/reference row.

    A synchronized triptych is legal only when every reference row has an
    explicit physics state.  Interpolation, event alignment, independent time
    warp and frozen source panels are intentionally unsupported here.
    """
    rows = np.asarray(reference_row_index)
    if rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("trace reference_row_index must be a nonempty integer vector")
    if (np.diff(rows) < 0).any() or rows[0] != 0 or rows[-1] != reference_count - 1:
        raise ValueError("trace must monotonically cover the full reference row range")
    selected = np.empty(reference_count, dtype=np.int64)
    for row in range(reference_count):
        matches = np.flatnonzero(rows == row)
        if not len(matches):
            raise ValueError(f"trace has no physics state for reference row {row}")
        selected[row] = int(matches[-1])
    return selected


def smoothstep_envelope(samples: np.ndarray, start: float, full: float) -> np.ndarray:
    """Bounded C1 ramp used by source-timed residual controls."""
    if not np.isfinite((start, full)).all() or full <= start:
        raise ValueError("residual full time must follow its start time")
    phase = np.clip(
        (np.asarray(samples, dtype=np.float64) - start) / (full - start), 0.0, 1.0,
    )
    return phase * phase * (3.0 - 2.0 * phase)


def longest_true_duration(flags: np.ndarray, durations_s: np.ndarray) -> float:
    """Measure a Boolean run using its real, possibly nonuniform step lengths."""
    values = np.asarray(flags, dtype=bool)
    durations = np.asarray(durations_s, dtype=np.float64)
    if (
        values.shape != durations.shape
        or not np.isfinite(durations).all()
        or np.any(durations <= 0.0)
    ):
        raise ValueError("Boolean flags and positive finite durations must align")
    longest = current = 0.0
    for value, duration in zip(values.tolist(), durations.tolist(), strict=True):
        current = current + duration if value else 0.0
        longest = max(longest, current)
    return float(longest)


def cumulative_physics_step_schedule(
    timestamps_s: np.ndarray, *, time_scale: float, physics_dt_s: float,
) -> np.ndarray:
    """Map source timestamps to cumulative physics steps without drift.

    Rounding every source interval independently loses the fractional step on
    every frame.  Rounding cumulative endpoint time makes those fractions
    cancel and bounds total timing error by half a physics step.
    """
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if (
        timestamps.ndim != 1 or not len(timestamps)
        or not np.isfinite(timestamps).all()
        or (len(timestamps) > 1 and (np.diff(timestamps) <= 0.0).any())
        or not np.isfinite((time_scale, physics_dt_s)).all()
        or min(time_scale, physics_dt_s) <= 0.0
    ):
        raise ValueError("timestamps, time scale and physics dt must be finite and positive")
    target_steps = (
        (timestamps - timestamps[0]) * float(time_scale) / float(physics_dt_s)
    )
    schedule = np.rint(target_steps).astype(np.int64)
    if schedule[0] != 0 or (len(schedule) > 1 and (np.diff(schedule) <= 0).any()):
        raise ValueError("source cadence is too fast for at least one physics step per row")
    return schedule


def expanded_source_timeline(
    timestamps_s: np.ndarray, *, time_scale: float, physics_dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand source rows/timestamps to the exact physics-state timeline.

    State zero belongs to source row zero.  Each subsequent state is the
    endpoint of one physics step and is labelled with the source row whose
    interval it is approaching.  This is the canonical layout shared by the
    nominal audit and every residual controller that claims to perturb it.
    """
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    schedule = cumulative_physics_step_schedule(
        timestamps, time_scale=time_scale, physics_dt_s=physics_dt_s,
    )
    rows: list[int] = [0]
    source_times: list[float] = [float(timestamps[0])]
    for row in range(1, len(timestamps)):
        steps = int(schedule[row] - schedule[row - 1])
        for step in range(1, steps + 1):
            phase = float(step) / float(steps)
            rows.append(row)
            source_times.append(float(
                timestamps[row - 1]
                + phase * (timestamps[row] - timestamps[row - 1])
            ))
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(source_times, dtype=np.float64),
    )


def validate_source_timed_trace_timeline(
    reference_row_index: np.ndarray, source_timestamp_s: np.ndarray,
    physics_time_s: np.ndarray, *, reference_timestamps_s: np.ndarray,
    time_scale: float, physics_dt_s: float,
) -> None:
    """Require the canonical row labels, source clock, and physics clock.

    Checking only each row's terminal timestamp still permits extra negative-
    time states or a physics clock unrelated to the declared time scale.  The
    audit, residual search, and renderer share this exact per-step contract.
    """
    rows = np.asarray(reference_row_index)
    source = np.asarray(source_timestamp_s, dtype=np.float64)
    physics = np.asarray(physics_time_s, dtype=np.float64)
    expected_rows, expected_source = expanded_source_timeline(
        reference_timestamps_s, time_scale=time_scale, physics_dt_s=physics_dt_s,
    )
    if (
        rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
        or source.shape != rows.shape or physics.shape != rows.shape
        or not np.isfinite(source).all() or not np.isfinite(physics).all()
    ):
        raise ValueError("source-timed trace clocks must be aligned finite vectors")
    if not np.array_equal(rows, expected_rows):
        raise ValueError("source-timed trace does not use the canonical physics-row schedule")
    if not np.allclose(source, expected_source, atol=1.0e-12, rtol=0.0):
        raise ValueError("source-timed trace source timestamps do not match the reference")
    expected_physics = np.arange(len(rows), dtype=np.float64) * float(physics_dt_s)
    if not np.allclose(physics, expected_physics, atol=1.0e-12, rtol=0.0):
        raise ValueError("source-timed trace does not use the canonical physics clock")
