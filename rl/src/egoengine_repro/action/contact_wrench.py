"""Object-centred contact regions and conservative frictional-wrench tests.

This module is deliberately a *filter*, not a grasp-success oracle.  It
converts GT fingertip events into regions on the collision meshes actually
used by MuJoCo, then asks two more precise questions than ``thumb + other``:

* do the observed XHand contacts match a demonstrated surface region, with
  optional robot-digit substitution; and
* can point contacts with the scene friction coefficients statically balance a
  requested object wrench under a declared normal-force cap?

The answer is only a local, polyhedral friction-cone approximation.  A
candidate still has to pass the normal free-object close/lift/hold rollout;
this code never writes an object pose or creates an object actuator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog

from .contracts import XHAND_FINGER_ORDER
from .replay import MujocoReplayBackend


FINGERS = XHAND_FINGER_ORDER
MANO_NEAR_JOINTS = (
    (1, "thumb_mcp"), (2, "thumb_pip"), (3, "thumb_dip"),
    (5, "index_mcp"), (6, "index_pip"), (7, "index_dip"),
    (9, "middle_mcp"), (10, "middle_pip"), (11, "middle_dip"),
    (13, "ring_mcp"), (14, "ring_pip"), (15, "ring_dip"),
    (17, "pinky_mcp"), (18, "pinky_pip"), (19, "pinky_dip"),
)


def _unit(vector: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{label} must be one finite 3-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-10:
        raise ValueError(f"{label} must be nonzero")
    return value / norm


def _finger_from_geom_name(name: str) -> str | None:
    """Resolve the physical XHand finger name without positional assumptions."""
    if "thumb" in name:
        return "thumb"
    if "index" in name:
        return "index"
    if "middle" in name or "mid" in name:
        return "middle"
    if "ring" in name:
        return "ring"
    if "pinky" in name:
        return "pinky"
    return None


def object_outward_normal_from_contact_frame(
    contact_frame: np.ndarray, *, object_is_geom1: bool,
) -> np.ndarray:
    """Return the physical object's outward normal from one MuJoCo contact.

    MuJoCo stores the contact-frame normal in the **first three contiguous
    elements** (the first row after a normal NumPy reshape), pointing from
    ``geom1`` toward ``geom2``.  ``[:, 0]`` is a tangent-component mixture and
    is not the contact normal.  This tiny layout detail can completely change
    a friction-cone/wrench result at a convex-hull edge.
    """
    frame = np.asarray(contact_frame, dtype=np.float64)
    if frame.shape not in ((9,), (3, 3)) or not np.isfinite(frame).all():
        raise ValueError("MuJoCo contact frame must be finite with shape (9,) or (3,3)")
    normal = frame.reshape(3, 3)[0]
    return _unit(normal if object_is_geom1 else -normal, label="MuJoCo contact normal")


@dataclass(frozen=True)
class SurfaceContactSample:
    """One GT contact projected onto a physical object collision mesh."""

    source_row: int
    finger: str
    position_object_local_m: np.ndarray
    outward_normal_object_local: np.ndarray
    confidence: float = 1.0
    projection_distance_m: float = 0.0


@dataclass(frozen=True)
class ContactRegion:
    """One demonstrated object-surface region with temporal evidence."""

    label: str
    start_row: int
    end_row: int
    center_object_local_m: np.ndarray
    outward_normal_object_local: np.ndarray
    radius_m: float
    support_count: int
    activation_by_source_row: np.ndarray | None = None

    def activation_weight_at(self, source_row: int) -> float:
        """Return the read-only demonstrated-contact weight at one source row."""
        row = int(source_row)
        if self.activation_by_source_row is None:
            return float(self.start_row <= row <= self.end_row)
        values = np.asarray(self.activation_by_source_row, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
            raise ValueError("region activation must be a finite one-dimensional probability")
        return float(values[row]) if 0 <= row < len(values) else 0.0

    def active_at(self, source_row: int) -> bool:
        return self.activation_weight_at(source_row) > 0.0


@dataclass(frozen=True)
class RobotContact:
    """One actual XHand--object contact expressed in the object's frame."""

    finger: str
    position_object_local_m: np.ndarray
    outward_normal_object_local: np.ndarray
    friction_coefficient: float
    normal_force_n: float
    hand_body_id: int | None = None
    position_world_m: np.ndarray | None = None


@dataclass(frozen=True)
class ContactModeMatch:
    """Best injective robot-contact to demonstrated-region correspondence."""

    feasible: bool
    cost: float | None
    assignments: tuple[tuple[str, str], ...]
    mean_region_excess_m: float | None
    mean_normal_alignment: float | None


@dataclass(frozen=True)
class DemonstratedOpposedSurfaceCoverage:
    """Two XHand contacts cover opposed demonstrated object-surface regions.

    This is deliberately a *geometric demonstration-consistency* predicate,
    not an opposition/force-closure/grasp claim.  It asks whether two distinct
    actual robot contacts can be assigned to two active demonstrated regions
    that are both locally close, normal-consistent, and point to opposite
    sides of the object.  Extra robot contacts are intentionally ignored: a
    morphology-compatible hand may use more fingers than the human did.
    """

    feasible: bool
    assignments: tuple[tuple[str, str], ...]
    region_indices: tuple[int, ...]
    mean_region_excess_m: float | None
    mean_normal_alignment: float | None
    demonstrated_normal_dot: float | None
    cost: float | None


@dataclass(frozen=True)
class WrenchFeasibility:
    """Result of a finite-force polyhedral-friction-cone feasibility test."""

    support_feasible: bool
    force_closure_approx: bool
    contact_count: int
    wrench_rank: int
    residual_norm: float | None
    total_normal_force_n: float | None
    minimum_friction_margin_n: float | None
    minimum_force_cap_margin_n: float | None
    minimum_required_peak_normal_force_n: float | None
    closure_interior_margin: float | None
    contact_forces_object_local_n: tuple[np.ndarray, ...]
    status: str


@dataclass(frozen=True)
class TorqueLimitedWrenchFeasibility:
    """Force-cone feasibility plus XHand finger-actuation feasibility.

    ``force_only`` is the geometric/static test.  ``finger_torque_feasible``
    additionally requires that one *same* cone-force assignment produces
    generalized torques within the named XHand torque-motor ctrlranges.  It is
    a local static capacity test, not a claim that the current PD controller
    has already realised those forces in a moving rollout.
    """

    force_only: WrenchFeasibility
    finger_torque_feasible: bool
    minimum_required_peak_normal_force_n: float | None
    minimum_finger_torque_margin_nm: float | None
    required_finger_joint_torques_nm: tuple[float, ...]
    finger_actuator_ids: tuple[int, ...]
    status: str


@dataclass(frozen=True)
class SurfaceProjection:
    position_world_m: np.ndarray
    outward_normal_world: np.ndarray
    distance_m: float
    geom_id: int


@dataclass(frozen=True)
class _CollisionMeshPart:
    geom_id: int
    mesh: object
    outward_face_normals: np.ndarray


class CollisionSurfaceProjector:
    """Nearest-surface queries over the collision hulls used by MuJoCo.

    The renderer mesh is deliberately excluded through ``object_geom_ids``.
    All transforms are read from the live MuJoCo state; the class never
    changes ``qpos`` by itself.
    """

    def __init__(self, backend: MujocoReplayBackend):
        try:
            import trimesh
        except ImportError as error:  # pragma: no cover - project dependency
            raise RuntimeError("collision surface regions require trimesh") from error

        self.backend = backend
        self._trimesh = trimesh
        parts: list[_CollisionMeshPart] = []
        for geom_id in sorted(backend.object_geom_ids):
            mesh_id = int(backend.model.geom_dataid[geom_id])
            if mesh_id < 0:
                name = backend.mujoco.mj_id2name(
                    backend.model, backend.mujoco.mjtObj.mjOBJ_GEOM, geom_id,
                ) or str(geom_id)
                raise ValueError(
                    f"physical object geom {name!r} is not a mesh; "
                    "add an analytic primitive projector before using contact regions"
                )
            vertex_start = int(backend.model.mesh_vertadr[mesh_id])
            vertex_count = int(backend.model.mesh_vertnum[mesh_id])
            face_start = int(backend.model.mesh_faceadr[mesh_id])
            face_count = int(backend.model.mesh_facenum[mesh_id])
            vertices = np.asarray(
                backend.model.mesh_vert[vertex_start : vertex_start + vertex_count],
                dtype=np.float64,
            )
            faces = np.asarray(
                backend.model.mesh_face[face_start : face_start + face_count],
                dtype=np.int64,
            )
            if (
                vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.min() < 0 or faces.max() >= len(vertices)
            ):
                raise ValueError(f"physical object mesh {mesh_id} has invalid local topology")
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            normals = np.asarray(mesh.face_normals, dtype=np.float64).copy()
            # Convex decomposition pieces should be watertight.  Do not trust
            # an arbitrary mesh winding: verify the direction just outside a
            # face and flip only when that probe is inside the piece.
            if bool(mesh.is_watertight):
                probes = np.asarray(mesh.triangles_center, dtype=np.float64) + 1.0e-4 * normals
                try:
                    normals[np.asarray(mesh.contains(probes), dtype=bool)] *= -1.0
                except Exception:
                    # The query is a validation aid; preserving trimesh's
                    # winding is safer than silently inventing a radial normal.
                    pass
            parts.append(_CollisionMeshPart(geom_id, mesh, normals))
        if not parts:
            raise ValueError("object has no physical collision meshes")
        self._parts = tuple(parts)

    def project_world(self, point_world_m: np.ndarray) -> SurfaceProjection:
        """Project a world point to the nearest physical collision surface."""
        point = np.asarray(point_world_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("surface query point must be finite with shape (3,)")
        best: SurfaceProjection | None = None
        for part in self._parts:
            rotation = self.backend.data.geom_xmat[part.geom_id].reshape(3, 3)
            translation = self.backend.data.geom_xpos[part.geom_id]
            local = rotation.T @ (point - translation)
            surface, distance, face_index = self._trimesh.proximity.closest_point(
                part.mesh, np.asarray([local]),
            )
            index = int(np.asarray(face_index, dtype=np.int64)[0])
            candidate = SurfaceProjection(
                position_world_m=rotation @ np.asarray(surface[0], dtype=np.float64) + translation,
                outward_normal_world=_unit(
                    rotation @ part.outward_face_normals[index], label="collision-mesh normal",
                ),
                distance_m=float(np.asarray(distance, dtype=np.float64)[0]),
                geom_id=part.geom_id,
            )
            if best is None or candidate.distance_m < best.distance_m:
                best = candidate
        if best is None:  # pragma: no cover - constructor rejects this
            raise RuntimeError("collision surface projection has no mesh parts")
        return best


def project_gt_contacts_to_collision_surface(
    backend: MujocoReplayBackend, reference_qpos: np.ndarray,
    contact_mask: np.ndarray, contact_positions_world_m: np.ndarray,
) -> tuple[SurfaceContactSample, ...]:
    """Project GT contacts to object-local collision regions without moving it."""
    qpos = np.asarray(reference_qpos, dtype=np.float64)
    mask = np.asarray(contact_mask, dtype=bool)
    positions = np.asarray(contact_positions_world_m, dtype=np.float64)
    if (
        qpos.ndim != 2 or qpos.shape[1] != backend.model.nq
        or mask.shape != (len(qpos), len(FINGERS))
        or positions.shape != (*mask.shape, 3)
    ):
        raise ValueError("reference qpos and GT contact arrays are not aligned")
    projector = CollisionSurfaceProjector(backend)
    samples: list[SurfaceContactSample] = []
    for row in range(len(qpos)):
        if not mask[row].any():
            continue
        backend.data.qpos[:] = qpos[row]
        backend.data.qvel[:] = 0.0
        backend.mujoco.mj_forward(backend.model, backend.data)
        object_rotation = backend.data.xmat[backend.object_body_id].reshape(3, 3)
        object_position = backend.data.xpos[backend.object_body_id]
        for finger_index in np.flatnonzero(mask[row]).tolist():
            point = positions[row, finger_index]
            if not np.isfinite(point).all():
                raise ValueError(f"GT contact point at row {row}, finger {finger_index} is non-finite")
            surface = projector.project_world(point)
            samples.append(SurfaceContactSample(
                source_row=row,
                finger=FINGERS[finger_index],
                position_object_local_m=object_rotation.T @ (
                    surface.position_world_m - object_position
                ),
                outward_normal_object_local=_unit(
                    object_rotation.T @ surface.outward_normal_world,
                    label="object-local collision normal",
                ),
                projection_distance_m=float(surface.distance_m),
            ))
    return tuple(samples)


def project_near_mano_joints_to_collision_surface(
    backend: MujocoReplayBackend, qpos: np.ndarray, joint_positions_world_m: np.ndarray,
    *, max_distance_m: float, confidence: float = 0.35,
) -> tuple[SurfaceContactSample, ...]:
    """Project only near intermediate MANO joints to collision-surface evidence.

    These are proximity observations, not GT contact labels.  Fingertips are
    intentionally excluded because the separate GT contact mask already
    records them.  The human wrist and palm are likewise not hallucinated as
    surface contact from skeletal keypoints.
    """
    reference = np.asarray(qpos, dtype=np.float64)
    joints = np.asarray(joint_positions_world_m, dtype=np.float64)
    if (
        reference.ndim != 2 or reference.shape[1] != backend.model.nq
        or joints.shape != (len(reference), 21, 3) or not np.isfinite(joints).all()
        or not np.isfinite((max_distance_m, confidence)).all()
        or max_distance_m <= 0.0 or not 0.0 < confidence <= 1.0
    ):
        raise ValueError("MANO near-surface evidence inputs are malformed")
    projector = CollisionSurfaceProjector(backend)
    samples: list[SurfaceContactSample] = []
    for row in range(len(reference)):
        backend.data.qpos[:] = reference[row]
        backend.data.qvel[:] = 0.0
        backend.mujoco.mj_forward(backend.model, backend.data)
        object_rotation = backend.data.xmat[backend.object_body_id].reshape(3, 3)
        object_position = backend.data.xpos[backend.object_body_id]
        for joint, label in MANO_NEAR_JOINTS:
            surface = projector.project_world(joints[row, joint])
            if surface.distance_m > max_distance_m:
                continue
            samples.append(SurfaceContactSample(
                source_row=row,
                finger=label,
                position_object_local_m=object_rotation.T @ (
                    surface.position_world_m - object_position
                ),
                outward_normal_object_local=_unit(
                    object_rotation.T @ surface.outward_normal_world,
                    label="object-local MANO near-surface normal",
                ),
                confidence=float(confidence),
                projection_distance_m=float(surface.distance_m),
            ))
    return tuple(samples)


def build_stable_object_contact_regions(
    samples: Iterable[SurfaceContactSample], *, source_frame_count: int,
    spatial_merge_radius_m: float = 0.018, min_normal_alignment: float = 0.85,
    min_radius_m: float = 0.004, padding_m: float = 0.002,
    temporal_taper_rows: int = 3,
) -> tuple[ContactRegion, ...]:
    """Aggregate noisy GT events into object-side, digit-agnostic regions.

    Each component is formed only from collision-surface samples that are near
    one another in the object's canonical frame and whose outward normals
    agree.  It therefore has no object semantic axis and no fixed XHand finger
    assignment.  The returned activation is a short, continuous temporal
    evidence field around the observed rows, rather than a hand-coded
    approach/transport/release state machine.

    The operation deliberately aggregates only *observed* GT contact evidence.
    It does not infer an unobserved human mesh area, alter the object, or turn
    a contact likelihood into force closure.
    """
    values = tuple(samples)
    if (
        source_frame_count < 1 or temporal_taper_rows < 0
        or not np.isfinite((
            spatial_merge_radius_m, min_normal_alignment, min_radius_m, padding_m,
        )).all()
        or spatial_merge_radius_m <= 0.0 or min_radius_m <= 0.0 or padding_m < 0.0
        or not -1.0 <= min_normal_alignment <= 1.0
    ):
        raise ValueError("stable object-contact region configuration is invalid")
    if not values:
        return ()
    positions = np.asarray([sample.position_object_local_m for sample in values], dtype=np.float64)
    normals = np.asarray([sample.outward_normal_object_local for sample in values], dtype=np.float64)
    rows = np.asarray([sample.source_row for sample in values], dtype=np.int64)
    confidence = np.asarray([sample.confidence for sample in values], dtype=np.float64)
    projection_distance = np.asarray([
        sample.projection_distance_m for sample in values
    ], dtype=np.float64)
    if (
        positions.shape != (len(values), 3) or normals.shape != positions.shape
        or not np.isfinite(positions).all() or not np.isfinite(normals).all()
        or not np.isfinite(confidence).all() or (confidence <= 0.0).any()
        or not np.isfinite(projection_distance).all() or (projection_distance < 0.0).any()
        or (rows < 0).any() or (rows >= source_frame_count).any()
    ):
        raise ValueError("stable object-contact samples are malformed or outside the source timeline")
    normals = np.asarray([_unit(normal, label="stable-region sample normal") for normal in normals])

    parent = np.arange(len(values), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for first in range(len(values)):
        for second in range(first + 1, len(values)):
            if (
                np.linalg.norm(positions[first] - positions[second]) <= spatial_merge_radius_m
                and float(np.dot(normals[first], normals[second])) >= min_normal_alignment
            ):
                union(first, second)
    components: dict[int, list[int]] = {}
    for index in range(len(values)):
        components.setdefault(find(index), []).append(index)

    output: list[ContactRegion] = []
    source_rows = np.arange(source_frame_count, dtype=np.float64)
    for component in sorted(components.values(), key=lambda item: (int(rows[item].min()), item)):
        indices = np.asarray(component, dtype=np.int64)
        weights = confidence[indices]
        center = np.average(positions[indices], axis=0, weights=weights)
        normal = _unit(
            np.average(normals[indices], axis=0, weights=weights),
            label="stable-region average normal",
        )
        distances = np.linalg.norm(positions[indices] - center, axis=1)
        radius = max(
            float(min_radius_m),
            float(np.quantile(distances, 0.90)) + float(padding_m),
        )
        evidence_rows = np.unique(rows[indices])
        evidence_strength = np.zeros(len(evidence_rows), dtype=np.float64)
        for evidence_index, evidence_row in enumerate(evidence_rows.tolist()):
            evidence_strength[evidence_index] = float(np.max(
                confidence[indices][rows[indices] == evidence_row]
            ))
        distance_to_evidence = np.abs(
            source_rows[:, None] - evidence_rows[None, :]
        )
        if temporal_taper_rows == 0:
            activation = np.max(
                (distance_to_evidence == 0.0) * evidence_strength[None, :],
                axis=1,
            )
        else:
            normalized = np.clip(
                1.0 - distance_to_evidence / float(temporal_taper_rows + 1), 0.0, 1.0,
            )
            kernels = normalized * normalized * (3.0 - 2.0 * normalized)
            activation = np.max(kernels * evidence_strength[None, :], axis=1)
        digits = tuple(sorted({values[index].finger for index in indices.tolist()}))
        output.append(ContactRegion(
            label=f"r{len(output)}[" + "+".join(digits) + "]",
            start_row=int(evidence_rows.min()),
            end_row=int(evidence_rows.max()),
            center_object_local_m=center,
            outward_normal_object_local=normal,
            radius_m=radius,
            support_count=len(indices),
            activation_by_source_row=activation,
        ))
    return tuple(output)


def _contacts_by_link(contacts: Iterable[RobotContact]) -> tuple[RobotContact, ...]:
    """Keep one strongest solver contact per physical hand link.

    A digit can touch an object with more than one phalanx.  Collapsing all of
    them to one contact silently removes real moment arms and can turn a valid
    wrench into a false negative.  MuJoCo body ids distinguish those links;
    synthetic callers without body ids retain the older one-per-digit rule.
    """
    best: dict[tuple[str, int | None], RobotContact] = {}
    for contact in contacts:
        if contact.finger not in FINGERS:
            raise ValueError(f"unsupported robot contact digit {contact.finger!r}")
        if (
            contact.position_object_local_m.shape != (3,)
            or contact.outward_normal_object_local.shape != (3,)
            or not np.isfinite(contact.position_object_local_m).all()
            or not np.isfinite(contact.outward_normal_object_local).all()
            or not np.isfinite((contact.friction_coefficient, contact.normal_force_n)).all()
            or contact.friction_coefficient < 0.0 or contact.normal_force_n < 0.0
        ):
            raise ValueError("robot contact data is malformed")
        if contact.hand_body_id is not None and int(contact.hand_body_id) < 0:
            raise ValueError("robot contact hand body id must be nonnegative")
        if contact.position_world_m is not None:
            world = np.asarray(contact.position_world_m, dtype=np.float64)
            if world.shape != (3,) or not np.isfinite(world).all():
                raise ValueError("robot contact world position is malformed")
        key = (contact.finger, contact.hand_body_id)
        previous = best.get(key)
        if previous is None or contact.normal_force_n > previous.normal_force_n:
            best[key] = contact
    order = {finger: index for index, finger in enumerate(FINGERS)}
    return tuple(
        contact for _, contact in sorted(
            best.items(), key=lambda item: (
                order[item[0][0]], -1 if item[0][1] is None else item[0][1],
            ),
        )
    )


def extract_live_hand_contacts(
    backend: MujocoReplayBackend, *, side: str,
) -> tuple[RobotContact, ...]:
    """Read actual hand--object contacts from the current free-object state.

    GT regions use collision-mesh normals, but a live force/wrench computation
    must use the normal MuJoCo actually gave its contact solver.  A nearest
    triangle can be a neighbouring facet at a convex-decomposition edge and
    must never replace this physical normal.
    """
    prefix = f"collision_hand_{side}_"
    object_rotation = backend.data.xmat[backend.object_body_id].reshape(3, 3)
    object_position = backend.data.xpos[backend.object_body_id]
    force = np.empty(6, dtype=np.float64)
    contacts: list[RobotContact] = []
    for contact_id in range(backend.data.ncon):
        contact = backend.data.contact[contact_id]
        if int(contact.efc_address) < 0:
            continue
        first, second = int(contact.geom1), int(contact.geom2)
        object_is_first = first in backend.object_geom_ids
        if not object_is_first and second not in backend.object_geom_ids:
            continue
        hand_geom = second if object_is_first else first
        name = backend.hand_geom_names.get(hand_geom, "")
        if not name.startswith(prefix):
            continue
        finger = _finger_from_geom_name(name)
        if finger is None:
            continue
        backend.mujoco.mj_contactForce(backend.model, backend.data, contact_id, force)
        outward_world = object_outward_normal_from_contact_frame(
            contact.frame, object_is_geom1=object_is_first,
        )
        contacts.append(RobotContact(
            finger=finger,
            position_object_local_m=object_rotation.T @ (contact.pos - object_position),
            outward_normal_object_local=_unit(
                object_rotation.T @ outward_world, label="live object contact normal",
            ),
            friction_coefficient=float(contact.friction[0]),
            normal_force_n=abs(float(force[0])),
            hand_body_id=int(backend.model.geom_bodyid[hand_geom]),
            position_world_m=np.asarray(contact.pos, dtype=np.float64).copy(),
        ))
    return _contacts_by_link(contacts)


def finger_object_gaps_m(backend: MujocoReplayBackend, *, side: str) -> np.ndarray:
    """Return named-geometry distance to the physical object for every digit.

    This is only a continuous *approach* signal for the sampler.  A zero or
    negative gap is never relabelled as force closure; that requires live
    contacts and ``evaluate_frictional_wrench`` separately.
    """
    fromto = np.empty(6, dtype=np.float64)
    result = np.full(len(FINGERS), np.inf, dtype=np.float64)
    prefix = f"collision_hand_{side}_"
    for finger_index, finger in enumerate(FINGERS):
        hand_geoms = [
            geom for geom, name in backend.hand_geom_names.items()
            if name.startswith(prefix) and _finger_from_geom_name(name) == finger
        ]
        if not hand_geoms:
            raise ValueError(f"runtime scene has no named collision geometry for {side} {finger}")
        result[finger_index] = min(
            float(backend.mujoco.mj_geomDistance(
                backend.model, backend.data, hand_geom, object_geom, 0.05, fromto,
            ))
            for hand_geom in hand_geoms for object_geom in backend.object_geom_ids
        )
    return result


def match_contact_mode(
    robot_contacts: Sequence[RobotContact], regions: Sequence[ContactRegion],
    *, allow_digit_substitution: bool = True, normal_weight: float = 0.25,
) -> ContactModeMatch:
    """Match actual robot contacts to demonstrated regions injectively."""
    if not np.isfinite(normal_weight) or normal_weight < 0.0:
        raise ValueError("normal_weight must be finite and nonnegative")
    contacts = _contacts_by_link(robot_contacts)
    available = tuple(regions)
    if not contacts or len(contacts) > len(available):
        return ContactModeMatch(False, None, (), None, None)
    best: tuple[float, tuple[tuple[str, str], ...], float, float] | None = None
    if allow_digit_substitution:
        assignments = permutations(range(len(available)), len(contacts))
    else:
        candidates: list[int] = []
        for contact in contacts:
            matching = [index for index, region in enumerate(available) if region.label == contact.finger]
            if len(matching) != 1:
                return ContactModeMatch(False, None, (), None, None)
            candidates.append(matching[0])
        assignments = (tuple(candidates),)
    for region_indices in assignments:
        cost = 0.0
        excesses: list[float] = []
        alignments: list[float] = []
        labels: list[tuple[str, str]] = []
        for contact, region_index in zip(contacts, region_indices, strict=True):
            region = available[region_index]
            distance = float(np.linalg.norm(contact.position_object_local_m - region.center_object_local_m))
            excess = max(0.0, distance - region.radius_m)
            alignment = float(np.clip(np.dot(
                _unit(contact.outward_normal_object_local, label="robot contact normal"),
                _unit(region.outward_normal_object_local, label="region normal"),
            ), -1.0, 1.0))
            cost += (excess / region.radius_m) ** 2 + normal_weight * (1.0 - alignment)
            excesses.append(excess)
            alignments.append(alignment)
            labels.append((contact.finger, region.label))
        candidate = (cost, tuple(labels), float(np.mean(excesses)), float(np.mean(alignments)))
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return ContactModeMatch(True, best[0], best[1], best[2], best[3])


def demonstrated_opposed_surface_coverage(
    robot_contacts: Sequence[RobotContact], regions: Sequence[ContactRegion],
    *, max_region_excess_m: float = 0.003, min_normal_alignment: float = 0.5,
    max_demonstrated_normal_dot: float = -0.5,
) -> DemonstratedOpposedSurfaceCoverage:
    """Find a partial, digit-agnostic match covering two opposed GT regions.

    ``match_contact_mode`` answers a softer diagnostic question: it forces all
    representative robot contacts into an injective region assignment and
    reports its least-bad cost.  That is useful for debugging, but is unsafe
    as a promotion criterion: a many-finger contact on the wrong side of the
    object may receive a finite cost and then be outweighed by a wrench reward.

    This predicate instead searches for *any two* distinct robot contacts and
    distinct active regions that satisfy hard local geometry tolerances and
    whose **demonstrated object-surface normals** oppose one another.  It does
    not require matching human digit names, does not penalise extra robot
    contacts, and never calls the result force closure.  Friction/wrench and
    live torque tests remain separate checks.
    """
    if (
        not np.isfinite((max_region_excess_m, min_normal_alignment, max_demonstrated_normal_dot)).all()
        or max_region_excess_m < 0.0
        or not -1.0 <= min_normal_alignment <= 1.0
        or not -1.0 <= max_demonstrated_normal_dot <= 1.0
    ):
        raise ValueError("demonstrated surface-coverage thresholds are invalid")
    contacts = _contacts_by_link(robot_contacts)
    available = tuple(regions)
    if len(contacts) < 2 or len(available) < 2:
        return DemonstratedOpposedSurfaceCoverage(False, (), (), None, None, None, None)

    # A feasible edge preserves object-side geometry, independently of the
    # source/robot finger names.  Cost only resolves ties among already-valid
    # edges; it cannot turn an invalid contact into a covered region.
    edges: list[tuple[int, int, float, float, float]] = []
    for contact_index, contact in enumerate(contacts):
        contact_normal = _unit(contact.outward_normal_object_local, label="robot contact normal")
        for region_index, region in enumerate(available):
            distance = float(np.linalg.norm(
                contact.position_object_local_m - region.center_object_local_m,
            ))
            excess = max(0.0, distance - region.radius_m)
            alignment = float(np.clip(np.dot(
                contact_normal,
                _unit(region.outward_normal_object_local, label="region normal"),
            ), -1.0, 1.0))
            if excess <= max_region_excess_m and alignment >= min_normal_alignment:
                normalized_excess = excess / max(max_region_excess_m, 1.0e-12)
                cost = normalized_excess * normalized_excess + (1.0 - alignment)
                edges.append((contact_index, region_index, excess, alignment, cost))

    best: tuple[
        float, tuple[int, int], tuple[int, int], tuple[float, float], tuple[float, float], float,
    ] | None = None
    for left in range(len(edges)):
        first = edges[left]
        for right in range(left + 1, len(edges)):
            second = edges[right]
            if first[0] == second[0] or first[1] == second[1]:
                continue
            normal_dot = float(np.clip(np.dot(
                _unit(available[first[1]].outward_normal_object_local, label="first region normal"),
                _unit(available[second[1]].outward_normal_object_local, label="second region normal"),
            ), -1.0, 1.0))
            if normal_dot > max_demonstrated_normal_dot:
                continue
            candidate = (
                first[4] + second[4],
                (first[0], second[0]),
                (first[1], second[1]),
                (first[2], second[2]),
                (first[3], second[3]),
                normal_dot,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        return DemonstratedOpposedSurfaceCoverage(False, (), (), None, None, None, None)
    cost, contact_indices, region_indices, excesses, alignments, normal_dot = best
    return DemonstratedOpposedSurfaceCoverage(
        feasible=True,
        assignments=tuple(
            (contacts[contact_index].finger, available[region_index].label)
            for contact_index, region_index in zip(contact_indices, region_indices, strict=True)
        ),
        region_indices=region_indices,
        mean_region_excess_m=float(np.mean(excesses)),
        mean_normal_alignment=float(np.mean(alignments)),
        demonstrated_normal_dot=normal_dot,
        cost=float(cost),
    )


def gravity_wrench_in_object_frame(
    object_rotation_world: np.ndarray, *, mass_kg: float, gravity_world_m_s2: np.ndarray,
) -> np.ndarray:
    """Return the hand wrench needed to statically cancel gravity at the COM."""
    rotation = np.asarray(object_rotation_world, dtype=np.float64)
    gravity = np.asarray(gravity_world_m_s2, dtype=np.float64)
    if (
        rotation.shape != (3, 3) or gravity.shape != (3,) or not np.isfinite(rotation).all()
        or not np.isfinite(gravity).all() or not np.isfinite(mass_kg) or mass_kg <= 0.0
    ):
        raise ValueError("object rotation, gravity, and positive mass are required")
    result = np.zeros(6, dtype=np.float64)
    result[:3] = rotation.T @ (-float(mass_kg) * gravity)
    return result


def _tangent_basis(outward_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _unit(outward_normal, label="outward contact normal")
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    first = _unit(np.cross(normal, axis), label="contact tangent")
    second = _unit(np.cross(normal, first), label="contact tangent")
    return first, second


def evaluate_frictional_wrench(
    robot_contacts: Sequence[RobotContact], required_wrench_object_local: np.ndarray,
    *, moment_origin_object_local_m: np.ndarray | None = None,
    max_normal_force_per_contact_n: float = 8.0, cone_sides: int = 32,
    closure_margin: float = 1.0e-5,
) -> WrenchFeasibility:
    """Test support and approximate force closure using bounded friction cones.

    ``moment_origin_object_local_m`` must be the same origin used by the
    requested torque.  Passing the object COM and a zero gravity torque is
    the normal hand-only support test.  ``max_normal_force_per_contact_n`` is an explicit planning cap, not an
    inferred XHand contact-force limit.  It keeps a geometric LP from claiming
    that arbitrarily large forces rescue an otherwise weak candidate.
    """
    contacts = _contacts_by_link(robot_contacts)
    wrench = np.asarray(required_wrench_object_local, dtype=np.float64)
    origin = (
        np.zeros(3, dtype=np.float64) if moment_origin_object_local_m is None
        else np.asarray(moment_origin_object_local_m, dtype=np.float64)
    )
    if (
        wrench.shape != (6,) or not np.isfinite(wrench).all()
        or origin.shape != (3,) or not np.isfinite(origin).all()
        or not np.isfinite(max_normal_force_per_contact_n)
        or max_normal_force_per_contact_n <= 0.0 or cone_sides < 4
        or not np.isfinite(closure_margin) or closure_margin <= 0.0
    ):
        raise ValueError("wrench test arguments are invalid")
    if not contacts:
        return WrenchFeasibility(
            False, False, 0, 0, None, None, None, None, None, None, (), "no hand-object contacts",
        )
    columns: list[np.ndarray] = []
    groups: list[list[int]] = []
    for contact in contacts:
        normal = _unit(contact.outward_normal_object_local, label="contact normal")
        tangent_a, tangent_b = _tangent_basis(normal)
        group: list[int] = []
        for side in range(cone_sides):
            angle = 2.0 * np.pi * side / cone_sides
            # The hand can push into the object (opposite its outward normal)
            # and apply bounded tangential friction around that direction.
            force = -normal + contact.friction_coefficient * (
                np.cos(angle) * tangent_a + np.sin(angle) * tangent_b
            )
            group.append(len(columns))
            moment_arm = contact.position_object_local_m - origin
            columns.append(np.concatenate([force, np.cross(moment_arm, force)]))
        groups.append(group)
    matrix = np.stack(columns, axis=1)
    bounds = [(0.0, None)] * matrix.shape[1]
    group_rows = np.zeros((len(groups), matrix.shape[1]), dtype=np.float64)
    for index, group in enumerate(groups):
        group_rows[index, group] = 1.0
    # This separate minimax LP quantifies a serious distinction that a binary
    # force-closure label loses: a contact set may geometrically span all
    # wrenches but require hundreds of newtons of internal squeeze to balance
    # one light object.  It intentionally has no arbitrary planning cap.
    minimax = linprog(
        c=np.append(np.zeros(matrix.shape[1]), 1.0),
        A_ub=np.hstack([group_rows, -np.ones((len(groups), 1))]),
        b_ub=np.zeros(len(groups), dtype=np.float64),
        A_eq=np.hstack([matrix, np.zeros((6, 1))]),
        b_eq=wrench,
        bounds=[(0.0, None)] * (matrix.shape[1] + 1),
        method="highs",
    )
    minimum_peak = float(minimax.x[-1]) if minimax.success else None
    support = linprog(
        c=np.ones(matrix.shape[1], dtype=np.float64) * 1.0e-6,
        A_ub=group_rows,
        b_ub=np.full(len(groups), float(max_normal_force_per_contact_n)),
        A_eq=matrix,
        b_eq=wrench,
        bounds=bounds,
        method="highs",
    )
    rank_scale = max(
        float(np.max(np.linalg.norm([
            contact.position_object_local_m - origin for contact in contacts
        ], axis=1))), 1.0e-3,
    )
    normalized_matrix = matrix.copy()
    normalized_matrix[3:] /= rank_scale
    rank = int(np.linalg.matrix_rank(normalized_matrix, tol=1.0e-8))
    # Origin in the interior of the normalized primitive-wrench convex hull is
    # a finite polyhedral approximation of force closure.  The max-min LP
    # intentionally does not reuse the gravity-support solution.
    closure_eq = np.vstack([
        np.hstack([normalized_matrix, np.zeros((6, 1))]),
        np.append(np.ones(matrix.shape[1]), 0.0),
    ])
    closure_rhs = np.zeros(7, dtype=np.float64)
    closure_rhs[-1] = 1.0
    closure_ub = np.zeros((matrix.shape[1], matrix.shape[1] + 1), dtype=np.float64)
    for index in range(matrix.shape[1]):
        closure_ub[index, index] = -1.0
        closure_ub[index, -1] = 1.0
    closure = linprog(
        c=np.append(np.zeros(matrix.shape[1]), -1.0),
        A_ub=closure_ub,
        b_ub=np.zeros(matrix.shape[1]),
        A_eq=closure_eq,
        b_eq=closure_rhs,
        bounds=[(0.0, None)] * (matrix.shape[1] + 1),
        method="highs",
    )
    force_closure = bool(
        rank == 6 and closure.success and float(closure.x[-1]) > closure_margin
    )
    interior_margin = float(closure.x[-1]) if closure.success else None
    if not support.success:
        return WrenchFeasibility(
            False, force_closure, len(contacts), rank, None, None, None, None, minimum_peak,
            interior_margin, (),
            f"support LP infeasible: {support.message}",
        )
    solution = np.asarray(support.x, dtype=np.float64)
    residual = float(np.linalg.norm(matrix @ solution - wrench))
    forces: list[np.ndarray] = []
    friction_margins: list[float] = []
    cap_margins: list[float] = []
    normal_sum = 0.0
    for contact, group in zip(contacts, groups, strict=True):
        force = matrix[:3, group] @ solution[group]
        normal = _unit(contact.outward_normal_object_local, label="contact normal")
        normal_force = float(-np.dot(force, normal))
        tangent = force + normal_force * normal
        forces.append(force)
        normal_sum += normal_force
        friction_margins.append(float(contact.friction_coefficient * normal_force - np.linalg.norm(tangent)))
        cap_margins.append(float(max_normal_force_per_contact_n - normal_force))
    return WrenchFeasibility(
        True, force_closure, len(contacts), rank, residual, normal_sum,
        float(min(friction_margins)), float(min(cap_margins)), minimum_peak, interior_margin, tuple(forces),
        "support LP feasible",
    )


def evaluate_finger_torque_limited_wrench(
    backend: MujocoReplayBackend, robot_contacts: Sequence[RobotContact],
    required_wrench_object_local: np.ndarray, *,
    moment_origin_object_local_m: np.ndarray | None = None,
    max_normal_force_per_contact_n: float = 8.0, cone_sides: int = 32,
) -> TorqueLimitedWrenchFeasibility:
    """Add XHand finger-motor limits to a bounded friction-cone wrench LP.

    For every candidate cone force ``f`` on the *object*, MuJoCo's point
    Jacobian maps the opposite force on the hand to a generalized joint load
    ``tau = J(q)^T (-R_object f)``.  The LP enforces each named finger motor's
    own ctrlrange (the URDF effort limit in this project), rather than treating
    force closure as if arbitrary finger torque were available.  Wrist loads
    are deliberately not constrained here: their six motors are task-space
    motion actuation, while this gate answers the narrower, reproducible
    question whether the XHand fingers can create the proposed squeeze.
    """
    contacts = _contacts_by_link(robot_contacts)
    force_only = evaluate_frictional_wrench(
        contacts, required_wrench_object_local,
        moment_origin_object_local_m=moment_origin_object_local_m,
        max_normal_force_per_contact_n=max_normal_force_per_contact_n,
        cone_sides=cone_sides,
    )
    actuator_ids = tuple(int(value) for value in backend.finger_torque_ids.tolist())
    if not force_only.support_feasible:
        return TorqueLimitedWrenchFeasibility(
            force_only, False, None, None, (), actuator_ids,
            "force-only bounded wrench LP is infeasible",
        )
    if not actuator_ids:
        raise ValueError("finger torque gate requires named torque-controlled XHand fingers")
    origin = (
        np.zeros(3, dtype=np.float64) if moment_origin_object_local_m is None
        else np.asarray(moment_origin_object_local_m, dtype=np.float64)
    )
    wrench = np.asarray(required_wrench_object_local, dtype=np.float64)
    # Recreate the same finite friction-cone columns as the force-only test,
    # while retaining the originating contact needed for its point Jacobian.
    columns: list[np.ndarray] = []
    primitive_forces: list[np.ndarray] = []
    primitive_contacts: list[RobotContact] = []
    groups: list[list[int]] = []
    for contact in contacts:
        normal = _unit(contact.outward_normal_object_local, label="contact normal")
        tangent_a, tangent_b = _tangent_basis(normal)
        group: list[int] = []
        for side in range(cone_sides):
            angle = 2.0 * np.pi * side / cone_sides
            force = -normal + contact.friction_coefficient * (
                np.cos(angle) * tangent_a + np.sin(angle) * tangent_b
            )
            group.append(len(columns))
            primitive_forces.append(force)
            primitive_contacts.append(contact)
            arm = contact.position_object_local_m - origin
            columns.append(np.concatenate([force, np.cross(arm, force)]))
        groups.append(group)
    matrix = np.stack(columns, axis=1)
    group_rows = np.zeros((len(groups), matrix.shape[1]), dtype=np.float64)
    for index, group in enumerate(groups):
        group_rows[index, group] = 1.0

    dof_addresses = np.asarray(backend.actuator_qvel_addresses, dtype=np.int64)
    actuator_index = {int(aid): index for index, aid in enumerate(range(backend.model.nu))}
    selected = np.asarray([actuator_index[aid] for aid in actuator_ids], dtype=np.int64)
    if len(np.unique(dof_addresses[selected])) != len(selected):
        raise ValueError("finger torque gate requires one unique actuator per finger DoF")
    torque_columns = np.empty((len(selected), matrix.shape[1]), dtype=np.float64)
    rotation = backend.data.xmat[backend.object_body_id].reshape(3, 3)
    jacp = np.empty((3, backend.model.nv), dtype=np.float64)
    jacr = np.empty((3, backend.model.nv), dtype=np.float64)
    for column, (force_object, contact) in enumerate(zip(primitive_forces, primitive_contacts, strict=True)):
        if contact.hand_body_id is None or contact.position_world_m is None:
            raise ValueError("finger torque gate requires live MuJoCo contact body and world position")
        jacp.fill(0.0)
        jacr.fill(0.0)
        backend.mujoco.mj_jac(
            backend.model, backend.data, jacp, jacr,
            np.asarray(contact.position_world_m, dtype=np.float64), int(contact.hand_body_id),
        )
        # Cone primitives are forces exerted on the object.  Equal and
        # opposite contact forces load the hand joints.
        force_hand_world = -rotation @ force_object
        torque_columns[:, column] = jacp[:, dof_addresses[selected]].T @ force_hand_world
    torque_lower = np.empty(len(selected), dtype=np.float64)
    torque_upper = np.empty(len(selected), dtype=np.float64)
    for index, aid in enumerate(actuator_ids):
        if not bool(backend.model.actuator_ctrllimited[aid]):
            raise ValueError("finger torque gate requires bounded actuator ctrlranges")
        gear = float(backend.model.actuator_gear[aid, 0])
        low, high = np.asarray(backend.model.actuator_ctrlrange[aid], dtype=np.float64) * gear
        torque_lower[index], torque_upper[index] = min(low, high), max(low, high)
    common_ub = np.vstack([group_rows, torque_columns, -torque_columns])
    common_b = np.concatenate([
        np.full(len(groups), float(max_normal_force_per_contact_n)),
        torque_upper, -torque_lower,
    ])
    support = linprog(
        c=np.full(matrix.shape[1], 1.0e-6, dtype=np.float64),
        A_ub=common_ub, b_ub=common_b, A_eq=matrix, b_eq=wrench,
        bounds=[(0.0, None)] * matrix.shape[1], method="highs",
    )
    minimax_ub = np.hstack([common_ub, np.zeros((common_ub.shape[0], 1))])
    minimax_ub[:len(groups), -1] = -1.0
    minimax = linprog(
        c=np.append(np.zeros(matrix.shape[1]), 1.0),
        A_ub=minimax_ub, b_ub=np.concatenate([np.zeros(len(groups)), torque_upper, -torque_lower]),
        A_eq=np.hstack([matrix, np.zeros((6, 1))]), b_eq=wrench,
        bounds=[(0.0, None)] * (matrix.shape[1] + 1), method="highs",
    )
    minimum_peak = float(minimax.x[-1]) if minimax.success else None
    if not support.success:
        return TorqueLimitedWrenchFeasibility(
            force_only, False, minimum_peak, None, (), actuator_ids,
            f"finger-torque LP infeasible: {support.message}",
        )
    loads = torque_columns @ np.asarray(support.x, dtype=np.float64)
    margins = np.minimum(torque_upper - loads, loads - torque_lower)
    return TorqueLimitedWrenchFeasibility(
        force_only, True, minimum_peak, float(np.min(margins)),
        tuple(float(value) for value in loads), actuator_ids,
        "force and named XHand finger-torque LP feasible",
    )
