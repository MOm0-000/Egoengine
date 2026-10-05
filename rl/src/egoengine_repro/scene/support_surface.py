"""Coordinate-free support-plane contracts.

Planes use ``normal dot point = offset``.  The module deliberately exposes no
``table_z`` shortcut: source ingestion, scene alignment and collision audits
must consume the same geometric object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


TACO_PUBLISHED_TABLE_HEIGHT_M = 0.72


def _unit_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _rigid_transform(value: Any) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be finite 4x4")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12):
        raise ValueError("invalid homogeneous row")
    rotation = transform[:3, :3]
    # Released TACO transforms are stored as float32 and exhibit ~1e-8
    # orthogonality error.  Accept that representation error without silently
    # projecting or otherwise changing the published pose.
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("plane transform requires a rigid rotation")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("plane transform rotation must be proper")
    return transform


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray
    offset: float
    frame: str

    def __post_init__(self) -> None:
        normal = _unit_vector(self.normal, name="plane normal")
        offset = float(self.offset)
        if not np.isfinite(offset) or not self.frame:
            raise ValueError("plane offset/frame must be finite and nonempty")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", offset)

    def signed_distance(self, points: Any) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1:] != (3,) or not np.isfinite(values).all():
            raise ValueError("points must be finite (...,3)")
        return values @ self.normal - self.offset

    def transform(self, transform_target_source: Any, *, target_frame: str) -> "Plane":
        transform = _rigid_transform(transform_target_source)
        normal = transform[:3, :3] @ self.normal
        offset = self.offset + float(normal @ transform[:3, 3])
        return Plane(normal=normal, offset=offset, frame=target_frame)

    def minimum_signed_distance(self, vertices: Any, pose: Any | None = None) -> float:
        raw_vertices = vertices.vertices if hasattr(vertices, "vertices") else vertices
        values = np.asarray(raw_vertices, dtype=np.float64)
        if pose is not None:
            transform = _rigid_transform(pose)
            values = values @ transform[:3, :3].T + transform[:3, 3]
        return float(self.signed_distance(values).min())

    def assert_entity_supported(self, vertices: Any, pose: Any | None = None,
                                *, tolerance_m: float = 1e-9) -> float:
        if not np.isfinite(tolerance_m) or tolerance_m < 0:
            raise ValueError("support tolerance must be finite and nonnegative")
        minimum = self.minimum_signed_distance(vertices, pose)
        if abs(minimum) > tolerance_m:
            raise ValueError(
                f"entity is not supported by plane within tolerance: {minimum} m"
            )
        return minimum

    def to_dict(self) -> dict[str, Any]:
        return {"normal": self.normal.tolist(), "offset_m": self.offset,
                "frame": self.frame, "equation": "normal dot x = offset"}


@dataclass(frozen=True)
class SupportSurfaceSpec:
    frame_index: int
    entity_role: str
    entity_id: str
    direction_world: np.ndarray
    extremum: str = "minimum"
    method: str = "support_entity_extreme_surface"

    def __post_init__(self) -> None:
        if self.frame_index < 0 or not self.entity_role or not self.entity_id:
            raise ValueError("invalid source support entity")
        if self.extremum not in {"minimum", "maximum"}:
            raise ValueError("support extremum must be minimum or maximum")
        object.__setattr__(self, "direction_world",
                           _unit_vector(self.direction_world, name="support direction"))

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "frame_index": self.frame_index,
                "entity_role": self.entity_role, "entity_id": self.entity_id,
                "direction_world": self.direction_world.tolist(),
                "extremum": self.extremum}


@dataclass(frozen=True)
class SupportSurfaceContract:
    source_frame: str
    source: SupportSurfaceSpec
    simulator: Plane
    provenance: str
    schema: str = "support_surface_contract_v1"

    def __post_init__(self) -> None:
        if self.schema != "support_surface_contract_v1":
            raise ValueError("unsupported support-surface schema")
        if not self.source_frame or not self.provenance:
            raise ValueError("support contract needs frame and provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "source_frame": self.source_frame,
                "source": self.source.to_dict(),
                "simulator": self.simulator.to_dict(),
                "provenance": self.provenance}


@dataclass(frozen=True)
class ResolvedSupportSurface:
    contract: SupportSurfaceContract
    source: Plane
    simulator: Plane
    support_entity_minimum_distance_m: float

    def signed_distance(self, points: Any) -> np.ndarray:
        """Measure points in the resolved source coordinate system."""
        return self.source.signed_distance(points)

    def transform(self, transform_sim_source: Any) -> Plane:
        """Transform the resolved source plane into the simulator frame."""
        return self.source.transform(
            transform_sim_source, target_frame=self.simulator.frame
        )

    def assert_entity_supported(self, mesh_or_vertices: Any, pose: Any | None = None,
                                *, tolerance_m: float = 1e-9) -> float:
        return self.source.assert_entity_supported(
            mesh_or_vertices, pose, tolerance_m=tolerance_m
        )

    def to_dict(self) -> dict[str, Any]:
        return {"contract": self.contract.to_dict(), "source": self.source.to_dict(),
                "simulator": self.simulator.to_dict(),
                "support_entity_minimum_distance_m": self.support_entity_minimum_distance_m}


def make_taco_support_contract(*, target_id: str, simulator_offset_m: float,
                               frame_index: int = 0,
                               provenance: str = "PROJECT_SAMPLE_CONTRACT") -> SupportSurfaceContract:
    return SupportSurfaceContract(
        source_frame="world",
        source=SupportSurfaceSpec(
            frame_index=frame_index, entity_role="target", entity_id=str(target_id),
            direction_world=np.array([0.0, 0.0, 1.0]), extremum="minimum",
        ),
        simulator=Plane(normal=np.array([0.0, 0.0, 1.0]),
                        offset=float(simulator_offset_m), frame="simulator"),
        provenance=provenance,
    )


def taco_project_sample_support_contract(*, target_id: str,
                                         frame_index: int = 0) -> SupportSurfaceContract:
    """Current project-wide TACO support convention.

    The table height is author-published.  Using the target extreme surface as
    the source support plane is explicitly a project sample contract.
    """
    return make_taco_support_contract(
        target_id=target_id,
        simulator_offset_m=TACO_PUBLISHED_TABLE_HEIGHT_M,
        frame_index=frame_index,
        provenance="PROJECT_SAMPLE_CONTRACT",
    )


def resolve_support_surface(
    contract: SupportSurfaceContract,
    source_scene: Mapping[str, Mapping[str, Any]],
) -> ResolvedSupportSurface:
    spec = contract.source
    if spec.entity_role not in source_scene:
        raise KeyError(f"missing support entity role: {spec.entity_role}")
    entity = source_scene[spec.entity_role]
    vertices = np.asarray(entity["vertices"], dtype=np.float64)
    poses = np.asarray(entity["poses"], dtype=np.float64)
    if poses.ndim == 2:
        pose = poses
    else:
        pose = poses[spec.frame_index]
    transform = _rigid_transform(pose)
    world = vertices @ transform[:3, :3].T + transform[:3, 3]
    projections = world @ spec.direction_world
    offset = float(projections.min() if spec.extremum == "minimum" else projections.max())
    source = Plane(normal=spec.direction_world, offset=offset,
                   frame=contract.source_frame)
    minimum = source.minimum_signed_distance(world)
    return ResolvedSupportSurface(
        contract=contract, source=source, simulator=contract.simulator,
        support_entity_minimum_distance_m=minimum,
    )


def build_taco_scene_alignment(
    support: ResolvedSupportSurface,
    *,
    source_scene_center: Any,
    desired_scene_center_sim: Any,
    rotation_sim_world: Any | None = None,
) -> np.ndarray:
    """Build one rigid alignment whose transformed source plane equals sim support."""
    rotation = np.eye(3) if rotation_sim_world is None else np.asarray(
        rotation_sim_world, dtype=np.float64)
    probe = np.eye(4)
    probe[:3, :3] = rotation
    _rigid_transform(probe)
    mapped_normal = rotation @ support.source.normal
    if not np.allclose(mapped_normal, support.simulator.normal, atol=1e-9):
        raise ValueError("alignment rotation does not map source support normal to simulator normal")
    source_center = np.asarray(source_scene_center, dtype=np.float64)
    desired_center = np.asarray(desired_scene_center_sim, dtype=np.float64)
    if source_center.shape != (3,) or desired_center.shape != (3,):
        raise ValueError("scene centers must be 3-vectors")
    translation = desired_center - rotation @ source_center
    transformed_offset = support.source.offset + float(mapped_normal @ translation)
    translation += (support.simulator.offset - transformed_offset) * mapped_normal
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mapped = support.transform(transform)
    if not np.allclose(mapped.normal, support.simulator.normal, atol=1e-10):
        raise AssertionError("support normal mapping failed")
    if not np.isclose(mapped.offset, support.simulator.offset, atol=1e-10):
        raise AssertionError("support offset mapping failed")
    return transform


def horizontal_floor_position(support: Plane) -> np.ndarray:
    """Return a MuJoCo plane position after explicitly proving horizontal +Z."""
    if not np.allclose(support.normal, [0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("current MuJoCo floor builder supports only simulator +Z planes")
    return np.array([0.0, 0.0, support.offset], dtype=np.float64)
