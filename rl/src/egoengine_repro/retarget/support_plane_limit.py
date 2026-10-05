"""Native material-mesh half-space constraints for kinematic retargeting.

This module is deliberately independent of MuJoCo collision proxies.  For an
infinite support plane, the minimum over a mesh equals the minimum over its
convex-hull vertices; no such equivalence is claimed for mesh--mesh contact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import numpy as np
from scipy.spatial import ConvexHull

from ..scene.support_surface import Plane


@dataclass(frozen=True)
class NativeSupportGeom:
    geom_id: int
    body_id: int
    name: str
    full_vertices: np.ndarray
    support_vertices: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "geom_id": self.geom_id,
            "body_id": self.body_id,
            "name": self.name,
            "full_vertex_count": int(len(self.full_vertices)),
            "support_vertex_count": int(len(self.support_vertices)),
            "full_vertices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.full_vertices).tobytes()
            ).hexdigest(),
            "support_vertices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.support_vertices).tobytes()
            ).hexdigest(),
        }


def native_hand_visual_geom_ids(model: Any, mujoco: Any) -> list[int]:
    """Return exactly the two hands' material visual mesh geoms."""
    result: list[int] = []
    for geom_id in range(int(model.ngeom)):
        name = model.geom(geom_id).name or ""
        selected = (
            name.startswith(("right_", "left_"))
            and name.endswith("_visual")
            and "_object_" not in name
        )
        if not selected:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError(f"native material geom is not a mesh: {name}")
        result.append(geom_id)
    if not result:
        raise ValueError("no native hand visual mesh geoms found")
    return result


def _mesh_vertices(model: Any, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise ValueError(f"geom {geom_id} has no mesh data")
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64).copy()
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"geom {geom_id} has invalid native mesh vertices")
    if not np.isfinite(vertices).all():
        raise ValueError(f"geom {geom_id} has nonfinite native mesh vertices")
    return vertices


def _hull_vertices(vertices: np.ndarray) -> np.ndarray:
    hull = ConvexHull(vertices)
    result = vertices[np.asarray(hull.vertices, dtype=np.int64)].copy()
    if not len(result):
        raise ValueError("convex hull has no support vertices")
    return result


def build_native_support_geoms(
    model: Any, mujoco: Any, visual_geom_ids: Iterable[int],
) -> tuple[NativeSupportGeom, ...]:
    ids = tuple(int(value) for value in visual_geom_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("native support geom ids must be unique")
    geoms: list[NativeSupportGeom] = []
    for geom_id in ids:
        if not 0 <= geom_id < int(model.ngeom):
            raise ValueError(f"invalid geom id: {geom_id}")
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError(f"support geom must be a mesh: {model.geom(geom_id).name}")
        vertices = _mesh_vertices(model, geom_id)
        geoms.append(NativeSupportGeom(
            geom_id=geom_id,
            body_id=int(model.geom_bodyid[geom_id]),
            name=model.geom(geom_id).name,
            full_vertices=vertices,
            support_vertices=_hull_vertices(vertices),
        ))
    if not geoms:
        raise ValueError("native support limit requires at least one geom")
    return tuple(geoms)


def geom_world_vertices(data: Any, geom: NativeSupportGeom, *, full: bool) -> np.ndarray:
    local = geom.full_vertices if full else geom.support_vertices
    rotation = np.asarray(data.geom_xmat[geom.geom_id], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(data.geom_xpos[geom.geom_id], dtype=np.float64)
    return local @ rotation.T + translation


def independent_native_floor_rows(
    model: Any, mujoco: Any, plane: Plane, qpos: np.ndarray,
    visual_geom_ids: Iterable[int],
) -> list[dict[str, Any]]:
    """Full-vertex audit, intentionally independent from the QP support set."""
    geoms = build_native_support_geoms(model, mujoco, visual_geom_ids)
    trajectory = np.asarray(qpos, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != int(model.nq):
        raise ValueError("native floor audit requires (frames,nq) qpos")
    data = mujoco.MjData(model)
    rows: list[dict[str, Any]] = []
    for frame, q in enumerate(trajectory):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for geom in geoms:
            distances = plane.signed_distance(geom_world_vertices(data, geom, full=True))
            rows.append({
                "frame": frame,
                "geom_id": geom.geom_id,
                "geom": geom.name,
                "side": geom.name.split("_", 1)[0],
                "minimum_signed_distance_m": float(distances.min()),
            })
    return rows


class NativeSupportPlaneLimit:
    """QP half-space limit evaluated on native visual-mesh support points."""

    def __init__(
        self,
        model: Any,
        mujoco: Any,
        plane: Plane,
        visual_geom_ids: Iterable[int],
        minimum_clearance_m: float = 0.0,
        activation_distance_m: float | None = None,
        gain: float = 0.85,
        depenetration_step_m: float = 0.002,
        constraint_type: Any | None = None,
    ) -> None:
        self.model = model
        self.mujoco = mujoco
        self.plane = plane
        self.geoms = build_native_support_geoms(model, mujoco, visual_geom_ids)
        self.minimum_clearance_m = float(minimum_clearance_m)
        self.activation_distance_m = (
            None if activation_distance_m is None else float(activation_distance_m)
        )
        self.gain = float(gain)
        self.depenetration_step_m = float(depenetration_step_m)
        if not np.isfinite(self.minimum_clearance_m):
            raise ValueError("minimum clearance must be finite")
        if self.activation_distance_m is not None and (
            not np.isfinite(self.activation_distance_m)
            or self.activation_distance_m < self.minimum_clearance_m
        ):
            raise ValueError("activation distance must be finite and >= clearance")
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("gain must be in (0,1]")
        if not np.isfinite(self.depenetration_step_m) or self.depenetration_step_m <= 0.0:
            raise ValueError("depenetration step must be positive and finite")
        if constraint_type is None:
            import mink
            constraint_type = mink.Constraint
        self.constraint_type = constraint_type
        self._jacobian = np.empty((3, int(model.nv)), dtype=np.float64)

    def _support(self, data: Any, geom: NativeSupportGeom) -> tuple[np.ndarray, float]:
        rotation = np.asarray(data.geom_xmat[geom.geom_id], dtype=np.float64).reshape(3, 3)
        local_normal = rotation.T @ self.plane.normal
        local = geom.support_vertices[
            int(np.argmin(geom.support_vertices @ local_normal))
        ]
        point = rotation @ local + np.asarray(data.geom_xpos[geom.geom_id])
        distance = float(self.plane.signed_distance(point))
        return point, distance

    def rows(self, configuration: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for geom in self.geoms:
            point, distance = self._support(configuration.data, geom)
            self._jacobian.fill(0.0)
            self.mujoco.mj_jac(
                self.model, configuration.data, self._jacobian, None,
                point, geom.body_id,
            )
            normal_jacobian = self.plane.normal @ self._jacobian
            result.append({
                "geom": geom,
                "point_world": point.copy(),
                "distance_m": distance,
                "normal_jacobian": normal_jacobian.copy(),
            })
        return result

    def minimum_distance(self, configuration: Any) -> float:
        return min(row["distance_m"] for row in self.rows(configuration))

    def compute_qp_inequalities(self, configuration: Any, _dt: float) -> Any:
        matrices: list[np.ndarray] = []
        bounds: list[float] = []
        for row in self.rows(configuration):
            distance = float(row["distance_m"])
            if (
                self.activation_distance_m is not None
                and distance > self.activation_distance_m
            ):
                continue
            residual = distance - self.minimum_clearance_m
            if residual >= 0.0:
                bound = self.gain * residual
            else:
                bound = -min(
                    self.depenetration_step_m,
                    self.gain * (-residual),
                )
            matrices.append(-np.asarray(row["normal_jacobian"], dtype=np.float64))
            bounds.append(bound)
        if not matrices:
            return self.constraint_type()
        return self.constraint_type(
            np.asarray(matrices, dtype=np.float64),
            np.asarray(bounds, dtype=np.float64),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "native_support_geom_manifest_v1",
            "selection": "left/right native material visual mesh geoms; object visuals excluded",
            "support_query": "convex-hull vertices for infinite-plane support only",
            "geom_count": len(self.geoms),
            "geoms": [geom.to_dict() for geom in self.geoms],
        }
