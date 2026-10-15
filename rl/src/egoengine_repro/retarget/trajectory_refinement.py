"""Local continuous-reference refinement utilities.

The optimizer-facing geometry is the native visual mesh, not MuJoCo's
collision proxy.  This module also performs an exact reachability precheck:
an active clearance violation on a body that has no authorized ancestor DOF
cannot be repaired by the candidate and must stop before a QP is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .support_plane_limit import NativeSupportGeom, geom_world_vertices


@dataclass(frozen=True)
class TrajectoryVariable:
    endpoint: int
    joint_name: str
    joint_id: int
    qpos_address: int
    dof_address: int
    budget: float


@dataclass(frozen=True)
class SupportWitness:
    endpoint: int
    geom_id: int
    geom_name: str
    body_id: int
    body_name: str
    vertex_index: int
    point_world: np.ndarray
    distance_m: float


def resolve_variables(
    model: Any,
    *,
    endpoints: Sequence[int],
    joint_names: Sequence[str],
    budgets: Mapping[str, float],
    robot_qpos_stop: int = 36,
) -> tuple[TrajectoryVariable, ...]:
    if len(endpoints) != len(set(endpoints)) or len(joint_names) != len(set(joint_names)):
        raise ValueError("candidate endpoints and joint names must be unique")
    variables: list[TrajectoryVariable] = []
    for endpoint in endpoints:
        if int(endpoint) < 0:
            raise ValueError("candidate endpoint must be nonnegative")
        for name in joint_names:
            if name not in budgets:
                raise ValueError(f"missing modification budget for {name}")
            budget = float(budgets[name])
            if not np.isfinite(budget) or budget <= 0.0:
                raise ValueError(f"invalid modification budget for {name}")
            joint = model.joint(name)
            joint_id = int(joint.id)
            qpos_address = int(model.jnt_qposadr[joint_id])
            dof_address = int(model.jnt_dofadr[joint_id])
            if qpos_address >= robot_qpos_stop:
                raise ValueError(f"candidate joint is not a robot joint: {name}")
            if int(model.jnt_type[joint_id]) not in (
                2,  # mjJNT_SLIDE
                3,  # mjJNT_HINGE
            ):
                raise ValueError(f"candidate joint is not scalar: {name}")
            variables.append(TrajectoryVariable(
                endpoint=int(endpoint),
                joint_name=name,
                joint_id=joint_id,
                qpos_address=qpos_address,
                dof_address=dof_address,
                budget=budget,
            ))
    return tuple(variables)


def apply_delta(
    reference_qpos: np.ndarray,
    variables: Sequence[TrajectoryVariable],
    delta: np.ndarray,
) -> np.ndarray:
    source = np.asarray(reference_qpos, dtype=np.float64)
    values = np.asarray(delta, dtype=np.float64)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("reference qpos must be one finite trajectory")
    if values.shape != (len(variables),) or not np.isfinite(values).all():
        raise ValueError("delta does not match the trajectory variables")
    if len({(item.endpoint, item.qpos_address) for item in variables}) != len(variables):
        raise ValueError("trajectory variables contain duplicate endpoint/joint entries")
    result = source.copy()
    for value, variable in zip(values, variables, strict=True):
        if abs(float(value)) > variable.budget + 1e-15:
            raise ValueError(f"modification budget exceeded for {variable.joint_name}")
        result[variable.endpoint, variable.qpos_address] += value
    return result


def interpolate_qpos(
    model: Any,
    mujoco: Any,
    qpos: np.ndarray,
    source_endpoint: int,
    alpha: float,
    *,
    freeze_dofs_from: int | None = None,
) -> np.ndarray:
    trajectory = np.asarray(qpos, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != int(model.nq):
        raise ValueError("qpos must have shape (frames, model.nq)")
    if not 0 <= source_endpoint < len(trajectory) - 1:
        raise ValueError("source endpoint is outside the trajectory")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    tangent = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        model,
        tangent,
        1.0,
        trajectory[source_endpoint],
        trajectory[source_endpoint + 1],
    )
    if freeze_dofs_from is not None:
        tangent[int(freeze_dofs_from):] = 0.0
    result = trajectory[source_endpoint].copy()
    mujoco.mj_integratePos(model, result, tangent, float(alpha))
    return result


def full_mesh_support_witnesses(
    model: Any,
    data: Any,
    geoms: Iterable[NativeSupportGeom],
    *,
    plane_z_m: float,
    endpoint: int,
) -> tuple[SupportWitness, ...]:
    witnesses: list[SupportWitness] = []
    for geom in geoms:
        world = geom_world_vertices(data, geom, full=True)
        signed = world[:, 2] - float(plane_z_m)
        vertex = int(np.argmin(signed))
        witnesses.append(SupportWitness(
            endpoint=int(endpoint),
            geom_id=geom.geom_id,
            geom_name=geom.name,
            body_id=geom.body_id,
            body_name=model.body(geom.body_id).name,
            vertex_index=vertex,
            point_world=world[vertex].copy(),
            distance_m=float(signed[vertex]),
        ))
    if not witnesses:
        raise ValueError("native support witness set is empty")
    return tuple(witnesses)


def ancestor_dofs(model: Any, body_id: int) -> frozenset[int]:
    result: set[int] = set()
    current = int(body_id)
    while current > 0:
        start = int(model.body_dofadr[current])
        count = int(model.body_dofnum[current])
        result.update(range(start, start + count))
        current = int(model.body_parentid[current])
    return frozenset(result)


def immutable_clearance_blockers(
    model: Any,
    mujoco: Any,
    reference_qpos: np.ndarray,
    variables: Sequence[TrajectoryVariable],
    geoms: Iterable[NativeSupportGeom],
    *,
    endpoints: Sequence[int],
    plane_z_m: float,
    minimum_clearance_m: float,
) -> list[dict[str, Any]]:
    candidate_dofs = {item.dof_address for item in variables}
    data = mujoco.MjData(model)
    blockers: list[dict[str, Any]] = []
    for endpoint in endpoints:
        data.qpos[:] = reference_qpos[int(endpoint)]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for witness in full_mesh_support_witnesses(
            model, data, geoms, plane_z_m=plane_z_m, endpoint=int(endpoint),
        ):
            if witness.distance_m >= minimum_clearance_m:
                continue
            influencing = sorted(candidate_dofs & ancestor_dofs(model, witness.body_id))
            if influencing:
                continue
            blockers.append({
                "endpoint": witness.endpoint,
                "required_clearance_m": float(minimum_clearance_m),
                "actual_clearance_m": witness.distance_m,
                "deficit_m": float(minimum_clearance_m - witness.distance_m),
                "geom_id": witness.geom_id,
                "geom": witness.geom_name,
                "body_id": witness.body_id,
                "body": witness.body_name,
                "vertex_index": witness.vertex_index,
                "point_world_m": witness.point_world.tolist(),
                "body_ancestor_dofs": sorted(ancestor_dofs(model, witness.body_id)),
                "candidate_dofs": sorted(candidate_dofs),
                "candidate_influencing_dofs": influencing,
                "reason": "CLEARANCE_DEFICIT_ON_IMMUTABLE_BODY",
            })
    return blockers


def interpolated_support_linearization(
    model: Any,
    mujoco: Any,
    qpos: np.ndarray,
    variables: Sequence[TrajectoryVariable],
    geom: NativeSupportGeom,
    *,
    source_endpoint: int,
    alpha: float,
    plane_z_m: float,
    freeze_dofs_from: int | None = 36,
) -> tuple[float, np.ndarray, np.ndarray]:
    state = interpolate_qpos(
        model, mujoco, qpos, source_endpoint, alpha,
        freeze_dofs_from=freeze_dofs_from,
    )
    data = mujoco.MjData(model)
    data.qpos[:] = state
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    world = geom_world_vertices(data, geom, full=True)
    vertex = int(np.argmin(world[:, 2]))
    point = world[vertex]
    jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacobian, None, point, geom.body_id)
    gradient = np.zeros(len(variables), dtype=np.float64)
    for index, variable in enumerate(variables):
        if variable.endpoint == source_endpoint:
            gradient[index] = (1.0 - alpha) * jacobian[2, variable.dof_address]
        elif variable.endpoint == source_endpoint + 1:
            gradient[index] = alpha * jacobian[2, variable.dof_address]
    return float(point[2] - plane_z_m), gradient, point.copy()


def validate_scalar_joint_ranges(
    model: Any,
    qpos: np.ndarray,
    variables: Sequence[TrajectoryVariable],
    *,
    tolerance: float = 1e-12,
) -> None:
    trajectory = np.asarray(qpos, dtype=np.float64)
    for variable in variables:
        if not bool(model.jnt_limited[variable.joint_id]):
            continue
        lower, upper = np.asarray(model.jnt_range[variable.joint_id], dtype=np.float64)
        value = float(trajectory[variable.endpoint, variable.qpos_address])
        if value < lower - tolerance or value > upper + tolerance:
            raise ValueError(f"joint range violated: {variable.joint_name}")


def differentiate_trajectory(
    model: Any,
    mujoco: Any,
    qpos: np.ndarray,
    timestamps_s: np.ndarray,
) -> np.ndarray:
    trajectory = np.asarray(qpos, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != int(model.nq):
        raise ValueError("qpos must have shape (frames, model.nq)")
    if timestamps.shape != (len(trajectory),) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    qvel = np.empty((len(trajectory), model.nv), dtype=np.float64)
    for endpoint in range(len(trajectory)):
        first = max(0, endpoint - 1)
        last = min(len(trajectory) - 1, endpoint + 1)
        mujoco.mj_differentiatePos(
            model,
            qvel[endpoint],
            float(timestamps[last] - timestamps[first]),
            trajectory[first],
            trajectory[last],
        )
    if not np.isfinite(qvel).all():
        raise ValueError("derived qvel contains nonfinite values")
    return qvel


def path_minimum_clearance(
    model: Any,
    mujoco: Any,
    qpos: np.ndarray,
    geoms: Iterable[NativeSupportGeom],
    *,
    intervals: Sequence[int],
    alphas: np.ndarray,
    plane_z_m: float,
    freeze_dofs_from: int | None = 36,
) -> dict[str, Any]:
    alpha_values = np.asarray(alphas, dtype=np.float64)
    if alpha_values.ndim != 1 or not len(alpha_values):
        raise ValueError("alpha grid must be one-dimensional and nonempty")
    best: dict[str, Any] | None = None
    data = mujoco.MjData(model)
    for interval in intervals:
        for alpha in alpha_values:
            data.qpos[:] = interpolate_qpos(
                model, mujoco, qpos, int(interval), float(alpha),
                freeze_dofs_from=freeze_dofs_from,
            )
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            witnesses = full_mesh_support_witnesses(
                model, data, geoms, plane_z_m=plane_z_m,
                endpoint=int(interval),
            )
            witness = min(witnesses, key=lambda row: row.distance_m)
            row = {
                "source_endpoint": int(interval),
                "outcome_endpoint": int(interval) + 1,
                "alpha": float(alpha),
                "distance_m": witness.distance_m,
                "geom": witness.geom_name,
                "body": witness.body_name,
                "vertex_index": witness.vertex_index,
                "point_world_m": witness.point_world.tolist(),
            }
            if best is None or row["distance_m"] < best["distance_m"]:
                best = row
    if best is None:
        raise ValueError("no interval samples were evaluated")
    return best


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
