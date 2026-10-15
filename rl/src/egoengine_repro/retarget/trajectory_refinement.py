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


def interval_clearance_minima(
    model: Any,
    mujoco: Any,
    qpos: np.ndarray,
    geoms: Iterable[NativeSupportGeom],
    *,
    intervals: Sequence[int],
    alphas: np.ndarray,
    plane_z_m: float,
    freeze_dofs_from: int | None = 36,
) -> list[dict[str, Any]]:
    """Return one full-native-mesh minimum for every requested interval."""
    return [
        path_minimum_clearance(
            model,
            mujoco,
            qpos,
            geoms,
            intervals=(int(interval),),
            alphas=alphas,
            plane_z_m=plane_z_m,
            freeze_dofs_from=freeze_dofs_from,
        )
        for interval in intervals
    ]


def fingertip_positions(
    model: Any,
    mujoco: Any,
    qpos: np.ndarray,
    *,
    endpoints: Sequence[int],
    site_names: Sequence[str],
) -> np.ndarray:
    """Evaluate named fingertip sites at selected trajectory endpoints."""
    trajectory = np.asarray(qpos, dtype=np.float64)
    result = np.empty((len(endpoints), len(site_names), 3), dtype=np.float64)
    site_ids = [int(model.site(name).id) for name in site_names]
    data = mujoco.MjData(model)
    for endpoint_index, endpoint in enumerate(endpoints):
        data.qpos[:] = trajectory[int(endpoint)]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        result[endpoint_index] = np.asarray(data.site_xpos[site_ids], dtype=np.float64)
    return result


def _objective_terms(
    model: Any,
    mujoco: Any,
    reference_qpos: np.ndarray,
    current_qpos: np.ndarray,
    variables: Sequence[TrajectoryVariable],
    delta: np.ndarray,
    *,
    modified_endpoints: Sequence[int],
    site_names: Sequence[str],
    reference_weight: float,
    smoothness_weight: float,
    fingertip_weight: float,
    object_relative_fingertip_weight: float,
    fingertip_scale_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the local quadratic objective for a delta update."""
    count = len(variables)
    hessian = np.zeros((count, count), dtype=np.float64)
    gradient = np.zeros(count, dtype=np.float64)
    budgets = np.asarray([item.budget for item in variables], dtype=np.float64)

    reference_hessian = 2.0 * float(reference_weight) * np.diag(1.0 / budgets**2)
    hessian += reference_hessian
    gradient += reference_hessian @ delta

    variable_index = {
        (item.endpoint, item.joint_name): index
        for index, item in enumerate(variables)
    }
    joint_names = tuple(dict.fromkeys(item.joint_name for item in variables))
    for joint_name in joint_names:
        budget = next(item.budget for item in variables if item.joint_name == joint_name)
        for endpoint in modified_endpoints:
            row = np.zeros(count, dtype=np.float64)
            for neighbor, coefficient in (
                (int(endpoint) - 1, 1.0),
                (int(endpoint), -2.0),
                (int(endpoint) + 1, 1.0),
            ):
                index = variable_index.get((neighbor, joint_name))
                if index is not None:
                    row[index] = coefficient / budget
            hessian_add = 2.0 * float(smoothness_weight) * np.outer(row, row)
            hessian += hessian_add
            gradient += hessian_add @ delta

    reference_data = mujoco.MjData(model)
    current_data = mujoco.MjData(model)
    site_ids = [int(model.site(name).id) for name in site_names]
    tip_weight = float(fingertip_weight) + float(object_relative_fingertip_weight)
    tip_factor = 2.0 * tip_weight / float(fingertip_scale_m) ** 2
    for endpoint in modified_endpoints:
        reference_data.qpos[:] = reference_qpos[int(endpoint)]
        reference_data.qvel[:] = 0.0
        current_data.qpos[:] = current_qpos[int(endpoint)]
        current_data.qvel[:] = 0.0
        mujoco.mj_forward(model, reference_data)
        mujoco.mj_forward(model, current_data)
        for site_id in site_ids:
            residual = (
                np.asarray(current_data.site_xpos[site_id], dtype=np.float64)
                - np.asarray(reference_data.site_xpos[site_id], dtype=np.float64)
            )
            point_jacobian = np.zeros((3, int(model.nv)), dtype=np.float64)
            mujoco.mj_jacSite(model, current_data, point_jacobian, None, site_id)
            local = np.zeros((3, count), dtype=np.float64)
            for variable_index_value, variable in enumerate(variables):
                if variable.endpoint == int(endpoint):
                    local[:, variable_index_value] = point_jacobian[
                        :, variable.dof_address
                    ]
            hessian += tip_factor * (local.T @ local)
            gradient += tip_factor * (local.T @ residual)

    hessian += np.eye(count, dtype=np.float64) * 1.0e-12
    return hessian, gradient


def _clearance_linearization_rows(
    model: Any,
    mujoco: Any,
    current_qpos: np.ndarray,
    variables: Sequence[TrajectoryVariable],
    geoms: Sequence[NativeSupportGeom],
    *,
    sample_alphas: Mapping[int, Sequence[float]],
    thresholds_m: Mapping[int, float],
    plane_z_m: float,
    activation_margin_m: float,
    feasibility_tolerance_m: float,
    freeze_dofs_from: int | None,
) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    immutable: list[dict[str, Any]] = []
    data = mujoco.MjData(model)
    point_jacobian = np.zeros((3, int(model.nv)), dtype=np.float64)
    for interval, alpha_values in sample_alphas.items():
        threshold = float(thresholds_m[int(interval)])
        for alpha in alpha_values:
            alpha = float(alpha)
            data.qpos[:] = interpolate_qpos(
                model,
                mujoco,
                current_qpos,
                int(interval),
                alpha,
                freeze_dofs_from=freeze_dofs_from,
            )
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            for geom in geoms:
                world = geom_world_vertices(data, geom, full=True)
                vertex = int(np.argmin(world[:, 2]))
                distance = float(world[vertex, 2] - plane_z_m)
                if distance > threshold + float(activation_margin_m):
                    continue
                point_jacobian.fill(0.0)
                mujoco.mj_jac(
                    model,
                    data,
                    point_jacobian,
                    None,
                    world[vertex],
                    geom.body_id,
                )
                normal_gradient = np.zeros(len(variables), dtype=np.float64)
                for index, variable in enumerate(variables):
                    if variable.endpoint == int(interval):
                        normal_gradient[index] = (
                            (1.0 - alpha)
                            * point_jacobian[2, variable.dof_address]
                        )
                    elif variable.endpoint == int(interval) + 1:
                        normal_gradient[index] = (
                            alpha * point_jacobian[2, variable.dof_address]
                        )
                if np.linalg.norm(normal_gradient) <= 1.0e-14:
                    if distance < threshold - float(feasibility_tolerance_m):
                        immutable.append({
                            "source_endpoint": int(interval),
                            "alpha": alpha,
                            "geom": geom.name,
                            "vertex_index": vertex,
                            "distance_m": distance,
                            "required_m": threshold,
                        })
                    continue
                rows.append(-normal_gradient)
                bounds.append(
                    distance - threshold + float(feasibility_tolerance_m)
                )
    return rows, bounds, immutable


def _velocity_constraint_rows(
    model: Any,
    reference_qpos: np.ndarray,
    variables: Sequence[TrajectoryVariable],
    delta: np.ndarray,
    *,
    intervals: Sequence[int],
    velocity_limits: Mapping[str, float],
    dt_s: float,
) -> tuple[list[np.ndarray], list[float]]:
    count = len(variables)
    index = {
        (item.endpoint, item.joint_name): variable_index
        for variable_index, item in enumerate(variables)
    }
    joint_names = tuple(dict.fromkeys(item.joint_name for item in variables))
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    for joint_name in joint_names:
        limit = float(velocity_limits[joint_name])
        qpos_address = int(model.joint(joint_name).qposadr[0])
        for interval in intervals:
            row = np.zeros(count, dtype=np.float64)
            source_index = index.get((int(interval), joint_name))
            outcome_index = index.get((int(interval) + 1, joint_name))
            if source_index is not None:
                row[source_index] -= 1.0
            if outcome_index is not None:
                row[outcome_index] += 1.0
            baseline = float(
                reference_qpos[int(interval) + 1, qpos_address]
                - reference_qpos[int(interval), qpos_address]
            )
            rows.extend((row, -row))
            bounds.extend((
                limit * dt_s - baseline - float(row @ delta),
                limit * dt_s + baseline + float(row @ delta),
            ))
    return rows, bounds


def refine_native_support_path(
    model: Any,
    mujoco: Any,
    reference_qpos: np.ndarray,
    variables: Sequence[TrajectoryVariable],
    geoms: Sequence[NativeSupportGeom],
    *,
    intervals: Sequence[int],
    thresholds_m: Mapping[int, float],
    plane_z_m: float,
    modified_endpoints: Sequence[int],
    site_names: Sequence[str],
    velocity_limits: Mapping[str, float],
    dt_s: float,
    optimization_samples_per_interval: int,
    nonlinear_check_samples_per_interval: int,
    activation_margin_m: float,
    maximum_iterations: int,
    feasibility_tolerance_m: float,
    initial_trust_region_fraction: float,
    minimum_trust_region_fraction: float,
    maximum_trust_region_fraction: float,
    trust_region_shrink: float,
    trust_region_grow: float,
    weights: Mapping[str, float],
    fingertip_scale_m: float,
    qp_solver: str,
    freeze_dofs_from: int | None = 36,
) -> dict[str, Any]:
    """Refine a short trajectory with sequential native-geometry QPs."""
    from qpsolvers import solve_qp

    if set(int(value) for value in intervals) != set(thresholds_m):
        raise ValueError("every optimized interval needs one clearance threshold")
    if optimization_samples_per_interval < 2 or nonlinear_check_samples_per_interval < 2:
        raise ValueError("path sampling requires at least two points per interval")
    delta = np.zeros(len(variables), dtype=np.float64)
    budgets = np.asarray([item.budget for item in variables], dtype=np.float64)
    sample_alphas: dict[int, set[float]] = {
        int(interval): set(np.linspace(0.0, 1.0, optimization_samples_per_interval))
        for interval in intervals
    }
    check_alphas = np.linspace(0.0, 1.0, nonlinear_check_samples_per_interval)
    trust_fraction = float(initial_trust_region_fraction)
    trace: list[dict[str, Any]] = []

    def nonlinear_minima(values: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
        trajectory = apply_delta(reference_qpos, variables, values)
        minima = interval_clearance_minima(
            model,
            mujoco,
            trajectory,
            geoms,
            intervals=intervals,
            alphas=check_alphas,
            plane_z_m=plane_z_m,
            freeze_dofs_from=freeze_dofs_from,
        )
        return trajectory, minima

    current_qpos, current_minima = nonlinear_minima(delta)
    for iteration in range(int(maximum_iterations)):
        violations = {
            int(row["source_endpoint"]): (
                float(thresholds_m[int(row["source_endpoint"])])
                - float(row["distance_m"])
            )
            for row in current_minima
        }
        maximum_violation = max(violations.values())
        trace.append({
            "iteration": iteration,
            "stage": "nonlinear_state",
            "status": "FEASIBLE" if maximum_violation <= feasibility_tolerance_m else "INFEASIBLE",
            "maximum_violation_m": maximum_violation,
            "trust_region_fraction": trust_fraction,
            "qp_step_maximum_budget_fraction": 0.0,
            "interval_minima": current_minima,
        })
        if maximum_violation <= feasibility_tolerance_m:
            return {
                "status": "SOLVED",
                "delta": delta,
                "qpos": current_qpos,
                "interval_minima": current_minima,
                "trace": trace,
                "iterations": iteration,
            }

        hessian, gradient = _objective_terms(
            model,
            mujoco,
            reference_qpos,
            current_qpos,
            variables,
            delta,
            modified_endpoints=modified_endpoints,
            site_names=site_names,
            reference_weight=float(weights["reference_fidelity"]),
            smoothness_weight=float(weights["second_difference_smoothness"]),
            fingertip_weight=float(weights["fingertip_preservation"]),
            object_relative_fingertip_weight=float(
                weights["object_relative_fingertip_preservation"]
            ),
            fingertip_scale_m=fingertip_scale_m,
        )
        rows, bounds, immutable = _clearance_linearization_rows(
            model,
            mujoco,
            current_qpos,
            variables,
            geoms,
            sample_alphas={
                interval: sorted(values)
                for interval, values in sample_alphas.items()
            },
            thresholds_m=thresholds_m,
            plane_z_m=plane_z_m,
            activation_margin_m=activation_margin_m,
            feasibility_tolerance_m=feasibility_tolerance_m,
            freeze_dofs_from=freeze_dofs_from,
        )
        if immutable:
            return {
                "status": "IMMUTABLE_LINEARIZED_CONSTRAINT",
                "delta": delta,
                "qpos": current_qpos,
                "interval_minima": current_minima,
                "trace": trace,
                "immutable": immutable,
                "iterations": iteration,
            }
        velocity_rows, velocity_bounds = _velocity_constraint_rows(
            model,
            reference_qpos,
            variables,
            delta,
            intervals=intervals,
            velocity_limits=velocity_limits,
            dt_s=dt_s,
        )
        rows.extend(velocity_rows)
        bounds.extend(velocity_bounds)
        lower = np.maximum(-budgets - delta, -trust_fraction * budgets)
        upper = np.minimum(budgets - delta, trust_fraction * budgets)
        for index, variable in enumerate(variables):
            if not bool(model.jnt_limited[variable.joint_id]):
                continue
            joint_lower, joint_upper = np.asarray(
                model.jnt_range[variable.joint_id], dtype=np.float64
            )
            current = float(current_qpos[variable.endpoint, variable.qpos_address])
            lower[index] = max(lower[index], joint_lower - current)
            upper[index] = min(upper[index], joint_upper - current)
        step = solve_qp(
            hessian,
            gradient,
            G=np.asarray(rows, dtype=np.float64),
            h=np.asarray(bounds, dtype=np.float64),
            lb=lower,
            ub=upper,
            solver=qp_solver,
            primal_tol=1.0e-9,
            dual_tol=1.0e-9,
        )
        if step is None or not np.isfinite(step).all():
            trace.append({
                "iteration": iteration,
                "stage": "qp",
                "status": "NO_SOLUTION",
                "maximum_violation_m": maximum_violation,
                "trust_region_fraction": trust_fraction,
                "qp_step_maximum_budget_fraction": None,
                "interval_minima": current_minima,
            })
            trust_fraction *= float(trust_region_shrink)
            if trust_fraction < float(minimum_trust_region_fraction):
                break
            continue

        candidate_delta = delta + np.asarray(step, dtype=np.float64)
        candidate_qpos, candidate_minima = nonlinear_minima(candidate_delta)
        candidate_violations = [
            float(thresholds_m[int(row["source_endpoint"])])
            - float(row["distance_m"])
            for row in candidate_minima
        ]
        candidate_maximum_violation = max(candidate_violations)
        step_fraction = float(np.max(np.abs(step) / budgets))
        accepted = (
            candidate_maximum_violation < maximum_violation - 1.0e-9
            or candidate_maximum_violation <= feasibility_tolerance_m
        )
        trace.append({
            "iteration": iteration,
            "stage": "nonlinear_candidate",
            "status": "ACCEPTED" if accepted else "REJECTED",
            "maximum_violation_m": candidate_maximum_violation,
            "trust_region_fraction": trust_fraction,
            "qp_step_maximum_budget_fraction": step_fraction,
            "interval_minima": candidate_minima,
        })
        for row in candidate_minima:
            sample_alphas[int(row["source_endpoint"])].add(float(row["alpha"]))
        if accepted:
            delta = candidate_delta
            current_qpos = candidate_qpos
            current_minima = candidate_minima
            trust_fraction = min(
                float(maximum_trust_region_fraction),
                trust_fraction * float(trust_region_grow),
            )
        else:
            trust_fraction *= float(trust_region_shrink)
            if trust_fraction < float(minimum_trust_region_fraction):
                break

    return {
        "status": "MAXIMUM_ITERATIONS_OR_TRUST_REGION_EXHAUSTED",
        "delta": delta,
        "qpos": current_qpos,
        "interval_minima": current_minima,
        "trace": trace,
        "iterations": int(maximum_iterations),
    }


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
