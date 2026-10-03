"""Bounded control-aware startup planning for the frozen Pour experiment.

The numerical update is deliberately small: retain the current nominal plan,
add linearly interpolated Gaussian noise at fixed control knots, execute every
candidate in the audited CPU world, and retain the lowest-cost complete
forecast.  This is a local port of the predictive-sampling mechanism, not a
general MPC framework and not an RL path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
import yaml

from video_to_spider.rl.replay_contact_trace import (
    canonical_contact_group,
    classify_geom_role,
)

from .env import make_startup_world
from .state_io import (
    load_torch_gzip,
    manifest_entry,
    sha256,
    validate_physics_snapshot,
    verify_artifact,
    write_json,
    write_torch_gzip_atomic,
)


HORIZON = 40
DOF = 36
SUBSTEPS = 10
REPLAN_SOURCES = (0, 5, 10, 15)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "taco_pour_control_aware_startup_v1"


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _endpoint(world: Any) -> int:
    start = np.asarray(world.start_indices).reshape(-1)
    time = np.asarray(world.time_indices).reshape(-1)
    if start.size != 1 or time.size != 1:
        raise RuntimeError("startup planner requires one scalar world")
    return int(start[0] + time[0])


def knot_sources(source: int, *, end: int = HORIZON, interval: int = 5) -> np.ndarray:
    """Return the frozen global control-knot indices for one replan."""
    if source < 0 or source >= end or interval < 1:
        raise ValueError("invalid knot schedule")
    values = list(range(source, end, interval))
    values.append(end - 1)
    return np.asarray(sorted(set(values)), dtype=np.int32)


def make_noise_schedule(
    *, seed: int = 0, sources: tuple[int, ...] = REPLAN_SOURCES,
    rounds: int = 4, slots: int = 32, dof: int = DOF, std: float = 0.20,
) -> dict[tuple[int, int], np.ndarray]:
    """Pre-generate noise once so A and L consume identical samples."""
    if rounds < 1 or slots < 3 or dof < 1 or not np.isfinite(std) or std <= 0:
        raise ValueError("invalid noise schedule")
    rng = np.random.default_rng(seed)
    schedule = {}
    for source in sources:
        count = len(knot_sources(source))
        for round_index in range(rounds):
            schedule[(source, round_index)] = rng.normal(
                0.0, std, size=(slots - 2, count, dof)
            ).astype(np.float64)
    return schedule


def expand_knot_noise(source: int, knot_noise: np.ndarray) -> np.ndarray:
    """Linearly interpolate node noise over control indices, not substeps."""
    knots = knot_sources(source)
    noise = np.asarray(knot_noise, dtype=np.float64)
    if noise.shape != (len(knots), DOF) or not np.isfinite(noise).all():
        raise ValueError("knot noise shape or values changed")
    rows = np.arange(source, HORIZON, dtype=np.float64)
    expanded = np.empty((HORIZON - source, DOF), dtype=np.float64)
    for index in range(DOF):
        expanded[:, index] = np.interp(rows, knots, noise[:, index])
    return expanded


def project_plan(plan: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply the experiment's sole explicit float32 support projection."""
    value = np.asarray(plan)
    lower = np.asarray(low)
    upper = np.asarray(high)
    if value.shape != (HORIZON, DOF) or lower.shape != value.shape or upper.shape != value.shape:
        raise ValueError("plan/support arrays must be (40, 36)")
    if lower.dtype != np.float32 or upper.dtype != np.float32:
        raise ValueError("support must be float32")
    if not all(np.isfinite(row).all() for row in (value, lower, upper)):
        raise ValueError("plan/support must be finite")
    if bool((lower > upper).any()):
        raise ValueError("action support is empty")
    raw = value.astype(np.float64, copy=False)
    projected = np.clip(raw, lower.astype(np.float64), upper.astype(np.float64)).astype(np.float32)
    count = int(np.count_nonzero(projected.astype(np.float64) != raw))
    return projected, count


def make_round_candidates(
    nominal: np.ndarray, low: np.ndarray, high: np.ndarray, *, source: int,
    noise: np.ndarray, slots: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct nominal, Replay, and thirty noisy candidates in fixed order."""
    nominal = np.asarray(nominal)
    noise = np.asarray(noise, dtype=np.float64)
    if nominal.shape != (HORIZON, DOF) or nominal.dtype != np.float32:
        raise ValueError("nominal plan must be float32 (40, 36)")
    if noise.shape != (slots - 2, len(knot_sources(source)), DOF):
        raise ValueError("round noise does not match the frozen slots/knots")
    rows = np.empty((slots, HORIZON, DOF), dtype=np.float32)
    projection_counts = np.zeros(slots, dtype=np.int32)
    # Slot zero is byte-preserved, not reconstructed through a spline.
    rows[0] = nominal
    rows[1] = 0.0
    rows[1], projection_counts[1] = project_plan(rows[1], low, high)
    for slot in range(2, slots):
        proposal = nominal.astype(np.float64)
        proposal[source:] += expand_knot_noise(source, noise[slot - 2])
        rows[slot], projection_counts[slot] = project_plan(proposal, low, high)
    if rows[0].tobytes() != nominal.tobytes():
        raise RuntimeError("nominal slot bytes changed")
    return rows, projection_counts


def improve_plan(
    nominal: np.ndarray, low: np.ndarray, high: np.ndarray, *, source: int,
    noise: np.ndarray,
    evaluator: Callable[[np.ndarray, int, int], "Forecast"],
) -> tuple[np.ndarray, "Forecast", list["Forecast"], np.ndarray]:
    """Evaluate one frozen 32-slot round and retain the best complete plan."""
    candidates, projection_counts = make_round_candidates(
        nominal, low, high, source=source, noise=noise,
    )
    results = [
        evaluator(candidates[slot], slot, int(projection_counts[slot]))
        for slot in range(len(candidates))
    ]
    complete = [(row.cost, slot, row) for slot, row in enumerate(results) if row.complete]
    if not complete:
        raise RuntimeError(f"NO_COMPLETE_FORECAST_AT_SOURCE_{source}")
    _, _, winner = min(complete, key=lambda item: (item[0], item[1]))
    return winner.sequence.copy(), winner, results, projection_counts


def _quat_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid quaternion")
    w, x, y, z = q / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def rotation_error_rad(actual: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


def object_pose(qpos: np.ndarray, role: str) -> tuple[np.ndarray, np.ndarray]:
    offset = {"tool": 36, "target": 43}[role]
    value = np.asarray(qpos)
    return value[offset:offset + 3].astype(np.float64), value[offset + 3:offset + 7].astype(np.float64)


def pair_pose(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tool_p, tool_q = object_pose(qpos, "tool")
    target_p, target_q = object_pose(qpos, "target")
    target_r = _quat_matrix_wxyz(target_q)
    relative_p = target_r.T @ (tool_p - target_p)
    relative_r = target_r.T @ _quat_matrix_wxyz(tool_q)
    return relative_p, relative_r


def matrix_rotation_error_rad(actual: np.ndarray, reference: np.ndarray) -> float:
    relative = np.asarray(reference).T @ np.asarray(actual)
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


def reference_tip_positions(model: mujoco.MjModel, reference_qpos: np.ndarray) -> np.ndarray:
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_tip")
        for side in ("right", "left") for finger in FINGERS
    ]
    if any(index < 0 for index in ids):
        raise ValueError("reference scene is missing calibrated fingertip sites")
    data = mujoco.MjData(model)
    result = np.empty((len(reference_qpos), 10, 3), dtype=np.float64)
    for endpoint, qpos in enumerate(np.asarray(reference_qpos)):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        result[endpoint] = data.site_xpos[ids]
    return result


def hand_relation_error(
    qpos: np.ndarray, tips: np.ndarray, reference_qpos: np.ndarray,
    reference_tips: np.ndarray,
) -> float:
    values = []
    for hand, role in ((slice(0, 5), "tool"), (slice(5, 10), "target")):
        p, q = object_pose(qpos, role)
        ref_p, ref_q = object_pose(reference_qpos, role)
        local = (np.asarray(tips[hand]) - p) @ _quat_matrix_wxyz(q)
        local_ref = (np.asarray(reference_tips[hand]) - ref_p) @ _quat_matrix_wxyz(ref_q)
        values.extend(np.sum(np.square(local - local_ref), axis=1).tolist())
    return float(np.mean(values) / (0.030 ** 2))


def step_cost_terms(
    *, qpos: np.ndarray, reference_qpos: np.ndarray, tips: np.ndarray,
    reference_tips: np.ndarray, action: np.ndarray, ctrl: np.ndarray,
    previous_ctrl: np.ndarray,
) -> dict[str, float]:
    object_terms = []
    output: dict[str, float] = {}
    for role in ("tool", "target"):
        p, q = object_pose(qpos, role)
        rp, rq = object_pose(reference_qpos, role)
        position = float(np.linalg.norm(p - rp))
        rotation = rotation_error_rad(q, rq)
        output[f"{role}_position_error_m"] = position
        output[f"{role}_rotation_error_rad"] = rotation
        object_terms.append((position / 0.020) ** 2 + (rotation / 0.20) ** 2)
    d_object = 0.5 * float(sum(object_terms))
    pair_p, pair_r = pair_pose(qpos)
    ref_pair_p, ref_pair_r = pair_pose(reference_qpos)
    pair_translation = float(np.linalg.norm(pair_p - ref_pair_p))
    pair_rotation = matrix_rotation_error_rad(pair_r, ref_pair_r)
    d_pair = (pair_translation / 0.020) ** 2 + (pair_rotation / 0.20) ** 2
    d_hand = hand_relation_error(qpos, tips, reference_qpos, reference_tips)
    a = np.asarray(action, dtype=np.float64)
    u = np.asarray(ctrl, dtype=np.float64)
    u_prev = np.asarray(previous_ctrl, dtype=np.float64)
    d_control = float(np.mean(np.square(a)) + np.mean(np.square((u - u_prev) / 0.05)))
    running = d_object + d_pair + 0.05 * d_hand + 0.01 * d_control
    output.update({
        "D_obj": d_object, "D_pair": d_pair, "D_hand": d_hand,
        "D_control": d_control, "L": running,
        "pair_translation_error_m": pair_translation,
        "pair_rotation_error_rad": pair_rotation,
    })
    return output


def trajectory_cost(rows: list[dict[str, float]]) -> tuple[float, dict[str, float]]:
    if not rows:
        raise ValueError("trajectory cost requires at least one complete outcome")
    running = float(np.mean([row["L"] for row in rows]))
    terminal = float(rows[-1]["D_obj"] + rows[-1]["D_pair"])
    return running + terminal, {
        "mean_D_obj": float(np.mean([row["D_obj"] for row in rows])),
        "mean_D_pair": float(np.mean([row["D_pair"] for row in rows])),
        "mean_D_hand": float(np.mean([row["D_hand"] for row in rows])),
        "mean_D_control": float(np.mean([row["D_control"] for row in rows])),
        "mean_L": running, "terminal_D_obj_plus_D_pair": terminal,
    }


@dataclass
class PhysicsBudget:
    limits: Mapping[str, int]
    all_in_limit: int
    physics: dict[str, int] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.limits = {str(k): int(v) for k, v in self.limits.items()}
        self.physics = {name: 0 for name in self.limits}
        self.attempts = {name: 0 for name in self.limits}

    @property
    def total(self) -> int:
        return int(sum(self.physics.values()))

    def reserve(self, phase: str, maximum: int) -> None:
        if phase not in self.limits or maximum < 0:
            raise ValueError("invalid budget reservation")
        if self.physics[phase] + maximum > self.limits[phase]:
            raise RuntimeError(f"{phase} budget lacks complete-candidate capacity")
        if self.total + maximum > self.all_in_limit:
            raise RuntimeError("all-in budget lacks complete-candidate capacity")
        self.attempts[phase] += 1

    def charge(self, phase: str, steps: int) -> None:
        if steps < 0 or self.physics[phase] + steps > self.limits[phase]:
            raise RuntimeError(f"{phase} physics budget exceeded")
        if self.total + steps > self.all_in_limit:
            raise RuntimeError("all-in physics budget exceeded")
        self.physics[phase] += steps

    def report(self) -> dict[str, Any]:
        return {
            "physics_steps": dict(self.physics), "attempts": dict(self.attempts),
            "total_physics_steps": self.total, "phase_limits": dict(self.limits),
            "all_in_limit": self.all_in_limit,
        }


@dataclass
class Forecast:
    complete: bool
    sequence: np.ndarray
    source: int
    cost: float
    components: dict[str, float]
    first_failure_endpoint: int | None
    projection_count: int
    actions: np.ndarray
    source_endpoint: np.ndarray
    outcome_endpoint: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    tips: np.ndarray
    tracking_score: np.ndarray
    cost_rows: tuple[dict[str, float], ...]
    ctrl_loss_max: float


def support_table(world: Any) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(world.time_indices).copy()
    start = np.asarray(world.start_indices).copy()
    if start.shape != (1,) or int(start[0]) != 0:
        raise RuntimeError("startup action support requires absolute endpoint cursor")
    low = np.empty((HORIZON, DOF), dtype=np.float32)
    high = np.empty_like(low)
    try:
        for source in range(HORIZON):
            world.time_indices[:] = source
            lo, hi = world.current_normalized_action_bounds()
            low[source] = _numpy(lo).reshape(1, -1)[0]
            high[source] = _numpy(hi).reshape(1, -1)[0]
    finally:
        world.time_indices = original
    if bool((low > 0).any() or (high < 0).any()):
        raise RuntimeError("zero Replay is outside startup action support")
    return low, high


def _world_row(world: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qpos = _numpy(world._mjwp.get_qpos(world.ego_cfg, world.env)).reshape(1, -1)[0].copy()
    qvel = _numpy(world._mjwp.get_qvel(world.ego_cfg, world.env)).reshape(1, -1)[0].copy()
    # Read the simulator's post-ctrlrange target, not the pre-clip request
    # cached in ``_last_ctrl``.
    ctrl = wp.to_torch(world.env.data_wp.ctrl).cpu().numpy().reshape(1, -1)[0].copy()
    site_xpos = wp.to_torch(world.env.data_wp.site_xpos).cpu().numpy()
    tips = site_xpos[0, list(world.env_cfg.fingertip_site_ids)].copy()
    return qpos, qvel, ctrl, tips


def evaluate_plan(
    world: Any, boundary: dict[str, Any], sequence: np.ndarray,
    low: np.ndarray, high: np.ndarray, reference_qpos: np.ndarray,
    reference_ctrl: np.ndarray, reference_tips: np.ndarray,
    budget: PhysicsBudget, *, source: int, projection_count: int = 0,
    phase: str = "search", observer: Callable[[int], None] | None = None,
    stop_endpoint: int = HORIZON,
) -> Forecast:
    validate_physics_snapshot(boundary)
    plan = np.asarray(sequence)
    if plan.shape != (HORIZON, DOF) or plan.dtype != np.float32:
        raise ValueError("executed startup plan must be float32 (40, 36)")
    if bool((plan < low).any() or (plan > high).any()):
        raise ValueError("startup plan exceeds original action support")
    if not source < stop_endpoint <= HORIZON:
        raise ValueError("invalid forecast stop endpoint")
    maximum = (stop_endpoint - source) * SUBSTEPS
    budget.reserve(phase, maximum)
    world.set_env_state(boundary)
    if _endpoint(world) != source:
        raise RuntimeError("forecast restored the wrong source endpoint")
    qpos_rows, qvel_rows, ctrl_rows, tip_rows = [], [], [], []
    score_rows, sources, outcomes, cost_rows = [], [], [], []
    actions = []
    previous_ctrl = _numpy(boundary["ctrl"]).reshape(1, -1)[0].copy()
    failure = None
    maximum_ctrl_loss = 0.0
    for endpoint_source in range(source, stop_endpoint):
        live_low, live_high = world.current_normalized_action_bounds()
        live_low = _numpy(live_low).reshape(1, -1)[0].astype(np.float32, copy=False)
        live_high = _numpy(live_high).reshape(1, -1)[0].astype(np.float32, copy=False)
        if live_low.tobytes() != low[endpoint_source].tobytes() or live_high.tobytes() != high[endpoint_source].tobytes():
            raise RuntimeError("live action support differs from frozen support table")
        base = world._reference_ctrls(world.time_indices, offset=1)
        requested = world._apply_residual(base, torch.as_tensor(plan[endpoint_source:endpoint_source + 1]))
        try:
            if observer is not None:
                observer(endpoint_source)
            _, _, done, info = world.step(
                plan[endpoint_source:endpoint_source + 1], auto_reset=False,
                **({"substep_observer": observer.observer} if observer is not None else {}),
            )
        finally:
            budget.charge(phase, SUBSTEPS)
        outcome = endpoint_source + 1
        if (
            int(info["source_reference_endpoint"][0]) != endpoint_source
            or int(info["outcome_reference_endpoint"][0]) != outcome
            or int(info["command_reference_endpoint"][0]) != outcome
            or int(info["reward_reference_endpoint"][0]) != outcome
        ):
            raise RuntimeError("startup reward/command endpoint contract changed")
        qpos, qvel, ctrl, tips = _world_row(world)
        ctrl_loss = float(np.max(np.abs(ctrl.astype(np.float64) - _numpy(requested)[0].astype(np.float64))))
        maximum_ctrl_loss = max(maximum_ctrl_loss, ctrl_loss)
        if ctrl_loss != 0.0:
            raise RuntimeError(
                f"undeclared residual/ctrlrange clipping at source {endpoint_source}: {ctrl_loss}"
            )
        row = step_cost_terms(
            qpos=qpos, reference_qpos=reference_qpos[outcome], tips=tips,
            reference_tips=reference_tips[outcome], action=plan[endpoint_source],
            ctrl=ctrl, previous_ctrl=previous_ctrl,
        )
        previous_ctrl = ctrl
        qpos_rows.append(qpos); qvel_rows.append(qvel); ctrl_rows.append(ctrl); tip_rows.append(tips)
        score_rows.append(float(info["object_tracking_error"][0]))
        sources.append(endpoint_source); outcomes.append(outcome); actions.append(plan[endpoint_source].copy())
        cost_rows.append(row)
        terminated = bool(info["terminated"][0])
        if terminated:
            failure = outcome
            break
        if bool(done[0]) and not bool(info["time_outs"][0]):
            raise RuntimeError("startup rollout ended without tracking termination")
    complete = (
        failure is None and len(outcomes) == stop_endpoint - source
        and outcomes[-1] == stop_endpoint
    )
    cost, components = trajectory_cost(cost_rows) if complete else (float("inf"), {})
    return Forecast(
        complete=complete, sequence=plan.copy(), source=source, cost=cost,
        components=components, first_failure_endpoint=failure,
        projection_count=int(projection_count), actions=np.asarray(actions, dtype=np.float32),
        source_endpoint=np.asarray(sources, dtype=np.int32),
        outcome_endpoint=np.asarray(outcomes, dtype=np.int32),
        qpos=np.asarray(qpos_rows, dtype=np.float32),
        qvel=np.asarray(qvel_rows, dtype=np.float32),
        ctrl=np.asarray(ctrl_rows, dtype=np.float32),
        tips=np.asarray(tip_rows, dtype=np.float32),
        tracking_score=np.asarray(score_rows, dtype=np.float64),
        cost_rows=tuple(cost_rows), ctrl_loss_max=maximum_ctrl_loss,
    )


def _array_digest(array: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _snapshot_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or not torch.equal(a, b):
                return False
        elif isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                return False
        elif a != b:
            return False
    return True


def _model_roles(model: mujoco.MjModel) -> dict[str, Any]:
    def object_id(kind: mujoco.mjtObj, name: str) -> int:
        value = int(mujoco.mj_name2id(model, kind, name))
        if value < 0:
            raise RuntimeError(f"model is missing {name}")
        return value
    return {
        "right_root": object_id(mujoco.mjtObj.mjOBJ_BODY, "right_hand_link"),
        "left_root": object_id(mujoco.mjtObj.mjOBJ_BODY, "left_hand_link"),
        "tool_root": object_id(mujoco.mjtObj.mjOBJ_BODY, "right_object"),
        "target_root": object_id(mujoco.mjtObj.mjOBJ_BODY, "left_object"),
        "floor_geom": 0,
        "body_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(model.nbody)],
        "geom_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}" for i in range(model.ngeom)],
        "body_parentid": model.body_parentid.copy(), "geom_bodyid": model.geom_bodyid.copy(),
    }


class StartupObserver:
    """Record solve-aligned substeps and active contact wrenches."""

    def __init__(self, world: Any) -> None:
        self.world = world
        self.rows: list[dict[str, Any]] = []
        self.contacts: list[dict[str, Any]] = []
        self.source = -1
        self.substep_base = 0
        self.roles = _model_roles(world.env.model_cpu)

    def __call__(self, source: int) -> None:
        self.source = int(source)
        self.substep_base = len(self.rows)

    def observer(self, substep: int) -> None:
        qpos, qvel, ctrl, _ = _world_row(self.world)
        data = self.world.env.data_wp
        nacon = int(wp.to_torch(data.nacon).cpu().numpy()[0])
        local = np.zeros((nacon, 6), dtype=np.float32)
        world_force = np.zeros((nacon, 6), dtype=np.float32)
        if nacon:
            ids = wp.array(np.arange(nacon, dtype=np.int32), dtype=wp.int32, device=self.world.env.device)
            local_wp = wp.zeros(nacon, dtype=wp.spatial_vector, device=self.world.env.device)
            world_wp = wp.zeros(nacon, dtype=wp.spatial_vector, device=self.world.env.device)
            mjwarp.contact_force(self.world.env.model_wp, data, ids, False, local_wp)
            mjwarp.contact_force(self.world.env.model_wp, data, ids, True, world_wp)
            wp.synchronize(); local = local_wp.numpy().copy(); world_force = world_wp.numpy().copy()
        contact_geom = wp.to_torch(data.contact.geom).cpu().numpy()[:nacon]
        contact_pos = wp.to_torch(data.contact.pos).cpu().numpy()[:nacon]
        contact_dist = wp.to_torch(data.contact.dist).cpu().numpy()[:nacon]
        global_substep = self.source * SUBSTEPS + int(substep) + 1
        for contact_id in range(nacon):
            geom1, geom2 = map(int, contact_geom[contact_id])
            kwargs = dict(
                geom_body=self.roles["geom_bodyid"], body_parents=self.roles["body_parentid"],
                body_names=self.roles["body_names"], right_root=self.roles["right_root"],
                left_root=self.roles["left_root"], tool_root=self.roles["tool_root"],
                target_root=self.roles["target_root"], floor_geom=self.roles["floor_geom"],
            )
            role1 = classify_geom_role(geom1, **kwargs); role2 = classify_geom_role(geom2, **kwargs)
            self.contacts.append({
                "global_substep": global_substep, "source_endpoint": self.source,
                "outcome_endpoint": self.source + 1, "substep": int(substep) + 1,
                "contact_id": contact_id, "geom1": geom1, "geom2": geom2,
                "geom1_name": self.roles["geom_names"][geom1], "geom2_name": self.roles["geom_names"][geom2],
                "role1": role1, "role2": role2, "group": canonical_contact_group(role1, role2),
                "pos": contact_pos[contact_id].copy(), "dist": float(contact_dist[contact_id]),
                "wrench_contact_force_torque": local[contact_id].copy(),
                "wrench_world_force_torque_on_geom2": world_force[contact_id].copy(),
            })
        self.rows.append({
            "global_substep": global_substep, "source_endpoint": self.source,
            "outcome_endpoint": self.source + 1, "substep": int(substep) + 1,
            "qpos": qpos, "qvel": qvel, "ctrl": ctrl, "nacon": nacon,
        })


def _save_observer(observer: StartupObserver, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if observer.rows:
        np.savez_compressed(directory / "substeps.npz", **{
            key: np.stack([np.asarray(row[key]) for row in observer.rows]) for key in observer.rows[0]
        })
    else:
        np.savez_compressed(directory / "substeps.npz")
    numeric = {}
    for key in ("global_substep", "source_endpoint", "outcome_endpoint", "substep", "contact_id", "geom1", "geom2", "pos", "dist", "wrench_contact_force_torque", "wrench_world_force_torque_on_geom2"):
        numeric[key] = np.stack([np.asarray(row[key]) for row in observer.contacts]) if observer.contacts else np.empty((0,))
    for key in ("geom1_name", "geom2_name", "role1", "role2", "group"):
        numeric[key] = np.asarray([row[key] for row in observer.contacts])
    np.savez_compressed(directory / "contacts_raw.npz", **numeric)
    with (directory / "contact_forces.csv").open("w", newline="") as stream:
        fields = ("global_substep", "source_endpoint", "outcome_endpoint", "substep", "geom1_name", "geom2_name", "role1", "role2", "group", "dist_m", "normal_force_N")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in observer.contacts:
            writer.writerow({
                "global_substep": row["global_substep"], "source_endpoint": row["source_endpoint"],
                "outcome_endpoint": row["outcome_endpoint"], "substep": row["substep"],
                "geom1_name": row["geom1_name"], "geom2_name": row["geom2_name"],
                "role1": row["role1"], "role2": row["role2"], "group": row["group"],
                "dist_m": row["dist"], "normal_force_N": row["wrench_contact_force_torque"][0],
            })


def _save_forecast_group(path: Path, forecasts: list[Forecast], rounds: np.ndarray, slots: np.ndarray) -> None:
    maximum = max((len(row.outcome_endpoint) for row in forecasts), default=0)
    count = len(forecasts)
    def padded(shape, dtype, fill): return np.full((count, maximum, *shape), fill, dtype=dtype)
    qpos = padded((50,), np.float32, np.nan); qvel = padded((48,), np.float32, np.nan)
    ctrl = padded((36,), np.float32, np.nan); actions = padded((36,), np.float32, np.nan)
    score = padded((), np.float64, np.nan); outcomes = padded((), np.int32, -1)
    mask = padded((), np.bool_, False)
    for index, row in enumerate(forecasts):
        n = len(row.outcome_endpoint); mask[index, :n] = True
        qpos[index, :n] = row.qpos; qvel[index, :n] = row.qvel; ctrl[index, :n] = row.ctrl
        actions[index, :n] = row.actions; score[index, :n] = row.tracking_score; outcomes[index, :n] = row.outcome_endpoint
    component_names = sorted({name for row in forecasts for name in row.components})
    component_arrays = {
        f"cost_{name}": np.asarray([row.components.get(name, np.nan) for row in forecasts], dtype=np.float64)
        for name in component_names
    }
    np.savez_compressed(
        path, sequence=np.stack([row.sequence for row in forecasts]), round=rounds,
        slot=slots, complete=np.asarray([row.complete for row in forecasts]),
        cost=np.asarray([row.cost for row in forecasts]),
        first_failure_endpoint=np.asarray([-1 if row.first_failure_endpoint is None else row.first_failure_endpoint for row in forecasts]),
        projection_count=np.asarray([row.projection_count for row in forecasts]),
        ctrl_loss_max=np.asarray([row.ctrl_loss_max for row in forecasts]),
        executed_mask=mask, outcome_endpoint=outcomes, action=actions, qpos=qpos,
        qvel=qvel, ctrl=ctrl, tracking_score=score, **component_arrays,
    )


def _load_config(config_path: Path, asset_root: Path | None) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema") != SCHEMA or config.get("status") != "authorized_single_run":
        raise ValueError("startup planning is not authorized")
    if config.get("baseline", {}).get("commit") != "0a51b1624189770022c369a62b3dbc265b1aea56":
        raise ValueError("startup baseline changed")
    root = (asset_root or Path(config["asset_root"])).resolve(strict=True)
    assets = {}
    for name, row in config["assets"].items():
        candidate = root / row["path"]
        assets[name] = verify_artifact(candidate, row["sha256"]) if row.get("sha256") else candidate.resolve(strict=True)
    expected_planner = {
        "replan_sources": [0, 5, 10, 15], "terminal_reference_endpoint": 40,
        "execute_controls_per_replan": 5,
        "suffix_20_to_40": "frozen_from_last_plan_no_more_optimization",
        "rounds_per_replan": 4, "slots_per_round": 32, "nominal_slot": 0,
        "zero_residual_replay_slot": 1, "noisy_slots": 30, "seed": 0,
        "rng": "numpy_default_rng_separate_from_world_and_torch",
        "noise_shared_between_initial_states": True, "knot_interval_controls": 5,
        "final_knot_source": 39, "interpolation": "linear_over_control_indices",
        "normalized_noise_std": 0.20, "residual_scale": 0.05,
        "candidate_projection": "once_to_original_float32_support",
        "preserve_nominal_actual_action_bytes": True,
        "complete_forecast_required_for_execution": True,
        "all_failed": "stop_arm_without_extra_sampling", "terminate_resampling": False,
        "covariance_refitting": False, "averaging_candidate_actions": False,
    }
    if config.get("planner") != expected_planner:
        raise ValueError("startup planner contract changed")
    return config, root, assets


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def _require_clean_and_baseline(config: dict[str, Any]) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip():
        raise RuntimeError("startup planning requires a clean worktree")
    baseline = config["baseline"]["commit"]
    if subprocess.run(["git", "merge-base", "--is-ancestor", baseline, _git_head()], cwd=PROJECT_ROOT, check=False).returncode:
        raise RuntimeError("required startup baseline is not an ancestor")


def preflight(config_path: Path, asset_root: Path | None, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable startup output exists: {output}")
    config, root, assets = _load_config(config_path, asset_root)
    _require_clean_and_baseline(config)
    for name in ("a_snapshot", "l_snapshot"):
        snapshot = load_torch_gzip(assets[name]); validate_physics_snapshot(snapshot)
        if len(snapshot) != 362 or len(snapshot["warp_state_keys"]) != 342:
            raise ValueError(f"{name} complete snapshot field contract changed")
        if int(snapshot["start_indices"][0] + snapshot["time_indices"][0]) != 0:
            raise ValueError(f"{name} is not an endpoint-0 snapshot")
    parity = json.loads(assets["replay_trace_parity"].read_text())
    left_status = json.loads(assets["left_alignment_status"].read_text())
    left_parity = json.loads(assets["left_alignment_parity"].read_text())
    if not parity.get("B_endpoint_arrays_bitwise_equal_historical"):
        raise ValueError("A Replay identity is not established")
    if (
        not left_parity.get("all_bitwise_equal")
        or left_status.get("status") != "complete_report_only_no_promotion"
        or left_status.get("conclusion") != "MIXED_EARLY_RESPONSE_WITH_ENDPOINT20_POSITION_ROTATION_IMPROVEMENT"
    ):
        raise ValueError("L cold Replay identity is not established")
    source_rows = {}
    for name, row in config["upstream"].items():
        checkout = Path(row["checkout"]).resolve(strict=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        if commit != row["commit"] or sha256(checkout / "LICENSE") != row["license_sha256"]:
            raise ValueError(f"upstream pin changed: {name}")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True).strip():
            raise ValueError(f"upstream checkout is dirty: {name}")
        source_rows[name] = {"path": str(checkout), "commit": commit, "license": manifest_entry(checkout / "LICENSE")}
    output.mkdir(parents=True)
    manifest = {
        "schema": f"{SCHEMA}_input_manifest", "status": "PREFLIGHT_COMPLETE",
        "baseline_commit": config["baseline"]["commit"], "implementation_commit": _git_head(),
        "config": manifest_entry(config_path), "asset_root": str(root),
        "assets": {name: manifest_entry(path) for name, path in assets.items()},
        "upstream_sources": source_rows,
        "snapshot_contract": {"field_count": 362, "warp_state_field_count": 342},
        "authorization": config["authorization"],
    }
    write_json(output / "input_manifest.json", manifest)
    write_json(output / "status.json", {"schema": SCHEMA, "status": "PREFLIGHT_COMPLETE", "implementation_commit": _git_head()})
    return manifest


def _make_world(assets: dict[str, Path], snapshot: dict[str, Any]) -> Any:
    return make_startup_world(
        simulator_config=assets["simulator"], protocol=assets["protocol"],
        objective_profile=assets["objective"], observation_profile=assets["observation"],
        action_profile=assets["action"], boundary=snapshot, seed=0,
    )


def _forecast_summary(row: Forecast, *, round_index: int, slot: int) -> dict[str, Any]:
    return {
        "round": round_index, "slot": slot, "complete": row.complete,
        "cost": row.cost if np.isfinite(row.cost) else None,
        "components": row.components, "first_failure_endpoint": row.first_failure_endpoint,
        "projection_count": row.projection_count, "ctrl_loss_max": row.ctrl_loss_max,
        "sequence_sha256": _array_digest(row.sequence), "executed_controls": len(row.outcome_endpoint),
    }


def _condition_arrays(initial: dict[str, Any], forecasts: list[Forecast]) -> dict[str, np.ndarray]:
    qpos0 = _numpy(initial["qpos"]).reshape(1, -1)[0].astype(np.float32)
    qvel0 = _numpy(initial["qvel"]).reshape(1, -1)[0].astype(np.float32)
    ctrl0 = _numpy(initial["ctrl"]).reshape(1, -1)[0].astype(np.float32)
    return {
        "endpoint": np.asarray([0, *[int(row.outcome_endpoint[-1]) for row in forecasts]], dtype=np.int32),
        "qpos": np.vstack([qpos0, *[row.qpos[-1] for row in forecasts]]).astype(np.float32),
        "qvel": np.vstack([qvel0, *[row.qvel[-1] for row in forecasts]]).astype(np.float32),
        "ctrl": np.vstack([ctrl0, *[row.ctrl[-1] for row in forecasts]]).astype(np.float32),
    }


def _run_sequence(
    world: Any, initial: dict[str, Any], sequence: np.ndarray, low: np.ndarray, high: np.ndarray,
    reference_qpos: np.ndarray, reference_ctrl: np.ndarray, reference_tips: np.ndarray,
    budget: PhysicsBudget, phase: str, *, observer: StartupObserver | None,
) -> Forecast:
    return evaluate_plan(
        world, initial, sequence, low, high, reference_qpos, reference_ctrl,
        reference_tips, budget, source=0, phase=phase, observer=observer,
    )


def run_startup_plan(
    label: str, world: Any, initial: dict[str, Any], low: np.ndarray, high: np.ndarray,
    reference_qpos: np.ndarray, reference_ctrl: np.ndarray, reference_tips: np.ndarray,
    noise: dict[tuple[int, int], np.ndarray], budget: PhysicsBudget, output: Path,
) -> tuple[Forecast, np.ndarray, dict[str, Any]]:
    nominal = np.zeros((HORIZON, DOF), dtype=np.float32)
    boundary = initial
    execution_parts: list[Forecast] = []
    selection_rows = []
    arm_dir = output / label
    arm_dir.mkdir(parents=True)
    observer = StartupObserver(world)
    executed_sequence = np.zeros((HORIZON, DOF), dtype=np.float32)
    for source in REPLAN_SOURCES:
        source_dir = arm_dir / "search" / f"source_{source:02d}"
        source_dir.mkdir(parents=True)
        all_forecasts: list[Forecast] = []
        all_rounds: list[int] = []
        all_slots: list[int] = []
        winner: Forecast | None = None
        for round_index in range(4):
            round_rows = []
            def evaluator(candidate: np.ndarray, slot: int, projected: int) -> Forecast:
                return evaluate_plan(
                    world, boundary, candidate, low, high, reference_qpos,
                    reference_ctrl, reference_tips, budget, source=source,
                    projection_count=projected, phase="search",
                )
            # Keep the callback explicit so every slot is a real rollout.
            candidates, projected = make_round_candidates(
                nominal, low, high, source=source, noise=noise[(source, round_index)]
            )
            results = [evaluator(candidates[slot], slot, int(projected[slot])) for slot in range(32)]
            complete_rows = []
            for slot, result in enumerate(results):
                all_forecasts.append(result); all_rounds.append(round_index); all_slots.append(slot)
                row = _forecast_summary(result, round_index=round_index, slot=slot)
                round_rows.append(row)
                if result.complete:
                    complete_rows.append((result.cost, slot, result))
            if not complete_rows:
                write_json(source_dir / f"round_{round_index}.json", {"status": "NO_COMPLETE_FORECAST", "candidates": round_rows})
                _save_forecast_group(source_dir / "candidates.npz", all_forecasts, np.asarray(all_rounds), np.asarray(all_slots))
                raise RuntimeError(f"NO_COMPLETE_FORECAST_AT_SOURCE_{source}")
            _, selected_slot, winner = min(complete_rows, key=lambda item: (item[0], item[1]))
            nominal = winner.sequence.copy()
            selection_rows.append({"source": source, "round": round_index, "selected_slot": selected_slot, **_forecast_summary(winner, round_index=round_index, slot=selected_slot)})
            write_json(source_dir / f"round_{round_index}.json", {"status": "COMPLETE", "selected_slot": selected_slot, "candidates": round_rows})
        assert winner is not None
        _save_forecast_group(source_dir / "candidates.npz", all_forecasts, np.asarray(all_rounds), np.asarray(all_slots))
        write_torch_gzip_atomic(source_dir / "source_snapshot.pt.gz", boundary)
        # Execute the five-step selected prefix from the same full state.
        stop = HORIZON if source == 15 else source + 5
        executed = evaluate_plan(
            world, boundary, nominal, low, high, reference_qpos, reference_ctrl,
            reference_tips, budget, source=source, phase="execution", observer=observer,
            stop_endpoint=stop,
        )
        n = stop - source
        if not executed.complete:
            raise RuntimeError(f"selected plan terminated during real execution at {executed.first_failure_endpoint}")
        parity = {
            "qpos": executed.qpos.tobytes() == winner.qpos[:n].tobytes(),
            "qvel": executed.qvel.tobytes() == winner.qvel[:n].tobytes(),
            "ctrl": executed.ctrl.tobytes() == winner.ctrl[:n].tobytes(),
            "actions": executed.actions.tobytes() == winner.actions[:n].tobytes(),
        }
        if not all(parity.values()):
            raise RuntimeError(f"selected forecast/execution prefix mismatch at source {source}: {parity}")
        execution_parts.append(executed)
        executed_sequence[source:stop] = nominal[source:stop]
        boundary = world.get_env_state()
        write_torch_gzip_atomic(arm_dir / "execution" / f"endpoint_{stop:02d}.pt.gz", boundary)
        write_json(arm_dir / "execution" / f"source_{source:02d}_parity.json", parity)
    actions = np.concatenate([row.actions for row in execution_parts])
    sources = np.concatenate([row.source_endpoint for row in execution_parts])
    outcomes = np.concatenate([row.outcome_endpoint for row in execution_parts])
    qpos = np.concatenate([row.qpos for row in execution_parts])
    qvel = np.concatenate([row.qvel for row in execution_parts])
    ctrl = np.concatenate([row.ctrl for row in execution_parts])
    tips = np.concatenate([row.tips for row in execution_parts])
    scores = np.concatenate([row.tracking_score for row in execution_parts])
    rows = tuple(item for part in execution_parts for item in part.cost_rows)
    if not np.array_equal(sources, np.arange(HORIZON)) or not np.array_equal(outcomes, np.arange(1, HORIZON + 1)):
        raise RuntimeError("real startup execution is not one uninterrupted 0->40 ancestry")
    total_cost, components = trajectory_cost(list(rows))
    combined = Forecast(
        complete=True, sequence=executed_sequence, source=0, cost=total_cost,
        components=components, first_failure_endpoint=None,
        projection_count=sum(row.projection_count for row in execution_parts),
        actions=actions, source_endpoint=sources, outcome_endpoint=outcomes,
        qpos=qpos, qvel=qvel, ctrl=ctrl, tips=tips, tracking_score=scores,
        cost_rows=rows, ctrl_loss_max=max(row.ctrl_loss_max for row in execution_parts),
    )
    condition = _condition_arrays(initial, execution_parts)
    np.savez_compressed(
        arm_dir / "trajectory.npz", endpoint=np.arange(HORIZON + 1, dtype=np.int32),
        qpos=np.vstack([_numpy(initial["qpos"]).reshape(1, -1), qpos]).astype(np.float32),
        qvel=np.vstack([_numpy(initial["qvel"]).reshape(1, -1), qvel]).astype(np.float32),
        ctrl=np.vstack([_numpy(initial["ctrl"]).reshape(1, -1), ctrl]).astype(np.float32),
        action=executed_sequence, tracking_score=scores, tips=tips,
        **{key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]},
    )
    _save_observer(observer, arm_dir)
    write_json(arm_dir / "selection.json", {
        "condition": label, "round_selections": selection_rows,
        "final_cost": total_cost, "final_components": components,
        "sequence_sha256": _array_digest(executed_sequence),
        "ctrl_loss_max": combined.ctrl_loss_max,
    })
    del condition
    return combined, executed_sequence, {
        "round_selections": selection_rows, "terminal_snapshot": boundary,
        "observer": observer,
    }


def _save_condition(
    directory: Path, initial: dict[str, Any], result: Forecast,
    observer: StartupObserver, *, label: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    qpos0 = _numpy(initial["qpos"]).reshape(1, -1).astype(np.float32)
    qvel0 = _numpy(initial["qvel"]).reshape(1, -1).astype(np.float32)
    ctrl0 = _numpy(initial["ctrl"]).reshape(1, -1).astype(np.float32)
    np.savez_compressed(
        directory / "trajectory.npz",
        endpoint=np.asarray([0, *result.outcome_endpoint.tolist()], dtype=np.int32),
        qpos=np.vstack([qpos0, result.qpos]).astype(np.float32),
        qvel=np.vstack([qvel0, result.qvel]).astype(np.float32),
        ctrl=np.vstack([ctrl0, result.ctrl]).astype(np.float32),
        action=result.actions.astype(np.float32), tracking_score=result.tracking_score,
        tips=result.tips.astype(np.float32),
        **({key: np.asarray([row[key] for row in result.cost_rows], dtype=np.float64) for key in result.cost_rows[0]} if result.cost_rows else {}),
    )
    _save_observer(observer, directory)
    write_json(directory / "result.json", {
        "condition": label, "complete_to_40": result.complete,
        "executed_controls": len(result.outcome_endpoint),
        "first_failure_endpoint": result.first_failure_endpoint,
        "cost": result.cost if np.isfinite(result.cost) else None,
        "cost_components": result.components,
        "ctrl_loss_max": result.ctrl_loss_max,
        "sequence_sha256": _array_digest(result.sequence),
    })


def _trajectory_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _array_maps_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    rows = {}
    for key in common:
        a, b = np.asarray(left[key]), np.asarray(right[key])
        rows[key] = {
            "equal": bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()),
            "shape_left": list(a.shape), "shape_right": list(b.shape),
            "dtype_left": str(a.dtype), "dtype_right": str(b.dtype),
        }
    return {
        "all_equal": bool(set(left) == set(right) and all(row["equal"] for row in rows.values())),
        "only_left": sorted(set(left) - set(right)), "only_right": sorted(set(right) - set(left)),
        "fields": rows,
    }


def _observer_map(observer: StartupObserver) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = ({key: np.stack([np.asarray(row[key]) for row in observer.rows]) for key in observer.rows[0]} if observer.rows else {})
    contacts: dict[str, np.ndarray] = {}
    if observer.contacts:
        for key in observer.contacts[0]:
            contacts[key] = np.stack([np.asarray(row[key]) for row in observer.contacts])
    return rows, contacts


def _cold_parity(actual: Forecast, actual_observer: StartupObserver, cold: Forecast, cold_observer: StartupObserver) -> dict[str, Any]:
    endpoint = _array_maps_equal(
        {name: getattr(actual, name) for name in ("actions", "source_endpoint", "outcome_endpoint", "qpos", "qvel", "ctrl", "tips", "tracking_score")},
        {name: getattr(cold, name) for name in ("actions", "source_endpoint", "outcome_endpoint", "qpos", "qvel", "ctrl", "tips", "tracking_score")},
    )
    actual_rows, actual_contacts = _observer_map(actual_observer)
    cold_rows, cold_contacts = _observer_map(cold_observer)
    substeps = _array_maps_equal(actual_rows, cold_rows)
    contacts = _array_maps_equal(actual_contacts, cold_contacts)
    return {
        "endpoints": endpoint, "substeps": substeps, "contacts": contacts,
        "all_bitwise_equal": bool(endpoint["all_equal"] and substeps["all_equal"] and contacts["all_equal"]),
    }


def execute(config_path: Path, asset_root: Path | None, output: Path) -> dict[str, Any]:
    """Execute the frozen four-cell experiment and both cold replays."""
    config, _, assets = _load_config(config_path, asset_root)
    _require_clean_and_baseline(config)
    manifest = json.loads((output / "input_manifest.json").read_text())
    status_path = output / "status.json"
    status = json.loads(status_path.read_text())
    if manifest.get("status") != "PREFLIGHT_COMPLETE" or status.get("status") != "PREFLIGHT_COMPLETE":
        raise RuntimeError("startup execute requires one unused preflight")
    if manifest.get("implementation_commit") != _git_head():
        raise RuntimeError("implementation changed after startup preflight")
    for name, recorded in manifest["assets"].items():
        if manifest_entry(assets[name]) != recorded:
            raise RuntimeError(f"startup input changed after preflight: {name}")
    a = load_torch_gzip(assets["a_snapshot"]); l = load_torch_gzip(assets["l_snapshot"])
    reference = _trajectory_arrays(assets["reference"])
    reference_qpos = np.asarray(reference["qpos"], dtype=np.float32)
    reference_ctrl = np.asarray(reference["ctrl"], dtype=np.float32)
    limits = config["budget"]
    budget = PhysicsBudget(
        limits={
            "search": limits["search_physics_max"], "baseline": limits["baseline_physics_max"],
            "execution": limits["real_execution_physics_max"], "cold": limits["cold_replay_physics_max"],
            "retest": limits["implementation_retest_physics_max"], "setup": limits["all_setup_physics_max"],
        },
        all_in_limit=limits["all_in_physics_hard_max"],
    )
    noise = make_noise_schedule()
    np.savez_compressed(output / "noise_schedule.npz", **{
        f"source_{source:02d}_round_{round_index}": value
        for (source, round_index), value in noise.items()
    })
    results: dict[str, Forecast] = {}
    plan_meta: dict[str, Any] = {}
    try:
        for arm, initial in (("A", a), ("L", l)):
            world = _make_world(assets, initial); budget.reserve("setup", 1); budget.charge("setup", 1)
            restored = world.get_env_state()
            if not _snapshot_equal(restored, initial):
                raise RuntimeError(f"{arm} initial full snapshot did not restore bitwise")
            low, high = support_table(world)
            np.savez_compressed(output / f"{arm}_action_support.npz", low=low, high=high)
            tips_ref = reference_tip_positions(world.env.model_cpu, reference_qpos)
            zero = np.zeros((HORIZON, DOF), dtype=np.float32)
            baseline_observer = StartupObserver(world)
            baseline = _run_sequence(
                world, initial, zero, low, high, reference_qpos, reference_ctrl,
                tips_ref, budget, "baseline", observer=baseline_observer,
            )
            results[f"{arm}_REPLAY"] = baseline
            _save_condition(output / f"{arm}_REPLAY", initial, baseline, baseline_observer, label=f"{arm}_REPLAY")
            # Historical identity applies only to endpoints 0..20.
            current = _trajectory_arrays(output / f"{arm}_REPLAY/trajectory.npz")
            historical_path = (
                assets["replay_trace_manifest"].parent / "endpoints.npz"
                if arm == "A" else assets["left_alignment_status"].parent / "conditions/LEFT_ALIGNED_REPLAY/endpoints.npz"
            )
            historical = _trajectory_arrays(historical_path)
            if arm == "A":
                expected = {name: historical[f"B_{name}"] for name in ("qpos", "qvel", "ctrl")}
            else:
                expected = {name: historical[name] for name in ("qpos", "qvel", "ctrl")}
            parity = _array_maps_equal(
                {name: current[name][:21] for name in expected}, expected,
            )
            write_json(output / f"{arm}_REPLAY/historical_0_20_parity.json", parity)
            if not parity["all_equal"]:
                raise RuntimeError(f"{arm} Replay 0->20 did not reproduce historical arrays")
            planned, sequence, meta = run_startup_plan(
                f"{arm}_PLAN", world, initial, low, high, reference_qpos,
                reference_ctrl, tips_ref, noise, budget, output,
            )
            results[f"{arm}_PLAN"] = planned; plan_meta[arm] = meta
            # One independent fresh-world cold replay per completed arm.
            cold_world = _make_world(assets, initial); budget.reserve("setup", 1); budget.charge("setup", 1)
            cold_low, cold_high = support_table(cold_world)
            if cold_low.tobytes() != low.tobytes() or cold_high.tobytes() != high.tobytes():
                raise RuntimeError(f"{arm} cold action support changed")
            cold_observer = StartupObserver(cold_world)
            cold = _run_sequence(
                cold_world, initial, sequence, low, high, reference_qpos,
                reference_ctrl, tips_ref, budget, "cold", observer=cold_observer,
            )
            _save_condition(output / f"{arm}_PLAN_COLD", initial, cold, cold_observer, label=f"{arm}_PLAN_COLD")
            cold_report = _cold_parity(planned, meta["observer"], cold, cold_observer)
            write_json(output / f"{arm}_PLAN/cold_replay_parity.json", cold_report)
            if not cold_report["all_bitwise_equal"]:
                raise RuntimeError(f"{arm} selected plan cold replay diverged")
            del meta["observer"], meta["terminal_snapshot"]
            write_json(output / "budget_ledger.json", budget.report())
            status.update({"status": f"{arm}_COMPLETE", "budget": budget.report()})
            write_json(status_path, status)
    except Exception as error:
        status.update({"status": "PHYSICS_STOPPED", "error": repr(error), "budget": budget.report()})
        write_json(status_path, status); write_json(output / "budget_ledger.json", budget.report())
        raise
    if budget.physics["search"] > limits["search_physics_max"] or budget.total > limits["all_in_physics_hard_max"]:
        raise RuntimeError("startup experiment exceeded the frozen physics budget")
    summary = {
        "schema": f"{SCHEMA}_execution", "status": "PHYSICS_COMPLETE_ANALYSIS_PENDING",
        "conditions": {
            name: {
                "complete_to_40": row.complete, "first_failure_endpoint": row.first_failure_endpoint,
                "cost": row.cost if np.isfinite(row.cost) else None,
                "cost_components": row.components, "ctrl_loss_max": row.ctrl_loss_max,
            } for name, row in results.items()
        },
        "budget": budget.report(), "actor_or_critic_forwards": 0,
        "optimizer_updates": 32, "optimizer_update_semantics": "best_executed_candidate_retention",
        "rl_optimizer_updates": 0, "chunk_commit": False,
    }
    write_json(output / "execution_summary.json", summary)
    status.update({"status": "PHYSICS_COMPLETE_ANALYSIS_PENDING", "budget": budget.report()})
    write_json(status_path, status); write_json(output / "budget_ledger.json", budget.report())
    return summary


def analyze(config_path: Path, asset_root: Path | None, output: Path) -> dict[str, Any]:
    config, _, assets = _load_config(config_path, asset_root)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("status") not in {"PHYSICS_COMPLETE_ANALYSIS_PENDING", "ANALYSIS_COMPLETE_VISUAL_REVIEW_PENDING"}:
        raise RuntimeError("startup analysis requires completed physical execution")
    reference = _trajectory_arrays(assets["reference"])
    conditions = {name: _trajectory_arrays(output / name / "trajectory.npz") for name in config["evaluation"]["four_cells"]}
    metric_names = (
        "tool_position_error_m", "tool_rotation_error_rad",
        "target_position_error_m", "target_rotation_error_rad",
        "pair_translation_error_m", "pair_rotation_error_rad",
    )
    rows: list[dict[str, Any]] = []
    endpoint_metrics: dict[str, dict[str, Any]] = {}
    for name, data in conditions.items():
        endpoints = np.asarray(data["endpoint"], dtype=np.int32)
        endpoint_metrics[name] = {}
        for endpoint in range(HORIZON + 1):
            found = np.flatnonzero(endpoints == endpoint)
            if not len(found):
                endpoint_metrics[name][str(endpoint)] = None
                rows.append({"condition": name, "endpoint": endpoint, "availability": "N/A"})
                continue
            index = int(found[0]); qpos = data["qpos"][index]; qvel = data["qvel"][index]
            ref = reference["qpos"][endpoint]
            values: dict[str, float] = {}
            for role in ("tool", "target"):
                p, q = object_pose(qpos, role); rp, rq = object_pose(ref, role)
                values[f"{role}_position_error_m"] = float(np.linalg.norm(p - rp))
                values[f"{role}_rotation_error_rad"] = rotation_error_rad(q, rq)
                qvel_offset = 36 if role == "tool" else 42
                values[f"{role}_linear_speed_m_s"] = float(np.linalg.norm(qvel[qvel_offset:qvel_offset + 3]))
                values[f"{role}_angular_speed_rad_s"] = float(np.linalg.norm(qvel[qvel_offset + 3:qvel_offset + 6]))
            pair_p, pair_r = pair_pose(qpos); ref_p, ref_r = pair_pose(ref)
            values["pair_translation_error_m"] = float(np.linalg.norm(pair_p - ref_p))
            values["pair_rotation_error_rad"] = matrix_rotation_error_rad(pair_r, ref_r)
            values["world_pair_translation_error_m"] = float(np.linalg.norm(
                (qpos[36:39] - qpos[43:46]) - (ref[36:39] - ref[43:46])
            ))
            endpoint_metrics[name][str(endpoint)] = values
            rows.append({"condition": name, "endpoint": endpoint, "availability": "measured", **values})
    fields = sorted({key for row in rows for key in row})
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)

    segments = {"0_10": (1, 10), "0_20": (1, 20), "20_40": (21, 40), "0_40": (1, 40)}
    summaries: dict[str, dict[str, Any]] = {}
    for name in conditions:
        summaries[name] = {}
        for segment, (first, last) in segments.items():
            values = [endpoint_metrics[name][str(endpoint)] for endpoint in range(first, last + 1)]
            if any(value is None for value in values):
                summaries[name][segment] = None
                continue
            assert all(value is not None for value in values)
            summary: dict[str, float] = {}
            for metric in metric_names:
                samples = np.asarray([value[metric] for value in values], dtype=np.float64)
                summary[f"{metric}_mean"] = float(samples.mean())
                summary[f"{metric}_rms"] = float(np.sqrt(np.mean(np.square(samples))))
                summary[f"{metric}_max"] = float(samples.max())
            summaries[name][segment] = summary
        summaries[name]["endpoints"] = {
            str(endpoint): endpoint_metrics[name][str(endpoint)]
            for endpoint in config["evaluation"]["endpoint_summaries"]
        }

    comparisons: dict[str, Any] = {}
    for plan in ("A_PLAN", "L_PLAN"):
        comparisons[plan] = {}
        for baseline in (("A_REPLAY", plan.replace("PLAN", "REPLAY")) if plan == "A_PLAN" else ("A_REPLAY", "L_REPLAY")):
            key = f"vs_{baseline}"
            comparisons[plan][key] = {}
            for segment in segments:
                left, right = summaries[plan][segment], summaries[baseline][segment]
                comparisons[plan][key][segment] = (
                    None if left is None or right is None else {
                        metric: left[f"{metric}_rms"] - right[f"{metric}_rms"] for metric in metric_names
                    }
                )

    guard_m = float(config["evaluation"]["numerical_guard_m"])
    guard_rad = float(config["evaluation"]["numerical_guard_rad"])
    guard = {metric: (guard_m if metric.endswith("_m") else guard_rad) for metric in metric_names}
    overall = {}
    for plan in ("A_PLAN", "L_PLAN"):
        checks = []
        improvements = []
        for segment in ("0_20", "0_40"):
            candidate = summaries[plan][segment]; baseline = summaries["A_REPLAY"][segment]
            if candidate is None or baseline is None:
                checks.append(False); continue
            for metric in metric_names:
                delta = candidate[f"{metric}_rms"] - baseline[f"{metric}_rms"]
                checks.append(delta <= guard[metric]); improvements.append(delta < -guard[metric])
        for endpoint in (20, 40):
            candidate = endpoint_metrics[plan][str(endpoint)]; baseline = endpoint_metrics["A_REPLAY"][str(endpoint)]
            if candidate is None or baseline is None:
                checks.append(False); continue
            for metric in metric_names:
                delta = candidate[metric] - baseline[metric]
                checks.append(delta <= guard[metric]); improvements.append(delta < -guard[metric])
        overall[plan] = {
            "numerical_no_tradeoff_gate": bool(checks and all(checks) and any(improvements)),
            "all_required_nonworsening": bool(checks and all(checks)),
            "at_least_one_clear_improvement": bool(any(improvements)),
            "visual_gate_pending": True,
        }

    controls = {}
    for plan in ("A_PLAN", "L_PLAN"):
        data = conditions[plan]; arm = plan[0]
        support = _trajectory_arrays(output / f"{arm}_action_support.npz")
        action = data["action"]
        on_bound = np.isclose(action, support["low"], rtol=0, atol=0) | np.isclose(action, support["high"], rtol=0, atol=0)
        projected = total = 0
        for source in REPLAN_SOURCES:
            candidates = _trajectory_arrays(output / plan / "search" / f"source_{source:02d}/candidates.npz")
            projected += int(candidates["projection_count"].sum())
            total += int(len(candidates["projection_count"]) * (HORIZON - source) * DOF)
        controls[plan] = {
            "deterministic_action_rms": float(np.sqrt(np.mean(np.square(action.astype(np.float64))))),
            "deterministic_bound_fraction": float(on_bound.mean()),
            "search_projected_components": projected,
            "search_projection_denominator": total,
            "search_projection_fraction": float(projected / total),
            "first_ctrl_change_l2": float(np.linalg.norm(data["ctrl"][1] - data["ctrl"][0])),
        }

    contacts = {}
    for name in config["evaluation"]["four_cells"]:
        raw = _trajectory_arrays(output / name / "contacts_raw.npz")
        groups = np.asarray(raw.get("group", []))
        force = np.asarray(raw.get("wrench_contact_force_torque", np.empty((0, 6))))
        contacts[name] = {
            group: {
                "contact_rows": int(np.count_nonzero(groups == group)),
                "maximum_normal_force_N": float(force[groups == group, 0].max()) if np.any(groups == group) else 0.0,
            }
            for group in sorted(set(groups.tolist()))
        }

    execution = json.loads((output / "execution_summary.json").read_text())
    report = {
        "schema": f"{SCHEMA}_analysis", "status": "ANALYSIS_COMPLETE_VISUAL_REVIEW_PENDING",
        "endpoint_metrics": endpoint_metrics, "segment_summaries": summaries,
        "comparisons": comparisons, "overall_numerical_gate": overall,
        "control_usage": controls, "contact_summary": contacts,
        "budget": execution["budget"],
        "interpretation_boundaries": {
            "planner_cost_is_not_success_criterion": True,
            "numerical_gate_is_not_full_pour_success": True,
            "visual_review_required_before_green_classification": True,
            "no_reset_or_chunk_promotion": True,
        },
    }
    write_json(output / "analysis.json", report)
    findings = [
        "# Control-aware startup v1 findings", "",
        "**Visual review is still pending; the numerical classification alone is not a task-quality pass.**", "",
    ]
    for name in config["evaluation"]["four_cells"]:
        row = execution["conditions"][name]
        findings.append(f"- {name}: complete_to_40=`{row['complete_to_40']}`, planner cost=`{row['cost']}`.")
    for plan in ("A_PLAN", "L_PLAN"):
        findings.append(f"- {plan} conservative no-tradeoff numerical gate vs A_REPLAY: `{overall[plan]['numerical_no_tradeoff_gate']}`.")
    findings.extend(["", "This is a bounded local predictive-sampling repair pilot. It does not recover an EgoEngine author MPC configuration, certify a deployable reset, or authorize RL/chunk commit."])
    (output / "findings.md").write_text("\n".join(findings) + "\n")
    status.update({"status": "ANALYSIS_COMPLETE_VISUAL_REVIEW_PENDING"})
    write_json(status_path, status)
    return report


def run_phase(
    config_path: Path, asset_root: Path | None, output: Path | None, phase: str,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    root = (asset_root or Path(config["asset_root"])).resolve(strict=True)
    destination = output or root / config["run_directory"]
    if phase == "preflight":
        return preflight(config_path, asset_root, destination)
    if phase == "execute":
        return execute(config_path, asset_root, destination)
    if phase == "analyze":
        return analyze(config_path, asset_root, destination)
    raise ValueError(f"unknown startup phase: {phase}")
