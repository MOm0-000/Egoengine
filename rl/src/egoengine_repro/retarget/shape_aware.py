"""Local left ring/pinky shape descriptors for the Pour initialization audit.

This module intentionally does not define a new retargeting framework.  It
contains the small, deterministic two-link mapping needed by the bounded
shape-aware endpoint-0 experiment.
"""

from __future__ import annotations

from typing import Any, Callable

import mujoco
import numpy as np
from scipy.optimize import minimize


FINGERS = ("ring", "pinky")
MANO_TWO_LINK = {"ring": (13, 14, 16), "pinky": (17, 18, 20)}


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("shape descriptor contains a degenerate bone")
    return vector / norm


def bend_angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(_unit(first), _unit(second)), -1.0, 1.0)))


def human_two_link_descriptor(
    joint_positions: np.ndarray,
    wrist_transform: np.ndarray,
) -> dict[str, Any]:
    """Describe MANO ring/pinky as MCP->PIP and PIP->tip palm-frame links."""
    points = np.asarray(joint_positions, dtype=np.float64)
    wrist = np.asarray(wrist_transform, dtype=np.float64)
    if points.shape != (21, 3) or wrist.shape != (4, 4):
        raise ValueError("expected MANO21 points and one wrist transform")
    local = (points - wrist[:3, 3]) @ wrist[:3, :3]
    result: dict[str, Any] = {}
    for finger, indices in MANO_TWO_LINK.items():
        selected = local[np.asarray(indices)]
        proximal = selected[1] - selected[0]
        distal = selected[2] - selected[1]
        result[finger] = {
            "landmark_indices": list(indices),
            "landmarks_palm_m": selected.tolist(),
            "proximal_direction": _unit(proximal).tolist(),
            "distal_direction": _unit(distal).tolist(),
            "proximal_length_m": float(np.linalg.norm(proximal)),
            "distal_length_m": float(np.linalg.norm(distal)),
            "bend_angle_rad": bend_angle(proximal, distal),
        }
    return result


def robot_two_link_descriptor(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fingertip_targets: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Return the matching two-link XHand descriptor in its palm frame."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("invalid robot qpos")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    palm = model.body("left_hand_link").id
    rotation = data.xmat[palm].reshape(3, 3)
    origin = data.xpos[palm]
    result: dict[str, Any] = {}
    for finger in FINGERS:
        world = np.stack(
            [
                data.xpos[model.body(f"left_hand_{finger}_link1").id],
                data.xpos[model.body(f"left_hand_{finger}_link2").id],
                data.site_xpos[model.site(f"left_{finger}_tip").id],
            ]
        )
        local = (world - origin) @ rotation
        proximal = local[1] - local[0]
        distal = local[2] - local[1]
        row: dict[str, Any] = {
            "landmarks_world_m": world.tolist(),
            "landmarks_palm_m": local.tolist(),
            "proximal_direction": _unit(proximal).tolist(),
            "distal_direction": _unit(distal).tolist(),
            "proximal_length_m": float(np.linalg.norm(proximal)),
            "distal_length_m": float(np.linalg.norm(distal)),
            "bend_angle_rad": bend_angle(proximal, distal),
            "joint_values_rad": [
                float(qpos[model.joint(f"left_hand_{finger}_joint1").qposadr[0]]),
                float(qpos[model.joint(f"left_hand_{finger}_joint2").qposadr[0]]),
            ],
        }
        if fingertip_targets is not None:
            target = np.asarray(fingertip_targets[finger], dtype=np.float64)
            row["fingertip_target_world_m"] = target.tolist()
            row["fingertip_error_m"] = float(np.linalg.norm(world[-1] - target))
        result[finger] = row
    return result


def shape_residual(
    robot: dict[str, Any],
    human: dict[str, Any],
) -> np.ndarray:
    """Equal-weight direction and length-normalized fingertip residual."""
    residual: list[float] = []
    for finger in FINGERS:
        row = robot[finger]
        target = human[finger]
        residual.extend(
            np.asarray(row["proximal_direction"], dtype=np.float64)
            - np.asarray(target["proximal_direction"], dtype=np.float64)
        )
        residual.extend(
            np.asarray(row["distal_direction"], dtype=np.float64)
            - np.asarray(target["distal_direction"], dtype=np.float64)
        )
        length = float(row["proximal_length_m"] + row["distal_length_m"])
        tip = np.asarray(row["landmarks_world_m"][-1], dtype=np.float64)
        target_tip = np.asarray(row["fingertip_target_world_m"], dtype=np.float64)
        residual.extend((tip - target_tip) / length)
    value = np.asarray(residual, dtype=np.float64)
    if value.shape != (18,) or not np.isfinite(value).all():
        raise ValueError("invalid shape residual")
    return value


def fit_left_ring_pinky_shape(
    model: mujoco.MjModel,
    accepted_qpos: np.ndarray,
    human_descriptor: dict[str, Any],
    fingertip_targets: dict[str, np.ndarray],
    distance_constraint: Callable[[np.ndarray], np.ndarray],
    *,
    max_iterations: int = 200,
    ftol: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one single-start constrained four-joint fit.

    ``distance_constraint`` returns nonnegative margins for every declared pair
    affected by these four joints.  Physics outcomes never enter this solve.
    """
    accepted = np.asarray(accepted_qpos, dtype=np.float64)
    names = [
        f"left_hand_{finger}_joint{joint}"
        for finger in FINGERS
        for joint in (1, 2)
    ]
    addresses = np.asarray([int(model.joint(name).qposadr[0]) for name in names])
    joint_ids = np.asarray([model.joint(name).id for name in names])
    lower = model.jnt_range[joint_ids, 0].astype(np.float64)
    upper = model.jnt_range[joint_ids, 1].astype(np.float64)
    start = np.clip(accepted[addresses], lower, upper)

    calls = {"objective": 0, "constraint": 0}

    def qpos_from(values: np.ndarray) -> np.ndarray:
        qpos = accepted.copy()
        qpos[addresses] = values
        return qpos

    def objective(values: np.ndarray) -> float:
        calls["objective"] += 1
        descriptor = robot_two_link_descriptor(
            model, qpos_from(values), fingertip_targets
        )
        residual = shape_residual(descriptor, human_descriptor)
        return float(np.dot(residual, residual))

    def constraint(values: np.ndarray) -> np.ndarray:
        calls["constraint"] += 1
        margin = np.asarray(distance_constraint(qpos_from(values)), dtype=np.float64)
        if margin.ndim != 1 or not np.isfinite(margin).all():
            raise ValueError("invalid declared-pair constraint margin")
        return margin

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=[{"type": "ineq", "fun": constraint}],
        options={"maxiter": int(max_iterations), "ftol": float(ftol), "disp": False},
    )
    candidate = qpos_from(result.x)
    before = robot_two_link_descriptor(model, accepted, fingertip_targets)
    after = robot_two_link_descriptor(model, candidate, fingertip_targets)
    return candidate, {
        "method": "deterministic_single_start_slsqp",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective_evaluations": int(result.nfev),
        "calls": calls,
        "joint_names": names,
        "qpos_addresses": addresses.tolist(),
        "joint_lower_rad": lower.tolist(),
        "joint_upper_rad": upper.tolist(),
        "accepted_joint_values_rad": start.tolist(),
        "candidate_joint_values_rad": result.x.tolist(),
        "accepted_shape_objective": objective(start),
        "candidate_shape_objective": objective(result.x),
        "accepted_descriptor": before,
        "candidate_descriptor": after,
        "minimum_constraint_margin": float(constraint(result.x).min()),
        "physics_outcomes_used": False,
        "multi_start": False,
        "weight_sweep": False,
    }

