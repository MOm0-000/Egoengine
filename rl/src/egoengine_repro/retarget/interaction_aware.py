"""Small interaction-aware retargeting primitives for the bounded Pour pilot.

The module deliberately exposes direct numerical functions rather than a new
planner or retargeting framework.  It only represents landmarks that exist in
the active XHand model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HUMAN_INDICES = {
    "thumb": (1, 2, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
ROBOT_ANCHORS = {
    "thumb": ("left_hand_thumb_bend_link", "left_hand_thumb_rota_link1", "left_thumb_tip"),
    "index": ("left_hand_index_bend_link", "left_hand_index_rota_link1", "left_index_tip"),
    "middle": ("left_hand_mid_link1", "left_hand_mid_link2", "left_middle_tip"),
    "ring": ("left_hand_ring_link1", "left_hand_ring_link2", "left_ring_tip"),
    "pinky": ("left_hand_pinky_link1", "left_hand_pinky_link2", "left_pinky_tip"),
}


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate semantic bone")
    return vector / norm


def semantic_keypoint_map(model: mujoco.MjModel) -> dict[str, Any]:
    """Resolve and freeze the 16 real robot/human semantic landmarks."""
    result: dict[str, Any] = {
        "count": 16,
        "wrist": {
            "human_joint_index": 0,
            "robot_body": "left_hand_link",
            "robot_body_id": int(model.body("left_hand_link").id),
        },
        "fingers": {},
    }
    for finger in FINGERS:
        first, second, tip = ROBOT_ANCHORS[finger]
        result["fingers"][finger] = {
            "human_joint_indices": list(HUMAN_INDICES[finger]),
            "robot_proximal_body": first,
            "robot_proximal_body_id": int(model.body(first).id),
            "robot_second_body": second,
            "robot_second_body_id": int(model.body(second).id),
            "robot_fingertip_site": tip,
            "robot_fingertip_site_id": int(model.site(tip).id),
        }
    return result


def human_semantic_points(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (21, 3) or not np.isfinite(joints).all():
        raise ValueError("expected finite MANO21 joint positions")
    rows = [joints[0]]
    for finger in FINGERS:
        rows.extend(joints[list(HUMAN_INDICES[finger])])
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (16, 3):
        raise AssertionError(result.shape)
    return result


def robot_semantic_points(
    model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray
) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("invalid robot qpos")
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    rows = [data.xpos[model.body("left_hand_link").id].copy()]
    for finger in FINGERS:
        first, second, tip = ROBOT_ANCHORS[finger]
        rows.extend(
            [
                data.xpos[model.body(first).id].copy(),
                data.xpos[model.body(second).id].copy(),
                data.site_xpos[model.site(tip).id].copy(),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _wrist_local(points: np.ndarray, wrist_transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(wrist_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("wrist transform must be 4x4")
    return (np.asarray(points) - transform[:3, 3]) @ transform[:3, :3]


def human_bone_directions(
    joints: np.ndarray, wrist_transform: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    local = _wrist_local(human_semantic_points(joints), wrist_transform)
    return {
        finger: (_unit(local[1 + 3 * index + 1] - local[1 + 3 * index]),
                 _unit(local[1 + 3 * index + 2] - local[1 + 3 * index + 1]))
        for index, finger in enumerate(FINGERS)
    }


def robot_bone_directions(
    model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    points = robot_semantic_points(model, data, qpos)
    palm = int(model.body("left_hand_link").id)
    local = (points - data.xpos[palm]) @ data.xmat[palm].reshape(3, 3)
    return {
        finger: (_unit(local[1 + 3 * index + 1] - local[1 + 3 * index]),
                 _unit(local[1 + 3 * index + 2] - local[1 + 3 * index + 1]))
        for index, finger in enumerate(FINGERS)
    }


def bone_error(
    robot: dict[str, tuple[np.ndarray, np.ndarray]],
    source: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, dict[str, float], dict[str, dict[str, float]]]:
    """TopoRetarget-style adjacent-direction mismatch and intuitive angles."""
    contributions: dict[str, float] = {}
    angles: dict[str, dict[str, float]] = {}
    for finger in FINGERS:
        rp, rd = robot[finger]
        sp, sd = source[finger]
        residual = (rp - rd) - (sp - sd)
        contributions[finger] = float(np.dot(residual, residual))
        angles[finger] = {
            "proximal_deg": float(np.degrees(np.arccos(np.clip(np.dot(rp, sp), -1.0, 1.0)))),
            "distal_deg": float(np.degrees(np.arccos(np.clip(np.dot(rd, sd), -1.0, 1.0)))),
        }
    return float(sum(contributions.values())), contributions, angles


def deterministic_surface_samples(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic area-uniform triangle samples in mesh-local space."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    ) * 0.5
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("invalid object surface area")
    rng = np.random.default_rng(int(seed))
    face_index = rng.choice(len(faces), size=int(count), replace=True, p=area / area.sum())
    uv = rng.random((int(count), 2))
    flip = uv.sum(axis=1) > 1.0
    uv[flip] = 1.0 - uv[flip]
    bary = np.c_[1.0 - uv.sum(axis=1), uv]
    points = np.einsum("ni,nij->nj", bary, triangles[face_index])
    return points, face_index, bary


def pose_local_points(
    local_points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return np.asarray(local_points) @ np.asarray(rotation).reshape(3, 3).T + np.asarray(translation)


def interaction_topology(source_vertices: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    tetra = Delaunay(np.asarray(source_vertices, dtype=np.float64), qhull_options="QJ Pp").simplices
    edges = set()
    for cell in tetra:
        for i in range(4):
            for j in range(i + 1, 4):
                edges.add(tuple(sorted((int(cell[i]), int(cell[j])))))
    neighbors = [set() for _ in range(len(source_vertices))]
    for first, second in edges:
        neighbors[first].add(second)
        neighbors[second].add(first)
    if any(not item for item in neighbors):
        raise ValueError("interaction mesh contains isolated vertices")
    return np.asarray(sorted(edges), dtype=np.int64), [np.asarray(sorted(item), dtype=np.int64) for item in neighbors]


def weighted_laplacian(
    vertices: np.ndarray,
    source_vertices: np.ndarray,
    neighbors: list[np.ndarray],
    kappa: float,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    source = np.asarray(source_vertices, dtype=np.float64)
    result = np.empty_like(vertices)
    for index, adjacent in enumerate(neighbors):
        distance = np.linalg.norm(source[index] - source[adjacent], axis=1)
        weight = np.exp(-float(kappa) * distance)
        weight /= weight.sum()
        result[index] = vertices[index] - np.sum(weight[:, None] * vertices[adjacent], axis=0)
    return result


def interaction_error(
    robot_vertices: np.ndarray,
    source_vertices: np.ndarray,
    neighbors: list[np.ndarray],
    kappa: float,
) -> float:
    source_lap = weighted_laplacian(source_vertices, source_vertices, neighbors, kappa)
    robot_lap = weighted_laplacian(robot_vertices, source_vertices, neighbors, kappa)
    residual = robot_lap - source_lap
    return float(np.mean(np.sum(residual * residual, axis=1)))


@dataclass(frozen=True)
class FrameTarget:
    human_points: np.ndarray
    human_bones: dict[str, tuple[np.ndarray, np.ndarray]]
    object_world_points: np.ndarray
    source_vertices: np.ndarray
    neighbors: list[np.ndarray]


def solve_frame_two_stage(
    model: mujoco.MjModel,
    old_qpos: np.ndarray,
    previous_qpos: np.ndarray,
    target: FrameTarget,
    feasibility_margin: Callable[[np.ndarray], np.ndarray],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    lambda_warm: float,
    lambda_smooth: float,
    lambda_im: float,
    lambda_bone: float,
    lambda_temporal: float,
    lambda_base_translation: float,
    lambda_base_rotation: float,
    kappa: float,
    max_iterations: int,
    ftol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run exactly one warm/refine solve for one frame of the sole candidate."""
    old = np.asarray(old_qpos, dtype=np.float64)
    previous = np.asarray(previous_qpos, dtype=np.float64)
    addresses = np.arange(18, 36, dtype=np.int64)
    data = mujoco.MjData(model)
    counts = {"warm_objective": 0, "refine_objective": 0, "constraint": 0}

    def assemble(value: np.ndarray) -> np.ndarray:
        qpos = old.copy()
        qpos[addresses] = value
        return qpos

    def bone(value: np.ndarray) -> float:
        directions = robot_bone_directions(model, data, assemble(value))
        return bone_error(directions, target.human_bones)[0]

    def constraint(value: np.ndarray) -> np.ndarray:
        counts["constraint"] += 1
        result = np.asarray(feasibility_margin(assemble(value)), dtype=np.float64)
        if result.ndim != 1 or not np.isfinite(result).all():
            raise ValueError("nonfinite feasibility margin")
        return result

    previous_left = previous[addresses]

    def warm_objective(value: np.ndarray) -> float:
        counts["warm_objective"] += 1
        return float(lambda_warm * bone(value) + lambda_smooth * np.dot(value - previous_left, value - previous_left))

    common = {
        "method": "SLSQP",
        "bounds": list(zip(lower, upper)),
        "constraints": [{"type": "ineq", "fun": constraint}],
        "options": {"maxiter": int(max_iterations), "ftol": float(ftol), "disp": False},
    }
    warm = minimize(warm_objective, np.clip(old[addresses], lower, upper), **common)

    def refine_objective(value: np.ndarray) -> float:
        counts["refine_objective"] += 1
        qpos = assemble(value)
        robot_points = robot_semantic_points(model, data, qpos)
        vertices = np.concatenate([robot_points, target.object_world_points])
        im = interaction_error(vertices, target.source_vertices, target.neighbors, kappa)
        delta = value - old[addresses]
        temporal = value - previous_left
        regularization = (
            lambda_temporal * np.dot(temporal, temporal)
            + lambda_base_translation * np.dot(delta[:3], delta[:3])
            + lambda_base_rotation * np.dot(delta[3:6], delta[3:6])
        )
        return float(lambda_im * im + lambda_bone * bone(value) + regularization)

    refined = minimize(refine_objective, np.clip(warm.x, lower, upper), **common)
    candidate = assemble(refined.x)
    return candidate, {
        "warm": {
            "success": bool(warm.success), "status": int(warm.status), "message": str(warm.message),
            "iterations": int(warm.nit), "objective": float(warm.fun),
        },
        "refine": {
            "success": bool(refined.success), "status": int(refined.status), "message": str(refined.message),
            "iterations": int(refined.nit), "objective": float(refined.fun),
        },
        "calls": counts,
        "minimum_margin": float(constraint(refined.x).min()),
        "candidate_count": 1,
        "multi_start": False,
        "parameter_sweep": False,
    }
