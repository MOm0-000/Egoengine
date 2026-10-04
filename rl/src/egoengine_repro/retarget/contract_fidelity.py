"""Pure kinematic primitives for the Pour retarget-contract fidelity audit.

The functions in this module never step a simulator and never optimize a
trajectory.  They only evaluate frozen states or deterministic FK probes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HUMAN_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
ROBOT_CHAINS = {
    "thumb": (
        "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1",
        "left_hand_thumb_bend_link", "left_hand_thumb_rota_link1", "left_thumb_tip",
    ),
    "index": (
        "left_hand_index_bend_joint", "left_hand_index_joint1",
        "left_hand_index_bend_link", "left_hand_index_rota_link1", "left_index_tip",
    ),
    "middle": (
        "left_hand_mid_joint1", "left_hand_mid_joint2",
        "left_hand_mid_link1", "left_hand_mid_link2", "left_middle_tip",
    ),
    "ring": (
        "left_hand_ring_joint1", "left_hand_ring_joint2",
        "left_hand_ring_link1", "left_hand_ring_link2", "left_ring_tip",
    ),
    "pinky": (
        "left_hand_pinky_joint1", "left_hand_pinky_joint2",
        "left_hand_pinky_link1", "left_hand_pinky_link2", "left_pinky_tip",
    ),
}
ROBOT_ACTIVE_JOINTS = {
    "thumb": (
        "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1",
        "left_hand_thumb_rota_joint2",
    ),
    "index": (
        "left_hand_index_bend_joint", "left_hand_index_joint1",
        "left_hand_index_joint2",
    ),
    "middle": ("left_hand_mid_joint1", "left_hand_mid_joint2"),
    "ring": ("left_hand_ring_joint1", "left_hand_ring_joint2"),
    "pinky": ("left_hand_pinky_joint1", "left_hand_pinky_joint2"),
}


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate direction")
    return value / norm


def angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(unit(first), unit(second)), -1.0, 1.0)))


def rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).reshape(3, 3).T @ np.asarray(second).reshape(3, 3)
    return float(Rotation.from_matrix(relative).magnitude())


def semantic_landmark_candidates(model: mujoco.MjModel) -> dict[str, Any]:
    result: dict[str, Any] = {
        "wrist": {
            "body": "left_hand_link",
            "body_id": int(model.body("left_hand_link").id),
            "site": "left_palm",
            "site_id": int(model.site("left_palm").id),
        },
        "fingers": {},
    }
    for finger, names in ROBOT_CHAINS.items():
        joint1, joint2, body1, body2, tip = names
        result["fingers"][finger] = {
            "human_full_chain_indices": list(HUMAN_CHAINS[finger]),
            "legacy": {
                "proximal_body": body1,
                "proximal_body_id": int(model.body(body1).id),
                "distal_body": body2,
                "distal_body_id": int(model.body(body2).id),
                "tip_site": tip,
                "tip_site_id": int(model.site(tip).id),
            },
            "joint_anchor": {
                "proximal_joint": joint1,
                "proximal_joint_id": int(model.joint(joint1).id),
                "distal_joint": joint2,
                "distal_joint_id": int(model.joint(joint2).id),
                "tip_site": tip,
                "tip_site_id": int(model.site(tip).id),
            },
        }
    return result


def robot_landmarks(
    model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, *, mapping: str,
) -> dict[str, np.ndarray]:
    data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_forward(model, data)
    result = {"wrist": data.site_xpos[model.site("left_palm").id].copy()}
    for finger, names in ROBOT_CHAINS.items():
        joint1, joint2, body1, body2, tip = names
        if mapping == "joint_anchor":
            proximal = data.xanchor[model.joint(joint1).id]
            distal = data.xanchor[model.joint(joint2).id]
        elif mapping == "legacy_body_origin":
            proximal = data.xpos[model.body(body1).id]
            distal = data.xpos[model.body(body2).id]
        else:
            raise ValueError(f"unknown mapping {mapping}")
        result[finger] = np.stack(
            [proximal.copy(), distal.copy(), data.site_xpos[model.site(tip).id].copy()]
        )
    return result


def human_chain(joints: np.ndarray, finger: str) -> np.ndarray:
    value = np.asarray(joints, dtype=np.float64)
    if value.shape != (21, 3):
        raise ValueError("expected MANO21 joint positions")
    return value[list(HUMAN_CHAINS[finger])]


def direct_directions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(points, dtype=np.float64)
    if value.shape == (4, 3):
        return unit(value[1] - value[0]), unit(value[3] - value[2])
    if value.shape == (3, 3):
        return unit(value[1] - value[0]), unit(value[2] - value[1])
    raise ValueError("expected a three- or four-point chain")


def palm_local(points: np.ndarray, origin: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - np.asarray(origin)) @ np.asarray(rotation).reshape(3, 3)


def finite_difference_jacobian(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_name: str,
    epsilon: float,
    *,
    mapping: str = "joint_anchor",
) -> dict[str, np.ndarray]:
    joint = int(model.joint(joint_name).id)
    address = int(model.jnt_qposadr[joint])
    lower, upper = map(float, model.jnt_range[joint])
    requested_center = float(qpos[address])
    step = min(float(epsilon), 0.25 * (upper - lower))
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"cannot form symmetric finite difference for {joint_name}")
    # Reference states may lie exactly on a physical joint limit.  Preserve a
    # genuinely symmetric finite difference by moving only the evaluation
    # centre to the closest point with +/-step clearance; never silently use a
    # one-sided derivative.
    center = float(np.clip(requested_center, lower + step, upper - step))
    minus = np.asarray(qpos, dtype=np.float64).copy()
    plus = minus.copy()
    minus[address] = center - step
    plus[address] = center + step
    data = mujoco.MjData(model)
    left = robot_landmarks(model, data, minus, mapping=mapping)
    right = robot_landmarks(model, data, plus, mapping=mapping)
    result = {"wrist": (right["wrist"] - left["wrist"]) / (2.0 * step)}
    for finger in FINGERS:
        result[finger] = (right[finger] - left[finger]) / (2.0 * step)
    result["effective_step_rad"] = np.asarray(step)
    result["requested_center_rad"] = np.asarray(requested_center)
    result["evaluation_center_rad"] = np.asarray(center)
    result["evaluation_center_shift_rad"] = np.asarray(center - requested_center)
    return result


def joint_chain_singular_values(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    finger: str,
    epsilon: float,
) -> np.ndarray:
    names = ROBOT_ACTIVE_JOINTS[finger]
    columns = []
    for name in names:
        jac = finite_difference_jacobian(model, qpos, name, epsilon)
        # The first anchor is a hinge point.  Identifiability lives in the
        # downstream second-anchor and physical-tip motion.
        columns.append(jac[finger][1:].reshape(-1))
    return np.linalg.svd(np.stack(columns, axis=1), compute_uv=False)


@dataclass(frozen=True)
class Probe:
    probe_id: str
    family: str
    endpoint: int
    qpos: np.ndarray
    metadata: dict[str, Any]


def finger_probes(
    model: mujoco.MjModel,
    old_qpos: np.ndarray,
    endpoints: list[int],
    fractions: list[float],
) -> list[Probe]:
    result: list[Probe] = []
    for endpoint in endpoints:
        for finger in ("ring", "pinky"):
            for joint_slot, joint_name in enumerate(ROBOT_CHAINS[finger][:2], 1):
                joint = int(model.joint(joint_name).id)
                address = int(model.jnt_qposadr[joint])
                lower, upper = map(float, model.jnt_range[joint])
                width = upper - lower
                for fraction in fractions:
                    requested = float(old_qpos[endpoint, address] + fraction * width)
                    applied = float(np.clip(requested, lower, upper))
                    qpos = old_qpos[endpoint].copy()
                    qpos[address] = applied
                    code = f"{fraction:+.3f}".replace("+", "p").replace("-", "m").replace(".", "d")
                    result.append(Probe(
                        f"finger_e{endpoint:02d}_{finger}_j{joint_slot}_{code}",
                        "finger_joint", endpoint, qpos,
                        {
                            "finger": finger,
                            "joint": joint_name,
                            "joint_slot": joint_slot,
                            "fraction_of_range": float(fraction),
                            "requested_qpos": requested,
                            "applied_qpos": applied,
                            "clipped": bool(applied != requested),
                        },
                    ))
    return result


def wrist_z_probes(
    old: np.ndarray, accepted: np.ndarray, fractions: list[float], address: int = 20,
) -> list[Probe]:
    result: list[Probe] = []
    for family, anchor in (("old_fixed", old[0]), ("accepted_a_fixed", accepted[0])):
        for fraction in fractions:
            qpos = np.asarray(anchor, dtype=np.float64).copy()
            qpos[address] = old[0, address] + fraction * (accepted[0, address] - old[0, address])
            code = f"{fraction:.2f}".replace(".", "d")
            result.append(Probe(
                f"wrist_z_{family}_{code}", "wrist_z", 0, qpos,
                {"fixed_state": family, "fraction": float(fraction), "qpos_address": address},
            ))
    return result


def deterministic_probe_arrays(probes: list[Probe]) -> dict[str, np.ndarray]:
    return {
        "probe_id": np.asarray([probe.probe_id for probe in probes]),
        "family": np.asarray([probe.family for probe in probes]),
        "endpoint": np.asarray([probe.endpoint for probe in probes], dtype=np.int32),
        "qpos": np.stack([probe.qpos for probe in probes]),
    }
