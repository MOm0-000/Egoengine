"""Read-only early hand/object contact origin audit.

This module deliberately exposes only static array, FK, distance-query, and
rendering operations.  It never constructs the runtime environment and never
calls a MuJoCo integrator, policy, IK solver, or optimizer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import uuid
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


HANDS = ("right", "left")
PARTS = ("palm", "thumb", "index", "middle", "ring", "pinky")
FINGERS = PARTS[1:]
HUMAN_TIPS = (4, 8, 12, 16, 20)
HUMAN_BONES = tuple(
    (0 if joint == start else joint - 1, joint)
    for start in (1, 5, 9, 13, 17)
    for joint in range(start, start + 4)
)
COMBINATIONS = ("RR", "AR", "RA", "AA")
KEY_ENDPOINTS = (0, 5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20)
EXPECTED_HEAD = "7babecb9a4845ab16b958b6ef9187af68604799e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def npz_schema(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
            for name in archive.files
        }


def pose7_to_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def transform_to_pose7(transform: np.ndarray) -> np.ndarray:
    quaternion = Rotation.from_matrix(np.asarray(transform)[:3, :3]).as_quat()
    return np.r_[transform[:3, 3], quaternion[[3, 0, 1, 2]]]


def object_local(point_world: np.ndarray, object_pose: np.ndarray) -> np.ndarray:
    transform = pose7_to_transform(object_pose)
    return transform[:3, :3].T @ (np.asarray(point_world) - transform[:3, 3])


def compose_qpos(reference: np.ndarray, actual: np.ndarray, mode: str) -> np.ndarray:
    """Compose hand/object saved poses without mutating either source."""
    reference = np.asarray(reference)
    actual = np.asarray(actual)
    if reference.shape != (50,) or actual.shape != (50,) or mode not in COMBINATIONS:
        raise ValueError("expected 50-D reference/actual qpos and RR/AR/RA/AA mode")
    result = np.empty(50, dtype=np.float64)
    result[:36] = reference[:36] if mode[0] == "R" else actual[:36]
    result[36:] = reference[36:] if mode[1] == "R" else actual[36:]
    return result


def factorization_vectors(
    human_target: np.ndarray, reference_marker: np.ndarray, actual_marker: np.ndarray
) -> dict[str, np.ndarray]:
    h_to_r = np.asarray(reference_marker) - np.asarray(human_target)
    r_to_a = np.asarray(actual_marker) - np.asarray(reference_marker)
    h_to_a = np.asarray(actual_marker) - np.asarray(human_target)
    if not np.allclose(h_to_a, h_to_r + r_to_a, atol=1e-12, rtol=0.0):
        raise ValueError("marker vector factorization failed")
    return {"H_to_R": h_to_r, "R_to_A": r_to_a, "H_to_A": h_to_a}


def command_reference_endpoint(endpoint: int) -> int:
    if endpoint < 0:
        raise ValueError("endpoint must be non-negative")
    return endpoint


def actuator_unit(joint_type: int) -> str:
    if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
        return "m"
    if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
        return "rad"
    return "unsupported"


def distance_semantics(raw: float, distmax: float, fromto: np.ndarray) -> tuple[str, float | None]:
    """Keep MuJoCo's distmax sentinel distinct from a measured clearance."""
    if not math.isfinite(raw):
        return "unsupported", None
    if raw >= distmax - 1e-12 and np.array_equal(np.asarray(fromto), np.zeros(6)):
        return "censored_at_distmax", None
    return "measured", float(raw)


def geom_part(body_name: str | None) -> tuple[str, str] | None:
    name = (body_name or "").lower()
    hand = "right" if name.startswith("right_hand") else "left" if name.startswith("left_hand") else None
    if hand is None:
        return None
    if name == f"{hand}_hand_link":
        return hand, "palm"
    aliases = {"mid": "middle", "middle": "middle"}
    for token in ("thumb", "index", "middle", "mid", "ring", "pinky"):
        if token in name:
            return hand, aliases.get(token, token)
    return hand, "unclassified"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    if not rows and fields is None:
        raise ValueError(f"cannot infer empty CSV schema for {path}")
    fieldnames = list(fields or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class StaticScene:
    def __init__(self, path: Path):
        self.path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        if (self.model.nq, self.model.nv, self.model.nu) != (50, 48, 36):
            raise ValueError("unexpected model dimensions")
        self.kinematics_calls = 0
        self.distance_queries = 0

    def set_qpos(self, qpos: np.ndarray) -> None:
        before_time = float(self.data.time)
        value = np.asarray(qpos, dtype=np.float64)
        self.data.qpos[:] = value
        copied = self.data.qpos.copy()
        mujoco.mj_kinematics(self.model, self.data)
        self.kinematics_calls += 1
        if self.data.time != before_time or not np.array_equal(self.data.qpos, copied):
            raise RuntimeError("static kinematics mutated time or qpos")

    def markers(self, qpos: np.ndarray) -> np.ndarray:
        self.set_qpos(qpos)
        values = np.empty((2, 6, 3), dtype=np.float64)
        for hi, hand in enumerate(HANDS):
            names = [f"{hand}_palm", *(f"{hand}_{finger}_tip" for finger in FINGERS)]
            for pi, name in enumerate(names):
                values[hi, pi] = self.data.site_xpos[self.model.site(name).id]
        return values

    def geom_distance(
        self, geom1: int, geom2: int, distmax: float = 0.2
    ) -> tuple[str, float | None, np.ndarray | None]:
        fromto = np.zeros(6, dtype=np.float64)
        raw = float(mujoco.mj_geomDistance(self.model, self.data, geom1, geom2, distmax, fromto))
        self.distance_queries += 1
        status, distance = distance_semantics(raw, distmax, fromto)
        return status, distance, None if distance is None else fromto.copy()


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or f"body_{body_id}"


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom_{geom_id}"


def relevant_explicit_pairs(model: mujoco.MjModel) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
    result = {(hand, part): [] for hand in HANDS for part in PARTS}
    object_bodies = {"right": model.body("right_object").id, "left": model.body("left_object").id}
    for pair_id in range(model.npair):
        g1, g2 = int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id])
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        for hand in HANDS:
            if b1 == object_bodies[hand]:
                parsed, hand_geom, object_geom = geom_part(_body_name(model, b2)), g2, g1
            elif b2 == object_bodies[hand]:
                parsed, hand_geom, object_geom = geom_part(_body_name(model, b1)), g1, g2
            else:
                continue
            if parsed is not None and parsed[0] == hand and parsed[1] in PARTS:
                result[parsed].append((pair_id, hand_geom, object_geom))
    return result


def relevant_floor_pairs(model: mujoco.MjModel) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
    result = {(hand, part): [] for hand in HANDS for part in PARTS}
    floor = model.geom("floor").id
    for pair_id in range(model.npair):
        g1, g2 = int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id])
        if g1 == floor:
            parsed, hand_geom = geom_part(_body_name(model, model.geom_bodyid[g2])), g2
        elif g2 == floor:
            parsed, hand_geom = geom_part(_body_name(model, model.geom_bodyid[g1])), g1
        else:
            continue
        if parsed is not None and parsed[1] in PARTS:
            result[parsed].append((pair_id, hand_geom, floor))
    return result


def relevant_object_floor_pairs(model: mujoco.MjModel, object_body: int) -> list[tuple[int, int, int]]:
    result = []
    floor = model.geom("floor").id
    for pair_id in range(model.npair):
        g1, g2 = int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id])
        if g1 == floor and int(model.geom_bodyid[g2]) == object_body:
            result.append((pair_id, g2, floor))
        elif g2 == floor and int(model.geom_bodyid[g1]) == object_body:
            result.append((pair_id, g1, floor))
    return result


def min_pair_distance(
    scene: StaticScene, pairs: list[tuple[int, int, int]]
) -> dict[str, Any]:
    measured: list[tuple[float, int, int, int]] = []
    censored = 0
    unsupported = 0
    for pair_id, hand_geom, object_geom in pairs:
        status, value, fromto = scene.geom_distance(hand_geom, object_geom)
        if status == "measured":
            assert fromto is not None
            measured.append((float(value), pair_id, hand_geom, object_geom, fromto))
        elif status == "censored_at_distmax":
            censored += 1
        else:
            unsupported += 1
    if not measured:
        return {
            "status": "censored_or_unsupported",
            "distance_m": None,
            "pair_id": None,
            "hand_geom": None,
            "object_geom": None,
            "censored_pairs": censored,
            "unsupported_pairs": unsupported,
            "nearest_point_hand": None,
            "nearest_point_other": None,
        }
    value, pair_id, hand_geom, object_geom, fromto = min(measured, key=lambda item: item[0])
    return {
        "status": "measured_min_over_explicit_pairs",
        "distance_m": value,
        "pair_id": pair_id,
        "hand_geom": _geom_name(scene.model, hand_geom),
        "object_geom": _geom_name(scene.model, object_geom),
        "censored_pairs": censored,
        "unsupported_pairs": unsupported,
        "nearest_point_hand": fromto[:3],
        "nearest_point_other": fromto[3:],
    }


def _object_pose(qpos: np.ndarray, hand: str) -> np.ndarray:
    return np.asarray(qpos[36:43] if hand == "right" else qpos[43:50])


def _human_markers(human: dict[str, np.ndarray], endpoint: int) -> tuple[np.ndarray, np.ndarray]:
    gt = np.empty((2, 6, 3), dtype=np.float64)
    target = np.empty_like(gt)
    gt[:, 0] = human["joint_positions_sim"][endpoint, :, 0]
    gt[:, 1:] = np.take(human["joint_positions_sim"][endpoint], HUMAN_TIPS, axis=1)
    target[:, 0] = human["T_sim_wrist_target"][endpoint, :, :3, 3]
    target[:, 1:] = human["T_sim_fingertip_target"][endpoint, :, :, :3, 3]
    return gt, target


def _load_mesh_queries(paths: dict[str, Path]) -> dict[str, trimesh.proximity.ProximityQuery]:
    result = {}
    for role in ("tool", "target"):
        mesh = trimesh.load_mesh(paths[f"{role}_mesh"], process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"{role} mesh is not a triangle mesh")
        mesh.apply_scale(0.01)
        result[role] = trimesh.proximity.ProximityQuery(mesh)
    return result


def _surface_distance(
    query: trimesh.proximity.ProximityQuery, point: np.ndarray, pose: np.ndarray
) -> float:
    local = object_local(point, pose)
    _, distance, _ = query.on_surface(local[None])
    return float(distance[0])


def _contact_index(
    contacts: dict[str, np.ndarray], counterpart: dict[str, str]
) -> dict[tuple[int, str, str], dict[str, Any]]:
    result: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in range(len(contacts["source_endpoint"])):
        source = int(contacts["source_endpoint"][row])
        roles = (str(contacts["role1"][row]), str(contacts["role2"][row]))
        for role in roles:
            if role.startswith("right_hand:") or role.startswith("left_hand:"):
                hand, part = role.split(":", 1)
                hand = hand.removesuffix("_hand")
                if counterpart.get(hand) not in roles:
                    continue
                key = (source, hand, part)
                item = result.setdefault(
                    key,
                    {"rows": 0, "normal_force_N": 0.0, "first_global_substep": None, "geom_pairs": set()},
                )
                item["rows"] += 1
                item["normal_force_N"] += float(contacts["wrench_contact_force_torque"][row, 0])
                global_step = int(contacts["global_substep"][row])
                item["first_global_substep"] = (
                    global_step
                    if item["first_global_substep"] is None
                    else min(global_step, item["first_global_substep"])
                )
                item["geom_pairs"].add(
                    (str(contacts["geom1_name"][row]), str(contacts["geom2_name"][row]))
                )
    return result


def _actuator_contract(model: mujoco.MjModel) -> list[dict[str, Any]]:
    rows = []
    seen_qpos = []
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        qpos_address = int(model.jnt_qposadr[joint])
        unit = actuator_unit(int(model.jnt_type[joint]))
        rows.append(
            {
                "actuator": actuator,
                "actuator_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator),
                "joint": joint,
                "joint_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                "qpos_address": qpos_address,
                "transmission_type": int(model.actuator_trntype[actuator]),
                "gear0": float(model.actuator_gear[actuator, 0]),
                "unit": unit,
                "ctrl_low": float(model.actuator_ctrlrange[actuator, 0]),
                "ctrl_high": float(model.actuator_ctrlrange[actuator, 1]),
            }
        )
        seen_qpos.append(qpos_address)
    if seen_qpos != list(range(36)) or any(row["unit"] == "unsupported" for row in rows):
        raise ValueError("actuator/qpos direct position contract changed")
    return rows


def _write_control_tracking(
    output: Path,
    contract: list[dict[str, Any]],
    endpoints: dict[str, np.ndarray],
    substeps: dict[str, np.ndarray],
    robot: dict[str, np.ndarray],
    initial: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    actuator_force_by_endpoint: dict[int, np.ndarray] = {}
    for endpoint in range(1, 21):
        matches = np.flatnonzero(
            (substeps["outcome_endpoint"] == endpoint) & (substeps["substep"] == 9)
        )
        if len(matches) == 1:
            actuator_force_by_endpoint[endpoint] = substeps["actuator_force"][matches[0]]
    for endpoint in range(21):
        reference_endpoint = command_reference_endpoint(endpoint)
        for item in contract:
            actuator = item["actuator"]
            qaddr = item["qpos_address"]
            actual_ctrl = float(endpoints["B_ctrl"][endpoint, actuator])
            reference_ctrl = float(robot["ctrl"][reference_endpoint, actuator])
            rows.append(
                {
                    "endpoint": endpoint,
                    "source_endpoint": "" if endpoint == 0 else endpoint - 1,
                    "command_reference_endpoint": reference_endpoint,
                    **item,
                    "ctrl_actual": actual_ctrl,
                    "ctrl_reference": reference_ctrl,
                    "ctrl_minus_reference": actual_ctrl - reference_ctrl,
                    "qpos_actual": float(endpoints["B_qpos"][endpoint, qaddr]),
                    "qpos_reference": float(robot["qpos"][endpoint, qaddr]),
                    "qpos_minus_reference": float(
                        endpoints["B_qpos"][endpoint, qaddr] - robot["qpos"][endpoint, qaddr]
                    ),
                    "qvel_actual": float(endpoints["B_qvel"][endpoint, qaddr]),
                    "actuator_force": (
                        ""
                        if endpoint not in actuator_force_by_endpoint
                        else float(actuator_force_by_endpoint[endpoint][actuator])
                    ),
                }
            )
    write_csv(output / "control_tracking.csv", rows)
    first_command = robot["ctrl"][1] - initial["ctrl"]
    return rows, {
        "s0_qpos_matches_accepted_initial_max_abs": float(
            np.max(np.abs(endpoints["B_qpos"][0] - initial["qpos"]))
        ),
        "s0_hand_qpos_vs_reference0_max_abs": float(
            np.max(np.abs(endpoints["B_qpos"][0, :36] - robot["qpos"][0, :36]))
        ),
        "s0_ctrl_vs_reference0_max_abs": float(
            np.max(np.abs(endpoints["B_ctrl"][0] - robot["ctrl"][0]))
        ),
        "first_command_right_l2": float(np.linalg.norm(first_command[:18])),
        "first_command_left_l2": float(np.linalg.norm(first_command[18:])),
        "first_command_right_max_abs": float(np.max(np.abs(first_command[:18]))),
        "first_command_left_max_abs": float(np.max(np.abs(first_command[18:]))),
        "endpoints1_20_ctrl_reference_max_abs": float(
            np.max(np.abs(endpoints["B_ctrl"][1:] - robot["ctrl"][1:21]))
        ),
    }


def _write_geometry(
    output: Path,
    scene: StaticScene,
    human: dict[str, np.ndarray],
    robot: dict[str, np.ndarray],
    endpoints: dict[str, np.ndarray],
    contacts: dict[str, np.ndarray],
    mesh_queries: dict[str, trimesh.proximity.ProximityQuery],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = relevant_explicit_pairs(scene.model)
    floor_pairs = relevant_floor_pairs(scene.model)
    object_contact_index = _contact_index(contacts, {"right": "tool", "left": "target"})
    floor_contact_index = _contact_index(contacts, {"right": "floor", "left": "floor"})
    reference_source = robot["qpos"].copy()
    actual_source = endpoints["B_qpos"].copy()
    rows: list[dict[str, Any]] = []
    factorization_max = 0.0
    for endpoint in range(21):
        gt, target = _human_markers(human, endpoint)
        marker_by_mode = {}
        qpos_by_mode = {}
        for mode in COMBINATIONS:
            qpos = compose_qpos(reference_source[endpoint], actual_source[endpoint], mode)
            qpos_by_mode[mode] = qpos
            marker_by_mode[mode] = scene.markers(qpos)
        if not np.array_equal(qpos_by_mode["AA"], actual_source[endpoint].astype(np.float64)):
            raise RuntimeError("AA composition differs from recorded qpos")
        for hi, hand in enumerate(HANDS):
            object_role = "tool" if hand == "right" else "target"
            for pi, part in enumerate(PARTS):
                factor = factorization_vectors(
                    target[hi, pi], marker_by_mode["RR"][hi, pi], marker_by_mode["AA"][hi, pi]
                )
                factorization_max = max(
                    factorization_max,
                    float(np.max(np.abs(factor["H_to_A"] - factor["H_to_R"] - factor["R_to_A"]))),
                )
                for mode in COMBINATIONS:
                    qpos = qpos_by_mode[mode]
                    scene.set_qpos(qpos)
                    marker = marker_by_mode[mode][hi, pi]
                    object_pose = _object_pose(qpos, hand)
                    local = object_local(marker, object_pose)
                    collision = min_pair_distance(scene, pairs[(hand, part)])
                    floor_collision = min_pair_distance(scene, floor_pairs[(hand, part)])
                    contact = object_contact_index.get((endpoint, hand, part), {}) if mode == "AA" else {}
                    floor_contact = floor_contact_index.get((endpoint, hand, part), {}) if mode == "AA" else {}
                    human_gt_surface = _surface_distance(
                        mesh_queries[object_role], gt[hi, pi], object_pose
                    )
                    human_target_surface = _surface_distance(
                        mesh_queries[object_role], target[hi, pi], object_pose
                    )
                    rows.append(
                        {
                            "endpoint": endpoint,
                            "timestamp_s": endpoint / 30.0,
                            "combination": mode,
                            "hand": hand,
                            "part": part,
                            "object_role": object_role,
                            "human_gt_x": float(gt[hi, pi, 0]),
                            "human_gt_y": float(gt[hi, pi, 1]),
                            "human_gt_z": float(gt[hi, pi, 2]),
                            "human_target_x": float(target[hi, pi, 0]),
                            "human_target_y": float(target[hi, pi, 1]),
                            "human_target_z": float(target[hi, pi, 2]),
                            "marker_world_x": float(marker[0]),
                            "marker_world_y": float(marker[1]),
                            "marker_world_z": float(marker[2]),
                            "marker_object_x": float(local[0]),
                            "marker_object_y": float(local[1]),
                            "marker_object_z": float(local[2]),
                            "human_gt_to_target_m": float(np.linalg.norm(gt[hi, pi] - target[hi, pi])),
                            "human_gt_visual_triangle_distance_m": human_gt_surface,
                            "human_target_visual_triangle_distance_m": human_target_surface,
                            "H_to_R_x": float(factor["H_to_R"][0]),
                            "H_to_R_y": float(factor["H_to_R"][1]),
                            "H_to_R_z": float(factor["H_to_R"][2]),
                            "H_to_R_norm_m": float(np.linalg.norm(factor["H_to_R"])),
                            "R_to_A_x": float(factor["R_to_A"][0]),
                            "R_to_A_y": float(factor["R_to_A"][1]),
                            "R_to_A_z": float(factor["R_to_A"][2]),
                            "R_to_A_norm_m": float(np.linalg.norm(factor["R_to_A"])),
                            "H_to_A_norm_m": float(np.linalg.norm(factor["H_to_A"])),
                            "visual_triangle_distance_m": _surface_distance(
                                mesh_queries[object_role], marker, object_pose
                            ),
                            "collision_distance_status": collision["status"],
                            "collision_distance_m": "" if collision["distance_m"] is None else collision["distance_m"],
                            "collision_pair_id": "" if collision["pair_id"] is None else collision["pair_id"],
                            "collision_hand_geom": collision["hand_geom"] or "",
                            "collision_object_geom": collision["object_geom"] or "",
                            "collision_censored_pairs": collision["censored_pairs"],
                            "collision_unsupported_pairs": collision["unsupported_pairs"],
                            "collision_nearest_point_hand_x": "" if collision["nearest_point_hand"] is None else collision["nearest_point_hand"][0],
                            "collision_nearest_point_hand_y": "" if collision["nearest_point_hand"] is None else collision["nearest_point_hand"][1],
                            "collision_nearest_point_hand_z": "" if collision["nearest_point_hand"] is None else collision["nearest_point_hand"][2],
                            "collision_nearest_point_object_x": "" if collision["nearest_point_other"] is None else collision["nearest_point_other"][0],
                            "collision_nearest_point_object_y": "" if collision["nearest_point_other"] is None else collision["nearest_point_other"][1],
                            "collision_nearest_point_object_z": "" if collision["nearest_point_other"] is None else collision["nearest_point_other"][2],
                            "floor_collision_distance_status": floor_collision["status"],
                            "floor_collision_distance_m": "" if floor_collision["distance_m"] is None else floor_collision["distance_m"],
                            "floor_collision_pair_id": "" if floor_collision["pair_id"] is None else floor_collision["pair_id"],
                            "floor_collision_hand_geom": floor_collision["hand_geom"] or "",
                            "floor_collision_geom": floor_collision["object_geom"] or "",
                            "measured_object_contact_rows_outgoing_interval": contact.get("rows", 0),
                            "measured_object_contact_normal_force_sum_N": contact.get("normal_force_N", 0.0),
                            "measured_object_first_global_substep": (
                                "" if contact.get("first_global_substep") is None else contact["first_global_substep"]
                            ),
                            "measured_floor_contact_rows_outgoing_interval": floor_contact.get("rows", 0),
                            "measured_floor_contact_normal_force_sum_N": floor_contact.get("normal_force_N", 0.0),
                            "measured_contact_scope": "source endpoint interval; AA solve data only; object and floor separated" if mode == "AA" else "not_applicable_static_combination",
                        }
                    )
    if not np.array_equal(reference_source, robot["qpos"]) or not np.array_equal(actual_source, endpoints["B_qpos"]):
        raise RuntimeError("source qpos arrays mutated")
    write_csv(output / "digit_geometry.csv", rows)
    return rows, {
        "marker_vector_factorization_max_abs_m": factorization_max,
        "object_explicit_pair_counts": {f"{hand}_{part}": len(value) for (hand, part), value in pairs.items()},
        "floor_explicit_pair_counts": {f"{hand}_{part}": len(value) for (hand, part), value in floor_pairs.items()},
    }


def _write_contact_paths(
    output: Path, model: mujoco.MjModel, contacts: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_pairs = relevant_explicit_pairs(model)
    floor_pairs = relevant_floor_pairs(model)
    rows = []
    for hand in HANDS:
        for counterpart_role, counterpart_body, pair_table in (
            (
                "tool" if hand == "right" else "target",
                model.body("right_object" if hand == "right" else "left_object").id,
                object_pairs,
            ),
            ("floor", model.body(0).id, floor_pairs),
        ):
            for part in PARTS:
                explicit = pair_table[(hand, part)]
                observed_rows = []
                for row in range(len(contacts["source_endpoint"])):
                    roles = (str(contacts["role1"][row]), str(contacts["role2"][row]))
                    expected_role = f"{hand}_hand:{part}"
                    if expected_role in roles and counterpart_role in roles:
                        observed_rows.append(row)
                auto_bits = []
                for _, hand_geom, object_geom in explicit:
                    auto_bits.append(
                        bool(
                            (model.geom_contype[hand_geom] & model.geom_conaffinity[object_geom])
                            or (model.geom_contype[object_geom] & model.geom_conaffinity[hand_geom])
                        )
                    )
                observed_pairs = sorted(
                    {
                        f"{contacts['geom1_name'][i]} <-> {contacts['geom2_name'][i]}"
                        for i in observed_rows
                    }
                )
                rows.append(
                    {
                        "primary_role": f"{hand}_hand:{part}",
                        "hand": hand,
                        "part": part,
                        "counterpart_role": counterpart_role,
                        "counterpart_body_id": int(counterpart_body),
                        "explicit_pair_count": len(explicit),
                        "automatic_bitmask_possible_for_any_explicit_pair": any(auto_bits),
                        "path_interpretation": "explicit pair path exists" if explicit else "no explicit pair found",
                        "observed_contact_rows_0_20": len(observed_rows),
                        "first_observed_global_substep": (
                            "" if not observed_rows else int(np.min(contacts["global_substep"][observed_rows]))
                        ),
                        "last_observed_global_substep": (
                            "" if not observed_rows else int(np.max(contacts["global_substep"][observed_rows]))
                        ),
                        "observed_geom_pairs": " | ".join(observed_pairs),
                    }
                )
    for primary_role, body_name in (("tool", "right_object"), ("target", "left_object")):
        body_id = model.body(body_name).id
        explicit = relevant_object_floor_pairs(model, body_id)
        observed_rows = [
            row
            for row in range(len(contacts["source_endpoint"]))
            if primary_role in (str(contacts["role1"][row]), str(contacts["role2"][row]))
            and "floor" in (str(contacts["role1"][row]), str(contacts["role2"][row]))
        ]
        rows.append(
            {
                "primary_role": primary_role,
                "hand": "",
                "part": "object",
                "counterpart_role": "floor",
                "counterpart_body_id": int(model.body(0).id),
                "explicit_pair_count": len(explicit),
                "automatic_bitmask_possible_for_any_explicit_pair": False,
                "path_interpretation": "explicit pair path exists" if explicit else "no explicit pair found",
                "observed_contact_rows_0_20": len(observed_rows),
                "first_observed_global_substep": (
                    "" if not observed_rows else int(np.min(contacts["global_substep"][observed_rows]))
                ),
                "last_observed_global_substep": (
                    "" if not observed_rows else int(np.max(contacts["global_substep"][observed_rows]))
                ),
                "observed_geom_pairs": " | ".join(
                    sorted(
                        {
                            f"{contacts['geom1_name'][i]} <-> {contacts['geom2_name'][i]}"
                            for i in observed_rows
                        }
                    )
                ),
            }
        )
    unclassified = sorted(
        {
            role
            for role in np.r_[contacts["role1"], contacts["role2"]].astype(str)
            if role.startswith("right_hand:") or role.startswith("left_hand:")
            if role.endswith(":unclassified") or role.endswith(":other")
        }
    )
    write_csv(output / "contact_path_checks.csv", rows)
    return rows, {"unclassified_or_other_hand_contact_roles": unclassified}


def _contact_episode_summary(
    contacts: dict[str, np.ndarray], role_a: str, role_b: str
) -> dict[str, Any]:
    role1 = contacts["role1"].astype(str)
    role2 = contacts["role2"].astype(str)
    mask = ((role1 == role_a) & (role2 == role_b)) | ((role1 == role_b) & (role2 == role_a))
    if not np.any(mask):
        return {
            "rows": 0,
            "physics_substeps_with_contact": 0,
            "first_global_substep": None,
            "last_global_substep": None,
            "first_substep_normal_force_sum_N": None,
            "max_substep_normal_force_sum_N": None,
        }
    steps = np.unique(contacts["global_substep"][mask])
    forces = contacts["wrench_contact_force_torque"][:, 0]
    per_step = [
        float(np.sum(forces[mask & (contacts["global_substep"] == step)])) for step in steps
    ]
    return {
        "rows": int(np.sum(mask)),
        "physics_substeps_with_contact": len(steps),
        "first_global_substep": int(steps[0]),
        "last_global_substep": int(steps[-1]),
        "first_substep_normal_force_sum_N": per_step[0],
        "max_substep_normal_force_sum_N": max(per_step),
    }


def _camera(lookat: np.ndarray, distance: float, azimuth: float = 90.0, elevation: float = -55.0) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def _scene_option(*, human: bool = False) -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    if not human:
        option.geomgroup[1] = 1
    option.sitegroup[:] = 0
    return option


def _annotate(image: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    result = image.copy()
    text = list(lines)
    cv2.rectangle(result, (0, 0), (result.shape[1], 10 + 22 * len(text)), (0, 0, 0), -1)
    for index, line in enumerate(text):
        cv2.putText(result, line, (9, 21 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _add_points(renderer: mujoco.Renderer, points: np.ndarray, hand: str) -> None:
    colors = {
        "right": np.array([0.15, 0.70, 1.0, 1.0], dtype=np.float32),
        "left": np.array([1.0, 0.52, 0.10, 1.0], dtype=np.float32),
    }
    for point in points:
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.0045), point, np.eye(3).reshape(-1), colors[hand])
        renderer.scene.ngeom += 1


def _add_human(renderer: mujoco.Renderer, joints: np.ndarray) -> None:
    colors = (np.array([0.15, 0.70, 1.0, 1.0], np.float32), np.array([1.0, 0.52, 0.10, 1.0], np.float32))
    for hi in range(2):
        for parent, child in HUMAN_BONES:
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), colors[hi])
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0028, joints[hi, parent], joints[hi, child])
            renderer.scene.ngeom += 1


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def _read_rgb(video: Path, count: int = 21) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("RGB stream ended early")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def _human_object_qpos(model: mujoco.MjModel, transforms: np.ndarray) -> np.ndarray:
    qpos = model.qpos0.copy()
    qpos[36:43] = transform_to_pose7(transforms[0])
    qpos[43:50] = transform_to_pose7(transforms[1])
    return qpos


def _render_outputs(
    output: Path,
    scene: StaticScene,
    human: dict[str, np.ndarray],
    robot: dict[str, np.ndarray],
    endpoints: dict[str, np.ndarray],
    rgb_video: Path,
) -> dict[str, int]:
    width, height = 480, 360
    renderer = mujoco.Renderer(scene.model, width=width, height=height, max_geom=10000)
    updates = 0
    rgb = _read_rgb(rgb_video)
    right_center = robot["qpos"][0, 36:39]
    left_center = robot["qpos"][0, 43:46]
    global_center = (right_center + left_center) / 2

    def render(qpos: np.ndarray, camera: mujoco.MjvCamera, label: str, highlights: bool = True) -> np.ndarray:
        nonlocal updates
        markers = scene.markers(qpos)
        renderer.update_scene(scene.data, camera=camera, scene_option=_scene_option())
        if highlights:
            _add_points(renderer, markers[0, [0, 1, 2]], "right")
            _add_points(renderer, markers[1, [0, 1, 4, 5]], "left")
        updates += 1
        return _annotate(renderer.render().copy(), [label, "sites: right palm/thumb/index; left palm/thumb/ring/pinky"])

    def render_human(endpoint: int, camera: mujoco.MjvCamera) -> np.ndarray:
        nonlocal updates
        qpos = _human_object_qpos(scene.model, human["T_sim_object_reference"][endpoint])
        scene.set_qpos(qpos)
        renderer.update_scene(scene.data, camera=camera, scene_option=_scene_option(human=True))
        _add_human(renderer, human["joint_positions_sim"][endpoint])
        updates += 1
        return _annotate(renderer.render().copy(), ["HUMAN GT + reference objects", "fixed sim camera; not RGB registered"])

    global_camera = _camera(global_center, 0.82, 90.0, -88.0)
    selected = (0, 10, 14, 15, 16, 20)
    sheet_rows = []
    for endpoint in selected:
        rgb_panel = _annotate(_letterbox(rgb[endpoint], width, height), [f"RGB frame {endpoint}", "independent camera"])
        human_panel = render_human(endpoint, global_camera)
        rr = render(robot["qpos"][endpoint], global_camera, f"MINK reference endpoint {endpoint}")
        aa = render(endpoints["B_qpos"][endpoint], global_camera, f"Replay actual endpoint {endpoint}")
        sheet_rows.append(np.concatenate([rgb_panel, human_panel, rr, aa], axis=1))
    _save_image(output / "review/human_reference_actual_early.png", np.concatenate(sheet_rows, axis=0))

    for endpoint in (14, 15, 16):
        panels = []
        for mode in COMBINATIONS:
            qpos = compose_qpos(robot["qpos"][endpoint], endpoints["B_qpos"][endpoint], mode)
            panels.append(render(qpos, global_camera, f"{mode} endpoint {endpoint} | offline geometry combination"))
        _save_image(output / f"review/pose_factorization_{endpoint:03d}.png", np.concatenate(panels, axis=1))

    def local_video(path: Path, endpoints_range: range, center: np.ndarray, distance: float, parts: str) -> None:
        camera = _camera(center, distance, 90.0, -52.0)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 3.0, (width * 3, height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open {path}")
        try:
            for endpoint in endpoints_range:
                human_panel = render_human(endpoint, camera)
                reference_panel = render(robot["qpos"][endpoint], camera, f"reference {parts} | endpoint {endpoint}")
                actual_panel = render(endpoints["B_qpos"][endpoint], camera, f"actual {parts} | endpoint {endpoint}")
                frame = np.concatenate([human_panel, reference_panel, actual_panel], axis=1)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    local_video(output / "review/right_thumb_index_12_20.mp4", range(12, 21), right_center, 0.38, "right thumb/index")
    local_video(output / "review/left_support_0_15.mp4", range(0, 16), left_center, 0.50, "left support")
    renderer.close()
    return {"renderer_scene_updates": updates, "rgb_frames_read": len(rgb)}


def _plot_curves(output: Path, geometry: list[dict[str, Any]], control: list[dict[str, Any]]) -> None:
    focus = (("right", "thumb"), ("right", "index"), ("left", "thumb"), ("left", "ring"), ("left", "pinky"))
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for hand, part in focus:
        rows = [r for r in geometry if r["combination"] == "AA" and r["hand"] == hand and r["part"] == part]
        axes[0].plot([r["endpoint"] for r in rows], [1000 * r["H_to_R_norm_m"] for r in rows], label=f"{hand} {part}")
        axes[1].plot([r["endpoint"] for r in rows], [1000 * r["R_to_A_norm_m"] for r in rows], label=f"{hand} {part}")
    axes[0].set_ylabel("H target -> reference site (mm)")
    axes[1].set_ylabel("reference site -> actual site (mm)")
    axes[1].set_xlabel("endpoint")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(output / "review/marker_factorization.png", dpi=160)
    plt.close(figure)


def _plot_object_and_contact_timeline(
    output: Path,
    robot: dict[str, np.ndarray],
    endpoints: dict[str, np.ndarray],
    contacts: dict[str, np.ndarray],
    geometry: list[dict[str, Any]],
    control: list[dict[str, Any]],
) -> None:
    focus = (("right", "thumb"), ("right", "index"), ("left", "thumb"), ("left", "ring"), ("left", "pinky"))
    endpoint = np.arange(21)
    actual = endpoints["B_qpos"]
    reference = robot["qpos"][:21]
    tool_position = 1000 * np.linalg.norm(actual[:, 36:39] - reference[:, 36:39], axis=1)
    target_position = 1000 * np.linalg.norm(actual[:, 43:46] - reference[:, 43:46], axis=1)
    pair_actual = actual[:, 36:39] - actual[:, 43:46]
    pair_reference = reference[:, 36:39] - reference[:, 43:46]
    pair_position = 1000 * np.linalg.norm(pair_actual - pair_reference, axis=1)

    def rotation_curve(start: int) -> np.ndarray:
        values = []
        for actual_pose, reference_pose in zip(actual[:, start:start + 7], reference[:, start:start + 7]):
            actual_rotation = Rotation.from_quat(actual_pose[[4, 5, 6, 3]])
            reference_rotation = Rotation.from_quat(reference_pose[[4, 5, 6, 3]])
            values.append((actual_rotation.inv() * reference_rotation).magnitude())
        return np.asarray(values)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(endpoint, tool_position, label="bowl/tool position error")
    axes[0].plot(endpoint, target_position, label="tray/target position error")
    axes[0].plot(endpoint, pair_position, label="bowl-tray relative position error", linewidth=2)
    axes[1].plot(endpoint, rotation_curve(36), label="bowl/tool rotation error")
    axes[1].plot(endpoint, rotation_curve(43), label="tray/target rotation error")
    axes[0].set_ylabel("position error (mm)")
    axes[1].set_ylabel("rotation error (rad)")
    axes[1].set_xlabel("endpoint")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "review/object_tracking_0_20.png", dpi=160)
    plt.close(figure)

    role1 = contacts["role1"].astype(str)
    role2 = contacts["role2"].astype(str)
    channels = (
        ("right index–bowl", "right_hand:index", "tool"),
        ("right thumb–bowl", "right_hand:thumb", "tool"),
        ("right thumb–floor", "right_hand:thumb", "floor"),
        ("left ring–tray", "left_hand:ring", "target"),
        ("left pinky–tray", "left_hand:pinky", "target"),
        ("tray–floor", "target", "floor"),
        ("bowl–floor", "tool", "floor"),
    )
    figure, axis = plt.subplots(figsize=(13, 5))
    for row, (label, first, second) in enumerate(channels):
        mask = ((role1 == first) & (role2 == second)) | ((role1 == second) & (role2 == first))
        steps = np.unique(contacts["global_substep"][mask])
        axis.scatter(steps / 10.0, np.full(len(steps), row), marker="|", s=80)
    axis.set_yticks(range(len(channels)), [item[0] for item in channels])
    axis.set_xlabel("source endpoint + substep/10 (saved solver contact rows)")
    axis.set_xlim(0, 20.1)
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "review/contact_timeline_0_20.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for hand, part in focus:
        rows = [r for r in geometry if r["combination"] == "AA" and r["hand"] == hand and r["part"] == part]
        axes[0].plot([r["endpoint"] for r in rows], [1000 * r["visual_triangle_distance_m"] for r in rows], label=f"{hand} {part}")
        values = [np.nan if r["collision_distance_m"] == "" else 1000 * float(r["collision_distance_m"]) for r in rows]
        axes[1].plot([r["endpoint"] for r in rows], values, label=f"{hand} {part}")
    axes[0].set_ylabel("tip/palm site to visual mesh (mm)")
    axes[1].set_ylabel("min queryable explicit collision pair (mm)")
    axes[1].set_xlabel("endpoint")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(output / "review/surface_and_collision_distance.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    selected_actuators = (0, 1, 2, 9, 10, 11, 18, 19, 20, 32, 33, 34, 35)
    for actuator in selected_actuators:
        rows = [r for r in control if r["actuator"] == actuator]
        axes[0].plot([r["endpoint"] for r in rows], [r["ctrl_actual"] for r in rows], label=rows[0]["actuator_name"])
        axes[1].plot([r["endpoint"] for r in rows], [r["qpos_minus_reference"] for r in rows], label=rows[0]["actuator_name"])
    axes[0].set_ylabel("position target (m or rad; separate labels)")
    axes[1].set_ylabel("qpos - reference (m or rad; do not norm)")
    axes[1].set_xlabel("endpoint")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=6, ncol=3)
    figure.tight_layout()
    figure.savefig(output / "review/control_tracking_selected.png", dpi=160)
    plt.close(figure)


def _write_intent_table(output: Path) -> None:
    rows = [
        {"frames": "0-10", "hand": "right", "object": "bowl/tool", "visible_parts": "thumb,index,palm", "observed_action": "approach; index side leads", "visibility": "clear overhead", "uncertainty": "no visible load/contact; RGB is not metric", "interpretation": "no evidence that thumb contact is required in this phase"},
        {"frames": "12-16", "hand": "right", "object": "bowl/tool", "visible_parts": "thumb,index,palm", "observed_action": "continued approach toward near rim", "visibility": "thumb and index visible; rim proximity partly occluded", "uncertainty": "30 Hz cannot resolve substep contact", "interpretation": "index remains closer than thumb in human GT"},
        {"frames": "17-20", "hand": "right", "object": "bowl/tool", "visible_parts": "thumb,index", "observed_action": "index nears rim; thumb follows", "visibility": "partial rim occlusion", "uncertainty": "finger-pad contact cannot be certified from RGB", "interpretation": "possible contact preparation, not demonstrated thumb-index force closure"},
        {"frames": "0-5", "hand": "left", "object": "tray/target", "visible_parts": "ring,pinky,palm", "observed_action": "hand approaches tray edge while tray remains table-supported", "visibility": "clear overhead", "uncertainty": "surface load cannot be inferred", "interpretation": "human ring-side landmarks are initially nearest"},
        {"frames": "6-15", "hand": "left", "object": "tray/target", "visible_parts": "thumb,ring,pinky,palm", "observed_action": "pose changes around near tray edge", "visibility": "partially blurred during motion", "uncertainty": "which finger carries load is not visually resolvable", "interpretation": "support is shared with table; do not require a fixed finger mode"},
        {"frames": "16-20", "hand": "left", "object": "tray/target", "visible_parts": "thumb,palm", "observed_action": "hand continues toward edge", "visibility": "ring/pinky partly occluded", "uncertainty": "contact intent remains ambiguous", "interpretation": "GT/reference geometry is required alongside RGB"},
    ]
    write_csv(output / "demonstration_contact_intent.csv", rows)


def _write_index(output: Path) -> None:
    document = """<!doctype html><meta charset='utf-8'><title>Early contact origin v1</title>
<style>body{background:#17191c;color:#eee;font-family:sans-serif;margin:24px}img{max-width:100%}video{width:100%;max-width:1440px}.card{margin:18px 0;padding:12px;background:#24272b}</style>
<h1>Early contact origin v1</h1><p>Static offline audit only. RR/AR/RA/AA are geometry combinations, not executable trajectories.</p>
<div class='card'><h2>RGB / human GT / reference / actual</h2><img src='review/human_reference_actual_early.png'></div>
<div class='card'><h2>Right thumb/index 12–20</h2><video controls src='review/right_thumb_index_12_20.mp4'></video></div>
<div class='card'><h2>Left support 0–15</h2><video controls src='review/left_support_0_15.mp4'></video></div>
<div class='card'><h2>Pose factorization</h2><img src='review/pose_factorization_014.png'><img src='review/pose_factorization_015.png'><img src='review/pose_factorization_016.png'></div>
<div class='card'><h2>Object/contact timeline</h2><img src='review/object_tracking_0_20.png'><img src='review/contact_timeline_0_20.png'></div>
<div class='card'><h2>Curves</h2><img src='review/marker_factorization.png'><img src='review/surface_and_collision_distance.png'><img src='review/control_tracking_selected.png'></div>"""
    (output / "index.html").write_text(document)


def _hash_output(output: Path) -> None:
    target = output / "server_artifacts.sha256"
    lines = [
        f"{sha256(path)}  {path.relative_to(output)}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != target
    ]
    target.write_text("\n".join(lines) + "\n")


def run_early_contact_origin(asset_root: Path, trace_root: Path, output: Path) -> dict[str, Any]:
    asset_root = asset_root.resolve(strict=True)
    trace_root = trace_root.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    repository = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"expected exact baseline {EXPECTED_HEAD}, got {head}")

    build = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    build.mkdir(parents=True)
    (build / "review").mkdir()
    initialization_report = asset_root / "runs/taco_pour_initialization_protocol_v2/candidate_a/report.json"
    initialization = json.loads(initialization_report.read_text())
    paths = {
        "trace_manifest": trace_root / "input_manifest.json",
        "trace_summary": trace_root / "summary.json",
        "trace_findings": trace_root / "findings.md",
        "endpoints": trace_root / "endpoints.npz",
        "substeps": trace_root / "substeps.npz",
        "contacts": trace_root / "contacts_raw.npz",
        "contact_forces": trace_root / "contact_forces.csv",
        "s0_snapshot": trace_root / "snapshots/B_s0.pt.gz",
        "historical_trajectory": asset_root / "runs/taco_pour_corrected_replay_rebase_v1/tool_only/optimized_trajectory.npz",
        "robot_reference": asset_root / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz",
        "retarget_report": asset_root / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/retarget_report.json",
        "human_reference": asset_root / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/human_reference.npz",
        "human_input_audit": asset_root / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/input_audit.json",
        "reference_model": asset_root / "runs/taco_pour_collision_semantics_combined_v1/combined_candidate_scene.xml",
        "runtime_model": asset_root / "runs/taco_pour_floor_contact_v1/candidate.xml",
        "initialization_report": initialization_report,
        "accepted_initial_state": Path(initialization["initial_state"]["path"]),
        "previous_visual_geometry": asset_root / "runs/taco_pour_hand_object_visual_audit_v1/geometry.csv",
        "previous_pair_relative": asset_root / "runs/taco_pour_hand_object_visual_audit_v1/object_pair_relative.csv",
        "rgb": asset_root / "data/taco_v1/pour_bowl_plate/rgb/taco_pour_bowl_plate_20230927_017.mp4",
        "tool_mesh": asset_root / "data/taco_v1/pour_bowl_plate/object_models/object_models_released/022_cm.obj",
        "target_mesh": asset_root / "data/taco_v1/pour_bowl_plate/object_models/object_models_released/135_cm.obj",
    }
    inputs = {
        "schema": "taco_pour_early_contact_origin_inputs_v1",
        "status": "written_before_static_model_loading",
        "source_commit": head,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "array_schemas": {
            name: npz_schema(paths[name])
            for name in ("endpoints", "substeps", "contacts", "historical_trajectory", "robot_reference", "human_reference", "accepted_initial_state")
        },
        "time_contract": {
            "endpoint": "saved state at 30 Hz; endpoint k aligns to RGB/reference frame k",
            "command": "source k -> outcome k+1 executes reference_ctrl[k+1]",
            "substep_contact": "solve_for_this_transition paired with qpos_before; qpos_after is post-integrator",
            "rgb": "one original 30 Hz frame per endpoint; never treated as ten independent substeps",
        },
        "cost_contract": {"physics": 0, "control": 0, "actions": 0, "network": 0, "optimizer": 0, "ik_solve": 0},
    }
    write_json(build / "inputs.json", inputs)

    with np.load(paths["endpoints"], allow_pickle=False) as archive:
        endpoints = {name: archive[name].copy() for name in archive.files}
    with np.load(paths["substeps"], allow_pickle=False) as archive:
        substeps = {name: archive[name].copy() for name in archive.files}
    with np.load(paths["contacts"], allow_pickle=False) as archive:
        contacts = {name: archive[name].copy() for name in archive.files}
    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        robot = {name: archive[name].copy() for name in archive.files}
    with np.load(paths["human_reference"], allow_pickle=False) as archive:
        human = {name: archive[name].copy() for name in archive.files}
    with np.load(paths["accepted_initial_state"], allow_pickle=False) as archive:
        initial = {name: archive[name].copy() for name in archive.files}
    if endpoints["reference_endpoint"].tolist() != list(range(21)):
        raise ValueError("endpoint coverage changed")
    if len(substeps["global_substep"]) != 200 or len(contacts["source_endpoint"]) == 0:
        raise ValueError("saved trace coverage changed")
    if not np.array_equal(endpoints["A_qpos"], endpoints["B_qpos"]):
        raise ValueError("observer identity regression")

    reference_scene = StaticScene(paths["reference_model"])
    runtime_scene = StaticScene(paths["runtime_model"])
    contract = _actuator_contract(runtime_scene.model)
    compatibility = {
        "dimensions_equal": (reference_scene.model.nq, reference_scene.model.nv, reference_scene.model.nu) == (runtime_scene.model.nq, runtime_scene.model.nv, runtime_scene.model.nu),
        "joint_names_equal": all(
            mujoco.mj_id2name(reference_scene.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            == mujoco.mj_id2name(runtime_scene.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(runtime_scene.model.njnt)
        ),
    }
    reference_markers = reference_scene.markers(robot["qpos"][0])
    runtime_markers = runtime_scene.markers(robot["qpos"][0])
    compatibility["reference_qpos0_marker_max_abs_m"] = float(np.max(np.abs(reference_markers - runtime_markers)))
    if not compatibility["dimensions_equal"] or not compatibility["joint_names_equal"]:
        raise ValueError(f"model qpos layout mismatch: {compatibility}")

    mesh_queries = _load_mesh_queries(paths)
    geometry, geometry_checks = _write_geometry(build, runtime_scene, human, robot, endpoints, contacts, mesh_queries)
    control, control_summary = _write_control_tracking(build, contract, endpoints, substeps, robot, initial)
    contact_paths, contact_checks = _write_contact_paths(build, runtime_scene.model, contacts)
    _write_intent_table(build)
    render_counts = _render_outputs(build, runtime_scene, human, robot, endpoints, paths["rgb"])
    _plot_curves(build, geometry, control)
    _plot_object_and_contact_timeline(build, robot, endpoints, contacts, geometry, control)
    _write_index(build)

    def geom_row(endpoint: int, mode: str, hand: str, part: str) -> dict[str, Any]:
        return next(r for r in geometry if r["endpoint"] == endpoint and r["combination"] == mode and r["hand"] == hand and r["part"] == part)

    def rotation_error(actual_pose: np.ndarray, reference_pose: np.ndarray) -> float:
        actual_rotation = Rotation.from_quat(actual_pose[[4, 5, 6, 3]])
        reference_rotation = Rotation.from_quat(reference_pose[[4, 5, 6, 3]])
        return float((actual_rotation.inv() * reference_rotation).magnitude())

    def object_error(endpoint: int) -> dict[str, float]:
        actual = endpoints["B_qpos"][endpoint]
        reference = robot["qpos"][endpoint]
        actual_pair = actual[36:39] - actual[43:46]
        reference_pair = reference[36:39] - reference[43:46]
        return {
            "tool_position_error_mm": 1000 * float(np.linalg.norm(actual[36:39] - reference[36:39])),
            "target_position_error_mm": 1000 * float(np.linalg.norm(actual[43:46] - reference[43:46])),
            "tool_target_relative_position_error_mm": 1000 * float(np.linalg.norm(actual_pair - reference_pair)),
            "tool_rotation_error_rad": rotation_error(actual[36:43], reference[36:43]),
            "target_rotation_error_rad": rotation_error(actual[43:50], reference[43:50]),
            "actual_target_motion_from_s0_mm": 1000 * float(np.linalg.norm(actual[43:46] - endpoints["B_qpos"][0, 43:46])),
            "reference_target_motion_from_0_mm": 1000 * float(np.linalg.norm(reference[43:46] - robot["qpos"][0, 43:46])),
        }

    right_contact_rows = [r for r in contact_paths if r["hand"] == "right"]
    left_contact_rows = [r for r in contact_paths if r["hand"] == "left"]
    summary = {
        "schema": "taco_pour_early_contact_origin_v1",
        "status": "STATIC_AUDIT_COMPLETE_VISUAL_REVIEW_PENDING",
        "right": {
            "task_phase": "approach/contact establishment, not demonstrated stable pinch",
            "human_gt_visual_surface_distance_mm": {
                "endpoint14_thumb": 1000 * geom_row(14, "RR", "right", "thumb")["human_gt_visual_triangle_distance_m"],
                "endpoint14_index": 1000 * geom_row(14, "RR", "right", "index")["human_gt_visual_triangle_distance_m"],
                "endpoint16_thumb": 1000 * geom_row(16, "RR", "right", "thumb")["human_gt_visual_triangle_distance_m"],
                "endpoint16_index": 1000 * geom_row(16, "RR", "right", "index")["human_gt_visual_triangle_distance_m"],
                "endpoint20_thumb": 1000 * geom_row(20, "RR", "right", "thumb")["human_gt_visual_triangle_distance_m"],
                "endpoint20_index": 1000 * geom_row(20, "RR", "right", "index")["human_gt_visual_triangle_distance_m"],
                "provenance": "fresh triangle-mesh query of human GT landmarks against reference object pose",
            },
            "H_to_R_tip_error_mm": {
                "endpoint14_thumb": 1000 * geom_row(14, "AA", "right", "thumb")["H_to_R_norm_m"],
                "endpoint14_index": 1000 * geom_row(14, "AA", "right", "index")["H_to_R_norm_m"],
                "endpoint16_thumb": 1000 * geom_row(16, "AA", "right", "thumb")["H_to_R_norm_m"],
                "endpoint16_index": 1000 * geom_row(16, "AA", "right", "index")["H_to_R_norm_m"],
            },
            "R_to_A_tip_error_mm": {
                "endpoint14_thumb": 1000 * geom_row(14, "AA", "right", "thumb")["R_to_A_norm_m"],
                "endpoint14_index": 1000 * geom_row(14, "AA", "right", "index")["R_to_A_norm_m"],
                "endpoint16_thumb": 1000 * geom_row(16, "AA", "right", "thumb")["R_to_A_norm_m"],
                "endpoint16_index": 1000 * geom_row(16, "AA", "right", "index")["R_to_A_norm_m"],
            },
            "contact_path": right_contact_rows,
        },
        "left": {
            "task_phase": "tray remains floor-supported while ring/pinky-side hand approaches/contacts",
            "human_gt_visual_surface_distance_mm": {
                "endpoint0_thumb": 1000 * geom_row(0, "RR", "left", "thumb")["human_gt_visual_triangle_distance_m"],
                "endpoint0_ring": 1000 * geom_row(0, "RR", "left", "ring")["human_gt_visual_triangle_distance_m"],
                "endpoint0_pinky": 1000 * geom_row(0, "RR", "left", "pinky")["human_gt_visual_triangle_distance_m"],
                "provenance": "fresh triangle-mesh query of human GT landmarks against reference object pose",
            },
            "H_to_R_tip_error_mm": {
                "endpoint0_thumb": 1000 * geom_row(0, "AA", "left", "thumb")["H_to_R_norm_m"],
                "endpoint0_ring": 1000 * geom_row(0, "AA", "left", "ring")["H_to_R_norm_m"],
                "endpoint0_pinky": 1000 * geom_row(0, "AA", "left", "pinky")["H_to_R_norm_m"],
            },
            "R_to_A_tip_error_mm": {
                "endpoint0_thumb": 1000 * geom_row(0, "AA", "left", "thumb")["R_to_A_norm_m"],
                "endpoint0_ring": 1000 * geom_row(0, "AA", "left", "ring")["R_to_A_norm_m"],
                "endpoint0_pinky": 1000 * geom_row(0, "AA", "left", "pinky")["R_to_A_norm_m"],
                "endpoint10_ring": 1000 * geom_row(10, "AA", "left", "ring")["R_to_A_norm_m"],
            },
            "contact_path": left_contact_rows,
        },
        "initialization_and_control": {
            **control_summary,
            "hand_qpos_provenance": str(initial["hand_qpos_provenance"]),
            "first_command_reference_index": int(initial["first_command_reference_index"]),
            "first_command_semantics": str(initial["first_command_semantics"]),
        },
        "model_compatibility": compatibility,
        "geometry_checks": geometry_checks,
        "contact_checks": contact_checks,
        "contact_episodes_from_saved_solver_rows": {
            "right_index_tool": _contact_episode_summary(contacts, "right_hand:index", "tool"),
            "right_thumb_tool": _contact_episode_summary(contacts, "right_hand:thumb", "tool"),
            "right_thumb_floor": _contact_episode_summary(contacts, "right_hand:thumb", "floor"),
            "left_ring_target": _contact_episode_summary(contacts, "left_hand:ring", "target"),
            "left_pinky_target": _contact_episode_summary(contacts, "left_hand:pinky", "target"),
            "target_floor": _contact_episode_summary(contacts, "target", "floor"),
            "force_note": "normal-force sums combine saved contact points within one solver substep; they are not summed across time",
        },
        "object_tracking": {
            f"endpoint{endpoint}": object_error(endpoint)
            for endpoint in (0, 1, 5, 10, 14, 15, 16, 20)
        },
        "execution_counts": {
            "physics_integrator_steps": 0,
            "simulation_control_intervals": 0,
            "new_actions": 0,
            "network_forwards": 0,
            "optimizer_updates": 0,
            "ik_solves": 0,
            "mj_forward_calls": 0,
            "mj_kinematics_calls_reference_model": reference_scene.kinematics_calls,
            "mj_kinematics_calls_runtime_model": runtime_scene.kinematics_calls,
            "mj_geomDistance_queries": runtime_scene.distance_queries,
            **render_counts,
        },
        "input_immutability": {
            name: sha256(Path(item["path"])) == item["sha256"]
            for name, item in inputs["artifacts"].items()
        },
        "unchanged_status": {
            "historical_tracking_criterion": "old tool-only criterion below boundary through endpoint20",
            "replay_identity": "established by prior trace; not rerun",
            "task_relation_review": "concern confirmed",
            "downstream_viability": "not established",
        },
    }
    write_json(build / "summary.json", summary)
    (build / "findings.md").write_text(
        "# Early contact origin v1\n\n"
        "Status: `STATIC_AUDIT_COMPLETE_VISUAL_REVIEW_PENDING`.\n\n"
        "No physics, actions, networks, IK, or optimization were executed. Quantitative tables and static RR/AR/RA/AA views are complete; direct visual observations must be recorded before changing this status.\n"
    )
    write_json(build / "visual_review.json", {"status": "pending", "required": ["human_reference_actual_early.png", "right_thumb_index_12_20.mp4", "left_support_0_15.mp4", "pose_factorization_014..016.png"]})
    _hash_output(build)
    os.rename(build, output)
    return summary
