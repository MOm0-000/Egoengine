"""Pure helpers for the zero-physics retarget-contract closure audit."""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


PALM_THUMB_PAIR = ("left_hand_link_visual", "left_thumb_rota1_visual")


def prior_native_stricter_breakdown(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Recompute A1 from the frozen CSV rows, never from its prose summary."""
    selected = [row for row in rows if row["classification"] == "NATIVE_STRICTER_THAN_PROXY"]
    result = {
        "native_stricter_row_count": len(selected),
        "rows_containing_palm_thumb_pair": 0,
        "left_hand_object_finding_count": 0,
        "left_hand_table_row_count": 0,
        "left_hand_table_finding_count": 0,
        "other_omitted_self_pair_count": 0,
        "unclassified_omitted_self_pair_count": 0,
        "states": [],
    }
    for row in selected:
        findings = json.loads(row["candidate_relevant_findings"])
        pairs = [tuple(value) for value in findings["left_omitted_nonadjacent"]]
        has_pair = PALM_THUMB_PAIR in pairs or PALM_THUMB_PAIR[::-1] in pairs
        result["rows_containing_palm_thumb_pair"] += int(has_pair)
        result["left_hand_object_finding_count"] += len(findings["left_hand_object"])
        result["left_hand_table_row_count"] += int(bool(findings["left_hand_table"]))
        result["left_hand_table_finding_count"] += len(findings["left_hand_table"])
        result["other_omitted_self_pair_count"] += sum(
            pair not in (PALM_THUMB_PAIR, PALM_THUMB_PAIR[::-1]) for pair in pairs
        )
        result["unclassified_omitted_self_pair_count"] += len(
            findings["left_unclassified_omitted_nonadjacent"]
        )
        result["states"].append({
            "state": row["state"],
            "endpoint": int(row["endpoint"]),
            "contains_palm_thumb_pair": has_pair,
            "left_hand_object": findings["left_hand_object"],
            "left_hand_table": findings["left_hand_table"],
            "other_omitted_self": [list(pair) for pair in pairs
                                   if pair not in (PALM_THUMB_PAIR, PALM_THUMB_PAIR[::-1])],
        })
    return result


def project_world_points(points: np.ndarray, intrinsic: np.ndarray,
                         camera_from_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with the released T_camera_world convention."""
    points = np.asarray(points, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    transform = np.asarray(camera_from_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("expected (N,3) world points")
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    if np.any(camera[:, 2] <= 0):
        raise ValueError("projection contains a point behind the released camera")
    homogeneous = camera @ intrinsic.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, camera[:, 2]


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))


def _rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(Rotation.from_matrix(np.asarray(first).T @ np.asarray(second)).magnitude())


def _local(point: np.ndarray, object_position: np.ndarray, object_rotation: np.ndarray) -> np.ndarray:
    return np.asarray(object_rotation).T @ (np.asarray(point) - np.asarray(object_position))


def objective_semantic_unit_tests(tolerance: float = 1e-10) -> list[dict[str, Any]]:
    """Analytic duty tests; holistic visual preference is deliberately absent."""
    quarter = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    common = Rotation.from_euler("xyz", [23, -17, 31], degrees=True).as_matrix()
    shift = np.array([0.7, -0.2, 1.1])
    point = np.array([0.2, -0.3, 0.5])
    target = np.array([-0.1, 0.4, 0.2])
    object_position = np.array([0.05, -0.15, 0.1])
    object_rotation = Rotation.from_euler("xyz", [7, 11, -9], degrees=True).as_matrix()

    position = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    local_position = lambda p, op, ro, q, oq, rq: position(
        _local(p, op, ro), _local(q, oq, rq)
    )
    transform_point = lambda p: common @ np.asarray(p) + shift
    transformed_object_position = transform_point(object_position)
    transformed_object_rotation = common @ object_rotation
    direction_a = np.array([1.0, 0.0, 0.0])
    direction_b = np.array([0.0, 1.0, 0.0])
    known_angle = np.pi / 2

    rows = []

    def add(component: str, duty: str, checks: dict[str, bool], evidence: dict[str, Any]) -> None:
        rows.append({
            "component": component,
            "duty": duty,
            "checks": checks,
            "evidence": evidence,
            "classification": "SEMANTICALLY_CERTIFIED" if all(checks.values())
                              else "SEMANTICALLY_FAILED",
        })

    p0 = position(point, point)
    p1 = position(point, target)
    p2 = position(point, point + 2 * (target - point))
    add("fingertip_position_error_m", "fingertip Cartesian position only", {
        "identity_zero": abs(p0) <= tolerance,
        "known_displacement": abs(p1 - np.linalg.norm(point - target)) <= tolerance,
        "monotonic_magnitude": p2 > p1 > p0,
        "common_rigid_transform_invariant": abs(
            position(transform_point(point), transform_point(target)) - p1
        ) <= tolerance,
    }, {"zero": p0, "unit_case": p1, "larger_case": p2})
    vector_checks = {
        "identity_zero": abs(p0) <= tolerance,
        "known_vector_difference": abs(p1 - np.linalg.norm(point - target)) <= tolerance,
        "common_rigid_transform_invariant": abs(
            position(common @ point, common @ target) - p1
        ) <= tolerance,
    }
    add("wrist_to_tip_vector_error_m", "finger vector relative to its wrist", dict(vector_checks),
        {"zero": p0, "known_difference": p1})
    add("thumb_to_tip_vector_error_m", "finger vector relative to thumb tip", dict(vector_checks),
        {"zero": p0, "known_difference": p1})

    r0 = _rotation_error(np.eye(3), np.eye(3))
    r1 = _rotation_error(np.eye(3), quarter)
    add("fingertip_orientation_error_rad", "fingertip/distal orientation only", {
        "identity_zero": abs(r0) <= tolerance,
        "known_quarter_turn": abs(r1 - known_angle) <= tolerance,
        "common_rotation_invariant": abs(_rotation_error(common, common @ quarter) - r1) <= tolerance,
    }, {"zero": r0, "quarter_turn_rad": r1})

    d0 = _angle(direction_a, direction_a)
    d1 = _angle(direction_a, direction_b)
    direction_checks = {
        "identity_zero": abs(d0) <= tolerance,
        "known_quarter_turn": abs(d1 - known_angle) <= tolerance,
        "common_rotation_invariant": abs(_angle(common @ direction_a, common @ direction_b) - d1) <= tolerance,
    }
    add("direct_proximal_orientation_error_rad", "proximal finger shape only",
        dict(direction_checks), {"zero": d0, "quarter_turn_rad": d1})
    add("direct_distal_orientation_error_rad", "distal finger shape only",
        dict(direction_checks), {"zero": d0, "quarter_turn_rad": d1})

    lp0 = local_position(point, object_position, object_rotation,
                         point, object_position, object_rotation)
    lp1 = local_position(point, object_position, object_rotation,
                         target, object_position, object_rotation)
    lp_common = local_position(
        transform_point(point), transformed_object_position, transformed_object_rotation,
        transform_point(target), transformed_object_position, transformed_object_rotation,
    )
    add("near_interaction_position_error_m", "finger segment position relative to tray", {
        "identity_zero": abs(lp0) <= tolerance,
        "relative_offset_sensitive": lp1 > tolerance,
        "common_rigid_transform_invariant": abs(lp_common - lp1) <= tolerance,
    }, {"zero": lp0, "relative_offset": lp1, "transformed": lp_common})

    ni0 = _angle(object_rotation.T @ direction_a, object_rotation.T @ direction_a)
    ni1 = _angle(object_rotation.T @ direction_a, object_rotation.T @ direction_b)
    ni_common = _angle(
        transformed_object_rotation.T @ (common @ direction_a),
        transformed_object_rotation.T @ (common @ direction_b),
    )
    add("near_interaction_orientation_error_rad", "finger segment orientation relative to tray", {
        "identity_zero": abs(ni0) <= tolerance,
        "known_relative_rotation": abs(ni1 - known_angle) <= tolerance,
        "common_rigid_transform_invariant": abs(ni_common - ni1) <= tolerance,
    }, {"zero": ni0, "quarter_turn_rad": ni1, "transformed": ni_common})

    wrist_local = _local(point, object_position, object_rotation)
    wrist_target_local = _local(target, object_position, object_rotation)
    wrist_error = position(wrist_local, wrist_target_local)
    transformed_wrist_error = position(
        _local(transform_point(point), transformed_object_position, transformed_object_rotation),
        _local(transform_point(target), transformed_object_position, transformed_object_rotation),
    )
    add("wrist_tray_position_error_m", "wrist position relative to tray", {
        "identity_zero": position(wrist_local, wrist_local) <= tolerance,
        "relative_offset_sensitive": wrist_error > tolerance,
        "common_rigid_transform_invariant": abs(wrist_error - transformed_wrist_error) <= tolerance,
    }, {"relative_offset": wrist_error, "transformed": transformed_wrist_error})

    add("wrist_orientation_error_rad", "wrist orientation only", {
        "identity_zero": abs(r0) <= tolerance,
        "known_quarter_turn": abs(r1 - known_angle) <= tolerance,
        "common_rotation_invariant": abs(_rotation_error(common, common @ quarter) - r1) <= tolerance,
    }, {"zero": r0, "quarter_turn_rad": r1})

    qref = np.array([0.2, -0.1, 0.4])
    nominal = [float(np.linalg.norm(qref + scale * np.array([0.1, -0.2, 0.3]) - qref))
               for scale in (0, 1, 2)]
    add("joint_nominal_deviation_rad_l2", "limit deviation from OLD_MINK nominal", {
        "nominal_zero": nominal[0] <= tolerance,
        "symmetric": abs(np.linalg.norm((qref - np.array([0.1, -0.2, 0.3])) - qref) - nominal[1]) <= tolerance,
        "monotonic_magnitude": nominal[2] > nominal[1] > nominal[0],
    }, {"magnitudes": nominal})

    temporal = [float(np.linalg.norm(scale * np.array([0.1, -0.2, 0.3]))) for scale in (0, 1, 2)]
    add("temporal_joint_change_rad_l2", "frame-to-frame continuity only", {
        "unchanged_zero": temporal[0] <= tolerance,
        "symmetric": abs(np.linalg.norm(-np.array([0.1, -0.2, 0.3])) - temporal[1]) <= tolerance,
        "monotonic_magnitude": temporal[2] > temporal[1] > temporal[0],
    }, {"magnitudes": temporal})
    return rows


def final_classification(*, collision_closed: bool, objective_closed: bool,
                         source_rgb_closed: bool) -> tuple[str, list[str]]:
    blockers = []
    if not collision_closed:
        blockers.append("COLLISION_SEMANTICS_BLOCKER_REMAINS")
    if not objective_closed:
        blockers.append("OBJECTIVE_SEMANTICS_BLOCKER_REMAINS")
    if not source_rgb_closed:
        blockers.append("SOURCE_RGB_ALIGNMENT_BLOCKER")
    if not blockers:
        return "RETARGET_CONTRACT_V2_CERTIFIED_NO_CANDIDATE", blockers
    if len(blockers) == 1:
        return blockers[0], blockers
    return "MULTIPLE_CONTRACT_BLOCKERS_REMAIN", blockers
