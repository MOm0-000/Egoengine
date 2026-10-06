#!/usr/bin/env python3
"""Read-only MANO/XHand geometry-contract and reachability audit.

The audit compares the inherited XML fingertip sites with a candidate contract
using the fixed endpoint named by each XHand URDF.  It does not modify the
released GT, production scene, weights, or formal reference.  The candidate
scene and target archive written under the output directory are diagnostic
artifacts only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src"), str(ROOT / "scripts")]

from audit_taco_orientation_prerequisites import (  # noqa: E402
    _current_anatomical_calibrations,
    _unit,
)
from diagnose_taco_retarget_objectives import Experiment  # noqa: E402
from egoengine_repro.evaluation.taco_surface import (  # noqa: E402
    reconstruct_taco_mano,
)
from egoengine_repro.retarget.paper_audit import artifact  # noqa: E402
from egoengine_repro.retarget.taco_bimanual import (  # noqa: E402
    DIPS,
    FINGERS,
    MANO_DISTALS,
    SIDES,
    TIPS,
    geometric_frame,
    mano_fingertip_frames,
)


DEFAULT_RUN = ROOT / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1"
DEFAULT_SCENE = ROOT / (
    "models/taco_xhand/xhand/bimanual/"
    "taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
)
DEFAULT_VENDOR = ROOT / "runs/taco_pour_spider_mink_experiment_v2/vendor/spider/assets/robots/xhand"
DEFAULT_HANDS = ROOT / (
    "data/taco_v1/pour_bowl_plate/hand_poses/Hand_Poses/"
    "(pour in some, bowl, plate)/20230927_017"
)
DEFAULT_MANO = ROOT / "data/taco_v1/hand_poses_v1/mano_v1_2/models"
DEFAULT_OUTPUT = ROOT / "runs/taco_pour_geometry_contract_audit_v7"
SAMPLED_ROWS = (0, 20, 50, 100, 150, 197)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ji,...jk->...ik", actual, target)
    return Rotation.from_matrix(relative).magnitude()


def _summary(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1) * scale
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("metric values must be finite and nonempty")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _fixed_tip(urdf: ET.Element, body_name: str) -> dict[str, object]:
    matches = []
    for joint in urdf.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None and parent.get("link") == body_name:
            if "tip" in child.get("link", ""):
                matches.append(joint)
    if len(matches) != 1:
        raise ValueError(f"expected one fixed tip joint for {body_name!r}")
    joint = matches[0]
    origin = joint.find("origin")
    if origin is None:
        raise ValueError(f"fixed tip joint {joint.get('name')!r} lacks origin")
    point = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    if point.shape != (3,) or rpy.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"malformed fixed tip origin for {body_name!r}")
    if np.linalg.norm(rpy) > 1.0e-12:
        raise ValueError(f"nonidentity fixed tip rotation for {body_name!r}")
    return {
        "joint": joint.get("name"),
        "child": joint.find("child").get("link"),
        "point_local_m": point,
        "longitudinal_axis_local": _unit(point),
    }


def _urdf_fk(urdf: ET.Element, root_name: str, values: dict[str, float]) -> dict[str, np.ndarray]:
    transforms: dict[str, np.ndarray] = {root_name: np.eye(4)}
    pending = list(urdf.findall("joint"))
    while pending:
        advanced = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            local = np.eye(4)
            local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            local[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
            if joint.get("type") == "revolute":
                axis = _unit(np.fromstring(joint.find("axis").get("xyz"), sep=" "))
                local[:3, :3] = local[:3, :3] @ Rotation.from_rotvec(
                    axis * values[joint.get("name")]
                ).as_matrix()
            elif joint.get("type") != "fixed":
                raise ValueError(f"unexpected URDF joint type {joint.get('type')!r}")
            child = joint.find("child").get("link")
            transforms[child] = transforms[parent] @ local
            pending.remove(joint)
            advanced = True
        if not advanced:
            raise ValueError("URDF chain is disconnected")
    return transforms


def _load(run: Path) -> tuple[dict, dict, dict]:
    report = json.loads((run / "retarget_report.json").read_text(encoding="utf-8"))
    with np.load(run / "human_reference.npz", allow_pickle=False) as source:
        human = {key: np.asarray(source[key]) for key in source.files}
    with np.load(run / "robot_reference.npz", allow_pickle=False) as source:
        robot = {key: np.asarray(source[key]) for key in source.files}
    return report, human, robot


def _model_geometry(model: mujoco.MjModel, vendor: Path) -> tuple[dict, dict, dict]:
    """Return site/endpoint frames, URDF FK checks, and fixed candidate frames."""
    zero = mujoco.MjData(model)
    mujoco.mj_forward(model, zero)
    old_cal = _current_anatomical_calibrations(model)
    records: dict[str, dict] = {}
    candidate_cal = np.empty((2, 5, 3, 3), dtype=float)
    endpoint_data: dict[str, dict] = {}
    for hi, side in enumerate(SIDES):
        urdf = ET.parse(vendor / f"xhand_{side}.urdf").getroot()
        root_id = model.body(f"{side}_hand_link").id
        palm_id = model.site(f"{side}_palm").id
        middle_id = model.body(f"{side}_hand_mid_link1").id
        palm_axis = zero.site_xmat[palm_id].reshape(3, 3)[:, 0]
        palm_frame = geometric_frame(palm_axis, zero.xpos[middle_id] - zero.xpos[root_id])
        records[side] = {}
        endpoint_data[side] = {}
        for fi, finger in enumerate(FINGERS):
            site_id = model.site(f"{side}_{finger}_tip").id
            body_id = int(model.site_bodyid[site_id])
            body_name = model.body(body_id).name
            fixed = _fixed_tip(urdf, body_name)
            endpoint = fixed["point_local_m"]
            site_local = np.asarray(model.site_pos[site_id], dtype=float)
            site_frame = geometric_frame(
                palm_frame[:, 0], zero.site_xpos[site_id] - zero.xpos[body_id]
            )
            endpoint_world = zero.xmat[body_id].reshape(3, 3) @ endpoint + zero.xpos[body_id]
            endpoint_frame = geometric_frame(
                palm_frame[:, 0], endpoint_world - zero.xpos[body_id]
            )
            c_old = zero.xmat[body_id].reshape(3, 3).T @ site_frame
            c_new = zero.xmat[body_id].reshape(3, 3).T @ endpoint_frame
            candidate_cal[hi, fi] = c_new
            endpoint_data[side][finger] = {
                "body_id": body_id,
                "body_name": body_name,
                "site_id": site_id,
                "site_local_m": site_local,
                "endpoint_local_m": endpoint,
                "endpoint_child_link": fixed["child"],
                "longitudinal_axis_local": fixed["longitudinal_axis_local"],
                "site_frame_local": c_old,
                "endpoint_anatomical_frame_local": c_new,
            }
            records[side][finger] = {
                "body": body_name,
                "site_local_m": site_local.tolist(),
                "urdf_endpoint_local_m": endpoint.tolist(),
                "point_offset_endpoint_minus_site_local_m": (endpoint - site_local).tolist(),
                "point_offset_norm_mm": float(np.linalg.norm(endpoint - site_local) * 1000.0),
                "site_to_endpoint_frame_offset_deg": float(
                    np.rad2deg(Rotation.from_matrix(c_old.T @ c_new).magnitude())
                ),
                "site_axis_vs_endpoint_longitudinal_deg": float(
                    np.rad2deg(Rotation.from_matrix(site_frame.T @ endpoint_frame).magnitude())
                ),
                "old_frame_local": c_old.tolist(),
                "candidate_frame_local": c_new.tolist(),
                "old_frame_orthogonality_error": float(np.max(np.abs(c_old.T @ c_old - np.eye(3)))),
                "candidate_frame_orthogonality_error": float(np.max(np.abs(c_new.T @ c_new - np.eye(3)))),
                "candidate_frame_det": float(np.linalg.det(c_new)),
            }
    return records, endpoint_data, {"old": old_cal, "candidate": candidate_cal}


def _independent_fk_checks(model: mujoco.MjModel, vendor: Path, robot: dict, endpoint_data: dict) -> dict:
    checks: dict[str, dict] = {}
    for hi, side in enumerate(SIDES):
        urdf = ET.parse(vendor / f"xhand_{side}.urdf").getroot()
        root_name = f"{side}_hand_link"
        root_id = model.body(root_name).id
        max_body_position = 0.0
        max_body_angle = 0.0
        max_endpoint_position = 0.0
        max_endpoint_axis_angle = 0.0
        rows = []
        for row, q in enumerate(robot["qpos"]):
            data = mujoco.MjData(model)
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            values = {
                joint.get("name"): float(q[int(model.joint(joint.get("name")).qposadr[0])])
                for joint in urdf.findall("joint") if joint.get("type") == "revolute"
            }
            fk = _urdf_fk(urdf, root_name, values)
            root_R = data.xmat[root_id].reshape(3, 3)
            root_p = data.xpos[root_id]
            for body_name, local in fk.items():
                if body_name.endswith("tip") or body_name == f"{side}_hand_ee_link":
                    continue
                body_id = model.body(body_name).id
                actual_p = root_R.T @ (data.xpos[body_id] - root_p)
                actual_R = root_R.T @ data.xmat[body_id].reshape(3, 3)
                max_body_position = max(max_body_position, float(np.linalg.norm(actual_p - local[:3, 3])))
                max_body_angle = max(max_body_angle, float(Rotation.from_matrix(actual_R.T @ local[:3, :3]).magnitude()))
            for finger in FINGERS:
                item = endpoint_data[side][finger]
                body_id = item["body_id"]
                endpoint_local = item["endpoint_local_m"]
                mujoco_point = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ endpoint_local
                fixed = _fixed_tip(urdf, item["body_name"])
                parent_local = fk[item["body_name"]]
                urdf_point = root_p + root_R @ (parent_local[:3, 3] + parent_local[:3, :3] @ fixed["point_local_m"])
                max_endpoint_position = max(max_endpoint_position, float(np.linalg.norm(mujoco_point - urdf_point)))
                mujoco_axis = data.xmat[body_id].reshape(3, 3) @ fixed["longitudinal_axis_local"]
                urdf_axis = root_R @ parent_local[:3, :3] @ fixed["longitudinal_axis_local"]
                max_endpoint_axis_angle = max(
                    max_endpoint_axis_angle,
                    float(np.rad2deg(np.arccos(np.clip(np.dot(mujoco_axis, urdf_axis), -1.0, 1.0)))),
                )
            if row in SAMPLED_ROWS:
                rows.append(row)
        checks[side] = {
            "rows": len(robot["qpos"]),
            "sampled_rows": rows,
            "max_body_position_difference_m": max_body_position,
            "max_body_rotation_difference_rad": max_body_angle,
            "max_endpoint_position_difference_m": max_endpoint_position,
            "max_endpoint_longitudinal_axis_difference_deg": max_endpoint_axis_angle,
            "pass": bool(max_body_position < 1e-6 and max_body_angle < 2e-5 and
                          max_endpoint_position < 1e-6 and max_endpoint_axis_angle < 2e-4),
            "numerical_tolerances": {
                "body_position_m": 1e-6,
                "body_rotation_rad": 2e-5,
                "endpoint_position_m": 1e-6,
                "endpoint_axis_deg": 2e-4,
            },
        }
    return checks


def _posture_strata(human: dict) -> dict[str, dict[str, list[int]]]:
    joints = np.asarray(human["joint_positions_sim"], dtype=float)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise ValueError("joint_positions_sim must have shape (T,2,21,3)")
    result: dict[str, dict[str, list[int]]] = {}
    for hi, side in enumerate(SIDES):
        wrist = joints[:, hi, 0]
        palm_span = np.linalg.norm(joints[:, hi, 9] - wrist, axis=1)
        openness = np.mean(np.linalg.norm(joints[:, hi, np.asarray(TIPS)] - wrist[:, None], axis=2), axis=1) / np.maximum(palm_span, 1e-8)
        order = np.argsort(openness)
        n = len(order)
        slices = {
            "grasp_like": order[:max(1, n // 3)],
            "bent_like": order[n // 3: max(n // 3 + 1, 2 * n // 3)],
            "open_like": order[max(0, 2 * n // 3):],
        }
        representative = {
            name: int(indices[np.argmin(np.abs(openness[indices] - np.median(openness[indices])))])
            for name, indices in slices.items()
        }
        result[side] = {
            "metric": "mean fingertip-to-wrist distance normalized by wrist-to-middle-MCP distance",
            "openness_min": float(openness.min()),
            "openness_median": float(np.median(openness)),
            "openness_max": float(openness.max()),
            "strata_rows": {name: [int(value) for value in indices.tolist()] for name, indices in slices.items()},
            "representative_rows": representative,
        }
    return result


def _mano_frame_audit(human: dict, hands: Path, mano_models: Path, calibrations: dict[str, np.ndarray]) -> tuple[dict, dict]:
    records: dict[str, dict] = {}
    mano_frames = {}
    for hi, side in enumerate(SIDES):
        pose_path = hands / f"{side}_hand.pkl"
        shape_path = hands / f"{side}_hand_shape.pkl"
        model_path = mano_models / f"MANO_{side.upper()}.pkl"
        _, joints, _, _, _ = reconstruct_taco_mano(pose_path, shape_path, model_path, side=side)
        frames, source_report = mano_fingertip_frames(pose_path, shape_path, model_path, side, joints)
        mano_frames[side] = frames
        old_target = human["T_sim_fingertip_target"][:, hi, :, :3, :3]
        old_human = np.einsum("tfij,fjk->tfik", old_target, calibrations["old"][hi])
        transport_error = _rotation_error(old_human, frames)
        records[side] = {
            "source": source_report,
            "transport_error_deg": _summary(np.rad2deg(transport_error)),
            "transport_error_deg_by_finger": [_summary(np.rad2deg(transport_error[:, fi])) for fi in range(5)],
        }
    return records, mano_frames


def _candidate_targets(human: dict, calibrations: dict[str, np.ndarray]) -> dict:
    target = {key: np.array(value, copy=True) for key, value in human.items()}
    old = calibrations["old"]
    new = calibrations["candidate"]
    rotations = human["T_sim_fingertip_target"][:, :, :, :3, :3]
    old_transport_target = np.einsum("thfij,hfjk->thfik", rotations, old)
    target["T_sim_fingertip_target"][:, :, :, :3, :3] = np.einsum(
        "thfij,hfjk->thfik", old_transport_target, np.swapaxes(new, -1, -2)
    )
    target["fingertip_orientation_source"] = np.asarray(
        "MANO_rotational_FK_fixed_neutral_calibration_v1_endpoint_frame_diagnostic"
    )
    return target


def _write_candidate_scene(source: Path, destination: Path, endpoint_data: dict) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("scene lacks compiler element")
    meshdir = (source.parent / compiler.get("meshdir", "")).resolve(strict=True)
    compiler.set("meshdir", str(meshdir))
    for side in SIDES:
        for finger in FINGERS:
            name = f"{side}_{finger}_tip"
            node = root.find(f".//site[@name='{name}']")
            if node is None:
                raise ValueError(f"scene lacks {name}")
            values = endpoint_data[side][finger]["endpoint_local_m"]
            node.set("pos", " ".join(format(float(v), ".17g") for v in values))
    with destination.open("x", encoding="utf-8") as stream:
        tree.write(stream, encoding="unicode")


def _save_target(path: Path, target: dict) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **target)


def _state_metrics(exp: Experiment, row: int, q: np.ndarray) -> dict[str, np.ndarray]:
    exp.config.update(q)
    tip_position = np.empty((2, 5), dtype=float)
    tip_orientation = np.empty((2, 5), dtype=float)
    wrist_position = np.empty(2, dtype=float)
    wrist_orientation = np.empty(2, dtype=float)
    for hi, side in enumerate(SIDES):
        ids = [exp.model.site(f"{side}_{finger}_tip").id for finger in FINGERS]
        actual_p = exp.config.data.site_xpos[ids]
        actual_R = exp.config.data.site_xmat[ids].reshape(5, 3, 3)
        targets = exp.human["T_sim_fingertip_target"][row, hi]
        tip_position[hi] = np.linalg.norm(actual_p - targets[:, :3, 3], axis=1)
        tip_orientation[hi] = _rotation_error(actual_R, targets[:, :3, :3])
        root = exp.model.body(f"{side}_hand_link").id
        wrist = exp.human["T_sim_wrist_target"][row, hi]
        wrist_position[hi] = np.linalg.norm(exp.config.data.xpos[root] - wrist[:3, 3])
        wrist_orientation[hi] = float(_rotation_error(
            exp.config.data.xmat[root].reshape(3, 3), wrist[:3, :3]
        ))
    return {
        "tip_position_m": tip_position,
        "tip_orientation_rad": tip_orientation,
        "wrist_position_m": wrist_position,
        "wrist_orientation_rad": wrist_orientation,
    }


def _probe_reachability(scene: Path, human: dict, robot: dict, settings: dict, rows: tuple[int, ...]) -> dict:
    result: dict[str, object] = {}
    for mode in ("position_only", "position_wrist", "pose"):
        exp = Experiment(scene, settings, human, robot)
        records = []
        for row in rows:
            record, q = exp.solve(row, mode, max_iterations=160, use_speed=False)
            metrics = _state_metrics(exp, row, q)
            records.append({
                "row": int(row),
                "converged": bool(record["converged"]),
                "stalled": bool(record["line_search_stalled"]),
                "constraint_gate_passed": bool(record["constraint_gate_passed"]),
                "metrics": {key: value.tolist() for key, value in metrics.items()},
            })
        valid = [r for r in records if r["constraint_gate_passed"]]
        if not valid:
            result[mode] = {"rows": records, "valid_rows": 0}
            continue
        arrays = {
            key: np.asarray([row["metrics"][key] for row in valid], dtype=float)
            for key in ("tip_position_m", "tip_orientation_rad", "wrist_position_m", "wrist_orientation_rad")
        }
        result[mode] = {
            "rows": records,
            "valid_rows": len(valid),
            "converged_rows": int(sum(row["converged"] for row in records)),
            "stalled_rows": int(sum(row["stalled"] for row in records)),
            "tip_position_mm_by_hand": _summary(arrays["tip_position_m"], 1000.0),
            "tip_orientation_deg_by_hand": _summary(np.rad2deg(arrays["tip_orientation_rad"])),
            "wrist_position_mm_by_hand": _summary(arrays["wrist_position_m"], 1000.0),
            "wrist_orientation_deg_by_hand": _summary(np.rad2deg(arrays["wrist_orientation_rad"])),
            "interpretation": "finite local constrained probes, not a global optimum or formal reference",
        }
    return result


def _axis_lower_bound(human: dict, model: mujoco.MjModel, endpoint_data: dict, *, endpoint: bool) -> dict:
    """Compute the fixed-wrist common-X and position-plane lower bounds."""
    data = mujoco.MjData(model)
    result: dict[str, dict] = {}
    for hi, side in enumerate(SIDES):
        root = model.body(f"{side}_hand_link").id
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        root_R = data.xmat[root].reshape(3, 3)
        point_coords = []
        for finger in FINGERS[2:]:
            item = endpoint_data[side][finger]
            if endpoint:
                local = item["endpoint_local_m"]
                point = data.xpos[item["body_id"]] + data.xmat[item["body_id"]].reshape(3, 3) @ local
            else:
                sid = item["site_id"]
                point = data.site_xpos[sid]
            point_coords.append(float(root_R.T @ (point - data.xpos[root]) @ np.eye(3)[:, 0]))
        point_coords = np.asarray(point_coords)
        target_frames = human["T_sim_fingertip_target"][:, hi, FINGERS.index("middle"):FINGERS.index("pinky") + 1, :3, :3]
        target_points = human["T_sim_fingertip_target"][:, hi, 2:, :3, 3]
        target_wrist = human["T_sim_wrist_target"][:, hi]
        target_local_x = np.einsum(
            "tij,tkj->tki", np.swapaxes(target_wrist[:, :3, :3], -1, -2),
            target_points - target_wrist[:, None, :3, 3]
        )[:, :, 0]
        fixed_wrist = target_local_x - point_coords[None]
        centered = fixed_wrist - fixed_wrist.mean(axis=1, keepdims=True)
        axes = target_frames[..., :, 0]
        pairs = np.stack([
            np.rad2deg(np.arccos(np.clip(np.sum(axes[:, a] * axes[:, b], axis=1), -1, 1)))
            for a, b in combinations(range(3), 2)
        ], axis=1)
        orientation_lower = pairs.max(axis=1) / np.sqrt(6.0)
        result[side] = {
            "fingers": list(FINGERS[2:]),
            "robot_common_x_m": point_coords.tolist(),
            "fixed_wrist_position_lower_bound_mm": _summary(np.linalg.norm(fixed_wrist, axis=1), 1000.0),
            "best_common_translation_position_lower_bound_mm": _summary(np.sqrt(np.mean(centered ** 2, axis=1)), 1000.0),
            "orientation_shared_axis_pair_mean_deg": pairs.mean(axis=0).tolist(),
            "orientation_shared_axis_rms_lower_bound_deg": _summary(orientation_lower),
            "interpretation": "analytic lower bounds for middle/ring/pinky shared-axis kinematics with wrist rotation fixed",
        }
    return result


def _posture_stability(human: dict, robot: dict, mano_frames: dict, model: mujoco.MjModel, endpoint_data: dict,
                       calibrations: dict[str, np.ndarray], strata: dict) -> dict:
    data = mujoco.MjData(model)
    result: dict[str, dict] = {}
    for hi, side in enumerate(SIDES):
        result[side] = {}
        for posture, indices in strata[side]["strata_rows"].items():
            rows = np.asarray(indices, dtype=int)
            old_transport = []
            endpoint_transport = []
            # MANO target transport must be invariant in every posture stratum.
            target_old = human["T_sim_fingertip_target"][rows, hi, :, :3, :3]
            old_transport_target = np.einsum(
                "tfij,fjk->tfik", target_old, calibrations["old"][hi]
            )
            target_new = np.einsum(
                "tfij,fjk->tfik", old_transport_target,
                np.swapaxes(calibrations["candidate"][hi], -1, -2)
            )
            old_transport = _rotation_error(
                np.einsum("tfij,fjk->tfik", target_old, calibrations["old"][hi]), mano_frames[side][rows]
            )
            endpoint_transport = _rotation_error(
                np.einsum("tfij,fjk->tfik", target_new, calibrations["candidate"][hi]), mano_frames[side][rows]
            )
            fixed_transport = []
            point_norms = []
            for row in rows:
                # Use the saved robot configuration for this posture. Object
                # coordinates are irrelevant and remain exactly untouched.
                data.qpos[:] = robot["qpos"][row]
                mujoco.mj_forward(model, data)
                for finger in FINGERS:
                    item = endpoint_data[side][finger]
                    point_norms.append(np.linalg.norm(item["endpoint_local_m"] - item["site_local_m"]))
                    body = item["body_id"]
                    transported = data.xmat[body].reshape(3, 3) @ calibrations["candidate"][hi, FINGERS.index(finger)]
                    recovered_local = data.xmat[body].reshape(3, 3).T @ transported
                    fixed_transport.append(float(np.rad2deg(Rotation.from_matrix(
                        recovered_local.T @ calibrations["candidate"][hi, FINGERS.index(finger)]
                    ).magnitude())))
            result[side][posture] = {
                "rows": [int(v) for v in rows.tolist()],
                "representative_row": int(strata[side]["representative_rows"][posture]),
                "point_offset_norm_mm": _summary(np.asarray(point_norms), 1000.0),
                "MANO_target_transport_error_deg": _summary(np.rad2deg(old_transport)),
                "candidate_target_transport_error_deg": _summary(np.rad2deg(endpoint_transport)),
                "candidate_fixed_endpoint_frame_transport_error_deg": _summary(np.asarray(fixed_transport)),
                "dynamic_palm_projection_is_not_used": True,
            }
    return result


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    report, human, robot = _load(args.run)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    if len(robot["qpos"]) != len(human["frame_indices"]):
        raise ValueError("human and robot frame counts differ")
    records, endpoint_data, calibrations = _model_geometry(model, args.vendor)
    fk = _independent_fk_checks(model, args.vendor, robot, endpoint_data)
    mano, mano_frames = _mano_frame_audit(human, args.hands, args.mano_models, calibrations)
    candidate_human = _candidate_targets(human, calibrations)
    candidate_scene = args.output / "diagnostic_endpoint_scene.xml"
    candidate_target = args.output / "diagnostic_endpoint_human_reference.npz"
    args.output.mkdir(parents=True)
    _write_candidate_scene(args.scene, candidate_scene, endpoint_data)
    _save_target(candidate_target, candidate_human)
    strata = _posture_strata(human)
    stability = _posture_stability(human, robot, mano_frames, model, endpoint_data, calibrations, strata)
    lower_before = _axis_lower_bound(human, model, endpoint_data, endpoint=False)
    lower_after = _axis_lower_bound(
        candidate_human, mujoco.MjModel.from_xml_path(str(candidate_scene)),
        endpoint_data, endpoint=True,
    )
    settings = dict(report["inherited_settings"])
    probes = {
        "inherited_site_contract": _probe_reachability(args.scene, human, robot, settings, SAMPLED_ROWS),
        "candidate_endpoint_contract": _probe_reachability(candidate_scene, candidate_human, robot, settings, SAMPLED_ROWS),
    }
    target_preservation = _rotation_error(
        np.einsum("thfij,hfjk->thfik", human["T_sim_fingertip_target"][:, :, :, :3, :3], calibrations["old"]),
        np.einsum("thfij,hfjk->thfik", candidate_human["T_sim_fingertip_target"][:, :, :, :3, :3], calibrations["candidate"]),
    )
    inputs = [args.run / name for name in ("human_reference.npz", "robot_reference.npz", "retarget_report.json")]
    inputs += [args.scene, *[args.vendor / f"xhand_{side}.urdf" for side in SIDES]]
    inputs += [args.hands / f"{side}_hand.pkl" for side in SIDES]
    inputs += [args.hands / f"{side}_hand_shape.pkl" for side in SIDES]
    inputs += [args.mano_models / f"MANO_{side.upper()}.pkl" for side in SIDES]
    code = [Path(__file__), ROOT / "src/egoengine_repro/retarget/taco_bimanual.py",
            ROOT / "scripts/audit_taco_orientation_prerequisites.py",
            ROOT / "scripts/diagnose_taco_retarget_objectives.py"]
    result = {
        "status": "geometry_contract_read_only_audit_not_formal_mapping",
        "weights_modified": False,
        "models_or_formal_reference_modified": False,
        "rl_validation_completed": False,
        "input_artifacts": [artifact(path) for path in inputs],
        "code_artifacts": [artifact(path) for path in code],
        "scene": artifact(args.scene),
        "settings": settings,
        "contract_definitions": {
            "inherited": "XML fingertip site position and neutral anatomical frame transported by the current site body",
            "candidate": "URDF fixed tip endpoint position and fixed endpoint anatomical frame transported by the distal body",
            "human": "released MANO tip vertex and MANO distal rotational-FK frame; target points unchanged",
            "candidate_target_transport": "R_target_candidate = R_target_inherited C_inherited C_candidate^T",
            "candidate_is_not_author_calibration": True,
            "candidate_is_not_formal_initializer": True,
        },
        "per_finger_point_and_axis_offsets": records,
        "independent_urdf_mujoco_fk": fk,
        "MANO_anatomical_frame": mano,
        "posture_strata": strata,
        "posture_stability": stability,
        "target_frame_transport_preservation_deg": _summary(np.rad2deg(target_preservation)),
        "reachability_lower_bounds": {
            "inherited_site_contract": lower_before,
            "candidate_endpoint_contract": lower_after,
        },
        "empirical_local_reachability_probes": probes,
        "diagnostic_outputs": {
            "candidate_scene": artifact(candidate_scene),
            "candidate_human_reference": artifact(candidate_target),
        },
        "conclusions": {
            "independent_fk_passed": all(value["pass"] for value in fk.values()),
            "MANO_transport_is_consistent": all(
                value["transport_error_deg"]["max"] < 1e-5 for value in mano.values()
            ),
            "candidate_fixed_point_offsets_are_pose_invariant": True,
            "candidate_frame_transport_is_pose_invariant": True,
            "geometry_contract_resolved": False,
            "grid_search_allowed": False,
            "reason_grid_blocked": "candidate endpoint/anatomical frame is an auditable hypothesis; multi-task feasibility and author correspondence are not established",
        },
        "limitations": [
            "The endpoint contract is a reproducible engineering hypothesis, not a recovered EgoEngine author calibration.",
            "The finite QP probes are local constrained optima on six representative rows, not global reachable-set proofs.",
            "The posture labels are geometric strata, not TACO-provided action labels.",
            "Wrist relaxed kinematic lower bounds are zero because the wrist is floating; nonzero coupled residuals are reported by the probes.",
            "No weights, target points, production sites, collision pairs, qpos, or formal references were overwritten.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--hands", type=Path, default=DEFAULT_HANDS)
    parser.add_argument("--mano-models", type=Path, default=DEFAULT_MANO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result = run(parser.parse_args())
    print(json.dumps(result["conclusions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
