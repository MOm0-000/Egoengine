#!/usr/bin/env python3
"""Bounded, fail-closed CPU physics Replay for the Brush sample.

This diagnostic never runs MINK, MPC, or RL.  It replays the frozen MINK-v2
position targets from endpoint 0 through endpoint 80 in the unmodified source
MuJoCo dynamics, except for an experiment-local copy of the already audited
Issue #14 floor height.  Runtime proxy contacts/forces are aligned with sampled
native visual-mesh distances; the latter never participate in simulation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_world_objects,
    pair_group_audit,
    table_audit,
)
from egoengine_repro.evaluation.controlled_physics_replay import (  # noqa: E402
    REFERENCE_CTRL_LINEAR_SUBSTEP_INTERPOLATION,
    REFERENCE_CTRL_OUTCOME_ENDPOINT,
    SUPPORTED_COMMAND_TIMINGS,
    controlled_feedforward_forces,
    control_substep_schedule,
    direct_joint_position_actuators,
    gravity_bias_at_state,
    integrate_freejoint_pose,
    longest_true_run,
    proxy_native_geometry_relation,
    proxy_native_mismatch,
    quaternion_distance_rad,
    reference_control_target,
    reference_velocity_between_endpoints,
    synchronized_forward_snapshot,
)
from egoengine_repro.retarget.collision_audit import collision_families  # noqa: E402
from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    build_native_support_geoms,
    geom_world_vertices,
    native_hand_visual_geom_ids,
)
from egoengine_repro.scene.support_surface import Plane  # noqa: E402
from run_taco_brush_issue14_mink_candidate_v1 import relevant_pairs  # noqa: E402


SCHEMA = "taco_brush_controlled_physics_replay_v1"
SUPPORTED_SCHEMAS = {
    SCHEMA,
    "taco_brush_20231006_145_initialization_controlled_replay_v1",
    "taco_brush_20231006_145_initialization_controlled_replay_interpolated_v1",
    "taco_brush_20231006_145_endpoint_1_4_hand_table_capture_v1",
    "taco_brush_20231006_145_vertical_feedforward_replay_v1",
}
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_controlled_physics_replay_v1.yaml"
VERTICAL_FEEDFORWARD_MODE = "vertical_gravity_reference_velocity_feedforward_v1"
VERTICAL_FEEDFORWARD_ACTUATORS = (
    "R_forearm_tz_position", "L_forearm_tz_position",
)
CONTACT_GROUPS = (
    "right_hand_brush", "left_hand_bowl", "cross_hand_object",
    "brush_floor", "bowl_floor", "hand_floor", "brush_bowl",
    "intrahand", "interhand", "other",
)
FINGER_JOINT_PREFIXES = ("left_hand_", "right_hand_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved)}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def is_finger_joint_name(name: str) -> bool:
    return name.startswith(FINGER_JOINT_PREFIXES)


def urdf_finger_effort_limits(paths: tuple[Path, ...]) -> dict[str, float]:
    limits: dict[str, float] = {}
    for path in paths:
        root = ET.parse(path).getroot()
        for joint in root.findall("joint"):
            name = joint.attrib.get("name", "")
            if not is_finger_joint_name(name) or joint.attrib.get("type") == "fixed":
                continue
            limit = joint.find("limit")
            if limit is None or "effort" not in limit.attrib:
                raise ValueError(f"URDF finger joint has no effort limit: {name}")
            if name in limits:
                raise ValueError(f"duplicate URDF finger joint: {name}")
            limits[name] = float(limit.attrib["effort"])
    if len(limits) != 24:
        raise ValueError(f"expected 24 URDF finger effort limits, found {len(limits)}")
    return limits


def build_local_scene(
    source: Path,
    destination: Path,
    floor_z_m: float,
    *,
    finger_dynamics: dict[str, Any] | None = None,
    finger_effort_limits: dict[str, float] | None = None,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("source scene has no compiler")
    for key in ("meshdir", "texturedir"):
        raw = compiler.attrib.get(key)
        if raw:
            compiler.set(key, str((source.parent / raw).resolve(strict=True)))
    floor = root.find("worldbody/geom[@name='floor']")
    if floor is None:
        raise ValueError("source scene has no floor geom")
    position = np.fromstring(floor.attrib.get("pos", "0 0 0"), sep=" ")
    if position.shape != (3,):
        raise ValueError("floor position is malformed")
    position[2] = floor_z_m
    floor.set("pos", " ".join(f"{value:.17g}" for value in position))
    if finger_dynamics is not None:
        if finger_effort_limits is None:
            raise ValueError("finger dynamics require URDF effort limits")
        if finger_dynamics.get("effort_limit_scope") != "joint_actuatorfrcrange":
            raise ValueError("finger effort limits must use joint actuatorfrcrange")
        armature_mode = str(finger_dynamics["armature_mode"])
        if armature_mode not in {"preserve_source", "uniform_explicit"}:
            raise ValueError(f"unsupported finger armature mode: {armature_mode}")
        armature = None
        if armature_mode == "uniform_explicit":
            armature = float(finger_dynamics["uniform_armature_kg_m2"])
            if not np.isfinite(armature) or armature <= 0.0:
                raise ValueError("finger armature must be positive and finite")
        applied: set[str] = set()
        for joint in root.findall(".//joint"):
            name = joint.attrib.get("name", "")
            if name not in finger_effort_limits:
                continue
            limit = finger_effort_limits[name]
            joint.set("actuatorfrclimited", "true")
            joint.set("actuatorfrcrange", f"{-limit:.17g} {limit:.17g}")
            if armature is not None:
                joint.set("armature", f"{armature:.17g}")
            applied.add(name)
        missing = set(finger_effort_limits) - applied
        if missing:
            raise ValueError(f"scene is missing finger joints: {sorted(missing)}")
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def compiled_finger_contracts(
    model: mujoco.MjModel,
    effort_limits: dict[str, float],
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for name, effort_limit in sorted(effort_limits.items()):
        joint_id = int(model.joint(name).id)
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"finger joint is not a hinge: {name}")
        actuator_ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(actuator_ids) != 1:
            raise ValueError(f"finger joint {name} has {len(actuator_ids)} actuators")
        actuator_id = int(actuator_ids[0])
        gear = np.asarray(model.actuator_gear[actuator_id], dtype=np.float64)
        if not np.isclose(gear[0], 1.0) or np.any(gear[1:] != 0.0):
            raise ValueError(f"finger actuator is not direct unit gear: {name}")
        if not bool(model.jnt_actfrclimited[joint_id]):
            raise ValueError(f"finger joint force limit did not compile: {name}")
        force_range = np.asarray(model.jnt_actfrcrange[joint_id], dtype=np.float64)
        expected_range = np.asarray([-effort_limit, effort_limit])
        if not np.allclose(force_range, expected_range, rtol=0.0, atol=1e-12):
            raise ValueError(f"compiled finger effort limit changed: {name}")
        dof = int(model.jnt_dofadr[joint_id])
        contracts.append({
            "joint_name": name,
            "joint_id": joint_id,
            "qpos_address": int(model.jnt_qposadr[joint_id]),
            "dof_address": dof,
            "actuator_name": model.actuator(actuator_id).name,
            "actuator_id": actuator_id,
            "transmission": "direct_joint_unit_gear",
            "urdf_effort_limit_nm": effort_limit,
            "compiled_joint_actuator_force_range_nm": force_range,
            "compiled_armature_kg_m2": float(model.dof_armature[dof]),
            "compiled_position_gain_nm_rad": float(
                model.actuator_gainprm[actuator_id, 0]
            ),
            "compiled_velocity_gain_nm_s_rad": float(
                -model.actuator_biasprm[actuator_id, 2]
            ),
            "actuator_force_limited_separately": bool(
                model.actuator_forcelimited[actuator_id]
            ),
        })
    return contracts


def is_descendant(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == int(root_id):
            return True
        current = int(model.body_parentid[current])
    return False


def geom_role(model: mujoco.MjModel, geom_id: int) -> str:
    if int(geom_id) == int(model.geom("floor").id):
        return "floor"
    body = int(model.geom_bodyid[geom_id])
    if body == int(model.body("right_object").id):
        return "brush"
    if body == int(model.body("left_object").id):
        return "bowl"
    if is_descendant(model, body, int(model.body("R_forearm_ty_link").id)):
        return "right_hand"
    if is_descendant(model, body, int(model.body("L_forearm_ty_link").id)):
        return "left_hand"
    return "other"


def contact_group(first: str, second: str) -> str:
    roles = {first, second}
    if roles == {"right_hand", "brush"}:
        return "right_hand_brush"
    if roles == {"left_hand", "bowl"}:
        return "left_hand_bowl"
    if (("right_hand" in roles or "left_hand" in roles)
            and ("brush" in roles or "bowl" in roles)):
        return "cross_hand_object"
    if roles == {"brush", "floor"}:
        return "brush_floor"
    if roles == {"bowl", "floor"}:
        return "bowl_floor"
    if "floor" in roles and ("right_hand" in roles or "left_hand" in roles):
        return "hand_floor"
    if roles == {"brush", "bowl"}:
        return "brush_bowl"
    if first == second and first in {"right_hand", "left_hand"}:
        return "intrahand"
    if roles == {"right_hand", "left_hand"}:
        return "interhand"
    return "other"


def empty_interval() -> dict[str, dict[str, Any]]:
    return {
        group: {
            "normal_impulse_ns": 0.0,
            "max_force_n": 0.0,
            "max_proxy_penetration_m": 0.0,
            "contact_substeps": 0,
            "max_force_pair": None,
        }
        for group in CONTACT_GROUPS
    }


def sample_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, dt: float,
    interval: dict[str, dict[str, Any]], *, source: int, substep: int,
    interval_start_time_s: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    maxima = {"force_n": 0.0, "proxy_penetration_m": 0.0}
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        role1, role2 = geom_role(model, first), geom_role(model, second)
        group = contact_group(role1, role2)
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal = max(0.0, float(wrench[0]))
        force = float(np.linalg.norm(wrench[:3]))
        penetration = max(0.0, -float(contact.dist))
        record = interval[group]
        record["normal_impulse_ns"] += normal * dt
        if force > record["max_force_n"]:
            record["max_force_n"] = force
            record["max_force_pair"] = [model.geom(first).name, model.geom(second).name]
        record["max_proxy_penetration_m"] = max(
            record["max_proxy_penetration_m"], penetration,
        )
        seen_groups.add(group)
        maxima["force_n"] = max(maxima["force_n"], force)
        maxima["proxy_penetration_m"] = max(
            maxima["proxy_penetration_m"], penetration,
        )
        rows.append({
            "source_endpoint": source,
            "outcome_endpoint": source + 1,
            "substep": substep,
            "contact_state_time_s": interval_start_time_s,
            "interval_start_time_s": interval_start_time_s,
            "interval_end_time_s": float(data.time),
            "contact_index": index,
            "group": group,
            "geom1": model.geom(first).name,
            "geom2": model.geom(second).name,
            "distance_m": float(contact.dist),
            "normal_force_n": normal,
            "force_norm_n": force,
            "efc_address": int(contact.efc_address),
        })
    for group in seen_groups:
        interval[group]["contact_substeps"] += 1
    return rows, maxima


def object_state(
    model: mujoco.MjModel, data: mujoco.MjData, joint_name: str,
) -> dict[str, Any]:
    joint = model.joint(joint_name)
    qadr, dadr = int(joint.qposadr[0]), int(joint.dofadr[0])
    return {
        "position_m": np.asarray(data.qpos[qadr:qadr + 3]).copy(),
        "quaternion_wxyz": np.asarray(data.qpos[qadr + 3:qadr + 7]).copy(),
        "linear_velocity_m_s": np.asarray(data.qvel[dadr:dadr + 3]).copy(),
        "angular_velocity_rad_s": np.asarray(data.qvel[dadr + 3:dadr + 6]).copy(),
    }


def native_minimum_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "body1": row.get("body1"), "body2": row.get("body2"),
        "native_visual_geom1": row.get("native_visual_geom1"),
        "native_visual_geom2": row.get("native_visual_geom2"),
        "classification": row.get("classification"),
        "distance_m": row.get("distance_m"),
        "reported_penetration_depth_m": row.get("reported_penetration_depth_m"),
        "deepest_contact_position_sim_m": row.get("deepest_contact_position_sim_m"),
        "nearest_points_sim_m": row.get("nearest_points_sim_m"),
    }


def native_audit(
    model: mujoco.MjModel, data: mujoco.MjData, plane: Plane,
    specs: list[dict[str, Any]], tolerance: float,
) -> dict[str, Any]:
    native = native_world_objects(model, data)
    families = collision_families(model)
    interactions = {}
    for spec in specs:
        family_name = "hand_tool" if spec["role"] == "tool" else "hand_target"
        pairs = relevant_pairs(
            model, families[family_name], spec["hand"], spec["object_side"],
        )
        audited = pair_group_audit(model, data, native, pairs, tolerance)
        minimum = native_minimum_summary(audited["native_minimum"])
        interactions[spec["object"]] = {
            "native_minimum": minimum,
            "proxy_minimum": audited["shell_minimum"],
            "native_penetrating_pair_count": len(audited["native_penetrating_pairs"]),
            "unknown_pair_count": len(audited["unknown_pairs"]),
        }
    table = table_audit(native, plane, tolerance)
    return {
        "interactions": interactions,
        "table": {
            name: {
                "minimum_signed_distance_m": row["minimum_signed_distance_m"],
                "classification": row["classification"],
                "worst_location": row["worst_location"],
            }
            for name, row in table["entities"].items()
        },
    }


def native_hand_table_minima(
    data: mujoco.MjData,
    plane: Plane,
    hand_geoms: tuple[Any, ...],
) -> dict[str, float]:
    """Return exact native mesh-to-plane minima for each hand."""
    result = {"left": np.inf, "right": np.inf}
    for geom in hand_geoms:
        side = geom.name.split("_", 1)[0]
        if side not in result:
            continue
        signed = plane.signed_distance(geom_world_vertices(data, geom, full=False))
        result[side] = min(result[side], float(np.min(signed)))
    if not all(np.isfinite(value) for value in result.values()):
        raise RuntimeError("native hand/table audit did not produce both hands")
    return result


def summarize_substep_native_hand_table(
    rows: list[dict[str, Any]], tolerance_m: float,
) -> dict[str, Any] | None:
    if not rows:
        return None

    def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            key: row[key]
            for key in (
                "source_endpoint", "outcome_endpoint", "substep", "substeps",
                "control_alpha", "state_time_s", "reference_minimum_distance_m",
                "actual_minimum_distance_m", "robot_qpos_error_max_abs",
                "preceding_interval_actuator_force_max_abs_n",
                "control_range_minimum_margin",
            )
        }

    def summarize_scope(scope: list[dict[str, Any]]) -> dict[str, Any]:
        reference_safe = [
            row["reference_minimum_distance_m"] >= -tolerance_m for row in scope
        ]
        actual_safe = [
            row["actual_minimum_distance_m"] >= -tolerance_m for row in scope
        ]
        reference_strict = [
            row["reference_minimum_distance_m"] >= 0.0 for row in scope
        ]
        actual_strict = [
            row["actual_minimum_distance_m"] >= 0.0 for row in scope
        ]
        first_actual_penetration = next((
            row for row, safe in zip(scope, actual_safe, strict=True) if not safe
        ), None)
        first_reference_safe_actual_penetration = next((
            row for row, ref_ok, actual_ok in zip(
                scope, reference_safe, actual_safe, strict=True,
            ) if ref_ok and not actual_ok
        ), None)
        return {
            "sample_count": len(scope),
            "reference_minimum_distance_m": min(
                row["reference_minimum_distance_m"] for row in scope
            ),
            "actual_minimum_distance_m": min(
                row["actual_minimum_distance_m"] for row in scope
            ),
            "reference_penetration_beyond_tolerance_count": int(
                len(scope) - sum(reference_safe)
            ),
            "actual_penetration_beyond_tolerance_count": int(
                len(scope) - sum(actual_safe)
            ),
            "reference_safe_actual_penetration_count": int(sum(
                ref_ok and not actual_ok
                for ref_ok, actual_ok in zip(reference_safe, actual_safe, strict=True)
            )),
            "reference_and_actual_safe_count": int(sum(
                ref_ok and actual_ok
                for ref_ok, actual_ok in zip(reference_safe, actual_safe, strict=True)
            )),
            "all_reference_and_actual_safe_within_tolerance": bool(
                all(ref_ok and actual_ok for ref_ok, actual_ok in zip(
                    reference_safe, actual_safe, strict=True,
                ))
            ),
            "strict_reference_negative_count": int(
                len(scope) - sum(reference_strict)
            ),
            "strict_actual_negative_count": int(len(scope) - sum(actual_strict)),
            "all_reference_and_actual_strictly_nonnegative": bool(
                all(ref_ok and actual_ok for ref_ok, actual_ok in zip(
                    reference_strict, actual_strict, strict=True,
                ))
            ),
            "maximum_robot_qpos_error_max_abs": max(
                row["robot_qpos_error_max_abs"] for row in scope
            ),
            "maximum_actuator_force_abs_n": max(
                row["preceding_interval_actuator_force_max_abs_n"] for row in scope
            ),
            "minimum_control_range_margin": min(
                row["control_range_minimum_margin"] for row in scope
                if row["control_range_minimum_margin"] is not None
            ),
            "first_actual_penetration_beyond_tolerance": compact(
                first_actual_penetration
            ),
            "first_reference_safe_actual_penetration": compact(
                first_reference_safe_actual_penetration
            ),
        }

    interval_4_to_5 = [row for row in rows if row["source_endpoint"] == 4]
    return {
        "tolerance_m": tolerance_m,
        "whole_replay": summarize_scope(rows),
        "interval_4_to_5": (
            summarize_scope(interval_4_to_5) if interval_4_to_5 else None
        ),
    }


def controls_within_range(model: mujoco.MjModel, controls: np.ndarray) -> bool:
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    return bool(
        np.all(controls[:, limited] >= low[limited] - 1.0e-12)
        and np.all(controls[:, limited] <= high[limited] + 1.0e-12)
    )


def warning_counts(data: mujoco.MjData) -> list[int]:
    return [int(row.number) for row in data.warning]


def hash_tree(output: Path) -> None:
    target = output / "server_artifacts.sha256"
    rows = []
    for path in sorted(value for value in output.rglob("*") if value.is_file()
                       and value != target):
        rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def settled_object_qpos(
    *, csv_path: Path, variants: dict[str, str], transform: np.ndarray,
    expected_final_time_s: float, isolated_scenes: dict[str, Path],
    timestep_s: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Recover final post-step poses and map world coordinates to simulator."""
    selected: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for name, variant in variants.items():
                if row["object_variant"] == f"{name}__{variant}":
                    selected[name] = row
    missing = sorted(set(variants) - set(selected))
    if missing:
        raise ValueError(f"missing settled object rows: {missing}")
    if any(not np.isclose(float(row["time_s"]), expected_final_time_s,
                          rtol=0.0, atol=1.0e-9)
           for row in selected.values()):
        raise ValueError("natural-seating timeseries does not end at frozen time")
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    transform_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(transform_quat, rotation.reshape(-1))
    result = {}
    reconstruction: dict[str, Any] = {}
    for name, row in selected.items():
        pre_integration_position = np.asarray([
            row["position_x_m"], row["position_y_m"], row["position_z_m"],
        ], dtype=np.float64)
        pre_integration_quaternion = np.asarray([
            row["quaternion_w"], row["quaternion_x"],
            row["quaternion_y"], row["quaternion_z"],
        ], dtype=np.float64)
        post_integration_velocity = np.asarray([
            row["linear_velocity_x_m_s"], row["linear_velocity_y_m_s"],
            row["linear_velocity_z_m_s"], row["angular_velocity_x_rad_s"],
            row["angular_velocity_y_rad_s"], row["angular_velocity_z_rad_s"],
        ], dtype=np.float64)
        isolated_model = mujoco.MjModel.from_xml_path(str(isolated_scenes[name]))
        world_pose = integrate_freejoint_pose(
            mujoco, isolated_model,
            np.concatenate((pre_integration_position, pre_integration_quaternion)),
            post_integration_velocity,
            timestep_s,
        )
        world_position = world_pose[:3]
        world_quaternion = world_pose[3:7]
        simulator_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(simulator_quaternion, transform_quat, world_quaternion)
        simulator_quaternion /= np.linalg.norm(simulator_quaternion)
        result[name] = np.concatenate((rotation @ world_position + translation,
                                       simulator_quaternion))
        reconstruction[name] = {
            "source_record_time_s": float(row["time_s"]),
            "source_geometry_state_time_s": float(row["time_s"]) - timestep_s,
            "source_velocity_state_time_s": float(row["time_s"]),
            "reconstructed_pose_time_s": float(row["time_s"]),
            "timestep_s": timestep_s,
            "pre_integration_world_pose": np.concatenate((
                pre_integration_position, pre_integration_quaternion,
            )),
            "post_integration_world_velocity": post_integration_velocity,
            "reconstructed_world_pose": world_pose,
            "simulator_pose": result[name],
        }
    return result, reconstruction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError("unexpected controlled Replay schema")
    forbidden = (
        "modify_source_scene", "modify_reference", "modify_active_support_contract",
        "modify_collision_geometry", "mink", "full_209_frame_replay", "mpc",
        "reinforcement_learning", "promotion", "chunk_commit",
    )
    if any(cfg["authorization"].get(key) is not False for key in forbidden):
        raise ValueError("authorization is not fail-closed")
    command_timing = str(cfg["runtime"]["command_timing"])
    if command_timing not in SUPPORTED_COMMAND_TIMINGS:
        raise ValueError(f"unsupported command timing: {command_timing}")
    reference_time_scale = float(cfg["runtime"].get("reference_time_scale", 1.0))
    if not np.isfinite(reference_time_scale) or reference_time_scale < 1.0:
        raise ValueError("reference time scale must be finite and at least one")
    controller_cfg = cfg.get("controller")
    if controller_cfg is None:
        controller_mode = "position_servo_only"
        gravity_feedforward = False
        reference_velocity_feedback = False
        controlled_actuator_names: tuple[str, ...] = ()
        maximum_extra_force_n = 0.0
    else:
        if controller_cfg.get("mode") != VERTICAL_FEEDFORWARD_MODE:
            raise ValueError("unsupported experimental controller mode")
        controller_mode = str(controller_cfg["mode"])
        for key in ("gravity_feedforward", "reference_velocity_feedback"):
            if not isinstance(controller_cfg.get(key), bool):
                raise ValueError(f"controller {key} must be a YAML boolean")
        gravity_feedforward = controller_cfg["gravity_feedforward"]
        reference_velocity_feedback = controller_cfg[
            "reference_velocity_feedback"
        ]
        controlled_actuator_names = tuple(
            str(name) for name in controller_cfg["controlled_actuators"]
        )
        if controlled_actuator_names != VERTICAL_FEEDFORWARD_ACTUATORS:
            raise ValueError(
                "vertical experiment must control only the frozen wrist actuators"
            )
        maximum_extra_force_n = float(
            controller_cfg["maximum_abs_extra_generalized_force_n"]
        )
        if not np.isfinite(maximum_extra_force_n) or maximum_extra_force_n <= 0.0:
            raise ValueError("controller force hard stop must be positive and finite")
        if reference_velocity_feedback and command_timing != (
            REFERENCE_CTRL_LINEAR_SUBSTEP_INTERPOLATION
        ):
            raise ValueError(
                "reference velocity feedback requires linear substep targets"
            )
    trace_substep_native = bool(
        cfg.get("diagnostics", {}).get("substep_native_hand_table_trace", False)
    )
    trace_substep_state = bool(
        cfg.get("diagnostics", {}).get("substep_state_trace", False)
    )
    source_endpoint = int(cfg["sample"]["replay_source_endpoint"])
    outcome_endpoint = int(cfg["sample"]["replay_outcome_endpoint"])
    if source_endpoint != 0 or outcome_endpoint <= source_endpoint:
        raise ValueError("controlled Replay requires a positive endpoint-0 window")

    paths = {
        key: Path(value).resolve(strict=(key != "output"))
        for key, value in cfg["paths"].items()
    }
    finger_dynamics = cfg.get("finger_dynamics")
    finger_effort_limits: dict[str, float] = {}
    if finger_dynamics is not None:
        if not bool(finger_dynamics.get("require_zero_finger_qfrc_applied")):
            raise ValueError("finger experiment must forbid external finger forces")
        finger_effort_limits = urdf_finger_effort_limits((
            paths["left_hand_urdf"], paths["right_hand_urdf"],
        ))
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    local_scene = output / "experiment_local_scene.xml"
    build_local_scene(
        paths["scene"], local_scene, float(cfg["support"]["simulator_z_m"]),
        finger_dynamics=finger_dynamics,
        finger_effort_limits=(
            finger_effort_limits if finger_dynamics is not None else None
        ),
    )
    model = mujoco.MjModel.from_xml_path(str(local_scene))
    if not np.isclose(model.geom("floor").pos[2], cfg["support"]["simulator_z_m"],
                      rtol=0.0, atol=0.0):
        raise ValueError("experiment-local floor height did not compile exactly")
    controlled_actuators = direct_joint_position_actuators(
        mujoco, model, controlled_actuator_names,
    ) if controlled_actuator_names else ()
    finger_contracts = (
        compiled_finger_contracts(model, finger_effort_limits)
        if finger_dynamics is not None else []
    )
    if finger_dynamics is not None:
        expected_armature = float(finger_dynamics["expected_compiled_armature_kg_m2"])
        compiled_armatures = np.asarray([
            contract["compiled_armature_kg_m2"] for contract in finger_contracts
        ])
        if not np.allclose(
            compiled_armatures, expected_armature, rtol=0.0, atol=1e-12,
        ):
            raise ValueError("compiled finger armature does not match frozen contract")
    if gravity_feedforward:
        if np.any(np.asarray(model.body_gravcomp) != 0.0):
            raise ValueError("model already contains body gravity compensation")
        for contract in controlled_actuators:
            if bool(model.jnt_actgravcomp[contract.joint_id]):
                raise ValueError(
                    f"{contract.actuator_name} already has gravity compensation"
                )

    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        reference = {name: np.asarray(archive[name]).copy()
                     for name in ("qpos", "qvel", "ctrl")}
        frequency = float(np.asarray(archive["frequency"]))
    reference_frames = int(cfg["sample"]["reference_frames"])
    if reference["qpos"].shape != (reference_frames, model.nq) \
            or reference["qvel"].shape != (reference_frames, model.nv) \
            or reference["ctrl"].shape != (reference_frames, model.nu):
        raise ValueError("reference dimensions changed")
    if frequency != float(cfg["sample"]["control_hz"]):
        raise ValueError("reference frequency changed")
    execution_control_hz = frequency / reference_time_scale
    controls = reference["ctrl"][:outcome_endpoint + 1]
    if not controls_within_range(model, controls):
        raise ValueError("Replay reference contains out-of-range controls")

    schedule = control_substep_schedule(
        control_endpoints=outcome_endpoint, control_hz=execution_control_hz,
        simulation_dt_s=float(model.opt.timestep),
    )
    if int(schedule.sum()) > int(cfg["hard_stop"]["maximum_physics_steps"]):
        raise ValueError("frozen physics budget is insufficient")
    support_cfg = yaml.safe_load(paths["support_contract"].read_text(encoding="utf-8"))
    simulator = support_cfg["simulator"]
    plane = Plane(
        normal=np.asarray(simulator["normal"], dtype=np.float64),
        offset=float(simulator["offset_m"]), frame=str(simulator["frame"]),
    )
    if plane.offset != float(cfg["support"]["simulator_z_m"]):
        raise ValueError("support contract and Replay contract disagree")
    hand_support_geoms = build_native_support_geoms(
        model, mujoco, native_hand_visual_geom_ids(model, mujoco),
    )

    manifest = {
        "schema": cfg["schema"],
        "status": "FROZEN_BEFORE_PHYSICS",
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
        "inputs": {
            "config": artifact(config_path), "source_scene": artifact(paths["scene"]),
            "local_scene": artifact(local_scene),
            "robot_reference": artifact(paths["robot_reference"]),
            "support_contract": artifact(paths["support_contract"]),
            "prior_collision_audit": artifact(paths["prior_collision_audit"]),
            "script": artifact(Path(__file__)),
            **({
                "left_hand_urdf": artifact(paths["left_hand_urdf"]),
                "right_hand_urdf": artifact(paths["right_hand_urdf"]),
                "finger_dynamics_ab_contract": artifact(
                    paths["finger_dynamics_ab_contract"]
                ),
            } if finger_dynamics is not None else {}),
        },
        "physics": {
            "backend": "mujoco_cpu", "mujoco_version": mujoco.__version__,
            "authored_timestep_s": float(model.opt.timestep),
            "authored_reference_hz": frequency,
            "reference_time_scale": reference_time_scale,
            "execution_control_hz": execution_control_hz,
            "substeps_per_interval": schedule.tolist(),
            "total_planned_physics_steps": int(schedule.sum()),
            "initial_qvel": "ZERO", "command_timing": command_timing,
            "control_target_rule": (
                "ctrl[outcome_endpoint] for every physics substep"
                if command_timing == REFERENCE_CTRL_OUTCOME_ENDPOINT
                else "ctrl[source] + ((substep + 1) / substeps) * "
                     "(ctrl[outcome] - ctrl[source])"
            ),
            "substep_native_hand_table_trace": trace_substep_native,
            "substep_state_trace": trace_substep_state,
            "native_geometry_state_alignment": (
                "copy post-step integration state into independent MjData, then mj_forward"
            ),
            "force_time_alignment": (
                "contact and actuator forces are the forces solved at interval_start_time_s "
                "and applied over [interval_start_time_s, interval_end_time_s]; post-step "
                "qpos/qvel and refreshed native geometry are at state_time_s=interval_end_time_s"
            ),
        },
        "controller": {
            "mode": controller_mode,
            "gravity_feedforward": gravity_feedforward,
            "reference_velocity_feedback": reference_velocity_feedback,
            "controlled_actuators": [
                {
                    "actuator_name": contract.actuator_name,
                    "actuator_id": contract.actuator_id,
                    "joint_name": contract.joint_name,
                    "joint_id": contract.joint_id,
                    "qpos_address": contract.qpos_address,
                    "dof_address": contract.dof_address,
                    "position_gain": contract.position_gain,
                    "velocity_gain": contract.velocity_gain,
                    "transmission": "direct_joint_unit_gear",
                    "force_limited": bool(
                        model.actuator_forcelimited[contract.actuator_id]
                    ),
                    "joint_actuator_gravity_compensation": bool(
                        model.jnt_actgravcomp[contract.joint_id]
                    ),
                    "body_gravity_compensation": float(
                        model.body_gravcomp[model.jnt_bodyid[contract.joint_id]]
                    ),
                }
                for contract in controlled_actuators
            ],
            "maximum_abs_extra_generalized_force_n": maximum_extra_force_n,
            "reference_velocity_rule": (
                "mj_differentiatePos(qpos[source], qpos[outcome], "
                "substeps * model.opt.timestep)"
                if controlled_actuators else "NONE"
            ),
            "gravity_rule": (
                "independent MjModel and MjData at live qpos with qvel=0; "
                "use qfrc_bias"
                if gravity_feedforward else "DISABLED"
            ),
            "application_rule": (
                "restore initial qfrc_applied each substep, then add sparse "
                "feedforward generalized force"
                if controlled_actuators else "NONE"
            ),
        },
        "finger_dynamics": ({
            **finger_dynamics,
            "compiled_contracts": finger_contracts,
            "external_generalized_force_rule": (
                "qfrc_applied must remain exactly zero at every finger DOF; "
                "joint actuatorfrcrange limits qfrc_actuator after transmission"
            ),
        } if finger_dynamics is not None else None),
        "hard_stop": cfg["hard_stop"],
        "strict_pickup_gate": cfg["strict_pickup_gate"],
        "authorization": cfg["authorization"],
    }
    write_json(output / "input_manifest.json", manifest)

    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"][0]
    initial_state_provenance: dict[str, Any] = {
        "robot": "robot_reference_endpoint_0",
        "objects": "robot_reference_endpoint_0",
        "object_velocity": "ZERO",
    }
    if "initial_state" in cfg:
        initialization = cfg["initial_state"]
        robot_results_path = Path(initialization["robot_results"]).resolve(strict=True)
        transform_report = json.loads(robot_results_path.read_text(encoding="utf-8"))
        candidate_name = str(initialization["robot_candidate"])
        candidate = transform_report["mink_candidates"][candidate_name]
        if candidate["status"] != "MINK_PASS":
            raise ValueError("selected robot initialization did not pass MINK gate")
        if candidate["robot_reference"]["sha256"] != sha256(paths["robot_reference"]):
            raise ValueError("robot reference does not match selected passing candidate")
        seating_results_path = Path(
            initialization["natural_seating_results"]
        ).resolve(strict=True)
        seating_results = json.loads(seating_results_path.read_text(encoding="utf-8"))
        if not seating_results["decision"]["object_settling_passed_all_variants"]:
            raise ValueError("natural-seating gate did not pass all frozen variants")
        transform = np.asarray(transform_report["T_sim_world"], dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("initial-state transform must be 4x4")
        seating_config_path = Path(
            seating_results["source_artifacts"]["config"]["path"]
        ).resolve(strict=True)
        seating_config = yaml.safe_load(
            seating_config_path.read_text(encoding="utf-8")
        )
        isolated_scenes = {
            name: Path(
                seating_results["source_artifacts"]["isolated_scenes"][name]["path"]
            ).resolve(strict=True)
            for name in initialization["object_variants"]
        }
        settled, seating_state_reconstruction = settled_object_qpos(
            csv_path=Path(initialization["natural_seating_timeseries"]).resolve(strict=True),
            variants=dict(initialization["object_variants"]), transform=transform,
            expected_final_time_s=float(initialization["expected_final_time_s"]),
            isolated_scenes=isolated_scenes,
            timestep_s=float(seating_config["physics"]["timestep_s"]),
        )
        for name, joint_name in (("brush", "right_object_joint"),
                                 ("bowl", "left_object_joint")):
            joint = model.joint(joint_name)
            qadr = int(joint.qposadr[0])
            data.qpos[qadr:qadr + 7] = settled[name]
        initial_state_provenance = {
            "robot": initialization["robot"],
            "objects": initialization["objects"],
            "object_velocity": "ZERO",
            "natural_seating_timeseries": artifact(
                Path(initialization["natural_seating_timeseries"])
            ),
            "natural_seating_results": artifact(seating_results_path),
            "robot_results": artifact(robot_results_path),
            "robot_candidate": candidate_name,
            "object_variants": initialization["object_variants"],
            "natural_seating_state_alignment": (
                "reconstructed post-step freejoint pose from the recorded pre-integration "
                "xpos/xquat and post-integration qvel with mj_integratePos"
            ),
            "natural_seating_state_reconstruction": seating_state_reconstruction,
            "settled_object_qpos_simulator": settled,
        }
    data.qvel[:] = 0.0
    data.ctrl[:] = reference["ctrl"][0]
    mujoco.mj_forward(model, data)
    initial_warning = warning_counts(data)
    if any(initial_warning):
        raise RuntimeError(f"MuJoCo warning during initialization: {initial_warning}")
    manifest["initial_state_provenance"] = initial_state_provenance
    initial_qfrc_applied = data.qfrc_applied.copy()
    finger_dofs = np.asarray([
        contract["dof_address"] for contract in finger_contracts
    ], dtype=np.int64)
    if len(finger_dofs) and np.any(initial_qfrc_applied[finger_dofs] != 0.0):
        raise ValueError("initial qfrc_applied is nonzero on a finger DOF")
    manifest["controller"]["preserved_initial_qfrc_applied"] = (
        initial_qfrc_applied
    )
    write_json(output / "input_manifest.json", manifest)

    object_specs = {
        "brush": {"joint": "right_object_joint", "contact": "right_hand_brush",
                  "floor": "brush_floor"},
        "bowl": {"joint": "left_object_joint", "contact": "left_hand_bowl",
                 "floor": "bowl_floor"},
    }
    initial_objects = {
        name: object_state(model, data, spec["joint"])
        for name, spec in object_specs.items()
    }
    audit_endpoints = set(int(value) for value in cfg["native_geometry_audit"]["endpoints"])
    native_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    substep_native_rows: list[dict[str, Any]] = []
    substep_state_rows: list[dict[str, Any]] = []
    controller_rows: list[dict[str, Any]] = []
    reference_data = (
        mujoco.MjData(model)
        if trace_substep_native or trace_substep_state else None
    )
    actual_trace_data = mujoco.MjData(model) if trace_substep_native else None
    endpoint_audit_data = mujoco.MjData(model)
    gravity_model = (
        mujoco.MjModel.from_xml_path(str(local_scene))
        if gravity_feedforward else None
    )
    gravity_data = (
        mujoco.MjData(gravity_model) if gravity_model is not None else None
    )
    initial_actual_qpos = data.qpos.copy()
    initial_actual_qvel = data.qvel.copy()
    initial_actual_ctrl = data.ctrl.copy()
    hard_stop: dict[str, Any] | None = None

    def append_endpoint(endpoint: int, interval: dict[str, dict[str, Any]]) -> None:
        row: dict[str, Any] = {"endpoint": endpoint, "time_s": float(data.time)}
        for name, spec in object_specs.items():
            state = object_state(model, data, spec["joint"])
            joint = model.joint(spec["joint"])
            qadr = int(joint.qposadr[0])
            ref_pos = reference["qpos"][endpoint, qadr:qadr + 3]
            ref_quat = reference["qpos"][endpoint, qadr + 3:qadr + 7]
            row.update({
                f"{name}_position_m": state["position_m"],
                f"{name}_quaternion_wxyz": state["quaternion_wxyz"],
                f"{name}_linear_velocity_m_s": state["linear_velocity_m_s"],
                f"{name}_angular_velocity_rad_s": state["angular_velocity_rad_s"],
                f"{name}_actual_lift_m": float(
                    state["position_m"][2] - initial_objects[name]["position_m"][2]
                ),
                f"{name}_reference_lift_m": float(
                    ref_pos[2] - reference["qpos"][0, qadr + 2]
                ),
                f"{name}_position_error_m": float(np.linalg.norm(state["position_m"] - ref_pos)),
                f"{name}_rotation_error_rad": quaternion_distance_rad(
                    state["quaternion_wxyz"], ref_quat,
                ),
                f"{name}_hand_normal_impulse_ns": interval[spec["contact"]]["normal_impulse_ns"],
                f"{name}_floor_normal_impulse_ns": interval[spec["floor"]]["normal_impulse_ns"],
                f"{name}_hand_max_force_n": interval[spec["contact"]]["max_force_n"],
                f"{name}_floor_max_force_n": interval[spec["floor"]]["max_force_n"],
                f"{name}_hand_contact_substeps": interval[spec["contact"]]["contact_substeps"],
                f"{name}_floor_contact_substeps": interval[spec["floor"]]["contact_substeps"],
            })
            if "strict_tracking_gate" in cfg:
                position_scale = float(cfg["strict_tracking_gate"]["position_scale_m"])
                rotation_scale = float(cfg["strict_tracking_gate"]["rotation_scale_rad"])
                row[f"{name}_tracking_score"] = float(np.sqrt(
                    (row[f"{name}_position_error_m"] / position_scale) ** 2
                    + (row[f"{name}_rotation_error_rad"] / rotation_scale) ** 2
                ))
        row["interval_groups"] = copy.deepcopy(interval)
        endpoint_rows.append(row)
        if endpoint in audit_endpoints:
            synchronized_forward_snapshot(
                mujoco, model, data, endpoint_audit_data,
            )
            audited = native_audit(
                model, endpoint_audit_data, plane,
                list(cfg["native_geometry_audit"]["interaction_pairs"]),
                float(cfg["native_geometry_audit"]["hand_object_and_hand_table_tolerance_m"]),
            )
            native_row = {"endpoint": endpoint, **audited}
            for name, spec in object_specs.items():
                minimum = audited["interactions"][name]["native_minimum"]
                if minimum is None or minimum["distance_m"] is None:
                    mismatch = "UNKNOWN_NATIVE_DISTANCE"
                else:
                    mismatch = proxy_native_mismatch(
                        proxy_contact=interval[spec["contact"]]["normal_impulse_ns"] > 0.0,
                        native_distance_m=float(minimum["distance_m"]),
                        tolerance_m=float(
                            cfg["native_geometry_audit"]["hand_object_and_hand_table_tolerance_m"]
                        ),
                    )
                proxy_minimum = native_row["interactions"][name]["proxy_minimum"]
                if minimum is None or minimum["distance_m"] is None or proxy_minimum is None:
                    geometry = "UNKNOWN_GEOMETRY_DISTANCE"
                else:
                    geometry = proxy_native_geometry_relation(
                        proxy_distance_m=float(proxy_minimum["distance_m"]),
                        native_distance_m=float(minimum["distance_m"]),
                        tolerance_m=float(
                            cfg["native_geometry_audit"]["hand_object_and_hand_table_tolerance_m"]
                        ),
                    )
                native_row["interactions"][name]["force_native_exposure_classification"] = mismatch
                native_row["interactions"][name]["proxy_native_geometry_relation"] = geometry
                native_row["interactions"][name]["preceding_interval_normal_impulse_ns"] = (
                    interval[spec["contact"]]["normal_impulse_ns"]
                )
                native_row["interactions"][name]["preceding_interval_object_displacement_m"] = (
                    0.0 if endpoint == 0 else float(np.linalg.norm(
                        row[f"{name}_position_m"]
                        - endpoint_rows[-2][f"{name}_position_m"]
                    ))
                )
            native_rows.append(native_row)

    append_endpoint(0, empty_interval())
    previous_objects = copy.deepcopy(initial_objects)
    physics_steps = 0
    limits = cfg["hard_stop"]
    for source, substeps in enumerate(schedule):
        outcome = source + 1
        interval = empty_interval()
        reference_interval_velocity = reference_velocity_between_endpoints(
            mujoco,
            model,
            reference["qpos"][source],
            reference["qpos"][outcome],
            float(substeps) * float(model.opt.timestep),
        )
        reference_tangent = np.empty(model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(
            model, reference_tangent, 1.0,
            reference["qpos"][source], reference["qpos"][outcome],
        )
        reference_tangent[model.nu:] = 0.0
        for substep in range(int(substeps)):
            target, control_alpha = reference_control_target(
                reference["ctrl"], source_endpoint=source,
                outcome_endpoint=outcome, substep=substep,
                substeps=int(substeps), command_timing=command_timing,
            )
            data.ctrl[:] = target
            interval_start_time_s = float(data.time)
            controller_step_rows: list[dict[str, Any]] = []
            applied_feedforward = np.zeros(model.nv, dtype=np.float64)
            if controlled_actuators:
                gravity_bias = np.zeros(model.nv, dtype=np.float64)
                if gravity_feedforward:
                    if gravity_model is None or gravity_data is None:
                        raise RuntimeError(
                            "gravity calculation buffer was not initialized"
                        )
                    gravity_bias = gravity_bias_at_state(
                        mujoco, gravity_model, data, gravity_data,
                    )
                feedforward = controlled_feedforward_forces(
                    nv=model.nv,
                    contracts=controlled_actuators,
                    gravity_bias=gravity_bias,
                    reference_velocity=reference_interval_velocity,
                    gravity_feedforward=gravity_feedforward,
                    reference_velocity_feedback=reference_velocity_feedback,
                )
                applied_feedforward = feedforward.total
                controlled_extra = np.asarray([
                    applied_feedforward[contract.dof_address]
                    for contract in controlled_actuators
                ])
                if np.max(np.abs(controlled_extra)) > maximum_extra_force_n:
                    hard_stop = {
                        "reasons": ["CONTROLLER_EXTRA_FORCE_LIMIT"],
                        "source_endpoint": source,
                        "planned_outcome_endpoint": outcome,
                        "substep": substep,
                        "physics_step": physics_steps,
                        "time_s": float(data.time),
                        "maximum_abs_extra_generalized_force_n": float(
                            np.max(np.abs(controlled_extra))
                        ),
                        "configured_limit_n": maximum_extra_force_n,
                    }
                    break
                if gravity_feedforward or reference_velocity_feedback:
                    data.qfrc_applied[:] = initial_qfrc_applied
                    data.qfrc_applied[:] += applied_feedforward
                for contract in controlled_actuators:
                    actuator = contract.actuator_id
                    qpos_address = contract.qpos_address
                    dof = contract.dof_address
                    position_force = contract.position_gain * (
                        data.ctrl[actuator] - data.qpos[qpos_address]
                    )
                    damping_force = -contract.velocity_gain * data.qvel[dof]
                    controller_step_rows.append({
                        "source_endpoint": source,
                        "outcome_endpoint": outcome,
                        "substep": substep,
                        "substeps": int(substeps),
                        "control_alpha": control_alpha,
                        "interval_start_time_s": interval_start_time_s,
                        "interval_end_time_s": (
                            interval_start_time_s + float(model.opt.timestep)
                        ),
                        "actuator_name": contract.actuator_name,
                        "actuator_id": actuator,
                        "joint_name": contract.joint_name,
                        "dof_address": dof,
                        "reference_position_target": float(data.ctrl[actuator]),
                        "reference_velocity": float(
                            reference_interval_velocity[dof]
                        ),
                        "actual_position_start": float(data.qpos[qpos_address]),
                        "actual_velocity_start": float(data.qvel[dof]),
                        "target_minus_actual_start_m": float(
                            data.ctrl[actuator] - data.qpos[qpos_address]
                        ),
                        "position_servo_force_n": float(position_force),
                        "velocity_damping_force_n": float(damping_force),
                        "original_actuator_force_n": float(
                            position_force + damping_force
                        ),
                        "gravity_bias_n": float(gravity_bias[dof]),
                        "gravity_feedforward_n": float(feedforward.gravity[dof]),
                        "reference_velocity_force_n": float(
                            feedforward.reference_velocity[dof]
                        ),
                        "extra_generalized_force_n": float(
                            applied_feedforward[dof]
                        ),
                        "preserved_initial_applied_force_n": float(
                            initial_qfrc_applied[dof]
                        ),
                        "total_commanded_generalized_force_n": float(
                            position_force + damping_force
                            + data.qfrc_applied[dof]
                        ),
                    })
            if len(finger_dofs) and np.any(data.qfrc_applied[finger_dofs] != 0.0):
                raise RuntimeError(
                    "external generalized force bypassed a finger effort limit"
                )
            mujoco.mj_step(model, data)
            physics_steps += 1
            if finger_contracts:
                for contract in finger_contracts:
                    dof = contract["dof_address"]
                    limit = contract["urdf_effort_limit_nm"]
                    total = float(data.qfrc_actuator[dof] + data.qfrc_applied[dof])
                    if abs(total) > limit + 1e-10:
                        raise RuntimeError(
                            "finger commanded generalized force exceeded URDF limit: "
                            f"{contract['joint_name']} {total} vs {limit}"
                        )
            rows, maxima = sample_contacts(
                model, data, float(model.opt.timestep), interval,
                source=source, substep=substep,
                interval_start_time_s=interval_start_time_s,
            )
            contact_rows.extend(rows)
            states = {
                name: object_state(model, data, spec["joint"])
                for name, spec in object_specs.items()
            }
            reference_qpos_at_substep = None
            actual_table = None
            reference_table = None
            if trace_substep_native or trace_substep_state:
                if reference_data is None:
                    raise RuntimeError("reference trace data was not initialized")
                reference_data.qpos[:] = reference["qpos"][source]
                mujoco.mj_integratePos(
                    model, reference_data.qpos,
                    reference_tangent, control_alpha,
                )
                reference_data.qvel[:] = 0.0
                mujoco.mj_forward(model, reference_data)
                reference_qpos_at_substep = reference_data.qpos.copy()
            if trace_substep_native:
                if actual_trace_data is None:
                    raise RuntimeError("actual trace data was not initialized")
                synchronized_forward_snapshot(
                    mujoco, model, data, actual_trace_data,
                )
                actual_table = native_hand_table_minima(
                    actual_trace_data, plane, hand_support_geoms,
                )
                reference_table = native_hand_table_minima(
                    reference_data, plane, hand_support_geoms,
                )
                qpos_error = np.empty(model.nv, dtype=np.float64)
                mujoco.mj_differentiatePos(
                    model, qpos_error, 1.0,
                    reference_data.qpos, data.qpos,
                )
                limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
                control_margin = np.minimum(
                    data.ctrl[limited] - model.actuator_ctrlrange[limited, 0],
                    model.actuator_ctrlrange[limited, 1] - data.ctrl[limited],
                )
                substep_native_rows.append({
                    "source_endpoint": source,
                    "outcome_endpoint": outcome,
                    "substep": substep,
                    "substeps": int(substeps),
                    "control_alpha": control_alpha,
                    "state_time_s": float(data.time),
                    "force_interval_start_time_s": interval_start_time_s,
                    "force_interval_end_time_s": float(data.time),
                    "reference_left_distance_m": reference_table["left"],
                    "reference_right_distance_m": reference_table["right"],
                    "reference_minimum_distance_m": min(reference_table.values()),
                    "actual_left_distance_m": actual_table["left"],
                    "actual_right_distance_m": actual_table["right"],
                    "actual_minimum_distance_m": min(actual_table.values()),
                    "robot_qpos_error_max_abs": float(
                        np.max(np.abs(qpos_error[:model.nu]))
                    ),
                    "preceding_interval_actuator_force_max_abs_n": float(
                        np.max(np.abs(data.actuator_force))
                    ),
                    "control_range_minimum_margin": (
                        float(np.min(control_margin)) if len(control_margin) else None
                    ),
                })
            if trace_substep_state:
                if reference_qpos_at_substep is None:
                    raise RuntimeError("reference substep state was not initialized")
                substep_state_rows.append({
                    "source_endpoint": source,
                    "outcome_endpoint": outcome,
                    "substep": substep,
                    "substeps": int(substeps),
                    "control_alpha": control_alpha,
                    "state_time_s": float(data.time),
                    "force_interval_start_time_s": interval_start_time_s,
                    "force_interval_end_time_s": float(data.time),
                    "reference_qpos": reference_qpos_at_substep,
                    "actual_qpos": data.qpos.copy(),
                    "actual_qvel": data.qvel.copy(),
                    "control_target": data.ctrl.copy(),
                    "reference_qvel": reference_interval_velocity.copy(),
                    "applied_feedforward_force": applied_feedforward.copy(),
                    "qfrc_applied": data.qfrc_applied.copy(),
                    "preceding_interval_actuator_force": data.actuator_force.copy(),
                    "preceding_interval_qfrc_actuator": data.qfrc_actuator.copy(),
                    "preceding_interval_qfrc_constraint": data.qfrc_constraint.copy(),
                })
            for controller_row, contract in zip(
                controller_step_rows, controlled_actuators, strict=True,
            ):
                dof = contract.dof_address
                qpos_address = contract.qpos_address
                side = "right" if contract.actuator_name.startswith("R_") else "left"
                controller_row.update({
                    "actual_position_end": float(data.qpos[qpos_address]),
                    "actual_velocity_end": float(data.qvel[dof]),
                    "reference_minus_actual_end_m": float(
                        data.ctrl[contract.actuator_id] - data.qpos[qpos_address]
                    ),
                    "solved_actuator_force_n": float(
                        data.actuator_force[contract.actuator_id]
                    ),
                    "solved_constraint_force_n": float(data.qfrc_constraint[dof]),
                    "reference_hand_table_distance_m": (
                        None if reference_table is None
                        else float(reference_table[side])
                    ),
                    "actual_hand_table_distance_m": (
                        None if actual_table is None else float(actual_table[side])
                    ),
                })
                controller_rows.append(controller_row)
            failures = []
            arrays = [data.qpos, data.qvel, data.ctrl, data.qacc]
            if not all(np.isfinite(value).all() for value in arrays):
                failures.append("NONFINITE_STATE")
            warnings = warning_counts(data)
            if limits["fail_on_any_mujoco_warning"] and any(warnings):
                failures.append("MUJOCO_WARNING")
            if maxima["force_n"] > float(limits["maximum_contact_force_n"]):
                failures.append("CONTACT_FORCE_LIMIT")
            if maxima["proxy_penetration_m"] > float(limits["maximum_proxy_penetration_m"]):
                failures.append("PROXY_PENETRATION_LIMIT")
            for name, state in states.items():
                linear_speed = float(np.linalg.norm(state["linear_velocity_m_s"]))
                angular_speed = float(np.linalg.norm(state["angular_velocity_rad_s"]))
                translation = float(np.linalg.norm(
                    state["position_m"] - previous_objects[name]["position_m"]
                ))
                rotation = quaternion_distance_rad(
                    state["quaternion_wxyz"], previous_objects[name]["quaternion_wxyz"],
                )
                if linear_speed > float(limits["maximum_object_linear_speed_m_s"]):
                    failures.append(f"{name.upper()}_LINEAR_SPEED_LIMIT")
                if angular_speed > float(limits["maximum_object_angular_speed_rad_s"]):
                    failures.append(f"{name.upper()}_ANGULAR_SPEED_LIMIT")
                if translation > float(limits["maximum_object_translation_per_substep_m"]):
                    failures.append(f"{name.upper()}_SUBSTEP_TRANSLATION_LIMIT")
                if rotation > float(limits["maximum_object_rotation_per_substep_rad"]):
                    failures.append(f"{name.upper()}_SUBSTEP_ROTATION_LIMIT")
            previous_objects = states
            if failures:
                hard_stop = {
                    "reasons": sorted(set(failures)), "source_endpoint": source,
                    "planned_outcome_endpoint": outcome, "substep": substep,
                    "physics_step": physics_steps, "time_s": float(data.time),
                    "warning_counts": warnings, "maximum_contact_force_n": maxima["force_n"],
                    "maximum_proxy_penetration_m": maxima["proxy_penetration_m"],
                    "object_states": states,
                }
                break
        if hard_stop is not None:
            break
        append_endpoint(outcome, interval)

    # Flatten endpoint records for portable CSV/NPZ evidence.
    scalar_fields = [
        "endpoint", "time_s",
        *[f"{name}_{suffix}" for name in object_specs for suffix in (
            "actual_lift_m", "reference_lift_m", "position_error_m", "rotation_error_rad",
            "hand_normal_impulse_ns", "floor_normal_impulse_ns", "hand_max_force_n",
            "floor_max_force_n", "hand_contact_substeps", "floor_contact_substeps",
        )],
    ]
    if "strict_tracking_gate" in cfg:
        scalar_fields.extend(
            f"{name}_tracking_score" for name in object_specs
        )
    with (output / "endpoint_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=scalar_fields, lineterminator="\n",
        )
        writer.writeheader()
        for row in endpoint_rows:
            writer.writerow({key: row[key] for key in scalar_fields})
    with gzip.open(output / "contact_trace.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        fields = [
            "source_endpoint", "outcome_endpoint", "substep",
            "contact_state_time_s", "interval_start_time_s", "interval_end_time_s",
            "contact_index",
            "group", "geom1", "geom2", "distance_m", "normal_force_n", "force_norm_n",
            "efc_address",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(contact_rows)
    if controlled_actuators:
        with (output / "controller_trace.csv").open(
            "w", newline="", encoding="utf-8",
        ) as stream:
            fields = [
                "source_endpoint", "outcome_endpoint", "substep", "substeps",
                "control_alpha", "interval_start_time_s", "interval_end_time_s",
                "actuator_name", "actuator_id", "joint_name", "dof_address",
                "reference_position_target", "reference_velocity",
                "actual_position_start", "actual_velocity_start",
                "target_minus_actual_start_m", "position_servo_force_n",
                "velocity_damping_force_n", "original_actuator_force_n",
                "gravity_bias_n", "gravity_feedforward_n",
                "reference_velocity_force_n", "extra_generalized_force_n",
                "preserved_initial_applied_force_n",
                "total_commanded_generalized_force_n",
                "actual_position_end", "actual_velocity_end",
                "reference_minus_actual_end_m", "solved_actuator_force_n",
                "solved_constraint_force_n", "reference_hand_table_distance_m",
                "actual_hand_table_distance_m",
            ]
            writer = csv.DictWriter(
                stream, fieldnames=fields, lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(controller_rows)
    if trace_substep_native:
        with gzip.open(
            output / "substep_native_hand_table_trace.csv.gz",
            "wt", newline="", encoding="utf-8",
        ) as stream:
            fields = [
                "source_endpoint", "outcome_endpoint", "substep", "substeps",
                "control_alpha", "state_time_s",
                "force_interval_start_time_s", "force_interval_end_time_s",
                "reference_left_distance_m", "reference_right_distance_m",
                "reference_minimum_distance_m",
                "actual_left_distance_m", "actual_right_distance_m",
                "actual_minimum_distance_m", "robot_qpos_error_max_abs",
                "preceding_interval_actuator_force_max_abs_n",
                "control_range_minimum_margin",
            ]
            writer = csv.DictWriter(
                stream, fieldnames=fields, lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(substep_native_rows)
    if trace_substep_state:
        scalar_names = (
            "source_endpoint", "outcome_endpoint", "substep", "substeps",
            "control_alpha", "state_time_s",
            "force_interval_start_time_s", "force_interval_end_time_s",
        )
        array_names = (
            "reference_qpos", "actual_qpos", "actual_qvel",
            "control_target", "reference_qvel", "applied_feedforward_force",
            "qfrc_applied", "preceding_interval_actuator_force",
            "preceding_interval_qfrc_actuator",
            "preceding_interval_qfrc_constraint",
        )
        np.savez_compressed(
            output / "substep_state_trace.npz",
            **{
                name: np.asarray([row[name] for row in substep_state_rows])
                for name in (*scalar_names, *array_names)
            },
            initial_actual_qpos=initial_actual_qpos,
            initial_actual_qvel=initial_actual_qvel,
            initial_actual_ctrl=initial_actual_ctrl,
            initial_reference_qpos=reference["qpos"][0],
        )
    np.savez_compressed(
        output / "endpoint_states.npz",
        endpoint=np.asarray([row["endpoint"] for row in endpoint_rows], dtype=np.int64),
        **{
            f"{name}_{field}": np.asarray([row[f"{name}_{field}"] for row in endpoint_rows])
            for name in object_specs
            for field in (
                "position_m", "quaternion_wxyz", "linear_velocity_m_s",
                "angular_velocity_rad_s",
            )
        },
    )
    write_json(output / "native_geometry_trace.json", native_rows)
    if hard_stop is not None:
        write_json(output / "hard_stop.json", hard_stop)

    pickup = {}
    stable_needed = int(cfg["strict_pickup_gate"]["stable_consecutive_control_intervals"])
    lift_needed = float(cfg["strict_pickup_gate"]["object_lift_delta_m"])
    for name, spec in object_specs.items():
        masks = [
            row[f"{name}_actual_lift_m"] >= lift_needed
            and row[f"{name}_hand_normal_impulse_ns"] > 0.0
            and row[f"{name}_floor_normal_impulse_ns"] == 0.0
            for row in endpoint_rows[1:]
        ]
        run = longest_true_run(masks)
        pickup[name] = {
            "maximum_actual_lift_m": max(row[f"{name}_actual_lift_m"] for row in endpoint_rows),
            "maximum_reference_lift_m": max(row[f"{name}_reference_lift_m"] for row in endpoint_rows),
            "lifted_hand_contact_no_floor_longest_run": run,
            "stable_pickup_pass": run >= stable_needed,
        }

    force_exposure_mismatches = []
    geometry_mismatches = []
    native_blockers = []
    total_impulse = {"brush": 0.0, "bowl": 0.0}
    mismatch_impulse = {"brush": 0.0, "bowl": 0.0}
    mismatch_displacement = {"brush": 0.0, "bowl": 0.0}
    for audited in native_rows:
        endpoint = int(audited["endpoint"])
        for name in object_specs:
            interaction = audited["interactions"][name]
            impulse = float(interaction["preceding_interval_normal_impulse_ns"])
            total_impulse[name] += impulse
            mismatch = interaction["force_native_exposure_classification"]
            if not mismatch.startswith("AGREEMENT_"):
                mismatch_row = {
                    "endpoint": endpoint, "object": name, "classification": mismatch,
                    "normal_impulse_ns": impulse,
                    "object_displacement_m": interaction["preceding_interval_object_displacement_m"],
                    "native_distance_m": (
                        None if interaction["native_minimum"] is None
                        else interaction["native_minimum"]["distance_m"]
                    ),
                    "proxy_distance_m": (
                        None if interaction["proxy_minimum"] is None
                        else interaction["proxy_minimum"]["distance_m"]
                    ),
                }
                force_exposure_mismatches.append(mismatch_row)
                mismatch_impulse[name] += impulse
                mismatch_displacement[name] += float(
                    interaction["preceding_interval_object_displacement_m"]
                )
            geometry = interaction["proxy_native_geometry_relation"]
            if geometry.startswith("PROXY_") or geometry.startswith("UNKNOWN_"):
                geometry_mismatches.append({
                    "endpoint": endpoint, "object": name, "classification": geometry,
                    "native_distance_m": (
                        None if interaction["native_minimum"] is None
                        else interaction["native_minimum"]["distance_m"]
                    ),
                    "proxy_distance_m": (
                        None if interaction["proxy_minimum"] is None
                        else interaction["proxy_minimum"]["distance_m"]
                    ),
                    "preceding_interval_normal_impulse_ns": impulse,
                    "preceding_interval_object_displacement_m": (
                        interaction["preceding_interval_object_displacement_m"]
                    ),
                })
            minimum = interaction["native_minimum"]
            if minimum is None or minimum["classification"] in {
                "PENETRATION", "UNKNOWN_INTERSECTION_DEPTH", "UNKNOWN_CONTAINMENT_OPEN_MESH",
            }:
                native_blockers.append({
                    "endpoint": endpoint, "entity": f"hand_{name}",
                    "classification": None if minimum is None else minimum["classification"],
                })
        for entity, table_row in audited["table"].items():
            limit = (
                float(cfg["native_geometry_audit"]["dynamic_object_floor_penetration_limit_m"])
                if entity in {"brush", "bowl"}
                else float(
                    cfg["native_geometry_audit"]["hand_object_and_hand_table_tolerance_m"]
                )
            )
            if float(table_row["minimum_signed_distance_m"]) < -limit:
                native_blockers.append({
                    "endpoint": endpoint, "entity": f"{entity}_table",
                    "classification": table_row["classification"],
                    "distance_m": table_row["minimum_signed_distance_m"],
                    "limit_m": limit,
                })

    tracking_failures = []
    if "strict_tracking_gate" in cfg:
        for row in endpoint_rows:
            for name in object_specs:
                score = float(row[f"{name}_tracking_score"])
                if score > float(cfg["strict_tracking_gate"]["maximum_score"]):
                    tracking_failures.append({
                        "endpoint": int(row["endpoint"]), "object": name,
                        "score": score,
                        "position_error_m": float(row[f"{name}_position_error_m"]),
                        "rotation_error_rad": float(row[f"{name}_rotation_error_rad"]),
                    })

    substep_native_summary = summarize_substep_native_hand_table(
        substep_native_rows,
        float(cfg["native_geometry_audit"]["hand_object_and_hand_table_tolerance_m"]),
    )
    controller_evidence: dict[str, Any] = {
        "mode": controller_mode,
        "gravity_feedforward": gravity_feedforward,
        "reference_velocity_feedback": reference_velocity_feedback,
        "maximum_abs_extra_generalized_force_n": max(
            (abs(row["extra_generalized_force_n"]) for row in controller_rows),
            default=0.0,
        ),
        "actuators": {},
    }
    for contract in controlled_actuators:
        rows = [
            row for row in controller_rows
            if row["actuator_name"] == contract.actuator_name
        ]
        endpoint_4 = next((
            row for row in reversed(rows) if row["outcome_endpoint"] == 4
        ), None)
        endpoint_5 = next((
            row for row in reversed(rows) if row["outcome_endpoint"] == 5
        ), None)
        controller_evidence["actuators"][contract.actuator_name] = {
            "maximum_abs_reference_tracking_error_m": max(
                (abs(row["reference_minus_actual_end_m"]) for row in rows),
                default=0.0,
            ),
            "endpoint_4_reference_minus_actual_m": (
                None if endpoint_4 is None
                else endpoint_4["reference_minus_actual_end_m"]
            ),
            "endpoint_5_reference_minus_actual_m": (
                None if endpoint_5 is None
                else endpoint_5["reference_minus_actual_end_m"]
            ),
            "endpoint_4_actual_hand_table_distance_m": (
                None if endpoint_4 is None
                else endpoint_4["actual_hand_table_distance_m"]
            ),
            "endpoint_5_actual_hand_table_distance_m": (
                None if endpoint_5 is None
                else endpoint_5["actual_hand_table_distance_m"]
            ),
        }

    strict_pass = bool(
        hard_stop is None
        and all(row["stable_pickup_pass"] for row in pickup.values())
        and not native_blockers
        and not force_exposure_mismatches
        and not geometry_mismatches
        and not tracking_failures
        and (
            substep_native_summary is None
            or substep_native_summary["whole_replay"][
                "actual_penetration_beyond_tolerance_count"
            ] == 0
        )
        and endpoint_rows[-1]["endpoint"] == outcome_endpoint
    )
    if hard_stop is not None:
        classification = "CONTROLLED_REPLAY_HARD_STOP"
    elif strict_pass:
        classification = "CONTROLLED_REPLAY_STRICT_PASS_NO_PROMOTION"
    else:
        classification = "CONTROLLED_REPLAY_FAILED_CLOSED"

    proxy_impact = {
        name: {
            "sampled_hand_object_normal_impulse_ns": total_impulse[name],
            "mismatch_normal_impulse_ns": mismatch_impulse[name],
            "mismatch_impulse_fraction": (
                mismatch_impulse[name] / total_impulse[name] if total_impulse[name] else None
            ),
            "sum_object_displacement_during_mismatch_intervals_m": mismatch_displacement[name],
            "causal_interpretation": (
                "DYNAMIC_EXPOSURE_NOT_A_TRUE_GEOMETRY_COUNTERFACTUAL"
            ),
        }
        for name in object_specs
    }
    result = {
        "schema": cfg["schema"], "classification": classification,
        "strict_gate_passed": strict_pass, "promotion_authorized": False,
        "chunk_commit_allowed": False, "reinforcement_learning_runs": 0,
        "completed_outcome_endpoint": int(endpoint_rows[-1]["endpoint"]),
        "physics_steps": physics_steps, "hard_stop": hard_stop,
        "pickup": pickup, "native_blockers": native_blockers,
        "tracking_failures": tracking_failures,
        "controller": controller_evidence,
        "substep_native_hand_table": substep_native_summary,
        "proxy_native_force_exposure_mismatches": force_exposure_mismatches,
        "proxy_native_geometry_mismatches": geometry_mismatches,
        "proxy_error_dynamic_exposure": proxy_impact,
        "maximums": {
            "contact_force_n": max(
                (row["force_norm_n"] for row in contact_rows), default=0.0,
            ),
            "proxy_penetration_m": max(
                (max(0.0, -row["distance_m"]) for row in contact_rows), default=0.0,
            ),
            **{
                f"{name}_position_error_m": max(
                    row[f"{name}_position_error_m"] for row in endpoint_rows
                ) for name in object_specs
            },
        },
        "source_artifacts": manifest["inputs"],
        "repository_head": manifest["repository_head"],
    }
    write_json(output / "results.json", result)

    endpoints = np.asarray([row["endpoint"] for row in endpoint_rows])
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for name, color in (("brush", "tab:blue"), ("bowl", "tab:orange")):
        axes[0].plot(endpoints, [row[f"{name}_actual_lift_m"] * 1000 for row in endpoint_rows],
                     color=color, label=f"{name} actual")
        axes[0].plot(endpoints, [row[f"{name}_reference_lift_m"] * 1000 for row in endpoint_rows],
                     color=color, linestyle="--", label=f"{name} reference")
        axes[1].plot(endpoints, [row[f"{name}_position_error_m"] * 1000 for row in endpoint_rows],
                     color=color, label=name)
        axes[2].plot(endpoints, [row[f"{name}_hand_normal_impulse_ns"] for row in endpoint_rows],
                     color=color, label=f"{name} hand")
        axes[2].plot(endpoints, [row[f"{name}_floor_normal_impulse_ns"] for row in endpoint_rows],
                     color=color, linestyle=":", label=f"{name} floor")
    axes[0].axhline(lift_needed * 1000, color="black", linewidth=1, linestyle=":")
    axes[0].set_ylabel("body-z lift (mm)"); axes[0].legend(ncol=2)
    axes[1].set_ylabel("reference position error (mm)"); axes[1].legend()
    axes[2].set_ylabel("interval normal impulse (N s)"); axes[2].set_xlabel("endpoint")
    axes[2].legend(ncol=2)
    figure.savefig(output / "replay_dynamics.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Brush controlled physics Replay v1", "",
        f"- Classification: `{classification}`",
        f"- Strict gate: `{'PASS' if strict_pass else 'FAIL'}`",
        f"- Completed: endpoint `0→{endpoint_rows[-1]['endpoint']}` / planned `0→{outcome_endpoint}`",
        f"- CPU MuJoCo physics steps: `{physics_steps}`",
        f"- Control target timing: `{command_timing}`",
        f"- Controller mode: `{controller_mode}`",
        f"- Gravity feedforward: `{gravity_feedforward}`",
        f"- Reference velocity feedback: `{reference_velocity_feedback}`",
        f"- Maximum applied extra generalized force: "
        f"`{controller_evidence['maximum_abs_extra_generalized_force_n']:.6f} N`",
        f"- Hard stop: `{hard_stop['reasons'] if hard_stop else 'NONE'}`",
        "- RL / MPC / MINK: `NOT RUN`",
        "- Promotion / chunk commit: `FORBIDDEN`", "",
    ]
    if controlled_actuators:
        lines += [
            "", "## Vertical wrist tracking", "",
            "| actuator | endpoint 4 below reference (mm) | "
            "endpoint 5 below reference (mm) |",
            "|---|---:|---:|",
        ]
        for name, evidence in controller_evidence["actuators"].items():
            endpoint_4_error = evidence["endpoint_4_reference_minus_actual_m"]
            endpoint_5_error = evidence["endpoint_5_reference_minus_actual_m"]
            lines.append(
                f"| {name} | "
                f"{'N/A' if endpoint_4_error is None else f'{endpoint_4_error * 1000:.6f}'} | "
                f"{'N/A' if endpoint_5_error is None else f'{endpoint_5_error * 1000:.6f}'} |"
            )
    lines += [
        "", "## Physical pickup gate", "",
        "| object | max actual/reference lift (mm) | "
        "longest lifted + hand-force + no-floor run | result |",
        "|---|---:|---:|---|",
    ]
    for name, row in pickup.items():
        lines.append(
            f"| {name} | {row['maximum_actual_lift_m']*1000:.3f} / "
            f"{row['maximum_reference_lift_m']*1000:.3f} | "
            f"{row['lifted_hand_contact_no_floor_longest_run']} | "
            f"{'PASS' if row['stable_pickup_pass'] else 'FAIL'} |"
        )
    lines += [
        "", "## Tracking gate", "",
        f"- Tracking failures: `{len(tracking_failures)}`",
        f"- First tracking failure: `{tracking_failures[0] if tracking_failures else 'NONE'}`",
        "", "## Collision/proxy evidence", "",
        f"- Native blocking observations: `{len(native_blockers)}`",
        f"- Endpoint proxy/native geometry mismatch observations: `{len(geometry_mismatches)}`",
        f"- Preceding-interval proxy-force/native-clear exposure observations: `{len(force_exposure_mismatches)}`",
        f"- Maximum runtime contact force: `{result['maximums']['contact_force_n']:.6f} N`",
        f"- Maximum runtime proxy penetration: `{result['maximums']['proxy_penetration_m']*1000:.6f} mm`",
    ]
    if substep_native_summary is not None:
        whole = substep_native_summary["whole_replay"]
        early = substep_native_summary["interval_4_to_5"]
        lines += [
            f"- Substep native hand/table actual minimum: "
            f"`{whole['actual_minimum_distance_m']*1000:.6f} mm`",
            f"- Substep native hand/table reference minimum: "
            f"`{whole['reference_minimum_distance_m']*1000:.6f} mm`",
            f"- Reference-safe / actual-penetrating substeps: "
            f"`{whole['reference_safe_actual_penetration_count']}`",
        ]
        if early is not None:
            lines.append(
                f"- Interval 4->5 both reference and actual safe: "
                f"`{early['all_reference_and_actual_safe_within_tolerance']}`"
            )
    for name, row in proxy_impact.items():
        fraction = row["mismatch_impulse_fraction"]
        lines.append(
            f"- {name}: sampled hand-object impulse `{row['sampled_hand_object_normal_impulse_ns']:.9f} N·s`; "
            f"mismatch impulse `{row['mismatch_normal_impulse_ns']:.9f} N·s` "
            f"(`{'N/A' if fraction is None else f'{fraction:.3%}'}`); mismatch-interval "
            f"object displacement sum `{row['sum_object_displacement_during_mismatch_intervals_m']*1000:.6f} mm`."
        )
    lines += [
        "", "These impulse/displacement values measure exposure along the proxy-driven Replay. "
        "They are not a counterfactual simulation with exact native-mesh contact and therefore do not prove how the true-geometry trajectory would evolve.",
        "", "A strict PASS is never inferred merely because the bounded Replay reached its final endpoint.", "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    hash_tree(output)
    print(json.dumps({
        "classification": classification, "strict_gate_passed": strict_pass,
        "completed_outcome_endpoint": endpoint_rows[-1]["endpoint"],
        "physics_steps": physics_steps, "hard_stop": hard_stop,
        "pickup": pickup, "native_blockers": len(native_blockers),
        "tracking_failures": len(tracking_failures),
        "proxy_native_geometry_mismatches": len(geometry_mismatches),
        "proxy_native_force_exposure_mismatches": len(force_exposure_mismatches),
    }, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
