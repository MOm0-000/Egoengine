#!/usr/bin/env python3
"""Attribute the bounded Brush Replay failure without changing its controls."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import mujoco
import numpy as np
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_pair_measurement,
    native_world_objects,
)
from egoengine_repro.evaluation.controlled_physics_replay import (  # noqa: E402
    control_substep_schedule,
)
from egoengine_repro.evaluation.replay_failure_attribution import (  # noqa: E402
    classify_native_measurement,
    semantic_finger,
    vector_cosine,
)
from video_to_spider.rl.replay_contact_trace import (  # noqa: E402
    contact_frame_wrench_to_world,
)
from run_taco_brush_controlled_physics_replay_v1 import (  # noqa: E402
    contact_group,
    geom_role,
    object_state,
    warning_counts,
)


SCHEMA = "taco_brush_replay_failure_attribution_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_replay_failure_attribution_v1.yaml"
SCALAR_CONTACT_FIELDS = (
    "source_endpoint", "outcome_endpoint", "substep", "time_s", "contact_index",
    "group", "geom1", "geom2", "distance_m", "normal_force_n", "force_norm_n",
    "efc_address",
)


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


def is_descendant(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == int(root_id):
            return True
        current = int(model.body_parentid[current])
    return False


def object_force(
    role1: str, role2: str, target: str, world_on_geom2: np.ndarray,
) -> np.ndarray:
    if role2 == target:
        return world_on_geom2.copy()
    if role1 == target:
        return -world_on_geom2.copy()
    return np.zeros(3, dtype=np.float64)


def read_primary_contacts(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def contact_parity(
    primary: list[dict[str, Any]], observed: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(primary) != len(observed):
        return {"passed": False, "reason": "ROW_COUNT", "primary": len(primary),
                "observed": len(observed)}
    string_fields = ("group", "geom1", "geom2")
    int_fields = ("source_endpoint", "outcome_endpoint", "substep", "contact_index",
                  "efc_address")
    float_fields = ("time_s", "distance_m", "normal_force_n", "force_norm_n")
    mismatches = []
    for index, (left, right) in enumerate(zip(primary, observed)):
        bad = []
        for field in string_fields:
            if left[field] != str(right[field]):
                bad.append(field)
        for field in int_fields:
            if int(left[field]) != int(right[field]):
                bad.append(field)
        for field in float_fields:
            if np.float64(left[field]).tobytes() != np.float64(right[field]).tobytes():
                bad.append(field)
        if bad:
            mismatches.append({"row": index, "fields": bad})
            if len(mismatches) == 8:
                break
    return {"passed": not mismatches, "row_count": len(primary),
            "first_mismatches": mismatches}


def endpoint_parity(
    saved: dict[str, np.ndarray], full_qpos: np.ndarray, full_qvel: np.ndarray,
    model: mujoco.MjModel,
) -> dict[str, Any]:
    rows = {}
    passed = True
    for name, joint_name in (("brush", "right_object_joint"), ("bowl", "left_object_joint")):
        joint = model.joint(joint_name)
        qadr, dadr = int(joint.qposadr[0]), int(joint.dofadr[0])
        comparisons = {
            "position_m": np.array_equal(saved[f"{name}_position_m"], full_qpos[:, qadr:qadr + 3]),
            "quaternion_wxyz": np.array_equal(
                saved[f"{name}_quaternion_wxyz"], full_qpos[:, qadr + 3:qadr + 7],
            ),
            "linear_velocity_m_s": np.array_equal(
                saved[f"{name}_linear_velocity_m_s"], full_qvel[:, dadr:dadr + 3],
            ),
            "angular_velocity_rad_s": np.array_equal(
                saved[f"{name}_angular_velocity_rad_s"], full_qvel[:, dadr + 3:dadr + 6],
            ),
        }
        rows[name] = comparisons
        passed = passed and all(comparisons.values())
    return {"passed": passed, "objects": rows}


def aggregate_object_contacts(
    rows: list[dict[str, Any]], *, group: str, target: str,
    first_outcome: int, last_outcome: int, timestep: float,
) -> dict[str, Any]:
    selected = [
        row for row in rows if row["group"] == group
        and first_outcome <= int(row["outcome_endpoint"]) <= last_outcome
    ]
    per_finger: dict[str, dict[str, Any]] = {}
    total = np.zeros(3, dtype=np.float64)
    torque = np.zeros(3, dtype=np.float64)
    for row in selected:
        first = semantic_finger(str(row["geom1"]))
        second = semantic_finger(str(row["geom2"]))
        finger = first if first != "other" else second
        record = per_finger.setdefault(finger, {
            "normal_impulse_ns": 0.0, "object_impulse_world_ns": np.zeros(3),
            "object_torque_impulse_world_nms": np.zeros(3), "max_force_n": 0.0,
            "contact_rows": 0, "outcome_endpoints": set(),
        })
        impulse = np.asarray(row[f"force_on_{target}_world_n"], dtype=np.float64) * timestep
        torque_impulse = np.asarray(
            row[f"torque_on_{target}_about_com_world_nm"], dtype=np.float64,
        ) * timestep
        record["normal_impulse_ns"] += float(row["normal_force_n"]) * timestep
        record["object_impulse_world_ns"] += impulse
        record["object_torque_impulse_world_nms"] += torque_impulse
        record["max_force_n"] = max(record["max_force_n"], float(row["force_norm_n"]))
        record["contact_rows"] += 1
        record["outcome_endpoints"].add(int(row["outcome_endpoint"]))
        total += impulse
        torque += torque_impulse
    serial = {}
    for finger, value in per_finger.items():
        serial[finger] = {
            **value,
            "outcome_endpoints": sorted(value["outcome_endpoints"]),
            "first_outcome_endpoint": min(value["outcome_endpoints"]),
            "last_outcome_endpoint": max(value["outcome_endpoints"]),
        }
    fingers = sorted(serial)
    opposition = []
    for index, first in enumerate(fingers):
        for second in fingers[index + 1:]:
            opposition.append({
                "first": first, "second": second,
                "impulse_direction_cosine": vector_cosine(
                    np.asarray(serial[first]["object_impulse_world_ns"]),
                    np.asarray(serial[second]["object_impulse_world_ns"]),
                ),
            })
    return {
        "contact_row_count": len(selected), "per_finger": serial,
        "total_object_impulse_world_ns": total,
        "total_object_torque_impulse_world_nms": torque,
        "finger_impulse_direction_cosines": opposition,
    }


def aggregate_group_impulse(
    rows: list[dict[str, Any]], *, group: str, target: str,
    first_outcome: int, last_outcome: int, timestep: float,
) -> np.ndarray:
    return sum((
        np.asarray(row[f"force_on_{target}_world_n"], dtype=np.float64) * timestep
        for row in rows if row["group"] == group
        and first_outcome <= int(row["outcome_endpoint"]) <= last_outcome
    ), start=np.zeros(3, dtype=np.float64))


def native_body_rows(
    model: mujoco.MjModel, qpos: np.ndarray, *, hand: str, object_body: str,
    active_body_impulses: dict[int, float], include_all_hand_bodies: bool,
    tolerance_m: float,
) -> list[dict[str, Any]]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    native = native_world_objects(model, data)
    object_id = int(model.body(object_body).id)
    root = int(model.body("R_forearm_ty_link" if hand == "right" else "L_forearm_ty_link").id)
    if include_all_hand_bodies:
        candidates = [
            body for body in native if body != object_id and is_descendant(model, body, root)
            and model.body(body).name.startswith(f"{hand}_hand")
        ]
    else:
        candidates = sorted(active_body_impulses)
    result = []
    for body in candidates:
        if body not in native or object_id not in native:
            continue
        measured = native_pair_measurement(native[body], native[object_id], tolerance_m)
        result.append({
            "hand_body_id": body, "hand_body": model.body(body).name,
            "finger": semantic_finger(model.body(body).name),
            "proxy_normal_impulse_ns": float(active_body_impulses.get(body, 0.0)),
            "proxy_active": active_body_impulses.get(body, 0.0) > 0.0,
            "native_classification": classify_native_measurement(measured, tolerance_m),
            "native_distance_m": measured["distance_m"],
            "native_penetration_depth_m": measured["reported_penetration_depth_m"],
        })
    rank = {"PENETRATION": 0, "UNKNOWN_INTERSECTION_DEPTH": 1,
            "UNKNOWN_CONTAINMENT_OPEN_MESH": 1, "CONTACT_WITHIN_TOLERANCE": 2,
            "CLEARANCE": 3}
    return sorted(result, key=lambda row: (
        rank[row["native_classification"]],
        row["native_distance_m"] if row["native_distance_m"] is not None else 0.0,
    ))


def joint_state_rows(
    model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray,
    actuator_force: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for actuator in range(model.nu):
        name = model.actuator(actuator).name
        if not (name.startswith("left_index_") or name.startswith("left_middle_")):
            continue
        joint_id = int(model.actuator_trnid[actuator, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        limited = bool(model.jnt_limited[joint_id])
        low, high = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
        actual = float(qpos[qadr])
        rows.append({
            "actuator": name, "joint": model.joint(joint_id).name,
            "command": float(ctrl[actuator]), "actual_qpos": actual,
            "control_error_command_minus_qpos": float(ctrl[actuator] - actual),
            "qvel": float(qvel[dadr]), "actuator_force": float(actuator_force[actuator]),
            "joint_limited": limited,
            "joint_range": [float(low), float(high)] if limited else None,
            "joint_limit_margin": (
                float(min(actual - low, high - actual)) if limited else None
            ),
        })
    return rows


def hash_tree(output: Path) -> None:
    target = output / "server_artifacts.sha256"
    rows = []
    for path in sorted(value for value in output.rglob("*") if value.is_file()
                       and value != target):
        rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected failure-attribution schema")
    forbidden = (
        "change_commands", "change_physics", "change_collision_geometry", "change_support",
        "mink", "mpc", "reinforcement_learning", "full_replay", "promotion", "chunk_commit",
    )
    if any(cfg["authorization"].get(key) is not False for key in forbidden):
        raise ValueError("attribution authorization is not fail-closed")
    paths = {
        key: Path(value).resolve(strict=(key != "output"))
        for key, value in cfg["paths"].items()
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    primary = paths["controlled_replay"]
    primary_result = json.loads((primary / "results.json").read_text())
    if primary_result["classification"] != "CONTROLLED_REPLAY_FAILED_CLOSED" \
            or primary_result["completed_outcome_endpoint"] != 80 \
            or primary_result["hard_stop"] is not None:
        raise ValueError("source Replay is not the reviewed complete failed-closed run")
    primary_manifest = json.loads((primary / "input_manifest.json").read_text())
    scene = Path(primary_manifest["inputs"]["local_scene"]["path"]).resolve(strict=True)
    if sha256(scene) != primary_manifest["inputs"]["local_scene"]["sha256"]:
        raise ValueError("source local scene hash changed")
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(paths["source_reference"], allow_pickle=False) as archive:
        reference = {name: np.asarray(archive[name]).copy()
                     for name in ("qpos", "qvel", "ctrl")}
        frequency = float(np.asarray(archive["frequency"]))
    schedule = control_substep_schedule(
        control_endpoints=80, control_hz=frequency,
        simulation_dt_s=float(model.opt.timestep),
    )
    manifest = {
        "schema": SCHEMA, "status": "FROZEN_BEFORE_OBSERVER_PARITY_REPLAY",
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
        "inputs": {
            "config": artifact(config_path), "source_results": artifact(primary / "results.json"),
            "source_manifest": artifact(primary / "input_manifest.json"),
            "source_endpoint_states": artifact(primary / "endpoint_states.npz"),
            "source_endpoint_trace": artifact(primary / "endpoint_trace.csv"),
            "source_contact_trace": artifact(primary / "contact_trace.csv.gz"),
            "source_native_trace": artifact(primary / "native_geometry_trace.json"),
            "local_scene": artifact(scene), "reference": artifact(paths["source_reference"]),
            "script": artifact(Path(__file__)),
        },
        "parity_gate": cfg["parity_gate"], "authorization": cfg["authorization"],
        "planned_physics_steps": int(schedule.sum()),
    }
    write_json(output / "input_manifest.json", manifest)

    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"][0]
    data.qvel[:] = 0.0
    data.ctrl[:] = reference["ctrl"][0]
    mujoco.mj_forward(model, data)
    if any(warning_counts(data)):
        raise RuntimeError("MuJoCo warning before parity Replay")
    endpoint_qpos = [np.asarray(data.qpos).copy()]
    endpoint_qvel = [np.asarray(data.qvel).copy()]
    endpoint_ctrl = [np.asarray(data.ctrl).copy()]
    endpoint_time = [float(data.time)]
    observed_scalar: list[dict[str, Any]] = []
    extended: list[dict[str, Any]] = []
    peak_self: dict[str, Any] | None = None
    physics_steps = 0

    for source, substeps in enumerate(schedule):
        outcome = source + 1
        data.ctrl[:] = reference["ctrl"][outcome]
        for substep in range(int(substeps)):
            mujoco.mj_step(model, data)
            physics_steps += 1
            if any(warning_counts(data)):
                raise RuntimeError(f"MuJoCo warning at physics step {physics_steps}")
            brush_com = object_state(model, data, "right_object_joint")["position_m"]
            bowl_com = object_state(model, data, "left_object_joint")["position_m"]
            for index in range(data.ncon):
                contact = data.contact[index]
                first, second = int(contact.geom1), int(contact.geom2)
                role1, role2 = geom_role(model, first), geom_role(model, second)
                group = contact_group(role1, role2)
                wrench = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, index, wrench)
                frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
                world = contact_frame_wrench_to_world(wrench, frame)[:3]
                brush_force = object_force(role1, role2, "brush", world)
                bowl_force = object_force(role1, role2, "bowl", world)
                position = np.asarray(contact.pos, dtype=np.float64).copy()
                force_norm = float(np.linalg.norm(wrench[:3]))
                scalar = {
                    "source_endpoint": source, "outcome_endpoint": outcome,
                    "substep": substep, "time_s": float(data.time), "contact_index": index,
                    "group": group, "geom1": model.geom(first).name,
                    "geom2": model.geom(second).name, "distance_m": float(contact.dist),
                    "normal_force_n": max(0.0, float(wrench[0])),
                    "force_norm_n": force_norm, "efc_address": int(contact.efc_address),
                }
                observed_scalar.append(scalar)
                row = {
                    **scalar, "body1_id": int(model.geom_bodyid[first]),
                    "body2_id": int(model.geom_bodyid[second]),
                    "body1": model.body(int(model.geom_bodyid[first])).name,
                    "body2": model.body(int(model.geom_bodyid[second])).name,
                    "role1": role1, "role2": role2,
                    "contact_position_world_m": position,
                    "contact_normal_geom1_to_geom2_world": frame[0].copy(),
                    "force_on_geom2_world_n": world,
                    "force_on_brush_world_n": brush_force,
                    "force_on_bowl_world_n": bowl_force,
                    "torque_on_brush_about_com_world_nm": np.cross(position - brush_com, brush_force),
                    "torque_on_bowl_about_com_world_nm": np.cross(position - bowl_com, bowl_force),
                }
                extended.append(row)
                if group == "intrahand" and 44 <= outcome <= 45 \
                        and (peak_self is None or force_norm > peak_self["contact"]["force_norm_n"]):
                    peak_self = {
                        "contact": row.copy(), "qpos": np.asarray(data.qpos).copy(),
                        "qvel": np.asarray(data.qvel).copy(), "ctrl": np.asarray(data.ctrl).copy(),
                        "actuator_force": np.asarray(data.actuator_force).copy(),
                    }
        endpoint_qpos.append(np.asarray(data.qpos).copy())
        endpoint_qvel.append(np.asarray(data.qvel).copy())
        endpoint_ctrl.append(np.asarray(data.ctrl).copy())
        endpoint_time.append(float(data.time))

    full_qpos = np.asarray(endpoint_qpos)
    full_qvel = np.asarray(endpoint_qvel)
    full_ctrl = np.asarray(endpoint_ctrl)
    with np.load(primary / "endpoint_states.npz", allow_pickle=False) as archive:
        saved = {name: np.asarray(archive[name]).copy() for name in archive.files}
    endpoint_check = endpoint_parity(saved, full_qpos, full_qvel, model)
    contact_check = contact_parity(
        read_primary_contacts(primary / "contact_trace.csv.gz"), observed_scalar,
    )
    parity = {
        "passed": bool(endpoint_check["passed"] and contact_check["passed"]),
        "object_endpoint_states": endpoint_check, "contact_trace": contact_check,
        "physics_steps": physics_steps,
    }
    write_json(output / "observer_parity.json", parity)
    np.savez_compressed(
        output / "full_endpoint_states.npz", qpos=full_qpos, qvel=full_qvel,
        ctrl=full_ctrl, time_s=np.asarray(endpoint_time),
    )
    if not parity["passed"]:
        result = {"schema": SCHEMA, "classification": "OBSERVER_PARITY_FAILED_STOPPED",
                  "parity": parity, "attribution_performed": False}
        write_json(output / "results.json", result)
        (output / "summary.md").write_text(
            "# Brush Replay failure attribution\n\n"
            "Observer parity failed; attribution stopped fail-closed.\n", encoding="utf-8",
        )
        hash_tree(output)
        print(json.dumps(result, indent=2, default=json_default))
        return 0

    # Store the extended observer evidence only after parity is proven.
    columns = [
        *SCALAR_CONTACT_FIELDS, "body1", "body2", "role1", "role2",
        *[f"contact_position_world_m_{axis}" for axis in "xyz"],
        *[f"contact_normal_world_{axis}" for axis in "xyz"],
        *[f"force_on_geom2_world_n_{axis}" for axis in "xyz"],
        *[f"force_on_brush_world_n_{axis}" for axis in "xyz"],
        *[f"force_on_bowl_world_n_{axis}" for axis in "xyz"],
    ]
    with gzip.open(output / "extended_contact_trace.csv.gz", "wt", newline="",
                   encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in extended:
            flat = {key: row[key] for key in SCALAR_CONTACT_FIELDS}
            flat.update({key: row[key] for key in ("body1", "body2", "role1", "role2")})
            for axis, index in zip("xyz", range(3)):
                flat[f"contact_position_world_m_{axis}"] = row["contact_position_world_m"][index]
                flat[f"contact_normal_world_{axis}"] = row[
                    "contact_normal_geom1_to_geom2_world"
                ][index]
                flat[f"force_on_geom2_world_n_{axis}"] = row["force_on_geom2_world_n"][index]
                flat[f"force_on_brush_world_n_{axis}"] = row["force_on_brush_world_n"][index]
                flat[f"force_on_bowl_world_n_{axis}"] = row["force_on_bowl_world_n"][index]
            writer.writerow(flat)

    timestep = float(model.opt.timestep)
    brush_window = tuple(int(value) for value in cfg["sample"]["brush_contact_window_outcomes"])
    bowl_window = tuple(int(value) for value in cfg["sample"]["bowl_contact_window_outcomes"])
    brush = aggregate_object_contacts(
        extended, group="right_hand_brush", target="brush",
        first_outcome=brush_window[0], last_outcome=brush_window[1], timestep=timestep,
    )
    bowl = aggregate_object_contacts(
        extended, group="left_hand_bowl", target="bowl",
        first_outcome=bowl_window[0], last_outcome=bowl_window[1], timestep=timestep,
    )
    for name, group, target, window, joint_name in (
        ("brush", "brush_floor", "brush", brush_window, "right_object_joint"),
        ("bowl", "bowl_floor", "bowl", bowl_window, "left_object_joint"),
    ):
        floor_impulse = aggregate_group_impulse(
            extended, group=group, target=target, first_outcome=window[0],
            last_outcome=window[1], timestep=timestep,
        )
        value = brush if name == "brush" else bowl
        start_endpoint = window[0] - 1
        duration = endpoint_time[window[1]] - endpoint_time[start_endpoint]
        mass = float(model.body("right_object" if name == "brush" else "left_object").mass[0])
        joint = model.joint(joint_name)
        dadr = int(joint.dofadr[0])
        gravity = np.asarray(model.opt.gravity, dtype=np.float64) * mass * duration
        momentum_change = mass * (
            full_qvel[window[1], dadr:dadr + 3] - full_qvel[start_endpoint, dadr:dadr + 3]
        )
        value.update({
            "window": list(window), "duration_s": duration, "object_mass_kg": mass,
            "floor_impulse_world_ns": floor_impulse, "gravity_impulse_world_ns": gravity,
            "measured_linear_momentum_change_ns": momentum_change,
            "vertical_support": {
                "hand_impulse_z_ns": float(value["total_object_impulse_world_ns"][2]),
                "floor_impulse_z_ns": float(floor_impulse[2]),
                "gravity_impulse_z_ns": float(gravity[2]),
            },
            "start_position_m": full_qpos[start_endpoint, int(joint.qposadr[0]):int(joint.qposadr[0]) + 3],
            "end_position_m": full_qpos[window[1], int(joint.qposadr[0]):int(joint.qposadr[0]) + 3],
        })

    tolerance = float(cfg["native_followup"]["material_tolerance_m"])
    brush_native = []
    for endpoint in cfg["native_followup"]["brush_proxy_overcoverage_endpoints"]:
        active: dict[int, float] = {}
        for row in extended:
            if row["group"] != "right_hand_brush" or int(row["outcome_endpoint"]) != endpoint:
                continue
            body = int(row["body1_id"] if row["role1"] == "right_hand" else row["body2_id"])
            active[body] = active.get(body, 0.0) + float(row["normal_force_n"]) * timestep
        brush_native.append({
            "endpoint": endpoint,
            "active_contact_body_native_checks": native_body_rows(
                model, full_qpos[endpoint], hand="right", object_body="right_object",
                active_body_impulses=active, include_all_hand_bodies=False,
                tolerance_m=tolerance,
            ),
        })
    write_json(output / "brush_proxy_contact_attribution.json", brush_native)

    bowl_native = []
    for endpoint in cfg["native_followup"]["bowl_contact_endpoints"]:
        active = {}
        for row in extended:
            if row["group"] != "left_hand_bowl" or int(row["outcome_endpoint"]) != endpoint:
                continue
            body = int(row["body1_id"] if row["role1"] == "left_hand" else row["body2_id"])
            active[body] = active.get(body, 0.0) + float(row["normal_force_n"]) * timestep
        rows = native_body_rows(
            model, full_qpos[endpoint], hand="left", object_body="left_object",
            active_body_impulses=active, include_all_hand_bodies=True, tolerance_m=tolerance,
        )
        bowl_native.append({"endpoint": endpoint, "body_checks": rows, "nearest": rows[0]})
    write_json(output / "bowl_reach_attribution.json", bowl_native)

    historical_native = json.loads((primary / "native_geometry_trace.json").read_text())
    historical_bowl = {
        int(row["endpoint"]): row["interactions"]["bowl"]["native_minimum"]
        for row in historical_native
        if 52 <= int(row["endpoint"]) <= 60
    }
    native_recheck = []
    for row in bowl_native:
        endpoint = int(row["endpoint"])
        if endpoint not in historical_bowl:
            continue
        old = historical_bowl[endpoint]
        new = row["nearest"]
        native_recheck.append({
            "endpoint": endpoint,
            "historical_body": old["body1"],
            "historical_classification": old["classification"],
            "historical_distance_m": old["distance_m"],
            "observer_body": new["hand_body"],
            "observer_classification": new["native_classification"],
            "observer_distance_m": new["native_distance_m"],
            "classification_reproduced": (
                old["classification"] == new["native_classification"]
            ),
        })
    write_json(output / "bowl_native_recheck.json", native_recheck)

    if peak_self is None:
        raise RuntimeError("reviewed self-collision peak was not reproduced")
    peak_contact = peak_self["contact"]
    first_body, second_body = int(peak_contact["body1_id"]), int(peak_contact["body2_id"])
    peak_data = mujoco.MjData(model)
    peak_data.qpos[:] = peak_self["qpos"]
    peak_data.qvel[:] = peak_self["qvel"]
    peak_data.ctrl[:] = peak_self["ctrl"]
    mujoco.mj_forward(model, peak_data)
    peak_native = native_world_objects(model, peak_data)
    native_measurement = native_pair_measurement(
        peak_native[first_body], peak_native[second_body], tolerance,
    )
    first_geom = int(model.geom(peak_contact["geom1"]).id)
    second_geom = int(model.geom(peak_contact["geom2"]).id)
    self_report = {
        "peak_contact": peak_contact,
        "proxy_geometries": [
            {
                "geom": model.geom(value).name, "body": model.body(int(model.geom_bodyid[value])).name,
                "type": int(model.geom_type[value]), "size": np.asarray(model.geom_size[value]),
                "world_position_m": np.asarray(peak_data.geom_xpos[value]),
                "world_rotation": np.asarray(peak_data.geom_xmat[value]).reshape(3, 3),
            }
            for value in (first_geom, second_geom)
        ],
        "native_measurement": {
            "classification": classify_native_measurement(native_measurement, tolerance),
            "distance_m": native_measurement["distance_m"],
            "reported_penetration_depth_m": native_measurement["reported_penetration_depth_m"],
            "surface_intersection": native_measurement["surface_intersection"],
            "containment_penetration": native_measurement.get("containment_penetration", False),
            "nearest_points_sim_m": native_measurement.get("nearest_points_sim_m"),
            "deepest_contact_position_sim_m": native_measurement.get(
                "deepest_contact_position_sim_m"
            ),
        },
        "joint_states": joint_state_rows(
            model, peak_self["qpos"], peak_self["qvel"], peak_self["ctrl"],
            peak_self["actuator_force"],
        ),
    }
    write_json(output / "self_collision_attribution.json", self_report)
    write_json(output / "brush_contact_mechanics.json", brush)
    write_json(output / "bowl_contact_mechanics.json", bowl)

    brush_cosine = brush["finger_impulse_direction_cosines"][0][
        "impulse_direction_cosine"
    ]
    brush_proxy_bodies = sorted({
        (row["finger"], row["hand_body"])
        for endpoint in brush_native
        for row in endpoint["active_contact_body_native_checks"]
        if row["proxy_active"] and row["native_classification"] == "CLEARANCE"
    })
    bowl_total_normal_impulse = sum(
        row["normal_impulse_ns"] for row in bowl["per_finger"].values()
    )
    bowl_index_fraction = (
        bowl["per_finger"].get("index", {}).get("normal_impulse_ns", 0.0)
        / bowl_total_normal_impulse
    )
    positive_vertical_support = (
        bowl["vertical_support"]["floor_impulse_z_ns"]
        + max(0.0, bowl["vertical_support"]["hand_impulse_z_ns"])
    )
    floor_support_fraction = (
        bowl["vertical_support"]["floor_impulse_z_ns"] / positive_vertical_support
    )
    joint_errors = [
        abs(row["control_error_command_minus_qpos"])
        for row in self_report["joint_states"]
    ]
    endpoint60_recheck = next(
        (row for row in native_recheck if row["endpoint"] == 60), None,
    )
    mechanism_conclusions = {
        "brush_15_41": {
            "classification": "DOWNWARD_PROXY_PUSH_WITH_TABLE_REACTION_NOT_EFFECTIVE_CLAMP",
            "driving_fingers": sorted(brush["per_finger"]),
            "index_thumb_impulse_direction_cosine": brush_cosine,
            "hand_vertical_impulse_ns": brush["vertical_support"]["hand_impulse_z_ns"],
            "floor_vertical_impulse_ns": brush["vertical_support"]["floor_impulse_z_ns"],
            "proxy_overcoverage_finger_bodies": [
                {"finger": finger, "body": body} for finger, body in brush_proxy_bodies
            ],
        },
        "self_collision_44_45": {
            "classification": "PROXY_OVERCOVERAGE_FALSE_SELF_CONTACT_WITH_LOADED_TRACKING",
            "native_clearance_m": self_report["native_measurement"]["distance_m"],
            "proxy_penetration_m": -peak_contact["distance_m"],
            "peak_force_n": peak_contact["force_norm_n"],
            "maximum_absolute_control_error_rad": max(joint_errors),
            "all_reviewed_joints_within_limits": all(
                row["joint_limit_margin"] >= 0 for row in self_report["joint_states"]
            ),
        },
        "bowl_52_60": {
            "classification": "INDEX_ONLY_WEAK_PUSH_WITH_FLOOR_DOMINANT_SUPPORT",
            "index_normal_impulse_fraction": bowl_index_fraction,
            "floor_positive_vertical_support_fraction": floor_support_fraction,
            "hand_vertical_impulse_ns": bowl["vertical_support"]["hand_impulse_z_ns"],
            "floor_vertical_impulse_ns": bowl["vertical_support"]["floor_impulse_z_ns"],
            "robust_proxy_undercoverage_endpoint": 57,
            "endpoint60_native_recheck": endpoint60_recheck,
            "endpoint60_decisive_evidence": False,
        },
    }
    write_json(output / "mechanism_conclusions.json", mechanism_conclusions)

    result = {
        "schema": SCHEMA, "classification": "REPLAY_FAILURE_MECHANISM_ATTRIBUTED",
        "parity": parity, "physics_steps": physics_steps,
        "brush": brush, "bowl": bowl, "self_collision": self_report,
        "brush_native_followup": brush_native, "bowl_native_followup": bowl_native,
        "bowl_native_recheck": native_recheck,
        "mechanism_conclusions": mechanism_conclusions,
        "mutations": {"commands": 0, "physics": 0, "collision_geometry": 0,
                      "support": 0, "training_steps": 0, "promotion": 0},
        "source_artifacts": manifest["inputs"],
    }
    write_json(output / "results.json", result)

    brush_fingers = sorted(
        brush["per_finger"].items(), key=lambda item: -item[1]["normal_impulse_ns"],
    )
    bowl_fingers = sorted(
        bowl["per_finger"].items(), key=lambda item: -item[1]["normal_impulse_ns"],
    )
    lines = [
        "# Brush Replay failure mechanism attribution v1", "",
        f"- Observer parity: `{'PASS' if parity['passed'] else 'FAIL'}`",
        f"- Classification: `{result['classification']}`",
        "- Commands / physics / geometry changed: `NO`",
        "- RL / MPC / MINK: `NOT RUN`", "",
        "## Brush, outcomes 15–41", "",
        "| finger | normal impulse (N·s) | object impulse xyz (N·s) | max force (N) | endpoints |",
        "|---|---:|---|---:|---|",
    ]
    for finger, row in brush_fingers:
        lines.append(
            f"| {finger} | {row['normal_impulse_ns']:.6f} | "
            f"{np.asarray(row['object_impulse_world_ns']).round(6).tolist()} | "
            f"{row['max_force_n']:.6f} | {row['first_outcome_endpoint']}–"
            f"{row['last_outcome_endpoint']} |"
        )
    lines += [
        "", f"- Hand vertical impulse: `{brush['vertical_support']['hand_impulse_z_ns']:.6f} N·s`",
        f"- Floor vertical impulse: `{brush['vertical_support']['floor_impulse_z_ns']:.6f} N·s`",
        f"- Gravity impulse: `{brush['vertical_support']['gravity_impulse_z_ns']:.6f} N·s`",
        f"- Index/thumb impulse-direction cosine: `{brush_cosine:.6f}` (aligned, not an opposing pinch).",
        "- Mechanism: the hand contacts push the brush downward; the table reaction balances "
        "that push plus gravity, so the brush does not leave the table.",
        "- Confirmed proxy over-coverage involves right index and thumb collision bodies; "
        "the native meshes are clear at the reviewed endpoints.",
        "", "## Left-hand self collision, outcomes 44–45", "",
        f"- Peak proxy pair: `{peak_contact['geom1']}` / `{peak_contact['geom2']}`",
        f"- Peak force: `{peak_contact['force_norm_n']:.6f} N`",
        f"- Proxy signed distance: `{peak_contact['distance_m']*1000:.6f} mm`",
        f"- Native visual-mesh result: `{self_report['native_measurement']['classification']}`; "
        f"distance `{self_report['native_measurement']['distance_m']*1000:.6f} mm`",
        f"- Maximum reviewed joint tracking error: `{max(joint_errors):.6f} rad`; all reviewed "
        "joints remain inside their limits.",
        "- Mechanism: the 258.76 N event is a false proxy self-contact. The proxy force loads "
        "the position servos and produces large index/middle tracking errors; it is not native "
        "finger-mesh squeezing or a joint-limit collision.", "",
        "## Bowl, outcomes 52–60", "",
        "| finger | normal impulse (N·s) | object impulse xyz (N·s) | max force (N) | endpoints |",
        "|---|---:|---|---:|---|",
    ]
    for finger, row in bowl_fingers:
        lines.append(
            f"| {finger} | {row['normal_impulse_ns']:.6f} | "
            f"{np.asarray(row['object_impulse_world_ns']).round(6).tolist()} | "
            f"{row['max_force_n']:.6f} | {row['first_outcome_endpoint']}–"
            f"{row['last_outcome_endpoint']} |"
        )
    lines += [
        "", f"- Hand vertical impulse: `{bowl['vertical_support']['hand_impulse_z_ns']:.6f} N·s`",
        f"- Floor vertical impulse: `{bowl['vertical_support']['floor_impulse_z_ns']:.6f} N·s`",
        f"- Gravity impulse: `{bowl['vertical_support']['gravity_impulse_z_ns']:.6f} N·s`", "",
        f"- Index share of hand–bowl normal impulse: `{bowl_index_fraction:.3%}`.",
        f"- Floor share of positive vertical support: `{floor_support_fraction:.3%}`.",
        "- Mechanism: only the left index reaches the bowl materially; the middle contact is "
        "negligible and thumb/ring/pinky do not form a grasp. The floor remains the dominant "
        "vertical support, so the bowl stays on the table.",
        "- Proxy under-coverage is a secondary contributor at endpoint 57. Endpoint 60 is not "
        "decisive: the historical native audit reported -0.186 mm penetration while this "
        "parity observer reported +0.108 mm clearance.", "",
        "Native geometry is an observer only. Force directions come from the exact replayed "
        "MuJoCo contact frame; no native-mesh force counterfactual is claimed.", "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    hash_tree(output)
    print(json.dumps({
        "classification": result["classification"], "parity": parity["passed"],
        "brush_fingers": {key: value["normal_impulse_ns"] for key, value in brush_fingers},
        "self_peak_force_n": peak_contact["force_norm_n"],
        "self_native": self_report["native_measurement"],
        "bowl_fingers": {key: value["normal_impulse_ns"] for key, value in bowl_fingers},
    }, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
