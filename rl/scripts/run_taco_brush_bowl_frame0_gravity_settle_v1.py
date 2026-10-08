#!/usr/bin/env python3
"""Isolated frame-0 bowl gravity-settling audit on the Issue #14 table."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
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

from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    build_native_support_geoms,
    geom_world_vertices,
)


SCHEMA = "taco_brush_bowl_frame0_gravity_settle_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_bowl_frame0_gravity_settle_v1.yaml"


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def quaternion_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    q1 = np.asarray(first, dtype=np.float64)
    q2 = np.asarray(second, dtype=np.float64)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    return float(2.0 * np.arccos(np.clip(abs(float(q1 @ q2)), 0.0, 1.0)))


def build_isolated_scene(source_scene: Path, destination: Path, table_z_m: float) -> None:
    source_root = ET.parse(source_scene).getroot()
    isolated = ET.Element("mujoco", {"model": "taco_bowl_146_frame0_isolated"})

    compiler = source_root.find("compiler")
    if compiler is None:
        raise ValueError("source scene has no compiler element")
    compiler_copy = copy.deepcopy(compiler)
    meshdir = compiler.attrib.get("meshdir", "")
    compiler_copy.set("meshdir", str((source_scene.parent / meshdir).resolve(strict=True)))
    isolated.append(compiler_copy)

    default = source_root.find("default")
    if default is None:
        raise ValueError("source scene has no default element")
    isolated.append(copy.deepcopy(default))

    source_asset = source_root.find("asset")
    if source_asset is None:
        raise ValueError("source scene has no asset element")
    asset = ET.SubElement(isolated, "asset")
    for child in source_asset:
        name = child.attrib.get("name", "")
        if ((child.tag == "mesh" and name.startswith("left_"))
                or (child.tag in {"texture", "material"} and name == "right_groundplane")):
            asset.append(copy.deepcopy(child))

    source_world = source_root.find("worldbody")
    if source_world is None:
        raise ValueError("source scene has no worldbody")
    source_floor = source_world.find("geom[@name='floor']")
    source_bowl = source_world.find("body[@name='left_object']")
    if source_floor is None or source_bowl is None:
        raise ValueError("source floor or bowl body is missing")
    world = ET.SubElement(isolated, "worldbody")
    floor = copy.deepcopy(source_floor)
    floor.set("pos", f"0 0 {table_z_m:.17g}")
    world.append(floor)
    world.append(copy.deepcopy(source_bowl))

    source_contact = source_root.find("contact")
    if source_contact is None:
        raise ValueError("source scene has no explicit contact pairs")
    contact = ET.SubElement(isolated, "contact")
    for pair in source_contact.findall("pair"):
        names = (pair.attrib.get("geom1", ""), pair.attrib.get("geom2", ""))
        if "floor" in names and any(name.startswith("left_object_") for name in names):
            contact.append(copy.deepcopy(pair))
    if len(contact) != 32:
        raise ValueError(f"expected 32 bowl-floor pairs, found {len(contact)}")

    ET.indent(isolated, space="  ")
    ET.ElementTree(isolated).write(destination, encoding="utf-8", xml_declaration=True)


def _mesh_arrays(model: mujoco.MjModel, geom_name: str) -> tuple[np.ndarray, np.ndarray]:
    geom_id = model.geom(geom_name).id
    mesh_id = int(model.geom_dataid[geom_id])
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    return (
        np.asarray(model.mesh_vert[vert_start:vert_start + vert_count]).copy(),
        np.asarray(model.mesh_face[face_start:face_start + face_count]).copy(),
    )


def verify_isolation_fidelity(
    source: mujoco.MjModel, isolated: mujoco.MjModel, table_z_m: float,
) -> dict[str, Any]:
    if (isolated.nbody, isolated.njnt, isolated.nq, isolated.nv, isolated.nu,
            isolated.npair) != (2, 1, 7, 6, 0, 32):
        raise ValueError("isolated model contains unexpected dynamic content")
    if isolated.geom("floor").pos[2] != table_z_m:
        raise ValueError("candidate table height was not compiled exactly")

    option_checks = {
        "timestep": float(source.opt.timestep),
        "gravity": np.asarray(source.opt.gravity).tolist(),
        "integrator": int(source.opt.integrator),
        "solver": int(source.opt.solver),
        "iterations": int(source.opt.iterations),
        "tolerance": float(source.opt.tolerance),
        "cone": int(source.opt.cone),
        "impratio": float(source.opt.impratio),
    }
    for field in ("timestep", "gravity", "integrator", "solver", "iterations",
                  "tolerance", "cone", "impratio"):
        if not np.array_equal(np.asarray(getattr(source.opt, field)),
                              np.asarray(getattr(isolated.opt, field))):
            raise ValueError(f"isolated option differs from source: {field}")

    source_body = source.body("left_object").id
    isolated_body = isolated.body("left_object").id
    body_fields = ("body_mass", "body_inertia", "body_ipos", "body_iquat")
    for field in body_fields:
        if not np.array_equal(getattr(source, field)[source_body],
                              getattr(isolated, field)[isolated_body]):
            raise ValueError(f"isolated bowl body differs from source: {field}")

    geom_fields = (
        "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction",
        "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "geom_condim",
        "geom_priority",
    )
    mesh_hashes = {}
    for index in range(32):
        name = f"left_object_{index}"
        source_id = source.geom(name).id
        isolated_id = isolated.geom(name).id
        for field in geom_fields:
            if not np.array_equal(getattr(source, field)[source_id],
                                  getattr(isolated, field)[isolated_id]):
                raise ValueError(f"isolated geom differs from source: {name}/{field}")
        source_mesh = _mesh_arrays(source, name)
        isolated_mesh = _mesh_arrays(isolated, name)
        if not all(np.array_equal(a, b) for a, b in zip(source_mesh, isolated_mesh)):
            raise ValueError(f"isolated mesh differs from source: {name}")
        mesh_hashes[name] = hashlib.sha256(
            source_mesh[0].tobytes() + source_mesh[1].tobytes()).hexdigest()

    pair_fields = ("pair_dim", "pair_friction", "pair_gap", "pair_margin",
                   "pair_solimp", "pair_solref", "pair_solreffriction")
    for index in range(41, 73):
        name = f"floor_{index}"
        source_id = source.pair(name).id
        isolated_id = isolated.pair(name).id
        for field in pair_fields:
            if not np.array_equal(getattr(source, field)[source_id],
                                  getattr(isolated, field)[isolated_id]):
                raise ValueError(f"isolated pair differs from source: {name}/{field}")

    return {
        "isolated_counts": {
            "nbody": int(isolated.nbody), "njnt": int(isolated.njnt),
            "nq": int(isolated.nq), "nv": int(isolated.nv),
            "nu": int(isolated.nu), "ngeom": int(isolated.ngeom),
            "npair": int(isolated.npair),
        },
        "source_options": option_checks,
        "bowl_mass_kg": float(source.body_mass[source_body]),
        "bowl_principal_inertia_kg_m2": source.body_inertia[source_body].tolist(),
        "bowl_inertial_position_m": source.body_ipos[source_body].tolist(),
        "bowl_inertial_quaternion": source.body_iquat[source_body].tolist(),
        "collision_mesh_hashes": mesh_hashes,
        "collision_geometry_count": 32,
        "explicit_bowl_floor_pair_count": 32,
        "only_dynamic_body": isolated.body("left_object").name,
    }


def extract_initial_state(
    source_model: mujoco.MjModel, reference_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(reference_path, allow_pickle=False) as archive:
        source_qpos = np.asarray(archive["qpos"][0], dtype=np.float64)
        source_qvel = np.asarray(archive["qvel"][0], dtype=np.float64)
    joint = source_model.joint("left_object_joint")
    qpos_address = int(joint.qposadr[0])
    dof_address = int(joint.dofadr[0])
    qpos = source_qpos[qpos_address:qpos_address + 7].copy()
    qvel = source_qvel[dof_address:dof_address + 6].copy()
    if qpos.shape != (7,) or qvel.shape != (6,) or not np.isfinite(qpos).all() \
            or not np.isfinite(qvel).all():
        raise ValueError("invalid saved frame-0 bowl state")
    return qpos, qvel, {
        "source_qpos_address": qpos_address,
        "source_dof_address": dof_address,
        "qpos": qpos.tolist(),
        "recorded_qvel": qvel.tolist(),
        "recorded_linear_speed_m_s": float(np.linalg.norm(qvel[:3])),
        "recorded_angular_speed_rad_s": float(np.linalg.norm(qvel[3:])),
    }


def verify_initial_pose_transport(
    source_model: mujoco.MjModel, isolated_model: mujoco.MjModel,
    reference_path: Path, isolated_qpos: np.ndarray,
) -> float:
    with np.load(reference_path, allow_pickle=False) as archive:
        source_qpos = np.asarray(archive["qpos"][0], dtype=np.float64)
        source_qvel = np.asarray(archive["qvel"][0], dtype=np.float64)
    source_data = mujoco.MjData(source_model)
    source_data.qpos[:] = source_qpos
    source_data.qvel[:] = source_qvel
    mujoco.mj_forward(source_model, source_data)
    isolated_data = mujoco.MjData(isolated_model)
    isolated_data.qpos[:] = isolated_qpos
    mujoco.mj_forward(isolated_model, isolated_data)
    source_body = source_model.body("left_object").id
    isolated_body = isolated_model.body("left_object").id
    error = max(
        float(np.max(np.abs(source_data.xpos[source_body] - isolated_data.xpos[isolated_body]))),
        float(np.max(np.abs(source_data.xquat[source_body] - isolated_data.xquat[isolated_body]))),
    )
    if error != 0.0:
        raise ValueError(f"isolated bowl pose transport is not bitwise exact: {error}")
    return error


def record_sample(
    model: mujoco.MjModel, data: mujoco.MjData, native_geom: Any,
    collision_geom_ids: list[int], floor_id: int, body_id: int,
    initial_position: np.ndarray, initial_quaternion: np.ndarray,
) -> dict[str, Any]:
    collision_distances = np.asarray([
        mujoco.mj_geomDistance(model, data, geom_id, floor_id, 1.0, None)
        for geom_id in collision_geom_ids
    ], dtype=np.float64)
    native_vertices = geom_world_vertices(data, native_geom, full=True)
    native_clearance = float(np.min(native_vertices[:, 2] - model.geom_pos[floor_id, 2]))

    normal_forces = []
    contact_distances = []
    collision_set = set(collision_geom_ids)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor_id not in pair or not pair.intersection(collision_set):
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        normal_forces.append(max(0.0, float(wrench[0])))
        contact_distances.append(float(contact.dist))

    position = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    quaternion = np.asarray(data.xquat[body_id], dtype=np.float64).copy()
    velocity = np.asarray(data.qvel[:6], dtype=np.float64).copy()
    minimum_collision_distance = float(np.min(collision_distances))
    return {
        "time_s": float(data.time),
        "position_x_m": float(position[0]),
        "position_y_m": float(position[1]),
        "position_z_m": float(position[2]),
        "quaternion_w": float(quaternion[0]),
        "quaternion_x": float(quaternion[1]),
        "quaternion_y": float(quaternion[2]),
        "quaternion_z": float(quaternion[3]),
        "linear_velocity_x_m_s": float(velocity[0]),
        "linear_velocity_y_m_s": float(velocity[1]),
        "linear_velocity_z_m_s": float(velocity[2]),
        "angular_velocity_x_rad_s": float(velocity[3]),
        "angular_velocity_y_rad_s": float(velocity[4]),
        "angular_velocity_z_rad_s": float(velocity[5]),
        "linear_speed_m_s": float(np.linalg.norm(velocity[:3])),
        "angular_speed_rad_s": float(np.linalg.norm(velocity[3:])),
        "horizontal_displacement_m": float(np.linalg.norm(position[:2] - initial_position[:2])),
        "orientation_change_rad": quaternion_distance_rad(initial_quaternion, quaternion),
        "native_visual_clearance_m": native_clearance,
        "collision_minimum_distance_m": minimum_collision_distance,
        "collision_penetration_depth_m": max(0.0, -minimum_collision_distance),
        "contact_count": len(normal_forces),
        "contact_minimum_distance_m": (
            None if not contact_distances else float(min(contact_distances))
        ),
        "normal_force_sum_N": float(sum(normal_forces)),
        "normal_force_max_N": float(max(normal_forces, default=0.0)),
    }


def simulation_warnings(data: mujoco.MjData) -> list[dict[str, int]]:
    rows = []
    for warning_type in range(mujoco.mjtWarning.mjNWARNING):
        warning = data.warning[warning_type]
        if int(warning.number):
            rows.append({"type": warning_type, "number": int(warning.number),
                         "lastinfo": int(warning.lastinfo)})
    return rows


def run_variant(
    model: mujoco.MjModel, initial_qpos: np.ndarray, initial_qvel: np.ndarray,
    duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    body_id = model.body("left_object").id
    floor_id = model.geom("floor").id
    collision_geom_ids = [model.geom(f"left_object_{index}").id for index in range(32)]
    visual_id = model.geom("left_object_visual").id
    native_geom = build_native_support_geoms(model, mujoco, [visual_id])[0]
    initial_position = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    initial_quaternion = np.asarray(data.xquat[body_id], dtype=np.float64).copy()
    steps_float = duration_s / model.opt.timestep
    steps = int(round(steps_float))
    if not np.isclose(steps_float, steps, atol=1e-12):
        raise ValueError("duration is not an exact number of source-scene time steps")
    rows = [record_sample(
        model, data, native_geom, collision_geom_ids, floor_id, body_id,
        initial_position, initial_quaternion,
    )]
    for _ in range(steps):
        mujoco.mj_step(model, data)
        rows.append(record_sample(
            model, data, native_geom, collision_geom_ids, floor_id, body_id,
            initial_position, initial_quaternion,
        ))
    for row in rows:
        if any(not np.isfinite(float(value)) for value in row.values() if value is not None):
            raise ValueError("nonfinite simulation time series")
    return rows, simulation_warnings(data)


def assess_variant(
    rows: list[dict[str, Any]], criteria: dict[str, Any], warnings: list[dict[str, int]],
) -> dict[str, Any]:
    time = np.asarray([row["time_s"] for row in rows])
    contact = np.asarray([row["contact_count"] > 0 for row in rows])
    linear = np.asarray([row["linear_speed_m_s"] for row in rows])
    angular = np.asarray([row["angular_speed_rad_s"] for row in rows])
    x = np.asarray([row["position_x_m"] for row in rows])
    y = np.asarray([row["position_y_m"] for row in rows])
    orientation = np.asarray([row["orientation_change_rad"] for row in rows])
    collision_distance = np.asarray([row["collision_minimum_distance_m"] for row in rows])
    penetration = np.maximum(0.0, -collision_distance)
    upward_velocity = np.asarray([row["linear_velocity_z_m_s"] for row in rows])
    force = np.asarray([row["normal_force_sum_N"] for row in rows])
    native_clearance = np.asarray([row["native_visual_clearance_m"] for row in rows])

    final_start = float(time[-1] - criteria["final_window_s"])
    final = time >= final_start - 1e-12
    contact_indices = np.flatnonzero(contact)
    first_contact_index = None if not len(contact_indices) else int(contact_indices[0])
    first_contact_s = None if first_contact_index is None else float(time[first_contact_index])
    if first_contact_index is None:
        post_contact_max_upward = None
        post_contact_max_separation = None
        obvious_rebound = False
    else:
        post = slice(first_contact_index, None)
        post_contact_max_upward = float(np.max(upward_velocity[post]))
        post_contact_max_separation = float(np.max(collision_distance[post]))
        obvious_rebound = bool(
            post_contact_max_upward > criteria["obvious_rebound_upward_speed_m_s"]
            or post_contact_max_separation > criteria["obvious_rebound_separation_m"]
        )

    final_horizontal_range = float(np.hypot(np.ptp(x[final]), np.ptp(y[final])))
    final_orientation_range = float(np.ptp(orientation[final]))
    metrics = {
        "initial_native_visual_clearance_m": float(native_clearance[0]),
        "initial_collision_minimum_distance_m": float(collision_distance[0]),
        "first_contact_s": first_contact_s,
        "final_contact_fraction": float(np.mean(contact[final])),
        "final_linear_speed_p95_m_s": float(np.quantile(linear[final], 0.95)),
        "final_angular_speed_p95_rad_s": float(np.quantile(angular[final], 0.95)),
        "final_horizontal_range_m": final_horizontal_range,
        "final_orientation_range_rad": final_orientation_range,
        "final_collision_penetration_p95_m": float(np.quantile(penetration[final], 0.95)),
        "final_native_visual_clearance_median_m": float(np.median(native_clearance[final])),
        "final_native_visual_clearance_range_m": [
            float(np.min(native_clearance[final])), float(np.max(native_clearance[final]))],
        "post_contact_max_upward_speed_m_s": post_contact_max_upward,
        "post_contact_max_collision_separation_m": post_contact_max_separation,
        "obvious_rebound": obvious_rebound,
        "maximum_contact_force_N": float(np.max(force)),
        "final_normal_force_median_N": float(np.median(force[final])),
        "maximum_collision_penetration_m": float(np.max(penetration)),
        "maximum_horizontal_displacement_m": float(np.max(
            [row["horizontal_displacement_m"] for row in rows])),
        "final_horizontal_displacement_m": float(rows[-1]["horizontal_displacement_m"]),
        "final_orientation_change_rad": float(rows[-1]["orientation_change_rad"]),
        "final_linear_speed_m_s": float(rows[-1]["linear_speed_m_s"]),
        "final_angular_speed_rad_s": float(rows[-1]["angular_speed_rad_s"]),
        "simulation_warnings": warnings,
    }
    checks = {
        "no_simulation_warning": not warnings,
        "contact_by_deadline": (
            first_contact_s is not None
            and first_contact_s <= criteria["first_contact_deadline_s"]
        ),
        "final_contact_fraction": (
            metrics["final_contact_fraction"] >= criteria["final_contact_fraction_min"]
        ),
        "final_linear_speed": (
            metrics["final_linear_speed_p95_m_s"]
            <= criteria["final_linear_speed_p95_max_m_s"]
        ),
        "final_angular_speed": (
            metrics["final_angular_speed_p95_rad_s"]
            <= criteria["final_angular_speed_p95_max_rad_s"]
        ),
        "final_horizontal_range": (
            metrics["final_horizontal_range_m"]
            <= criteria["final_horizontal_range_max_m"]
        ),
        "final_orientation_range": (
            metrics["final_orientation_range_rad"]
            <= criteria["final_orientation_range_max_rad"]
        ),
        "final_collision_penetration": (
            metrics["final_collision_penetration_p95_m"]
            <= criteria["final_collision_penetration_p95_max_m"]
        ),
        "no_obvious_rebound": not obvious_rebound,
    }
    return {"metrics": metrics, "checks": checks, "stable": bool(all(checks.values()))}


def write_timeseries(
    path: Path, all_rows: dict[str, list[dict[str, Any]]],
) -> None:
    fieldnames = ["variant", *all_rows[next(iter(all_rows))][0].keys()]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for variant, rows in all_rows.items():
            for row in rows:
                writer.writerow({"variant": variant, **row})


def write_npz(path: Path, all_rows: dict[str, list[dict[str, Any]]]) -> None:
    arrays = {}
    for variant, rows in all_rows.items():
        for field in rows[0]:
            values = [np.nan if row[field] is None else row[field] for row in rows]
            arrays[f"{variant}__{field}"] = np.asarray(values)
    np.savez_compressed(path, **arrays)


def write_plot(path: Path, all_rows: dict[str, list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for variant, rows in all_rows.items():
        time = np.asarray([row["time_s"] for row in rows])
        axes[0].plot(time, 1000 * np.asarray([
            row["native_visual_clearance_m"] for row in rows]), label=f"{variant}: visual")
        axes[0].plot(time, 1000 * np.asarray([
            row["collision_minimum_distance_m"] for row in rows]), linestyle="--",
            label=f"{variant}: collision")
        axes[1].plot(time, [row["normal_force_sum_N"] for row in rows], label=variant)
        axes[2].plot(time, 1000 * np.asarray([
            row["linear_speed_m_s"] for row in rows]), label=f"{variant}: linear mm/s")
        axes[2].plot(time, [row["angular_speed_rad_s"] for row in rows], linestyle="--",
                     label=f"{variant}: angular rad/s")
        axes[3].plot(time, 1000 * np.asarray([
            row["horizontal_displacement_m"] for row in rows]), label=f"{variant}: XY mm")
        axes[3].plot(time, np.degrees([row["orientation_change_rad"] for row in rows]),
                     linestyle="--", label=f"{variant}: angle deg")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("table distance (mm)")
    axes[1].set_ylabel("normal force sum (N)")
    axes[2].set_ylabel("speed")
    axes[3].set_ylabel("motion from frame 0")
    axes[3].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    fig.suptitle("Bowl 146 isolated gravity settling on Issue #14 candidate table")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected schema")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], "HEAD"],
        cwd=REPO_ROOT, check=True,
    )
    forbidden = (
        "modify_original_data", "modify_source_scene", "modify_active_support_contract",
        "modify_mink_v2_trajectory", "add_auxiliary_constraint", "apply_external_force",
        "tune_physics", "replay", "mpc", "reinforcement_learning", "promotion",
    )
    if any(cfg["authorization"][name] for name in forbidden):
        raise ValueError("a forbidden operation is authorized")
    source_scene = Path(cfg["paths"]["source_scene"]).resolve(strict=True)
    reference_path = Path(cfg["paths"]["robot_reference"]).resolve(strict=True)
    issue14_path = Path(cfg["paths"]["issue14_static_audit"]).resolve(strict=True)
    output = Path(cfg["paths"]["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    issue14 = json.loads(issue14_path.read_text(encoding="utf-8"))
    table_z = float(cfg["candidate_table"]["simulator_z_m"])
    verified_z = float(issue14["candidate_support"]["simulator_plane"]["offset_m"])
    if table_z != verified_z or issue14["decision"]["candidate_support_promoted"]:
        raise ValueError("candidate table does not exactly match the verified read-only audit")

    isolated_scene = output / "isolated_bowl_table_scene.xml"
    build_isolated_scene(source_scene, isolated_scene, table_z)
    source_model = mujoco.MjModel.from_xml_path(str(source_scene))
    isolated_model = mujoco.MjModel.from_xml_path(str(isolated_scene))
    fidelity = verify_isolation_fidelity(source_model, isolated_model, table_z)
    initial_qpos, recorded_qvel, initial_state = extract_initial_state(
        source_model, reference_path)
    initial_state["pose_transport_max_abs_error"] = verify_initial_pose_transport(
        source_model, isolated_model, reference_path, initial_qpos)

    expected_variants = ["recorded_frame0_velocity", "zero_initial_velocity"]
    if cfg["simulation"]["variants"] != expected_variants:
        raise ValueError("variant contract changed")
    variant_velocities = {
        "recorded_frame0_velocity": recorded_qvel,
        "zero_initial_velocity": np.zeros(6, dtype=np.float64),
    }
    all_rows = {}
    assessments = {}
    for variant in expected_variants:
        rows, warnings = run_variant(
            isolated_model, initial_qpos, variant_velocities[variant],
            float(cfg["simulation"]["duration_s"]),
        )
        all_rows[variant] = rows
        assessments[variant] = assess_variant(
            rows, cfg["frozen_stability_criteria"], warnings)
    static_weight = float(
        isolated_model.body_mass[isolated_model.body("left_object").id]
        * np.linalg.norm(isolated_model.opt.gravity)
    )
    for assessment in assessments.values():
        assessment["metrics"]["expected_static_weight_N"] = static_weight
        assessment["metrics"]["final_normal_force_to_weight_ratio"] = (
            assessment["metrics"]["final_normal_force_median_N"] / static_weight
        )

    if any(assessment["metrics"]["simulation_warnings"] for assessment in assessments.values()):
        verdict = "INDETERMINATE"
    elif all(assessment["stable"] for assessment in assessments.values()):
        verdict = "STABLE_SEATING"
    else:
        verdict = "NOT_STABLE"

    write_timeseries(output / "timeseries.csv", all_rows)
    write_npz(output / "timeseries.npz", all_rows)
    write_plot(output / "settling_curves.png", all_rows)
    report = {
        "schema": SCHEMA,
        "sample": cfg["sample"],
        "candidate_table": cfg["candidate_table"],
        "source_artifacts": {
            "source_scene": artifact(source_scene),
            "robot_reference": artifact(reference_path),
            "issue14_static_audit": artifact(issue14_path),
            "config": artifact(config_path),
            "isolated_scene": artifact(isolated_scene),
        },
        "isolation_fidelity": fidelity,
        "initial_state": initial_state,
        "simulation": {
            "duration_s": float(cfg["simulation"]["duration_s"]),
            "timestep_s": float(isolated_model.opt.timestep),
            "step_count": int(round(
                float(cfg["simulation"]["duration_s"]) / isolated_model.opt.timestep)),
            "backend": "CPU_MUJOCO",
            "external_force_applied": False,
            "auxiliary_constraint_added": False,
            "physics_parameters_tuned": False,
            "other_dynamic_bodies": 0,
        },
        "frozen_stability_criteria": cfg["frozen_stability_criteria"],
        "variants": assessments,
        "verdict": verdict,
        "limitations": [
            "This isolated result does not validate the complete robot scene.",
            "The Issue #14 candidate table remains unpromoted.",
            "No claim is made about future trajectory frames.",
        ],
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    }
    write_json(output / "results.json", report)

    label = {"STABLE_SEATING": "稳定落座", "NOT_STABLE": "未稳定",
             "INDETERMINATE": "无法判定"}[verdict]
    lines = [
        "# Brush bowl frame-0 isolated gravity-settling audit", "",
        f"Final verdict: **{label}** (`{verdict}`).", "",
        f"The verified Issue #14 candidate table is fixed at simulator `Z={table_z:.12f} m`. "
        "The isolated model contains only the fixed plane and dynamic bowl 146; the source "
        "mass, inertia, 32 convex collision meshes, contact pairs, gravity, time step, solver, "
        "and contact parameters are unchanged.", "",
        "## Frozen criteria", "",
        "The final 0.5 s requires contact coverage >=99%, linear-speed P95 <=5 mm/s, "
        "angular-speed P95 <=0.05 rad/s, horizontal range <=0.5 mm, orientation range "
        "<=0.5 degrees, and collision-penetration P95 <=0.5 mm. First contact must occur "
        "by 1.0 s. Post-contact upward speed >0.05 m/s or separation >1 mm is an obvious "
        "rebound. Both velocity variants must pass.", "",
        "## Results", "",
        "| variant | first contact (s) | final contact | linear P95 (mm/s) | angular P95 (rad/s) | horizontal range (mm) | orientation range (deg) | penetration P95 (mm) | rebound | stable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for variant in expected_variants:
        result = assessments[variant]
        metric = result["metrics"]
        contact_text = "N/A" if metric["first_contact_s"] is None else f"{metric['first_contact_s']:.6f}"
        lines.append(
            f"| {variant} | {contact_text} | {metric['final_contact_fraction']:.3%} | "
            f"{metric['final_linear_speed_p95_m_s']*1000:.6f} | "
            f"{metric['final_angular_speed_p95_rad_s']:.6f} | "
            f"{metric['final_horizontal_range_m']*1000:.6f} | "
            f"{math.degrees(metric['final_orientation_range_rad']):.6f} | "
            f"{metric['final_collision_penetration_p95_m']*1000:.6f} | "
            f"{'YES' if metric['obvious_rebound'] else 'NO'} | "
            f"{'YES' if result['stable'] else 'NO'} |"
        )
    lines += ["", "## Contact and net motion", "",
              "| variant | initial visual gap (mm) | peak force (N) | final force / weight | max transient penetration (mm) | settled visual distance (mm) | final XY displacement (mm) | final orientation change (deg) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in expected_variants:
        metric = assessments[variant]["metrics"]
        lines.append(
            f"| {variant} | {metric['initial_native_visual_clearance_m']*1000:.6f} | "
            f"{metric['maximum_contact_force_N']:.6f} | "
            f"{metric['final_normal_force_to_weight_ratio']:.6f} | "
            f"{metric['maximum_collision_penetration_m']*1000:.6f} | "
            f"{metric['final_native_visual_clearance_median_m']*1000:.6f} | "
            f"{metric['final_horizontal_displacement_m']*1000:.6f} | "
            f"{math.degrees(metric['final_orientation_change_rad']):.6f} |"
        )
    lines += ["", "The settled median normal force is compared with `mass * |gravity|`; a ratio "
              "near 1 is the expected static weight balance. Negative settled visual distance "
              "denotes the soft-contact overlap retained by the unchanged MuJoCo parameters.",
              "", "## Scope", "",
              "No source asset, active SupportSurfaceContract, MINK v2 trajectory, or formal "
              "scene was modified. No auxiliary constraint or external force was applied, and "
              "no parameter was tuned. Replay, MPC, and RL did not run. Passing this isolated "
              "test would not imply that the full robot scene passes.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "variants": assessments}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
