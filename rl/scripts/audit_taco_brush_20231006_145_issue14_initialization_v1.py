#!/usr/bin/env python3
"""Audit Issue-14 support initialization and isolated natural settling."""

from __future__ import annotations

import argparse
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
from scipy.spatial.transform import Rotation
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_pair_measurement,
    object_global_minimum,
    triangle_object,
)
from egoengine_repro.evaluation.taco_surface import reconstruct_taco_mano  # noqa: E402


SCHEMA = "taco_brush_20231006_145_issue14_initialization_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_20231006_145_issue14_initialization_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def classify(distance_m: float, tolerance_m: float) -> str:
    if distance_m < -tolerance_m:
        return "PENETRATION"
    if distance_m <= tolerance_m:
        return "CONTACT_WITHIN_TOLERANCE"
    return "CLEARANCE"


def load_object(data_root: Path, sequence: str, spec: dict[str, str]) -> dict[str, Any]:
    mesh_path = data_root / "object_models_released" / f"{spec['id']}_cm.obj"
    pose_path = data_root / "Object_Poses" / sequence / f"{spec['role']}_{spec['id']}.npy"
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{mesh_path} is not one triangle mesh")
    mesh.apply_scale(0.01)
    poses = np.asarray(np.load(pose_path, allow_pickle=False), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"invalid pose array: {pose_path}")
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError(f"source mesh is not a closed positive-volume solid: {mesh_path}")
    return {"spec": spec, "mesh": mesh, "poses": poses, "mesh_path": mesh_path, "pose_path": pose_path}


def world_mesh(entry: dict[str, Any], frame: int) -> trimesh.Trimesh:
    mesh = entry["mesh"].copy()
    mesh.apply_transform(entry["poses"][frame])
    return mesh


def fcl_entry(name: str, mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return {
        "geom": name,
        "vertices_sim_m": vertices,
        "faces": faces,
        "mesh": mesh,
        "fcl": triangle_object(vertices, faces),
    }


def quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def qpos_from_pose(pose: np.ndarray) -> np.ndarray:
    return np.concatenate((np.asarray(pose[:3, 3], dtype=np.float64), quaternion_wxyz(pose[:3, :3])))


def mass_properties(mesh: trimesh.Trimesh, density: float) -> dict[str, Any]:
    copy = mesh.copy()
    copy.density = density
    props = copy.mass_properties
    inertia = np.asarray(props["inertia"], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.any(eigenvalues <= 0) or not np.isfinite(inertia).all():
        raise ValueError("invalid source-mesh mass properties")
    return {
        "density_kg_m3": density,
        "mass_kg": float(props["mass"]),
        "center_mass_m": np.asarray(props["center_mass"], dtype=np.float64),
        "inertia_kg_m2": inertia,
        "principal_checks": eigenvalues.tolist(),
    }


def build_scene(
    destination: Path,
    entry: dict[str, Any],
    collision_dir: Path,
    table_z: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    spec = entry["spec"]
    parts = sorted(collision_dir.glob("*.obj"), key=lambda path: int(path.stem))
    if not parts:
        raise ValueError(f"no collision parts: {collision_dir}")
    props = mass_properties(entry["mesh"], float(cfg["physics"]["density_kg_m3"]))
    root = ET.Element("mujoco", {"model": f"taco_{spec['name']}_{spec['id']}_frame0_isolated"})
    ET.SubElement(root, "compiler", {
        "angle": "radian", "inertiafromgeom": "false", "balanceinertia": "false",
    })
    physics = cfg["physics"]
    ET.SubElement(root, "option", {
        "timestep": str(physics["timestep_s"]),
        "gravity": " ".join(map(str, physics["gravity_m_s2"])),
        "integrator": physics["integrator"], "solver": physics["solver"],
        "iterations": str(physics["iterations"]), "tolerance": str(physics["tolerance"]),
        "cone": physics["cone"], "impratio": str(physics["impratio"]),
    })
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {
        "name": "native_visual", "file": str(entry["mesh_path"].resolve(strict=True)),
        "scale": "0.01 0.01 0.01",
    })
    for index, path in enumerate(parts):
        ET.SubElement(asset, "mesh", {"name": f"collision_{index}", "file": str(path.resolve(strict=True))})
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", {
        "name": "floor", "type": "plane", "size": "0 0 0.05",
        "pos": f"0 0 {table_z:.17g}", "contype": "0", "conaffinity": "0",
        "condim": "3", "friction": " ".join(map(str, physics["floor_friction"])),
    })
    body = ET.SubElement(world, "body", {"name": "object"})
    ET.SubElement(body, "freejoint", {"name": "object_joint"})
    center = props["center_mass_m"]
    inertia = props["inertia_kg_m2"]
    ET.SubElement(body, "inertial", {
        "pos": " ".join(f"{value:.17g}" for value in center),
        "mass": f"{props['mass_kg']:.17g}",
        "fullinertia": " ".join(f"{value:.17g}" for value in (
            inertia[0, 0], inertia[1, 1], inertia[2, 2],
            inertia[0, 1], inertia[0, 2], inertia[1, 2],
        )),
    })
    ET.SubElement(body, "geom", {
        "name": "native_visual", "type": "mesh", "mesh": "native_visual",
        "density": "0", "contype": "0", "conaffinity": "0", "rgba": "0.7 0.7 0.7 0.35",
    })
    for index, _ in enumerate(parts):
        ET.SubElement(body, "geom", {
            "name": f"collision_{index}", "type": "mesh", "mesh": f"collision_{index}",
            "density": "0", "contype": "0", "conaffinity": "0", "condim": "3",
            "friction": " ".join(map(str, physics["floor_friction"])), "rgba": "0 1 0 0.1",
        })
    contact = ET.SubElement(root, "contact")
    for index, _ in enumerate(parts):
        ET.SubElement(contact, "pair", {
            "name": f"floor_{index}", "geom1": f"collision_{index}", "geom2": "floor",
            "condim": "3", "friction": " ".join(map(str, physics["pair_friction"])),
        })
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return {"collision_part_count": len(parts), **props}


def native_clearance(data: mujoco.MjData, body_id: int, vertices: np.ndarray, table_z: float) -> float:
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    world = np.asarray(vertices, dtype=np.float64) @ rotation.T + np.asarray(data.xpos[body_id])
    return float(np.min(world[:, 2] - table_z))


def quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64) / np.linalg.norm(first)
    second = np.asarray(second, dtype=np.float64) / np.linalg.norm(second)
    return float(2 * np.arccos(np.clip(abs(float(first @ second)), 0.0, 1.0)))


def sample_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    raw_vertices: np.ndarray,
    table_z: float,
    initial_position: np.ndarray,
    initial_quaternion: np.ndarray,
) -> dict[str, Any]:
    body_id = model.body("object").id
    floor_id = model.geom("floor").id
    collision_ids = [model.geom(f"collision_{index}").id for index in range(model.npair)]
    proxy_distances = np.asarray([
        mujoco.mj_geomDistance(model, data, geom_id, floor_id, 1.0, None)
        for geom_id in collision_ids
    ], dtype=np.float64)
    forces = []
    contact_distances = []
    collision_set = set(collision_ids)
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor_id not in pair or not pair.intersection(collision_set):
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, wrench)
        forces.append(max(0.0, float(wrench[0])))
        contact_distances.append(float(contact.dist))
    position = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    quaternion = np.asarray(data.xquat[body_id], dtype=np.float64).copy()
    qvel = np.asarray(data.qvel[:6], dtype=np.float64).copy()
    proxy = float(np.min(proxy_distances))
    return {
        "time_s": float(data.time),
        "position_x_m": float(position[0]), "position_y_m": float(position[1]), "position_z_m": float(position[2]),
        "quaternion_w": float(quaternion[0]), "quaternion_x": float(quaternion[1]),
        "quaternion_y": float(quaternion[2]), "quaternion_z": float(quaternion[3]),
        "linear_velocity_x_m_s": float(qvel[0]), "linear_velocity_y_m_s": float(qvel[1]),
        "linear_velocity_z_m_s": float(qvel[2]), "angular_velocity_x_rad_s": float(qvel[3]),
        "angular_velocity_y_rad_s": float(qvel[4]), "angular_velocity_z_rad_s": float(qvel[5]),
        "linear_speed_m_s": float(np.linalg.norm(qvel[:3])),
        "angular_speed_rad_s": float(np.linalg.norm(qvel[3:])),
        "horizontal_displacement_m": float(np.linalg.norm(position[:2] - initial_position[:2])),
        "orientation_change_rad": quaternion_distance(initial_quaternion, quaternion),
        "native_visual_clearance_m": native_clearance(data, body_id, raw_vertices, table_z),
        "collision_minimum_distance_m": proxy,
        "collision_penetration_depth_m": max(0.0, -proxy),
        "contact_count": len(forces),
        "contact_minimum_distance_m": None if not contact_distances else float(min(contact_distances)),
        "normal_force_sum_N": float(sum(forces)),
        "normal_force_max_N": float(max(forces, default=0.0)),
    }


def warnings(data: mujoco.MjData) -> list[dict[str, int]]:
    result = []
    for warning_type in range(mujoco.mjtWarning.mjNWARNING):
        item = data.warning[warning_type]
        if int(item.number):
            result.append({"type": warning_type, "number": int(item.number), "lastinfo": int(item.lastinfo)})
    return result


def run_variant(
    model: mujoco.MjModel,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    raw_vertices: np.ndarray,
    table_z: float,
    duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    body_id = model.body("object").id
    initial_position = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    initial_quaternion = np.asarray(data.xquat[body_id], dtype=np.float64).copy()
    count = int(round(duration_s / model.opt.timestep))
    if not np.isclose(count * model.opt.timestep, duration_s, atol=1e-12):
        raise ValueError("duration must be divisible by timestep")
    rows = [sample_state(model, data, raw_vertices, table_z, initial_position, initial_quaternion)]
    for _ in range(count):
        mujoco.mj_step(model, data)
        rows.append(sample_state(model, data, raw_vertices, table_z, initial_position, initial_quaternion))
    return rows, warnings(data)


def assess(rows: list[dict[str, Any]], criteria: dict[str, Any], warning_rows: list[dict[str, int]]) -> dict[str, Any]:
    values = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in (
        "time_s", "linear_speed_m_s", "angular_speed_rad_s", "position_x_m", "position_y_m",
        "orientation_change_rad", "collision_minimum_distance_m", "linear_velocity_z_m_s",
        "normal_force_sum_N", "native_visual_clearance_m", "horizontal_displacement_m",
    )}
    contact = np.asarray([row["contact_count"] > 0 for row in rows])
    final = values["time_s"] >= values["time_s"][-1] - float(criteria["final_window_s"]) - 1e-12
    contact_indices = np.flatnonzero(contact)
    first_index = None if not len(contact_indices) else int(contact_indices[0])
    first_contact = None if first_index is None else float(values["time_s"][first_index])
    if first_index is None:
        max_upward = None
        max_separation = None
        rebound = False
    else:
        max_upward = float(np.max(values["linear_velocity_z_m_s"][first_index:]))
        max_separation = float(np.max(values["collision_minimum_distance_m"][first_index:]))
        rebound = bool(max_upward > criteria["obvious_rebound_upward_speed_m_s"] or max_separation > criteria["obvious_rebound_separation_m"])
    penetration = np.maximum(0.0, -values["collision_minimum_distance_m"])
    metrics = {
        "initial_native_visual_clearance_m": float(values["native_visual_clearance_m"][0]),
        "initial_collision_minimum_distance_m": float(values["collision_minimum_distance_m"][0]),
        "first_contact_s": first_contact,
        "final_contact_fraction": float(np.mean(contact[final])),
        "final_linear_speed_p95_m_s": float(np.quantile(values["linear_speed_m_s"][final], 0.95)),
        "final_angular_speed_p95_rad_s": float(np.quantile(values["angular_speed_rad_s"][final], 0.95)),
        "final_horizontal_range_m": float(np.hypot(np.ptp(values["position_x_m"][final]), np.ptp(values["position_y_m"][final]))),
        "final_orientation_range_rad": float(np.ptp(values["orientation_change_rad"][final])),
        "final_collision_penetration_p95_m": float(np.quantile(penetration[final], 0.95)),
        "final_native_visual_clearance_median_m": float(np.median(values["native_visual_clearance_m"][final])),
        "final_native_visual_clearance_range_m": [float(np.min(values["native_visual_clearance_m"][final])), float(np.max(values["native_visual_clearance_m"][final]))],
        "post_contact_max_upward_speed_m_s": max_upward,
        "post_contact_max_collision_separation_m": max_separation,
        "obvious_rebound": rebound,
        "maximum_contact_force_N": float(np.max(values["normal_force_sum_N"])),
        "final_normal_force_median_N": float(np.median(values["normal_force_sum_N"][final])),
        "maximum_collision_penetration_m": float(np.max(penetration)),
        "maximum_horizontal_displacement_m": float(np.max(values["horizontal_displacement_m"])),
        "final_horizontal_displacement_m": float(values["horizontal_displacement_m"][-1]),
        "final_orientation_change_rad": float(values["orientation_change_rad"][-1]),
        "final_linear_speed_m_s": float(values["linear_speed_m_s"][-1]),
        "final_angular_speed_rad_s": float(values["angular_speed_rad_s"][-1]),
        "simulation_warnings": warning_rows,
    }
    checks = {
        "no_simulation_warning": not warning_rows,
        "contact_by_deadline": first_contact is not None and first_contact <= criteria["first_contact_deadline_s"],
        "final_contact_fraction": metrics["final_contact_fraction"] >= criteria["final_contact_fraction_min"],
        "final_linear_speed": metrics["final_linear_speed_p95_m_s"] <= criteria["final_linear_speed_p95_max_m_s"],
        "final_angular_speed": metrics["final_angular_speed_p95_rad_s"] <= criteria["final_angular_speed_p95_max_rad_s"],
        "final_horizontal_range": metrics["final_horizontal_range_m"] <= criteria["final_horizontal_range_max_m"],
        "final_orientation_range": metrics["final_orientation_range_rad"] <= criteria["final_orientation_range_max_rad"],
        "final_collision_penetration": metrics["final_collision_penetration_p95_m"] <= criteria["final_collision_penetration_p95_max_m"],
        "no_obvious_rebound": not rebound,
    }
    return {"metrics": metrics, "checks": checks, "stable": bool(all(checks.values()))}


def write_timeseries(path: Path, rows_by_key: dict[str, list[dict[str, Any]]]) -> None:
    first = next(iter(rows_by_key.values()))[0]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["object_variant", *first.keys()])
        writer.writeheader()
        for key, rows in rows_by_key.items():
            for row in rows:
                writer.writerow({"object_variant": key, **row})


def write_plot(path: Path, rows_by_key: dict[str, list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for key, rows in rows_by_key.items():
        time = np.asarray([row["time_s"] for row in rows])
        axes[0].plot(time, 1000 * np.asarray([row["native_visual_clearance_m"] for row in rows]), label=key)
        axes[1].plot(time, [row["normal_force_sum_N"] for row in rows], label=key)
        axes[2].plot(time, 1000 * np.asarray([row["linear_speed_m_s"] for row in rows]), label=key)
        axes[3].plot(time, 1000 * np.asarray([row["horizontal_displacement_m"] for row in rows]), label=f"{key}: xy mm")
        axes[3].plot(time, np.degrees([row["orientation_change_rad"] for row in rows]), linestyle="--", label=f"{key}: deg")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("native table gap (mm)")
    axes[1].set_ylabel("normal force (N)")
    axes[2].set_ylabel("linear speed (mm/s)")
    axes[3].set_ylabel("motion from frame 0")
    axes[3].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6, ncol=2)
    fig.suptitle("20231006_145 isolated object settling on Issue #14 table")
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
    subprocess.run(["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], "HEAD"], cwd=REPO_ROOT, check=True)
    forbidden = ("modify_original_data", "modify_official_pose", "modify_active_support_contract", "run_mink", "replay", "mpc", "reinforcement_learning", "promotion")
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("forbidden operation authorized")
    data_root = Path(cfg["paths"]["data_root"]).resolve(strict=True)
    output = Path(cfg["paths"]["output"]).resolve(strict=True)
    # A failed run may leave derived scenes/curves, but never a completed
    # decision.  Completed evidence remains immutable while an incomplete run
    # can be reproduced from scratch under the same contract.
    reserved = [output / name for name in ("results.json", "summary.md")]
    if any(path.exists() for path in reserved):
        raise FileExistsError("audit output already exists")
    sequence = cfg["sample"]["sequence"]
    objects = {spec["name"]: load_object(data_root, sequence, spec) for spec in cfg["sample"]["objects"]}
    if any(len(entry["poses"]) != cfg["sample"]["frames"] for entry in objects.values()):
        raise ValueError("official object timeline length mismatch")

    minima = [object_global_minimum(np.asarray(entry["mesh"].vertices), entry["poses"], role=entry["spec"]["role"], object_id=entry["spec"]["id"]) for entry in objects.values()]
    winner = min(minima, key=lambda row: row["minimum_world_z_m"])
    table_z = float(winner["minimum_world_z_m"])
    tolerance = float(cfg["static_audit"]["material_penetration_tolerance_m"])
    static_objects = {}
    frame0_meshes = {}
    for name, entry in objects.items():
        mesh = world_mesh(entry, 0)
        frame0_meshes[name] = mesh
        signed = np.asarray(mesh.vertices)[:, 2] - table_z
        index = int(np.argmin(signed))
        static_objects[name] = {
            "minimum_signed_distance_m": float(signed[index]),
            "status": classify(float(signed[index]), tolerance),
            "minimum_vertex_index": index,
            "minimum_point_world_m": np.asarray(mesh.vertices[index]).tolist(),
            "vertices_below_material_tolerance": int(np.sum(signed < -tolerance)),
            "mesh_watertight": bool(entry["mesh"].is_watertight),
            "mesh_winding_consistent": bool(entry["mesh"].is_winding_consistent),
            "mesh_volume_m3": float(entry["mesh"].volume),
        }
    object_pair = native_pair_measurement(fcl_entry("brush", frame0_meshes["brush"]), fcl_entry("bowl", frame0_meshes["bowl"]), tolerance)

    mano = {}
    for side in ("right", "left"):
        vertices, _, faces, _, keys = reconstruct_taco_mano(
            data_root / "Hand_Poses" / sequence / f"{side}_hand.pkl",
            data_root / "Hand_Poses" / sequence / f"{side}_hand_shape.pkl",
            data_root / "mano_v1_2" / "models" / f"MANO_{side.upper()}.pkl",
            side=side,
        )
        expected = tuple(f"{index:05d}" for index in range(1, cfg["sample"]["frames"] + 1))
        if keys != expected:
            raise ValueError(f"{side} MANO keys mismatch")
        signed = np.asarray(vertices[0, :, 2], dtype=np.float64) - table_z
        index = int(np.argmin(signed))
        hand_mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces, process=False)
        pair_rows = {}
        for name, object_mesh in frame0_meshes.items():
            measured = native_pair_measurement(fcl_entry(f"{side}_mano", hand_mesh), fcl_entry(name, object_mesh), tolerance)
            pair_rows[name] = {
                "distance_m": measured["distance_m"],
                "surface_intersection": measured["surface_intersection"],
                "reported_penetration_depth_m": measured["reported_penetration_depth_m"],
                "contact_count": measured["contact_count"],
            }
        mano[side] = {
            "minimum_table_signed_distance_m": float(signed[index]),
            "table_status": classify(float(signed[index]), tolerance),
            "minimum_vertex_index": index,
            "minimum_point_world_m": np.asarray(vertices[0, index]).tolist(),
            "vertices_below_material_tolerance": int(np.sum(signed < -tolerance)),
            "object_pairs": pair_rows,
        }

    rows_by_key: dict[str, list[dict[str, Any]]] = {}
    settling = {}
    scene_artifacts = {}
    collision_provenance = {}
    for name, entry in objects.items():
        collision_dir = output / "collision" / entry["spec"]["id"]
        collision_provenance[name] = artifact(collision_dir / "provenance.json")
        scene_path = output / f"isolated_{name}_table_scene.xml"
        props = build_scene(scene_path, entry, collision_dir, table_z, cfg)
        scene_artifacts[name] = artifact(scene_path)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        qpos0 = qpos_from_pose(entry["poses"][0])
        qpos1 = qpos_from_pose(entry["poses"][1])
        derived_qvel = np.zeros(model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(model, derived_qvel, 1.0 / 30.0, qpos0, qpos1)
        velocities = {
            "official_frame0_to_1_velocity": derived_qvel,
            "zero_initial_velocity": np.zeros(model.nv, dtype=np.float64),
        }
        assessments = {}
        for variant in cfg["physics"]["variants"]:
            key = f"{name}__{variant}"
            rows, warning_rows = run_variant(
                model, qpos0, velocities[variant], np.asarray(entry["mesh"].vertices),
                table_z, float(cfg["physics"]["duration_s"]),
            )
            rows_by_key[key] = rows
            assessments[variant] = assess(rows, cfg["frozen_stability_criteria"], warning_rows)
            expected_weight = props["mass_kg"] * abs(float(model.opt.gravity[2]))
            assessments[variant]["metrics"]["expected_static_weight_N"] = expected_weight
            assessments[variant]["metrics"]["final_normal_force_to_weight_ratio"] = assessments[variant]["metrics"]["final_normal_force_median_N"] / expected_weight
        serializable_props = {
            **props,
            "center_mass_m": props["center_mass_m"].tolist(),
            "inertia_kg_m2": props["inertia_kg_m2"].tolist(),
        }
        settling[name] = {
            "mass_and_collision": serializable_props,
            "initial_qpos": qpos0.tolist(),
            "derived_official_frame0_to_1_qvel": derived_qvel.tolist(),
            "derived_linear_speed_m_s": float(np.linalg.norm(derived_qvel[:3])),
            "derived_angular_speed_rad_s": float(np.linalg.norm(derived_qvel[3:])),
            "assessments": assessments,
            "stable_all_variants": bool(all(row["stable"] for row in assessments.values())),
        }

    write_timeseries(output / "timeseries.csv", rows_by_key)
    write_plot(output / "settling_curves.png", rows_by_key)
    object_settling_ok = all(row["stable_all_variants"] for row in settling.values())
    hand_table_ok = all(row["table_status"] != "PENETRATION" for row in mano.values())
    if not object_settling_ok:
        verdict = "OBJECT_SETTLING_NOT_STABLE"
    elif not hand_table_ok:
        verdict = "OBJECTS_SETTLE_BUT_OFFICIAL_HAND_INITIALIZATION_PENETRATES_TABLE"
    else:
        verdict = "SOURCE_INITIALIZATION_GEOMETRY_PASSES_CHECKED_CONTRACT"
    report = {
        "schema": SCHEMA,
        "sample": cfg["sample"],
        "candidate_support": {
            **cfg["candidate_support"],
            "table_world_z_m": table_z,
            "global_minimum": winner,
            "per_object_global_minima": minima,
            "promoted_to_active_contract": False,
        },
        "static_frame0": {
            "objects": static_objects,
            "official_mano_hands": mano,
            "brush_bowl_pair": object_pair,
            "material_penetration_tolerance_m": tolerance,
        },
        "isolated_settling": settling,
        "source_artifacts": {
            "config": artifact(config_path),
            "object_meshes": {name: artifact(entry["mesh_path"]) for name, entry in objects.items()},
            "object_poses": {name: artifact(entry["pose_path"]) for name, entry in objects.items()},
            "collision_provenance": collision_provenance,
            "isolated_scenes": scene_artifacts,
        },
        "limitations": [
            "TACO does not publish measured mass, inertia, friction, restitution, or contact-solver parameters for object 171/180.",
            "The isolated diagnostic uses closed source-mesh mass properties at the frozen local density and a fixed disclosed CoACD approximation.",
            "No XHand/MINK trajectory exists yet for this sequence; robot initialization is not assessed.",
            "Official MANO table penetration is a source-geometry finding and is not repaired by object settling.",
        ],
        "decision": {
            "verdict": verdict,
            "object_settling_passed_all_variants": object_settling_ok,
            "official_mano_table_clear": hand_table_ok,
            "candidate_support_promoted": False,
            "mink_authorized": False,
            "replay_or_rl_authorized": False,
        },
    }
    write_json(output / "results.json", report)

    def mm(value: float | None) -> str:
        return "n/a" if value is None else f"{1000 * value:.3f}"
    lines = [
        "# 20231006_145 Issue #14 initialization audit", "",
        f"- Verdict: **{verdict}**",
        f"- Candidate table: world `z={table_z:.12f} m`, from `{winner['role']} {winner['object_id']}` frame {winner['frame']} vertex {winner['vertex_index']}.",
        "- The candidate remains read-only and was not promoted to the active support contract.", "",
        "## Frame-0 static geometry", "",
        "| entity | signed table distance (mm) | status | vertices materially below |",
        "|---|---:|---|---:|",
    ]
    for name, row in static_objects.items():
        lines.append(f"| {name} | {mm(row['minimum_signed_distance_m'])} | {row['status']} | {row['vertices_below_material_tolerance']} |")
    for side, row in mano.items():
        lines.append(f"| official MANO {side} | {mm(row['minimum_table_signed_distance_m'])} | {row['table_status']} | {row['vertices_below_material_tolerance']} |")
    lines += ["", "## Isolated natural settling", "", "| object | velocity variant | stable | first contact (s) | final raw-mesh gap median (mm) | final linear P95 (mm/s) | final angular P95 (rad/s) | final XY drift (mm) | final rotation (deg) |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in settling.items():
        for variant, row in result["assessments"].items():
            metric = row["metrics"]
            lines.append(
                f"| {name} | {variant} | {row['stable']} | {metric['first_contact_s']} | "
                f"{mm(metric['final_native_visual_clearance_median_m'])} | {1000 * metric['final_linear_speed_p95_m_s']:.3f} | "
                f"{metric['final_angular_speed_p95_rad_s']:.6f} | {1000 * metric['final_horizontal_displacement_m']:.3f} | "
                f"{math.degrees(metric['final_orientation_change_rad']):.3f} |"
            )
    lines += [
        "", "## Interpretation", "",
        "The object check and the hand check are separate. Natural settling may remove a positive object-table gap, but it cannot repair official MANO vertices that start below the same table.",
        "This is a local engineering diagnostic, not a paper-faithful physics reconstruction: measured physical parameters for object 171/180 were not published.",
        "No MINK, full-scene Replay, MPC, or RL was run.", "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "validation_commands.md").write_text(
        "# Validation\n\n"
        f"- `{sys.executable} {Path(__file__).resolve()} --config {config_path}`\n"
        "- `python -m pytest -q rl/tests/core`\n",
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
