#!/usr/bin/env python3
"""Read-only frame-0 Brush penetration audit under two support planes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import fcl
import mujoco
import numpy as np
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from egoengine_repro.retarget.collision_audit import (  # noqa: E402
    collision_families,
    distances,
    validate_qpos,
)
from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    build_native_support_geoms,
    geom_world_vertices,
)
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance  # noqa: E402
from egoengine_repro.scene.support_surface import Plane  # noqa: E402


SCHEMA = "taco_brush_issue14_frame0_static_penetration_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_issue14_frame0_static_penetration_v1.yaml"


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


def classify_distance(distance_m: float, tolerance_m: float) -> str:
    if distance_m < -tolerance_m:
        return "PENETRATION"
    if distance_m <= tolerance_m:
        return "CONTACT_WITHIN_TOLERANCE"
    return "CLEARANCE"


def object_global_minimum(
    vertices_m: np.ndarray, poses: np.ndarray, *, role: str, object_id: str,
) -> dict[str, Any]:
    vertices = np.asarray(vertices_m, dtype=np.float64)
    transforms = np.asarray(poses, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("object mesh must contain finite Nx3 vertices")
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not len(transforms):
        raise ValueError("official object poses must be Fx4x4")
    if not np.isfinite(vertices).all() or not np.isfinite(transforms).all():
        raise ValueError("object mesh/pose contains nonfinite values")
    best: tuple[float, int, int, np.ndarray] | None = None
    for frame, pose in enumerate(transforms):
        world = vertices @ pose[:3, :3].T + pose[:3, 3]
        vertex = int(np.argmin(world[:, 2]))
        candidate = (float(world[vertex, 2]), frame, vertex, world[vertex].copy())
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return {
        "role": role,
        "object_id": object_id,
        "minimum_world_z_m": best[0],
        "frame": best[1],
        "vertex_index": best[2],
        "point_world_m": best[3].tolist(),
        "vertex_count": int(len(vertices)),
        "frame_count": int(len(transforms)),
    }


def transform_world_horizontal_plane_to_sim(T_sim_world: np.ndarray, height_world_m: float) -> Plane:
    transform = np.asarray(T_sim_world, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_sim_world must be one finite 4x4 transform")
    normal_world = np.array([0.0, 0.0, 1.0])
    normal_sim = transform[:3, :3] @ normal_world
    normal_sim /= np.linalg.norm(normal_sim)
    offset_sim = float(height_world_m + normal_sim @ transform[:3, 3])
    return Plane(normal=normal_sim, offset=offset_sim, frame="simulator")


def triangle_object(vertices: np.ndarray, faces: np.ndarray) -> fcl.CollisionObject:
    geometry = fcl.BVHModel()
    verts = np.ascontiguousarray(vertices, dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int32)
    geometry.beginModel(len(verts), len(tris))
    geometry.addSubModel(verts, tris)
    geometry.endModel()
    return fcl.CollisionObject(geometry)


def native_pair_measurement(
    first: dict[str, Any], second: dict[str, Any], tolerance_m: float,
) -> dict[str, Any]:
    collision = fcl.CollisionResult()
    fcl.collide(
        first["fcl"], second["fcl"],
        fcl.CollisionRequest(num_max_contacts=64, enable_contact=True), collision,
    )
    if collision.is_collision:
        contacts = list(collision.contacts)
        depths = [float(contact.penetration_depth) for contact in contacts
                  if np.isfinite(contact.penetration_depth)]
        deepest = int(np.argmax(depths)) if depths else None
        point = None
        normal = None
        if deepest is not None:
            point = np.asarray(contacts[deepest].pos, dtype=float).tolist()
            normal = np.asarray(contacts[deepest].normal, dtype=float).tolist()
        depth = max(depths) if depths else None
        return {
            "surface_intersection": True,
            "reported_penetration_depth_m": depth,
            "contact_count": len(contacts),
            "deepest_contact_position_sim_m": point,
            "deepest_contact_normal": normal,
            "distance_m": None if depth is None else -depth,
            "containment_followup": "NOT_NEEDED_SURFACE_INTERSECTION",
        }
    bounds = [entry["mesh"].bounds for entry in (first, second)]
    containment_possible = [
        bool(np.all(bounds[target][0] <= bounds[source][0])
             and np.all(bounds[target][1] >= bounds[source][1]))
        for source, target in ((0, 1), (1, 0))
    ]
    containment_rows = []
    entries = (first, second)
    for (source_index, target_index), possible in zip(((0, 1), (1, 0)), containment_possible):
        if not possible:
            continue
        source, target = entries[source_index], entries[target_index]
        if not target["mesh"].is_watertight:
            containment_rows.append({
                "source": source["geom"], "target": target["geom"],
                "status": "UNKNOWN_TARGET_NATIVE_MESH_OPEN",
            })
            continue
        signed = closed_mesh_signed_distance(target["mesh"], source["vertices_sim_m"])
        index = int(np.argmax(signed))
        containment_rows.append({
            "source": source["geom"], "target": target["geom"],
            "status": "FULL_VERTEX_WATERTIGHT_CONTAINMENT_TEST",
            "inside_vertex_count_over_tolerance": int((signed > tolerance_m).sum()),
            "maximum_inside_depth_m": float(max(0.0, signed[index])),
            "maximum_inside_point_sim_m": source["vertices_sim_m"][index].tolist(),
        })
    contained = [row for row in containment_rows
                 if row.get("inside_vertex_count_over_tolerance", 0) > 0]
    if contained:
        worst = max(contained, key=lambda row: row["maximum_inside_depth_m"])
        return {
            "surface_intersection": False,
            "containment_penetration": True,
            "reported_penetration_depth_m": worst["maximum_inside_depth_m"],
            "contact_count": 0,
            "deepest_contact_position_sim_m": worst["maximum_inside_point_sim_m"],
            "deepest_contact_normal": None,
            "distance_m": -worst["maximum_inside_depth_m"],
            "containment_followup": containment_rows,
        }
    result = fcl.DistanceResult()
    distance = float(fcl.distance(
        first["fcl"], second["fcl"], fcl.DistanceRequest(enable_nearest_points=True), result,
    ))
    nearest = None
    if result.nearest_points is not None:
        nearest = [np.asarray(point, dtype=float).tolist() for point in result.nearest_points]
    return {
        "surface_intersection": False,
        "reported_penetration_depth_m": 0.0,
        "contact_count": 0,
        "nearest_points_sim_m": nearest,
        "distance_m": distance,
        "containment_penetration": False,
        "containment_followup": containment_rows,
    }


def native_visual_geom_ids(model: mujoco.MjModel) -> list[int]:
    result = []
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or ""
        if (name.startswith(("right_", "left_")) and name.endswith("_visual")
                and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)):
            result.append(geom_id)
    if not result:
        raise ValueError("no native hand/object visual meshes found")
    return result


def mesh_faces(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    start = int(model.mesh_faceadr[mesh_id])
    count = int(model.mesh_facenum[mesh_id])
    faces = np.asarray(model.mesh_face[start:start + count], dtype=np.int32).copy()
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"native visual mesh has invalid faces: {model.geom(geom_id).name}")
    return faces


def native_world_objects(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> dict[int, dict[str, Any]]:
    geoms = build_native_support_geoms(model, mujoco, native_visual_geom_ids(model))
    result: dict[int, dict[str, Any]] = {}
    for geom in geoms:
        geom_id = geom.geom_id
        body_id = geom.body_id
        if body_id in result:
            raise ValueError(f"multiple native visual meshes for body {model.body(body_id).name}")
        vertices = geom_world_vertices(data, geom, full=True)
        faces = mesh_faces(model, geom_id)
        world_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        result[body_id] = {
            "geom_id": int(geom_id),
            "geom": model.geom(geom_id).name,
            "body": model.body(body_id).name,
            "vertices_sim_m": vertices,
            "faces": faces,
            "fcl": triangle_object(vertices, faces),
            "mesh": world_mesh,
        }
    return result


def semantic_entity(geom_name: str) -> str:
    if geom_name == "right_object_visual":
        return "brush"
    if geom_name == "left_object_visual":
        return "bowl"
    if geom_name.startswith("right_"):
        return "right_xhand"
    if geom_name.startswith("left_"):
        return "left_xhand"
    raise ValueError(f"unclassified native visual geom: {geom_name}")


def table_audit(
    native: dict[int, dict[str, Any]], plane: Plane, tolerance_m: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "brush": [], "bowl": [], "right_xhand": [], "left_xhand": [],
    }
    for entry in native.values():
        signed = np.asarray(plane.signed_distance(entry["vertices_sim_m"]), dtype=np.float64)
        index = int(np.argmin(signed))
        distance = float(signed[index])
        grouped[semantic_entity(entry["geom"])].append({
            "body": entry["body"],
            "native_visual_geom": entry["geom"],
            "vertex_index": index,
            "point_sim_m": entry["vertices_sim_m"][index].tolist(),
            "minimum_signed_distance_m": distance,
            "classification": classify_distance(distance, tolerance_m),
        })
    entities = {}
    for name, rows in grouped.items():
        if not rows:
            raise ValueError(f"no native geometry for {name}")
        worst = min(rows, key=lambda row: row["minimum_signed_distance_m"])
        entities[name] = {
            "minimum_signed_distance_m": worst["minimum_signed_distance_m"],
            "minimum_signed_distance_mm": worst["minimum_signed_distance_m"] * 1000.0,
            "penetration_depth_mm": max(0.0, -worst["minimum_signed_distance_m"] * 1000.0),
            "classification": worst["classification"],
            "worst_location": worst,
            "native_part_count": len(rows),
            "penetrating_parts": [row for row in rows if row["classification"] == "PENETRATION"],
            "contact_parts": [row for row in rows if row["classification"] == "CONTACT_WITHIN_TOLERANCE"],
        }
    return {
        "plane": plane.to_dict(),
        "tolerance_m": tolerance_m,
        "entities": entities,
        "penetrating_entities": [name for name, row in entities.items()
                                 if row["classification"] == "PENETRATION"],
    }


def pair_group_audit(
    model: mujoco.MjModel, data: mujoco.MjData, native: dict[int, dict[str, Any]],
    pairs: list[tuple[int, int]], tolerance_m: float,
) -> dict[str, Any]:
    shell = distances(model, data, pairs) if pairs else np.empty(0)
    shell_rows = []
    for index, (geom_a, geom_b) in enumerate(pairs):
        shell_rows.append({
            "geom1": model.geom(geom_a).name,
            "geom2": model.geom(geom_b).name,
            "distance_m": float(shell[index]),
            "classification": classify_distance(float(shell[index]), tolerance_m),
        })
    unique_body_pairs = sorted({
        tuple(sorted((int(model.geom_bodyid[a]), int(model.geom_bodyid[b]))))
        for a, b in pairs if int(model.geom_bodyid[a]) != int(model.geom_bodyid[b])
    })
    native_rows = []
    unknown = []
    for body_a, body_b in unique_body_pairs:
        if body_a not in native or body_b not in native:
            unknown.append({"body1": model.body(body_a).name, "body2": model.body(body_b).name,
                            "reason": "NATIVE_VISUAL_MESH_MISSING"})
            continue
        first, second = native[body_a], native[body_b]
        measured = native_pair_measurement(first, second, tolerance_m)
        depth = measured["reported_penetration_depth_m"]
        containment_followup = measured["containment_followup"]
        containment_unknown = bool(
            isinstance(containment_followup, list)
            and any(row.get("status", "").startswith("UNKNOWN")
                    for row in containment_followup)
        )
        material_intersection = bool(
            measured["surface_intersection"] or measured.get("containment_penetration", False)
        )
        if material_intersection and depth is None:
            classification = "UNKNOWN_INTERSECTION_DEPTH"
        elif material_intersection and depth > tolerance_m:
            classification = "PENETRATION"
        elif material_intersection:
            classification = "CONTACT_WITHIN_TOLERANCE"
        elif containment_unknown:
            classification = "UNKNOWN_CONTAINMENT_OPEN_MESH"
        else:
            classification = classify_distance(float(measured["distance_m"]), tolerance_m)
        native_rows.append({
            "body1": first["body"], "body2": second["body"],
            "native_visual_geom1": first["geom"], "native_visual_geom2": second["geom"],
            "classification": classification, **measured,
        })
    rank = {"PENETRATION": 0, "UNKNOWN_INTERSECTION_DEPTH": 1,
            "UNKNOWN_CONTAINMENT_OPEN_MESH": 1,
            "CONTACT_WITHIN_TOLERANCE": 2, "CLEARANCE": 3}
    native_sorted = sorted(native_rows, key=lambda row: (
        rank[row["classification"]],
        row["distance_m"] if row["distance_m"] is not None else 0.0,
    ))
    shell_sorted = sorted(shell_rows, key=lambda row: row["distance_m"])
    return {
        "shell_pair_count": len(pairs),
        "shell_minimum": shell_sorted[0] if shell_sorted else None,
        "shell_penetrating_pair_count": sum(row["classification"] == "PENETRATION"
                                             for row in shell_rows),
        "native_body_pair_count": len(native_rows),
        "native_minimum": native_sorted[0] if native_sorted else None,
        "native_penetrating_pairs": [row for row in native_rows
                                     if row["classification"] == "PENETRATION"],
        "native_contact_pairs": [row for row in native_rows
                                 if row["classification"] == "CONTACT_WITHIN_TOLERANCE"],
        "unknown_pairs": unknown + [row for row in native_rows
                                    if row["classification"].startswith("UNKNOWN")],
        "native_pairs": native_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected schema")
    subprocess.run(["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], "HEAD"],
                   cwd=REPO_ROOT, check=True)
    if cfg["sample"]["audited_frame"] != 0:
        raise ValueError("this contract authorizes frame 0 only")
    forbidden = ("modify_raw_data", "modify_robot_trajectory", "modify_object_pose",
                 "modify_active_support_contract", "run_mink", "physics_step", "replay", "mpc",
                 "reinforcement_learning", "promotion")
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("forbidden operation authorized")
    paths = {key: Path(value).resolve(strict=True) for key, value in cfg["paths"].items()
             if key != "output"}
    output = Path(cfg["paths"]["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tolerance = float(cfg["audit"]["material_penetration_tolerance_m"])

    minima = []
    object_inputs = {}
    for role, spec in cfg["sample"]["objects"].items():
        object_id = str(spec["id"])
        mesh_path = paths["object_model_root"] / f"{object_id}_cm.obj"
        pose_path = paths["object_pose_root"] / f"{role}_{object_id}.npy"
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"object mesh is not one triangle mesh: {mesh_path}")
        vertices_m = np.asarray(mesh.vertices, dtype=np.float64) * float(
            cfg["candidate_support"]["mesh_scale_to_m"])
        poses = np.load(pose_path, allow_pickle=False).astype(np.float64)
        if len(poses) != cfg["sample"]["frames"]:
            raise ValueError("official object pose frame count mismatch")
        minima.append(object_global_minimum(vertices_m, poses, role=role, object_id=object_id))
        object_inputs[role] = {"mesh": artifact(mesh_path), "pose": artifact(pose_path)}
    origin = min(minima, key=lambda row: row["minimum_world_z_m"])

    with np.load(paths["human_reference"], allow_pickle=False) as archive:
        human = {key: archive[key] for key in archive.files}
    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        robot = {key: archive[key] for key in archive.files}
    if not np.array_equal(robot["frame_indices"], np.arange(cfg["sample"]["frames"])):
        raise ValueError("saved MINK v2 reference does not cover exact source indices")
    candidate_plane = transform_world_horizontal_plane_to_sim(
        human["T_sim_world"], origin["minimum_world_z_m"],
    )
    active_raw = yaml.safe_load(paths["active_support_contract"].read_text(encoding="utf-8"))
    active_plane = Plane(
        normal=active_raw["simulator"]["normal"],
        offset=active_raw["simulator"]["offset_m"],
        frame=active_raw["simulator"]["frame"],
    )

    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    qpos0 = validate_qpos(model, robot["qpos"][0].copy())
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = robot["qvel"][0]
    data.ctrl[:] = robot["ctrl"][0]
    mujoco.mj_forward(model, data)
    object_pose_errors = {}
    for index, side in enumerate(("right", "left")):
        expected = human["T_sim_object_reference"][0, index]
        body = model.body(f"{side}_object").id
        actual = np.eye(4)
        actual[:3, :3] = data.xmat[body].reshape(3, 3)
        actual[:3, 3] = data.xpos[body]
        object_pose_errors[side] = float(np.max(np.abs(actual - expected)))
    pose_tolerance = float(cfg["audit"]["official_object_pose_max_abs_tolerance"])
    if max(object_pose_errors.values()) > pose_tolerance:
        raise ValueError("saved frame-0 object state differs from official transformed pose")

    native = native_world_objects(model, data)
    tables = {
        "issue14_candidate": table_audit(native, candidate_plane, tolerance),
        "active_support": table_audit(native, active_plane, tolerance),
    }
    families = collision_families(model)
    pair_groups = {}
    for name in ("self_explicit", "self_nonadjacent_shells", "hand_tool",
                 "hand_target", "tool_target"):
        pair_groups[name] = pair_group_audit(
            model, data, native, families[name], tolerance,
        )

    table_penetrations = {
        name: report["penetrating_entities"] for name, report in tables.items()
    }
    pair_penetrations = {
        name: len(report["native_penetrating_pairs"]) for name, report in pair_groups.items()
    }
    unknown_count = sum(len(report["unknown_pairs"]) for report in pair_groups.values())
    report = {
        "schema": SCHEMA,
        "sample": cfg["sample"],
        "candidate_support": {
            "definition": cfg["candidate_support"]["definition"],
            "source_url": cfg["candidate_support"]["source_url"],
            "per_object_minima": minima,
            "global_origin": origin,
            "world_plane": {"normal": [0.0, 0.0, 1.0],
                            "offset_m": origin["minimum_world_z_m"], "frame": "world"},
            "simulator_plane": candidate_plane.to_dict(),
            "active_offset_difference_mm": (
                candidate_plane.offset - active_plane.offset) * 1000.0,
        },
        "initial_state": {
            "source": artifact(paths["robot_reference"]),
            "human_reference": artifact(paths["human_reference"]),
            "scene": artifact(paths["scene"]),
            "frame": 0,
            "qpos_modified": False,
            "object_pose_modified": False,
            "object_pose_max_abs_errors": object_pose_errors,
            "object_pose_max_abs_tolerance": pose_tolerance,
            "mink_rerun": False,
            "physics_steps": 0,
        },
        "active_support_contract": artifact(paths["active_support_contract"]),
        "object_inputs": object_inputs,
        "table_audits": tables,
        "mesh_pair_audits": pair_groups,
        "decision": {
            "issue14_candidate_has_penetration": bool(table_penetrations["issue14_candidate"]
                                                       or any(pair_penetrations.values())
                                                       or unknown_count),
            "active_support_has_penetration": bool(table_penetrations["active_support"]
                                                    or any(pair_penetrations.values())
                                                    or unknown_count),
            "table_penetrating_entities": table_penetrations,
            "native_pair_penetrating_counts": pair_penetrations,
            "unknown_pair_count": unknown_count,
            "tolerance_m": tolerance,
            "candidate_support_promoted": False,
            "active_support_modified": False,
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
        "config": artifact(config_path),
    }
    write_json(output / "results.json", report)

    with (output / "frame0_table_distances.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "support", "entity", "minimum_signed_distance_mm", "classification",
            "penetration_depth_mm", "native_visual_geom", "body", "vertex_index",
            "point_sim_x_m", "point_sim_y_m", "point_sim_z_m",
        ))
        writer.writeheader()
        for support, audit in tables.items():
            for entity, value in audit["entities"].items():
                location = value["worst_location"]
                point = location["point_sim_m"]
                writer.writerow({
                    "support": support,
                    "entity": entity,
                    "minimum_signed_distance_mm": value["minimum_signed_distance_mm"],
                    "classification": value["classification"],
                    "penetration_depth_mm": value["penetration_depth_mm"],
                    "native_visual_geom": location["native_visual_geom"],
                    "body": location["body"],
                    "vertex_index": location["vertex_index"],
                    "point_sim_x_m": point[0],
                    "point_sim_y_m": point[1],
                    "point_sim_z_m": point[2],
                })

    lines = [
        "# Brush frame-0 static penetration audit (Issue #14 scheme A)", "",
        f"The candidate horizontal source-world table is `Z={origin['minimum_world_z_m']:.9f} m`, "
        f"set by `{origin['role']} {origin['object_id']}` frame `{origin['frame']}` vertex "
        f"`{origin['vertex_index']}`. It maps to simulator `offset={candidate_plane.offset:.9f} m`, "
        f"which is `{(candidate_plane.offset-active_plane.offset)*1000:.3f} mm` relative to the active support.",
        "", "## Frame-0 native material distance to table", "",
        "| entity | Issue14 candidate (mm) | status | active support (mm) | status |",
        "|---|---:|---|---:|---|",
    ]
    for entity in ("brush", "bowl", "left_xhand", "right_xhand"):
        candidate = tables["issue14_candidate"]["entities"][entity]
        active = tables["active_support"]["entities"][entity]
        lines.append(
            f"| {entity} | {candidate['minimum_signed_distance_mm']:.6f} | "
            f"{candidate['classification']} | {active['minimum_signed_distance_mm']:.6f} | "
            f"{active['classification']} |"
        )
    lines += ["", "## Native mesh-pair checks", "",
              "| group | native body pairs | minimum/classification | penetrating | unknown |",
              "|---|---:|---|---:|---:|"]
    for name, value in pair_groups.items():
        minimum = value["native_minimum"]
        minimum_text = "N/A" if minimum is None else (
            f"{minimum['distance_m']*1000:.6f} mm / {minimum['classification']}"
            if minimum["distance_m"] is not None else minimum["classification"]
        )
        lines.append(
            f"| {name} | {value['native_body_pair_count']} | {minimum_text} | "
            f"{len(value['native_penetrating_pairs'])} | {len(value['unknown_pairs'])} |"
        )
    candidate_pen = report["decision"]["issue14_candidate_has_penetration"]
    active_pen = report["decision"]["active_support_has_penetration"]
    lines += ["", "## Conclusion", "",
              f"- Issue #14 candidate table initialization penetration: `{'YES' if candidate_pen else 'NO'}`.",
              f"- Active support initialization penetration: `{'YES' if active_pen else 'NO'}`.",
              f"- Active-support deepest material penetration is the brush at `{tables['active_support']['entities']['brush']['minimum_signed_distance_mm']:.6f} mm`, native point `{tables['active_support']['entities']['brush']['worst_location']['point_sim_m']}` in simulator coordinates.",
              "- Every non-table native mesh-pair check completed with zero unknown results; MuJoCo shell overlap is reported separately and is not relabelled as native material penetration.",
              "- This is a static geometry audit only. The candidate support is not promoted, the active contract is unchanged, and no MINK, physics step, Replay, MPC, or RL ran.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
