"""Build an isolated palm/thumb guard from inward native-triangle prisms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import fcl
import mujoco
import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from audit_taco_initialization import visual_meshes
from egoengine_repro.retarget.paper_audit import artifact, verify_artifacts
from fit_taco_left_palm_thumb_boundary_cegis import _extract_rows
from fit_taco_left_palm_thumb_semantic_guard import Oracle

DEFAULT_PROTOCOL = ROOT / "configs/taco_pour_left_palm_thumb_local_surface_guard_v1.yaml"
DEFAULT_OUTPUT = ROOT / "runs/taco_pour_left_palm_thumb_local_surface_guard_v1"


def _fmt(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _collect_pairs(oracle: Oracle, rows: list[dict], offsets: list[float]) -> set[tuple[int, int]]:
    result = set()
    request = fcl.CollisionRequest(num_max_contacts=512, enable_contact=True)
    for row in rows:
        intervals = row["native"]["collision_intervals_rad"]
        if not intervals:
            continue
        bend = float(row["thumb_bend_rad"])
        boundary = float(intervals[0][0])
        for offset in offsets:
            rotation, translation = oracle.transform(bend, boundary + offset)
            oracle.thumb.setTransform(fcl.Transform(rotation, translation))
            collision = fcl.CollisionResult()
            fcl.collide(oracle.palm, oracle.thumb, request, collision)
            if not collision.is_collision or not collision.contacts:
                raise RuntimeError("native collision boundary probe returned no triangle contacts")
            result.update((int(contact.b1), int(contact.b2))
                          for contact in collision.contacts)
    return result


def _prism(mesh: trimesh.Trimesh, face: int, thickness: float,
           shrink_fraction: float = 0.0) -> trimesh.Trimesh:
    triangle = np.asarray(mesh.vertices[mesh.faces[face]], dtype=float)
    centre = triangle.mean(axis=0)
    triangle = centre + (1.0 - shrink_fraction) * (triangle - centre)
    normal = np.asarray(mesh.face_normals[face], dtype=float)
    points = np.concatenate([triangle, triangle - thickness * normal])
    prism = trimesh.convex.convex_hull(points, qhull_options="QJ")
    if not prism.is_watertight or prism.volume <= 0.0:
        raise ValueError(f"face {face} did not form a positive watertight prism")
    return prism


def run(protocol_path: Path, output: Path) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text())
    verify_artifacts(protocol["inputs"].values())
    scene = Path(protocol["inputs"]["scene"]["path"])
    reference = Path(protocol["inputs"]["robot_reference"]["path"])
    topology = json.loads(Path(protocol["inputs"]["topology_v1"]["path"]).read_text())
    oracle = Oracle(scene, reference)

    construction = protocol["construction"]
    fractions = list(map(float, construction["calibration_bend_fractions"]))
    if fractions != [0.0, 0.5]:
        raise ValueError("v1 requires primary and midpoint calibration bends")
    offsets = list(map(float, construction["native_contact_offsets_inside_rad"]))
    primary_rows = topology["slice_results"]
    primary_bends = np.linspace(0.0, 1.83, 257)
    midpoint_bends = (primary_bends[:-1] + primary_bends[1:]) / 2.0
    rotas = np.linspace(-1.05, 1.57, 513)
    midpoint_rows = _extract_rows(oracle, None, midpoint_bends, rotas, 1e-7, 24)
    face_pairs = _collect_pairs(oracle, primary_rows, offsets)
    face_pairs |= _collect_pairs(oracle, midpoint_rows, offsets)
    palm_faces = sorted({pair[0] for pair in face_pairs})
    thumb_faces = sorted({pair[1] for pair in face_pairs})

    source_model = mujoco.MjModel.from_xml_path(str(scene))
    meshes, _ = visual_meshes(scene, source_model)
    palm_mesh = meshes[source_model.geom("left_hand_link_visual").id]
    thumb_mesh = meshes[source_model.geom("left_thumb_rota1_visual").id]
    if max(palm_faces) >= len(palm_mesh.faces) or max(thumb_faces) >= len(thumb_mesh.faces):
        raise ValueError("FCL triangle id is outside its native source mesh")

    output.mkdir(parents=True, exist_ok=True)
    asset_root = output / "assets"
    palm_dir, thumb_dir = asset_root / "palm", asset_root / "thumb"
    palm_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thickness = float(construction["inward_prism_thickness_m"])
    shrink = float(construction["in_plane_triangle_shrink_fraction"])
    if not 0.0 <= shrink < 1.0:
        raise ValueError("in-plane triangle shrink must be in [0,1)")
    expected_paths = set()
    for face in palm_faces:
        path = palm_dir / f"face_{face:05d}.obj"
        _prism(palm_mesh, face, thickness, shrink).export(path)
        expected_paths.add(path.resolve())
    for face in thumb_faces:
        path = thumb_dir / f"face_{face:05d}.obj"
        _prism(thumb_mesh, face, thickness, shrink).export(path)
        expected_paths.add(path.resolve())
    actual_paths = {path.resolve() for path in asset_root.rglob("*.obj")}
    if actual_paths != expected_paths:
        raise ValueError("surface-guard asset directory contains stale or missing files")

    tree = ET.parse(scene)
    root = tree.getroot()
    assets = root.find("asset")
    contact_table = root.find("contact")
    palm_body = root.find(".//body[@name='left_hand_link']")
    thumb_body = root.find(".//body[@name='left_hand_thumb_rota_link1']")
    if None in (assets, contact_table, palm_body, thumb_body):
        raise ValueError("source scene is missing required XML sections or bodies")
    compiler = root.find("compiler")
    meshdir = (scene.parent / compiler.get("meshdir", ".")).resolve()
    compiler.set("meshdir", str(meshdir))

    palm_names, thumb_names = {}, {}
    for side, faces, directory, body, names in (
        ("palm", palm_faces, palm_dir, palm_body, palm_names),
        ("thumb", thumb_faces, thumb_dir, thumb_body, thumb_names),
    ):
        for face in faces:
            mesh_name = f"left_palm_thumb_guard_{side}_mesh_{face:05d}"
            geom_name = f"collision_hand_left_palm_thumb_surface_guard_{side}_{face:05d}"
            path = directory / f"face_{face:05d}.obj"
            ET.SubElement(
                assets, "mesh", name=mesh_name,
                file=os.path.relpath(path.resolve(), meshdir),
            )
            ET.SubElement(
                body, "geom", name=geom_name, type="mesh", mesh=mesh_name,
                group="3", density="0", rgba="1 0.35 0.1 0.25",
                contype="0", conaffinity="0", condim="3", friction="1 0.1 0",
            )
            names[face] = geom_name

    pair_records = []
    for index, (palm_face, thumb_face) in enumerate(sorted(face_pairs)):
        name = f"semantic_self_left_palm_thumb_surface_{index:03d}"
        ET.SubElement(
            contact_table, "pair", name=name,
            geom1=palm_names[palm_face], geom2=thumb_names[thumb_face],
            condim="3", friction="1 1 0.1 0 0",
        )
        pair_records.append({
            "name": name,
            "palm_face": palm_face,
            "thumb_face": thumb_face,
            "palm_geom": palm_names[palm_face],
            "thumb_geom": thumb_names[thumb_face],
        })

    ET.indent(tree, space="  ")
    candidate = output / "candidate_self_guard_scene.xml"
    tree.write(candidate, encoding="unicode")
    compiled = mujoco.MjModel.from_xml_path(str(candidate))
    report = {
        "status": "isolated_local_surface_self_guard_built_not_promoted",
        "protocol": artifact(protocol_path),
        "source_scene": artifact(scene),
        "candidate_scene": artifact(candidate),
        "construction": {
            "inward_prism_thickness_m": thickness,
            "in_plane_triangle_shrink_fraction": shrink,
            "palm_face_geoms": len(palm_faces),
            "thumb_face_geoms": len(thumb_faces),
            "observed_native_triangle_pairs": len(pair_records),
            "calibration_bend_slices": len(primary_rows) + len(midpoint_rows),
            "contact_offsets_inside_rad": offsets,
            "compiled_ngeom": int(compiled.ngeom),
            "compiled_npair": int(compiled.npair),
        },
        "palm_assets": [artifact(palm_dir / f"face_{face:05d}.obj") for face in palm_faces],
        "thumb_assets": [artifact(thumb_dir / f"face_{face:05d}.obj") for face in thumb_faces],
        "guard_pairs": pair_records,
        "formal_scene_modified": False,
        "reference_modified": False,
        "training_ready": False,
        "code": artifact(Path(__file__)),
    }
    (output / "build_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": report["status"], **report["construction"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.protocol, args.output)
