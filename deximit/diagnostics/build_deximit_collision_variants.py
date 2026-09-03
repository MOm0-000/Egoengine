#!/usr/bin/env python3
"""Build isolated MuJoCo collision-representation variants.

Only diagnostic collision geoms are changed.  Visual meshes, body inertias,
drives, contact parameters, and the single fixed camera remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh


SCHEMA = (
    "deximit_collision_representation_variants_"
    "v6_hand_rigid_flex_diagnostic_only"
)
ACTIVE_CONTACT_LINKS = (
    "right_hand_thumb_rota_link2",
    "right_hand_mid_link2",
    "right_hand_ring_link2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sdf-iterations", type=int, default=10)
    parser.add_argument("--sdf-initpoints", type=int, default=40)
    parser.add_argument(
        "--object-cooked-mesh", type=Path,
        help="Add an oriented-box proxy derived from the exact cooked object hull.",
    )
    parser.add_argument(
        "--rigid-flex-margin-m", type=float, default=0.002,
        help="Broad-phase search margin for exact rigid-flex object variants.",
    )
    args = parser.parse_args()
    source = args.scene.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite collision variants {output}")
    if args.sdf_iterations <= 0 or args.sdf_initpoints <= 0:
        raise ValueError("SDF solver counts must be positive")

    if not np.isfinite(args.rigid_flex_margin_m) or args.rigid_flex_margin_m <= 0.0:
        raise ValueError("rigid-flex margin must be finite and positive")
    variants = {
        "object_sdf": (True, False, False, False, False, None, False),
        "hand_sdf": (False, True, False, False, False, None, False),
        "both_sdf": (True, True, False, False, False, None, False),
        "hand_zero_margin": (False, False, False, True, False, None, False),
    }
    box_proxy: dict[str, object] | None = None
    if args.object_cooked_mesh is not None:
        object_mesh = args.object_cooked_mesh.expanduser().resolve(strict=True)
        loaded = trimesh.load(object_mesh, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError("cooked object collision is not one triangle mesh")
        hull = loaded.convex_hull
        box = hull.bounding_box_oriented
        transform = np.asarray(box.primitive.transform, dtype=np.float64)
        extents = np.asarray(box.primitive.extents, dtype=np.float64)
        local_vertices = (
            np.c_[np.asarray(hull.vertices), np.ones(len(hull.vertices))]
            @ np.linalg.inv(transform).T
        )[:, :3]
        face_clearance = np.min(
            extents[None, :] / 2.0 - np.abs(local_vertices), axis=1,
        )
        equations = ConvexHull(np.asarray(hull.vertices)).equations
        corners = np.asarray(box.vertices)
        corner_outside = np.maximum(
            corners @ equations[:, :3].T + equations[:, 3], 0.0,
        ).max(axis=1)
        box_proxy = {
            "source": str(object_mesh),
            "source_sha256": sha256(object_mesh),
            "local_position_m": transform[:3, 3].tolist(),
            "local_quaternion_wxyz": Rotation.from_matrix(
                transform[:3, :3],
            ).as_quat(scalar_first=True).tolist(),
            "half_size_m": (extents / 2.0).tolist(),
            "hull_volume_m3": float(hull.volume),
            "box_volume_m3": float(np.prod(extents)),
            "box_to_hull_volume_ratio": float(np.prod(extents) / hull.volume),
            "hull_vertex_to_box_face_clearance_m_mean": float(
                face_clearance.mean(),
            ),
            "hull_vertex_to_box_face_clearance_m_max": float(
                face_clearance.max(),
            ),
            "box_corner_outside_hull_plane_distance_m_max": float(
                corner_outside.max(),
            ),
        }
        variants["object_obb"] = (
            False, False, True, False, False, None, False,
        )
        variants["object_rigid_flex"] = (
            False, False, False, False, True, 0, False,
        )
        variants["object_rigid_flex_refined"] = (
            False, False, False, False, True, 1, False,
        )
        variants["active_hand_rigid_flex"] = (
            False, False, False, False, False, None, True,
        )
    output.mkdir(parents=True)
    records: list[dict[str, object]] = []
    source_model = mujoco.MjModel.from_xml_path(str(source))
    source_object_body = mujoco.mj_name2id(
        source_model, mujoco.mjtObj.mjOBJ_BODY, "right_object",
    )
    source_object_mass = float(source_model.body_mass[source_object_body])
    source_object_inertia = source_model.body_inertia[source_object_body].copy()
    source_body_mass = source_model.body_mass.copy()
    source_body_inertia = source_model.body_inertia.copy()
    for name, (
        object_sdf, hand_sdf, object_obb, hand_zero_margin, object_rigid_flex,
        rigid_flex_subdivision, active_hand_rigid_flex,
    ) in variants.items():
        tree = ET.parse(source)
        root = tree.getroot()
        option = root.find("option")
        if option is None:
            raise ValueError("source scene lacks the bound option block")
        option.set("sdf_iterations", str(args.sdf_iterations))
        option.set("sdf_initpoints", str(args.sdf_initpoints))
        changed: list[str] = []
        for geom in root.iter("geom"):
            geom_name = geom.get("name", "")
            should_change = (
                object_sdf and geom_name == "right_object_single"
            ) or (
                hand_sdf and geom_name.startswith("collision_hand_")
            )
            if not should_change:
                continue
            if not geom.get("mesh"):
                raise ValueError(f"diagnostic SDF geom {geom_name!r} lacks a mesh")
            geom.set("type", "sdf")
            changed.append(geom_name)
        if object_obb:
            if box_proxy is None:
                raise RuntimeError("box variant lacks its fitted proxy")
            matches = [
                geom for geom in root.iter("geom")
                if geom.get("name") == "right_object_single"
            ]
            if len(matches) != 1:
                raise ValueError("source scene does not have one object collision geom")
            geom = matches[0]
            if geom.get("pos") is not None or geom.get("quat") is not None:
                raise ValueError("box proxy expects an identity source collision pose")
            geom.set("type", "box")
            geom.attrib.pop("mesh", None)
            geom.set("pos", " ".join(map(str, box_proxy["local_position_m"])))
            geom.set("quat", " ".join(map(str, box_proxy["local_quaternion_wxyz"])))
            geom.set("size", " ".join(map(str, box_proxy["half_size_m"])))
            changed.append("right_object_single")
        expected = (
            (1 if object_sdf else 0) + (13 if hand_sdf else 0)
            + (1 if object_obb else 0)
        )
        if len(changed) != expected:
            raise ValueError(
                f"variant {name} changed {len(changed)} geoms, expected {expected}",
            )
        changed_pairs: list[str] = []
        removed_pairs: list[str] = []
        if hand_zero_margin:
            contact = root.find("contact")
            if contact is None:
                raise ValueError("source scene lacks explicit contact pairs")
            for pair in contact.findall("pair"):
                geoms = (pair.get("geom1", ""), pair.get("geom2", ""))
                if (
                    "right_object_single" in geoms
                    and any(value.startswith("collision_hand_") for value in geoms)
                ):
                    pair.set("margin", "0")
                    pair.set("gap", "0")
                    changed_pairs.append(pair.get("name", ""))
            if len(changed_pairs) != 13:
                raise ValueError(
                    f"zero-margin variant changed {len(changed_pairs)} pairs, expected 13",
                )
        if object_rigid_flex:
            if args.object_cooked_mesh is None:
                raise RuntimeError("rigid-flex variant lacks its exact cooked mesh")
            object_bodies = [
                body for body in root.iter("body")
                if body.get("name") == "right_object"
            ]
            if len(object_bodies) != 1:
                raise ValueError("source scene does not have one object body")
            flex_mesh = loaded.copy()
            assert rigid_flex_subdivision is not None
            for _ in range(rigid_flex_subdivision):
                flex_mesh = flex_mesh.subdivide()
            if rigid_flex_subdivision:
                flex_mesh_path = output / (
                    f"right_object_flex_subdiv_{rigid_flex_subdivision}.obj"
                )
                flex_mesh.export(flex_mesh_path)
            else:
                flex_mesh_path = args.object_cooked_mesh.expanduser().resolve(
                    strict=True,
                )
            flexcomp = ET.SubElement(object_bodies[0], "flexcomp", {
                "name": "right_object_flex", "type": "mesh",
                "file": str(flex_mesh_path),
                "rigid": "true", "radius": "0", "rgba": "0 1 0 0",
            })
            ET.SubElement(flexcomp, "contact", {
                "internal": "false", "selfcollide": "none",
                "contype": "2", "conaffinity": "1", "condim": "3",
                "priority": "10", "friction": "0.5 0 0",
                "solref": "0.00035 4", "solimp": "0.95 0.95 0.001 0.5 2",
                "margin": str(args.rigid_flex_margin_m),
                "gap": str(args.rigid_flex_margin_m),
            })
            for geom in root.iter("geom"):
                if geom.get("name", "").startswith("collision_hand_"):
                    geom.set("contype", "1")
                    geom.set("conaffinity", "2")
            contact = root.find("contact")
            if contact is None:
                raise ValueError("source scene lacks explicit contact pairs")
            for pair in list(contact.findall("pair")):
                geoms = (pair.get("geom1", ""), pair.get("geom2", ""))
                if (
                    "right_object_single" in geoms
                    and any(value.startswith("collision_hand_") for value in geoms)
                ):
                    removed_pairs.append(pair.get("name", ""))
                    contact.remove(pair)
            if len(removed_pairs) != 13:
                raise ValueError(
                    f"rigid-flex variant removed {len(removed_pairs)} pairs, expected 13",
                )
        if active_hand_rigid_flex:
            assets = {
                mesh.get("name", ""): mesh
                for mesh in root.find("asset").findall("mesh")
            }
            object_geoms = [
                geom for geom in root.iter("geom")
                if geom.get("name") == "right_object_single"
            ]
            if len(object_geoms) != 1:
                raise ValueError("hand flex variant lacks one object geom")
            object_geoms[0].set("contype", "2")
            object_geoms[0].set("conaffinity", "1")
            for link_name in ACTIVE_CONTACT_LINKS:
                bodies = [
                    body for body in root.iter("body")
                    if body.get("name") == link_name
                ]
                geoms = [
                    geom for geom in root.iter("geom")
                    if geom.get("name") == f"collision_hand_{link_name}"
                ]
                if len(bodies) != 1 or len(geoms) != 1:
                    raise ValueError(f"hand flex source is ambiguous for {link_name}")
                mesh_name = geoms[0].get("mesh", "")
                mesh_asset = assets.get(mesh_name)
                if mesh_asset is None or not mesh_asset.get("file"):
                    raise ValueError(f"hand flex mesh is missing for {link_name}")
                flexcomp = ET.SubElement(bodies[0], "flexcomp", {
                    "name": f"{link_name}_flex", "type": "mesh",
                    "file": mesh_asset.get("file", ""),
                    "rigid": "true", "radius": "0", "rgba": "0 1 0 0",
                })
                ET.SubElement(flexcomp, "contact", {
                    "internal": "false", "selfcollide": "none",
                    "contype": "1", "conaffinity": "2", "condim": "3",
                    "priority": "10", "friction": "0.5 0 0",
                    "solref": "0.00035 4",
                    "solimp": "0.95 0.95 0.001 0.5 2",
                    "margin": str(args.rigid_flex_margin_m),
                    "gap": str(args.rigid_flex_margin_m),
                })
            contact = root.find("contact")
            if contact is None:
                raise ValueError("source scene lacks explicit contact pairs")
            for pair in list(contact.findall("pair")):
                geoms = (pair.get("geom1", ""), pair.get("geom2", ""))
                if (
                    "right_object_single" in geoms
                    and any(
                        f"collision_hand_{link_name}" in geoms
                        for link_name in ACTIVE_CONTACT_LINKS
                    )
                ):
                    removed_pairs.append(pair.get("name", ""))
                    contact.remove(pair)
            if len(removed_pairs) != len(ACTIVE_CONTACT_LINKS):
                raise ValueError(
                    "active hand flex variant did not replace exactly three pairs",
                )
        path = output / f"{name}.xml"
        ET.indent(tree, space="  ")
        tree.write(path, encoding="unicode", xml_declaration=False)
        model = mujoco.MjModel.from_xml_path(str(path))
        cameras = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
            for index in range(model.ncam)
        ]
        object_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_object",
        )
        mass_unchanged = float(model.body_mass[object_body]) == source_object_mass
        inertia_unchanged = bool(np.array_equal(
            model.body_inertia[object_body], source_object_inertia,
        ))
        all_body_inertias_unchanged = bool(
            np.array_equal(model.body_mass, source_body_mass)
            and np.array_equal(model.body_inertia, source_body_inertia)
        )
        if (
            model.nq != 25 or model.nv != 24 or model.nu != 18
            or cameras != ["front"] or not mass_unchanged or not inertia_unchanged
            or not all_body_inertias_unchanged
        ):
            raise RuntimeError(
                f"compiled variant contract failed for {name}: "
                f"nq={model.nq}, nv={model.nv}, nu={model.nu}, cameras={cameras}, "
                f"mass={mass_unchanged}, inertia={inertia_unchanged}",
            )
        records.append({
            "name": name,
            "scene": str(path),
            "scene_sha256": sha256(path),
            "object_sdf": object_sdf,
            "hand_sdf": hand_sdf,
            "object_obb": object_obb,
            "hand_zero_margin": hand_zero_margin,
            "object_rigid_flex": object_rigid_flex,
            "active_hand_rigid_flex": active_hand_rigid_flex,
            "active_hand_rigid_flex_links": (
                list(ACTIVE_CONTACT_LINKS) if active_hand_rigid_flex else []
            ),
            "rigid_flex_subdivision_level": rigid_flex_subdivision,
            "rigid_flex_margin_m": (
                args.rigid_flex_margin_m if object_rigid_flex else None
            ),
            "rigid_flex_vertex_count": (
                int(model.flex_vertnum[0]) if object_rigid_flex else 0
            ),
            "rigid_flex_element_count": (
                int(model.flex_elemnum[0]) if object_rigid_flex else 0
            ),
            "changed_collision_geoms": changed,
            "changed_contact_pairs": changed_pairs,
            "removed_contact_pairs": removed_pairs,
            "compiled_ngeom": int(model.ngeom),
            "compiled_nflex": int(model.nflex),
            "object_mass_unchanged": mass_unchanged,
            "object_inertia_unchanged": inertia_unchanged,
            "all_body_masses_and_inertias_unchanged": (
                all_body_inertias_unchanged
            ),
            "camera": {"count": 1, "name": "front", "mode": "fixed"},
        })

    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "source_scene": str(source),
        "source_scene_sha256": sha256(source),
        "sdf_iterations": args.sdf_iterations,
        "sdf_initpoints": args.sdf_initpoints,
        "rigid_flex_margin_m": args.rigid_flex_margin_m,
        "object_box_proxy": box_proxy,
        "variants": records,
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
