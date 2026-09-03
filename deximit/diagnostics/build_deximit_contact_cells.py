#!/usr/bin/env python3
"""Split active convex fingertip hulls into exact diagnostic contact cells.

MuJoCo returns at most a small manifold for one convex pair.  PhysX PCM keeps
more points for the same pair.  This isolated variant partitions each selected
convex fingertip into surface-triangle tetrahedra sharing one interior point.
The cells exactly tile the original convex volume; body mass, inertia, visuals,
drives, object, table, and camera are not changed.

The variant is diagnostic because independent cell pairs can also expose
internal cell faces after penetration.  It must pass identical-state manifold
and one-step audits before it is allowed to advance a grasp candidate.
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
import trimesh


SCHEMA = "deximit_contact_cells_v1_diagnostic_only"
DEFAULT_LINKS = (
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


def one(root: ET.Element, tag: str, attr: str, value: str) -> ET.Element:
    rows = [node for node in root.iter(tag) if node.get(attr) == value]
    if len(rows) != 1:
        raise ValueError(f"expected one {tag} with {attr}={value!r}, got {len(rows)}")
    return rows[0]


def mesh_path(scene: Path, root: ET.Element, asset: ET.Element) -> Path:
    value = asset.get("file")
    if not value:
        raise ValueError(f"mesh asset {asset.get('name')!r} has no file")
    path = Path(value).expanduser()
    if not path.is_absolute():
        compiler = root.find("compiler")
        mesh_dir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
        path = (scene.parent / mesh_dir / path).resolve()
    return path.resolve(strict=True)


def write_tetra(path: Path, vertices: np.ndarray) -> None:
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))),
        process=False,
    )
    mesh.fix_normals()
    mesh.export(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--link", action="append", dest="links",
        help="Selected fingertip body; repeat as needed (default: three active tips).",
    )
    args = parser.parse_args()
    source = args.scene.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    links = tuple(args.links or DEFAULT_LINKS)
    if len(set(links)) != len(links):
        raise ValueError("selected links must be unique")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite contact-cell variant {output}")

    source_model = mujoco.MjModel.from_xml_path(str(source))
    source_mass = source_model.body_mass.copy()
    source_inertia = source_model.body_inertia.copy()
    tree = ET.parse(source)
    root = tree.getroot()
    assets = root.find("asset")
    contact = root.find("contact")
    if assets is None or contact is None:
        raise ValueError("source scene lacks asset/contact blocks")

    output.mkdir(parents=True)
    mesh_dir = output / "cells"
    mesh_dir.mkdir()
    link_reports: list[dict[str, object]] = []
    total_cells = 0
    for link in links:
        body = one(root, "body", "name", link)
        geom_name = f"collision_hand_{link}"
        geom = one(root, "geom", "name", geom_name)
        mesh_name = geom.get("mesh")
        if not mesh_name:
            raise ValueError(f"selected geom {geom_name!r} is not a mesh")
        mesh_asset = one(assets, "mesh", "name", mesh_name)
        source_mesh = mesh_path(source, root, mesh_asset)
        loaded = trimesh.load(source_mesh, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"collision asset for {link!r} is not one mesh")
        hull = loaded.convex_hull
        center = np.asarray(hull.center_mass, dtype=np.float64)
        if not hull.contains(center[None, :])[0]:
            center = np.mean(np.asarray(hull.vertices), axis=0)
        if not hull.contains(center[None, :])[0]:
            raise ValueError(f"could not find an interior point for {link!r}")

        parent = next(node for node in root.iter() if geom in list(node))
        insert_at = list(parent).index(geom)
        parent.remove(geom)
        pair_templates = [
            pair for pair in contact.findall("pair")
            if geom_name in (pair.get("geom1"), pair.get("geom2"))
        ]
        if len(pair_templates) != 2:
            raise ValueError(
                f"selected geom {geom_name!r} must have object and table pairs; "
                f"found {len(pair_templates)}",
            )
        for pair in pair_templates:
            contact.remove(pair)

        cell_volumes: list[float] = []
        for index, face in enumerate(np.asarray(hull.faces, dtype=np.int32)):
            vertices = np.vstack((np.asarray(hull.vertices)[face], center))
            cell_path = mesh_dir / f"{link}_{index:03d}.obj"
            write_tetra(cell_path, vertices)
            cell_mesh = trimesh.load(cell_path, force="mesh", process=False)
            cell_volumes.append(abs(float(cell_mesh.volume)))
            cell_mesh_name = f"contact_cell_{link}_{index:03d}"
            ET.SubElement(assets, "mesh", {
                "name": cell_mesh_name,
                "file": str(cell_path),
            })
            cell_name = f"{geom_name}_cell_{index:03d}"
            attrs = dict(geom.attrib)
            attrs["name"] = cell_name
            attrs["mesh"] = cell_mesh_name
            parent.insert(insert_at + index, ET.Element("geom", attrs))
            for template in pair_templates:
                pair_attrs = dict(template.attrib)
                pair_attrs["name"] = f"{template.get('name')}_cell_{index:03d}"
                if pair_attrs.get("geom1") == geom_name:
                    pair_attrs["geom1"] = cell_name
                if pair_attrs.get("geom2") == geom_name:
                    pair_attrs["geom2"] = cell_name
                contact.append(ET.Element("pair", pair_attrs))

        cell_volume = float(np.sum(cell_volumes))
        hull_volume = abs(float(hull.volume))
        rel_error = abs(cell_volume - hull_volume) / hull_volume
        if rel_error > 1.0e-9:
            raise RuntimeError(
                f"contact cells do not tile {link}: relative volume error {rel_error}",
            )
        total_cells += len(hull.faces)
        link_reports.append({
            "link": link,
            "source_geom": geom_name,
            "source_mesh": str(source_mesh),
            "source_mesh_sha256": sha256(source_mesh),
            "cell_count": int(len(hull.faces)),
            "convex_hull_volume_m3": hull_volume,
            "cell_volume_sum_m3": cell_volume,
            "relative_volume_error": rel_error,
            "interior_point_m": center.tolist(),
            "outer_surface_uses_exact_hull_vertices": True,
        })

    path = output / "scene.xml"
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=False)
    model = mujoco.MjModel.from_xml_path(str(path))
    cameras = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
        for index in range(model.ncam)
    ]
    mass_exact = bool(np.array_equal(model.body_mass, source_mass))
    inertia_exact = bool(np.array_equal(model.body_inertia, source_inertia))
    if (
        model.nq != source_model.nq or model.nv != source_model.nv
        or model.nu != source_model.nu or cameras != ["front"]
        or not mass_exact or not inertia_exact
    ):
        raise RuntimeError("compiled contact-cell scene changed the physical contract")
    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "rendered": False,
        "source_scene": str(source),
        "source_scene_sha256": sha256(source),
        "scene": str(path),
        "scene_sha256": sha256(path),
        "selected_links": list(links),
        "links": link_reports,
        "cell_count": total_cells,
        "compiled_ngeom": int(model.ngeom),
        "compiled_npair": int(model.npair),
        "all_body_masses_exact": mass_exact,
        "all_body_inertias_exact": inertia_exact,
        "camera": {"count": 1, "name": "front", "mode": "fixed"},
        "known_risk": (
            "independent convex cells may expose shared internal faces after "
            "penetration; identical-state and one-step audits are mandatory"
        ),
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
