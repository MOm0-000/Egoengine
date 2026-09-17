#!/usr/bin/env python3
"""Materialize an isolated TACO bimanual xHand scene.

The Spider/OakInk bimanual xHand scene is used only as a robot/XML skeleton.
The object bodies are replaced with explicitly selected TACO object meshes;
no source project files are modified.  The generated scene keeps the naming
contract consumed by the renderer and RL adapter: right/left hand branches
and right/left passive free-object joints (seven qpos, six qvel each).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from itertools import product
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_TEMPLATE = Path(
    "/data_all/zzx/egoengine/spider/example_datasets/processed/"
    "oakink/xhand/bimanual/wipe_board/scene_act.xml"
)
DEFAULT_TEMPLATE = ROOT / "models/taco_xhand/templates/xhand_bimanual_source.xml"
DEFAULT_ASSETS = ROOT / "models" / "taco_xhand" / "assets"
DEFAULT_OUTPUT = (
    ROOT
    / "models"
    / "taco_xhand"
    / "xhand"
    / "bimanual"
    / "taco_brush_brush_bowl_20230927_027"
    / "scene_source_contacts_mass.xml"
)


def _mesh(parent: ET.Element, *, name: str, filename: str) -> None:
    ET.SubElement(parent, "mesh", {"name": name, "file": filename})


def selected_objects(data_root: Path | None = None, tool_id: str | None = None,
                     target_id: str | None = None) -> dict:
    if data_root is None:
        if tool_id is not None or target_id is not None:
            raise ValueError("object IDs require --data-root")
        data_root = ROOT / "data/taco_v1/dev4"
        specs = {"right": ("071", 1.0, "convex"), "left": ("146", 0.01, "convex_m")}
    else:
        if any(value is None or len(value) != 3 or not value.isdigit() for value in (tool_id, target_id)):
            raise ValueError("explicit data root requires three-digit tool and target IDs")
        specs = {"right": (tool_id, 0.01, "convex_m"), "left": (target_id, 0.01, "convex_m")}
    return {side: dict(object_id=obj, visual_scale=scale, collision_subdir=parts,
                       source=data_root / "object_models/object_models_released" / f"{obj}_cm.obj")
            for side, (obj, scale, parts) in specs.items()}


def _replace_object_assets(root: ET.Element, assets: Path, objects: dict) -> None:
    asset = root.find("asset")
    if asset is None:
        raise ValueError("template has no <asset> section")

    # Keep robot meshes and materials; replace all object mesh declarations.
    for element in list(asset):
        name = element.attrib.get("name", "")
        if name.startswith(("right_visual", "right_", "left_visual", "left_")):
            filename = element.attrib.get("file", "")
            if filename.startswith("objects/"):
                asset.remove(element)

    for side, spec in objects.items():
        relative = Path("objects") / spec["object_id"]
        directory = assets / relative
        visual = directory / "visual.obj"
        if spec["visual_scale"] == 0.01:
            source_bytes = spec["source"].read_bytes()
            if visual.exists():
                if visual.is_symlink() or visual.read_bytes() != source_bytes:
                    raise ValueError(f"existing visual differs from selected source: {visual}")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                with visual.open("xb") as stream:
                    stream.write(source_bytes)
        elif not visual.is_file():
            raise FileNotFoundError(visual)
        ET.SubElement(asset, "mesh", dict(name=f"{side}_visual", file=str(relative / "visual.obj"),
                                          scale=" ".join([str(spec["visual_scale"])] * 3)))
        parts = sorted((directory / spec["collision_subdir"]).glob("*.obj"), key=lambda p: int(p.stem))
        if not parts or (spec["collision_subdir"] == "convex_m" and len(parts) < 2):
            raise ValueError(f"prepare and audit collision parts before scene construction: {directory}")
        for path in parts:
            _mesh(asset, name=f"{side}_{path.stem}", filename=str(relative / spec["collision_subdir"] / path.name))


def _replace_object_geoms(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("template has no <worldbody> section")

    right = worldbody.find("body[@name='right_object']")
    left = worldbody.find("body[@name='left_object']")
    if right is None or left is None:
        raise ValueError("template must contain right_object and left_object bodies")

    def reset_body(body: ET.Element, side: str, meshes: list[str]) -> None:
        for child in list(body):
            if child.tag == "joint":
                body.remove(child)
        ET.SubElement(body, "freejoint", {"name": f"{side}_object_joint"})
        for child in list(body):
            if child.tag == "geom":
                body.remove(child)
        # Physical pairs are assigned centrally in _replace_contacts.
        for index, mesh_name in enumerate(meshes):
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{side}_object_{index}",
                    "type": "mesh",
                    "group": "3",
                    "density": "1000",
                    "contype": "0",
                    "conaffinity": "0",
                    "condim": "3",
                    "friction": "1 0.1 0",
                    "rgba": "0 1 0 1",
                    "mesh": mesh_name,
                },
            )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{side}_object_visual",
                "type": "mesh",
                "density": "0",
                "contype": "0",
                "conaffinity": "0",
                "rgba": "1 1 1 1",
                "mesh": f"{side}_visual",
            },
        )

    for body, side in ((right, "right"), (left, "left")):
        names = [mesh.get("name") for mesh in root.findall("asset/mesh")
                 if mesh.get("name", "").startswith(f"{side}_")
                 and mesh.get("name", "").split("_", 1)[1].isdigit()]
        reset_body(body, side, names)
        for site in body.findall("site"):
            if site.get("name", "").startswith("track_object_"):
                site.set("pos", "0 0 0")

    # Explicit runtime pairs below are the single collision contract. Broad
    # same-hand masks would reintroduce the source shells' structural overlaps.
    for geom in worldbody.iter("geom"):
        name = geom.attrib.get("name", "")
        if name == "floor" or name.startswith("collision_hand_"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("condim", "3")
            geom.set("friction", "1 0.1 0")
        if name == "floor":
            geom.set("pos", "0 0 0.72")


def _replace_contacts(root: ET.Element) -> dict:
    contact = root.find("contact")
    source_self = []
    if contact is not None:
        for pair in contact.findall("pair"):
            a, b = pair.get("geom1", ""), pair.get("geom2", "")
            if (a.startswith("collision_hand_") and b.startswith("collision_hand_")
                    and a.split("_")[2] == b.split("_")[2]):
                source_self.append((a, b))
        root.remove(contact)
    if len(source_self) != 30:
        raise ValueError("expected the pinned template's 15 explicit self pairs per hand")
    contact = ET.SubElement(root, "contact")
    hands, objects = [], []
    for side in ("right", "left"):
        hands.append([g.get("name") for g in root.findall(".//geom")
                      if g.get("name", "").startswith(f"collision_hand_{side}_")])
        objects.append([g.get("name") for g in root.findall(".//geom")
                        if g.get("name", "").startswith(f"{side}_object_")
                        and not g.get("name").endswith("visual")])
    families = {
        "source_intrahand": source_self,
        "full_interhand": list(product(*hands)),
        "hand_object": list(product(hands[0] + hands[1], objects[0] + objects[1])),
        "object_object": list(product(*objects)),
        "floor": [(g, "floor") for group in hands + objects for g in group],
    }
    for geom in root.findall(".//geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    for family, pairs in families.items():
        for index, (a, b) in enumerate(pairs):
            ET.SubElement(contact, "pair", dict(name=f"{family}_{index}", geom1=a, geom2=b,
                          condim="3", friction="1 1 0.1 0 0"))
    return dict(policy="pinned_source_intrahand_plus_full_interhand",
                pair_counts={name: len(pairs) for name, pairs in families.items()},
                geometry_unchanged=True,
                complete_intrahand_geometry_coverage=False,
                limitation="source self model tests palm/distal and distal/distal only; omitted shell pairs audited separately")


def _keep_fixed_front_camera(root: ET.Element) -> None:
    front = root.find(".//camera[@name='front']")
    if front is None:
        raise ValueError("template lacks front camera")
    front.set("mode", "fixed")
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "camera" and child is not front:
                parent.remove(child)


def _remove_object_actuators(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        return
    for element in list(actuator):
        joint = element.attrib.get("joint", "")
        name = element.attrib.get("name", "")
        if "object" in joint or "object" in name:
            actuator.remove(element)


def _set_object_inertials(root: ET.Element, objects: dict | None = None) -> dict:
    report = {}
    for side, spec in (objects or selected_objects()).items():
        path = spec["source"]
        mesh = trimesh.load_mesh(path, process=True)
        mesh.apply_scale(0.01)
        if not mesh.is_watertight or mesh.volume <= 0:
            raise ValueError(f"{path}: a closed oriented mesh is required for inertial integration")
        mesh.density = 1000.0
        inertia = mesh.moment_inertia
        if not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0:
            raise ValueError(f"{path}: invalid integrated inertia")
        body = root.find(f"worldbody/body[@name='{side}_object']")
        for element in list(body):
            if element.tag == "inertial":
                body.remove(element)
            if element.tag == "geom":
                element.set("density", "0")
        ET.SubElement(body, "inertial", dict(
            pos=" ".join(map(str, mesh.center_mass)), mass=str(mesh.mass),
            fullinertia=" ".join(map(str, inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))))
        report[side] = dict(source=str(path.resolve()),
                            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            mass_kg=float(mesh.mass), com_m=mesh.center_mass.tolist(),
                            inertia_kg_m2=inertia.tolist(), density_kg_m3=1000.0,
                            density_provenance="inherited nominal value, not a measured TACO/paper mass",
                            method="closed visual-mesh volume integral; no sum of overlapping convex parts")
    return report


def snapshot_robot_assets(root: ET.Element, assets: Path, name: str) -> list[dict]:
    """Resolve old links once and freeze only the robot files used by this scene."""
    if not name or Path(name).name != name or name in (".", "..", "xhand"):
        raise ValueError("robot snapshot requires a new single directory name")
    destination = assets / "robots" / name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    entries = []
    for mesh in root.findall("asset/mesh"):
        relative = Path(mesh.get("file", ""))
        if relative.parts[:2] != ("robots", "xhand"):
            continue
        if ".." in relative.parts:
            raise ValueError(f"unexpected robot asset path: {relative}")
        source = (assets / relative).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"robot mesh is not a file: {source}")
        local = Path("robots") / name / Path(*relative.parts[2:])
        entries.append((mesh, source, local))
    if not entries:
        raise ValueError("template has no expected XHand mesh assets")
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for mesh, source, relative in entries:
        data = source.read_bytes()
        output = assets / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(data)
        mesh.set("file", str(relative))
        records.append(dict(source=str(source), output=str(output.resolve()),
                            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    return records


def build(template: Path, output: Path, assets: Path, *, objects: dict | None = None,
          model_name: str = "taco_brush_brush_bowl_bimanual",
          robot_snapshot_name: str | None = None) -> None:
    if output.exists() or output.with_suffix(".build.json").exists():
        raise FileExistsError(f"preserve generated scene; choose a new --output: {output}")
    if template == DEFAULT_TEMPLATE and not template.exists():
        template.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(UPSTREAM_TEMPLATE, template)
    tree = ET.parse(template)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("template lacks compiler mesh directory")
    compiler.set("meshdir", os.path.relpath(assets.resolve(), output.resolve().parent))
    objects = objects or selected_objects()
    root.set("model", model_name)
    _replace_object_assets(root, assets, objects)
    _replace_object_geoms(root)
    collision_policy = _replace_contacts(root)
    _remove_object_actuators(root)
    inertials = _set_object_inertials(root, objects)
    _keep_fixed_front_camera(root)
    snapshot = snapshot_robot_assets(root, assets, robot_snapshot_name) if robot_snapshot_name else []
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)
    report = dict(template=str(template.resolve()), template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
                  upstream_template=str(UPSTREAM_TEMPLATE), table_height_m=0.72,
                  object_roles={"right_object": f"tool_{objects['right']['object_id']}",
                                "left_object": f"target_{objects['left']['object_id']}"},
                  passive_objects=True, hand_control_dof=36,
                  object_assets={side: {**spec, "source": str(spec["source"].resolve())}
                                 for side, spec in objects.items()},
                  collision_units="m", collision_provenance="inherited local CoACD pipeline; approximation requires audit; not author geometry",
                  contact_condim=3, contact_friction=[1, 0.1, 0],
                  collision_policy=collision_policy,
                  independent_robot_asset_snapshot=snapshot,
                  object_inertials=inertials,
                  source_contact_sites_replaced_with_zero_placeholders=True,
                  physics_validated=False)
    output.with_suffix(".build.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--tool-id")
    parser.add_argument("--target-id")
    parser.add_argument("--model-name", default="taco_brush_brush_bowl_bimanual")
    parser.add_argument("--robot-snapshot-name")
    args = parser.parse_args()
    build(args.template, args.output, args.assets,
          objects=selected_objects(args.data_root, args.tool_id, args.target_id), model_name=args.model_name,
          robot_snapshot_name=args.robot_snapshot_name)
    print(args.output)


if __name__ == "__main__":
    main()
