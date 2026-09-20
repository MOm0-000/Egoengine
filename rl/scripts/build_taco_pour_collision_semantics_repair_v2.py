"""Build the runtime-only external-geometry candidate for repair protocol v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from audit_taco_initialization import visual_meshes
PROTOCOL = ROOT / "configs/taco_pour_collision_semantics_repair_v2.yaml"
OUTPUT = ROOT / "runs/taco_pour_collision_semantics_repair_v2"


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(values):
    return " ".join(f"{float(value):.12g}" for value in values)


def run(protocol_path: Path, output: Path):
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol["protocol_name"] != "taco_pour_collision_semantics_repair_v2":
        raise ValueError("unsupported protocol")
    if protocol["training_ready"] is not False:
        raise ValueError("repair candidate must remain training-blocked")
    for row in protocol["inputs"].values():
        if _sha(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"frozen input changed: {row['path']}")

    scene = Path(protocol["inputs"]["scene"]["path"])
    tree = ET.parse(scene)
    root = tree.getroot()
    source_model = mujoco.MjModel.from_xml_path(str(scene))
    native_meshes, _ = visual_meshes(scene, source_model)
    contact = root.find("contact")
    if contact is None:
        raise ValueError("scene has no explicit contact table")
    before_ranges = {
        joint.get("name"): joint.get("range") for joint in root.findall(".//joint")
    }
    before_pairs = len(contact.findall("pair"))
    removed = []
    added = []
    for spec in protocol["external_semantic_geoms"]:
        body = root.find(f".//body[@name='{spec['body']}']")
        if body is None:
            raise ValueError(f"missing body {spec['body']}")
        if root.find(f".//geom[@name='{spec['name']}']") is not None:
            raise ValueError(f"duplicate semantic geom {spec['name']}")
        if "native_aabb_scale" in spec:
            visual = source_model.geom(spec["source_visual"]).id
            bounds = native_meshes[visual].bounds
            pos = bounds.mean(axis=0)
            size = np.diff(bounds, axis=0)[0] * 0.5 * float(spec["native_aabb_scale"])
        else:
            pos, size = spec["pos"], spec["size"]
        ET.SubElement(
            body,
            "geom",
            name=spec["name"],
            type="ellipsoid",
            pos=_fmt(pos),
            size=_fmt(size),
            group="3",
            density="0",
            rgba="0.2 0.7 1 0.35",
            contype="0",
            conaffinity="0",
            condim="3",
            friction="1 0.1 0",
        )
        replaced = spec.get("replaces_geom_for_object")
        if replaced:
            for pair in list(contact.findall("pair")):
                values = (pair.get("geom1", ""), pair.get("geom2", ""))
                if replaced in values and any(
                    value.startswith(spec["object_prefix"]) for value in values
                ):
                    removed.append(pair.get("name"))
                    contact.remove(pair)
        object_geoms = [
            geom.get("name") for geom in root.findall(".//geom")
            if geom.get("name", "").startswith(spec["object_prefix"])
            and not geom.get("name", "").endswith("visual")
        ]
        if len(object_geoms) != 32:
            raise ValueError(f"expected 32 object pieces for {spec['object_prefix']}")
        for object_geom in object_geoms:
            name = f"semantic_external_{len(added):04d}"
            ET.SubElement(
                contact, "pair", name=name, geom1=spec["name"], geom2=object_geom,
                condim="3", friction="1 1 0.1 0 0",
            )
            added.append(name)

    # Floor collision needs only the native mesh support function.  MuJoCo's
    # convex mesh representation preserves the exact minimum support point
    # against a plane, while remaining isolated from object/self contacts.
    legacy_floor_pairs = []
    for pair in list(contact.findall("pair")):
        values = (pair.get("geom1", ""), pair.get("geom2", ""))
        if "floor" in values and any(value.startswith("collision_hand_") for value in values):
            legacy_floor_pairs.append(pair.get("name"))
            contact.remove(pair)
    floor_added = []
    hand_visuals = []
    for body in root.findall(".//body"):
        for visual in list(body.findall("geom")):
            name = visual.get("name", "")
            if not name.endswith("_visual") or not name.startswith(("right_", "left_")):
                continue
            if "object_visual" in name:
                continue
            side = "right" if name.startswith("right_") else "left"
            finger = next(
                (value for value in ("thumb", "index", "middle", "ring", "pinky") if value in name),
                "palm",
            )
            semantic_name = f"collision_hand_{side}_{finger}_floor_semantic_{len(hand_visuals):02d}"
            attributes = {
                "name": semantic_name,
                "type": "mesh",
                "mesh": visual.get("mesh"),
                "group": "3",
                "density": "0",
                "rgba": "0.7 0.4 1 0.25",
                "contype": "0",
                "conaffinity": "0",
                "condim": "3",
                "friction": "1 0.1 0",
            }
            for key in ("pos", "quat", "euler", "axisangle"):
                if visual.get(key) is not None:
                    attributes[key] = visual.get(key)
            ET.SubElement(body, "geom", **attributes)
            pair_name = f"semantic_floor_{len(floor_added):02d}"
            ET.SubElement(
                contact, "pair", name=pair_name, geom1=semantic_name, geom2="floor",
                condim="3", friction="1 1 0.1 0 0",
            )
            hand_visuals.append(name)
            floor_added.append(pair_name)

    after_ranges = {
        joint.get("name"): joint.get("range") for joint in root.findall(".//joint")
    }
    if before_ranges != after_ranges:
        raise ValueError("joint ranges changed")
    if len(removed) != 4 * 32 or len(added) != 9 * 32:
        raise ValueError(f"unexpected pair edit count: removed={len(removed)} added={len(added)}")
    if len(legacy_floor_pairs) != 24 or len(floor_added) != 26:
        raise ValueError(
            f"unexpected floor pair edit count: removed={len(legacy_floor_pairs)} added={len(floor_added)}"
        )
    output.mkdir(parents=True, exist_ok=True)
    candidate = output / "candidate_runtime_semantics_scene.xml"
    compiler = root.find("compiler")
    meshdir = (scene.parent / compiler.get("meshdir", ".")).resolve()
    compiler.set("meshdir", str(meshdir))
    ET.indent(tree, space="  ")
    tree.write(candidate, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(candidate))
    report = {
        "status": "runtime_semantics_candidate_built_not_formal_scene",
        "protocol": {"path": str(protocol_path), "sha256": _sha(protocol_path)},
        "source_scene": {"path": str(scene), "sha256": _sha(scene)},
        "candidate_scene": {"path": str(candidate), "sha256": _sha(candidate)},
        "changes": {
            "external_semantic_ellipsoids_added": 9,
            "old_specific_object_pairs_removed": len(removed),
            "semantic_object_pairs_added": len(added),
            "legacy_hand_floor_pairs_removed": len(legacy_floor_pairs),
            "semantic_mesh_floor_pairs_added": len(floor_added),
            "pair_count_before": before_pairs,
            "pair_count_after": len(contact.findall("pair")),
            "compiled_ngeom": int(model.ngeom),
            "compiled_npair": int(model.npair),
        },
        "mink_hand_object_constraint_added": False,
        "self_guard_added": False,
        "formal_scene_modified": False,
        "training_ready": False,
        "pending": [
            "left palm-thumb self guard passes independent holdout",
            "external attribution acceptance",
            "hand-floor regression",
            "contact-role mapping",
            "capacity and physics-contract revalidation",
        ],
    }
    (output / "build_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.protocol, args.output)
