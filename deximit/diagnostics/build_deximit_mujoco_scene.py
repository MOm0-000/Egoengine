#!/usr/bin/env python3
"""Build one isolated MuJoCo scene that mirrors DexImit's object setup.

The generated scene is diagnostic-only.  It replaces the seven CoACD object
pieces with the convex hull of the visual mesh and preserves the fixed
third-person camera.  Runtime code converts SAPIEN damping/contact behavior
into MuJoCo parameters.  This builder never edits the input scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


OBJECT_NAME = "right_object"
OBJECT_JOINT = "right_object_joint"
SINGLE_COLLISION = "right_object_single"
VISUAL_MESH = "right_visual"
OLD_COLLISION_PREFIX = "right_object_"
HAND_COLLISION_PREFIX = "collision_hand_right_"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def keep_fixed_front_camera_only(root: ET.Element) -> list[str]:
    """Remove every camera except the fixed third-person front camera."""
    front = root.find(".//camera[@name='front']")
    if front is None or front.get("mode", "fixed") != "fixed":
        raise ValueError("source scene lacks the required fixed front camera")
    removed: list[str] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag != "camera" or child is front:
                continue
            removed.append(child.get("name", "<unnamed>"))
            parent.remove(child)
    remaining = root.findall(".//camera")
    if remaining != [front]:
        raise ValueError("failed to enforce one fixed third-person camera")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source scene is not a file: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic scene: {output}")

    tree = ET.parse(source)
    root = tree.getroot()
    removed_cameras = keep_fixed_front_camera_only(root)
    body = root.find(f".//body[@name='{OBJECT_NAME}']")
    if body is None:
        raise ValueError(f"source scene lacks body {OBJECT_NAME!r}")
    joint = body.find(f"joint[@name='{OBJECT_JOINT}']")
    if joint is None or joint.get("type") != "free":
        raise ValueError("object is not attached by the expected free joint")
    if root.find(f".//mesh[@name='{VISUAL_MESH}']") is None:
        raise ValueError("source scene lacks the visual object mesh")
    if root.find(f".//geom[@name='{SINGLE_COLLISION}']") is not None:
        raise ValueError("source scene already contains the diagnostic collision geom")

    removed_names: set[str] = set()
    for geom in list(body.findall("geom")):
        name = geom.get("name", "")
        suffix = name.removeprefix(OLD_COLLISION_PREFIX)
        if name.startswith(OLD_COLLISION_PREFIX) and suffix.isdigit():
            removed_names.add(name)
            body.remove(geom)
    if len(removed_names) != 7:
        raise ValueError(f"expected seven CoACD pieces, found {sorted(removed_names)}")

    # MuJoCo uses the convex hull of a mesh geom for collision by default.  The
    # alpha-zero duplicate keeps the original textured visual geom unchanged.
    body.append(ET.Element("geom", {
        "name": SINGLE_COLLISION,
        "type": "mesh",
        "mesh": VISUAL_MESH,
        "group": "3",
        "density": "300",
        "rgba": "0 1 0 0",
    }))
    joint.set("armature", "0")
    joint.set("frictionloss", "0")
    # Runtime conversion derives a MuJoCo viscous coefficient from the PhysX
    # velocity-decay rate.  The engine-specific number 20 must not be copied.
    joint.set("damping", "0")

    contact = root.find("contact")
    if contact is None:
        raise ValueError("source scene lacks an explicit contact graph")
    removed_pairs = 0
    for pair in list(contact.findall("pair")):
        if pair.get("geom1") in removed_names or pair.get("geom2") in removed_names:
            contact.remove(pair)
            removed_pairs += 1
    if removed_pairs == 0:
        raise ValueError("source contact graph did not reference the old object pieces")

    hand_geoms = sorted({
        geom.get("name", "")
        for geom in root.findall(".//geom")
        if geom.get("name", "").startswith(HAND_COLLISION_PREFIX)
    })
    if not hand_geoms:
        raise ValueError("source scene has no right-hand collision geoms")
    # PhysX has separate static/dynamic friction.  MuJoCo has one sliding
    # coefficient, so 0.7 preserves DexImit's static holding coefficient.
    contact.append(ET.Element("pair", {
        "name": f"floor_{SINGLE_COLLISION}",
        "geom1": "floor",
        "geom2": SINGLE_COLLISION,
        "friction": "1 1 0.1 0 0",
        "margin": "0",
        "gap": "0.04",
    }))
    for hand in hand_geoms:
        contact.append(ET.Element("pair", {
            "name": f"{hand}_{SINGLE_COLLISION}",
            "geom1": hand,
            "geom2": SINGLE_COLLISION,
            "condim": "4",
            "friction": "0.7 0.7 0.1 0 0",
            # Two PhysX 20 mm contact offsets yield a 40 mm detection range,
            # while rest_offset=0 means geometric contact remains the force
            # origin.  MuJoCo therefore uses zero margin plus a 40 mm inactive
            # detection gap.  margin=gap=40 mm would create a false air shell.
            "margin": "0",
            "gap": "0.04",
        }))

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode", xml_declaration=False)
    provenance = {
        "schema": "deximit_mujoco_scene_v1_diagnostic_only",
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "fixed_camera": "front",
        "camera_count": 1,
        "removed_cameras": removed_cameras,
        "camera_following_allowed": False,
        "object_collision": "single convex hull of right_visual",
        "object_density_kg_m3": 300.0,
        "object_damping_in_scene": 0.0,
        "object_damping_runtime_policy": "match recorded PhysX one-step velocity decay",
        "hand_object_sliding_friction": 0.7,
        "physx_static_friction_reference": 0.7,
        "physx_dynamic_friction_reference": 0.5,
        "contact_detection_distance_m": 0.04,
        "mujoco_contact_margin_m": 0.0,
        "mujoco_contact_gap_m": 0.04,
        "geometric_force_onset_m": 0.0,
        "limitations": [
            "MuJoCo and PhysX use different contact solvers.",
            "MuJoCo has no separate static/dynamic sliding coefficient per pair.",
            "The 40 mm gap records early detections but cannot reproduce PhysX speculative-contact impulses by itself.",
            "Runtime code converts the PhysX decay rate instead of copying its damping number.",
        ],
    }
    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
