"""Merge the passed external/floor and isolated self-guard candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoengine_repro.action.geometry import explicit_collision_pairs, physical_object_geom_ids
from egoengine_repro.retarget.paper_audit import artifact, verify_artifacts

DEFAULT_PROTOCOL = ROOT / "configs/taco_pour_collision_semantics_combined_v1.yaml"
DEFAULT_OUTPUT = ROOT / "runs/taco_pour_collision_semantics_combined_v1"


def _copy(element: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(element, encoding="unicode"))


def run(protocol_path: Path, output: Path) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text())
    verify_artifacts(protocol["inputs"].values())
    external_audit = json.loads(Path(protocol["inputs"]["external_floor_audit"]["path"]).read_text())
    self_audit = json.loads(Path(protocol["inputs"]["local_self_guard_audit"]["path"]).read_text())
    if not external_audit["external_candidate_passed"]:
        raise ValueError("external/floor candidate has not passed")
    if not self_audit["passed_local_self_guard_gate"]:
        raise ValueError("local self guard has not passed")

    external_path = Path(protocol["inputs"]["external_floor_candidate"]["path"])
    self_build = json.loads(Path(protocol["inputs"]["local_self_guard_build"]["path"]).read_text())
    self_path = Path(self_build["candidate_scene"]["path"])
    tree = ET.parse(external_path)
    root = tree.getroot()
    self_root = ET.parse(self_path).getroot()
    assets = root.find("asset")
    contact = root.find("contact")
    if assets is None or contact is None:
        raise ValueError("candidate scene is missing asset or contact section")

    asset_prefix = "left_palm_thumb_guard_"
    geom_prefix = "collision_hand_left_palm_thumb_surface_guard_"
    pair_prefix = "semantic_self_left_palm_thumb_surface_"
    copied_assets = [row for row in self_root.findall("asset/mesh")
                     if row.get("name", "").startswith(asset_prefix)]
    copied_geoms = [row for row in self_root.findall(".//geom")
                    if row.get("name", "").startswith(geom_prefix)]
    copied_pairs = [row for row in self_root.findall("contact/pair")
                    if row.get("name", "").startswith(pair_prefix)]
    if (len(copied_assets), len(copied_geoms), len(copied_pairs)) != (107, 107, 241):
        raise ValueError("unexpected local self-guard inventory")
    existing_names = {row.get("name") for row in root.iter() if row.get("name")}
    if any(row.get("name") in existing_names for row in copied_assets + copied_geoms + copied_pairs):
        raise ValueError("self-guard name collides with external/floor candidate")

    for row in copied_assets:
        assets.append(_copy(row))
    for row in copied_geoms:
        parent_name = (
            "left_hand_link"
            if "_surface_guard_palm_" in row.get("name")
            else "left_hand_thumb_rota_link1"
        )
        parent = root.find(f".//body[@name='{parent_name}']")
        if parent is None:
            raise ValueError(f"missing target body {parent_name}")
        parent.append(_copy(row))
    for row in copied_pairs:
        contact.append(_copy(row))

    output.mkdir(parents=True, exist_ok=True)
    candidate = output / "combined_candidate_scene.xml"
    ET.indent(tree, space="  ")
    tree.write(candidate, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(candidate))
    object_ids = set()
    for body in ("right_object", "left_object"):
        object_ids.update(physical_object_geom_ids(model, mujoco, model.body(body).id))
    families = explicit_collision_pairs(
        model, mujoco, object_geom_ids=frozenset(object_ids),
        hand_sides=("right", "left"),
    )
    guard_ids = {
        model.geom(row.get("name")).id for row in copied_geoms
    }
    guard_pairs = {
        tuple(sorted((int(model.pair_geom1[index]), int(model.pair_geom2[index]))))
        for index in range(model.npair)
        if int(model.pair_geom1[index]) in guard_ids or int(model.pair_geom2[index]) in guard_ids
    }
    if len(guard_pairs) != 241 or any(
        first not in guard_ids or second not in guard_ids for first, second in guard_pairs
    ):
        raise ValueError("combined candidate changed self-guard pair isolation")
    report = {
        "status": "combined_collision_candidate_built_retarget_pending",
        "protocol": artifact(protocol_path),
        "candidate_scene": artifact(candidate),
        "copied": {
            "local_guard_mesh_assets": len(copied_assets),
            "local_guard_geoms": len(copied_geoms),
            "local_guard_pairs": len(copied_pairs),
        },
        "compiled": {
            "ngeom": int(model.ngeom),
            "npair": int(model.npair),
            "self_pairs": len(families["self"]),
            "floor_pairs": len(families["floor"]),
            "object_pairs": len(families["object"]),
        },
        "guard_pair_isolation_passed": True,
        "hand_object_mink_limit_enabled": False,
        "formal_scene_modified": False,
        "reference_modified": False,
        "training_ready": False,
        "next_steps": protocol["next_steps"],
        "code": artifact(Path(__file__)),
    }
    (output / "build_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.protocol, args.output)
