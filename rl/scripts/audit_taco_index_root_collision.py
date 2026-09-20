"""Screen palm/index shapes before enabling their previously omitted contacts."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from build_taco_collision_repair import replace_body_shapes
from audit_taco_initialization import visual_meshes
from audit_taco_remaining_contacts import native_pair_evidence, relative_mesh
from egoengine_repro.retarget.paper_audit import artifact


def classify_pose(record):
    """Classify this measured pose only; never exempt a body pair globally."""
    native = record.get("native", {})
    volume = native.get("intersection_volume_mm3")
    samples = native.get("surface_containment_first_in_second_then_reverse", [])
    depths = [s.get("sampled_max_inside_m", np.nan) for s in samples]
    if volume is not None and (not np.isfinite(volume) or volume < 0):
        raise ValueError("native intersection volume must be finite and nonnegative")
    if any(np.isfinite(d) and d > 1e-6 for d in depths) or (volume is not None and volume > 0):
        return "native_overlap_confirmed"
    # Triangle nonintersection alone cannot exclude a contained solid. An open
    # or auxiliary component is also insufficient to clear the complete CAD.
    clear = (native.get("solid_sources") == ["full_native", "full_native"]
             and native.get("surface_crossing") is False and volume == 0
             and len(depths) == 2 and np.isfinite(depths).all()
             and all(0 <= d <= 1e-6 for d in depths))
    if not clear:
        return "native_overlap_unresolved"
    distance = record["shell_minimum_m"]
    if not np.isfinite(distance):
        raise ValueError("shell distance must be finite")
    return ("shell_false_positive_at_audited_pose" if distance < -1e-6
            else "no_native_overlap_detected_at_audited_pose")


def run(scene, reference, output, *, root_parts=None, palm_parts=None, sides=("right", "left")):
    if output.exists():
        raise FileExistsError(output)
    root = ET.parse(scene).getroot()
    # Only the two isolated assembly pairs are compiled for this shape screen.
    # All other geometry and the 50-coordinate FK are unchanged. This is not a
    # capacity test or a scene exported for training.
    root.remove(root.find("contact"))
    contact = ET.SubElement(root, "contact")
    sources = [artifact(scene), artifact(reference)]
    for side in sides:
        for suffix, directory in (("hand_index_bend_link", root_parts), ("hand_link", palm_parts)):
            if directory is not None:
                body_name = f"{side}_{suffix}"
                body = root.find(f".//body[@name='{body_name}']")
                old = [g.get("name") for g in body.findall("geom")
                       if g.get("name", "").startswith("collision_hand_")]
                names, record = replace_body_shapes(root, body_name, old,
                    Path(str(directory).replace("{side}", side)), f"collision_hand_{side}_{suffix}_screen")
                sources.append(record)
        bodies = [root.find(f".//body[@name='{side}_{suffix}']")
                  for suffix in ("hand_link", "hand_index_bend_link")]
        groups = [[g.get("name") for g in b.findall("geom")
                   if g.get("name", "").startswith("collision_hand_")] for b in bodies]
        for a in groups[0]:
            for b in groups[1]:
                ET.SubElement(contact, "pair", geom1=a, geom2=b)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    meshes, _ = visual_meshes(scene, model)
    qpos = np.load(reference, allow_pickle=False)["qpos"]
    results = {}
    for side in sides:
        joint = model.joint(f"{side}_hand_index_bend_joint")
        address = int(joint.qposadr[0])
        a, b = [model.geom(name).id for name in (f"{side}_hand_link_visual", f"{side}_index_bend_visual")]
        body_a, body_b = model.geom_bodyid[[a, b]]
        pairs = [(int(x), int(y)) for x, y in zip(model.pair_geom1, model.pair_geom2)
                 if set(model.geom_bodyid[[x, y]]) == {body_a, body_b}]
        records = []
        for angle in sorted(set([float(qpos[0, address]), 0.0, *np.linspace(-.175, .175, 9)])):
            data.qpos[:] = qpos[0]
            data.qpos[address] = angle
            mujoco.mj_forward(model, data)
            values = [mujoco.mj_geomDistance(model, data, x, y, .05, None) for x, y in pairs]
            record = dict(angle_rad=angle, shell_minimum_m=float(min(values)),
                          penetrating_piece_pairs=int(np.count_nonzero(np.asarray(values) < -1e-6)))
            if angle in (0.0, -.175, .175, float(qpos[0, address])):
                record["native"] = native_pair_evidence(meshes[a], relative_mesh(model, data, meshes, b, a))
            record["classification"] = classify_pose(record)
            records.append(record)
        results[side] = dict(piece_pairs=len(pairs), best_sampled_clearance_m=max(r["shell_minimum_m"] for r in records),
                             native_overlap_confirmed_poses=sum(r["classification"] == "native_overlap_confirmed" for r in records),
                             shell_false_positive_poses=sum(r["classification"] == "shell_false_positive_at_audited_pose" for r in records),
                             native_overlap_unresolved_poses=sum(r["classification"] == "native_overlap_unresolved" for r in records),
                             poses=records)
        print(side, json.dumps(results[side]), flush=True)
    report = dict(status="isolated_shape_screen_not_training_scene", results=results, sources=sources,
                  root_parts=str(root_parts), palm_parts=str(palm_parts),
                  scene_or_GT_changed=False, full_model_capacity_test=False,
                  classification_scope="audited_poses_only_not_pair_exemptions_or_physical_shape_repair",
                  generation_code=artifact(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=ROOT / "runs/taco_pour_collision_repair/scene_refined.xml")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/robot_reference.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-parts", type=Path)
    parser.add_argument("--palm-parts", type=Path)
    parser.add_argument("--sides", choices=("right", "left"), nargs="+", default=["right", "left"])
    args = parser.parse_args()
    run(args.scene, args.reference, args.output, root_parts=args.root_parts, palm_parts=args.palm_parts, sides=args.sides)
