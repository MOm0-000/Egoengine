"""Build collision-shape comparisons; never silently select one for RL.

Modes isolate object decomposition, six closed hand links, all native hand hulls,
and subdivided palms. Original meshes, dynamics and references stay intact.
"""

import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts
from audit_taco_initialization import visual_meshes

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
OUTPUT = ROOT / "runs/taco_pour_collision_repair"


def replace_body_shapes(root, body_name, old_names, directory, prefix, *, allow_nonmanifold=False):
    report_path = directory / "provenance.json"
    report = json.loads(report_path.read_text())
    verify_artifacts([dict(path=r["path"], sha256=r["sha256"]) for r in report["artifacts"]])
    if artifact(Path(report["source"]))["sha256"] != report["source_sha256"]:
        raise ValueError("decomposition source changed")
    if not report["source_is_positive_volume"] and not allow_nonmanifold:
        raise ValueError("this candidate accepts only closed positive-volume sources")
    body = root.find(f".//body[@name='{body_name}']")
    if body is None or body.find("inertial") is None:
        raise ValueError("replacement needs a known body with explicit unchanged inertia")
    if old_names:
        prototype = deepcopy(body.find(f"geom[@name='{old_names[0]}']"))
        if prototype is None:
            raise ValueError("source collision geometry missing")
    else:
        prototype = ET.Element("geom", dict(contype="0", conaffinity="0", group="3",
            condim="3", friction="1 0.1 0", rgba="0 1 0 1"))
    for key in ("name", "type", "size", "pos", "quat", "fromto", "mesh", "mass", "density"):
        prototype.attrib.pop(key, None)
    prototype.set("type", "mesh")
    prototype.set("density", "0")
    for name in old_names:
        geom = body.find(f"geom[@name='{name}']")
        if geom is None:
            raise ValueError(f"missing original geom {name}")
        old_mesh = geom.get("mesh")
        body.remove(geom)
        if old_mesh and not root.findall(f".//geom[@mesh='{old_mesh}']"):
            root.find("asset").remove(root.find(f"asset/mesh[@name='{old_mesh}']"))
    names = []
    for i, part in enumerate(report["artifacts"]):
        name = f"{prefix}_{i}"
        mesh_name = f"repair_mesh_{prefix}_{i}"
        ET.SubElement(root.find("asset"), "mesh", name=mesh_name, file=part["path"])
        geom = deepcopy(prototype)
        geom.set("name", name)
        geom.set("mesh", mesh_name)
        body.append(geom)
        names.append(name)
    return names, artifact(report_path)


def check_unchanged_dynamics(before, after):
    if (before.nq, before.nv, before.nu, before.nbody, before.njnt) != (
            after.nq, after.nv, after.nu, after.nbody, after.njnt):
        raise ValueError("collision replacement changed state dimensions")
    fields = ("body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
              "jnt_type", "jnt_pos", "jnt_axis", "jnt_range", "jnt_qposadr", "jnt_dofadr", "qpos0",
              "dof_armature", "dof_damping", "actuator_trnid", "actuator_gear", "actuator_gainprm",
              "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange")
    for field in fields:
        if not np.array_equal(getattr(before, field), getattr(after, field)):
            raise ValueError(f"collision replacement changed {field}")
    for body in range(before.nbody):
        if before.body(body).name != after.body(body).name:
            raise ValueError("body ordering changed")
    return dict(unchanged_fields=list(fields), state_dimensions=[after.nq, after.nv, after.nu],
                object_actuators_added=False, original_joint_order_preserved=True)


def build(scene, output, *, objects_only=False, source_mesh_hands=False, budgeted_palms=False, refinements=None):
    if objects_only and source_mesh_hands:
        raise ValueError("choose object-only or source-mesh hand comparison")
    if budgeted_palms and not source_mesh_hands:
        raise ValueError("budgeted palms belong to the source-mesh hand comparison")
    if refinements and not (source_mesh_hands and budgeted_palms):
        raise ValueError("targeted refinement requires the budgeted-palms comparison")
    selected = json.loads(refinements.read_text()) if refinements else {}
    destination = output / ("scene_refined.xml" if refinements else "scene_objects_only.xml" if objects_only else
                            "scene_budgeted_palms.xml" if budgeted_palms else
                            "scene_mesh_hands.xml" if source_mesh_hands else "scene.xml")
    report_path = destination.with_suffix(".json")
    if destination.exists() or report_path.exists():
        raise FileExistsError(destination)
    preserved = [artifact(scene)] + scene_mesh_artifacts(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    native_meshes, _ = visual_meshes(scene, model)
    root = ET.parse(scene).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((scene.parent / compiler.get("meshdir", "")).resolve(strict=True)))
    original_pairs = [deepcopy(p) for p in root.findall("contact/pair")]
    replacements, by_body, provenance = {}, {}, []
    if refinements:
        provenance.append(artifact(refinements))
        if set(selected) - {body.get("name") for body in root.findall(".//body")}:
            raise ValueError("unknown refinement body")
    for side, label in (("right", "bowl"), ("left", "plate")):
        old = [g.get("name") for g in root.findall(f".//body[@name='{side}_object']/geom")
               if g.get("name", "").startswith(f"{side}_object_") and not g.get("name").endswith("visual")]
        directory = Path(selected.get(f"{side}_object", output / "assets" / label))
        names, source = replace_body_shapes(root, f"{side}_object", old, directory,
                                           f"{side}_object")
        replacements.update({name: names for name in old})
        provenance.append(source)
        if not objects_only:
            changes = [
                ("palm", f"{side}_hand_link", f"collision_hand_{side}_palm_0", f"{side}_hand_link_visual"),
                ("thumb", f"{side}_hand_thumb_rota_link1", f"collision_hand_{side}_thumb_1", f"{side}_thumb_rota1_visual"),
                ("index_root", f"{side}_hand_index_bend_link", None, f"{side}_index_bend_visual"),
            ]
            if source_mesh_hands:
                changes = []
                for body in root.findall(".//body"):
                    visuals = [g for g in body.findall("geom") if g.get("name", "").startswith(f"{side}_")
                               and g.get("name", "").endswith("_visual") and g.get("name") != f"{side}_object_visual"]
                    shells = [g.get("name") for g in body.findall("geom")
                              if g.get("name", "").startswith("collision_hand_")]
                    if visuals:
                        if len(visuals) != 1 or len(shells) > 1:
                            raise ValueError("expected one source hand surface and at most one inherited shell per body")
                        v = visuals[0].get("name")
                        label = v.removeprefix(f"{side}_").removesuffix("_visual")
                        changes.append((label, body.get("name"), shells[0] if shells else None, v))
            for label, body_name, old_name, visual_name in changes:
                if not source_mesh_hands and not native_meshes[model.geom(visual_name).id].is_volume:
                    raise ValueError("targeted native hand source is not a closed positive volume")
                body = root.find(f".//body[@name='{body_name}']")
                visual = body.find(f"geom[@name='{visual_name}']")
                name = f"collision_hand_{side}_{label}_hull"
                if body_name in selected:
                    directory = Path(selected[body_name])
                    record = json.loads((directory / "provenance.json").read_text())
                    visual_asset = root.find(f"asset/mesh[@name='{visual.get('mesh')}']")
                    visual_path = Path(compiler.get("meshdir")) / visual_asset.get("file")
                    if artifact(visual_path)["sha256"] != record["source_sha256"] or record["unit_scale"] != 1:
                        raise ValueError("hand refinement does not match the native metric source")
                    names, source = replace_body_shapes(root, body_name, [old_name] if old_name else [],
                        directory, f"collision_hand_{side}_{label}_part", allow_nonmanifold=True)
                    if old_name:
                        replacements[old_name] = names
                    by_body[body_name] = names
                    provenance.append(source)
                    continue
                if budgeted_palms and body_name == f"{side}_hand_link":
                    names, source = replace_body_shapes(root, body_name, [old_name],
                        output / "assets" / f"{side}_palm_budgeted", f"collision_hand_{side}_palm_part")
                    replacements[old_name] = names
                    by_body[body_name] = names
                    provenance.append(source)
                    continue
                if old_name:
                    body.remove(body.find(f"geom[@name='{old_name}']"))
                ET.SubElement(body, "geom", name=name, type="mesh", mesh=visual.get("mesh"),
                              density="0", group="3", contype="0", conaffinity="0", condim="3",
                              friction="1 0.1 0", rgba="0 1 0 1")
                names = [name]
                if old_name:
                    replacements[old_name] = names
                by_body[body_name] = names

    # Expand old semantic body contacts once: all old pieces of an object map
    # to the same new parts, so deduplicate before the Cartesian product.
    contact = root.find("contact")
    for pair in list(contact.findall("pair")):
        contact.remove(pair)
    seen, expanded = {}, set()

    def add(a, b, attributes):
        key = tuple(sorted((a, b)))
        settings = {k: v for k, v in attributes.items() if k not in ("geom1", "geom2", "name")}
        if key in seen:
            if seen[key] != settings:
                raise ValueError("inconsistent contact settings in source pair expansion")
            return
        seen[key] = settings
        ET.SubElement(contact, "pair", dict(settings, geom1=a, geom2=b))

    for pair in original_pairs:
        ga, gb = pair.get("geom1"), pair.get("geom2")
        names_a, names_b = replacements.get(ga, [ga]), replacements.get(gb, [gb])
        settings = {k: v for k, v in pair.attrib.items() if k not in ("geom1", "geom2", "name")}
        signature = (tuple(names_a), tuple(names_b), tuple(sorted(settings.items())))
        if signature not in expanded:
            for a, b in product(names_a, names_b):
                add(a, b, settings)
            expanded.add(signature)

    added_pairs = len(seen)
    if not objects_only:
        self_settings = next(p.attrib for p in original_pairs if
            p.get("geom1").startswith("collision_hand_") and p.get("geom2").startswith("collision_hand_"))
        object_settings = next(p.attrib for p in original_pairs if
            p.get("geom1").startswith("collision_hand_") and p.get("geom2").startswith("right_object_"))
        floor_settings = next(p.attrib for p in original_pairs if
            {p.get("geom1"), p.get("geom2")} == {"floor", "collision_hand_right_palm_0"})
        hand_geoms = [(g.get("name"), model.body(body.get("name")).id)
                      for body in root.findall(".//body") for g in body.findall("geom")
                      if g.get("name", "").startswith("collision_hand_")]
        for side in ("right", "left"):
            for a, b in product(by_body[f"{side}_hand_link"], by_body[f"{side}_hand_thumb_rota_link1"]):
                add(a, b, self_settings)
            index_body = model.body(f"{side}_hand_index_bend_link").id
            wa = model.body_weldid[index_body]
            pa = model.body_weldid[model.body_parentid[wa]]
            for a in by_body[f"{side}_hand_index_bend_link"]:
                for b, body in hand_geoms:
                    wb = model.body_weldid[body]
                    pb = model.body_weldid[model.body_parentid[wb]]
                    if wa != wb and wa != pb and wb != pa:
                        add(a, b, self_settings)
                for other in ("right", "left"):
                    for g in root.findall(f".//body[@name='{other}_object']/geom"):
                        if g.get("name", "").startswith(f"{other}_object_") and not g.get("name").endswith("visual"):
                            add(a, g.get("name"), object_settings)
                add(a, "floor", floor_settings)
        if source_mesh_hands:
            # This is the native-mesh collision candidate's explicit policy,
            # not a new hole-filling operation or a certified hardware exemption.
            for i, (a, ba) in enumerate(hand_geoms):
                wa = model.body_weldid[ba]
                pa = model.body_weldid[model.body_parentid[wa]]
                for b, bb in hand_geoms[i + 1:]:
                    wb = model.body_weldid[bb]
                    pb = model.body_weldid[model.body_parentid[wb]]
                    if wa != wb and wa != pb and wb != pa:
                        add(a, b, self_settings)
    added_pairs = len(seen) - added_pairs
    ET.indent(root)
    # Compile before exporting: never leave a scene with unknown dynamics.
    print(f"Compiling candidate: {len(root.findall('.//geom'))} geoms, {len(seen)} explicit pairs", flush=True)
    candidate = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    unchanged = check_unchanged_dynamics(model, candidate)
    verify_artifacts(preserved + provenance)
    output.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(ET.tostring(root, encoding="unicode") + "\n")
    report = dict(status="partial_collision_repair_candidate_not_selected_for_training",
        objects_only=objects_only, source_mesh_hands=source_mesh_hands, budgeted_palms=budgeted_palms,
        targeted_refinements=selected,
        source_scene=artifact(scene), candidate_scene=artifact(destination),
        before=dict(geoms=model.ngeom, pairs=model.npair), after=dict(geoms=candidate.ngeom, pairs=candidate.npair),
        new_semantic_pair_expansion_count=added_pairs, dynamics=unchanged,
        source_and_geometry_preserved=preserved, candidate_part_sources=provenance,
        full_collision_coverage=False, accepted_for_training=False,
        hand_shape=("budgeted palms plus listed source-matched refinements; remaining links are native single hulls" if refinements else
                    "unchanged source hand primitives" if objects_only else
                    "32 convex pieces per closed native palm; single hulls for other 24 native links, including open CAD" if budgeted_palms else
                    "single convex hulls of all 26 unchanged native hand surfaces, including open CAD; no original hole filling" if source_mesh_hands else
                    "single convex hulls of six unchanged closed native meshes; approximation, not native solid"),
        retained_omissions=("source adjacent-body omissions only; not certified exemptions" if source_mesh_hands else
                            "unchanged source adjacent and other nonadjacent hand-pair omissions; not certified exemptions"),
        added_policy=("none; hand-pair policy unchanged" if objects_only else
                      "all nonadjacent hand-body pairs, including both index roots; hands against both objects and table" if source_mesh_hands else
                      "both palm/proximal-thumb pairs; both index roots against nonadjacent hands, both objects and table"),
        generation_code=artifact(Path(__file__)))
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(scene=str(destination), before=report["before"], after=report["after"])), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--objects-only", action="store_true")
    parser.add_argument("--source-mesh-hands", action="store_true")
    parser.add_argument("--budgeted-palms", action="store_true")
    parser.add_argument("--refinements", type=Path)
    args = parser.parse_args()
    build(args.scene, args.output, objects_only=args.objects_only, source_mesh_hands=args.source_mesh_hands,
          budgeted_palms=args.budgeted_palms, refinements=args.refinements)
