"""Compare partial collision repair with native meshes and the unchanged baseline."""

from itertools import product
import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from audit_taco_initialization import visual_meshes
from audit_taco_initialization_preflight import object_overfill_samples
from audit_taco_thumb_assembly import native_intersection
from build_taco_collision_repair import SCENE, OUTPUT, check_unchanged_dynamics
from egoengine_repro.retarget.initial_hand import state_summary
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance, mesh_surface_distance
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts

BASELINE = ROOT / "runs/taco_pour_bimanual_mano_fk_right_guard_v1"


def body_hulls(model, body_name):
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    body = model.body(body_name).id
    hulls = []
    for g in range(model.ngeom):
        name = model.geom(g).name
        physical = name.startswith("collision_hand_") or (name.startswith(("right_object_", "left_object_"))
                                                           and not name.endswith("visual"))
        if model.geom_bodyid[g] != body or not physical:
            continue
        mid = model.geom_dataid[g]
        start, size = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        points = model.mesh_vert[start:start + size] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        points = (points - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        hulls.append(trimesh.convex.convex_hull(points))
    return hulls


def object_hulls(model, side):
    return body_hulls(model, f"{side}_object")


def union_contains(hulls, points, tolerance=1e-9):
    """Exact convex half-space membership up to disclosed floating tolerance."""
    points = np.asarray(points, dtype=float)
    if not len(hulls) or not np.isfinite(points).all():
        raise ValueError("finite probes and nonempty convex union required")
    contained = np.zeros(len(points), dtype=bool)
    for hull in hulls:
        # Face normals point outwards. Dot(n, p - face_center) <= 0 is inside.
        normal = hull.face_normals
        offset = np.einsum("ij,ij->i", normal, hull.triangles_center)
        indices = np.flatnonzero(~contained)
        for block in np.array_split(indices, max(1, len(indices) // 512)):
            contained[block] |= np.all(points[block] @ normal.T - offset <= tolerance, axis=1)
    return contained


def object_probe_comparison(before, after, native):
    report = {}
    for side in ("right", "left"):
        hullsets = [object_hulls(model, side) for model in (before, after)]
        mesh = native[before.geom(f"{side}_object_visual").id]
        probes = []
        for hulls in hullsets:
            full = np.concatenate([np.concatenate([h.vertices, h.triangles_center]) for h in hulls])
            probes.append(full[np.linspace(0, len(full) - 1, min(4096, len(full)), dtype=int)])
        points = np.concatenate(probes)  # Same balanced probe set for both models.
        signed = closed_mesh_signed_distance(mesh, points)
        surface = mesh.vertices[np.linspace(0, len(mesh.vertices) - 1, 4096, dtype=int)]
        variants = []
        for hulls in hullsets:
            inside = union_contains(hulls, points)
            missed = ~union_contains(hulls, surface)
            # For points outside every component, minimum distance to all
            # triangles equals distance to the union. Build the tree once, not
            # one spatial tree/query per piece. Inside points already have zero.
            surfaces = trimesh.util.concatenate(hulls)
            outside = surface[missed]
            gaps = mesh_surface_distance(surfaces, outside)
            gap_all = np.zeros(len(surface))
            gap_all[missed] = gaps
            variants.append(dict(parts=len(hulls),
                common_probe_false_solid_over_1mm=int((inside & (signed < -.001)).sum()),
                common_probe_false_solid_over_50um=int((inside & (signed < -5e-5)).sum()),
                common_probe_max_overfill_m=float(np.maximum(0, -signed[inside]).max()) if inside.any() else 0,
                native_surface_missing_over_1mm=int((gap_all > .001).sum()),
                native_surface_gap_m_percentiles=np.percentile(gap_all, [50, 95, 99, 100]).tolist()))
        report[side] = dict(before=variants[0], after=variants[1], common_probes=len(points),
                            native_probes=len(surface), halfspace_tolerance_m=1e-9,
                            distance_backend="Open3D signed distance, 11 rays, positive inside",
                            global_geometry_certificate=False)
        print(f"{side} common probes: {variants}", flush=True)
    return report


def body_pair_distances(model, data, side):
    geoms = []
    for name in (f"{side}_hand_link", f"{side}_hand_thumb_rota_link1"):
        body = model.body(name).id
        geoms.append([g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                      and model.geom(g).name.startswith("collision_hand_")])
    pairs = list(product(*geoms))
    distances = [mujoco.mj_geomDistance(model, data, a, b, .05, None) for a, b in pairs]
    return float(min(distances)), pairs


def compare():
    destination = OUTPUT / "comparison.json"
    if destination.exists():
        raise FileExistsError(destination)
    candidate_scene = OUTPUT / "scene.xml"
    inputs = [artifact(p) for p in (SCENE, candidate_scene, BASELINE / "robot_reference.npz",
        ROOT / "runs/taco_pour_initialization_preflight_right_guard_v1/report.json",
        ROOT / "runs/taco_pour_initial_hand_v4/initial_hand_candidate.npz")]
    dependencies = scene_mesh_artifacts(SCENE) + scene_mesh_artifacts(candidate_scene)
    before, after = [mujoco.MjModel.from_xml_path(str(p)) for p in (SCENE, candidate_scene)]
    check_unchanged_dynamics(before, after)
    native, _ = visual_meshes(SCENE, before)
    candidate_native, _ = visual_meshes(candidate_scene, after)
    preflight = json.loads((ROOT / "runs/taco_pour_initialization_preflight_right_guard_v1/report.json").read_text())
    verify_artifacts(preflight["preserved_artifacts"] + preflight["scene_meshes"])
    with np.load(BASELINE / "robot_reference.npz", allow_pickle=False) as data:
        qpos = data["qpos"]
    with np.load(ROOT / "runs/taco_pour_initial_hand_v4/initial_hand_candidate.npz", allow_pickle=False) as data:
        negative_control = data["qpos"][0]
    hand_results = {}
    declared = {frozenset((int(a), int(b))) for a, b in zip(after.pair_geom1, after.pair_geom2)}
    state = mujoco.MjData(after)
    timers, all_contacts = [], []
    values = {s: [] for s in ("right", "left")}
    for q in qpos:
        state.qpos[:] = q
        start = time.perf_counter()
        mujoco.mj_forward(after, state)
        timers.append(time.perf_counter() - start)
        all_contacts.append(state.ncon)
        for side in values:
            value, pairs = body_pair_distances(after, state, side)
            if not all(frozenset(pair) in declared for pair in pairs):
                raise ValueError("repaired palm/thumb pair absent at runtime")
            values[side].append(value)
    for side in values:
        native_pair = next(p for p in preflight["native_surfaces"]["pairs"] if
            p["geoms"] == [f"{side}_hand_link_visual", f"{side}_thumb_rota1_visual"])
        native_crossing = np.zeros(len(qpos), dtype=bool)
        native_crossing[native_pair["source_rows"]] = True
        values[side] = np.asarray(values[side])
        intersecting = values[side] < -5e-5
        state.qpos[:] = negative_control
        mujoco.mj_forward(after, state)
        control_distance, _ = body_pair_distances(after, state, side)
        closed_control = native_intersection(after, state, candidate_native, side)
        hand_results[side] = dict(initial_distance_m=float(values[side][0]),
            minimum_distance_m=float(values[side].min()),
            native_surface_crossing_frames=int(native_crossing.sum()),
            hull_overlap_frames=int(intersecting.sum()),
            hull_overlap_without_native_surface_crossing=int((intersecting & ~native_crossing).sum()),
            native_surface_crossing_without_50um_hull_overlap=int((~intersecting & native_crossing).sum()),
            historical_v4_negative_control_native_intersection_m3=closed_control["intersection_volume_m3"],
            historical_v4_negative_control_hull_distance_m=control_distance,
            comparison_tolerance_m=5e-5)
    pairs_by_body = {frozenset(map(int, after.geom_bodyid[[a, b]])) for a, b in zip(after.pair_geom1, after.pair_geom2)}
    omissions = []
    for pair in preflight["native_surfaces"]["pairs"]:
        if pair["family"] == "intrahand" and not pair["assembly_adjacent"] and pair["surface_contact_frames"]:
            bodies = frozenset(int(after.geom_bodyid[after.geom(n).id]) for n in pair["geoms"])
            if bodies not in pairs_by_body:
                omissions.append(pair)
    print(f"Palm/thumb comparison: {hand_results}", flush=True)
    result = dict(status="partial_model_comparison_not_initialization_comparison", source=inputs,
        scene_dependencies=dependencies, hand_palm_thumb=hand_results,
        remaining_observed_nonadjacent_pair_omissions=omissions,
        object_overfill_before=object_overfill_samples(before, native),
        object_overfill_after=object_overfill_samples(after, candidate_native),
        shared_object_probe_comparison=object_probe_comparison(before, after, native),
        initial_declared_state=state_summary(after, qpos[0]),
        performance=dict(mj_forward_seconds_median=float(np.median(timers)),
                         mj_forward_seconds_max=max(timers), max_contacts=max(all_contacts),
                         measured_calls=len(qpos), benchmark="CPU read-only forward dynamics, not physics stepping or MJWarp"),
        full_collision_coverage=False, accepted_for_training=False, source_reference_modified=False,
        initialization_candidates_compared=False, simulation_steps_executed=0,
        limitations="sampled shapes; native surface crossing is not signed penetration depth; adjacent assembly parts unresolved",
        distance_code=artifact(ROOT / "src/egoengine_repro/retarget/mesh_distance.py"),
        code=artifact(Path(__file__)))
    verify_artifacts(inputs + dependencies)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    np.savez_compressed(OUTPUT / "palm_thumb_distances.npz", **values)
    return result


def screen_object_parameters():
    """Compare alternative piece sets in geometry-only models, not an RL scene."""
    destination = OUTPUT / "object_parameter_screen.json"
    if destination.exists():
        raise FileExistsError(destination)
    before = mujoco.MjModel.from_xml_path(str(SCENE))
    native, _ = visual_meshes(SCENE, before)
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    result = dict(status="object_geometry_parameter_screen_not_task_training",
        capped_original=comparison["object_overfill_before"],
        uncapped_default=comparison["object_overfill_after"], variants={},
        parameters_are_local_not_paper=True, acceptance_tolerance_selected=False,
        code=artifact(Path(__file__)),
        distance_code=artifact(ROOT / "src/egoengine_repro/retarget/mesh_distance.py"))
    for label in ("no_preprocess", "tighter"):
        root = ET.Element("mujoco")
        assets = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        provenance = []
        for side, obj, oid in (("right", "bowl", "022"), ("left", "plate", "135")):
            record_path = OUTPUT / "assets" / f"{obj}_{label}" / "provenance.json"
            record = json.loads(record_path.read_text())
            verify_artifacts([dict(path=p["path"], sha256=p["sha256"]) for p in record["artifacts"]])
            if artifact(Path(record["source"]))["sha256"] != record["source_sha256"]:
                raise ValueError("alternative source changed")
            provenance.append(artifact(record_path))
            ET.SubElement(assets, "mesh", name=f"{side}_native", file=record["source"], scale=".01 .01 .01")
            body = ET.SubElement(world, "body", name=f"{side}_object")
            ET.SubElement(body, "geom", name=f"{side}_object_visual", type="mesh", mesh=f"{side}_native",
                          density="0", contype="0", conaffinity="0")
            for i, piece in enumerate(record["artifacts"]):
                name = f"{side}_object_{i}"
                ET.SubElement(assets, "mesh", name=name, file=piece["path"])
                ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, density="0", contype="0", conaffinity="0")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        meshes = {model.geom(f"{side}_object_visual").id: native[before.geom(f"{side}_object_visual").id]
                  for side in ("right", "left")}
        result["variants"][label] = dict(part_sources=provenance,
            overfill=object_overfill_samples(model, meshes),
            common_probe_comparison=object_probe_comparison(before, model, native))
        print(f"Object variant completed: {label}", flush=True)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def contact_discrepancy_masks(native, minimum, present):
    """A 50 um reporting cut is not the simulator's contact activation rule."""
    if native.shape != minimum.shape or native.shape != present.shape or not np.isfinite(minimum).all():
        raise ValueError("matching contact arrays and finite distances required")
    detected = minimum < -5e-5
    return dict(shell_overlap_over_50um=detected,
                shell_overlap_without_surface_crossing=detected & ~native,
                native_crossing_not_over_50um=native & ~detected,
                native_crossing_without_runtime_contact=native & ~present)


def screen_hand_geometry(labels=("mesh_hands", "budgeted_palms"), destination=OUTPUT / "hand_geometry_screen.json", models=None):
    if destination is not None and destination.exists():
        raise FileExistsError(destination)
    report = json.loads((ROOT / "runs/taco_pour_initialization_preflight_right_guard_v1/report.json").read_text())
    verify_artifacts(report["preserved_artifacts"] + report["scene_meshes"])
    pairs = [p for p in report["native_surfaces"]["pairs"]
             if p["family"] in ("intrahand", "interhand") and not p["assembly_adjacent"]]
    native_flags = np.zeros((198, len(pairs)), dtype=bool)
    for i, p in enumerate(pairs):
        native_flags[p["source_rows"], i] = True
    with np.load(BASELINE / "robot_reference.npz", allow_pickle=False) as data:
        qpos = data["qpos"]
    result = dict(status="native_mesh_hand_shape_screen_not_reset_comparison", frames=198,
                  body_pairs=len(pairs), tolerance_m=5e-5, variants={}, accepted_for_training=False)
    for label in labels:
        scene = OUTPUT / f"scene_{label}.xml"
        model = models[label] if models is not None else mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        lookup = {frozenset(int(model.geom_bodyid[model.geom(n).id]) for n in p["geoms"]): i
                  for i, p in enumerate(pairs)}
        declared = {frozenset(map(int, model.geom_bodyid[[a, b]])) for a, b in zip(model.pair_geom1, model.pair_geom2)}
        if set(lookup) - declared:
            raise ValueError("full nonadjacent hand-body coverage was not generated")
        minimum = np.zeros_like(native_flags, dtype=float)
        present = np.zeros_like(native_flags)
        timing, ncon = [], []
        for frame, q in enumerate(qpos):
            data.qpos[:] = q
            start = time.perf_counter()
            mujoco.mj_forward(model, data)
            timing.append(time.perf_counter() - start)
            ncon.append(data.ncon)
            for contact in data.contact:
                key = frozenset(map(int, model.geom_bodyid[[contact.geom1, contact.geom2]]))
                if key in lookup:
                    i = lookup[key]
                    present[frame, i] = True
                    minimum[frame, i] = min(minimum[frame, i], float(contact.dist))
        masks = contact_discrepancy_masks(native_flags, minimum, present)
        extra = masks["shell_overlap_without_surface_crossing"]
        below = masks["native_crossing_not_over_50um"]
        absent = masks["native_crossing_without_runtime_contact"]
        discrepancies = []
        for i, p in enumerate(pairs):
            if extra[:, i].any() or below[:, i].any():
                discrepancies.append(dict(geoms=p["geoms"],
                    shell_overlap_without_surface_crossing_rows=np.flatnonzero(extra[:, i]).tolist(),
                    native_crossing_not_over_50um_rows=np.flatnonzero(below[:, i]).tolist(),
                    native_crossing_without_runtime_contact_rows=np.flatnonzero(absent[:, i]).tolist(),
                    extra_overlap_max_m=float(-minimum[extra[:, i], i].min()) if extra[:, i].any() else 0.0))
        result["variants"][label] = dict(scene=artifact(scene),
            scene_dependencies=scene_mesh_artifacts(scene), geoms=model.ngeom, physical_pairs=model.npair,
            native_surface_crossing_frame_pairs=int(native_flags.sum()),
            **{f"{name}_frame_pairs": int(mask.sum()) for name, mask in masks.items()},
            shell_only_is_not_confirmed_false_positive=True,
            absent_runtime_contact_requires_review_not_automatic_missed_penetration=True,
            discrepancies=discrepancies,
            forward_median_s=float(np.median(timing)), max_contacts=max(ncon),
            mj_warnings=int(sum(w.number for w in data.warning)),
            original_native_meshes_or_reference_modified=False, simulation_steps=0)
        print(f"{label}: shell-only {extra.sum()}, below statistic {below.sum()}, runtime absent {absent.sum()}; worst extra "
              f"{max((p['extra_overlap_max_m'] for p in discrepancies), default=0)*1000:.3f} mm", flush=True)
    if destination is not None:
        with destination.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def compare_refinement():
    destination = OUTPUT / "refinement_comparison.json"
    if destination.exists():
        raise FileExistsError(destination)
    scenes = [OUTPUT / f"scene_{name}.xml" for name in ("budgeted_palms", "refined")]
    before, after = [mujoco.MjModel.from_xml_path(str(p)) for p in scenes]
    unchanged = check_unchanged_dynamics(before, after)
    native, _ = visual_meshes(scenes[0], before)
    refined_native, _ = visual_meshes(scenes[1], after)
    result = dict(status="targeted_shape_refinement_not_formal_model", dynamics=unchanged,
        object_overfill_before=object_overfill_samples(before, native),
        object_overfill_after=object_overfill_samples(after, refined_native),
        shared_object_probes=object_probe_comparison(before, after, native),
        hands=screen_hand_geometry(("budgeted_palms", "refined"), None,
                                   {"budgeted_palms": before, "refined": after}),
        refined_finger_surfaces=finger_surface_comparison(before, after, native),
        source_reference=artifact(BASELINE / "robot_reference.npz"),
        native_preflight=artifact(ROOT / "runs/taco_pour_initialization_preflight_right_guard_v1/report.json"),
        code=artifact(Path(__file__)),
        distance_code=artifact(ROOT / "src/egoengine_repro/retarget/mesh_distance.py"),
        accepted_for_training=False)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def finger_surface_comparison(before, after, native):
    """Measure all native surface vertices/centres, including nonmanifold sheets."""
    result = {}
    for visual in ("left_index_rota2_visual", "left_middle_link2_visual"):
        mesh = native[before.geom(visual).id]
        body = before.body(before.geom_bodyid[before.geom(visual).id]).name
        points = np.concatenate([mesh.vertices, mesh.triangles_center])
        # A closed main component exists in these two specific CAD files.
        # Keep the original mesh unchanged and disclose the auxiliary solid.
        closed = [m for m in mesh.split(only_watertight=False, repair=False) if m.is_volume]
        if len(closed) != 1:
            raise ValueError("expected exactly one closed main component for distal diagnostic")
        core = closed[0]
        rows = []
        for model in (before, after):
            hulls = body_hulls(model, body)
            missing = ~union_contains(hulls, points)
            gaps = np.zeros(len(points))
            surfaces = trimesh.util.concatenate(hulls)
            if missing.any():
                gaps[missing] = mesh_surface_distance(surfaces, points[missing])
            shell = np.concatenate([np.concatenate([h.vertices, h.triangles_center]) for h in hulls])
            signed = closed_mesh_signed_distance(core, shell)
            rows.append(dict(parts=len(hulls), native_probes=len(points),
                native_surface_max_missing_m=float(gaps.max()),
                native_surface_missing_over_50um=int((gaps > 5e-5).sum()),
                native_surface_missing_over_0p5mm=int((gaps > .0005).sum()),
                sampled_overfill_vs_closed_main_component_m=float(np.maximum(-signed, 0).max())))
        result[visual] = dict(before=rows[0], after=rows[1],
            native_faces=len(mesh.faces), auxiliary_closed_component_faces=len(core.faces),
            open_or_nonmanifold_original_not_modified=True, global_certificate=False)
        print(f"{visual} surface comparison: {rows}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-screen", action="store_true")
    parser.add_argument("--hand-screen", action="store_true")
    parser.add_argument("--refined", action="store_true")
    args = parser.parse_args()
    if sum((args.object_screen, args.hand_screen, args.refined)) > 1:
        parser.error("choose one screen")
    compare_refinement() if args.refined else screen_hand_geometry() if args.hand_screen else screen_object_parameters() if args.object_screen else compare()
