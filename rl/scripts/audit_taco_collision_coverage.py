"""Inventory collision coverage without changing models, states, pairs or gates."""

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from audit_taco_initialization import visual_meshes
from audit_taco_thumb_assembly import kinematic_source_check, native_intersection
from egoengine_repro.retarget.collision_audit import (
    collision_families, explicit_hand_pairs, hand_ids, nonadjacent_hand_pairs, validate_qpos,
)
from egoengine_repro.retarget.initial_hand import state_summary
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts


def coverage_inventory(model, meshes):
    hands = hand_ids(model)
    native = [g for g in meshes if model.geom(g).name not in ("right_object_visual", "left_object_visual")]
    explicit = set(explicit_hand_pairs(model))
    records = []
    for g in native:
        body = int(model.geom_bodyid[g])
        shell = [s for s in hands if model.geom_bodyid[s] == body]
        records.append(dict(visual=model.geom(g).name, body=model.body(body).name,
            same_body_shells=[model.geom(s).name for s in shell],
            watertight=bool(meshes[g].is_watertight), positive_closed_volume=bool(meshes[g].is_volume)))
    side = lambda g: "right" if model.geom(g).name.startswith(("right_", "collision_hand_right_")) else "left"
    intra = {p for p in combinations(hands, 2) if side(p[0]) == side(p[1])}
    omitted = intra - explicit
    nonadjacent = set(nonadjacent_hand_pairs(model))
    body_pairs = {tuple(sorted(int(model.geom_bodyid[g]) for g in p)) for p in explicit}
    native_intra = [p for p in combinations(native, 2) if side(p[0]) == side(p[1])]
    native_cross = list(product([g for g in native if side(g) == "right"], [g for g in native if side(g) == "left"]))
    counts = {}
    for label, pairs in (("intrahand", native_intra), ("interhand", native_cross)):
        represented = sum(tuple(sorted(int(model.geom_bodyid[g]) for g in p)) in body_pairs for p in pairs)
        counts[label] = dict(possible_native_body_pairs=len(pairs), directly_represented_by_declared_shell_pairs=represented,
                             not_directly_represented=len(pairs) - represented)
    return dict(hand_shell_count=len(hands), native_hand_mesh_count=len(native),
        native_open_mesh_count=sum(not r["watertight"] for r in records),
        hand_meshes=records,
        native_meshes_without_same_body_shell=[r["visual"] for r in records if not r["same_body_shells"]],
        shell_intrahand_possible=len(intra), shell_intrahand_declared=len(intra & explicit),
        shell_intrahand_omitted=len(omitted), omitted_nonadjacent_shell_pairs=len(omitted & nonadjacent),
        omitted_adjacent_shell_pairs=len(omitted - nonadjacent),
        shell_interhand_declared=len(explicit - intra),
        native_body_pair_counts=counts,
        declared_physical_pair_count=int(model.npair),
        automatic_mask_collision_disabled=bool(not model.geom_contype.any() and not model.geom_conaffinity.any()),
        family_pair_counts={k: len(v) for k, v in collision_families(model).items()},
        coverage_complete=False,
        caveats=["same-body shell inventory is not a surface-containment or hardware-coverage certificate",
                 "an absent shell may be partly covered by a neighboring shell; this has not been certified",
                 "omitted adjacent pairs are not automatically permitted to intersect",
                 "counting all existing shell pairs does not establish native surface coverage"])


def run(scene, baseline, candidate_directory, assembly, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = [scene, baseline / "robot_reference.npz", candidate_directory / "initial_hand_candidate.npz",
              candidate_directory / "report.json", candidate_directory / "audit.json",
              candidate_directory / "native_geometry_audit.json", candidate_directory / "initial_control_audit.json",
              assembly / "report.json", assembly / "xhand_left.urdf", assembly / "xhand_right.urdf"]
    preserved = [artifact(p) for p in inputs]
    mesh_snapshot = scene_mesh_artifacts(scene)
    for path in inputs[3:8]:
        old = json.loads(path.read_text())
        for key in ("preserved_artifacts", "source_assets", "source_meshes", "scene_meshes"):
            verify_artifacts(old.get(key, []))
    model = mujoco.MjModel.from_xml_path(str(scene))
    meshes, _ = visual_meshes(scene, model)
    with np.load(inputs[1], allow_pickle=False) as source:
        original = validate_qpos(model, source["qpos"], trajectory=True)
    with np.load(inputs[2], allow_pickle=False) as source:
        if not bool(source["diagnostic_only"]) or bool(source["accepted_as_reset"]):
            raise ValueError("expected the preserved temporary diagnostic candidate")
        candidate = validate_qpos(model, source["qpos"], trajectory=True)
    if len(candidate) != 1 or not np.array_equal(candidate[0, 36:], original[0, 36:]):
        raise ValueError("expected one candidate with unchanged object coordinates")
    records = []
    for label, qpos in (("original_first_frame", original[0]), ("temporary_v4_candidate", candidate[0])):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        records.append(dict(state=label, declared_state=state_summary(model, qpos),
                            native_thumb={s: native_intersection(model, data, meshes, s) for s in ("right", "left")}))
    old_native = json.loads(inputs[5].read_text())
    object_provenance = {}
    for object_id in ("022", "135"):
        path = ROOT / f"models/taco_xhand/assets/objects/{object_id}/convex_m/provenance.json"
        record = json.loads(path.read_text())
        verify_artifacts([dict(path=a["path"], sha256=a["sha256"]) for a in record["artifacts"]])
        if artifact(Path(record["source"]))["sha256"] != record["source_sha256"]:
            raise ValueError("object source mesh changed")
        object_provenance[object_id] = dict(provenance=artifact(path), parts=record["parts"],
            parameters=record["parameters"], requested_threshold_certified=record["requested_threshold_certified"])
    old_state = json.loads(inputs[3].read_text())["after"]
    result = dict(status="collision_coverage_review_not_a_model_repair_or_formal_initialization",
        scope="temporary_sample_diagnostics_and_collision_model_inventory",
        inventory=coverage_inventory(model, meshes), states=records,
        existing_v4_declared_state_results_unchanged=records[1]["declared_state"] == old_state,
        original_frames_validated=len(original),
        source_checks={s: kinematic_source_check(model, assembly / f"xhand_{s}.urdf", s) for s in ("right", "left")},
        object_decompositions=object_provenance,
        old_native_audit_asset_hashes_verified=True,
        old_native_audit_had_full_collision_mesh_snapshot=bool(old_native.get("scene_meshes")),
        current_full_scene_mesh_snapshot=mesh_snapshot,
        known_bug_fixes=["reject_nonfinite_states_distances_and_nonunit_quaternions",
                         "derive_source_kinematic_conclusion_from_numeric_and_topology_checks",
                         "verify_native_mesh_hashes_before_reusing_reports_and_snapshot_external_collision_meshes"],
        model_or_pairs_modified=False, reference_or_candidate_modified=False, accepted_as_reset=False,
        formal_pipeline_method_selected=False, generalized_initialization_claim=False,
        simulation_steps_executed=0, physical_rollout_executed=False, training_ready=False,
        preserved_artifacts=preserved,
        audit_code=[artifact(Path(__file__)), artifact(ROOT / "scripts/audit_taco_thumb_assembly.py"),
                    artifact(ROOT / "src/egoengine_repro/retarget/collision_audit.py")])
    verify_artifacts(preserved)
    verify_artifacts(mesh_snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result["inventory"].items() if k != "hand_meshes"}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.scene, args.baseline, args.candidate_directory, args.assembly, args.output)


if __name__ == "__main__":
    main()
