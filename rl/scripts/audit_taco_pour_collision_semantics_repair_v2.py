"""Audit the v2 external collision candidate without promoting it to formal."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "scripts"), str(ROOT / "src"),
    "/data_all/zzx/egoengine/spider",
]

from audit_taco_initialization import visual_meshes
from audit_taco_initialization_preflight import triangle_object
from audit_taco_pour_first40_collision_attribution import (
    _collides,
    _declared_pairs,
    _proxy_geoms,
    _runtime_min_distance,
    _runtime_object,
    _set_transforms,
)
from build_taco_pour_initialization_candidates import _world_mesh
from egoengine_repro.retarget.mink import _explicit_collision_groups
from spider.config import build_force_closure_geom_maps

PROTOCOL = ROOT / "configs/taco_pour_collision_semantics_repair_v2.yaml"
RUN = ROOT / "runs/taco_pour_collision_semantics_repair_v2"
REJECTED = ROOT / "TRASH/rejected_candidates/2026-09-20_collision_semantics_repair_v2_self_guard"


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(protocol_path: Path):
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol["protocol_name"] != "taco_pour_collision_semantics_repair_v2":
        raise ValueError("unsupported protocol")
    for row in protocol["inputs"].values():
        if _sha(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"frozen input changed: {row['path']}")
    return protocol


def _runtime_inputs(scene: Path):
    model = mujoco.MjModel.from_xml_path(str(scene))
    meshes, _ = visual_meshes(scene, model)
    native = {geom: triangle_object(mesh) for geom, mesh in meshes.items()}
    runtime_ids = [
        geom for geom in range(model.ngeom)
        if model.geom(geom).name.startswith("collision_hand_")
        or (
            model.geom(geom).name.startswith(("right_object_", "left_object_"))
            and not model.geom(geom).name.endswith("visual")
        )
    ]
    runtime = {geom: _runtime_object(model, geom) for geom in runtime_ids}
    return model, meshes, native, runtime


def _evaluate_pair(
    model, data, qpos, names, meshes, native, runtime, expected_native=None,
):
    ga, gb = (model.geom(name).id for name in names)
    ba, bb = map(int, model.geom_bodyid[[ga, gb]])
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    _set_transforms(model, data, native, runtime)
    native_collision = _collides([native[ga]], [native[gb]])
    if expected_native is not None and native_collision != expected_native:
        raise ValueError("candidate changed native CAD oracle")
    declared = _declared_pairs(model, ba, bb)
    minimum = _runtime_min_distance(model, data, declared)
    actual = bool(minimum is not None and minimum <= 0.0)
    return {
        "native_native_surface_crossing": native_collision,
        "declared_runtime_pair_collision": actual,
        "declared_runtime_min_distance_m": minimum,
        "classification": (
            "runtime_false_positive" if actual and not native_collision
            else "runtime_detects_native_crossing" if actual
            else "runtime_pair_missing" if native_collision and not declared
            else "runtime_missed_native_crossing" if native_collision
            else "both_clear"
        ),
    }


def _selected_pair_audit(protocol, model, meshes, native, runtime):
    baseline = json.loads(Path(protocol["inputs"]["attribution_v1"]["path"]).read_text())
    with np.load(protocol["inputs"]["robot_reference"]["path"], allow_pickle=False) as source:
        qpos = np.asarray(source["qpos"], dtype=float)
    data = mujoco.MjData(model)
    records = []
    for old in baseline["pairs"]:
        rows = []
        for endpoint, old_row in enumerate(old["rows"]):
            new = _evaluate_pair(
                model, data, qpos[endpoint], old["geoms"], meshes, native, runtime,
                expected_native=old_row["native_native_surface_crossing"],
            )
            rows.append({
                "endpoint": endpoint,
                "native_crossing": new["native_native_surface_crossing"],
                "material_over_50um": bool(
                    old_row["native_material_evidence"]
                    and old_row["native_material_evidence"]["material_interference_over_50um"]
                ),
                "runtime_collision": new["declared_runtime_pair_collision"],
                "runtime_min_distance_m": new["declared_runtime_min_distance_m"],
                "classification": new["classification"],
            })
        records.append({
            "geoms": old["geoms"], "family": old["family"], "rows": rows,
            "material_false_negative_rows": [
                row["endpoint"] for row in rows
                if row["material_over_50um"] and not row["runtime_collision"]
            ],
            "cad_clear_runtime_false_positive_rows": [
                row["endpoint"] for row in rows
                if not row["native_crossing"] and row["runtime_collision"]
            ],
        })
    return records


def _candidate_a(protocol, model, meshes, native, runtime):
    baseline = json.loads(Path(protocol["inputs"]["attribution_v1"]["path"]).read_text())
    old_rows = baseline["diagnostic_probes"][0]["pairs"]
    with np.load(protocol["inputs"]["candidate_a"]["path"], allow_pickle=False) as source:
        qpos = np.asarray(source["qpos"], dtype=float)
    data = mujoco.MjData(model)
    return [
        {"geoms": row["geoms"], **_evaluate_pair(
            model, data, qpos, row["geoms"], meshes, native, runtime,
            expected_native=row["native_native_surface_crossing"],
        )}
        for row in old_rows
    ]


def _contact_and_mink_contract(protocol, model):
    finger_map, hand_map, object_map = build_force_closure_geom_maps(model, "bimanual")
    expected = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}
    semantic = []
    semantic_ids = set()
    for spec in protocol["external_semantic_geoms"]:
        geom = model.geom(spec["name"]).id
        semantic_ids.add(geom)
        side = 0 if "_right_" in spec["name"] else 1
        finger = next(name for name in expected if f"_{name}_" in spec["name"])
        semantic.append({
            "geom": spec["name"],
            "finger_map": int(finger_map[geom]),
            "expected_finger_map": side * 5 + expected[finger],
            "hand_group_map": int(hand_map[geom]),
            "expected_hand_group_map": side,
        })
    floor_ids = {
        geom for geom in range(model.ngeom)
        if "_floor_semantic_" in model.geom(geom).name
    }
    object_ids = {geom for geom, group in enumerate(object_map) if group >= 0}
    hand_ids = {
        geom for geom in range(model.ngeom)
        if model.geom(geom).name.startswith("collision_hand_")
    }
    self_groups = _explicit_collision_groups(
        model, mujoco, hand_geom_ids=hand_ids, object_geom_ids=None
    )
    bad_pairs = []
    semantic_object_pairs = 0
    semantic_floor_pairs = 0
    for first, second in zip(model.pair_geom1, model.pair_geom2):
        first, second = int(first), int(second)
        if first not in semantic_ids and second not in semantic_ids:
            if first in floor_ids or second in floor_ids:
                other = second if first in floor_ids else first
                if model.geom(other).name == "floor":
                    semantic_floor_pairs += 1
                else:
                    bad_pairs.append([model.geom(first).name, model.geom(second).name])
            continue
        other = second if first in semantic_ids else first
        if other in object_ids:
            semantic_object_pairs += 1
        else:
            bad_pairs.append([model.geom(first).name, model.geom(second).name])
    passed = (
        all(row["finger_map"] == row["expected_finger_map"] for row in semantic)
        and all(row["hand_group_map"] == row["expected_hand_group_map"] for row in semantic)
        and semantic_object_pairs == 9 * 32
        and len(floor_ids) == 26
        and semantic_floor_pairs == 26
        and not bad_pairs
        and not any(
            first[0] in {model.geom(g).name for g in semantic_ids}
            or second[0] in {model.geom(g).name for g in semantic_ids}
            for first, second in self_groups
        )
    )
    return {
        "passed": passed,
        "semantic_geoms": semantic,
        "semantic_object_pair_count": semantic_object_pairs,
        "semantic_floor_geom_count": len(floor_ids),
        "semantic_floor_pair_count": semantic_floor_pairs,
        "semantic_nonobject_pairs": bad_pairs,
        "mink_self_group_count": len(self_groups),
        "external_semantic_geoms_in_mink_self_groups": 0,
        "hand_object_mink_limit_enabled": False,
    }


def _hand_floor_regression(source_scene, candidate_scene, qpos):
    source, source_meshes, _, _ = _runtime_inputs(source_scene)
    candidate, candidate_meshes, _, _ = _runtime_inputs(candidate_scene)
    source_data, candidate_data = mujoco.MjData(source), mujoco.MjData(candidate)
    floor_source = source.geom("floor").id
    floor_candidate = candidate.geom("floor").id
    if int(source.geom_type[floor_source]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
        floor_top = float(source.geom_pos[floor_source, 2])
    else:
        floor_top = float(source.geom_pos[floor_source, 2] + source.geom_size[floor_source, 2])
    rows = []
    for endpoint in range(41):
        endpoint_rows = []
        for name in sorted(
            source.geom(geom).name for geom in source_meshes
            if "_hand_" in source.geom(geom).name or any(
                token in source.geom(geom).name
                for token in ("thumb_", "index_", "middle_", "ring_", "pinky_")
            )
        ):
            sg = source.geom(name).id
            cg = candidate.geom(name).id
            source_data.qpos[:] = qpos[endpoint]
            candidate_data.qpos[:] = qpos[endpoint]
            mujoco.mj_forward(source, source_data)
            mujoco.mj_forward(candidate, candidate_data)
            native_min = float(_world_mesh(source, source_data, sg, source_meshes[sg]).vertices[:, 2].min())
            sb = int(source.geom_bodyid[sg])
            cb = int(candidate.geom_bodyid[cg])
            spairs = _declared_pairs(source, sb, int(source.geom_bodyid[floor_source]))
            cpairs = _declared_pairs(candidate, cb, int(candidate.geom_bodyid[floor_candidate]))
            sdist = min((mujoco.mj_geomDistance(source, source_data, *p, 1.0, None) for p in spairs), default=None)
            cdist = min((mujoco.mj_geomDistance(candidate, candidate_data, *p, 1.0, None) for p in cpairs), default=None)
            endpoint_rows.append({
                "visual": name,
                "native_table_distance_m": native_min - floor_top,
                "source_runtime_distance_m": sdist,
                "candidate_runtime_distance_m": cdist,
            })
        rows.extend(endpoint_rows)
    exact = all(
        row["source_runtime_distance_m"] == row["candidate_runtime_distance_m"]
        for row in rows
    )
    false_negatives = [
        {"endpoint": index // 26, **row}
        for index, row in enumerate(rows)
        if row["native_table_distance_m"] < -5e-5
        and (row["candidate_runtime_distance_m"] is None or row["candidate_runtime_distance_m"] >= 0)
    ]
    false_positives = [
        {"endpoint": index // 26, **row}
        for index, row in enumerate(rows)
        if row["native_table_distance_m"] > 0
        and row["candidate_runtime_distance_m"] is not None
        and row["candidate_runtime_distance_m"] < 0
    ]
    native_runtime_match = not false_negatives and not false_positives
    return {
        "passed_native_runtime_contract": native_runtime_match,
        "frame_link_checks": len(rows),
        "native_material_below_table_over_50um": sum(
            row["native_table_distance_m"] < -5e-5 for row in rows
        ),
        "runtime_false_negatives_over_50um": len(false_negatives),
        "cad_clear_runtime_false_positives": len(false_positives),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "source_candidate_runtime_distances_bit_equal": exact,
        "source_candidate_difference_is_expected_repair": not exact and native_runtime_match,
    }


def run(protocol_path: Path, run_dir: Path):
    protocol = _load(protocol_path)
    build = json.loads((run_dir / "build_report.json").read_text())
    candidate_scene = Path(build["candidate_scene"]["path"])
    if _sha(candidate_scene) != build["candidate_scene"]["sha256"]:
        raise ValueError("candidate scene changed after build")
    model, meshes, native, runtime = _runtime_inputs(candidate_scene)
    records = _selected_pair_audit(protocol, model, meshes, native, runtime)
    external = [row for row in records if row["family"] == "hand_object"]
    intrahand = [row for row in records if row["family"] == "intrahand"]
    candidate_a = _candidate_a(protocol, model, meshes, native, runtime)
    contact = _contact_and_mink_contract(protocol, model)
    with np.load(protocol["inputs"]["robot_reference"]["path"], allow_pickle=False) as source:
        qpos = np.asarray(source["qpos"], dtype=float)
    floor = _hand_floor_regression(
        Path(protocol["inputs"]["scene"]["path"]), candidate_scene, qpos
    )
    summary = {
        "selected_pairs": len(records),
        "external_material_false_negatives": sum(
            len(row["material_false_negative_rows"]) for row in external
        ),
        "external_cad_clear_runtime_false_positives": sum(
            len(row["cad_clear_runtime_false_positive_rows"]) for row in external
        ),
        "intrahand_material_false_negatives_pending": sum(
            len(row["material_false_negative_rows"]) for row in intrahand
        ),
        "candidate_a_external_seeds_detected": sum(
            row["declared_runtime_pair_collision"] for row in candidate_a
            if row["geoms"][1] == "left_object_visual"
        ),
        "candidate_a_self_seed_detected": sum(
            row["declared_runtime_pair_collision"] for row in candidate_a
            if row["geoms"][1] != "left_object_visual"
        ),
    }
    external_passed = (
        summary["external_material_false_negatives"] == 0
        and summary["external_cad_clear_runtime_false_positives"] == 0
        and summary["candidate_a_external_seeds_detected"] == 2
        and contact["passed"] and floor["passed_native_runtime_contract"]
    )
    guard_fit = json.loads((REJECTED / "left_palm_thumb_guard_fit.json").read_text())
    convex_fit = json.loads((REJECTED / "left_palm_thumb_convex_guard_fit.json").read_text())
    hybrid_fit = json.loads((REJECTED / "left_palm_thumb_hybrid_guard_fit.json").read_text())
    report = {
        "status": "external_candidate_passed_self_guard_still_blocked" if external_passed else "external_candidate_rejected",
        "protocol": {"path": str(protocol_path), "sha256": _sha(protocol_path)},
        "candidate_scene": build["candidate_scene"],
        "summary": summary,
        "selected_pair_results": records,
        "candidate_a": candidate_a,
        "contact_and_mink_contract": contact,
        "hand_floor_regression": floor,
        "left_palm_thumb_guard_candidates": {
            "sphere": {"path": str(REJECTED / "left_palm_thumb_guard_fit.json"),
                       "status": guard_fit["status"]},
            "convex": {"path": str(REJECTED / "left_palm_thumb_convex_guard_fit.json"),
                       "status": convex_fit["status"]},
            "hybrid": {"path": str(REJECTED / "left_palm_thumb_hybrid_guard_fit.json"),
                       "status": hybrid_fit["status"]},
        },
        "external_candidate_passed": external_passed,
        "formal_promotion_passed": external_passed and hybrid_fit["status"] == "candidate_hybrid_guard_fit_passed",
        "formal_scene_modified": False,
        "reference_modified": False,
        "simulation_steps_executed": 0,
        "training_ready": False,
    }
    (run_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(report["status"], summary, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    args = parser.parse_args()
    run(args.protocol, args.run_dir)
