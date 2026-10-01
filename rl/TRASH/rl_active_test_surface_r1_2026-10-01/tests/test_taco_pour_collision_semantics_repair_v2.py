"""The v2 repair candidate must remain explicit, isolated and fail-closed."""

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoengine_repro.action.geometry import explicit_collision_pairs, physical_object_geom_ids

PROTOCOL = ROOT / "configs/taco_pour_collision_semantics_repair_v2.yaml"
RUN = ROOT / "runs/taco_pour_collision_semantics_repair_v2"
REJECTED = ROOT / "TRASH/rejected_candidates/2026-09-20_collision_semantics_repair_v2_self_guard"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_protocol_binds_inputs_and_does_not_add_object_avoidance_to_mink():
    protocol = yaml.safe_load(PROTOCOL.read_text())
    assert protocol["training_ready"] is False
    assert protocol["formal_promotion_requires_all_gates"] is True
    for row in protocol["inputs"].values():
        assert _sha(row["path"]) == row["sha256"]
    contract = protocol["contracts"]
    assert contract["add_hand_object_avoidance_to_mink"] is False
    assert contract["external_repairs_are_runtime_physics_only"] is True
    assert contract["object_gt_trajectory_unchanged"] is True


def test_candidate_changes_only_declared_semantic_families():
    protocol = yaml.safe_load(PROTOCOL.read_text())
    source = ET.parse(protocol["inputs"]["scene"]["path"]).getroot()
    build = json.loads((RUN / "build_report.json").read_text())
    candidate_path = Path(build["candidate_scene"]["path"])
    assert candidate_path.name == "candidate_runtime_semantics_scene.xml"
    assert _sha(candidate_path) == build["candidate_scene"]["sha256"]
    candidate = ET.parse(candidate_path).getroot()
    source_names = {geom.get("name", "") for geom in source.findall(".//geom")}
    assert not any("_external_semantic_" in name or "_floor_semantic_" in name
                   for name in source_names)
    candidate_names = {geom.get("name", "") for geom in candidate.findall(".//geom")}
    assert sum("_external_semantic_" in name for name in candidate_names) == 9
    assert sum("_floor_semantic_" in name for name in candidate_names) == 26
    changes = build["changes"]
    assert changes["old_specific_object_pairs_removed"] == 128
    assert changes["semantic_object_pairs_added"] == 288
    assert changes["legacy_hand_floor_pairs_removed"] == 24
    assert changes["semantic_mesh_floor_pairs_added"] == 26
    assert changes["compiled_npair"] == 2988
    source_ranges = {row.get("name"): row.get("range") for row in source.findall(".//joint")}
    candidate_ranges = {row.get("name"): row.get("range") for row in candidate.findall(".//joint")}
    assert source_ranges == candidate_ranges


def test_candidate_runtime_pair_roles_and_action_geometry_parser_agree():
    build = json.loads((RUN / "build_report.json").read_text())
    model = mujoco.MjModel.from_xml_path(build["candidate_scene"]["path"])
    object_ids = set()
    for body in ("right_object", "left_object"):
        object_ids.update(physical_object_geom_ids(model, mujoco, model.body(body).id))
    families = explicit_collision_pairs(
        model, mujoco, object_geom_ids=frozenset(object_ids),
        hand_sides=("right", "left"),
    )
    assert {name: len(rows) for name, rows in families.items()} == {
        "self": 178, "floor": 26, "object": 1696,
    }
    audit = json.loads((RUN / "audit_report.json").read_text())
    contact = audit["contact_and_mink_contract"]
    assert contact["passed"]
    assert contact["semantic_object_pair_count"] == 288
    assert contact["semantic_floor_pair_count"] == 26
    assert contact["external_semantic_geoms_in_mink_self_groups"] == 0
    assert contact["hand_object_mink_limit_enabled"] is False
    assert all(row["finger_map"] == row["expected_finger_map"]
               for row in contact["semantic_geoms"])


def test_external_and_floor_candidate_pass_but_formal_promotion_is_blocked():
    audit = json.loads((RUN / "audit_report.json").read_text())
    assert audit["status"] == "external_candidate_passed_self_guard_still_blocked"
    assert audit["summary"] == {
        "selected_pairs": 15,
        "external_material_false_negatives": 0,
        "external_cad_clear_runtime_false_positives": 0,
        "intrahand_material_false_negatives_pending": 19,
        "candidate_a_external_seeds_detected": 2,
        "candidate_a_self_seed_detected": 0,
    }
    floor = audit["hand_floor_regression"]
    assert floor["frame_link_checks"] == 1066
    assert floor["runtime_false_negatives_over_50um"] == 0
    assert floor["cad_clear_runtime_false_positives"] == 0
    assert floor["passed_native_runtime_contract"]
    assert audit["external_candidate_passed"]
    assert not audit["formal_promotion_passed"]
    assert not audit["formal_scene_modified"] and not audit["reference_modified"]
    assert audit["simulation_steps_executed"] == 0
    assert not audit["training_ready"]


def test_all_left_palm_thumb_candidates_are_rejected_and_quarantined():
    sphere = json.loads((REJECTED / "left_palm_thumb_guard_fit.json").read_text())
    convex = json.loads((REJECTED / "left_palm_thumb_convex_guard_fit.json").read_text())
    hybrid = json.loads((REJECTED / "left_palm_thumb_hybrid_guard_fit.json").read_text())
    assert sphere["status"] == "candidate_guard_fit_rejected"
    assert convex["status"] == "candidate_convex_guard_fit_rejected"
    assert hybrid["status"] == "candidate_hybrid_guard_fit_rejected"
    assert convex["fit"]["uncovered_calibration_positive_count"] == 1
    assert hybrid["validation"]["final_offset_holdout_a"]["false_negative_count"] == 1
    assert hybrid["validation"]["final_offset_holdout_b"]["false_positive_count"] == 1
    audit = json.loads((RUN / "audit_report.json").read_text())
    assert {row["status"] for row in audit["left_palm_thumb_guard_candidates"].values()} == {
        "candidate_guard_fit_rejected",
        "candidate_convex_guard_fit_rejected",
        "candidate_hybrid_guard_fit_rejected",
    }
