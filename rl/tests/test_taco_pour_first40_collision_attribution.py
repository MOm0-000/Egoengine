"""The first-40 audit must separate missing pairs from shape under-coverage."""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/taco_pour_first40_collision_attribution_v1.yaml"
REPORT = ROOT / "runs/taco_pour_first40_collision_attribution_v1/report.json"


def _pair(report, names):
    return next(row for row in report["pairs"] if row["geoms"] == list(names))


def test_protocol_is_read_only_and_binds_every_input():
    protocol = yaml.safe_load(PROTOCOL.read_text())
    assert protocol["scope"] == "read_only_collision_diagnostic"
    assert not protocol["training_ready"] and not protocol["repair_allowed"]
    assert protocol["simulation_steps"] == 0
    bound = [
        (protocol["formal_scene"]["path"], protocol["formal_scene"]["sha256"]),
        (protocol["robot_reference"]["path"], protocol["robot_reference"]["sha256"]),
        (protocol["native_preflight"]["report"], protocol["native_preflight"]["report_sha256"]),
        (protocol["native_preflight"]["checks"], protocol["native_preflight"]["checks_sha256"]),
    ]
    bound += [(row["state"], row["state_sha256"])
              for row in protocol["diagnostic_probes"]]
    for path, digest in bound:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_first40_counts_are_explicit_and_do_not_promote_training():
    report = json.loads(REPORT.read_text())
    assert report["selection"]["selected_pairs"] == 15
    assert report["window"] == {
        "first_endpoint": 0, "last_endpoint_inclusive": 40, "endpoints": 41,
    }
    assert report["summary"] == {
        "native_crossing_frame_pairs": 157,
        "native_material_interference_over_50um_frame_pairs": 155,
        "runtime_detected_native_crossing_frame_pairs": 132,
        "runtime_missed_native_crossing_frame_pairs": 25,
        "runtime_missed_material_interference_frame_pairs": 23,
        "runtime_false_positive_frame_pairs_within_selected_pairs": 23,
        "miss_attribution_counts": {
            "first_runtime_proxy_underfills_contact_region": 6,
            "runtime_pair_missing": 19,
        },
    }
    assert not report["formal_scene_modified"] and not report["reference_modified"]
    assert report["simulation_steps_executed"] == 0
    assert not report["repair_applied"] and not report["training_ready"]


def test_candidate_a_seeds_identify_hand_underfill_and_missing_pair():
    report = json.loads(REPORT.read_text())
    probe = report["diagnostic_probes"][0]
    values = {tuple(row["geoms"]): row for row in probe["pairs"]}
    for names in (
        ("left_middle_link2_visual", "left_object_visual"),
        ("left_pinky_link2_visual", "left_object_visual"),
    ):
        row = values[names]
        assert row["classification"] == "first_runtime_proxy_underfills_contact_region"
        assert row["declared_runtime_geom_pair_count"] == 32
        assert not row["runtime_first_x_native_second"]
        assert row["native_first_x_runtime_second"]
    palm_thumb = values[("left_hand_link_visual", "left_thumb_rota1_visual")]
    assert palm_thumb["classification"] == "runtime_pair_missing"
    assert palm_thumb["declared_runtime_geom_pair_count"] == 0
    assert palm_thumb["counterfactual_proxy_proxy_shape_collision"]


def test_existing_palm_thumb_shapes_would_false_positive_on_clear_rows():
    report = json.loads(REPORT.read_text())
    pair = _pair(report, ("left_hand_link_visual", "left_thumb_rota1_visual"))
    assert len(pair["native_crossing_rows"]) == 19
    clear = [row for row in pair["rows"] if not row["native_native_surface_crossing"]]
    assert len(clear) == 22
    assert all(row["counterfactual_proxy_proxy_shape_collision"] for row in clear)
