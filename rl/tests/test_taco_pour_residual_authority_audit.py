"""Evidence checks for the frozen CPU residual-authority audit."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "runs/taco_pour_residual_authority_audit_v1/report.json"
SOURCE_PATH = ROOT / "runs/taco_pour_tail_curriculum_3plus1_v1/report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text())


def test_authority_audit_is_read_only_and_hash_bound():
    report = _report()
    assert report["schema"] == "taco_pour_residual_authority_audit_v1"
    assert report["status"] == "all_groups_use_limit_finger_ctrlrange_further_truncates_no_new_scale_selected"
    assert report["scope"]["training_executed"] is False
    assert report["scope"]["policy_inference_executed"] is False
    assert report["scope"]["physics_rollout_executed"] is False
    assert report["scope"]["new_action_scale_selected"] is False
    assert report["scope"]["next_training_authorized"] is False
    for record in report["preserved_inputs"] + report["audit_code"]:
        path = Path(record["path"])
        assert path.is_file()
        assert _sha256(path) == record["sha256"]


def test_requested_saturation_counts_reproduce_the_frozen_cpu_trace():
    report = _report()
    source = json.loads(SOURCE_PATH.read_text())
    trace = next(row for row in source["chunks"][0]["validation_traces"] if row["mode"] == "rl")
    actions = np.asarray([row["applied_residual"] for row in trace["steps"]])
    groups = {
        "both_wrist_translation": [0, 1, 2, 18, 19, 20],
        "both_wrist_rotation": [3, 4, 5, 21, 22, 23],
        "both_fingers": [*range(6, 18), *range(24, 36)],
    }
    expected_counts = {
        "both_wrist_translation": 36,
        "both_wrist_rotation": 39,
        "both_fingers": 166,
    }
    full = report["group_windows"]["full_endpoint_21_to_53"]
    for name, indices in groups.items():
        saturated = np.abs(actions[:, indices]) >= 0.05 - 2e-7
        assert int(saturated.sum()) == expected_counts[name]
        assert full[name]["requested_residual_at_0p05_limit"]["count"] == int(saturated.sum())
        assert full[name]["requested_residual_at_0p05_limit"]["steps_with_any"] == int(saturated.any(axis=1).sum())


def test_actuator_range_truncation_is_separate_from_policy_saturation():
    report = _report()
    full = report["group_windows"]["full_endpoint_21_to_53"]
    assert full["both_wrist_translation"]["range_truncated_requested_residual"]["count"] == 0
    assert full["both_wrist_rotation"]["range_truncated_requested_residual"]["count"] == 0
    finger = full["both_fingers"]
    assert finger["range_truncated_requested_residual"]["count"] == 62
    assert finger["range_truncated_requested_residual"]["steps_with_any"] == 28
    assert finger["fully_blocked_requested_residual"]["count"] == 57
    truncated = {
        row["actuator"]: row["windows"]["full_endpoint_21_to_53"]["range_truncated_requested_residual"]["count"]
        for row in report["coordinate_audit"]
        if row["windows"]["full_endpoint_21_to_53"]["range_truncated_requested_residual"]["count"]
    }
    assert truncated == {
        "right_thumb_rota2_position": 15,
        "right_index_joint1_position": 5,
        "right_ring_joint2_position": 4,
        "right_pinky_joint2_position": 16,
        "left_thumb_rota2_position": 11,
        "left_pinky_joint1_position": 11,
    }


def test_result_does_not_overclaim_a_new_group_scale():
    report = _report()
    decision = report["decision"]
    assert decision["case_A_angular_limited_while_translation_rarely_limited_supported"] is False
    assert decision["case_B_no_category_materially_uses_limit_supported"] is False
    assert decision["mixed_unit_scale_proven_as_endpoint53_primary_cause"] is False
    assert decision["increasing_only_angular_scale_supported"] is False
    assert decision["increasing_finger_scale_without_ctrlrange_handling_supported"] is False
    assert decision["new_group_scale_values_selected"] is False
