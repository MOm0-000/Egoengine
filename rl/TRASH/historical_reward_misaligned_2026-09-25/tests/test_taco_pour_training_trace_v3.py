"""Evidence checks for the v3 residual-semantics logging gate."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_training_trace_transparency_v3"


def test_v3_logger_is_passive_and_reproduces_frozen_ctrlrange_counts():
    report = json.loads((RUN / "report.json").read_text())
    assert report["schema"] == "taco_pour_training_trace_transparency_v3"
    assert report["status"] == "logging_transparency_gate_passed"
    assert all(report["initial_state_bitwise_equal"].values())
    assert all(report["final_state_bitwise_equal"].values())
    audit_script = report["inputs"]["audit_script"]
    assert hashlib.sha256(
        (ROOT / "scripts/audit_taco_pour_training_trace_transparency.py").read_bytes()
    ).hexdigest() == audit_script["sha256"]

    contract = report["training_trace_contract"]
    assert contract["schema"] == "taco_ppo_training_visitation_v3"
    assert contract["sample_count"] == 4
    assert contract["legacy_ambiguous_field_absent"] is True
    assert contract["requested_equals_effective_plus_lost"] is True
    assert contract["decomposition_max_abs_error"] == 0.0
    assert set(map(tuple, contract["full_residual_array_shapes"].values())) == {(4, 36)}

    regression = report["frozen_3plus1_numeric_regression"]
    assert regression["step_count"] == 33
    assert regression["wrist_translation_lost_component_count"] == 0
    assert regression["wrist_rotation_lost_component_count"] == 0
    assert regression["finger_truncated_component_count"] == 62
    assert regression["finger_fully_blocked_component_count"] == 57
    assert regression["steps_with_any_finger_truncation"] == 28
    assert regression["decomposition_max_abs_error"] == 0.0


def test_v3_raw_trace_uses_full_36d_unambiguous_residual_fields():
    trace_dir = RUN / "with_logging/training_visitation"
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    assert manifest["schema"] == "taco_ppo_training_visitation_v3"
    semantics = manifest["logging_semantics"]
    assert "not realized qpos motion" in semantics[
        "effective_residual_after_ctrlrange"
    ]
    assert "requested_residual" in semantics["legacy_v2_applied_residual"]

    with np.load(trace_dir / "epoch_0001_visits.npz", allow_pickle=False) as arrays:
        for name in (
            "requested_residual",
            "effective_residual_after_ctrlrange",
            "residual_lost_to_ctrlrange",
        ):
            assert arrays[name].shape == (4, 36)
            assert arrays[name].dtype == np.float64
        assert "right_wrist_applied_residual" not in arrays.files
        np.testing.assert_array_equal(
            arrays["requested_residual"],
            arrays["effective_residual_after_ctrlrange"]
            + arrays["residual_lost_to_ctrlrange"],
        )


def test_v2_evidence_remains_immutable_and_explicitly_legacy():
    legacy = ROOT / "runs/taco_pour_training_trace_transparency_v2"
    manifest = json.loads(
        (legacy / "with_logging/training_visitation/manifest.json").read_text()
    )
    assert manifest["schema"] == "taco_ppo_training_visitation_v2"
    with np.load(
        legacy / "with_logging/training_visitation/epoch_0001_visits.npz",
        allow_pickle=False,
    ) as arrays:
        assert "right_wrist_applied_residual" in arrays.files
        assert "requested_residual" not in arrays.files
