"""Contract tests for PPO visitation logging; no simulator is required."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_spider.rl.training_trace import PpoTrainingTrace, SCHEMA


def _trace(path: Path) -> PpoTrainingTrace:
    return PpoTrainingTrace(
        path,
        actuator_names=tuple(f"actuator_{index}" for index in range(36)),
        object_roles=("tool", "target"),
        hand_roles=("right", "left"),
        residual_scale=0.05,
        residual_clip=0.05,
    )


def _info() -> dict:
    contacts = np.zeros((2, 2, 2, 5), dtype=bool)
    contacts[0, 0, 0, 0] = True
    contacts[1, 1, 1, 2] = True
    return {
        "object_position_error": np.array([[0.04, 0.02], [0.06, 0.03]], np.float32),
        "object_rotation_error": np.array([[0.4, 0.2], [0.6, 0.3]], np.float32),
        "object_tracking_error_per_object": np.array([[0.5, 0.25], [0.75, 0.375]], np.float32),
        "contact_flags": contacts,
        "terminated": np.array([False, True]),
        "time_outs": np.array([False, False]),
    }


def test_training_trace_is_lossless_and_flushes_only_at_epoch_boundaries(tmp_path):
    output = tmp_path / "visits"
    trace = _trace(output)
    trace.begin_epoch(1, 0)
    prelimit = np.zeros((2, 36), np.float32)
    prelimit[:, :6] = [[1.2, 0.5, 0, 0, 0, 0], [-1.4, -0.5, 0, 0, 0, 0]]
    bounded = np.clip(prelimit, -1.0, 1.0)
    applied = np.clip(0.05 * bounded, -0.05, 0.05)
    trace.record(
        source_endpoint=np.array([20, 20]),
        outcome_endpoint=np.array([21, 21]),
        policy_output_prelimit=prelimit,
        policy_output_bounded=bounded,
        applied_residual=applied,
        info=_info(),
    )
    assert list(output.iterdir()) == []

    trace.begin_epoch(2, 2)
    assert {path.name for path in output.iterdir()} == {
        "epoch_0001_visits.npz",
        "epoch_0001_summary.json",
    }
    trace.record(
        source_endpoint=np.array([21, 21]),
        outcome_endpoint=np.array([22, 22]),
        policy_output_prelimit=np.zeros((2, 36), np.float32),
        policy_output_bounded=np.zeros((2, 36), np.float32),
        applied_residual=np.zeros((2, 36), np.float32),
        info=_info(),
    )
    report = trace.finalize(completed=True)

    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "epoch_0001_summary.json").read_text())
    raw = np.load(output / "epoch_0001_visits.npz", allow_pickle=False)
    assert manifest["schema"] == report["schema"] == SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["incomplete_step_discarded"] is False
    assert manifest["action_contract"]["right_wrist_coordinate_units"] == [
        "m", "m", "m", "rad", "rad", "rad"
    ]
    assert [row["sample_count"] for row in manifest["epochs"]] == [2, 2]
    assert summary["source_endpoint_visit_counts"] == {"20": 2}
    assert summary["outcome_endpoint_visit_counts"] == {"21": 2}
    assert summary["tracking_termination_endpoint_counts"] == {"21": 1}
    assert summary["coarse_contact_pattern_counts"] == {
        "left-target:middle": 1,
        "right-tool:thumb": 1,
    }
    assert summary["right_wrist_translation"]["prelimit_fraction_abs_gt_1"] == pytest.approx(2 / 6)
    np.testing.assert_array_equal(raw["right_wrist_policy_output_prelimit"], prelimit[:, :6])
    for artifact in manifest["epochs"]:
        for key in ("visits", "summary"):
            path = Path(artifact[key]["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact[key]["sha256"]


def test_training_trace_fails_closed_on_bad_shape_and_duplicate_finalize(tmp_path):
    trace = _trace(tmp_path / "visits")
    with pytest.raises(RuntimeError, match="without an active"):
        trace.record(
            source_endpoint=np.array([0]), outcome_endpoint=np.array([1]),
            policy_output_prelimit=np.zeros((1, 36)),
            policy_output_bounded=np.zeros((1, 36)),
            applied_residual=np.zeros((1, 36)), info=_info(),
        )
    trace.begin_epoch(1, 0)
    bad = _info()
    bad["contact_flags"] = np.zeros((2, 1, 2, 5), dtype=bool)
    with pytest.raises(ValueError, match="contact flags must have shape"):
        trace.record(
            source_endpoint=np.array([0, 0]), outcome_endpoint=np.array([1, 1]),
            policy_output_prelimit=np.zeros((2, 36)),
            policy_output_bounded=np.zeros((2, 36)),
            applied_residual=np.zeros((2, 36)), info=bad,
        )


def test_failed_training_before_rollout_writes_explicit_empty_manifest(tmp_path):
    trace = _trace(tmp_path / "visits")
    trace.begin_epoch(1, 0)
    report = trace.finalize(completed=False)
    assert report["epochs"] == []
    assert report["status"] == "training_failed_before_logged_rollout"
