"""Regression checks for the read-only MANO/XHand morphology audit."""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "runs/taco_pour_morphology_audit_v3/report.json"


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_report_is_read_only_and_keeps_grid_blocked(report):
    assert report["status"] == "neutral_morphology_audit_read_only"
    assert not report["weights_modified"]
    assert not report["formal_reference_modified"]
    assert report["conclusions"]["grid_search_allowed"] is False


def test_neutral_geometry_has_finite_metric_contract(report):
    for side in ("right", "left"):
        human = report["human_neutral_geometry"][side]
        robot = report["robot_neutral_geometry"][side]
        assert human["palm_width_index_mcp_to_pinky_mcp_mm"] > 0
        assert robot["palm_width_index_mcp_to_pinky_mcp_mm"] > 0
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            for geometry in (human, robot):
                record = geometry["fingers"][finger]
                assert record["chain_length_mm"] > record["wrist_to_tip_mm"] > 0
                assert np.isfinite(record["segment_lengths_mm"]).all()
                assert np.isfinite(record["neutral_tip_vector_in_palm_frame_mm"]).all()


def test_scale_is_stable_by_posture_but_not_identical_across_fingers(report):
    for side in ("right", "left"):
        spreads = np.asarray(
            report["trajectory_morphology_stability"][side][
                "all_rows_ratio_p95_minus_p05_by_finger"
            ]
        )
        assert spreads.shape == (5,)
        assert np.isfinite(spreads).all()
        assert spreads.max() < 0.05
        scales = [
            report["morphology_comparison"][side]["per_finger"][finger][
                "chain_scale_robot_over_human"
            ]
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        ]
        assert max(scales) - min(scales) > 0.1


def test_diagnostic_scaling_preserves_orientation_and_wrist_targets(report):
    for mode, candidate in report["empirical_scaled_target_probes"].items():
        assert candidate["target_points_only_changed"]
        assert candidate["orientation_targets_unchanged"]
        assert candidate["wrist_targets_unchanged"]
        for solve_mode in ("position_only", "position_wrist", "pose"):
            probe = candidate["probes"][solve_mode]
            assert probe["valid_rows"] == 6
            assert np.isfinite(probe["tip_position_mm_by_hand"]["mean"])
