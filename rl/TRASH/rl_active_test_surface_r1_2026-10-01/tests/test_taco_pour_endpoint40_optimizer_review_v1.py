import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_endpoint40_optimizer_review_v1"


def load_json(name: str):
    return json.loads((RUN / name).read_text())


def test_decision_stops_before_candidate_e():
    decision = load_json("decision.json")
    assert decision["classification"] == "LOOKAHEAD_VISITATION_BOTTLENECK"
    assert decision["action"] == "stop_no_candidate_E"
    assert decision["candidate_E_authorized"] is False
    assert decision["candidate_E_executed"] is False
    assert decision["new_simulation_or_training_steps"] == 0
    assert decision["immutable_progress"] == {
        "endpoint20_to40_committed": True,
        "endpoint40_to60_committed": False,
    }
    assert not (RUN / "candidate_E").exists()


def test_prefrozen_visitation_rule_is_satisfied():
    decision = load_json("decision.json")
    evidence = decision["evidence"]
    np.testing.assert_allclose(
        evidence["failed_second_half_deep_lookahead_fractions"],
        [0.00225, 0.0066, 0.00255],
        rtol=0.0,
        atol=0.0,
    )
    assert evidence["failed_second_half_median_deep_lookahead_fraction"] == 0.00255
    assert evidence["successful_control_deep_lookahead_fraction"] == (
        0.43214285714285716
    )
    assert evidence["failed_to_successful_control_ratio"] == (
        0.005900826446280992
    )
    assert evidence["lookahead_condition_passed"] is True
    assert evidence["optimizer_branch_visitation_condition_passed"] is False


def test_visitation_curves_preserve_k20_vs_deep_lookahead():
    report = load_json("visitation_comparison.json")
    control = report["successful_control"]["visitation_segments"]["middle"]
    assert control["sample_count"] == 63 * 160
    assert control["deep_lookahead_fraction_k_21_39"] > 0.43
    for run in report["failed_restarts"].values():
        second_half = run["failed_run_second_half"]
        assert second_half["sample_count"] == 125 * 160
        assert second_half["deep_lookahead_fraction_k_21_39"] < 0.012
        assert second_half["fraction_k_ge_25"] == 0.0
        assert second_half["next_chunk_boundary_action_fraction_k_20"] > 0.0
        assert len(second_half["sample_count_by_relative_source_offset"]) == 40


def test_credit_join_fails_closed_without_row_identity():
    report = load_json("visitation_comparison.json")
    join = report["credit_join"]
    assert join["status"] == "credit_join_not_proven"
    assert join["joined_statistics_emitted"] is False
    assert all(
        item["shared_sample_id_field"] is False
        for item in join["inspected_schema_examples"]
    )


def test_optimizer_evidence_is_descriptive_only():
    report = load_json("optimizer_comparison.json")
    assert "cannot authorize Candidate E" in report["interpretation_boundary"]
    control = report["successful_control"]["middle"]
    assert control["epoch_count"] == 63
    for run in report["failed_restarts"].values():
        assert set(run) == {"early", "middle", "late_a", "late_b"}
        for segment in run.values():
            assert segment["exact_truncated_KL"][
                "sample_weighted_mean_from_epoch_means"
            ] > 0.0
            assert segment["ratio_outside_0p8_1p2"]["mean"] > 0.0


def test_contract_and_protocol_keep_commit_boundary_frozen():
    contract = yaml.safe_load((RUN / "contract.yaml").read_text())
    canonical = yaml.safe_load(
        (ROOT / "configs/taco_pour_endpoint40_optimizer_review_v1.yaml").read_text()
    )
    assert contract == canonical
    assert contract["immutable_progress"]["endpoint20_to40_committed"] is True
    assert contract["immutable_progress"]["endpoint40_to60_committed"] is False
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    state = protocol["evaluation"]["progressive_chunk_execution_v1"]
    review = state["endpoint40_optimizer_review_v1"]
    assert review["classification"] == "LOOKAHEAD_VISITATION_BOTTLENECK"
    assert review["Candidate_E_authorized"] is False
    assert state["chunk_local_solver"]["chunk_commit_written"] is False
