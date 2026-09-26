import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_source45_prefix_source50_refinement_gate_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_source45_prefix_source50_refinement_gate_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_source50_gate_is_read_only_and_exactly_reproduced():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert report["status"] == "completed_read_only_gate"
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert contract["primary_passing_condition"]["contact_required"] is False
    assert contract["runtime"]["training_allowed"] is False
    assert report["regression_gate"] == {
        "formal_Replay_trace_bitwise_reproduced": True,
        "formal_PPO_trace_bitwise_reproduced": True,
        "validated_source45_translation_prefix_arrays_bitwise_reproduced": True,
        "restored_prefix_source50_suffix_bitwise_reproduced": True,
        "restored_prefix_endpoint51_physical_record_exact": True,
        "restored_prefix_source50_actor_action_exact": True,
        "formal_Replay_successful_intervals": 30,
        "formal_PPO_successful_intervals": 28,
        "prefix_successful_intervals": 30,
    }


def test_prefix_matches_failure_endpoint_but_not_Replay_physical_basin():
    report = _report()
    result = report["suppression_or_refinement_classification"]
    assert result["Replay_failure_endpoint"] == 51
    assert result["prefix_failure_endpoint"] == 51
    assert result["same_failure_endpoint"] is True
    assert result["same_failure_reason"] is True
    assert result["same_physical_basin_or_failure_mechanism_established"] is False
    assert result["endpoint51_error_decomposition"] == {
        "Replay": {
            "position": 0.6442332555849586,
            "rotation": 0.40550625793356687,
        },
        "source45_translation_suppression_prefix": {
            "position": 0.9303857905024812,
            "rotation": 0.10883657053727962,
        },
    }
    distances = result[
        "endpoint51_prefix_minus_Replay_separate_physical_distances"
    ]
    assert distances["tool_position_l2_m"] == 0.032897932479413076
    assert distances["combined_mixed_unit_distance"] is None
    assert report["measurement_boundaries"][
        "same_failure_endpoint_does_not_by_itself_prove_same_physical_basin"
    ] is True


def test_all_source50_branches_cross_endpoint51_with_ty_as_narrowest():
    expected = {
        "zero_R_forearm_ty_residual_only": (0.9947279095649719, 52),
        "zero_right_wrist_translation": (0.9920123219490051, 53),
        "zero_complete_right_wrist": (0.9920122623443604, 53),
        "zero_entire_right_hand_residual": (0.9920122623443604, 53),
        "zero_full_36d_residual": (0.9920122623443604, 53),
    }
    report = _report()
    for row in report["source50_branches"]:
        score, failure = expected[row["name"]]
        assert row["untargeted_action_dimensions_bitwise_unchanged"] is True
        assert row["endpoint_51_survives"] is True
        assert row["endpoint_51_score"] == score
        assert row["endpoint_51_score_below_prefix_baseline"] is True
        assert row["endpoint_51_score_below_same_run_Replay"] is True
        assert row["first_failure_endpoint_or_60"] == failure
        assert row["passes_predeclared_primary_gate"] is True
        assert row["contact_used_as_acceptance_condition"] is False
    decision = report["decision"]
    assert decision["gate_passed"] is True
    assert decision["narrowest_passing_branch"] == (
        "zero_R_forearm_ty_residual_only"
    )
    assert report["two_stage_result"]["learned_PPO_refinement_proven"] is False


def test_protocol_moves_to_candidate_design_without_authorizing_training():
    report = _report()
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "binary_translation_oracle_36_of_40_endpoint57_failure_attribution_required"
    assert report["decision"]["new_training_authorized"] is False
    assert report["decision"]["actor_LR_5e_minus_5_unblocked"] is False
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"][
        "source45_prefix_source50_refinement_gate"
    ]
    assert gate["narrowest_passing_branch"] == "zero_R_forearm_ty_residual_only"
    assert gate["learned_PPO_refinement_proven"] is False
    assert gate["contact_is_secondary_not_acceptance"] is True
    assert gate["new_training_authorized"] is False
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_binary_translation_oracle_gate"
    )
    assert half_lr["latest_state_entry_evidence"][
        "half_LR_selected_for_next_experiment"
    ] is False
    assert half_lr["execution_gate"]["fresh_training_authorized"] is False
