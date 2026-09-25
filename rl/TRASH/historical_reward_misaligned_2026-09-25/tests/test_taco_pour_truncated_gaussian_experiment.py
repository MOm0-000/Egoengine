"""Evidence checks for the single authorized no-commit PPO experiment."""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_state_feasible_truncated_gaussian_experiment_v1"


def test_authorized_experiment_is_complete_but_did_not_pass_40_of_40():
    report = json.loads((RUN / "report.json").read_text())
    assert report["schema"] == (
        "taco_pour_state_feasible_truncated_gaussian_experiment_run_v1"
    )
    assert report["status"] == "diagnostic_complete_no_commit"
    traces = {
        trace["mode"]: trace
        for trace in report["diagnostic"]["validation_traces"]
    }
    assert traces["diagnostic_replay"]["validated_steps"] == 29
    assert traces["diagnostic_rl"]["validated_steps"] == 37
    assert traces["diagnostic_rl"]["first_failure"]["endpoint"] == 58
    assert traces["diagnostic_rl"]["feasible"] is False


def test_experiment_removed_training_action_clamps_without_promoting_chunk():
    analysis = json.loads((RUN / "analysis.json").read_text())
    channel = analysis["training_action_channel"]
    assert channel["sample_count"] == 1280
    assert channel["ordinary_minus1_plus1_clamp_changed_components"] == 0
    assert channel["ctrlrange_lost_components_exact"] == 0
    assert channel["residual_identity_max_abs_error"] == 0.0
    assert analysis["deterministic_CPU_validation"]["strict_40_of_40_passed"] is False
    assert analysis["promotion"] == {
        "chunk_committed": False,
        "optimized_trajectory_written": False,
        "incoming_endpoint_20_boundary_restored": True,
        "full_RL_authorized_by_this_result": False,
    }
