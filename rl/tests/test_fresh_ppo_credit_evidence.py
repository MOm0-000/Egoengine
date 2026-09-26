import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "runs/taco_pour_credit_instrumentation_gate_v1/report.json"
RUN = ROOT / "runs/taco_pour_corrected_fresh_ppo_credit_instrumented_v1/report.json"
EVIDENCE = ROOT / "runs/taco_pour_fresh_ppo_credit_evidence_v1/report.json"


def load(path):
    return json.loads(path.read_text())


def test_credit_logger_is_bitwise_transparent_on_real_cpu_pour():
    report = load(GATE)
    assert report["status"] == "passed"
    assert report["task_level_training_executed"] is False
    assert report["chunk_commit_written"] is False
    assert all(report["comparisons"].values())
    assert all(report["checks"].values())


def test_fresh_credit_run_is_nonpromotable_and_unchanged_algorithm():
    report = load(RUN)
    assert report["status"] == "completed_diagnostic_not_promotable"
    assert report["chunk_commit_written"] is False
    assert report["task_success_claimed"] is False
    assert report["performance_comparison_to_prior_gpu_run_allowed"] is False
    assert report["training"] == {
        "worlds": 4,
        "epochs": 8,
        "horizon_per_world_per_epoch": 40,
        "samples_per_epoch": 160,
        "total_samples": 1280,
        "seed": 0,
        "fixed_reset_endpoints": [20, 20, 20, 20],
        "old_actor_or_checkpoint_resume": False,
        "tail_curriculum": False,
    }
    assert report["replay_validation"]["first_failure"]["endpoint"] == 51
    assert report["ppo_validation"]["first_failure"]["endpoint"] == 50
    manifest = load(Path(report["training_audit"]["credit_audit"]["path"]))
    assert manifest["epochs"] == 8
    assert manifest["actor_updates"] == 32
    assert "lossless XOR" in manifest["exact_intermediate_actor_reconstruction"]


def test_evidence_separates_normalizer_state_from_optimizer_update():
    report = load(EVIDENCE)
    assert report["status"] == "completed_read_only_reconstruction"
    assert report["training_executed"] is False
    assert report["actor_update_executed"] is False
    assert report["patch_chain"]["updates_reconstructed"] == 32
    assert report["patch_chain"]["bitwise_chain_complete"] is True
    assert report["first_forward_changed_tensors"] == [
        "running_mean_std.count",
        "running_mean_std.running_mean",
        "running_mean_std.running_var",
    ]
    first = report["first_update"]
    assert first["maximum_abs_mu_y_change_from_normalization_forward"] > 0.34
    assert first["maximum_abs_mu_y_change_from_optimizer"] < 0.106
    ratio = first["ratio_before_optimizer"]
    assert ratio["outside_PPO_clip_count"] == 134
    assert ratio["source_43_46_outside_clip_count"] == 9
    assert ratio["source_43_46_count"] == 10
    assert ratio["minimum"] < 0.001
    assert ratio["maximum"] > 19.0
    assert report["advantage_credit"]["source_43_46_sign_flip_count"] == 39
    assert max(
        row["rollout_logprob_direct_recompute_max_abs_error"]
        for row in report["advantage_credit"]["epochs"]
    ) < 4e-6


def test_protocol_records_the_consumed_single_pass_result_without_authorizing_more_training():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    assert protocol["training_ready"] is False
    blocker = "source46_state_entry_gate_failed_source45_attribution_required"
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    repair = protocol["runtime_contract"]["observation_normalization"]
    assert repair["status"] == "resolved_local_engineering_contract_gate_passed"
    assert repair["first_pre_optimizer_ratio_max_abs_error_from_one"] == 0.0
    assert repair["full_lr_zero_dry_run_ratio_max_abs_error_from_one"] == 0.0
    assert repair["fresh_post_fix_task_PPO_executed"] is True
    old = protocol["historical_observation_normalization_misaligned"]
    assert old["active_algorithm_evidence"] is False
    assert old["old_policy_resume_allowed"] is False
    audit = protocol["evaluation"]["postfix_policy_extremization_audit"]
    assert audit["formal_CPU_traces_bitwise_reproduced"] is True
    assert audit["next_single_variable_candidate"]["status"] == (
        "completed_once_no_rerun_authorized"
    )
