"""Static contract checks for the prospective Pour coverage experiment."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_training_coverage_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_coverage_analysis_keeps_claims_within_measured_scope():
    analysis = json.loads((RUN / "coverage_analysis.json").read_text())
    assert analysis["schema"] == "taco_pour_prospective_training_coverage_v1"
    assert analysis["status"] == "target_range_visited_but_sparse_with_termination_attrition"
    assert analysis["scope"]["prospective_training_only"] is True
    assert analysis["scope"]["historical_8epoch_coverage_inference_allowed"] is False
    assert analysis["scope"]["actor_mean_mu_recorded"] is False
    assert analysis["scope"]["policy_variance_recorded"] is False
    for artifact in analysis["artifacts"].values():
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert _sha256(path) == artifact["sha256"]
    assert analysis["decision"]["zero_coverage_hypothesis_rejected"] is True
    assert analysis["decision"]["coverage_adequacy_predeclared_threshold_available"] is False
    assert analysis["decision"]["coverage_alone_proven_as_failure_cause"] is False


def test_coverage_counts_and_cpu_validation_match_lossless_trace():
    analysis = json.loads((RUN / "coverage_analysis.json").read_text())
    samples = analysis["training_samples"]
    assert samples["epochs"] == 8
    assert samples["samples_per_epoch"] == 40
    assert samples["total_samples"] == 320
    assert samples["episode_segments_started"] == 13
    assert samples["completed_episode_segments"] == 12
    assert samples["incomplete_segments_at_training_end"] == 1
    assert samples["target_endpoint_total_visits"] == 24
    assert samples["target_endpoint_visit_fraction_of_all_samples"] == 0.075
    assert {
        endpoint: row["visit_count"]
        for endpoint, row in samples["per_target_endpoint"].items()
    } == {"46": 6, "47": 6, "48": 4, "49": 4, "50": 4}
    assert samples["episode_segments_reaching_endpoint"] == {
        "46": 6, "47": 6, "48": 4, "49": 4, "50": 4,
    }
    assert samples["termination_endpoint_counts"] == {
        "34": 1, "39": 1, "40": 1, "43": 1, "45": 2,
        "47": 2, "50": 2, "51": 1, "55": 1,
    }
    validation = analysis["run_outcome"]["cpu_validation"]
    assert validation["replay"]["validated_steps"] == 29
    assert validation["replay"]["first_failure"]["endpoint"] == 50
    assert validation["rl"]["validated_steps"] == 28
    assert validation["rl"]["first_failure"]["endpoint"] == 49
    assert analysis["run_outcome"]["committed_reference_index"] == 20
    assert analysis["run_outcome"]["task_success"] is False


def test_all_epoch_artifacts_are_v2_hash_bound_and_total_320_samples():
    manifest_path = RUN / "ppo_chunk_20/training_visitation/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "taco_ppo_training_visitation_v2"
    assert manifest["status"] == "complete"
    assert len(manifest["epochs"]) == 8
    total = 0
    for artifact in manifest["epochs"]:
        visits = manifest_path.parent / Path(artifact["visits"]["path"]).name
        summary = manifest_path.parent / Path(artifact["summary"]["path"]).name
        assert _sha256(visits) == artifact["visits"]["sha256"]
        assert _sha256(summary) == artifact["summary"]["sha256"]
        with np.load(visits, allow_pickle=False) as arrays:
            count = len(arrays["source_endpoint"])
            assert count == len(arrays["outcome_endpoint"]) == 40
            assert "right_wrist_sampled_action_preclamp" in arrays.files
            assert "right_wrist_sampled_action_clamped" in arrays.files
            assert "right_wrist_applied_residual" in arrays.files
            total += count
    assert total == 320
