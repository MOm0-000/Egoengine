"""Evidence contracts for the four-world Pour training intervention."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "runs/taco_pour_multiworld_state_gate_v1/report.json"
RUN = ROOT / "runs/taco_pour_multiworld_training_v1"
COMPARISON = RUN / "comparison.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_engineering_gate_uses_complete_independent_world_states():
    report = json.loads(GATE.read_text())
    assert report["schema"] == "taco_pour_multiworld_state_gate_v1"
    assert report["status"] == "passed"
    assert report["worlds"] == 4
    assert report["layout"] == "independent_one_world_mjwp_instances"
    assert report["exact_start_state"]["warp_state_fields_per_world"] == 342
    assert report["exact_start_state"]["all_worlds_bitwise_equal"] is True
    assert report["isolated_reset"]["reset_world_matches_complete_boundary"] is True
    assert report["isolated_reset"]["other_worlds_bitwise_unchanged"] is True
    assert report["same_action_baseline"]["gpu_bitwise_repeatability_claimed"] is False
    assert "nondeterminism" in report["short_rollout"]["attribution_limit"]


def test_four_world_coverage_and_cpu_result_match_predeclared_metrics():
    comparison = json.loads(COMPARISON.read_text())
    assert comparison["status"] == (
        "absolute_tail_coverage_and_CPU_validation_improved_but_40_of_40_failed"
    )
    training = comparison["four_world_training"]
    assert training["samples"] == 1280
    assert training["epochs"] == 8
    assert training["worlds"] == 4
    assert {
        endpoint: row["count"]
        for endpoint, row in training["per_target_endpoint"].items()
    } == {"46": 23, "47": 23, "48": 20, "49": 18, "50": 12}
    assert training["target_endpoint_total_visits"] == 96
    assert training["target_endpoint_visit_fraction"] == 0.075
    assert training["per_target_endpoint"]["50"]["epochs_with_visit"] == 6
    validation = comparison["formal_outcome"]["CPU_validation"]
    assert validation["replay"]["validated_steps"] == 29
    assert validation["replay"]["first_failure"]["endpoint"] == 50
    assert validation["rl"]["validated_steps"] == 31
    assert validation["rl"]["first_failure"]["endpoint"] == 52
    assert comparison["formal_outcome"]["committed_reference_index"] == 20
    assert comparison["decision"]["absolute_tail_samples_increased"] is True
    assert comparison["decision"]["tail_sample_fraction_increased"] is False
    assert comparison["decision"]["passed_40_of_40"] is False


def test_four_world_artifacts_and_epoch_logs_are_hash_bound():
    comparison = json.loads(COMPARISON.read_text())
    for artifact in comparison["artifacts"].values():
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert _sha256(path) == artifact["sha256"]
    manifest_path = RUN / "ppo_chunk_20/training_visitation/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "taco_ppo_training_visitation_v2"
    assert len(manifest["epochs"]) == 8
    total = 0
    for artifact in manifest["epochs"]:
        visits = manifest_path.parent / Path(artifact["visits"]["path"]).name
        summary = manifest_path.parent / Path(artifact["summary"]["path"]).name
        assert _sha256(visits) == artifact["visits"]["sha256"]
        assert _sha256(summary) == artifact["summary"]["sha256"]
        with np.load(visits, allow_pickle=False) as arrays:
            assert len(arrays["source_endpoint"]) == 160
            total += len(arrays["source_endpoint"])
    assert total == 1280
