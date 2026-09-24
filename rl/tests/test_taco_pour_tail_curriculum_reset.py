"""Evidence contract for the no-training tail curriculum reset gate."""

import gzip
import hashlib
import io
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_tail_curriculum_reset_gate_v1"
REPORT = RUN / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tail_reset_gate_is_exact_and_does_not_authorize_training():
    report = json.loads(REPORT.read_text())
    assert report["schema"] == "taco_pour_tail_curriculum_reset_gate_v1"
    assert report["status"] == "passed"
    assert report["scope"]["training_executed"] is False
    assert report["natural_rollout"]["gt_state_injected"] is False
    assert report["natural_rollout"]["captured_endpoints"] == [46, 47, 48, 49, 50]
    assert report["paired_boundary_artifact"]["warp_state_fields_per_boundary"] == [342] * 5
    assert report["four_world_restore"]["physics_bitwise_equal"] == [True] * 4
    assert report["four_world_restore"]["observation_bitwise_equal"] is True
    assert report["deterministic_one_interval_repeat"]["all_bitwise_equal"] is True
    assert report["stale_actor_memory_rejection"]["passed"] is True
    refresh = report["actor_update_memory_refresh"]
    assert refresh["same_actor_replay_matches_saved_RNN_bitwise"] == [True] * 5
    assert refresh["changed_actor_old_memory_rejected"] is True
    assert refresh["changed_actor_refreshed_memory_accepted"] is True
    assert refresh["physical_state_unchanged_by_refresh"] is True
    assert refresh["passed"] is True
    assert report["semantic_limit"][
        "saved_memory_reusable_after_actor_update_without_refresh"
    ] is False
    assert report["PPO_training_authorized"] is False


def test_paired_boundary_artifact_is_hash_bound_and_complete():
    report = json.loads(REPORT.read_text())
    record = report["paired_boundary_artifact"]
    artifact = ROOT / record["path"]
    assert artifact == RUN / "paired_boundaries.pt.gz"
    assert _sha256(artifact) == record["sha256"]
    payload = torch.load(
        io.BytesIO(gzip.decompress(artifact.read_bytes())),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["schema"] == "taco_pour_tail_curriculum_boundaries_v1"
    boundaries = payload["boundaries"]
    assert [row["reference_endpoint"] for row in boundaries] == [46, 47, 48, 49, 50]
    for boundary in boundaries:
        assert boundary["schema"] == "egoengine_physics_rnn_boundary_v1"
        assert len(boundary["physics_state"]["warp_state_keys"]) == 342
        assert [tuple(state.shape) for state in boundary["rnn_states"]] == [
            (1, 1, 1024), (1, 1, 1024)
        ]
        assert boundary["provenance"]["natural_simulator_rollout"] is True
        assert boundary["provenance"]["gt_state_injected"] is False
        assert len(boundary["observation_prefix"]) == (
            boundary["reference_endpoint"] - boundary["rollout_start_endpoint"]
        )
        assert boundary["exploration_rng_restored"] is False


def test_reset_contract_is_bound_and_formal_sampler_gate_is_separate():
    report = json.loads(REPORT.read_text())
    contract = report["contracts"]["tail_curriculum_reset"]
    path = ROOT / contract["path"]
    assert path == ROOT / "configs/taco_pour_tail_curriculum_reset_v1.yaml"
    assert _sha256(path) == contract["sha256"]
    protocol = (ROOT / "configs/replay_rl_protocol.yaml").read_text()
    assert "curriculum_training_ready: false" in protocol
    assert "fixed_world_start_endpoints: [20, 20, 20, 46]" in protocol
    assert "remaining_engineering_blocker: null" in protocol
    assert "strict_40_of_40_achieved: false" in protocol
