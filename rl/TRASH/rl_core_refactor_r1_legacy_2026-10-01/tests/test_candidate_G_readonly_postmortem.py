from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_taco_pour_candidate_G_readonly_postmortem_v1.py"
SPEC = importlib.util.spec_from_file_location("candidate_g_readonly_postmortem", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def synthetic_grid() -> dict[str, np.ndarray]:
    step = np.repeat(np.arange(40, dtype=np.int32), 4)
    world = np.tile(np.arange(4, dtype=np.int32), 40)
    source = np.full(160, 40, np.int32) + step
    terminated = np.zeros(160, bool)
    timeout = np.zeros(160, bool)
    serial = np.zeros(160, np.int32)
    return {
        "run_id": np.full(160, "candidate_G_seed_0"),
        "training_seed": np.zeros(160, np.int32),
        "world_index": world,
        "rollout_step": step,
        "episode_serial": serial,
        "source_endpoint": source,
        "outcome_endpoint": source + 1,
        "command_reference_endpoint": source + 1,
        "reward_reference_endpoint": source + 1,
        "next_observation_goal_reference_endpoint": source + 2,
        "tracking_terminated": terminated,
        "time_out": timeout,
        "ppo_flat_index": world * 40 + step,
        "ppo_world_index": world.copy(),
        "ppo_time_index": step.copy(),
        "actor_hash": np.full(160, "a" * 64),
        "RMS_hash": np.full(160, "b" * 64),
        "normalization_version": np.ones(160, np.int32),
        "reset_context_hash": np.full(160, "c" * 64),
    }


def action_row(step: int, source: int, *, score: float = 0.9, terminated: bool = False,
               timeout: bool = False, serial: int = 0) -> dict:
    return {
        "rollout_step": step,
        "world_index": 0,
        "episode_serial": serial,
        "source": source,
        "outcome": source + 1,
        "score": score,
        "tracking_terminated": terminated,
        "time_out": timeout,
    }


def terminal_trace(terminal: bool = True) -> dict[str, np.ndarray]:
    count = 3
    rewards = np.asarray([0.5, 0.25, -0.1], np.float32)
    values = np.asarray([0.2, 0.1, 0.3], np.float32)
    gae = np.float32(0.0)
    advantages = np.zeros(count, np.float32)
    for i in range(count - 1, -1, -1):
        if i == count - 1:
            nonterminal = np.float32(0.0 if terminal else 1.0)
            next_value = np.float32(0.0)
        else:
            nonterminal = np.float32(1.0)
            next_value = values[i + 1]
        delta = np.float32(rewards[i] + audit.GAMMA * next_value * nonterminal - values[i])
        gae = np.float32(delta + audit.GAMMA * audit.GAE_TAU * nonterminal * gae)
        advantages[i] = gae
    return {
        "world_index": np.zeros(count, np.int32),
        "episode_serial": np.zeros(count, np.int32),
        "rollout_step": np.arange(count, dtype=np.int32),
        "tracking_terminated": np.asarray([False, False, terminal]),
        "time_out": np.zeros(count, bool),
        "shaped_training_reward": rewards[:, None],
        "rollout_value_before_update": values[:, None],
        "raw_advantage": advantages,
        "GAE_return": (advantages + values)[:, None],
    }


def test_valid_grid_and_world_major_mapping() -> None:
    data = synthetic_grid()
    result = audit.validate_grid(data, seed=0, epoch=1)
    assert all(result.values())
    world_major = np.arange(160)
    time_major = audit.world_major_to_time_major(world_major)
    assert np.array_equal(time_major, data["ppo_flat_index"])


def test_duplicate_or_missing_primary_key_fails() -> None:
    data = synthetic_grid()
    data["world_index"][1] = 0
    with pytest.raises(audit.ContractFailure, match="duplicate or missing"):
        audit.validate_grid(data, seed=0, epoch=1)


def test_episode_serial_must_reset_after_done_within_epoch() -> None:
    data = synthetic_grid()
    terminal_row = 2 * 4
    data["tracking_terminated"][terminal_row] = True
    # world zero rows after step two must belong to episode one.
    data["episode_serial"][(data["world_index"] == 0) & (data["rollout_step"] > 2)] = 1
    assert audit.validate_grid(data, seed=0, epoch=1)["episode_serial_reset_exact"]
    data["episode_serial"][3 * 4] = 0
    with pytest.raises(audit.ContractFailure, match="episode_serial"):
        audit.validate_grid(data, seed=0, epoch=1)


def test_pass61_then_endpoint62_failure() -> None:
    row = action_row(10, 60)
    successor = action_row(11, 61, score=1.1, terminated=True)
    result, observed = audit.classify_pass61_successor(row, {(0, 11): successor})
    assert result == "continued_then_tracking_terminated_at62"
    assert observed is successor


def test_pass61_at_step39_is_right_censored() -> None:
    result, successor = audit.classify_pass61_successor(action_row(39, 60), {})
    assert result == "right_censored_at_rollout_boundary"
    assert successor is None


def test_pass62_at_rollout_boundary_is_not_failure() -> None:
    row = action_row(38, 60)
    successor = action_row(39, 61, score=0.99)
    result, _ = audit.classify_pass61_successor(row, {(0, 39): successor})
    assert result == "continued_and_passed62_then_right_censored"


def test_missing_successor_is_not_joined_from_another_epoch() -> None:
    result, successor = audit.classify_pass61_successor(action_row(20, 60), {})
    assert result == "unexplained_gap"
    assert successor is None


def test_timeout_is_distinct_from_tracking_termination() -> None:
    row = action_row(10, 60, timeout=True)
    result, _ = audit.classify_pass61_successor(row, {})
    assert result == "window_timeout_at61"


def test_terminal_gae_reproduces_and_missing_bootstrap_is_not_fabricated() -> None:
    complete = audit.terminal_gae_audit(terminal_trace(True))
    assert complete["rows_recomputed"] == 3
    assert complete["maximum_abs_GAE_error"] == 0.0
    incomplete = audit.terminal_gae_audit(terminal_trace(False))
    assert incomplete["rows_recomputed"] == 0
    assert incomplete["maximum_abs_GAE_error"] is None


def test_advantage_normalization_uses_unbiased_full_batch_std() -> None:
    raw = np.asarray([1.0, 2.0, 4.0, 8.0], np.float32)
    observed = audit.normalized_advantage(raw)
    expected = (raw - raw.mean()) / (raw.std(ddof=1) + 1.0e-8)
    assert np.array_equal(observed, expected)


def test_truncated_joint_logprob_supports_asymmetric_intervals() -> None:
    action = np.asarray([[0.0, -0.25], [0.8, 0.1]], np.float32)
    mu = np.asarray([[0.0, 0.0], [0.5, 0.0]], np.float32)
    sigma = np.full((2, 2), 0.25, np.float32)
    low = np.asarray([[0.0, -1.0], [0.0, -0.2]], np.float32)
    high = np.ones((2, 2), np.float32)
    value = audit.truncated_joint_logprob(action, mu, sigma, low, high)
    assert value.shape == (2,)
    assert np.isfinite(value).all()
    bad = action.copy(); bad[0, 0] = -0.1
    with pytest.raises(audit.ContractFailure, match="outside"):
        audit.truncated_joint_logprob(bad, mu, sigma, low, high)


def test_raw_ratio_outside_and_true_clip_active_are_distinct() -> None:
    advantages = np.asarray([1.0, -1.0, 1.0, -1.0])
    ratios = np.asarray([0.5, 1.5, 1.5, 0.5])
    assert np.all((ratios < 0.8) | (ratios > 1.2))
    assert np.array_equal(audit.clip_active(advantages, ratios), [False, False, True, True])


def test_input_ledger_detects_old_input_mutation(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text("one")
    ledger = audit.InputLedger()
    ledger.add(path, "test", hashlib.sha256(b"one").hexdigest())
    path.write_text("two")
    report = ledger.finish()
    assert not report["all_old_inputs_unchanged"]
    assert report["changed"] == [str(path.resolve())]


def test_audit_module_has_no_project_runtime_imports() -> None:
    source = SCRIPT.read_text()
    forbidden = (
        "import mujoco", "import warp", "from video_to_spider", "import video_to_spider",
        "from human2sim2robot", "import human2sim2robot",
    )
    assert not any(token in source for token in forbidden)
