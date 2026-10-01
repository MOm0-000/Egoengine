"""Contract tests for PPO visitation logging; no simulator is required."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_spider.rl.training_trace import (
    PpoTrainingTrace,
    SCHEMA,
    WORLD_INDEX_SCHEMA,
)


def _trace(
    path: Path,
    *,
    include_world_index: bool = False,
    run_id: str | None = None,
    training_seed: int | None = None,
) -> PpoTrainingTrace:
    return PpoTrainingTrace(
        path,
        actuator_names=tuple(f"actuator_{index}" for index in range(36)),
        actuator_units=tuple(
            "m" if index in (0, 1, 2, 18, 19, 20) else "rad"
            for index in range(36)
        ),
        object_roles=("tool", "target"),
        hand_roles=("right", "left"),
        residual_scale=0.05,
        residual_clip=0.05,
        ctrlrange_contract={
            "control_clamping_enabled": True,
            "model_disableflags": 0,
            "ctrllimited": [True] * 36,
            "ctrlrange": [[-1.0, 1.0]] * 36,
        },
        include_world_index=include_world_index,
        run_id=run_id,
        training_seed=training_seed,
    )


def _info(outcome_endpoint: int = 21) -> dict:
    contacts = np.zeros((2, 2, 2, 5), dtype=bool)
    contacts[0, 0, 0, 0] = True
    contacts[1, 1, 1, 2] = True
    return {
        "command_reference_endpoint": np.array([outcome_endpoint] * 2, np.int32),
        "reward_reference_endpoint": np.array([outcome_endpoint] * 2, np.int32),
        "next_observation_goal_reference_endpoint": np.array(
            [outcome_endpoint + 1] * 2, np.int32
        ),
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
    mu = np.full((2, 36), 0.1, np.float32)
    sigma = np.full((2, 36), 0.6, np.float32)
    reference_ctrl = np.zeros((2, 36), np.float64)
    requested = np.clip(0.05 * bounded, -0.05, 0.05).astype(np.float64)
    requested[:, 8] = -0.03
    effective = requested.copy()
    effective[:, 8] = 0.0
    lost = requested - effective
    trace.record(
        source_endpoint=np.array([20, 20]),
        outcome_endpoint=np.array([21, 21]),
        sampled_action_preclamp=prelimit,
        sampled_action_clamped=bounded,
        actor_mu=mu,
        actor_sigma=sigma,
        reference_ctrl=reference_ctrl,
        requested_residual=requested,
        effective_residual_after_ctrlrange=effective,
        residual_lost_to_ctrlrange=lost,
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
        sampled_action_preclamp=np.zeros((2, 36), np.float32),
        sampled_action_clamped=np.zeros((2, 36), np.float32),
        actor_mu=np.zeros((2, 36), np.float32),
        actor_sigma=np.ones((2, 36), np.float32),
        reference_ctrl=np.zeros((2, 36), np.float64),
        requested_residual=np.zeros((2, 36), np.float64),
        effective_residual_after_ctrlrange=np.zeros((2, 36), np.float64),
        residual_lost_to_ctrlrange=np.zeros((2, 36), np.float64),
        info=_info(22),
    )
    report = trace.finalize(completed=True)

    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "epoch_0001_summary.json").read_text())
    raw = np.load(output / "epoch_0001_visits.npz", allow_pickle=False)
    assert manifest["schema"] == report["schema"] == SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["incomplete_step_discarded"] is False
    assert manifest["action_contract"]["dimensions"] == 36
    assert manifest["action_contract"]["coordinate_units"][:6] == [
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
    assert summary["right_wrist_translation"][
        "sampled_preclamp_fraction_abs_gt_1"
    ] == pytest.approx(2 / 6)
    np.testing.assert_array_equal(raw["sampled_action_preclamp"], prelimit)
    np.testing.assert_array_equal(raw["sampled_action_clamped"], bounded)
    np.testing.assert_array_equal(raw["actor_mu"], mu)
    np.testing.assert_array_equal(raw["actor_sigma"], sigma)
    np.testing.assert_array_equal(raw["command_reference_endpoint"], [21, 21])
    np.testing.assert_array_equal(raw["reward_reference_endpoint"], [21, 21])
    np.testing.assert_array_equal(
        raw["next_observation_goal_reference_endpoint"], [22, 22]
    )
    np.testing.assert_array_equal(raw["reference_ctrl"], reference_ctrl)
    for name in (
        "requested_residual",
        "effective_residual_after_ctrlrange",
        "residual_lost_to_ctrlrange",
    ):
        assert raw[name].shape == (2, 36)
    assert "right_wrist_applied_residual" not in raw.files
    np.testing.assert_array_equal(
        raw["requested_residual"],
        raw["effective_residual_after_ctrlrange"]
        + raw["residual_lost_to_ctrlrange"],
    )
    assert summary["residual_groups"]["right_fingers"][
        "range_truncated_component_count"
    ] == 2
    for artifact in manifest["epochs"]:
        for key in ("visits", "summary"):
            path = Path(artifact[key]["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact[key]["sha256"]


def test_training_trace_fails_closed_on_bad_shape_and_duplicate_finalize(tmp_path):
    trace = _trace(tmp_path / "visits")
    with pytest.raises(RuntimeError, match="without an active"):
        trace.record(
            source_endpoint=np.array([0]), outcome_endpoint=np.array([1]),
            sampled_action_preclamp=np.zeros((1, 36)),
            sampled_action_clamped=np.zeros((1, 36)),
            actor_mu=np.zeros((1, 36)),
            actor_sigma=np.ones((1, 36)),
            reference_ctrl=np.zeros((1, 36)),
            requested_residual=np.zeros((1, 36)),
            effective_residual_after_ctrlrange=np.zeros((1, 36)),
            residual_lost_to_ctrlrange=np.zeros((1, 36)),
            info=_info(),
        )
    trace.begin_epoch(1, 0)
    bad = _info(1)
    bad["contact_flags"] = np.zeros((2, 1, 2, 5), dtype=bool)
    with pytest.raises(ValueError, match="contact flags must have shape"):
        trace.record(
            source_endpoint=np.array([0, 0]), outcome_endpoint=np.array([1, 1]),
            sampled_action_preclamp=np.zeros((2, 36)),
            sampled_action_clamped=np.zeros((2, 36)),
            actor_mu=np.zeros((2, 36)),
            actor_sigma=np.ones((2, 36)),
            reference_ctrl=np.zeros((2, 36)),
            requested_residual=np.zeros((2, 36)),
            effective_residual_after_ctrlrange=np.zeros((2, 36)),
            residual_lost_to_ctrlrange=np.zeros((2, 36)),
            info=bad,
        )


def test_failed_training_before_rollout_writes_explicit_empty_manifest(tmp_path):
    trace = _trace(tmp_path / "visits")
    trace.begin_epoch(1, 0)
    report = trace.finalize(completed=False)
    assert report["epochs"] == []
    assert report["status"] == "training_failed_before_logged_rollout"


def test_world_indexed_trace_records_stable_batch_positions(tmp_path):
    output = tmp_path / "visits"
    trace = _trace(output, include_world_index=True)
    trace.begin_epoch(63, 99_200)
    zeros = np.zeros((2, 36), np.float32)
    info = _info(41)
    for name in ("command_reference_endpoint", "reward_reference_endpoint"):
        info[name] = np.array([41, 61], np.int32)
    info["next_observation_goal_reference_endpoint"] = np.array([42, 62], np.int32)
    trace.record(
        source_endpoint=np.array([40, 60], np.int32),
        outcome_endpoint=np.array([41, 61], np.int32),
        sampled_action_preclamp=zeros,
        sampled_action_clamped=zeros,
        actor_mu=zeros,
        actor_sigma=np.ones((2, 36), np.float32),
        reference_ctrl=zeros.astype(np.float64),
        requested_residual=zeros.astype(np.float64),
        effective_residual_after_ctrlrange=zeros.astype(np.float64),
        residual_lost_to_ctrlrange=zeros.astype(np.float64),
        info=info,
    )
    report = trace.finalize(completed=True)
    with np.load(output / "epoch_0063_visits.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["world_index"], [0, 1])
    assert report["schema"] == WORLD_INDEX_SCHEMA
    assert "stable world-major batch position" in report[
        "logging_semantics"
    ]["world_index"]


def test_world_indexed_trace_joins_world_major_credit_with_explicit_permutation(tmp_path):
    output = tmp_path / "visits"
    trace = _trace(
        output,
        include_world_index=True,
        run_id="candidate_G_seed_2",
        training_seed=2,
    )
    trace.begin_epoch(1, 0)
    zeros = np.zeros((2, 36), np.float32)
    for step in range(2):
        info = _info(41 + step)
        trace.record(
            source_endpoint=np.array([40 + step, 40 + step], np.int32),
            outcome_endpoint=np.array([41 + step, 41 + step], np.int32),
            sampled_action_preclamp=zeros,
            sampled_action_clamped=zeros,
            actor_mu=zeros,
            actor_sigma=np.ones((2, 36), np.float32),
            reference_ctrl=zeros.astype(np.float64),
            requested_residual=zeros.astype(np.float64),
            effective_residual_after_ctrlrange=zeros.astype(np.float64),
            residual_lost_to_ctrlrange=zeros.astype(np.float64),
            info=info,
        )
    # PPO order is w0t0,w0t1,w1t0,w1t1; visitation order is t0w0,t0w1,t1w0,t1w1.
    credit = np.arange(4, dtype=np.float32)
    trace.attach_credit(
        rollout_value_before_update=credit,
        gae_return=credit + 10,
        raw_advantage=credit + 20,
        normalized_advantage=credit + 30,
        canonical_old_logprob=credit + 40,
        shaped_training_reward=credit + 50,
        actor_hash="actor",
        rms_hash="rms",
        normalization_version=125,
        reset_context_hash="context",
    )
    report = trace.finalize(completed=True)
    with np.load(output / "epoch_0001_visits.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["raw_advantage"], [20, 22, 21, 23])
        np.testing.assert_array_equal(data["ppo_flat_index"], [0, 2, 1, 3])
        np.testing.assert_array_equal(data["ppo_world_index"], [0, 1, 0, 1])
        np.testing.assert_array_equal(data["ppo_time_index"], [0, 0, 1, 1])
        np.testing.assert_array_equal(data["rollout_step"], [0, 0, 1, 1])
        np.testing.assert_array_equal(data["episode_serial"], [0, 0, 0, 1])
        assert set(data["run_id"].tolist()) == {"candidate_G_seed_2"}
        assert set(data["actor_hash"].tolist()) == {"actor"}
    assert report["row_identity"]["fields"][:3] == [
        "run_id", "training_seed", "epoch"
    ]
