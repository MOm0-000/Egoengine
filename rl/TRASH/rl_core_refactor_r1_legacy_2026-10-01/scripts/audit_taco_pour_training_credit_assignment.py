#!/usr/bin/env python3
"""Audit corrected Pour PPO credit assignment without training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_corrected_endpoint44_48_failure_attribution import (  # noqa: E402
    _load_gzip_torch,
    _require_exact_trace,
    _sha256,
)
from audit_corrected_input_bifurcation_attribution import (  # noqa: E402
    _bounds_and_mode,
    _forward_raw,
    _normalization_state,
)
from audit_corrected_local_controllability import _clone_rnn  # noqa: E402
from audit_taco_pour_reference_timing_action_frame import (  # noqa: E402
    REFERENCE_SLICE,
    _reference_context,
)


SOURCE_ENDPOINTS = (43, 44, 45, 46)
TAIL_TRAINING_ENDPOINTS = (43, 44, 45, 46, 47)
ACTION_Y = 1
ACTION_GRID = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _load_contract(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_training_credit_assignment_audit_v1":
        raise ValueError("unsupported training-credit audit contract")
    if contract.get("status") != "authorized_read_only_credit_audit":
        raise ValueError("training-credit audit is not authorized")
    if contract.get("paper_faithful") is not False:
        raise ValueError("local training-credit audit cannot be paper-faithful")
    required = {
        "backend": "CPU_MuJoCo_Warp",
        "optimizer_steps": 0,
        "actor_frozen": True,
        "normalization_frozen": True,
        "reward_changed": False,
        "objective_changed": False,
        "residual_scale_changed": False,
        "residual_bound_changed": False,
        "reference_timing_changed_in_formal_path": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    if any(contract["runtime"].get(key) != value for key, value in required.items()):
        raise ValueError("training-credit runtime contract is not fail-closed")
    for row in contract["inputs"].values():
        artifact = Path(row["path"])
        if _sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
    counterfactual = contract["one_action_counterfactual"]
    if (
        counterfactual["source_endpoints"] != list(SOURCE_ENDPOINTS)
        or counterfactual["absolute_normalized_action_grid"] != list(ACTION_GRID)
        or counterfactual["right_wrist_y_action_index"] != ACTION_Y
        or counterfactual["intervention_duration_control_steps"] != 1
        or counterfactual["final_endpoint_exclusive"] != 60
        or counterfactual["object_tracking_return"]["gamma"] != 0.99
    ):
        raise ValueError("one-action counterfactual differs from frozen definition")
    return contract, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _zero_hidden(policy):
    return [state.to("cpu").zero_() for state in policy.model.get_default_rnn_state()]


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64).copy()


def _active_contacts(flags: np.ndarray) -> list[str]:
    labels = []
    for hand_index, hand in enumerate(("right", "left")):
        for object_index, role in enumerate(("tool", "target")):
            for finger_index, finger in enumerate(FINGERS):
                if bool(flags[hand_index, object_index, finger_index]):
                    labels.append(f"{hand}_{role}_{finger}")
    return labels


def _step_contact_record(backend) -> dict:
    flags = np.asarray(backend.last_info["contact_flags"][0], dtype=bool)
    return {
        "flags": flags.tolist(),
        "active": _active_contacts(flags),
    }


def _capture_formal_ppo(
    *, env, backend, policy, boundary, expected_trace: dict
) -> tuple[dict, dict[int, dict]]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = _zero_hidden(policy)
    backend.begin_trial("rl", 20, 60)
    sources = {}
    validated = 0
    for source in range(20, 60):
        packed = policy.obs_to_tensors(backend.observation())
        hidden = _clone_rnn(policy.rnn_states)
        forward = _forward_raw(policy, packed["obs"], hidden)
        bounded = _bounds_and_mode(env, forward)
        if source in SOURCE_ENDPOINTS:
            sources[source] = {
                "environment_state": backend.snapshot(),
                "pre_forward_hidden": hidden,
                "actor_mu": forward["actor_mu"],
                "state_low": bounded["state_low"],
                "state_high": bounded["state_high"],
                "deterministic_action": bounded["deterministic_action"],
            }
        policy.rnn_states = _clone_rnn(forward["next_hidden"])
        feasible = backend.step(
            bounded["deterministic_action"][None].astype(np.float32), source
        )
        if not feasible:
            break
        validated += 1
    backend.end_trial(validated == 40, validated)
    trace = backend.validation_traces[-1]
    _require_exact_trace(trace, expected_trace, "formal PPO")
    if set(sources) != set(SOURCE_ENDPOINTS):
        raise RuntimeError("formal PPO did not capture every counterfactual source")
    return trace, sources


def _run_replay(*, backend, boundary, expected_trace: dict) -> tuple[dict, dict]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    backend.begin_trial("replay", 20, 60)
    contacts = {}
    validated = 0
    for source in range(20, 60):
        feasible = backend.step(np.zeros((1, 36), dtype=np.float32), source)
        contacts[int(backend.env.time_indices[0])] = _step_contact_record(backend)
        if not feasible:
            break
        validated += 1
    backend.end_trial(validated == 40, validated)
    trace = backend.validation_traces[-1]
    _require_exact_trace(trace, expected_trace, "formal Replay")
    return trace, contacts


def _run_timing_minus_one(
    *, env, backend, policy, boundary, contexts: dict[int, np.ndarray]
) -> tuple[dict, dict]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = _zero_hidden(policy)
    backend.begin_trial("diagnostic_actor_reference_shift_-1", 20, 60)
    contacts = {}
    validated = 0
    for source in range(20, 60):
        packed = policy.obs_to_tensors(backend.observation())
        raw = np.asarray(packed["obs"].detach().cpu().numpy(), dtype=np.float32)
        raw[0, REFERENCE_SLICE] = contexts[source - 1]
        hidden = _clone_rnn(policy.rnn_states)
        forward = _forward_raw(policy, raw, hidden)
        bounded = _bounds_and_mode(env, forward)
        policy.rnn_states = _clone_rnn(forward["next_hidden"])
        feasible = backend.step(
            bounded["deterministic_action"][None].astype(np.float32), source
        )
        contacts[int(env.time_indices[0])] = _step_contact_record(backend)
        if not feasible:
            break
        validated += 1
    backend.end_trial(validated == 40, validated)
    return backend.validation_traces[-1], contacts


def _path_alignment(
    replay: dict,
    replay_contacts: dict,
    timing: dict,
    timing_contacts: dict,
    formal_ppo: dict,
) -> dict:
    replay_rows = {row["endpoint"]: row for row in replay["steps"]}
    timing_rows = {row["endpoint"]: row for row in timing["steps"]}
    ppo_rows = {row["endpoint"]: row for row in formal_ppo["steps"]}
    rows = {}
    timing_qpos_distances = []
    ppo_qpos_distances = []
    common_ppo = []
    for endpoint in range(40, 52):
        replay_row = replay_rows[endpoint]
        timing_row = timing_rows[endpoint]
        replay_qpos = np.asarray(replay_row["endpoint_qpos"], np.float64)
        replay_qvel = np.asarray(replay_row["endpoint_qvel"], np.float64)
        timing_qpos = np.asarray(timing_row["endpoint_qpos"], np.float64)
        timing_qvel = np.asarray(timing_row["endpoint_qvel"], np.float64)
        timing_qpos_distance = float(np.linalg.norm(timing_qpos - replay_qpos))
        timing_qpos_distances.append(timing_qpos_distance)
        row = {
            "endpoint": endpoint,
            "replay": {
                "position_error_m": replay_row["position_error_m"][0],
                "rotation_error_rad": replay_row["rotation_error_rad"][0],
                "tracking_score": replay_row["objective_score"][0],
                "normalized_residual_l2": float(np.linalg.norm(
                    replay_row["raw_residual_action"]
                )),
                "contacts": replay_contacts[endpoint],
            },
            "timing_minus1": {
                "position_error_m": timing_row["position_error_m"][0],
                "rotation_error_rad": timing_row["rotation_error_rad"][0],
                "tracking_score": timing_row["objective_score"][0],
                "normalized_residual_l2": float(np.linalg.norm(
                    timing_row["raw_residual_action"]
                )),
                "contacts": timing_contacts[endpoint],
            },
            "timing_minus1_distance_to_replay": {
                "qpos_l2": timing_qpos_distance,
                "qvel_l2": float(np.linalg.norm(timing_qvel - replay_qvel)),
                "commanded_ctrl_l2": float(np.linalg.norm(
                    np.asarray(timing_row["commanded_ctrl"], np.float64)
                    - np.asarray(replay_row["commanded_ctrl"], np.float64)
                )),
                "tool_position_m": float(np.linalg.norm(
                    timing_qpos[36:39] - replay_qpos[36:39]
                )),
            },
        }
        if endpoint in ppo_rows:
            ppo_distance = float(np.linalg.norm(
                np.asarray(ppo_rows[endpoint]["endpoint_qpos"], np.float64)
                - replay_qpos
            ))
            ppo_qpos_distances.append(ppo_distance)
            common_ppo.append(endpoint)
            row["formal_ppo_qpos_l2_to_replay"] = ppo_distance
        rows[str(endpoint)] = row

    timing_common = [
        rows[str(endpoint)]["timing_minus1_distance_to_replay"]["qpos_l2"]
        for endpoint in common_ppo
    ]
    same_failure = (
        timing["first_failure"] is not None
        and replay["first_failure"] is not None
        and timing["first_failure"]["endpoint"] == replay["first_failure"]["endpoint"]
        and timing["first_failure"]["reason"] == replay["first_failure"]["reason"]
    )
    no_beyond = timing["validated_steps"] <= replay["validated_steps"]
    closer = float(np.mean(timing_common)) < float(np.mean(ppo_qpos_distances))
    return {
        "rows": rows,
        "comparison_endpoints": list(range(40, 52)),
        "formal_ppo_common_endpoints": common_ppo,
        "mean_timing_minus1_qpos_l2_to_replay_all_40_51": float(
            np.mean(timing_qpos_distances)
        ),
        "mean_timing_minus1_qpos_l2_to_replay_common_ppo": float(
            np.mean(timing_common)
        ),
        "mean_formal_ppo_qpos_l2_to_replay_common": float(
            np.mean(ppo_qpos_distances)
        ),
        "same_failure_endpoint_and_reason_as_replay": same_failure,
        "no_validated_interval_beyond_replay": no_beyond,
        "closer_to_replay_than_formal_ppo_on_common_endpoints": closer,
        "classified_as_suppression_candidate": same_failure and no_beyond and closer,
        "classified_as_refinement_candidate": timing["validated_steps"] > replay["validated_steps"],
    }


def _restore_source(backend, policy, source: dict) -> None:
    backend.restore(source["environment_state"])
    backend.verify_restored_snapshot(source["environment_state"])
    policy.rnn_states = _clone_rnn(source["pre_forward_hidden"])


def _run_counterfactual(
    *, env, backend, policy, source_endpoint: int, source: dict, action_y: float
) -> tuple[dict, dict[str, np.ndarray]]:
    _restore_source(backend, policy, source)
    backend.begin_trial(
        f"diagnostic_source_{source_endpoint}_wrist_y_{action_y:+.2f}",
        source_endpoint,
        60,
    )
    contacts = []
    validated = 0
    first = True
    while int(env.time_indices[0]) < 60:
        current = int(env.time_indices[0])
        packed = policy.obs_to_tensors(backend.observation())
        hidden = _clone_rnn(policy.rnn_states)
        forward = _forward_raw(policy, packed["obs"], hidden)
        bounded = _bounds_and_mode(env, forward)
        policy.rnn_states = _clone_rnn(forward["next_hidden"])
        action = bounded["deterministic_action"].copy()
        if first:
            low = float(bounded["state_low"][ACTION_Y])
            high = float(bounded["state_high"][ACTION_Y])
            if not low - 1e-7 <= action_y <= high + 1e-7:
                raise RuntimeError(
                    f"counterfactual y={action_y} leaves [{low}, {high}] at {current}"
                )
            action[ACTION_Y] = action_y
            first = False
        feasible = backend.step(action[None].astype(np.float32), current)
        contacts.append(_step_contact_record(backend))
        if not feasible:
            break
        validated += 1
    backend.end_trial(validated == 60 - source_endpoint, validated)
    trace = backend.validation_traces[-1]
    tracking = np.asarray(
        [row["aggregate_tracking_reward"] for row in trace["steps"]], np.float64
    )
    total = np.asarray([row["total_reward"] for row in trace["steps"]], np.float64)
    gamma = 0.99
    discounts = gamma ** np.arange(len(tracking), dtype=np.float64)
    termination_endpoint = (
        trace["first_failure"]["endpoint"] if trace["first_failure"] else 60
    )
    summary = {
        "source_endpoint": source_endpoint,
        "one_step_action_y": action_y,
        "formal_actor_mu_y": float(source["actor_mu"][ACTION_Y]),
        "formal_deterministic_action_y": float(
            source["deterministic_action"][ACTION_Y]
        ),
        "state_feasible_y": [
            float(source["state_low"][ACTION_Y]),
            float(source["state_high"][ACTION_Y]),
        ],
        "validated_remaining_intervals": trace["validated_steps"],
        "termination_endpoint_or_60": int(termination_endpoint),
        "first_failure": trace["first_failure"],
        "full_remaining_horizon_passed": trace["validated_steps"] == 60 - source_endpoint,
        "object_tracking_return_undiscounted": float(tracking.sum()),
        "object_tracking_return_discounted_gamma_0_99": float(
            np.sum(discounts * tracking)
        ),
        "total_reward_return_undiscounted": float(total.sum()),
        "outcome_endpoints": [row["endpoint"] for row in trace["steps"]],
        "tracking_scores": [row["objective_score"][0] for row in trace["steps"]],
        "active_contacts": [row["active"] for row in contacts],
        "formal_success_or_commit_allowed": False,
    }
    arrays = {
        "outcome_endpoint": np.asarray(summary["outcome_endpoints"], np.int64),
        "tracking_score": np.asarray(summary["tracking_scores"], np.float64),
        "tracking_reward": tracking,
        "total_reward": total,
        "qpos": np.asarray([row["endpoint_qpos"] for row in trace["steps"]], np.float32),
        "qvel": np.asarray([row["endpoint_qvel"] for row in trace["steps"]], np.float32),
        "normalized_action": np.asarray(
            [row["raw_residual_action"] for row in trace["steps"]], np.float32
        ),
        "commanded_ctrl": np.asarray(
            [row["commanded_ctrl"] for row in trace["steps"]], np.float32
        ),
        "contact_flags": np.asarray([row["flags"] for row in contacts], bool),
    }
    return summary, arrays


def _counterfactual_analysis(branches: list[dict]) -> dict:
    by_source = {}
    for source in SOURCE_ENDPOINTS:
        rows = sorted(
            (row for row in branches if row["source_endpoint"] == source),
            key=lambda row: row["one_step_action_y"],
        )
        best = max(
            rows,
            key=lambda row: (
                row["termination_endpoint_or_60"],
                row["object_tracking_return_discounted_gamma_0_99"],
            ),
        )
        formal_action = rows[-1]
        zero = next(row for row in rows if row["one_step_action_y"] == 0.0)
        better_than_formal = [
            row["one_step_action_y"] for row in rows
            if (
                row["termination_endpoint_or_60"],
                row["object_tracking_return_discounted_gamma_0_99"],
            ) > (
                formal_action["termination_endpoint_or_60"],
                formal_action["object_tracking_return_discounted_gamma_0_99"],
            )
        ]
        by_source[str(source)] = {
            "rows": rows,
            "formal_saturated_action": formal_action,
            "zero_y_action": zero,
            "best_action_y": best["one_step_action_y"],
            "best_termination_endpoint_or_60": best["termination_endpoint_or_60"],
            "actions_better_than_formal_by_termination_then_discounted_return": better_than_formal,
            "formal_mean_is_counterfactually_suboptimal": bool(better_than_formal),
        }
    rescues = {
        str(source): {
            "termination_endpoint_or_60": by_source[str(source)]["zero_y_action"][
                "termination_endpoint_or_60"
            ],
            "full_remaining_horizon_passed": by_source[str(source)]["zero_y_action"][
                "full_remaining_horizon_passed"
            ],
            "passes_beyond_replay_failure_endpoint_51": by_source[str(source)][
                "zero_y_action"
            ]["termination_endpoint_or_60"] > 51,
        }
        for source in (44, 45, 46)
    }
    return {
        "by_source": by_source,
        "full_zero_y_rescues": rescues,
        "any_zero_y_rescue_completes_40_step_window": any(
            row["full_remaining_horizon_passed"] for row in rescues.values()
        ),
        "any_zero_y_rescue_passes_endpoint_51": any(
            row["passes_beyond_replay_failure_endpoint_51"] for row in rescues.values()
        ),
    }


def _statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "p50": float(np.median(values)),
        "max": float(values.max()),
    }


def _historical_credit_inventory(paths: dict[str, Path], checkpoint: dict) -> dict:
    required_keys = {
        "source_endpoint", "outcome_endpoint", "actor_mu", "actor_sigma",
        "sampled_action_preclamp", "sampled_action_clamped",
        "tool_objective_score", "tracking_terminated",
    }
    absent_credit_fields = {
        "reward", "critic_observation", "critic_value", "return", "advantage",
        "raw_observation", "rnn_hidden",
    }
    epoch_rows = []
    observed_keys = None
    for epoch in range(1, 9):
        path = paths[f"training_epoch_{epoch}"]
        with np.load(path) as arrays:
            keys = set(arrays.files)
            if not required_keys <= keys:
                raise RuntimeError(f"training epoch {epoch} is missing visitation fields")
            if observed_keys is None:
                observed_keys = keys
            elif keys != observed_keys:
                raise RuntimeError("training visitation schemas differ by epoch")
            epoch_rows.append({name: arrays[name].copy() for name in arrays.files})
    tail = {}
    for source in TAIL_TRAINING_ENDPOINTS:
        selections = []
        for epoch, row in enumerate(epoch_rows, start=1):
            mask = row["source_endpoint"] == source
            if mask.any():
                selections.append((epoch, row, mask))
        mu = np.concatenate([row["actor_mu"][mask, ACTION_Y] for _, row, mask in selections])
        sigma = np.concatenate([
            row["actor_sigma"][mask, ACTION_Y] for _, row, mask in selections
        ])
        sampled = np.concatenate([
            row["sampled_action_clamped"][mask, ACTION_Y]
            for _, row, mask in selections
        ])
        score = np.concatenate([
            row["tool_objective_score"][mask] for _, row, mask in selections
        ])
        terminated = np.concatenate([
            row["tracking_terminated"][mask] for _, row, mask in selections
        ]).astype(bool)
        tail[str(source)] = {
            "visits": int(mu.size),
            "epochs_with_visit": [epoch for epoch, _, _ in selections],
            "actor_mu_y": _statistics(mu),
            "actor_sigma_y": _statistics(sigma),
            "sampled_action_y": _statistics(sampled),
            "outcome_tracking_score": _statistics(score),
            "tracking_termination_count": int(terminated.sum()),
            "actor_mu_y_above_plus_one_count": int((mu > 1.0).sum()),
        }
    checkpoint_keys = sorted(checkpoint)
    exact_available = absent_credit_fields.issubset(observed_keys)
    return {
        "visitation_schema": "taco_ppo_training_visitation_v5",
        "observed_npz_fields": sorted(observed_keys),
        "missing_credit_fields": sorted(absent_credit_fields - observed_keys),
        "tail_source_summary": tail,
        "checkpoint_top_level_fields": checkpoint_keys,
        "checkpoint_contains_rollout_buffer": False,
        "per_epoch_actor_or_critic_checkpoints_available": False,
        "exact_historical_advantage_value_return_available": exact_available,
        "exact_historical_credit_reconstruction_allowed": False,
        "reason": (
            "v5 visitation omitted rollout reward, critic observation/value, return, "
            "advantage, raw observation and recurrent hidden; the final checkpoint "
            "contains only final networks/optimizers/environment state, not historical "
            "rollout buffers or per-epoch critics"
        ),
        "final_critic_backfill_rejected": True,
    }


def _write_summary(path: Path, report: dict) -> None:
    alignment = report["timing_minus1_replay_alignment"]
    counterfactual = report["one_action_counterfactual"]
    lines = [
        "# Pour corrected PPO training-credit assignment audit v1",
        "",
        "Read-only local audit; no optimizer step, training resume or chunk commit occurred.",
        "",
        "## Timing -1 versus Replay",
        "",
        f"- Replay: {report['regression_gate']['replay_validated_intervals']}/40.",
        f"- timing -1: {report['regression_gate']['timing_minus1_validated_intervals']}/40.",
        f"- suppression candidate: {alignment['classified_as_suppression_candidate']}.",
        f"- refinement candidate: {alignment['classified_as_refinement_candidate']}.",
        "",
        "## Full one-step counterfactuals",
        "",
    ]
    for source, row in counterfactual["by_source"].items():
        formal = row["formal_saturated_action"]
        lines.append(
            f"- source {source}: formal +1 ends at {formal['termination_endpoint_or_60']}; "
            f"best y={row['best_action_y']:+.2f} ends at "
            f"{row['best_termination_endpoint_or_60']}."
        )
    lines.extend(["", "## Historical credit availability", ""])
    inventory = report["historical_training_credit"]
    lines.append(
        "- Exact historical advantage/value/return available: "
        f"{inventory['exact_historical_advantage_value_return_available']}."
    )
    lines.append(f"- Reason: {inventory['reason']}.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/taco_pour_training_credit_assignment_audit_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/taco_pour_training_credit_assignment_audit_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        _build_network_config,
        _build_ppo_config,
        _load_ego_config,
        _load_reference,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import MJWPChunkBackend, _model_state_sha256
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    contract, contract_artifact = _load_contract(args.contract)
    paths = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    formal = json.loads(paths["formal_report"].read_text())
    expected = {row["mode"]: row for row in formal["chunks"][0]["validation_traces"]}
    objective = load_runtime_objective(
        paths["replay_rl_protocol_snapshot"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        paths["replay_rl_protocol_snapshot"], paths["observation_profile"],
        require_run_ready=True,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    spec, distribution = load_truncated_gaussian_profile(paths["distribution_profile"])
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    config = _load_ego_config(str(paths["simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    env = MJWPVectorEnv(
        config, reference, num_envs=1,
        env_config=MJWPVectorEnvConfig(
            reference_start_index=0,
            asymmetric_critic=False,
            max_episode_length=len(reference[0]) - 1,
            tracked_object_indices=(0,),
            object_roles=("tool", "target"),
            objective=objective,
            observation=observation,
            residual=residual,
        ),
        seed=0,
    )
    verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
    backend = MJWPChunkBackend(env)
    boundary, _, _ = _load_gzip_torch(paths["endpoint_20_boundary"], torch)
    checkpoint, _, checkpoint_raw = _load_gzip_torch(paths["checkpoint_artifact"], torch)
    actor_hash = formal["chunks"][0]["training_runs"][0]["actor_state_sha256"]
    if _model_state_sha256(checkpoint["model"]) != actor_hash:
        raise ValueError("checkpoint actor differs from formal corrected run")

    with tempfile.TemporaryDirectory(prefix=".training_credit_", dir=ROOT / "runs") as tmp:
        temp = Path(tmp)
        # Replay precedes policy construction in the formal runner.  The
        # state-feasible PPO agent installs reference snapping on the shared
        # environment, which must not leak backward into the Replay baseline.
        replay, replay_contacts = _run_replay(
            backend=backend, boundary=boundary, expected_trace=expected["replay"]
        )
        ppo_config = replace(
            _build_ppo_config(
                num_envs=1, horizon_length=40, seq_length=4, max_epochs=8,
                learning_rate=1e-4, device="cpu", asymmetric_critic=None,
            ),
            clip_actions=False,
        )
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=temp / "policy_runtime",
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=spec,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        actor_before = _model_state_sha256(policy.model.state_dict())
        normalization_before = _normalization_state(policy)

        formal_ppo, sources = _capture_formal_ppo(
            env=env, backend=backend, policy=policy, boundary=boundary,
            expected_trace=expected["rl"],
        )
        contexts = {
            endpoint: _reference_context(env, backend, endpoint)
            for endpoint in range(19, 60)
        }
        timing, timing_contacts = _run_timing_minus_one(
            env=env, backend=backend, policy=policy, boundary=boundary,
            contexts=contexts,
        )
        prior = json.loads(paths["prior_timing_frame_audit"].read_text())
        prior_timing = prior["closed_loop_counterfactuals"][0]
        if (
            timing["validated_steps"] != prior_timing["validated_intervals"]
            or timing["first_failure"] != prior_timing["first_failure"]
        ):
            raise RuntimeError("timing -1 branch does not reproduce prior audit")
        alignment = _path_alignment(
            replay, replay_contacts, timing, timing_contacts, formal_ppo
        )

        branches = []
        branch_arrays = {}
        formal_branch_regression = {}
        for source_endpoint, source in sources.items():
            low = float(source["state_low"][ACTION_Y])
            high = float(source["state_high"][ACTION_Y])
            if min(ACTION_GRID) < low - 1e-7 or max(ACTION_GRID) > high + 1e-7:
                raise RuntimeError("frozen y grid leaves state-feasible support")
            for action_y in ACTION_GRID:
                summary, arrays = _run_counterfactual(
                    env=env, backend=backend, policy=policy,
                    source_endpoint=source_endpoint, source=source,
                    action_y=action_y,
                )
                branches.append(summary)
                if action_y == float(source["deterministic_action"][ACTION_Y]):
                    expected_suffix = [
                        row for row in expected["rl"]["steps"]
                        if row["control_interval"] >= source_endpoint
                    ]
                    exact = all(
                        np.array_equal(
                            arrays[name],
                            np.asarray(
                                [row[field] for row in expected_suffix], dtype=dtype
                            ),
                        )
                        for name, field, dtype in (
                            ("qpos", "endpoint_qpos", np.float32),
                            ("qvel", "endpoint_qvel", np.float32),
                            ("normalized_action", "raw_residual_action", np.float32),
                            ("commanded_ctrl", "commanded_ctrl", np.float32),
                        )
                    )
                    if not exact:
                        raise RuntimeError(
                            f"formal +y branch from source {source_endpoint} changed"
                        )
                    formal_branch_regression[str(source_endpoint)] = True
                prefix = f"source_{source_endpoint}_y_{action_y:+.2f}"
                for name, value in arrays.items():
                    branch_arrays[f"{prefix}_{name}"] = value
        counterfactual = _counterfactual_analysis(branches)
        historical = _historical_credit_inventory(paths, checkpoint)

        actor_after = _model_state_sha256(policy.model.state_dict())
        normalization_after = _normalization_state(policy)
        normalizer_unchanged = all(
            np.array_equal(normalization_before[key], normalization_after[key])
            if isinstance(normalization_before[key], np.ndarray)
            else normalization_before[key] == normalization_after[key]
            for key in normalization_before
        )
        if actor_before != actor_after or not normalizer_unchanged:
            raise RuntimeError("read-only credit audit changed actor or normalization")

        arrays_path = temp / contract["artifacts"]["arrays"]
        np.savez_compressed(arrays_path, **branch_arrays)
        report = {
            "schema": contract["schema"],
            "status": "completed_read_only_training_credit_audit",
            "paper_faithful": False,
            "scope": {
                "training_performed": False,
                "optimizer_steps": 0,
                "actor_frozen": True,
                "normalization_frozen": True,
                "reward_objective_scale_bound_and_formal_timing_unchanged": True,
                "chunk_acceptance_allowed": False,
                "chunk_commit_allowed": False,
            },
            "contract": contract_artifact,
            "inputs": {
                name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
            "actor": {
                "state_sha256": actor_hash,
                "checkpoint_uncompressed_sha256": hashlib.sha256(
                    checkpoint_raw
                ).hexdigest(),
                "distribution": distribution,
            },
            "regression_gate": {
                "formal_replay_trace_exactly_reproduced": True,
                "formal_ppo_trace_exactly_reproduced": True,
                "replay_validated_intervals": replay["validated_steps"],
                "ppo_validated_intervals": formal_ppo["validated_steps"],
                "timing_minus1_validated_intervals": timing["validated_steps"],
                "formal_plus_y_suffix_exactly_reproduced": formal_branch_regression,
                "actor_state_unchanged": actor_before == actor_after,
                "normalization_state_unchanged": normalizer_unchanged,
            },
            "timing_minus1_replay_alignment": alignment,
            "one_action_counterfactual": counterfactual,
            "historical_training_credit": historical,
            "full_arrays": {
                "path": str((args.output / arrays_path.name).resolve()),
                "sha256": _sha256(arrays_path),
                "keys": sorted(branch_arrays),
            },
            "findings": {
                "timing_minus1_is_suppression_candidate": alignment[
                    "classified_as_suppression_candidate"
                ],
                "timing_minus1_is_refinement_candidate": alignment[
                    "classified_as_refinement_candidate"
                ],
                "zero_y_rescue_passes_replay_failure_endpoint_51": counterfactual[
                    "any_zero_y_rescue_passes_endpoint_51"
                ],
                "zero_y_rescue_completes_full_window": counterfactual[
                    "any_zero_y_rescue_completes_40_step_window"
                ],
                "historical_advantage_sign_recoverable": historical[
                    "exact_historical_advantage_value_return_available"
                ],
                "formal_success_or_promotion_claimed": False,
            },
            "interpretation_limits": [
                "Counterfactual rollout returns use the frozen final actor and are not the historical critic Q estimate.",
                "One-action branches restore exact physical and recurrent states, then replace only wrist-y for one control interval.",
                "Historical visitation rows identify endpoint-level samples, not exact equality with any CPU validation state.",
                "Missing historical value/return/advantage fields are not backfilled with the final critic.",
                "All timing and action branches are diagnostic and cannot accept or commit a chunk.",
            ],
        }
        (temp / "contract_snapshot.yaml").write_bytes(args.contract.read_bytes())
        (temp / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        _write_summary(temp / "summary.md", report)
        policy.writer.close()
        runtime = temp / "policy_runtime"
        if runtime.exists():
            shutil.rmtree(runtime)
        temp.rename(args.output)

    print(json.dumps({
        "status": report["status"],
        "regression_gate": report["regression_gate"],
        "timing_alignment": {
            key: alignment[key] for key in (
                "classified_as_suppression_candidate",
                "classified_as_refinement_candidate",
            )
        },
        "full_zero_y_rescues": counterfactual["full_zero_y_rescues"],
        "historical_advantage_available": historical[
            "exact_historical_advantage_value_return_available"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
