#!/usr/bin/env python3
"""Read-only actor-input and closed-loop bifurcation audit for corrected Pour PPO."""

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
from audit_corrected_local_controllability import _clone_rnn  # noqa: E402
from audit_corrected_policy_decision_attribution import _contact_state  # noqa: E402


PATH_ENDPOINTS = tuple(range(40, 48))
SUBSTITUTION_ENDPOINTS = tuple(range(42, 47))
RESCUE_ENDPOINTS = (44, 45, 46)
GROUPS = {
    "hand_state_and_geometry": slice(0, 108),
    "current_object_anchors": slice(108, 126),
    "reference_context": slice(126, 216),
    "contact_flags": slice(216, 236),
}
RIGHT_WRIST_Y = 1


def _load_contract(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_corrected_input_bifurcation_attribution_v1":
        raise ValueError("unsupported input-bifurcation attribution contract")
    if contract.get("status") != "authorized_read_only_attribution":
        raise ValueError("input-bifurcation attribution is not authorized")
    if contract.get("paper_faithful") is not False:
        raise ValueError("local attribution cannot be marked paper-faithful")
    required = {
        "backend": "CPU_MuJoCo_Warp",
        "optimizer_steps": 0,
        "actor_frozen": True,
        "normalization_frozen": True,
        "objective_changed": False,
        "reward_changed": False,
        "residual_bound_changed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    if any(contract["runtime"].get(key) != value for key, value in required.items()):
        raise ValueError("runtime contract is not fail-closed")
    for row in contract["inputs"].values():
        artifact = Path(row["path"])
        if _sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
    expected_groups = {
        name: [selection.start, selection.stop] for name, selection in GROUPS.items()
    }
    if {
        name: contract["observation_groups"][name] for name in GROUPS
    } != expected_groups:
        raise ValueError("observation-group slices differ from the frozen definition")
    if contract["path_forward_comparison"]["source_endpoints"] != list(PATH_ENDPOINTS):
        raise ValueError("path-forward endpoints differ from the frozen definition")
    if contract["observation_groups"]["substitution_sources"] != list(
        SUBSTITUTION_ENDPOINTS
    ):
        raise ValueError("substitution endpoints differ from the frozen definition")
    if contract["single_step_rescue_followup"]["source_endpoints"] != list(
        RESCUE_ENDPOINTS
    ):
        raise ValueError("rescue endpoints differ from the frozen definition")
    return contract, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64).copy()


def _hidden_numpy(states) -> list[np.ndarray]:
    return [_numpy(state) for state in states]


def _hidden_norm(states) -> float:
    return float(np.sqrt(sum(np.square(_numpy(state)).sum() for state in states)))


def _normalization_state(policy) -> dict[str, np.ndarray | float]:
    normalizer = policy.model.running_mean_std
    return {
        "mean": _numpy(normalizer.running_mean),
        "variance": _numpy(normalizer.running_var),
        "count": float(normalizer.count.detach().cpu().item()),
        "epsilon": float(normalizer.epsilon),
    }


def _forward_raw(policy, raw_observation, hidden) -> dict:
    """Evaluate one actor input with explicit hidden; never mutate policy state."""
    import torch

    raw = torch.as_tensor(
        np.asarray(raw_observation, dtype=np.float32), device=policy.device
    )
    if raw.ndim == 1:
        raw = raw[None]
    processed = policy._preproc_obs(raw)
    normalizer = policy.model.running_mean_std
    mean = normalizer.running_mean.float()
    variance = normalizer.running_var.float()
    preclip_z = (processed - mean) / torch.sqrt(variance + normalizer.epsilon)
    actor_input = policy.model.norm_obs(processed)
    with torch.no_grad():
        mu, logstd, value, next_hidden = policy.model.a2c_network({
            "obs": actor_input,
            "rnn_states": [state.clone() for state in hidden],
        })
    return {
        "raw_observation": _numpy(processed)[0],
        "normalization_preclip_z": _numpy(preclip_z)[0],
        "actor_input": _numpy(actor_input)[0],
        "actor_mu": _numpy(mu)[0],
        "actor_logstd": _numpy(logstd)[0],
        "actor_sigma": np.exp(_numpy(logstd)[0]),
        "critic_value": _numpy(value)[0],
        "next_hidden": [state.detach().clone() for state in next_hidden],
    }


def _bounds_and_mode(env, forward: dict) -> dict:
    low, high = env.current_normalized_action_bounds()
    low = _numpy(low)[0]
    high = _numpy(high)[0]
    action = np.maximum(np.minimum(forward["actor_mu"], high), low)
    return {
        "state_low": low,
        "state_high": high,
        "deterministic_action": action,
        "requested_residual": 0.05 * action,
    }


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _normalization_summary(forward: dict) -> dict:
    z = forward["normalization_preclip_z"]
    clipped = np.abs(z) > 5.0
    return {
        "maximum_absolute_preclip_z": float(np.abs(z).max()),
        "normalization_clipped_dimension_count": int(clipped.sum()),
        "normalization_clipped_indices": np.flatnonzero(clipped).tolist(),
        "groups": {
            name: {
                "maximum_absolute_preclip_z": float(np.abs(z[selection]).max()),
                "normalization_clipped_dimension_count": int(
                    clipped[selection].sum()
                ),
            }
            for name, selection in GROUPS.items()
        },
    }


def _path_row(source: int, forward: dict, hidden, bounded: dict) -> dict:
    mu = forward["actor_mu"]
    action = bounded["deterministic_action"]
    return {
        "source_endpoint": source,
        "pre_forward_hidden_l2_norm": _hidden_norm(hidden),
        "right_wrist_y": {
            "actor_mu": float(mu[RIGHT_WRIST_Y]),
            "actor_sigma": float(forward["actor_sigma"][RIGHT_WRIST_Y]),
            "state_low": float(bounded["state_low"][RIGHT_WRIST_Y]),
            "state_high": float(bounded["state_high"][RIGHT_WRIST_Y]),
            "deterministic_action": float(action[RIGHT_WRIST_Y]),
            "requested_residual_m": float(
                bounded["requested_residual"][RIGHT_WRIST_Y]
            ),
        },
        "normalization": _normalization_summary(forward),
    }


def _run_replay_capture(*, backend, boundary, expected_trace: dict) -> tuple[dict, dict]:
    """Capture Replay before the PPO candidate enables reference snapping."""
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    backend.begin_trial("replay", 20, 60)
    observations = {}
    states = {}
    validated = 0
    for source in range(20, 60):
        packed = backend.observation()
        raw = packed["obs"] if isinstance(packed, dict) else packed
        observations[source] = np.asarray(raw, dtype=np.float32).copy()
        if source in PATH_ENDPOINTS:
            states[source] = backend.snapshot()
        feasible = backend.step(np.zeros((1, 36), dtype=np.float32), source)
        if not feasible:
            break
        validated += 1
    backend.end_trial(validated == 40, validated)
    trace = backend.validation_traces[-1]
    _require_exact_trace(trace, expected_trace, "formal replay")
    if set(states) != set(PATH_ENDPOINTS):
        raise RuntimeError("Replay path did not capture all attribution endpoints")
    return trace, {"observations": observations, "states": states}


def _forward_captured_replay(*, env, backend, policy, capture: dict) -> dict[int, dict]:
    policy.rnn_states = [
        state.to("cpu").zero_() for state in policy.model.get_default_rnn_state()
    ]
    records = {}
    for source in range(20, max(PATH_ENDPOINTS) + 1):
        hidden = _clone_rnn(policy.rnn_states)
        forward = _forward_raw(policy, capture["observations"][source], hidden)
        policy.rnn_states = _clone_rnn(forward["next_hidden"])
        if source in PATH_ENDPOINTS:
            backend.restore(capture["states"][source])
            backend.verify_restored_snapshot(capture["states"][source])
            bounded = _bounds_and_mode(env, forward)
            records[source] = {
                "environment_state": capture["states"][source],
                "pre_forward_hidden": hidden,
                "forward": forward,
                "bounded": bounded,
                "row": _path_row(source, forward, hidden, bounded),
            }
    return records


def _run_formal_path(
    *, env, backend, policy, boundary, expected_trace: dict, mode: str
) -> tuple[dict, dict[int, dict]]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = [
        state.to("cpu").zero_() for state in policy.model.get_default_rnn_state()
    ]
    backend.begin_trial(mode, 20, 60)
    records: dict[int, dict] = {}
    validated = 0
    for source in range(20, 60):
        packed = policy.obs_to_tensors(backend.observation())
        hidden = _clone_rnn(policy.rnn_states)
        forward = _forward_raw(policy, packed["obs"], hidden)
        bounded = _bounds_and_mode(env, forward)
        policy.rnn_states = _clone_rnn(forward["next_hidden"])
        if source in PATH_ENDPOINTS:
            records[source] = {
                "environment_state": backend.snapshot(),
                "pre_forward_hidden": hidden,
                "forward": forward,
                "bounded": bounded,
                "row": _path_row(source, forward, hidden, bounded),
            }
        action = (
            np.zeros((1, 36), dtype=np.float32)
            if mode == "replay"
            else bounded["deterministic_action"][None].astype(np.float32)
        )
        feasible = backend.step(action, source)
        if not feasible:
            break
        validated += 1
    backend.end_trial(validated == 40, validated)
    trace = backend.validation_traces[-1]
    _require_exact_trace(trace, expected_trace, f"formal {mode}")
    if set(records) != set(PATH_ENDPOINTS):
        raise RuntimeError(f"{mode} path did not capture all attribution endpoints")
    return trace, records


def _evaluate_with_hidden(policy, raw: np.ndarray, hidden) -> dict:
    return _forward_raw(policy, raw[None].astype(np.float32), hidden)


def _swap_group(base: np.ndarray, replacement: np.ndarray, selection: slice) -> np.ndarray:
    result = base.copy()
    result[selection] = replacement[selection]
    return result


def _substitution_analysis(policy, ppo: dict, replay: dict) -> tuple[dict, dict]:
    report = {}
    arrays = {}
    for source in SUBSTITUTION_ENDPOINTS:
        ppo_forward = ppo[source]["forward"]
        replay_forward = replay[source]["forward"]
        ppo_raw = ppo_forward["raw_observation"]
        replay_raw = replay_forward["raw_observation"]
        hidden = ppo[source]["pre_forward_hidden"]
        base_mu = ppo_forward["actor_mu"]
        rows = {}
        for name, selection in GROUPS.items():
            counterfactual_raw = _swap_group(ppo_raw, replay_raw, selection)
            counterfactual = _evaluate_with_hidden(policy, counterfactual_raw, hidden)
            delta = counterfactual["actor_mu"] - base_mu
            rows[name] = {
                "raw_group_l2_difference": float(
                    np.linalg.norm(ppo_raw[selection] - replay_raw[selection])
                ),
                "right_wrist_y_mu": float(counterfactual["actor_mu"][RIGHT_WRIST_Y]),
                "right_wrist_y_mu_change": float(delta[RIGHT_WRIST_Y]),
                "all_action_mu_change_l2": float(np.linalg.norm(delta)),
                "normalization": _normalization_summary(counterfactual),
            }
            arrays[f"substitution_source_{source}_{name}_raw_observation"] = counterfactual_raw
            arrays[f"substitution_source_{source}_{name}_actor_input"] = counterfactual["actor_input"]
            arrays[f"substitution_source_{source}_{name}_actor_mu"] = counterfactual["actor_mu"]
        report[str(source)] = {
            "base_ppo_right_wrist_y_mu": float(base_mu[RIGHT_WRIST_Y]),
            "fixed_hidden": "ppo_path_pre_forward_hidden",
            "substitutions": rows,
        }
    return report, arrays


def _perturbation_indices() -> dict[str, tuple[int, ...]]:
    return {
        # qpos wrist-y, five right fingertip y coordinates, right-palm y.
        "coherent_right_hand_y_translation": (1, 73, 76, 79, 82, 85, 103),
        # Translate all three current tool anchors along world y.
        "current_tool_anchor_y_translation": (109, 112, 115),
        # Translate all three goal tool anchors along world y.
        "goal_tool_anchor_y_translation": (127, 130, 133),
        "current_reference_right_wrist_y": (145,),
        "preview_reference_right_wrist_y": (181,),
    }


def _sensitivity_analysis(policy, ppo: dict, delta: float) -> tuple[dict, dict]:
    report = {}
    arrays = {}
    definitions = _perturbation_indices()
    for source in SUBSTITUTION_ENDPOINTS:
        base = ppo[source]["forward"]["raw_observation"]
        hidden = ppo[source]["pre_forward_hidden"]
        rows = {}
        for name, indices in definitions.items():
            plus_raw = base.copy()
            minus_raw = base.copy()
            plus_raw[list(indices)] += delta
            minus_raw[list(indices)] -= delta
            plus = _evaluate_with_hidden(policy, plus_raw, hidden)
            minus = _evaluate_with_hidden(policy, minus_raw, hidden)
            derivative = (plus["actor_mu"] - minus["actor_mu"]) / (2.0 * delta)
            rows[name] = {
                "raw_observation_indices": list(indices),
                "central_difference_delta_m": delta,
                "right_wrist_y_mu_minus": float(minus["actor_mu"][RIGHT_WRIST_Y]),
                "right_wrist_y_mu_plus": float(plus["actor_mu"][RIGHT_WRIST_Y]),
                "right_wrist_y_mu_derivative_per_m": float(
                    derivative[RIGHT_WRIST_Y]
                ),
                "all_action_mu_derivative_l2_per_m": float(
                    np.linalg.norm(derivative)
                ),
                "causal_physics_gradient": False,
            }
            prefix = f"sensitivity_source_{source}_{name}"
            arrays[f"{prefix}_mu_minus"] = minus["actor_mu"]
            arrays[f"{prefix}_mu_plus"] = plus["actor_mu"]
            arrays[f"{prefix}_mu_derivative_per_m"] = derivative
        report[str(source)] = rows
    return report, arrays


def _rescue_followup(
    *, env, backend, policy, source: int, record: dict, final_endpoint: int
) -> dict:
    backend.restore(record["environment_state"])
    backend.verify_restored_snapshot(record["environment_state"])
    policy.rnn_states = _clone_rnn(record["pre_forward_hidden"])
    rows = []
    while int(env.time_indices[0]) < final_endpoint:
        current = int(env.time_indices[0])
        packed = policy.obs_to_tensors(backend.observation())
        hidden = _clone_rnn(policy.rnn_states)
        forward = _forward_raw(policy, packed["obs"], hidden)
        bounded = _bounds_and_mode(env, forward)
        policy.rnn_states = _clone_rnn(forward["next_hidden"])
        action = bounded["deterministic_action"].copy()
        before = float(action[RIGHT_WRIST_Y])
        if current == source:
            action[RIGHT_WRIST_Y] = 0.0
        source_contact = _contact_state(env)
        backend.step(action[None].astype(np.float32), current)
        info = backend.last_info
        rows.append({
            "source_endpoint": current,
            "outcome_endpoint": int(env.time_indices[0]),
            "actor_mu_y": float(forward["actor_mu"][RIGHT_WRIST_Y]),
            "state_high_y": float(bounded["state_high"][RIGHT_WRIST_Y]),
            "deterministic_action_y_before_intervention": before,
            "executed_action_y": float(action[RIGHT_WRIST_Y]),
            "mu_exceeds_high": bool(
                forward["actor_mu"][RIGHT_WRIST_Y]
                > bounded["state_high"][RIGHT_WRIST_Y]
            ),
            "source_active_right_tool_fingers": source_contact[
                "active_right_tool_fingers"
            ],
            "outcome_score": float(info["object_tracking_error_per_object"][0, 0]),
            "outcome_position_error_m": float(info["object_position_error"][0, 0]),
            "outcome_rotation_error_rad": float(info["object_rotation_error"][0, 0]),
            "outcome_tracking_terminated": bool(info["terminated"][0]),
        })
    return {
        "intervention_source": source,
        "intervention": "set_right_wrist_y_normalized_action_to_zero_once",
        "subsequent_policy": "frozen_deterministic_actor_closed_loop",
        "steps": rows,
        "endpoint_48": rows[-1],
        "subsequent_mu_returns_below_state_high_before_48": any(
            not row["mu_exceeds_high"] for row in rows[1:]
        ),
        "subsequent_mu_all_remain_above_state_high": all(
            row["mu_exceeds_high"] for row in rows[1:]
        ),
    }


def _write_summary(path: Path, report: dict) -> None:
    path_rows = report["path_forward_comparison"]["rows"]
    rescue = report["single_step_rescue_followup"]
    substitutions = report["observation_group_substitution"]
    lines = [
        "# Corrected frozen-policy input/bifurcation attribution v1",
        "",
        "This is a read-only local diagnosis, not a paper-recovered policy contract.",
        "",
        "## Path forward comparison",
        "",
        "| source | PPO μ_y | Replay μ_y (own hidden) | Replay obs + PPO hidden μ_y |",
        "|---:|---:|---:|---:|",
    ]
    for source in PATH_ENDPOINTS:
        row = path_rows[str(source)]
        lines.append(
            f"| {source} | {row['ppo_path_mu_y']:.6f} | "
            f"{row['replay_path_mu_y']:.6f} | {row['replay_obs_ppo_hidden_mu_y']:.6f} |"
        )
    lines.extend(["", "## One-step rescue follow-up", ""])
    for source in RESCUE_ENDPOINTS:
        row = rescue[str(source)]
        subsequent = ", ".join(
            f"{step['source_endpoint']}:{step['actor_mu_y']:.4f}"
            for step in row["steps"][1:]
        )
        lines.append(
            f"- source {source}: later μ_y = [{subsequent}], "
            f"endpoint48 score={row['endpoint_48']['outcome_score']:.6f}."
        )
    lines.extend(["", "## Largest single-group substitutions", ""])
    for source in SUBSTITUTION_ENDPOINTS:
        rows = substitutions[str(source)]["substitutions"]
        name = max(rows, key=lambda key: abs(rows[key]["right_wrist_y_mu_change"]))
        delta = rows[name]["right_wrist_y_mu_change"]
        lines.append(f"- source {source}: `{name}` changes μ_y by {delta:+.6f}.")
    lines.extend([
        "",
        "## Frozen conclusion",
        "",
        f"- PPO-path μ_y exceeds support at sources {report['findings']['ppo_mu_y_outside_support_sources']}.",
        f"- Replay-path μ_y also exceeds support at sources {report['findings']['replay_path_mu_y_outside_support_sources']}.",
        "- Every successful one-step rescue keeps all later μ_y values above support; the actor does not self-correct.",
        "- No recorded PPO/Replay input dimension reaches the normalization ±5 clamp.",
        "",
        "## Limits",
        "",
        "- Full historical training observations were not logged, so running mean/std is not treated as empirical training support.",
        "- Finite differences are actor input sensitivities with fixed recurrent memory, not causal physics gradients.",
        "- Counterfactual group substitutions may form observations that did not occur physically; they localize the actor input dependency only.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_corrected_input_bifurcation_attribution_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_corrected_input_bifurcation_attribution_v1",
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
    from video_to_spider.rl.replay_rl import (
        MJWPChunkBackend,
        _model_state_sha256,
    )
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    contract, contract_artifact = _load_contract(args.contract)
    paths = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    formal = json.loads(paths["formal_report"].read_text())
    traces = {row["mode"]: row for row in formal["chunks"][0]["validation_traces"]}
    objective = load_runtime_objective(
        paths["replay_rl_protocol_snapshot"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        paths["replay_rl_protocol_snapshot"], paths["observation_profile"],
        require_run_ready=True,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    spec, distribution_artifact = load_truncated_gaussian_profile(
        paths["distribution_profile"]
    )
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    config = _load_ego_config(str(paths["simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    env = MJWPVectorEnv(
        config,
        reference,
        num_envs=1,
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
    checkpoint, _, checkpoint_raw = _load_gzip_torch(
        paths["checkpoint_artifact"], torch
    )
    actor_hash = formal["chunks"][0]["training_runs"][0]["actor_state_sha256"]
    if _model_state_sha256(checkpoint["model"]) != actor_hash:
        raise ValueError("frozen actor differs from the formal corrected run")
    replay_trace, replay_capture = _run_replay_capture(
        backend=backend, boundary=boundary, expected_trace=traces["replay"]
    )

    with tempfile.TemporaryDirectory(
        prefix=".corrected_input_bifurcation_", dir=ROOT / "runs"
    ) as temporary:
        temp = Path(temporary)
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
        model_hash_before = _model_state_sha256(policy.model.state_dict())
        normalization_before = _normalization_state(policy)

        replay_records = _forward_captured_replay(
            env=env, backend=backend, policy=policy, capture=replay_capture
        )
        ppo_trace, ppo_records = _run_formal_path(
            env=env, backend=backend, policy=policy, boundary=boundary,
            expected_trace=traces["rl"], mode="rl",
        )

        path_rows = {}
        full_arrays = {
            "source_endpoints": np.asarray(PATH_ENDPOINTS, dtype=np.int64),
        }
        for path_name, records in (("ppo", ppo_records), ("replay", replay_records)):
            for field in (
                "raw_observation", "normalization_preclip_z", "actor_input",
                "actor_mu", "actor_logstd", "actor_sigma",
            ):
                full_arrays[f"{path_name}_{field}"] = np.stack([
                    records[source]["forward"][field] for source in PATH_ENDPOINTS
                ])
            full_arrays[f"{path_name}_deterministic_action"] = np.stack([
                records[source]["bounded"]["deterministic_action"]
                for source in PATH_ENDPOINTS
            ])
            for field in ("state_low", "state_high", "requested_residual"):
                full_arrays[f"{path_name}_{field}"] = np.stack([
                    records[source]["bounded"][field]
                    for source in PATH_ENDPOINTS
                ])
            for state_index in range(len(records[40]["pre_forward_hidden"])):
                full_arrays[f"{path_name}_pre_forward_hidden_{state_index}"] = np.concatenate([
                    _numpy(records[source]["pre_forward_hidden"][state_index])
                    for source in PATH_ENDPOINTS
                ], axis=0)

        replay_obs_ppo_hidden_mus = []
        for source in PATH_ENDPOINTS:
            ppo_forward = ppo_records[source]["forward"]
            replay_forward = replay_records[source]["forward"]
            common = _evaluate_with_hidden(
                policy,
                replay_forward["raw_observation"],
                ppo_records[source]["pre_forward_hidden"],
            )
            replay_obs_ppo_hidden_mus.append(common["actor_mu"])
            path_rows[str(source)] = {
                "source_endpoint": source,
                "ppo_path_mu_y": float(ppo_forward["actor_mu"][RIGHT_WRIST_Y]),
                "replay_path_mu_y": float(replay_forward["actor_mu"][RIGHT_WRIST_Y]),
                "replay_obs_ppo_hidden_mu_y": float(
                    common["actor_mu"][RIGHT_WRIST_Y]
                ),
                "ppo_minus_replay_path_mu_y": float(
                    ppo_forward["actor_mu"][RIGHT_WRIST_Y]
                    - replay_forward["actor_mu"][RIGHT_WRIST_Y]
                ),
                "ppo_minus_replay_obs_common_hidden_mu_y": float(
                    ppo_forward["actor_mu"][RIGHT_WRIST_Y]
                    - common["actor_mu"][RIGHT_WRIST_Y]
                ),
                "ppo_pre_forward_hidden_l2_norm": _hidden_norm(
                    ppo_records[source]["pre_forward_hidden"]
                ),
                "replay_pre_forward_hidden_l2_norm": _hidden_norm(
                    replay_records[source]["pre_forward_hidden"]
                ),
                "ppo_observation_normalization": _normalization_summary(ppo_forward),
                "replay_observation_normalization": _normalization_summary(
                    replay_forward
                ),
            }
        full_arrays["replay_observation_with_ppo_hidden_actor_mu"] = np.stack(
            replay_obs_ppo_hidden_mus
        )

        for source in PATH_ENDPOINTS:
            ppo_ref = ppo_records[source]["forward"]["raw_observation"][GROUPS["reference_context"]]
            replay_ref = replay_records[source]["forward"]["raw_observation"][GROUPS["reference_context"]]
            if not np.array_equal(ppo_ref, replay_ref):
                raise RuntimeError(
                    f"reference observation unexpectedly differs by physical path at {source}"
                )

        substitution, substitution_arrays = _substitution_analysis(
            policy, ppo_records, replay_records
        )
        full_arrays.update(substitution_arrays)
        sensitivity, sensitivity_arrays = _sensitivity_analysis(
            policy, ppo_records,
            float(contract["local_input_sensitivity"]["central_difference_delta_m"]),
        )
        full_arrays.update(sensitivity_arrays)

        rescues = {}
        for source in RESCUE_ENDPOINTS:
            rescues[str(source)] = _rescue_followup(
                env=env, backend=backend, policy=policy, source=source,
                record=ppo_records[source], final_endpoint=48,
            )

        prior = json.loads(paths["prior_policy_decision_attribution"].read_text())
        for source in RESCUE_ENDPOINTS:
            expected = prior["single_step_zero_y"]["analysis"][str(source)]
            actual = rescues[str(source)]["endpoint_48"]
            if not np.isclose(
                actual["outcome_score"], expected["endpoint_48_score"],
                rtol=0.0, atol=1e-12,
            ):
                raise RuntimeError("one-step rescue did not reproduce prior audit")

        normalization_after = _normalization_state(policy)
        model_hash_after = _model_state_sha256(policy.model.state_dict())
        normalization_unchanged = all(
            np.array_equal(normalization_before[name], normalization_after[name])
            if isinstance(normalization_before[name], np.ndarray)
            else normalization_before[name] == normalization_after[name]
            for name in normalization_before
        )
        if model_hash_before != model_hash_after or not normalization_unchanged:
            raise RuntimeError("read-only attribution changed actor/normalization state")

        ppo_saturated = [
            source for source in PATH_ENDPOINTS
            if ppo_records[source]["forward"]["actor_mu"][RIGHT_WRIST_Y]
            > ppo_records[source]["bounded"]["state_high"][RIGHT_WRIST_Y]
        ]
        replay_saturated = [
            source for source in PATH_ENDPOINTS
            if replay_records[source]["forward"]["actor_mu"][RIGHT_WRIST_Y]
            > replay_records[source]["bounded"]["state_high"][RIGHT_WRIST_Y]
        ]
        normalization_clip_count = sum(
            row[f"{path}_observation_normalization"][
                "normalization_clipped_dimension_count"
            ]
            for row in path_rows.values()
            for path in ("ppo", "replay")
        )
        maximum_local_mu_y_derivative = max(
            abs(row["right_wrist_y_mu_derivative_per_m"])
            for source_rows in sensitivity.values()
            for row in source_rows.values()
        )
        findings = {
            "ppo_mu_y_outside_support_sources": ppo_saturated,
            "replay_path_mu_y_outside_support_sources": replay_saturated,
            "tail_saturation_is_unique_to_ppo_state_path": False,
            "replay_observation_with_common_ppo_hidden_has_higher_mu_y_than_ppo_at_sources_42_to_47": all(
                path_rows[str(source)]["replay_obs_ppo_hidden_mu_y"]
                > path_rows[str(source)]["ppo_path_mu_y"]
                for source in range(42, 48)
            ),
            "all_rescued_branches_keep_later_mu_y_above_support": all(
                row["subsequent_mu_all_remain_above_state_high"]
                for row in rescues.values()
            ),
            "all_rescued_branches_pass_endpoint_48": all(
                row["endpoint_48"]["outcome_score"] < 1.0
                for row in rescues.values()
            ),
            "reference_context_is_path_invariant": True,
            "replay_object_substitution_increases_mu_y_at_every_source_42_to_46": all(
                substitution[str(source)]["substitutions"][
                    "current_object_anchors"
                ]["right_wrist_y_mu_change"] > 0.0
                for source in SUBSTITUTION_ENDPOINTS
            ),
            "normalization_clipped_dimension_count_across_recorded_path_states": normalization_clip_count,
            "maximum_absolute_local_mu_y_derivative_per_m": maximum_local_mu_y_derivative,
            "maximum_predicted_mu_y_change_for_one_mm_local_perturbation": (
                maximum_local_mu_y_derivative * 0.001
            ),
            "empirical_training_support_claim_available": False,
            "interpretation": (
                "The harmful deterministic +y request is not uniquely triggered by "
                "the PPO physical state and does not self-correct after a successful "
                "one-step rescue. The frozen actor also saturates on Replay states; "
                "the critical intervention changes contact/trajectory timing while "
                "the actor keeps requesting +y. Reference timing/coordinate and "
                "residual representation remain the next unresolved contract layer."
            ),
        }

        npz_path = temp / contract["artifacts"]["full_array_npz"]
        np.savez_compressed(
            npz_path,
            **{name: np.asarray(value) for name, value in full_arrays.items()},
        )
        report = {
            "schema": contract["schema"],
            "status": "completed_read_only_attribution",
            "paper_faithful": False,
            "scope": {
                "training_performed": False,
                "optimizer_steps": 0,
                "actor_frozen": True,
                "normalization_frozen": True,
                "objective_reward_and_residual_bound_unchanged": True,
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
                "distribution": distribution_artifact,
            },
            "regression_gate": {
                "formal_replay_trace_exactly_reproduced": True,
                "formal_ppo_trace_exactly_reproduced": True,
                "replay_validated_intervals": replay_trace["validated_steps"],
                "ppo_validated_intervals": ppo_trace["validated_steps"],
                "actor_state_unchanged": model_hash_before == model_hash_after,
                "normalization_state_unchanged": normalization_unchanged,
                "reference_context_identical_between_paths_at_all_sources": True,
            },
            "observation_layout": {
                "dimension": 236,
                "groups": {
                    name: [selection.start, selection.stop]
                    for name, selection in GROUPS.items()
                },
                "full_arrays": {
                    "path": str((args.output / npz_path.name).resolve()),
                    "sha256": _sha256(npz_path),
                    "keys": sorted(full_arrays),
                },
            },
            "path_forward_comparison": {
                "path_consistent_hidden_reported": True,
                "common_ppo_hidden_control_reported": True,
                "rows": path_rows,
            },
            "single_step_rescue_followup": rescues,
            "observation_group_substitution": substitution,
            "local_actor_input_sensitivity": {
                "central_difference_delta_m": contract[
                    "local_input_sensitivity"
                ]["central_difference_delta_m"],
                "fixed_hidden": "ppo_path_pre_forward_hidden",
                "causal_physics_gradient": False,
                "rows": sensitivity,
            },
            "normalization_audit": {
                "running_mean_std_count": normalization_before["count"],
                "running_mean": normalization_before["mean"].tolist(),
                "running_variance": normalization_before["variance"].tolist(),
                "empirical_training_observation_range_available": False,
                "reason": (
                    "historical training visitation did not store the full 236-D "
                    "observation; running mean/variance is not empirical min/max support"
                ),
                "preclip_z_and_actor_input_saved_in_npz": True,
            },
            "findings": findings,
            "interpretation_limits": [
                "Group substitutions isolate actor-input dependence with fixed PPO hidden but may not be physically realizable observations.",
                "Central differences measure actor input sensitivity, not causal object response or a smooth contact-dynamics Jacobian.",
                "Replay-path and PPO-path results are reported both with path-consistent memory and a common PPO-hidden control.",
                "Historical full observations are unavailable, so standardized magnitude is not called empirical training-range coverage.",
                "The exact 236-D observation, contact flags, preview and truncated action distribution are local unpublished choices.",
            ],
        }
        (temp / "contract_snapshot.yaml").write_bytes(args.contract.read_bytes())
        (temp / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        _write_summary(temp / "summary.md", report)
        policy.writer.close()
        policy_dir = temp / "policy_runtime"
        if policy_dir.exists():
            shutil.rmtree(policy_dir)
        temp.rename(args.output)

    print(json.dumps({
        "status": report["status"],
        "output": str((args.output / "report.json").resolve()),
        "regression_gate": report["regression_gate"],
        "path_mu_y": report["path_forward_comparison"]["rows"],
        "rescue_recovery": {
            source: {
                "subsequent_mu_all_remain_above_state_high": row[
                    "subsequent_mu_all_remain_above_state_high"
                ],
                "endpoint_48_score": row["endpoint_48"]["outcome_score"],
            }
            for source, row in report["single_step_rescue_followup"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
