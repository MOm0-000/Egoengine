#!/usr/bin/env python3
"""Attribute the corrected PPO wrist-y decision without training or changing gates."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_corrected_endpoint44_48_failure_attribution import (  # noqa: E402
    FINGER_NAMES,
    _active_contacts,
    _load_gzip_torch,
    _require_exact_trace,
    _sha256,
)
from audit_corrected_local_controllability import _clone_rnn  # noqa: E402


def _load_contract(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_corrected_policy_decision_attribution_v1":
        raise ValueError("unsupported policy-decision attribution contract")
    if contract.get("status") != "authorized_read_only_attribution":
        raise ValueError("policy-decision attribution is not authorized")
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
    if contract["policy_decision_trace"] != {
        "source_endpoints": [40, 41, 42, 43, 44, 45, 46, 47],
        "current_formal_selection": "deterministic_truncated_mode_clip_mu_to_state_bounds",
        "stochastic_sample_used_by_cpu_validation": False,
        "squash_transform": "none",
        "distinct_deterministic_vs_current_rollout_exists": False,
        "record_full_36d": [
            "actor_mu", "actor_sigma", "state_low", "state_high",
            "deterministic_action",
        ],
        "record_right_wrist_y_mapping": [
            "actor_mu", "actor_sigma", "deterministic_action",
            "requested_residual", "effective_residual_after_ctrlrange", "final_ctrl",
        ],
    }:
        raise ValueError("policy-decision trace differs from its frozen definition")
    probe = contract["control_transmission_probe"]
    if (
        probe["source_endpoints"] != [42, 43, 44, 45, 46, 47]
        or probe["normalized_action_decrements"] != [0.0, -0.25]
        or probe["residual_delta_m_for_negative_probe"] != -0.0125
        or probe["rollout_final_endpoint"] != 48
    ):
        raise ValueError("control-transmission probe differs from its frozen definition")
    return contract, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _gzip_torch(value, torch) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    raw = buffer.getvalue()
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as stream:
        stream.write(raw)
    return compressed.getvalue(), hashlib.sha256(raw).hexdigest()


def _restore_source(env, policy, source: dict, snapshot_equal) -> None:
    env.set_env_state(source["environment_state"])
    if not snapshot_equal(env.get_env_state(), source["environment_state"]):
        raise RuntimeError("restored source differs from the complete snapshot")
    policy.rnn_states = [state.to("cpu").clone() for state in source["rnn_states"]]


def _policy_decision(policy, backend) -> tuple[dict, np.ndarray]:
    observation = policy.obs_to_tensors(backend.observation())
    result = policy.get_deterministic_action_values(observation)
    policy.rnn_states = result["rnn_states"]
    action = result["deterministic_actions"][0].detach().cpu().numpy().copy()
    return result, action


def _policy_record(result: dict) -> dict:
    def array(name):
        return result[name][0].detach().cpu().numpy().astype(np.float64)

    mu = array("mus")
    sigma = array("sigmas")
    low = array("action_lows")
    high = array("action_highs")
    action = array("deterministic_actions")
    if not np.array_equal(action, np.maximum(np.minimum(mu, high), low)):
        raise RuntimeError("deterministic action is not clip(mu, low, high)")
    return {
        "actor_mu": mu.tolist(),
        "actor_sigma": sigma.tolist(),
        "state_feasible_low": low.tolist(),
        "state_feasible_high": high.tolist(),
        "deterministic_truncated_mode": action.tolist(),
        "stochastic_sample": None,
        "pre_squash_action": None,
        "post_squash_action": None,
        "sample_and_squash_not_applicable_reason": (
            "formal CPU acceptance calls get_deterministic_action_values; "
            "the truncated distribution has no squash transform"
        ),
    }


def _contact_state(env) -> dict:
    import warp as wp

    flags, force_vectors = env._live_contact_features()
    flags = flags[0, 0, 0].detach().cpu().numpy().astype(bool)
    force_vectors = force_vectors[0, 0, 0].detach().cpu().numpy().astype(np.float64)
    qpos = env._mjwp.get_qpos(env.ego_cfg, env.env)[0].detach().cpu().numpy()
    model = env.env.model_cpu
    finger_map = np.asarray(env.ego_cfg.force_closure_geom_finger_map, dtype=np.int64)
    object_map = np.asarray(env.ego_cfg.force_closure_geom_object_group_map, dtype=np.int64)
    contact = env.env.data_wp.contact
    geom = wp.to_torch(contact.geom).detach().cpu().numpy().astype(np.int64)
    distance = wp.to_torch(contact.dist).detach().cpu().numpy().astype(np.float64)
    address = wp.to_torch(contact.efc_address).detach().cpu().numpy().astype(np.int64)
    frame = wp.to_torch(contact.frame).detach().cpu().numpy().astype(np.float64)
    forces = wp.to_torch(env.env.data_wp.efc.force)[0].detach().cpu().numpy()
    count = int(wp.to_torch(env.env.data_wp.nacon)[0].item())
    live_contacts = []
    for index in range(count):
        first, second = map(int, geom[index])
        if 0 <= finger_map[first] < 5 and object_map[second] == 0:
            hand, tool = first, second
            finger = int(finger_map[first])
            normal = frame[index, 0]
        elif 0 <= finger_map[second] < 5 and object_map[first] == 0:
            hand, tool = second, first
            finger = int(finger_map[second])
            normal = -frame[index, 0]
        else:
            continue
        valid_addresses = address[index][
            (address[index] >= 0) & (address[index] < len(forces))
        ]
        normal_force = float(np.maximum(forces[valid_addresses], 0.0).sum())
        if distance[index] > 0.0 or normal_force <= 0.0:
            continue
        live_contacts.append({
            "finger": FINGER_NAMES[finger],
            "hand_geom": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, hand
            ),
            "tool_geom": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, tool
            ),
            "contact_distance_m": float(distance[index]),
            "normal_force": normal_force,
            "hand_to_tool_contact_normal_world": normal.tolist(),
        })
    site_positions = wp.to_torch(env.env.data_wp.site_xpos)[
        0, list(env.env_cfg.fingertip_site_ids[:5])
    ].detach().cpu().numpy().astype(np.float64)
    tool_origin = qpos[36:39].astype(np.float64)
    site_to_origin = np.linalg.norm(site_positions - tool_origin, axis=-1)
    return {
        "right_tool_finger_flags": flags.tolist(),
        "active_right_tool_fingers": [
            FINGER_NAMES[index] for index, active in enumerate(flags) if active
        ],
        "aggregated_normal_force_vectors_world": force_vectors.tolist(),
        "aggregated_normal_force_magnitudes": np.linalg.norm(
            force_vectors, axis=-1
        ).tolist(),
        "live_mjwp_contacts": live_contacts,
        "right_fingertip_site_to_tool_origin_distance_m": site_to_origin.tolist(),
        "distance_semantics": (
            "site-to-tool-origin is a continuous pose surrogate, not surface distance; "
            "live contact_distance_m is available only for active MJWP contacts"
        ),
    }


def _step(
    env,
    action: np.ndarray,
    result: dict,
    reference_qpos: np.ndarray,
) -> dict:
    from video_to_spider.rl.residual_semantics import control_target_residuals

    source = int(env.time_indices[0])
    reference_ctrl = env._reference_ctrls(env.time_indices, offset=1)
    if env._state_feasible_action_contract is not None:
        reference_ctrl, _ = env._snap_state_feasible_reference(reference_ctrl)
    reference_ctrl_np = reference_ctrl.detach().cpu().numpy()
    _, reward, _, info = env.step(action[None].astype(np.float32), auto_reset=False)
    endpoint = int(env.time_indices[0])
    qpos = env._mjwp.get_qpos(env.ego_cfg, env.env)[0].detach().cpu().numpy()
    vector = qpos[36:39].astype(np.float64) - reference_qpos[endpoint, 36:39]
    position = float(info["object_position_error"][0, 0])
    if not np.isclose(np.linalg.norm(vector), position, rtol=0.0, atol=2e-7):
        raise RuntimeError("position vector does not reproduce runtime norm")
    semantics = control_target_residuals(
        env.env.model_cpu,
        reference_ctrl_np,
        env._last_ctrl.detach().cpu().numpy(),
    )
    lost = semantics["residual_lost_to_ctrlrange"][0]
    policy = _policy_record(result)
    return {
        "source_endpoint": source,
        "outcome_endpoint": endpoint,
        "bowl_position_error_vector_actual_minus_reference_m": vector.tolist(),
        "bowl_position_error_m": position,
        "bowl_rotation_error_rad": float(info["object_rotation_error"][0, 0]),
        "normalized_ellipse_score": float(info["object_tracking_error_per_object"][0, 0]),
        "tracking_terminated": bool(info["terminated"][0]),
        "reward": float(reward[0]),
        "contact": _contact_state(env),
        "policy": policy,
        "executed_normalized_action": action.tolist(),
        "reference_ctrl": reference_ctrl_np[0].tolist(),
        "final_ctrl": env._last_ctrl[0].detach().cpu().numpy().tolist(),
        "requested_residual": semantics["requested_residual"][0].tolist(),
        "effective_residual_after_ctrlrange": semantics[
            "effective_residual_after_ctrlrange"
        ][0].tolist(),
        "residual_lost_to_ctrlrange": lost.tolist(),
        "maximum_ctrlrange_loss": float(np.abs(lost).max()),
        "endpoint_qpos_sha256": hashlib.sha256(
            np.ascontiguousarray(qpos).tobytes()
        ).hexdigest(),
        "_qpos": qpos,
    }


def _run_branch(
    env,
    backend,
    policy,
    source: dict,
    reference_qpos: np.ndarray,
    snapshot_equal,
    *,
    intervention: str,
    normalized_delta: float,
) -> dict:
    _restore_source(env, policy, source, snapshot_equal)
    source_contact = _contact_state(env)
    rows = []
    first_original_action = None
    while int(env.time_indices[0]) < 48:
        result, action = _policy_decision(policy, backend)
        if not rows:
            first_original_action = action.copy()
            if intervention == "add_y_delta_once":
                action[1] += normalized_delta
                low = float(result["action_lows"][0, 1])
                high = float(result["action_highs"][0, 1])
                if action[1] < low - 1e-7 or action[1] > high + 1e-7:
                    raise ValueError("control-transmission probe left formal support")
            elif intervention == "zero_y_once":
                action[1] = 0.0
            else:
                raise ValueError(f"unknown intervention {intervention}")
        rows.append(_step(env, action, result, reference_qpos))
    final_qpos = rows[-1].pop("_qpos")
    for row in rows[:-1]:
        row.pop("_qpos")
    return {
        "source_endpoint": int(source["endpoint"]),
        "intervention": intervention,
        "normalized_y_delta_first_interval": normalized_delta,
        "source_contact": source_contact,
        "original_first_normalized_action": first_original_action.tolist(),
        "executed_first_normalized_action": rows[0]["executed_normalized_action"],
        "continued_after_tracking_termination_for_diagnosis": True,
        "first_tracking_failure_endpoint": next(
            (row["outcome_endpoint"] for row in rows if row["tracking_terminated"]),
            None,
        ),
        "steps": rows,
        "final": rows[-1],
        "_final_qpos": final_qpos,
    }


def _branch_analysis(probes: list[dict], single_steps: list[dict]) -> dict:
    by_key = {
        (row["source_endpoint"], row["normalized_y_delta_first_interval"]): row
        for row in probes
    }
    transmission = {}
    for source in range(42, 48):
        baseline = by_key[(source, 0.0)]
        negative = by_key[(source, -0.25)]
        immediate_base = baseline["steps"][0]
        immediate_negative = negative["steps"][0]
        delta_residual = (
            negative["steps"][0]["effective_residual_after_ctrlrange"][1]
            - immediate_base["effective_residual_after_ctrlrange"][1]
        )
        if not np.isclose(delta_residual, -0.0125, rtol=0.0, atol=2e-7):
            raise RuntimeError("negative probe did not apply the frozen residual delta")
        immediate_delta_y = (
            immediate_negative["bowl_position_error_vector_actual_minus_reference_m"][1]
            - immediate_base["bowl_position_error_vector_actual_minus_reference_m"][1]
        )
        final_delta_y = (
            negative["final"]["bowl_position_error_vector_actual_minus_reference_m"][1]
            - baseline["final"]["bowl_position_error_vector_actual_minus_reference_m"][1]
        )
        transmission[str(source)] = {
            "effective_residual_delta_m": delta_residual,
            "immediate_bowl_y_delta_m": immediate_delta_y,
            "immediate_one_sided_response_ratio": immediate_delta_y / delta_residual,
            "endpoint_48_bowl_y_delta_m": final_delta_y,
            "endpoint_48_response_ratio": final_delta_y / delta_residual,
            "baseline_source_contact": baseline["source_contact"],
            "negative_probe_source_contact": negative["source_contact"],
            "baseline_immediate_contact": immediate_base["contact"],
            "negative_probe_immediate_contact": immediate_negative["contact"],
            "baseline_final_contact": baseline["final"]["contact"],
            "negative_probe_final_contact": negative["final"]["contact"],
            "smooth_jacobian_claim": False,
        }

    single = {}
    for branch in single_steps:
        source = branch["source_endpoint"]
        baseline = by_key[(source, 0.0)]
        single[str(source)] = {
            "first_action_y_before": branch["original_first_normalized_action"][1],
            "first_action_y_after": branch["executed_first_normalized_action"][1],
            "endpoint_48_score": branch["final"]["normalized_ellipse_score"],
            "change_from_baseline_score": (
                branch["final"]["normalized_ellipse_score"]
                - baseline["final"]["normalized_ellipse_score"]
            ),
            "endpoint_48_bowl_y_error_m": branch["final"][
                "bowl_position_error_vector_actual_minus_reference_m"
            ][1],
            "change_from_baseline_bowl_y_m": (
                branch["final"]["bowl_position_error_vector_actual_minus_reference_m"][1]
                - baseline["final"]["bowl_position_error_vector_actual_minus_reference_m"][1]
            ),
            "right_tool_contact_endpoints": [
                row["outcome_endpoint"] for row in branch["steps"]
                if row["contact"]["active_right_tool_fingers"]
            ],
            "first_tracking_failure_endpoint": branch["first_tracking_failure_endpoint"],
        }
    return {"control_transmission": transmission, "single_step_zero_y": single}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_corrected_policy_decision_attribution_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_corrected_policy_decision_attribution_v1",
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
        _snapshot_value_equal,
    )
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    contract, contract_artifact = _load_contract(args.contract)
    paths = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    formal = json.loads(paths["formal_report"].read_text())
    expected_rl = next(
        row for row in formal["chunks"][0]["validation_traces"] if row["mode"] == "rl"
    )
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

    with tempfile.TemporaryDirectory(
        prefix=".corrected_policy_decision_", dir=ROOT / "runs"
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
        policy.rnn_states = [
            state.to("cpu").zero_() for state in policy.model.get_default_rnn_state()
        ]
        reference_qpos = reference[0].cpu().numpy().astype(np.float64)

        backend.restore(boundary)
        backend.verify_restored_snapshot(boundary)
        sources = {}
        decisions = {}
        backend.begin_trial("rl", 20, 60)
        validated = 0
        for source_endpoint in range(20, 60):
            if source_endpoint in contract["snapshots"]["source_endpoints"]:
                sources[source_endpoint] = {
                    "endpoint": source_endpoint,
                    "environment_state": backend.snapshot(),
                    "rnn_states": _clone_rnn(policy.rnn_states),
                }
            source_contact = _contact_state(env)
            result, action = _policy_decision(policy, backend)
            policy_row = _policy_record(result)
            feasible = backend.step(action[None], source_endpoint)
            outcome_endpoint = int(env.time_indices[0])
            if source_endpoint in contract["policy_decision_trace"]["source_endpoints"]:
                decisions[str(source_endpoint)] = {
                    "source_endpoint": source_endpoint,
                    "outcome_endpoint": outcome_endpoint,
                    "source_contact": source_contact,
                    "policy": policy_row,
                    "right_wrist_y": {
                        "actor_mu": policy_row["actor_mu"][1],
                        "actor_sigma": policy_row["actor_sigma"][1],
                        "state_low": policy_row["state_feasible_low"][1],
                        "state_high": policy_row["state_feasible_high"][1],
                        "deterministic_action": policy_row[
                            "deterministic_truncated_mode"
                        ][1],
                        "mu_minus_high": (
                            policy_row["actor_mu"][1]
                            - policy_row["state_feasible_high"][1]
                        ),
                    },
                }
            if not feasible:
                break
            validated += 1
        backend.end_trial(validated == 40, validated)
        baseline_trace = backend.validation_traces[-1]
        _require_exact_trace(baseline_trace, expected_rl, "frozen deterministic PPO")
        if set(sources) != set(contract["snapshots"]["source_endpoints"]):
            raise RuntimeError("formal trace did not capture every requested source")

        trace_steps = {int(row["endpoint"]): row for row in baseline_trace["steps"]}
        for source_key, decision in decisions.items():
            outcome = decision["outcome_endpoint"]
            trace = trace_steps[outcome]
            decision["right_wrist_y"].update({
                "requested_residual_m": trace["requested_residual"][1],
                "effective_residual_after_ctrlrange_m": trace[
                    "effective_residual_after_ctrlrange"
                ][1],
                "residual_lost_to_ctrlrange_m": trace[
                    "residual_lost_to_ctrlrange"
                ][1],
                "final_ctrl": trace["commanded_ctrl"][1],
            })

        snapshot_dir = temp / "source_snapshots"
        snapshot_dir.mkdir()
        snapshot_artifacts = {}
        for endpoint, source in sources.items():
            artifact, raw_sha = _gzip_torch({
                "schema": "corrected_policy_decision_source_v1",
                "endpoint": endpoint,
                "diagnostic_only": True,
                "resume_authorized": False,
                "actor_state_sha256": actor_hash,
                "environment_state": source["environment_state"],
                "rnn_states": source["rnn_states"],
            }, torch)
            destination = snapshot_dir / f"endpoint_{endpoint}.pt.gz"
            destination.write_bytes(artifact)
            snapshot_artifacts[str(endpoint)] = {
                "path": str((args.output / "source_snapshots" / destination.name).resolve()),
                "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                "uncompressed_pt_sha256": raw_sha,
                "snapshot_field_count": len(source["environment_state"]),
                "diagnostic_only": True,
                "resume_authorized": False,
            }

        probes = []
        formal_endpoint_48_qpos = np.asarray(
            trace_steps[48]["endpoint_qpos"], dtype=np.float32
        )
        for source in contract["control_transmission_probe"]["source_endpoints"]:
            for delta in contract["control_transmission_probe"][
                "normalized_action_decrements"
            ]:
                branch = _run_branch(
                    env, backend, policy, sources[source], reference_qpos,
                    _snapshot_value_equal,
                    intervention="add_y_delta_once", normalized_delta=float(delta),
                )
                if delta == 0.0 and not np.array_equal(
                    branch["_final_qpos"], formal_endpoint_48_qpos
                ):
                    raise RuntimeError("zero probe does not reproduce formal endpoint 48")
                branch.pop("_final_qpos")
                probes.append(branch)

        single_steps = []
        for source in contract["single_step_intervention"]["source_endpoints"]:
            branch = _run_branch(
                env, backend, policy, sources[source], reference_qpos,
                _snapshot_value_equal,
                intervention="zero_y_once", normalized_delta=0.0,
            )
            branch.pop("_final_qpos")
            single_steps.append(branch)

        analysis = _branch_analysis(probes, single_steps)
        all_steps = [
            step for branch in (*probes, *single_steps) for step in branch["steps"]
        ]
        report = {
            "schema": "taco_pour_corrected_policy_decision_attribution_v1",
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
                "formal_cpu_trace_exactly_reproduced": True,
                "validated_intervals": baseline_trace["validated_steps"],
                "first_failure": baseline_trace["first_failure"],
                "restored_complete_state_count": len(probes) + len(single_steps),
            },
            "deterministic_vs_current": {
                "distinct_rollout_exists": False,
                "current_formal_cpu_selector": "clip_mu_to_state_bounds",
                "current_formal_cpu_is_deterministic": True,
                "stochastic_sample_used": False,
                "squash_transform_used": False,
                "reason": (
                    "formal CPU validation already calls the deterministic truncated "
                    "mode; running a second deterministic-vs-current branch would "
                    "duplicate the same action sequence"
                ),
            },
            "policy_decisions": decisions,
            "source_snapshots": snapshot_artifacts,
            "single_step_zero_y": {
                "branch_count": len(single_steps),
                "branches": single_steps,
                "analysis": analysis["single_step_zero_y"],
            },
            "control_transmission_probe": {
                "branch_count": len(probes),
                "branches": probes,
                "analysis": analysis["control_transmission"],
            },
            "global_checks": {
                "zero_probes_reproduce_formal_endpoint_48": True,
                "all_interventions_remain_within_formal_action_support": True,
                "maximum_ctrlrange_loss": float(
                    max(step["maximum_ctrlrange_loss"] for step in all_steps)
                ),
                "complete_physics_and_rnn_state_restored_for_every_branch": True,
            },
            "interpretation_limits": [
                "CPU acceptance already uses deterministic clip(mu,low,high); no stochastic CPU comparison exists.",
                "The one-sided probe is a finite counterfactual response, not a smooth Jacobian claim.",
                "Fingertip-site to tool-origin distance is not surface distance; positive mesh-SDF separation is unavailable here.",
                "Contact loss, wrist pose and finger configuration remain coupled; this report cannot name one unique root cause.",
            ],
        }
        (temp / "contract_snapshot.yaml").write_bytes(args.contract.read_bytes())
        (temp / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        policy.writer.close()
        policy_dir = temp / "policy_runtime"
        if policy_dir.exists():
            shutil.rmtree(policy_dir)
        temp.rename(args.output)

    print(json.dumps({
        "status": report["status"],
        "output": str((args.output / "report.json").resolve()),
        "regression_gate": report["regression_gate"],
        "deterministic_vs_current": report["deterministic_vs_current"],
        "single_step_zero_y": report["single_step_zero_y"]["analysis"],
        "right_wrist_y_policy": {
            source: row["right_wrist_y"]
            for source, row in report["policy_decisions"].items()
        },
        "control_transmission_ratios": {
            source: {
                "immediate": row["immediate_one_sided_response_ratio"],
                "endpoint_48": row["endpoint_48_response_ratio"],
            }
            for source, row in report["control_transmission_probe"][
                "analysis"
            ].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
