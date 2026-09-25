#!/usr/bin/env python3
"""Closed-loop CPU audit of the three predeclared Pour objective mappings.

This is an inference-only diagnostic.  It loads one frozen actor and one exact
endpoint-20 simulator boundary, then creates a fresh CPU environment for each
termination rule.  It never trains, commits a chunk, or changes the formal
objective profile.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from egoengine_repro.action.paper_rewards import TrackingObjective
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
    _AgentPolicy,
    _model_state_sha256,
)
from video_to_spider.rl.state_feasible_truncated_gaussian import (
    StateFeasibleTruncatedGaussianPpoAgent,
    load_truncated_gaussian_profile,
)


POSITION_THRESHOLD_M = 0.12
ROTATION_THRESHOLD_RAD = 1.5
START_ENDPOINT = 20
END_ENDPOINT = 60


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_yaml(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value, raw


def trace_signature(trace: dict[str, Any]) -> str:
    chunks: list[bytes] = []
    for row in trace["steps"]:
        chunks.extend(
            (
                np.asarray(row["endpoint_qpos"], np.float32).tobytes(),
                np.asarray(row["endpoint_qvel"], np.float32).tobytes(),
                np.asarray(row["raw_residual_action"], np.float32).tobytes(),
                np.asarray(row["objective_score"], np.float32).tobytes(),
            )
        )
    failure = json.dumps(
        trace["first_failure"], sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256_bytes(b"".join(chunks) + failure)


class IndependentThresholdMJWPVectorEnv(MJWPVectorEnv):
    """Use the predeclared independent position-and-rotation termination rule."""

    def _compute_reward(self, current_objects, goal_objects, contact_flags):
        super()._compute_reward(current_objects, goal_objects, contact_flags)
        score = torch.maximum(
            self._last_tracking_position_error / POSITION_THRESHOLD_M,
            self._last_tracking_rotation_error / ROTATION_THRESHOLD_RAD,
        )
        self._last_tracking_errors = score
        self._last_tracking_rewards = 1.0 - score
        self._last_tracking_error = score.mean(dim=1)
        self._last_object_terminated = (
            (self._last_tracking_position_error > POSITION_THRESHOLD_M)
            | (self._last_tracking_rotation_error > ROTATION_THRESHOLD_RAD)
        )
        self._last_terminated = self._last_object_terminated.any(dim=1)
        # Reward does not affect this frozen-policy rollout.  Keeping its
        # decomposition coherent prevents the diagnostic trace from combining
        # an independent-threshold score with an axis-ellipse tracking reward.
        return (
            self._last_tracking_rewards.mean(dim=1)
            + self._last_contact_score
            + self._last_lift_reward
        )


def validate_contract(
    contract_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract, raw = load_yaml(contract_path)
    if contract.get("schema") != "taco_pour_objective_mapping_closed_loop_v1":
        raise ValueError("unsupported objective-mapping diagnostic contract")
    if contract.get("status") != "frozen_inference_only_no_promotion":
        raise ValueError("objective-mapping diagnostic is not frozen")
    execution = contract.get("execution", {})
    if (
        execution.get("device") != "cpu"
        or execution.get("training_allowed") is not False
        or execution.get("optimizer_updates") != 0
        or execution.get("start_endpoint") != START_ENDPOINT
        or execution.get("end_endpoint") != END_ENDPOINT
        or execution.get("control_intervals") != END_ENDPOINT - START_ENDPOINT
        or execution.get("fresh_environment_per_branch") is not True
        or execution.get("fresh_zero_RNN_state_per_branch") is not True
    ):
        raise ValueError("objective-mapping execution contract changed")
    promotion = contract.get("promotion", {})
    if (
        promotion.get("chunk_commit_allowed") is not False
        or promotion.get("optimized_trajectory_output_allowed") is not False
        or promotion.get("formal_objective_change_allowed") is not False
        or promotion.get("task_success_claim_allowed") is not False
    ):
        raise ValueError("diagnostic contract must forbid every promotion path")
    branches = contract.get("branches")
    expected_branches = {
        "axis_intercept_normalized_ellipse": {
            "lambda_p": 1.0 / POSITION_THRESHOLD_M**2,
            "lambda_R": 1.0 / ROTATION_THRESHOLD_RAD**2,
            "C": 1.0,
        },
        "corner_intercept_ellipse": {
            "lambda_p": 1.0 / POSITION_THRESHOLD_M**2,
            "lambda_R": 1.0 / ROTATION_THRESHOLD_RAD**2,
            "C": math.sqrt(2.0),
        },
        "independent_thresholds": {
            "position_m": POSITION_THRESHOLD_M,
            "rotation_rad": ROTATION_THRESHOLD_RAD,
            "combination": "position_and_rotation",
        },
    }
    if not isinstance(branches, dict) or set(branches) != set(expected_branches):
        raise ValueError("the diagnostic must contain exactly the three predeclared mappings")
    for name, expected in expected_branches.items():
        actual = branches[name]
        for key, value in expected.items():
            if isinstance(value, float):
                if not math.isclose(float(actual.get(key, math.nan)), value):
                    raise ValueError(f"{name}.{key} changed")
            elif actual.get(key) != value:
                raise ValueError(f"{name}.{key} changed")

    bound_inputs = {
        "boundary": args.boundary,
        "actor_manifest": args.actor_manifest,
        "actor_artifact": args.actor,
        "prior_experiment_report": args.prior_experiment_report,
        "config": args.config,
        "initialization_report": args.initialization_report,
        "protocol": args.protocol,
        "objective_profile": args.objective_profile,
        "observation_profile": args.observation_profile,
        "action_profile": args.action_profile,
        "action_distribution_profile": args.action_distribution_profile,
        "implementation": Path(__file__).resolve(),
    }
    expected_hashes = contract.get("input_sha256", {})
    actual_hashes = {name: sha256_path(path) for name, path in bound_inputs.items()}
    if expected_hashes != actual_hashes:
        changed = sorted(
            name
            for name in set(expected_hashes) | set(actual_hashes)
            if expected_hashes.get(name) != actual_hashes.get(name)
        )
        raise ValueError("hash-bound diagnostic inputs changed: " + ", ".join(changed))
    return contract, {
        "path": str(contract_path.resolve()),
        "sha256": sha256_bytes(raw),
        "input_sha256": actual_hashes,
    }


def load_boundary(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = path.read_bytes()
    decoded = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    boundary = torch.load(io.BytesIO(decoded), map_location="cpu", weights_only=False)
    if boundary.get("snapshot_schema") != "egoengine_mjwp_snapshot_v2":
        raise ValueError("objective audit requires a complete v2 simulator snapshot")
    indices = np.asarray(boundary.get("time_indices"), dtype=np.int64)
    if indices.shape != (1,) or int(indices[0]) != START_ENDPOINT:
        raise ValueError("objective audit requires the exact endpoint-20 boundary")
    return boundary, {
        "path": str(path.resolve()),
        "artifact_sha256": sha256_bytes(encoded),
        "uncompressed_pt_sha256": sha256_bytes(decoded),
        "reference_endpoint": int(indices[0]),
    }


def load_actor(
    manifest_path: Path, artifact_path: Path
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    encoded = artifact_path.read_bytes()
    if manifest.get("schema") != "taco_pour_truncated_gaussian_actor_artifact_v1":
        raise ValueError("unsupported frozen actor manifest")
    if manifest.get("artifact", {}).get("sha256") != sha256_bytes(encoded):
        raise ValueError("frozen actor artifact differs from its manifest")
    decoded = gzip.decompress(encoded)
    if manifest["artifact"].get("uncompressed_pt_sha256") != sha256_bytes(decoded):
        raise ValueError("frozen actor payload differs from its manifest")
    payload = torch.load(io.BytesIO(decoded), map_location="cpu", weights_only=False)
    if payload.get("schema") != "taco_pour_truncated_gaussian_actor_v1":
        raise ValueError("unsupported frozen actor payload")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("frozen actor payload has no model state")
    actor_hash = _model_state_sha256(model)
    if actor_hash != manifest.get("actor_state_sha256") or actor_hash != payload.get(
        "actor_state_sha256"
    ):
        raise ValueError("frozen actor state hash changed")
    return model, {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_bytes(manifest_raw),
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": sha256_bytes(encoded),
        "uncompressed_pt_sha256": sha256_bytes(decoded),
        "actor_state_sha256": actor_hash,
    }


def branch_objective(base, branch: str):
    if branch == "axis_intercept_normalized_ellipse":
        return base
    if branch == "corner_intercept_ellipse":
        return replace(
            base,
            objective_id="taco_pour_diagnostic_corner_intercept_ellipse",
            tracking=TrackingObjective(
                lambda_p=1.0 / POSITION_THRESHOLD_M**2,
                lambda_r=1.0 / ROTATION_THRESHOLD_RAD**2,
                boundary=math.sqrt(2.0),
            ),
            tracking_metric_name="corner_intercept_ellipse_score",
            provenance=(
                "Predeclared local objective-mapping diagnostic; not an "
                "author-recovered EgoEngine objective."
            ),
        )
    if branch == "independent_thresholds":
        return replace(
            base,
            objective_id="taco_pour_diagnostic_independent_thresholds",
            tracking_metric_name="max_independent_threshold_ratio",
            provenance=(
                "Predeclared independent 0.12 m AND 1.5 rad diagnostic; not "
                "an author-recovered EgoEngine success rule."
            ),
        )
    raise ValueError(f"unsupported branch: {branch}")


def run_branch(
    *,
    branch: str,
    base_objective,
    observation,
    residual_action,
    action_spec,
    config_path: Path,
    initialization: dict[str, Any],
    boundary: dict[str, Any],
    actor_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    objective = branch_objective(base_objective, branch)
    config = _load_ego_config(str(config_path), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    env_class = (
        IndependentThresholdMJWPVectorEnv
        if branch == "independent_thresholds"
        else MJWPVectorEnv
    )
    env = env_class(
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
            residual=residual_action,
        ),
    )
    verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
    ppo_config = replace(
        _build_ppo_config(
            num_envs=1,
            horizon_length=40,
            seq_length=4,
            max_epochs=8,
            learning_rate=1e-4,
            device="cpu",
            asymmetric_critic=None,
        ),
        clip_actions=False,
    )
    with tempfile.TemporaryDirectory(
        prefix=f"egoengine_objective_mapping_{branch}_"
    ) as temp_dir:
        agent = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temp_dir),
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=action_spec,
        )
        agent.model.load_state_dict(actor_state)
        loaded_hash = _model_state_sha256(agent.model.state_dict())
        expected_hash = _model_state_sha256(actor_state)
        if loaded_hash != expected_hash:
            raise ValueError("actor changed while loading a diagnostic branch")
        agent.set_eval()
        agent.rnn_states = [
            state.to("cpu").zero_() for state in agent.model.get_default_rnn_state()
        ]
        policy = _AgentPolicy(agent, {}, None)
        backend = MJWPChunkBackend(env)
        backend.restore(boundary)
        backend.verify_restored_snapshot(boundary)
        backend.begin_trial(branch, START_ENDPOINT, END_ENDPOINT)
        valid_steps = 0
        for reference_step in range(START_ENDPOINT, END_ENDPOINT):
            action = policy(backend, reference_step)
            if not backend.step(action, reference_step):
                break
            valid_steps += 1
        backend.end_trial(valid_steps == END_ENDPOINT - START_ENDPOINT, valid_steps)
        trace = backend.validation_traces[-1]
        agent.writer.close()
        policy.closed = True
    rows = trace["steps"]
    last = rows[-1]
    return {
        "branch": branch,
        "objective": objective.as_report(),
        "validated_steps": trace["validated_steps"],
        "strict_40_of_40_passed": trace["validated_steps"] == 40,
        "first_failure": trace["first_failure"],
        "last_endpoint": last["endpoint"],
        "last_position_error_m": last["position_error_m"][0],
        "last_rotation_error_rad": last["rotation_error_rad"][0],
        "last_objective_score": last["objective_score"][0],
        "last_independent_threshold_pass": last["independent_threshold_pass"][0],
        "trajectory_signature_sha256": trace_signature(trace),
        "actor_state_sha256": loaded_hash,
        "restored_complete_boundary_verified": backend.verified_restore_count == 1,
        "trace": trace,
    }


def prior_axis_trace(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    report = json.loads(raw)
    traces = {
        trace["mode"]: trace
        for trace in report.get("diagnostic", {}).get("validation_traces", [])
    }
    trace = traces.get("diagnostic_rl")
    if trace is None:
        raise ValueError("prior authorized experiment has no diagnostic_rl trace")
    return trace, {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(raw),
        "validated_steps": trace["validated_steps"],
        "first_failure": trace["first_failure"],
        "trajectory_signature_sha256": trace_signature(trace),
    }


def shared_physics_gate(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Prove that changing termination semantics did not change common states."""
    fields = (
        "endpoint_qpos",
        "endpoint_qvel",
        "raw_residual_action",
        "commanded_ctrl",
    )
    names = tuple(results)
    rows = {
        name: {row["endpoint"]: row for row in results[name]["trace"]["steps"]}
        for name in names
    }
    common = sorted(set.intersection(*(set(value) for value in rows.values())))
    mismatches = []
    for endpoint in common:
        for field in fields:
            values = [
                np.asarray(rows[name][endpoint][field], np.float32).tobytes()
                for name in names
            ]
            if any(value != values[0] for value in values[1:]):
                mismatches.append({"endpoint": endpoint, "field": field})
    alternatives = ("corner_intercept_ellipse", "independent_thresholds")
    alternative_endpoints = sorted(set(rows[alternatives[0]]) & set(rows[alternatives[1]]))
    alternative_mismatches = []
    for endpoint in alternative_endpoints:
        for field in fields:
            left = np.asarray(rows[alternatives[0]][endpoint][field], np.float32)
            right = np.asarray(rows[alternatives[1]][endpoint][field], np.float32)
            if left.tobytes() != right.tobytes():
                alternative_mismatches.append({"endpoint": endpoint, "field": field})
    gate = {
        "compared_fields": list(fields),
        "all_three_common_endpoint_range": [common[0], common[-1]],
        "all_three_common_mismatches": mismatches,
        "alternative_branch_endpoint_range": [
            alternative_endpoints[0], alternative_endpoints[-1]
        ],
        "alternative_branch_mismatches": alternative_mismatches,
        "passed": not mismatches and not alternative_mismatches,
    }
    if not gate["passed"]:
        raise RuntimeError("objective branches changed a common physical trajectory")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--actor-manifest", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--prior-experiment-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--objective-profile", type=Path, required=True)
    parser.add_argument("--observation-profile", type=Path, required=True)
    parser.add_argument("--action-profile", type=Path, required=True)
    parser.add_argument("--action-distribution-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    contract, contract_artifact = validate_contract(args.contract, args)
    boundary, boundary_artifact = load_boundary(args.boundary)
    actor_state, actor_artifact = load_actor(args.actor_manifest, args.actor)
    prior_trace, prior_artifact = prior_axis_trace(args.prior_experiment_report)
    _, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )
    base_objective = load_runtime_objective(
        args.protocol,
        args.objective_profile,
        tracking_variant="tool_only",
        require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True
    )
    residual_action, residual_action_artifact = load_residual_action_profile(
        args.action_profile
    )
    action_spec, action_distribution_artifact = load_truncated_gaussian_profile(
        args.action_distribution_profile
    )
    if not math.isclose(action_spec.residual_scale, residual_action.residual_scale):
        raise ValueError("action distribution and residual profile disagree")

    results = {}
    for branch in contract["branches"]:
        results[branch] = run_branch(
            branch=branch,
            base_objective=base_objective,
            observation=observation,
            residual_action=residual_action,
            action_spec=action_spec,
            config_path=args.config,
            initialization=initialization,
            boundary=boundary,
            actor_state=actor_state,
        )

    axis = results["axis_intercept_normalized_ellipse"]
    physics_gate = shared_physics_gate(results)
    axis_regression = {
        "required_validated_steps": 37,
        "actual_validated_steps": axis["validated_steps"],
        "required_failure_endpoint": 58,
        "actual_first_failure": axis["first_failure"],
        "prior_signature_sha256": prior_artifact["trajectory_signature_sha256"],
        "rerun_signature_sha256": axis["trajectory_signature_sha256"],
    }
    axis_regression["passed"] = (
        axis["validated_steps"] == 37
        and axis["first_failure"] is not None
        and axis["first_failure"]["endpoint"] == 58
        and axis["trajectory_signature_sha256"]
        == prior_artifact["trajectory_signature_sha256"]
    )
    if not axis_regression["passed"]:
        raise RuntimeError("axis-ellipse closed-loop regression did not reproduce exactly")

    alternative_passes = {
        name: results[name]["strict_40_of_40_passed"]
        for name in ("corner_intercept_ellipse", "independent_thresholds")
    }
    if all(alternative_passes.values()):
        interpretation = (
            "Only the current axis-intercept ellipse blocks this frozen actor; "
            "the remaining endpoint-20 window blocker is objective-contract ambiguity."
        )
        classification = "objective_contract_ambiguity"
    elif not any(alternative_passes.values()):
        interpretation = (
            "All three predeclared mappings fail the 40-step window; the remaining "
            "blocker includes control performance rather than only objective mapping."
        )
        classification = "control_failure_under_all_three_mappings"
    else:
        interpretation = (
            "The two alternative mappings disagree; objective-contract ambiguity "
            "remains and no task promotion is justified."
        )
        classification = "mixed_objective_mapping_outcome"

    report = {
        "schema": "taco_pour_objective_mapping_closed_loop_report_v1",
        "status": "completed_inference_only_no_promotion",
        "paper_faithful": False,
        "scope": {
            "training_executed": False,
            "optimizer_updates": 0,
            "device": "cpu",
            "same_frozen_actor_all_branches": True,
            "same_endpoint_20_boundary_all_branches": True,
            "fresh_environment_and_zero_RNN_per_branch": True,
            "physical_closed_loop_replay_not_trace_rescoring": True,
            "control_intervals_requested_per_branch": 40,
        },
        "contract": contract_artifact,
        "boundary": boundary_artifact,
        "actor": actor_artifact,
        "prior_axis_experiment": prior_artifact,
        "formal_inputs": {
            "objective": base_objective.as_report(),
            "observation": observation.as_report(),
            "residual_action": residual_action_artifact,
            "action_distribution": action_distribution_artifact,
            "initialization_report": {
                "path": str(args.initialization_report.resolve()),
                "sha256": sha256_path(args.initialization_report),
                "accepted_for_replay_rl": initialization["accepted_for_replay_rl"],
                "physics_contract_sha256": initialization[
                    "validated_physics_contract"
                ]["physics_contract_sha256"],
            },
        },
        "axis_regression_gate": axis_regression,
        "shared_physics_gate": physics_gate,
        "branches": results,
        "diagnostic_conclusion": {
            "classification": classification,
            "interpretation": interpretation,
            "alternative_40_of_40": alternative_passes,
            "endpoint_60": {
                "position_error_m": results["independent_thresholds"][
                    "last_position_error_m"
                ],
                "position_threshold_excess_m": (
                    results["independent_thresholds"]["last_position_error_m"]
                    - POSITION_THRESHOLD_M
                ),
                "rotation_error_rad": results["independent_thresholds"][
                    "last_rotation_error_rad"
                ],
                "independent_threshold_pass": results[
                    "independent_thresholds"
                ]["last_independent_threshold_pass"],
                "corner_ellipse_score": results["corner_intercept_ellipse"][
                    "last_objective_score"
                ],
                "corner_ellipse_boundary": math.sqrt(2.0),
            },
        },
        "promotion": {
            "chunk_committed": False,
            "optimized_trajectory_written": False,
            "formal_objective_changed": False,
            "full_RL_authorized": False,
            "task_success_claimed": False,
        },
    }
    args.output.mkdir(parents=True)
    output = args.output / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "axis_regression": axis_regression,
                "outcomes": {
                    name: {
                        "validated_steps": result["validated_steps"],
                        "first_failure": result["first_failure"],
                    }
                    for name, result in results.items()
                },
                "classification": classification,
                "report": str(output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
