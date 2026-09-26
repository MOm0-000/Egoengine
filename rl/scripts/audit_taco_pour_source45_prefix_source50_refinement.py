#!/usr/bin/env python3
"""Read-only suppression-versus-refinement audit for the frozen Pour actor."""

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
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]

from audit_corrected_endpoint44_48_failure_attribution import (  # noqa: E402
    _actuator_metadata,
    _require_exact_trace,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
    zero_hidden,
)
from audit_taco_pour_single_pass_endpoint47_49 import (  # noqa: E402
    step_map,
    successful_intervals,
)
from audit_taco_pour_source45_state_entry import (  # noqa: E402
    actor_record,
    has_live_right_tool_contact,
    state_record,
)
from audit_taco_pour_source46_state_entry import (  # noqa: E402
    outcome_state,
    state_distance,
)


PREFIX_SOURCE = 45
BRANCH_SOURCE = 50
COMPARE_ENDPOINTS = tuple(range(46, 52))
PREFIX_ZERO_INDICES = (0, 1, 2)
EXPECTED_BRANCHES = {
    "zero_R_forearm_ty_residual_only": (1,),
    "zero_right_wrist_translation": tuple(range(0, 3)),
    "zero_complete_right_wrist": tuple(range(0, 6)),
    "zero_entire_right_hand_residual": tuple(range(0, 18)),
    "zero_full_36d_residual": tuple(range(0, 36)),
}


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_source45_prefix_source50_refinement_gate_v1":
        raise ValueError("unsupported source-45/source-50 contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("source-45/source-50 gate is not authorized")
    required = {
        "backend": "CPU_MuJoCo_Warp",
        "training_allowed": False,
        "optimizer_steps": 0,
        "actor_frozen": True,
        "critic_frozen": True,
        "observation_normalization_frozen": True,
        "reward_changed": False,
        "objective_changed": False,
        "action_scale_changed": False,
        "action_bound_changed": False,
        "action_distribution_changed": False,
        "reference_timing_changed": False,
        "action_frame_changed": False,
        "PPO_hyperparameters_changed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    if any(contract["runtime"].get(key) != value for key, value in required.items()):
        raise ValueError("source-45/source-50 runtime is not fail-closed")
    actual = {
        row["name"]: tuple(row["zero_normalized_action_indices"])
        for row in contract["branches"]
    }
    prefix = contract["validated_prefix"]
    primary = contract["primary_passing_condition"]
    if (
        actual != EXPECTED_BRANCHES
        or prefix["source_endpoint"] != PREFIX_SOURCE
        or tuple(prefix["zero_normalized_action_indices"]) != PREFIX_ZERO_INDICES
        or contract["source50"]["endpoint"] != BRANCH_SOURCE
        or contract["intervention_control_steps"] != 1
        or contract["continuation"] != "frozen_deterministic_PPO"
        or contract["rollout_end_endpoint"] != 60
        or contract["individual_actuator_sweep_allowed"] is not False
        or contract["scale_sweep_allowed"] is not False
        or contract["multi_step_intervention_allowed"] is not False
        or primary["contact_required"] is not False
        or contract["frozen_half_LR_candidate"]["remains_blocked"] is not True
    ):
        raise ValueError("source-45/source-50 gate definition changed")
    paths = {}
    for name, row in contract["inputs"].items():
        artifact = Path(row["path"])
        if sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
        paths[name] = artifact
    return contract, paths, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def trace_arrays(name: str, trace: dict) -> dict[str, np.ndarray]:
    return {
        f"{name}_endpoint": np.asarray(
            [row["endpoint"] for row in trace["steps"]], np.int64
        ),
        f"{name}_score": np.asarray(
            [row["objective_score"][0] for row in trace["steps"]], np.float64
        ),
        f"{name}_qpos": np.asarray(
            [row["endpoint_qpos"] for row in trace["steps"]], np.float32
        ),
        f"{name}_qvel": np.asarray(
            [row["endpoint_qvel"] for row in trace["steps"]], np.float32
        ),
        f"{name}_normalized_action": np.asarray(
            [row["raw_residual_action"] for row in trace["steps"]], np.float32
        ),
    }


def require_prefix_arrays(trace: dict, path: Path) -> None:
    expected = np.load(path)
    suffix = {
        "steps": [
            row for row in trace["steps"]
            if int(row["endpoint"]) > PREFIX_SOURCE
        ]
    }
    produced = trace_arrays("zero_right_wrist_translation", suffix)
    for key, value in produced.items():
        if key not in expected.files or not np.array_equal(value, expected[key]):
            raise RuntimeError(f"validated source-45 prefix changed: {key}")


def run_replay(*, backend, boundary) -> tuple[dict, dict[int, dict]]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    backend.begin_trial("Replay", 20, 60)
    states = {}
    attempted = 0
    feasible = True
    for source in range(20, 60):
        feasible = backend.step(np.zeros((1, 36), np.float32), source)
        attempted += 1
        endpoint = int(backend.env.time_indices[0])
        if endpoint in COMPARE_ENDPOINTS:
            states[endpoint] = state_record(backend.env)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    return backend.validation_traces[-1], states


def run_policy_from_boundary(
    *, backend, policy, boundary, prefix_suppression: bool,
) -> tuple[dict, dict[int, dict], dict[int, dict], dict | None]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = zero_hidden(policy)
    name = "source45_translation_suppression_prefix" if prefix_suppression else "PPO"
    backend.begin_trial(name, 20, 60)
    states = {}
    actors = {}
    capture = None
    attempted = 0
    feasible = True
    for source in range(20, 60):
        if prefix_suppression and source == BRANCH_SOURCE:
            capture = {
                "environment_state": backend.snapshot(),
                "pre_forward_hidden": clone_hidden(policy.rnn_states),
            }
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        action = policy.preprocess_actions(result["deterministic_actions"])
        if prefix_suppression and source == PREFIX_SOURCE:
            low = result["action_lows"][0].detach().cpu().numpy()
            high = result["action_highs"][0].detach().cpu().numpy()
            indices = np.asarray(PREFIX_ZERO_INDICES, np.int64)
            if np.any(low[indices] > 1e-7) or np.any(high[indices] < -1e-7):
                raise RuntimeError("source-45 translation zero leaves feasible support")
            action[0, indices] = 0.0
        if prefix_suppression and source in range(46, 51):
            actors[source] = actor_record(result, action[0])
        feasible = backend.step(action, source)
        attempted += 1
        endpoint = int(backend.env.time_indices[0])
        if prefix_suppression and endpoint in COMPARE_ENDPOINTS:
            states[endpoint] = state_record(backend.env)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    if prefix_suppression and (capture is None or set(actors) != set(range(46, 51))):
        raise RuntimeError("prefix did not capture source-50 state and actor decisions")
    return backend.validation_traces[-1], states, actors, capture


def run_source50_branch(
    *, backend, policy, capture: dict, name: str,
    zero_indices: tuple[int, ...],
) -> tuple[dict, dict[int, dict], dict, dict[str, np.ndarray]]:
    backend.restore(capture["environment_state"])
    backend.verify_restored_snapshot(capture["environment_state"])
    policy.rnn_states = clone_hidden(capture["pre_forward_hidden"])
    states = {BRANCH_SOURCE: state_record(backend.env)}
    backend.begin_trial(name, BRANCH_SOURCE, 60)
    original_action = None
    executed_action = None
    attempted = 0
    feasible = True
    for source in range(BRANCH_SOURCE, 60):
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        action = policy.preprocess_actions(result["deterministic_actions"])
        if source == BRANCH_SOURCE:
            original_action = action[0].copy()
            indices = np.asarray(zero_indices, np.int64)
            low = result["action_lows"][0].detach().cpu().numpy()
            high = result["action_highs"][0].detach().cpu().numpy()
            if np.any(low[indices] > 1e-7) or np.any(high[indices] < -1e-7):
                raise RuntimeError(f"zero leaves state-feasible support for {name}")
            action[0, indices] = 0.0
            executed_action = action[0].copy()
        feasible = backend.step(action, source)
        attempted += 1
        endpoint = int(backend.env.time_indices[0])
        if endpoint >= 51:
            states[endpoint] = state_record(backend.env)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    trace = backend.validation_traces[-1]
    return trace, states, {
        "original_action": original_action,
        "executed_action": executed_action,
    }, trace_arrays(name, trace)


def tracking_state(
    *, states: dict[int, dict], rows: dict[int, dict], endpoint: int,
    reference_qpos: np.ndarray,
) -> dict | None:
    if endpoint not in states or endpoint not in rows:
        return None
    return outcome_state(states[endpoint], rows[endpoint], reference_qpos)


def comparison_record(
    *, endpoint: int, replay_state: dict, prefix_state: dict,
    prefix_actor: dict | None,
) -> dict:
    return {
        "endpoint": endpoint,
        "Replay": replay_state,
        "source45_translation_suppression_prefix": prefix_state,
        "prefix_minus_Replay_separate_physical_distances": state_distance(
            prefix_state, replay_state
        ),
        "actions_at_source_endpoint": {
            "Replay_normalized_residual": (
                [0.0] * 36 if endpoint <= 50 else None
            ),
            "prefix_actor": prefix_actor,
            "semantics": (
                "Action at source endpoint t controls outcome endpoint t+1; "
                "there is no action record at terminal endpoint 51."
            ),
        },
    }


def summarize_branch(
    *, name: str, zero_indices: tuple[int, ...], trace: dict,
    states: dict[int, dict], detail: dict, reference_qpos: np.ndarray,
    actuator_names: tuple[str, ...], prefix_score_51: float,
    replay_score_51: float,
) -> dict:
    rows = step_map(trace)
    endpoint_51 = tracking_state(
        states=states, rows=rows, endpoint=51, reference_qpos=reference_qpos
    )
    survives = endpoint_51 is not None and not endpoint_51["tracking"]["terminated"]
    score = None if endpoint_51 is None else endpoint_51["tracking"]["objective_score"]
    below_prefix = score is not None and score < prefix_score_51
    below_replay = score is not None and score < replay_score_51
    original = np.asarray(detail["original_action"])
    executed = np.asarray(detail["executed_action"])
    failure = trace["first_failure"]
    return {
        "name": name,
        "source_endpoint": BRANCH_SOURCE,
        "zero_normalized_action_indices": list(zero_indices),
        "zeroed_actuators": [actuator_names[index] for index in zero_indices],
        "formal_prefix_normalized_action": original.tolist(),
        "executed_normalized_action": executed.tolist(),
        "untargeted_action_dimensions_bitwise_unchanged": bool(np.array_equal(
            np.delete(original, zero_indices), np.delete(executed, zero_indices)
        )),
        "source_50_state": states[BRANCH_SOURCE],
        "endpoint_51_state": endpoint_51,
        "endpoint_51_has_live_right_hand_tool_contact": (
            None if endpoint_51 is None else has_live_right_tool_contact(endpoint_51)
        ),
        "endpoint_51_survives": survives,
        "endpoint_51_score": score,
        "prefix_baseline_endpoint_51_score": prefix_score_51,
        "same_run_Replay_endpoint_51_score": replay_score_51,
        "endpoint_51_score_below_prefix_baseline": below_prefix,
        "endpoint_51_score_below_same_run_Replay": below_replay,
        "successful_intervals_from_source": successful_intervals(trace),
        "first_failure_endpoint_or_60": (
            60 if failure is None else int(failure["endpoint"])
        ),
        "passes_predeclared_primary_gate": bool(
            survives and below_prefix and below_replay
        ),
        "contact_used_as_acceptance_condition": False,
        "formal_acceptance_or_chunk_commit_allowed": False,
    }


def write_summary(path: Path, report: dict) -> None:
    classification = report["suppression_or_refinement_classification"]
    lines = [
        "# Source-45 suppression / source-50 refinement gate",
        "",
        f"- Replay failure: endpoint {classification['Replay_failure_endpoint']}",
        f"- prefix failure: endpoint {classification['prefix_failure_endpoint']}",
        f"- same failure reason: {classification['same_failure_reason']}",
        "",
    ]
    for row in report["source50_branches"]:
        lines.append(
            f"- {row['name']}: survive51={row['endpoint_51_survives']}, "
            f"score@51={row['endpoint_51_score']}, "
            f"failure={row['first_failure_endpoint_or_60']}, "
            f"pass={row['passes_predeclared_primary_gate']}"
        )
    lines.extend([
        "",
        f"- gate passed: {report['decision']['gate_passed']}",
        f"- next direction: {report['decision']['next_direction']}",
        "",
        "Contact was diagnostic only. No training, task acceptance or chunk commit occurred.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_source45_prefix_source50_refinement_gate_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_source45_prefix_source50_refinement_gate_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)

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
    from video_to_spider.rl.replay_rl import MJWPChunkBackend
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    formal = json.loads(paths["formal_report"].read_text())
    source45 = json.loads(paths["source45_gate_report"].read_text())
    selected = next(
        row for row in source45["branches"]
        if row["name"] == "zero_right_wrist_translation"
    )
    if (
        selected["first_failure_endpoint_or_60"] != 51
        or selected["passes_predeclared_primary_gate"] is not True
        or source45["decision"]["narrowest_passing_branch"]
        != "zero_right_wrist_translation"
    ):
        raise ValueError("source-45 validated prefix evidence changed")

    objective = load_runtime_objective(
        paths["protocol"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol"], paths["observation_profile"], require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    distribution_spec, _ = load_truncated_gaussian_profile(
        paths["distribution_profile"]
    )
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    config = _load_ego_config(str(paths["simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    reference_qpos = reference[0].detach().cpu().numpy().astype(np.float64)
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
    actuator_names, _, _ = _actuator_metadata(
        env.env.model_cpu, tuple(env.env_cfg.residual.hand_control_indices)
    )
    if actuator_names[1] != "R_forearm_ty_position":
        raise RuntimeError("action index 1 is not R_forearm_ty_position")
    backend = MJWPChunkBackend(env)
    boundary = load_gzip_torch(paths["boundary"])

    replay, replay_states = run_replay(backend=backend, boundary=boundary)
    _require_exact_trace(replay, formal["replay_validation"], "single-pass Replay")

    checkpoint = load_gzip_torch(paths["checkpoint"])
    temporary = tempfile.TemporaryDirectory(prefix=".source50_refine_", dir=ROOT / "runs")
    try:
        ppo_config = replace(
            _build_ppo_config(
                num_envs=1, horizon_length=40, seq_length=4, max_epochs=8,
                learning_rate=1e-4, device="cpu", asymmetric_critic=None,
            ),
            clip_actions=False,
        )
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temporary.name) / "policy",
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=distribution_spec,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()

        ppo, _, _, _ = run_policy_from_boundary(
            backend=backend, policy=policy, boundary=boundary,
            prefix_suppression=False,
        )
        _require_exact_trace(ppo, formal["ppo_validation"], "single-pass PPO")

        prefix, prefix_states, prefix_actors, capture = run_policy_from_boundary(
            backend=backend, policy=policy, boundary=boundary,
            prefix_suppression=True,
        )
        require_prefix_arrays(prefix, paths["source45_branch_arrays"])
        if capture is None:
            raise RuntimeError("source-50 capture missing")

        baseline, baseline_states, baseline_detail, _ = run_source50_branch(
            backend=backend, policy=policy, capture=capture,
            name="restored_prefix_source50_suffix", zero_indices=(),
        )
        expected_suffix = [
            row for row in prefix["steps"] if int(row["endpoint"]) > BRANCH_SOURCE
        ]
        if baseline["steps"] != expected_suffix:
            raise RuntimeError("restored prefix source-50 suffix is not exact")
        if baseline_states[51] != prefix_states[51]:
            raise RuntimeError("restored prefix endpoint-51 physical record differs")
        baseline_action = np.asarray(baseline_detail["original_action"])
        if not np.array_equal(
            baseline_action,
            np.asarray(prefix_actors[50]["deterministic_action"]),
        ):
            raise RuntimeError("restored prefix source-50 actor action differs")

        replay_rows = step_map(replay)
        prefix_rows = step_map(prefix)
        replay_outcomes = {
            endpoint: tracking_state(
                states=replay_states, rows=replay_rows, endpoint=endpoint,
                reference_qpos=reference_qpos,
            )
            for endpoint in COMPARE_ENDPOINTS
        }
        prefix_outcomes = {
            endpoint: tracking_state(
                states=prefix_states, rows=prefix_rows, endpoint=endpoint,
                reference_qpos=reference_qpos,
            )
            for endpoint in COMPARE_ENDPOINTS
        }
        if any(value is None for value in (*replay_outcomes.values(), *prefix_outcomes.values())):
            raise RuntimeError("Replay/prefix comparison does not cover endpoints 46-51")
        comparisons = [
            comparison_record(
                endpoint=endpoint,
                replay_state=replay_outcomes[endpoint],
                prefix_state=prefix_outcomes[endpoint],
                prefix_actor=prefix_actors.get(endpoint),
            )
            for endpoint in COMPARE_ENDPOINTS
        ]
        replay_score_51 = replay_outcomes[51]["tracking"]["objective_score"]
        prefix_score_51 = prefix_outcomes[51]["tracking"]["objective_score"]

        branches = []
        arrays = {}
        for name, indices in EXPECTED_BRANCHES.items():
            trace, states, detail, branch_arrays = run_source50_branch(
                backend=backend, policy=policy, capture=capture,
                name=name, zero_indices=indices,
            )
            branches.append(summarize_branch(
                name=name, zero_indices=indices, trace=trace, states=states,
                detail=detail, reference_qpos=reference_qpos,
                actuator_names=actuator_names,
                prefix_score_51=prefix_score_51,
                replay_score_51=replay_score_51,
            ))
            arrays.update(branch_arrays)
        policy.writer.close()
    finally:
        temporary.cleanup()

    passing = [row["name"] for row in branches if row["passes_predeclared_primary_gate"]]
    narrowest_passing = next(
        (name for name in EXPECTED_BRANCHES if name in passing), None
    )
    replay_failure = int(replay["first_failure"]["endpoint"])
    prefix_failure = int(prefix["first_failure"]["endpoint"])
    same_reason = replay["first_failure"]["reason"] == prefix["first_failure"]["reason"]
    decision = {
        "passing_branches": passing,
        "gate_passed": bool(passing),
        "narrowest_passing_branch": narrowest_passing,
        "next_direction": (
            contract["decision_if_any_pass"] if passing
            else contract["decision_if_all_fail"]
        ),
        "new_training_authorized": False,
        "actor_LR_5e_minus_5_unblocked": False,
        "reward_change_authorized": False,
        "chunk_acceptance_or_commit_authorized": False,
    }

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["branch_arrays"]
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "schema": contract["schema"],
        "status": "completed_read_only_gate",
        "paper_faithful": False,
        "training_executed": False,
        "optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "regression_gate": {
            "formal_Replay_trace_bitwise_reproduced": True,
            "formal_PPO_trace_bitwise_reproduced": True,
            "validated_source45_translation_prefix_arrays_bitwise_reproduced": True,
            "restored_prefix_source50_suffix_bitwise_reproduced": True,
            "restored_prefix_endpoint51_physical_record_exact": True,
            "restored_prefix_source50_actor_action_exact": True,
            "formal_Replay_successful_intervals": successful_intervals(replay),
            "formal_PPO_successful_intervals": successful_intervals(ppo),
            "prefix_successful_intervals": successful_intervals(prefix),
        },
        "suppression_or_refinement_classification": {
            "Replay_failure_endpoint": replay_failure,
            "prefix_failure_endpoint": prefix_failure,
            "Replay_failure_reason": replay["first_failure"]["reason"],
            "prefix_failure_reason": prefix["first_failure"]["reason"],
            "same_failure_endpoint": replay_failure == prefix_failure,
            "same_failure_reason": same_reason,
            "Replay_endpoint51_score": replay_score_51,
            "prefix_endpoint51_score": prefix_score_51,
            "prefix_exceeds_Replay_failure_boundary": prefix_failure > replay_failure,
            "endpoint51_error_decomposition": {
                "Replay": replay_outcomes[51]["tracking"][
                    "ellipse_squared_contributions"
                ],
                "source45_translation_suppression_prefix": prefix_outcomes[51][
                    "tracking"
                ]["ellipse_squared_contributions"],
            },
            "endpoint51_prefix_minus_Replay_separate_physical_distances": (
                state_distance(prefix_outcomes[51], replay_outcomes[51])
            ),
            "same_physical_basin_or_failure_mechanism_established": False,
            "interpretation": (
                "The prefix reaches the same endpoint-51 tracking boundary, "
                "but not the same measured physical state or error trade-off: "
                "it has more position contribution and less rotation "
                "contribution than Replay. Matching the failure endpoint alone "
                "does not establish Replay-basin restoration."
            ),
            "classification_before_source50_gate": (
                "suppression_candidate_not_demonstrated_refinement"
            ),
        },
        "prefix_vs_Replay_endpoints_46_to_51": comparisons,
        "source50_branches": branches,
        "two_stage_result": {
            "counterfactual_exceeds_same_run_Replay_failure_boundary": bool(passing),
            "narrowest_passing_intervention": narrowest_passing,
            "narrowest_passing_intervention_is_source50_R_forearm_ty_suppression": (
                narrowest_passing == "zero_R_forearm_ty_residual_only"
            ),
            "learned_PPO_refinement_proven": False,
            "interpretation": (
                "The frozen trajectory can cross endpoint 51 after two "
                "time-local suppressions. This is evidence for a temporal, "
                "state-dependent gating candidate, not proof that the learned "
                "deterministic PPO policy itself performs the refinement."
            ),
        },
        "branch_arrays": {
            "path": str(arrays_path.resolve()),
            "sha256": sha256(arrays_path),
        },
        "measurement_boundaries": {
            "contact_is_secondary_and_does_not_affect_gate": True,
            "positive_mesh_surface_gap_computed": False,
            "mj_geomDistance_used": False,
            "position_and_rotation_contributions_reported_separately": True,
            "mixed_unit_combined_distance_reported": False,
            "same_failure_endpoint_does_not_by_itself_prove_same_physical_basin": True,
        },
        "decision": decision,
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "Replay_score_51": replay_score_51,
        "prefix_score_51": prefix_score_51,
        "passing_branches": passing,
        "branches": [
            {
                "name": row["name"],
                "survive51": row["endpoint_51_survives"],
                "score51": row["endpoint_51_score"],
                "failure": row["first_failure_endpoint_or_60"],
                "pass": row["passes_predeclared_primary_gate"],
            }
            for row in branches
        ],
        "next_direction": decision["next_direction"],
    }, indent=2))


if __name__ == "__main__":
    main()
