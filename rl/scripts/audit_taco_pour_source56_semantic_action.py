#!/usr/bin/env python3
"""Read-only semantic action attribution at the source-55-reversal source 56."""

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

from audit_corrected_endpoint44_48_failure_attribution import _actuator_metadata  # noqa: E402
from audit_taco_pour_binary_translation_last_off_reversal import (  # noqa: E402
    ARRAY_KEYS,
    append_arrays,
    array_result,
    evaluate_source,
    reproduce_baseline,
    restore_choice,
)
from audit_taco_pour_binary_translation_oracle import (  # noqa: E402
    TRANSLATION_INDICES,
    choose_branch,
    compact_outcome,
    one_step,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
)


SOURCE = 56
ANCHORS = {
    "complete_PPO_translation_ON": (),
    "translation_OFF": (0, 1, 2),
}
BRANCHES = {
    "zero_right_wrist_rotation": (3, 4, 5),
    "zero_right_fingers": tuple(range(6, 18)),
    "zero_complete_right_wrist": tuple(range(0, 6)),
    "zero_entire_right_hand": tuple(range(0, 18)),
    "zero_full_36d_residual": tuple(range(0, 36)),
}


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_source56_semantic_action_gate_v1":
        raise ValueError("unsupported source-56 semantic gate contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("source-56 semantic gate is not authorized")
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
        "learned_gate_training_allowed": False,
        "actor_learning_rate_5e_minus_5_training_allowed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    anchors = {
        row["name"]: tuple(row["zero_normalized_action_indices"])
        for row in contract["anchors"]
    }
    branches = {
        row["name"]: tuple(row["zero_normalized_action_indices"])
        for row in contract["branches"]
    }
    source = contract["source_state"]
    intervention = contract["intervention"]
    if (
        any(contract["runtime"].get(key) != value for key, value in required.items())
        or contract.get("paper_faithful") is not False
        or anchors != ANCHORS
        or branches != BRANCHES
        or source["source_endpoint"] != SOURCE
        or source["source55_branch_arrays_must_be_bitwise_exact"] is not True
        or source["actor_forward_calls_at_source56"] != 1
        or source["same_pre_step_snapshot_and_post_forward_hidden_for_all_candidates"] is not True
        or intervention["control_steps"] != 1
        or intervention["source_endpoint"] != SOURCE
        or intervention["individual_actuator_sweep_allowed"] is not False
        or intervention["scale_sweep_allowed"] is not False
        or intervention["additional_action_subspaces_allowed"] is not False
        or intervention["contact_used_for_pass_fail"] is not False
        or contract["passing_condition"]["contact_required"] is not False
        or contract["frozen_candidates"] != {
            "learned_translation_gate": "blocked",
            "actor_learning_rate_5e_minus_5": "blocked",
            "reward_change": "blocked",
        }
    ):
        raise ValueError("source-56 semantic gate definition changed")
    paths: dict[str, Path] = {}
    for name, row in contract["inputs"].items():
        artifact = Path(row["path"])
        if sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
        paths[name] = artifact
    return contract, paths, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def reproduce_source56(*, backend, policy, boundary, reference_qpos,
                       reference_qvel, objective, paths, expected_off_sources):
    captures, baseline = reproduce_baseline(
        backend=backend, policy=policy, boundary=boundary,
        reference_qpos=reference_qpos, reference_qvel=reference_qvel,
        objective=objective, expected_arrays=paths["baseline_oracle_arrays"],
        expected_off_sources=expected_off_sources,
    )
    backend.restore(captures[55]["snapshot"])
    backend.verify_restored_snapshot(captures[55]["snapshot"])
    policy.rnn_states = clone_hidden(captures[55]["pre_hidden"])
    arrays = {key: [] for key in ARRAY_KEYS}

    source55 = evaluate_source(
        backend=backend, policy=policy, source=55,
        reference_qpos=reference_qpos, reference_qvel=reference_qvel,
        objective=objective,
    )
    if source55["greedy_selected"] != "OFF":
        raise RuntimeError("source 55 is no longer a baseline OFF decision")
    source55_on = restore_choice(
        backend=backend, policy=policy, evaluated=source55, selected="ON"
    )
    if source55_on["row"]["terminated"]:
        raise RuntimeError("forced source-55 ON no longer reaches source 56")
    append_arrays(arrays, source55, "ON")

    source56 = evaluate_source(
        backend=backend, policy=policy, source=56,
        reference_qpos=reference_qpos, reference_qvel=reference_qvel,
        objective=objective,
    )
    if source56["greedy_selected"] != "OFF":
        raise RuntimeError("source-55 reversal no longer selects OFF at source 56")
    append_arrays(arrays, source56, "OFF")
    actual = array_result(arrays)
    expected = np.load(paths["last_OFF_reversal_arrays"])
    for key in ARRAY_KEYS:
        expected_key = f"force_source55_ON__{key}"
        if expected_key not in expected.files or not np.array_equal(
            actual[key], expected[expected_key]
        ):
            raise RuntimeError(f"source-55 reversal changed: {key}")
    return baseline, source56


def candidate_step(*, backend, policy, evaluated: dict, name: str,
                   zero_indices: tuple[int, ...], reference_qpos: np.ndarray) -> dict:
    action = evaluated["action_on"].copy()
    if zero_indices:
        indices = np.asarray(zero_indices, np.int64)
        low = np.asarray(evaluated["action_lows"])
        high = np.asarray(evaluated["action_highs"])
        if np.any(low[indices] > 1e-7) or np.any(high[indices] < -1e-7):
            raise RuntimeError(f"zero leaves state-feasible support for {name}")
        action[0, indices] = 0.0
    backend.restore(evaluated["pre_snapshot"])
    backend.verify_restored_snapshot(evaluated["pre_snapshot"])
    policy.rnn_states = clone_hidden(evaluated["post_hidden"])
    raw = one_step(
        backend=backend, action=action, source=SOURCE,
        label=f"source56_{name}",
    )
    return {
        "name": name,
        "zero_normalized_action_indices": list(zero_indices),
        "normalized_action": action[0].copy(),
        "raw": raw,
        "outcome": compact_outcome(raw["row"], raw["state"], reference_qpos),
    }


def continue_oracle(*, backend, policy, candidate: dict, post_source56_hidden,
                    reference_qpos, reference_qvel, objective) -> dict:
    outcome = candidate["outcome"]
    rows = [candidate["raw"]["row"]]
    actions = [candidate["normalized_action"]]
    decisions = []
    if not outcome["terminated"] and outcome["objective_score"] < 1.0:
        backend.restore(candidate["raw"]["snapshot"])
        backend.verify_restored_snapshot(candidate["raw"]["snapshot"])
        policy.rnn_states = clone_hidden(post_source56_hidden)
        for source in range(57, 60):
            evaluated = evaluate_source(
                backend=backend, policy=policy, source=source,
                reference_qpos=reference_qpos, reference_qvel=reference_qvel,
                objective=objective,
            )
            selected, reason = choose_branch(
                evaluated["ON_raw"], evaluated["OFF_raw"]
            )
            chosen = restore_choice(
                backend=backend, policy=policy, evaluated=evaluated,
                selected=selected,
            )
            rows.append(chosen["row"])
            actions.append(
                evaluated["action_off"][0].copy()
                if selected == "OFF" else evaluated["action_on"][0].copy()
            )
            decisions.append({
                "source_endpoint": source,
                "selected": selected,
                "selection_reason": reason,
                "ON_score": evaluated["ON"]["objective_score"],
                "OFF_score": evaluated["OFF"]["objective_score"],
                "selected_outcome": compact_outcome(
                    chosen["row"], chosen["state"], reference_qpos
                ),
            })
            if chosen["row"]["terminated"]:
                break
    failure = next((row for row in rows if row["terminated"]), None)
    successful = 36 + sum(not row["terminated"] for row in rows)
    return {
        "successful_intervals": successful,
        "first_failure_endpoint": None if failure is None else int(failure["endpoint"]),
        "binary_oracle_continuation": decisions,
        "rows": rows,
        "actions": actions,
    }


def summarize_candidate(candidate: dict, continuation: dict) -> dict:
    endpoint57 = candidate["outcome"]
    passed = bool(
        not endpoint57["terminated"] and endpoint57["objective_score"] < 1.0
    )
    return {
        "name": candidate["name"],
        "zero_normalized_action_indices": candidate[
            "zero_normalized_action_indices"
        ],
        "normalized_action": candidate["normalized_action"].tolist(),
        "endpoint57": endpoint57,
        "endpoint57_position_contribution": endpoint57[
            "ellipse_squared_contributions"
        ]["position"],
        "endpoint57_rotation_contribution": endpoint57[
            "ellipse_squared_contributions"
        ]["rotation"],
        "passes_endpoint57": passed,
        "successful_intervals": continuation["successful_intervals"],
        "first_failure_endpoint": continuation["first_failure_endpoint"],
        "binary_oracle_continuation": continuation[
            "binary_oracle_continuation"
        ],
    }


def write_summary(path: Path, report: dict) -> None:
    lines = [
        "# Source-56 semantic action gate",
        "",
        f"- passing semantic branches: {report['decision']['passing_semantic_branches']}",
        f"- next blocker: {report['decision']['next_blocker']}",
        "",
    ]
    for row in report["anchors"] + report["branches"]:
        lines.append(
            f"- {row['name']}: score57={row['endpoint57']['objective_score']:.9g}, "
            f"position={row['endpoint57_position_contribution']:.9g}, "
            f"rotation={row['endpoint57_rotation_contribution']:.9g}, "
            f"pass={row['passes_endpoint57']}, failure={row['first_failure_endpoint']}"
        )
    lines.extend([
        "",
        "Contact is diagnostic only and is not part of pass/fail.",
        "No training, learned-gate fitting, task acceptance or chunk commit occurred.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_source56_semantic_action_gate_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_source56_semantic_action_gate_v1",
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

    prior_report = json.loads(paths["last_OFF_reversal_report"].read_text())
    baseline_report = json.loads(paths["baseline_oracle_report"].read_text())
    if (
        prior_report["decision"]["any_branch_survives_endpoint57"] is not False
        or prior_report["decision"]["source55_reversal_restores_endpoint57_feasibility"] is not False
        or baseline_report["oracle_result"]["successful_intervals"] != 36
    ):
        raise ValueError("source-56 prerequisite evidence changed")
    expected_off_sources = baseline_report["oracle_result"]["OFF_sources"]
    objective = load_runtime_objective(
        paths["protocol"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol"], paths["observation_profile"], require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    distribution_spec, _ = load_truncated_gaussian_profile(paths["distribution_profile"])
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    config = _load_ego_config(str(paths["simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    reference_qpos = reference[0].detach().cpu().numpy().astype(np.float64)
    reference_qvel = reference[1].detach().cpu().numpy().astype(np.float64)
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
    backend = MJWPChunkBackend(env)
    boundary = load_gzip_torch(paths["boundary"])
    checkpoint = load_gzip_torch(paths["checkpoint"])

    temporary = tempfile.TemporaryDirectory(prefix=".source56_semantic_", dir=ROOT / "runs")
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
        _, source56 = reproduce_source56(
            backend=backend, policy=policy, boundary=boundary,
            reference_qpos=reference_qpos, reference_qvel=reference_qvel,
            objective=objective, paths=paths,
            expected_off_sources=expected_off_sources,
        )

        all_candidates = []
        arrays = {}
        for name, zero_indices in {**ANCHORS, **BRANCHES}.items():
            if name == "complete_PPO_translation_ON":
                raw = source56["ON_raw"]
                candidate = {
                    "name": name,
                    "zero_normalized_action_indices": [],
                    "normalized_action": source56["action_on"][0].copy(),
                    "raw": raw,
                    "outcome": source56["ON"],
                }
            elif name == "translation_OFF":
                raw = source56["OFF_raw"]
                candidate = {
                    "name": name,
                    "zero_normalized_action_indices": list(TRANSLATION_INDICES),
                    "normalized_action": source56["action_off"][0].copy(),
                    "raw": raw,
                    "outcome": source56["OFF"],
                }
            else:
                candidate = candidate_step(
                    backend=backend, policy=policy, evaluated=source56,
                    name=name, zero_indices=zero_indices,
                    reference_qpos=reference_qpos,
                )
            continuation = continue_oracle(
                backend=backend, policy=policy, candidate=candidate,
                post_source56_hidden=source56["post_hidden"],
                reference_qpos=reference_qpos, reference_qvel=reference_qvel,
                objective=objective,
            )
            summary = summarize_candidate(candidate, continuation)
            all_candidates.append(summary)
            prefix = name
            arrays[f"{prefix}__endpoint"] = np.asarray(
                [row["endpoint"] for row in continuation["rows"]], np.int64
            )
            arrays[f"{prefix}__score"] = np.asarray(
                [row["objective_score"][0] for row in continuation["rows"]],
                np.float32,
            )
            arrays[f"{prefix}__terminated"] = np.asarray(
                [row["terminated"] for row in continuation["rows"]], np.bool_
            )
            arrays[f"{prefix}__qpos"] = np.asarray(
                [row["endpoint_qpos"] for row in continuation["rows"]],
                np.float32,
            )
            arrays[f"{prefix}__qvel"] = np.asarray(
                [row["endpoint_qvel"] for row in continuation["rows"]],
                np.float32,
            )
            arrays[f"{prefix}__normalized_action"] = np.asarray(
                continuation["actions"], np.float32
            )
        policy.writer.close()
    finally:
        temporary.cleanup()

    anchors = [row for row in all_candidates if row["name"] in ANCHORS]
    branches = [row for row in all_candidates if row["name"] in BRANCHES]
    passing = [row["name"] for row in branches if row["passes_endpoint57"]]
    next_blocker = (
        contract["decision_if_any_pass"] if passing else contract["decision_if_all_fail"]
    )
    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["branch_arrays"]
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "schema": contract["schema"],
        "status": "completed_read_only_gate",
        "paper_faithful": False,
        "classification": contract["classification"],
        "training_executed": False,
        "optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "regression_gate": {
            "baseline_oracle_arrays_bitwise_reproduced": True,
            "source55_ON_reversal_arrays_bitwise_reproduced": True,
            "source56_actor_forward_calls": 1,
            "same_source56_snapshot_and_post_forward_hidden_for_all_candidates": True,
        },
        "action_contract": {
            "actuator_names": list(actuator_names),
            "anchors": {key: list(value) for key, value in ANCHORS.items()},
            "semantic_branches": {key: list(value) for key, value in BRANCHES.items()},
            "contact_used_for_pass_fail": False,
            "individual_actuator_or_scale_sweep": False,
        },
        "source56_state": source56["source_state"],
        "source56_actor": source56["actor"],
        "anchors": anchors,
        "branches": branches,
        "decision": {
            "passing_semantic_branches": passing,
            "any_semantic_branch_passes_endpoint57": bool(passing),
            "next_blocker": next_blocker,
            "learned_gate_training_authorized": False,
            "actor_LR_5e_minus_5_unblocked": False,
            "reward_change_authorized": False,
            "chunk_acceptance_or_commit_authorized": False,
        },
        "branch_arrays": {
            "path": str(arrays_path.resolve()),
            "sha256": sha256(arrays_path),
        },
        "measurement_boundaries": {
            "position_and_rotation_contributions_reported_separately": True,
            "contact_is_mechanistic_diagnostic_only": True,
            "not_a_learned_gate_or_EgoEngine_component": True,
            "no_training_or_chunk_commit": True,
        },
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "passing_semantic_branches": passing,
        "next_blocker": next_blocker,
        "candidates": [
            {
                "name": row["name"],
                "score57": row["endpoint57"]["objective_score"],
                "passes57": row["passes_endpoint57"],
                "successful_intervals": row["successful_intervals"],
                "first_failure_endpoint": row["first_failure_endpoint"],
            }
            for row in all_candidates
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
