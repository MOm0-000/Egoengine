#!/usr/bin/env python3
"""Read-only endpoint-47/49 gate for the frozen single-pass Pour actor."""

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
    contact_record,
    group_values,
    load_gzip_torch,
    sha256,
    zero_hidden,
)


SOURCES = (46, 47, 48)
INTERVENTION_SOURCES = (47, 48)


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_single_pass_endpoint47_49_gate_v1":
        raise ValueError("unsupported endpoint-47/49 gate contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("endpoint-47/49 gate is not authorized")
    required = {
        "backend": "CPU_MuJoCo_Warp",
        "training_allowed": False,
        "optimizer_steps": 0,
        "actor_frozen": True,
        "critic_frozen": True,
        "observation_normalization_frozen": True,
        "reward_changed": False,
        "objective_changed": False,
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
        raise ValueError("read-only runtime contract is not fail-closed")
    counterfactual = contract["counterfactual"]
    if (
        counterfactual["source_endpoints"] != list(INTERVENTION_SOURCES)
        or counterfactual["intervention_control_steps"] != 1
        or counterfactual["intervention"]
        != "replace_full_36d_PPO_residual_with_zero"
        or counterfactual["continuation"] != "frozen_deterministic_PPO"
        or counterfactual["rollout_end_endpoint"] != 60
    ):
        raise ValueError("counterfactual definition changed")
    candidate = contract["frozen_next_candidate"]
    if (
        candidate["sole_future_change"] != "actor_learning_rate"
        or candidate["baseline"] != 1e-4
        or candidate["candidate"] != 5e-5
        or candidate["actor_mini_epochs"] != 1
        or candidate["fresh_training_authorized"] is not False
    ):
        raise ValueError("learning-rate candidate changed")
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


def capture_formal_ppo(*, backend, policy, boundary) -> tuple[dict, dict, dict]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = zero_hidden(policy)
    backend.begin_trial("PPO", 20, 60)
    captures = {}
    contacts = {}
    attempted = 0
    feasible = True
    for source in range(20, 60):
        if source in INTERVENTION_SOURCES:
            captures[source] = {
                "environment_state": backend.snapshot(),
                "pre_forward_hidden": clone_hidden(policy.rnn_states),
            }
        raw = backend.observation()
        packed = policy.obs_to_tensors(raw)
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        selected = result["deterministic_actions"]
        action = policy.preprocess_actions(selected)
        if source in SOURCES:
            captures.setdefault(source, {})
            captures[source].update({
                "actor_mu": result["mus"][0].detach().cpu().numpy().copy(),
                "actor_sigma": result["sigmas"][0].detach().cpu().numpy().copy(),
                "action_low": result["action_lows"][0].detach().cpu().numpy().copy(),
                "action_high": result["action_highs"][0].detach().cpu().numpy().copy(),
                "normalized_action": selected[0].detach().cpu().numpy().copy(),
            })
        feasible = backend.step(action, source)
        attempted += 1
        contacts[int(backend.env.time_indices[0])] = contact_record(backend)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    return backend.validation_traces[-1], contacts, captures


def run_replay(*, backend, boundary) -> tuple[dict, dict]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    backend.begin_trial("Replay", 20, 60)
    contacts = {}
    attempted = 0
    feasible = True
    for source in range(20, 60):
        feasible = backend.step(np.zeros((1, 36), dtype=np.float32), source)
        attempted += 1
        contacts[int(backend.env.time_indices[0])] = contact_record(backend)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    return backend.validation_traces[-1], contacts


def run_suffix(
    *, backend, policy, source: int, capture: dict, replace_first_with_zero: bool,
) -> tuple[dict, dict]:
    backend.restore(capture["environment_state"])
    backend.verify_restored_snapshot(capture["environment_state"])
    policy.rnn_states = clone_hidden(capture["pre_forward_hidden"])
    label = "zero_residual_once" if replace_first_with_zero else "formal_action"
    backend.begin_trial(f"source_{source}_{label}", source, 60)
    contacts = {}
    attempted = 0
    feasible = True
    for current in range(source, 60):
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        action = policy.preprocess_actions(result["deterministic_actions"])
        if replace_first_with_zero and current == source:
            action = np.zeros((1, 36), dtype=np.float32)
        feasible = backend.step(action, current)
        attempted += 1
        contacts[int(backend.env.time_indices[0])] = contact_record(backend)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    return backend.validation_traces[-1], contacts


def step_map(trace: dict) -> dict[int, dict]:
    return {int(row["endpoint"]): row for row in trace["steps"]}


def successful_intervals(trace: dict) -> int:
    return sum(not bool(row["terminated"]) for row in trace["steps"])


def row_metrics(row: dict, contact: dict, reference_qpos: np.ndarray) -> dict:
    endpoint = int(row["endpoint"])
    qpos = np.asarray(row["endpoint_qpos"], np.float64)
    position = float(row["position_error_m"][0])
    rotation = float(row["rotation_error_rad"][0])
    return {
        "endpoint": endpoint,
        "position_error_m": position,
        "rotation_error_rad": rotation,
        "objective_score": float(row["objective_score"][0]),
        "ellipse_squared_contributions": {
            "position": (position / 0.12) ** 2,
            "rotation": (rotation / 1.5) ** 2,
        },
        "tool_position_error_xyz_actual_minus_reference_m": (
            qpos[36:39] - reference_qpos[endpoint, 36:39]
        ).tolist(),
        "contact": {
            "active": contact["active"],
            "flags": contact["flags"].tolist(),
        },
        "terminated": bool(row["terminated"]),
    }


def aligned_attribution(
    *, replay: dict, ppo: dict, replay_contacts: dict, ppo_contacts: dict,
    captures: dict, reference_qpos: np.ndarray, actuator_names: tuple[str, ...],
) -> dict:
    maps = {"Replay": step_map(replay), "PPO": step_map(ppo)}
    contacts = {"Replay": replay_contacts, "PPO": ppo_contacts}
    result = {}
    for source in SOURCES:
        rows = {}
        for mode in ("Replay", "PPO"):
            current = row_metrics(
                maps[mode][source], contacts[mode][source], reference_qpos
            )
            outcome = row_metrics(
                maps[mode][source + 1], contacts[mode][source + 1], reference_qpos
            )
            if mode == "Replay":
                action = np.zeros(36, np.float64)
                action_row = {
                    "normalized_action": action.tolist(),
                    "named_36d_action": group_values(action, actuator_names),
                    "unit_bound_count": 0,
                    "state_feasible_bound_count": 0,
                }
            else:
                capture = captures[source]
                action = np.asarray(capture["normalized_action"], np.float64)
                low = np.asarray(capture["action_low"], np.float64)
                high = np.asarray(capture["action_high"], np.float64)
                state_bound = (action - low <= 2e-6) | (high - action <= 2e-6)
                unit_bound = np.abs(action) >= 1.0 - 2e-6
                action_row = {
                    "actor_mu": np.asarray(capture["actor_mu"]).tolist(),
                    "actor_sigma": np.asarray(capture["actor_sigma"]).tolist(),
                    "normalized_action": action.tolist(),
                    "named_36d_action": group_values(action, actuator_names),
                    "state_feasible_low": low.tolist(),
                    "state_feasible_high": high.tolist(),
                    "unit_bound_count": int(unit_bound.sum()),
                    "state_feasible_bound_count": int(state_bound.sum()),
                    "unit_bound_actuators": [
                        actuator_names[index] for index in np.flatnonzero(unit_bound)
                    ],
                    "state_feasible_bound_actuators": [
                        actuator_names[index] for index in np.flatnonzero(state_bound)
                    ],
                }
            residual = np.asarray(
                maps[mode][source + 1]["requested_residual"], np.float64
            )
            action_row["requested_residual"] = residual.tolist()
            action_row["named_requested_residual"] = group_values(
                residual, actuator_names
            )
            rows[mode] = {
                "state_at_source": current,
                "action_from_source": action_row,
                "outcome": outcome,
            }
        result[str(source)] = {
            **rows,
            "PPO_minus_Replay_outcome": {
                "position_error_m": (
                    rows["PPO"]["outcome"]["position_error_m"]
                    - rows["Replay"]["outcome"]["position_error_m"]
                ),
                "rotation_error_rad": (
                    rows["PPO"]["outcome"]["rotation_error_rad"]
                    - rows["Replay"]["outcome"]["rotation_error_rad"]
                ),
                "objective_score": (
                    rows["PPO"]["outcome"]["objective_score"]
                    - rows["Replay"]["outcome"]["objective_score"]
                ),
                "tool_position_error_xyz_m": (
                    np.asarray(rows["PPO"]["outcome"][
                        "tool_position_error_xyz_actual_minus_reference_m"
                    ])
                    - np.asarray(rows["Replay"]["outcome"][
                        "tool_position_error_xyz_actual_minus_reference_m"
                    ])
                ).tolist(),
            },
        }
    return result


def branch_summary(
    *, source: int, trace: dict, contacts: dict, formal_ppo: dict,
    reference_qpos: np.ndarray, formal_normalized_action: np.ndarray,
    controlled_qpos_indices: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    rows = step_map(trace)
    formal_rows = step_map(formal_ppo)
    formal_49 = float(formal_rows[49]["objective_score"][0])
    metrics = {
        str(endpoint): row_metrics(rows[endpoint], contacts[endpoint], reference_qpos)
        for endpoint in sorted(rows)
        if endpoint in (48, 49, 50)
    }
    first_failure = trace["first_failure"]
    failure_endpoint = 60 if first_failure is None else int(first_failure["endpoint"])
    endpoint_49_score = (
        float(rows[49]["objective_score"][0]) if 49 in rows else None
    )
    first_row = trace["steps"][0]
    formal_49_qpos = np.asarray(formal_rows[49]["endpoint_qpos"], np.float64)
    branch_49_qpos = (
        np.asarray(rows[49]["endpoint_qpos"], np.float64) if 49 in rows else None
    )
    qpos_delta = (
        branch_49_qpos - formal_49_qpos if branch_49_qpos is not None else None
    )
    summary = {
        "source_endpoint": source,
        "intervention": "one_step_full_36d_zero_residual",
        "replay_action_equivalent": True,
        "then_resumed_frozen_deterministic_PPO": True,
        "intervention_step": {
            "formal_normalized_action_l2": float(np.linalg.norm(
                formal_normalized_action
            )),
            "replacement_normalized_action_l2": 0.0,
            "executed_requested_residual": first_row["requested_residual"],
            "executed_effective_residual_after_ctrlrange": first_row[
                "effective_residual_after_ctrlrange"
            ],
            "executed_residual_lost_to_ctrlrange": first_row[
                "residual_lost_to_ctrlrange"
            ],
        },
        "successful_intervals_from_source": successful_intervals(trace),
        "first_failure_endpoint_or_60": failure_endpoint,
        "endpoint_metrics": metrics,
        "endpoint_49_score_delta_vs_formal_PPO": (
            endpoint_49_score - formal_49 if endpoint_49_score is not None else None
        ),
        "endpoint_49_score_strictly_lower_than_formal_PPO": (
            endpoint_49_score is not None and endpoint_49_score < formal_49
        ),
        "endpoint_49_state_delta_vs_formal_PPO": None if qpos_delta is None else {
            "full_qpos_l2": float(np.linalg.norm(qpos_delta)),
            "controlled_hand_qpos_l2": float(np.linalg.norm(
                qpos_delta[controlled_qpos_indices]
            )),
            "tool_freejoint_qpos_l2": float(np.linalg.norm(qpos_delta[36:43])),
            "target_freejoint_qpos_l2": float(np.linalg.norm(qpos_delta[43:50])),
        },
        "does_not_terminate_earlier_than_formal_PPO": failure_endpoint >= 49,
        "survives_endpoint_49": first_failure is None or failure_endpoint > 49,
        "formal_acceptance_or_commit_allowed": False,
    }
    arrays = {
        f"source_{source}_endpoint": np.asarray(
            [row["endpoint"] for row in trace["steps"]], np.int64
        ),
        f"source_{source}_score": np.asarray(
            [row["objective_score"][0] for row in trace["steps"]], np.float64
        ),
        f"source_{source}_qpos": np.asarray(
            [row["endpoint_qpos"] for row in trace["steps"]], np.float32
        ),
        f"source_{source}_qvel": np.asarray(
            [row["endpoint_qvel"] for row in trace["steps"]], np.float32
        ),
        f"source_{source}_normalized_action": np.asarray(
            [row["raw_residual_action"] for row in trace["steps"]], np.float32
        ),
        f"source_{source}_contact_flags": np.asarray(
            [contacts[row["endpoint"]]["flags"] for row in trace["steps"]], bool
        ),
    }
    return summary, arrays


def write_summary(path: Path, report: dict) -> None:
    gate = report["candidate_evidence_gate"]
    lines = [
        "# Single-pass endpoint 47--49 read-only gate",
        "",
        f"- formal Replay: 30 successful intervals, failure endpoint 51",
        f"- formal PPO: 28 successful intervals, failure endpoint 49",
    ]
    for row in report["counterfactuals"]:
        score = row["endpoint_metrics"].get("49", {}).get("objective_score")
        lines.append(
            f"- source {row['source_endpoint']} one-step zero residual: "
            f"endpoint-49 score {score}, failure endpoint "
            f"{row['first_failure_endpoint_or_60']}"
        )
    lines.extend([
        f"- evidence gate passed: {gate['passed']}",
        f"- frozen LR candidate may await authorization: "
        f"{gate['candidate_may_await_explicit_training_authorization']}",
        "",
        "No training, optimizer update, task acceptance or chunk commit occurred.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_single_pass_endpoint47_49_gate_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_single_pass_endpoint47_49_gate_v1",
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
    objective = load_runtime_objective(
        paths["protocol"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol"], paths["observation_profile"], require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    distribution, _ = load_truncated_gaussian_profile(paths["distribution_profile"])
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
    boundary = load_gzip_torch(paths["boundary"])
    replay, replay_contacts = run_replay(backend=backend, boundary=boundary)
    _require_exact_trace(replay, formal["replay_validation"], "single-pass Replay")

    checkpoint = load_gzip_torch(paths["checkpoint"])
    temp = tempfile.TemporaryDirectory(prefix=".endpoint47_49_gate_", dir=ROOT / "runs")
    try:
        ppo_config = replace(
            _build_ppo_config(
                num_envs=1, horizon_length=40, seq_length=4, max_epochs=8,
                learning_rate=1e-4, device="cpu", asymmetric_critic=None,
            ),
            clip_actions=False,
        )
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temp.name) / "policy",
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=distribution,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        ppo, ppo_contacts, captures = capture_formal_ppo(
            backend=backend, policy=policy, boundary=boundary
        )
        _require_exact_trace(ppo, formal["ppo_validation"], "single-pass PPO")

        control_indices = tuple(env.env_cfg.residual.hand_control_indices)
        actuator_names, controlled_qpos_indices, actuator_units = _actuator_metadata(
            env.env.model_cpu, control_indices
        )

        formal_rows = step_map(ppo)
        suffix_gates = {}
        branches = []
        branch_arrays = {}
        reference_qpos = reference[0].detach().cpu().numpy().astype(np.float64)
        for source in INTERVENTION_SOURCES:
            baseline, _ = run_suffix(
                backend=backend, policy=policy, source=source,
                capture=captures[source], replace_first_with_zero=False,
            )
            expected_steps = [
                row for row in ppo["steps"] if int(row["endpoint"]) > source
            ]
            exact = baseline["steps"] == expected_steps
            if not exact:
                raise RuntimeError(f"source-{source} restored suffix is not exact")
            suffix_gates[str(source)] = exact
            branch, contacts = run_suffix(
                backend=backend, policy=policy, source=source,
                capture=captures[source], replace_first_with_zero=True,
            )
            summary, arrays = branch_summary(
                source=source, trace=branch, contacts=contacts, formal_ppo=ppo,
                reference_qpos=reference_qpos,
                formal_normalized_action=np.asarray(
                    captures[source]["normalized_action"], np.float64
                ),
                controlled_qpos_indices=controlled_qpos_indices,
            )
            branches.append(summary)
            branch_arrays.update(arrays)

        aligned = aligned_attribution(
            replay=replay, ppo=ppo,
            replay_contacts=replay_contacts, ppo_contacts=ppo_contacts,
            captures=captures, reference_qpos=reference_qpos,
            actuator_names=actuator_names,
        )
        policy.writer.close()
    finally:
        temp.cleanup()

    requirements = {
        str(row["source_endpoint"]): {
            "suffix_exact": suffix_gates[str(row["source_endpoint"])],
            "does_not_terminate_earlier": row[
                "does_not_terminate_earlier_than_formal_PPO"
            ],
            "endpoint_49_score_strictly_lower": row[
                "endpoint_49_score_strictly_lower_than_formal_PPO"
            ],
            "survives_endpoint_49": row["survives_endpoint_49"],
        }
        for row in branches
    }
    per_source_pass = all(
        item["suffix_exact"]
        and item["does_not_terminate_earlier"]
        and item["endpoint_49_score_strictly_lower"]
        for item in requirements.values()
    )
    joint_pass = any(item["survives_endpoint_49"] for item in requirements.values())
    gate_passed = per_source_pass and joint_pass

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["branches"]
    np.savez_compressed(arrays_path, **branch_arrays)
    report = {
        "schema": contract["schema"],
        "status": "completed_read_only_gate",
        "paper_faithful": False,
        "training_executed": False,
        "optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "formal_trace_bitwise_reproduced": {"Replay": True, "PPO": True},
        "formal_results": {
            "Replay": {
                "successful_intervals": successful_intervals(replay),
                "first_failure": replay["first_failure"],
            },
            "PPO": {
                "successful_intervals": successful_intervals(ppo),
                "first_failure": ppo["first_failure"],
            },
        },
        "actuator_contract": {
            "names": list(actuator_names),
            "units": list(actuator_units),
        },
        "aligned_source46_48_attribution": aligned,
        "counterfactuals": branches,
        "counterfactual_arrays": {
            "path": str(arrays_path.resolve()),
            "sha256": sha256(arrays_path),
        },
        "candidate_evidence_gate": {
            "predeclared_requirements": requirements,
            "per_source_requirements_passed": per_source_pass,
            "joint_survival_requirement_passed": joint_pass,
            "passed": gate_passed,
            "candidate": {
                "actor_mini_epochs": 1,
                "actor_learning_rate_baseline": 1e-4,
                "actor_learning_rate_candidate": 5e-5,
            },
            "candidate_may_await_explicit_training_authorization": gate_passed,
            "fresh_training_authorized_by_this_gate": False,
            "automatic_sweep_allowed": False,
            "chunk_commit_allowed": False,
        },
        "interpretation_boundary": {
            "claim": (
                "A one-step weaker residual intervention mitigates the current "
                "endpoint-49 divergence only if the predeclared gate passes."
            ),
            "not_claimed": [
                "actor LR 5e-5 is guaranteed to improve training",
                "the local intervention identifies a smooth physical gradient",
                "the candidate is an EgoEngine author-recovered parameter",
            ],
        },
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "gate_passed": gate_passed,
        "counterfactuals": [
            {
                "source": row["source_endpoint"],
                "endpoint49_delta": row[
                    "endpoint_49_score_delta_vs_formal_PPO"
                ],
                "failure": row["first_failure_endpoint_or_60"],
            }
            for row in branches
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
