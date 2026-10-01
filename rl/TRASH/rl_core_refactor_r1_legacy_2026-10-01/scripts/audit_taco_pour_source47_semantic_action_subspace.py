#!/usr/bin/env python3
"""Read-only semantic action-subspace gate at the formal Pour source 47."""

from __future__ import annotations

import argparse
from collections import Counter
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
    _require_exact_trace,
)
from audit_corrected_policy_decision_attribution import _contact_state  # noqa: E402
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
)
from audit_taco_pour_single_pass_endpoint47_49 import (  # noqa: E402
    capture_formal_ppo,
    run_replay,
    step_map,
    successful_intervals,
)


SOURCE = 47
EXPECTED_BRANCHES = {
    "zero_right_wrist_translation": tuple(range(0, 3)),
    "zero_right_wrist_rotation": tuple(range(3, 6)),
    "zero_right_fingers": tuple(range(6, 18)),
    "zero_right_wrist_translation_and_rotation": tuple(range(0, 6)),
    "zero_entire_right_hand_residual": tuple(range(0, 18)),
}


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_source47_semantic_action_subspace_gate_v1":
        raise ValueError("unsupported source-47 semantic gate contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("source-47 semantic gate is not authorized")
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
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    if any(contract["runtime"].get(key) != value for key, value in required.items()):
        raise ValueError("semantic gate runtime is not fail-closed")
    actual_branches = {
        row["name"]: tuple(row["zero_normalized_action_indices"])
        for row in contract["branches"]
    }
    if actual_branches != EXPECTED_BRANCHES:
        raise ValueError("semantic branch set changed")
    if (
        contract["source"]["endpoint"] != SOURCE
        or contract["intervention_control_steps"] != 1
        or contract["continuation"] != "frozen_deterministic_PPO"
        or contract["rollout_end_endpoint"] != 60
        or contract["left_hand_action_always_formal"] is not True
    ):
        raise ValueError("semantic intervention definition changed")
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


def run_branch(
    *, env, backend, policy, capture: dict, name: str, zero_indices: tuple[int, ...],
) -> tuple[dict, dict, dict[str, np.ndarray]]:
    backend.restore(capture["environment_state"])
    backend.verify_restored_snapshot(capture["environment_state"])
    policy.rnn_states = clone_hidden(capture["pre_forward_hidden"])
    source_contact = _contact_state(env)
    backend.begin_trial(name, SOURCE, 60)
    detailed_contacts = {}
    original_action = None
    executed_action = None
    attempted = 0
    feasible = True
    for current in range(SOURCE, 60):
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        action = policy.preprocess_actions(result["deterministic_actions"])
        if current == SOURCE:
            original_action = action[0].copy()
            low = result["action_lows"][0].detach().cpu().numpy()
            high = result["action_highs"][0].detach().cpu().numpy()
            indices = np.asarray(zero_indices, np.int64)
            if np.any(low[indices] > 1e-7) or np.any(high[indices] < -1e-7):
                raise RuntimeError(f"zero leaves state-feasible support for {name}")
            action[0, indices] = 0.0
            executed_action = action[0].copy()
            if not np.array_equal(executed_action[18:], original_action[18:]):
                raise RuntimeError("semantic intervention changed the left hand")
        feasible = backend.step(action, current)
        attempted += 1
        endpoint = int(env.time_indices[0])
        detailed_contacts[endpoint] = _contact_state(env)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    trace = backend.validation_traces[-1]
    arrays = {
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
    return trace, {
        "source_contact": source_contact,
        "outcome_contacts": detailed_contacts,
        "original_action": original_action,
        "executed_action": executed_action,
    }, arrays


def outcome_metrics(row: dict, reference_qpos: np.ndarray) -> dict:
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
        "tool_freejoint_qpos": qpos[36:43].tolist(),
        "terminated": bool(row["terminated"]),
    }


def summarize_branch(
    *, name: str, zero_indices: tuple[int, ...], trace: dict, detail: dict,
    formal_ppo: dict, reference_qpos: np.ndarray,
) -> dict:
    rows = step_map(trace)
    formal_rows = step_map(formal_ppo)
    endpoint_48 = outcome_metrics(rows[48], reference_qpos)
    endpoint_49 = outcome_metrics(rows[49], reference_qpos) if 49 in rows else None
    contact_48 = detail["outcome_contacts"][48]
    live_contact = bool(
        contact_48["active_right_tool_fingers"]
        and contact_48["live_mjwp_contacts"]
    )
    first_failure = trace["first_failure"]
    failure_endpoint = 60 if first_failure is None else int(first_failure["endpoint"])
    formal_49_score = float(formal_rows[49]["objective_score"][0])
    formal_48_tool = np.asarray(formal_rows[48]["endpoint_qpos"], np.float64)[36:43]
    formal_49_tool = np.asarray(formal_rows[49]["endpoint_qpos"], np.float64)[36:43]
    endpoint_49_score = None if endpoint_49 is None else endpoint_49["objective_score"]
    passes = bool(
        live_contact
        and endpoint_49 is not None
        and not endpoint_49["terminated"]
        and endpoint_49_score < formal_49_score
    )
    return {
        "name": name,
        "source_endpoint": SOURCE,
        "zero_normalized_action_indices": list(zero_indices),
        "original_normalized_action": detail["original_action"].tolist(),
        "executed_normalized_action": detail["executed_action"].tolist(),
        "left_hand_action_bitwise_unchanged": bool(np.array_equal(
            detail["original_action"][18:], detail["executed_action"][18:]
        )),
        "source_47_contact": detail["source_contact"],
        "endpoint_48": {
            **endpoint_48,
            "right_hand_tool_contact": contact_48,
            "has_effective_live_right_hand_tool_contact": live_contact,
            "tool_freejoint_qpos_l2_delta_vs_formal": float(np.linalg.norm(
                np.asarray(endpoint_48["tool_freejoint_qpos"]) - formal_48_tool
            )),
        },
        "endpoint_49": None if endpoint_49 is None else {
            **endpoint_49,
            "score_delta_vs_formal_PPO": endpoint_49_score - formal_49_score,
            "tool_freejoint_qpos_l2_delta_vs_formal": float(np.linalg.norm(
                np.asarray(endpoint_49["tool_freejoint_qpos"]) - formal_49_tool
            )),
        },
        "successful_intervals_from_source": successful_intervals(trace),
        "first_failure_endpoint_or_60": failure_endpoint,
        "passes_predeclared_gate": passes,
        "formal_acceptance_or_chunk_commit_allowed": False,
    }


def training_credit_groups(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("epochs") != 8 or manifest.get("actor_updates") != 8:
        raise ValueError("credit manifest is not the single-pass run")
    rows = []
    for epoch_row in manifest["epoch_reports"]:
        artifact = Path(epoch_row["credit_rows"]["path"])
        if sha256(artifact) != epoch_row["credit_rows"]["sha256"]:
            raise ValueError(f"credit rows changed: {artifact}")
        with np.load(artifact, allow_pickle=False) as data:
            for source in (46, 47):
                selected = np.flatnonzero(data["source_endpoint"] == source)
                for index in selected:
                    flags = data["contact_flags"][index, 0, 0]
                    rows.append({
                        "epoch": int(epoch_row["epoch"]),
                        "source_endpoint": source,
                        "next_step_any_right_tool_contact": bool(flags.any()),
                        "next_step_right_tool_fingers": [
                            finger for finger, active in zip(
                                ("thumb", "index", "middle", "ring", "pinky"), flags
                            ) if active
                        ],
                        "paper_bonus_contact_condition": bool(
                            flags[0] and flags[1:].any()
                        ),
                        "return": float(data["return"][index, 0]),
                        "raw_advantage": float(data["raw_advantage"][index]),
                        "normalized_advantage": float(
                            data["normalized_advantage"][index]
                        ),
                        "future_termination_endpoint": int(
                            data["termination_outcome_endpoint"][index]
                        ),
                    })

    grouped = {}
    for source in (46, 47):
        grouped[str(source)] = {}
        for contact in (False, True):
            subset = [
                row for row in rows
                if row["source_endpoint"] == source
                and row["next_step_any_right_tool_contact"] is contact
            ]
            key = "next_step_contact" if contact else "next_step_no_contact"
            termination = Counter(
                str(row["future_termination_endpoint"]) for row in subset
            )
            grouped[str(source)][key] = {
                "sample_count": len(subset),
                "epochs": sorted({row["epoch"] for row in subset}),
                "return": distribution([row["return"] for row in subset]),
                "raw_advantage": distribution([
                    row["raw_advantage"] for row in subset
                ]),
                "normalized_advantage": distribution([
                    row["normalized_advantage"] for row in subset
                ]),
                "future_termination_endpoint_counts": dict(sorted(
                    termination.items(), key=lambda item: int(item[0])
                )),
                "paper_bonus_contact_condition_count": sum(
                    row["paper_bonus_contact_condition"] for row in subset
                ),
                "right_tool_finger_pattern_counts": dict(Counter(
                    "+".join(row["next_step_right_tool_fingers"]) or "none"
                    for row in subset
                )),
            }
    return {
        "definition": (
            "contact_flags are the next physical endpoint produced by each "
            "source action; exact final CPU state coverage is not claimed"
        ),
        "total_samples": len(rows),
        "by_source_and_next_step_contact": grouped,
        "rows": rows,
    }


def write_summary(path: Path, report: dict) -> None:
    lines = [
        "# Source-47 semantic action-subspace gate",
        "",
    ]
    for row in report["branches"]:
        endpoint_49 = row["endpoint_49"]
        lines.append(
            f"- {row['name']}: contact@48="
            f"{row['endpoint_48']['has_effective_live_right_hand_tool_contact']}, "
            f"score@49={None if endpoint_49 is None else endpoint_49['objective_score']}, "
            f"failure={row['first_failure_endpoint_or_60']}, "
            f"pass={row['passes_predeclared_gate']}"
        )
    lines.extend([
        "",
        f"- gate passed: {report['decision']['gate_passed']}",
        f"- next step: {report['decision']['next_read_only_direction']}",
        "",
        "No training, optimizer update, reward change, task acceptance or chunk commit occurred.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_source47_semantic_action_subspace_gate_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_source47_semantic_action_subspace_gate_v1",
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
    backend = MJWPChunkBackend(env)
    boundary = load_gzip_torch(paths["boundary"])
    replay, _ = run_replay(backend=backend, boundary=boundary)
    _require_exact_trace(replay, formal["replay_validation"], "single-pass Replay")

    checkpoint = load_gzip_torch(paths["checkpoint"])
    temp = tempfile.TemporaryDirectory(prefix=".source47_semantic_", dir=ROOT / "runs")
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
            distribution_spec=distribution_spec,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        ppo, _, captures = capture_formal_ppo(
            backend=backend, policy=policy, boundary=boundary
        )
        _require_exact_trace(ppo, formal["ppo_validation"], "single-pass PPO")

        baseline, baseline_detail, _ = run_branch(
            env=env, backend=backend, policy=policy, capture=captures[SOURCE],
            name="formal_source47_suffix", zero_indices=(),
        )
        expected_steps = [
            row for row in ppo["steps"] if int(row["endpoint"]) > SOURCE
        ]
        if baseline["steps"] != expected_steps:
            raise RuntimeError("restored formal source-47 suffix is not exact")

        branches = []
        arrays = {}
        for name, indices in EXPECTED_BRANCHES.items():
            trace, detail, branch_arrays = run_branch(
                env=env, backend=backend, policy=policy,
                capture=captures[SOURCE], name=name, zero_indices=indices,
            )
            branches.append(summarize_branch(
                name=name, zero_indices=indices, trace=trace, detail=detail,
                formal_ppo=ppo, reference_qpos=reference_qpos,
            ))
            arrays.update(branch_arrays)
        policy.writer.close()
    finally:
        temp.cleanup()

    credit = training_credit_groups(paths["credit_manifest"])
    passed = [row["name"] for row in branches if row["passes_predeclared_gate"]]
    if passed:
        next_direction = "freeze_one_semantic_subspace_candidate_await_user_authorization"
    else:
        next_direction = "move_read_only_state_entry_attribution_to_source_46"

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["branch_arrays"]
    np.savez_compressed(arrays_path, **arrays)
    formal_49 = step_map(ppo)[49]
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
            "restored_source47_suffix_bitwise_reproduced": True,
            "formal_Replay_successful_intervals": successful_intervals(replay),
            "formal_PPO_successful_intervals": successful_intervals(ppo),
            "formal_PPO_endpoint49_score": float(formal_49["objective_score"][0]),
        },
        "formal_source47": {
            "normalized_action": baseline_detail["original_action"].tolist(),
            "source_contact": baseline_detail["source_contact"],
        },
        "branches": branches,
        "branch_arrays": {
            "path": str(arrays_path.resolve()),
            "sha256": sha256(arrays_path),
        },
        "training_credit_source46_47": credit,
        "paper_contact_bonus_boundary": {
            "requires_thumb_and_non_thumb": True,
            "index_only_transmission_contact_earns_bonus": False,
            "reward_change_authorized": False,
        },
        "decision": {
            "passing_branches": passed,
            "gate_passed": bool(passed),
            "new_training_authorized": False,
            "global_half_LR_candidate_unblocked": False,
            "next_read_only_direction": next_direction,
            "reason": (
                "A semantic branch must preserve or restore live right-tool "
                "contact at endpoint 48, survive endpoint 49, and strictly "
                "improve endpoint-49 score."
            ),
        },
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "passing_branches": passed,
        "next_read_only_direction": next_direction,
        "branches": [
            {
                "name": row["name"],
                "contact48": row["endpoint_48"][
                    "has_effective_live_right_hand_tool_contact"
                ],
                "score49": None if row["endpoint_49"] is None else row[
                    "endpoint_49"
                ]["objective_score"],
                "failure": row["first_failure_endpoint_or_60"],
            }
            for row in branches
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
