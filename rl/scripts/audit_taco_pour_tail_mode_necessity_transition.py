#!/usr/bin/env python3
"""Read-only margin and source56--58 transition audit for the tail oracle."""

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
    evaluate_source,
    reproduce_baseline,
    restore_choice,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
)
from audit_taco_pour_tail_semantic_suppression_oracle import (  # noqa: E402
    CANDIDATES,
    evaluate_candidates,
    load_contract as load_parent_contract,
    restore_semantic_choice,
)
from audit_taco_pour_binary_translation_oracle import compact_source_state  # noqa: E402


SOURCES = tuple(range(44, 56))
NESTED_CHAIN = (
    "zero_right_wrist_translation",
    "zero_complete_right_wrist",
    "zero_entire_right_hand",
    "zero_full_36d_residual",
)


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_tail_mode_necessity_transition_audit_v1":
        raise ValueError("unsupported tail mode/transition audit contract")
    if contract.get("status") != "authorized_read_only_audit":
        raise ValueError("tail mode/transition audit is not authorized")
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
        "PPO_retraining_allowed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
    }
    mode = contract["mode_necessity"]
    transition = contract["transition_reproduction"]
    if (
        any(contract["runtime"].get(key) != value for key, value in required.items())
        or contract.get("paper_faithful") is not False
        or tuple(mode["sources"]) != SOURCES
        or mode["numerical_tolerance_for_merging_modes"] is not None
        or tuple(mode["nested_chain"]) != NESTED_CHAIN
        or mode["causal_claim_from_near_tie_allowed"] is not False
        or mode["learned_gate_label_generation_allowed"] is not False
        or transition["untouched_prefix_last_source"] != 43
        or tuple(transition["reproduce_parent_selected_sources"])
        != tuple(range(44, 57))
        or transition["require_parent_scores_exact"] is not True
        or transition["source57_new_semantic_candidates_allowed"] is not False
        or transition["reproduce_existing_source57_binary_ON_OFF_only"] is not True
        or tuple(transition["required_state_endpoints"]) != (56, 57, 58)
        or contract["frozen_decisions"] != {
            "actor_learning_rate_5e_minus_5": "blocked",
            "learned_gate": "blocked",
            "reward_change": "blocked",
            "PPO_retraining": "blocked",
            "chunk_commit": "blocked",
        }
    ):
        raise ValueError("tail mode/transition audit definition changed")
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


def mode_margin_table(parent_report: dict) -> list[dict]:
    rows = []
    for decision in parent_report["tail_decisions"]:
        scores = {
            row["name"]: float(row["outcome"]["objective_score"])
            for row in decision["candidates"]
        }
        ordered = sorted(
            scores,
            key=lambda name: (
                scores[name],
                len(CANDIDATES[name]),
                list(CANDIDATES).index(name),
            ),
        )
        minimum = scores[ordered[0]]
        exact_minimum = [name for name in scores if scores[name] == minimum]
        nested = []
        for less, more in zip(NESTED_CHAIN, NESTED_CHAIN[1:]):
            nested.append({
                "less_suppression": less,
                "more_suppression": more,
                "score_more_minus_less": scores[more] - scores[less],
                "negative_means_incremental_improvement": True,
            })
        rows.append({
            "source_endpoint": int(decision["source_endpoint"]),
            "reported_winner": decision["selected"],
            "winner_score": scores[decision["selected"]],
            "runner_up_by_exact_score": ordered[1],
            "runner_up_score": scores[ordered[1]],
            "winner_runner_up_absolute_margin": scores[ordered[1]] - minimum,
            "exact_minimum_score_candidates": exact_minimum,
            "minimal_zeroed_dimensions_exact_minimum": min(
                exact_minimum, key=lambda name: (
                    len(CANDIDATES[name]), list(CANDIDATES).index(name)
                )
            ),
            "candidate_score_delta_from_winner": {
                name: score - minimum for name, score in scores.items()
            },
            "nested_incremental_effects": nested,
            "numerical_tolerance_used_to_merge_modes": None,
        })
    return rows


def _assert_scores_exact(produced: dict, expected: dict) -> None:
    expected_scores = {
        row["name"]: row["outcome"]["objective_score"]
        for row in expected["candidates"]
    }
    for row in produced["candidates"]:
        name = row["name"]
        if row["outcome"]["objective_score"] != expected_scores[name]:
            raise RuntimeError(
                f"parent candidate score changed at source {produced['source']}: {name}"
            )


def _vector_delta(next_values, current_values, scale: float = 1.0) -> list[float]:
    return ((np.asarray(next_values, np.float64) - np.asarray(current_values, np.float64)) * scale).tolist()


def _contact_summary(state: dict) -> dict:
    contact = state["right_hand_tool_contact"]
    live = contact["live_contacts"]
    return {
        "active_finger_roles": contact["active_finger_roles"],
        "live_contact_count": len(live),
        "sum_normal_force": float(sum(row["normal_force"] for row in live)),
        "live_contacts": live,
    }


def _state_transition(current: dict, following: dict, reference_qpos: np.ndarray) -> dict:
    source = int(current["endpoint"])
    endpoint = int(following["endpoint"])
    current_error = np.asarray(
        current["tracking"]["tool_position_error_xyz_actual_minus_reference_m"],
        np.float64,
    )
    next_error = np.asarray(
        following["tracking"]["tool_position_error_xyz_actual_minus_reference_m"],
        np.float64,
    )
    absolute_error_growth = np.abs(next_error) - np.abs(current_error)
    axes = ("x", "y", "z")
    dominant = axes[int(np.argmax(absolute_error_growth))]
    return {
        "source_endpoint": source,
        "outcome_endpoint": endpoint,
        "actual_tool_displacement_world_mm": _vector_delta(
            following["tool"]["position_world_m"],
            current["tool"]["position_world_m"],
            1000.0,
        ),
        "reference_tool_displacement_world_mm": _vector_delta(
            reference_qpos[endpoint, 36:39],
            reference_qpos[source, 36:39],
            1000.0,
        ),
        "position_error_xyz_change_mm": ((next_error - current_error) * 1000.0).tolist(),
        "absolute_position_error_growth_mm": (absolute_error_growth * 1000.0).tolist(),
        "dominant_absolute_position_error_growth_axis": dominant,
        "position_error_norm_change_mm": 1000.0 * (
            following["tracking"]["position_error_m"]
            - current["tracking"]["position_error_m"]
        ),
        "rotation_error_change_rad": (
            following["tracking"]["rotation_error_rad"]
            - current["tracking"]["rotation_error_rad"]
        ),
        "objective_score_change": (
            following["tracking"]["objective_score"]
            - current["tracking"]["objective_score"]
        ),
        "squared_contribution_change": {
            key: (
                following["tracking"]["ellipse_squared_contributions"][key]
                - current["tracking"]["ellipse_squared_contributions"][key]
            )
            for key in ("position", "rotation")
        },
        "actual_tool_freejoint_linear_velocity": current["tool"]["freejoint_qvel"][:3],
        "next_actual_tool_freejoint_linear_velocity": following["tool"]["freejoint_qvel"][:3],
        "reference_tool_freejoint_linear_velocity": current["reference_tool_freejoint_qvel"][:3],
        "next_reference_tool_freejoint_linear_velocity": following["reference_tool_freejoint_qvel"][:3],
        "actual_minus_reference_linear_velocity": (
            np.asarray(current["tool"]["freejoint_qvel"][:3], np.float64)
            - np.asarray(current["reference_tool_freejoint_qvel"][:3], np.float64)
        ).tolist(),
        "next_actual_minus_reference_linear_velocity": (
            np.asarray(following["tool"]["freejoint_qvel"][:3], np.float64)
            - np.asarray(following["reference_tool_freejoint_qvel"][:3], np.float64)
        ).tolist(),
        "right_wrist_position_in_tool_frame_change_mm": _vector_delta(
            following["right_wrist_relative_to_tool"]["position_in_tool_frame_m"],
            current["right_wrist_relative_to_tool"]["position_in_tool_frame_m"],
            1000.0,
        ),
        "right_wrist_relative_linear_velocity_change_m_s": _vector_delta(
            following["right_wrist_relative_to_tool"][
                "linear_velocity_relative_to_tool_in_tool_frame_m_s"
            ],
            current["right_wrist_relative_to_tool"][
                "linear_velocity_relative_to_tool_in_tool_frame_m_s"
            ],
        ),
        "source_contact": _contact_summary(current),
        "outcome_contact": _contact_summary(following),
    }


def _actor_delta(source56: dict, source57: dict, actuator_names: tuple[str, ...]) -> dict:
    action56 = np.asarray(source56["complete_deterministic_action"], np.float64)
    action57 = np.asarray(source57["complete_deterministic_action"], np.float64)
    delta = action57 - action56
    groups = {
        "right_wrist_translation": slice(0, 3),
        "right_wrist_rotation": slice(3, 6),
        "right_fingers": slice(6, 18),
        "left_wrist_translation": slice(18, 21),
        "left_wrist_rotation": slice(21, 24),
        "left_fingers": slice(24, 36),
    }
    order = np.argsort(-np.abs(delta))
    return {
        "source56_normalized_action": action56.tolist(),
        "source57_normalized_action": action57.tolist(),
        "source57_minus_source56": delta.tolist(),
        "group_delta_l2": {
            name: float(np.linalg.norm(delta[index])) for name, index in groups.items()
        },
        "largest_absolute_changes": [
            {
                "actuator": actuator_names[int(index)],
                "source56": float(action56[index]),
                "source57": float(action57[index]),
                "delta": float(delta[index]),
            }
            for index in order[:8]
        ],
    }


def write_summary(path: Path, report: dict) -> None:
    transition = report["transition_audit"]
    lines = [
        "# Tail mode necessity and transition audit",
        "",
        "- no numerical tolerance was used to merge mode labels",
        "- distinct finger causality established: false",
        "- distinct bilateral causality established: false",
        f"- source56->57 dominant position-growth axis: {transition['source56_to_57']['dominant_absolute_position_error_growth_axis']}",
        f"- source57->58 dominant position-growth axis: {transition['source57_to_58']['dominant_absolute_position_error_growth_axis']}",
        "- source56 contact: none; endpoint57: pinky contact; endpoint58: weak pinky contact",
        "- endpoint58 remains a position-dominated failure",
        "- training, learned gate, reward changes and chunk commit remain blocked",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/taco_pour_tail_mode_necessity_transition_audit_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/taco_pour_tail_mode_necessity_transition_audit_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    parent_report = json.loads(paths["parent_report"].read_text())
    parent_contract, parent_paths, _ = load_parent_contract(paths["parent_contract"])
    margins = mode_margin_table(parent_report)

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

    objective = load_runtime_objective(
        parent_paths["protocol"], parent_paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        parent_paths["protocol"], parent_paths["observation_profile"],
        require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(parent_paths["action_profile"])
    distribution, _ = load_truncated_gaussian_profile(
        parent_paths["distribution_profile"]
    )
    _, initialization = load_accepted_initialization(
        parent_paths["initialization_report"], parent_paths["simulator_config"]
    )
    config = _load_ego_config(str(parent_paths["simulator_config"]), "cpu")
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
    boundary = load_gzip_torch(parent_paths["boundary"])
    checkpoint = load_gzip_torch(parent_paths["checkpoint"])

    temporary = tempfile.TemporaryDirectory(prefix=".tail_transition_", dir=ROOT / "runs")
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
            distribution_spec=distribution,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        captures, _ = reproduce_baseline(
            backend=backend,
            policy=policy,
            boundary=boundary,
            reference_qpos=reference_qpos,
            reference_qvel=reference_qvel,
            objective=objective,
            expected_arrays=parent_paths["baseline_oracle_arrays"],
            expected_off_sources=json.loads(
                parent_paths["baseline_oracle_report"].read_text()
            )["oracle_result"]["OFF_sources"],
            capture_sources=(44,),
        )
        backend.restore(captures[44]["snapshot"])
        backend.verify_restored_snapshot(captures[44]["snapshot"])
        policy.rnn_states = clone_hidden(captures[44]["pre_hidden"])

        source56_evaluated = None
        for source in range(44, 57):
            evaluated = evaluate_candidates(
                backend=backend,
                policy=policy,
                source=source,
                reference_qpos=reference_qpos,
                reference_qvel=reference_qvel,
                objective=objective,
            )
            expected = (
                parent_report["tail_decisions"][source - 44]
                if source <= 55 else parent_report["source56_fork"]
            )
            _assert_scores_exact(evaluated, expected)
            expected_selected = (
                expected["selected"] if source <= 55
                else expected["selected_for_continuation"]
            )
            if evaluated["selected"]["name"] != expected_selected:
                raise RuntimeError(f"parent selected mode changed at source {source}")
            restore_semantic_choice(
                backend=backend,
                policy=policy,
                evaluated=evaluated,
                candidate=evaluated["selected"],
            )
            if source == 56:
                source56_evaluated = evaluated

        if source56_evaluated is None:
            raise RuntimeError("source56 was not reproduced")
        source57_evaluated = evaluate_source(
            backend=backend,
            policy=policy,
            source=57,
            reference_qpos=reference_qpos,
            reference_qvel=reference_qvel,
            objective=objective,
        )
        expected_continuation = parent_report["source56_fork"][
            "binary_oracle_continuation"
        ][0]
        if (
            source57_evaluated["ON"]["objective_score"]
            != expected_continuation["ON_score"]
            or source57_evaluated["OFF"]["objective_score"]
            != expected_continuation["OFF_score"]
            or source57_evaluated["greedy_selected"] != "ON"
        ):
            raise RuntimeError("existing source57 binary continuation changed")
        selected58 = restore_choice(
            backend=backend,
            policy=policy,
            evaluated=source57_evaluated,
            selected="ON",
        )
        state56 = source56_evaluated["source_state"]
        state57 = source57_evaluated["source_state"]
        state58 = compact_source_state(
            selected58["state"], 58, reference_qpos, reference_qvel, objective
        )
        actor_delta = _actor_delta(
            source56_evaluated["actor"],
            source57_evaluated["actor"],
            tuple(actuator_names),
        )
        policy.writer.close()
    finally:
        temporary.cleanup()

    transition56 = _state_transition(state56, state57, reference_qpos)
    transition57 = _state_transition(state57, state58, reference_qpos)
    source44 = margins[0]
    source45 = margins[1]
    wrist_increment = lambda row: next(
        item["score_more_minus_less"] for item in row["nested_incremental_effects"]
        if item["more_suppression"] == "zero_complete_right_wrist"
    )
    report = {
        "schema": contract["schema"],
        "status": "completed_read_only_audit",
        "paper_faithful": False,
        "classification": contract["classification"],
        "training_executed": False,
        "optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "parent_evidence": {
            "report": str(paths["parent_report"].resolve()),
            "report_sha256": sha256(paths["parent_report"]),
            "saved_results_only_for_mode_margins": True,
            "selected_path_and_existing_binary_continuation_exactly_reproduced_for_transition": True,
        },
        "mode_necessity": {
            "numerical_tolerance_used_to_merge_modes": None,
            "near_tie_promoted_to_causal_label": False,
            "learned_gate_labels_generated": False,
            "sources": margins,
            "source44_complete_wrist_improvement_vs_translation": -wrist_increment(source44),
            "source45_complete_wrist_improvement_vs_translation": -wrist_increment(source45),
            "distinct_finger_causality_established": False,
            "distinct_bilateral_causality_established": False,
            "interpretation": (
                "Complete-wrist suppression has measurable additional benefit at "
                "sources 44--45. Translation is the principal narrower mode through "
                "most later sources. Broader winners at 48/52/53 include exact or "
                "near score ties; their names alone do not establish finger or "
                "bilateral causality."
            ),
        },
        "transition_audit": {
            "source56_state": state56,
            "source57_state": state57,
            "source58_state": state58,
            "source56_to_57": transition56,
            "source57_to_58": transition57,
            "actor_source56_to_57": actor_delta,
            "source57_existing_binary_scores": {
                "ON": source57_evaluated["ON"]["objective_score"],
                "OFF": source57_evaluated["OFF"]["objective_score"],
                "selected": "ON",
            },
            "interpretation": (
                "The tool barely follows the reference displacement from 56 to 57; "
                "the dominant absolute position-error growth is identified per axis. "
                "Pinky contact is reacquired at 57 but weakens sharply by 58, where "
                "position remains the dominant failure contribution. This audit does "
                "not test new source57 semantic actions."
            ),
        },
        "decision": {
            "translation_only_gate_sufficient": False,
            "finger_or_bilateral_causality_claim_authorized": False,
            "source57_new_semantic_gate_executed": False,
            "actor_LR_5e_minus_5_unblocked": False,
            "learned_gate_training_authorized": False,
            "reward_change_authorized": False,
            "PPO_retraining_authorized": False,
            "chunk_acceptance_or_commit_authorized": False,
            "next_blocker": "tail_mode_state_criterion_and_endpoint58_active_correction_unresolved",
        },
    }

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["arrays"]
    np.savez_compressed(
        arrays_path,
        source=np.asarray([row["source_endpoint"] for row in margins], np.int64),
        winner_margin=np.asarray(
            [row["winner_runner_up_absolute_margin"] for row in margins], np.float64
        ),
        nested_increment=np.asarray([
            [item["score_more_minus_less"] for item in row["nested_incremental_effects"]]
            for row in margins
        ], np.float64),
        source56_action=np.asarray(
            actor_delta["source56_normalized_action"], np.float32
        ),
        source57_action=np.asarray(
            actor_delta["source57_normalized_action"], np.float32
        ),
    )
    report["arrays"] = {
        "path": str(arrays_path.resolve()),
        "sha256": sha256(arrays_path),
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "source44_wrist_benefit": report["mode_necessity"][
            "source44_complete_wrist_improvement_vs_translation"
        ],
        "source45_wrist_benefit": report["mode_necessity"][
            "source45_complete_wrist_improvement_vs_translation"
        ],
        "source56_to_57_dominant_axis": transition56[
            "dominant_absolute_position_error_growth_axis"
        ],
        "source57_to_58_dominant_axis": transition57[
            "dominant_absolute_position_error_growth_axis"
        ],
        **report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
