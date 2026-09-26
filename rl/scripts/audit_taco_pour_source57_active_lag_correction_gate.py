#!/usr/bin/env python3
"""Read-only source-57 active position-lag correction gate for Pour."""

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
    reproduce_baseline,
)
from audit_taco_pour_binary_translation_oracle import (  # noqa: E402
    compact_outcome,
    compact_source_state,
    one_step,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
)
from audit_taco_pour_source57_temporal_hold_gate import (  # noqa: E402
    _assert_scores_exact,
    _contact_summary,
    continue_selector,
)
from audit_taco_pour_source45_state_entry import state_record  # noqa: E402
from audit_taco_pour_tail_semantic_suppression_oracle import (  # noqa: E402
    evaluate_candidates,
    load_contract as load_tail_contract,
    restore_semantic_choice,
)


SOURCE = 57
FORMAL = "formal_source57_PPO"
OFF = "translation_OFF"
ACTIVE = "PPO_plus_unit_tool_position_lag_correction"
CANDIDATES = (FORMAL, OFF, ACTIVE)


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_source57_active_lag_correction_gate_v1":
        raise ValueError("unsupported source57 active-lag contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("source57 active-lag gate is not authorized")
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
        "actor_learning_rate_5e_minus_5_training_allowed": False,
        "learned_gate_training_allowed": False,
        "PPO_retraining_allowed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
    }
    source = contract["source_state"]
    correction = contract["correction"]
    primary = contract["primary_gate"]
    if (
        any(contract["runtime"].get(key) != value for key, value in required.items())
        or contract.get("paper_faithful") is not False
        or tuple(contract["candidates"]) != CANDIDATES
        or source["source_endpoint"] != SOURCE
        or tuple(source["parent_sources_reproduced"]) != tuple(range(44, 57))
        or source["actor_forward_calls"] != 1
        or source["same_complete_physics_snapshot_for_all_candidates"] is not True
        or source["same_pre_forward_observation_and_RNN_hidden"] is not True
        or source["same_post_forward_RNN_hidden_for_all_physics_branches"] is not True
        or correction["candidate_name"] != ACTIVE
        or tuple(correction["modified_normalized_action_indices"]) != (0, 1, 2)
        or correction["coordinate_frame"] != "world"
        or correction["measured_error"]
        != "source57_actual_tool_position_minus_source57_reference_tool_position"
        or correction["normalized_delta_formula"]
        != "-(actual_minus_reference_m / residual_scale_m)"
        or correction["gain"] != 1.0
        or correction["residual_scale_m"] != 0.05
        or correction["add_to_formal_PPO_action"] is not True
        or correction[
            "project_only_modified_indices_to_source57_state_feasible_bounds"
        ] is not True
        or correction["other_action_dimensions_unchanged"] is not True
        or correction["gain_sweep_allowed"] is not False
        or correction["axis_sweep_allowed"] is not False
        or correction["extra_candidates_allowed"] is not False
        or primary["endpoint58_does_not_terminate"] is not True
        or primary["endpoint58_objective_score_strictly_below"] != 1.0
        or primary["contact_used_for_pass_fail"] is not False
        or contract["frozen_decisions"] != {
            "actor_learning_rate_5e_minus_5": "blocked",
            "learned_gate": "blocked",
            "reward_change": "blocked",
            "PPO_retraining": "blocked",
            "chunk_commit": "blocked",
        }
    ):
        raise ValueError("source57 active-lag definition changed")
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


def build_candidates(
    action57: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    position_error_xyz: np.ndarray,
    residual_scale_m: float,
) -> tuple[list[dict], dict]:
    formal = action57.copy()
    off = action57.copy()
    off[0, :3] = 0.0

    normalized_delta = -position_error_xyz / residual_scale_m
    preprojection = action57[0, :3].astype(np.float64) + normalized_delta
    projected = np.clip(preprojection, low[:3], high[:3])
    active = action57.copy()
    active[0, :3] = projected.astype(active.dtype)
    if not np.array_equal(active[0, 3:], action57[0, 3:]):
        raise RuntimeError("active correction changed a non-translation action")

    rows = []
    for name, action in ((FORMAL, formal), (OFF, off), (ACTIVE, active)):
        feasible = bool(
            np.all(action[0] >= low - 1e-7)
            and np.all(action[0] <= high + 1e-7)
        )
        rows.append({
            "name": name,
            "normalized_action": action,
            "inside_source57_state_feasible_bounds": feasible,
        })
    correction = {
        "source57_tool_position_error_xyz_actual_minus_reference_m": (
            position_error_xyz.tolist()
        ),
        "residual_scale_m": residual_scale_m,
        "fixed_gain": 1.0,
        "normalized_delta_before_projection": normalized_delta.tolist(),
        "formal_PPO_translation": action57[0, :3].tolist(),
        "translation_before_projection": preprojection.tolist(),
        "source57_state_feasible_low_translation": low[:3].tolist(),
        "source57_state_feasible_high_translation": high[:3].tolist(),
        "translation_after_projection": projected.tolist(),
        "projection_delta": (projected - preprojection).tolist(),
        "projection_changed_axes": [
            axis for axis, changed in zip(
                ("x", "y", "z"), np.abs(projected - preprojection) > 0.0
            ) if changed
        ],
    }
    return rows, correction


def write_summary(path: Path, report: dict) -> None:
    correction = report["active_correction"]
    lines = [
        "# Source-57 active lag correction gate",
        "",
        f"- correction passed endpoint58: {report['decision']['active_correction_passes_endpoint58']}",
        f"- best successful intervals: {report['decision']['best_successful_intervals']}/40",
        f"- projected axes: {correction['projection_changed_axes']}",
        f"- next blocker: {report['decision']['next_blocker']}",
        "",
        "The only active candidate adds unit-gain current tool-position lag to the",
        "source-57 right-wrist world-translation residual and projects those three",
        "values to the already frozen state-feasible support. No gain or axis sweep",
        "was performed. This is a local engineering diagnostic, not EgoEngine RL.",
        "No training or chunk commit is authorized.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/taco_pour_source57_active_lag_correction_gate_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/taco_pour_source57_active_lag_correction_gate_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    tail_report = json.loads(paths["tail_report"].read_text())
    transition_report = json.loads(paths["transition_report"].read_text())
    temporal_hold_report = json.loads(paths["temporal_hold_report"].read_text())
    _, parent_paths, _ = load_tail_contract(paths["tail_contract"])

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

    temporary = tempfile.TemporaryDirectory(prefix=".source57_active_lag_", dir=ROOT / "runs")
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
                tail_report["tail_decisions"][source - 44]
                if source <= 55 else tail_report["source56_fork"]
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

        pre_snapshot = backend.snapshot()
        pre_state = compact_source_state(
            state_record(backend.env), SOURCE, reference_qpos, reference_qvel, objective
        )
        expected_state = transition_report["transition_audit"]["source57_state"]
        if pre_state != expected_state:
            raise RuntimeError("source57 compact state changed")
        packed = policy.obs_to_tensors(backend.observation())
        actor_result = policy.get_deterministic_action_values(packed)
        post_hidden = clone_hidden(actor_result["rnn_states"])
        action57 = policy.preprocess_actions(actor_result["deterministic_actions"])
        low = actor_result["action_lows"][0].detach().cpu().numpy()
        high = actor_result["action_highs"][0].detach().cpu().numpy()
        expected_action = np.asarray(
            transition_report["transition_audit"]["actor_source56_to_57"][
                "source57_normalized_action"
            ],
            np.float32,
        )
        if not np.array_equal(action57[0], expected_action):
            raise RuntimeError("source57 deterministic action changed")

        position_error_xyz = np.asarray(
            pre_state["tracking"][
                "tool_position_error_xyz_actual_minus_reference_m"
            ],
            np.float64,
        )
        candidates, correction = build_candidates(
            action57=action57,
            low=low,
            high=high,
            position_error_xyz=position_error_xyz,
            residual_scale_m=float(contract["correction"]["residual_scale_m"]),
        )
        branch_results = []
        for candidate in candidates:
            if not candidate["inside_source57_state_feasible_bounds"]:
                raise RuntimeError(f"candidate outside support: {candidate['name']}")
            backend.restore(pre_snapshot)
            backend.verify_restored_snapshot(pre_snapshot)
            policy.rnn_states = clone_hidden(post_hidden)
            raw = one_step(
                backend=backend,
                action=candidate["normalized_action"],
                source=SOURCE,
                label=f"source57_active_lag_{candidate['name']}",
            )
            outcome = compact_outcome(raw["row"], raw["state"], reference_qpos)
            full_state = compact_source_state(
                raw["state"], 58, reference_qpos, reference_qvel, objective
            )
            passes = not outcome["terminated"] and outcome["objective_score"] < 1.0
            branch_results.append({
                "name": candidate["name"],
                "normalized_action": candidate["normalized_action"][0].tolist(),
                "inside_source57_state_feasible_bounds": True,
                "executed": True,
                "raw": raw,
                "outcome": outcome,
                "endpoint58_contact": _contact_summary(full_state),
                "endpoint58_tool_freejoint_velocity": full_state["tool"]["freejoint_qvel"],
                "passes_endpoint58": passes,
                "continuation": None,
            })

        formal = next(row for row in branch_results if row["name"] == FORMAL)
        off = next(row for row in branch_results if row["name"] == OFF)
        temporal = {row["name"]: row for row in temporal_hold_report["branches"]}
        if (
            formal["outcome"]["objective_score"]
            != temporal[FORMAL]["outcome"]["objective_score"]
            or off["outcome"]["objective_score"]
            != temporal[OFF]["outcome"]["objective_score"]
        ):
            raise RuntimeError("source57 temporal-hold anchors changed")

        active = next(row for row in branch_results if row["name"] == ACTIVE)
        if active["passes_endpoint58"]:
            active["continuation"] = continue_selector(
                backend=backend,
                policy=policy,
                branch=active,
                post_hidden=post_hidden,
                reference_qpos=reference_qpos,
                reference_qvel=reference_qvel,
                objective=objective,
            )
        policy.writer.close()
    finally:
        temporary.cleanup()

    serializable = [
        {key: value for key, value in row.items() if key != "raw"}
        for row in branch_results
    ]
    active = next(row for row in serializable if row["name"] == ACTIVE)
    best = (
        active["continuation"]["successful_intervals"]
        if active["continuation"] is not None else 37
    )
    if active["passes_endpoint58"]:
        next_blocker = (
            "active_lag_correction_mechanism_requires_deployable_policy_candidate"
            if best == 40 else
            "active_lag_correction_passes_endpoint58_later_failure_requires_attribution"
        )
    else:
        next_blocker = "unit_gain_active_lag_correction_insufficient_no_parameter_sweep_authorized"
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
            "tail_selected_sources44_through56_exactly_reproduced": True,
            "formal_source57_PPO_anchor_exactly_reproduced": True,
            "translation_OFF_anchor_exactly_reproduced": True,
            "source57_actor_forward_calls": 1,
            "same_source57_snapshot_and_post_forward_hidden_for_all_candidates": True,
        },
        "actuator_names": list(actuator_names),
        "source57_state": pre_state,
        "source57_state_feasible_low": low.tolist(),
        "source57_state_feasible_high": high.tolist(),
        "active_correction": correction,
        "branches": serializable,
        "decision": {
            "active_correction_passes_endpoint58": active["passes_endpoint58"],
            "active_correction_continuation": active["continuation"],
            "best_successful_intervals": best,
            "forty_of_forty": best == 40,
            "gain_or_axis_sweep_authorized": False,
            "actor_LR_5e_minus_5_unblocked": False,
            "learned_gate_training_authorized": False,
            "reward_change_authorized": False,
            "PPO_retraining_authorized": False,
            "chunk_acceptance_or_commit_authorized": False,
            "next_blocker": next_blocker,
        },
        "measurement_boundaries": {
            "correction_uses_current_source57_tool_position_lag_only": True,
            "correction_is_unit_gain_in_existing_world_translation_residual_units": True,
            "only_right_wrist_translation_is_projected_to_existing_support": True,
            "other_33_action_dimensions_are_bitwise_unchanged": True,
            "no_gain_or_axis_sweep": True,
            "contact_is_diagnostic_only": True,
            "no_training_or_chunk_commit": True,
        },
    }

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["arrays"]
    np.savez_compressed(
        arrays_path,
        candidate=np.asarray([row["name"] for row in serializable], dtype="U64"),
        passes_endpoint58=np.asarray([
            row["passes_endpoint58"] for row in serializable
        ], np.bool_),
        score=np.asarray([
            row["outcome"]["objective_score"] for row in serializable
        ], np.float64),
        position_error_xyz=np.asarray([
            row["outcome"]["tool_position_error_xyz_actual_minus_reference_m"]
            for row in serializable
        ], np.float64),
        contact_force=np.asarray([
            row["endpoint58_contact"]["sum_normal_force"] for row in serializable
        ], np.float64),
        action=np.asarray([
            row["normalized_action"] for row in serializable
        ], np.float32),
        correction_before_projection=np.asarray(
            correction["translation_before_projection"], np.float64
        ),
        correction_after_projection=np.asarray(
            correction["translation_after_projection"], np.float64
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
        "active_correction": correction,
        **report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
