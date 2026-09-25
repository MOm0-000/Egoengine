#!/usr/bin/env python3
"""Frozen-state, frozen-policy local controllability audit for Pour endpoints 44--47."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import io
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
    _active_contacts,
    _load_gzip_torch,
    _require_exact_trace,
    _sha256,
)


def _gzip_torch(value, torch) -> tuple[bytes, str]:
    raw_buffer = io.BytesIO()
    torch.save(value, raw_buffer)
    raw = raw_buffer.getvalue()
    compressed_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_buffer, mode="wb", mtime=0) as stream:
        stream.write(raw)
    return compressed_buffer.getvalue(), hashlib.sha256(raw).hexdigest()


def _load_contract(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_corrected_local_controllability_v1":
        raise ValueError("unsupported local-controllability contract")
    if contract.get("status") != "authorized_read_only_counterfactual":
        raise ValueError("counterfactual audit is not authorized")
    if contract.get("paper_faithful") is not False:
        raise ValueError("local controllability audit cannot be paper-faithful")
    runtime = contract.get("runtime", {})
    required_runtime = {
        "backend": "CPU_MuJoCo_Warp",
        "optimizer_steps": 0,
        "actor_frozen": True,
        "normalization_frozen": True,
        "objective_changed": False,
        "reward_changed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    if any(runtime.get(name) != value for name, value in required_runtime.items()):
        raise ValueError("runtime contract is not fail-closed")
    for row in contract.get("inputs", {}).values():
        artifact = Path(row["path"])
        if _sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
    sweep = contract["single_axis_sweep"]
    if (
        contract["snapshots"]["source_endpoints"] != [44, 45, 46, 47]
        or sweep["fractions_of_formal_residual_bound"]
        != [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        or sweep["formal_residual_bound_numeric"] != 0.05
        or sweep["diagnostic_first_step_clip_numeric"] != 0.10
        or sweep["rollout_final_endpoint"] != 48
    ):
        raise ValueError("counterfactual grid differs from its frozen definition")
    return contract, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _clone_rnn(states):
    return [state.detach().cpu().clone() for state in states]


def _restore_source(env, policy, source: dict, snapshot_equal) -> None:
    env.set_env_state(source["environment_state"])
    if not snapshot_equal(env.get_env_state(), source["environment_state"]):
        raise RuntimeError("restored counterfactual source differs from saved snapshot")
    policy.rnn_states = [state.to("cpu").clone() for state in source["rnn_states"]]


def _policy_decision(policy, backend):
    observation = policy.obs_to_tensors(backend.observation())
    result = policy.get_deterministic_action_values(observation)
    policy.rnn_states = result["rnn_states"]
    return result, result["deterministic_actions"][0].detach().cpu().numpy().copy()


def _step_record(
    env,
    action: np.ndarray,
    result,
    reference_qpos: np.ndarray,
    right_wrist_qpos_indices: np.ndarray,
    *,
    diagnostic_clip: float | None,
):
    from video_to_spider.rl.residual_semantics import control_target_residuals

    source_endpoint = int(env.time_indices[0])
    reference_ctrl = env._reference_ctrls(env.time_indices, offset=1)
    if env._state_feasible_action_contract is not None:
        reference_ctrl, _ = env._snap_state_feasible_reference(reference_ctrl)
    reference_ctrl_np = reference_ctrl.detach().cpu().numpy()
    formal_residual = env.env_cfg.residual
    if diagnostic_clip is not None:
        env.env_cfg = replace(
            env.env_cfg,
            residual=replace(formal_residual, residual_clip=float(diagnostic_clip)),
        )
    try:
        _, reward, _, info = env.step(action[None].astype(np.float32), auto_reset=False)
    finally:
        env.env_cfg = replace(env.env_cfg, residual=formal_residual)

    endpoint = int(env.time_indices[0])
    qpos = env._mjwp.get_qpos(env.ego_cfg, env.env)[0].detach().cpu().numpy()
    vector = qpos[36:39].astype(np.float64) - reference_qpos[endpoint, 36:39]
    position = float(info["object_position_error"][0, 0])
    rotation = float(info["object_rotation_error"][0, 0])
    score = float(info["object_tracking_error_per_object"][0, 0])
    if not np.isclose(np.linalg.norm(vector), position, rtol=0.0, atol=2e-7):
        raise RuntimeError("counterfactual position vector does not match runtime norm")
    semantics = control_target_residuals(
        env.env.model_cpu,
        reference_ctrl_np,
        env._last_ctrl.detach().cpu().numpy(),
    )
    lost = semantics["residual_lost_to_ctrlrange"][0]
    contact_flags = np.asarray(info["contact_flags"][0], dtype=bool)
    low = result["action_lows"][0].detach().cpu().numpy()
    high = result["action_highs"][0].detach().cpu().numpy()
    return {
        "source_endpoint": source_endpoint,
        "outcome_endpoint": endpoint,
        "bowl_position_error_vector_actual_minus_reference_m": vector.tolist(),
        "bowl_position_error_m": position,
        "bowl_rotation_error_rad": rotation,
        "right_wrist_qpos": qpos[right_wrist_qpos_indices].astype(
            np.float64
        ).tolist(),
        "normalized_ellipse_score": score,
        "ellipse_squared_contributions": {
            "position": (position / 0.12) ** 2,
            "rotation": (rotation / 1.5) ** 2,
        },
        "reward": float(reward[0]),
        "tracking_terminated": bool(info["terminated"][0]),
        "right_hand_contact_flags_object_finger": contact_flags[0].tolist(),
        "active_contacts": _active_contacts(contact_flags),
        "policy_normalized_action": result["deterministic_actions"][0]
        .detach().cpu().numpy().tolist(),
        "executed_normalized_action": action.tolist(),
        "formal_state_feasible_low": low.tolist(),
        "formal_state_feasible_high": high.tolist(),
        "formal_support_exceeded_count": int(
            np.count_nonzero((action < low - 1e-7) | (action > high + 1e-7))
        ),
        "requested_residual": semantics["requested_residual"][0].tolist(),
        "effective_residual_after_ctrlrange": semantics[
            "effective_residual_after_ctrlrange"
        ][0].tolist(),
        "residual_lost_to_ctrlrange": lost.tolist(),
        "residual_lost_above_1e_8_count": int(
            np.count_nonzero(np.abs(lost) > 1e-8)
        ),
        "residual_lost_max_abs": float(np.abs(lost).max()),
        "diagnostic_residual_clip_numeric": (
            float(diagnostic_clip) if diagnostic_clip is not None else 0.05
        ),
        "endpoint_qpos_sha256": hashlib.sha256(
            np.ascontiguousarray(qpos).tobytes()
        ).hexdigest(),
        "_endpoint_qpos": qpos,
    }


def _run_branch(
    env,
    backend,
    policy,
    source: dict,
    reference_qpos: np.ndarray,
    right_wrist_qpos_indices: np.ndarray,
    *,
    kind: str,
    axis_index: int,
    snapshot_equal,
    fraction: float | None = None,
):
    _restore_source(env, policy, source, snapshot_equal)
    rows = []
    while int(env.time_indices[0]) < 48:
        result, action = _policy_decision(policy, backend)
        diagnostic_clip = None
        if kind == "additive_first_step" and not rows:
            action[axis_index] += float(fraction)
            diagnostic_clip = 0.10
        elif kind == "drop_axis":
            action[axis_index] = 0.0
        elif kind == "keep_only_axis":
            kept = float(action[axis_index])
            action[:] = 0.0
            action[axis_index] = kept
        elif kind != "additive_first_step":
            raise ValueError(f"unknown intervention {kind}")
        rows.append(
            _step_record(
                env,
                action,
                result,
                reference_qpos,
                right_wrist_qpos_indices,
                diagnostic_clip=diagnostic_clip,
            )
        )
    final = rows[-1]
    first_failure = next(
        (row["outcome_endpoint"] for row in rows if row["tracking_terminated"]),
        None,
    )
    output_rows = []
    for row in rows:
        output = dict(row)
        output.pop("_endpoint_qpos")
        output_rows.append(output)
    return {
        "source_endpoint": int(source["endpoint"]),
        "intervention": kind,
        "axis_index": axis_index,
        "fraction_of_formal_bound": fraction,
        "continued_after_tracking_termination_for_diagnosis": True,
        "first_tracking_failure_endpoint": first_failure,
        "steps": output_rows,
        "final": output_rows[-1],
        "_final_qpos": final["_endpoint_qpos"],
    }


def _sweep_analysis(branches: list[dict], axes: dict[str, int]) -> dict:
    report = {}
    for source in (44, 45, 46, 47):
        source_rows = {}
        for axis_name, axis_index in axes.items():
            rows = sorted(
                (
                    row for row in branches
                    if row["source_endpoint"] == source
                    and row["axis_index"] == axis_index
                ),
                key=lambda row: row["fraction_of_formal_bound"],
            )
            fractions = np.asarray(
                [row["fraction_of_formal_bound"] for row in rows], np.float64
            )
            vectors = np.asarray(
                [
                    row["final"][
                        "bowl_position_error_vector_actual_minus_reference_m"
                    ]
                    for row in rows
                ],
                np.float64,
            )
            wrist_qpos = np.asarray(
                [row["final"]["right_wrist_qpos"] for row in rows], np.float64
            )
            scores = np.asarray(
                [row["final"]["normalized_ellipse_score"] for row in rows]
            )
            baseline_index = int(np.flatnonzero(fractions == 0.0)[0])
            negative_index = int(np.flatnonzero(fractions == -0.25)[0])
            positive_index = int(np.flatnonzero(fractions == 0.25)[0])
            baseline_abs_y = abs(vectors[baseline_index, 1])
            candidates = {
                "negative_0.25": abs(vectors[negative_index, 1]),
                "positive_0.25": abs(vectors[positive_index, 1]),
            }
            selected = min(candidates, key=candidates.get)
            if candidates[selected] >= baseline_abs_y:
                selected = None
            contact_patterns = [
                tuple(row["final"]["active_contacts"]) for row in rows
            ]
            source_rows[axis_name] = {
                "axis_index": axis_index,
                "axis_unit": "m" if axis_index < 3 else "rad",
                "fractions": fractions.tolist(),
                "final_bowl_position_error_vectors_m": vectors.tolist(),
                "final_right_wrist_qpos": wrist_qpos.tolist(),
                "final_scores": scores.tolist(),
                "final_active_contacts": [list(pattern) for pattern in contact_patterns],
                "baseline_final_abs_bowl_y_error_m": baseline_abs_y,
                "small_probe_abs_bowl_y_error_m": candidates,
                "small_probe_direction_reducing_abs_bowl_y": selected,
                "small_probe_improvement_m": (
                    baseline_abs_y - candidates[selected] if selected else 0.0
                ),
                "central_signed_bowl_y_gain": float(
                    (vectors[positive_index, 1] - vectors[negative_index, 1])
                    / (0.5 * 0.05)
                ),
                "central_selected_wrist_axis_qpos_gain": float(
                    (
                        wrist_qpos[positive_index, axis_index]
                        - wrist_qpos[negative_index, axis_index]
                    )
                    / (0.5 * 0.05)
                ),
                "signed_bowl_y_monotonic_over_full_grid": bool(
                    np.all(np.diff(vectors[:, 1]) >= -1e-8)
                    or np.all(np.diff(vectors[:, 1]) <= 1e-8)
                ),
                "best_fraction_for_abs_bowl_y": float(
                    fractions[int(np.argmin(np.abs(vectors[:, 1])))]
                ),
                "best_abs_bowl_y_improvement_m": float(
                    baseline_abs_y - np.min(np.abs(vectors[:, 1]))
                ),
                "contact_pattern_changes_across_grid": len(set(contact_patterns)) - 1,
                "formal_support_exceeding_branch_count": int(
                    sum(
                        row["steps"][0]["formal_support_exceeded_count"] > 0
                        for row in rows
                    )
                ),
                "maximum_ctrlrange_loss": float(
                    max(
                        step["residual_lost_max_abs"]
                        for row in rows for step in row["steps"]
                    )
                ),
            }
        report[str(source)] = source_rows
    return report


def _ablation_analysis(
    branches: list[dict], baseline_endpoint_48: dict, axes: dict[str, int]
) -> dict:
    baseline_vector = np.asarray(
        baseline_endpoint_48["bowl_position_error_vector_actual_minus_reference_m"]
    )
    baseline_score = baseline_endpoint_48["normalized_ellipse_score"]
    result = {}
    for branch in branches:
        final = branch["final"]
        vector = np.asarray(
            final["bowl_position_error_vector_actual_minus_reference_m"]
        )
        key = (
            f"source_{branch['source_endpoint']}/"
            f"{branch['intervention']}/"
            f"{next(name for name, index in axes.items() if index == branch['axis_index'])}"
        )
        result[key] = {
            "final_bowl_position_error_vector_m": vector.tolist(),
            "change_from_baseline_vector_m": (vector - baseline_vector).tolist(),
            "final_score": final["normalized_ellipse_score"],
            "change_from_baseline_score": (
                final["normalized_ellipse_score"] - baseline_score
            ),
            "final_active_contacts": final["active_contacts"],
            "first_tracking_failure_endpoint": branch[
                "first_tracking_failure_endpoint"
            ],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_corrected_local_controllability_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_corrected_local_controllability_v1",
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
    inputs = contract["inputs"]
    path = lambda name: Path(inputs[name]["path"])
    formal = json.loads(path("formal_report").read_text())
    expected_rl = next(
        trace for trace in formal["chunks"][0]["validation_traces"]
        if trace["mode"] == "rl"
    )
    objective = load_runtime_objective(
        path("replay_rl_protocol_snapshot"), path("objective_profile"),
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        path("replay_rl_protocol_snapshot"), path("observation_profile"),
        require_run_ready=True,
    )
    residual, _ = load_residual_action_profile(path("action_profile"))
    spec, distribution_artifact = load_truncated_gaussian_profile(
        path("distribution_profile")
    )
    _, initialization = load_accepted_initialization(
        path("initialization_report"), path("simulator_config")
    )
    config = _load_ego_config(str(path("simulator_config")), "cpu")
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
    boundary, _, _ = _load_gzip_torch(path("endpoint_20_boundary"), torch)
    checkpoint, _, checkpoint_raw = _load_gzip_torch(
        path("checkpoint_artifact"), torch
    )
    actor_hash = formal["chunks"][0]["training_runs"][0]["actor_state_sha256"]
    if _model_state_sha256(checkpoint["model"]) != actor_hash:
        raise ValueError("frozen actor differs from the formal corrected run")

    with tempfile.TemporaryDirectory(
        prefix=".corrected_local_controllability_", dir=ROOT / "runs"
    ) as temporary:
        temp = Path(temporary)
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
            state.to("cpu").zero_()
            for state in policy.model.get_default_rnn_state()
        ]

        backend.restore(boundary)
        backend.verify_restored_snapshot(boundary)
        sources = {}
        backend.begin_trial("rl", 20, 60)
        validated = 0
        for reference_step in range(20, 60):
            if reference_step in contract["snapshots"]["source_endpoints"]:
                sources[reference_step] = {
                    "endpoint": reference_step,
                    "environment_state": backend.snapshot(),
                    "rnn_states": _clone_rnn(policy.rnn_states),
                }
            result, action = _policy_decision(policy, backend)
            feasible = backend.step(action[None], reference_step)
            if not feasible:
                break
            validated += 1
        backend.end_trial(validated == 40, validated)
        baseline_trace = backend.validation_traces[-1]
        _require_exact_trace(baseline_trace, expected_rl, "PPO baseline")
        if set(sources) != {44, 45, 46, 47}:
            raise RuntimeError("baseline did not capture every frozen source endpoint")

        snapshot_artifacts = {}
        snapshot_dir = temp / "source_snapshots"
        snapshot_dir.mkdir()
        for endpoint, source in sources.items():
            artifact, raw_sha = _gzip_torch(
                {
                    "schema": "corrected_local_controllability_source_v1",
                    "endpoint": endpoint,
                    "diagnostic_only": True,
                    "resume_authorized": False,
                    "actor_state_sha256": actor_hash,
                    "environment_state": source["environment_state"],
                    "rnn_states": source["rnn_states"],
                },
                torch,
            )
            destination = snapshot_dir / f"endpoint_{endpoint}.pt.gz"
            destination.write_bytes(artifact)
            snapshot_artifacts[str(endpoint)] = {
                "path": str((args.output / "source_snapshots" / destination.name).resolve()),
                "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                "uncompressed_pt_sha256": raw_sha,
                "diagnostic_only": True,
                "resume_authorized": False,
                "snapshot_field_count": len(source["environment_state"]),
            }

        reference_qpos = reference[0].cpu().numpy().astype(np.float64)
        control_indices = tuple(env.env_cfg.residual.hand_control_indices)
        right_wrist_qpos_indices = np.asarray(
            [
                int(
                    env.env.model_cpu.jnt_qposadr[
                        int(env.env.model_cpu.actuator_trnid[actuator, 0])
                    ]
                )
                for actuator in control_indices[:6]
            ],
            dtype=np.int64,
        )
        axes = contract["single_axis_sweep"]["axes"]
        fractions = contract["single_axis_sweep"][
            "fractions_of_formal_residual_bound"
        ]
        sweep_branches = []
        formal_endpoint_48 = next(
            step for step in expected_rl["steps"] if step["endpoint"] == 48
        )
        for source_endpoint, source in sources.items():
            for axis_index in axes.values():
                for fraction in fractions:
                    branch = _run_branch(
                        env,
                        backend,
                        policy,
                        source,
                        reference_qpos,
                        right_wrist_qpos_indices,
                        kind="additive_first_step",
                        axis_index=axis_index,
                        snapshot_equal=_snapshot_value_equal,
                        fraction=float(fraction),
                    )
                    if fraction == 0.0 and not np.array_equal(
                        branch["_final_qpos"],
                        np.asarray(formal_endpoint_48["endpoint_qpos"], np.float32),
                    ):
                        raise RuntimeError(
                            "zero-perturbation branch does not reproduce formal endpoint 48"
                        )
                    branch.pop("_final_qpos")
                    sweep_branches.append(branch)

        ablation_branches = []
        for source_endpoint, source in sources.items():
            for axis_index in axes.values():
                for kind in ("drop_axis", "keep_only_axis"):
                    branch = _run_branch(
                        env,
                        backend,
                        policy,
                        source,
                        reference_qpos,
                        right_wrist_qpos_indices,
                        kind=kind,
                        axis_index=axis_index,
                        snapshot_equal=_snapshot_value_equal,
                    )
                    branch.pop("_final_qpos")
                    ablation_branches.append(branch)

        baseline_endpoint_48 = {
            "bowl_position_error_vector_actual_minus_reference_m": (
                np.asarray(formal_endpoint_48["endpoint_qpos"], np.float64)[36:39]
                - reference_qpos[48, 36:39]
            ).tolist(),
            "normalized_ellipse_score": formal_endpoint_48["objective_score"][0],
        }
        sweep_analysis = _sweep_analysis(sweep_branches, axes)
        ablation_analysis = _ablation_analysis(
            ablation_branches, baseline_endpoint_48, axes
        )
        all_steps = [
            step
            for branch in (*sweep_branches, *ablation_branches)
            for step in branch["steps"]
        ]
        report = {
            "schema": "taco_pour_corrected_local_controllability_v1",
            "status": "completed_read_only_counterfactual",
            "paper_faithful": False,
            "scope": {
                "optimizer_steps": 0,
                "training_performed": False,
                "actor_frozen": True,
                "objective_and_reward_unchanged": True,
                "chunk_acceptance_allowed": False,
                "chunk_commit_allowed": False,
                "diagnostic_continuation_after_tracking_failure": True,
                "diagnostic_actions_may_exceed_formal_residual_support": True,
            },
            "contract": contract_artifact,
            "inputs": {
                name: {"path": row["path"], "sha256": row["sha256"]}
                for name, row in inputs.items()
            },
            "actor": {
                "state_sha256": actor_hash,
                "checkpoint_uncompressed_sha256": hashlib.sha256(
                    checkpoint_raw
                ).hexdigest(),
                "distribution": distribution_artifact,
            },
            "baseline_regression": {
                "formal_trace_exactly_reproduced": True,
                "validated_intervals": baseline_trace["validated_steps"],
                "first_failure": baseline_trace["first_failure"],
            },
            "source_snapshots": snapshot_artifacts,
            "single_axis_sweep": {
                "branch_count": len(sweep_branches),
                "branches": sweep_branches,
                "analysis": sweep_analysis,
            },
            "axis_ablation": {
                "branch_count": len(ablation_branches),
                "branches": ablation_branches,
                "analysis": ablation_analysis,
            },
            "global_checks": {
                "zero_perturbation_branches_reproduce_formal_endpoint_48": True,
                "material_ctrlrange_loss_count_gt_1e_8": int(
                    sum(step["residual_lost_above_1e_8_count"] for step in all_steps)
                ),
                "maximum_ctrlrange_loss": float(
                    max(step["residual_lost_max_abs"] for step in all_steps)
                ),
                "all_sources_restored_from_complete_state_and_rnn": True,
            },
            "interpretation_limits": [
                "This diagnostic changes one action coordinate or ablates residuals; it does not estimate a smooth gradient.",
                "Branches outside the formal +/-0.05 residual support are counterfactual only and cannot be used for task acceptance.",
                "Contact flags are coarse finger-object indicators, not complete contact geometry or force state.",
                "A monotonic response in this frozen local sweep does not by itself authorize a new action scale.",
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
        "status": "completed_read_only_counterfactual",
        "output": str((args.output / "report.json").resolve()),
        "sweep_branches": len(sweep_branches),
        "ablation_branches": len(ablation_branches),
        "global_checks": report["global_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
