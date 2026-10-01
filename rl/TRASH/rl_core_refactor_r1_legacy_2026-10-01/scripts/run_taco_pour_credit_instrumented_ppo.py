#!/usr/bin/env python3
"""Run one hash-bound fresh PPO diagnostic with complete credit evidence."""

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

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
POSTFIX_SCHEMA = "taco_pour_postfix_fresh_ppo_credit_instrumented_v1"
SINGLE_PASS_SCHEMA = "taco_pour_postfix_single_actor_pass_candidate_v1"
LEGACY_SCHEMA = "taco_pour_corrected_fresh_ppo_credit_instrumented_v1"
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def validate_window(backend, policy, *, mode: str, start: int, end: int) -> dict:
    backend.begin_trial(mode, start, end)
    validated = 0
    feasible = True
    error = None
    try:
        for source in range(start, end):
            if policy is None:
                action = np.zeros(
                    (1, backend.env.env_cfg.residual.hand_dof), dtype=np.float32
                )
            else:
                action = policy(backend, source)
            feasible = backend.step(action, source)
            validated += 1
            if not feasible:
                break
    except BaseException as exc:
        feasible = False
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        backend.end_trial(feasible, validated, error=error)
    return backend.validation_traces[-1]


def build_credit_analysis(credit_manifest: Path) -> dict:
    manifest = json.loads(credit_manifest.read_text())
    epoch_rows = []
    focus_rows = []
    update_rows = []
    for epoch_report in manifest["epoch_reports"]:
        epoch = int(epoch_report["epoch"])
        with np.load(epoch_report["credit_rows"]["path"], allow_pickle=False) as data:
            source = np.asarray(data["source_endpoint"]).reshape(-1)
            y = np.asarray(data["sampled_action"])[:, 1]
            raw_adv = np.asarray(data["raw_advantage"]).reshape(-1)
            norm_adv = np.asarray(data["normalized_advantage"]).reshape(-1)
            returns = np.asarray(data["return"]).reshape(-1)
            values = np.asarray(data["critic_value"]).reshape(-1)
            term = np.asarray(data["termination_outcome_endpoint"]).reshape(-1)
            selected = (source >= 43) & (source <= 46)
            positive = selected & (y > 0.5)
            nonpositive = selected & (y <= 0.0)
            sign_flip = selected & (raw_adv * norm_adv < 0.0)

            def summary(mask):
                if not mask.any():
                    return {"count": 0}
                return {
                    "count": int(mask.sum()),
                    "mean_action_y": float(y[mask].mean()),
                    "mean_return": float(returns[mask].mean()),
                    "mean_value": float(values[mask].mean()),
                    "mean_raw_advantage": float(raw_adv[mask].mean()),
                    "mean_normalized_advantage": float(norm_adv[mask].mean()),
                    "positive_raw_advantage_fraction": float((raw_adv[mask] > 0).mean()),
                    "termination_outcome_counts": {
                        str(item): int((term[mask] == item).sum())
                        for item in sorted(set(term[mask].tolist()))
                    },
                }

            row = {
                "epoch": epoch,
                "focus_source_43_46": summary(selected),
                "large_positive_y": summary(positive),
                "nonpositive_y": summary(nonpositive),
                "raw_to_normalized_advantage_sign_flip_count": int(sign_flip.sum()),
            }
            epoch_rows.append(row)
            for sample_id in np.flatnonzero(selected):
                focus_rows.append({
                    "epoch": epoch,
                    "sample_id": int(sample_id),
                    "source_endpoint": int(source[sample_id]),
                    "action_y": float(y[sample_id]),
                    "raw_advantage": float(raw_adv[sample_id]),
                    "normalized_advantage": float(norm_adv[sample_id]),
                    "return": float(returns[sample_id]),
                    "value": float(values[sample_id]),
                    "termination_outcome_endpoint": int(term[sample_id]),
                })

    first_saturation = None
    for update in manifest["update_reports"]:
        probe = update["probe"]
        labels = probe["labels"]
        mu_y = np.asarray(probe["mu_y"])
        selected = np.asarray([
            label.startswith("ppo_source_")
            and 43 <= int(label.rsplit("_", 1)[1]) <= 46
            for label in labels
        ])
        maximum = float(mu_y[selected].max())
        row = {
            "epoch": int(update["epoch"]),
            "update_in_epoch": int(update["update_in_epoch"]),
            "global_actor_update": int(update["global_actor_update"]),
            "probe_source_43_46_mu_y": {
                labels[index]: float(mu_y[index]) for index in np.flatnonzero(selected)
            },
            "maximum_probe_mu_y": maximum,
            "gradient_norm_before_clip": update["gradient_norm_before_clip"],
            "actor_loss": update["actor_loss"],
            "ratio": update["ratio"],
        }
        update_rows.append(row)
        if first_saturation is None and maximum > 1.0:
            first_saturation = row

    return {
        "schema": "taco_pour_fresh_ppo_credit_analysis_v1",
        "purpose": (
            "Locate harmful wrist-y mean formation without treating final-policy "
            "counterfactual returns as historical PPO advantages."
        ),
        "epoch_credit_summary": epoch_rows,
        "focus_samples": focus_rows,
        "actor_update_probe_trajectory": update_rows,
        "first_probe_mu_y_above_one": first_saturation,
        "interpretation_rule": {
            "A": "large +y receives favorable on-policy raw advantage",
            "B": "raw credit is materially altered by global advantage normalization",
            "C": "credit opposes +y but a specific PPO update raises fixed-probe mu_y",
            "D": "many small updates accumulate the bias without a single discontinuity",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_corrected_fresh_ppo_credit_instrumented_v1.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs/taco_pour_corrected_fresh_ppo_credit_instrumented_v1",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = yaml.safe_load(args.contract.read_text())
    schema = contract.get("schema")
    expected_status = (
        "authorized_single_fresh_diagnostic_training"
        if schema == SINGLE_PASS_SCHEMA
        else "authorized_diagnostic_training"
    )
    if (
        schema not in {LEGACY_SCHEMA, POSTFIX_SCHEMA, SINGLE_PASS_SCHEMA}
        or contract.get("status") != expected_status
        or contract.get("paper_faithful") is not False
        or contract.get("authorized_output_directory") != str(args.output_dir.resolve())
    ):
        raise ValueError("fresh credit-training contract is not authorized")
    training = contract["training"]
    if training != {
        "worlds": 4,
        "epochs": 8,
        "horizon_per_world_per_epoch": 40,
        "samples_per_epoch": 160,
        "total_samples": 1280,
        "seed": 0,
        "fixed_reset_endpoints": [20, 20, 20, 20],
        "old_actor_or_checkpoint_resume": False,
        "tail_curriculum": False,
    }:
        raise ValueError("fresh credit-training settings changed")

    paths = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
    gate = json.loads(paths["credit_instrumentation_gate"].read_text())
    if gate.get("status") != "passed" or not all(gate.get("checks", {}).values()):
        raise ValueError("credit instrumentation gate has not passed")
    if schema in {POSTFIX_SCHEMA, SINGLE_PASS_SCHEMA}:
        normalization_gate = json.loads(
            paths["observation_normalization_commit_gate"].read_text()
        )
        if (
            normalization_gate.get("schema")
            != "taco_pour_observation_normalization_commit_gate_v1"
            or normalization_gate.get("status") != "passed"
            or not all(normalization_gate.get("checks", {}).values())
            or normalization_gate.get("dry_run", {}).get("epochs") != 2
            or normalization_gate.get("dry_run", {}).get(
                "observation_stats_commit_enabled"
            ) is not True
        ):
            raise ValueError("commit-enabled normalization gate has not passed")
        eligibility = yaml.safe_load(paths["checkpoint_eligibility"].read_text())
        post_fix_checkpoint = eligibility.get("post_fix_checkpoint", {})
        if schema == POSTFIX_SCHEMA:
            if post_fix_checkpoint != {
                "exists": False,
                "fresh_training_required": True,
                "old_checkpoint_resume_allowed": False,
            }:
                raise ValueError("checkpoint eligibility no longer authorizes a fresh start")
        elif (
            post_fix_checkpoint.get("exists") is not True
            or post_fix_checkpoint.get("eligible_for_warm_start") is not False
            or post_fix_checkpoint.get("old_checkpoint_resume_allowed") is not False
        ):
            raise ValueError("single-pass candidate must not resume the old checkpoint")
    for name, row in contract["implementation"].items():
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"credit implementation changed after authorization: {name}")

    import torch
    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        _load_ego_config,
        _load_reference,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.credit_audit import load_fixed_probe_panel
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import (
        MJWPChunkBackend,
        MJWPIndependentTrainingBackend,
        train_chunk_ppo,
    )
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        load_truncated_gaussian_profile,
    )

    objective = load_runtime_objective(
        paths["protocol"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol"], paths["observation_profile"], require_run_ready=False,
    )
    residual, residual_report = load_residual_action_profile(paths["action_profile"])
    base_spec, distribution_report = load_truncated_gaussian_profile(
        paths["distribution_profile"]
    )
    spec = replace(base_spec, optimizer_training_authorized=True)
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    raw_boundary = paths["source_boundary"].read_bytes()
    source = torch.load(
        io.BytesIO(gzip.decompress(raw_boundary)), map_location="cpu", weights_only=False
    )
    if (
        source.get("snapshot_schema") != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or np.asarray(source["time_indices"]).tolist() != [20]
    ):
        raise ValueError("fresh credit run requires corrected endpoint-20 boundary")

    def make_world(device: str, *, asymmetric: bool):
        config = _load_ego_config(str(paths["simulator_config"]), device)
        reference = _load_reference(config.data_path, device, expected_frequency=30)
        world = MJWPVectorEnv(
            config,
            reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=asymmetric,
                max_episode_length=len(reference[0]) - 1,
                tracked_object_indices=(0,),
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
                residual=residual,
            ),
            seed=0,
        )
        verify_runtime_model(
            world.env.model_cpu, initialization["validated_physics_contract"]
        )
        world.set_env_state(source)
        return world

    args.output_dir.mkdir(parents=True)
    snapshots = args.output_dir / "input_contracts"
    snapshots.mkdir()
    for name, path in {"run_contract": args.contract, **paths}.items():
        destination = snapshots / f"{name}{''.join(path.suffixes) or '.bin'}"
        shutil.copy2(path, destination)

    training_env = IndependentMJWPTrainingEnv([
        make_world("cuda:0", asymmetric=True) for _ in range(4)
    ])
    training_backend = MJWPIndependentTrainingBackend(training_env)
    training_backend.verify_restored_snapshot(source)
    validation_env = make_world("cpu", asymmetric=False)
    validation_backend = MJWPChunkBackend(validation_env)
    validation_backend.verify_restored_snapshot(source)

    replay_trace = validate_window(
        validation_backend, None, mode="Replay", start=20, end=60
    )
    validation_backend.restore(source)
    validation_backend.verify_restored_snapshot(source)

    probe_panel = load_fixed_probe_panel(paths["fixed_probe_panel"])
    actor_mini_epochs = 1 if schema == SINGLE_PASS_SCHEMA else 4
    if schema == SINGLE_PASS_SCHEMA and contract.get("single_change") != {
        "field": "PPO_actor_mini_epochs",
        "baseline": 4,
        "candidate": 1,
        "actor_updates_per_training_epoch": 1,
        "total_actor_updates_over_8_epochs": 8,
    }:
        raise ValueError("single-pass candidate changed more than actor mini-epochs")
    policy = train_chunk_ppo(
        training_backend,
        20,
        60,
        args.output_dir / "ppo_chunk_20",
        epochs=8,
        horizon=40,
        seed=0,
        validation_env=validation_env,
        action_distribution_spec=spec,
        credit_audit_dir=args.output_dir / "credit_audit",
        credit_probe_panel=probe_panel,
        actor_mini_epochs=actor_mini_epochs,
    )
    validation_backend.restore(source)
    validation_backend.verify_restored_snapshot(source)
    try:
        ppo_trace = validate_window(
            validation_backend, policy, mode="PPO", start=20, end=60
        )
        training_audit = policy.audit
    finally:
        policy.close()

    credit_path = Path(training_audit["credit_audit"]["path"])
    analysis = build_credit_analysis(credit_path)
    analysis_path = args.output_dir / "credit_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    post_fix = schema in {POSTFIX_SCHEMA, SINGLE_PASS_SCHEMA}
    report = {
        "schema": contract["schema"],
        "status": (
            "completed_postfix_diagnostic_no_commit"
            if post_fix else "completed_diagnostic_not_promotable"
        ),
        "paper_faithful": False,
        "valid_postfix_algorithm_evidence": post_fix,
        "chunk_commit_written": False,
        "task_success_claimed": False,
        "performance_comparison_to_prior_gpu_run_allowed": False,
        "training": training,
        "actor_mini_epochs": actor_mini_epochs,
        "critic_mini_epochs": 4,
        "objective": objective.as_report(),
        "residual_action": residual_report,
        "action_distribution": distribution_report,
        "probe_panel": {
            "source_artifact": probe_panel["source_artifact"],
            "source_sha256": probe_panel["source_sha256"],
            "labels": probe_panel["labels"],
            "limitations": probe_panel["limitations"],
        },
        "replay_validation": replay_trace,
        "ppo_validation": ppo_trace,
        "training_audit": training_audit,
        "credit_analysis": artifact(analysis_path),
        "decision": (
            "This is valid post-fix algorithm evidence, but this diagnostic run "
            "commits no chunk and authorizes no checkpoint resume. CPU 40/40 "
            "remains required for any later formal commit."
            if post_fix else
            "This run restores historical credit evidence only. It neither commits "
            "a chunk nor authorizes checkpoint resume or a new algorithm change."
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "Replay": {
            "validated": replay_trace["validated_steps"],
            "failure": replay_trace["first_failure"],
        },
        "PPO": {
            "validated": ppo_trace["validated_steps"],
            "failure": ppo_trace["first_failure"],
        },
        "first_probe_mu_y_above_one": analysis["first_probe_mu_y_above_one"],
    }, indent=2))


if __name__ == "__main__":
    main()
