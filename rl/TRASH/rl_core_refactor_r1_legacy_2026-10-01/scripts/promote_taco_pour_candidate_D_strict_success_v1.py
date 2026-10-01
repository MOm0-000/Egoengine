#!/usr/bin/env python3
"""Promote the pre-frozen Candidate-D strict-success milestone, fail closed."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external/human2sim2robot"),
    str(ROOT / "external/spider_compat"),
]

from run_taco_pour_algorithmic_candidate_C_v5 import (  # noqa: E402
    artifact,
    exact_equal,
    load_checkpoint,
    sha256,
    validation_summary,
)


CONTRACT = ROOT / "configs/taco_pour_candidate_D_strict_success_promotion_v1.yaml"
RUN_ROOT = ROOT / "runs/taco_pour_candidate_D_strict_success_promotion_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_snapshot(path: Path, state: dict[str, Any]) -> dict[str, object]:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    raw = buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(compressed)
    temporary.replace(path)
    restored = load_checkpoint(path)
    if not exact_equal(state, restored):
        raise RuntimeError("committed endpoint-40 snapshot serialization changed state")
    return {
        "path": str(path.resolve()),
        "artifact_sha256": _sha256_bytes(compressed),
        "uncompressed_sha256": _sha256_bytes(raw),
        "bytes": len(compressed),
        "payload_roundtrip_exact": True,
    }


def _array_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    if (
        contract.get("schema")
        != "taco_pour_candidate_D_strict_success_promotion_v1"
        or contract.get("status") != "authorized_exact_revalidation_then_chunk_commit"
        or contract.get("paper_faithful") is not False
        or contract.get("promotion", {}).get("candidate") != "D"
        or contract.get("promotion", {}).get("seed") != 2
        or contract.get("promotion", {}).get("epoch") != 125
        or contract.get("promotion", {}).get("source_endpoint") != 20
        or contract.get("promotion", {}).get("target_endpoint") != 40
        or contract.get("promotion", {}).get("lookahead_endpoint") != 60
        or contract.get("promotion", {}).get("commit_only_after_exact_revalidation")
        is not True
    ):
        raise ValueError("strict-success promotion contract changed")
    if args.run_root.resolve() != Path(contract["output_directory"]).resolve():
        raise ValueError("promotion output directory changed")
    if args.run_root.exists():
        raise FileExistsError(args.run_root)
    args.run_root.mkdir(parents=True)

    inputs = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in inputs.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen promotion input changed: {name}")
    implementations = {
        "state_feasible_distribution_sha256": (
            ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py"
        ),
        "v6_algorithmic_training_sha256": (
            ROOT / "src/video_to_spider/rl/algorithmic_training_v6.py"
        ),
        "MJWP_environment_sha256": ROOT / "src/video_to_spider/rl/mjwp_env.py",
        "official_PPO_agent_sha256": (
            ROOT / "external/human2sim2robot/human2sim2robot/ppo/ppo_agent.py"
        ),
        "PPO_config_builder_sha256": ROOT / "scripts/run_mjwp_ppo.py",
        "v6_training_runner_sha256": (
            ROOT / "scripts/run_taco_pour_algorithmic_candidate_D_v6.py"
        ),
        "promotion_runner_sha256": Path(__file__).resolve(),
    }
    for name, path in implementations.items():
        if sha256(path) != contract["implementation_contract"][name]:
            raise ValueError(f"promotion implementation changed: {name}")

    checkpoint_bytes = inputs["checkpoint"].read_bytes()
    checkpoint_raw = gzip.decompress(checkpoint_bytes)
    if _sha256_bytes(checkpoint_raw) != contract["inputs"]["checkpoint"][
        "uncompressed_sha256"
    ]:
        raise ValueError("promotion checkpoint uncompressed hash changed")
    checkpoint = load_checkpoint(inputs["checkpoint"])
    expected_checkpoint = {
        "schema": "taco_pour_algorithmic_benchmark_checkpoint_v1",
        "candidate": "D",
        "seed": 2,
        "agent_epoch": 125,
        "simulation_physics_steps": 200000,
        "simulation_control_intervals": 20000,
        "chunk_commit_allowed": False,
    }
    for name, expected in expected_checkpoint.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"checkpoint field {name!r} changed")

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
    from video_to_spider.rl.algorithmic_training_v6 import (
        SupportAnchoredBoundedMeanPpoAgent,
        load_support_anchored_profile,
    )
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        CanonicalOldPolicyGateSpec,
        LikelihoodIdentityGateSpec,
    )

    objective = load_runtime_objective(
        inputs["protocol_at_authorization"],
        inputs["objective_profile"],
        tracking_variant="tool_only",
        require_run_ready=False,
    )
    observation = load_runtime_observation(
        inputs["protocol_at_authorization"],
        inputs["observation_profile"],
        require_run_ready=False,
    )
    residual, residual_report = load_residual_action_profile(inputs["action_profile"])
    distribution, distribution_report = load_support_anchored_profile(
        inputs["bounded_mean_distribution_profile"]
    )
    _, initialization = load_accepted_initialization(
        inputs["initialization_report"], inputs["simulator_config"]
    )
    boundary = load_checkpoint(inputs["source_boundary"])
    if (
        boundary.get("snapshot_schema")
        != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or boundary.get("mujoco_warp_version") != "3.13.0"
        or np.asarray(boundary["time_indices"]).tolist() != [20]
    ):
        raise ValueError("promotion requires the exact MJWP-3.13 endpoint-20 boundary")

    cpu_config = _load_ego_config(str(inputs["simulator_config"]), "cpu")
    cpu_reference = _load_reference(cpu_config.data_path, "cpu", expected_frequency=30)

    def make_world(seed: int) -> MJWPVectorEnv:
        world = MJWPVectorEnv(
            cpu_config,
            cpu_reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=False,
                max_episode_length=len(cpu_reference[0]) - 1,
                tracked_object_indices=(0,),
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
                residual=residual,
            ),
            seed=seed,
        )
        verify_runtime_model(world.env.model_cpu, initialization["validated_physics_contract"])
        world.set_env_state(boundary)
        return world

    world = make_world(2)
    ppo_config = replace(
        _build_ppo_config(
            num_envs=1,
            horizon_length=40,
            seq_length=4,
            max_epochs=1,
            learning_rate=1.0e-4,
            device="cpu",
            asymmetric_critic=None,
            actor_mini_epochs=1,
        ),
        clip_actions=False,
        bounds_loss_coef=0.0,
        bound_loss_type="regularisation",
        lr_schedule=None,
    )
    agent = SupportAnchoredBoundedMeanPpoAgent(
        experiment_dir=args.run_root / "validation_agent",
        ppo_config=ppo_config,
        network_config=_build_network_config(4),
        env=world,
        distribution_spec=distribution,
        likelihood_identity_gate=LikelihoodIdentityGateSpec.float32_ulp_aware_v1(),
        canonical_old_policy_gate=CanonicalOldPolicyGateSpec.v1(),
        audit_dir=args.run_root / "validation_audit_unused",
    )
    agent.model.load_state_dict(checkpoint["actor"], strict=True)
    agent.set_eval()
    agent.rnn_states = [
        state.to("cpu").zero_() for state in agent.model.get_default_rnn_state()
    ]
    world.set_env_state(boundary)

    rows: dict[str, list[Any]] = {
        "endpoint": [],
        "terminated": [],
        "tracking_score": [],
        "position_error": [],
        "rotation_error": [],
        "ctrl": [],
        "qpos": [],
        "qvel": [],
        "contact_flags": [],
        "deterministic_action": [],
        "raw_location": [],
        "bounded_mu": [],
        "actor_sigma": [],
        "action_low": [],
        "action_high": [],
    }
    endpoint40_state: dict[str, Any] | None = None
    try:
        for source in range(20, 60):
            obs = agent.obs_to_tensors(world.current_observation())
            result = agent.get_deterministic_action_values(obs)
            agent.rnn_states = result["rnn_states"]
            action = agent.preprocess_actions(result["deterministic_actions"])
            _, _, _, info = world.step(action, auto_reset=False)
            rows["endpoint"].append(source + 1)
            rows["terminated"].append(bool(info["terminated"][0]))
            rows["tracking_score"].append(float(info["object_tracking_error"][0]))
            rows["position_error"].append(float(info["object_position_error"][0, 0]))
            rows["rotation_error"].append(float(info["object_rotation_error"][0, 0]))
            rows["ctrl"].append(world._last_ctrl[0].detach().cpu().numpy().copy())
            qpos = world._mjwp.get_qpos(cpu_config, world.env)[0].detach().cpu().numpy().copy()
            rows["qpos"].append(qpos)
            rows["qvel"].append(
                world._mjwp.get_qvel(cpu_config, world.env)[0].detach().cpu().numpy().copy()
            )
            rows["contact_flags"].append(np.asarray(info["contact_flags"][0]).copy())
            rows["deterministic_action"].append(np.asarray(action[0]).copy())
            rows["raw_location"].append(result["raw_locations"][0].detach().cpu().numpy().copy())
            rows["bounded_mu"].append(result["mus"][0].detach().cpu().numpy().copy())
            rows["actor_sigma"].append(result["sigmas"][0].detach().cpu().numpy().copy())
            rows["action_low"].append(result["action_lows"][0].detach().cpu().numpy().copy())
            rows["action_high"].append(result["action_highs"][0].detach().cpu().numpy().copy())
            if source + 1 == 40:
                endpoint40_state = world.get_env_state()
    finally:
        if agent.writer is not None:
            agent.writer.close()

    arrays = {
        name: np.asarray(values, dtype=np.int32) if name == "endpoint" else np.asarray(values)
        for name, values in rows.items()
    }
    arrays["actor_mu"] = arrays["bounded_mu"]
    arrays["object_pose"] = arrays["qpos"][:, -14:].copy()
    historical = np.load(inputs["historical_validation"], allow_pickle=False)
    exact_checks = {
        name: _array_exact(arrays[name], historical[name])
        for name in historical.files
    }
    exact_checks["object_pose_derived_from_exact_qpos"] = bool(
        exact_checks["qpos"]
        and _array_exact(arrays["object_pose"], historical["qpos"][:, -14:])
    )
    all_exact = all(exact_checks.values())
    summary = validation_summary(arrays)
    bounded_outside = int(np.count_nonzero(
        (arrays["bounded_mu"] < arrays["action_low"])
        | (arrays["bounded_mu"] > arrays["action_high"])
    ))
    strict_success = bool(
        summary["successful_intervals"] == 40
        and summary["first_failure_endpoint"] is None
        and bounded_outside == 0
    )
    if not all_exact or not strict_success or endpoint40_state is None:
        failed_path = args.run_root / "failed_revalidation.npz"
        np.savez_compressed(failed_path, **arrays)
        failure = {
            "schema": "taco_pour_candidate_D_strict_success_promotion_failure_v1",
            "status": "failed_closed_no_chunk_commit",
            "exact_revalidation": all_exact,
            "strict_40_of_40": strict_success,
            "array_exact_checks": exact_checks,
            "summary": summary,
            "bounded_mu_outside_support_count": bounded_outside,
            "diagnostic_validation": artifact(failed_path),
            "chunk_commit_written": False,
            "retraining_executed": False,
        }
        (args.run_root / "failure_report.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise RuntimeError("Candidate-D milestone did not reproduce exactly; no chunk committed")

    promoted_path = args.run_root / "promoted_validation.npz"
    np.savez_compressed(promoted_path, **arrays)
    committed_arrays = {
        "source_endpoint": np.arange(20, 40, dtype=np.int32),
        **{name: value[:20].copy() for name, value in arrays.items()},
    }
    committed_path = args.run_root / "committed_chunk_20_40.npz"
    np.savez_compressed(committed_path, **committed_arrays)
    boundary_artifact = _write_snapshot(
        args.run_root / "committed_boundary_endpoint_40.pt.gz", endpoint40_state
    )

    restore_world = make_world(2)
    restore_world.set_env_state(load_checkpoint(args.run_root / "committed_boundary_endpoint_40.pt.gz"))
    restored_state = restore_world.get_env_state()
    boundary_restore_exact = exact_equal(endpoint40_state, restored_state)
    if not boundary_restore_exact:
        raise RuntimeError("endpoint-40 snapshot did not restore bitwise exactly")

    endpoint_scores = {
        str(endpoint): float(arrays["tracking_score"][endpoint - 21])
        for endpoint in (40, 50, 60)
    }
    report = {
        "schema": "taco_pour_candidate_D_strict_success_promotion_report_v1",
        "status": "promoted_exact_40_of_40_chunk_20_40_committed",
        "paper_faithful": False,
        "classification": "local_algorithmic_chunk_generation",
        "paper_cost_comparison_eligible": False,
        "promotion_reason": "prefrozen_D5_fixed_milestone_strict_40_of_40",
        "candidate": "D",
        "seed": 2,
        "epoch": 125,
        "source_endpoint": 20,
        "target_endpoint": 40,
        "lookahead_endpoint": 60,
        "lookahead_validated": True,
        "lookahead_committed": False,
        "committed_intervals": 20,
        "committed_endpoints": [21, 40],
        "exact_revalidation": {
            "all_historical_arrays_bitwise_equal": True,
            "array_checks": exact_checks,
            "summary": summary,
            "endpoint_scores": endpoint_scores,
            "bounded_mu_outside_support_count": bounded_outside,
        },
        "boundary": {
            "incoming": artifact(inputs["source_boundary"]),
            "committed_endpoint_40": boundary_artifact,
            "restore_bitwise_exact": boundary_restore_exact,
            "mujoco_warp_version": endpoint40_state["mujoco_warp_version"],
            "state_field_count": len(endpoint40_state),
        },
        "provenance": {
            "promotion_contract": artifact(args.contract),
            "v6_contract": artifact(inputs["v6_contract"]),
            "bounded_mean_distribution_profile": artifact(
                inputs["bounded_mean_distribution_profile"]
            ),
            "checkpoint": {
                **artifact(inputs["checkpoint"]),
                "uncompressed_sha256": _sha256_bytes(checkpoint_raw),
            },
            "historical_validation": artifact(inputs["historical_validation"]),
        },
        "artifacts": {
            "promoted_validation": artifact(promoted_path),
            "committed_chunk_20_40": artifact(committed_path),
            "committed_boundary_endpoint_40": boundary_artifact,
        },
        "residual_action": residual_report,
        "action_distribution": distribution_report,
        "fresh_training_executed": False,
        "warm_start_executed": False,
        "chunk_commit_written": True,
    }
    report_path = args.run_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": artifact(report_path)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print("".join(traceback.format_exception(error)), file=sys.stderr)
        raise
