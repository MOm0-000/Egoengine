#!/usr/bin/env python3
"""CPU transparency gate for the corrected fresh-PPO credit logger."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "runs/taco_pour_floor_contact_v1/candidate_ppo_config.yaml",
    )
    parser.add_argument(
        "--initialization-report", type=Path,
        default=ROOT / "runs/taco_pour_initialization_protocol_v2/candidate_a/report.json",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml",
    )
    parser.add_argument(
        "--objective-profile", type=Path,
        default=ROOT / "configs/taco_pour_local_normalized_ellipse_v1.yaml",
    )
    parser.add_argument(
        "--observation-profile", type=Path,
        default=ROOT / "configs/taco_pour_observation_local_236d_v1.yaml",
    )
    parser.add_argument(
        "--action-profile", type=Path,
        default=ROOT / "configs/taco_pour_residual_action_scaled_v1.yaml",
    )
    parser.add_argument(
        "--distribution-profile", type=Path,
        default=ROOT / "configs/taco_pour_state_feasible_truncated_gaussian_v1.yaml",
    )
    parser.add_argument(
        "--source-boundary", type=Path,
        default=(ROOT / "runs/taco_pour_corrected_replay_rebase_v1/tool_only"
                 / "committed_boundary_endpoint_20.pt.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs/taco_pour_credit_instrumentation_gate_v1",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import torch
    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        _build_asymmetric_critic_config,
        _build_network_config,
        _build_ppo_config,
        _load_ego_config,
        _load_reference,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.credit_audit import (
        CreditInstrumentedTruncatedGaussianPpoAgent,
        model_state_sha256,
    )
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import _snapshot_value_equal
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(args.action_profile)
    base_spec, profile = load_truncated_gaussian_profile(args.distribution_profile)
    spec = replace(base_spec, optimizer_training_authorized=True)
    _, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )
    artifact = args.source_boundary.read_bytes()
    source = torch.load(
        io.BytesIO(gzip.decompress(artifact)), map_location="cpu", weights_only=False
    )
    if (
        source.get("snapshot_schema") != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or np.asarray(source["time_indices"]).tolist() != [20]
    ):
        raise ValueError("credit gate requires the corrected endpoint-20 boundary")

    def make_env():
        config = _load_ego_config(str(args.config), "cpu")
        reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
        worlds = []
        for _ in range(4):
            world = MJWPVectorEnv(
                config,
                reference,
                num_envs=1,
                env_config=MJWPVectorEnvConfig(
                    reference_start_index=0,
                    asymmetric_critic=True,
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
            worlds.append(world)
        env = IndependentMJWPTrainingEnv(worlds)
        env.set_chunk_reset(start=20, end=60)
        return env

    config = replace(
        _build_ppo_config(
            num_envs=4,
            horizon_length=4,
            seq_length=4,
            max_epochs=1,
            learning_rate=1e-4,
            device="cpu",
            asymmetric_critic=_build_asymmetric_critic_config(16),
        ),
        clip_actions=False,
        print_stats=False,
    )

    def run(label: str, instrumented: bool):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        env = make_env()
        cls = (
            CreditInstrumentedTruncatedGaussianPpoAgent
            if instrumented else StateFeasibleTruncatedGaussianPpoAgent
        )
        kwargs = {}
        if instrumented:
            kwargs["credit_audit_dir"] = args.output_dir / "credit_audit"
        agent = cls(
            experiment_dir=args.output_dir / label,
            ppo_config=config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=spec,
            **kwargs,
        )
        agent.train()
        credit = agent.finalize_credit_audit() if instrumented else None
        result = {
            "actor": copy.deepcopy(agent.model.state_dict()),
            "optimizer": copy.deepcopy(agent.optimizer.state_dict()),
            "critic": copy.deepcopy(agent.asymmetric_critic_net.state_dict()),
            "critic_optimizer": copy.deepcopy(
                agent.asymmetric_critic_net.optimizer.state_dict()
            ),
            "physics": env.get_env_state(),
            "rnn": [state.detach().clone() for state in agent.rnn_states],
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state().clone(),
            "actor_sha256": model_state_sha256(agent.model.state_dict()),
            "credit": credit,
        }
        agent.writer.close()
        return result

    baseline = run("baseline", False)
    audited = run("audited", True)
    fields = (
        "actor", "optimizer", "critic", "critic_optimizer", "physics", "rnn",
        "python_rng", "numpy_rng", "torch_rng",
    )
    comparisons = {
        name: _snapshot_value_equal(baseline[name], audited[name])
        for name in fields
    }
    manifest = json.loads(Path(audited["credit"]["path"]).read_text())
    patch = manifest["update_reports"][-1]["optimizer_state_patch"]
    checks = {
        "all_training_state_bitwise_equal": all(comparisons.values()),
        "actor_hash_equal": baseline["actor_sha256"] == audited["actor_sha256"],
        "one_rollout_logged": manifest["epochs"] == 1,
        "four_actor_updates_logged": manifest["actor_updates"] == 4,
        "lossless_patch_chain_hashes_match": (
            patch["after_model_state_sha256"]
            == manifest["final_actor"]["model_state_sha256"]
        ),
        "no_task_level_result_or_commit": True,
    }
    report = {
        "schema": "taco_pour_credit_instrumentation_gate_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "task_level_training_executed": False,
        "chunk_commit_written": False,
        "gate_training": {
            "backend": "CPU_MuJoCo_Warp",
            "worlds": 4,
            "horizon": 4,
            "epochs": 1,
            "samples": 16,
            "purpose": "logger transparency only",
        },
        "comparisons": comparisons,
        "checks": checks,
        "credit_audit": audited["credit"],
        "profile": profile,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in {
                "config": args.config,
                "initialization_report": args.initialization_report,
                "protocol": args.protocol,
                "objective_profile": args.objective_profile,
                "observation_profile": args.observation_profile,
                "action_profile": args.action_profile,
                "distribution_profile": args.distribution_profile,
                "source_boundary": args.source_boundary,
            }.items()
        },
        "implementation": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in {
                "credit_audit": ROOT / "src/video_to_spider/rl/credit_audit.py",
                "state_feasible_distribution": ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py",
                "ppo_chunk_adapter": ROOT / "src/video_to_spider/rl/replay_rl.py",
                "gate_script": Path(__file__).resolve(),
            }.items()
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
