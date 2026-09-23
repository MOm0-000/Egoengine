#!/usr/bin/env python3
"""Prove that PPO visitation logging does not change one deterministic CPU epoch."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
from video_to_spider.rl.action_contract import load_residual_action_profile
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation
from video_to_spider.rl.replay_rl import _snapshot_value_equal


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_all(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _clone(value):
    return copy.deepcopy(value)


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
        "--boundary", type=Path,
        default=ROOT / (
            "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1/"
            "endpoint20_complete_boundary.pt.gz"
        ),
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml"
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
        "--output-dir", type=Path,
        default=ROOT / "runs/taco_pour_training_trace_transparency_v1",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    from run_mjwp_ppo import (
        MJWPVectorEnv, MJWPVectorEnvConfig, PpoAgent,
        _build_asymmetric_critic_config, _build_network_config,
        _build_ppo_config, _load_ego_config, _load_reference, torch,
    )
    from video_to_spider.rl.replay_rl import MJWPChunkBackend

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    residual, residual_report = load_residual_action_profile(args.action_profile)
    _, initialization = load_accepted_initialization(args.initialization_report, args.config)
    config = _load_ego_config(str(args.config), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    env = MJWPVectorEnv(
        config, reference, num_envs=1,
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
    verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
    boundary_raw = gzip.decompress(args.boundary.read_bytes())
    boundary = torch.load(io.BytesIO(boundary_raw), map_location="cpu", weights_only=False)
    if int(boundary["time_indices"][0]) != 20:
        raise ValueError("transparency audit requires the frozen endpoint-20 boundary")
    backend = MJWPChunkBackend(env)

    def train_once(name: str, *, logging: bool) -> dict:
        backend.restore(boundary)
        backend.verify_restored_snapshot(boundary)
        env.set_chunk_reset(start=20, end=60)
        _seed_all(0, torch)
        run_dir = args.output_dir / name
        agent = PpoAgent(
            experiment_dir=run_dir,
            ppo_config=_build_ppo_config(
                num_envs=1,
                horizon_length=4,
                seq_length=4,
                max_epochs=1,
                learning_rate=1e-4,
                device="cpu",
                asymmetric_critic=_build_asymmetric_critic_config(4),
            ),
            network_config=_build_network_config(4),
            env=env,
        )
        initial = {
            "model": _clone(agent.model.state_dict()),
            "optimizer": _clone(agent.optimizer.state_dict()),
            "critic": _clone(agent.asymmetric_critic_net.state_dict()),
            "critic_optimizer": _clone(agent.asymmetric_critic_net.optimizer.state_dict()),
        }
        if logging:
            env.enable_training_trace(run_dir / "training_visitation")
        completed = False
        try:
            agent.train()
            completed = True
        finally:
            trace = env.finalize_training_trace(completed=completed) if logging else None
            if agent.writer is not None:
                agent.writer.close()
        final = {
            "model": _clone(agent.model.state_dict()),
            "optimizer": _clone(agent.optimizer.state_dict()),
            "critic": _clone(agent.asymmetric_critic_net.state_dict()),
            "critic_optimizer": _clone(agent.asymmetric_critic_net.optimizer.state_dict()),
            "env": _clone(env.get_env_state()),
            "rnn_states": _clone(agent.rnn_states),
            "torch_rng": torch.get_rng_state().clone(),
            "numpy_rng": _clone(np.random.get_state()),
            "python_rng": _clone(random.getstate()),
        }
        return {"initial": initial, "final": final, "trace": trace}

    without_logging = train_once("without_logging", logging=False)
    with_logging = train_once("with_logging", logging=True)
    initial_equal = {
        key: _snapshot_value_equal(without_logging["initial"][key], with_logging["initial"][key])
        for key in without_logging["initial"]
    }
    final_equal = {
        key: _snapshot_value_equal(without_logging["final"][key], with_logging["final"][key])
        for key in without_logging["final"]
    }
    trace = with_logging["trace"]
    summary = json.loads(Path(trace["epochs"][0]["summary"]["path"]).read_text())
    trace_contract = {
        "epoch_count": len(trace["epochs"]),
        "sample_count": sum(row["sample_count"] for row in trace["epochs"]),
        "source_endpoint_visit_counts": summary["source_endpoint_visit_counts"],
        "outcome_endpoint_visit_counts": summary["outcome_endpoint_visit_counts"],
        "manifest_sha256_matches": _sha256(Path(trace["path"])) == trace["sha256"],
    }
    expected_trace = (
        trace_contract["epoch_count"] == 1
        and trace_contract["sample_count"] == 4
        and trace_contract["source_endpoint_visit_counts"]
        == {str(endpoint): 1 for endpoint in range(20, 24)}
        and trace_contract["outcome_endpoint_visit_counts"]
        == {str(endpoint): 1 for endpoint in range(21, 25)}
        and trace_contract["manifest_sha256_matches"]
    )
    passed = all(initial_equal.values()) and all(final_equal.values()) and expected_trace
    report = {
        "schema": "taco_pour_training_trace_transparency_v1",
        "status": "logging_transparency_gate_passed" if passed else "logging_changed_training_behavior",
        "scope": {
            "purpose": "implementation equivalence audit, not an algorithm experiment",
            "device": "cpu",
            "epochs": 1,
            "horizon": 4,
            "seed": 0,
            "reference_start_endpoint": 20,
            "ppo_algorithm_changed": False,
            "reward_changed": False,
            "action_mapping_changed": False,
            "sampling_changed": False,
        },
        "inputs": {
            "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config)},
            "initialization_report": {
                "path": str(args.initialization_report.resolve()),
                "sha256": _sha256(args.initialization_report),
            },
            "boundary": {"path": str(args.boundary.resolve()), "sha256": _sha256(args.boundary)},
            "objective_profile": objective.as_report(),
            "observation_profile": observation.as_report(),
            "action_profile": residual_report,
        },
        "initial_state_bitwise_equal": initial_equal,
        "final_state_bitwise_equal": final_equal,
        "training_trace_contract": trace_contract,
        "training_trace": trace,
        "decision": {
            "logging_may_be_enabled_for_next_training": passed,
            "historical_8_epoch_training_coverage_remains_unknown": True,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "initial_state_bitwise_equal": initial_equal,
        "final_state_bitwise_equal": final_equal,
        "training_trace_contract": trace_contract,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
