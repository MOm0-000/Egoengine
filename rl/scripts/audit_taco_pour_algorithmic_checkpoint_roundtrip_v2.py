#!/usr/bin/env python3
"""Real fresh-agent checkpoint roundtrip before benchmark-v2 training."""

from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

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
CONTRACT = ROOT / "configs/taco_pour_algorithmic_reproduction_training_v2.yaml"
OUTPUT = ROOT / "runs/taco_pour_algorithmic_checkpoint_roundtrip_v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left, right) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["schema"] != "taco_pour_algorithmic_reproduction_training_benchmark_v2":
        raise ValueError("checkpoint gate requires benchmark v2")
    b0_path = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v2/candidate_B/seed_0/report.json"
    b0 = json.loads(b0_path.read_text())
    if b0["status"] != "passed_training_may_start" or not all(b0["checks"].values()):
        raise ValueError("checkpoint gate requires the exact B0 identity pass")

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
    from video_to_spider.rl.algorithmic_benchmark import (
        build_checkpoint_payload,
        replay_preserving_initialization,
        validate_checkpoint_payload,
        write_checkpoint,
    )
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    paths = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen checkpoint-gate input changed: {name}")
    objective = load_runtime_objective(
        paths["protocol_at_authorization"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol_at_authorization"], paths["observation_profile"],
        require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    distribution, _ = load_truncated_gaussian_profile(paths["distribution_profile"])
    distribution = replace(distribution, optimizer_training_authorized=True)
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    boundary = torch.load(
        io.BytesIO(gzip.decompress(paths["source_boundary"].read_bytes())),
        map_location="cpu", weights_only=False,
    )
    config = _load_ego_config(str(paths["simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    env = MJWPVectorEnv(
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
    verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
    env.set_env_state(boundary)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    ppo = _build_ppo_config(
        num_envs=1,
        horizon_length=40,
        seq_length=4,
        max_epochs=1,
        learning_rate=1.0e-4,
        device="cpu",
        asymmetric_critic=_build_asymmetric_critic_config(40),
        actor_mini_epochs=1,
    )
    ppo = replace(ppo, clip_actions=False)
    temporary_agent = Path(tempfile.mkdtemp(prefix=".algorithmic_checkpoint_agent_", dir=ROOT / "runs"))
    temporary_output = Path(tempfile.mkdtemp(prefix=".algorithmic_checkpoint_gate_", dir=ROOT / "runs"))
    agent = StateFeasibleTruncatedGaussianPpoAgent(
        experiment_dir=temporary_agent,
        ppo_config=ppo,
        network_config=_build_network_config(4),
        env=env,
        distribution_spec=distribution,
    )
    try:
        replay_preserving_initialization(agent, sigma_multiplier=0.25)
        # A freshly constructed rl-games agent has model/optimizers but does not
        # allocate its rollout tensors or recurrent state until training starts.
        # The benchmark checkpoint must exercise the real initialized runtime
        # shape, not serialize constructor-time ``None`` placeholders.
        agent.init_tensors()
        agent.rnn_states = [
            state.to("cpu").zero_()
            for state in agent.model.get_default_rnn_state()
        ]
        agent.obs = agent.obs_to_tensors(env.current_observation())
        payload = build_checkpoint_payload(
            agent,
            candidate="B",
            seed=0,
            simulation_physics_steps=0,
            simulation_control_intervals=0,
            config_hashes={
                "benchmark_contract": sha256(CONTRACT),
                "B0_report": sha256(b0_path),
            },
        )
        artifact_path = temporary_output / "fresh_agent_roundtrip.pt.gz"
        artifact = write_checkpoint(artifact_path, payload)
        restored = torch.load(
            io.BytesIO(gzip.decompress(artifact_path.read_bytes())),
            map_location="cpu", weights_only=False,
        )
        validate_checkpoint_payload(restored)
        fields = [
            "actor", "critic", "actor_optimizer", "critic_optimizer",
            "observation_normalization_version", "agent_rnn_states",
            "agent_observation", "agent_dones", "environment", "rng_states",
        ]
        field_equality = {name: equal(payload[name], restored[name]) for name in fields}
        if not all(field_equality.values()):
            raise RuntimeError(f"checkpoint roundtrip differs: {field_equality}")
        report = {
            "schema": "taco_pour_algorithmic_checkpoint_roundtrip_v2",
            "status": "passed_no_optimizer_step",
            "paper_faithful": False,
            "real_fresh_agent": True,
            "backend": "CPU_MuJoCo_Warp",
            "optimizer_updates": 0,
            "simulation_physics_steps": 0,
            "field_equality": field_equality,
            "payload_schema": restored["schema"],
            "normalization_version": restored["observation_normalization_version"],
            "environment_schema": restored["environment"]["snapshot_schema"],
            "chunk_commit_allowed": restored["chunk_commit_allowed"],
            "artifact": {
                **artifact,
                "path": str((OUTPUT / "fresh_agent_roundtrip.pt.gz").resolve()),
                "repository_retention": "local_only_not_versioned",
                "available_in_clean_checkout": False,
                "reproduction_script": (
                    "scripts/audit_taco_pour_algorithmic_checkpoint_roundtrip_v2.py"
                ),
                "eligible_for_warm_start": False,
                "purpose": "infrastructure_roundtrip_only",
            },
            "B0_report_sha256": sha256(b0_path),
            "benchmark_contract_sha256": sha256(CONTRACT),
        }
        (temporary_output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary_output.replace(OUTPUT)
    finally:
        if agent.writer is not None:
            agent.writer.close()
        shutil.rmtree(temporary_agent, ignore_errors=True)
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
