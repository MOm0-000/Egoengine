#!/usr/bin/env python3
"""Build the two natural endpoint-60 Candidate-F curriculum boundaries."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
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
)


CONTRACT = ROOT / "configs/taco_pour_candidate_F_two_chunk_curriculum_v1.yaml"
OUTPUT = ROOT / "runs/taco_pour_candidate_F_two_chunk_curriculum_v1/tail_boundary_builder"


def write_snapshot(path: Path, state: dict[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("curriculum boundary serialization changed its payload")
    return {
        "path": str(path.resolve()),
        "artifact_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(compressed),
        "payload_roundtrip_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    if (
        contract.get("schema") != "taco_pour_candidate_F_two_chunk_curriculum_v1"
        or contract.get("status") != "authorized_fixed_two_chunk_curriculum_200k"
        or contract.get("paper_faithful") is not False
        or contract["curriculum"]["epochs_63_through_125_world_starts"]
        != [40, 40, 60, 60]
    ):
        raise ValueError("Candidate-F contract changed")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    inputs = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in inputs.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen Candidate-F input changed: {name}")

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
    from video_to_spider.rl.algorithmic_benchmark import validate_checkpoint_payload
    from video_to_spider.rl.algorithmic_training_v6 import (
        SupportAnchoredBoundedMeanPpoAgent,
        load_support_anchored_profile,
    )
    from video_to_spider.rl.curriculum_reset import (
        capture_physics_rnn_boundary,
        restore_physics_rnn_boundaries,
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
    residual, _ = load_residual_action_profile(inputs["action_profile"])
    distribution, _ = load_support_anchored_profile(
        inputs["bounded_mean_distribution_profile"]
    )
    _, initialization = load_accepted_initialization(
        inputs["initialization_report"], inputs["simulator_config"]
    )
    source_boundary = load_checkpoint(inputs["source_boundary_endpoint40"])
    if np.asarray(source_boundary["time_indices"]).tolist() != [40]:
        raise ValueError("Candidate-F boundary builder requires endpoint40")
    cpu_config = _load_ego_config(str(inputs["simulator_config"]), "cpu")
    reference = _load_reference(cpu_config.data_path, "cpu", expected_frequency=30)

    def make_world(seed: int) -> MJWPVectorEnv:
        world = MJWPVectorEnv(
            cpu_config,
            reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                # Candidate F trains with the same asymmetric actor/critic
                # observation contract as Candidate D.  The boundary must
                # therefore preserve both the actor observation and critic
                # privileged state, even though the deterministic teacher
                # rollout itself only queries the actor branch.
                asymmetric_critic=True,
                max_episode_length=len(reference[0]) - 1,
                tracked_object_indices=(0,),
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
                residual=residual,
            ),
            seed=seed,
        )
        verify_runtime_model(world.env.model_cpu, initialization["validated_physics_contract"])
        world.set_env_state(source_boundary)
        return world

    def make_agent(world: MJWPVectorEnv, seed: int, output: Path):
        config = replace(
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
        return SupportAnchoredBoundedMeanPpoAgent(
            experiment_dir=output / "agent",
            ppo_config=config,
            network_config=_build_network_config(4),
            env=world,
            distribution_spec=distribution,
            likelihood_identity_gate=LikelihoodIdentityGateSpec.float32_ulp_aware_v1(),
            canonical_old_policy_gate=CanonicalOldPolicyGateSpec.v1(),
            audit_dir=output / "unused_update_audit",
        )

    teacher_specs = (
        (0, inputs["tail_A_teacher_checkpoint"], inputs["tail_A_historical_validation"]),
        (1, inputs["tail_B_teacher_checkpoint"], inputs["tail_B_historical_validation"]),
    )
    rows = []
    for seed, checkpoint_path, historical_path in teacher_specs:
        teacher_dir = args.output_dir / f"teacher_seed_{seed}"
        teacher_dir.mkdir()
        checkpoint = load_checkpoint(checkpoint_path)
        validate_checkpoint_payload(checkpoint)
        if (
            checkpoint.get("candidate") != "D"
            or int(checkpoint.get("seed", -1)) != seed
            or int(checkpoint.get("simulation_physics_steps", -1)) != 200_000
        ):
            raise ValueError("tail teacher checkpoint lineage changed")
        world = make_world(seed)
        agent = make_agent(world, seed, teacher_dir)
        agent.model.load_state_dict(checkpoint["actor"], strict=True)
        agent.set_eval()
        agent.rnn_states = [
            state.to("cpu").zero_() for state in agent.model.get_default_rnn_state()
        ]
        agent.dones = torch.zeros(1, dtype=torch.uint8)
        agent.current_rewards = torch.zeros(1, 1, dtype=torch.float32)
        agent.current_shaped_rewards = torch.zeros(1, 1, dtype=torch.float32)
        agent.current_lengths = torch.zeros(1, dtype=torch.float32)
        world.set_env_state(source_boundary)
        prefix = []
        collected: dict[str, list[Any]] = {
            "endpoint": [], "terminated": [], "tracking_score": [],
            "position_error": [], "rotation_error": [], "ctrl": [],
            "qpos": [], "qvel": [], "contact_flags": [],
            "deterministic_action": [], "raw_location": [], "bounded_mu": [],
            "actor_sigma": [], "action_low": [], "action_high": [],
        }
        for source in range(40, 60):
            raw_observation = world.current_observation()
            prefix.append(deepcopy(raw_observation))
            obs = agent.obs_to_tensors(raw_observation)
            result = agent.get_deterministic_action_values(obs)
            agent.rnn_states = result["rnn_states"]
            action = agent.preprocess_actions(result["deterministic_actions"])
            _, _, _, info = world.step(action, auto_reset=False)
            collected["endpoint"].append(source + 1)
            collected["terminated"].append(bool(info["terminated"][0]))
            collected["tracking_score"].append(float(info["object_tracking_error"][0]))
            collected["position_error"].append(float(info["object_position_error"][0, 0]))
            collected["rotation_error"].append(float(info["object_rotation_error"][0, 0]))
            collected["ctrl"].append(world._last_ctrl[0].detach().cpu().numpy().copy())
            collected["qpos"].append(
                world._mjwp.get_qpos(cpu_config, world.env)[0].detach().cpu().numpy().copy()
            )
            collected["qvel"].append(
                world._mjwp.get_qvel(cpu_config, world.env)[0].detach().cpu().numpy().copy()
            )
            collected["contact_flags"].append(np.asarray(info["contact_flags"][0]).copy())
            collected["deterministic_action"].append(np.asarray(action[0]).copy())
            collected["raw_location"].append(result["raw_locations"][0].detach().cpu().numpy().copy())
            collected["bounded_mu"].append(result["mus"][0].detach().cpu().numpy().copy())
            collected["actor_sigma"].append(result["sigmas"][0].detach().cpu().numpy().copy())
            collected["action_low"].append(result["action_lows"][0].detach().cpu().numpy().copy())
            collected["action_high"].append(result["action_highs"][0].detach().cpu().numpy().copy())
        arrays = {
            name: np.asarray(values, dtype=np.int32) if name == "endpoint" else np.asarray(values)
            for name, values in collected.items()
        }
        arrays["actor_mu"] = arrays["bounded_mu"]
        with np.load(historical_path, allow_pickle=False) as historical:
            missing = sorted(set(arrays) - set(historical.files))
            if missing:
                raise ValueError(f"historical validation lacks fields: {missing}")
            exact_checks = {
                name: bool(arrays[name].tobytes() == historical[name][:20].tobytes())
                for name in arrays
            }
        if not all(exact_checks.values()):
            raise RuntimeError(f"teacher seed {seed} did not reproduce historical validation")
        if arrays["terminated"].any() or arrays["endpoint"].tolist() != list(range(41, 61)):
            raise RuntimeError("tail teacher did not naturally reach endpoint60 nonterminal")
        boundary = capture_physics_rnn_boundary(
            agent,
            world,
            rollout_start_endpoint=40,
            observation_prefix=prefix,
            provenance={
                "role": f"Candidate_F_tail_{'A' if seed == 0 else 'B'}",
                "teacher_candidate": "D",
                "teacher_seed": seed,
                "teacher_epoch": 125,
                "teacher_physics_steps": 200_000,
                "teacher_checkpoint": artifact(checkpoint_path),
                "historical_validation": artifact(historical_path),
                "endpoint41_60_historical_validation_bitwise_equal": True,
                "natural_simulator_rollout": True,
                "gt_or_reference_state_injected": False,
                "off_policy_curriculum_state": True,
                "direct_chunk_commit_allowed": False,
            },
        )
        # Freeze the curriculum episode end in the serialized boundary.  This
        # is simulator bookkeeping rather than GT/reference state injection,
        # and makes a later endpoint-60 restore exact without mutating the
        # saved physics payload during sampler preparation.
        boundary["physics_state"]["episode_lengths"] = np.array(
            [80], dtype=np.int32
        )
        boundary["provenance"]["curriculum_window_end_endpoint"] = 80
        boundary["provenance"]["episode_bookkeeping_frozen_at_capture"] = True
        if (
            boundary["reference_endpoint"] != 60
            or boundary["rollout_start_endpoint"] != 40
            or len(boundary["observation_prefix"]) != 20
        ):
            raise RuntimeError("captured tail boundary contract changed")
        boundary_path = args.output_dir / f"seed{seed}_endpoint60.pt.gz"
        saved = write_snapshot(boundary_path, boundary)
        restored_boundary = load_checkpoint(boundary_path)
        restore_world = make_world(seed + 10)
        restore_agent = make_agent(restore_world, seed + 10, teacher_dir / "restore")
        restore_agent.model.load_state_dict(checkpoint["actor"], strict=True)
        restore = restore_physics_rnn_boundaries(
            restore_agent, restore_world, [restored_boundary]
        )
        if restore["physics_bitwise_equal"] != [True] or not restore["observation_bitwise_equal"]:
            raise RuntimeError("tail boundary did not restore exactly")
        validation_path = args.output_dir / f"seed{seed}_endpoint41_60_validation.npz"
        np.savez_compressed(validation_path, **arrays)
        rows.append({
            "teacher_seed": seed,
            "teacher_epoch": 125,
            "teacher_checkpoint": artifact(checkpoint_path),
            "historical_validation": artifact(historical_path),
            "endpoint41_60_array_exact_checks": exact_checks,
            "all_endpoint41_60_arrays_bitwise_equal": all(exact_checks.values()),
            "nonterminal_through_endpoint60": True,
            "reference_endpoint": boundary["reference_endpoint"],
            "rollout_start_endpoint": boundary["rollout_start_endpoint"],
            "observation_prefix_length": len(boundary["observation_prefix"]),
            "boundary": saved,
            "boundary_restore": restore,
            "captured_validation": artifact(validation_path),
            "off_policy_curriculum_state": True,
            "direct_chunk_commit_allowed": False,
        })
        if agent.writer is not None:
            agent.writer.close()
        if restore_agent.writer is not None:
            restore_agent.writer.close()

    report = {
        "schema": "taco_pour_candidate_F_tail_boundary_builder_v1",
        "status": "passed_two_natural_endpoint60_boundaries",
        "paper_faithful": False,
        "simulation_role": "exact_revalidation_and_boundary_capture_only",
        "training_executed": False,
        "optimizer_steps": 0,
        "reference_or_GT_state_injection": False,
        "boundaries": rows,
        "sampler_gate_authorized": True,
        "candidate_F_training_authorized": False,
        "endpoint20_to40_committed": True,
        "endpoint40_to60_committed": False,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": artifact(report_path)}, indent=2))


if __name__ == "__main__":
    main()
