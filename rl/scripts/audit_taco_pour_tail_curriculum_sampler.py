#!/usr/bin/env python3
"""No-training gate for the frozen 3-anchor + 1-tail PPO sampler."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def project_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


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
        "--source-boundary", type=Path,
        default=(ROOT / "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1"
                 / "endpoint20_complete_boundary.pt.gz"),
    )
    parser.add_argument(
        "--paired-boundaries", type=Path,
        default=ROOT / "runs/taco_pour_tail_curriculum_reset_gate_v1/paired_boundaries.pt.gz",
    )
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_tail_curriculum_3plus1_v1.yaml",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import torch
    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        PpoAgent,
        _build_asymmetric_critic_config,
        _build_network_config,
        _build_ppo_config,
        _load_ego_config,
        _load_reference,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import _model_state_sha256, _snapshot_value_equal
    from video_to_spider.rl.curriculum_reset import restore_physics_rnn_boundaries

    contract_raw = args.contract.read_bytes()
    contract = yaml.safe_load(contract_raw)
    if contract.get("schema") != "taco_pour_tail_curriculum_3plus1_v1":
        raise ValueError("unsupported curriculum contract")
    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    residual, residual_report = load_residual_action_profile(args.action_profile)
    _, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )
    config = _load_ego_config(str(args.config), "cuda:0")
    reference = _load_reference(config.data_path, "cuda:0", expected_frequency=30)

    source_raw = args.source_boundary.read_bytes()
    source_decoded = gzip.decompress(source_raw)
    source = torch.load(
        io.BytesIO(source_decoded), map_location="cpu", weights_only=False
    )
    if int(np.asarray(source["time_indices"])[0]) != 20:
        raise ValueError("source boundary must be endpoint 20")
    paired_raw = args.paired_boundaries.read_bytes()
    paired = torch.load(
        io.BytesIO(gzip.decompress(paired_raw)), map_location="cpu", weights_only=False
    )
    tail_rows = [
        row for row in paired.get("boundaries", ())
        if int(row.get("reference_endpoint", -1)) == 46
    ]
    if paired.get("schema") != "taco_pour_tail_curriculum_boundaries_v1" or len(tail_rows) != 1:
        raise ValueError("paired artifact must contain exactly one endpoint-46 boundary")
    tail = tail_rows[0]

    def make_world():
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
        return world

    with tempfile.TemporaryDirectory(prefix="egoengine_tail_sampler_gate_") as temp:
        env = IndependentMJWPTrainingEnv([make_world() for _ in range(4)])
        env.set_chunk_reset(start=20, end=60)
        env.enable_tail_curriculum(
            tail,
            anchor_endpoint=20,
            tail_endpoint=46,
            window_end_endpoint=60,
        )
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        agent = PpoAgent(
            experiment_dir=Path(temp),
            ppo_config=_build_ppo_config(
                num_envs=4,
                horizon_length=40,
                seq_length=4,
                max_epochs=8,
                learning_rate=1e-4,
                device="cuda:0",
                asymmetric_critic=_build_asymmetric_critic_config(160),
            ),
            network_config=_build_network_config(4),
            env=env,
        )
        agent.init_tensors()
        agent.obs = agent.env_reset()
        agent.epoch_num = 1
        model_before = _model_state_sha256(agent.model.state_dict())
        env.set_train_info(0, agent)
        model_after = _model_state_sha256(agent.model.state_dict())
        first_audit = env.tail_curriculum_audit()
        starts = [int(world.time_indices[0]) for world in env.worlds]
        prepared_rnn = tuple(
            state.detach().cpu().clone()
            for state in env._curriculum_reset_rnn_states
        )
        reset_physics = tuple(world.get_env_state() for world in env.worlds)

        tail_reset_seen = False
        tail_reset_physics_equal = False
        tail_reset_rnn_equal = False
        steps = []
        for control_step in range(40):
            values = agent.get_action_values(agent.obs)
            agent.rnn_states = values["rnn_states"]
            actions = np.asarray(
                agent.preprocess_actions(values["mus"]), dtype=np.float32
            )
            obs, _, done, info = env.step(actions)
            agent.obs = agent.obs_to_tensors(obs)
            done_tensor = torch.as_tensor(done, device=agent.device)
            if bool(done_tensor.any()):
                indices = done_tensor.nonzero(as_tuple=False)
                agent.rnn_states = env.reset_rnn_states_after_done(
                    agent.rnn_states, indices
                )
            steps.append({
                "control_step": control_step,
                "source_endpoints": np.asarray(
                    info["source_reference_endpoint"]
                ).tolist(),
                "outcome_endpoints": np.asarray(
                    info["outcome_reference_endpoint"]
                ).tolist(),
                "done": np.asarray(done, bool).tolist(),
            })
            if bool(done[3]):
                tail_reset_seen = True
                tail_reset_physics_equal = _snapshot_value_equal(
                    env.worlds[3].get_env_state(), reset_physics[3]
                )
                tail_reset_rnn_equal = all(
                    torch.equal(
                        current[:, 3, :].detach().cpu(),
                        expected[:, 3, :],
                    )
                    for current, expected in zip(
                        agent.rnn_states, prepared_rnn, strict=True
                    )
                )
                break

        first_actor_hash = first_audit["epochs_prepared"][0]["actor_state_sha256"]
        first_tail_hidden = tuple(state[:, 3:4, :].clone() for state in prepared_rnn)
        named_parameters = dict(agent.model.named_parameters())
        recurrent_bias_name = "a2c_network.rnn.rnn.bias_ih_l0"
        if recurrent_bias_name not in named_parameters:
            raise ValueError("expected actor LSTM input bias is missing")
        first_parameter = named_parameters[recurrent_bias_name]
        with torch.no_grad():
            first_parameter.reshape(-1)[0] += 1e-2
        changed_hash = _model_state_sha256(agent.model.state_dict())
        stale_hidden_rejected = False
        try:
            restore_physics_rnn_boundaries(agent, env, [tail] * 4)
        except ValueError as error:
            stale_hidden_rejected = "different actor/normalization" in str(error)
        agent.epoch_num = 2
        env.set_train_info(160, agent)
        second_audit = env.tail_curriculum_audit()
        second_tail_hidden = tuple(
            state[:, 3:4, :].detach().cpu().clone()
            for state in env._curriculum_reset_rnn_states
        )
        second_actor_hash = second_audit["epochs_prepared"][1]["actor_state_sha256"]
        changed_hidden = any(
            not torch.equal(left, right)
            for left, right in zip(first_tail_hidden, second_tail_hidden, strict=True)
        )
        agent.writer.close()

    normalization_passed = bool(
        model_before == model_after
        and first_audit["epochs_prepared"][0][
            "normalization_unchanged_during_prefix_replay"
        ] is True
        and second_audit["epochs_prepared"][1][
            "normalization_unchanged_during_prefix_replay"
        ] is True
    )
    episode_reset_passed = bool(
        tail_reset_seen and tail_reset_physics_equal and tail_reset_rnn_equal
    )
    actor_refresh_passed = bool(
        changed_hash != first_actor_hash
        and second_actor_hash == changed_hash
        and stale_hidden_rejected
        and changed_hidden
    )
    passed = bool(
        starts == [20, 20, 20, 46]
        and normalization_passed
        and episode_reset_passed
        and actor_refresh_passed
        and len(first_audit["epochs_prepared"]) == 1
        and len(second_audit["epochs_prepared"]) == 2
    )
    report = {
        "schema": "taco_pour_tail_curriculum_sampler_gate_v1",
        "status": "passed" if passed else "failed",
        "PPO_training_executed": False,
        "optimizer_steps": 0,
        "frozen_world_start_endpoints": starts,
        "contract": {
            "path": project_path(args.contract), "sha256": sha256(contract_raw),
        },
        "source_boundary": {
            "path": project_path(args.source_boundary), "sha256": sha256(source_raw),
            "reference_endpoint": 20,
        },
        "paired_boundaries": {
            "path": project_path(args.paired_boundaries), "sha256": sha256(paired_raw),
            "selected_reference_endpoint": 46,
        },
        "normalization_immutability": {
            "model_state_sha256_before_prefix_replay": model_before,
            "model_state_sha256_after_prefix_replay": model_after,
            "passed": normalization_passed,
        },
        "episode_reset": {
            "tail_done_observed": tail_reset_seen,
            "tail_physics_returned_bitwise_to_endpoint_46": tail_reset_physics_equal,
            "tail_RNN_returned_bitwise_to_current_actor_memory": tail_reset_rnn_equal,
            "passed": episode_reset_passed,
        },
        "actor_update_refresh": {
            "first_actor_sha256": first_actor_hash,
            "modified_actor_sha256": changed_hash,
            "second_epoch_actor_sha256": second_actor_hash,
            "stale_hidden_rejected_after_actor_change": stale_hidden_rejected,
            "modified_parameter": recurrent_bias_name,
            "tail_hidden_changed_after_actor_change": changed_hidden,
            "passed": actor_refresh_passed,
        },
        "rollout_until_first_tail_reset": steps,
        "sampler_audit_after_second_prepare": second_audit,
        "semantic_limit": {
            "tail_physical_state_generated_by_current_actor": False,
            "tail_start_is_off_policy_curriculum": True,
            "CPU_acceptance_start_endpoint": 20,
            "CPU_acceptance_required_steps": 40,
            "curriculum_rollout_direct_commit_allowed": False,
        },
        "residual_action": residual_report,
    }
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "starts": starts,
        "normalization_immutability": normalization_passed,
        "episode_reset": episode_reset_passed,
        "actor_refresh": actor_refresh_passed,
        "PPO_training_executed": False,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
