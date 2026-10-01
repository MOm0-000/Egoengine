#!/usr/bin/env python3
"""No-training integration gate for Candidate F's fixed 2+2 sampler."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import random
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
    load_checkpoint,
    sha256,
)


CONTRACT = ROOT / "configs/taco_pour_candidate_F_two_chunk_curriculum_v1.yaml"
OUTPUT = ROOT / "runs/taco_pour_candidate_F_two_chunk_curriculum_v1/sampler_gate"


def _normalization_hash(model: torch.nn.Module) -> str:
    payload = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if "running_mean_std" in name
    }
    if not payload:
        raise RuntimeError("actor has no input-normalization state")
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _states_equal(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> bool:
    return len(left) == len(right) and all(
        torch.equal(a.detach().cpu(), b.detach().cpu())
        for a, b in zip(left, right, strict=True)
    )


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
    boundary_root = args.output_dir.parent / "tail_boundary_builder"
    boundary_report_path = boundary_root / "report.json"
    boundary_report = json.loads(boundary_report_path.read_text())
    if (
        boundary_report.get("status") != "passed_two_natural_endpoint60_boundaries"
        or boundary_report.get("training_executed") is not False
        or boundary_report.get("optimizer_steps") != 0
    ):
        raise ValueError("tail-boundary builder gate is not valid")
    tail_boundaries = tuple(
        load_checkpoint(boundary_root / f"seed{seed}_endpoint60.pt.gz")
        for seed in (0, 1)
    )

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
    from video_to_spider.rl.algorithmic_benchmark import replay_preserving_initialization
    from video_to_spider.rl.algorithmic_training_v6 import (
        SupportAnchoredBoundedMeanPpoAgent,
        load_support_anchored_profile,
    )
    from video_to_spider.rl.curriculum_reset import restore_physics_rnn_boundaries
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import _model_state_sha256, _snapshot_value_equal
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
    cpu_config = _load_ego_config(str(inputs["simulator_config"]), "cpu")
    reference = _load_reference(cpu_config.data_path, "cpu", expected_frequency=30)

    def make_world(seed: int) -> MJWPVectorEnv:
        world = MJWPVectorEnv(
            cpu_config,
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
            seed=seed,
        )
        verify_runtime_model(
            world.env.model_cpu, initialization["validated_physics_contract"]
        )
        world.set_env_state(source_boundary)
        return world

    training_env = IndependentMJWPTrainingEnv([make_world(i) for i in range(4)])
    training_env.set_chunk_reset(start=40, end=80)
    training_env.enable_fixed_two_chunk_curriculum(
        tail_boundaries,
        anchor_endpoint=40,
        tail_endpoint=60,
        window_end_endpoint=80,
        activation_epoch=63,
    )
    ppo_config = replace(
        _build_ppo_config(
            num_envs=4,
            horizon_length=40,
            seq_length=4,
            max_epochs=1,
            learning_rate=1.0e-4,
            device="cpu",
            asymmetric_critic=_build_asymmetric_critic_config(160),
            actor_mini_epochs=1,
        ),
        clip_actions=False,
        bounds_loss_coef=0.0,
        bound_loss_type="regularisation",
        lr_schedule=None,
        schedule_type="legacy",
        kl_threshold=None,
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    agent = SupportAnchoredBoundedMeanPpoAgent(
        experiment_dir=args.output_dir / "agent",
        ppo_config=ppo_config,
        network_config=_build_network_config(4),
        env=training_env,
        distribution_spec=distribution,
        likelihood_identity_gate=LikelihoodIdentityGateSpec.float32_ulp_aware_v1(),
        canonical_old_policy_gate=CanonicalOldPolicyGateSpec.v1(),
        audit_dir=args.output_dir / "unused_update_audit",
    )
    initialization_audit = replay_preserving_initialization(
        agent, sigma_multiplier=0.25
    )

    report: dict[str, Any] = {
        "schema": "taco_pour_candidate_F_no_training_sampler_gate_v1",
        "paper_faithful": False,
        "training_executed": False,
        "optimizer_steps": 0,
        "tail_boundary_builder": artifact(boundary_report_path),
        "initialization": initialization_audit,
        "candidate_F_training_authorized": False,
        "endpoint20_to40_committed": True,
        "endpoint40_to60_committed": False,
    }
    try:
        agent.init_tensors()
        agent.obs = agent.obs_to_tensors(training_env.reset())
        actor_before = _model_state_sha256(agent.model.state_dict())
        rms_before = _normalization_hash(agent.model)

        stale_rejected = False
        stale_message = None
        try:
            restore_physics_rnn_boundaries(
                agent, training_env, [tail_boundaries[0]] * 4
            )
        except ValueError as error:
            stale_rejected = "different actor/normalization" in str(error)
            stale_message = str(error)
        if not stale_rejected:
            raise RuntimeError("stale teacher recurrent memory was not rejected")

        agent.epoch_num = 63
        training_env.set_train_info(0, agent)
        first_audit = deepcopy(training_env.fixed_two_chunk_curriculum_audit())
        actor_after_first = _model_state_sha256(agent.model.state_dict())
        rms_after_first = _normalization_hash(agent.model)
        first_states = training_env.curriculum_rnn_reset_states()
        first_world_states = training_env.get_env_states()
        anchor_expected = tuple(world._chunk_reset_state for world in training_env.worlds[:2])
        anchor_exact = [
            _snapshot_value_equal(expected, actual)
            for expected, actual in zip(anchor_expected, first_world_states[:2], strict=True)
        ]
        tail_exact = [
            _snapshot_value_equal(boundary["physics_state"], actual)
            for boundary, actual in zip(tail_boundaries, first_world_states[2:], strict=True)
        ]
        if (
            first_audit["epochs_prepared"][-1]["world_start_endpoints"]
            != [40, 40, 60, 60]
            or not all(anchor_exact)
            or not all(tail_exact)
            or actor_before != actor_after_first
            or rms_before != rms_after_first
        ):
            raise RuntimeError("first fixed-curriculum preparation gate failed")

        changed_parameter = None
        with torch.no_grad():
            for name, parameter in agent.model.named_parameters():
                if "rnn" in name.lower() and parameter.numel():
                    parameter.reshape(-1)[0].add_(1.0e-3)
                    changed_parameter = name
                    break
        if changed_parameter is None:
            raise RuntimeError("no recurrent actor parameter was available for refresh gate")
        actor_mutated = _model_state_sha256(agent.model.state_dict())
        rms_before_second = _normalization_hash(agent.model)
        if actor_mutated == actor_after_first:
            raise RuntimeError("actor mutation did not change the actor hash")
        agent.epoch_num = 64
        training_env.set_train_info(0, agent)
        second_audit = deepcopy(training_env.fixed_two_chunk_curriculum_audit())
        second_states = training_env.curriculum_rnn_reset_states()
        actor_after_second = _model_state_sha256(agent.model.state_dict())
        rms_after_second = _normalization_hash(agent.model)
        tail_hidden_changed = any(
            not torch.equal(before[:, 2:, :], after[:, 2:, :])
            for before, after in zip(first_states, second_states, strict=True)
        )
        if (
            actor_mutated != actor_after_second
            or rms_before_second != rms_after_second
            or not tail_hidden_changed
            or second_audit["epochs_prepared"][-1]["actor_state_sha256"]
            != actor_mutated
        ):
            raise RuntimeError("current-actor tail-memory refresh gate failed")

        expected_reset_physics = tuple(
            deepcopy(world._chunk_reset_state) for world in training_env.worlds
        )
        expected_reset_rnn = training_env.curriculum_rnn_reset_states()
        tail_done = np.zeros(2, dtype=bool)
        reset_physics_exact = [False, False]
        reset_rnn_exact = [False, False]
        reset_outcomes: list[list[int]] = [[], []]
        for _ in range(20):
            _, _, dones, info = training_env.step(
                np.zeros(
                    (4, training_env.env_cfg.residual.hand_dof), dtype=np.float32
                )
            )
            done_indices = torch.as_tensor(np.flatnonzero(dones), dtype=torch.long)
            working = [state + 1.0 for state in expected_reset_rnn]
            if len(done_indices):
                working = training_env.reset_rnn_states_after_done(
                    working, done_indices
                )
            for local, world_index in enumerate((2, 3)):
                reset_outcomes[local].append(int(info["outcome_reference_endpoint"][world_index]))
                if dones[world_index] and not tail_done[local]:
                    tail_done[local] = True
                    reset_physics_exact[local] = _snapshot_value_equal(
                        expected_reset_physics[world_index],
                        training_env.worlds[world_index].get_env_state(),
                    )
                    reset_rnn_exact[local] = all(
                        torch.equal(
                            state[:, world_index : world_index + 1, :],
                            expected[:, world_index : world_index + 1, :],
                        )
                        for state, expected in zip(
                            working, expected_reset_rnn, strict=True
                        )
                    )
            if tail_done.all():
                break
        if not tail_done.all() or not all(reset_physics_exact) or not all(reset_rnn_exact):
            raise RuntimeError("tail done/reset did not restore exact physics and RNN")

        report.update({
            "status": "passed_no_training_sampler_gate",
            "stale_teacher_memory": {
                "rejected": stale_rejected,
                "message": stale_message,
            },
            "first_prepare": {
                "epoch": 63,
                "world_starts": [40, 40, 60, 60],
                "anchor_physics_bitwise_exact": anchor_exact,
                "tail_physics_bitwise_exact": tail_exact,
                "actor_hash_before": actor_before,
                "actor_hash_after": actor_after_first,
                "RMS_hash_before": rms_before,
                "RMS_hash_after": rms_after_first,
                "actor_and_RMS_unchanged": (
                    actor_before == actor_after_first and rms_before == rms_after_first
                ),
                "audit": first_audit["epochs_prepared"][-1],
            },
            "current_actor_refresh": {
                "epoch": 64,
                "changed_parameter": changed_parameter,
                "actor_hash_after_mutation": actor_mutated,
                "actor_hash_after_refresh": actor_after_second,
                "RMS_hash_before": rms_before_second,
                "RMS_hash_after": rms_after_second,
                "tail_hidden_changed": tail_hidden_changed,
                "audit": second_audit["epochs_prepared"][-1],
            },
            "tail_done_reset": {
                "tail_done_observed": tail_done.tolist(),
                "physics_bitwise_exact": reset_physics_exact,
                "RNN_bitwise_exact": reset_rnn_exact,
                "outcome_endpoints_until_reset": reset_outcomes,
            },
            "candidate_F_training_authorized": True,
        })
    except BaseException as error:
        report.update({
            "status": "failed_closed_no_training_authorized",
            "exception": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            },
            "candidate_F_training_authorized": False,
        })
        path = args.output_dir / "report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        if agent.writer is not None:
            agent.writer.close()

    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": artifact(path)}, indent=2))


if __name__ == "__main__":
    main()
