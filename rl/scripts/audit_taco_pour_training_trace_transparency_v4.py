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
from video_to_spider.rl.residual_semantics import (
    control_target_residuals,
    residual_from_trace_step,
)


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
        default=ROOT / "runs/taco_pour_training_trace_transparency_v4",
    )
    parser.add_argument(
        "--frozen-3plus1-report", type=Path,
        default=ROOT / "runs/taco_pour_tail_curriculum_3plus1_v1/report.json",
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
    with np.load(trace["epochs"][0]["visits"]["path"], allow_pickle=False) as raw:
        required_action_fields = (
            "sampled_action_preclamp",
            "sampled_action_clamped",
            "actor_mu",
            "actor_sigma",
            "reference_ctrl",
            "requested_residual",
            "effective_residual_after_ctrlrange",
            "residual_lost_to_ctrlrange",
        )
        action_shapes = {
            name: list(raw[name].shape) for name in required_action_fields
        }
        decomposition_error = float(np.max(np.abs(
            raw["requested_residual"]
            - raw["effective_residual_after_ctrlrange"]
            - raw["residual_lost_to_ctrlrange"]
        )))
        decomposition_valid = bool(
            decomposition_error <= np.finfo(np.float64).eps
        )
        legacy_field_absent = "right_wrist_applied_residual" not in raw.files
        legacy_partial_fields_absent = all(
            name not in raw.files for name in (
                "right_wrist_sampled_action_preclamp",
                "right_wrist_sampled_action_clamped",
            )
        )
        sigma_positive = bool((raw["actor_sigma"] > 0.0).all())
        official_clamp_exact = bool(np.array_equal(
            raw["sampled_action_clamped"],
            np.clip(raw["sampled_action_preclamp"], -1.0, 1.0),
        ))
    trace_contract = {
        "schema": trace["schema"],
        "epoch_count": len(trace["epochs"]),
        "sample_count": sum(row["sample_count"] for row in trace["epochs"]),
        "source_endpoint_visit_counts": summary["source_endpoint_visit_counts"],
        "outcome_endpoint_visit_counts": summary["outcome_endpoint_visit_counts"],
        "manifest_sha256_matches": _sha256(Path(trace["path"])) == trace["sha256"],
        "full_action_array_shapes": action_shapes,
        "requested_equals_effective_plus_lost": decomposition_valid,
        "decomposition_max_abs_error": decomposition_error,
        "legacy_ambiguous_field_absent": legacy_field_absent,
        "legacy_partial_action_fields_absent": legacy_partial_fields_absent,
        "actor_sigma_strictly_positive": sigma_positive,
        "sampled_action_clamp_exact": official_clamp_exact,
        "actor_distribution_reused_from_official_rollout_result": True,
        "extra_actor_forward_for_logging": False,
    }
    expected_trace = (
        trace_contract["schema"] == "taco_ppo_training_visitation_v4"
        and trace_contract["epoch_count"] == 1
        and trace_contract["sample_count"] == 4
        and trace_contract["source_endpoint_visit_counts"]
        == {str(endpoint): 1 for endpoint in range(20, 24)}
        and trace_contract["outcome_endpoint_visit_counts"]
        == {str(endpoint): 1 for endpoint in range(21, 25)}
        and trace_contract["manifest_sha256_matches"]
        and all(shape == [4, 36] for shape in action_shapes.values())
        and decomposition_valid
        and legacy_field_absent
        and legacy_partial_fields_absent
        and sigma_positive
        and official_clamp_exact
    )

    frozen = json.loads(args.frozen_3plus1_report.read_text())
    chunks = frozen.get("chunks", [])
    if len(chunks) != 1:
        raise ValueError("frozen 3+1 report must contain exactly one chunk")
    rl_traces = [
        row for row in chunks[0]["validation_traces"] if row.get("mode") == "rl"
    ]
    if len(rl_traces) != 1 or len(rl_traces[0]["steps"]) != 33:
        raise ValueError("frozen 3+1 report must contain the 33-step CPU RL trace")
    frozen_steps = rl_traces[0]["steps"]
    reference_ctrl = np.asarray([
        np.asarray(row["commanded_ctrl"], np.float64)
        - residual_from_trace_step(row)
        for row in frozen_steps
    ])
    requested_ctrl = np.asarray([
        row["commanded_ctrl"] for row in frozen_steps
    ], np.float64)
    frozen_terms = control_target_residuals(
        env.env.model_cpu, reference_ctrl, requested_ctrl
    )
    frozen_requested = frozen_terms["requested_residual"]
    frozen_effective = frozen_terms["effective_residual_after_ctrlrange"]
    frozen_lost = frozen_terms["residual_lost_to_ctrlrange"]
    tolerance = 2e-7
    wrist_translation = np.asarray([0, 1, 2, 18, 19, 20])
    wrist_rotation = np.asarray([3, 4, 5, 21, 22, 23])
    fingers = np.asarray([*range(6, 18), *range(24, 36)])
    finger_truncated = np.abs(frozen_lost[:, fingers]) > tolerance
    finger_blocked = (
        (np.abs(frozen_requested[:, fingers]) > tolerance)
        & (np.abs(frozen_effective[:, fingers]) <= tolerance)
    )
    frozen_regression = {
        "source_report": {
            "path": str(args.frozen_3plus1_report.resolve()),
            "sha256": _sha256(args.frozen_3plus1_report),
        },
        "step_count": len(frozen_steps),
        "tolerance": tolerance,
        "wrist_translation_lost_component_count": int(
            (np.abs(frozen_lost[:, wrist_translation]) > tolerance).sum()
        ),
        "wrist_rotation_lost_component_count": int(
            (np.abs(frozen_lost[:, wrist_rotation]) > tolerance).sum()
        ),
        "finger_truncated_component_count": int(finger_truncated.sum()),
        "finger_fully_blocked_component_count": int(finger_blocked.sum()),
        "steps_with_any_finger_truncation": int(finger_truncated.any(axis=1).sum()),
        "decomposition_max_abs_error": float(np.max(np.abs(
            frozen_requested - frozen_effective - frozen_lost
        ))),
    }
    frozen_regression_passed = bool(
        frozen_regression["wrist_translation_lost_component_count"] == 0
        and frozen_regression["wrist_rotation_lost_component_count"] == 0
        and frozen_regression["finger_truncated_component_count"] == 62
        and frozen_regression["finger_fully_blocked_component_count"] == 57
        and frozen_regression["steps_with_any_finger_truncation"] == 28
        and frozen_regression["decomposition_max_abs_error"]
        <= np.finfo(np.float64).eps
    )
    passed = bool(
        all(initial_equal.values())
        and all(final_equal.values())
        and expected_trace
        and frozen_regression_passed
    )
    report = {
        "schema": "taco_pour_training_trace_transparency_v4",
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
            "extra_actor_forward_for_logging": False,
            "actor_sigma_semantics": "standard deviation = exp(logstd), not variance",
        },
        "inputs": {
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
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
        "frozen_3plus1_numeric_regression": frozen_regression,
        "training_trace": trace,
        "decision": {
            "logging_may_be_enabled_for_next_training": passed,
            "v4_logger_is_observational_only": passed,
            "actor_mu_sigma_capture_may_be_used_for_nonpromotable_attribution": passed,
            "legacy_v2_artifacts_modified": False,
            "new_action_scale_selected": False,
            "algorithm_training_authorized_by_this_audit": False,
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
        "frozen_3plus1_numeric_regression": frozen_regression,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
