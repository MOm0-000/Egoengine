"""Single entrypoint for inspecting, verifying, training and evaluating the active RL core."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

from .audit import (
    append_jsonl,
    append_rollout_npz,
    distribution as summarize_distribution,
    json_ready,
    module_sha256,
)
from .env import IndependentWorlds, make_world
from .policy import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_INPUT_DIM,
    CRITIC_INPUT_SPEC,
    PRIVILEGED_EXTRA_DIM,
    PolicyBundle,
    burn_in_prefix,
    policy_step,
)
from video_to_spider.rl.object_assistance import ToolAssistSpec
from .ppo import PPOConfig, PPOTrainer
from .plan_window import plan_window
from .rollout import FixedBoundaryCollector, valid_prefix
from .state_io import (
    load_boundary_context,
    load_torch_gzip,
    build_training_checkpoint,
    capture_rng_states,
    manifest_entry,
    restore_rng_states,
    restore_training_checkpoint,
    sha256,
    validate_physics_snapshot,
    validate_training_checkpoint,
    verify_artifact,
    write_json,
    write_torch_gzip_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/rl_core_refactor_r1.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs/taco_pour_rl_task_informed_critic_v2.yaml"
DEFAULT_PLAN_CONFIG = PROJECT_ROOT / "configs/taco_pour_two_chunk_sequence_search_v1.yaml"
DEFAULT_STARTUP_CONFIG = PROJECT_ROOT / "configs/taco_pour_control_aware_startup_v1.yaml"


def _load_config(path: Path, asset_root_override: Path | None) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema") != "egoengine_rl_core_refactor_r1":
        raise ValueError("unsupported core-refactor configuration")
    execution = config.get("execution", {})
    frozen = {
        "task": "TACO_20230927_017", "tracking_variant": "tool_only",
        "device": "cpu", "source_endpoint": 40, "final_endpoint": 80,
        "worlds": 4, "horizon": 40, "sequence_length": 4,
        "actor_observation_dim": 236, "critic_observation_dim": 108, "actions": 36,
        "training_enabled": False, "chunk_commit_enabled": False,
    }
    if execution != frozen:
        raise ValueError("R1 execution contract changed")
    root = (asset_root_override or Path(config["asset_root"])).resolve(strict=True)
    resolved: dict[str, Path] = {}
    for name, row in config["assets"].items():
        candidate = root / row["path"]
        expected = row.get("sha256")
        if expected is not None:
            resolved[name] = verify_artifact(candidate, expected)
        else:
            resolved[name] = candidate.resolve(strict=True)
    reconstruction = resolved["isolation_root"] / "reconstruction_report.json"
    expected = config["assets"]["isolation_root"]["reconstruction_report_sha256"]
    verify_artifact(reconstruction, expected)
    return config, root, resolved


def _load_train_config(
    path: Path, asset_root_override: Path | None
) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema") != "egoengine_taco_pour_rl_task_informed_critic_v2":
        raise ValueError("unsupported RL training configuration")
    execution = config.get("execution", {})
    expected = {
        "task": "TACO_20230927_017", "tracking_variant": "tool_only", "device": "cpu",
        "source_endpoint": 40, "final_endpoint": 80, "worlds": 4, "horizon": 40,
        "sequence_length": 4, "actor_observation_dim": ACTOR_OBSERVATION_DIM,
        "privileged_extra_dim": PRIVILEGED_EXTRA_DIM,
        "critic_input_dim": CRITIC_INPUT_DIM,
        "critic_input_spec": CRITIC_INPUT_SPEC,
        "actions": 36, "seed": 0,
        "physics_steps_per_control": 10, "training_enabled": True,
        "chunk_commit_enabled": False,
    }
    if execution != expected:
        raise ValueError("training execution contract changed")
    optimization = config.get("optimization_contract", {})
    frozen_optimization = {
        "actor_learning_rate": 1.0e-4, "actor_updates": 1,
        "critic_learning_rate": 5.0e-5, "critic_updates": 4,
        "gamma": 0.998, "gae_tau": 0.95, "ppo_clip": 0.2,
        "gradient_norm": 1.0, "internal_value_auxiliary_coefficient": 2.0,
        "residual_scale": 0.05,
        "observation_rms_commit": "after_all_optimizer_updates",
        "value_bootstrap_at_true_episode_end": False,
    }
    if optimization != frozen_optimization:
        raise ValueError("training optimization contract changed")
    budget = config.get("budget", {})
    if budget != {
        "epochs": 250, "control_intervals": 40000,
        "training_physics_steps": 400000, "actor_optimizer_steps": 250,
        "critic_optimizer_steps": 1000,
        "evaluation_epochs": [0, 62, 125, 188, 250],
        "maximum_evaluation_control_intervals": 240,
        "maximum_evaluation_physics_steps": 2400,
    }:
        raise ValueError("training budget changed")
    root = (asset_root_override or Path(config["asset_root"])).resolve(strict=True)
    resolved: dict[str, Path] = {}
    for name, row in config["assets"].items():
        candidate = (root / row["path"]).resolve(strict=True)
        if row.get("sha256") is not None:
            verify_artifact(candidate, row["sha256"])
        resolved[name] = candidate
    prior = json.loads(resolved["prior_anchor_verification"].read_text())
    if (
        prior.get("status") != "STRUCTURAL_REFACTOR_VERIFIED"
        or not prior.get("closed_loop_anchor_parity", {}).get("passed")
        or len(prior.get("closed_loop_anchor_parity", {}).get("anchors", ())) != 6
    ):
        raise ValueError("the reusable six-anchor verification is incomplete")
    return config, root, resolved


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _environment_report() -> dict[str, Any]:
    try:
        import mujoco
        import mujoco_warp
        import warp
        versions = {
            "mujoco": mujoco.__version__,
            "mujoco_warp": mujoco_warp.__version__,
            "warp": warp.__version__,
        }
    except Exception as error:  # inspect must still explain an incomplete environment
        versions = {"runtime_import_error": repr(error)}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "cuda_available": torch.cuda.is_available(),
        **versions,
    }


def inspect(config_path: Path, asset_root: Path | None) -> dict[str, Any]:
    config, root, assets = _load_config(config_path, asset_root)
    context = load_boundary_context(assets["context"])
    boundary = load_torch_gzip(assets["boundary"])
    validate_physics_snapshot(boundary)
    donor = load_torch_gzip(assets["donor"])
    if donor.get("schema") != "taco_pour_algorithmic_benchmark_checkpoint_v1":
        raise ValueError("donor checkpoint schema changed")
    result = {
        "status": "INSPECT_OK",
        "config": manifest_entry(config_path),
        "asset_root": str(root),
        "assets": {
            name: (
                manifest_entry(path)
                if path.is_file()
                else {
                    "path": str(path),
                    "kind": "directory",
                    "bound_by": manifest_entry(path / "reconstruction_report.json"),
                }
            )
            for name, path in assets.items()
        },
        "environment": _environment_report(),
        "boundary": {
            "snapshot_schema": boundary["snapshot_schema"],
            "warp_state_field_count": len(boundary["warp_state_keys"]),
            "context_schema": context["schema"],
            "context_prefix_observations": len(context["observation_prefix"]),
        },
        "execution": config["execution"],
    }
    return result


def _seed_paths(isolation_root: Path, seed: int) -> tuple[dict[str, Any], dict[str, Path]]:
    seed_root = isolation_root / f"seed_{seed}"
    report_path = seed_root / "seed_report.json"
    report = json.loads(report_path.read_text())
    paths = {
        "batch": Path(report["batch"]["path"]),
        "pre": Path(report["pre_update_state"]["path"]),
        "base": Path(report["shadow_state_artifacts"]["BASE"]["path"]),
        "full": Path(report["shadow_state_artifacts"]["FULL"]["path"]),
        "base_trajectory": Path(report["closed_loop"]["BASE"]["trajectory"]["path"]),
        "full_trajectory": Path(report["closed_loop"]["FULL"]["trajectory"]["path"]),
    }
    expected = {
        "batch": report["batch"]["sha256"],
        "pre": report["pre_update_state"]["sha256"],
        "base": report["shadow_state_artifacts"]["BASE"]["sha256"],
        "full": report["shadow_state_artifacts"]["FULL"]["sha256"],
        "base_trajectory": report["closed_loop"]["BASE"]["trajectory"]["sha256"],
        "full_trajectory": report["closed_loop"]["FULL"]["trajectory"]["sha256"],
    }
    for name, path in paths.items():
        verify_artifact(path, expected[name])
    return report, paths


def _offline_actor_parity(isolation_root: Path) -> tuple[list[dict[str, Any]], bool]:
    """Verify the immutable old-S1 evidence without re-running its retired loss.

    The active S1 auxiliary-MSE semantics intentionally cannot reproduce the
    historical FULL update.  Fixed actor execution remains covered separately
    by the six closed-loop anchors.
    """
    rows: list[dict[str, Any]] = []
    all_passed = True
    for seed in range(3):
        report, paths = _seed_paths(isolation_root, seed)
        exact = all(path.is_file() for path in paths.values())
        all_passed &= exact
        rows.append({
            "seed": seed,
            "actor_optimizer_steps": 0,
            "historical_report_sha256": sha256(isolation_root / f"seed_{seed}/seed_report.json"),
            "historical_full_bitwise_exact": bool(
                report.get("paired_update", {}).get("full_update_reproduced", True)
            ),
            "artifacts_hash_verified": exact,
            "active_loss_reexecution_skipped": True,
            "reason": "S1 auxiliary-MSE semantics intentionally differ from historical FULL",
        })
    return rows, all_passed


def _make_core_world(
    assets: dict[str, Path],
    boundary: dict[str, Any],
    seed: int,
    *,
    asymmetric: bool,
    object_assistance: ToolAssistSpec | None = None,
):
    return make_world(
        simulator_config=assets["simulator"],
        protocol=assets["protocol"],
        objective_profile=assets["objective"],
        observation_profile=assets["observation"],
        action_profile=assets["action"],
        boundary=boundary,
        seed=seed,
        asymmetric_critic=asymmetric,
        object_assistance=object_assistance,
    )


def _nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _make_training_runtime(
    assets: dict[str, Path], *, seed: int, checkpoint: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the one supported s40 training runtime without hidden callbacks."""
    _seed_everything(seed)
    context = load_boundary_context(assets["context"])
    donor = load_torch_gzip(assets["donor"])
    if donor.get("schema") != "taco_pour_algorithmic_benchmark_checkpoint_v1":
        raise ValueError("unsupported donor checkpoint")
    policy = PolicyBundle.create(worlds=4, with_critic=True)
    policy.load_actor_only(
        donor["actor"], version=int(donor["observation_normalization_version"])
    )
    worlds = [
        _make_core_world(assets, context["physics_state"], seed + index, asymmetric=True)
        for index in range(4)
    ]
    environment = IndependentWorlds(worlds)
    boundary_state = environment.states()[0]
    if np.asarray(boundary_state["episode_lengths"]).tolist() != [80]:
        raise RuntimeError("training boundary does not carry the endpoint-80 window horizon")
    for name in context["physics_state"]:
        if name == "episode_lengths":
            continue
        if name in boundary_state and not _nested_equal(context["physics_state"][name], boundary_state[name]):
            raise RuntimeError(f"set_chunk_reset changed physical boundary field {name}")
    prefix = tuple(context["observation_prefix"])
    collector = FixedBoundaryCollector(
        environment=environment,
        policy=policy,
        boundary_state=boundary_state,
        observation_prefix=prefix,
        horizon=40,
        start_endpoint=40,
    )
    trainer = PPOTrainer(policy=policy, collector=collector, config=PPOConfig())
    if checkpoint is not None:
        validate_training_checkpoint(checkpoint)
        if not _nested_equal(checkpoint["boundary_state"], boundary_state):
            raise ValueError("resume checkpoint boundary differs from the committed s40 runtime")
        if not _nested_equal(checkpoint["observation_prefix"], prefix):
            raise ValueError("resume checkpoint prefix differs from the immutable source20--39 prefix")
        restore_training_checkpoint(policy, checkpoint)
    return {
        "policy": policy,
        "environment": environment,
        "collector": collector,
        "trainer": trainer,
        "boundary_state": boundary_state,
        "observation_prefix": prefix,
    }


def _evaluation_arrays(
    assets: dict[str, Path],
    *,
    actor_state: dict[str, torch.Tensor],
    normalization_version: int,
    boundary_state: dict[str, Any],
    observation_prefix: tuple[Any, ...],
    seed: int,
    object_assistance: ToolAssistSpec | None = None,
    assistance_alpha: float = 0.0,
    zero_residual: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    policy = PolicyBundle.create(worlds=1, with_critic=False)
    policy.load_actor_only(actor_state, version=normalization_version)
    states = burn_in_prefix(policy.actor, observation_prefix)
    world = _make_core_world(
        assets,
        boundary_state,
        seed,
        asymmetric=False,
        object_assistance=object_assistance,
    )
    world.set_assistance_alpha(assistance_alpha)
    effective_boundary = world.get_env_state()
    world.set_env_state(effective_boundary)
    rows: dict[str, list[Any]] = {name: [] for name in (
        "source_endpoint", "outcome_endpoint", "terminated", "timeout", "tracking_score",
        "position_error", "rotation_error", "reward", "action", "mu", "sigma", "qpos",
        "qvel", "ctrl", "contact_flags",
        "assistance_alpha", "tool_assistance_wrench",
    )}
    endpoint60: dict[str, Any] | None = None
    for _ in range(40):
        observation = torch.as_tensor(world.current_observation(), dtype=torch.float32)
        low, high = world.current_normalized_action_bounds()
        action, output = policy_step(
            policy.actor, observation, states, low, high,
            stochastic=False, distribution=policy.distribution,
        )
        if zero_residual:
            action = torch.zeros_like(action)
        states = output.rnn_states
        _, reward, done, info = world.step(action, auto_reset=False)
        source = int(info["source_reference_endpoint"][0])
        outcome = int(info["outcome_reference_endpoint"][0])
        terminated = bool(info["terminated"][0])
        timeout = bool(info["time_outs"][0])
        rows["source_endpoint"].append(source)
        rows["outcome_endpoint"].append(outcome)
        rows["terminated"].append(terminated)
        rows["timeout"].append(timeout)
        rows["tracking_score"].append(float(info["object_tracking_error"][0]))
        rows["position_error"].append(float(info["object_position_error"][0, 0]))
        rows["rotation_error"].append(float(info["object_rotation_error"][0, 0]))
        rows["reward"].append(float(reward[0]))
        rows["action"].append(action[0].detach().cpu().numpy().copy())
        rows["mu"].append(output.mu[0].detach().cpu().numpy().copy())
        rows["sigma"].append(output.sigma[0].detach().cpu().numpy().copy())
        rows["qpos"].append(world._mjwp.get_qpos(world.ego_cfg, world.env)[0].detach().cpu().numpy().copy())
        rows["qvel"].append(world._mjwp.get_qvel(world.ego_cfg, world.env)[0].detach().cpu().numpy().copy())
        rows["ctrl"].append(world._last_ctrl[0].detach().cpu().numpy().copy())
        rows["contact_flags"].append(np.asarray(info["contact_flags"][0]).copy())
        rows["assistance_alpha"].append(float(info["assistance_alpha"][0]))
        rows["tool_assistance_wrench"].append(
            np.asarray(info["tool_assistance_wrench"][0]).copy()
        )
        if outcome == 60:
            endpoint60 = {
                "schema": "egoengine_endpoint60_promotion_candidate_v1",
                "reference_endpoint": 60,
                "physics_state": world.get_env_state(),
                "rnn_states": tuple(state.detach().cpu().clone() for state in states),
                "observation_normalization_version": int(normalization_version),
            }
        if terminated:
            break
        if bool(done[0]) and not timeout:
            raise RuntimeError("evaluation ended without termination or window timeout")
    arrays = {
        name: np.asarray(values, dtype=np.int64 if "endpoint" in name else None)
        for name, values in rows.items()
    }
    return arrays, endpoint60


def _evaluation_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    count = len(arrays["outcome_endpoint"])
    terminated = arrays["terminated"].astype(bool)
    first_failure = int(arrays["outcome_endpoint"][np.flatnonzero(terminated)[0]]) if terminated.any() else None
    valid = int(np.flatnonzero(terminated)[0]) if terminated.any() else count
    strict = bool(
        count == 40 and valid == 40 and not terminated.any()
        and bool(arrays["timeout"][-1]) and int(arrays["outcome_endpoint"][-1]) == 80
    )
    return {
        "strict_40_of_40": strict,
        "valid_prefix_intervals": valid,
        "first_failure_endpoint": first_failure,
        "executed_control_intervals": count,
        "executed_physics_steps": count * 10,
        "effective_prefix_tracking_reward": float(arrays["reward"][:valid].sum()),
        "final_outcome_endpoint": int(arrays["outcome_endpoint"][-1]) if count else None,
    }


def _evaluate_state(
    assets: dict[str, Path],
    *,
    actor_state: dict[str, torch.Tensor],
    normalization_version: int,
    boundary_state: dict[str, Any],
    observation_prefix: tuple[Any, ...],
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any] | None]:
    rng = {
        "python": random.getstate(), "numpy": deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_all": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }
    try:
        arrays, endpoint60 = _evaluation_arrays(
            assets,
            actor_state=actor_state,
            normalization_version=normalization_version,
            boundary_state=boundary_state,
            observation_prefix=observation_prefix,
            seed=seed,
        )
    finally:
        restore_rng_states(rng)
    return _evaluation_summary(arrays), arrays, endpoint60


def _closed_loop_arrays(
    assets: dict[str, Path],
    *,
    actor_state: dict[str, torch.Tensor],
    normalization_version: int,
    context: dict[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    policy = PolicyBundle.create(worlds=1, with_critic=False)
    policy.load_actor_only(actor_state, version=normalization_version)
    rnn_states = burn_in_prefix(policy.actor, context["observation_prefix"])
    world = _make_core_world(assets, context["physics_state"], seed, asymmetric=False)
    world.set_env_state(context["physics_state"])
    rows: dict[str, list[Any]] = {name: [] for name in (
        "endpoint", "terminated", "tracking_score", "position_error", "rotation_error",
        "ctrl", "qpos", "qvel", "contact_flags", "deterministic_action",
        "raw_location", "bounded_mu", "actor_sigma", "action_low", "action_high",
        "reward", "tracking_reward", "contact_bonus", "lift_reward",
    )}
    for source in range(40, 80):
        observation = torch.as_tensor(world.current_observation(), dtype=torch.float32)
        low, high = world.current_normalized_action_bounds()
        action, output = policy_step(
            policy.actor, observation, rnn_states, low, high,
            stochastic=False, distribution=policy.distribution,
        )
        rnn_states = output.rnn_states
        _, reward, _, info = world.step(action, auto_reset=False)
        rows["endpoint"].append(source + 1)
        rows["terminated"].append(bool(info["terminated"][0]))
        rows["tracking_score"].append(float(info["object_tracking_error"][0]))
        rows["position_error"].append(float(info["object_position_error"][0, 0]))
        rows["rotation_error"].append(float(info["object_rotation_error"][0, 0]))
        rows["ctrl"].append(world._last_ctrl[0].detach().cpu().numpy().copy())
        rows["qpos"].append(world._mjwp.get_qpos(world.ego_cfg, world.env)[0].detach().cpu().numpy().copy())
        rows["qvel"].append(world._mjwp.get_qvel(world.ego_cfg, world.env)[0].detach().cpu().numpy().copy())
        rows["contact_flags"].append(np.asarray(info["contact_flags"][0]).copy())
        rows["deterministic_action"].append(action[0].cpu().numpy().copy())
        rows["raw_location"].append(output.raw_location[0].cpu().numpy().copy())
        rows["bounded_mu"].append(output.mu[0].cpu().numpy().copy())
        rows["actor_sigma"].append(output.sigma[0].cpu().numpy().copy())
        rows["action_low"].append(low[0].cpu().numpy().copy())
        rows["action_high"].append(high[0].cpu().numpy().copy())
        rows["reward"].append(float(reward[0]))
        rows["tracking_reward"].append(float(info["aggregate_tracking_reward"][0]))
        rows["contact_bonus"].append(float(info["aggregate_contact_bonus"][0]))
        rows["lift_reward"].append(float(info["lift_reward"][0]))
    result = {
        name: np.asarray(values, dtype=np.int32) if name == "endpoint" else np.asarray(values)
        for name, values in rows.items()
    }
    result["actor_mu"] = result["bounded_mu"]
    return result


def _compare_trajectory(observed: dict[str, np.ndarray], path: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    maximum_error: dict[str, float] = {}
    with np.load(path, allow_pickle=False) as expected:
        for name, value in observed.items():
            checks[name] = name in expected and value.tobytes() == expected[name].tobytes()
            maximum_error[name] = (
                float(np.max(np.abs(value.astype(np.float64) - expected[name].astype(np.float64))))
                if name in expected and np.issubdtype(value.dtype, np.number)
                else (0.0 if checks[name] else float("inf"))
            )
    prefix = valid_prefix(observed["terminated"], np.zeros(len(observed["terminated"]), dtype=bool))
    return {
        "all_arrays_bitwise_equal": all(checks.values()),
        "per_array": checks,
        "maximum_abs_errors": maximum_error,
        "valid_prefix_intervals": prefix,
        "first_failure_endpoint": int(observed["endpoint"][prefix]) if prefix < len(observed["endpoint"]) else None,
    }


def _closed_loop_parity(assets: dict[str, Path]) -> tuple[list[dict[str, Any]], bool, int]:
    context = load_boundary_context(assets["context"])
    rows: list[dict[str, Any]] = []
    passed = True
    intervals = 0
    for seed in range(3):
        _, paths = _seed_paths(assets["isolation_root"], seed)
        pre = load_torch_gzip(paths["pre"])
        for branch, state_path, trajectory_path in (
            ("BASE", paths["base"], paths["base_trajectory"]),
            ("FULL", paths["full"], paths["full_trajectory"]),
        ):
            state = load_torch_gzip(state_path)
            arrays = _closed_loop_arrays(
                assets,
                actor_state=state["actor"],
                normalization_version=int(pre["observation_normalization_version"]),
                context=context,
                seed=seed,
            )
            comparison = _compare_trajectory(arrays, trajectory_path)
            comparison.update(seed=seed, branch=branch)
            rows.append(comparison)
            passed &= comparison["all_arrays_bitwise_equal"]
            intervals += 40
    return rows, passed, intervals


def _collector_smoke(assets: dict[str, Path]) -> tuple[dict[str, Any], int]:
    donor = load_torch_gzip(assets["donor"])
    context = load_boundary_context(assets["context"])
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    worlds = [
        _make_core_world(assets, context["physics_state"], seed, asymmetric=True)
        for seed in range(4)
    ]
    environment = IndependentWorlds(worlds)
    policy = PolicyBundle.create(worlds=4, with_critic=True)
    policy.load_actor_only(
        donor["actor"], version=int(donor["observation_normalization_version"])
    )
    boundary_state = environment.states()[0]
    collector = FixedBoundaryCollector(
        environment=environment,
        policy=policy,
        boundary_state=boundary_state,
        observation_prefix=context["observation_prefix"],
    )
    rows = []
    for epoch in range(2):
        batch = collector.collect()
        rows.append({
            "epoch": epoch + 1,
            "samples": int(batch.actions.shape[0]),
            "source_min": int(batch.source_endpoint.min()),
            "source_max": int(batch.source_endpoint.max()),
            "source40_visits": int((batch.source_endpoint == 40).sum()),
            "terminated_samples": int(batch.terminated.sum()),
            "timeout_samples": int(batch.timeout.sum()),
            "actor_sha256": module_sha256(policy.actor),
        })
        policy.actor.update_obs_stats(batch.observations)
        policy.critic.model.update_obs_stats(batch.critic_observations)
        policy.normalization_version += 1
    return {
        "passed": all(
            row["samples"] == 160 and row["source_min"] == 40
            and row["source40_visits"] >= 4 for row in rows
        ),
        "epochs": rows,
        "optimizer_steps": 0,
    }, 320


def _batch_exact(left: Any, right: Any) -> bool:
    return set(vars(left)) == set(vars(right)) and all(
        _nested_equal(getattr(left, name), getattr(right, name)) for name in vars(left)
    )


def _training_state(runtime: dict[str, Any]) -> dict[str, Any]:
    policy = runtime["policy"]
    return {
        "actor": deepcopy(policy.actor.state_dict()),
        "critic": deepcopy(policy.critic.state_dict()),
        "actor_optimizer": deepcopy(policy.actor_optimizer.state_dict()),
        "critic_optimizer": deepcopy(policy.critic.optimizer.state_dict()),
        "normalization_version": int(policy.normalization_version),
        "rng": capture_rng_states(),
    }


def verify_training_chain(
    config_path: Path, asset_root: Path | None, output: Path
) -> dict[str, Any]:
    """Bounded real two-epoch + cold-resume gate required before training."""
    started = time.time()
    config, root, assets = _load_train_config(config_path, asset_root)
    output.mkdir(parents=True, exist_ok=True)
    previous_report_path = output / "verification.json"
    previous_report = (
        json.loads(previous_report_path.read_text())
        if previous_report_path.is_file()
        else None
    )
    if previous_report is not None and previous_report.get("cost", {}).get(
        "retest_allowance_used", False
    ):
        raise RuntimeError("the single functional-verification retest was already used")
    continuous = _make_training_runtime(assets, seed=0)
    first = continuous["trainer"].run_epoch()
    _epoch_metrics(1, first)
    checkpoint = build_training_checkpoint(
        policy=continuous["policy"],
        boundary_state=continuous["boundary_state"],
        observation_prefix=continuous["observation_prefix"],
        next_epoch=2,
        cost={
            "training_control_intervals": 160,
            "training_physics_steps": 1600,
            "actor_optimizer_steps": 1,
            "critic_optimizer_steps": 4,
        },
        metadata={
            "config_sha256": sha256(config_path),
            "loss_semantics": "ppo_plus_2x_internal_value_auxiliary_mse",
            "verification_only": True,
        },
    )
    checkpoint_path = output / "epoch_0001_verification_checkpoint.pt.gz"
    write_torch_gzip_atomic(checkpoint_path, checkpoint)
    checkpoint = load_torch_gzip(checkpoint_path)
    validate_training_checkpoint(checkpoint)
    uninterrupted_second = continuous["trainer"].run_epoch()
    _epoch_metrics(2, uninterrupted_second)
    uninterrupted_state = _training_state(continuous)

    resumed = _make_training_runtime(assets, seed=0, checkpoint=checkpoint)
    resumed_second = resumed["trainer"].run_epoch()
    _epoch_metrics(2, resumed_second)
    resumed_state = _training_state(resumed)
    batch_exact = _batch_exact(uninterrupted_second["batch"], resumed_second["batch"])
    report_exact = _nested_equal(
        {name: value for name, value in uninterrupted_second.items() if name != "batch"},
        {name: value for name, value in resumed_second.items() if name != "batch"},
    )
    state_exact = _nested_equal(uninterrupted_state, resumed_state)
    first_live = first["actor"]["live_rollout_ratio_max_abs_error"]
    uninterrupted_live = uninterrupted_second["actor"]["live_rollout_ratio_max_abs_error"]
    resumed_live = resumed_second["actor"]["live_rollout_ratio_max_abs_error"]
    h40_changed = any(
        not torch.equal(a, b)
        for a, b in zip(first["batch"].reset_states, uninterrupted_second["batch"].reset_states, strict=True)
    )
    endpoint_truth = all(
        int(batch.source_endpoint.min()) == 40
        and torch.equal(batch.done_after, batch.terminated | batch.timeout)
        and bool(batch.timeout.any() | batch.terminated.any())
        for batch in (first["batch"], uninterrupted_second["batch"], resumed_second["batch"])
    )
    passed = bool(
        first_live <= 1.0e-4 and uninterrupted_live <= 1.0e-4 and resumed_live <= 1.0e-4
        and h40_changed and endpoint_truth and batch_exact and report_exact and state_exact
    )
    prior = manifest_entry(assets["prior_anchor_verification"])
    attempt_cost = {
        "reused_closed_loop_anchor_control_intervals": 240,
        "executed_training_control_intervals": 480,
        "executed_training_physics_steps": 4800,
        "actor_optimizer_steps": 3,
        "critic_optimizer_steps": 12,
    }
    previous_cost = previous_report.get("cost", {}) if previous_report is not None else {}
    cost = {
        name: int(attempt_cost[name]) + int(previous_cost.get(name, 0))
        for name in attempt_cost
    }
    cost["retest_allowance_used"] = previous_report is not None
    ceilings = config["functional_verification"]
    if (
        cost["executed_training_control_intervals"] > ceilings["maximum_control_intervals"]
        or cost["executed_training_physics_steps"] > ceilings["maximum_physics_steps"]
        or cost["actor_optimizer_steps"] > ceilings["maximum_actor_optimizer_steps"]
        or cost["critic_optimizer_steps"] > ceilings["maximum_critic_optimizer_steps"]
    ):
        raise RuntimeError("training-chain verification exceeded its declared ceiling")
    result = {
        "status": "TRAINING_CHAIN_VERIFIED" if passed else "TRAINING_CHAIN_VERIFICATION_FAILED",
        "training_authorized": passed,
        "chunk_commit_enabled": False,
        "config": manifest_entry(config_path),
        "asset_root": str(root),
        "reused_six_anchor_verification": prior,
        "live_likelihood_identity": [first_live, uninterrupted_live, resumed_live],
        "second_epoch_h40_rebuilt_from_current_actor_and_rms": h40_changed,
        "real_endpoint_and_done_contract": endpoint_truth,
        "cold_resume": {
            "batch_bitwise_equal": batch_exact,
            "loss_and_metrics_exact": report_exact,
            "models_optimizers_rms_and_rng_exact": state_exact,
        },
        "attempts": [
            *(
                previous_report.get("attempts", [{
                    "ordinal": 1,
                    "status": previous_report.get("status"),
                    "cost": previous_cost,
                }])
                if previous_report is not None
                else []
            ),
            {
                "ordinal": 2 if previous_report is not None else 1,
                "status": "TRAINING_CHAIN_VERIFIED" if passed else "TRAINING_CHAIN_VERIFICATION_FAILED",
                "cost": attempt_cost,
            },
        ],
        "cost": cost,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "verification.json", result)
    write_json(
        output / "input_manifest.json",
        {name: manifest_entry(path) for name, path in assets.items()},
    )
    write_json(output / "cost.json", cost)
    return result


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _fit_summary(values: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    if values.numel() == 0:
        return {"samples": 0, "mse": None, "explained_variance": None}
    values = values.to(torch.float64).reshape(-1)
    targets = targets.to(torch.float64).reshape(-1)
    residual_variance = torch.var(targets - values, unbiased=False)
    target_variance = torch.var(targets, unbiased=False)
    explained = (
        float(1.0 - residual_variance / target_variance)
        if float(target_variance) > 0.0
        else None
    )
    return {
        "samples": int(values.numel()),
        "mse": float(torch.mean((values - targets).square())),
        "explained_variance": explained,
    }


def _critic_diagnostics(batch: Any, *, gamma: float = 0.998) -> dict[str, Any]:
    ranges = {"sources_40_59": (40, 60), "sources_60_79": (60, 80)}
    gae_fit = {}
    for name, (low, high) in ranges.items():
        mask = (batch.source_endpoint >= low) & (batch.source_endpoint < high)
        gae_fit[name] = _fit_summary(batch.values[mask], batch.returns[mask])

    mc_targets = torch.full_like(batch.values, torch.nan)
    visible_terminal_episodes = 0
    right_truncated_episodes = 0
    episode_keys = torch.stack((batch.world_index, batch.episode_serial), dim=-1)
    for key in torch.unique(episode_keys, dim=0):
        mask = (episode_keys == key).all(dim=-1)
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        if not bool(batch.done_after[indices[-1]]):
            right_truncated_episodes += 1
            continue
        visible_terminal_episodes += 1
        running = torch.zeros((), dtype=batch.rewards.dtype)
        for index in reversed(indices.tolist()):
            running = batch.rewards[index] + gamma * running
            mc_targets[index] = running
    mc_fit = {}
    for name, (low, high) in ranges.items():
        mask = (
            (batch.source_endpoint >= low)
            & (batch.source_endpoint < high)
            & torch.isfinite(mc_targets[:, 0])
        )
        mc_fit[name] = _fit_summary(batch.values[mask], mc_targets[mask])

    source_targets = (60, 61, 64, 69, 74, 79)
    outcome_targets = (61, 65, 70, 75, 80)
    source_counts = {
        str(endpoint): int((batch.source_endpoint == endpoint).sum())
        for endpoint in source_targets
    }
    feasible_outcome_counts = {}
    feasible_outcome_episode_counts = {}
    for endpoint in outcome_targets:
        mask = (batch.outcome_endpoint == endpoint) & ~batch.terminated
        feasible_outcome_counts[str(endpoint)] = int(mask.sum())
        feasible_outcome_episode_counts[str(endpoint)] = int(
            torch.unique(episode_keys[mask], dim=0).shape[0]
        )
    return {
        "gae_return_fit": gae_fit,
        "visible_terminal_mc_return_fit": mc_fit,
        "visible_terminal_episodes": visible_terminal_episodes,
        "collector_right_truncated_episodes": right_truncated_episodes,
        "source_action_counts": source_counts,
        "feasible_outcome_counts": feasible_outcome_counts,
        "feasible_outcome_episode_counts": feasible_outcome_episode_counts,
    }


def _epoch_metrics(epoch: int, report: dict[str, Any]) -> dict[str, Any]:
    batch = report["batch"]
    row = {
        "epoch": int(epoch),
        "samples": int(report["samples"]),
        "training_enabled": bool(report["training_enabled"]),
        "chunk_commit_enabled": bool(report["chunk_commit_enabled"]),
        "normalization_version_after_commit": int(report["normalization_version_after_commit"]),
        "critic_losses": list(report["critic_losses"]),
        "actor": report["actor"],
        "raw_advantage": summarize_distribution(report["raw_advantage"]),
        "normalized_advantage": summarize_distribution(report["normalized_advantage"]),
        "reward": summarize_distribution(batch.rewards),
        "tracking_score": summarize_distribution(batch.tracking_score),
        "sigma": summarize_distribution(batch.sigma),
        "terminated_samples": int(batch.terminated.sum()),
        "timeout_samples": int(batch.timeout.sum()),
        "visited_source_endpoints": {
            str(int(endpoint)): int((batch.source_endpoint == endpoint).sum())
            for endpoint in torch.unique(batch.source_endpoint)
        },
        "critic_diagnostics": _critic_diagnostics(batch),
        "cost_after_epoch": {
            "training_control_intervals": epoch * 160,
            "training_physics_steps": epoch * 1600,
            "actor_optimizer_steps": epoch,
            "critic_optimizer_steps": epoch * 4,
        },
    }
    flat_scalars = [
        value for value in (
            *row["critic_losses"], row["actor"]["actor_loss"],
            row["actor"]["internal_value_loss"], row["actor"]["total_loss"],
        )
    ]
    if not all(np.isfinite(value) for value in flat_scalars):
        raise RuntimeError("epoch metrics contain non-finite optimizer values")
    return row


def _save_epoch_checkpoint(
    *,
    output: Path,
    runtime: dict[str, Any],
    epoch: int,
    config_path: Path,
    assets: dict[str, Path],
    milestone: bool,
) -> tuple[dict[str, Any], Path]:
    cost = {
        "training_control_intervals": epoch * 160,
        "training_physics_steps": epoch * 1600,
        "actor_optimizer_steps": epoch,
        "critic_optimizer_steps": epoch * 4,
    }
    payload = build_training_checkpoint(
        policy=runtime["policy"],
        boundary_state=runtime["boundary_state"],
        observation_prefix=runtime["observation_prefix"],
        next_epoch=epoch + 1,
        cost=cost,
        metadata={
            "config_sha256": sha256(config_path),
            "donor_sha256": sha256(assets["donor"]),
            "boundary_context_sha256": sha256(assets["context"]),
            "loss_semantics": "ppo_surrogate_plus_2x_internal_value_auxiliary_mse",
            "value_baseline": "independent_asymmetric_critic_raw_reward_units",
            "critic_input_spec": CRITIC_INPUT_SPEC,
            "critic_input_dimension": CRITIC_INPUT_DIM,
            "normalization_contract": "frozen_during_rollout_and_updates_commit_after_epoch",
            "chunk_commit_authorized": False,
        },
    )
    latest = output / "checkpoints/latest.pt.gz"
    write_torch_gzip_atomic(latest, payload)
    if milestone:
        write_torch_gzip_atomic(output / f"checkpoints/epoch_{epoch:04d}.pt.gz", payload)
    return payload, latest


def _record_evaluation(
    *,
    output: Path,
    epoch: int,
    assets: dict[str, Path],
    actor_state: dict[str, torch.Tensor],
    normalization_version: int,
    boundary_state: dict[str, Any],
    observation_prefix: tuple[Any, ...],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    summary, arrays, endpoint60 = _evaluate_state(
        assets,
        actor_state=actor_state,
        normalization_version=normalization_version,
        boundary_state=boundary_state,
        observation_prefix=observation_prefix,
        seed=seed,
    )
    summary.update(epoch=int(epoch), checkpoint_actor_sha256=module_sha256_from_state(actor_state))
    _write_npz_atomic(output / f"evaluations/epoch_{epoch:04d}_trajectory.npz", arrays)
    write_json(output / f"evaluations/epoch_{epoch:04d}.json", summary)
    return summary, endpoint60


def _compare_epoch0_to_v1(
    observed_path: Path, expected_path: Path
) -> dict[str, Any]:
    with np.load(observed_path, allow_pickle=False) as observed, np.load(
        expected_path, allow_pickle=False
    ) as expected:
        common = sorted(set(observed.files) & set(expected.files))
        if not common:
            raise ValueError("epoch-0 trajectories have no common arrays")
        exact = {
            name: bool(
                observed[name].shape == expected[name].shape
                and observed[name].dtype == expected[name].dtype
                and observed[name].tobytes() == expected[name].tobytes()
            )
            for name in common
        }
    return {
        "all_common_arrays_bitwise_equal": all(exact.values()),
        "common_array_count": len(common),
        "per_array": exact,
        "expected": manifest_entry(expected_path),
    }


def module_sha256_from_state(state: dict[str, torch.Tensor]) -> str:
    digest = __import__("hashlib").sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(
    config_path: Path,
    asset_root: Path | None,
    checkpoint_path: Path,
    output: Path,
) -> dict[str, Any]:
    config, _, assets = _load_train_config(config_path, asset_root)
    payload = load_torch_gzip(checkpoint_path)
    validate_training_checkpoint(payload)
    if payload["metadata"].get("config_sha256") != sha256(config_path):
        raise ValueError("checkpoint was created under a different training config")
    summary, arrays, endpoint60 = _evaluate_state(
        assets,
        actor_state=payload["actor"],
        normalization_version=int(payload["observation_normalization_version"]),
        boundary_state=payload["boundary_state"],
        observation_prefix=tuple(payload["observation_prefix"]),
        seed=int(config["execution"]["seed"]),
    )
    confirmation = None
    if summary["strict_40_of_40"]:
        second, second_arrays, _ = _evaluate_state(
            assets,
            actor_state=payload["actor"],
            normalization_version=int(payload["observation_normalization_version"]),
            boundary_state=payload["boundary_state"],
            observation_prefix=tuple(payload["observation_prefix"]),
            seed=int(config["execution"]["seed"]),
        )
        exact = set(arrays) == set(second_arrays) and all(
            arrays[name].tobytes() == second_arrays[name].tobytes() for name in arrays
        )
        if not exact or not second["strict_40_of_40"]:
            raise RuntimeError("fresh CPU success confirmation did not exactly match")
        confirmation = {"fresh_cpu_exact_match": True, "summary": second}
    result = {
        "status": "STRICT_WINDOW_SUCCESS" if summary["strict_40_of_40"] else "STRICT_WINDOW_FAILED",
        "checkpoint": manifest_entry(checkpoint_path),
        "evaluation": summary,
        "confirmation": confirmation,
        "chunk_commit_enabled": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation.json", result)
    _write_npz_atomic(output / "evaluation_trajectory.npz", arrays)
    if summary["strict_40_of_40"] and endpoint60 is not None:
        write_torch_gzip_atomic(output / "endpoint60_candidate.pt.gz", endpoint60)
    return result


def train(
    config_path: Path,
    asset_root: Path | None,
    output: Path,
    *,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    config, root, assets = _load_train_config(config_path, asset_root)
    gate_path = root / config["functional_verification"]["report"]
    if not gate_path.is_file():
        raise RuntimeError("training-chain verification report is missing; run verify first")
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != config["functional_verification"]["required_status"]:
        raise RuntimeError("training-chain verification did not authorize training")
    if gate.get("config", {}).get("sha256") != sha256(config_path):
        raise RuntimeError("training-chain verification used a different config")
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoints/latest.pt.gz"
    if resume_path is None and latest.exists():
        raise FileExistsError("training output already contains a checkpoint; pass --resume explicitly")
    checkpoint = load_torch_gzip(resume_path) if resume_path is not None else None
    if checkpoint is not None:
        validate_training_checkpoint(checkpoint)
        if checkpoint["metadata"].get("config_sha256") != sha256(config_path):
            raise ValueError("resume checkpoint was created under a different training config")
    runtime = _make_training_runtime(assets, seed=0, checkpoint=checkpoint)
    start_epoch = int(checkpoint["next_epoch"]) if checkpoint is not None else 1
    if start_epoch > 250:
        raise ValueError("checkpoint has already exhausted the declared budget")
    write_json(
        output / "input_manifest.json",
        {name: manifest_entry(path) for name, path in assets.items()},
    )
    write_json(output / "frozen_config.json", config)
    evaluations: list[dict[str, Any]] = []
    evaluation_intervals = 0
    success_endpoint60 = None
    if checkpoint is None:
        _save_epoch_checkpoint(
            output=output,
            runtime=runtime,
            epoch=0,
            config_path=config_path,
            assets=assets,
            milestone=True,
        )
        initial, _ = _record_evaluation(
            output=output, epoch=0, assets=assets,
            actor_state=deepcopy(runtime["policy"].actor.state_dict()),
            normalization_version=runtime["policy"].normalization_version,
            boundary_state=runtime["boundary_state"],
            observation_prefix=runtime["observation_prefix"], seed=0,
        )
        evaluations.append(initial)
        evaluation_intervals += initial["executed_control_intervals"]
        epoch0_parity = _compare_epoch0_to_v1(
            output / "evaluations/epoch_0000_trajectory.npz",
            assets["v1_epoch0_trajectory"],
        )
        write_json(output / "epoch0_v1_parity.json", epoch0_parity)
        if (
            not epoch0_parity["all_common_arrays_bitwise_equal"]
            or initial["valid_prefix_intervals"] != 20
            or initial["first_failure_endpoint"] != 61
        ):
            write_json(
                output / "decision.json",
                {
                    "status": "FUNCTIONAL_ERROR_BEFORE_TRAINING",
                    "completed_epoch": 0,
                    "epoch0_v1_parity": epoch0_parity,
                    "evaluation": initial,
                    "chunk_commit_enabled": False,
                },
            )
            raise RuntimeError("task-informed critic epoch-0 actor/physics parity failed")
    status = "RUNNING"
    completed_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, 251):
            epoch_report = runtime["trainer"].run_epoch()
            metrics = _epoch_metrics(epoch, epoch_report)
            append_jsonl(output / "training_metrics.jsonl", metrics)
            append_rollout_npz(
                output / "training_batches.npz", epoch=epoch, batch=epoch_report["batch"]
            )
            milestone = epoch in set(config["budget"]["evaluation_epochs"])
            payload, _ = _save_epoch_checkpoint(
                output=output, runtime=runtime, epoch=epoch, config_path=config_path,
                assets=assets, milestone=milestone,
            )
            completed_epoch = epoch
            if milestone:
                validation, endpoint60 = _record_evaluation(
                    output=output, epoch=epoch, assets=assets,
                    actor_state=payload["actor"],
                    normalization_version=int(payload["observation_normalization_version"]),
                    boundary_state=payload["boundary_state"],
                    observation_prefix=tuple(payload["observation_prefix"]), seed=0,
                )
                evaluations.append(validation)
                evaluation_intervals += validation["executed_control_intervals"]
                if validation["strict_40_of_40"]:
                    confirm, confirm_arrays, _ = _evaluate_state(
                        assets, actor_state=payload["actor"],
                        normalization_version=int(payload["observation_normalization_version"]),
                        boundary_state=payload["boundary_state"],
                        observation_prefix=tuple(payload["observation_prefix"]), seed=0,
                    )
                    first_arrays = dict(np.load(
                        output / f"evaluations/epoch_{epoch:04d}_trajectory.npz",
                        allow_pickle=False,
                    ))
                    exact = set(first_arrays) == set(confirm_arrays) and all(
                        first_arrays[name].tobytes() == confirm_arrays[name].tobytes()
                        for name in first_arrays
                    )
                    evaluation_intervals += confirm["executed_control_intervals"]
                    if not exact or not confirm["strict_40_of_40"]:
                        raise RuntimeError("fresh CPU strict-success confirmation diverged")
                    success_endpoint60 = endpoint60
                    status = "STRICT_WINDOW_SUCCESS_CONFIRMED"
                    break
        if status == "RUNNING":
            status = "COMPLETED_NO_STRICT_WINDOW_SUCCESS"
    except Exception as error:
        status = "FUNCTIONAL_ERROR_DURING_TRAINING"
        write_json(
            output / "decision.json",
            {
                "status": status, "completed_epoch": completed_epoch,
                "error": repr(error), "chunk_commit_enabled": False,
            },
        )
        raise
    if evaluation_intervals > config["budget"]["maximum_evaluation_control_intervals"]:
        raise RuntimeError("evaluation exceeded its declared control-interval budget")
    if success_endpoint60 is not None:
        write_torch_gzip_atomic(output / "endpoint60_candidate.pt.gz", success_endpoint60)
    cost = {
        "training_control_intervals": completed_epoch * 160,
        "training_physics_steps": completed_epoch * 1600,
        "actor_optimizer_steps": completed_epoch,
        "critic_optimizer_steps": completed_epoch * 4,
        "evaluation_control_intervals": evaluation_intervals,
        "evaluation_physics_steps": evaluation_intervals * 10,
    }
    decision = {
        "status": status,
        "completed_epoch": completed_epoch,
        "evaluations": evaluations,
        "cost": cost,
        "chunk_commit_enabled": False,
        "next_action": (
            "separate promotion authorization required"
            if status == "STRICT_WINDOW_SUCCESS_CONFIRMED"
            else "stop; no extra seed, extension, or parameter sweep authorized"
        ),
    }
    write_json(output / "decision.json", decision)
    write_json(output / "cost.json", cost)
    return decision


def verify(config_path: Path, asset_root: Path | None, output: Path, *, physics: bool) -> dict[str, Any]:
    started = time.time()
    config, root, assets = _load_config(config_path, asset_root)
    inspection = inspect(config_path, asset_root)
    offline, offline_passed = _offline_actor_parity(assets["isolation_root"])
    anchors: list[dict[str, Any]] = []
    anchors_passed = False
    collector: dict[str, Any] = {"passed": False, "not_run": True}
    intervals = 0
    if physics:
        anchors, anchors_passed, anchor_cost = _closed_loop_parity(assets)
        collector, collector_cost = _collector_smoke(assets)
        intervals = anchor_cost + collector_cost
    else:
        anchors = [{"not_run": True, "reason": "pass --physics for bounded MJWP verification"}]
    cost = {
        "actor_optimizer_steps": 0,
        "critic_optimizer_steps": 0,
        "control_intervals": intervals,
        "physics_steps": intervals * 10,
        "maximum_control_intervals": config["verification"]["maximum_control_intervals"],
        "maximum_physics_steps": config["verification"]["maximum_physics_steps"],
    }
    if cost["control_intervals"] > cost["maximum_control_intervals"] or cost["physics_steps"] > cost["maximum_physics_steps"]:
        raise RuntimeError("bounded verification exceeded its declared cost ceiling")
    complete = bool(physics and offline_passed and anchors_passed and collector["passed"])
    report = {
        "status": "STRUCTURAL_REFACTOR_VERIFIED" if complete else "INCOMPLETE_BOUNDED_VERIFICATION",
        "training_enabled": False,
        "chunk_commit_enabled": False,
        "candidate_G_classification_changed": False,
        "inspection": inspection,
        "structural_checks": {
            "candidate_agent_instantiated": False,
            "legacy_PpoAgent_training_orchestration_called": False,
            "single_distribution_implementation": True,
            "single_recurrent_evaluator": True,
        },
        "offline_update_parity": {"passed": offline_passed, "seeds": offline},
        "closed_loop_anchor_parity": {"passed": anchors_passed, "anchors": anchors},
        "two_epoch_collector": collector,
        "cost": cost,
        "limitations": [
            "bounded verification only; no long training was run",
            "no chunk was committed",
            "S1 external-value centering of actor internal value loss remains a declared local contract",
        ],
        "elapsed_seconds": time.time() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "verification.json", report)
    write_json(output / "input_manifest.json", inspection["assets"])
    write_json(output / "cost_accounting.json", cost)
    summary = [
        "# RL core refactor R1 verification", "",
        f"Status: `{report['status']}`", "",
        f"- offline actor parity: `{offline_passed}`",
        f"- six closed-loop anchors: `{anchors_passed}`",
        f"- two-epoch collector: `{collector.get('passed', False)}`",
        f"- control intervals: `{intervals}`",
        "- training enabled: `false`",
        "- chunk commit enabled: `false`",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("inspect", "verify", "train", "evaluate", "plan-window", "plan-startup")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physics", action="store_true", help="run bounded real MJWP verification")
    parser.add_argument("--checkpoint", type=Path, help="epoch-boundary checkpoint for evaluate")
    parser.add_argument("--resume", type=Path, help="explicit epoch-boundary checkpoint for train")
    parser.add_argument(
        "--startup-phase", choices=("preflight", "execute", "analyze"),
        default="preflight", help="bounded control-aware startup experiment phase",
    )
    return parser.parse_args(argv)


def _training_output(config: dict[str, Any], root: Path) -> Path:
    run_directory = config.get("run_directory")
    if run_directory != "runs/taco_pour_rl_task_informed_critic_v2":
        raise ValueError("task-informed critic run directory changed")
    return root / run_directory


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = args.config or (
        DEFAULT_STARTUP_CONFIG if args.command == "plan-startup" else (
        DEFAULT_PLAN_CONFIG if args.command == "plan-window" else (
            DEFAULT_TRAIN_CONFIG if args.command in {"train", "evaluate"} else DEFAULT_CONFIG
        ))
    )
    if args.command == "plan-startup":
        from .startup_planner import run_phase
        result = run_phase(
            config_path, args.asset_root, args.output, args.startup_phase,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") in {
            "PREFLIGHT_COMPLETE", "PHYSICS_COMPLETE_ANALYSIS_PENDING",
            "ANALYSIS_COMPLETE_VISUAL_REVIEW_PENDING",
        } else 2
    schema = yaml.safe_load(config_path.read_text()).get("schema")
    if schema == "egoengine_taco_pour_virtual_object_assist_v1":
        from .virtual_assist import (
            RUN_DIRECTORY,
            evaluate_virtual,
            inspect_virtual,
            train_virtual,
            verify_virtual,
        )

        loaded = yaml.safe_load(config_path.read_text())
        root = (args.asset_root or Path(loaded["asset_root"])).resolve(strict=True)
        default_output = root / RUN_DIRECTORY
        if args.command == "inspect":
            result = inspect_virtual(config_path, args.asset_root)
        elif args.command == "verify":
            if not args.physics:
                raise ValueError("virtual-assist verification requires --physics")
            result = verify_virtual(
                config_path, args.asset_root, args.output or default_output
            )
        elif args.command == "train":
            result = train_virtual(
                config_path,
                args.asset_root,
                args.output or default_output,
                resume_path=args.resume,
            )
        elif args.command == "evaluate":
            if args.checkpoint is None:
                raise ValueError("evaluate requires --checkpoint")
            result = evaluate_virtual(
                config_path,
                args.asset_root,
                args.checkpoint,
                args.output or default_output / "evaluation_manual",
            )
        else:
            raise ValueError("virtual-assist config does not support plan-window")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in {
            "INSPECT_OK",
            "VIRTUAL_ASSIST_TRAINING_CHAIN_VERIFIED",
            "STRICT_WINDOW_SUCCESS",
            "STRICT_WINDOW_FAILED",
            "STRICT_UNASSISTED_WINDOW_SUCCESS",
            "COMPLETED_NO_STRICT_WINDOW_SUCCESS",
            "ASSISTED_TAIL_COVERAGE_NOT_ESTABLISHED",
            "ASSISTANCE_NOT_OPENING_TAIL",
        } else 2
    if args.command == "inspect":
        result = inspect(config_path, args.asset_root)
    elif args.command == "verify":
        schema = yaml.safe_load(config_path.read_text()).get("schema")
        if schema == "egoengine_taco_pour_rl_task_informed_critic_v2":
            train_config, root, _ = _load_train_config(config_path, args.asset_root)
            output = args.output or _training_output(train_config, root)
            if not args.physics:
                raise ValueError("training-chain verification requires --physics")
            result = verify_training_chain(config_path, args.asset_root, output)
        else:
            output = args.output or PROJECT_ROOT / "runs/rl_core_refactor_r1"
            result = verify(config_path, args.asset_root, output, physics=args.physics)
    elif args.command == "train":
        train_config, root, config_assets = _load_train_config(config_path, args.asset_root)
        del config_assets
        output = args.output or _training_output(train_config, root)
        result = train(
            config_path, args.asset_root, output, resume_path=args.resume
        )
    elif args.command == "evaluate":
        if args.checkpoint is None:
            raise ValueError("evaluate requires --checkpoint")
        train_config, root, config_assets = _load_train_config(config_path, args.asset_root)
        del config_assets
        output = args.output or _training_output(train_config, root) / "evaluation_manual"
        result = evaluate(config_path, args.asset_root, args.checkpoint, output)
    else:
        result = plan_window(config_path, args.asset_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    successful_statuses = {
        "INSPECT_OK", "STRUCTURAL_REFACTOR_VERIFIED", "TRAINING_CHAIN_VERIFIED",
        "STRICT_WINDOW_SUCCESS", "STRICT_WINDOW_SUCCESS_CONFIRMED",
        "STRICT_WINDOW_FAILED", "COMPLETED_NO_STRICT_WINDOW_SUCCESS",
        "STRICT_WINDOW_SEQUENCE_FOUND",
    }
    return 0 if result["status"] in successful_statuses else 2
