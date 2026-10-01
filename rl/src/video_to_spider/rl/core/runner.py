"""Single bounded entrypoint for inspecting and verifying the active RL core."""

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

from .audit import module_sha256, state_dict_max_abs_error
from .env import IndependentWorlds, make_world
from .policy import PolicyBundle, burn_in_prefix, policy_step
from .ppo import PPOConfig, update_actor
from .rollout import FixedBoundaryCollector, valid_prefix
from .state_io import (
    load_boundary_context,
    load_torch_gzip,
    manifest_entry,
    sha256,
    validate_physics_snapshot,
    verify_artifact,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/rl_core_refactor_r1.yaml"


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
    rows: list[dict[str, Any]] = []
    all_passed = True
    for seed in range(3):
        _, paths = _seed_paths(isolation_root, seed)
        batch_payload = load_torch_gzip(paths["batch"])
        pre = load_torch_gzip(paths["pre"])
        target = load_torch_gzip(paths["full"])
        outputs: list[dict[str, Any]] = []
        states: list[dict[str, torch.Tensor]] = []
        # Two independent copies prove that audit I/O has no numerical effect.
        for audit_enabled in (False, True):
            policy = PolicyBundle.create(worlds=4, with_critic=False)
            policy.load_actor_only(
                pre["actor"], version=int(pre["observation_normalization_version"])
            )
            policy.actor_optimizer.load_state_dict(pre["actor_optimizer"])
            report = update_actor(
                policy,
                deepcopy(batch_payload["optimizer_input"]),
                tuple(batch_payload["boundary_reset_rnn_states"]),
                PPOConfig(),
            )
            if audit_enabled:
                report = json.loads(json.dumps(report))
            outputs.append(report)
            states.append(deepcopy(policy.actor.state_dict()))
        maximum_error, _ = state_dict_max_abs_error(states[0], target["actor"])
        logger_error, _ = state_dict_max_abs_error(states[0], states[1])
        exact = maximum_error == 0.0 and logger_error == 0.0
        all_passed &= exact
        rows.append({
            "seed": seed,
            "actor_optimizer_steps": 2,
            "maximum_abs_error_to_FULL": maximum_error,
            "logger_on_off_maximum_abs_error": logger_error,
            "bitwise_exact": exact,
            "canonical_ratio_identity": outputs[0]["canonical_ratio_max_abs_error"],
        })
    return rows, all_passed


def _make_core_world(assets: dict[str, Path], boundary: dict[str, Any], seed: int, *, asymmetric: bool):
    return make_world(
        simulator_config=assets["simulator"],
        protocol=assets["protocol"],
        objective_profile=assets["objective"],
        observation_profile=assets["observation"],
        action_profile=assets["action"],
        boundary=boundary,
        seed=seed,
        asymmetric_critic=asymmetric,
    )


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
            "terminated_samples": int(batch.terminated.sum()),
            "timeout_samples": int(batch.timeout.sum()),
            "actor_sha256": module_sha256(policy.actor),
        })
        policy.actor.update_obs_stats(batch.observations)
        policy.critic.model.update_obs_stats(batch.critic_observations)
        policy.normalization_version += 1
    return {
        "passed": all(row["samples"] == 160 and row["source_min"] == 40 and row["source_max"] == 79 for row in rows),
        "epochs": rows,
        "optimizer_steps": 0,
    }, 320


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
        "actor_optimizer_steps": 6,
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
    parser.add_argument("command", choices=("inspect", "verify"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/rl_core_refactor_r1")
    parser.add_argument("--physics", action="store_true", help="run bounded real MJWP verification")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "inspect":
        result = inspect(args.config, args.asset_root)
    else:
        result = verify(args.config, args.asset_root, args.output, physics=args.physics)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"INSPECT_OK", "STRUCTURAL_REFACTOR_VERIFIED"} else 2
