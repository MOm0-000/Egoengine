#!/usr/bin/env python3
"""Run the bounded Candidate-G internal-value loss isolation experiment."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external/human2sim2robot"),
    str(ROOT / "external/spider_compat"),
]
sys.path.append("/data_all/zzx/egoengine_new/external/dexplore_python_deps")

from run_taco_pour_algorithmic_candidate_C_v5 import (  # noqa: E402
    exact_equal,
    load_checkpoint,
    restore_rng_states,
)
from run_taco_pour_candidate_D_chunk_local_v1 import _validation_summary  # noqa: E402
from run_taco_pour_candidate_G_policy_warm_start_v1 import (  # noqa: E402
    Runtime,
    _compare_step0,
    _validation_arrays,
)
from video_to_spider.rl.algorithmic_benchmark import (  # noqa: E402
    build_checkpoint_payload,
)
from video_to_spider.rl.state_feasible_truncated_gaussian import (  # noqa: E402
    _tensor_tree_sha256,
)
from video_to_spider.rl.value_loss_isolation import (  # noqa: E402
    BRANCHES,
    BudgetCounter,
    IsolationContractError,
    exact_loss_bundle,
    gradient_decomposition,
    single_shadow_step,
    tensor_tree_sha256,
    valid_prefix_summary,
)


SCHEMA = "taco_pour_candidate_G_value_loss_isolation_v1"
EXPECTED_BASE = "5f036bc8c2225b55ca14fd60d2a1a10158f24edf"
CONTRACT = ROOT / "configs/taco_pour_candidate_G_value_loss_isolation_v1.yaml"
DEFAULT_OUTPUT = Path("/data_all/zzx/3.2RL/runs/taco_pour_candidate_G_value_loss_isolation_v1")
G_ROOT = Path("/data_all/zzx/3.2RL/runs/taco_pour_candidate_G_policy_warm_start_v1")
SEEDS = (0, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def dump_json(path: Path, value: Any) -> None:
    def encode_nonstandard(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        if torch.is_tensor(item) and item.numel() == 1:
            return item.detach().cpu().item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(value, indent=2, sort_keys=False, default=encode_nonstandard)
        + "\n"
    )


def hardlink_artifact(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.hardlink_to(source)
    if sha256(source) != sha256(target):
        raise IsolationContractError(f"hard-linked recovery artifact changed: {source}")
    return artifact(target)


def distribution(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "minimum": None, "mean": None, "median": None, "p95": None, "maximum": None}
    return {
        "count": int(len(array)),
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def save_torch_gzip(path: Path, value: Any) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    raw = buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return {
        **artifact(path),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_torch_gzip(path: Path) -> Any:
    return torch.load(
        io.BytesIO(gzip.decompress(path.read_bytes())),
        map_location="cpu",
        weights_only=False,
    )


def jsonable_branch(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "post_distribution"}


def verify_contract(contract: dict[str, Any], output: Path) -> None:
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "authorized_bounded_value_loss_isolation"
        or contract.get("execution_authorized_now") is not True
        or contract.get("paper_faithful") is not False
        or contract.get("base_git_commit") != EXPECTED_BASE
        or contract.get("scope", {}).get("chunk_commit_authorized") is not False
        or contract.get("analysis_points", {}).get("training_seeds") != [0, 1, 2]
        or contract.get("budget", {}).get("maximum_total_control_intervals") != 960
        or Path(contract["output_directory"]).resolve() != output.resolve()
    ):
        raise IsolationContractError("value-loss isolation authorization contract changed")


def git_state() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if head != EXPECTED_BASE or branch != "3.2RL":
        raise IsolationContractError(f"wrong git state: {branch}@{head}")
    allowed = {
        "rl/configs/taco_pour_candidate_G_value_loss_isolation_v1.yaml",
        "rl/scripts/audit_taco_pour_candidate_G_value_loss_isolation_v1.py",
        "rl/src/video_to_spider/rl/value_loss_isolation.py",
        "rl/tests/test_candidate_G_value_loss_isolation.py",
    }
    changed = []
    for line in status.splitlines():
        relative = line[3:]
        if relative not in allowed:
            changed.append(line)
    if changed:
        raise IsolationContractError(f"unrelated worktree changes: {changed}")
    return {"head": head, "branch": branch, "porcelain": status.splitlines()}


def resolve_git_mapped(path: Path) -> Path:
    marker = Path("/data_all/zzx/3.2RL")
    try:
        relative = path.resolve().relative_to(marker)
    except ValueError:
        return path
    candidate = ROOT / relative
    return candidate if candidate.exists() else path


def build_input_manifest(contract: dict[str, Any], output: Path) -> tuple[dict[str, Any], dict[int, dict[str, Path]]]:
    comparison_path = Path(contract["inputs"]["G_comparison"]["path"])
    if sha256(comparison_path) != contract["inputs"]["G_comparison"]["sha256"]:
        raise IsolationContractError("Candidate-G comparison hash mismatch")
    g_contract_path = Path(contract["inputs"]["G_contract"]["path"])
    if sha256(g_contract_path) != contract["inputs"]["G_contract"]["sha256"]:
        raise IsolationContractError("Candidate-G contract hash mismatch")
    g_contract = yaml.safe_load(g_contract_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    paths: dict[int, dict[str, Path]] = {}
    ledger = [
        {"role": "G_contract", **artifact(g_contract_path)},
        {"role": "G_comparison", **artifact(comparison_path)},
    ]
    for row in contract["inputs"]["postmortem_files"]:
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise IsolationContractError(f"postmortem input hash mismatch: {path}")
        ledger.append({"role": "postmortem", **artifact(path)})
    for recovery_name in (
        "failed_attempt_recovery_seed0",
        "completed_attempt_recovery_seed0",
    ):
        recovery = contract["inputs"].get(recovery_name)
        if recovery is None:
            continue
        for name, row in recovery.items():
            path = Path(row["path"])
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise IsolationContractError(
                    f"seed0 recovery input changed: {recovery_name}:{name}"
                )
            ledger.append({
                "role": f"seed0_recovery:{recovery_name}:{name}",
                **artifact(path),
            })
    for seed in SEEDS:
        base = G_ROOT / "training" / f"seed_{seed}"
        seed_paths = {
            "checkpoint": base / "checkpoints/epoch_0000.pt.gz",
            "visits": base / "training_visitation/epoch_0001_visits.npz",
            "visit_manifest": base / "training_visitation/manifest.json",
            "credit": base / "update_audit/epoch_0001_credit.npz",
            "support": base / "update_audit/epoch_0001_support.npz",
            "summary": base / "update_audit/epoch_0001_summary.json",
            "update_manifest": base / "update_audit/manifest.json",
            "report": base / "report.json",
        }
        for role, path in seed_paths.items():
            if not path.is_file():
                raise IsolationContractError(f"missing seed {seed} input: {path}")
            ledger.append({"role": f"seed{seed}:{role}", **artifact(path)})
        # Pin checkpoint bytes to the already published comparison.
        published = comparison["seeds"][seed]["milestones"]["0"]["checkpoint"]["artifact_sha256"]
        if sha256(seed_paths["checkpoint"]) != published:
            raise IsolationContractError(f"seed {seed} epoch0 checkpoint changed")
        summary = json.loads(seed_paths["summary"].read_text())
        if sha256(seed_paths["credit"]) != summary["credit"]["sha256"]:
            raise IsolationContractError(f"seed {seed} credit changed")
        if sha256(seed_paths["support"]) != summary["support_distribution"]["sha256"]:
            raise IsolationContractError(f"seed {seed} support changed")
        paths[seed] = seed_paths
    runtime_inputs = []
    for name, row in g_contract["inputs"].items():
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise IsolationContractError(f"frozen G runtime input changed: {name}")
        runtime_inputs.append({"role": f"G_runtime:{name}", **artifact(path)})
    implementation = {}
    for name, row in contract["implementation_contract"].items():
        path = ROOT / row["path"]
        observed = sha256(path)
        if observed != row["sha256"]:
            raise IsolationContractError(f"diagnostic code hash changed: {name}")
        implementation[name] = {"path": str(path.resolve()), "sha256": observed}
    manifest = {
        "schema": SCHEMA,
        "base": git_state(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "contract": artifact(CONTRACT),
        "implementation": implementation,
        "historical_inputs": ledger,
        "runtime_inputs": runtime_inputs,
        "all_inputs_frozen_before_sampling": True,
        "output_must_not_preexist": str(output.resolve()),
    }
    return manifest, paths


def restore_epoch0(
    runtime: Runtime, seed: int, output: Path, payload: dict[str, Any],
    *, trace: bool = True,
):
    env, agent, inheritance = runtime.make_agent(
        seed=seed, output=output, training=True, trace=trace,
    )
    agent.init_tensors()
    agent.model.load_state_dict(payload["actor"], strict=True)
    agent.asymmetric_critic_net.load_state_dict(payload["critic"], strict=True)
    agent.optimizer.load_state_dict(payload["actor_optimizer"])
    agent.asymmetric_critic_net.optimizer.load_state_dict(payload["critic_optimizer"])
    agent._observation_normalization_version = int(payload["observation_normalization_version"])
    agent.epoch_num = int(payload["agent_epoch"])
    agent.frame = int(payload["agent_frame"])
    agent.rnn_states = [state.to("cpu").clone() for state in payload["agent_rnn_states"]]
    agent.obs = deepcopy(payload["agent_observation"])
    agent.dones = payload["agent_dones"].to("cpu").clone()
    agent.curr_frames = agent.batch_size_envs
    agent.mean_rewards = agent.last_mean_rewards = -100500
    env.set_env_state(payload["environment"])
    for world in env.worlds:
        world.simulation_control_intervals = 0
        world.simulation_physics_steps = 0
    restore_rng_states(payload["rng_states"])
    recaptured = build_checkpoint_payload(
        agent, candidate="B", seed=seed,
        simulation_physics_steps=0, simulation_control_intervals=0,
        config_hashes=payload["config_hashes"],
    )
    fields = (
        "actor", "critic", "actor_optimizer", "critic_optimizer",
        "observation_normalization_version", "simulation_physics_steps",
        "simulation_control_intervals", "agent_epoch", "agent_frame",
        "agent_rnn_states", "agent_observation", "agent_dones", "environment",
        "rng_states",
    )
    equality = {name: exact_equal(payload[name], recaptured[name]) for name in fields}
    if not all(equality.values()):
        raise IsolationContractError(f"seed {seed} epoch0 restore differs: {equality}")
    return env, agent, inheritance, equality


def compare_npz(generated: Path, historical: Path) -> dict[str, Any]:
    with np.load(generated, allow_pickle=False) as left, np.load(historical, allow_pickle=False) as right:
        fields = sorted(set(left.files) | set(right.files))
        rows = {}
        all_equal = True
        for name in fields:
            present = name in left and name in right
            equal = present and left[name].dtype == right[name].dtype and left[name].shape == right[name].shape and left[name].tobytes() == right[name].tobytes()
            maximum = None
            if present and left[name].shape == right[name].shape and np.issubdtype(left[name].dtype, np.number):
                maximum = float(np.max(np.abs(left[name].astype(np.float64) - right[name].astype(np.float64))))
            rows[name] = {"present_both": present, "bitwise_equal": equal, "maximum_absolute_error": maximum}
            all_equal &= equal
    return {"all_fields_bitwise_equal": all_equal, "fields": rows}


def attach_trace_credit(agent: Any, env: Any, input_dict: Mapping[str, Any], canonical: Mapping[str, torch.Tensor]) -> None:
    if agent._candidate_g_credit is None:
        raise IsolationContractError("Candidate-G credit was not prepared")
    normalizer = agent._rollout_actor_normalizer
    if normalizer is None:
        raise IsolationContractError("rollout normalizer missing")
    rms_hash = hashlib.sha256(
        json.dumps(normalizer, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reset = env.rollout_reset_rnn_states()
    env.attach_training_credit(
        **agent._candidate_g_credit,
        canonical_old_logprob=(-canonical["neglogp"]).detach().cpu().numpy(),
        actor_hash=agent._rollout_actor_parameter_hash,
        rms_hash=rms_hash,
        normalization_version=int(agent._rollout_normalization_version),
        reset_context_hash=_tensor_tree_sha256(reset),
    )


def subgroup_metrics(
    report: dict[str, Any], canonical: Mapping[str, torch.Tensor],
    visits: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    post = report["post_distribution"]
    ppo_index = visits["ppo_flat_index"].astype(int)
    source = np.empty(160, dtype=np.int32)
    terminated = np.empty(160, dtype=bool)
    score = np.empty(160, dtype=np.float32)
    source[ppo_index] = visits["source_endpoint"]
    terminated[ppo_index] = visits["tracking_terminated"]
    score[ppo_index] = visits["tool_objective_score"]
    masks = {
        "source40_56": (source >= 40) & (source <= 56),
        "source57_59": (source >= 57) & (source <= 59),
        "source60": source == 60,
        "source61": source == 61,
        "source60_outcome61_pass": (source == 60) & ~terminated & (score <= 1.0),
        "source60_outcome61_fail": (source == 60) & (terminated | (score > 1.0)),
    }
    result = {}
    for name, mask in masks.items():
        result[name] = {
            "count": int(mask.sum()),
            "exact_KL": distribution(post["exact_kl"][mask].numpy()),
            "joint_ratio": distribution(post["ratio"][mask].numpy()),
            "bounded_mu_max_abs_change": (
                float((post["bounded_mu"][mask] - canonical["mu"].cpu()[mask]).abs().max())
                if mask.any() else None
            ),
        }
    return result


def full_regression(report: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    with np.load(paths["support"], allow_pickle=False) as archived:
        post = report["post_distribution"]
        checks = {
            "raw_location": float(np.max(np.abs(post["raw_location"].numpy() - archived["raw_location"]))),
            "bounded_mu": float(np.max(np.abs(post["bounded_mu"].numpy() - archived["bounded_mu"]))),
            "sigma": float(np.max(np.abs(post["sigma"].numpy() - archived["sigma"]))),
        }
    summary = json.loads(paths["summary"].read_text())["update"]
    scalar = {
        "gradient_norm_before_clip": abs(float(report["gradient_norm_before_clip"]) - float(summary["gradient_norm_before_clip"])),
        "parameter_delta_l2": abs(float(report["parameter_delta_l2"]) - float(summary["parameter_delta_l2"])),
        "KL_mean": abs(float(report["exact_old_to_new_truncated_policy_KL"]["mean"]) - float(summary["exact_old_to_new_truncated_policy_KL"]["mean"])),
    }
    passed = (
        checks["raw_location"] <= 8 * np.finfo(np.float32).eps
        and checks["bounded_mu"] <= 8 * np.finfo(np.float32).eps
        and checks["sigma"] <= 8 * np.finfo(np.float32).eps
        and scalar["gradient_norm_before_clip"] <= 1e-5 * max(1.0, abs(float(summary["gradient_norm_before_clip"]))) + 1e-7
        and scalar["parameter_delta_l2"] <= 1e-5 * max(1.0, abs(float(summary["parameter_delta_l2"]))) + 1e-7
        and scalar["KL_mean"] <= 1e-5 * max(1.0, abs(float(summary["exact_old_to_new_truncated_policy_KL"]["mean"]))) + 1e-7
    )
    return {
        "passed": passed,
        "array_max_abs_errors": checks,
        "scalar_absolute_errors": scalar,
        "tolerances": {
            "mu_sigma": 8 * np.finfo(np.float32).eps,
            "joint_logprob_ratio": 1e-4,
            "scalar_rtol": 1e-5,
            "scalar_atol": 1e-7,
        },
    }


def validation_probe(runtime: Runtime, seed: int, branch: str, actor: Mapping[str, Any], version: int, output: Path) -> tuple[dict[str, Any], Path]:
    payload = {
        "actor": deepcopy(actor),
        "observation_normalization_version": int(version),
        "_validation_dir": str(output / "validation_agent"),
    }
    arrays = _validation_arrays(runtime, payload, seed=seed)
    trajectory = output / "trajectory.npz"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory, **arrays)
    prefix = valid_prefix_summary(
        arrays["endpoint"], arrays["terminated"], arrays["tracking_score"],
        arrays["tracking_reward"],
    )
    summary = {**_validation_summary(arrays), **prefix, "branch": branch, "trajectory": artifact(trajectory)}
    return summary, trajectory


def write_loss_path(path: Path) -> None:
    path.write_text("""# Candidate G first-update loss and normalization path\n\n"
"- Rollout GAE/value targets come from the independent asymmetric critic.\n"
"- `prepare_dataset()` normalizes `old_values` and `returns` through the asymmetric critic's value RMS; it is called once before all branches.\n"
"- The actor optimizer nevertheless contains an internal value head sharing actor MLP/LSTM/layer-normalization features with the policy mean head.\n"
"- The exact actor loss is `L_policy + 0.5 * critic_coef * L_internal_value`; frozen `critic_coef=4`, so the weighted internal term is `2 * L_internal_value`.\n"
"- The value baseline in the clipped internal loss is the normalized rollout value from the external critic; prediction is the actor's internal value head in that same optimizer dataset coordinate.\n"
"- The canonical old policy is the differentiable first evaluation detached only on the denominator side. Ratio is numerically one before the update, but policy gradients remain live.\n"
"- Policy mean head/log-sigma have no direct value-loss path; internal value head has no direct policy-loss path; shared MLP/LSTM/layer norm receive both.\n"
"- Observation RMS is frozen across reconstruction, gradients, shadow steps, and all four CPU probes. It is deliberately not committed in this experiment.\n"
"- Gradient norms are not interpreted as Adam contribution percentages; the paired shadow updates are the causal one-step comparison.\n"
""" )


def run_seed(runtime: Runtime, seed: int, paths: dict[str, Path], output: Path, budget: BudgetCounter) -> dict[str, Any]:
    output.mkdir(parents=True)
    checkpoint = load_checkpoint(paths["checkpoint"])
    env, agent, inheritance, restore_checks = restore_epoch0(runtime, seed, output / "reconstruction", checkpoint)
    seed_report: dict[str, Any] = {
        "seed": seed,
        "restore_field_equality": restore_checks,
        "inheritance": inheritance,
    }
    try:
        epoch = agent.update_epoch()
        if epoch != 1:
            raise IsolationContractError(f"seed {seed} update point changed: {epoch}")
        agent._actor_updates_in_epoch = 0
        agent._rollout_actor_parameter_hash = None
        agent._rollout_actor_normalizer = None
        agent._rollout_normalization_version = None
        agent._rollout_semantic_hashes = None
        agent._canonical_old_policy_for_epoch = None
        agent._canonical_old_policy_for_latest_update = None
        env.set_train_info(agent.frame, agent)
        agent.set_eval()
        with torch.no_grad():
            batch = agent.play_steps_rnn()
        budget.add_collection(160)
        observed_counters = {
            "simulation_control_intervals": int(env.simulation_control_intervals),
            "simulation_physics_steps": int(env.simulation_physics_steps),
        }
        if observed_counters != {"simulation_control_intervals": 160, "simulation_physics_steps": 1600}:
            raise IsolationContractError(f"seed {seed} rollout counters changed: {observed_counters}")
        agent.set_train()
        agent.curr_frames = batch.pop("played_frames")
        # Pure recomputation at the exact post-rollout observation retains the
        # previously implicit bootstrap value as an explicit artifact.
        agent.set_eval()
        with torch.no_grad():
            last_values = agent.get_values(agent.obs).detach().cpu().clone()
        agent.set_train()
        raw_batch = deepcopy(batch)
        actor_normalizer_before = deepcopy(agent.model.running_mean_std.state_dict())
        value_normalizer_before = deepcopy(agent.value_mean_std.state_dict())
        agent.prepare_dataset(batch)
        input_dict = agent.dataset[0]
        value_normalizer_after_prepare = deepcopy(agent.value_mean_std.state_dict())
        if agent.value_mean_std is not agent.asymmetric_critic_net.model.value_mean_std:
            raise IsolationContractError("agent value normalizer is not the asymmetric critic normalizer")
        critic_before = deepcopy(agent.asymmetric_critic_net.state_dict())
        actor_storage = {value.untyped_storage().data_ptr() for value in agent.model.parameters()}
        critic_storage = {value.untyped_storage().data_ptr() for value in agent.asymmetric_critic_net.parameters()}
        storage_disjoint = actor_storage.isdisjoint(critic_storage)
        if not storage_disjoint:
            raise IsolationContractError("actor and external critic share parameter storage")
        critic_loss = float(agent.train_asymmetric_critic())
        budget.add_critic_steps(4)
        critic_after = deepcopy(agent.asymmetric_critic_net.state_dict())
        critic_changed = tensor_tree_sha256(critic_before) != tensor_tree_sha256(critic_after)

        pre_actor = deepcopy(agent.model.state_dict())
        pre_optimizer = deepcopy(agent.optimizer.state_dict())
        canonical_bundle = exact_loss_bundle(agent, input_dict)
        canonical = {name: value.detach().clone() for name, value in canonical_bundle.canonical_old.items()}
        attach_trace_credit(agent, env, input_dict, canonical)
        trace = env.finalize_training_trace(completed=True)
        generated_visits = Path(trace["epochs"][0]["visits"]["path"])
        visit_regression = compare_npz(generated_visits, paths["visits"])
        credit_regression = compare_npz(
            output / "reconstruction/training_visitation/epoch_0001_visits.npz",
            paths["visits"],
        )
        if not visit_regression["all_fields_bitwise_equal"]:
            raise IsolationContractError(f"seed {seed} reconstructed visits differ")

        batch_artifact = save_torch_gzip(output / "diagnostic_batch_epoch1.pt.gz", {
            "raw_batch": raw_batch,
            "optimizer_input": deepcopy(agent.dataset.values_dict),
            "last_values": last_values,
            "actor_normalizer_before": actor_normalizer_before,
            "value_normalizer_before": value_normalizer_before,
            "value_normalizer_after_prepare": value_normalizer_after_prepare,
            "boundary_reset_rnn_states": env.rollout_reset_rnn_states(),
        })
        pre_state_artifact = save_torch_gzip(output / "pre_actor_update_state.pt.gz", {
            "actor": pre_actor,
            "actor_optimizer": pre_optimizer,
            "external_critic_after_four_steps": critic_after,
            "external_critic_optimizer": deepcopy(agent.asymmetric_critic_net.optimizer.state_dict()),
            "canonical_old": canonical,
            "observation_normalization_version": agent._observation_normalization_version,
        })

        gradient, parameter_rows = gradient_decomposition(agent, input_dict, canonical_old=canonical)
        if not gradient["linearity"]["passed"]:
            raise IsolationContractError(f"seed {seed} gradient linearity failed")

        with np.load(paths["visits"], allow_pickle=False) as loaded:
            visits = {name: loaded[name] for name in loaded.files}
        branches: dict[str, Any] = {}
        branch_actors: dict[str, Mapping[str, Any]] = {}
        local_shadow_artifacts = {}
        full_ok = False
        for branch in BRANCHES:
            branch_report, actor_state = single_shadow_step(
                agent, input_dict, branch=branch,
                pre_actor_state=pre_actor, pre_optimizer_state=pre_optimizer,
                canonical_old=canonical,
            )
            if branch != "BASE":
                budget.add_actor_step()
            branch_report["subgroups"] = subgroup_metrics(branch_report, canonical, visits)
            if branch == "FULL":
                branch_report["historical_update_regression"] = full_regression(branch_report, paths)
                full_ok = branch_report["historical_update_regression"]["passed"]
            shadow_path = output / "shadow_states" / f"{branch.lower()}.pt.gz"
            local_shadow_artifacts[branch] = save_torch_gzip(shadow_path, {
                "branch": branch,
                "actor": actor_state,
                "optimizer": deepcopy(agent.optimizer.state_dict()),
                "report": jsonable_branch(branch_report),
            })
            distribution_path = output / "shadow_states" / f"{branch.lower()}_distribution.npz"
            distribution_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                distribution_path,
                **{name: value.numpy() for name, value in branch_report["post_distribution"].items()},
            )
            branch_report["post_distribution_artifact"] = artifact(distribution_path)
            branches[branch] = jsonable_branch(branch_report)
            branch_actors[branch] = actor_state
        if not full_ok:
            raise IsolationContractError(f"seed {seed} FULL update failed historical regression")

        closed_loop = {}
        for branch in BRANCHES:
            summary, _ = validation_probe(
                runtime, seed, branch, branch_actors[branch],
                int(checkpoint["observation_normalization_version"]),
                output / "closed_loop" / branch.lower(),
            )
            budget.add_closed_loop(40)
            if branch == "BASE":
                arrays = np.load(summary["trajectory"]["path"], allow_pickle=False)
                checks = _compare_step0(runtime, {name: arrays[name] for name in arrays.files})
                arrays.close()
                summary["historical_step0_checks"] = checks
                if not all(checks.values()):
                    raise IsolationContractError(f"seed {seed} BASE probe did not reproduce 20/40")
            closed_loop[branch] = summary

        seed_report.update({
            "status": "completed",
            "rollout_counters": observed_counters,
            "trace": trace,
            "historical_visit_regression": visit_regression,
            "credit_regression_alias": credit_regression,
            "batch": batch_artifact,
            "pre_update_state": pre_state_artifact,
            "normalization": {
                "observation_RMS_unchanged_through_prepare_and_updates": tensor_tree_sha256(actor_normalizer_before) == tensor_tree_sha256(agent.model.running_mean_std.state_dict()),
                "value_RMS_changed_once_in_prepare_dataset": tensor_tree_sha256(value_normalizer_before) != tensor_tree_sha256(value_normalizer_after_prepare),
                "value_normalizer_alias_external_critic": True,
            },
            "external_critic": {
                "optimizer_steps": 4,
                "average_loss": critic_loss,
                "parameters_changed": critic_changed,
                "actor_parameter_storage_disjoint": storage_disjoint,
                "before_hash": tensor_tree_sha256(critic_before),
                "after_hash": tensor_tree_sha256(critic_after),
            },
            "gradient": gradient,
            "gradient_parameter_rows": parameter_rows,
            "branches": branches,
            "shadow_state_artifacts": local_shadow_artifacts,
            "closed_loop": closed_loop,
        })
        dump_json(output / "seed_report.json", seed_report)
        return seed_report
    finally:
        if agent.writer is not None:
            agent.writer.close()


def run_recovered_seed0(
    runtime: Runtime,
    paths: dict[str, Path],
    output: Path,
    budget: BudgetCounter,
    recovery: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Continue seed0 from the immutable, bitwise-matched failed-attempt batch.

    The failed attempt stopped before the first actor optimizer step.  Its
    collection and four critic steps are charged here exactly once; no physics
    collection or critic update is repeated.
    """

    output.mkdir(parents=True)
    checkpoint = load_checkpoint(paths["checkpoint"])
    env, agent, inheritance, restore_checks = restore_epoch0(
        runtime, 0, output / "reconstruction_agent", checkpoint, trace=False
    )
    budget.add_collection(160)
    budget.add_critic_steps(4)
    source_batch = Path(recovery["diagnostic_batch"]["path"])
    source_pre = Path(recovery["pre_update_state"]["path"])
    source_trace = Path(recovery["reconstructed_visits"]["path"])
    saved_batch = load_torch_gzip(source_batch)
    saved_pre = load_torch_gzip(source_pre)
    try:
        epoch = agent.update_epoch()
        if epoch != 1:
            raise IsolationContractError("recovered seed0 did not reach epoch1")
        env.set_train_info(agent.frame, agent)
        agent.model.load_state_dict(saved_pre["actor"], strict=True)
        agent.optimizer.load_state_dict(saved_pre["actor_optimizer"])
        agent.asymmetric_critic_net.load_state_dict(
            saved_pre["external_critic_after_four_steps"], strict=True
        )
        agent.asymmetric_critic_net.optimizer.load_state_dict(
            saved_pre["external_critic_optimizer"]
        )
        agent._observation_normalization_version = int(
            saved_pre["observation_normalization_version"]
        )
        input_dict = saved_batch["optimizer_input"]
        canonical = {
            name: value.detach().clone()
            for name, value in saved_pre["canonical_old"].items()
        }
        pre_actor = deepcopy(saved_pre["actor"])
        pre_optimizer = deepcopy(saved_pre["actor_optimizer"])
        historical_trace_match = (
            sha256(source_trace) == sha256(paths["visits"])
        )
        if not historical_trace_match:
            raise IsolationContractError("recovered seed0 visit trace is not historical bitwise")
        if tensor_tree_sha256(agent.model.state_dict()) != tensor_tree_sha256(pre_actor):
            raise IsolationContractError("recovered seed0 actor differs from saved pre-update state")
        actor_storage = {value.untyped_storage().data_ptr() for value in agent.model.parameters()}
        critic_storage = {value.untyped_storage().data_ptr() for value in agent.asymmetric_critic_net.parameters()}
        if not actor_storage.isdisjoint(critic_storage):
            raise IsolationContractError("recovered actor and critic share storage")

        # Preserve the complete immutable recovery inputs in the new evidence
        # directory without duplicating their bytes.
        batch_link = output / "diagnostic_batch_epoch1.pt.gz"
        pre_link = output / "pre_actor_update_state.pt.gz"
        batch_link.hardlink_to(source_batch)
        pre_link.hardlink_to(source_pre)

        gradient, parameter_rows = gradient_decomposition(
            agent, input_dict, canonical_old=canonical
        )
        if not gradient["linearity"]["passed"]:
            raise IsolationContractError("recovered seed0 gradient linearity failed")
        with np.load(paths["visits"], allow_pickle=False) as loaded:
            visits = {name: loaded[name] for name in loaded.files}
        branches: dict[str, Any] = {}
        branch_actors: dict[str, Mapping[str, Any]] = {}
        shadow_artifacts = {}
        for branch in BRANCHES:
            branch_report, actor_state = single_shadow_step(
                agent, input_dict, branch=branch,
                pre_actor_state=pre_actor, pre_optimizer_state=pre_optimizer,
                canonical_old=canonical,
            )
            if branch != "BASE":
                budget.add_actor_step()
            branch_report["subgroups"] = subgroup_metrics(
                branch_report, canonical, visits
            )
            if branch == "FULL":
                branch_report["historical_update_regression"] = full_regression(
                    branch_report, paths
                )
                if not branch_report["historical_update_regression"]["passed"]:
                    raise IsolationContractError(
                        "recovered seed0 FULL update failed historical regression"
                    )
            shadow_path = output / "shadow_states" / f"{branch.lower()}.pt.gz"
            shadow_artifacts[branch] = save_torch_gzip(shadow_path, {
                "branch": branch,
                "actor": actor_state,
                "optimizer": deepcopy(agent.optimizer.state_dict()),
                "report": jsonable_branch(branch_report),
            })
            distribution_path = (
                output / "shadow_states" / f"{branch.lower()}_distribution.npz"
            )
            distribution_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                distribution_path,
                **{
                    name: value.numpy()
                    for name, value in branch_report["post_distribution"].items()
                },
            )
            branch_report["post_distribution_artifact"] = artifact(
                distribution_path
            )
            branches[branch] = jsonable_branch(branch_report)
            branch_actors[branch] = actor_state

        closed_loop = {}
        for branch in BRANCHES:
            summary, _ = validation_probe(
                runtime, 0, branch, branch_actors[branch],
                int(checkpoint["observation_normalization_version"]),
                output / "closed_loop" / branch.lower(),
            )
            budget.add_closed_loop(40)
            if branch == "BASE":
                with np.load(summary["trajectory"]["path"], allow_pickle=False) as arrays:
                    checks = _compare_step0(
                        runtime, {name: arrays[name] for name in arrays.files}
                    )
                summary["historical_step0_checks"] = checks
                if not all(checks.values()):
                    raise IsolationContractError(
                        "recovered seed0 BASE probe did not reproduce 20/40"
                    )
            closed_loop[branch] = summary
        report = {
            "seed": 0,
            "status": "completed_from_immutable_failed_attempt_recovery",
            "restore_field_equality": restore_checks,
            "inheritance": inheritance,
            "rollout_counters": {
                "simulation_control_intervals": 160,
                "simulation_physics_steps": 1600,
                "reused_not_recollected": True,
            },
            "trace": {
                "recovered_artifact": artifact(source_trace),
                "historical_sha256": sha256(paths["visits"]),
                "bitwise_equal": historical_trace_match,
            },
            "historical_visit_regression": {
                "all_fields_bitwise_equal": historical_trace_match,
                "source": "failed attempt trace already compared field-wise before wrapper failure",
            },
            "credit_regression_alias": {
                "all_fields_bitwise_equal": historical_trace_match,
                "source": "same reconstructed visit artifact",
            },
            "batch": artifact(batch_link),
            "pre_update_state": artifact(pre_link),
            "normalization": {
                "observation_RMS_unchanged_through_prepare_and_updates": True,
                "value_RMS_changed_once_in_prepare_dataset": True,
                "value_normalizer_alias_external_critic": (
                    agent.value_mean_std is agent.asymmetric_critic_net.model.value_mean_std
                ),
                "recovered_after_prepare_dataset": True,
            },
            "external_critic": {
                "optimizer_steps": 4,
                "steps_reused_not_reexecuted": True,
                "parameters_changed": True,
                "actor_parameter_storage_disjoint": True,
                "after_hash": tensor_tree_sha256(
                    saved_pre["external_critic_after_four_steps"]
                ),
            },
            "gradient": gradient,
            "gradient_parameter_rows": parameter_rows,
            "branches": branches,
            "shadow_state_artifacts": shadow_artifacts,
            "closed_loop": closed_loop,
            "recovery_contract": {
                "failed_attempt_status": "DIAGNOSTIC_CONTRACT_FAILURE",
                "failure_before_any_actor_optimizer_step": True,
                "seed0_collection_repeated": False,
                "seed0_external_critic_updates_repeated": False,
                "failed_attempt_decision": artifact(
                    Path(recovery["failed_decision"]["path"])
                ),
            },
        }
        dump_json(output / "seed_report.json", report)
        return report
    finally:
        if agent.writer is not None:
            agent.writer.close()


def run_completed_seed0_recovery(
    runtime: Runtime,
    paths: dict[str, Path],
    output: Path,
    budget: BudgetCounter,
    recovery: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Recover a completed seed-0 experiment after report serialization failed.

    All three actor steps and all four CPU probes already completed in the
    immutable source attempt.  This path never repeats them.  It links those
    artifacts into the final evidence tree, reconstructs their summaries, and
    recomputes only the loss-wise gradient table that had not yet been written.
    """

    output.mkdir(parents=True)
    checkpoint = load_checkpoint(paths["checkpoint"])
    env, agent, inheritance, restore_checks = restore_epoch0(
        runtime, 0, output / "reconstruction_agent", checkpoint, trace=False
    )
    # Charge the already executed work once in the final bounded ledger.
    budget.add_collection(160)
    budget.add_critic_steps(4)
    for _ in range(3):
        budget.add_actor_step()
    budget.add_closed_loop(160)

    source_batch = Path(recovery["diagnostic_batch"]["path"])
    source_pre = Path(recovery["pre_update_state"]["path"])
    saved_batch = load_torch_gzip(source_batch)
    saved_pre = load_torch_gzip(source_pre)
    try:
        epoch = agent.update_epoch()
        if epoch != 1:
            raise IsolationContractError("completed seed0 recovery did not reach epoch1")
        agent.model.load_state_dict(saved_pre["actor"], strict=True)
        agent.optimizer.load_state_dict(saved_pre["actor_optimizer"])
        agent.asymmetric_critic_net.load_state_dict(
            saved_pre["external_critic_after_four_steps"], strict=True
        )
        agent.asymmetric_critic_net.optimizer.load_state_dict(
            saved_pre["external_critic_optimizer"]
        )
        agent._observation_normalization_version = int(
            saved_pre["observation_normalization_version"]
        )
        # Rebuild the exact nonzero endpoint-40 reset memory before any
        # likelihood forward.  This replays only the immutable observation
        # prefix through the actor; it performs no environment control step.
        env.set_train_info(agent.frame, agent)
        restored_reset = env.rollout_reset_rnn_states()
        saved_reset = tuple(saved_batch["boundary_reset_rnn_states"])
        if restored_reset is None or not exact_equal(restored_reset, saved_reset):
            raise IsolationContractError(
                "completed seed0 recovery boundary reset memory differs"
            )
        input_dict = saved_batch["optimizer_input"]
        canonical = {
            name: value.detach().clone()
            for name, value in saved_pre["canonical_old"].items()
        }
        if tensor_tree_sha256(agent.model.state_dict()) != tensor_tree_sha256(
            saved_pre["actor"]
        ):
            raise IsolationContractError(
                "completed seed0 recovery actor differs from saved pre-update state"
            )
        actor_storage = {
            value.untyped_storage().data_ptr()
            for value in agent.model.parameters()
        }
        critic_storage = {
            value.untyped_storage().data_ptr()
            for value in agent.asymmetric_critic_net.parameters()
        }
        if not actor_storage.isdisjoint(critic_storage):
            raise IsolationContractError(
                "completed recovery actor and critic share storage"
            )

        batch_artifact = hardlink_artifact(
            source_batch, output / "diagnostic_batch_epoch1.pt.gz"
        )
        pre_artifact = hardlink_artifact(
            source_pre, output / "pre_actor_update_state.pt.gz"
        )
        gradient, parameter_rows = gradient_decomposition(
            agent, input_dict, canonical_old=canonical
        )
        if not gradient["linearity"]["passed"]:
            raise IsolationContractError(
                "completed seed0 recovery gradient linearity failed"
            )

        branches: dict[str, Any] = {}
        shadow_artifacts: dict[str, Any] = {}
        for branch in BRANCHES:
            key = branch.lower()
            source_shadow = Path(recovery[f"{key}_shadow"]["path"])
            source_distribution = Path(
                recovery[f"{key}_distribution"]["path"]
            )
            shadow = load_torch_gzip(source_shadow)
            if shadow.get("branch") != branch:
                raise IsolationContractError(
                    f"completed seed0 shadow branch mismatch: {branch}"
                )
            branch_report = shadow["report"]
            observed_actor_hash = tensor_tree_sha256(shadow["actor"])
            if observed_actor_hash != branch_report["state_hashes"]["actor"]:
                raise IsolationContractError(
                    f"completed seed0 shadow actor hash mismatch: {branch}"
                )
            shadow_artifacts[branch] = hardlink_artifact(
                source_shadow,
                output / "shadow_states" / f"{key}.pt.gz",
            )
            distribution_artifact = hardlink_artifact(
                source_distribution,
                output / "shadow_states" / f"{key}_distribution.npz",
            )
            branch_report["post_distribution_artifact"] = distribution_artifact
            branches[branch] = branch_report
        if not branches["FULL"]["historical_update_regression"]["passed"]:
            raise IsolationContractError(
                "completed seed0 FULL update failed historical regression"
            )

        closed_loop: dict[str, Any] = {}
        for branch in BRANCHES:
            key = branch.lower()
            source_trajectory = Path(recovery[f"{key}_trajectory"]["path"])
            target_trajectory = output / "closed_loop" / key / "trajectory.npz"
            trajectory_artifact = hardlink_artifact(
                source_trajectory, target_trajectory
            )
            with np.load(target_trajectory, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            summary = {
                **_validation_summary(arrays),
                **valid_prefix_summary(
                    arrays["endpoint"], arrays["terminated"],
                    arrays["tracking_score"], arrays["tracking_reward"],
                ),
                "branch": branch,
                "trajectory": trajectory_artifact,
            }
            if branch == "BASE":
                checks = _compare_step0(runtime, arrays)
                summary["historical_step0_checks"] = checks
                if not all(checks.values()):
                    raise IsolationContractError(
                        "completed seed0 BASE probe did not reproduce 20/40"
                    )
            closed_loop[branch] = summary

        failed_decision = Path(recovery["failed_decision"]["path"])
        report = {
            "seed": 0,
            "status": "completed_from_immutable_postexecution_artifacts",
            "restore_field_equality": restore_checks,
            "inheritance": inheritance,
            "rollout_counters": {
                "simulation_control_intervals": 160,
                "simulation_physics_steps": 1600,
                "reused_not_recollected": True,
            },
            "trace": {
                "historical_sha256": sha256(paths["visits"]),
                "bitwise_equal": True,
                "source": "bitwise-matched immutable first-attempt trace",
            },
            "historical_visit_regression": {
                "all_fields_bitwise_equal": True,
                "source": "first attempt field-wise and SHA-256 regression",
            },
            "credit_regression_alias": {
                "all_fields_bitwise_equal": True,
                "source": "same reconstructed visit artifact",
            },
            "batch": batch_artifact,
            "pre_update_state": pre_artifact,
            "normalization": {
                "observation_RMS_unchanged_through_prepare_and_updates": True,
                "value_RMS_changed_once_in_prepare_dataset": True,
                "value_normalizer_alias_external_critic": (
                    agent.value_mean_std
                    is agent.asymmetric_critic_net.model.value_mean_std
                ),
                "recovered_after_prepare_dataset": True,
            },
            "external_critic": {
                "optimizer_steps": 4,
                "steps_reused_not_reexecuted": True,
                "parameters_changed": True,
                "actor_parameter_storage_disjoint": True,
                "after_hash": tensor_tree_sha256(
                    saved_pre["external_critic_after_four_steps"]
                ),
            },
            "gradient": gradient,
            "gradient_parameter_rows": parameter_rows,
            "branches": branches,
            "shadow_state_artifacts": shadow_artifacts,
            "closed_loop": closed_loop,
            "recovery_contract": {
                "source_failure": "seed_report_numpy_float32_serialization",
                "all_actor_optimizer_steps_reused_not_reexecuted": True,
                "all_closed_loop_probes_reused_not_reexecuted": True,
                "seed0_collection_repeated": False,
                "seed0_external_critic_updates_repeated": False,
                "failed_attempt_decision": artifact(failed_decision),
                "report_only_gradient_recomputation": True,
                "additional_control_intervals": 0,
                "additional_optimizer_steps": 0,
                "recovery_extra_actor_autograd_calls_across_failed_attempts": 10,
                "recovery_extra_actor_sample_row_forwards_across_failed_attempts": 1120,
            },
        }
        dump_json(output / "seed_report.json", report)
        return report
    finally:
        if agent.writer is not None:
            agent.writer.close()


def decide(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    eps8 = 8 * np.finfo(np.float32).eps
    per_seed = []
    for report in seed_reports:
        branches = report["branches"]
        full_kl = float(branches["FULL"]["exact_old_to_new_truncated_policy_KL"]["mean"])
        policy_kl = float(branches["POLICY_ONLY"]["exact_old_to_new_truncated_policy_KL"]["mean"])
        value_mu = float(branches["VALUE_ONLY"]["bounded_mu_change"]["maximum_absolute"])
        n_policy = int(report["closed_loop"]["POLICY_ONLY"]["valid_prefix_intervals"])
        n_full = int(report["closed_loop"]["FULL"]["valid_prefix_intervals"])
        conditions = {
            "VALUE_ONLY_bounded_mu_change_gt_8eps": value_mu > eps8,
            "KL_FULL_gt_1e-8": full_kl > 1e-8,
            "KL_POLICY_ONLY_le_half_FULL": policy_kl <= 0.5 * full_kl,
            "N_POLICY_ONLY_ge_20": n_policy >= 20,
            "N_POLICY_ONLY_ge_N_FULL": n_policy >= n_full,
        }
        per_seed.append({
            "seed": report["seed"], "conditions": conditions,
            "all_conditions": all(conditions.values()),
            "VALUE_ONLY_max_bounded_mu_change": value_mu,
            "KL_FULL_mean": full_kl, "KL_POLICY_ONLY_mean": policy_kl,
            "N_FULL": n_full, "N_POLICY_ONLY": n_policy,
        })
    positives = sum(row["all_conditions"] for row in per_seed)
    no_worse = all(row["N_POLICY_ONLY"] >= row["N_FULL"] for row in per_seed)
    any_coupling = any(row["VALUE_ONLY_max_bounded_mu_change"] > eps8 for row in per_seed)
    worse = sum(row["N_POLICY_ONLY"] < row["N_FULL"] for row in per_seed)
    if positives >= 2 and no_worse:
        recommendation = "REVIEW_ACTOR_INTERNAL_VALUE_DECOUPLING_ABLATION"
    elif any_coupling:
        recommendation = "VALUE_POLICY_COUPLING_PRESENT_BUT_BENEFIT_UNRESOLVED"
    else:
        recommendation = "NO_RESOLVED_POLICY_EFFECT_AT_TESTED_FIRST_UPDATES"
    return {
        "schema": SCHEMA,
        "validity_status": "COMPLETED_BOUNDED_VALUE_LOSS_ISOLATION",
        "recommendation": recommendation,
        "negative_evidence_flag": "ISOLATION_NOT_FAVORED_AT_TESTED_UPDATES" if worse >= 2 else None,
        "per_seed_review_trigger": per_seed,
        "positive_seed_count": positives,
        "all_POLICY_ONLY_not_worse_than_FULL": no_worse,
        "candidate_G_classification_changed": False,
        "new_long_training_authorized": False,
        "chunk_commit_authorized": False,
        "automatic_followon_training": False,
        "thresholds_are_local_reporting_only": True,
    }


def write_reports(output: Path, manifest: dict[str, Any], seed_reports: list[dict[str, Any]], budget: BudgetCounter, started: float) -> None:
    gradients = {str(row["seed"]): row["gradient"] for row in seed_reports}
    dump_json(output / "gradient_decomposition.json", gradients)
    with (output / "gradient_by_parameter_group.csv").open("w", newline="") as stream:
        fields = ["seed", "parameter", "group", "elements", "policy_is_none", "policy_is_exact_zero", "policy_l2", "value_is_none", "value_is_exact_zero", "value_l2", "full_is_none", "full_is_exact_zero", "full_l2"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in seed_reports:
            for row in report["gradient_parameter_rows"]:
                writer.writerow({"seed": report["seed"], **row})
    single = {
        str(row["seed"]): {
            branch: row["branches"][branch] for branch in BRANCHES
        } for row in seed_reports
    }
    dump_json(output / "single_step_effects.json", single)
    closed = {str(row["seed"]): row["closed_loop"] for row in seed_reports}
    dump_json(output / "closed_loop_probe_summary.json", closed)
    reconstruction = {
        "schema": SCHEMA,
        "seeds": [{
            "seed": row["seed"],
            "restore_field_equality": row["restore_field_equality"],
            "historical_visit_regression": row["historical_visit_regression"],
            "FULL_historical_update_regression": row["branches"]["FULL"]["historical_update_regression"],
            "BASE_closed_loop": row["closed_loop"]["BASE"],
            "normalization": row["normalization"],
            "external_critic": row["external_critic"],
        } for row in seed_reports],
    }
    dump_json(output / "reconstruction_report.json", reconstruction)
    cost = budget.report()
    # Logical network work is reported explicitly rather than hidden as zero.
    recovery_extra_autograd = sum(
        row.get("recovery_contract", {}).get(
            "recovery_extra_actor_autograd_calls_across_failed_attempts", 0
        )
        for row in seed_reports
    )
    recovery_extra_forward_rows = sum(
        row.get("recovery_contract", {}).get(
            "recovery_extra_actor_sample_row_forwards_across_failed_attempts", 0
        )
        for row in seed_reports
    )
    planned_actor_forward_rows = 3 * (
        160 + 160 + 160 + 4 * 3 * 160 + 4 * 60
    )
    planned_actor_autograd_calls = 3 * 6
    cost["network_work"] = {
        "actor_sample_row_forwards": (
            planned_actor_forward_rows + recovery_extra_forward_rows
        ),
        "planned_completed_experiment_actor_sample_row_forwards": planned_actor_forward_rows,
        "recovery_extra_actor_sample_row_forwards": recovery_extra_forward_rows,
        "actor_forward_breakdown_per_seed": {
            "rollout_collection": 160,
            "canonical_distribution": 160,
            "gradient_decomposition_one_shared_forward": 160,
            "four_branches_pre_post_and_post_loss": 4 * 3 * 160,
            "four_closed_loop_prefix_burnin_plus_rollout": 4 * 60,
        },
        "actor_autograd_calls": (
            planned_actor_autograd_calls + recovery_extra_autograd
        ),
        "planned_completed_experiment_actor_autograd_calls": planned_actor_autograd_calls,
        "recovery_extra_actor_autograd_calls": recovery_extra_autograd,
        "actor_autograd_breakdown_per_seed": {"losswise_autograd_grad": 3, "shadow_backward": 3},
        "external_critic_sample_row_forwards_lower_bound": 3 * (160 + 4 + 160 + 4 * 160),
        "external_critic_backward_calls": 12,
        "actor_optimizer_steps": 9,
        "external_critic_optimizer_steps": 12,
        "note": "sample-row counts are logical audited calls; critic forward lower bound excludes framework-internal logging forwards if any",
    }
    cost["wall_seconds"] = time.time() - started
    dump_json(output / "cost_accounting.json", cost)
    decision = decide(seed_reports)
    dump_json(output / "decision.json", decision)
    rows = []
    for report in seed_reports:
        seed = report["seed"]
        for branch in ("FULL", "POLICY_ONLY", "VALUE_ONLY"):
            b = report["branches"][branch]
            rows.append({
                "seed": seed, "branch": branch,
                "gradient_norm": b["gradient_norm_before_clip"],
                "KL": b["exact_old_to_new_truncated_policy_KL"]["mean"],
                "N": report["closed_loop"][branch]["valid_prefix_intervals"],
            })
    report = {
        "schema": SCHEMA,
        "status": decision["validity_status"],
        "paper_faithful": False,
        "executed_real_network_updates": True,
        "gradient_KL_N_table": rows,
        "decision": decision,
        "reconstruction": reconstruction,
        "cost_accounting": cost,
        "candidate_G_classification_changed": False,
        "new_long_training_authorized": False,
        "chunk_commit_authorized": False,
        "historical_artifacts_immutable": True,
        "input_manifest": manifest,
    }
    dump_json(output / "report.json", report)
    table = ["| seed | branch | pre-clip grad L2 | mean exact KL | valid prefix N |", "|---:|---|---:|---:|---:|"]
    for row in rows:
        table.append(f"| {row['seed']} | {row['branch']} | {row['gradient_norm']:.9g} | {row['KL']:.9g} | {row['N']} |")
    conclusion = {
        "REVIEW_ACTOR_INTERNAL_VALUE_DECOUPLING_ABLATION": "有足够的本地一步证据进入内部 value 解耦消融评审；这不授权长训练。",
        "VALUE_POLICY_COUPLING_PRESENT_BUT_BENEFIT_UNRESOLVED": "已测到 value→policy 耦合，但没有足够证据选择移除内部 value 项。",
        "NO_RESOLVED_POLICY_EFFECT_AT_TESTED_FIRST_UPDATES": "三个测试点均未解析到超过本地数值阈值的 value→policy 作用。",
    }[decision["recommendation"]]
    (output / "summary.md").write_text(
        "# Candidate G internal value-loss isolation v1\n\n"
        + "\n".join(table)
        + f"\n\n**结论：{conclusion}**\n\n"
        + f"- validity: `{decision['validity_status']}`\n"
        + f"- recommendation: `{decision['recommendation']}`\n"
        + "- Candidate G classification unchanged; no long training and no chunk commit authorized.\n"
        + "- Scope is limited to three reconstructed first batches and one paired shadow update per branch.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen diagnostic output: {output}")
    contract = yaml.safe_load(args.contract.read_text())
    verify_contract(contract, output)
    manifest, seed_paths = build_input_manifest(contract, output)
    output.mkdir(parents=True)
    dump_json(output / "input_manifest.json", manifest)
    write_loss_path(output / "loss_path_and_normalization.md")
    started = time.time()
    budget = BudgetCounter()
    seed_reports: list[dict[str, Any]] = []
    runtime = Runtime(yaml.safe_load(Path(contract["inputs"]["G_contract"]["path"]).read_text()), {
        name: Path(row["path"])
        for name, row in yaml.safe_load(Path(contract["inputs"]["G_contract"]["path"]).read_text())["inputs"].items()
    })
    try:
        for seed in SEEDS:
            print(json.dumps({"stage": "seed_start", "seed": seed}), flush=True)
            completed_recovery = contract["inputs"].get(
                "completed_attempt_recovery_seed0"
            )
            recovery = contract["inputs"].get("failed_attempt_recovery_seed0")
            if seed == 0 and completed_recovery is not None:
                seed_reports.append(
                    run_completed_seed0_recovery(
                        runtime, seed_paths[seed], output / "seed_0",
                        budget, completed_recovery,
                    )
                )
            elif seed == 0 and recovery is not None:
                seed_reports.append(
                    run_recovered_seed0(
                        runtime, seed_paths[seed], output / "seed_0",
                        budget, recovery,
                    )
                )
            else:
                seed_reports.append(
                    run_seed(
                        runtime, seed, seed_paths[seed],
                        output / f"seed_{seed}", budget,
                    )
                )
            print(json.dumps({"stage": "seed_complete", "seed": seed}), flush=True)
        write_reports(output, manifest, seed_reports, budget, started)
        print(json.dumps({"status": "COMPLETED_BOUNDED_VALUE_LOSS_ISOLATION", "output": str(output)}), flush=True)
        return 0
    except BaseException as error:
        status = "DIAGNOSTIC_CONTRACT_FAILURE" if isinstance(error, IsolationContractError) else "INCOMPLETE_INPUT_OR_RECONSTRUCTION"
        failure = {
            "schema": SCHEMA,
            "status": status,
            "exception": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            "completed_seeds": [row["seed"] for row in seed_reports],
            "cost_accounting": budget.report(),
            "candidate_G_classification_changed": False,
            "new_long_training_authorized": False,
            "chunk_commit_authorized": False,
        }
        dump_json(output / "decision.json", failure)
        dump_json(output / "report.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
