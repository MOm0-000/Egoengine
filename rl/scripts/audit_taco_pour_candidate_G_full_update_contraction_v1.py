#!/usr/bin/env python3
"""Execute the bounded Candidate-G saved FULL-update contraction diagnostic."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

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

from run_taco_pour_candidate_D_chunk_local_v1 import _validation_summary  # noqa: E402
from run_taco_pour_candidate_G_policy_warm_start_v1 import (  # noqa: E402
    Runtime,
    _validation_arrays,
)
from video_to_spider.rl.algorithmic_training import exact_truncated_normal_kl  # noqa: E402
from video_to_spider.rl.algorithmic_training_v6 import support_anchored_bounded_mean  # noqa: E402
from video_to_spider.rl.full_update_contraction import (  # noqa: E402
    ALPHAS,
    LABELS,
    NEW_ALPHAS,
    BudgetLedger,
    ContractionContractError,
    decide_matrix,
    interpolate_actor_state,
    prefix_summary,
)
from video_to_spider.rl.state_feasible_truncated_gaussian import (  # noqa: E402
    truncated_normal_entropy,
    truncated_normal_log_prob,
)
from video_to_spider.rl.value_loss_isolation import tensor_tree_sha256  # noqa: E402


SCHEMA = "taco_pour_candidate_G_full_update_contraction_v1"
EXPECTED_BASE = "5f243e9864da6a7b71ce2cdc4a304aa7d5996496"
CONTRACT = ROOT / "configs/taco_pour_candidate_G_full_update_contraction_v1.yaml"
DEFAULT_OUTPUT = Path("/data_all/zzx/3.2RL/runs/taco_pour_candidate_G_full_update_contraction_v1")
SEEDS = (0, 1, 2)
ANCHOR_ORDER = tuple((seed, alpha) for seed in SEEDS for alpha in (0.0, 1.0))
NEW_ORDER = tuple((seed, alpha) for seed in SEEDS for alpha in NEW_ALPHAS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value) and value.numel() == 1:
        return value.detach().cpu().item()
    raise TypeError(type(value).__name__)


def load_torch_gzip(path: Path) -> Any:
    with gzip.open(path, "rb") as stream:
        return torch.load(io.BytesIO(stream.read()), map_location="cpu", weights_only=False)


def save_torch_gzip(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = io.BytesIO()
    torch.save(value, raw)
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0) as stream:
        stream.write(raw.getvalue())
    return artifact(path)


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    return {
        "head": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "status_porcelain": run("git", "status", "--porcelain"),
    }


def verify_file(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    path = Path(row["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{role}: {path}")
    observed = artifact(path)
    if observed["sha256"] != row["sha256"]:
        raise ContractionContractError(f"hash mismatch for {role}")
    if "bytes" in row and int(observed["bytes"]) != int(row["bytes"]):
        raise ContractionContractError(f"size mismatch for {role}")
    return {"role": role, "reported": dict(row), "resolved": observed}


def resolve_inputs(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, Path]], dict[str, Path], dict[str, Any]]:
    parent = Path(contract["parent"]["root"])
    rows: list[dict[str, Any]] = []
    summary_paths: dict[str, Path] = {}
    for name in contract["parent"]["required_summary_files"]:
        path = parent / name
        if not path.is_file():
            raise FileNotFoundError(path)
        summary_paths[name] = path
        rows.append({"role": f"parent:{name}", "resolved": artifact(path)})
    parent_manifest = json.loads(summary_paths["input_manifest.json"].read_text())
    historical_by_role = {row["role"]: row for row in parent_manifest["historical_inputs"]}
    runtime_rows = {
        row["role"].split(":", 1)[1]: row
        for row in parent_manifest["runtime_inputs"]
        if row["role"].startswith("G_runtime:")
    }
    runtime_paths: dict[str, Path] = {}
    for name, row in runtime_rows.items():
        rows.append(verify_file(row, f"runtime:{name}"))
        runtime_paths[name] = Path(row["path"])
    g_contract_row = historical_by_role["G_contract"]
    rows.append(verify_file(g_contract_row, "parent:G_contract"))
    g_contract = yaml.safe_load(Path(g_contract_row["path"]).read_text())

    seeds: dict[int, dict[str, Path]] = {}
    for seed in SEEDS:
        report_path = parent / f"seed_{seed}/seed_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        seed_report = json.loads(report_path.read_text())
        rows.append({"role": f"seed{seed}:seed_report", "resolved": artifact(report_path)})
        roles = {
            "batch": seed_report["batch"],
            "pre_update_state": seed_report["pre_update_state"],
            "BASE_actor": seed_report["shadow_state_artifacts"]["BASE"],
            "FULL_actor": seed_report["shadow_state_artifacts"]["FULL"],
            "BASE_distribution": seed_report["branches"]["BASE"]["post_distribution_artifact"],
            "FULL_distribution": seed_report["branches"]["FULL"]["post_distribution_artifact"],
            "BASE_trajectory": seed_report["closed_loop"]["BASE"]["trajectory"],
            "FULL_trajectory": seed_report["closed_loop"]["FULL"]["trajectory"],
        }
        paths: dict[str, Path] = {"seed_report": report_path}
        for role, row in roles.items():
            rows.append(verify_file(row, f"seed{seed}:{role}"))
            paths[role] = Path(row["path"])
        visits_row = historical_by_role[f"seed{seed}:visits"]
        rows.append(verify_file(visits_row, f"seed{seed}:historical_epoch1_visits"))
        paths["visits"] = Path(visits_row["path"])
        seeds[seed] = paths
    if sha256(runtime_paths["source_boundary"]) != contract["input_resolution"]["committed_endpoint40_sha256"]:
        raise ContractionContractError("committed endpoint40 identity changed")
    if sha256(runtime_paths["boundary_context"]) != contract["input_resolution"]["boundary_context_sha256"]:
        raise ContractionContractError("endpoint40 boundary context changed")
    manifest = {
        "schema": SCHEMA,
        "base": git_state(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "contract": artifact(CONTRACT),
        "resolved_inputs": rows,
        "all_required_bytes_verified_before_physics": True,
        "no_TRASH_discovery_or_substitution": True,
    }
    return manifest, seeds, runtime_paths, g_contract


def distribution(values: torch.Tensor) -> dict[str, Any]:
    array = values.detach().double().reshape(-1).cpu().numpy()
    if array.size == 0:
        return {"count": 0, "minimum": None, "mean": None, "median": None, "p95": None, "maximum": None}
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def evaluate_fixed_batch(agent: Any, inputs: Mapping[str, Any], reset_states: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    """One exact no-grad Candidate-G full-batch recurrent forward."""
    actions = inputs["actions"]
    low, high = inputs["action_lows"], inputs["action_highs"]
    obs = agent._preproc_obs(inputs["obs"])
    normalized = agent.model.norm_obs(obs, update_stats=False)
    obs_by_world = normalized.reshape(4, 40, *normalized.shape[1:])
    dones = inputs["dones"].reshape(4, 40).bool()
    initial_states = inputs["rnn_states"]
    blocks, sequence = 10, 4
    raw_rows = [[None] * 40 for _ in range(4)]
    logstd_rows = [[None] * 40 for _ in range(4)]
    value_rows = [[None] * 40 for _ in range(4)]
    last_states = None
    agent.model.eval()
    for block in range(blocks):
        index = torch.as_tensor([world * blocks + block for world in range(4)], dtype=torch.long)
        states = [state.index_select(1, index) for state in initial_states]
        for local in range(sequence):
            t = block * sequence + local
            done = dones[:, t]
            if bool(done.any().item()):
                states = [
                    torch.where(done.reshape(1, 4, 1), reset.to(state.device), state)
                    for state, reset in zip(states, reset_states, strict=True)
                ]
            raw, logstd, value, states = agent.model.a2c_network({
                "obs": obs_by_world[:, t], "rnn_states": states,
            })
            for world in range(4):
                raw_rows[world][t] = raw[world]
                logstd_rows[world][t] = logstd[world]
                value_rows[world][t] = value[world]
        last_states = states
    def flatten(rows):
        return torch.stack([value for row in rows for value in row])
    raw = flatten(raw_rows)
    sigma = torch.exp(flatten(logstd_rows))
    values = flatten(value_rows)
    mu = support_anchored_bounded_mean(raw, low, high)
    neglogp = -truncated_normal_log_prob(
        actions, mu, sigma, low, high,
        minimum_mass=agent.distribution_spec.minimum_normalization_mass,
    ).sum(dim=-1)
    entropy = truncated_normal_entropy(
        mu, sigma, low, high,
        minimum_mass=agent.distribution_spec.minimum_normalization_mass,
    ).sum(dim=-1)
    return {"raw_location": raw, "mu": mu, "sigma": sigma, "values": values,
            "neglogp": neglogp, "entropy": entropy, "rnn_states": last_states}


def load_visits(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def source_masks(visits: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    mapping = visits["ppo_flat_index"].astype(int)
    if sorted(mapping.tolist()) != list(range(160)):
        raise ContractionContractError("ppo_flat_index is not one-to-one")
    source = np.empty(160, dtype=np.int32)
    terminated = np.empty(160, dtype=bool)
    score = np.empty(160, dtype=np.float32)
    source[mapping] = visits["source_endpoint"]
    terminated[mapping] = visits["tracking_terminated"]
    score[mapping] = visits["tool_objective_score"]
    return {
        "source40_56": (source >= 40) & (source <= 56),
        "source57_59": (source >= 57) & (source <= 59),
        "source60": source == 60,
        "source61": source == 61,
        "source60_outcome61_pass": (source == 60) & ~terminated & (score <= 1.0),
        "source60_outcome61_fail": (source == 60) & (terminated | (score > 1.0)),
    }


def fixed_metrics(agent: Any, inputs: Mapping[str, Any], canonical: Mapping[str, torch.Tensor], current: Mapping[str, torch.Tensor], masks: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ratio = torch.exp(canonical["neglogp"] - current["neglogp"])
    kl = exact_truncated_normal_kl(
        canonical["mu"], canonical["sigma"], current["mu"], current["sigma"],
        inputs["action_lows"], inputs["action_highs"],
    )
    advantages = inputs["advantages"]
    clipped = torch.clamp(ratio, 1.0 - agent.cfg.e_clip, 1.0 + agent.cfg.e_clip)
    policy = torch.max(-advantages * ratio, -advantages * clipped).mean()
    baseline, returns = inputs["old_values"], inputs["returns"]
    value_clipped = baseline + (current["values"] - baseline).clamp(-agent.cfg.e_clip, agent.cfg.e_clip)
    value_unweighted = torch.max(
        (current["values"] - returns).square(),
        (value_clipped - returns).square(),
    ).squeeze(1).mean()
    weighted = value_unweighted * (0.5 * float(agent.cfg.critic_coef))
    true_clip = (((advantages > 0) & (ratio > 1.0 + agent.cfg.e_clip)) |
                 ((advantages < 0) & (ratio < 1.0 - agent.cfg.e_clip)))
    metrics = {
        "exact_KL": distribution(kl),
        "joint_ratio": distribution(ratio),
        "ratio_outside_0p8_1p2_fraction": float(((ratio < 0.8) | (ratio > 1.2)).float().mean()),
        "true_clip_active_fraction": float(true_clip.float().mean()),
        "bounded_mu_change_max_abs": float((current["mu"] - canonical["mu"]).abs().max()),
        "bounded_mu_change_RMS": float((current["mu"] - canonical["mu"]).square().mean().sqrt()),
        "sigma_change_max_abs": float((current["sigma"] - canonical["sigma"]).abs().max()),
        "sigma_change_RMS": float((current["sigma"] - canonical["sigma"]).square().mean().sqrt()),
        "loss": {"policy": float(policy), "internal_value_unweighted": float(value_unweighted),
                 "internal_value_weighted": float(weighted), "full": float(policy + weighted)},
        "subgroups": {},
    }
    for name, mask_np in masks.items():
        mask = torch.as_tensor(mask_np, dtype=torch.bool)
        metrics["subgroups"][name] = {
            "count": int(mask.sum()),
            "exact_KL": distribution(kl[mask]),
            "joint_ratio": distribution(ratio[mask]),
            "bounded_mu_max_abs_change": (
                None if not bool(mask.any()) else float((current["mu"][mask] - canonical["mu"][mask]).abs().max())
            ),
        }
    arrays = {
        "raw_location": current["raw_location"].detach().cpu().numpy(),
        "bounded_mu": current["mu"].detach().cpu().numpy(),
        "sigma": current["sigma"].detach().cpu().numpy(),
        "neglogp": current["neglogp"].detach().cpu().numpy(),
        "values": current["values"].detach().cpu().numpy(),
        "ratio": ratio.detach().cpu().numpy(),
        "exact_kl": kl.detach().cpu().numpy(),
    }
    return metrics, arrays


def compare_distribution(current: Mapping[str, np.ndarray], path: Path, metrics: Mapping[str, Any], parent_branch: Mapping[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as old:
        errors = {}
        for name in ("raw_location", "bounded_mu", "sigma", "neglogp", "ratio", "exact_kl"):
            errors[name] = float(np.max(np.abs(current[name].astype(np.float64) - old[name].astype(np.float64))))
    array_pass = (
        max(errors[name] for name in ("raw_location", "bounded_mu", "sigma"))
        <= 8 * np.finfo(np.float32).eps
        and max(errors[name] for name in ("neglogp", "ratio")) <= 1.0e-4
    )
    scalar_errors = {}
    for local, parent in (("policy", "policy"), ("internal_value_unweighted", "internal_value_unweighted"),
                          ("internal_value_weighted", "internal_value_weighted"), ("full", "full")):
        scalar_errors[local] = abs(float(metrics["loss"][local]) - float(parent_branch["loss_after"][parent]))
    scalar_pass = all(
        error <= 1.0e-7 + 1.0e-5 * max(1.0, abs(float(parent_branch["loss_after"][name])))
        for name, error in scalar_errors.items()
    )
    return {"passed": array_pass and scalar_pass, "array_max_abs_errors": errors,
            "scalar_absolute_errors": scalar_errors, "array_passed": array_pass,
            "scalar_passed": scalar_pass}


def compare_npz_bitwise(left: Mapping[str, np.ndarray], path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as old:
        names = sorted(set(left) | set(old.files))
        fields = {}
        for name in names:
            present = name in left and name in old
            equal = present and left[name].dtype == old[name].dtype and left[name].shape == old[name].shape and left[name].tobytes() == old[name].tobytes()
            fields[name] = {"present_both": present, "bitwise_equal": equal}
    return {"all_fields_bitwise_equal": all(row["bitwise_equal"] for row in fields.values()), "fields": fields}


@contextmanager
def no_training_guard():
    original_tensor_backward = torch.Tensor.backward
    original_autograd_backward = torch.autograd.backward
    optimizer_classes = [torch.optim.Optimizer, torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD]
    originals = {cls: cls.step for cls in optimizer_classes}
    def forbidden(*args, **kwargs):
        raise ContractionContractError("backward/optimizer step forbidden")
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    for cls in optimizer_classes:
        cls.step = forbidden
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        torch.set_grad_enabled(previous)
        torch.Tensor.backward = original_tensor_backward
        torch.autograd.backward = original_autograd_backward
        for cls, step in originals.items():
            cls.step = step


def condition_key(seed: int, alpha: float) -> str:
    return f"seed_{seed}_{LABELS[alpha].lower()}"


def run_condition(runtime: Runtime, seed: int, alpha: float, paths: Mapping[str, Path], output: Path, ledger: BudgetLedger, execution: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    label = LABELS[alpha]
    condition = output / f"seed_{seed}" / label.lower()
    condition.mkdir(parents=True)
    entry = {"seed": seed, "alpha": alpha, "label": label, "status": "started", "started_unix": time.time()}
    execution.append(entry)
    dump_json(output / "execution_ledger.json", {"conditions": execution, "budget": ledger.as_dict()})
    batch_payload = load_torch_gzip(paths["batch"])
    pre = load_torch_gzip(paths["pre_update_state"])
    base_shadow = load_torch_gzip(paths["BASE_actor"])
    full_shadow = load_torch_gzip(paths["FULL_actor"])
    parent_report = json.loads(paths["seed_report"].read_text())

    # A fresh inference wrapper is used only to recover the exact named-parameter set.
    env, agent, _ = runtime.make_agent(seed=seed, output=condition / "fixed_batch_agent", training=False, trace=False)
    try:
        parameter_names = [name for name, _ in agent.model.named_parameters()]
        if tensor_tree_sha256(base_shadow["actor"]) != tensor_tree_sha256(pre["actor"]):
            raise ContractionContractError(f"seed {seed} BASE != pre-update actor")
        actor, interpolation = interpolate_actor_state(
            base_shadow["actor"], full_shadow["actor"], parameter_names, alpha,
        )
        agent.model.load_state_dict(actor, strict=True)
        agent._observation_normalization_version = int(pre["observation_normalization_version"])
        actor_artifact = save_torch_gzip(condition / "diagnostic_actor.pt.gz", {
            "schema": "egoengine_diagnostic_actor_interpolation_v1",
            "seed": seed, "alpha": alpha, "label": label,
            "actor": actor,
            "eligible_for_training_resume": False,
            "eligible_for_chunk_commit": False,
            "optimizer_state": None,
        })
        inputs = batch_payload["optimizer_input"]
        reset = tuple(state.detach().cpu().clone() for state in batch_payload["boundary_reset_rnn_states"])
        current = evaluate_fixed_batch(agent, inputs, reset)
        ledger.add_batch(160)
        masks = source_masks(load_visits(paths["visits"]))
        canonical = {name: value.detach().cpu() for name, value in pre["canonical_old"].items()}
        batch_metrics, batch_arrays = fixed_metrics(agent, inputs, canonical, current, masks)
        np.savez_compressed(condition / "fixed_batch_distribution.npz", **batch_arrays)
        batch_artifact_out = artifact(condition / "fixed_batch_distribution.npz")
        anchor_distribution = None
        if alpha in (0.0, 1.0):
            branch = "BASE" if alpha == 0.0 else "FULL"
            anchor_distribution = compare_distribution(
                batch_arrays, paths[f"{branch}_distribution"], batch_metrics,
                parent_report["branches"][branch],
            )
            if not anchor_distribution["passed"]:
                raise ContractionContractError(f"seed {seed} {branch} distribution anchor failed")

        validation_payload = {
            "actor": actor,
            "observation_normalization_version": int(pre["observation_normalization_version"]),
            "_validation_dir": str(condition / "closed_loop_agent"),
        }
        arrays = _validation_arrays(runtime, validation_payload, seed=seed)
        ledger.add_rollout(40, 400)
        trajectory_path = condition / "trajectory.npz"
        np.savez_compressed(trajectory_path, **arrays)
        prefix = prefix_summary(arrays["endpoint"], arrays["terminated"], arrays["tracking_score"], arrays["tracking_reward"])
        closed = {**_validation_summary(arrays), **prefix, "trajectory": artifact(trajectory_path)}
        anchor_trajectory = None
        if alpha in (0.0, 1.0):
            branch = "BASE" if alpha == 0.0 else "FULL"
            anchor_trajectory = compare_npz_bitwise(arrays, paths[f"{branch}_trajectory"])
            expected = {0.0: 20, 1.0: (20 if seed == 2 else 14)}[alpha]
            anchor_trajectory["expected_N"] = expected
            anchor_trajectory["observed_N"] = prefix["N"]
            anchor_trajectory["passed"] = anchor_trajectory["all_fields_bitwise_equal"] and prefix["N"] == expected
            if not anchor_trajectory["passed"]:
                raise ContractionContractError(f"seed {seed} {branch} closed-loop anchor failed")
        report = {
            "seed": seed, "alpha": alpha, "label": label,
            "actor": actor_artifact,
            "actor_state_hash": tensor_tree_sha256(actor),
            "interpolation": interpolation,
            "fixed_batch": {**batch_metrics, "artifact": batch_artifact_out},
            "closed_loop": closed,
            "anchor_distribution_regression": anchor_distribution,
            "anchor_trajectory_regression": anchor_trajectory,
        }
        dump_json(condition / "condition_report.json", report)
        entry.update(status="completed", completed_unix=time.time(),
                     actor_sha256=actor_artifact["sha256"], trajectory_sha256=closed["trajectory"]["sha256"],
                     control_intervals=40, physics_steps=400)
        dump_json(condition / "COMPLETED.json", entry)
        dump_json(output / "execution_ledger.json", {"conditions": execution, "budget": ledger.as_dict()})
        return report, interpolation
    finally:
        if agent.writer is not None:
            agent.writer.close()


def write_csvs(output: Path, reports: list[dict[str, Any]]) -> None:
    batch_fields = ["seed", "alpha", "label", "KL_mean", "KL_p95", "KL_max", "ratio_mean", "ratio_outside", "true_clip", "mu_RMS", "sigma_RMS", "policy_loss", "value_loss", "full_loss"]
    with (output / "fixed_batch_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=batch_fields); writer.writeheader()
        for row in reports:
            fixed = row["fixed_batch"]
            writer.writerow({"seed": row["seed"], "alpha": row["alpha"], "label": row["label"],
                "KL_mean": fixed["exact_KL"]["mean"], "KL_p95": fixed["exact_KL"]["p95"], "KL_max": fixed["exact_KL"]["maximum"],
                "ratio_mean": fixed["joint_ratio"]["mean"], "ratio_outside": fixed["ratio_outside_0p8_1p2_fraction"],
                "true_clip": fixed["true_clip_active_fraction"], "mu_RMS": fixed["bounded_mu_change_RMS"],
                "sigma_RMS": fixed["sigma_change_RMS"], "policy_loss": fixed["loss"]["policy"],
                "value_loss": fixed["loss"]["internal_value_weighted"], "full_loss": fixed["loss"]["full"]})
    closed_fields = ["seed", "alpha", "label", "N", "first_failure_endpoint", "retained_base_intervals", "lost_base_intervals", "added_intervals_beyond_base", "valid_prefix_reward", "first20_reward", "endpoint60_score"]
    with (output / "closed_loop_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=closed_fields); writer.writeheader()
        for row in reports:
            closed = row["closed_loop"]
            writer.writerow({"seed": row["seed"], "alpha": row["alpha"], "label": row["label"],
                "N": closed["N"], "first_failure_endpoint": closed["first_failure_endpoint"],
                "retained_base_intervals": closed["retained_base_intervals"], "lost_base_intervals": closed["lost_base_intervals"],
                "added_intervals_beyond_base": closed["added_intervals_beyond_base"],
                "valid_prefix_reward": closed["valid_prefix_tracking_reward_sum"],
                "first20_reward": closed["first20_or_valid_tracking_reward_sum"],
                "endpoint60_score": closed["endpoint60"]["score"] if closed["N"] >= 20 else "post_failure_diagnostic"})


def matrix_markdown(reports: list[dict[str, Any]], metric) -> str:
    lookup = {(row["seed"], row["alpha"]): metric(row) for row in reports}
    header = "| seed | BASE 0 | FULL 1 | 1/2 | 1/4 | 1/8 |\n|---:|---:|---:|---:|---:|---:|"
    lines = [header]
    for seed in SEEDS:
        lines.append("| " + str(seed) + " | " + " | ".join(str(lookup[(seed, alpha)]) for alpha in ALPHAS) + " |")
    return "\n".join(lines)


def finalize(output: Path, reports: list[dict[str, Any]], interpolations: list[dict[str, Any]], ledger: BudgetLedger, manifest: Mapping[str, Any], implementation: Mapping[str, Any]) -> dict[str, Any]:
    rows = [{"seed": row["seed"], "alpha": row["alpha"], "label": row["label"], **row["closed_loop"]} for row in reports]
    decision_core = decide_matrix(rows)
    decision = {
        "schema": SCHEMA,
        "validity": "COMPLETED_BOUNDED_FULL_UPDATE_CONTRACTION",
        **decision_core,
        "diagnostic_strict_40_events": [
            {"seed": row["seed"], "alpha": row["alpha"]} for row in rows if row["N"] == 40
        ],
        "candidate_G_classification_changed": False,
        "candidate_G_classification": "G3_NO_PREDECLARED_SUSTAINED_EXTENSION",
        "new_training_authorized": False,
        "automatic_followon_experiment": False,
        "chunk_commit_authorized": False,
    }
    dump_json(output / "decision.json", decision)
    dump_json(output / "condition_matrix.json", {"rows": rows, **decision_core})
    dump_json(output / "parameter_interpolation_audit.json", {"conditions": interpolations})
    anchors = [{"seed": row["seed"], "alpha": row["alpha"],
                "distribution": row["anchor_distribution_regression"],
                "trajectory": row["anchor_trajectory_regression"]}
               for row in reports if row["alpha"] in (0.0, 1.0)]
    dump_json(output / "anchor_regression.json", {"passed": all(a["distribution"]["passed"] and a["trajectory"]["passed"] for a in anchors), "anchors": anchors})
    costs = {**ledger.as_dict(), "planned_actor_sample_rows": 3300,
             "all_planned_conditions_completed": len(reports) == 15,
             "new_training_performed": False, "autograd_or_optimizer_used": False}
    dump_json(output / "cost_accounting.json", costs)
    report = {"schema": SCHEMA, "status": decision["validity"], "decision": decision,
              "matrix": rows, "cost": costs, "input_manifest": artifact(output / "resolved_input_manifest.json"),
              "implementation_manifest": implementation}
    dump_json(output / "report.json", report)
    write_csvs(output, reports)
    n_table = matrix_markdown(reports, lambda row: row["closed_loop"]["N"])
    kl_table = matrix_markdown(reports, lambda row: f"{row['fixed_batch']['exact_KL']['mean']:.6g}")
    summary = f"""# Candidate G saved FULL-update contraction v1

Validity: `{decision['validity']}`  
Classification: `{decision['classification']}`

## Valid-prefix intervals N

{n_table}

## Fixed-batch exact KL mean

{kl_table}

Common preservation set C: `{decision['common_preservation_set_C']}`  
Common extension set E: `{decision['common_extension_set_E']}`

## Cost

- fixed-batch forward rows: {costs['fixed_batch_actor_sample_rows']} in {costs['fixed_batch_forwards']} forwards
- deterministic CPU rollouts: {costs['rollouts']}
- control intervals / physics steps: {costs['control_intervals']} / {costs['physics_steps']}
- burn-in / closed-loop actor rows: {costs['burnin_actor_sample_rows']} / {costs['closed_loop_actor_sample_rows']}
- training samples, backward calls, optimizer steps: 0

This is a local, non-paper-faithful parameter intervention. Candidate G remains
`G3_NO_PREDECLARED_SUSTAINED_EXTENSION`; endpoint40→60 remains uncommitted.
"""
    (output / "summary.md").write_text(summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    if (contract.get("schema") != SCHEMA or contract.get("status") != "authorized_bounded_full_update_contraction"
            or contract.get("execution_authorized_now") is not True or contract.get("paper_faithful") is not False
            or contract.get("scope", {}).get("training_allowed") is not False
            or contract.get("scope", {}).get("chunk_commit_authorized") is not False
            or contract.get("conditions", {}).get("new_alphas") != [0.5, 0.25, 0.125]
            or Path(contract["output"]["root"]).resolve() != args.output.resolve()):
        raise ContractionContractError("authorization contract changed")
    state = git_state()
    if state["head"] != EXPECTED_BASE or state["branch"] != "3.2RL":
        raise ContractionContractError(f"unexpected base: {state}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    try:
        manifest, seeds, runtime_paths, g_contract = resolve_inputs(contract)
        dump_json(args.output / "resolved_input_manifest.json", manifest)
        (args.output / "contract_snapshot.yaml").write_text(args.contract.read_text())
        implementation = {
            "files": {
                "runner": artifact(Path(__file__).resolve()),
                "helper": artifact(ROOT / "src/video_to_spider/rl/full_update_contraction.py"),
                "tests": artifact(ROOT / "tests/test_candidate_G_full_update_contraction.py"),
                "policy_warm_start": artifact(ROOT / "src/video_to_spider/rl/policy_warm_start.py"),
                "validation_runner": artifact(ROOT / "scripts/run_taco_pour_candidate_G_policy_warm_start_v1.py"),
                "distribution": artifact(ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py"),
            },
            "hashes_frozen_before_model_or_physics_execution": True,
        }
        frozen_hashes = contract.get("implementation_contract", {}).get("hashes", {})
        observed_hashes = {name: row["sha256"] for name, row in implementation["files"].items()}
        if contract.get("implementation_contract", {}).get("status") != "hashes_frozen_before_execution" or frozen_hashes != observed_hashes:
            raise ContractionContractError("implementation hashes are not frozen")
        dump_json(args.output / "implementation_manifest.json", implementation)

        runtime = Runtime(g_contract, runtime_paths)
        ledger = BudgetLedger()
        execution: list[dict[str, Any]] = []
        reports: list[dict[str, Any]] = []
        interpolations: list[dict[str, Any]] = []
        with no_training_guard():
            for seed, alpha in ANCHOR_ORDER:
                report, interpolation = run_condition(runtime, seed, alpha, seeds[seed], args.output, ledger, execution)
                reports.append(report); interpolations.append({"seed": seed, **interpolation})
            # No contraction condition starts until all six endpoint anchors pass.
            for seed, alpha in NEW_ORDER:
                report, interpolation = run_condition(runtime, seed, alpha, seeds[seed], args.output, ledger, execution)
                reports.append(report); interpolations.append({"seed": seed, **interpolation})
        # Canonical display order is seed x [BASE,FULL,HALF,QUARTER,EIGHTH].
        reports.sort(key=lambda row: (row["seed"], ALPHAS.index(row["alpha"])))
        interpolations.sort(key=lambda row: (row["seed"], ALPHAS.index(row["alpha"])))
        finalize(args.output, reports, interpolations, ledger, manifest, implementation)
    except Exception as exc:
        failure = {"schema": SCHEMA, "validity": "PARTIAL_EXECUTION",
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                   "chunk_commit_authorized": False, "new_training_authorized": False}
        dump_json(args.output / "decision.json", failure)
        raise


if __name__ == "__main__":
    main()
