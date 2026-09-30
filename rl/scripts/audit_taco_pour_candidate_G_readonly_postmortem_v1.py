#!/usr/bin/env python3
"""Pure-data, zero-simulation postmortem for Candidate G.

This module intentionally imports no project runtime, agent, simulator, or
network code.  It reads immutable Candidate-G arrays, reproduces their joins
and truncated-normal likelihood arithmetic, and emits bounded audit reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import torch
import yaml


SCHEMA = "taco_pour_candidate_G_readonly_postmortem_v1"
EXPECTED_BASE = "577d0f08172d011cefc53b87215470aaf722f1da"
SEEDS = (0, 1, 2)
EPOCHS = 250
WORLDS = 4
HORIZON = 40
SAMPLES = WORLDS * HORIZON
GAMMA = np.float32(0.998)
GAE_TAU = np.float32(0.95)
NUMERICAL_LIMIT = 1.0e-4
EPOCH_BINS = ((1, 10), (11, 62), (63, 125), (126, 188), (189, 250))


class ContractFailure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def truncated_joint_logprob(
    action: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    minimum_mass: float = 1.0e-12,
) -> np.ndarray:
    """Match the frozen torch float64-CDF/float32-reduction likelihood."""
    arrays = [np.asarray(x, np.float32) for x in (action, mu, sigma, low, high)]
    if len({x.shape for x in arrays}) != 1 or arrays[0].ndim != 2:
        raise ContractFailure("truncated-normal arrays must share a 2-D shape")
    a, m, s, lo, hi = [torch.from_numpy(x) for x in arrays]
    if not all(torch.isfinite(x).all() for x in (a, m, s, lo, hi)):
        raise ContractFailure("nonfinite truncated-normal input")
    if bool((s <= 0).any()) or bool((lo >= hi).any()):
        raise ContractFailure("invalid sigma or support")
    m64, s64, lo64, hi64, a64 = (x.double() for x in (m, s, lo, hi, a))
    mass = torch.special.ndtr((hi64 - m64) / s64) - torch.special.ndtr((lo64 - m64) / s64)
    if bool((mass < minimum_mass).any()):
        raise ContractFailure("truncated normalization mass below frozen threshold")
    tolerance = 8.0 * torch.finfo(a.dtype).eps
    if bool(((a64 < lo64 - tolerance) | (a64 > hi64 + tolerance)).any()):
        raise ContractFailure("sampled action outside saved support")
    z = (a64 - m64) / s64
    per_coordinate = (
        -0.5 * z.square()
        - torch.log(s64)
        - 0.5 * math.log(2.0 * math.pi)
        - torch.log(mass)
    ).to(m.dtype)
    return per_coordinate.sum(dim=-1).numpy()


def world_major_to_time_major(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[0] != SAMPLES:
        raise ContractFailure("world-major array does not contain 160 rows")
    return array.reshape(WORLDS, HORIZON, *array.shape[1:]).swapaxes(0, 1).reshape(
        SAMPLES, *array.shape[1:]
    )


def normalized_advantage(raw: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(raw, np.float32).reshape(-1))
    return ((tensor - tensor.mean()) / (tensor.std() + 1.0e-8)).numpy()


def clip_active(advantage: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    advantage = np.asarray(advantage)
    ratio = np.asarray(ratio)
    return ((advantage > 0) & (ratio > 1.2)) | ((advantage < 0) & (ratio < 0.8))


def classify_pass61_successor(
    row: dict[str, Any],
    rows_by_world_step: dict[tuple[int, int], dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    """Classify only a recorded same-epoch successor; never stitch epochs."""
    if row["time_out"]:
        return "window_timeout_at61", None
    if row["rollout_step"] == HORIZON - 1:
        return "right_censored_at_rollout_boundary", None
    successor = rows_by_world_step.get((row["world_index"], row["rollout_step"] + 1))
    if successor is None:
        return "unexplained_gap", None
    if successor["episode_serial"] != row["episode_serial"] or successor["source"] != 61:
        return "unexplained_gap", None
    if successor["tracking_terminated"]:
        return "continued_then_tracking_terminated_at62", successor
    if successor["time_out"]:
        return "continued_then_window_timeout_at62", successor
    if successor["score"] <= 1:
        if successor["rollout_step"] == HORIZON - 1:
            return "continued_and_passed62_then_right_censored", successor
        return "continued_and_passed62", successor
    return "continued_nonterminal_score_failure_at62", successor


def validate_grid(data: dict[str, np.ndarray], *, seed: int, epoch: int) -> dict[str, Any]:
    required = (
        "run_id", "training_seed", "world_index", "rollout_step", "episode_serial",
        "source_endpoint", "outcome_endpoint", "command_reference_endpoint",
        "reward_reference_endpoint", "next_observation_goal_reference_endpoint",
        "tracking_terminated", "time_out", "ppo_flat_index", "ppo_world_index",
        "ppo_time_index", "actor_hash", "RMS_hash", "normalization_version",
        "reset_context_hash",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ContractFailure(f"seed {seed} epoch {epoch}: missing fields {missing}")
    if any(np.asarray(data[name]).shape[0] != SAMPLES for name in data):
        raise ContractFailure(f"seed {seed} epoch {epoch}: a visit field is not length 160")
    world = data["world_index"].astype(int)
    step = data["rollout_step"].astype(int)
    if len(set(zip(step.tolist(), world.tolist()))) != SAMPLES:
        raise ContractFailure(f"seed {seed} epoch {epoch}: duplicate or missing time/world key")
    expected = {(t, w) for t in range(HORIZON) for w in range(WORLDS)}
    if set(zip(step.tolist(), world.tolist())) != expected:
        raise ContractFailure(f"seed {seed} epoch {epoch}: incomplete time/world grid")
    expected_flat = world * HORIZON + step
    checks = {
        "time_major_storage_order": bool(np.array_equal(world, np.tile(np.arange(4), 40)) and np.array_equal(step, np.repeat(np.arange(40), 4))),
        "ppo_flat_index_exact": bool(np.array_equal(data["ppo_flat_index"], expected_flat)),
        "ppo_world_index_exact": bool(np.array_equal(data["ppo_world_index"], world)),
        "ppo_time_index_exact": bool(np.array_equal(data["ppo_time_index"], step)),
        "source_outcome_plus_one": bool(np.array_equal(data["outcome_endpoint"], data["source_endpoint"] + 1)),
        "command_matches_outcome": bool(np.array_equal(data["command_reference_endpoint"], data["outcome_endpoint"])),
        "reward_matches_outcome": bool(np.array_equal(data["reward_reference_endpoint"], data["outcome_endpoint"])),
        "next_goal_tail_rule": bool(np.all((data["next_observation_goal_reference_endpoint"] >= data["outcome_endpoint"]) & (data["next_observation_goal_reference_endpoint"] <= data["outcome_endpoint"] + 1))),
        "seed_exact": bool(np.all(data["training_seed"] == seed)),
        "run_id_exact": bool(np.all(data["run_id"] == f"candidate_G_seed_{seed}")),
    }
    serial_ok = True
    for w in range(WORLDS):
        indices = np.where(world == w)[0]
        indices = indices[np.argsort(step[indices])]
        expected_serial = 0
        for index in indices:
            if int(data["episode_serial"][index]) != expected_serial:
                serial_ok = False
                break
            if bool(data["tracking_terminated"][index] or data["time_out"][index]):
                expected_serial += 1
    checks["episode_serial_reset_exact"] = serial_ok
    for name in ("actor_hash", "RMS_hash", "normalization_version", "reset_context_hash"):
        checks[f"{name}_epoch_constant"] = len(np.unique(data[name])) == 1
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ContractFailure(f"seed {seed} epoch {epoch}: row contract failed {failed}")
    return checks


def terminal_gae_audit(data: dict[str, np.ndarray]) -> dict[str, Any]:
    errors: list[float] = []
    suffix_errors: list[float] = []
    observed_returns: list[float] = []
    terminal_episodes = 0
    rows_checked = 0
    world = data["world_index"].astype(int)
    serial = data["episode_serial"].astype(int)
    step = data["rollout_step"].astype(int)
    reward = data["shaped_training_reward"].reshape(-1).astype(np.float32)
    value = data["rollout_value_before_update"].reshape(-1).astype(np.float32)
    recorded = data["raw_advantage"].reshape(-1).astype(np.float32)
    for key in sorted(set(zip(world.tolist(), serial.tolist()))):
        idx = np.where((world == key[0]) & (serial == key[1]))[0]
        idx = idx[np.argsort(step[idx])]
        if not len(idx) or not bool(data["tracking_terminated"][idx[-1]]):
            continue
        terminal_episodes += 1
        gae = np.float32(0.0)
        discounted = np.float32(0.0)
        for offset in range(len(idx) - 1, -1, -1):
            row = idx[offset]
            if offset == len(idx) - 1:
                next_nonterminal = np.float32(0.0)
                next_value = np.float32(0.0)
            else:
                next_nonterminal = np.float32(1.0)
                next_value = value[idx[offset + 1]]
            delta = np.float32(reward[row] + GAMMA * next_value * next_nonterminal - value[row])
            gae = np.float32(delta + GAMMA * GAE_TAU * next_nonterminal * gae)
            discounted = np.float32(reward[row] + GAMMA * discounted)
            observed_returns.append(float(discounted))
            errors.append(abs(float(gae) - float(recorded[row])))
            reconstructed_return = np.float32(gae + value[row])
            stored_return = np.float32(data["GAE_return"][row].reshape(-1)[0])
            suffix_errors.append(abs(float(reconstructed_return) - float(stored_return)))
            rows_checked += 1
    return {
        "true_tracking_terminal_episode_count": terminal_episodes,
        "rows_recomputed": rows_checked,
        "maximum_abs_GAE_error": max(errors, default=None),
        "maximum_abs_return_error": max(suffix_errors, default=None),
        "observed_discounted_returns": observed_returns,
        "observed_discounted_return_semantics": "discounted recorded rewards to the visible tracking terminal; computed for cohort interpretation, not substituted for logged GAE",
        "rollout_boundary_nonterminal_GAE": "not recomputed because last_values were not saved",
    }


@dataclass
class LedgerEntry:
    path: str
    role: str
    expected_sha256: str | None
    sha256_before: str
    bytes: int


class InputLedger:
    def __init__(self) -> None:
        self.entries: dict[str, LedgerEntry] = {}

    def add(self, path: Path, role: str, expected: str | None = None) -> Path:
        path = path.resolve()
        if not path.is_file():
            raise ContractFailure(f"missing input: {path}")
        observed = sha256(path)
        if expected is not None and observed != expected:
            raise ContractFailure(f"hash mismatch for {path}: {observed} != {expected}")
        key = str(path)
        old = self.entries.get(key)
        if old is not None and old.sha256_before != observed:
            raise ContractFailure(f"input changed during discovery: {path}")
        self.entries[key] = LedgerEntry(key, role, expected, observed, path.stat().st_size)
        return path

    def finish(self) -> dict[str, Any]:
        rows = []
        changed = []
        for item in sorted(self.entries.values(), key=lambda value: value.path):
            after = sha256(Path(item.path))
            same = after == item.sha256_before
            if not same:
                changed.append(item.path)
            rows.append({
                "path": item.path,
                "role": item.role,
                "expected_sha256": item.expected_sha256,
                "sha256_before": item.sha256_before,
                "sha256_after": after,
                "unchanged": same,
                "bytes": item.bytes,
            })
        return {"count": len(rows), "all_old_inputs_unchanged": not changed, "changed": changed, "files": rows}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def resolve_recorded_path(path: str, root: Path) -> Path:
    value = Path(path)
    if value.is_file():
        return value
    marker = "/data_all/zzx/3.2RL/"
    if marker in path:
        return root / path.split(marker, 1)[1]
    return value


def cohort_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "tracking_reward", "contact_bonus", "lift_reward", "raw_total_reward",
        "shaped_training_reward", "value", "return", "raw_advantage",
        "normalized_advantage", "delta_logp", "ratio",
    )
    result = {"n": len(rows)}
    for metric in metrics:
        result[metric] = distribution(row[metric] for row in rows)
    if rows:
        norm = np.asarray([row["normalized_advantage"] for row in rows])
        raw = np.asarray([row["raw_advantage"] for row in rows])
        result["normalized_advantage_sign_fraction"] = {
            "positive": float(np.mean(norm > 0)),
            "zero": float(np.mean(norm == 0)),
            "negative": float(np.mean(norm < 0)),
        }
        result["raw_to_normalized_sign_change_count"] = int(np.sum(np.sign(raw) != np.sign(norm)))
        result["true_clip_active_count"] = int(sum(row["clip_active"] for row in rows))
        result["update_direction"] = {
            key: int(sum(row["direction"] == key for row in rows))
            for key in ("consistent", "opposed", "numerically_ambiguous", "zero_advantage")
        }
    return result


def compare_cohorts(records: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_rows = [row for row in records if row["cohort"] == left]
    right_rows = [row for row in records if row["cohort"] == right]
    left_epochs = {(row["seed"], row["epoch"]) for row in left_rows}
    right_epochs = {(row["seed"], row["epoch"]) for row in right_rows}
    common = sorted(left_epochs & right_epochs)
    union = left_epochs | right_epochs
    metrics = ("tracking_reward", "raw_total_reward", "return", "value", "raw_advantage", "normalized_advantage", "delta_logp", "ratio")
    equal_weight = {}
    for metric in metrics:
        diffs = []
        for key in common:
            a = [row[metric] for row in left_rows if (row["seed"], row["epoch"]) == key]
            b = [row[metric] for row in right_rows if (row["seed"], row["epoch"]) == key]
            diffs.append(float(np.mean(a) - np.mean(b)))
        equal_weight[metric] = distribution(diffs)
    return {
        "left": left,
        "right": right,
        "left_stats": cohort_stats(left_rows),
        "right_stats": cohort_stats(right_rows),
        "matched_epoch_count": len(common),
        "epoch_union_count": len(union),
        "matching_coverage_fraction": (float(len(common) / len(union)) if union else None),
        "equal_epoch_weighted_left_minus_right": equal_weight,
        "sample_weighted_left_minus_right": {
            metric: (
                float(np.mean([row[metric] for row in left_rows]) - np.mean([row[metric] for row in right_rows]))
                if left_rows and right_rows else None
            ) for metric in metrics
        },
        "interpretation_limit": "observational same-source cohorts, not paired counterfactuals",
    }


def epoch_bin(epoch: int) -> str:
    for low, high in EPOCH_BINS:
        if low <= epoch <= high:
            return f"{low}-{high}"
    raise ContractFailure(f"epoch outside frozen bins: {epoch}")


def code_hashes(root: Path, ledger: InputLedger, contract: dict[str, Any]) -> dict[str, str]:
    result = {}
    expected_hashes = contract.get("source_sha256", {})
    if set(expected_hashes) != set(contract["source_files"]):
        raise ContractFailure("source_sha256 must bind every frozen source file exactly once")
    for name, relative in contract["source_files"].items():
        path = ledger.add(
            root / relative,
            f"frozen_source:{name}",
            expected_hashes[name],
        )
        result[name] = sha256(path)
    return result


def git_head(repo: Path) -> tuple[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    return head, status


def event_row(
    *, seed: int, epoch: int, index: int, data: dict[str, np.ndarray], visit_hash: str,
    delta_logp: np.ndarray, ratio: np.ndarray,
) -> dict[str, Any]:
    norm = float(data["normalized_advantage"][index])
    raw = float(data["raw_advantage"][index])
    delta = float(delta_logp[index])
    if abs(delta) <= NUMERICAL_LIMIT:
        direction = "numerically_ambiguous"
    elif norm == 0:
        direction = "zero_advantage"
    elif norm * delta > 0:
        direction = "consistent"
    else:
        direction = "opposed"
    return {
        "seed": seed,
        "epoch": epoch,
        "row_index": int(index),
        "visit_sha256": visit_hash,
        "run_id": str(data["run_id"][index]),
        "rollout_step": int(data["rollout_step"][index]),
        "world_index": int(data["world_index"][index]),
        "episode_serial": int(data["episode_serial"][index]),
        "source": int(data["source_endpoint"][index]),
        "outcome": int(data["outcome_endpoint"][index]),
        "score": float(data["tool_objective_score"][index]),
        "tracking_terminated": bool(data["tracking_terminated"][index]),
        "time_out": bool(data["time_out"][index]),
        "tracking_reward": float(data["tracking_reward"][index]),
        "contact_bonus": float(data["contact_bonus"][index]),
        "lift_reward": float(data["lift_reward"][index]),
        "raw_total_reward": float(data["raw_total_reward"][index]),
        "shaped_training_reward": float(data["shaped_training_reward"][index].reshape(-1)[0]),
        "value": float(data["rollout_value_before_update"][index].reshape(-1)[0]),
        "return": float(data["GAE_return"][index].reshape(-1)[0]),
        "raw_advantage": raw,
        "normalized_advantage": norm,
        "canonical_old_logprob": float(data["canonical_old_logprob"][index]),
        "post_logprob": float(data["canonical_old_logprob"][index] + delta),
        "delta_logp": delta,
        "ratio": float(ratio[index]),
        "clip_active": bool(clip_active(np.asarray([norm]), np.asarray([ratio[index]]))[0]),
        "direction": direction,
    }


def audit(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    contract_path = args.contract.resolve()
    contract = yaml.safe_load(contract_path.read_text())
    if contract.get("schema") != SCHEMA:
        raise ContractFailure("wrong postmortem contract schema")
    if contract.get("base_git_commit") != EXPECTED_BASE:
        raise ContractFailure("contract base commit changed")
    repo = Path(contract["git_repository"]).resolve()
    head, status = git_head(repo)
    if head != EXPECTED_BASE or status:
        raise ContractFailure(f"git baseline mismatch: head={head}, dirty={bool(status)}")
    output = args.output.resolve()
    if not args.validate_only and output.exists():
        raise FileExistsError(f"refusing to overwrite prior review: {output}")

    ledger = InputLedger()
    ledger.add(contract_path, "postmortem_contract")
    script_path = Path(__file__).resolve()
    observed_script_hash = sha256(script_path)
    if observed_script_hash != contract["implementation"]["audit_script_sha256"]:
        raise ContractFailure("audit script hash does not match frozen contract")
    ledger.add(script_path, "audit_script", observed_script_hash)
    frozen_code_hashes = code_hashes(root, ledger, contract)

    old_root = root / contract["candidate_G_run_root"]
    primary = contract["primary_inputs"]
    for name, spec in primary.items():
        ledger.add(root / spec["path"], f"primary:{name}", spec["sha256"])

    row_checks: list[dict[str, Any]] = []
    per_epoch: list[dict[str, Any]] = []
    source60_events: list[dict[str, Any]] = []
    pass_chains: list[dict[str, Any]] = []
    cohort_records: list[dict[str, Any]] = []
    continuation_bins: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    validation_audit = []
    numeric = {
        "maximum_reported_ratio_scalar_error": 0.0,
        "maximum_credit_join_abs_error": 0.0,
        "maximum_return_minus_value_advantage_error": 0.0,
        "maximum_advantage_normalization_error": 0.0,
        "maximum_reward_component_sum_error": 0.0,
        "maximum_shaped_reward_error": 0.0,
        "maximum_rollout_logprob_crosscheck_error": 0.0,
        "all_update_recomputations_usable": True,
    }
    terminal_gae_rows = []
    old_regression = {}

    for seed in SEEDS:
        seed_root = old_root / "training" / f"seed_{seed}"
        report_path = ledger.add(seed_root / "report.json", f"seed{seed}:report")
        report = load_json(report_path)
        if report.get("seed") != seed or report.get("training", {}).get("completed_actor_updates") != EPOCHS:
            raise ContractFailure(f"seed {seed} report is not a complete 250-epoch G run")
        visits_manifest_path = ledger.add(
            resolve_recorded_path(report["training_visitation"]["path"], root),
            f"seed{seed}:visitation_manifest", report["training_visitation"]["sha256"],
        )
        update_manifest_path = ledger.add(
            resolve_recorded_path(report["update_audit"]["path"], root),
            f"seed{seed}:update_manifest", report["update_audit"]["sha256"],
        )
        visits_manifest = load_json(visits_manifest_path)
        update_manifest = load_json(update_manifest_path)
        if len(visits_manifest.get("epochs", [])) != EPOCHS or len(update_manifest.get("epoch_reports", [])) != EPOCHS:
            raise ContractFailure(f"seed {seed}: manifest does not contain 250 epochs")
        # Bind small copied input contracts without opening checkpoint payloads.
        for path in sorted((seed_root / "input_contracts").iterdir()):
            if path.suffix in {".yaml", ".json", ".npz"}:
                ledger.add(path, f"seed{seed}:copied_input_contract")
        for epoch_text, validation in report["validations"].items():
            path = ledger.add(
                resolve_recorded_path(validation["trajectory"]["path"], root),
                f"seed{seed}:fixed_validation:{epoch_text}", validation["trajectory"]["sha256"],
            )
            with np.load(path, allow_pickle=False) as arrays:
                if arrays["endpoint"].shape != (40,):
                    raise ContractFailure(f"seed {seed} validation {epoch_text} is not 40 steps")
                validation_audit.append({"seed": seed, "epoch": int(epoch_text), "sha256": sha256(path), "rows": 40})

        seed_attempts = seed_passes = seed_source61 = 0
        for vm, um in zip(visits_manifest["epochs"], update_manifest["epoch_reports"], strict=True):
            epoch = int(vm["epoch"])
            if epoch != int(um["epoch"]):
                raise ContractFailure(f"seed {seed}: visit/update epoch mismatch")
            visit_path = ledger.add(resolve_recorded_path(vm["visits"]["path"], root), f"seed{seed}:epoch{epoch}:visits", vm["visits"]["sha256"])
            ledger.add(resolve_recorded_path(vm["summary"]["path"], root), f"seed{seed}:epoch{epoch}:visit_summary", vm["summary"]["sha256"])
            credit_path = ledger.add(resolve_recorded_path(um["credit"]["path"], root), f"seed{seed}:epoch{epoch}:credit", um["credit"]["sha256"])
            support_spec = um.get("support_distribution")
            if support_spec is None:
                raise ContractFailure(f"seed {seed} epoch {epoch}: no post support arrays")
            support_path = ledger.add(resolve_recorded_path(support_spec["path"], root), f"seed{seed}:epoch{epoch}:support", support_spec["sha256"])
            summary_path = ledger.add(seed_root / "update_audit" / f"epoch_{epoch:04d}_summary.json", f"seed{seed}:epoch{epoch}:update_summary")
            update_summary = load_json(summary_path)
            if update_summary != um:
                raise ContractFailure(
                    f"seed {seed} epoch {epoch}: update summary content differs from hashed manifest entry"
                )
            with np.load(visit_path, allow_pickle=False) as archive:
                data = {name: archive[name] for name in archive.files}
            checks = validate_grid(data, seed=seed, epoch=epoch)
            with np.load(credit_path, allow_pickle=False) as archive:
                credit = {name: archive[name] for name in archive.files}
            mapping_errors = {}
            mapping = data["ppo_flat_index"].astype(int)
            credit_pairs = {
                "raw_advantage": "raw_advantage",
                "normalized_advantage": "normalized_advantage",
                "return": "GAE_return",
                "value": "rollout_value_before_update",
            }
            for source_name, visit_name in credit_pairs.items():
                mapped = credit[source_name][mapping]
                error = float(np.max(np.abs(mapped.astype(np.float64) - data[visit_name].astype(np.float64))))
                mapping_errors[source_name] = error
                numeric["maximum_credit_join_abs_error"] = max(numeric["maximum_credit_join_abs_error"], error)
                if error != 0.0:
                    raise ContractFailure(f"seed {seed} epoch {epoch}: credit join mismatch {source_name}={error}")
            raw = data["raw_advantage"].reshape(-1)
            ret_value_error = float(np.max(np.abs(raw.astype(np.float64) - (data["GAE_return"].reshape(-1) - data["rollout_value_before_update"].reshape(-1)).astype(np.float64))))
            numeric["maximum_return_minus_value_advantage_error"] = max(numeric["maximum_return_minus_value_advantage_error"], ret_value_error)
            normalized = normalized_advantage(raw)
            norm_error = float(np.max(np.abs(normalized.astype(np.float64) - data["normalized_advantage"].astype(np.float64))))
            numeric["maximum_advantage_normalization_error"] = max(numeric["maximum_advantage_normalization_error"], norm_error)
            reward_sum = data["tracking_reward"] + data["contact_bonus"] + data["lift_reward"]
            reward_error = float(np.max(np.abs(reward_sum.astype(np.float64) - data["raw_total_reward"].astype(np.float64))))
            shaped_error = float(np.max(np.abs(data["shaped_training_reward"].reshape(-1).astype(np.float64) - data["raw_total_reward"].astype(np.float64))))
            numeric["maximum_reward_component_sum_error"] = max(numeric["maximum_reward_component_sum_error"], reward_error)
            numeric["maximum_shaped_reward_error"] = max(numeric["maximum_shaped_reward_error"], shaped_error)
            if max(ret_value_error, norm_error, reward_error, shaped_error) > NUMERICAL_LIMIT:
                raise ContractFailure(f"seed {seed} epoch {epoch}: credit/reward arithmetic exceeds 1e-4")
            terminal_gae_rows.append(terminal_gae_audit(data))

            with np.load(support_path, allow_pickle=False) as archive:
                support = {name: archive[name] for name in archive.files}
            if any(support[name].shape != (SAMPLES, 36) for name in ("raw_location", "bounded_mu", "sigma", "low", "high")):
                raise ContractFailure(f"seed {seed} epoch {epoch}: bad support array shape")
            post_logp_world = truncated_joint_logprob(
                data["sampled_action_preclamp"][np.argsort(mapping)],
                support["bounded_mu"], support["sigma"], support["low"], support["high"],
            )
            post_logp = post_logp_world[mapping]
            old_logp = data["canonical_old_logprob"].astype(np.float32)
            delta_logp = (post_logp.astype(np.float32) - old_logp).astype(np.float32)
            ratio = torch.exp(torch.from_numpy(delta_logp)).numpy()
            reported = update_summary["update"]["post_optimizer_ratio"]
            computed = distribution(ratio)
            scalar_errors = {
                "min": abs(computed["min"] - reported["min"]),
                "mean": abs(computed["mean"] - reported["mean"]),
                "p50": abs(computed["median"] - reported["p50"]),
                "p95": abs(computed["p95"] - reported["p95"]),
                "max": abs(computed["max"] - reported["max"]),
            }
            max_scalar_error = max(scalar_errors.values())
            numeric["maximum_reported_ratio_scalar_error"] = max(numeric["maximum_reported_ratio_scalar_error"], max_scalar_error)
            if max_scalar_error > NUMERICAL_LIMIT:
                numeric["all_update_recomputations_usable"] = False
                raise ContractFailure(f"seed {seed} epoch {epoch}: post ratio reproduction error {max_scalar_error}")
            rollout_logp = truncated_joint_logprob(
                data["sampled_action_preclamp"], data["actor_mu"], data["actor_sigma"],
                support["low"][mapping], support["high"][mapping],
            )
            cross_error = float(np.max(np.abs(rollout_logp.astype(np.float64) - old_logp.astype(np.float64))))
            numeric["maximum_rollout_logprob_crosscheck_error"] = max(numeric["maximum_rollout_logprob_crosscheck_error"], cross_error)

            indices_by_key = {(int(data["world_index"][i]), int(data["rollout_step"][i])): i for i in range(SAMPLES)}
            row_objects = [event_row(seed=seed, epoch=epoch, index=i, data=data, visit_hash=sha256(visit_path), delta_logp=delta_logp, ratio=ratio) for i in range(SAMPLES)]
            episode_outcome: dict[tuple[int, int], str] = {}
            for key in set((row["world_index"], row["episode_serial"]) for row in row_objects):
                episode_rows = [row for row in row_objects if (row["world_index"], row["episode_serial"]) == key]
                s60 = [row for row in episode_rows if row["source"] == 60]
                if any(row["score"] <= 1 and not row["tracking_terminated"] and not row["time_out"] for row in s60):
                    episode_outcome[key] = "later_pass61"
                elif any(row["tracking_terminated"] for row in s60):
                    episode_outcome[key] = "later_terminate61"
                else:
                    episode_outcome[key] = "unresolved_before61"

            epoch_source60_pass = epoch_source60_term = epoch_censor = 0
            for row in row_objects:
                source = row["source"]
                if source in (57, 58, 59):
                    outcome = episode_outcome[(row["world_index"], row["episode_serial"])]
                    if outcome != "unresolved_before61":
                        item = dict(row); item["cohort"] = f"source{source}_{outcome}"; cohort_records.append(item)
                elif source == 60:
                    seed_attempts += 1
                    passed = row["score"] <= 1 and not row["tracking_terminated"] and not row["time_out"]
                    prior = [r for r in row_objects if r["world_index"] == row["world_index"] and r["episode_serial"] == row["episode_serial"] and r["rollout_step"] < row["rollout_step"]]
                    if any(r["tracking_terminated"] for r in prior):
                        passed = False
                    classification = "tracking_terminated_at61" if row["tracking_terminated"] else "timeout_at61" if row["time_out"] else "nonterminated_score_failure_at61"
                    successor = None
                    if passed:
                        seed_passes += 1; epoch_source60_pass += 1
                        item = dict(row); item["cohort"] = "source60_pass61"; cohort_records.append(item)
                        rows_by_world_step = {
                            (candidate["world_index"], candidate["rollout_step"]): candidate
                            for candidate in row_objects
                        }
                        classification, successor = classify_pass61_successor(
                            row, rows_by_world_step
                        )
                        epoch_censor += int(classification == "right_censored_at_rollout_boundary")
                        chain_rows = [r for r in row_objects if r["world_index"] == row["world_index"] and r["episode_serial"] == row["episode_serial"] and 57 <= r["source"] <= 61]
                        pass_chains.append({
                            "event_key": {k: row[k] for k in ("run_id", "seed", "epoch", "rollout_step", "world_index", "episode_serial")},
                            "classification": classification,
                            "visible_sources57_61": chain_rows,
                            "successor": successor,
                        })
                    else:
                        epoch_source60_term += int(row["tracking_terminated"])
                        item = dict(row); item["cohort"] = "source60_terminate61" if row["tracking_terminated"] else "source60_other_failure61"; cohort_records.append(item)
                    continuation_bins[(seed, epoch_bin(epoch))]["source60_attempts"] += 1
                    continuation_bins[(seed, epoch_bin(epoch))]["pass61"] += int(passed)
                    continuation_bins[(seed, epoch_bin(epoch))][classification] += 1
                    source60_events.append({
                        **{k: row[k] for k in ("run_id", "seed", "epoch", "rollout_step", "world_index", "episode_serial", "row_index", "visit_sha256", "score", "tracking_terminated", "time_out", "raw_advantage", "normalized_advantage", "delta_logp", "ratio")},
                        "pass61": passed,
                        "classification": classification,
                        "successor_source": successor["source"] if successor else None,
                        "successor_outcome": successor["outcome"] if successor else None,
                        "successor_score": successor["score"] if successor else None,
                        "successor_tracking_terminated": successor["tracking_terminated"] if successor else None,
                    })
                elif source == 61:
                    seed_source61 += 1
                    passed62 = row["score"] <= 1 and not row["tracking_terminated"] and not row["time_out"]
                    item = dict(row); item["cohort"] = "source61_pass62" if passed62 else "source61_terminate62" if row["tracking_terminated"] else "source61_other62"; cohort_records.append(item)

            adv = data["normalized_advantage"].astype(np.float64)
            ambiguous = np.abs(delta_logp.astype(np.float64)) <= NUMERICAL_LIMIT
            direction_consistent = (~ambiguous) & (adv * delta_logp.astype(np.float64) > 0)
            direction_opposed = (~ambiguous) & (adv * delta_logp.astype(np.float64) < 0)
            update = update_summary["update"]
            per_epoch.append({
                "seed": seed, "epoch": epoch, "epoch_bin": epoch_bin(epoch), "samples": SAMPLES,
                "source60_attempts": int(np.sum(data["source_endpoint"] == 60)),
                "source60_pass61": epoch_source60_pass,
                "source60_terminate61": epoch_source60_term,
                "source60_boundary_censored": epoch_censor,
                "ratio_min": computed["min"], "ratio_mean": computed["mean"], "ratio_median": computed["median"], "ratio_p95": computed["p95"], "ratio_max": computed["max"],
                "ratio_outside_0p8_1p2_fraction": float(np.mean((ratio < 0.8) | (ratio > 1.2))),
                "true_clip_active_fraction": float(np.mean(clip_active(adv, ratio))),
                "direction_consistent_fraction": float(np.mean(direction_consistent)),
                "direction_opposed_fraction": float(np.mean(direction_opposed)),
                "direction_ambiguous_fraction": float(np.mean(ambiguous)),
                "reported_ratio_max_scalar_error": max_scalar_error,
                "exact_KL_mean": float(update["exact_old_to_new_truncated_policy_KL"]["mean"]),
                "exact_KL_max": float(update["exact_old_to_new_truncated_policy_KL"]["max"]),
                "gradient_norm_before_clip": float(update["gradient_norm_before_clip"]),
                "gradient_norm_after_clip": float(update["gradient_norm_after_clip"]),
                "parameter_delta_l2": float(update["parameter_delta_l2"]),
                "sigma_post_mean": float(update["sigma_post_optimizer"]["mean"]),
                "support_minimum_mass": float(update["minimum_truncated_normalization_mass"]),
                "bounded_mu_outside_support_count": int(update["bounded_mu_outside_support_count"]),
                "normalization_version_used": int(um["observation_normalization_version_used"]),
                "normalization_version_after_commit": int(um["observation_normalization_version_after_commit"]),
            })
            row_checks.append({"seed": seed, "epoch": epoch, **checks, "credit_mapping_max_errors": mapping_errors})
        old_regression[str(seed)] = {"source60_attempts": seed_attempts, "pass61": seed_passes, "source61_rows": seed_source61}

    expected_old = {"0": (722, 128, 126), "1": (604, 49, 43), "2": (553, 62, 60)}
    for key, expected in expected_old.items():
        observed = old_regression[key]
        if tuple(observed[name] for name in ("source60_attempts", "pass61", "source61_rows")) != expected:
            raise ContractFailure(f"old comparison regression mismatch seed {key}: {observed} != {expected}")
    if len(source60_events) != 1879 or len(pass_chains) != 239:
        raise ContractFailure("source60/pass61 aggregate does not reproduce 1879/239")

    # Aggregate continuation outcomes and explain the 239/229 difference.
    continuation_classes = defaultdict(int)
    for event in source60_events:
        if event["pass61"]:
            continuation_classes[event["classification"]] += 1
    source61_total = sum(value["source61_rows"] for value in old_regression.values())
    difference = len(pass_chains) - source61_total
    censor_count = continuation_classes["right_censored_at_rollout_boundary"]
    if difference != 10 or censor_count != difference or continuation_classes.get("unexplained_gap", 0):
        raise ContractFailure(f"239/229 gap not fully explained by boundary censoring: difference={difference}, censor={censor_count}, classes={dict(continuation_classes)}")

    # Cohort reports.
    comparisons = {
        "source60_pass_vs_terminate61": compare_cohorts(cohort_records, "source60_pass61", "source60_terminate61"),
        "source57_prefix": compare_cohorts(cohort_records, "source57_later_pass61", "source57_later_terminate61"),
        "source58_prefix": compare_cohorts(cohort_records, "source58_later_pass61", "source58_later_terminate61"),
        "source59_prefix": compare_cohorts(cohort_records, "source59_later_pass61", "source59_later_terminate61"),
        "source61_pass_vs_terminate62": compare_cohorts(cohort_records, "source61_pass62", "source61_terminate62"),
    }
    early = [row for row in per_epoch if row["epoch"] <= 10]
    update_epoch_bins = []
    update_metric_names = (
        "exact_KL_mean",
        "ratio_outside_0p8_1p2_fraction",
        "true_clip_active_fraction",
        "gradient_norm_before_clip",
        "parameter_delta_l2",
    )
    for seed in SEEDS:
        for low, high in EPOCH_BINS:
            rows = [
                row for row in per_epoch
                if row["seed"] == seed and low <= row["epoch"] <= high
            ]
            if len(rows) != high - low + 1:
                raise ContractFailure(
                    f"seed {seed} epoch bin {low}-{high} is incomplete"
                )
            update_epoch_bins.append({
                "seed": seed,
                "epoch_bin": f"{low}-{high}",
                "update_count": len(rows),
                "metrics": {
                    name: distribution(row[name] for row in rows)
                    for name in update_metric_names
                },
            })
    update_report = {
        "schema": f"{SCHEMA}_update_by_cohort",
        "all_750_updates_recomputed": len(per_epoch) == 750,
        "numerical_contract": {"dtype": "torch float64 CDF, float32 per-coordinate output and joint reduction", "hard_fail": NUMERICAL_LIMIT, **numeric},
        "early_epochs_1_10": early,
        "by_seed_and_epoch_bin": update_epoch_bins,
        "all_update_aggregate": {
            "exact_KL_mean": distribution(row["exact_KL_mean"] for row in per_epoch),
            "ratio_outside_fraction": distribution(row["ratio_outside_0p8_1p2_fraction"] for row in per_epoch),
            "true_clip_active_fraction": distribution(row["true_clip_active_fraction"] for row in per_epoch),
            "gradient_norm_before_clip": distribution(row["gradient_norm_before_clip"] for row in per_epoch),
            "parameter_delta_l2": distribution(row["parameter_delta_l2"] for row in per_epoch),
        },
        "cohort_comparisons": comparisons,
        "surrogate_field_note": "legacy surrogate_changed_by_clip_fraction used |unclipped-clipped|>1e-8; true_clip_active here is A>0,r>1.2 or A<0,r<0.8",
        "post_context_note": "saved support arrays are the post-update replay under the rollout-frozen RMS and saved BPTT/reset context, not a natural updated-policy episode replay",
    }
    observed_discounted_returns = [
        value
        for row in terminal_gae_rows
        for value in row["observed_discounted_returns"]
    ]
    gae_summary = {
        "true_tracking_terminal_episodes": sum(row["true_tracking_terminal_episode_count"] for row in terminal_gae_rows),
        "rows_recomputed": sum(row["rows_recomputed"] for row in terminal_gae_rows),
        "maximum_abs_GAE_error": max((row["maximum_abs_GAE_error"] or 0.0) for row in terminal_gae_rows),
        "maximum_abs_return_error": max((row["maximum_abs_return_error"] or 0.0) for row in terminal_gae_rows),
        "observed_discounted_return_distribution": distribution(observed_discounted_returns),
        "observed_discounted_return_semantics": "discounted recorded rewards to each visible tracking terminal; descriptive only and not substituted for GAE returns",
        "rollout_boundary_limit": "last_values absent; nonterminal step39 GAE is not reconstructed or zero-bootstrapped",
    }
    credit_report = {
        "schema": f"{SCHEMA}_credit_by_cohort",
        "units": "rollout value/return/raw advantage are denormalized original shaped-reward units from the asymmetric critic; normalized advantage uses the full 160-row batch with torch unbiased std and epsilon 1e-8",
        "arithmetic": {
            "return_minus_value_equals_raw_advantage_max_abs_error": numeric["maximum_return_minus_value_advantage_error"],
            "full_batch_advantage_normalization_max_abs_error": numeric["maximum_advantage_normalization_error"],
            "reward_component_sum_max_abs_error": numeric["maximum_reward_component_sum_error"],
            "shaped_equals_raw_reward_max_abs_error": numeric["maximum_shaped_reward_error"],
        },
        "terminal_GAE_recomputation": gae_summary,
        "cohort_comparisons": comparisons,
        "interpretation_limit": "negative normalized advantage is relative batch credit, not by itself evidence of a reward bug",
    }
    continuation_report = {
        "schema": f"{SCHEMA}_episode_continuation",
        "old_stat_regression": old_regression,
        "aggregate": {
            "source60_attempts": len(source60_events),
            "pass61": len(pass_chains),
            "source61_action_rows": source61_total,
            "pass61_minus_source61_rows": difference,
            "classification": dict(sorted(continuation_classes.items())),
            "gap_explanation": "all 10 missing source61 actions are successful 60->61 transitions at rollout_step=39 and are right-censored by the frozen 40-step collector boundary",
        },
        "by_seed_and_epoch_bin": [],
        "cross_epoch_join_forbidden": True,
    }
    continuation_class_names = sorted(
        {event["classification"] for event in source60_events}
    )
    for seed in SEEDS:
        for low, high in EPOCH_BINS:
            band = f"{low}-{high}"
            counts = continuation_bins[(seed, band)]
            attempts = int(counts.get("source60_attempts", 0))
            continuation_report["by_seed_and_epoch_bin"].append({
                "seed": seed,
                "epoch_bin": band,
                "source60_attempts": attempts,
                "pass61": int(counts.get("pass61", 0)),
                "pass61_fraction": (
                    float(counts.get("pass61", 0) / attempts) if attempts else None
                ),
                "classification": {
                    name: int(counts.get(name, 0))
                    for name in continuation_class_names
                },
            })

    manifest = ledger.finish()
    if not manifest["all_old_inputs_unchanged"]:
        raise ContractFailure("an old Candidate-G input changed during the review")
    if args.validate_only:
        print(json.dumps({
            "status": "input_contract_valid",
            "input_files": manifest["count"],
            "rows": len(per_epoch) * SAMPLES,
            "updates": len(per_epoch),
            "source60_attempts": len(source60_events),
            "pass61": len(pass_chains),
            "new_physics_steps": 0,
            "new_network_forwards": 0,
        }, indent=2))
        return 0

    output.mkdir(parents=True)
    json_dump(output / "input_manifest.json", {
        "schema": f"{SCHEMA}_input_manifest",
        "base_git_commit": head,
        "git_worktree_clean_before_analysis": True,
        "contract_sha256": sha256(contract_path),
        "audit_script_sha256": observed_script_hash,
        "source_hashes": frozen_code_hashes,
        "dependency_versions": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "pyyaml": yaml.__version__},
        "old_inputs": manifest,
        "checkpoint_payloads_opened": False,
    })
    json_dump(output / "row_join_audit.json", {
        "schema": f"{SCHEMA}_row_join_audit",
        "status": "passed",
        "epochs": len(row_checks),
        "rows": len(row_checks) * SAMPLES,
        "primary_key": ["run_id", "training_seed", "epoch", "rollout_step", "world_index"],
        "episode_key_extension": "episode_serial resets within each epoch",
        "layout": {"visits": "time-major", "PPO_credit_and_support": "world-major", "mapping": "ppo_flat_index=world*40+time"},
        "checks_all_passed": True,
        "maximum_credit_join_abs_error": numeric["maximum_credit_join_abs_error"],
        "per_epoch_checks": row_checks,
    })
    json_dump(output / "episode_continuation.json", continuation_report)
    json_dump(output / "credit_by_cohort.json", credit_report)
    json_dump(output / "update_by_cohort.json", update_report)
    with (output / "source60_events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(source60_events[0]),
            lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(source60_events)
    with (output / "pass61_event_chains.jsonl").open("w") as stream:
        for row in pass_chains:
            stream.write(json.dumps(row) + "\n")
    with (output / "per_epoch_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(per_epoch[0]),
            lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(per_epoch)

    first_update_regression = {
        str(seed): {
            "exact_KL_mean": next(row["exact_KL_mean"] for row in per_epoch if row["seed"] == seed and row["epoch"] == 1),
            "ratio_outside_fraction": next(row["ratio_outside_0p8_1p2_fraction"] for row in per_epoch if row["seed"] == seed and row["epoch"] == 1),
        } for seed in SEEDS
    }
    loss_text = f"""# Candidate G static training signal and loss path

This is a source-only audit. No checkpoint, model forward, gradient, optimizer, or simulator was invoked.

| Path | Frozen behavior in Candidate G |
|---|---|
| Rollout value | `PpoAgent.get_action_values()` replaces the actor model value with the independent asymmetric critic value when `has_asymmetric_critic`; GAE therefore starts from the fresh privileged critic. |
| GAE | `discount_values()` uses gamma=0.998 and tau=0.95. Stored visits contain the denormalized rollout value, return, and raw advantage before dataset value normalization. |
| External critic | A separate `[1024,512]` privileged-state MLP, fresh per seed, trained for 4 mini-epochs against the same GAE return with clipped value loss. |
| Actor optimizer loss | `actor_loss + 0.5 * critic_coef * internal_value_loss - entropy_coef * entropy + bounds_coef * bounds_loss`; frozen coefficients are critic_coef=4, entropy=0, bounds=0. Thus internal actor value loss has coefficient 2.0. |
| Actor internal value head | The actor has `separate_value_mlp=false`; its value head and policy heads share the 512-MLP, LSTM-1024, layer norm, and concatenated representation. Internal value loss can update shared policy representation even though rollout GAE uses an external critic. |
| Inheritance distinction | `actor-only inheritance` describes checkpoint state provenance. It does **not** mean actor optimizer uses policy-surrogate-only loss. |

Static source hashes are recorded in `input_manifest.json`. This establishes an influence path, not historical per-loss gradient magnitude. Missing historical per-loss gradients, natural closed-loop performance immediately after every update, and isolated RMS causal effects remain unknown.
"""
    (output / "loss_path.md").write_text(loss_text)

    # The recommendation is deliberately an audit direction, never a training authorization.
    recommendation = "REVIEW_ACTOR_INTERNAL_VALUE_LOSS_AND_SHARED_REPRESENTATION_CREDIT_PATH"
    decision = {
        "schema": f"{SCHEMA}_decision",
        "status": "COMPLETED_READ_ONLY_REVIEW",
        "candidate_G_original_classification": "G3_NO_PREDECLARED_SUSTAINED_EXTENSION",
        "candidate_G_classification_changed": False,
        "new_training_authorized": False,
        "chunk_commit_authorized": False,
        "next_review_recommendation": recommendation,
        "recommendation_basis": "Candidate G used a fresh asymmetric critic for rollout credit while the actor optimizer simultaneously applied a coefficient-2 internal value loss through the shared policy MLP/LSTM representation; the saved data cannot separate that loss contribution, so this is the narrowest supported next code/evidence review rather than an LR, reward, or curriculum change.",
        "not_selected": {
            "learning_rate_change": "large early updates are observed, but retrospective arrays do not establish LR as the unique cause",
            "reward_change": "successful continuation credit is observational and no reward arithmetic mismatch was found",
            "curriculum_or_more_training": "G3 remains closed and no new training is authorized",
        },
    }
    json_dump(output / "decision.json", decision)
    report = {
        "schema": f"{SCHEMA}_report",
        "status": "COMPLETED_READ_ONLY_REVIEW",
        "research_answers": {
            "Q1_continuation": continuation_report["aggregate"],
            "Q2_credit": comparisons,
            "Q3_probability_change": {
                "first_updates": first_update_regression,
                "all_updates": update_report["all_update_aggregate"],
                "cohort_comparisons": comparisons,
            },
        },
        "static_loss_path": {
            "rollout_credit_network": "fresh independent asymmetric critic",
            "actor_optimizer_contains_internal_value_loss": True,
            "actor_internal_value_loss_effective_coefficient": 2.0,
            "policy_and_internal_value_share_MLP_LSTM": True,
        },
        "unknowns": [
            "historical per-loss gradient contributions were not saved",
            "nonterminal rollout-boundary last_values were not saved, so boundary GAE cannot be independently reconstructed",
            "post-update support replay is saved-BPTT-context evidence, not a natural updated-policy closed-loop trajectory",
            "observational pass/fail cohorts are not paired counterfactual states",
            "no new simulation establishes mathematical feasibility or a unique root cause",
        ],
        "budgets": {"new_physics_steps": 0, "new_control_intervals": 0, "new_network_forwards": 0, "optimizer_updates": 0},
        "old_inputs_unchanged": True,
        "validation_arrays_read": len(validation_audit),
        "decision": decision,
    }
    json_dump(output / "report.json", report)
    source60_comparison = comparisons["source60_pass_vs_terminate61"]
    source60_pass_stats = source60_comparison["left_stats"]
    source60_fail_stats = source60_comparison["right_stats"]
    source61_fail_stats = comparisons["source61_pass_vs_terminate62"]["right_stats"]
    update_aggregate = update_report["all_update_aggregate"]
    summary = f"""# Candidate G read-only postmortem v1

Status: **COMPLETED_READ_ONLY_REVIEW**. Candidate G remains **G3 / NO_PREDECLARED_SUSTAINED_EXTENSION**. No simulator, network forward, gradient, optimizer, rollout, or new training was run.

## Q1 — what followed successful endpoint61 transitions?

- Recomputed source60 attempts: **{len(source60_events)}**.
- Successful 60→61 transitions: **{len(pass_chains)}**.
- Recorded source61 action rows: **{source61_total}**.
- The exact **{difference}**-row difference consists entirely of successful source60 transitions at rollout step 39, hence right-censoring at the 40-step collector boundary. It is not endpoint62 failure, and no cross-epoch join was made.
- Continuation classes: `{json.dumps(dict(sorted(continuation_classes.items())), sort_keys=True)}`.

## Q2 — recorded credit

The logged value/return/raw advantage are in original shaped-reward units from the independent asymmetric critic. `return - value == raw advantage` has maximum absolute error **{numeric['maximum_return_minus_value_advantage_error']:.3g}**; full-batch normalized-advantage reproduction error is **{numeric['maximum_advantage_normalization_error']:.3g}**.

| source60 cohort | n | mean reward | mean return | mean value | mean raw advantage | mean normalized advantage |
|---|---:|---:|---:|---:|---:|---:|
| pass endpoint61 | {source60_pass_stats['n']} | {source60_pass_stats['raw_total_reward']['mean']:.6f} | {source60_pass_stats['return']['mean']:.6f} | {source60_pass_stats['value']['mean']:.6f} | {source60_pass_stats['raw_advantage']['mean']:.6f} | {source60_pass_stats['normalized_advantage']['mean']:.6f} |
| terminate at endpoint61 | {source60_fail_stats['n']} | {source60_fail_stats['raw_total_reward']['mean']:.6f} | {source60_fail_stats['return']['mean']:.6f} | {source60_fail_stats['value']['mean']:.6f} | {source60_fail_stats['raw_advantage']['mean']:.6f} | {source60_fail_stats['normalized_advantage']['mean']:.6f} |

All **{source61_fail_stats['n']}** visible source61 actions terminate at endpoint62; their mean raw/normalized advantages are **{source61_fail_stats['raw_advantage']['mean']:.6f} / {source61_fail_stats['normalized_advantage']['mean']:.6f}**. The pooled source60 normalized-advantage difference (pass minus fail) is **{source60_comparison['sample_weighted_left_minus_right']['normalized_advantage']:.6f}**, while equal weighting over the **{source60_comparison['matched_epoch_count']}** matched epochs gives **{source60_comparison['equal_epoch_weighted_left_minus_right']['normalized_advantage']['mean']:.6f}**. This sign change is direct evidence of epoch-mixture confounding. See `credit_by_cohort.json` for all distributions. These are observational cohorts, not paired counterfactuals.

## Q3 — update direction

All **750** actor updates were recomputed from saved canonical old joint log-probabilities and post-update truncated-normal support arrays. Maximum disagreement with the recorded ratio aggregate is **{numeric['maximum_reported_ratio_scalar_error']:.3g}**, below the frozen `1e-4` diagnostic limit.

| source60 cohort | mean Δlog p | mean ratio | direction consistent | direction opposed |
|---|---:|---:|---:|---:|
| pass endpoint61 | {source60_pass_stats['delta_logp']['mean']:.6f} | {source60_pass_stats['ratio']['mean']:.6f} | {source60_pass_stats['update_direction']['consistent']} | {source60_pass_stats['update_direction']['opposed']} |
| terminate at endpoint61 | {source60_fail_stats['delta_logp']['mean']:.6f} | {source60_fail_stats['ratio']['mean']:.6f} | {source60_fail_stats['update_direction']['consistent']} | {source60_fail_stats['update_direction']['opposed']} |

Across all updates, the per-update exact-KL mean has mean/median/max **{update_aggregate['exact_KL_mean']['mean']:.6f} / {update_aggregate['exact_KL_mean']['median']:.6f} / {update_aggregate['exact_KL_mean']['max']:.6f}**; the ratio-outside-`[0.8,1.2]` fraction has mean/max **{update_aggregate['ratio_outside_fraction']['mean']:.6f} / {update_aggregate['ratio_outside_fraction']['max']:.6f}**. First-update regressions are preserved in `report.json`; epochs 1–10 and all frozen epoch-bin aggregates are in `update_by_cohort.json`, while all 750 rows are in `per_epoch_metrics.csv`.

## Static loss path

Rollout GAE uses the fresh independent asymmetric critic, but the actor optimizer still includes an internal value loss with effective coefficient **2.0** (`0.5 × critic_coef=4`). Because `separate_value_mlp=false`, this internal value head shares the actor MLP/LSTM representation. Actor-only checkpoint inheritance is therefore not actor-only training loss. This proves an influence path, not its historical gradient share.

## What cannot be inferred

- No unique root cause is claimed.
- Rollout-boundary GAE is not reconstructed without saved `last_values`.
- Saved post distributions use the rollout RMS and BPTT/reset context; they are not natural updated-policy trajectories.
- No mathematical infeasibility, reward bug, LR root cause, or new candidate authorization follows from this review.

## One next review direction

**{recommendation}**: statically and prospectively isolate the actor's internal value-loss/shared-representation path before choosing any training change. `new_training_authorized=false`.
"""
    (output / "summary.md").write_text(summary)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "source60_attempts": len(source60_events),
        "pass61": len(pass_chains),
        "source61_rows": source61_total,
        "right_censored": censor_count,
        "updates": len(per_epoch),
        "old_inputs_unchanged": True,
        "budgets": report["budgets"],
    }, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(audit(parse_args()))
    except (ContractFailure, FileExistsError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2)
