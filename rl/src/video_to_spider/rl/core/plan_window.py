"""Orchestration and artifact I/O for the fixed two-chunk sequence search."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml

from .audit import append_jsonl
from .env import make_world
from .sequence_search import (
    BudgetLedger,
    BudgetStopped,
    GenerationResult,
    HORIZON as PLAN_HORIZON,
    DOF as PLAN_DOF,
    SequenceResult,
    SlotResult,
    evaluate_sequence,
    sample_population,
    search_window,
)
from .state_io import (
    load_boundary_context,
    load_torch_gzip,
    manifest_entry,
    sha256,
    validate_physics_snapshot,
    verify_artifact,
    write_json,
    write_torch_gzip_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]


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


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_plan_config(
    path: Path, asset_root_override: Path | None,
) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema") != "egoengine_taco_pour_two_chunk_sequence_search_v1":
        raise ValueError("unsupported sequence-search configuration")
    if config.get("status") != "authorized_single_run":
        raise ValueError("sequence search is not authorized")
    expected_execution = {
        "task": "TACO_20230927_017", "tracking_variant": "tool_only",
        "device": "cpu", "worlds": 1, "source_endpoint": 40,
        "final_endpoint": 80, "horizon": 40, "actions": 36,
        "physics_steps_per_control": 10, "auto_reset": False,
        "training_enabled": False, "chunk_commit_enabled": False,
        "automatic_followon": False,
    }
    if config.get("execution") != expected_execution:
        raise ValueError("sequence-search execution contract changed")
    search = config.get("search", {})
    expected_search = {
        "seed": 0, "rng": "numpy_Generator_PCG64",
        "proposal": "fft_colored_gaussian_then_float32_support_projection",
        "noise_beta": 2.5, "iterations": 6,
        "new_candidate_slots": [256, 204, 163, 131, 104, 83],
        "maximum_new_candidate_slots": 941,
        "mean_candidate_in_each_generation": True,
        "mean_candidate_consumes_slot": True, "elite_count": 10,
        "elite_reuse_count": 3, "old_moment_weight": 0.1,
        "initial_std_half_range_factor": 0.5,
        "std_floor_half_range_factor": 0.02,
        "fit_only_executed_elite_rows": True,
        "minimum_observed_elites_for_row_update": 2,
        "time_shift_elites": False, "freeze_prefix": False,
        "rank_infeasible": [
            "valid_prefix_N_desc", "first_failure_score_asc",
            "valid_prefix_object_reward_desc", "candidate_id_asc",
        ],
        "stop_on_first_strict_success": True,
    }
    if search != expected_search:
        raise ValueError("sequence-search sampling contract changed")
    expected_preflight = {
        "seed": 991,
        "ordered_probes": [
            "zero_replay", "saved_donor_sequence", "one_colored_proposal",
            "saved_donor_sequence_repeat",
        ],
        "maximum_attempts": 2, "expected_donor_prefix": 20,
        "expected_donor_first_failure": 61,
    }
    if config.get("preflight") != expected_preflight:
        raise ValueError("sequence-search preflight contract changed")
    budget = config.get("budget", {})
    expected_budget = {
        "preflight_max_control_intervals": 320,
        "preflight_max_physics_steps": 3200,
        "search_max_control_intervals": 37640,
        "search_max_physics_steps": 376400,
        "final_validation_reserved_control_intervals": 80,
        "final_validation_reserved_physics_steps": 800,
        "all_in_max_control_intervals": 38040,
        "all_in_max_physics_steps": 380400,
        "actor_network_forward_calls": 0, "critic_network_forward_calls": 0,
        "backward_calls": 0, "actor_optimizer_steps": 0,
        "critic_optimizer_steps": 0, "max_cem_refits": 6,
        "reinvest_early_stop_savings": False,
        "automatic_formal_restart": False,
    }
    if budget != expected_budget:
        raise ValueError("sequence-search budget contract changed")
    root = (asset_root_override or Path(config["asset_root"])).resolve(strict=True)
    assets: dict[str, Path] = {}
    for name, row in config["assets"].items():
        assets[name] = verify_artifact(root / row["path"], row["sha256"])
    return config, root, assets


def _resolve_mjcf_dependencies(simulator_config: Path) -> list[Path]:
    """Resolve the frozen reference, MJCF, includes and file-backed assets."""
    simulator = yaml.safe_load(simulator_config.read_text())
    reference = Path(simulator["data_path"]).resolve(strict=True)
    model = Path(simulator["model_path"]).resolve(strict=True)
    found: set[Path] = {simulator_config.resolve(), reference, model}
    pending = [model]
    parsed: set[Path] = set()
    while pending:
        xml_path = pending.pop()
        if xml_path in parsed:
            continue
        parsed.add(xml_path)
        root = ET.parse(xml_path).getroot()
        compiler = root.find("compiler")
        meshdir = xml_path.parent
        texturedir = xml_path.parent
        if compiler is not None:
            if compiler.get("meshdir"):
                meshdir = Path(compiler.get("meshdir"))
                if not meshdir.is_absolute():
                    meshdir = xml_path.parent / meshdir
            if compiler.get("texturedir"):
                texturedir = Path(compiler.get("texturedir"))
                if not texturedir.is_absolute():
                    texturedir = xml_path.parent / texturedir
        for element in root.iter():
            value = element.get("file")
            if not value:
                continue
            raw = Path(value)
            if raw.is_absolute():
                resolved = raw.resolve(strict=True)
            elif element.tag == "include":
                resolved = (xml_path.parent / raw).resolve(strict=True)
            elif element.tag == "mesh":
                resolved = (meshdir / raw).resolve(strict=True)
            elif element.tag == "texture":
                resolved = (texturedir / raw).resolve(strict=True)
            else:
                resolved = (xml_path.parent / raw).resolve(strict=True)
            found.add(resolved)
            if element.tag == "include":
                pending.append(resolved)
    return sorted(found)


def _plan_input_manifest(
    config_path: Path, assets: dict[str, Path], simulator_dependencies: list[Path],
) -> dict[str, Any]:
    source = Path(__file__).with_name("sequence_search.py")
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable"
    return {
        "schema": "egoengine_taco_pour_two_chunk_sequence_search_input_manifest_v1",
        "git_head_at_execution": head,
        "config": manifest_entry(config_path),
        "implementation": manifest_entry(source),
        "sampler": {
            "name": "local_numpy_fft_colored_noise_v1",
            "beta": 2.5,
            "fft_axis": "time",
            "retains_dc": True,
            "per_sequence_normalization": False,
        },
        "assets": {name: manifest_entry(path) for name, path in assets.items()},
        "transitive_simulator_dependencies": [
            manifest_entry(path) for path in simulator_dependencies
        ],
        "network_forward_calls_authorized": 0,
        "training_enabled": False,
        "chunk_commit_enabled": False,
    }


def _load_plan_arrays(carryover: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(carryover, allow_pickle=False) as source:
        history = {name: source[name].copy() for name in source.files}
    endpoints = history.get("endpoint")
    if endpoints is None or endpoints.ndim != 1:
        raise ValueError("carryover has no one-dimensional endpoint array")
    selected = np.flatnonzero((endpoints >= 41) & (endpoints <= 80))
    if (
        selected.size != PLAN_HORIZON
        or not np.array_equal(endpoints[selected], np.arange(41, 81, dtype=endpoints.dtype))
    ):
        raise ValueError("carryover does not contain endpoints 41 through 80 exactly once")
    required = ("deterministic_action", "action_low", "action_high")
    if any(name not in history for name in required):
        raise ValueError("carryover is missing the frozen action sequence/support")
    u0, low, high = (np.asarray(history[name][selected]) for name in required)
    if any(value.shape != (40, 36) or value.dtype != np.float32 for value in (u0, low, high)):
        raise ValueError("carryover action arrays must be float32 (40, 36)")
    if not all(np.isfinite(value).all() for value in (u0, low, high)):
        raise ValueError("carryover action arrays contain non-finite values")
    if bool((low >= high).any() or (u0 < low).any() or (u0 > high).any()):
        raise ValueError("carryover action/support contract is invalid")
    return u0.copy(), low.copy(), high.copy(), history


def _plan_world(assets: dict[str, Path], boundary: dict[str, Any], seed: int = 0):
    return make_world(
        simulator_config=assets["simulator"], protocol=assets["protocol"],
        objective_profile=assets["objective"], observation_profile=assets["observation"],
        action_profile=assets["action"], boundary=boundary, seed=seed,
        asymmetric_critic=False,
    )


_RESULT_TRAJECTORY_FIELDS = (
    "executed_mask", "source_endpoint", "outcome_endpoint", "terminated", "timeout",
    "tracking_score", "position_error", "rotation_error", "reward",
    "tracking_reward", "contact_bonus", "lift_reward", "action_low", "action_high",
    "ctrl", "qpos", "qvel", "contact_flags", "observation",
)


def _result_exact(left: SequenceResult, right: SequenceResult) -> tuple[bool, dict[str, bool]]:
    checks = {
        name: getattr(left, name).dtype == getattr(right, name).dtype
        and getattr(left, name).shape == getattr(right, name).shape
        and getattr(left, name).tobytes() == getattr(right, name).tobytes()
        for name in _RESULT_TRAJECTORY_FIELDS
    }
    checks.update({
        "sequence": left.sequence.tobytes() == right.sequence.tobytes(),
        "valid_prefix": left.valid_prefix == right.valid_prefix,
        "executed_controls": left.executed_controls == right.executed_controls,
        "first_failure_endpoint": left.first_failure_endpoint == right.first_failure_endpoint,
        "first_failure_score": left.first_failure_score == right.first_failure_score,
        "prefix_tracking_reward": left.prefix_tracking_reward == right.prefix_tracking_reward,
        "strict_success": left.strict_success == right.strict_success,
    })
    return all(checks.values()), checks


def _carryover_parity(
    result: SequenceResult, history: dict[str, np.ndarray]
) -> dict[str, Any]:
    indices = np.flatnonzero((history["endpoint"] >= 41) & (history["endpoint"] <= 61))
    count = result.executed_controls
    if count != len(indices):
        return {"all_common_arrays_bitwise_equal": False, "reason": "row_count_mismatch"}
    mapping = {
        "outcome_endpoint": "endpoint", "terminated": "terminated",
        "tracking_score": "tracking_score", "position_error": "position_error",
        "rotation_error": "rotation_error", "ctrl": "ctrl", "qpos": "qpos",
        "qvel": "qvel", "contact_flags": "contact_flags",
        "sequence": "deterministic_action", "action_low": "action_low",
        "action_high": "action_high", "reward": "reward",
        "tracking_reward": "tracking_reward", "contact_bonus": "contact_bonus",
        "lift_reward": "lift_reward",
    }
    checks: dict[str, bool] = {}
    for observed_name, historical_name in mapping.items():
        observed = result.sequence[:count] if observed_name == "sequence" else getattr(result, observed_name)[:count]
        expected = history[historical_name][indices]
        checks[observed_name] = (
            observed.dtype == expected.dtype and observed.shape == expected.shape
            and observed.tobytes() == expected.tobytes()
        )
    return {
        "all_common_arrays_bitwise_equal": all(checks.values()),
        "per_array": checks,
        "compared_arrays": sorted(checks),
        "historical_rows": int(len(indices)),
    }


def _write_slot(output: Path, slot: SlotResult) -> None:
    arrays = slot.result.arrays()
    arrays.update({
        "slot_candidate_id": np.asarray(slot.candidate_id, dtype=np.int64),
        "slot_generation": np.asarray(slot.generation, dtype=np.int32),
        "slot_index": np.asarray(slot.slot, dtype=np.int32),
        "physically_evaluated": np.asarray(slot.evaluated, dtype=np.bool_),
        "original_candidate_id": np.asarray(slot.original_candidate_id, dtype=np.int64),
    })
    _write_npz_atomic(output / "candidates" / f"candidate_{slot.candidate_id:06d}.npz", arrays)


def _write_generation(output: Path, generation: GenerationResult, ledger: BudgetLedger) -> None:
    slots = generation.slots
    arrays = {
        "proposals": generation.proposals,
        "candidate_id": np.asarray([row.candidate_id for row in slots], dtype=np.int64),
        "physically_evaluated": np.asarray([row.evaluated for row in slots], dtype=np.bool_),
        "original_candidate_id": np.asarray(
            [row.original_candidate_id for row in slots], dtype=np.int64
        ),
        "valid_prefix": np.asarray([row.result.valid_prefix for row in slots], dtype=np.int32),
        "first_failure_endpoint": np.asarray([
            -1 if row.result.first_failure_endpoint is None else row.result.first_failure_endpoint
            for row in slots
        ], dtype=np.int32),
        "first_failure_score": np.asarray([
            np.nan if row.result.first_failure_score is None else row.result.first_failure_score
            for row in slots
        ], dtype=np.float64),
        "strict_success": np.asarray([row.result.strict_success for row in slots], dtype=np.bool_),
        "mean_after_refit": generation.mean,
        "std_after_refit": generation.std,
        "elite_candidate_id": np.asarray(
            [row.candidate_id for row in generation.elites], dtype=np.int64
        ),
        "observed_elites_per_row": generation.observed_elites_per_row,
    }
    _write_npz_atomic(
        output / "generations" / f"generation_{generation.generation:02d}.npz", arrays
    )
    prefixes = arrays["valid_prefix"]
    failures = arrays["first_failure_endpoint"]
    append_jsonl(output / "generations.jsonl", {
        "generation": generation.generation,
        "candidate_slots_completed": len(slots),
        "unique_physical_evaluations": int(arrays["physically_evaluated"].sum()),
        "exact_duplicate_reuses": int((~arrays["physically_evaluated"]).sum()),
        "valid_prefix_histogram": {
            str(int(value)): int((prefixes == value).sum()) for value in np.unique(prefixes)
        },
        "first_failure_endpoint_histogram": {
            str(int(value)): int((failures == value).sum()) for value in np.unique(failures)
        },
        "std": {
            "minimum": float(generation.std.min()),
            "mean": float(generation.std.mean()),
            "maximum": float(generation.std.max()),
        },
        "best_candidate_id": generation.best.candidate_id,
        "best_valid_prefix": generation.best.valid_prefix,
        "best_first_failure_endpoint": generation.best.first_failure_endpoint,
        "observed_elites_per_row": generation.observed_elites_per_row,
        "cumulative_cost": ledger.report(),
    })


def _merge_segments(first: SequenceResult, second: SequenceResult) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for name in _RESULT_TRAJECTORY_FIELDS:
        left = getattr(first, name)
        right = getattr(second, name)
        if name == "observation":
            combined = left.copy()
            mask = np.isfinite(right).all(axis=1)
            overlap = mask & np.isfinite(combined).all(axis=1)
            if overlap.any() and combined[overlap].tobytes() != right[overlap].tobytes():
                raise RuntimeError("split observation overlap differs at endpoint 60")
            combined[mask] = right[mask]
        else:
            combined = left.copy()
            mask = right if name == "executed_mask" else second.executed_mask
            combined[mask] = right[mask]
        merged[name] = combined
    merged["sequence"] = first.sequence.copy()
    return merged


def _trajectory_matches_result(arrays: dict[str, np.ndarray], result: SequenceResult) -> bool:
    return all(
        arrays[name].dtype == getattr(result, name).dtype
        and arrays[name].shape == getattr(result, name).shape
        and arrays[name].tobytes() == getattr(result, name).tobytes()
        for name in _RESULT_TRAJECTORY_FIELDS
    ) and arrays["sequence"].tobytes() == result.sequence.tobytes()


def _artifact_hashes(root: Path) -> None:
    rows = []
    target = root / "server_artifacts.sha256"
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != target):
        rows.append(f"{sha256(path)}  {path.relative_to(root)}")
    target.write_text("\n".join(rows) + "\n")


def _write_plan_summary(output: Path, decision: dict[str, Any]) -> None:
    best = decision.get("best", {})
    cost = decision["cost"]
    lines = [
        "# TACO Pour two-chunk sequence search v1", "",
        f"Status: `{decision['status']}`", "",
        "This is one local iCEM-inspired fixed-window planner run, not author-parameter recovery.",
        "No actor/critic forward, training update, automatic follow-on, or chunk commit occurred.", "",
        f"- candidate slots processed: `{decision.get('candidate_slots', 0)}`",
        f"- unique formal evaluations: `{decision.get('unique_evaluations', 0)}`",
        f"- best valid prefix: `{best.get('valid_prefix')}/40`",
        f"- best first failure endpoint: `{best.get('first_failure_endpoint')}`",
        f"- cold full validation executed: `{decision.get('cold_full_validation_executed', False)}`",
        f"- cold split-at-s60 validation executed: `{decision.get('cold_split_validation_executed', False)}`",
        f"- actual control intervals: `{cost['total_control_intervals']}` / `{cost['all_in_control_limit']}`",
        f"- actual physics steps: `{cost['total_physics_steps']}` / `{cost['all_in_physics_limit']}`",
        "- chunk committed: `false`",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def plan_window(config_path: Path, asset_root: Path | None, output: Path | None) -> dict[str, Any]:
    """Run the one authorized fixed s40-to-s80 sequence-search experiment."""
    config, root, assets = _load_plan_config(config_path, asset_root)
    output = output or root / config["run_directory"]
    state_path = output / "run_state.json"
    prior_state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if prior_state.get("search_started") or prior_state.get("terminal"):
        raise FileExistsError("this output already started formal search or reached a terminal state")
    attempt = int(prior_state.get("preflight_attempts", 0)) + 1
    if attempt > config["preflight"]["maximum_attempts"]:
        raise RuntimeError("preflight retest allowance is exhausted")
    output.mkdir(parents=True, exist_ok=True)
    dependencies = _resolve_mjcf_dependencies(assets["simulator"])
    write_json(output / "input_manifest.json", _plan_input_manifest(config_path, assets, dependencies))
    write_json(output / "frozen_config.json", config)
    ledger = BudgetLedger(
        phase_limits={
            "preflight": config["budget"]["preflight_max_control_intervals"],
            "search": config["budget"]["search_max_control_intervals"],
            "validation": config["budget"]["final_validation_reserved_control_intervals"],
        },
        all_in_limit=config["budget"]["all_in_max_control_intervals"],
        final_reserve=config["budget"]["final_validation_reserved_control_intervals"],
    )
    prior_cost_path = output / "cost_accounting.json"
    if prior_cost_path.is_file():
        prior_cost = json.loads(prior_cost_path.read_text())
        for phase in ledger.controls:
            ledger.controls[phase] = int(prior_cost.get("control_intervals", {}).get(phase, 0))
            ledger.attempts[phase] = int(prior_cost.get("candidate_attempts", {}).get(phase, 0))
    write_json(state_path, {
        "status": "PREFLIGHT_RUNNING", "preflight_attempts": attempt,
        "search_started": False, "terminal": False,
    })
    boundary = load_torch_gzip(assets["boundary"])
    validate_physics_snapshot(boundary)
    context = load_boundary_context(assets["context"])
    if not _nested_equal(boundary, context["physics_state"]):
        raise RuntimeError("formal boundary and carryover context physics state differ")
    u0, low, high, history = _load_plan_arrays(assets["carryover"])
    world = _plan_world(assets, boundary, seed=0)
    s40_window = world.get_env_state()
    for name, value in boundary.items():
        if name == "episode_lengths":
            continue
        if name not in s40_window or not _nested_equal(value, s40_window[name]):
            raise RuntimeError(f"window construction changed boundary field {name}")
    if np.asarray(s40_window["episode_lengths"]).tolist() != [80]:
        raise RuntimeError("window boundary does not end at endpoint 80")

    try:
        preflight_rng = np.random.Generator(np.random.PCG64(config["preflight"]["seed"]))
        half_range = (high.astype(np.float64) - low.astype(np.float64)) / 2.0
        probe = sample_population(
            preflight_rng, u0.astype(np.float64), 0.5 * half_range,
            low, high, 2, beta=config["search"]["noise_beta"],
        )[1]
        named_sequences = (
            ("zero_replay", np.zeros_like(u0), -4),
            ("saved_donor_sequence", u0, -3),
            ("one_colored_proposal", probe, -2),
            ("saved_donor_sequence_repeat", u0, -1),
        )
        preflight_results: dict[str, SequenceResult] = {}
        for slot, (name, sequence, candidate_id) in enumerate(named_sequences):
            result = evaluate_sequence(
                world, sequence, low, high, s40_window, ledger,
                phase="preflight", candidate_id=candidate_id,
                generation=-1, slot=slot,
            )
            preflight_results[name] = result
            _write_npz_atomic(
                output / "preflight" / f"attempt_{attempt}_{name}.npz", result.arrays()
            )
            write_json(output / "cost_accounting.json", ledger.report())
        donor = preflight_results["saved_donor_sequence"]
        repeat = preflight_results["saved_donor_sequence_repeat"]
        repeat_exact, repeat_checks = _result_exact(donor, repeat)
        parity = _carryover_parity(donor, history)
        zero = preflight_results["zero_replay"]
        preflight_report = {
            "attempt": attempt,
            "ordered_probes": [name for name, _, _ in named_sequences],
            "zero_replay": {
                "valid_prefix": zero.valid_prefix,
                "first_failure_endpoint": zero.first_failure_endpoint,
                "historical_expected_valid_prefix": 9,
                "historical_expected_first_failure_endpoint": 50,
            },
            "donor": {
                "valid_prefix": donor.valid_prefix,
                "first_failure_endpoint": donor.first_failure_endpoint,
                "historical_parity": parity,
            },
            "donor_repeat_bitwise_equal": repeat_exact,
            "donor_repeat_per_array": repeat_checks,
            "colored_probe": {
                "valid_prefix": preflight_results["one_colored_proposal"].valid_prefix,
                "first_failure_endpoint": preflight_results["one_colored_proposal"].first_failure_endpoint,
            },
            "cost": ledger.report(),
        }
        write_json(output / "preflight_report.json", preflight_report)
        preflight_passed = bool(
            zero.valid_prefix == 9 and zero.first_failure_endpoint == 50
            and donor.valid_prefix == config["preflight"]["expected_donor_prefix"]
            and donor.first_failure_endpoint == config["preflight"]["expected_donor_first_failure"]
            and parity["all_common_arrays_bitwise_equal"] and repeat_exact
        )
        if not preflight_passed:
            decision = {
                "status": "PREFLIGHT_MISMATCH", "preflight": preflight_report,
                "training_enabled": False, "chunk_commit_enabled": False,
                "automatic_followon": False, "cost": ledger.report(),
            }
            write_json(output / "decision.json", decision)
            write_json(state_path, {
                "status": decision["status"], "preflight_attempts": attempt,
                "search_started": False, "terminal": True,
            })
            _write_plan_summary(output, decision)
            _artifact_hashes(output)
            return decision
    except Exception as error:
        write_json(output / "cost_accounting.json", ledger.report())
        write_json(state_path, {
            "status": "PREFLIGHT_IMPLEMENTATION_ERROR", "preflight_attempts": attempt,
            "search_started": False, "terminal": False, "error": repr(error),
        })
        raise

    write_json(state_path, {
        "status": "FORMAL_SEARCH_RUNNING", "preflight_attempts": attempt,
        "search_started": True, "terminal": False,
    })
    donor = preflight_results["saved_donor_sequence"]
    _write_npz_atomic(output / "best_sequence.npz", donor.arrays())
    rng = np.random.Generator(np.random.PCG64(config["search"]["seed"]))

    def evaluator(sequence: np.ndarray, candidate_id: int, generation: int, slot: int):
        return evaluate_sequence(
            world, sequence, low, high, s40_window, ledger, phase="search",
            candidate_id=candidate_id, generation=generation, slot=slot,
        )

    def on_proposals(generation: int, candidate_ids: np.ndarray, proposals: np.ndarray):
        _write_npz_atomic(
            output / "generations" / f"generation_{generation:02d}_proposals.npz",
            {"candidate_id": candidate_ids, "proposal": proposals},
        )

    def on_slot(slot: SlotResult):
        _write_slot(output, slot)
        write_json(output / "cost_accounting.json", ledger.report())

    try:
        outcome = search_window(
            initial=donor, low=low, high=high, evaluator=evaluator, rng=rng,
            slots_per_generation=config["search"]["new_candidate_slots"],
            elite_count=config["search"]["elite_count"],
            elite_reuse_count=config["search"]["elite_reuse_count"],
            beta=config["search"]["noise_beta"],
            old_moment_weight=config["search"]["old_moment_weight"],
            on_proposals=on_proposals, on_slot=on_slot,
            on_generation=lambda row: _write_generation(output, row, ledger),
        )
    except BudgetStopped as error:
        decision = {
            "status": "BUDGET_STOPPED", "error": str(error),
            "training_enabled": False, "chunk_commit_enabled": False,
            "automatic_followon": False, "cost": ledger.report(),
        }
        write_json(output / "decision.json", decision)
        write_json(state_path, {
            "status": decision["status"], "preflight_attempts": attempt,
            "search_started": True, "terminal": True,
        })
        _write_plan_summary(output, decision)
        _artifact_hashes(output)
        return decision
    except Exception as error:
        write_json(output / "cost_accounting.json", ledger.report())
        write_json(output / "decision.json", {
            "status": "IMPLEMENTATION_ERROR", "error": repr(error),
            "training_enabled": False, "chunk_commit_enabled": False,
            "automatic_followon": False, "cost": ledger.report(),
        })
        write_json(state_path, {
            "status": "IMPLEMENTATION_ERROR", "preflight_attempts": attempt,
            "search_started": True, "terminal": True, "error": repr(error),
        })
        raise

    best = outcome.best
    _write_npz_atomic(output / "best_sequence.npz", best.arrays())
    _write_npz_atomic(output / "best_search_trajectory.npz", best.arrays())
    cold_full_executed = True
    cold_split_executed = False
    validation_exact = False
    split_exact = None
    snapshots: dict[int, dict[str, Any]] = {}
    cold_world = _plan_world(assets, boundary, seed=0)
    cold_boundary = cold_world.get_env_state()
    if not _nested_equal(cold_boundary, s40_window):
        raise RuntimeError("cold validation s40 differs from search s40")
    cold = evaluate_sequence(
        cold_world, best.sequence, low, high, cold_boundary, ledger,
        phase="validation", candidate_id=best.candidate_id,
        generation=best.generation, slot=best.slot,
        preserve_final_validation=False,
        snapshot_callback=lambda endpoint, state: snapshots.__setitem__(endpoint, state)
        if endpoint in {60, 80} else None,
    )
    validation_exact, validation_checks = _result_exact(best, cold)
    _write_npz_atomic(output / "best_cold_full_trajectory.npz", cold.arrays())
    status = "COMPLETED_NO_STRICT_WINDOW_SUCCESS"
    if outcome.strict_success:
        if not validation_exact or not cold.strict_success:
            status = "VALIDATION_MISMATCH"
        else:
            cold_split_executed = True
            split_first_snapshots: dict[int, dict[str, Any]] = {}
            first_world = _plan_world(assets, boundary, seed=0)
            first_boundary = first_world.get_env_state()
            first = evaluate_sequence(
                first_world, best.sequence, low, high, first_boundary, ledger,
                phase="validation", candidate_id=best.candidate_id,
                generation=best.generation, slot=best.slot,
                preserve_final_validation=False, start_row=0, stop_row=20,
                snapshot_callback=lambda endpoint, state: split_first_snapshots.__setitem__(endpoint, state)
                if endpoint == 60 else None,
            )
            s60 = split_first_snapshots.get(60)
            if s60 is None or first.valid_prefix != 20:
                raise RuntimeError("split validation did not reach endpoint 60")
            second_world = _plan_world(assets, boundary, seed=0)
            split_second_snapshots: dict[int, dict[str, Any]] = {}
            second = evaluate_sequence(
                second_world, best.sequence, low, high, s60, ledger,
                phase="validation", candidate_id=best.candidate_id,
                generation=best.generation, slot=best.slot,
                preserve_final_validation=False, start_row=20, stop_row=40,
                snapshot_callback=lambda endpoint, state: split_second_snapshots.__setitem__(endpoint, state)
                if endpoint == 80 else None,
            )
            stitched = _merge_segments(first, second)
            split_exact = bool(
                first.valid_prefix == 20 and second.valid_prefix == 20
                and _trajectory_matches_result(stitched, cold)
                and _nested_equal(s60, snapshots[60])
                and _nested_equal(split_second_snapshots.get(80), snapshots[80])
            )
            _write_npz_atomic(output / "best_cold_split_trajectory.npz", stitched)
            write_torch_gzip_atomic(output / "candidate_boundary_s60.pt.gz", s60)
            write_torch_gzip_atomic(
                output / "candidate_boundary_s80.pt.gz", split_second_snapshots[80]
            )
            _write_npz_atomic(
                output / "candidate_observation_history.npz",
                {"observation": cold.observation, "executed_mask": cold.executed_mask},
            )
            status = "STRICT_WINDOW_SEQUENCE_FOUND" if split_exact else "VALIDATION_MISMATCH"
    elif not validation_exact:
        status = "VALIDATION_MISMATCH"

    cost = ledger.report()
    decision = {
        "status": status,
        "training_enabled": False, "chunk_commit_enabled": False,
        "automatic_followon": False,
        "candidate_slots": outcome.candidate_slots,
        "unique_evaluations": outcome.unique_evaluations,
        "duplicate_reuses": outcome.duplicate_reuses,
        "cem_refits": outcome.cem_refits,
        "best": {
            "candidate_id": best.candidate_id,
            "generation": best.generation,
            "valid_prefix": best.valid_prefix,
            "first_failure_endpoint": best.first_failure_endpoint,
            "first_failure_score": best.first_failure_score,
            "strict_success": best.strict_success,
            "improved_over_donor_prefix": best.valid_prefix > donor.valid_prefix,
        },
        "preflight": preflight_report,
        "cold_full_validation_executed": cold_full_executed,
        "cold_full_bitwise_equal": validation_exact,
        "cold_full_per_array": validation_checks,
        "cold_split_validation_executed": cold_split_executed,
        "cold_split_bitwise_equal": split_exact,
        "cost": cost,
        "limitations": [
            "single seed and fixed local proposal family",
            "planner success would apply only to the frozen s40-to-s80 window",
            "no PPO-policy success or whole-task success is claimed",
            "no boundary was committed",
        ],
    }
    write_json(output / "decision.json", decision)
    write_json(output / "cost_accounting.json", cost)
    write_json(state_path, {
        "status": status, "preflight_attempts": attempt,
        "search_started": True, "terminal": True,
    })
    _write_plan_summary(output, decision)
    _artifact_hashes(output)
    return decision



