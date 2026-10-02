"""Fixed-window, actor-free sequence search for the active Pour runtime.

This module deliberately contains one concrete optimizer.  It is inspired by
iCEM's temporally correlated sampling, but the early-termination ranking and
masked moment fitting are local engineering choices for this experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from .state_io import validate_physics_snapshot


HORIZON = 40
DOF = 36
PHYSICS_STEPS_PER_CONTROL = 10


class BudgetStopped(RuntimeError):
    """Raised before a new candidate would exceed a frozen phase budget."""


@dataclass
class BudgetLedger:
    """Run-level cost ledger that is intentionally independent of snapshots."""

    phase_limits: Mapping[str, int]
    all_in_limit: int
    final_reserve: int
    controls: dict[str, int] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.phase_limits = {str(k): int(v) for k, v in self.phase_limits.items()}
        self.controls = {name: 0 for name in self.phase_limits}
        self.attempts = {name: 0 for name in self.phase_limits}
        if self.all_in_limit <= 0 or self.final_reserve < 0:
            raise ValueError("invalid all-in budget")

    @property
    def total_controls(self) -> int:
        return sum(self.controls.values())

    def reserve_candidate(
        self, phase: str, *, maximum_controls: int = HORIZON,
        preserve_final_validation: bool = True,
    ) -> None:
        """Fail before starting a candidate whose worst case cannot fit."""
        if phase not in self.phase_limits:
            raise ValueError(f"unknown budget phase {phase}")
        maximum_controls = int(maximum_controls)
        if maximum_controls < 1:
            raise ValueError("candidate reservation must be positive")
        if self.controls[phase] + maximum_controls > self.phase_limits[phase]:
            raise BudgetStopped(f"{phase} phase has no complete-candidate capacity")
        reserve = self.final_reserve if preserve_final_validation else 0
        if self.total_controls + maximum_controls + reserve > self.all_in_limit:
            raise BudgetStopped("all-in budget has no complete-candidate capacity")
        self.attempts[phase] += 1

    def record_control(self, phase: str) -> None:
        if phase not in self.phase_limits:
            raise ValueError(f"unknown budget phase {phase}")
        if self.controls[phase] + 1 > self.phase_limits[phase]:
            raise BudgetStopped(f"{phase} phase budget exhausted during execution")
        if self.total_controls + 1 > self.all_in_limit:
            raise BudgetStopped("all-in budget exhausted during execution")
        self.controls[phase] += 1

    def report(self) -> dict[str, Any]:
        return {
            "control_intervals": dict(self.controls),
            "physics_steps": {
                name: value * PHYSICS_STEPS_PER_CONTROL
                for name, value in self.controls.items()
            },
            "candidate_attempts": dict(self.attempts),
            "total_control_intervals": self.total_controls,
            "total_physics_steps": self.total_controls * PHYSICS_STEPS_PER_CONTROL,
            "phase_control_limits": dict(self.phase_limits),
            "phase_physics_limits": {
                name: value * PHYSICS_STEPS_PER_CONTROL
                for name, value in self.phase_limits.items()
            },
            "all_in_control_limit": self.all_in_limit,
            "all_in_physics_limit": self.all_in_limit * PHYSICS_STEPS_PER_CONTROL,
            "final_validation_reserved_control_intervals": self.final_reserve,
        }


def colored_noise(
    rng: np.random.Generator, count: int, horizon: int = HORIZON,
    dof: int = DOF, beta: float = 2.5,
) -> np.ndarray:
    """Draw colored Gaussian noise, applying the FFT only along time.

    The spectral construction follows iCEM's colored-noise idea.  The retained
    DC component and analytic energy normalization are frozen local choices;
    samples are not individually centered or rescaled.
    """
    if (
        not isinstance(rng, np.random.Generator) or count < 1 or horizon < 2
        or dof < 1 or not np.isfinite(beta) or beta < 0
    ):
        raise ValueError("invalid colored-noise arguments")
    white = rng.standard_normal((count, dof, horizon))
    frequencies = np.fft.rfftfreq(horizon)
    weights = np.maximum(frequencies, 1.0 / horizon) ** (-0.5 * beta)
    energy = weights[0] ** 2
    if horizon % 2 == 0:
        energy += weights[-1] ** 2 + 2.0 * np.sum(weights[1:-1] ** 2)
    else:
        energy += 2.0 * np.sum(weights[1:] ** 2)
    scale = np.sqrt(energy / horizon)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("colored-noise spectrum has invalid energy")
    noise = np.fft.irfft(
        np.fft.rfft(white, axis=-1) * weights, n=horizon, axis=-1
    ) / scale
    result = np.transpose(noise, (0, 2, 1))
    if result.shape != (count, horizon, dof) or not np.isfinite(result).all():
        raise ValueError("colored-noise construction produced invalid samples")
    return result


def _validate_support(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(low)
    high = np.asarray(high)
    if low.shape != (HORIZON, DOF) or high.shape != (HORIZON, DOF):
        raise ValueError("action support must have shape (40, 36)")
    if low.dtype != np.float32 or high.dtype != np.float32:
        raise ValueError("action support must be stored as float32")
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("action support must be finite")
    if bool((low >= high).any()):
        raise ValueError("every action-support interval must have positive width")
    return low, high


def project_sequence(sequence: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Perform the sole float32 projection used by the search."""
    low, high = _validate_support(low, high)
    value = np.asarray(sequence, dtype=np.float64)
    if value.shape != (HORIZON, DOF) or not np.isfinite(value).all():
        raise ValueError("sequence must be finite with shape (40, 36)")
    return np.clip(value, low.astype(np.float64), high.astype(np.float64)).astype(np.float32)


def sample_population(
    rng: np.random.Generator, mean: np.ndarray, std: np.ndarray,
    low: np.ndarray, high: np.ndarray, slots: int, *, beta: float,
) -> np.ndarray:
    """Make one fixed-order population; slot zero is the projected mean."""
    low, high = _validate_support(low, high)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != low.shape or std.shape != low.shape:
        raise ValueError("mean/std shape changed")
    if slots < 1 or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("invalid population arguments")
    if bool((std < 0).any()):
        raise ValueError("standard deviation cannot be negative")
    rows = np.empty((slots, HORIZON, DOF), dtype=np.float32)
    rows[0] = project_sequence(mean, low, high)
    if slots > 1:
        noise = colored_noise(rng, slots - 1, beta=beta)
        proposals = mean[None] + std[None] * noise
        rows[1:] = np.clip(
            proposals, low.astype(np.float64), high.astype(np.float64)
        ).astype(np.float32)
    return rows


def sequence_key(sequence: np.ndarray) -> bytes:
    value = np.asarray(sequence)
    if value.shape != (HORIZON, DOF) or value.dtype != np.float32:
        raise ValueError("sequence identity requires float32 (40, 36) bytes")
    return np.ascontiguousarray(value).tobytes()


@dataclass(frozen=True)
class SequenceResult:
    candidate_id: int
    generation: int
    slot: int
    sequence: np.ndarray
    executed_mask: np.ndarray
    source_endpoint: np.ndarray
    outcome_endpoint: np.ndarray
    terminated: np.ndarray
    timeout: np.ndarray
    tracking_score: np.ndarray
    position_error: np.ndarray
    rotation_error: np.ndarray
    reward: np.ndarray
    tracking_reward: np.ndarray
    contact_bonus: np.ndarray
    lift_reward: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    ctrl: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    contact_flags: np.ndarray
    observation: np.ndarray
    valid_prefix: int
    executed_controls: int
    first_failure_endpoint: int | None
    first_failure_score: float | None
    prefix_tracking_reward: float
    strict_success: bool

    def rank_key(self) -> tuple[float, float, float, int]:
        failure_component = (
            float("inf") if self.strict_success else -float(self.first_failure_score)
        )
        return (
            float(self.valid_prefix), failure_component,
            float(self.prefix_tracking_reward), -int(self.candidate_id),
        )

    def arrays(self, *, prefix: str = "") -> dict[str, np.ndarray]:
        scalars = {
            "candidate_id": np.asarray(self.candidate_id, dtype=np.int64),
            "generation": np.asarray(self.generation, dtype=np.int32),
            "slot": np.asarray(self.slot, dtype=np.int32),
            "valid_prefix": np.asarray(self.valid_prefix, dtype=np.int32),
            "executed_controls": np.asarray(self.executed_controls, dtype=np.int32),
            "first_failure_endpoint": np.asarray(
                -1 if self.first_failure_endpoint is None else self.first_failure_endpoint,
                dtype=np.int32,
            ),
            "first_failure_score": np.asarray(
                np.nan if self.first_failure_score is None else self.first_failure_score,
                dtype=np.float64,
            ),
            "prefix_tracking_reward": np.asarray(
                self.prefix_tracking_reward, dtype=np.float64
            ),
            "strict_success": np.asarray(self.strict_success, dtype=np.bool_),
        }
        arrays = {
            "sequence": self.sequence,
            "executed_mask": self.executed_mask,
            "source_endpoint": self.source_endpoint,
            "outcome_endpoint": self.outcome_endpoint,
            "terminated": self.terminated,
            "timeout": self.timeout,
            "tracking_score": self.tracking_score,
            "position_error": self.position_error,
            "rotation_error": self.rotation_error,
            "reward": self.reward,
            "tracking_reward": self.tracking_reward,
            "contact_bonus": self.contact_bonus,
            "lift_reward": self.lift_reward,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "ctrl": self.ctrl,
            "qpos": self.qpos,
            "qvel": self.qvel,
            "contact_flags": self.contact_flags,
            "observation": self.observation,
            **scalars,
        }
        return {prefix + name: value for name, value in arrays.items()}


def better(left: SequenceResult, right: SequenceResult) -> SequenceResult:
    return left if left.rank_key() >= right.rank_key() else right


def deduplicate_results(results: Iterable[SequenceResult]) -> list[SequenceResult]:
    unique: dict[bytes, SequenceResult] = {}
    for result in results:
        key = sequence_key(result.sequence)
        if key not in unique or result.rank_key() > unique[key].rank_key():
            unique[key] = result
    return list(unique.values())


def masked_refit(
    elites: Iterable[SequenceResult], mean: np.ndarray, std: np.ndarray,
    std_floor: np.ndarray, *, old_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refit only rows observed in at least two different elite sequences."""
    elite_rows = deduplicate_results(elites)
    old_mean = np.asarray(mean, dtype=np.float64)
    old_std = np.asarray(std, dtype=np.float64)
    floor = np.asarray(std_floor, dtype=np.float64)
    if any(value.shape != (HORIZON, DOF) for value in (old_mean, old_std, floor)):
        raise ValueError("moment arrays must have shape (40, 36)")
    if not 0 <= old_weight <= 1:
        raise ValueError("old moment weight must be in [0, 1]")
    new_mean = old_mean.copy()
    new_std = old_std.copy()
    observed = np.zeros(HORIZON, dtype=np.int32)
    for row in range(HORIZON):
        values = [
            result.sequence[row].astype(np.float64)
            for result in elite_rows if bool(result.executed_mask[row])
        ]
        observed[row] = len(values)
        if len(values) < 2:
            continue
        stacked = np.stack(values)
        empirical_mean = stacked.mean(axis=0)
        empirical_std = stacked.std(axis=0, ddof=0)
        new_mean[row] = old_weight * old_mean[row] + (1.0 - old_weight) * empirical_mean
        updated_std = old_weight * old_std[row] + (1.0 - old_weight) * empirical_std
        new_std[row] = np.maximum(updated_std, floor[row])
    return new_mean, new_std, observed


def _endpoint(world: Any) -> int:
    start = np.asarray(world.start_indices).reshape(-1)
    time = np.asarray(world.time_indices).reshape(-1)
    if start.size != 1 or time.size != 1:
        raise RuntimeError("sequence search requires one scalar world cursor")
    return int(start[0] + time[0])


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _observation_row(observation: Any) -> np.ndarray:
    if isinstance(observation, dict):
        observation = observation.get("obs")
    value = np.asarray(observation, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] != 1:
        raise RuntimeError("world observation is not a single row")
    return value[0].copy()


def evaluate_sequence(
    world: Any, sequence: np.ndarray, low_table: np.ndarray, high_table: np.ndarray,
    boundary_state: dict[str, Any], ledger: BudgetLedger, *, phase: str,
    candidate_id: int, generation: int, slot: int,
    preserve_final_validation: bool = True,
    start_row: int = 0,
    stop_row: int = HORIZON,
    snapshot_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> SequenceResult:
    """Execute a normalized residual sequence from the exact full s40 state."""
    validate_physics_snapshot(boundary_state)
    low_table, high_table = _validate_support(low_table, high_table)
    sequence = np.asarray(sequence)
    if sequence.shape != (HORIZON, DOF) or sequence.dtype != np.float32:
        raise ValueError("executed sequence must be float32 with shape (40, 36)")
    if not np.isfinite(sequence).all():
        raise ValueError("executed sequence must be finite")
    if bool((sequence < low_table).any() or (sequence > high_table).any()):
        raise ValueError("sequence is outside the frozen support table")
    if not (0 <= start_row < stop_row <= HORIZON):
        raise ValueError("invalid sequence segment")
    ledger.reserve_candidate(
        phase, maximum_controls=stop_row - start_row,
        preserve_final_validation=preserve_final_validation,
    )
    world.set_env_state(boundary_state)
    if _endpoint(world) != 40 + start_row:
        raise RuntimeError("candidate did not restore the expected source endpoint")

    shape_qpos = _numpy(boundary_state["qpos"]).reshape(1, -1).shape[1]
    shape_qvel = _numpy(boundary_state["qvel"]).reshape(1, -1).shape[1]
    shape_ctrl = _numpy(boundary_state["ctrl"]).reshape(1, -1).shape[1]
    observation0 = _observation_row(world.current_observation())
    arrays: dict[str, np.ndarray] = {
        "executed_mask": np.zeros(HORIZON, dtype=np.bool_),
        "source_endpoint": np.full(HORIZON, -1, dtype=np.int32),
        "outcome_endpoint": np.full(HORIZON, -1, dtype=np.int32),
        "terminated": np.zeros(HORIZON, dtype=np.bool_),
        "timeout": np.zeros(HORIZON, dtype=np.bool_),
        "tracking_score": np.full(HORIZON, np.nan, dtype=np.float64),
        "position_error": np.full(HORIZON, np.nan, dtype=np.float64),
        "rotation_error": np.full(HORIZON, np.nan, dtype=np.float64),
        "reward": np.full(HORIZON, np.nan, dtype=np.float64),
        "tracking_reward": np.full(HORIZON, np.nan, dtype=np.float64),
        "contact_bonus": np.full(HORIZON, np.nan, dtype=np.float64),
        "lift_reward": np.full(HORIZON, np.nan, dtype=np.float64),
        "action_low": np.full((HORIZON, DOF), np.nan, dtype=np.float32),
        "action_high": np.full((HORIZON, DOF), np.nan, dtype=np.float32),
        "ctrl": np.full((HORIZON, shape_ctrl), np.nan, dtype=np.float32),
        "qpos": np.full((HORIZON, shape_qpos), np.nan, dtype=np.float32),
        "qvel": np.full((HORIZON, shape_qvel), np.nan, dtype=np.float32),
        "contact_flags": np.full((HORIZON, 2, 2, 5), False, dtype=np.bool_),
        "observation": np.full((HORIZON + 1, observation0.size), np.nan, dtype=np.float32),
    }
    arrays["observation"][start_row] = observation0
    valid_prefix = 0
    failure_endpoint: int | None = None
    failure_score: float | None = None
    prefix_reward = 0.0
    final_timeout = False
    for row in range(start_row, stop_row):
        source = _endpoint(world)
        expected_source = 40 + row
        if source != expected_source:
            raise RuntimeError(f"source endpoint mismatch: {source} != {expected_source}")
        live_low, live_high = world.current_normalized_action_bounds()
        live_low = _numpy(live_low).astype(np.float32, copy=False).reshape(1, -1)[0]
        live_high = _numpy(live_high).astype(np.float32, copy=False).reshape(1, -1)[0]
        if (
            live_low.tobytes() != low_table[row].tobytes()
            or live_high.tobytes() != high_table[row].tobytes()
        ):
            raise RuntimeError(f"action support mismatch at source endpoint {source}")
        arrays["action_low"][row] = live_low
        arrays["action_high"][row] = live_high
        try:
            observation, reward, done, info = world.step(
                sequence[row : row + 1], auto_reset=False
            )
        finally:
            # A failing simulator call may already have integrated an unknown
            # number of substeps; charge the complete control conservatively.
            ledger.record_control(phase)
        arrays["executed_mask"][row] = True
        arrays["source_endpoint"][row] = int(info["source_reference_endpoint"][0])
        arrays["outcome_endpoint"][row] = int(info["outcome_reference_endpoint"][0])
        if arrays["source_endpoint"][row] != source:
            raise RuntimeError("step reported a different source endpoint")
        outcome = source + 1
        if (
            arrays["outcome_endpoint"][row] != outcome
            or int(info["command_reference_endpoint"][0]) != outcome
            or int(info["reward_reference_endpoint"][0]) != outcome
            or int(info["next_observation_goal_reference_endpoint"][0]) != outcome + 1
        ):
            raise RuntimeError("command/reward/observation endpoint contract changed")
        terminated = bool(info["terminated"][0])
        timeout = bool(info["time_outs"][0])
        arrays["terminated"][row] = terminated
        arrays["timeout"][row] = timeout
        arrays["tracking_score"][row] = float(info["object_tracking_error"][0])
        arrays["position_error"][row] = float(info["object_position_error"][0, 0])
        arrays["rotation_error"][row] = float(info["object_rotation_error"][0, 0])
        arrays["reward"][row] = float(reward[0])
        arrays["tracking_reward"][row] = float(info["aggregate_tracking_reward"][0])
        arrays["contact_bonus"][row] = float(info["aggregate_contact_bonus"][0])
        arrays["lift_reward"][row] = float(info["lift_reward"][0])
        arrays["ctrl"][row] = _numpy(world._last_ctrl).reshape(1, -1)[0]
        arrays["qpos"][row] = _numpy(
            world._mjwp.get_qpos(world.ego_cfg, world.env)
        ).reshape(1, -1)[0]
        arrays["qvel"][row] = _numpy(
            world._mjwp.get_qvel(world.ego_cfg, world.env)
        ).reshape(1, -1)[0]
        flags = np.asarray(info["contact_flags"][0], dtype=np.bool_)
        if flags.shape != (2, 2, 5):
            raise RuntimeError("contact flag shape changed")
        arrays["contact_flags"][row] = flags
        arrays["observation"][row + 1] = _observation_row(observation)
        if snapshot_callback is not None:
            snapshot_callback(outcome, world.get_env_state())
        numeric = np.concatenate((
            arrays["qpos"][row].astype(np.float64),
            arrays["qvel"][row].astype(np.float64),
            arrays["ctrl"][row].astype(np.float64),
            np.asarray([
                arrays["tracking_score"][row], arrays["position_error"][row],
                arrays["rotation_error"][row], arrays["reward"][row],
            ]),
        ))
        if not np.isfinite(numeric).all():
            raise RuntimeError("non-finite physics/result value")
        final_timeout = timeout
        if terminated:
            failure_endpoint = outcome
            failure_score = float(arrays["tracking_score"][row])
            if not bool(done[0]):
                raise RuntimeError("tracking termination did not end the episode")
            break
        valid_prefix += 1
        prefix_reward += 1.0 - float(arrays["tracking_score"][row])
        if timeout:
            if outcome != 80 or row != HORIZON - 1 or not bool(done[0]):
                raise RuntimeError("window timed out before endpoint 80")
        elif bool(done[0]):
            raise RuntimeError("episode ended without termination or timeout")
    executed = int(arrays["executed_mask"].sum())
    strict = bool(
        start_row == 0 and stop_row == HORIZON
        and executed == HORIZON and valid_prefix == HORIZON
        and failure_endpoint is None and final_timeout
    )
    if start_row == 0 and stop_row == HORIZON and strict != bool(valid_prefix == HORIZON):
        raise RuntimeError("strict-success/window-timeout contract changed")
    return SequenceResult(
        candidate_id=int(candidate_id), generation=int(generation), slot=int(slot),
        sequence=sequence.copy(), valid_prefix=valid_prefix,
        executed_controls=executed, first_failure_endpoint=failure_endpoint,
        first_failure_score=failure_score, prefix_tracking_reward=prefix_reward,
        strict_success=strict, **arrays,
    )


@dataclass(frozen=True)
class SlotResult:
    candidate_id: int
    generation: int
    slot: int
    evaluated: bool
    original_candidate_id: int
    result: SequenceResult


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    proposals: np.ndarray
    slots: tuple[SlotResult, ...]
    elites: tuple[SequenceResult, ...]
    observed_elites_per_row: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    best: SequenceResult


@dataclass(frozen=True)
class SearchOutcome:
    best: SequenceResult
    generations: tuple[GenerationResult, ...]
    strict_success: bool
    candidate_slots: int
    unique_evaluations: int
    duplicate_reuses: int
    cem_refits: int


def search_window(
    *, initial: SequenceResult, low: np.ndarray, high: np.ndarray,
    evaluator: Callable[[np.ndarray, int, int, int], SequenceResult],
    rng: np.random.Generator, slots_per_generation: Iterable[int],
    elite_count: int = 10, elite_reuse_count: int = 3, beta: float = 2.5,
    old_moment_weight: float = 0.1,
    on_proposals: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    on_slot: Callable[[SlotResult], None] | None = None,
    on_generation: Callable[[GenerationResult], None] | None = None,
) -> SearchOutcome:
    """Run the one authorized fixed-window search."""
    low, high = _validate_support(low, high)
    if sequence_key(initial.sequence) != sequence_key(
        project_sequence(initial.sequence, low, high)
    ):
        raise ValueError("initial sequence is not exactly inside frozen support")
    if elite_count < 2 or elite_reuse_count < 0:
        raise ValueError("invalid elite configuration")
    mean = initial.sequence.astype(np.float64)
    half_range = (high.astype(np.float64) - low.astype(np.float64)) / 2.0
    std = 0.5 * half_range
    std_floor = 0.02 * half_range
    cache: dict[bytes, SequenceResult] = {sequence_key(initial.sequence): initial}
    best_seen = initial
    previous_elites: list[SequenceResult] = []
    generations: list[GenerationResult] = []
    next_id = 0
    duplicate_reuses = 0
    refits = 0
    stop = initial.strict_success
    for generation, slot_count in enumerate(tuple(int(x) for x in slots_per_generation)):
        if stop:
            break
        proposals = sample_population(
            rng, mean, std, low, high, slot_count, beta=beta
        )
        candidate_ids = np.arange(next_id, next_id + slot_count, dtype=np.int64)
        if on_proposals is not None:
            on_proposals(generation, candidate_ids.copy(), proposals.copy())
        slot_rows: list[SlotResult] = []
        for slot, candidate_id in enumerate(candidate_ids.tolist()):
            proposal = proposals[slot]
            key = sequence_key(proposal)
            if key in cache:
                result = cache[key]
                evaluated = False
                duplicate_reuses += 1
            else:
                result = evaluator(proposal, candidate_id, generation, slot)
                if sequence_key(result.sequence) != key:
                    raise RuntimeError("evaluator changed candidate action bytes")
                cache[key] = result
                evaluated = True
            slot_result = SlotResult(
                candidate_id=candidate_id, generation=generation, slot=slot,
                evaluated=evaluated, original_candidate_id=result.candidate_id,
                result=result,
            )
            slot_rows.append(slot_result)
            if on_slot is not None:
                on_slot(slot_result)
            best_seen = better(best_seen, result)
            if result.strict_success:
                stop = True
                break
        pool = deduplicate_results([
            *(slot_result.result for slot_result in slot_rows),
            *previous_elites[:elite_reuse_count],
            best_seen,
        ])
        elites = sorted(pool, key=SequenceResult.rank_key, reverse=True)[:elite_count]
        mean, std, observed = masked_refit(
            elites, mean, std, std_floor, old_weight=old_moment_weight
        )
        refits += 1
        previous_elites = elites
        report = GenerationResult(
            generation=generation, proposals=proposals,
            slots=tuple(slot_rows), elites=tuple(elites),
            observed_elites_per_row=observed, mean=mean.copy(), std=std.copy(),
            best=best_seen,
        )
        generations.append(report)
        if on_generation is not None:
            on_generation(report)
        next_id += slot_count
    return SearchOutcome(
        best=best_seen, generations=tuple(generations),
        strict_success=best_seen.strict_success,
        candidate_slots=sum(len(g.slots) for g in generations),
        unique_evaluations=len(cache) - 1,
        duplicate_reuses=duplicate_reuses, cem_refits=refits,
    )
