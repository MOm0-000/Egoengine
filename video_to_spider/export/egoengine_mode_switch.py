"""EgoEngine 3.2.2 cheapest-first Replay -> MPC -> RL chunk switching.

The object-centric objective (paper Eq. 2) already exists in the SPIDER
simulator under ``paper_objective=True``; SPIDER itself does not implement the
temporal chunk decomposition or the cheapest-first solver escalation described
in EgoEngine Fig. 2.  This module adds that orchestration layer without
modifying the cloned SPIDER checkout.

Solver integration is deliberately injected as callables so the decision core
stays unit-testable on CPU and the senior's future RL residual policy can plug
into the same escalation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

SOLVER_REPLAY = "replay"
SOLVER_MPC = "mpc"
SOLVER_RL = "rl"
SOLVER_ORDER: tuple[str, ...] = (SOLVER_REPLAY, SOLVER_MPC, SOLVER_RL)


@dataclass(frozen=True)
class ModeSwitchConfig:
    chunk_steps: int = 20
    lookahead_chunks: int = 1
    object_pos_threshold_m: float = 0.1
    object_rot_threshold_rad: float = 0.3
    lambda_pos: float = 1.0
    lambda_rot: float = 1.0
    rl_enabled: bool = False


@dataclass(frozen=True)
class ChunkWindow:
    index: int
    start: int
    current_stop: int
    next_stop: int


@dataclass
class RolloutResult:
    position_error_m: np.ndarray
    rotation_error_rad: np.ndarray
    feasible: bool
    mode: str
    info: Mapping[str, Any] = field(default_factory=dict)


# A solver produces per-chunk rollout diagnostics for the half-open window
# ``[start, stop)``.  Return None when that solver is unavailable for the chunk.
Solver = Callable[[int, int], RolloutResult | None]


def feasibility_constant(config: ModeSwitchConfig) -> float:
    return float(
        np.sqrt(
            config.lambda_pos * config.object_pos_threshold_m**2
            + config.lambda_rot * config.object_rot_threshold_rad**2
        )
    )


def object_tracking_error(
    position_error_m: np.ndarray,
    rotation_error_rad: np.ndarray,
    config: ModeSwitchConfig,
) -> np.ndarray:
    """Paper Eq. (2): Euclidean position plus geodesic rotation combined.

    SPIDER's MJWP backend expresses rotation error as an axis-angle norm in
    ``qpos_diff``, matching the geodesic distance convention used by the paper.
    """
    position = np.asarray(position_error_m, dtype=np.float64)
    rotation = np.asarray(rotation_error_rad, dtype=np.float64)
    if position.shape != rotation.shape:
        raise ValueError("position and rotation error arrays must share a shape")
    return np.sqrt(
        config.lambda_pos * position**2 + config.lambda_rot * rotation**2
    )


def is_feasible(error: np.ndarray | float, constant: float) -> bool:
    values = np.asarray(error, dtype=np.float64)
    return bool(np.isfinite(values).all() and (values <= constant).all())


def chunk_windows(total_steps: int, config: ModeSwitchConfig) -> list[ChunkWindow]:
    """Decompose a trajectory into chunks with a two-chunk lookahead window.

    EgoEngine jointly solves the current and next chunks but executes only the
    current chunk.  ``current_stop`` is the execution boundary and ``next_stop``
    is the end of the lookahead window.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if config.chunk_steps < 1:
        raise ValueError("chunk_steps must be positive")
    if config.lookahead_chunks < 0:
        raise ValueError("lookahead_chunks must be non-negative")
    windows: list[ChunkWindow] = []
    index = 0
    start = 0
    while start < total_steps:
        current_stop = min(start + config.chunk_steps, total_steps)
        next_stop = min(
            start + config.chunk_steps * (1 + config.lookahead_chunks),
            total_steps,
        )
        windows.append(ChunkWindow(index, start, current_stop, next_stop))
        index += 1
        start = current_stop
    return windows


def select_solver(
    replay_feasible: bool,
    mpc_feasible: bool,
    rl_feasible: bool,
    *,
    rl_enabled: bool,
) -> str:
    """Return the cheapest feasible solver, escalating Replay -> MPC -> RL."""
    if replay_feasible:
        return SOLVER_REPLAY
    if mpc_feasible:
        return SOLVER_MPC
    if rl_enabled and rl_feasible:
        return SOLVER_RL
    return SOLVER_RL if rl_enabled else SOLVER_MPC


@dataclass
class ModeSwitchRunner:
    """Cheapest-first solver escalation over the two-chunk optimization window.

    ``replay_solver`` and ``mpc_solver`` accept ``(start, stop)`` and return a
    ``RolloutResult`` or ``None``.  ``rl_solver`` is reserved for the senior's
    residual policy and defaults to a documented unavailable stub.
    """

    replay_solver: Solver
    mpc_solver: Solver
    config: ModeSwitchConfig = field(default_factory=ModeSwitchConfig)
    rl_solver: Solver | None = None

    def _resolve_rl_solver(self) -> Solver:
        if self.config.rl_enabled:
            if self.rl_solver is None:
                raise NotImplementedError(
                    "RL residual policy is enabled in the mode-switch config but "
                    "no rl_solver callable was injected"
                )
            return self.rl_solver
        return lambda start, stop: None

    def _feasible(self, result: RolloutResult | None, constant: float) -> bool:
        if result is None or not result.feasible:
            return False
        error = object_tracking_error(
            result.position_error_m, result.rotation_error_rad, self.config
        )
        return is_feasible(error, constant)

    def run(
        self,
        total_steps: int,
        *,
        constant: float | None = None,
    ) -> dict[str, Any]:
        constant = (
            feasibility_constant(self.config) if constant is None else float(constant)
        )
        windows = chunk_windows(total_steps, self.config)
        rl_solver = self._resolve_rl_solver()
        decisions: list[dict[str, Any]] = []
        final_mode = SOLVER_MPC
        any_infeasible = False
        for window in windows:
            replay = self.replay_solver(window.start, window.current_stop)
            replay_ok = self._feasible(replay, constant)
            mpc: RolloutResult | None = None
            rl: RolloutResult | None = None
            chosen: RolloutResult | None
            if replay_ok:
                mode = SOLVER_REPLAY
                chosen = replay
            else:
                mpc = self.mpc_solver(window.start, window.current_stop)
                mpc_ok = self._feasible(mpc, constant)
                if mpc_ok:
                    mode = SOLVER_MPC
                    chosen = mpc
                else:
                    rl = rl_solver(window.start, window.current_stop)
                    rl_ok = self._feasible(rl, constant)
                    mode = select_solver(
                        False, False, rl_ok, rl_enabled=self.config.rl_enabled
                    )
                    chosen = rl if rl_ok else mpc
                    if not rl_ok:
                        any_infeasible = True
            decisions.append(
                {
                    "window_index": window.index,
                    "start": window.start,
                    "current_stop": window.current_stop,
                    "next_stop": window.next_stop,
                    "mode": mode,
                    "replay_feasible": replay_ok,
                    "mpc_feasible": (mpc is not None and mpc.feasible),
                    "rl_feasible": (rl is not None and rl.feasible),
                    "chosen_feasible": bool(chosen is not None and chosen.feasible),
                    "info": {} if chosen is None else dict(chosen.info),
                }
            )
            final_mode = mode
        return {
            "schema_version": "1.0",
            "solver_order": list(SOLVER_ORDER),
            "rl_enabled": bool(self.config.rl_enabled),
            "feasibility_constant": constant,
            "chunk_count": len(decisions),
            "decisions": decisions,
            "final_mode": final_mode,
            "any_chunk_infeasible": any_infeasible,
        }
