"""Two-mode variant of Appendix C.1; no MPC or replacement local-search mode.

This scheduler is backend-independent. A training adapter must snapshot *all*
state affecting transitions (physics, control, RNG, reference cursor and prior
action). Its total simulation-work counter must NOT be rolled back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class ChunkBackend(Protocol):
    def snapshot(self) -> Any: ...
    def restore(self, state: Any) -> None: ...
    def step(self, action: Any, reference_step: int) -> bool:
        """Advance one control interval; return its object-tracking feasibility."""
        ...


Policy = Callable[[ChunkBackend, int], Any]
RLTrainer = Callable[[ChunkBackend, int, int], Policy | None]


@dataclass(frozen=True)
class ModeTrial:
    mode: str
    feasible: bool
    validated_steps: int


@dataclass(frozen=True)
class ChunkResult:
    start: int
    committed_end: int
    lookahead_end: int
    mode: str | None
    trials: tuple[ModeTrial, ...]


def solve_chunk(backend: ChunkBackend, replay: Policy, train_rl: RLTrainer,
                *, start: int, total_steps: int) -> ChunkResult:
    """Validate current+next chunk; commit only the current chunk's exact state.

    Step indices denote control intervals, not raw source-video rows or physics
    substeps. Tail windows are truncated without repeating the last reference.
    Every call begins with Replay, even after an RL chunk. Failed lookahead and
    training mutations cannot leak into the next candidate or committed state.
    """
    if not isinstance(start, int) or not isinstance(total_steps, int) or not 0 <= start < total_steps:
        raise ValueError("expected integer 0 <= start < total_steps")
    committed_end = min(start + 20, total_steps)
    lookahead_end = min(start + 40, total_steps)
    initial = backend.snapshot()
    committed = initial
    trials = []
    try:
        for mode in ("replay", "rl"):
            backend.restore(initial)
            policy = replay if mode == "replay" else train_rl(backend, start, lookahead_end)
            # PPO may have advanced its training worlds or this backend.
            backend.restore(initial)
            if policy is None:
                trials.append(ModeTrial(mode, False, 0))
                continue
            accepted = initial
            valid_steps = 0
            begin_trial = getattr(backend, "begin_trial", None)
            end_trial = getattr(backend, "end_trial", None)
            if begin_trial is not None:
                begin_trial(mode, start, lookahead_end)
            try:
                for index in range(start, lookahead_end):
                    valid = backend.step(policy(backend, index), index)
                    if not isinstance(valid, bool):
                        raise ValueError("backend.step must return an explicit boolean feasibility result")
                    if not valid:
                        break
                    valid_steps += 1
                    if index + 1 == committed_end:
                        accepted = backend.snapshot()
            except Exception as error:
                if end_trial is not None:
                    end_trial(False, valid_steps, error=f"{type(error).__name__}: {error}")
                raise
            feasible = valid_steps == lookahead_end - start
            if end_trial is not None:
                end_trial(feasible, valid_steps)
            trials.append(ModeTrial(mode, feasible, valid_steps))
            if feasible:
                # Reuse the first-chunk state captured during validation. This
                # avoids another stochastic execution and extra simulator work.
                committed = accepted
                return ChunkResult(start, committed_end, lookahead_end, mode, tuple(trials))
        return ChunkResult(start, start, lookahead_end, None, tuple(trials))
    finally:
        backend.restore(committed)
