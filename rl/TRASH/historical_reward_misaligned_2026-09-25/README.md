# Historical reward-misaligned artifacts

These files were removed from the active Replay→RL evidence chain on
2026-09-25. The old runtime advanced the physical state from `t` to `t+1` but
scored it against the next actor goal `ref[t+2]`. The correct reward target is
`ref[t+1]`.

The archived runs, experiment-only configs, scripts, tests and documentation
remain recoverable for bug-history inspection. They must not be used to resume
a boundary, warm-start an actor, compare corrected task performance, or decide
whether a corrected chunk passes.

The archive also contains the early `taco_pour_ppo_smoke_v1` through `v4`,
`taco_pour_replay_rl_full_v2`/`v3`, and the superseded dual-backend runner
smoke.  Those runs used the same misaligned reward path and therefore are not
kept beside the active reward-aligned reports.

Still-valid engineering lessons, such as GPU non-determinism and the need for a
state-feasible action distribution, were revalidated in the corrected active
chain rather than inherited from these task scores.
