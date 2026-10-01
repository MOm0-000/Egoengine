# RL experiment ledger

## `rl_core_refactor_r1`

- Type: behavior-preserving structural refactor and bounded verification.
- Algorithm candidate: none.
- Active task classification changed: no.
- Long training: forbidden.
- Chunk commit: forbidden.
- Baseline implementation: `a1622a4` (includes B1/B2/B3 fixes).
- Historical numerical oracle: `96e8c03` plus the immutable Candidate-G
  value-isolation/full-contraction artifacts.
- Report: `runs/rl_core_refactor_r1/verification.json`.
- Legacy relocation: `TRASH/rl_core_refactor_r1_legacy_2026-10-01/`
  (`146` retired files, original hashes in `MANIFEST.sha256`; two read-only
  evidence contracts remain under `configs/`).

The run is successful only when the report says
`STRUCTURAL_REFACTOR_VERIFIED`; skipped physics checks are not a pass.

## `mjwp_env_active_surface_audit_r1`

- Type: active-call-graph audit and historical API removal.
- Physics/contact/reward/snapshot behavior: unchanged.
- Removed: H2S2R/Gym trainer compatibility and environment-owned training
  tracing; the pre-narrowing source remains in the R1 legacy TRASH archive.
- Report: `docs/mjwp_env_active_surface_audit_r1.md`.
- Long training: forbidden.
- Chunk commit: forbidden.

## `rl_active_test_surface_r1`

- Type: test-surface cleanup; no algorithm or runtime behavior change.
- Default surface: fixture-free core plus active runtime contracts.
- Explicit integration: saved-batch parity under `tests/integration_core/`.
- Historical relocation:
  `TRASH/rl_active_test_surface_r1_2026-10-01/` (55 tests, original hashes in
  `MANIFEST.sha256`).
- Long training: forbidden.
- Chunk commit: forbidden.

## `runner_surface_audit_r1`

- Type: read-only responsibility audit.
- Decision: no split; the current bounded call chain is shorter and already
  delegates hashing, model audit and physics construction to their core owners.
- Report: `docs/runner_surface_audit_r1.md`.
- Runtime behavior changed: no.

## `taco_pour_rl_train_v1`

- Type: corrected-core training enablement plus one bounded seed-0 pilot.
- Task/window: TACO `20230927_017`, `tool_only`, committed s40 to endpoint80.
- Donor: Candidate D seed2 epoch125 actor/RMS/log-sigma only.
- Fresh state: external critic and both optimizers.
- Collector corrections: raw critic value, post-action GAE mask, real endpoint
  and timeout fields, world-major RNN block starts, per-epoch h40 rebuild and
  live rollout/recomputation likelihood gate.
- S1 local decision: PPO surrogate plus `2 ×` internal value auxiliary MSE;
  external critic remains the sole GAE baseline.
- Functional gate: continuous epochs 1–2 versus cold checkpoint resume of
  epoch2, with batch/endpoints/loss/model/optimizer/RMS/RNG exact comparison.
- Training budget: one seed, 250 epochs, at most 400,000 training physics
  steps; fixed evaluations at epochs 0/62/125/188/250.
- Functional gate result: `TRAINING_CHAIN_VERIFIED`; continuous/cold-resume
  epoch-2 batch, loss/metrics, model, optimizer, RMS and RNG state matched.
- Pilot result: `COMPLETED_NO_STRICT_WINDOW_SUCCESS`. Fixed evaluations were
  `20/40`, `18/40`, `20/40`, `20/40`, and `20/40`; the final first failure was
  endpoint 61.
- Exact training cost: 400,000 physics steps, 250 actor steps and 1,000 critic
  steps. Evaluation cost: 1,030 physics steps.
- No chunk commit occurred. The frozen decision is to stop without an extra
  seed, budget extension, intermediate-checkpoint selection or parameter
  sweep.
- Lightweight report: `runs/taco_pour_rl_train_v1/summary.md`; server-only
  artifact hashes: `runs/taco_pour_rl_train_v1/server_artifacts.sha256`.
- Historical Candidate-G evidence remains valid history but is not a numeric
  baseline for the corrected collector/loss semantics.
