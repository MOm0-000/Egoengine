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

## `rl_active_test_surface_r1`

- Type: test-surface cleanup; no algorithm or runtime behavior change.
- Default surface: fixture-free core plus active runtime contracts.
- Explicit integration: saved-batch parity under `tests/integration_core/`.
- Historical relocation:
  `TRASH/rl_active_test_surface_r1_2026-10-01/` (55 tests, original hashes in
  `MANIFEST.sha256`).
- Long training: forbidden.
- Chunk commit: forbidden.
