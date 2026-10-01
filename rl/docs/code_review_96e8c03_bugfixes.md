# Fixes for the `96e8c03` code review

This patch closes the three implementation bugs classified as B1--B3 in
`CODE_REVIEW_96e8c03.md` without rewriting historical experiment artifacts.

## Closed bugs

- **B1 -- value-normalization coordinate drift.** Old value predictions and
  returns are now accumulated as one combined statistics update and transformed
  together under that one committed `RunningMeanStd` state. Raw advantages are
  still computed before value normalization.
- **B2 -- partial snapshot restore.** `MJWPVectorEnv` now validates the exact
  payload field set, manifest, schema/runtime metadata, RNG state, tensor and
  array types, shapes, dtypes, and world dimensions before writing any live
  state. Independent worlds are all preflighted before the first world is
  restored.
- **B3 -- ignored independent-critic freeze.** Critic optimization is disabled
  when either the outer PPO global freeze or the independent critic's local
  freeze is enabled. All four boolean combinations have regression coverage.

## Deliberately unchanged

- **S1** remains a design decision: the actor's internal auxiliary value head
  is still clipped relative to the external critic's old value. This patch does
  not silently change that objective or claim it caused a historical result.
- **R1** remains an architecture limitation, not an implementation bug: the
  privileged critic input is not expanded in this patch.
- Existing checkpoints, reports, run contracts, and their captured source
  hashes remain immutable. Historical behavior traces remain valid as facts of
  the code that produced them; algorithm comparisons affected by B1 must not be
  reclassified without a fresh post-fix run.

## Verification

The formal runtime-mirror suite passed on 2026-10-01:

```text
851 passed, 18 warnings, 57 subtests passed
```

The regression suite includes the original synthetic value-normalization
counterexample, fail-closed snapshot mutations (missing/extra field, wrong
dtype, wrong world count), bitwise state preservation after failed restores,
and all independent-critic freeze combinations.
