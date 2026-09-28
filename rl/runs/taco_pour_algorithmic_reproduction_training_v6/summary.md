# Candidate D v6 fixed 400k summary

Candidate D is a local, non-paper-faithful ablation. Relative to Candidate B, the only algorithm change is a support-anchored bounded distribution mean; mean regularization and LR scheduling are disabled.

| seed | 0 | 100k | 200k | 300k | 400k | status |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 30 | 37 | 31 | 39 | 31 | completed_fixed_400k_no_chunk_commit |
| 1 | 30 | 34 | 31 | 35 | 36 | completed_fixed_400k_no_chunk_commit |
| 2 | 30 | 35 | 40 | 34 | 33 | completed_fixed_400k_no_chunk_commit |

Common-milestone medians: `30 → 35 → 31 → 35 → 33`.

Global minimum normalization mass: `0.499965943376135`; maximum bounded-mean outside-support count: `0`.

## Frozen classification

- D1 structural support fix: **True**
- D2 stable and useful learning: **True**
- D3 stable but performance drift: **False**
- D4 in-support mass-floor failure: **False**
- D5 strict 40/40 event: **True**
- Active Candidate-D baseline: **True**

No intermediate checkpoint was selected, no earlier candidate was resumed, and no chunk was committed.
