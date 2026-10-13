# Candidate C v5 fixed 400k summary

Candidate C is a local, non-paper-faithful ablation. Relative to Candidate B, the only algorithm change is H2S2R-LSTM-style actor-mean regularization (`bounds_loss_coef=0.005`); the learning rate remains constant.

| seed | 0 | 100k | 200k | 300k | 400k | status |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 30 | 38 | 37 | 29 | 33 | completed_fixed_400k_no_chunk_commit |
| 1 | 30 | 30 | 34 | 22 | 38 | completed_fixed_400k_no_chunk_commit |
| 2 | 30 | 31 | 29 | 23 | — | failed_closed_before_fixed_400k_budget |

## Frozen classification

- C1 structural stabilization: **False**
- C2 performance drift: **not evaluable at 400k**
- C3 mass-floor recurrence: **True**
- C4 strict 40/40 event: **False**
- Active Candidate-C baseline under the frozen rule: **False**

No intermediate best checkpoint was selected, no v4 checkpoint was resumed, and no chunk was committed.
