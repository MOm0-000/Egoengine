# Progressive Candidate-D chunk solve: endpoint 40

The promoted source20 solver committed only endpoint21–40. Zero-residual Replay from the exact endpoint40 boundary passed 9/40 and wrote no endpoint60 commit.

| seed | step0 | 100k | 200k | 300.8k | 400k | result |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 9 | 17 | 20 | 20 | 14 | no strict success |
| 1 | 9 | 19 | 20 | 20 | 20 | no strict success |
| 2 | 9 | 20 | 19 | 20 | 17 | no strict success |

Fixed-milestone medians: 9 → 19 → 20 → 20 → 17.

No seed reached strict 40/40 at any predeclared fixed milestone. The three-restart protocol consumed 1,200,000 training physics steps after endpoint40; no non-strict checkpoint was selected and endpoint41–60 remains uncommitted.

The frozen protocol now permits review of an optimizer change, but does not authorize or execute one automatically. These local runs are not eligible for paper Cost comparison.
