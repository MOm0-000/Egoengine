# Candidate G — policy warm start with recurrent boundary context

Formal classification: **G3 — NO_PREDECLARED_SUSTAINED_EXTENSION**.

Candidate G is a local, non-paper-faithful actor/RMS warm-start experiment. The critic and both optimizers are fresh, and the endpoint-40 RNN context is causally rebuilt from the committed source20–39 observation prefix.

| seed | epoch 0 | 62 / 99.2k | 125 / 200k | 188 / 300.8k | 250 / 400k | status |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 20 | 21 | 20 | 20 | 17 | completed_400k_without_strict_success_no_chunk_commit |
| 1 | 20 | 15 | 20 | 19 | 19 | completed_400k_without_strict_success_no_chunk_commit |
| 2 | 20 | 17 | 17 | 20 | 18 | completed_400k_without_strict_success_no_chunk_commit |

Milestone medians: `20 → 17 → 20 → 20 → 18`.

Final seeds reaching the local G2 reporting threshold (>=30/40): `[]`.

No arbitrary intermediate checkpoint was selected and no endpoint40→60 chunk was committed. The endpoint20→40 chunk remains the only committed chunk.

Repository verification: 822 tests + 57 subtests (19 warnings).
