# TACO Pour two-chunk sequence search v1

Status: `COMPLETED_NO_STRICT_WINDOW_SUCCESS`.

The one authorized seed-0, local iCEM-inspired fixed-window search processed
all 941 frozen candidate slots.  There were 940 new physical evaluations and
one exact-byte cache reuse.  The saved donor started at `20/40, fail@61`; the
best evaluated sequence was candidate 877 from generation 5 at
`21/40, fail@62`, with first-failure score `1.0152033567428589`.

The best sequence was replayed in a newly constructed CPU environment and all
declared trajectory arrays were bitwise equal.  Because no strict 40/40
sequence was found, the split-at-s60 success validation was not run.  No
candidate boundary was promoted or committed.

- preflight: 63 control intervals / 630 physics steps
- formal search: 16,167 control intervals / 161,670 physics steps
- cold best replay: 22 control intervals / 220 physics steps
- all-in: 16,252 / 38,040 control intervals; 162,520 / 380,400 physics steps
- actor/critic forwards, backward calls, optimizer steps: all zero
- CEM refits: 6
- training enabled: `false`
- chunk commit enabled: `false`
- automatic follow-on: `false`

This negative result applies only to this single seed, fixed proposal family,
initialization, feasibility-first ranking and finite budget.  It does not prove
the window globally infeasible or recover the unpublished EgoEngine MPC
parameters.

Server evidence root:
`/data_all/zzx/3.2RL/runs/taco_pour_two_chunk_sequence_search_v1`.
The immutable full artifact listing is `server_artifacts.sha256` with SHA256
`fff49843528f0f5d8c4aceb5024ce2ffa9f018b3b4190cb38b2e9681c8389893`.
