# TACO Pour task-informed critic v2

## Outcome

The one authorized seed-0 experiment completed its full fixed budget and did
not solve the committed source-40 to endpoint-80 window. The terminal status is
`COMPLETED_NO_STRICT_WINDOW_SUCCESS`; no chunk was committed and no extra seed,
budget extension, checkpoint selection, or parameter sweep is authorized.

The only algorithm intervention was the external critic input:

```text
raw actor observation 236
+ raw privileged velocity/force extras 108
+ physical reference phase 1
= 345 dimensions
```

The actor, reward, action support, reference timing, physics, PPO update
settings, and deterministic acceptance rule were unchanged. This is a local
information-design experiment (`paper_faithful=false`), not recovery of an
EgoEngine or Human2Sim2Robot author dimension or parameter.

| Evaluation epoch | Training physics steps | Valid intervals | First failure | Effective prefix tracking reward |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 20/40 | 61 | 4.6509750299 |
| 62 | 99,200 | 18/40 | 59 | 5.2725666678 |
| 125 | 200,000 | 20/40 | 61 | 6.6713791713 |
| 188 | 300,800 | 19/40 | 60 | 5.5891359895 |
| 250 | 400,000 | 19/40 | 60 | 6.2592120990 |

No milestone passed strict `40/40`; the best interval count was the unchanged
epoch-0/125 result of `20/40`. The final failed endpoint-60 score was
`1.0077156`, which remains a strict failure. The prior v1 milestone sequence
was `20/18/20/20/20`; v2 therefore did not improve the fixed deterministic
window under this seed and budget.

## Functional evidence

- Baseline source commit: `fb0a512839253608f1edc310a72342d190f3b9cc`.
- Critic implementation commits: `826f6c8`, `7daf465`; runtime compatibility
  correction: `dd04ad8`.
- Epoch-0 deterministic validation reproduced all 15 common v1 trajectory
  arrays bitwise and reproduced `20/40, fail@61` before formal optimization.
- The bounded functional gate consumed its full one-retest allowance:
  continuous/cold-resume batch, loss/metrics, model, optimizer, RMS, and RNG
  state were exact; cumulative gate cost was 9,600 physics steps.
- Maximum live rollout/recomputation likelihood error over formal training was
  `5.7220458984375e-06`, below the fail-closed `1e-4` bound.
- The active default test surface completed with `52 passed`; the explicit
  saved-batch integration test completed with `1 passed`.

## Critic and tail diagnostics

The task-informed critic learned the on-policy prefix targets, but that did not
unlock multi-step tail experience:

- source40--59 pre-update GAE-return explained variance changed from
  `-0.0777` at epoch 1 to `0.9888` at epoch 250; visible-terminal MC-return
  explained variance changed from `-0.0588` to `0.9853`.
- Across all 250 batches, source60--79 contributed only 729 of 40,000 samples;
  239 epochs had at least one such row, with at most 6 tail rows in one epoch.
- Recorded action counts were source60 `654`, source61 `75`, and source64/69/74/79
  all `0`.
- A feasible endpoint61 outcome occurred 80 times across 68 epochs. Feasible
  outcomes at endpoints65/70/75/80 were all `0`.
- Tail explained variance is not interpretable as evidence of fit: each batch
  had very few tail samples and its median reported value was strongly
  negative. The GAE target is also not an independent ground-truth label.

These are descriptive on-policy diagnostics. Different policies induce
different sample distributions, so they do not isolate critic input as a
causal success or failure mechanism. The bounded result supports only the
narrow conclusion that this 345-D critic package did not unlock the window.

## Cost and invalid attempt

The valid formal experiment used exactly 400,000 training physics steps,
250 actor updates, 1,000 critic updates, and 1,010 evaluation physics steps.

An initial formal invocation completed one epoch and then failed inside the new
read-only diagnostic because the runtime Torch lacks `torch.flatnonzero`. Its
state was never resumed and remains isolated under `invalid_attempts/`. The
compatibility correction was covered by a regression test and the only allowed
functional retest before restarting fresh.

Including both functional gates and the invalid invocation, actual execution
was 412,420 physics steps, 420 above the declared all-in ceiling of 412,000.
This overrun is an implementation-accounting defect, is reported explicitly in
`total_cost_accounting.json`, and is not counted as evidence for the valid
algorithm result.

## Evidence retention

This directory tracks lightweight reports, five fixed evaluation summaries,
the exact metrics stream, the invalid-attempt record, and server artifact
hashes. Large checkpoints, trajectories, the full rollout-batch NPZ, and the
functional-gate checkpoint remain under
`/data_all/zzx/3.2RL/runs/taco_pour_rl_task_informed_critic_v2/`.

The H2S2R basis was limited to the published information-organization
principle that critic state contains actor observations plus task information:
the [official paper](https://proceedings.mlr.press/v305/lum25a.html) and the
[inspected official code revision](https://raw.githubusercontent.com/tylerlum/human2sim2robot/894eae2ec3ae39a573b81bd1860d14cc6bdfa6df/human2sim2robot/sim_training/tasks/cross_embodiment/env.py)
were not imported as runtime dependencies.
