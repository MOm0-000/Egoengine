# TACO Pour RL training v1

## Outcome

The corrected training chain completed its one authorized seed-0 budget, but
did not solve the committed source-40 to endpoint-80 window. The final status
is `COMPLETED_NO_STRICT_WINDOW_SUCCESS`; no chunk was committed and no extra
seed, budget extension, checkpoint selection, or parameter sweep is
authorized.

| Evaluation epoch | Training physics steps | Valid intervals | First failure | Effective prefix tracking reward |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 20/40 | 61 | 4.6509750299 |
| 62 | 99,200 | 18/40 | 59 | 5.8320815787 |
| 125 | 200,000 | 20/40 | 61 | 5.1926156115 |
| 188 | 300,800 | 20/40 | 61 | 6.6441972926 |
| 250 | 400,000 | 20/40 | 61 | 6.8730324768 |

No milestone produced a strict `40/40`. The pilot therefore provides a valid
negative algorithm result under the frozen local contract, not a promoted
trajectory or complete Pour success.

## Training-chain evidence

- Real two-epoch plus cold-resume gate: `TRAINING_CHAIN_VERIFIED`.
- Cold-resumed batch, loss/metrics, models, optimizers, RMS, and RNG were exact.
- Maximum live rollout/recomputation ratio error across all 250 training
  epochs: `3.814697265625e-06`, below the fail-closed `1e-4` limit.
- Training budget consumed exactly `400,000` physics steps, `250` actor
  optimizer steps, and `1,000` critic optimizer steps.
- Milestone evaluation consumed `1,030` physics steps. No success-confirmation
  replay was triggered.
- Default active tests: `46 passed`; explicit integration surface: `1 passed`.

The first-to-final training-batch mean tracking score changed from
`0.74717334` to `0.68200884`, while mean reward changed from `0.27734684` to
`0.32401821`. These stochastic training metrics did not translate into a
strict deterministic-window improvement at any fixed milestone.

## Evidence retention

This directory tracks the lightweight decision, verification, input manifest,
cost, frozen configuration, fixed evaluation JSON files, and epoch metrics.
Server-only checkpoints, trajectories, the full training-batch NPZ, and the
functional-gate checkpoint remain under
`/data_all/zzx/3.2RL/runs/taco_pour_rl_train_v1/`; their hashes are recorded in
`server_artifacts.sha256`. They are not copied into Git.

The experiment is a local reproduction run (`paper_faithful=false`). Its S1
auxiliary-value choice is not an author-parameter recovery and this negative
result is not attributed to S1 in isolation.
