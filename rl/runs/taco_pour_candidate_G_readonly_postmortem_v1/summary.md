# Candidate G read-only postmortem v1

Status: **COMPLETED_READ_ONLY_REVIEW**. Candidate G remains **G3 / NO_PREDECLARED_SUSTAINED_EXTENSION**. No simulator, network forward, gradient, optimizer, rollout, or new training was run.

## Q1 — what followed successful endpoint61 transitions?

- Recomputed source60 attempts: **1879**.
- Successful 60→61 transitions: **239**.
- Recorded source61 action rows: **229**.
- The exact **10**-row difference consists entirely of successful source60 transitions at rollout step 39, hence right-censoring at the 40-step collector boundary. It is not endpoint62 failure, and no cross-epoch join was made.
- Continuation classes: `{"continued_then_tracking_terminated_at62": 229, "right_censored_at_rollout_boundary": 10}`.

## Q2 — recorded credit

The logged value/return/raw advantage are in original shaped-reward units from the independent asymmetric critic. `return - value == raw advantage` has maximum absolute error **0**; full-batch normalized-advantage reproduction error is **4.77e-07**.

| source60 cohort | n | mean reward | mean return | mean value | mean raw advantage | mean normalized advantage |
|---|---:|---:|---:|---:|---:|---:|
| pass endpoint61 | 239 | 0.012006 | 0.019220 | 0.375342 | -0.356122 | -0.928575 |
| terminate at endpoint61 | 1640 | -0.028206 | -0.028206 | 0.326034 | -0.354240 | -0.986300 |

All **229** visible source61 actions terminate at endpoint62; their mean raw/normalized advantages are **-0.582723 / -1.497475**. The pooled source60 normalized-advantage difference (pass minus fail) is **0.057724**, while equal weighting over the **176** matched epochs gives **-0.160238**. This sign change is direct evidence of epoch-mixture confounding. See `credit_by_cohort.json` for all distributions. These are observational cohorts, not paired counterfactuals.

## Q3 — update direction

All **750** actor updates were recomputed from saved canonical old joint log-probabilities and post-update truncated-normal support arrays. Maximum disagreement with the recorded ratio aggregate is **0**, below the frozen `1e-4` diagnostic limit.

| source60 cohort | mean Δlog p | mean ratio | direction consistent | direction opposed |
|---|---:|---:|---:|---:|
| pass endpoint61 | -0.164216 | 0.897264 | 159 | 80 |
| terminate at endpoint61 | -0.173681 | 0.887675 | 1141 | 499 |

Across all updates, the per-update exact-KL mean has mean/median/max **0.059676 / 0.052109 / 0.605544**; the ratio-outside-`[0.8,1.2]` fraction has mean/max **0.518625 / 0.893750**. First-update regressions are preserved in `report.json`; epochs 1–10 and all frozen epoch-bin aggregates are in `update_by_cohort.json`, while all 750 rows are in `per_epoch_metrics.csv`.

## Static loss path

Rollout GAE uses the fresh independent asymmetric critic, but the actor optimizer still includes an internal value loss with effective coefficient **2.0** (`0.5 × critic_coef=4`). Because `separate_value_mlp=false`, this internal value head shares the actor MLP/LSTM representation. Actor-only checkpoint inheritance is therefore not actor-only training loss. This proves an influence path, not its historical gradient share.

## What cannot be inferred

- No unique root cause is claimed.
- Rollout-boundary GAE is not reconstructed without saved `last_values`.
- Saved post distributions use the rollout RMS and BPTT/reset context; they are not natural updated-policy trajectories.
- No mathematical infeasibility, reward bug, LR root cause, or new candidate authorization follows from this review.

## One next review direction

**REVIEW_ACTOR_INTERNAL_VALUE_LOSS_AND_SHARED_REPRESENTATION_CREDIT_PATH**: statically and prospectively isolate the actor's internal value-loss/shared-representation path before choosing any training change. `new_training_authorized=false`.
