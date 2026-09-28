# Candidate B fixed 500k seed extension

No hyperparameter, policy, physics, or objective setting changed. Each seed continued in place from its exact epoch-62 checkpoint.

| seed | step 0 | 100k | 200k | 300k | 400k | 500k | 100k→500k | failure@500k |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30/40 | 36/40 | 30/40 | not reached | fail-closed | fail-closed | n/a | n/a |
| 1 | 30/40 | 29/40 | 34/40 | 33/40 | fail-closed | fail-closed | n/a | n/a |
| 2 | 30/40 | 37/40 | 32/40 | 35/40 | fail-closed | fail-closed | n/a | n/a |

500k median: **not evaluable**; complete 500k trajectories: **0/3**; strict successes: **0/3**.

Pre-registered cases:
- `B500_1_budget_resolves_seed_sensitivity`: `not evaluable (fixed 500k budget not completed)`
- `B500_2_partial_robustness_remains`: `not evaluable (fixed 500k budget not completed)`
- `B500_3_overtraining_or_policy_drift`: `not evaluable (fixed 500k budget not completed)`
- `B500_4_strict_success_exists`: `not evaluable (fixed 500k budget not completed)`

Fail-closed records:
- seed 0: epoch 161 last complete update; failure at 259,200 physics steps — `ValueError: truncated-Gaussian normalization mass 6.48980869e-13 is below the 1e-12 fail-closed threshold`
- seed 1: epoch 194 last complete update; failure at 312,000 physics steps — `RuntimeError: canonical old-policy gate failed: classification=rollout_canonical_likelihood_mismatch, semantic identity all true, rollout_ratio_error=1.62124634e-05, canonical_ratio_error=0`
- seed 2: epoch 191 last complete update; failure at 307,080 physics steps — `ValueError: truncated-Gaussian normalization mass 6.6346928e-13 is below the 1e-12 fail-closed threshold`

All three fixed Candidate-B continuations failed closed before 500k: two crossed the frozen truncated-Gaussian normalization-mass floor and one crossed the frozen rollout-to-canonical likelihood tolerance. Seed sensitivity is not resolved, no intermediate milestone is a fallback, and no automatic 1M continuation is authorized.

No automatic 1M continuation, checkpoint selection, or chunk commit was executed or authorized.
