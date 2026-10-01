# RL core known issues

| ID | Status | Current location / meaning |
| --- | --- | --- |
| B1 | closed at `a1622a4` | `core.ppo.prepare_batch` uses one value-RMS update/transform for old values and returns. |
| B2 | closed at `a1622a4` | `core.state_io.validate_physics_snapshot` and `MJWPVectorEnv.set_env_state` reject missing declared fields before mutation. |
| B3 | closed at `a1622a4` | independent critic freeze is authoritative in addition to any outer freeze. |
| C1 | closed in training enablement v1 | collector critic values are raw-reward values after exactly one observation normalization. |
| C2 | closed in training enablement v1 | post-action termination and pre-forward recurrent reset use separate `done_after` / `episode_start` fields. |
| C3 | closed in training enablement v1 | RNN block starts are packed world-major and live rollout likelihood is checked before the canonical PPO denominator is formed. |
| C4 | closed in training enablement v1 | endpoint/timeout rows come from required environment info; h40 is rebuilt from the immutable prefix each epoch. |
| C5 | closed in task-informed critic v2 | the active external critic input is explicitly assembled as raw actor236 + privileged108 + physical phase1; checkpoint metadata, first-layer shape, RMS shape, transition values, and bootstrap values are fail-closed on this 345-D contract. |
| S1 | resolved local design decision | actor-internal value head is a `2.0 ×` auxiliary MSE on the normalized GAE return. It is not the GAE baseline; the independent critic retains clipped value loss. |

The S1 choice is not claimed paper-faithful and is not isolated as a success
cause. The bounded seed-0 s40→80 pilot passed its training-chain gate and used
the full authorized 400,000 training physics steps, but no fixed milestone
passed strict 40/40 (`20/18/20/20/20`). Its terminal status is
`COMPLETED_NO_STRICT_WINDOW_SUCCESS`; chunk commit remains disabled and the
frozen decision forbids an automatic extra seed, extension or parameter sweep.

The task-informed critic v2 experiment is also complete without strict window
success (`20/18/20/19/19`). Its critic fit improved strongly on source40--59,
but no training batch reached source64 and no fixed evaluation exceeded the v1
interval count. This single-seed bounded result does not prove task-informed
critics useless; it only closes the exact local 345-D package and budget.

One invalid v2 invocation exposed a runtime-only `torch.flatnonzero`
compatibility error after epoch 1. The attempt is isolated and ineligible for
resume. Total execution consequently exceeded the declared all-in physics
ceiling by 420 steps; see the v2 `total_cost_accounting.json`.
