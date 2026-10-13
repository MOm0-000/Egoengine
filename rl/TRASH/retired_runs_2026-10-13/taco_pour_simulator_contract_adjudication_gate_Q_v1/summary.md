# TACO Pour simulator-contract adjudication Gate Q v1

Gate Q is the final finite physics gate. No training, controller/action search, physics-parameter sweep, chunk acceptance, or chunk commit occurred.

## Q1 — formal actuator demand

- `Replay_zero_residual`: 3486/5760 focus-window actuator samples exceed declared effort; max ratio `267.066`.
- `frozen_single_pass_PPO`: 3530/5760 focus-window actuator samples exceed declared effort; max ratio `278.017`.
- `frozen_binary_prefix_plus_tail_semantic_oracle`: 3526/5760 focus-window actuator samples exceed declared effort; max ratio `285.639`.

## Q2 — collision/backend contract

- Passing variants: `[]`
- Unique frozen variant: `None`
- Backend-dependent physics acknowledged: `True`

## Q3 — complete-contact finite wrench

- All reference endpoints feasible: `True`
- Infeasible endpoints: `[]`

## Final decision

- Gate B/C reopened: `False`
- Exact reproduction status: `blocked_pending_unpublished_simulator_contact_actuation_object_physics_and_objective_details`
- Gate Q is closed. The protocol forbids additional physics gates or automatic fallback experiments.
