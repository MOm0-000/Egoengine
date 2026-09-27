# TACO Pour physics/reference validity Gate P v1

Gate P was executed in the frozen order `P1 → P2 → P0 → P3 → P4`.
No PPO training, controller tuning, task-parameter sweep, model mutation, chunk acceptance, or chunk commit occurred.

## Results

- P1: 678/1080 formal and 683/1080 feedback effort requests exceeded declared limits.
- P2: source57 same visual-patch duplicate diagnostic = True.
- P0: all three finite native-MuJoCo/MJWP parity probes passed = False.
- P3: endpoints lacking any reachable right-finger sample = [].
- P3: endpoints lacking reachable thumb + non-thumb samples = [].
- P4: current-model finite wrench LP infeasible endpoints = [48, 55, 59, 60].

## Decision

- Structural issues: `['P1_declared_effort_saturation', 'P2_convex_decomposition_duplicate_contact_patch', 'P0_native_MJWP_backend_parity_failure', 'P4_reference_wrench_infeasible_under_finite_contact_model']`
- Gate B/C remain blocked: `True`
- P5 remains an external-author-information blocker; no local objective recovery was attempted.
- Gate P is closed after these finite tests. The results are diagnostics, not global impossibility proofs.
