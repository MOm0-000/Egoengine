# Retired tail/source diagnostics (2026-10-03)

This batch removes the obsolete endpoint-44--58, source-by-source and Gate A--L
diagnostic records from the active `runs/` surface. Those experiments were useful
while the investigation treated a late-frame control failure as the primary
blocker. The current evidence instead places the unresolved discontinuity at the
startup transition, so these records are no longer active inputs or current
decision evidence.

The records were retired from Git HEAD at source commit
`90a0bb1ff6ecfc680b7797e5f193337e8d86c386`. Git history remains intact; this is
not a history rewrite.

The complete server copies are recoverable at:

`/data_all/zzx/3.2RL/TRASH/retired_tail_source_diagnostics_2026-10-03/runs/`

The server batch contains 23 run directories (107 files) plus 21 matching
experiment configs. The aggregate digests of the sorted relative-path
`sha256sum` streams are:

- runs: `fad9fcd93bb40d15de1170b0b2b5a6b061f9e4f808cb5e585971bd9ff823842d`
- configs: `352f8820f28d2cb9d44576c8b7a007c04b5b1a170a28931d703320c9ae7a2efb`

Retired runs:

- `corrected_endpoint44_48_failure_attribution_v1`
- `taco_pour_corrected_local_controllability_v1`
- `taco_pour_corrected_policy_decision_attribution_v1`
- `taco_pour_corrected_input_bifurcation_attribution_v1`
- `taco_pour_reference_timing_action_frame_audit_v1`
- `taco_pour_training_credit_assignment_audit_v1`
- `taco_pour_single_pass_endpoint47_49_gate_v1`
- `taco_pour_source47_semantic_action_subspace_gate_v1`
- `taco_pour_source46_state_entry_gate_v1`
- `taco_pour_source45_state_entry_gate_v1`
- `taco_pour_source45_prefix_source50_refinement_gate_v1`
- `taco_pour_binary_translation_oracle_gate_v1`
- `taco_pour_binary_translation_last_off_reversal_gate_v1`
- `taco_pour_source56_semantic_action_gate_v1`
- `taco_pour_tail_semantic_suppression_oracle_v1`
- `taco_pour_tail_mode_necessity_transition_audit_v1`
- `taco_pour_source57_temporal_hold_gate_v1`
- `taco_pour_source57_active_lag_correction_gate_v1`
- `taco_pour_action_feasibility_cem_v1`
- `taco_pour_action_feasibility_minimum_rho_v1`
- `taco_pour_action_feasibility_reference_object_frame_v1`
- `taco_pour_action_feasibility_gate_A1_v1`
- `taco_pour_low_level_contact_controllability_gate_L_v1`

Explicitly retained outside this batch are the current R1/task-informed-critic
inputs, `taco_pour_endpoint60_viability_adjudication_v1`, Gate P/Q, collision and
initialization contracts, observation-normalization evidence, Replay 0--20, and
the startup-transition isolation run.

The 21 matching configs and the stale historical narrative
`docs/test_environment.md` were also removed from Git HEAD. Current developers
should use `docs/ACTIVE_RUNTIME.md` and the latest startup-transition report
instead. Immutable snapshots that quote the former paths were intentionally not
rewritten; their source files remain recoverable from this archive and Git
history.
