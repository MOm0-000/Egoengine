# Active run surface

This directory is intentionally limited to evidence used by the current
runtime, tests, or the active TACO Brush initialization study.

The active task evidence is the seven-run chain for
`(brush, brush, bowl)/20231006_145`:

- `taco_brush_20231006_145_acquisition_v1`
- `taco_brush_20231006_145_issue14_initialization_v1`
- `taco_brush_20231006_145_initialization_feasibility_v1`
- `taco_brush_20231006_145_safe_entry_mink_v1`
- `taco_brush_20231006_145_natural_seating_extension_v1`
- `taco_brush_20231006_145_initialization_controlled_replay_v1`
- `taco_brush_20231006_145_initialization_study_v1`

The `rl_core_refactor_r1`, `rl_active_test_surface_r1`, and
`mjwp_env_active_surface_audit_r1` directories document the active runtime
surface. The remaining four directories are immutable fixtures read directly
by the core/integration tests.

Completed older experiments were moved without deletion to
`../TRASH/retired_runs_2026-10-13/`. They are historical evidence and are not
part of the active execution or promotion chain.
