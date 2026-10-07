# Code difference summary

Compared with frozen baseline
`fde6692aa81a12fd6c4a9a0f3b1558b4fe35f52d`, implementation commit
`6343fa1371bcd584c73c61845dbdaf2c4a711c7b` changes only measurement,
reporting, validation selection, calibration result plumbing, tests and the
long-term handoff.

- `Plane` now normalizes both sides of `normal dot x = offset`, rejects invalid
  values and preserves normal orientation.
- Brush post-correction audits rigidly transform the exact baseline plane;
  they no longer refit a replacement near-horizontal plane.
- Validation reports every preselected background point without filtering by
  distance to the candidate plane.  It explicitly marks independent table
  validation and absolute position precision as unavailable.
- Measured target selection requires a boolean forbidden mask.  Nominal hand
  projection is excluded regardless of depth ordering.
- Finite custom and Open3D candidates retain complete transform/status data;
  formal application is a separate flag.  Component search boxes and vector
  norm acceptance bounds are recorded separately.
- Per-entity summaries distinguish missing, zero, improvement, deterioration
  and mixed outcomes.

All active `Plane(...)` callers were inspected.  Existing production callers
provide unit normals (or a `SupportSurfaceSpec` direction normalized by its
constructor); `_plane_from_coefficients` already scales normal and offset
together.  The constructor fix therefore does not introduce a second offset
scaling in those paths.  There is no evidence that the old Brush result
actually triggered the normalization defect.

The new re-audit is read-only: it reads saved corrections, full historical
planes, released poses/models and hands.  It does not read Depth video, invoke
a calibration solver, refit a table, or execute MINK/physics/Replay/MPC/RL.

The task referenced three review attachments that were not present in the
provided attachment directory: `Brush穿透问题_代码只读审计_2026-10-07.md`,
`reproduce_checks.py`, and `checks.json`.  Their absence is recorded in
`source_pins.json`; no contents were inferred or fabricated.
