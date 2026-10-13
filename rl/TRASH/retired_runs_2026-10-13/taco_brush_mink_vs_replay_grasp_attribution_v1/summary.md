# Brush MINK reference vs Replay execution attribution v1

- Classification: `REFERENCE_VS_EXECUTION_GEOMETRY_ATTRIBUTED`
- Physics / MINK / MPC / RL executed: `NO`
- Geometry source: native visual meshes

## Brush 15–41

- MINK median index/thumb side cosine: `0.717135`
- Replay median index/thumb side cosine: `0.469492`

- Reference classification: index/thumb are usually on the same object side; index and middle also penetrate the native brush mesh. Replay adds loss of thumb/middle reach.
- MINK thumb median relative step: `-0.800610 mm` along the surface normal and `-1.141107 mm` in world Z.

## Bowl 52–60

- MINK median fingers within 5 mm: `3.000`
- Replay median fingers within 5 mm: `1.000`

- Reference classification: thumb/index/middle reach the bowl but remain concentrated on the same side and include native penetrations. Replay loses thumb and middle reach, leaving index-only contact.

## Left hand 44–60

- Endpoint 44 MINK proxy index/middle distance: `0.953384 mm`
- Endpoint 44 Replay proxy index/middle distance: `-2.161202 mm`

- Actual proxy-overlap endpoints in 44–60: `[44, 45]`.
- The false overlap is execution-induced and clears after endpoint 45, but finger tracking RMS remains material through endpoint 60; this audit does not claim the transient collision is the sole cause of every later error.

- Overall attribution: `BOTH_REFERENCE_GEOMETRY_AND_PHYSICAL_TRACKING_ARE_DEFECTIVE`.

Contact-side cosine and the 5 mm band are descriptive diagnostics, not pass/fail gates. See the CSV files for raw native distances, surface locations/normals, and per-actuator errors.
