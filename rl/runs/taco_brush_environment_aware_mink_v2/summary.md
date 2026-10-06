# TACO Brush environment-aware MINK v2

- Classification: `ENVIRONMENT_AWARE_MINK_V2_HAND_FLOOR_CLOSED_EXTERNAL_BLOCKER_REMAINS`
- Architecture contract: `PASS`
- Refactor frame 0–12 equivalence: `PASS`
- Candidate frames available: `209/209`
- Native hand-floor minimum: `-0.000015116 m`
- Native hand-floor status: `PASS`
- Self-collision status: `PASS`
- Brush-floor minimum retained: `-0.001311310 m`
- Physics / Replay / MPC / RL / promotion / chunk commit: `0`

The only candidate algorithm change is the final feasibility closure: self-only to unified self-collision plus full native support acceptance. No tolerance, task weight, target, support plane, or object pose changed.
