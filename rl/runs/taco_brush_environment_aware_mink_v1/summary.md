# TACO Brush environment-aware MINK v1

- Classification: `ENVIRONMENT_AWARE_MINK_FLOOR_CONSTRAINT_MISSED`
- Extension: `LOCAL_ENVIRONMENT_NONPENETRATION_EXTENSION`
- Accepted complete frames before failure: `13/209`
- Failed frame: `13`
- Failed-state native floor minimum: `-0.000100776 m`
- Frozen material tolerance: `0.000050000 m`
- Explicit self-collision prefix status: `PASS`
- Remaining brush-floor clearance: `-0.001311310 m`

The one authorized candidate passed frame 0, but its linearized QP constraint missed the independently audited native material support-plane gate at frame 13. The QP itself did not report infeasibility. No tolerance, target, weight, table, or object pose was changed, and no second candidate was run.

No complete `robot_reference.npz` is emitted because a 209-frame feasible candidate does not exist. The frozen failed state is retained in `failed_kinematic_prefix.npz`.
