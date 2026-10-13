# Objective semantics v2

- Component contract certified: `True`.
- Metrics are certified against their own analytic/FK duties, not a holistic visual vote.
- No scalar composite and no objective weights were fitted.

## Components

- `fingertip_position_error_m` — fingertip Cartesian position only: `SEMANTICALLY_CERTIFIED`.
- `wrist_to_tip_vector_error_m` — finger vector relative to its wrist: `SEMANTICALLY_CERTIFIED`.
- `thumb_to_tip_vector_error_m` — finger vector relative to thumb tip: `SEMANTICALLY_CERTIFIED`.
- `fingertip_orientation_error_rad` — fingertip/distal orientation only: `SEMANTICALLY_CERTIFIED`.
- `direct_proximal_orientation_error_rad` — proximal finger shape only: `SEMANTICALLY_CERTIFIED`.
- `direct_distal_orientation_error_rad` — distal finger shape only: `SEMANTICALLY_CERTIFIED`.
- `near_interaction_position_error_m` — finger segment position relative to tray: `SEMANTICALLY_CERTIFIED`.
- `near_interaction_orientation_error_rad` — finger segment orientation relative to tray: `SEMANTICALLY_CERTIFIED`.
- `wrist_tray_position_error_m` — wrist position relative to tray: `SEMANTICALLY_CERTIFIED`.
- `wrist_orientation_error_rad` — wrist orientation only: `SEMANTICALLY_CERTIFIED`.
- `joint_nominal_deviation_rad_l2` — limit deviation from OLD_MINK nominal: `SEMANTICALLY_CERTIFIED`.
- `temporal_joint_change_rad_l2` — frame-to-frame continuity only: `SEMANTICALLY_CERTIFIED`.

## Sample-type evidence

- `P070`: 1/8 narrow metrics move with the holistic visual preference; this is retained as a counterexample to the old all-components-must-agree rule, not used to invalidate each metric.
- `P072–P075`: 48/48 applicable ring/pinky shape and near-interaction comparisons correctly increase on the clearly worse trajectories.
- Small deterministic probes have no unambiguous holistic labels and are used only for component/FK duties.
