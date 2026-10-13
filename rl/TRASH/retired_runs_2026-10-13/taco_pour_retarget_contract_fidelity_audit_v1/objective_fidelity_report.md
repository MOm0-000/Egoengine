# Objective fidelity report

- Factorized objective contract certified: `False`.
- Collision representation contract certified: `False`.
- No scalar objective was fitted to visual labels.
- `global_E_bone` and `global_E_IM` remain diagnostic-only.

## Component certification

- `direct_distal_orientation_error_rad`: 8/10 comparisons; certified `False`.
- `direct_proximal_orientation_error_rad`: 9/10 comparisons; certified `False`.
- `fingertip_orientation_error_rad`: 8/10 comparisons; certified `False`.
- `fingertip_position_error_m`: 8/10 comparisons; certified `False`.
- `near_interaction_orientation_error_rad`: 8/8 comparisons; certified `True`.
- `near_interaction_position_error_m`: 8/8 comparisons; certified `True`.

## Missing or failed required evidence

- No unambiguous visual evidence: `['wrist_tray_position_error_m']`.
- Failed at least one applicable comparison: `['direct_distal_orientation_error_rad', 'direct_proximal_orientation_error_rad', 'fingertip_orientation_error_rad', 'fingertip_position_error_m']`.

## Collision representation

- Mismatches/unknowns: `25`.
- Proxy-stricter lower-wrist findings: `0`.
