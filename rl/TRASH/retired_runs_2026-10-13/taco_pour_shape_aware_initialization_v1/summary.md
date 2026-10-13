# TACO Pour shape-aware initialization v1

- Old endpoint-0 geometry: `NEAR_CONTACT_NO_MATERIAL_PENETRATION`; no certified visual-material penetration, runtime minimum distance `2.0000004e-06 m`.
- The apparent contradiction is resolved as a shape/near-contact blind spot: the old gate certified material legality, not human-like intermediate-phalanx shape.
- Exactly one deterministic four-joint candidate was generated.
- Static result: `FAIL`; classification `STATIC_GATE_FAILED_NO_PHYSICS`.
- Ring joints: `[0.09431927567741433, 0.3537453919733936]` -> `[0.14068001995123688, 0.2232810586509323]` rad.
- Pinky joints: `[0.0, 0.5324345448586136]` -> `[0.04056241003234996, 0.3660361694378195]` rad.
- Ring fingertip error: `0.0363340775` -> `0.0359050605 m`.
- Pinky fingertip error: `0.0392808566` -> `0.0410375354 m` (worsened; fail-closed).
- No candidate physics was run; physics steps remain zero.
- No reset promotion, RL, planner continuation, or chunk commit is authorized.
