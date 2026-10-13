# Left initial alignment v1 findings

**Conclusion: `MIXED_EARLY_RESPONSE_WITH_ENDPOINT20_POSITION_ROTATION_IMPROVEMENT`.**

The candidate was selected from one deterministic static left-only MINK trajectory before any physics result was observed. Right-hand/object initial primitives, initial velocities, Replay commands, scene and physics were unchanged; left qpos and its consistent initial position target changed together.

- Selected solver iteration: `8`.
- This is a bounded initialization attribution, not a reset promotion, downstream feasibility result, or RL authorization.
- Early target position mean, ORIGINAL → LEFT_ALIGNED: `0.002564945 → 0.002638459 m`.
- Early target position maximum: `0.004960198 → 0.003808468 m`.
- Early target rotation mean: `0.036274602 → 0.026265356 rad`.
- Early object-frame pair translation mean: `0.009618044 → 0.006849768 m`.
- Endpoint20 target position error: `0.025284542 → 0.018787399 m`.
- Selected iteration passed both float64 and runtime-float32 geometry gates: `True`.
