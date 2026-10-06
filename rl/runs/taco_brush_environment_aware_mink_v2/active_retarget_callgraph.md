# Active TACO bimanual retarget call graph

```text
v2 runner + v2 contract + minimal retarget settings
  -> taco_bimanual.retarget (typed settings)
     -> kinematic_limits
     -> support_plane_limit
     -> collision_audit / schema
     -> MINK + MuJoCo
  -> independent audits / reports / visuals
```

The active path does not call `retarget_with_mink`, a Pour candidate framework, generic reward/chunk/PPO configuration, or the archived v1 runner.
