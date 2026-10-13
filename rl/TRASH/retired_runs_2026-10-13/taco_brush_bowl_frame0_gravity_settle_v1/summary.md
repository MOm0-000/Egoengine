# Brush bowl frame-0 isolated gravity-settling audit

Final verdict: **稳定落座** (`STABLE_SEATING`).

The verified Issue #14 candidate table is fixed at simulator `Z=0.717359939227 m`. The isolated model contains only the fixed plane and dynamic bowl 146; the source mass, inertia, 32 convex collision meshes, contact pairs, gravity, time step, solver, and contact parameters are unchanged.

## Frozen criteria

The final 0.5 s requires contact coverage >=99%, linear-speed P95 <=5 mm/s, angular-speed P95 <=0.05 rad/s, horizontal range <=0.5 mm, orientation range <=0.5 degrees, and collision-penetration P95 <=0.5 mm. First contact must occur by 1.0 s. Post-contact upward speed >0.05 m/s or separation >1 mm is an obvious rebound. Both velocity variants must pass.

## Results

| variant | first contact (s) | final contact | linear P95 (mm/s) | angular P95 (rad/s) | horizontal range (mm) | orientation range (deg) | penetration P95 (mm) | rebound | stable |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| recorded_frame0_velocity | 0.024000 | 100.000% | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.155143 | NO | YES |
| zero_initial_velocity | 0.026000 | 100.000% | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.155143 | NO | YES |

## Contact and net motion

| variant | initial visual gap (mm) | peak force (N) | final force / weight | max transient penetration (mm) | settled visual distance (mm) | final XY displacement (mm) | final orientation change (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| recorded_frame0_velocity | 2.640059 | 2.468565 | 1.000000 | 1.222964 | -0.157914 | 1.348977 | 2.651977 |
| zero_initial_velocity | 2.640059 | 2.901199 | 1.000000 | 1.582560 | -0.157914 | 1.476556 | 2.651691 |

The settled median normal force is compared with `mass * |gravity|`; a ratio near 1 is the expected static weight balance. Negative settled visual distance denotes the soft-contact overlap retained by the unchanged MuJoCo parameters.

## Scope

No source asset, active SupportSurfaceContract, MINK v2 trajectory, or formal scene was modified. No auxiliary constraint or external force was applied, and no parameter was tuned. Replay, MPC, and RL did not run. Passing this isolated test would not imply that the full robot scene passes.
