# Pour MANO FK Correction And Remaining Error

Follow-up: [three-direction diagnosis](pour_retarget_three_direction_audit.md)
contains the subsequent point/axis provenance checks, objective ablations,
and convergence/constraint probes. The correction measurements below remain
historical results on the preserved reference, not newly accepted fidelity.

Input: all 198 frames (30 Hz) of `(pour in some, bowl, plate)/20230927_017`,
the released MANO poses/shapes and 21-joint GT, MANO v1.2 models, native
object meshes/poses, and the isolated two-hand XHand scene. The right hand
corresponds to bowl/tool `022`; the left hand to plate/target `135`.

## Status

The time-varying fingertip-orientation construction defect is repaired in
`src/egoengine_repro/retarget/taco_bimanual.py`. A separately named, complete
reference is saved in `runs/taco_pour_bimanual_mano_fk_v1/`. The original
`runs/taco_pour_bimanual_gt_v1/` and previous diagnostic candidates are preserved.

**The remaining 11.81/12.27 mm right/left mean fingertip errors are unresolved,
not accepted accuracy.** Removing artificial rotation jumps does not establish
correct absolute human-to-robot frame calibration, equivalent tracking-point
definitions, physical feasibility, or completed RL reproduction.

## Repair And Controlled Comparison

Previously a per-frame palm normal was projected perpendicular to each distal
bone. This can flip the constructed frame near a singularity. The replacement
transports a fixed neutral MANO fingertip frame using the released local
rotations and `smplx.lbs.batch_rigid_transform`. Neutral calibration uses zero
shape coefficients and identity joint rotations, separately for each side;
it is not fitted to this episode or smoothed over time.

The corrected rotation target is:

```text
R_sim_world @ R_MANO_global_distal @ C_MANO_neutral @ C_robot_local_tip.T
```

The MANO distal indices in thumb/index/middle/ring/pinky order are
`(15, 3, 6, 12, 9)`. Reconstruction matches every released joint on both hands
within 7.60e-8 m. An independent parent-chain rotation implementation agrees
with the library FK. These tests validate the rotation transport and source
correspondence, not the unpublished absolute MANO/XHand axis mapping.

The input positions, wrist targets, object trajectories, scene, joint limits,
174 declared hand collision pairs, solver settings, and iteration counts are
unchanged. All original human-reference arrays other than fingertip rotations
are byte-identical; saved object qpos are also byte-identical. No first-frame
posture compensation, collision-shape change, physics rollout, formal reset,
rendering, MPC, or RL training was performed.

| Metric | Original | Corrected |
| --- | ---: | ---: |
| Fingertip rotation transitions above 90 degrees | 5 | 0 |
| Maximum fingertip rotation step | 163.147 deg | 19.165 deg |
| Right mean position error | 14.924 mm | 11.810 mm |
| Left mean position error | 19.411 mm | 12.266 mm |
| Right maximum position error | 84.583 mm | 33.384 mm |
| Left maximum position error | 73.833 mm | 30.310 mm |

For example, left middle finger source keys `00103 -> 00104` now rotate
3.936 degrees instead of 160.866 degrees. The 90-degree cutoff is only an
event-reporting filter, not a paper acceptance threshold.

## Remaining Centimeter Residual

Whole-episode Euclidean position errors measured independently from MuJoCo
sites and the unchanged GT positions reproduce the saved error arrays.

| Finger | Right mean | Left mean |
| --- | ---: | ---: |
| Thumb | 11.130 mm | 11.333 mm |
| Index | 6.952 mm | 9.315 mm |
| Middle | 12.252 mm | 11.682 mm |
| Ring | 5.008 mm | 5.145 mm |
| Pinky | 23.707 mm | 23.852 mm |

Three issues require separation before interpreting these as unavoidable
human/robot morphology differences:

1. **Tracking-point and frame semantics remain unverified.** Human GT tips
   are MANO surface vertices. The inherited robot sites are offset from the
   native distal mesh surface: 3.450 mm for each nonthumb finger, and
   3.853/4.045 mm for the right/left thumb. These are unsigned nearest-surface
   distances, not established inside/outside classifications or evidence of
   manufacturer contact-pad intent. A virtual site can be legitimate, but
   its correspondence to the human landmark must be established. This offset
   alone does not account for the roughly 24 mm pinky error, and snapping a
   site to its nearest surface point is not an established semantic fix.
2. **The inherited objective does not enforce millimeter tracking.** MINK
   multiplies residuals by `cost` before squaring. The current position,
   fingertip-rotation and wrist-rotation costs are 10, 1 and 3. For pure errors,
   a 1 cm translation and a 0.1 rad (5.730 deg) fingertip rotation have equal
   weighted cost. At row 0, summed weighted rotation loss is 0.876716 versus
   0.135294 for SE(3) translation, a ratio of 6.48. Under simultaneous rotation
   error the SE(3) translation residual is not simply Euclidean displacement.
   These coefficients are inherited local choices, not recovered author
   coefficients; the loss ratio alone is not a causal ablation.
3. **More iterations are not a demonstrated first-frame solution.** Querying
   the same QP at the saved row 0 yields wrist-translation increments below
   2.12e-16 m and finger increments below 6.98e-15 rad. The proposed increment
   was not applied. This is a numerical fixed point of the current local QP,
   not proof of global optimality or position-only infeasibility. Other sampled
   frames have nonzero proposals, so convergence is not asserted for all rows.
   Joint limits are also active: the right/left distal thumb reaches its lower
   bound on 167/177 of 198 rows. Active limits do not by themselves establish
   an irreducible morphology error, and have not been relaxed.

Next diagnostic order, not changes already adopted: establish the native
robot tracking-point and absolute orientation-frame definitions; then use
isolated, explicitly labeled reachability and position/orientation-conflict
tests to separate semantic offsets, objective tradeoffs, limits and local
convergence. Do not promote those test states to a formal initializer. The
paper supplies no fingertip millimeter acceptance threshold here; object
tracking success thresholds must not be reused as a fingertip tolerance.

## Penetration Remeasurement

All table measurements below use the same **independently uncalibrated**
horizontal plane at simulator Z=0.72 m and native visual mesh vertices,
not collision shells. Negative clearance means below this assumed plane.

| Row-0 native clearance | Original | Corrected |
| --- | ---: | ---: |
| Right palm | -11.099 mm | +4.663 mm |
| Left palm | -7.746 mm | -1.994 mm |
| Right whole-hand minimum | -11.099 mm | -20.793 mm (thumb) |
| Left whole-hand minimum | -7.746 mm | -9.653 mm (thumb) |

Palm improvement must not hide the worse initial whole-hand minima. Across
all 198 rows, the deepest native hand/table intersections change from
28.431 to 24.726 mm on the right and 44.781 to 26.979 mm on the left.
Rows with native hand clearance below -50 micrometers change from 136 to
138 on the right and 191 to 180 on the left. Initial native clearances were
also cross-checked against MuJoCo's compiled mesh vertices.

The corrected left hand/plate audit finds surface samples inside the closed
native plate, up to 1.768 mm. The historical reported maximum was 1.509 mm.
These are sampled evidence, not complete-surface bounds; the historical audit
selected pairs through penetrating shells, whereas the corrected audit checks
all native hand/object AABBs independently of shells. Do not present their
maxima as an equal-coverage, exhaustive penetration comparison.

Declared joint, speed and self-pair checks pass, but coverage is incomplete:
102 same-hand pairs are omitted, with 23 omitted nonadjacent shell pairs
penetrating over the corrected trajectory. Shell overlap is not automatically
native mesh intersection. Both object decompositions still hit the 32-part
cap; neither decomposition nor the pair contract was changed in this repair.

## Independent Table Check

The source height 0.549062228 m comes from the first-frame native plate
minimum, then is translated to 0.72 m. Paper Appendix A.1 specifies the 0.6 m
object-center offset and 0.72 m table; A.3 describes approximate alignment.
It does not identify the first plate minimum as a measured tabletop plane.

The available Pour depth video in this bundle is H.264 `yuv420p`, 512x376, with
198 frames, so its pixels were not interpreted as metric distances. This is an
availability statement about the local bundle, not a statement that TACO lacks
metric depth: the official TACO README defines the original depth AVI as uint16
with scale 4000 (0.25 mm per raw unit). The original Pour member is still absent
from this local bundle and must be acquired before using depth for calibration.

An independent RGB/extrinsic check used eight frames and excluded projected
hand/object regions from tabletop feature support. The camera baseline from
the first frame reaches 73.9 mm, but there were no reliable tabletop
correspondences after filtering. No plane was fitted or adopted. The roughly
Z-aligned object geometry and plate-height consistency are not independent
measurements of the real tabletop.

Table calibration remains unresolved. It needs original metric depth with
verified units/registration, or independently measured plane and footprint.
No table height, rotation or footprint was adjusted to reduce robot
penetration. Table clearance and hand-to-hand tracking residual are distinct
measurements: applying a common rigid transform to two tracking points does
not change their Euclidean separation.

## Evidence And Verification

- `runs/taco_pour_bimanual_mano_fk_v1/input_audit.json`: source reconstruction
  and orientation provenance.
- `runs/taco_pour_bimanual_mano_fk_v1/final_comparison.json`: controlled
  before/after results; use this instead of preliminary `comparison.json`.
- `runs/taco_pour_bimanual_mano_fk_v1/residual_audit_final.json`: per-finger
  errors, native site offsets, weighted losses and unapplied QP queries.
- `runs/taco_pour_bimanual_mano_fk_v1/initialization_audit.json`: omitted pairs.
- `runs/taco_pour_table_calibration_v1/report.json`: unresolved table evidence.

The 109 historical data/model/reference input hashes remain unchanged. All
recorded current input and code hashes in the new input, final comparison,
final residual and table reports were verified against the filesystem.

The full suite passes **139 tests**, with 10 legacy MANO NumPy/SciPy pickle
deprecation warnings. Tests verify the defect repair and diagnostics, not
physical feasibility or task success. The selected downstream workflow remains
Replay -> residual PPO without MPC, but this reference has not passed a formal
initialization or physics validation gate.
