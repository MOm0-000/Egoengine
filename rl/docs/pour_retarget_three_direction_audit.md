# Pour Retargeting: Three-Direction Diagnosis

Follow-up: [common fingertip geometry diagnostic](pour_common_tip_geometry.md)
defines and tests one isolated native-surface/longitudinal-axis convention.
The remaining centimeter error is not repaired; that candidate is not adopted.

Input: the preserved 198-frame Pour `20230927_017` MANO-FK reference, its
unchanged human fingertip positions/orientations and wrist targets, the
isolated XHand scene, source SPIDER XHand XML/URDF, and the original IK costs.
Hand order in every result is **right, left**. This is a diagnostic study,
not a new retargeted demonstration or formal initializer.

## Outcome

The remaining centimeter error cannot be dismissed as inevitable position
unreachability. There is measured conflict between the current orientation
targets and the robot kinematics, combined with objective tradeoffs and
trajectory constraints. No additional production FK, joint-axis, range,
Jacobian-sign or position-unit defect was found in the checks below. This is
not a claim that all mapping or optimization code is correct.

The exact source-to-robot tracking-point and absolute orientation-frame
calibration remains a local convention, not recovered paper information.
No production model, tracking site, target, weight, reference, collision
pair, initial-state rule or renderer was changed in this study.

## 1. Tracking Points And Orientation Axes

All ten copied tip sites match both the isolated template and SPIDER's
`spider/assets/robots/xhand/{right,left}.xml`. They are not the same as the
fixed joints leading to links named `tip` in the corresponding URDF:

| Point definition | Nonthumb example, right | Thumb example, right |
| --- | --- | --- |
| Inherited site, distal-link local meters | `(0, .004, -.027)` | `(.04, 0, -.012)` |
| URDF named tip, local meters | `(0, 0, -.0425)` | `(.05040245, 0, 0)` |
| Distance between the two points | 16.008 mm | 15.881 mm |
| Difference in constructed distal frame | 8.427 deg | 16.735 deg |

The right middle URDF tip is at Z=-.042107 m, giving a 15.628 mm point
difference. Left nonthumb sites have Y=-.004 m. Human tips remain selected
MANO surface vertices. The previous 3.45-4.04 mm numbers measured nearest
native-surface distance, not distance to the URDF named endpoint; these are
different measurements, not contradictory estimates.

The XML sites may serve a different purpose from the named URDF endpoints.
Neither source establishes the paper's intended MANO correspondence. Simply
changing the reported measurement to the URDF endpoints at the same saved
qpos actually increases pinky position error from 23.71/23.85 mm to
37.82/37.64 mm. This is not a test of re-optimizing for those endpoints and
does not select either definition as the correct formal mapping.

The source XML/URDF files are clean in their SPIDER Git checkout at
`4bd2756720ea95b9da126d98da3bf414fe964849`. They are read-only evidence;
they were not modified or imported into the active scene during diagnosis.

### Independent Kinematic Check

All 24 actuated finger joint axes and ranges exactly match the source URDF.
An independent XML-parsed URDF FK was compared with MuJoCo at all 198 saved
configurations, in each wrist frame. Across the root and articulated hand
links, the maximum position discrepancy is 5.82e-8 m and the maximum rotation
discrepancy is 1.06e-6 rad. Fixed tip/EE marker links, fused in the MJCF, are
not counted as separately articulated bodies. No centimeter-scale model
conversion or mirrored-joint defect was found in this path.

### A Provable Orientation Conflict

Middle, ring and pinky have two parallel X-axis revolute joints each and no
independent lateral-spreading joint. Their actual tip-frame X axes must all
coincide with the wrist X axis, regardless of the finger angles. This identity
holds numerically to below 1e-12 on the saved sequence and follows directly
from their source kinematic chains.

The current target tip-frame X axes do not share this property:

| Target-axis separation, episode mean | Right | Left |
| --- | ---: | ---: |
| Middle vs ring | 15.73 deg | 13.40 deg |
| Middle vs pinky | 44.12 deg | 56.61 deg |
| Ring vs pinky | 39.49 deg | 52.94 deg |

Consequently all three current full orientation targets cannot be matched
exactly at once, even without joint-angle or speed limits. For any common
actual axis `u`, let `e_i` denote each finger's SO(3) orientation error and
`delta_ij` the angle between two desired X axes. The angular triangle
inequality gives `e_i + e_j >= delta_ij`, hence:

```text
RMS orientation error over the three fingers >= max(delta_ij) / sqrt(6)
```

The episode mean of this per-frame lower bound is 18.94/24.28 degrees,
with maxima 33.44/33.45 degrees. This is a derived diagnostic for the current
calibration, not a paper threshold, and does not prove that the calibration
itself is appropriate.

A separate fixed-wrist position check also finds that the lateral coordinates
of these three robot sites cannot change through finger flexion. After the
best common translation, the episode mean of the per-frame three-finger RMS
position lower bound is 7.87/6.73 mm **if the target wrist rotation is held
exactly fixed**. This does not bound a freely rotating wrist or the error of
any individual finger.

## 2. Objective Conflict Experiments

Five modes were checked on all 198 frames, each starting from the same saved
frame. Joint limits, all 174 declared self pairs, target points and object
poses are retained. The speed envelope is anchored to the original saved
previous frame in every branch; these independent endpoints cannot be
concatenated into a new trajectory.

The diagnostic permits up to 160 additional iterations. It uses explicitly
diagnostic-only backtracking to reject nonlinear loss increases and endpoint
constraint violations. This is not the production full-step integration rule.
Stalled line searches and iteration caps are reported separately from
convergence. All 198 endpoints in each mode pass the declared joint/self/speed
checks, but not a complete physical-feasibility gate.

| Objective | Mean right error | Mean left error | Converged / stalled / cap |
| --- | ---: | ---: | --- |
| Original saved reference | 11.810 mm | 12.266 mm | Not reassessed here |
| Original pose objective, continued | 11.785 mm | 12.249 mm | 79 / 72 / 47 |
| Separate Euclidean position and rotation terms, original costs | 11.789 mm | 12.253 mm | 79 / 71 / 48 |
| SE(3) task with fingertip rotation cost zero | 9.657 mm | 8.953 mm | 86 / 63 / 49 |
| True position + wrist orientation | 9.627 mm | 8.911 mm | 87 / 67 / 44 |
| True position only, no wrist orientation | 7.258 mm | 6.832 mm | 82 / 58 / 58 |

MINK's SE(3) translation error can depend on the target orientation even when
its explicit rotation cost is zero. The diagnostic therefore implements a
three-dimensional Euclidean site task using MuJoCo's analytic site Jacobian.
Finite-difference tests verify this Jacobian. Original MINK FrameTask errors,
Jacobians and weighted QP residuals also pass independent checks on both hands'
thumb/pinky at rows 0, 100 and 197. Splitting the full-pose loss changes the
episode means by only about .005/.004 mm: the SE(3) coupling is not the main
explanation of the centimeter error in this episode.

The original inherited costs remain position=10, tip rotation=1 and wrist
rotation=3, squared after weighting. They are not recovered author values.
No sweep or new production weight was selected.

### The First Frame Is A Useful Counterexample, Not A Reset

Keeping joint/self constraints and wrist orientation, removing only tip
orientation changes the row-0 right/left mean position error from
10.666/8.829 mm to .627/1.993 mm. Both runs converge under their own objectives.
The wrist errors in the latter are .020/.107 degrees, but the pinky orientation
errors become 118.11/88.29 degrees. The position improvement therefore comes
with a large orientation tradeoff, not a complete successful retargeting.

Even that low-position-error state still has native hand vertices below the
unchanged, uncalibrated table by 15.76/9.21 mm. It is not a legal physical
initializer. No first-frame modification was applied to the reference.

## 3. Convergence, Speed And Joint Bounds

Continuing the original objective changes the episode mean by only
.025/.016 mm. Row 0 is a numerical fixed point; three independently perturbed
seeds satisfying the declared joint/self constraints converge to the same
10.666/8.829 mm original-objective result. These are local seeds, not an
exhaustive global search. Other frames still show stalls or capped iterations.

Six rows (0, 20, 50, 100, 150, 197) were additionally solved without the
previous-frame speed envelope, retaining the same joint/self constraints:

| Row, zero-based | Position + wrist, speed envelope | Position + wrist, no speed envelope |
| --- | --- | --- |
| 0 | .627 / 1.993 mm | .627 / 1.993 mm |
| 150 | 10.426 / 8.438 mm | 2.805 / 2.527 mm |
| 197 | 9.592 / 7.935 mm | 2.072 / 2.424 mm |

With wrist orientation also removed, rows 150 and 197 reach essentially zero
site-position error on both hands while retaining joint/self checks. Their
wrist orientations deviate by roughly 28-35 degrees. Row 0 reaches .156 mm
on the right and essentially zero on the left, with roughly 36/26-degree
wrist errors. The right row-0 positional residual is small but its joint
increment criterion has not converged, so no exact optimum is claimed.

The other sampled rows demonstrate local-search sensitivity: changing legal
seeds can improve the position-only result substantially, while several
searches still stop at a constraint boundary or iteration cap. All seeds and
endpoints are isolated diagnostics, not trajectory states for execution.

Joint-bound effects were queried without applying an out-of-range step. At
row 0 under the original pose objective, removing joint limits from the local
QP lowers its predicted weighted loss from 1.01201 to 1.00307 but proposes
six out-of-range joints. Under the converged position+wrist objective at
the same row, removing limits gives no meaningful additional step. This
shows that active bounds depend on the objective; it does not justify
relaxing physical limits or blame all error on a hard workspace boundary.

## Paper Boundary And Next Decision

Section 3.2.1 specifies fingertip poses, wrist orientation, joint limits and
self-collision constraints. It does not supply the point coordinates, fixed
frame calibration, present costs, the above analytic diagnostics or these
ablation protocols. Appendix C's object success thresholds do not certify
fingertip fidelity. Removing orientation targets is only a diagnosis, not a
paper-faithful production change.

Before a formal correction, specify and verify a common geometric meaning
for each human landmark, robot tracking point and orientation frame. Then
evaluate position/orientation tradeoffs with all paper-required terms kept,
and regenerate a sequential reference before assessing speed or collision
feasibility. Do not directly switch to named URDF tips, delete orientation
losses, relax joint limits or promote these diagnostic endpoints to resets.

The frozen table and collision coverage are unchanged and still unresolved.
No Replay rollout or RL training was started.

## Reproducible Evidence

- `scripts/diagnose_taco_retarget_objectives.py`
- `runs/taco_pour_objective_audit_v2/report.json`
- The five mode JSON files in that directory, plus
  `diagnostic_endpoints_not_reference.npz`.
- `scripts/audit_taco_finger_axis_compatibility.py`
- `runs/taco_pour_finger_axis_v1/report.json`
- `tests/test_retarget_objective_diagnosis.py`

`runs/taco_pour_objective_probe_v1/` is preserved pilot evidence with raw
full-step iterations. Some endpoints violated the declared self gate; its
filtered means are not the final equal-coverage comparison. The current script
has evolved since that pilot; use the v2 report with matching current code hashes.

All 109 historical input hashes and all recorded current data/model/code
hashes in the new canonical reports were checked unchanged. The full test
suite passes 171 tests, including analytic/finite-difference checks and
independent remeasurement of all five saved 198-frame endpoint sets. Ten
legacy MANO pickle NumPy/SciPy deprecation warnings remain. Passing tests
does not certify complete collision coverage or physical task success.
