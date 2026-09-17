# Pour Initial-Hand Diagnostic

Input: the preserved Pour scene, source row 0 of the complete 198-frame MINK
robot reference, the corresponding human GT targets, and unchanged first-frame
bowl/plate poses. The original reference's row 1 is used only for an independent
connection diagnostic. Original GT, object coordinates, table, scene XML and
scoring parameters are unchanged.

The user authorized a separate first-frame hand-only feasibility attempt.
This is a local engineering extension, not a TACO reset published in Section
3.2 or Appendix C. It neither replaces the source reference nor selects a
physical initial velocity, command or reset state. No physical rollout, RL,
settling, rendered video, synthetic pregrasp or new reference segment was run.

The user explicitly clarified that this is a temporary single-sample test,
not a formal pipeline stage. Generalization to later samples is unproven.
These corrections must not become automatic preprocessing or a default reset;
a formal method requires separate design, cross-sample validation and user
confirmation, and may discard these diagnostic candidates entirely.

## Method And Attempts

Post-fix historical revalidation confirms the four saved candidates' numerical
results, without rerunning pose optimization or changing a candidate. The earlier
malformed-state bug demonstrations used injected faults, not these real inputs.
See [impact corrections](/data_all/zzx/3.2RL/docs/pour_bug_impact_corrections.md).

The search varies only the two hands' 36 scalar coordinates. All 12 free-object
tangent DOFs are locked in MINK, and all 14 object qpos components are verified
unchanged. It enforces the existing joint limits and all 1,734 declared hand
pairs: 174 hand/self, 1,536 hand/object and 24 hand/table. The remaining 1,088
fixed object/object and object/table pairs are checked independently.

The local posture objective penalizes changes from the original robot row 0,
normalizing each scalar coordinate by its inherited one-frame speed envelope:
sum_j ((q_j - q_original_j) / (v_limit_j * dt_source))^2. MINK's PostureTask and
DAQP solve the local QPs. This is not a new RL reward or the authors' Eq. (1)
objective. Costs, numerical seeds and solver iterates are disclosed local
choices; no claim of a globally minimum correction is made.

Existing settings are retained: 30 Hz reference, per-QP trust bounds equal to
the inherited speed limits times 1/240 s, 0.002 m separating step, zero target
penetration, 1e-6 m acceptance tolerance, and DAQP tolerances 1e-9. Solver
iterations are not physical time. In backtracking mode only the requested
separating step is halved down to the acceptance-tolerance scale after an
infeasible QP; no collision pair or final acceptance threshold is relaxed.

| Attempt | Numerical initialization | Outcome |
| --- | --- | --- |
| v1 | Original penetrating row 0; fixed separating step | QP failed after 1 accepted iteration; failed candidate retained |
| v2 | Same row 0; separating-step backtracking | QP failed after 61 accepted iterations; failed candidate retained |
| v3 | Geometry-derived above-object seed; backtracking | Declared-feasible candidate retained at iteration 22; subsequent QP failed while trying to reduce posture cost |

The v3 numerical seed raises each hand AABB above the fixed objects' AABBs,
using a 1e-6 m numerical padding. Its right/left z shifts are 94.291/91.962 mm,
computed from current geometry rather than tuned. Optimization then brings
the hands back toward the original posture. The seed was not executed as an
action, adopted as the candidate, added to the video or inserted into the
reference. The final candidate's much smaller adjustments are below.

All three attempts are preserved. The failed attempts are explicitly named
`failed_initial_hand_candidate.npz`; the retained v3 file contains one qpos
row and diagnostic flags only, deliberately no qvel or control command.

## v3 Candidate Result

| Quantity | Right hand | Left hand |
| --- | --- | --- |
| Wrist translation delta x/y/z | 0 / 0 / +14.877 mm | -3.698 / +4.917 / +30.242 mm |
| Wrist translation norm | 14.877 mm | 30.861 mm |
| Wrist rotation change | 0.002970 rad | 0.011557 rad |
| Largest finger-joint change | 0 | 0.053412 rad |
| Initial fingertip mean fitting error, baseline | 9.131 mm | 9.356 mm |
| Initial fingertip mean fitting error, candidate | 18.127 mm | 34.952 mm |
| Native palm/table clearance, candidate | +3.870 mm | +22.522 mm |

The candidate's independent runtime audit finds zero joint or declared
hand/self, hand/object and hand/table violations at the existing tolerance.
Minimum hand/plate distance is +0.04235 mm; the lowest hand collision shell
is effectively tangent to the table (+2.50e-11 m). This is not a clearance
margin robust to subsequent forces or numerical dynamics.

Native checks do not select pairs solely by shell penetration. All 26 native
hand mesh bodies are checked against both native objects (52 AABB checks),
and the seven potentially overlapping hand/plate pairs are sampled in every
direction with a watertight counterpart, up to 256 vertices per surface.
No sampled points are inside an object. All native hand vertices are above
the table, minimum 3.870 mm. Sparse sampling is not a complete triangle-level
intersection certificate, particularly for open finger meshes.

Hand-internal coverage is still incomplete: 17 of the 82 omitted nonadjacent
shell pairs penetrate in this candidate. Native samples for the left palm
and thumb proximal link find up to 0.698 mm containment. Its classification
as assembly overlap versus a forbidden collision is unresolved. No omitted
pair or collision mesh was changed or silently exempted. The capped CoACD
object approximations retain their previously disclosed limitations.

## Connection To The Reference

The scalar joint changes needed to reach unchanged source row 1 in 1/30 s
have maximum speed/limit ratio 0.6053, with no inherited speed-limit violations.
However, a diagnostic straight/manifold interpolation sampled at 11 points
re-enters the table and plate; the original row 1 itself has about 15.060 mm
hand/table penetration. There is no collision-free interpolation certificate.

This is not a simulated Replay failure and does not imply that later reference
frames must all be projected before RL. Section 3.2.2 permits reference/contact
mismatch for object-centric refinement. No later row was edited, and no
synthetic connecting trajectory was added. Whether actual controlled motion
from a validated reset can track the objects remains a separate experiment.

## Subsequent Thumb And Control Audit

The native overlap has now been checked using complete closed-mesh Boolean
intersection and source-joint comparisons. It is pose dependent, so no blanket
assembly exemption was justified. A separately preserved v4 candidate reduces
only the left proximal-thumb rotation by 0.056940 rad relative to v3, removing
that Boolean intersection while retaining the declared constraints. Objects,
GT and the original reference remain unchanged.

The four-way instantaneous qvel/control comparison also shows that sending the
old reference command at the raised hand posture creates downward wrist forces
and a thumb force toward the old intersecting angle. No velocity/control was
adopted, no physics steps were taken, and complete collision coverage remains
unresolved. See [the follow-up report](/data_all/zzx/3.2RL/docs/pour_thumb_and_reset_inputs.md).
All v3 measurements above remain the original measurements, not v4 results.

## Status And Next Boundary

The authorized temporary diagnostic is completed. A declared-model-feasible
initial hand posture exists, but is not a formal reset or an established
generalizable initialization method. Classify the remaining native hand-internal
coverage independently of designing a formal initialization method. Any later
temporary physical comparison also requires explicit initial velocity/command
specification and the same chosen state across solver comparisons. Missing reward coefficients
are still unspecified. Training readiness, physical validation and the strict
renderer gate remain false.

The local regression suite passes 74 tests, including 10 new checks for
object locks, geometric seed scope, failed-candidate handling, preserved source
hashes, omitted-pair reporting, native checks independent of shells and the
distinction between interpolation and physical execution.

## Artifacts

- [Solver and comparison report](/data_all/zzx/3.2RL/runs/taco_pour_initial_hand_v3/report.json)
- [Candidate posture](/data_all/zzx/3.2RL/runs/taco_pour_initial_hand_v3/initial_hand_candidate.npz)
- [Independent audit](/data_all/zzx/3.2RL/runs/taco_pour_initial_hand_v3/audit.json)
- [Native geometry audit](/data_all/zzx/3.2RL/runs/taco_pour_initial_hand_v3/native_geometry_audit.json)
- [First failed attempt](/data_all/zzx/3.2RL/runs/taco_pour_initial_hand_v1/report.json)
- [Second failed attempt](/data_all/zzx/3.2RL/runs/taco_pour_initial_hand_v2/report.json)
- [Diagnostic runner](/data_all/zzx/3.2RL/scripts/diagnose_taco_initial_hands.py)
- [Regression tests](/data_all/zzx/3.2RL/tests/test_initial_hand_diagnostic.py)

Run the diagnostic runner with `--scene <preserved_scene> --baseline <v1_GT_run>
--output <new_directory> --backtrack-separation --seed-mode above_fixed_geometry`.
Run `audit_taco_initial_contacts.py` on the resulting candidate with
`--native-aabb`, writing `native_geometry_audit.json` inside the new directory,
then run the diagnostic runner again with `--audit-only`. All existing result
files are preserved; these commands reject overwrites.
