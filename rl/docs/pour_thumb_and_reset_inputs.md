# Pour Thumb Geometry And Reset-Input Diagnostics

Input: the unchanged Pour scene, all 198 rows of the original human/robot
references, the separate v3 first-frame hand-only posture, and independently
copied XHand URDFs. Objects, table, meshes, collision pairs and reward settings
remain unchanged. Only a new diagnostic posture and diagnostic reports are
produced. Nothing here is an adopted reset, executed Replay, PPO result or video.

User clarification: this is a temporary test for this one Pour sample, not a
formal pipeline stage. The first-frame posture adjustments, numerical seeds
and single-thumb search are not assumed to generalize to later samples. They
may be discarded. No automatic new-sample preprocessing or default reset may
use them; any formal replacement requires a separate design, validation across
samples and explicit user confirmation. Fixing a diagnosed modeling defect
and promoting this sample-specific workaround are different decisions.

## Paper Evidence And Local Scope

The subsequent [collision coverage review](/data_all/zzx/3.2RL/docs/pour_collision_coverage_review.md)
fixes diagnostic false-pass bugs and independently rechecks the unchanged v4
result. It distinguishes missing shapes/pairs from shell false positives and
proposes model work without applying it. The 88-test count below is this earlier
milestone; the newer review passes 107 tests.
The subsequent [bug-impact revalidation](/data_all/zzx/3.2RL/docs/pour_bug_impact_corrections.md)
recomputes the full saved thumb scan and all four control cases with no metric
changes. The injected 10 mm source mismatch was not an actual robot-model offset.
That latest review passes 119 tests and does not advance the collision proposal.

Section 3.2.1, Eq. (1), PDF pp.3-4 requires joint limits and self-collision
constraints; it does not publish the collision geometry, exemptions or TACO
physical reset. Section 3.2.2 and Appendix C.1 (p.20) prescribe reference Replay,
residual PPO and current-plus-next-chunk evaluation, not an initial qvel/control
recipe. Appendix C.2 (pp.21-22) does not resolve those choices either.

Full native-mesh Boolean intersection, joint-angle scanning/bisection, and
instantaneous control comparisons are disclosed local diagnostics. The source
URDF is from the locally copied robot asset chain, not verified EgoEngine
author code. No source omission is treated as permission for parts to intersect.

## Native Thumb Finding

The original URDF and compiled MJCF have identical thumb joint origins in
translation, axes and limits. The largest orientation difference is 1.0523e-6
rad, consistent with the MJCF export precision. There is no located explicit
permission for palm/proximal-thumb overlap.

Both native palm and proximal-thumb meshes are closed, positively oriented
solids. Their full intersection was computed with the installed Manifold
Boolean engine through trimesh, in palm coordinates. This avoids substituting
MuJoCo's convex hull for the concave source CAD surface.

| State | Right intersection | Left intersection |
| --- | --- | --- |
| Source joint zero | Empty | Empty |
| Original first frame | Empty | 204.288967 mm^3 |
| v3 hand-only candidate | Empty | 204.288967 mm^3 |
| v4 single-thumb candidate | Empty | Empty at Boolean numerical precision |

The baseline and v3 values match because their thumb angles and palm-relative
thumb transforms match: raising/rotating a whole wrist cannot remove an internal
intersection. The intersection extends 15.57-42.90 mm from the proximal rotation
axis and spans approximately 2.18 x 13.96 x 29.54 mm. A 5-by-7 left-thumb sweep
also shows pose dependence. These findings do not support a blanket assembly
exemption. They establish intersection in the available native solids, not a
hardware-manufacturer certificate about permissible real contact.

## Separate v4 Candidate

Only `left_hand_thumb_rota_joint1` changes relative to v3:

- Previous angle: 1.4969428616961766 rad.
- Candidate angle: 1.4400026504045849 rad.
- Correction: -0.05694021129159177 rad, approximately -3.2624 degrees.

The search starts from the nearest lower clear angle in the existing sweep
(1.1333333333333335 rad) and bisects toward the intersecting v3 angle. It retains
a tested empty-intersection endpoint with a bracket narrower than 1e-6 rad.
This numerical tolerance is not a physical clearance margin. Monotonicity,
global optimality and robust positive clearance are not claimed.

All other 49 qpos coordinates are byte-identical to v3, including both complete
object poses. The original 198-frame reference is untouched. The independent
audit checks frozen coordinates against the explicitly supplied preserved v3
parent, and separately checks objects against the original reference.

The unchanged declared joint and 2,822 collision-pair checks pass. Both native
palms remain above the table, with a minimum native hand clearance of 3.870 mm.
The 52 native hand/object AABB checks and resulting surface samples find no
sampled containment. Left first-frame mean fingertip error changes from
34.952 mm (v3) to 35.218 mm (v4); right-hand results remain unchanged.

Coverage is not complete. Seventeen omitted nonadjacent shell pairs still
overlap; both palm/thumb pairs are clear in full native Boolean checks, but
many other finger surfaces are open and cannot be certified by signed samples.
The broad palm/thumb shell distance remains -8.601 mm even when the native
meshes are clear. Therefore simply enabling that broad shell pair is not yet
an evidence-backed correction. The existing runtime model still lacks a
validated palm/thumb self-collision constraint: this first-frame posture change
does not prevent a later action from reintroducing native overlap.

The object CoACD approximations remain capped and uncertified at the requested
concavity. No pair was added/exempted and no geometry was shrunk/replaced.

## Initial Velocity And Command Evidence

Four combinations were evaluated at the same v4 qpos with `mj_forward` only:
all 48 qvel entries set to zero, or the original reference first-row qvel;
each paired with the original reference command or a pose-matched command.
The latter is calculated from actual actuator transmissions. The former qvel
is verified against the unchanged row0-to-row1 manifold finite difference,
not recomputed from the projected posture and not a measured robot velocity.

Generalized actuator forces at the input state:

| Velocity | Command | Right wrist z (N) | Left wrist z (N) | Left thumb rotation (Nm) |
| --- | --- | --- | --- | --- |
| Zero | Original reference | -14.8773 | -30.2420 | +17.0821 |
| Zero | Candidate pose | 0 | 0 | 0 |
| Original reference | Original reference | -14.2474 | -30.1718 | +18.8128 |
| Original reference | Candidate pose | +0.6299 | +0.0702 | +1.7307 |

Negative wrist-z force points downward. The original position command pulls
the raised wrists back down and the corrected thumb toward the original
intersecting angle. Pose matching removes the position-error component; with
nonzero qvel, actuator damping still produces force. These gains and resulting
forces come from the inherited local scene, not published author parameters.

No runtime contacts are active at this exact qpos: the minimum declared table
gaps are positive, including approximately 2.50e-11 m for the right palm shell
and 9.23e-7 m for the plate. Thus zero reported closing contacts is not evidence
of stability. With zero qvel and pose-matched control the hands still have
downward acceleration, and both passive objects accelerate at approximately
-9.81 m/s^2 vertically. Pose matching is not gravity compensation or a supported
grasp. There was no integration, settling or trajectory execution.

The copied MJWP adapter's `_write_state` currently loads the original reference
control even when supplied another qpos. Directly importing a projected posture
there would not implement a consistently specified initial state. This adapter
was inspected but not modified or used for training.

## Decision Boundary

No initial qvel or control has been selected. A zero-velocity, pose-matched
initial command is a possible local baseline for later controlled testing,
not a recovered paper setting or a guarantee of stability. It must also specify
when the first Replay command is applied; matching a command only during reset
would not prevent the next original reference command from pulling toward the
same obstacles. No synthetic warm-up, blending, gravity compensation, changed
reference or hidden settling has been added.

Next resolve the demonstrated missing self-collision coverage and remaining
open-mesh pairs. A formal initialization method remains a separate design
question, not the automatic next version of this posture adjustment. Any
later temporary physical comparison must explicitly specify its initial state
and command timing, shared by both solver comparisons. Do not require
all later reference frames to be environment-collision-free before RL. Missing
tracking weights, combined boundary C and contact coefficient stay null.

## Artifacts

- [Native source/Boolean audit](/data_all/zzx/3.2RL/runs/taco_pour_thumb_assembly_v1/report.json)
- [v4 candidate and search report](/data_all/zzx/3.2RL/runs/taco_pour_initial_hand_v4/report.json)
- [v4 independent audit](/data_all/zzx/3.2RL/runs/taco_pour_initial_hand_v4/audit.json)
- [Four-way instantaneous comparison](/data_all/zzx/3.2RL/runs/taco_pour_initial_hand_v4/initial_control_audit.json)
- [Focused tests](/data_all/zzx/3.2RL/tests/test_thumb_and_initial_controls.py)

The focused suite passed 24 tests (14 new plus the 10 prior first-frame checks).
The full regression suite passed 88 tests. The earlier 74-test milestone remains
historical; this work does not change
training readiness, physical validation or the strict renderer gate.
