# Post-Fix Revalidation And Interpretation Corrections

Input: the preserved Pour scene, original 198-frame human/robot references,
four temporary first-frame candidates, three saved numerical-iteration traces,
16 historical diagnostic JSON reports, source URDF copies and native/collision
meshes. This review tests the impact of previously fixed audit bugs on existing
evidence. It does not advance the collision-model proposal or design a reset.

## Outcome

All 33 groups of recomputed historical metrics agree with their recorded values
within the disclosed numerical comparison tolerance. No actual invalid Pour
numeric state, mismatching recorded mesh hash, or failing actual thumb-source
kinematic comparison was found. No historical numerical result was therefore
retracted or rewritten. The interpretation of the bug demonstrations and the
limits of the feasibility claims are clarified below.

Scope correction from the subsequent input-logic investigation: agreement of
these 33 measurements does not exclude shared upstream errors. A separate
audit found artificial fingertip-orientation jumps in the saved targets and
unverified tabletop assumptions, and isolated the IK wrist descent responsible
for the initial palm-height change. See
[input-logic findings](/data_all/zzx/3.2RL/docs/pour_input_logic_findings.md).
The original measurements remain reproducible, but this reference must not be
treated as a clean test of correctly supplied fingertip-pose GT.

The previous review had screened all reference rows for numeric validity but
recomputed selected first-frame results. This review goes further: it recomputes
the whole 198-frame collision report, FK fitting metrics, derivative/control
arrays and omitted-pair topology, not merely their saved summaries.

## What The Bug Demonstrations Actually Established

| Demonstration | Correct interpretation after checking the real artifacts |
| --- | --- |
| NaN/Inf coordinate accepted as feasible | A malformed-state test exposed a real guard defect. The corruption was injected into a temporary array; it was not found in the saved Pour references or candidates. |
| Zero/nonunit quaternion accepted | Another injected malformed-state test. All 289 saved Pour qpos rows screened here have valid unit quaternions; none was normalized or repaired. |
| 10 mm joint-origin shift still labeled agreement | The shift was applied to an in-memory test model. The actual checked thumb joints do not have that shift. The corrected numerical/topology checks pass for the four audited joints. This is not verification of every joint or the entire robot model. |
| Old native report could ignore changed mesh hashes | A forged-hash/modified-fixture test exposed an input-binding defect. The 92 actual current scene mesh files match recorded historical mesh/provenance hashes. No actual hash mismatch was found. |

Finding a vulnerable code path does not establish that real historical data
triggered it. Conversely, agreement on current artifacts does not invalidate
the bugs: the negative tests still must fail closed for malformed future inputs.

The historical reports were retained unchanged. Recomputed artifacts and this
correction document are separate, so readers can inspect both the earlier
evidence and the post-fix validation. Hash agreement compares recorded/current
endpoints; it cannot prove that no unrecorded intermediate change ever occurred.

## Checks Executed

- Numeric screening of all eight Pour qpos archives: 198 reference rows, four
  candidate rows and 87 numerical-iteration rows, for 289 rows total. Available
  qvel/control arrays were checked for shape and finiteness. Maximum reference
  quaternion-norm error was 2.22e-16; no correction was applied.
- All 198 FK fingertip/wrist errors, joint-limit margins, interval velocity
  ratios, manifold qvel and original position commands were recomputed and
  compared with the NPZ and retarget report. Both object trajectories were
  compared against aligned GT at every source row.
- All 198 rows of every collision family and the original omitted-pair/native
  topology audit were recomputed, including recorded worst-pair diagnostics.
- Original first-frame hand/object native samples were repeated. Four candidate
  state/posture/next-reference diagnostics were repeated. The v3/v4 native AABB,
  table-clearance and omitted-pair sample results were recomputed as well.
- All six saved zero/original/v3 left/right thumb Boolean results and all 35
  saved joint-sweep results were recomputed. Both v4 final thumb intersections
  were recomputed separately. The corrected source check compares four thumb
  joints, with conditional rather than hardcoded agreement.
- All four v4 instantaneous velocity/control cases were recomputed, including
  actuator forces, accelerations and contact records. No time integration ran.
- Both objects' table clearances were recomputed for all 198 rows. The original
  2,048-point-per-object surface samples were repeated in both directions at
  all 13 previously selected diagnostic frames, without changing sample counts
  or replacing failed evidence with easier frames.
- 291 recorded input-hash entries from 16 old reports were verified. The 92
  current mesh dependencies were matched to historical records, including all
  64 CoACD pieces through their existing provenance. Code hashes were not
  expected to match old versions because the code was intentionally fixed.
- Three historical Brush robot references, totaling 627 rows, received numeric
  screening only. Their full collision results were not recomputed here;
  the Pour-specific conclusion must not be extended to those experiments.

An additional regression independently recomputes all 28 original native
hand/object table-support measurements, including the two palm penetrations.
Comparison tests reject changed booleans/counts, missing fields, changed list
lengths and nonfinite values, so a missing recomputation cannot be treated as
agreement. The numeric comparison tolerance is 1e-12 absolute / 1e-9 relative,
not a replacement collision tolerance, reward coefficient or task threshold.

## Which Conclusions Remain Valid

| Preserved result | Post-fix interpretation |
| --- | --- |
| Original reference passes declared joint/self/speed checks | Reconfirmed for all 198 rows. The declared self-pair list is incomplete, so this is not complete physical feasibility. |
| Original first frame penetrates the table/plate | Reconfirmed, including native surface evidence. These findings were not caused by the NaN/quaternion guard defects. |
| Temporary v1/v2 candidates fail declared feasibility | Both failures remain failures. They were not silently repaired or replaced. |
| Temporary v3/v4 candidates pass declared feasibility | Both retain that limited result. Neither is a complete no-collision certificate or a formal reset. |
| Left palm/thumb intersection about 204.289 mm^3 in original/v3 | Reconfirmed with full closed native meshes. It is not explained away by the source-agreement text bug. |
| v4 palm/thumb Boolean intersections are empty | Reconfirmed at numerical precision. It applies to that configuration and pair, not all future actions or samples. |
| Original commands pull adjusted hands toward old poses | Reconfirmed in all four instantaneous cases. It is a force calculation, not an executed trajectory or a stability verdict. |
| Seventeen omitted shell pairs overlap in v4 | Reconfirmed as shell overlaps, not seventeen proven native collisions. Two palm/thumb examples remain clear in native Boolean tests. |
| Two index-root bodies lack same-body shells | This is a representation-inventory finding, not proof that neighboring shells leave a specific exposed surface. |

Do not collapse numerical input validity, declared-model feasibility, native
surface evidence, physical execution success and formal-pipeline readiness into
one "valid" flag. They answer different questions.

## Current Boundary

The first-frame posture adjustments remain temporary single-sample tests.
They are neither promoted into the formal pipeline nor presumed to generalize.
Collision-model/pair changes and further collision-coverage proposal work were
paused for this review. Original GT, models, candidate NPZs, old reports, scoring
and the exact renderer are untouched. No new pose optimization, MPC, physical
rollout, settling, RL training or video generation was performed.

The full regression suite passes 119 tests. Test success and numerical agreement
do not close the still-explicit collision coverage, initialization and missing
reward-parameter questions.

## Artifacts

- [33-group comparison and provenance report](/data_all/zzx/3.2RL/runs/taco_pour_bug_revalidation_v1/report.json)
- [Recomputed full collision report](/data_all/zzx/3.2RL/runs/taco_pour_bug_revalidation_v1/reference_collision_audit.json)
- [Recomputed full omitted-pair report](/data_all/zzx/3.2RL/runs/taco_pour_bug_revalidation_v1/reference_omitted_pairs.json)
- [Recomputed thumb scan](/data_all/zzx/3.2RL/runs/taco_pour_bug_revalidation_v1/thumb/report.json)
- [Revalidation runner](/data_all/zzx/3.2RL/scripts/revalidate_taco_pour_diagnostics.py)
- [Regression tests](/data_all/zzx/3.2RL/tests/test_pour_bug_revalidation.py)
