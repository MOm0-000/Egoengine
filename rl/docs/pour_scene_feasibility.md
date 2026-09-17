# Pour Scene And Retargeting Feasibility

Input: all 198 synchronized source rows of both hands' TACO GT, bowl `022`
and plate/tray `135` world-pose GT and released centimeter meshes, camera
transforms and 30 Hz RGB for `(pour in some, bowl, plate)/20230927_017`.
The output is a robot motion reference and diagnostic evidence, not a physically
validated demonstration or a PPO result. The user-approved GT-oracle setting
and Replay -> RL protocol remain unchanged.

## Completed

After fixing diagnostic guards, all 198 reference rows' recorded kinematic and
collision metrics were recomputed without discrepancies. The prior bug
counterexamples were test injections, not detected corruption in this baseline.
See [post-fix impact corrections](/data_all/zzx/3.2RL/docs/pour_bug_impact_corrections.md).

- Built an independent two-XHand/two-passive-object scene. State dimensions are
  nq=50, nv=48, nu=36. Only the 36 hand coordinates are actuated; objects have
  free joints, not actuators. Object GT is imposed during reference construction
  only, not through an executable physical rollout.
- Found that the older robot mesh directory was still an external-project link.
  Copied all 26 referenced robot mesh files into a new local snapshot with
  SHA-256 provenance. Every mesh used by the Pour XML now resolves inside this
  project. Existing links, brush assets and earlier outputs were preserved.
- Generated independent collision approximations for both new objects, using
  the unchanged local CoACD settings. Each object has 32 metric parts. The
  original visual meshes remain byte-identical and get exactly one 0.01 scale.
- Integrated each object's inertia from its closed native mesh rather than
  summing overlapping hull volumes. At the inherited density 1000 kg/m^3,
  bowl mass is 0.235121 kg and plate mass is 0.253393 kg. These are nominal
  engineering values, not measured TACO masses or published author settings.
- Ran MINK on all 198 frames. No trimming, synthetic hand, per-frame world
  shift, GT correction or environment-constrained initial projection was used.
- Independently audited declared and omitted collision pairs, all reference
  frames, native mesh/table clearance, initial native hand/plate containment,
  common transforms, scale, reference object poses and finite-difference qvel.
- Verified the approved exact renderer's two-hand/two-object mapping without
  generating a robot video, modifying its source or bypassing its gate.

## Paper Versus Implementation

Section 3.2.1, Eq. (1), PDF pages 3-4 supplies fingertip pose and wrist-orientation
tracking with joint/self constraints. Appendix A.1/A.3, pages 15-16 supplies
the 0.6 m base offset, 0.72 m table and approximate TACO alignment. The copied
landmark-to-robot orientation offsets, first-target-bottom vertical anchor,
collision shapes/pairs, friction, density, IK solver settings and reference
control rate are inherited local implementations, not recovered author values.

This milestone uses the existing effective MINK settings: fingertip position
cost 10, fingertip orientation cost 1, wrist orientation cost 3, zero wrist
position/posture cost, DAQP tolerances 1e-9, eight subiterations per 30 Hz source
frame and 80 on the first frame. The aggregate velocity envelope is unchanged.
Only the retarget subsection of the copied YAML is read; its old Aria-derived
reward numbers are not imported into the active TACO protocol.

Section 3.2.2 expects reference replay/contact failures, so later hand/object
reference overlaps are not new paper success criteria or an automatic reason
to demand successful Replay before RL. However, genuine initial intersections
and unresolved model defects still need an explicit treatment. Neither Eq. (1)
nor A.3 publishes a TACO reset, table-constrained initial projection or settling
recipe. No new such procedure was applied here.

## Retargeting And Coverage

The 174 declared hand pairs comprise 30 source intra-hand pairs plus all 144
inter-hand pairs. MINK and runtime enumerate exactly the same set. All
hand/object, inter-object and floor pairs are enabled, for 2,822 physical pairs.

| Check | Result |
| --- | --- |
| Frames retained | 198 / 198 |
| Joint-limit violating frames | 0 |
| Frame-velocity violating intervals | 0 |
| Minimum declared self distance | -0.0009996 mm, within existing 0.001 mm numerical tolerance |
| Right / left mean fingertip error | 14.924 / 19.411 mm |
| Right / left maximum fingertip error | 84.583 / 73.833 mm |
| Omitted intra-hand shell pairs | 102: 20 assembly-adjacent candidates, 82 nonadjacent |
| Omitted nonadjacent pairs with shell penetration | 23, still requiring classification |
| Minimum broader nonadjacent-shell distance | -15.907 mm |

Thus `kinematic_model_feasible=true` applies only to the declared model.
`complete_intrahand_geometry_coverage=false` remains explicit. Native sampling
on watertight counterparts supplies positive overlap evidence for some omitted
pairs; open finger meshes and sparse samples cannot certify nonintersection.
No omitted pair was automatically enabled, excused or fixed by shrinking shells.

Both object decompositions hit the inherited 32-part cap. CoACD's console
reported maximum concavity 0.0754895054 for the bowl and 0.0283087329 for the
plate, above requested 0.02. These are CoACD diagnostics, not meter-valued
penetration bounds. The approximation is not certified at the requested
threshold; parameters were not tuned after looking at task success.

## Initial State Findings

These are measurements at robot reference row 0, not an accepted reset.

Subsequent [input-logic investigation](/data_all/zzx/3.2RL/docs/pour_input_logic_findings.md)
identified a real orientation-target construction defect and showed that the
initial palm-height drop is introduced by IK wrist displacement. The table
anchor is still uncalibrated independently. The numbers below describe the
preserved local reference/scene, not an unavoidable morphology error or a
proof of correctly constructed upstream targets.

| Initial check | Result |
| --- | --- |
| Right / left hand shell penetration into table | 14.961 / 13.287 mm |
| Right / left native palm penetration into table | 11.099 / 7.746 mm |
| Maximum hand/plate shell penetration | 14.009 mm |
| Initial hand/bowl shell distance | +24.778 mm |
| Initial bowl native table clearance | +1.488 mm |
| Initial plate native table clearance | approximately 0 |
| Native-vs-compiled world-bounds discrepancy | below 4.4e-9 m |

Initial native hand/plate tests used 256 deterministic vertex samples per
surface on the three unique body pairs identified by shell penetration.
Left ring distal, pinky proximal and pinky distal surfaces yielded respectively
13, 5 and 34 points inside the watertight plate, with maximum sampled depths
1.457, 0.891 and 1.509 mm. The reverse direction was not signed-tested against
open finger meshes. These are positive intersection evidence, not global
penetration bounds. They rule out attributing every initial overlap solely to
coarse collision shells.

One fixed transform preserves relative hand/object and camera/object geometry
to numerical precision; exported object poses agree exactly with the aligned
GT. Native/compiled scaling is consistent. These checks do not independently
validate the table calibration or prove temporal correspondence.

Vertical translation lower bounds of 14.961/13.287 mm for the two hands were
measured only. A common upward shift would lift the initially supported plate,
and independent shifts alter hand/object relations. Neither is an approved
reset, a solution to all contacts or a calibration correction. Reference qvel
is the existing manifold finite difference; choosing it versus zero as the
physical initial velocity remains unresolved.

Over the full reference, shell penetrations reach 26.745 mm hand/bowl,
22.407 mm hand/plate and 48.328 mm hand/table. The plate's native GT reaches
1.230 mm below the fixed table. The object/object shell audit found no negative
distance, and 2,048-vertex samples in each direction at endpoints and selected
diagnostic frames found no object/object containment. Sampling is not a complete
surface-intersection certificate. These later-reference findings are separate
from the invalid/unvalidated initial state and from Appendix C.3 scoring.

## Boundary And Next Decision

No physics integration, Replay rollout, PPO training, reset correction or
robot rendering was performed. Training readiness and strict-renderer success
remain false. The published Pour example thresholds (0.12 m, 1.5 rad) are
unchanged; lambda_p/lambda_R/C and contact coefficients remain unspecified.

The user subsequently authorized the first-frame hand-only diagnostic, which
has now been run separately from this preserved baseline. See
[the diagnostic report](/data_all/zzx/3.2RL/docs/pour_initial_hand_diagnostic.md).
It found a declared-model-feasible candidate without changing objects, table,
GT or scoring. This is not the paper's published reset and has not been adopted
as one. Unclassified hand geometry, approximate object collisions and the
physical initial velocity still need resolution before physical validation.
The subsequent v4 thumb/control audit removes the demonstrated native thumb
overlap in a separate posture, but does not fix runtime coverage or adopt any
initial state. See [the follow-up](/data_all/zzx/3.2RL/docs/pour_thumb_and_reset_inputs.md).

This scene milestone passed 64 tests. This includes the old sample
tests and 11 new scene/isolation/kinematic/native-contact checks. It does not
constitute a physical task success or completion of Section 3.2/Appendix C.

## Artifacts

- [Scene and build provenance](/data_all/zzx/3.2RL/models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml)
- [MINK report](/data_all/zzx/3.2RL/runs/taco_pour_bimanual_gt_v1/retarget_report.json)
- [Robot reference](/data_all/zzx/3.2RL/runs/taco_pour_bimanual_gt_v1/robot_reference.npz)
- [Trajectory collision audit](/data_all/zzx/3.2RL/runs/taco_pour_bimanual_gt_v1/collision_audit.json)
- [Initialization and omitted-pair audit](/data_all/zzx/3.2RL/runs/taco_pour_bimanual_gt_v1/initialization_audit.json)
- [Initial native hand/plate audit](/data_all/zzx/3.2RL/runs/taco_pour_bimanual_gt_v1/initial_native_contact_audit.json)
- [Native object surface audit](/data_all/zzx/3.2RL/runs/taco_pour_bimanual_gt_v1/object_surface_audit.json)
- [Scene regression tests](/data_all/zzx/3.2RL/tests/test_taco_pour_scene.py)

Commands use `/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python` with
`PYTHONPATH=src:external/mink/src`. Build with
`scripts/build_taco_bimanual_scene.py --data-root data/taco_v1/pour_bowl_plate
--tool-id 022 --target-id 135 --model-name taco_pour_bowl_plate_20230927_017
--robot-snapshot-name xhand_pour_20230927_017 --output <scene>` after running
`scripts/prepare_taco_bowl_collision.py --source <each_cm_mesh> --output <parts>`
once per object. Retarget with `scripts/retarget_taco_bimanual_gt.py --data-root
data/taco_v1/pour_bowl_plate --sequence '(pour in some, bowl, plate)/20230927_017'
--episode taco_pour_bowl_plate_20230927_017 --scene <scene> --output <new_run>`.
Existing outputs and snapshot directories are never overwritten; new runs need
new output paths. The linked JSON reports retain the exact source paths/hashes.
