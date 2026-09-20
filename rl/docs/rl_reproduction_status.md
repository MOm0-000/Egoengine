# RL Reproduction Status

Input: synchronized left/right TACO 21-joint world-space GT, tool/target 6D
world-space pose trajectories, metric object meshes, robot XML/actuation,
30 Hz source frame IDs, and a single frozen world-to-simulation transform.
Raw human GT is input to MINK, not robot actions ready for PPO.
This is the user-approved GT-oracle validation setting, not a reproduction of
the vision-based pose reconstruction in Appendix A.3. The physical reset state
is still unresolved and unvalidated.

## Active Protocol (User Decision, 2026-09-06)

The latest user decision prioritizes **Pour/Bowl/Plate `20230927_017`** instead
of smear/brush. Its independent 198-frame input bundle has been acquired and
audited; see [Pour status](/data_all/zzx/3.2RL/docs/pour_sample_status.md).
The Pour scene and complete 198-frame MINK reference now exist. The declared
kinematic model passes, but native initial hand/table and hand/plate penetrations
prevent accepting the unchanged initial state. There is no validated reset or
RL result. See [Pour scene feasibility](/data_all/zzx/3.2RL/docs/pour_scene_feasibility.md).

The local reset-only protocol is now frozen and executed as
`taco_pour_initialization_protocol_v1`. It keeps object qpos bit-identical to
reference endpoint 0, uses zero qvel and candidate-hand-qpos control, binds the
formal 128/512 MJWP physics contract, and refuses release dynamics until the t0
gate passes. Candidate A passes the runtime shells, all 178 declared self pairs,
bilateral guards, joint limits and table checks, but is rejected by native CAD
evidence: left middle/pinky material enters the plate and the omitted left
palm/proximal-thumb pair interferes. Candidate B's collision-aware projection
from the original row 0 stops without a legal seed, so its held-object pre-roll
is not run. Consequently both passive release validations are correctly skipped
with zero physics steps, neither report sets `accepted_for_replay_rl=true`, and
`training_ready` remains false. See
`runs/taco_pour_initialization_protocol_v1/comparison.json`.
The subsequently authorized initial-hand diagnostic has produced a separate
declared-feasible candidate, with unchanged object coordinates and source GT.
The follow-up v4 diagnostic removes the demonstrated native palm/thumb
intersection by changing only one thumb joint relative to v3. Full collision
coverage is still unresolved. Instantaneous control comparisons reveal that
the old reference command pulls the raised hands back toward the obstacles.
No physical qvel/control was selected and no candidate was adopted as a reset.
See [the follow-up](/data_all/zzx/3.2RL/docs/pour_thumb_and_reset_inputs.md).
The latest collision-diagnostic review fixes invalid-state false passes,
unconditional source-agreement text, and stale native-mesh report reuse. It
also inventories two index-root bodies lacking same-body collision shells and
17 open native hand meshes. That milestone passed 107 tests; see
[coverage problems and proposed work](/data_all/zzx/3.2RL/docs/pour_collision_coverage_review.md).
The user then prioritized correcting bug-related misunderstandings. A full
historical Pour revalidation now compares 33 groups of recomputed results with
zero discrepancies; the NaN/quaternion/10 mm counterexamples were injected
faults, not observed bad Pour data. The full suite passes 119 tests. See
[corrected interpretation and evidence](/data_all/zzx/3.2RL/docs/pour_bug_impact_corrections.md).
No further collision-model proposal work was advanced during this review.
The original Pour uint16 depth has now been decoded with scale 4000 and checked
against all 198 camera poses. It places the observed support surface 16.50 mm
above the paper table after the current transform, but object and MANO surface
comparisons have opposite-signed residuals. This rules out a single global
depth/scene translation as a valid repair. No table, GT, reset or collision
model was changed; see
[the raw-depth audit](/data_all/zzx/3.2RL/docs/pour_raw_depth_table_audit.md).
The earlier 74/88-test milestones are historical. No collision model was
changed. The user explicitly limits all these posture adjustments to a
temporary single-sample test, not a formal pipeline or assumed generalizable
reset. They must not be automatically applied to subsequent samples.
The original four inputs and brush results below remain preserved artifacts.

The MANO-XHand geometry audit is now complete at the level of neutral frames,
FK, fixed point/frame offsets, morphology ratios, and six-row reachability
probes. A Pour-only per-finger target scale improved the local Pour probe but
made the existing Brush/Bowl probe worse, so the geometry contract remains
unresolved and the weight grid stays blocked. See
[the Pour morphology audit](/data_all/zzx/3.2RL/docs/pour_morphology_audit.md)
and [the cross-task counterexample](/data_all/zzx/3.2RL/docs/morphology_cross_task_audit.md).
The proposed next step is the read-only object-local contact mapper in
[the contact-geometry plan](/data_all/zzx/3.2RL/docs/contact_geometry_mapping_plan.md).

Use **Replay -> residual PPO**, not Replay -> MPC -> RL. MPC implementation,
sampling, tuning, and experiments are out of scope. This is an intentional
two-mode variant of Appendix C.1, not a reproduction of the paper's full
three-mode algorithm or its reported simulation cost.

Work order: collision-model audit and retargeting feasibility first; exact
Eq. (2), C.1-C.13 reward/evaluation implementation next; then H=20 control-step
chunks with current-plus-next-chunk lookahead. At every boundary restore the
same full simulator state for each candidate, try Replay before RL, validate
up to 40 remaining control steps, and commit only the first 20 (or the remaining
tail). A failed lookahead must not advance the committed simulator state.
The second chunk is revalidated from the actually committed state at the next
boundary. No guessed MPC substitute is allowed.

## Paper-First Plan Revision (2026-09-06)

The supplied PDF takes precedence over copied defaults and earlier local
proposals. Detailed evidence and next steps are recorded in
`docs/replay_rl_implementation.md`; page numbers there are 1-based PDF pages.

- Section 4.1 selects compatible TACO demonstrations. Audit the four approved
  inputs without claiming the authors' episode IDs or unpublished selection
  procedure, choosing by eventual success, or silently trimming source frames.
- A.1/A.3 specify a 0.6 m base offset and 0.72 m table with approximated TACO
  alignment. Check units and common transforms first. The existing first-frame
  bowl-bottom vertical anchor is local; depth-based realignment is on hold.
- Eq. (1) specifies joint limits and self collision, not collision shapes/pairs
  or environment-constrained initial-state projection. Audit actual pair/model
  defects without treating either all shell overlaps or the sparse copied pair
  list as a complete physical certificate. Wholesale geometry replacement is
  not the default next step.
- Section 3.2.2 explicitly expects reference replay to fail from embodiment and
  contact dynamics. Do not require successful Replay or collision-free geometry
  for every reference/GT frame before RL. This does not waive proven model
  defects, genuine self-collision issues, or invalid simulator initialization.
- The paper does not publish the TACO physical reset procedure. Initial-hand
  projection, object pose correction, and adopting a settled state as the task
  origin are on hold pending evidence and an explicit decision. A.3's "After
  initialization" concerns scene reconstruction, not such a reset algorithm.
- Retain C.1-C.3 rewards and two-chunk scheduling, with the approved MPC omission.
  Engineering/renderer gates and C.3 success remain different reports. Section 6
  acknowledges contact-model and deformable-object limitations; it does not
  supply a repair method for the observed rigid GT/mesh overlap.

This revision changes documentation and the non-runnable protocol specification
only. No model, GT, initial state, runtime constraint, renderer gate, reward code,
checkpoint, or prior measured result is changed. No new physical validation or
training is claimed.

## Paper-First Audits Executed (2026-09-06)

See [the execution report](/data_all/zzx/3.2RL/docs/paper_first_audit_results.md).
Four-input, shared-transform, compiled/native scale, named collision-pair, and
initial-state audits have now been executed without modifying the preserved
scene, GT or reference. The reports are in `runs/paper_first_audit_v1/`.

All four inputs have structurally usable GT. Cut has 361 GT rows but 358 RGB
and 356 depth frames. Skim's released camera puts all hand GT behind the camera;
it was not automatically inverted or replaced. Neither finding was hidden by
trimming or discarding a sample. Original depth container timing (15 fps) must
not be mistaken for the 30 Hz GT timeline.

Brush's applied common transforms and native/compiled metric mesh bounds agree.
MINK/runtime hand pairs match and source hand-shell geometry is unchanged, but
25 of 82 omitted nonadjacent shell pairs penetrate in the reference. The other
20 omitted pairs are assembly-adjacent candidates, not automatic exclusions.
Native mesh samples and open-surface limitations are recorded individually.

Initial native palm penetration remains 7.186 mm right and 8.665 mm left, so
coarse-shell replacement alone cannot solve the reset. Translation-only table
clearance lower bounds are 11.686/13.399 mm for right/left hand and 2.273 mm for
the brush. No shifts, projections, settling, qvel changes or gate changes were
applied. These measurements do not define an approved or validated reset.

At the time of this historical audit, 48 local tests passed and real PPO had not
yet been integrated. The current adapter and smoke results are recorded below;
the physical reset and task validation remain unresolved. The exact reward and
mock-backend two-chunk tests are not RL results.

## Brush Scene Baseline (Historical)

- Initially approved scene: brush/brush/bowl, episode 20230927_027. Preserved after the later Pour acquisition.
- Isolated scene: `models/taco_xhand/xhand/bimanual/taco_brush_brush_bowl_20230927_027/scene_act.xml`.
- Two floating XHands: 18 coordinates each (6 wrist + 12 finger), 36 actuators total.
- Tool 071 and target 146 are passive free bodies: nq=50, nv=48, nu=36.
- Robot template snapshot is local under `models/taco_xhand/templates/`.
- Bowl visual uses the released centimeter mesh with exactly one 0.01 scale.
- Bowl collision uses 32 metric CoACD parts, not a solid convex bowl hull.
  CoACD reported residual concavity 0.049087 > requested 0.02 because of the
  32-part limit. This approximation still needs contact-geometry validation.
- Hand/object/floor contacts have sliding friction (condim=3). Both hands,
  both objects and inter-object collisions are enabled. Nothing actuates objects.
- Base placement follows Appendix A.1: pair center at x=0.6, y=0 and table z=0.72.
  The +X convention is inherited from the copied adapter; the paper does not
  publish the complete axis convention. One transform is shared by all frames.

## Original GT Candidate (Failed, Preserved)

Artifacts: `runs/taco_brush_bimanual_gt_v1/` contains `human_reference.npz`,
`input_audit.json`, `robot_reference.npz`, and `retarget_report.json`.

- All 209 RGB/hand/object rows are preserved; no 205-row unilateral artifact is
  used. The PKL root translations independently verify left/right joint order.
- MINK runs from the local pinned clone. Both hands use actual respective GT,
  with all 12 object tangent DOFs locked during IK only.
- Wrist and distal frames are landmark-derived, calibrated to the robot's
  native geometry. These are disclosed implementation conventions, not the
  authors' unpublished wrist offsets. No hand is synthesized or mirrored.
- Candidate fingertip mean error: right 0.04491 m, left 0.02980 m.
- Minimum self clearance: -0.008812 m; all 209 frames contain violations.
  Typical pairs are palm/thumb proximal, palm/index proximal, and adjacent
  finger capsules. **This is a failed kinematic feasibility candidate.**
- `strict_gate_passed=false`, `rl_validation_completed=false` remain explicit.

## Latest Collision/Retargeting Work (2026-09-06)

Latest scene: `models/taco_xhand/xhand/bimanual/taco_brush_brush_bowl_20230927_027/scene_source_contacts_mass.xml`.
Latest candidate: `runs/taco_brush_bimanual_gt_v4/robot_reference.npz`, with
`retarget_report.json` and `collision_audit.json` in the same directory.
The old scene and v1 results remain unchanged. v2 stopped at frame index 22;
v3 tightened QP numerical tolerances and completed; v4 additionally uses the
corrected object inertials and independently reports joint/velocity bounds.

- Native source XML has only 15 intra-hand self pairs per hand, primarily
  palm/distal and distal/distal. Restored those exact 30 pairs and added all
  144 inter-hand pairs, including pairs missing from the original template.
- IK and runtime share all 174 explicit hand pairs. All hand/object,
  inter-object, and floor pairs remain enabled (1,967 physical pairs total).
  Hand shell geometry was not shrunk. This is a **limited source collision
  model**, not a complete CAD self-collision certificate.
- MINK now applies bounded separating QPs, checks runtime/planning pair equality,
  preserves the aggregate frame velocity envelope, and exports failures as
  failed prefixes rather than usable robot references. DAQP primal/dual
  tolerances are 1e-9; self-distance acceptance tolerance is 1e-6 m.
- All 209 rows completed. Declared-model self distance >= -9.807e-7 m; zero
  joint-limit or frame-velocity violations; maximum speed/limit ratio 0.99999625.
  Right/left mean fingertip errors are 0.03220/0.03224 m. These are fitting
  errors, not published paper pass thresholds.
- Independent broad-shell audit still reports 209 penetrating frames and
  minimum -0.01271 m. Restoring source pairs is not evidence that these omitted
  geometric overlaps vanished. Some are shell artifacts, while others require
  a fuller hand-model audit.
- Full-reference maximum hand/tool, hand/target, and hand/floor penetrations
  are 0.02137, 0.01579, and 0.03649 m. First-frame hand/floor penetration is
  0.01340 m, so this candidate is not ready for unmodified physics initialization.
- Tool/target collision pieces penetrate by up to 0.007487 m. Independent
  2,048-vertex-per-object signed-distance samples on the released watertight
  meshes also find real GT/mesh overlap, e.g. up to 0.004029 m at frame 78.
  This is sampled evidence, not a global surface intersection bound. Details:
  `runs/taco_brush_bimanual_gt_v3/object_surface_audit.json`; human GT is
  byte-identical between v3 and v4.
- Native mesh/floor audit finds initial brush clearance -0.001311 m and full
  trajectory minimum -0.002640 m. The old shared transform anchored only the
  bowl's first-frame bottom. This is a local vertical alignment convention;
  audit it against A.1/A.3 before proposing support-plane or initial-state
  changes. GT poses and floor height were not silently moved.
- Fixed double-counted object mass from overlapping convex parts. Closed native
  mesh integration gives brush 0.179365 kg and bowl 0.113284 kg at the inherited
  nominal density 1000 kg/m^3, versus 0.283784/0.195252 kg previously. This density
  is not a measured TACO mass or an author-published value.

`kinematic_model_feasible=true` refers only to the declared joint/self model.
`complete_intrahand_geometry_coverage=false`, `strict_gate_passed=false`, and
`rl_validation_completed=false` must remain visible.

## Intrahand Audit In Every Retarget

Every completed MINK retarget now writes `intrahand_collision_audit.json` beside
the robot reference. `audit_intrahand_trajectory()` enumerates all same-hand
shell pairs and labels each as `declared`, `omitted_assembly_adjacent`, or
`omitted_nonadjacent`; it records each pair's minimum distance, worst frame, and
penetrating-frame count, plus per-class summaries. The audit is diagnostic only:
it does not add omitted pairs to the runtime model, alter qpos, or claim a
physics-valid reset. The standard TACO adapter, generic MINK adapter, and
Spider candidate all use this same check.

## Reward And Scheduler Modules

`src/egoengine_repro/action/paper_rewards.py` now implements Eq. (2), C.1-C.13:
SO(3) geodesic tracking, C-e reward, strict e>C termination, mimic, total-action
smoothness, per-hand/per-object opposition-contact formula, signed lifting,
TACO auxiliary selection, and auxiliary-free evaluation metrics. Failed steps
are excluded from the valid-prefix evaluation sum; no-success Cost is undefined.
The MJWP adapter now supplies physical MuJoCo/Warp contact flags and forces for
every hand-object-finger combination. TACO keeps mimic and smoothness disabled;
the configured contact coefficient is a disclosed local value because the paper
does not publish one.

`src/egoengine_repro/action/replay_rl.py` implements 20/40 control-step scheduling,
Replay before RL at every boundary, current-chunk-only commit, short tails,
and rollback after lookahead/training failures. The backend contract includes
RNG, previous action, warmstart and reference cursor; simulation cost must not
roll back. Tests use a deterministic mock backend. The real MJWP adapter now
captures/restores the MuJoCo/Warp state, reference cursor, reset prior, action,
contact/EFC workspaces and RNG. GPU contact solving is numerically non-bitwise
deterministic after rollback; this remains a tolerance to record, not a claim of
exact replay.

Active protocol: `configs/replay_rl_protocol.yaml`. Unpublished tracking/contact
coefficients remain null instead of importing Aria values under a TACO label.
This is not a runnable training configuration. See `docs/replay_rl_implementation.md`.

## Verified Versus Unverified

Previously verified: MuJoCo compilation, GT frame/coordinate invariants, passive-object
actuation, metric meshes, quaternion-manifold qvel, exact renderer mapping of both
hands and both objects, the real MJWP/PPO interface, and bimanual physical-contact
extraction. The current regression suite passes 479 tests and 57 subtests (with one skipped),
including reward equations, mock-backend lookahead/rollback, real-backend reset,
snapshot and endpoint checks, failed-QP artifact handling, and renderer mapping.
The previously measured renderer wrist position residual is 0; rotation
residual <1e-15 rad. No new video was rendered in this milestone.

Not verified: physical feasibility, contact acquisition/hold, or scientific PPO
improvement. The PPO smoke only proves that the official trainer can consume the
real bimanual MJWP adapter; it is not a trained task result.
No passing trace or passing video was fabricated. Rendering still uses only
the copied `diagnostics/render_exact_deximit_triptych.py` and its strict gate.

Historical bimanual smoke checkpoints used provisional data and earlier scene
variants; their tiny training loops proved interface connectivity only. They
remain preserved as non-scientific artifacts. The current smoke output uses the
real Pour GT/MINK reference and the current bimanual scene. The legacy reference
generator requires an explicit fixture flag.

## Remaining Work, In Order

1. Complete contact-geometry validation of Pour's independently generated 022/135
   collision parts. Scene compilation, common-frame/metric checks, passive
   actuation and full 198-frame MINK retargeting are done. Both 32-part CoACD
   approximations exceeded their requested concavity threshold; no geometry
   tuning or task-success claim was made.
2. Follow up the executed Eq. (1) joint/self and named-pair audit. Classify
   assembly exclusions, shell artifacts, genuine collisions, and copy/scale/pair
   defects. The limited source-model candidate remains an audited comparison;
   do not ignore unresolved pairs or automatically replace all collision shapes.
3. Resolve the unpublished reset specification using Pour's measured initial
   penetrations: native palms are 11.10/7.75 mm below the table and native left
   fingers enter the plate in deterministic surface samples. The independent
   hand-only diagnostic now clears the declared environment constraints and
   native hand/table tests, but remains unadopted because native hand-internal
   coverage, initial velocity and full physical validation remain unresolved.
   If no paper-supported explanation is
   found, present measured minimal alternatives for confirmation before changing
   the reset. Later execution/contact failures go to Replay -> RL unless a model
   defect or simulator breakdown is established; a fully successful Replay is
   not required. GT/mesh disagreement remains a separate diagnostic. Nominal
   density remains an assumption, not a published or measured value.
4. Run separate tool-only and approved tool+target evaluations using C.9-C.13,
   then strict-gated rendering. Full PPO/task validation has not occurred.
   Adapter timing, endpoint indexing, bimanual sites/contact extraction, paper
   rewards, zero-randomization TACO configuration, full snapshots, and PPO smoke
   integration are complete and covered by regression tests.
8. Other samples are deferred while Pour is the current priority. Preserve cut's
   known frame mismatch and skim's camera issue; do not silently repair or crop.

Paper omissions must stay visible: exact lambda_p/lambda_R, most TACO position
boundaries, the conversion from component boundaries to C, contact coefficient,
and some MINK calibration details are not numerically specified. Copied defaults
are implementation settings, not newly established paper values. In particular,
0.08 m / 2.5 rad in a copied TACO YAML originate from the Aria paragraph and
must not be presented as TACO paper thresholds.
