# Replay -> RL Implementation Contract

Input to RL: a timestamped MINK robot reference (`qpos`, manifold `qvel`,
36 hand-only reference commands), synchronized tool/target world-pose GT,
the isolated two-hand/two-passive-object MuJoCo scene, and an explicitly
specified initial simulator state. That initial state is not yet validated;
its reset rule is an unresolved reproduction detail, not a paper-provided recipe.
Human hand GT is input to retargeting, not directly to PPO.
RGB/depth are used for input/phase/support audits, not actor observations.
Using TACO GT is the user-approved oracle-input validation setting. It does not
reproduce the vision-based pose reconstruction described in Appendix A.3.

## Scope And Status

### Current checkpoint: real switching is connected and tested; Pour reset still blocked

The active Pour input remains the corrected full 198-frame MANO/MINK reference
in `runs/taco_pour_bimanual_mano_fk_v1`. Collision-shape experiments, including
the refined candidate and its actual 4-/16-world GPU capacity checks, are in
`runs/taco_pour_collision_repair`. They have not replaced the formal model or
selected a physical reset. The current geometry classification is recorded in
[the collision review](/data_all/zzx/3.2RL/docs/pour_collision_coverage_review.md).

The former standalone components now have an explicit connection:

- `action/replay_rl.py::solve_chunk` retains the two-chunk scheduling rules.
- `video_to_spider/rl/replay_rl.py` connects actual MJWP transitions and the
  official H2S2R PPO trainer to that scheduler. Every rollout reset during chunk
  training restores the incoming physical boundary, not an ideal reference pose.
- `scripts/run_taco_replay_rl.py` is the guarded entry point. It refuses old
  qpos-only diagnostics and requires a hash-matched, accepted initialization
  report containing qpos, qvel, ctrl and source index 0. No Pour report currently
  passes that gate. Full source length is 198 endpoints / 197 transitions.
- `scripts/run_mjwp_ppo.py` runs standalone PPO smoke training. It does not call
  `solve_chunk`; its trainer helpers are reused by the new connection.

Actual GPU tests now cover: a deliberately rejected Replay transition followed
by real PPO optimization and validation from the saved boundary; and a static
in-memory control fixture that checks 40 real transitions but commits exactly
the saved state after transition 20. Neither test is a Pour success experiment.
The first uses an injected Replay rejection to guarantee fallback coverage;
the second keeps the hands away from resting objects. Files/GT are unchanged.
Both task tracking variants remain explicit runner options.

This initial correctness path uses one GPU world, no noise, deterministic mean
actions for validation, and a fresh PPO policy per failed chunk. These are local
settings, not recovered author parameters. Multi-world boundary cloning is
intentionally rejected: packed contact buffers cannot be partially copied as
ordinary world-indexed arrays. Total work is not rolled back. The reset also
preserves the episode's original lifting-height origin.

Two concrete contact-state bugs were fixed while connecting the loop: snapshots
now restore packed contact/broadphase counts, and contact rewards ignore stale
buffer entries outside the live contact count. Capacity is checked after every
physics substep. Validation disables autoreset, so neither failure nor timeout
can substitute a freshly reset state for the actual terminal state.

Successful Replay, a pre-existing successful grasp, and eventual task success
are not prerequisites for entering residual RL (paper Section 3.2.2 and C.1).
The required entry checks are a consistent collision/retarget model, a declared
and reproducible legal initial state, explicit objective parameters, and the
real switching/state-restore connection. Collision approximation need not be
mathematically zero-error; unexplained false task contacts must not be silently
treated as true ones. Published Pour thresholds do not uniquely determine the
current adapter's threshold-to-C formula, which remains a local configuration.

### Historical implementation checkpoints (not the current readiness state)

Current priority is Pour/Bowl/Plate `20230927_017`, following the user's later
request to acquire and prioritize that task. Its input bundle is available at
`data/taco_v1/pour_bowl_plate/`. Its isolated scene and full 198-frame MINK
reference now exist; the declared kinematic model passes, but the initial
native palms penetrate the table and left fingers penetrate the plate.
See [Pour scene audit](/data_all/zzx/3.2RL/docs/pour_scene_feasibility.md).
The later user-authorized hand-only initial diagnostic produced a separate
declared-model-feasible candidate. It did not replace the reference, select
initial qvel or become an accepted reset. Native hand/object samples are clear,
but omitted hand-internal geometry remains unresolved; see
[initial-hand diagnostic](/data_all/zzx/3.2RL/docs/pour_initial_hand_diagnostic.md).
The v4 follow-up removes the demonstrated native palm/thumb overlap in a
separate posture, without fixing runtime pair coverage. Instantaneous input
comparisons show why qpos, qvel and first-command timing must be specified
together; no combination has been adopted. See
[thumb/control audit](/data_all/zzx/3.2RL/docs/pour_thumb_and_reset_inputs.md).
User clarification: these posture adjustments are temporary single-sample
tests, not part of the formal Replay -> RL pipeline. No generalization to later
samples is established and no automatic reuse is authorized. Formal
initialization needs a separate design and validation decision; these
diagnostic candidates may be discarded.
The later collision-diagnostic bug review and native-body inventory are recorded
in [coverage findings and proposed work](/data_all/zzx/3.2RL/docs/pour_collision_coverage_review.md).
The proposal is not an applied geometry/pair change or formal reset design.
The user subsequently requested historical bug-impact verification first.
[Full revalidation](/data_all/zzx/3.2RL/docs/pour_bug_impact_corrections.md) finds no
changes to 33 groups of recomputed Pour metrics. The fault-injection examples
are not evidence of actual historical data corruption. Further collision
proposal work was paused; no reset or physical integration was introduced.
The brush results below and in earlier reports are historical, not Pour reset
measurements. See [Pour status](/data_all/zzx/3.2RL/docs/pour_sample_status.md).

The latest MANO-XHand morphology audit confirms that neutral palm and finger
ratios are measurable, but a Pour-improving per-finger target scale worsens the
complete Brush/Bowl probe. It is therefore diagnostic only; no scaled target,
geometry contract, or weight-grid result has entered Replay or RL. See
[the morphology report](/data_all/zzx/3.2RL/docs/morphology_cross_task_audit.md).

The test environment is now complete as well: MINK's `robot_descriptions`
development dependency is installed, the local MINK checkout is selected, and
its MuJoCo 3.7 inertia-matrix call is compatible with the pinned Spider runtime.
The root command passes 481 tests and 57 subtests; details are in
[the environment note](/data_all/zzx/3.2RL/docs/test_environment.md).

MPC is deliberately omitted at the user's request. No replacement action
sampler or MPC-like solver will be added. The paper reward/evaluation equations,
two-mode chunk scheduler, real MJWP state snapshots, bimanual contact features,
and residual PPO smoke path were originally separately tested components. The
current real switching tests above supersede that integration gap, but do not
prove a trained Pour task result; do not
compare it to the paper's three-mode SR/Cost
tables.

The paper-first input/model/reset audits have since been executed; see
[the execution report](/data_all/zzx/3.2RL/docs/paper_first_audit_results.md).
Their results do not validate the current initialization or complete collision
coverage, and do not constitute actual PPO integration or task success.

## Paper-First Evidence And Decisions

Sources below refer to the supplied PDF; page numbers are 1-based PDF pages.
An implementation needed to realize a published constraint is not itself an
author-published procedure. Copied SPIDER/OakInk geometry and MINK/H2S2R defaults
are not evidence of EgoEngine's collision model or numerical settings.

| Source | Published evidence | Consequence and boundary |
| --- | --- | --- |
| Section 4.1, p.5 | 16 selected TACO demonstration pairs compatible with the robot embodiment and digital-twin pipeline. | Audit compatibility of approved inputs; exact episode IDs and selection tests are not published. Do not claim these are the authors' episodes. |
| Section 3.2.1, pp.3-4, Eq. (1) | MINK aligns fingertip positions/orientations and wrist orientation under joint limits and self-collision constraints. | Audit those constraints first. Collision shapes/pairs, environment-constrained IK, and initial-state projection are not specified. |
| Appendix A.1, p.15 | Tool/target center with a fixed 0.6 m base offset; table height 0.72 m. | Check existing scale and common-frame alignment. The axis convention and complete vertical alignment recipe are not published. |
| Appendix A.3, p.16 | TACO reconstruction uses a fixed-offset heuristic and approximated alignment. | "After initialization" describes digital-twin reconstruction, not a physical reset, settling, or depenetration procedure. |
| Appendix A, p.14; Fig. A.4, p.17; C.2, p.21 | Bimanual simulation with XHands; floating Cartesian wrist/base action abstraction. | Preserve the approved bimanual architecture. Exact XML mappings and two-object score reduction are local implementation choices. |
| Section 3.2.2, p.4; Section 4.3, pp.6-7 | Reference replay often fails from embodiment and contact-dynamics discrepancies; residual PPO refines execution. | Successful Replay or collision-free hand/object geometry at every GT/reference frame is not an RL entry requirement. This does not waive simulator defects or unresolved reset validity. |
| Appendix C.1, p.20 | H=20, current-plus-next-chunk validation, current-chunk-only execution, cheaper modes reconsidered at each boundary. | Preserve this schedule; removing MPC is a user-approved deviation. |
| Appendix C.2-C.3, pp.21-23 | Tracking and auxiliary reward equations; no TACO randomization/mimic/smoothness; object-only evaluation. | Keep paper scoring separate from engineering and renderer gates; do not loosen C to hide model mismatch. |
| Section 6, p.8 | Contact modeling and deformable objects remain limitations. | The paper supplies no cure for rigid GT/mesh overlap or a compliant brush-bristle model. |

The original four approved inputs remain preserved: brush `20230927_027`, cut
`20230917_020`, skim `20230926_004`, and smear `20231103_071`. The user subsequently
authorized Pour/Bowl/Plate acquisition and priority; `20230927_017` is the first
metadata-complete, calibration-good candidate in sequence-ID order. The previous
four-only acquisition restriction is superseded only for this task.
Compatibility screening must
precede evaluation, preserve all source frames, and record any ineligibility
separately from rollout failure. It is a disclosed local audit, not a recovered
author selection algorithm. Do not select inputs by eventual task success,
silently trim them, or add unrelated episodes without authorization.

## Exact Mathematical Contract

For one evaluated object, C.1-C.4 and C.9 are:

```text
ep = ||p_sim - p_ref||_2
eR = acos(clamp((trace(R_ref^T R_sim) - 1) / 2, -1, 1))
e = sqrt(lambda_p * ep^2 + lambda_R * eR^2)
r_obj = C - e
terminate iff e > C (not e >= C)
```

The implementation returns raw C-e and the terminal mask separately. Only
valid steps contribute to C.3 evaluation. Clamping rotation roundoff is not a
change to the SO(3) metric; replacing the reward with exp(-e) would be a change.

```text
C.5: r_human = -(beta_x ||x-x_ref||^2
                + beta_R d_SO3(R,R_ref)^2 + beta_q ||q_fingers-q_ref||^2)
C.6: r_smooth = -||a_t-a_(t-1)||^2
C.7: r_contact = c_contact * I[thumb_contact AND any_other_finger_contact]
C.8: r_lift = lambda_z * (z_object-z_initial)
```

Use total executed commands for C.6, not residual commands alone. C.8 is
signed, not max(0, dz). Contact flags must come from the physical backend;
fingertip proximity or human GT contact labels cannot replace them. A thumb
on one object and an index on another object do not earn opposition contact;
neither do fingers belonging to different hands. Helpers retain the hand and
object axes without inventing a two-object reduction rule.

For TACO: no domain randomization, no mimic reward, no smoothness reward.
Lifting is enabled only for a relevant task. Evaluation uses only object
tracking, never contact/mimic/smoothness/lifting bonuses.

```text
tau_i = number of valid steps before first boundary violation or completion
SR = mean(I[tau_i == T_i])
Step = mean(tau_i / T_i)
Reward = mean(sum_valid(C_i - e_it) / (C_i * T_i))
Cost = sum_success(M_i) / sum_success(T_i)
```

Cost is undefined if there are no successful trajectories. Count all actual
optimization/validation simulator work, including failed attempts. Record both
physics substeps and control steps to avoid silently conflating them.

## Two-Chunk Contract

`solve_chunk` uses H=20 control intervals, not 20 video rows or physics steps.
The currently copied timing is 30 Hz control/reference with ten physics
substeps per interval; this is an implementation setting, not a published
EgoEngine control rate. The real adapter still needs endpoint indexing tests.

1. Snapshot the committed simulator state at the chunk boundary.
2. Execute Replay over min(40, remaining) control intervals and score every
   endpoint against the matching reference object pose.
3. If either chunk fails, restore the identical boundary state, train residual
   PPO over that window, restore again, and validate the resulting policy.
4. Only if the entire window passes, commit the exact state captured after
   min(20, remaining) intervals. Never commit the lookahead's second chunk.
5. At the next boundary begin with Replay again. Final windows are truncated,
   not padded with repeated final-reference rows. Both modes failing leaves
   the committed state unchanged; exceptions also roll back.

The adapter must include MuJoCo/Warp integration state, control/actuator state,
warmstart, reference cursor, previous command and RNG in its snapshot. The PPO
policy used for validation must reset its recurrent inference state to the
same boundary context. A policy callable returned by the training callback is
responsible for this; the scheduler cannot serialize an opaque policy's RNN.
Snapshot/restore must not reset the cumulative simulation-work counter.

## Before Training

1. Prioritize the acquired Pour input against Section 4.1's compatibility premise: synchronized
   frames, metric assets, hand/object coordinate consistency, and whether the
   demonstrated interaction is representable by the chosen rigid XHand scene.
   Manual phase labels are diagnostic input, not new reward or success terms.
2. Check the existing scene against A.1/A.3: scale applied once, common transform
   for hands/objects/camera, preserved relative poses, base offset, and support
   placement. The current first-frame target-bottom table anchor is a local
   convention, not the authors' vertical alignment procedure. Depth can provide
   diagnostic evidence; it does not automatically authorize a new calibration.
3. Audit Eq. (1) feasibility using named collision pairs and existing geometry.
   Match IK and runtime pairs; distinguish assembly exclusions, shell artifacts,
   and genuine possible self collisions. Fix demonstrated copy/scale/pair-parsing
   errors before proposing geometry redesign. Neither enabling every shell pair
   nor retaining the sparse source list proves a correct collision model.
4. Classify failures before deciding what to change. Joint/self violations are
   retargeting issues. Initial penetration and velocity inconsistency are reset
   or model issues. Later contact/tracking failures belong to Replay -> RL unless
   evidence establishes a model/implementation defect or simulator breakdown.
   Full-reference hand/object intersections and GT/mesh mismatch remain visible
   diagnostics; they are not an extra paper-wide no-intersection success test.
5. Resolve and freeze the initial-state specification before physical validation.
   The paper does not publish TACO initial qpos/qvel, reference row selection,
   settling, depenetration, or allowable offsets. The measured first-frame
   13.40 mm hand/table overlap and brush/table overlap are historical brush
   findings; do not transfer them to Pour or infer an offset from them. Pour's
   robot initialization has now been measured: native palms penetrate the table
   by 11.10/7.75 mm (right/left), and sampled left fingers enter the plate by
   up to 1.51 mm. A subsequently authorized hand-only diagnostic produced a
   separate declared-feasible candidate; the baseline reset was not replaced.
   If the
   paper-backed checks above do not explain them, report the missing decisions
   and measured minimal alternatives for user confirmation before changing the
   reset. Do not declare the current initialization valid by relabeling its gate.
6. The MJWP adapter now uses one 30 Hz endpoint command per ten physics steps,
   returns post-reset observations, enumerates both hands' sites by name, exposes
   per hand-object contact features, executes reference command + residual, and
   restores full simulator state for lookahead. The active TACO configuration
   has zero reset and domain-randomization noise. Keep the initial state and
   model fixed across solver comparisons; record failures rather than requiring
   Replay success in advance.
7. Configure unpublished coefficients explicitly. Pour's published 0.12 m and
   1.5 rad example is recorded, but does not uniquely specify lambda_p/lambda_R/C.
   Aria's 0.08 m / 2.5 rad / contact bonus 2.0 are not TACO paper parameters.
   The PPO entry point exposes separate `tool_only` and `tool_and_target`
   tracking variants. Run and report them separately after reset validation; do
   not use old smoke checkpoints as scientific baselines.

## Proposals On Hold

The following earlier engineering proposals are not described by the paper and
are not the default next implementation steps: multi-frame depth-based table
realignment; adopting the diagnostic hand projection as an accepted reset; initial
object pose corrections; settling followed by adoption of the settled state as
the new task origin; wholesale hand collision replacement by native-mesh convex
decomposition; and object decomposition chosen merely to make a rollout pass.

The first-frame hand-only diagnostic itself was subsequently authorized and
executed. That authorization does not install its candidate, choose initial
velocities, change geometry or extend the reference trajectory.

Compliant brush bristles, synthetic pregrasp, frame trimming, per-frame world
shifts, and changing C to conceal mismatch are not part of the active plan.
Do not teleport objects to GT every step or actuate either object. A diagnostic
must not silently change reset states, reference targets, or success criteria.
Any necessary new modeling/reset procedure must be labeled as a local extension
and confirmed before implementation, not attributed to EgoEngine.

This paper-first revision changes the plan only. Existing runtime constraints,
renderer gates, model assets, GT, initial states, and measured results are not
changed. Engineering validity and rendering gates remain distinct from C.3 SR.

No rendering path was added. A future passing rollout must still use only the
approved copied `diagnostics/render_exact_deximit_triptych.py` and its gate.
