# Early contact origin v1

Status: `complete` (static/offline attribution only).

This audit reused the frozen 0→20 Replay trace. It executed **0 physics steps, 0 control intervals, 0 new actions, 0 network forwards, 0 optimizer updates, and 0 IK solves**. RR/AR/RA/AA are offline geometry combinations and are not executable trajectories.

## Core localization

| hand/object | observed task phase | earliest observable mismatch | H→R evidence | R/U→A evidence | collision/record evidence | uncertainty | next specific change to review |
|---|---|---|---|---|---|---|---|
| right / bowl | approach and contact establishment; the inspected human sequence does not yet demonstrate stable thumb–index force closure | the accepted s0 hand pose already differs from reference[0]; during the bowl event the actual hand also lags the saved reference | at ep14 the human GT visual-mesh distances are thumb `70.641 mm`, index `32.706 mm`; at ep16 they are `47.211/13.624 mm`. The reference preserves the index-led relation. H-target→R-site errors are only `9.084/4.242 mm` at ep14 and `6.991/6.159 mm` at ep16 (thumb/index) | s0 hand qpos differs from reference[0] by `0.055850`; at ep14 R→A site errors are `25.013/15.761 mm`, growing to `36.660/31.849 mm` at ep16. ep14 index RR→AR visual distance changes `34.985→42.037 mm`, while RR→RA is only `34.985→35.042 mm` | thumb→tool has 65 explicit-pair paths and index→tool has 33, so thumb absence is not a missing pair/filter path. Saved solver rows contain index→tool from source14/substep7, lose it at source15/substep3, and reacquire at source16/substep5; there are no thumb→tool rows. Right thumb→floor begins at global substep5 | RGB is not registered to the simulation camera; pad contact is partly occluded. Static site/mesh or queryable-pair distances are not solver contact tests | review the accepted initial hand pose and first-command transition, including right-thumb/floor loading and the distal index exact-mesh contact location. Do **not** add a mandatory thumb objective from this evidence |
| left / tray | hand approaches a tray that remains supported by the table; support is shared with the floor | two layers differ: H→R changes the nearest landmark from human ring-side to robot pinky-side, while actual tray drift begins immediately under the custom s0/contact load | at ep0 human GT ring/pinky visual distances are `2.512/12.411 mm`; RR robot sites are `6.779/0.168 mm`. H-target→R-site errors are ring `4.518 mm`, pinky `20.735 mm` | s0 R→A site errors are thumb/ring/pinky `30.932/35.777/30.969 mm`; the first left command changes by up to `0.046301` (m or rad per actuator). Ring+pinky contact the tray at source0/substep0; the tray has moved `1.010 mm` by ep1 while its reference moved `0.026 mm` | ring/pinky→target each have 33 explicit-pair paths and saved exact-mesh contact rows from global substep1; target→floor begins at global substep2. No evidence of a missing recorder path was found | human ring/pinky load sharing is not visually resolvable; the changed nearest landmark does not by itself prove the retarget is wrong; per-contact-point forces are aggregated only within each saved solver substep | first review the candidate-A initial left-hand pose and source0 first-command transition under ring/pinky→tray plus tray→floor load. Separately review whether the H→R ring/pinky support-mode change was intended; do not tune all gains globally |

## Bowl–tray relation: the visual concern is confirmed

The reference moves the tray by only `0.098 mm` from ep0 to ep20, while the Replay tray moves `25.211 mm`. The bowl–tray relative-position error is already `4.863 mm` at ep10, `10.560 mm` at ep15, `24.111 mm` at ep16, and `72.675 mm` at ep20. The tray therefore drifts before the source14→15 right-index/bowl event; that later event cannot explain the entire early mismatch.

Both objects are floor-supported in the saved solver rows. The first ring+pinky→tray load occurs at source0/substep0; tray→floor begins one substep later. At that first substep, the saved normal-force sums across contact points are `36.973 N` (ring) and `62.327 N` (pinky). These are per-solver-substep sums, not forces accumulated over time, and no claim is made that a particular magnitude is itself invalid.

## Right-hand event and four-way factorization

The human phase is index-led: the index landmark is closer to the bowl than the thumb at ep14, ep16, and ep20. RR reproduces that ordering. Therefore “no recorded thumb contact at source14–16” is not sufficient evidence of an H→R bug.

At ep14, object-only substitution barely changes the right-index visual distance (`RR 34.985 mm`, `RA 35.042 mm`), while actual-hand substitution increases it (`AR 42.037 mm`, `AA 41.974 mm`). At ep16 both layers matter (`RR 15.855`, `AR 27.048`, `RA 22.413`, `AA 37.011 mm`). This establishes an execution-posture gap before the large bowl motion, followed by combined hand and object divergence. It does not assign a percentage of causality.

The exact external mesh–SDF pair that produced the index contact is not reliably represented by static `mj_geomDistance`; many such queries are censored. Queryable semantic-pair distances and tip→visual-mesh distances are retained only as descriptive geometry. Actual contact claims use the saved solver rows.

## Initialization and control mapping

- The runtime actuator contract maps the 36 position actuators directly to qpos addresses `0..35`; slide joints are reported in metres and hinges in radians.
- Recorded ctrl at endpoints1…20 equals the corresponding saved reference ctrl to `1.184e-7` max absolute error. The transition source k→outcome k+1 uses reference ctrl[k+1]; no one-frame command-index error was found.
- Endpoint0 matches the accepted initialization state to `5.79e-8`, but that accepted state is a local `collision_free_pre_manipulation_pose_v2`, not reference[0].
- The first reference command differs from the accepted initial ctrl by right/left L2 `0.044646/0.078575`, with max component `0.033417/0.046301` (mixed units are not combined into a physical norm).

Thus this audit does not identify a wrong ctrl column or wrong reference index. It identifies a task-specific initialization/contact-loading transition that deserves the next review.

## Visual review

Reviewed directly:

- `review/human_reference_actual_early.png` at endpoints 0, 10, 14, 15, 16, 20;
- every frame of `review/right_thumb_index_12_20.mp4` and `review/left_support_0_15.mp4` via contact sheets;
- fixed-camera `review/pose_factorization_014.png`, `015.png`, `016.png`;
- marker, surface/collision, control, object-tracking, and contact-timeline plots.

The original RGB supports only an approach/contact-establishment interpretation. The right index visibly leads the thumb toward the bowl; no stable thumb-index pinch can be certified. The tray remains visibly table-supported while the left ring/pinky-side hand approaches. Occlusion and the independent RGB camera prevent visual certification of force-bearing finger pads.

## Limits and unchanged status

- RR/AR/RA/AA isolate saved geometry only; they do not prove dynamic realizability.
- Human landmarks, robot sites, visual triangles, collision geoms, and solver contacts are different observables and remain separate in the CSV.
- `mj_geomDistance == distmax` with zero `fromto` is recorded as censored, not as a measured clearance. Exact mesh–SDF contacts can exist when the queryable semantic-pair minimum is positive.
- `left_hand:other` and `right_hand:other` roles remain explicitly reported rather than silently assigned to a named digit.
- Historical tool-only tracking remains below its old boundary through ep20; prior Replay identity is unchanged. Task-relation concern is confirmed; downstream viability is not established.
- No collision, retarget, initialization, or controller change is authorized by this audit.

## Artifact index

- Quantitative: `digit_geometry.csv`, `control_tracking.csv`, `contact_path_checks.csv`, `demonstration_contact_intent.csv`
- Main figures: `review/human_reference_actual_early.png`, `review/object_tracking_0_20.png`, `review/contact_timeline_0_20.png`
- Local views: `review/right_thumb_index_12_20.mp4`, `review/left_support_0_15.mp4`, `review/pose_factorization_014.png` through `016.png`
- Machine-readable provenance and result: `inputs.json`, `summary.json`, `visual_review.json`, `server_artifacts.sha256`
