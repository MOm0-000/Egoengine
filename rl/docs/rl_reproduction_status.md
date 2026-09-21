# RL reproduction status

Updated: 2026-09-21. Active sample: TACO Pour/Bowl/Plate `20230927_017`.

## What enters RL

The action-generation input is:

- a 198-endpoint, 30 Hz bimanual XHand MINK reference (`qpos/qvel/ctrl`);
- synchronized Bowl/Plate GT poses, used as an approved oracle for this validation;
- the hash-bound MuJoCo scene and collision assets;
- one accepted full simulator reset (`qpos/qvel/ctrl`, reference index 0);
- an explicit local reward profile and an explicit local observation profile.

Human MANO joints feed MINK, not PPO directly. RGB/depth are audit inputs and
are not actor observations. MPC is intentionally omitted, so this project is a
Replay→residual-PPO variant of EgoEngine Appendix C.1, not a reproduction of
the paper's full Replay→MPC→RL cost result.

## Current gates

The formal runtime collision model is
`runs/taco_pour_floor_contact_v1/candidate.xml`. Its object SDFs are compiled
at depth 8 (bowl) and 9 (plate), producing 1,240,801 and 6,484,721 octree nodes.
The runtime model and independently rebuilt physics contract match exactly:

- physics contract: `fa778d1e1654f2c1dc016280fa4e7c05b2ffa6416b3b7f063c6e0f94d5c79f11`;
- capacity: 256 contacts and 1024 constraints per world;
- 198 reference snapshots plus 40 held-control stress records pass with no
  overflow; tested maxima are 108 contacts and 556 constraints per world.

Initialization protocol v2 accepts Candidate A. It preserves object endpoint 0,
uses zero qvel, releases with no object equality hold, and passes five passive
control intervals (50 physics steps). Candidate B remains rejected and is not
silently substituted. The accepted report is
`runs/taco_pour_initialization_protocol_v2/candidate_a/report.json`.

The exact EgoEngine actor vector is not published. The selected
`taco_pour_transition_aligned_236d_v1` profile is therefore explicitly
`local_unpublished`, not paper-faithful. Before transition `t→t+1` it contains:

| Field | Dim | Timestamp |
| --- | ---: | --- |
| hand qpos / qvel | 36 + 36 | `t` |
| fingertips / palms | 30 + 6 | `t` |
| current object anchors | 18 | `t` |
| target object anchors | 18 | `t+1` |
| reference command | 36 | `t+1` |
| reference command preview | 36 | `t+2`, local causal extension |
| contact flags | 20 | `t`, local extension |

The asymmetric critic has 108 dimensions. No measured future state is used;
`t+1/t+2` reference rows are available because the entire offline reference is
known. The `t+2` command is not justified by the paper's two-chunk scheduling
window and is never described as an author setting.

The paper's TACO reward coefficients remain unpublished. Formal paper-faithful
training therefore remains blocked. The first engineering run uses only the
named `taco_pour_local_unpublished_v1` objective. Its values and provenance are
stored separately; they are never inferred silently from the Pour example
thresholds.

## Executed results

The first 40-step window passes Replay for both tracking variants and commits
endpoint 20. The scheduler was then run over the full 197 transitions and now
exports every committed endpoint and action (`qpos/qvel/ctrl`, raw and applied
residual, and selected mode) to a hash-bound NPZ.

The main `tool_only` result is not yet accepted. Across otherwise identical GPU
runs, the single RL chunk appeared at chunk 20 or 120; a later run needed RL at
chunk 120 and then failed after endpoint 140 because the two-epoch PPO fallback
could validate only 20 steps before the tool crossed the local boundary at
endpoint 161. The runner correctly committed nothing after endpoint 140.

A previous run demonstrates that PPO is genuinely connected: from the same
endpoint-20 boundary, Replay crossed the local boundary at endpoint 50 with
weighted error `1.5354`, while the trained residual policy reduced the same
endpoint to `0.5914` and passed 40/40 validation steps. However, the saved action
sequence from that run failed an independent open-loop replay at endpoint 24.
It is evidence that PPO can repair one validation window, not yet a reproducible
full action trajectory.

The `tool_and_target` extension has one complete all-Replay run whose 197 saved
actions also pass an independent stitched replay. This still is not accepted as
paper task success: the local scalar objective permits 38 tool-position rows
above the paper's published Pour example threshold of `0.12 m` (first at
endpoint 59, maximum `0.1875 m`). Opposition contact bonus is nonzero for only
3/197 steps and the largest lift implied by the lift term is about `14.9 mm`, so
the result does not establish a stable grasp or convincing pour.

The current consolidated evidence is
`runs/taco_pour_replay_rl_full_v3/summary.json`.

## What is ready, and what is not

Ready:

- collision/retarget contract for this Pour validation;
- accepted reset and passive release;
- hash-bound runtime physics and tested capacity;
- explicit local objective and observation encodings;
- first formal Replay window and endpoint-20 commit;
- real PPO fallback, complete action export, and independent saved-action replay gate.

Not yet established:

- paper-faithful numerical reward or exact actor encoding;
- repeatable full 197-transition main-result success;
- a paper-supported mapping from the published position/rotation thresholds to
  `lambda_p/lambda_R/C`;
- stable grasp or a convincing physical pour;
- independent full-horizon replay of a trajectory containing PPO actions;
- generalization of this reset/collision calibration to other TACO samples;
- capacity for arbitrary unseen PPO states beyond the recorded stress tests.

Both MuJoCo/MJWP stacks pass the complete repository suite: 566 tests and 57
subtests. The MuJoCo 3.13 high-SDF stack additionally warns that capsule–mesh
CCD pairs support at most one contact; this is recorded as a backend limitation,
not hidden as a successful multicontact guarantee.
