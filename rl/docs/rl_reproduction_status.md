# RL reproduction status

Updated: 2026-09-22. Active sample: TACO Pour/Bowl/Plate `20230927_017`.

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
training therefore remains blocked. The active engineering objective is now
the explicitly local `taco_pour_local_normalized_ellipse_v1` proxy:

```text
sqrt((position_error / 0.12)^2 + (rotation_error / 1.5)^2) <= 1
```

It uses the paper-reported Pour scales as axis intercepts, not as recovered
author coefficients. Every new validation row also logs the independent
diagnostic `(position <= 0.12) AND (rotation <= 1.5)`. The previous raw-unit
objective and its results remain available as a sensitivity comparison.

## Previous raw-unit objective results

Under the previous `taco_pour_local_unpublished_v1` objective, the first
40-step window passed Replay for both tracking variants and committed
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

An objective-mapping sensitivity audit re-scored the saved traces without
changing controls or training. On the 197-step two-object trace, the current
raw-unit diagonal mapping accepts 197/197 steps. Three plausible readings of
the published `0.12 m / 1.5 rad` example instead reject the trace:

| Diagnostic interpretation | Violating steps | First violation |
| --- | ---: | ---: |
| axis-intercept normalized ellipse | 51 | 32 |
| ellipse passing through the threshold corner | 11 | 66 |
| independent position/rotation limits | 38 | 59 |

On the 140-step committed `tool_only` prefix, the same alternatives reject 98,
28, and 70 steps respectively. This audit originally selected no mapping. The
axis-intercept ellipse is now explicitly promoted only as the approved local
proxy; it remains unresolved as a paper mapping, and the independent-limit
case is not Eq. C.3/C.4. The audit is
`runs/taco_pour_objective_mapping_sensitivity_v1/`.

## Normalized-ellipse rerun

The fresh `tool_only` Replay audit passed its first 40 steps with maximum
ellipse score `0.8427`. A separate `tool_and_target` rollout failed at endpoint
31 with tool score `1.0223`; its position (`0.0796 m`) and rotation (`1.1673
rad`) still passed the independent thresholds. This is expected because the
ellipse is stricter than the rectangular independent check, and it confirms
that both metrics are being logged separately.

The bounded two-epoch PPO smoke did not pass its failed window. One additional
bounded eight-epoch smoke provided positive but insufficient evidence: from the
same endpoint-20 boundary, Replay validated 27 steps and failed at endpoint 48,
whereas PPO validated 35 and failed at endpoint 56. Across their 28 common
endpoints, PPO reduced the score on 22 and lowered the mean from `0.7340` to
`0.6280`; at endpoint 48 it reduced `1.0508` to `0.4304`. It finally missed the
ellipse boundary by a small but real margin (`1.00133`) while still passing the
independent thresholds.

The original GPU validation was not repeatable across runs. Auditing found a
real snapshot bug: the old snapshot omitted several MuJoCo-Warp 3.13 Data,
Contact, and Constraint arrays. Snapshot v2 now captures all 342 current and
previous runtime arrays and refuses legacy partial snapshots. After this fix,
the restored state was bitwise exact, but GPU rollouts still diverged inside
the first physics substep. The contact pair set was equal when order was
ignored; contact ordering, constraint row allocation/forces, qvel, and qpos
were already different. This matches MJWP's documented GPU atomic-operation
non-determinism rather than another missing state field.

The same full snapshot and physics contract were then tested with MJWP on CPU.
All ten substeps in the first control interval were bitwise identical, followed
by exact trajectory signatures for three repeats in the same environment and
three fresh environments. CPU validation of the recurrent eight-epoch
checkpoint gave the following paired result on one deterministic endpoint-20
boundary:

| Candidate | Validated intervals | First failed endpoint |
| --- | ---: | ---: |
| Replay | 29/40 | 50 |
| 8-epoch closed-loop PPO | 38/40 | 59 |

Thus the PPO improvement is real and repeatable (`+9` intervals), but still
fails the strict 40/40 gate. One predeclared 16-epoch run was then allowed. Its
deterministic CPU result was 27/40 and failed at endpoint 48, two steps before
Replay. These checkpoints came from separate non-deterministic GPU training
runs. The result therefore establishes that the frozen 8-epoch policy is better
than the frozen 16-epoch policy; it does not establish that one policy was
damaged by continuing its training from epoch 8 to epoch 16. The 16-epoch run
did not preserve an epoch-8 checkpoint from the same optimization path. No seed
sweep, 24/32-epoch escalation, or normalized-ellipse full run was started.

A subsequent read-only policy audit checked the two checkpoint payloads and
the deterministic CPU traces before authorizing any further training. Both
checkpoints contain the 236-dimensional input-normalization mean, variance, and
count as registered model buffers. Strict loading into the CPU actor reproduced
all 21 model tensors bitwise, including those buffers and the recurrent weights.
Training starts from the default zero recurrent state, clears it on every done
endpoint, and restores the exact chunk boundary; CPU validation also clears the
state before each trial. No inference-state mismatch was found.

The same audit found a much stronger action-contract issue. PPO exposes a
normalized `[-1,1]^36` action, while the runtime currently applies
`clip(action, -0.05, +0.05)` rad. Consequently, `95.94%` of all 8-epoch action
components and `95.63%` of all 16-epoch components were exactly saturated at
the residual limit. Every recorded policy step saturated at least one joint,
and some saturated all 36. At the 8-epoch failure tail (endpoints 55--59), the
contact bonus was zero and lift reward stayed below `0.00072`, while tracking
reward fell by about `0.188`; the saved trace therefore does not support reward
masking as the cause of that final failure. Per-finger contact flags/forces were
not saved, so a finer contact-switch claim is deliberately not made.

The predeclared decision rule therefore selected one action unit/range
experiment. Appendix C.2 explicitly says that action-smoothness reward is
disabled for TACO, so no smoothing term was added. The controlled candidate
mapped normalized output linearly across the existing `+/-0.05 rad` limit,
`delta_a = clip(0.05 u, -0.05, +0.05)`, while holding the endpoint-20 state,
seed, eight-epoch budget, reward, observation, network, optimizer, horizon, and
CPU acceptance backend fixed. The paper does not publish this residual scale;
the candidate is explicitly local rather than author-recovered.

The exact-boundary experiment did repair the action representation but did not
improve the strict task gate. The copied endpoint-20 snapshot has the same
artifact hash as the historical boundary, and its Replay trace is bitwise
equal. Runtime action saturation fell from `95.94%` to `26.57%`; the maximum
adjacent scalar jump fell from `0.10000` to `0.03465 rad`. Nevertheless, the
new eight-epoch policy validated only `29/40` intervals and failed at endpoint
50, compared with `38/40` and endpoint 59 for the frozen old policy. Three CPU
repeats of the new policy were bitwise identical. The first failed endpoint was
dominated by position error (`0.11565 m`) with rotation error `0.68332 rad`.
The candidate therefore is not promoted, no additional epoch/seed search was
run, and endpoint 55--59 cannot be compared because the candidate never reached
them. This result does not make the original unit-mismatch diagnosis false: it
shows that fixing the interface alone, without retuning PPO for the smaller
executed-action scale, is insufficient. Full evidence is in
`runs/taco_pour_normalized_action_scale_v1/comparison.json`.

The evidence and hashes are frozen in
`runs/taco_pour_normalized_ellipse_v1/summary.json`. GPU remains the PPO
training backend, but it no longer has acceptance or commit authority. The
formal runner now implements the separate
`taco_pour_gpu_train_cpu_validate_v1` contract: Replay and trained policies are
evaluated closed-loop on CPU; actor inference is also moved to CPU; 40/40 is
required; and the next chunk can start only from the CPU validation rollout's
saved endpoint-20 state. GPU and CPU environments have separate runtime records
and hashes even though they share one physics contract.

The complete 342-field CPU snapshot has been copied to GPU and restored
bitwise at both endpoint 0 and after one real CPU control interval. A bounded
one-epoch integration test also verifies GPU training, bitwise actor-weight
transfer to CPU, recurrent CPU inference, and a CPU physics transition. This
integration did not grant another training-budget search and did not change the
8/16-epoch results above.

The formal runner was then exercised for one Replay-only chunk. CPU Replay
passed 40/40 and committed the CPU rollout's endpoint-20 state; GPU training
work was exactly zero, while CPU validation used 40 control intervals / 400
physics steps. The resulting report remains explicitly
`chunk_budget_reached_not_full_task_success`.

## What is ready, and what is not

Ready:

- collision/retarget contract for this Pour validation;
- accepted reset and passive release;
- hash-bound runtime physics and tested capacity;
- explicit local objective and observation encodings;
- first formal Replay window and endpoint-20 commit;
- real PPO fallback, complete action export, and independent saved-action replay gate.
- integrated GPU-training / deterministic-CPU-validation scheduler, including
  CPU-only acceptance and CPU commit-state ownership.

Not yet established:

- paper-faithful numerical reward or exact actor encoding;
- repeatable full 197-transition main-result success;
- a paper-supported mapping from the published position/rotation thresholds to
  `lambda_p/lambda_R/C`;
- stable grasp or a convincing physical pour;
- independent full-horizon replay of a trajectory containing PPO actions;
- generalization of this reset/collision calibration to other TACO samples;
- capacity for arbitrary unseen PPO states beyond the recorded stress tests.

The MuJoCo 3.13 high-SDF stack passes the complete repository suite: 579 tests
and 57 subtests. At the preceding checkpoint, the isolated CPU stack passed 559
tests and 57 subtests with the CUDA/MJWP module skipped. The high-SDF stack
additionally warns that capsule–mesh
CCD pairs support at most one contact; this is recorded as a backend limitation,
not hidden as a successful multicontact guarantee.
