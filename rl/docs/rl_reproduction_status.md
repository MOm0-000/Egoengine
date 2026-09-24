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

A follow-up read-only attribution replayed both frozen checkpoints while
recording the network mean before the PPO `[-1,1]` action limit. Both augmented
rollouts remained bitwise equal to their frozen CPU traces. Across endpoints
40--50, the scaled policy had some output beyond the limit (`27.02%` above 1,
`1.01%` above 2, maximum `2.41`), but this does not explain the critical bowl
translation failure. At endpoints 46--50, none of the new right-wrist
translation outputs exceeded 1; their maximum magnitude was `0.9184`.

The critical right-wrist translation direction instead diverged from the old
policy: flattened action cosine `-0.493`, signed cumulative cosine `-0.594`,
and only `40%` matching signs. The old command offsets stayed approximately
`(+x,-y,+z)` and reduced the bowl's y error by `38.96 mm`; the scaled policy
turned toward `(-x,+y,near-zero/-z)`, while y error increased `34.99 mm` and z
error worsened `26.56 mm`. Its absolute right-wrist translation command was
also only `34.52%` of the old policy's, but the negative direction agreement
means this is not a pure strength shortage. Scalar scale increase/sweep is
therefore rejected as the next experiment. The next read-only question is
whether these terminal states and useful correction directions were represented
in the PPO training data.

The actuator-name audit also exposed a separate contract risk: the first three
controls of each wrist are slide-joint targets in metres, while wrist rotations
and fingers are in radians. One scalar residual scale currently spans both
units. This is recorded as a risk, not claimed as the cause of the endpoint-50
failure. Full evidence is in
`runs/taco_pour_action_attribution_v1/report.json`.

The requested one-step CPU sensitivity audit then restored the exact policy
states that emit the actions producing outcome endpoints 46--50 (source states
45--49). It held the other 33 policy controls fixed and tested symmetric right-
wrist translation changes of `+/-0.5 mm` and `+/-1.0 mm`. Because old policy
commands often lie on the formal `+/-0.05 m` boundary, policy-centred probes use
a diagnostic-only `+/-0.051 m` cap for one step; this is not a runtime or
training proposal.

The experiment did **not** recover a scale-stable local favourable direction.
For policy-centred probes, the per-axis gradient sign agreed between the two
epsilon sizes on only `53.3%` of old-policy axes and `40.0%` of scaled-policy
axes. The mean cosine between the two estimated gradient vectors was only
`0.271` for the old states and `-0.206` for the scaled states. The scaled policy
action had negative cosine with the estimated favourable increment at four of
five endpoints, but the old policy also had negative cosine at three of five;
therefore those cosines cannot be treated as a trustworthy causal label.

This result preserves the trajectory-level fact that the policies issue
different late wrist directions, but it does not prove that one-step local
physics identifies the old direction as correct. The likely limitation is the
non-smooth contact response and one-control-period horizon, but the report does
not turn that explanation into a fact. No scale sweep or training is authorized
by this audit. A follow-up, if approved, would need a predeclared contact-mode-
aware multi-step response or local system-identification contract. Full evidence
is in `runs/endpoint46_50_local_control_sensitivity_v1/report.json`.

The approved three-step follow-up was limited to the common frozen-action
support at source endpoints 45--47. It perturbed only the first right-wrist
translation command and then replayed the next two already-recorded actions;
neither policy was called again. Under the predeclared strict contact criterion,
all feasible perturbations changed the 30-substep global contact geometry
multiset (old 18/18, scaled 36/36), while repeated unperturbed baselines were
bitwise exact. This proves there was no sample with globally unchanged contact
topology; it does not prove that the right-hand/bowl subsystem admits no local
approximation under a different, post-hoc contact definition. The local finite-
difference route is therefore stopped rather than repeatedly redefining the
filter. Evidence is in `runs/endpoint45_49_contact_mode_response_v1/summary.json`.

The action-log contract has since been upgraded to
`taco_ppo_training_visitation_v3`. Its raw files record three full 36-D arrays:
the requested control-target residual, the residual remaining after the
compiled model's enabled actuator `ctrlrange`, and the signed part lost to that
range. The after-range quantity is still only a control target; it is not the
motion physically realized by the robot. New CPU validation traces and
`optimized_trajectory.npz` use the same names. Historical v2 evidence remains
immutable; its `applied_residual` name means the requested pre-`ctrlrange`
quantity.

The v3 logger passed the same paired CPU transparency check: initial and final
actor, critic, optimizers, complete physics state, RNN state, and all recorded
RNG states stayed bitwise identical with logging off versus on. Applied to the
frozen 3+1 CPU trace, it reproduced 0 wrist translation losses, 0 wrist
rotation losses, 62 truncated finger requests, 57 fully blocked requests, and
28 affected steps out of 33. See
`runs/taco_pour_training_trace_transparency_v3/report.json` and
`docs/taco_pour_residual_logging_v3.md`. No new scale or training run was
selected by this audit.

Training coverage in the historical 8-epoch run remains unknown because that
run did not save visited states, and a fresh non-deterministic GPU rerun cannot
reconstruct them. The formal PPO fallback now records each epoch's visited
reference endpoints, tool position/rotation/ellipse errors, right-wrist sampled
action distributions, coarse endpoint contacts, and termination endpoints. The
three action layers are the pre-clamp stochastic sample, the same sample after
the official `[-1,1]` clamp, and the requested control-target residual after
local scale/safety clipping but before actuator `ctrlrange`. The first layer
includes exploration noise and is not the
actor mean `mu`; neither `mu` nor policy variance is recorded in this schema.
Raw samples are retained alongside summaries and artifact hashes.

This logging was admitted only after a paired CPU transparency gate. From the
same complete endpoint-20 boundary and seed, one four-step epoch with logging
off and one with logging on had bitwise-identical initial actor, actor optimizer,
critic and critic optimizer. Their final actor, both optimizers, critic,
complete simulator state, recurrent state, and Python/NumPy/Torch RNG states
were also bitwise identical. The logger captured exactly four visits, sources
20--23 and outcomes 21--24. No PPO setting, reward, action mapping, sampling or
acceptance rule changed. The audit is
`runs/taco_pour_training_trace_transparency_v2/report.json`; it is an
implementation check, not a new task-performance experiment.

One prospective eight-epoch coverage run was then made from the exact saved
endpoint-20 CPU boundary with the scaled-action profile. It changed no PPO,
reward, observation, physics, seed, or acceptance setting. Across 320 training
samples, outcome endpoints 46--50 were visited `6, 6, 4, 4, 4` times,
respectively: six episode segments reached endpoint 46, but only four reached
endpoint 50. These five endpoints account for 24 samples (`7.5%`) in total.
The zero-coverage hypothesis is therefore rejected, while the attrition caused
by earlier tracking terminations is directly observed.

This result does not establish an adequate-coverage threshold or prove sparse
coverage is the cause of failure. The newly trained deterministic CPU policy
validated only `28/40` intervals and failed at endpoint 49; the paired Replay
baseline validated `29/40` and failed at endpoint 50. No chunk was committed,
no full-horizon run was started, and no algorithm change has been selected from
this result. The report is
`runs/taco_pour_training_coverage_v1/coverage_analysis.json`. It applies only
to this prospective run and does not reconstruct historical eight-epoch
training. It also makes no claim about actor mean `mu` or policy variance.

The next predeclared intervention changed only the number of independent GPU
training rollouts from one to four. Four separate one-world MJWP instances are
used rather than partially overwriting one batched contact buffer. Before
training, all four instances restored the same endpoint-20 source across all
342 Warp state fields bitwise, including current/previous contacts and
constraints. Resetting one instance restored its complete boundary while the
other three snapshots remained bitwise unchanged. The same-action GPU control
baseline nevertheless diverged by up to `7.38 mm` in qpos after one control
interval, so the gate retains the known GPU atomic-order nondeterminism and
does not claim all physical divergence is caused solely by action sampling.

The single authorized four-world/eight-epoch run recorded 1,280 samples.
Endpoint 46--50 visits increased from `6/6/4/4/4` to
`23/23/20/18/12`, and endpoint 50 appeared in six rather than four epochs.
However, these endpoints still represented exactly `7.5%` of all samples
(`96/1280`, versus `24/320`): absolute data increased, but the rollout
distribution did not become more tail-focused. Deterministic CPU validation
improved from the prospective one-world policy's `28/40` to `31/40`, and
exceeded the same-run Replay baseline `29/40`; it failed at endpoint 52, so no
state was committed and no full-horizon run was started. This is modest
evidence that more independent rollout samples helped, not proof that sparse
coverage was the sole cause or a sufficient solution. Evidence is in
`runs/taco_pour_multiworld_training_v1/comparison.json`.

That comparison also changes PPO's per-epoch batch from 40 to 160 because the
official configuration uses `batch_size = worlds * horizon` and one full-batch
minibatch. The `28/40 -> 31/40` change therefore combines more independent
rollouts, a larger update batch, and potentially lower gradient sampling noise;
it cannot be attributed to tail coverage alone. The tail share stayed at
`7.5%`, and `31/40` remains below the historical saturated-action policy's
`38/40`. That historical mapping is still rejected because it collapses most
policy outputs to the action limit.

The next engineering-only gate captured endpoint 46--50 from a natural,
deterministic CPU rollout starting at endpoint 20. Every artifact pairs all 342
Warp fields with both `1 x 1 x 1024` LSTM tensors and binds them to the complete
actor state hash, including input-normalization statistics. Four mixed tail
states restored as `1 x 4 x 1024`; repeating restore plus one control interval
gave bitwise-identical actions, next RNN states, physics, observations, rewards,
dones, and info. Changing one actor parameter correctly rejected the old
memory. No GT state was injected and no PPO training was run.

The actor-update refresh path is also now tested. Every boundary stores the
natural observation prefix from endpoint 20. Replaying it under the unchanged
actor reproduces all five saved LSTM states bitwise. After one actor parameter
is changed, the old hidden state is rejected; replaying the same prefix under
the changed actor produces a newly hash-bound hidden state that restores with
the unchanged physical state. This makes recurrent memory consistent with the
current actor and recorded physical history.

The real-GPU no-training sampler gate then fixed the starts to
`[20,20,20,46]`. Prefix replay did not change the actor/input-normalization
hash; a real tail termination restored both complete endpoint-46 physics and
the current-actor LSTM memory. A changed LSTM parameter rejected stale memory
and regenerated a newly bound hidden state. The tail physics remains an
explicit off-policy start generated by the earlier behavior policy.

The one authorized 3+1 experiment kept the declared `4 worlds / 8 epochs /
1280 PPO samples` budget and CPU endpoint-20 `40/40` acceptance unchanged.
Endpoint 46--50 visits increased from `96 (7.5%)` to `252 (19.6875%)`, and
endpoint 50 appeared in all eight epochs. The endpoint-46 world contributed
190 tail visits but terminated on tracking 43 times and never reached the
endpoint-60 timeout. CPU Replay remained `29/40`; PPO reached `32/40` and first
failed at endpoint 53. This did not pass the gate. Compared with the prior
four-anchor policy's `31/40`, the one-step difference from a single
non-deterministic GPU training run is not a causal estimate. No curriculum
ratio, endpoint, epoch, seed or world-count sweep was run, and tail sparsity is
not established as the root cause.

That closes the tail-sample-count line for now. No 2+2/1+3 variant, alternate
tail endpoint, additional world, epoch, or seed was run. The next read-only
audit instead examined the unresolved mixed-unit action mapping over all 197
reference transitions. The common numeric limit of `0.05` is 5 cm for wrist
translation but 0.05 rad for wrist rotation and fingers. Its ratio to the
aggregate P95 reference increment is `5.438 / 0.618 / 0.355` for the right
hand and `7.395 / 0.591 / 0.412` for the left (translation / wrist rotation /
finger components). Thus the reference changes do not balance the shared
numeric scale: translations receive several P95 steps of authority, while the
rotation and aggregate finger limits are below one P95 step. This is a
command-space diagnostic, not realized robot motion or proof of the endpoint-53
cause. No split scale was selected and no training was authorized. Full values
are in `runs/taco_pour_reference_action_scale_audit_v1/report.json`.

The follow-up authority audit used the frozen deterministic CPU validation
rows rather than reference increments alone. Full-window component saturation
was `18.18% / 19.70% / 20.96%` for wrist translation / wrist rotation /
fingers; in the final ten endpoints it was `16.67% / 30.00% / 22.08%`.
Translation therefore is not merely an unused over-wide channel, while angular
and finger bounds are also clearly active. Current-state to next-reference P95
gaps reached `0.0610 m / 0.1472 rad / 0.4546 rad` overall and `0.0489 m /
0.1119 rad / 0.6532 rad` in the last ten endpoints. These gaps are descriptive
servo/reference errors, not optimal residual labels.

Actuator `ctrlrange` adds a distinct finger-only restriction. Sixty-two
requested finger residuals were truncated across 28 of 33 validation steps;
57 were fully blocked at a bound. The last ten steps contained 22 truncations
and every one of those steps was affected. Wrist translation and rotation had
no range truncation. The mixed-unit action interface remains structurally
unclean, but this one trace does not establish it as the primary endpoint-53
cause, and simply increasing angular/finger scale is not supported. No new
scale or training run was selected.

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
- epoch-level PPO visitation logging, admitted by a bitwise CPU transparency
  gate and attached to the formal PPO fallback.
- one prospective endpoint 46--50 coverage measurement, with raw epoch samples,
  hashes, and deterministic CPU validation retained.
- a four-world complete-state/reset gate and one frozen four-world/eight-epoch
  comparison, without seed, epoch, reward, action, or success-threshold search.
- a no-training natural-tail physics/RNN paired-reset gate, including stale
  actor-memory rejection.
- a real-GPU fixed 3+1 sampler/reset gate and one frozen 1,280-sample
  tail-focused experiment, without a curriculum-ratio or endpoint sweep.

Not yet established:

- paper-faithful numerical reward or exact actor encoding;
- repeatable full 197-transition main-result success;
- a paper-supported mapping from the published position/rotation thresholds to
  `lambda_p/lambda_R/C`;
- stable grasp or a convincing physical pour;
- independent full-horizon replay of a trajectory containing PPO actions;
- generalization of this reset/collision calibration to other TACO samples;
- capacity for arbitrary unseen PPO states beyond the recorded stress tests.

The current isolated-suite count is recorded in `docs/test_environment.md`.
At an earlier checkpoint, the MuJoCo 3.13 high-SDF GPU stack passed
579 tests and 57 subtests. The high-SDF stack additionally warns that capsule–mesh
CCD pairs support at most one contact; this is recorded as a backend limitation,
not hidden as a successful multicontact guarantee.
