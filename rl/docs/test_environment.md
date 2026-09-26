# Test environment

## Active reward-aligned baseline

The active Replay→RL chain uses:

- Python from `/data_all/zzx/egoengine/spider/.venv/bin/python`;
- the local `.env_mjwp313_overlay` containing MuJoCo 3.13,
  MuJoCo-Warp 3.13 and Warp 1.15;
- project sources in `src` and the checked-in MINK source in
  `external/mink/src`.

Run the complete suite from the project root with:

```bash
PYTHONPATH="$PWD/.env_mjwp313_overlay:$PWD/src:$PWD/scripts:$PWD/external/mink/src:$PWD/external/human2sim2robot:$PWD/external/spider_compat" \
  OMP_NUM_THREADS=4 \
  /data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q
```

On 2026-09-26 this command passed **705 tests and 57 subtests**.
The 19 warnings are known upstream/diagnostic warnings: capsule-mesh MULTICCD
capacity, PyTorch AMP deprecations, two Trimesh degenerate-volume warnings and
seven SciPy pickle deprecations.

The active runtime contract is reward-aligned:

```text
state t + command ref[t+1]
        -> physical state t+1
reward/termination against ref[t+1]
returned next observation goal ref[t+2]
```

The no-training temporal gate, corrected Replay rebase, CPU repeatability
gate, dual-backend transfer gate and the first corrected PPO run are under:

```text
runs/taco_pour_reward_alignment_gate_v1/
runs/taco_pour_corrected_replay_rebase_v1/
runs/taco_pour_cpu_backend_repeatability_v1/
runs/taco_pour_dual_backend_contract_v1/
runs/taco_pour_corrected_ppo_gate_v1/
runs/taco_pour_corrected_first_ppo_v1/
runs/corrected_endpoint44_48_failure_attribution_v1/
runs/taco_pour_corrected_local_controllability_v1/
runs/taco_pour_corrected_policy_decision_attribution_v1/
runs/taco_pour_corrected_input_bifurcation_attribution_v1/
runs/taco_pour_reference_timing_action_frame_audit_v1/
runs/taco_pour_training_credit_assignment_audit_v1/
runs/taco_pour_credit_instrumentation_gate_v1/
runs/taco_pour_corrected_fresh_ppo_credit_instrumented_v1/
runs/taco_pour_fresh_ppo_credit_evidence_v1/
runs/taco_pour_observation_normalization_gate_v1/
runs/taco_pour_observation_normalization_commit_gate_v1/
runs/taco_pour_postfix_fresh_ppo_credit_instrumented_v1/
runs/taco_pour_postfix_policy_extremization_audit_v1/
runs/taco_pour_postfix_single_actor_pass_diagnostic_v1/
runs/taco_pour_postfix_single_actor_pass_audit_v1/
runs/taco_pour_single_pass_endpoint47_49_gate_v1/
runs/taco_pour_source47_semantic_action_subspace_gate_v1/
runs/taco_pour_source46_state_entry_gate_v1/
runs/taco_pour_source45_state_entry_gate_v1/
runs/taco_pour_source45_prefix_source50_refinement_gate_v1/
runs/taco_pour_binary_translation_oracle_gate_v1/
runs/taco_pour_binary_translation_last_off_reversal_gate_v1/
runs/taco_pour_source56_semantic_action_gate_v1/
runs/taco_pour_tail_semantic_suppression_oracle_v1/
runs/taco_pour_tail_mode_necessity_transition_audit_v1/
```

The first corrected PPO authorization is consumed. Replay passed the first
40-step lookahead and committed the CPU endpoint-20 boundary. In the second
lookahead Replay passed 30 intervals and PPO passed 27; neither passed 40/40,
so no new boundary was committed and formal training is closed pending a new
algorithm decision.

The last-OFF reversal gate reproduces the 36/40 binary-translation oracle
decision arrays bitwise, then forces source 52, 53, 54 or 55 from OFF to ON
for one step before resuming the same one-step oracle. None survives endpoint
57: the four branches fail at endpoints 55, 55, 56 and 57 respectively. The
source-55 reversal does produce a measurable but small source-56 bowl response:
ON/OFF differ by `0.171 mm` in position and `0.00440 rad` in orientation, and
the ON branch makes pinky--bowl contact. Both source-56 choices still fail
endpoint 57. Thus the primary gate does not support a last-decision
greedy-myopia explanation. It also does not support either extreme claim that
source 56 has exactly no influence or that useful task-scale authority has
been restored. Training, learned-gate fitting and chunk commit remain blocked.

The source-56 semantic gate starts from the exact state produced by forcing
source 55 ON. It uses one shared actor forward and compares the two existing
translation anchors with five predeclared semantic suppressions: wrist
rotation, right fingers, complete right wrist, complete right hand and all 36
residuals. Every candidate remains at `36/40` and fails endpoint 57. Scores
range only from `1.00305593` to `1.00422549`; position contributes about
`0.8225--0.8247` of the squared ellipse while rotation contributes about
`0.1836--0.1838`. Contact is diagnostic only: rotation/finger suppression can
retain pinky contact and still fails. Source-56 semantic subspace selection is
therefore exhausted at the declared resolution, and attribution must move to
an earlier state. No training or commit is authorized.

The tail semantic oracle leaves sources 20--43 bitwise identical to the
binary translation oracle, then compares seven predeclared semantic actions
from the same snapshot and post-forward hidden at every source 44--55. Its
selected modes are two complete-wrist suppressions, six translation
suppressions, two full-36D suppressions, one complete-right-hand suppression
and one complete PPO action. This broader state-entry correction changes the
source-56 result decisively: all seven source-56 candidates now pass endpoint
57, and complete PPO is the lowest-score candidate at `0.99185961`. The
continued binary selector then fails endpoint 58 (`ON=1.02272546`,
`OFF=1.02330267`), so the result is `37/40`, not task success. The frozen PPO
suppression family is therefore not exhausted, but no deployable selector,
new training, reward change or chunk commit is authorized.

The follow-up necessity audit reads the saved seven-candidate scores directly
and uses no tolerance to merge modes. Complete-wrist suppression improves on
translation-only by `0.00217515` at source 44 and `0.00637680` at source 45.
In contrast, the full-36D winner margins at sources 48 and 52 are only
`2.98e-7` and `2.38e-6`; source 53 is an exact right-hand/full-36D tie, and
six candidates tie exactly at source 55. These winner names are therefore not
used as learned-gate labels and do not establish separate finger or bilateral
causality.

The same audit bitwise reproduces the selected tail path and the already
existing source-57 ON/OFF continuation; it does not introduce a source-57
semantic search. From source 56 to 57, the bowl moves only about `0.285 mm`
while its reference moves about `4.74 mm`; the z absolute error grows by
`4.44 mm`. Pinky contact is reacquired with summed normal force `10.689`.
From source 57 to 58, y and z absolute errors grow by `4.218 mm` and
`2.484 mm`, while that pinky force falls to `0.174`. Endpoint 58 is therefore
a new position-dominated failure involving reference lag and weakened contact
transmission, not evidence that a near-tied broad suppression mode is needed.
All algorithm changes and commit remain blocked.

The endpoint 44--48 attribution is a no-training replay of the frozen actor.
It requires exact equality with both saved formal CPU traces before reporting
new contact flags, action bounds, actor distribution values or training-v5
coverage statistics.

The local-controllability run is also no-training. It restores hash-bound
complete physics plus RNN states at endpoints 44--47, checks every zero
perturbation branch against formal endpoint 48, and marks all actions outside
the formal residual support as diagnostic-only.

The policy-decision attribution remains fully within the formal residual
support. It proves that CPU validation already uses deterministic truncated
mode, tests one-step wrist-y interventions, and records live MJWP contact
geometry/normal-force evidence from complete endpoint 42--47 snapshots. It
does not train, accept a chunk or authorize checkpoint resume.

The input/bifurcation attribution exactly reproduces both formal CPU traces,
then evaluates the frozen actor without training. The actor requests saturated
right-wrist `+y` on both PPO and Replay states from source 43 onward. A
successful one-step rescue does not make later policy means fall back inside
support, so the failure is not uniquely triggered by the PPO state path and is
not a mean-level self-correcting feedback loop. Full raw/normalized 236-D
inputs, recurrent states, actor outputs and action bounds are hash-bound in the
run directory. No recorded input reaches the normalization clamp; historical
empirical training-observation ranges remain unavailable.

The reference-timing/action-frame audit is the next read-only semantic gate.
It explicitly confirms the active `state[t] + ref[t+1] -> state[t+1]` ledger,
then varies only the reference context shown to the frozen actor. Showing the
actor the preceding reference context reduces its summed positive wrist-y mean
excess by 11.54%, but its diagnostic rollout still passes only 30/40 and fails
at endpoint 51. Replay, zero and previous-step recurrent states all leave the
mean outside support at sources 43--47, so hidden-state mismatch does not
remove the tail bias. Independent FK confirms that the current wrist slide
axes are world axes. A reference-palm local translation basis is geometrically
less anti-aligned with the bowl correction, but is still anti-aligned over
sources 43--46 and transforming the complete bilateral translation action
would leave the frozen action support from source 42 onward. That frame
candidate is therefore rejected without clipping, rescaling or rollout. No
timing or frame diagnostic is a formal success or a paper-recovered choice.

The training-credit assignment audit keeps the final actor and every runtime
contract frozen. It classifies actor timing shift `-1` as suppression rather
than refinement: it is slightly closer to Replay than the formal PPO path on
their common endpoints, but both shift `-1` and Replay fail at endpoint 51.
Extending the old single-step `y=0` rescues through the complete lookahead also
shows that none succeeds: interventions at sources 44/45/46 fail at endpoints
49/51/49. A fixed nine-point, state-feasible wrist-y sweep provides stronger
evidence about the final actor decision. The formal `+1` branch is
counterfactually suboptimal at every source 43--46; the best one-step choices
extend failure to endpoints 50/51/56/51, but no branch reaches endpoint 60.
These are frozen final-actor rollout returns, not historical critic Q values.
The v5 visitation artifacts did not save rollout reward, critic observation or
value, return, advantage, raw observation, or recurrent hidden state, and the
checkpoint has no rollout buffer or per-epoch critics. Consequently the sign
of the original PPO advantage cannot be recovered and is not backfilled using
the final critic.

The subsequent fresh diagnostic run changes no reward, objective, action
support, timing, frame, observation, critic or PPO hyperparameter. Its CPU
transparency gate shows that credit logging leaves the actor, critic, both
optimizers, complete physics state, RNN and Python/NumPy/Torch RNG bitwise
unchanged. The diagnostic preserves all 1,280 rollout rows, exact GAE inputs,
raw and normalized advantages, all 32 actor updates and a lossless actor-state
patch chain. It is explicitly non-promotable and commits no chunk: Replay
fails at endpoint 51 and the fresh actor fails at endpoint 50.

The recovered credit evidence revealed a concrete likelihood-contract defect.
On the first mini-epoch the actor weights are still unchanged, but the input
running mean/variance is updated before recomputing the PPO likelihood. That
normalizer-only change moves 134/160 old/new likelihood ratios outside the
configured `[0.8, 1.2]` clip interval (9/10 samples at sources 43--46), with a
range of `0.000517...` to `19.487...`. In the fixed probe panel the same
normalizer update moves wrist-y mean by as much as `0.344869`; the subsequent
optimizer step adds at most `0.105659`. Thus the first large `+y` transition is
not solely an optimizer response to GAE credit. The rollout log-probability
itself recomputes from the stored action, mu, sigma and feasible bounds within
`3.82e-6`, so the mismatch is specifically between the rollout observation
transform and the first update observation transform, not the truncated
Gaussian formula or RNN likelihood replay.

Global advantage normalization is also material but is not labeled a bug by
itself. Across the fresh run, 39 source-43--46 samples change advantage sign;
in epoch 1 all ten tail actions are negative wrist-y, while eight have positive
raw advantage but negative normalized advantage. This is direct historical
credit evidence, not a final-checkpoint backfill.

That implementation blocker is now repaired by the local frozen-rollout
normalization contract:

```text
freeze actor/critic RMS snapshot
        -> collect one rollout
        -> recompute every PPO likelihood and critic value with that snapshot
        -> finish all actor/critic optimizer passes
        -> update RMS explicitly from the rollout's raw observations
        -> next rollout uses the new RMS version
```

Normalization forward calls are pure and statistics mutation is a separate
explicit operation. The no-learning gate uses four independent GPU worlds, a
40-step horizon, 160 samples and all four PPO mini-epochs. Actor and critic
learning rates are zero. Before the first optimizer call and after the full
dry-run, the maximum `|ratio-1|` is exactly `0.0`; repeated actor outputs,
log-probabilities and critic outputs are also bitwise unchanged, and neither
RMS hash moves. A second commit-enabled gate runs two complete epochs. Epoch 1
uses normalization version 0 and commits version 1 only at its tail; epoch 2
uses version 1 and commits version 2 only at its tail. Both first-update ratios
are exactly 1, both `before` hashes equal their `frozen_before_commit` hashes,
and actor/critic weights remain bitwise unchanged. Neither gate performs
task-level training or writes a checkpoint or chunk commit.

All checkpoints trained under the former likelihood-misaligned path are now
classified by `configs/taco_pour_ppo_checkpoint_eligibility_v1.yaml` as
behavioral-audit-only. They cannot be warm-started or used as algorithm
performance baselines. Their frozen-policy diagnostics remain valid as
behavior facts, but claims about how valid PPO credit created those policies
are withdrawn.

The authorized post-fix fresh PPO starts from the hash-bound CPU endpoint-20
snapshot with newly initialized actor, critic and optimizers. It uses four GPU
training worlds, eight epochs and the unchanged 1,280-sample budget, then
transfers the actor bitwise to the CPU judge. Every epoch uses normalization
versions 0 through 7 in order; all eight first-update ratios are exactly 1,
every `before` hash equals `frozen_before_commit`, and the commit chain is
exact. It is therefore valid post-fix algorithm evidence. The result is still
a strict failure: Replay first fails at endpoint 51, while PPO first fails at
endpoint 40 (`20` validation rows including the failed row). No chunk is
committed, the checkpoint is not authorized for warm start, and cross-run GPU
performance comparison remains forbidden.

The read-only post-fix policy-extremization audit reproduces both formal CPU
traces exactly and reconstructs all 32 actor updates plus eight deferred RMS
commits from lossless patches. On final-failure probes at sources 34--39, the
optimizer updates have net signed mean movement `+27.9179` toward the seven
endpoint-40 saturated directions; 17 optimizer transitions increase the
focused saturation count and none decrease it. RMS commits have net movement
`-4.94558`, so the first large global RMS effect does not explain the final
failure directions. Later passes over one rollout reach mean exact truncated
KL `0.21234` and an outside-`[0.8,1.2]` ratio fraction of `0.8375`.

Thirty-two fixed-seed stochastic final-policy rollouts are diagnostic only:
29 reach endpoint 40, 28 pass it, but none pass the complete 40-step window.
Deterministic CPU validation remains the sole judge.

The one authorized single-variable follow-up changed only actor mini-epochs
from four to one. It used a fresh actor, critic and optimizer, kept the critic
at four mini-epochs, and retained the same four worlds, eight epochs, 1,280
samples, seed, reward, action distribution and CPU acceptance rule. All eight
first-update ratios are exactly 1. Replay again passes 30 intervals and fails
at endpoint 51. The deterministic PPO passes 28 intervals and fails at
endpoint 49, so it materially improves over the valid four-pass PPO's 19
intervals but remains below Replay and strict 40/40. No chunk is committed and
the retained compressed checkpoint is ineligible for warm start.
Because the two actors came from separate non-deterministic GPU trainings, the
19-to-28 change is the predeclared single-variable evidence classification,
not a bitwise causal ablation.

The matching read-only audit shows that PPO is no worse than Replay through
endpoint 47 and first becomes worse at endpoint 48. It restores right-hand
tool contact over endpoints 37--40 and scores 0.586106 at endpoint 40. All
eight optimizer transitions still increase the failure-focused saturation
count; their net signed movement toward the final saturated directions is
`+13.6272`, while RMS commits contribute `-2.15111`. Post-optimizer exact KL
on the fixed source-32--39 probe panel reaches `0.0397362`. Of 32 fixed-seed
stochastic rollouts, 31 pass endpoint 40 and none finish the 40-step window.
The single authorization is consumed; no rerun, warm start, sweep or further
algorithm change is currently authorized.

The subsequent endpoint-47/49 gate is read-only and bitwise reproduces both
formal CPU traces. Replacing the complete source-47 PPO residual with the
Replay-equivalent zero residual lowers endpoint-49 score by `0.0255933`, but
still fails at endpoint 49. The same replacement at source 48 changes the
controlled hand state while changing the tool free joint by only `7.90e-9` in
qpos norm; right-hand/tool contact is already absent and the endpoint-49 score
is unchanged. Neither branch survives endpoint 49, so the predeclared gate
fails. The `actor_learning_rate: 1e-4 -> 5e-5` candidate is recorded but not
authorized for training, sweeping, warm start or chunk commit.

The source-47 semantic action-subspace gate then zeros, for one control step,
right-wrist translation, right-wrist rotation, right fingers, the whole wrist,
or the whole right hand. Every branch keeps the left hand bitwise unchanged
and resumes the frozen actor. None preserves or restores a live right-hand/tool
contact at endpoint 48, and all still fail at endpoint 49. Removing wrist
translation or the whole wrist improves endpoint-49 score by about `0.024` to
`0.026`; removing fingers alone is slightly worse. Training samples at sources
46/47 that retain next-step right-tool contact have higher relative return and
advantage, but are only `4/23` and `3/20` samples and never satisfy the
thumb-plus-non-thumb bonus condition. No source-47 training candidate is
authorized; the next read-only decision moves to source-46 state entry.

The source-46 state-entry gate changes exactly one declared residual group for
one control interval from the bitwise-reproduced formal state and RNN hidden,
then resumes the frozen actor. Zeroing only `R_forearm_ty`, all wrist
translation, or the complete wrist postpones failure from endpoint 49 to
endpoint 50, despite having no live right-hand/tool contact at endpoint 48.
Zeroing the complete right hand or all 36 residuals restores index contact at
endpoint 48, but both still terminate at endpoint 49. Thus source 46 remains
tracking-controllable; it only fails the previous joint contact-plus-survival
gate. Endpoint-48 contact is neither necessary nor sufficient for the observed
short-horizon feasibility.

The source-45 gate therefore uses only the original object-tracking boundary
as its primary condition. Zeroing right-wrist translation or the complete
right wrist for one source-45 interval makes endpoint 50 feasible and moves
failure to endpoint 51; zeroing y alone, the complete right hand or all 36
residuals still fails at endpoint 50. At source 46, neither passing branch
reduces the saturated y decision: the raw y mean rises slightly and the
deterministic y action remains `+1`. The narrowest supported direction is a
right-wrist-translation temporal or state-dependent gate, not the frozen
half-LR candidate. No new training or chunk commit is authorized.

## Archived invalid performance evidence

Runs and diagnostics produced by the former off-by-one reward path are not
active evidence. They are isolated under:

```text
TRASH/historical_reward_misaligned_2026-09-25/
```

That archive must not be used to resume a boundary, warm-start an actor, or
compare task performance. Its README records the exact reason and scope. The
old files remain recoverable only as bug history.

Uncompressed duplicate checkpoint files from the observation-normalization
defect are isolated under:

```text
TRASH/ppo_obsnorm_likelihood_misaligned_2026-09-26/
```

The hash-bound compressed artifact required to reproduce read-only behavioral
audits remains in its evidence directory, but its eligibility contract marks
it fail-closed for resume and algorithm comparison.

## Other environments

The CPU-only collision and mesh audit environment remains:

```text
/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python
```

It uses Python 3.11, NumPy 1.26.4 and MuJoCo 3.12.0. Native triangle checks
require `python-fcl==0.7.0.11`; closed-solid and distance audits use the
already-installed Manifold/Open3D tools. These diagnostics do not replace the
MuJoCo-Warp 3.13 runtime used by Replay→RL.

The legacy Spider environment without the local overlay carries a different
MuJoCo/MuJoCo-Warp version and is not accepted as task-performance evidence.
Every runtime report records its backend contract and physics-contract hash so
results from the two stacks cannot be silently mixed.
