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
PYTHONPATH="$PWD/.env_mjwp313_overlay:$PWD/src:$PWD/external/mink/src" \
  OMP_NUM_THREADS=4 \
  /data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q
```

On 2026-09-26 this command passed **651 tests and 57 subtests**.
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
```

The first corrected PPO authorization is consumed. Replay passed the first
40-step lookahead and committed the CPU endpoint-20 boundary. In the second
lookahead Replay passed 30 intervals and PPO passed 27; neither passed 40/40,
so no new boundary was committed and formal training is closed pending a new
algorithm decision.

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
RMS hash moves. The gate performs no task-level training and writes no
checkpoint or chunk commit.

All checkpoints trained under the former likelihood-misaligned path are now
classified by `configs/taco_pour_ppo_checkpoint_eligibility_v1.yaml` as
behavioral-audit-only. They cannot be warm-started or used as algorithm
performance baselines. Their frozen-policy diagnostics remain valid as
behavior facts, but claims about how valid PPO credit created those policies
are withdrawn. A new post-fix task PPO has not been run; it requires separate
authorization, so `training_ready` remains false.

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
