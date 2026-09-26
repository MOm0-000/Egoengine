# Pour Replay→RL current status

Only reward-aligned evidence is active.

- Source state `t` receives command `ref[t+1]` and produces physical state
  `t+1`.
- Reward and termination compare state `t+1` with object reference `ref[t+1]`.
- The next observation uses goal `ref[t+2]`.
- CPU MuJoCo-Warp is the sole acceptance backend; GPU is training-only.
- Strict acceptance remains 40/40 intervals. Diagnostic runs cannot commit a
  chunk.

Replay committed the accepted CPU endpoint-20 boundary after passing the first
lookahead. From that boundary it passes 30 intervals and first fails at outcome
endpoint 51.

The active local action contract is a state-feasible truncated Gaussian with
frozen-per-rollout observation normalization. Its likelihood gate proves that
old/new ratios are exactly one before the first optimizer update, critic values
are reproducible under the same transform, and RMS changes only after an epoch.
Historical checkpoints produced by the former reward-index or normalization
bugs are behavioral-audit-only and may not be resumed.

The first valid post-normalization PPO used four actor mini-epochs and passed
19 intervals. Its read-only update audit showed optimizer-driven deterministic
policy extremization rather than RMS-commit drift. A single authorized fresh
experiment therefore changed only actor mini-epochs from four to one. The
single-pass run retained four worlds, eight epochs, 1,280 samples, seed 0,
critic mini-epochs four, learning rate `1e-4`, reward, observation, action
distribution and CPU acceptance.

The single-pass result is:

```text
Replay: 30 successful intervals; failure endpoint 51
PPO:    28 successful intervals; failure endpoint 49
```

No chunk was committed. All eight actor updates start with an exact likelihood
ratio of one. Every optimizer transition still moves the fixed failure-focused
probes toward greater saturation, while RMS commits have the opposite net
effect. The largest one-update fixed-probe exact KL is `0.0397362`. Reducing
actor reuse is therefore a supported contributor, but is not a complete fix.
Because GPU training is not bitwise deterministic, the historical 19-to-28
change is the predeclared evidence classification rather than a bitwise causal
ablation.

## Endpoint 47--49 pretraining gate

Before authorizing the next candidate, a frozen CPU audit aligned Replay and
PPO at sources 46--48 and replaced exactly one complete PPO residual with the
Replay-equivalent zero residual at source 47 or 48. Each branch then resumed
the unchanged deterministic actor. The formal Replay and PPO traces and both
restored PPO suffixes were reproduced exactly before interpreting results.

```text
source 47 zero residual:
  endpoint-49 score change = -0.0255933
  first failure             = endpoint 49

source 48 zero residual:
  endpoint-49 score change = 0
  first failure             = endpoint 49
```

At source 47, weakening one action helps but not enough to postpone failure.
At source 48, zeroing an action with normalized L2 norm `3.4805` changes the
controlled-hand qpos at endpoint 49 by L2 `0.0276354`, but changes the tool
free-joint qpos by only `7.90e-9`; right-hand/tool contact is already absent.
The intervention is therefore real, but too late to affect the bowl.

The predeclared gate required both interventions to lower endpoint-49 score
and at least one to survive endpoint 49. It fails both the source-48 improvement
condition and the joint survival condition.

## Frozen next candidate

The single-variable local candidate remains recorded as:

```text
actor_mini_epochs:    1      (unchanged)
actor_learning_rate:  1e-4 -> 5e-5
all other settings:   unchanged
```

The half learning rate was selected before the gate from the observed
single-update KL and a disclosed local quadratic step-size approximation. It is
not an EgoEngine author-recovered parameter. Because the endpoint gate failed,
fresh training is not authorized. No warm start, seed sweep, fallback learning
rate sweep, diagnostic chunk commit or automatic `2.5e-5` follow-up is allowed.

## Source-47 semantic action-subspace gate

The next frozen audit kept every non-selected action component equal to the
formal policy and, for one source-47 interval only, zeroed one of five declared
right-hand semantic groups: translation, rotation, fingers, the complete wrist,
or the complete right hand. The left-hand action remained bitwise unchanged.
Each branch immediately resumed the same deterministic policy.

No branch retains a live right-hand/tool contact at endpoint 48 and every branch
still terminates at endpoint 49:

```text
branch                                      score@49
zero right-wrist translation                1.090214
zero right-wrist rotation                   1.112277
zero right fingers                          1.115082
zero complete right wrist                   1.088325
zero complete right hand                    1.088700
formal PPO                                  1.114293
```

Removing translation or the whole wrist reduces the score by about `0.024` to
`0.026`, but does not restore contact or feasibility. Removing fingers alone is
slightly worse, so full-action suppression had indeed mixed beneficial and
harmful components; nevertheless none supplies an admissible source-47 repair.

Across the saved eight-epoch training data, next-step right-tool contact occurs
in only `4/23` source-46 samples and `3/20` source-47 samples. In both endpoint
groups these samples have higher mean return, raw advantage and normalized
advantage than no-contact samples. They are index-only or middle-only contacts,
so none meets the paper-style thumb-plus-non-thumb contact-bonus condition.
This is endpoint-level training evidence, not proof that the exact final CPU
state was visited, and it does not authorize a reward change.

The active blocker is now:

> Source 47 is already too late for every declared semantic subspace repair:
> none preserves object transmission or survives endpoint 49. The next
> read-only attribution must move to source 46 and examine state entry before
> any new training candidate can be authorized. The half-LR candidate remains
> blocked.

## Source-46 state-entry gate

The source-46 audit starts from the exact formal PPO physics snapshot and RNN
hidden. It changes one declared action group for one control interval, then
returns to the unchanged deterministic actor. Formal Replay, formal PPO and
the restored source-46 PPO suffix are reproduced exactly before interpreting
any branch.

```text
branch                                contact@48  score@49  first failure
zero R_forearm_ty only                no          0.958052  endpoint 50
zero right-wrist translation          no          0.987501  endpoint 50
zero complete right wrist             no          0.973379  endpoint 50
zero complete right hand              yes         1.029588  endpoint 49
zero all 36 residuals                 yes         1.044594  endpoint 49
formal PPO                            no          1.114293  endpoint 49
```

Endpoint-47 contact is deliberately not a success condition: formal Replay
has no right-hand/tool contact there and reacquires index contact at endpoint
48. The actual gate requires a live endpoint-48 contact, survival through
endpoint 49, and a lower endpoint-49 score than formal PPO.

The result exposes a tradeoff rather than an admissible repair under that
joint gate. Source 46 is not too late to affect task feasibility: suppressing
wrist motion makes endpoint 49 feasible and postpones failure to endpoint 50,
but does not restore live transmission at endpoint 48. Suppressing the
complete right hand or all 36 residuals restores index contact, but
position/rotation tracking still crosses the local ellipse at endpoint 49.
Therefore endpoint-48 contact is neither necessary nor sufficient for this
short-horizon tracking feasibility. Endpoint-47 comparisons report
position, angle, velocity and fingertip distances separately; they do not mix
metres and radians into a synthetic success distance, and no unreliable
positive mesh gap is computed.

All five branches fail the earlier contact-plus-feasibility joint condition,
not all forms of task control. No reward, action scale, learning rate or
training setting changes; no chunk is accepted or committed. The
`actor_learning_rate=5e-5` candidate remains blocked, and the next read-only
state-entry attribution moves to source 45.

## Source-45 state-entry gate

The source-45 audit repeats exactly the same five one-step interventions and
then resumes the frozen actor. Its primary gate is deliberately task-centric:
endpoint 49 and endpoint 50 must both remain inside the existing object
tracking boundary. Contact over endpoints 46--50 is retained as a detailed
mechanistic diagnostic, but does not decide pass or fail.

```text
branch                                score@50  survives 50  first failure
zero R_forearm_ty only                1.059187  no           endpoint 50
zero right-wrist translation          0.993756  yes          endpoint 51
zero complete right wrist             0.986163  yes          endpoint 51
zero complete right hand              1.034220  no           endpoint 50
zero all 36 residuals                 1.038889  no           endpoint 50
```

Right-wrist translation is the narrowest passing subspace. Zeroing only the y
component is insufficient, while additionally zeroing wrist rotation is not
required. The translation-only branch has no live right-hand/tool contact at
endpoints 46--50 and still passes endpoint 50, further confirming that contact
is not a short-horizon success criterion.

The improvement is primarily a state-entry effect rather than self-correction
of the previously saturated y decision. At source 46, the translation-only
branch changes the full deterministic action by L2 `0.29738`, including wrist
x/z and fingers, but `R_forearm_ty` remains clipped at `+1`; its raw mean rises
from `1.54656` to `1.60884` instead of falling. The complete-wrist branch shows
the same qualitative result: y remains clipped at `+1` and its raw mean rises
slightly to `1.55688`.

Both passing branches subsequently fail at endpoint 51, so this is not a
40/40 success and no chunk is committed. The half-LR candidate is no longer
the selected next direction. The evidence now points to a separately designed
single-variable right-wrist-translation temporal or state-dependent gating
candidate; its exact rule and promotion contract remain unresolved, and no
new training is authorized yet.

## Source-45 suppression / source-50 refinement gate

The follow-up audit fixes `zero right-wrist translation` at source 45 as the
validated prefix, reproduces its six saved endpoint arrays bitwise, and then
captures the exact source-50 physics state plus pre-forward RNN hidden. It also
reproduces formal Replay, formal PPO and the restored source-50 suffix before
interpreting any counterfactual.

The source-45 prefix and Replay both cross the tracking boundary at endpoint
51, but they are not in the same measured physical basin. At that endpoint the
tool origins differ by `32.90 mm`; Replay's squared ellipse contributions are
`0.6442` position and `0.4055` rotation, whereas the prefix has `0.9304`
position and `0.1088` rotation. Thus the matching failure endpoint means only
that both hit the same formal boundary, not that the prefix has literally
restored the Replay trajectory or failure mechanism.

From the exact prefix source-50 state, every predeclared one-step suppression
crosses endpoint 51 with a score below both the prefix baseline (`1.019423`)
and same-run Replay (`1.024568`):

```text
branch                                score@51  first failure
zero R_forearm_ty only                0.994728  endpoint 52
zero right-wrist translation          0.992012  endpoint 53
zero complete right wrist             0.992012  endpoint 53
zero complete right hand              0.992012  endpoint 53
zero all 36 residuals                 0.992012  endpoint 53
```

The narrowest passing intervention is the single `R_forearm_ty` component.
This establishes a two-stage counterfactual path that exceeds Replay's
endpoint-51 boundary: source-45 translation suppression improves entry, then
source-50 y suppression crosses the boundary. It does **not** establish that
the learned deterministic actor performs refinement; both improvements are
time-local suppressions of its output, and even the best branches still fail
by endpoint 53 rather than completing 40/40.

Consequently no training or chunk commit is authorized. The next blocker is
to define one auditable temporal/state-dependent wrist-translation candidate
without turning the two discovered timestamps into an ad-hoc rule. The
historical `actor_learning_rate=5e-5` candidate remains blocked.

## Full-window binary translation oracle

The first non-time-hardcoded candidate is a read-only one-step simulator
oracle. At every source from endpoint 20 onward, the frozen actor is forwarded
exactly once. The complete PPO action (`ON`) and the same action with only the
right-wrist translation residual zeroed (`OFF`) are each simulated from the
same complete physics snapshot with the same post-forward RNN hidden. The
selector uses only next-endpoint tracking termination and score: prefer the
sole survivor, otherwise the lower score, with exact ties retaining `ON`.
Contact, source index and any manually chosen threshold are excluded.

The oracle extends deterministic feasibility from Replay's `30/40` and formal
PPO's `28/40` to `36/40`, failing at endpoint 57. It automatically selects
`OFF` at both source 45 and source 50, confirming that the earlier hand-picked
interventions belong to one state-dependent pattern rather than two isolated
timestamps.

The pattern is not sparse. The oracle selects `OFF` at 19 of 37 evaluated
sources:

```text
27, 31, 35, 36, 37, 38, 42,
44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55
```

In particular, every source from 44 through 55 selects `OFF`. Thus the current
PPO translation branch is harmful over a broad tail region, not merely at
source 45 and 50, even though `ON` remains useful at other earlier sources.

At source 56 both candidates terminate at endpoint 57 with the exactly equal
score `1.00463891`. ON/OFF changes the wrist world position by `6.33 mm` and
hand qpos by L2 `0.00744`, but changes the bowl free-joint qpos only by about
`1.77e-8` and neither branch has right-hand/tool contact. The new failure is
therefore not a remaining choice between translation ON and OFF: translation
has lost task-scale one-step authority over the bowl at that state.

This oracle is a local engineering shield, not an EgoEngine component or a
deployable policy. It does not complete 40/40, so no learned gate, PPO run,
chunk acceptance or commit is authorized. The next read-only blocker is
endpoint-57 failure attribution; the half-LR candidate remains blocked.

## Final-OFF reversal gate

The endpoint-57 follow-up tests whether the one-step oracle sacrificed future
controllability when it selected OFF at sources 52--55. Each branch starts
from the exact saved baseline physics snapshot and pre-forward RNN hidden,
forces exactly that source back to the complete PPO translation-ON action for
one control interval, then resumes the unchanged one-step oracle. The original
36/40 oracle decision arrays are reproduced bitwise before any branch runs.

```text
forced ON source  successful intervals  first failure
52                34                    endpoint 55
53                34                    endpoint 55
54                35                    endpoint 56
55                36                    endpoint 57
```

No reversal survives endpoint 57. The earlier reversals are actively worse,
not delayed-benefit choices hidden by the one-step score. Source 55 is more
subtle: its immediate ON score is only `0.00040954` worse than OFF, and that
changed state reaches source 56 with a measurable but small bowl response. At
source 56, ON versus OFF changes bowl position by `0.171 mm` and orientation
by `0.00440 rad`; the ON outcome creates a pinky--bowl contact while OFF has
none. The old full free-joint qpos L2 mixes metres with quaternion components
and is therefore retained only as a legacy diagnostic, never interpreted as a
metric distance. Both candidates terminate at endpoint 57, with scores
`1.004050` and `1.003057` respectively.

Therefore the predeclared primary gate does not support the claim that a final
greedy OFF decision caused the endpoint-57 failure. It would also be wrong to
retain the stronger baseline description that source 56 categorically lacks
influence. At the same time, this small response is not evidence of useful
task-scale authority. The binary translation subspace cannot turn the changed
state into feasibility. The next read-only direction is semantic source-56
action attribution from that source-55-reversal state. Learned-gate training,
the half-LR PPO candidate, task acceptance and chunk commit remain blocked.

Superseded or invalid evidence remains isolated under `TRASH/` and is not part
of the active decision chain.
