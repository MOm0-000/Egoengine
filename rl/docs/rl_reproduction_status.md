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

The active blocker is now:

> The source-47 weaker action gives only partial improvement, while source 48
> is already physically decoupled from the bowl. The current evidence does not
> support weaker local residual as a sufficient mitigation of the endpoint-49
> divergence, so the half-LR candidate remains frozen but blocked.

Superseded or invalid evidence remains isolated under `TRASH/` and is not part
of the active decision chain.
