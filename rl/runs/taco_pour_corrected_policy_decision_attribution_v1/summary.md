# Corrected policy-decision attribution

This audit freezes the corrected actor, normalization, objective, residual
bound and complete CPU state. It performs no training and cannot accept or
commit a chunk. The formal CPU PPO trace is reproduced exactly: 27/40, first
failure at endpoint 48.

## Formal CPU action selection is already deterministic

The formal validator calls the state-feasible truncated distribution's
deterministic mode:

```text
action = clip(actor_mu, state_low, state_high)
```

It does not sample an action and does not apply a tanh/squash transform.
Therefore a separate "deterministic versus current" rollout would duplicate
the same action sequence. Exploration noise cannot explain the formal
endpoint-48 failure.

For right-wrist y, the frozen actor produces:

| source endpoint | actor mu | deterministic action | residual |
|---:|---:|---:|---:|
| 40 | 0.4981 | 0.4981 | +24.90 mm |
| 41 | 0.7407 | 0.7407 | +37.03 mm |
| 42 | 0.9475 | 0.9475 | +47.37 mm |
| 43 | 1.2942 | 1.0000 | +50.00 mm |
| 44 | 1.6861 | 1.0000 | +50.00 mm |
| 45 | 1.9866 | 1.0000 | +50.00 mm |
| 46 | 2.1051 | 1.0000 | +50.00 mm |
| 47 | 2.1384 | 1.0000 | +50.00 mm |

The harmful positive-y decision is therefore present in the deterministic
policy function itself. Saturation begins at source 43 and is not caused by a
stochastic CPU sample.

## A single intervention is sufficient

Only the current source action's wrist-y component is set to zero; all later
actions again come from the unmodified frozen actor.

| intervention source | endpoint-48 score | change | bowl-y change | result |
|---:|---:|---:|---:|---:|
| 44 | 0.95594 | -0.11018 | -13.24 mm | pass at 48 |
| 45 | 0.91712 | -0.14900 | -19.43 mm | pass at 48 |
| 46 | 0.98475 | -0.08138 | -8.57 mm | pass at 48 |

Continuous suppression across 44--46 is not required to recover endpoint 48.
Any one of these three one-step changes moves the closed-loop trajectory onto
a better subsequent path. This is evidence of high-leverage local policy
decisions, not merely insufficient action magnitude.

## Contact and control transmission

The formal right-index/tool contact is absent at sources 40--41, present at
42--46 and absent again at 47. The MJWP contact force is strongly non-monotonic,
so contact must not be reduced to a binary or monotonic explanation. The
continuous fingertip-site-to-tool-origin distance decreases through source 43
and then grows from about 88.7 mm at source 44 to 112.6 mm at source 47.

A within-support one-sided probe reduces only the first wrist-y action by
`0.25` normalized units, or 12.5 mm of control-target residual. Its response is:

| source | immediate bowl-y change | endpoint-48 bowl-y change | endpoint-48 response ratio |
|---:|---:|---:|---:|
| 42 | -1.24 mm | -1.93 mm | 0.154 |
| 43 | +0.09 mm | -11.59 mm | 0.927 |
| 44 | -1.97 mm | -7.48 mm | 0.598 |
| 45 | -1.59 mm | -11.41 mm | 0.913 |
| 46 | -2.43 mm | -5.94 mm | 0.475 |
| 47 | -0.05 mm | -0.05 mm | 0.004 |

The response is non-smooth and not monotonic, so these finite differences are
not presented as a smooth Jacobian. The decisive observation is the collapse
between sources 46 and 47: source 46 still begins with index/tool contact and a
useful y response; source 47 has no right-tool contact and almost no response.
The wrist remains movable, but a correction issued only at 47 is too late.

Live contacts are taken directly from the MJWP contact/constraint buffers and
include geom names, penetration distances, normals and normal-force values.
CPU `mj_geomDistance` is deliberately not used for positive separation because
it does not provide a valid distance for this mesh--SDF pair. The reported
fingertip-site-to-tool-origin distance is only a pose surrogate, not surface
distance.

## Classification

- The formal failure is not caused by stochastic evaluation action selection.
- Increasing residual scale is contradicted by the counterfactual evidence.
- The deterministic actor mean itself asks for the harmful positive-y action
  and exceeds its support from source 43 onward.
- Contact loss is coupled to the state drift and marks loss of later control
  authority, but is not proven to be the unique root cause.
- The next read-only question should concern the actor inputs, reference-command
  timing and world/reference/object action frame that produce this deterministic
  mean. Reward, bound and exploration variance should remain unchanged.
