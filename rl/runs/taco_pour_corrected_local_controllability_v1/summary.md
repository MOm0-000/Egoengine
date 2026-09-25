# Corrected local controllability result

This diagnostic freezes the corrected PPO actor, objective, reward and complete
CPU state. It performs no training and cannot accept or commit a chunk.

The exact formal PPO trace is reproduced first. Complete 362-field physics
snapshots and matching LSTM hidden states are then captured at endpoints
44--47. Every zero-perturbation branch reproduces the formal endpoint-48 state.

## Right-wrist y result

The policy's deterministic right-wrist y residual is already at `+0.05 m`.
The first-step sweep adds `fraction * 0.05 m`; therefore fraction `-1` removes
that y residual, while `+1` extends the diagnostic target to `+0.10 m`.

Final signed bowl-y error at endpoint 48:

| source | remove first y residual (`-1`) | formal policy (`0`) | add another `+0.05 m` (`+1`) |
|---:|---:|---:|---:|
| 44 | 94.08 mm | 107.32 mm | 113.97 mm |
| 45 | 87.89 mm | 107.32 mm | 131.03 mm |
| 46 | 98.74 mm | 107.32 mm | 120.73 mm |
| 47 | 107.24 mm | 107.32 mm | 107.27 mm |

Source 45 is monotonic over the entire nine-point y grid: more positive wrist-y
residual produces more positive bowl-y error. Sources 44 and 46 are not fully
monotonic because contact dynamics are nonlinear, but both show the same local
direction around the baseline. Thus the evidence rejects the simple claim that
the current `+y` action only needs more positive authority.

The multi-step ablation is stronger. Setting only right-wrist y residual to
zero from each source onward gives:

| source | endpoint-48 score | score change | first tracking failure |
|---:|---:|---:|---:|
| 44 | 0.78347 | -0.28265 | none through 48 |
| 45 | 0.82702 | -0.23910 | none through 48 |
| 46 | 0.98475 | -0.08138 | none through 48 |
| 47 | 1.06562 | -0.00051 | 48 |

The harmful y decision is therefore made before endpoint 47. Removing it only
at endpoint 47 is too late.

## Loss of object-control transmission at endpoint 47

At source 47, the actuated wrist still responds to the full perturbation grid:

- wrist x qpos span: about 14.43 mm;
- wrist y qpos span: about 14.87 mm;
- wrist z qpos span: about 15.34 mm;
- wrist pitch qpos span: about 28.70 mrad.

Over those same branches the final bowl-y span is about 0.625 mm for wrist x,
0.078 mm for wrist y, 0.090 mm for wrist z and 0.557 mm for pitch; every
response is weak relative to the wrist motion, and none is monotonic. No
recorded right fingertip/tool contact is active in any source-47
branch. The actuator is moving, but its local ability to move the bowl has
become very weak by this point. This supports lost contact transmission, not
an inability of the wrist actuator to move, as the reason a late correction
cannot recover the bowl.

The contact mechanism is still nonlinear. For example, a `+0.25`-bound pitch
perturbation at source 44 retains right-index/tool contact through endpoint 48
and reduces the score from `1.06612` to `0.99707`. Other passing branches do not
retain that contact, so this does not prove one required contact pattern.

## Classification

- More positive right-wrist y authority is contradicted by the sweep.
- Meaningful actuator `ctrlrange` loss remains zero.
- The frozen policy's early, saturated `+y` decision is causally harmful in
  this window.
- By endpoint 47 the wrist still moves but the bowl is nearly uncontrollable
  from these wrist perturbations, coincident with absent recorded fingertip
  contact.
- The next candidate should therefore target action parameterization / the
  deterministic tail decision and earlier state entry, not increase the y
  scale. Contact-aware stabilization remains a secondary coupled mechanism.

All actions outside the formal `+-0.05` support are diagnostic-only. They are
not task-performance evidence and cannot be used for Replay--RL acceptance.
