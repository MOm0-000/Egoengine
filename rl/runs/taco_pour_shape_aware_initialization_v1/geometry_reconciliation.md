# Endpoint-0 left ring/pinky–tray geometry reconciliation

## Exact classification

The accepted endpoint-0 state is classified as
`NEAR_CONTACT_NO_MATERIAL_PENETRATION`.

- The four declared left ring/pinky visual-link checks against the watertight,
  concave tray visual mesh found no triangle-surface crossing and no sampled
  material containment deeper than the declared `50 µm` reporting threshold.
- The active runtime collision representation has zero contacts after
  `mj_forward` at endpoint 0.
- Its minimum declared collision distance is
  `2.0000004044155566e-06 m` (approximately `2 µm`), so the state is extremely
  close to the boundary even though material penetration is not certified.
- The independent regression set distinguishes separated geometry, `25 µm`
  near-contact, `1 mm` material penetration, and an open/non-watertight tray;
  the last case fails closed as `UNKNOWN_NOT_CERTIFIED`.

No convex hull or top-view silhouette was used to make this classification.
The exact visual meshes and model-native collision geometry are both recorded
in `geometry_reconciliation.json`.

## Why the old gate passed while the render looked wrong

The earlier initialization gate answered a narrower question: whether the
accepted state was materially legal under its declared collision contracts.
It did not certify that the intermediate ring and pinky phalanges reproduced
the human-hand shape.

The retargeting targets constrain wrist and fingertip placement, while the
posture-preserving initialization can leave proximal joints too straight. At
the open tray edge/cavity, a top view then makes the fingers appear to pass
through the tray even though the exact three-dimensional meshes do not cross
the tray material. The two observations are therefore compatible:

1. the old state passes the material-legality gate; and
2. its ring/pinky articulation is visibly unlike the human reference and is
   only about `2 µm` from the active collision boundary.

## Sole shape-aware candidate

Exactly one deterministic four-joint SLSQP candidate was evaluated. It changed
only the two ring and two pinky joints, improved the declared shape objective
from `0.5787481649` to `0.5530038648`, and left the right hand and object state
bit-identical. It was rejected by the static gate because:

- pinky fingertip error worsened from `0.0392808566 m` to `0.0410375354 m`; and
- the full existing endpoint-0 legality gate failed a declared hand–environment
  pair at the frozen numerical boundary.

The resulting classification is `STATIC_GATE_FAILED_NO_PHYSICS`. In accordance
with the frozen contract, no second fit, tolerance adjustment, physics rollout,
promotion, RL run, planner continuation, or chunk commit was performed.
