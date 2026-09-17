# Contact-Geometry Mapping Plan

## What the measurements say

The MANO and XHand proportions are different, but the difference is not a
single scale that can be applied to every task. The measured XHand/MANO chain
ratios are approximately `1.27, 1.15, 1.09, 1.16, 1.38` for thumb through
pinky. They are reasonably stable across open, bent and grasp-like postures,
and the same robot ratios appear in Pour and Brush.

That stability only says that the robot has a different hand shape. It does
not say that a scaled MANO fingertip is the correct point on an object. In the
read-only probes, a Pour-fitted scale reduced mean fingertip position error from
about `11.52 mm` to `7.52 mm`, but the same candidate increased Brush/Bowl from
about `26.18 mm` to `29.76 mm`. Therefore a fixed fingertip scale is not a
cross-task contract and must not enter Replay or RL.

## Proposed mapping

The input to the mapper is synchronized MANO21 hand motion, the two object
poses and metric meshes, per-finger contact evidence (contact/free/unknown),
and the previous accepted XHand state. All contact calculations are done in the
object's local coordinates, not directly in the simulator world frame.

For every observed finger-object contact, build a small surface patch rather
than treating the MANO tip vertex as an exact robot target:

* patch centre: the MANO contact point projected onto the object mesh;
* object-frame outward normal;
* one tangent direction when the source motion gives a stable one;
* a radius/uncertainty that covers mesh thickness, MANO noise and the XHand
  fingertip-pad offset.

The XHand target is then obtained by a constrained solve, not by multiplying a
world-space fingertip vector. For the active contacts, minimize a normalized
sum of:

```text
surface position error
+ normal/approach-direction error
+ tangent-direction error (when observed)
+ wrist pose error
+ distance from the previous accepted posture
+ joint-limit and collision slack penalties
```

The physical constraints are enforced separately: object poses stay at their
GT values during planning, joint limits remain active, self/hand/floor
penetration is forbidden, and a contact finger is allowed only inside its
surface patch and a small positive distance band. Before the first contact
evidence, the existing pre-contact clearance rule remains active; unknown is
not treated as either contact or free.

## When contacts disagree

The XHand cannot always satisfy every MANO finger, wrist and object condition
simultaneously. The solve therefore uses explicit priorities:

1. keep the object pose, collision legality and joint limits;
2. keep high-confidence load-bearing contacts (for example the grasping hand
   on the tool or the supporting hand on the target);
3. keep stable contact normals and approach directions;
4. preserve wrist motion and secondary fingertips;
5. let low-confidence or redundant contacts move inside their patch.

This is implemented with bounded slack variables and a report of which
contacts had to give way. A solver success by itself is not accepted: a result
with a large slack, penetration or an unexpected contact is a failed candidate.

## How morphology is used

The measured per-finger ratios are used only as an initial prior for the
joint-space solve. They are not used to scale the final task targets. If a
constant anatomical offset is needed, it is estimated in the local hand/object
frames from several tasks and checked with leave-one-task-out validation. A
single Pour-derived offset is never frozen as a universal rule.

## Cross-task test before RL

Each candidate is generated in a separate run directory while the formal GT
and robot references remain unchanged. It must be compared with the current
baseline on every approved task, first Pour and Brush/Bowl and then the other
approved TACO profiles. The report includes:

* object-local contact position error (median and 95th percentile, in mm);
* contact-normal/approach angle error;
* contact-state precision and recall;
* wrist position/orientation error;
* self, hand-object and floor penetration;
* joint-limit margin and temporal jerk;
* task-level success and the exact source-frame prefix used.

Hard failures (illegal state, penetration, wrong object pose or missing source
frames) reject the candidate. Among legal candidates, the candidate is kept
only when the combined normalized error improves across tasks without making
any task's contact metrics worse than the baseline. The paper's published
thresholds are used where available; unpublished thresholds remain explicitly
labelled engineering gates rather than being presented as paper values.

Only after this leave-one-task-out check passes do we connect the candidate to
Replay -> residual RL. The two-block lookahead and rollback rules remain the
same, and the exact triptych renderer is used for the required visual sanity
check. Until then, all mappings are diagnostics and no weight grid is run.

## Implementation order

1. Extract object-local contact patches and confidence from the existing GT and
   surface audits.
2. Add the constrained per-frame/per-chunk mapper with bounded slack and a
   complete residual report.
3. Run it read-only on Pour and Brush/Bowl and inspect the exact triptych.
4. Validate on the remaining approved profiles with leave-one-task-out
   fitting.
5. Freeze only the surviving mapping contract, then connect it to Replay -> RL.
