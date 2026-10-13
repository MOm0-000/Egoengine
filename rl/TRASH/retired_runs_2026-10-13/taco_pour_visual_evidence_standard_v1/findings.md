# Findings

Status: **COMPLETE visual review**, bound to image manifest
`f73275c18ab78fe547045db61ce7935fb210c310046a279a6ccaaed8661ddbf6`.
This status certifies review coverage, not task success, promotion, or the parent
experiment's decision.

## Direct observations

- All endpoints 0–60 were reviewed in both fixed views through all 14 contact
  sheets. Selected full-resolution early/middle/late frames, six saved-substep
  comparisons, and all three curve pages were opened separately.
- Endpoint 0 actual states are coincident with the kinematic reference to
  numerical precision. The simulated tray is left of the bowl as expected; no
  initial relocation is visible.
- Endpoints 0–14 evolve continuously. No object teleport, floor penetration,
  or explosive motion is visible.
- The first dynamic contact/relative-pose turning point is 15–17: the saved
  target-frame tool/target translation error is 25.37 mm at endpoint 16, a
  right-index/tool contact appears at 16, and it disappears at 17.
- Reference/actual separation grows smoothly through 23–32. The two actual
  conditions separate after the declared overlap boundary.
- By 43–60, bowl tracking and bowl/tray relative-pose disagreement are visually
  obvious. OVERLAP_PLAN generally tracks the tray better late, but does not
  uniformly improve bowl rotation or the relative pose.

## Data-supported interpretations

- The apparent bowl/tray mismatch is not created at endpoint 0. The earliest
  material interval requiring close review is 15–17, not an initialization
  offset.
- Later disagreement is accumulated tracking divergence rather than a render
  discontinuity.
- Saved legacy tracking scores below 1 are neutral evidence. They do not
  certify the task relation or paper-level success.

## Hypotheses requiring new evidence

- The 15–17 contact timing change may seed later relative-pose divergence, but
  this package does not establish causality.
- Cross-column pixel displacement against REAL RGB may partly be camera-view
  mismatch because the RGB camera is time-aligned but not calibrated or
  pixel-registered to the simulator cameras.

## Earliest review intervals

| Relation | Last clearly plausible | First suspicious interval | Meaning |
|---|---:|---:|---|
| right hand–tool | 15 | 15–17 | contact/motion transition, not a declared failure |
| left hand–target | 24 | 25–27 | first clearly growing reference/actual tray separation |
| tool–target | 15 | 15–17 | first material saved relative-pose change |

## Limitations and minimum next evidence

REAL RGB is an independent uncalibrated camera. Occlusion limits fingertip
contact judgment. Substep RGB is nearest-frame only, endpoint reference is
held, and historical force vectors are not overlaid without verified
solve-time geometry.

The smallest next evidence for the early bowl/tray question is calibrated RGB
camera extrinsics or an independent synchronized 3-D pose check for endpoints
15–17. A contact-causality claim additionally needs solve-time contact geometry
aligned with the saved wrench.
