# Pour Input-Logic Investigation

Historical diagnosis of `runs/taco_pour_bimanual_gt_v1/`. The orientation
defect described below has since been repaired in the generator and tested
in a separate `runs/taco_pour_bimanual_mano_fk_v1/` reference. See
[the correction and remaining-error report](pour_mano_fk_correction.md).
This document's measurements and statements that production was untouched
describe the earlier audit, not the current code. The old artifacts are kept.
The original uint16 Pour depth was acquired later and the table statement in
section 3 below is now superseded by
[the raw-depth audit](/data_all/zzx/3.2RL/docs/pour_raw_depth_table_audit.md).

Input: original Pour `20230927_017` MANO pose/shape PKLs, all 198 released
21-joint rows, native object meshes and poses, the preserved human/robot
references, and the current isolated XHand scene. Nothing is retargeted,
projected, reset, trained, rendered, or changed in those inputs in this audit.

## Outcome And Correction

The earlier 33-group numerical revalidation did **not** exclude upstream
conceptual errors. It established reproducible measurements on the saved
reference, not correct construction of that reference. This investigation
identifies a real orientation-target defect and isolates the stage at which
the initial robot palms move below the assumed table. Do not characterize
all of these effects as inevitable human/robot morphology mismatch.

## 1. Confirmed Orientation-Target Defect

`taco_bimanual.py:33` projects the palm normal onto the plane perpendicular to
the DIP-to-tip direction and normalizes the result to construct a full frame.
`prepare()` applies it to each finger at line 148. When the distal bone passes
near parallel to the palm normal, the projected axis can reverse even while
the physical finger rotates smoothly. Its absolute degeneracy check does not
protect against this near-singular orientation construction.

The full PKLs already contain MANO local rotations. Reconstructing MANO with
the existing `smplx` helper reproduces all released joints on both hands to
less than 7.60e-8 m. Thus the alternative rotation evidence is tied to the
actual episode, not to an invented hand trajectory or a guessed hand order.

| Hand/finger | Original PKL keys | Stored target rotation step | MANO distal rotation step |
| --- | --- | ---: | ---: |
| Right middle | 00031 -> 00032 | 152.53 deg | 6.13 deg |
| Right pinky | 00047 -> 00048 | 159.11 deg | 10.08 deg |
| Right pinky | 00072 -> 00073 | 163.15 deg | 6.56 deg |
| Left middle | 00103 -> 00104 | 160.87 deg | 3.94 deg |
| Left middle | 00132 -> 00133 | 93.29 deg | 9.62 deg |

For the left middle example, the independently observed DIP-to-tip direction
changes by 3.99 deg. The maximum MANO distal step anywhere in the episode is
19.17 deg on the right and 13.61 deg on the left. All stored fingertip
orientation-valid flags nevertheless remain true and IK applies their full
orientation costs. A unit counterexample reproduces an approximately 180 deg
constructed-frame flip under a smooth 2 deg finger rotation.

This angular-step comparison is invariant to any fixed MANO-to-robot axis
calibration; an unknown constant offset cannot explain it away. The 90 deg
event-selection level is a diagnostic display filter, not a paper task
threshold or an adopted new orientation gate. This finding invalidates the
use of this reference as a clean test of correctly supplied fingertip poses.
It does **not** establish that those later jumps caused row-0 penetrations.

## 2. Initial Palm Penetration Is Introduced During Retargeting

The same native palm vertices were evaluated at the input wrist target and
the saved robot wrist pose. The intermediate columns below are a rigid-mesh
height decomposition, not proposed candidate states or a legal initializer.

| Height term relative to current table | Left | Right |
| --- | ---: | ---: |
| Native robot palm at input wrist position/orientation | +12.134 mm | +9.188 mm |
| Wrist translation contribution after IK | -19.288 mm | -18.196 mm |
| Additional wrist rotation contribution | -0.592 mm | -2.091 mm |
| Saved robot palm clearance | -7.746 mm | -11.099 mm |

At row 0, IK also moves the wrists back along simulator Y by 32.65/33.62 mm
(left/right). The achieved five-fingertip centroid residuals are below
0.008 mm componentwise. These observations are consistent with fitting
different-sized finger chains using a freely translating wrist; they do not
look like a fixed 10 mm mesh-origin arithmetic error.

The active objective tracks fingertip poses and wrist orientation, with zero
wrist-position cost (`taco_bimanual.py:187`). Its collision constraints cover
declared hand/hand pairs only (`:194`), not table or object avoidance. A result
that reduces fingertip loss can therefore place the palm below the table.
This explains why the measured error appears at this stage; it does not prove
the current orientation mapping, local weights, or attained IK optimum are
the correct reproduction choices.

Importantly, paper Eq. (1) also specifies wrist **orientation**, not wrist
position, and joint/self-collision constraints. Automatically adding a strong
wrist-position penalty is therefore not a demonstrated paper-derived fix.
Nor does this experiment establish that copying the human wrist position is
a feasible or generalizable formal reset.

## 3. Table Alignment Is Still An Unverified Physical Assumption

The applied rule is: assume source +Z up, take the minimum Z of the native
plate at row 0 (0.549062 m), and translate that point onto an infinite
horizontal table at 0.72 m (`taco_bimanual.py:114`).

Before any robot retargeting, the reconstructed MANO surface already has
vertices below that assumed plane by 1.095 mm on the left and 4.034 mm on the
right. The MANO palm regions themselves are above it by 12.896/10.491 mm.
All 21 joint centers/tip landmarks being above a plane does not prove the
finite-thickness hand surface is above it.

This establishes incompatibility between **annotation geometry and the
assumed table**, not that a real person penetrated a real table. Sources still
to separate include tabletop alignment/footprint, object reconstruction,
annotation fitting and the difference between MANO skin and the real hand.
Camera projection/common-transform agreement does not measure the physical
table plane. At the time of this historical audit, the local Pour bundle only
had resized depth, whose pixels are not metric values. The original uint16
source was subsequently acquired and decoded with `raw_uint16 / 4000.0`; its
current conclusions are in the raw-depth audit linked at the top of this file.

## 4. Native Hand/Plate Evidence Is Not Just A Shell Issue

At row 0 all 778 reconstructed left MANO vertices are outside the closed native
plate; the closest tested vertex is about 2.185 mm away. The preserved robot
reference, however, has native finger-surface samples inside the plate, up to
1.509 mm as measured in the earlier native audit. Do not attribute that robot
finding directly to overlapping source-hand GT. Vertex checks do not certify
that no triangle interiors intersect, so no complete human/plate clearance
certificate is claimed.

The 32-part CoACD caps and omitted self pairs are separate representation
issues. CoACD concavity values are not meter-valued errors, and a capped
decomposition cannot directly explain native palm/table intersection because
that calculation does not use either object's collision pieces. Neither issue
is resolved or automatically reclassified in this investigation.

## Paper Boundary And Next Work

Section 3.2.2 explicitly says retargeting/replay can fail under embodiment and
contact mismatch; Table 2 reports TACO Replay SR 0.17. The paper's not listing
these millimeter values is not evidence that its initial references were all
collision-free. Conversely, that statement cannot excuse artificial target
orientation jumps in this implementation.

Section 3.2.1 requires fingertip poses, but does not prescribe the current
palm-projection proxy. Appendix A.1/A.3 specifies the 0.6 m offset, 0.72 m
table and approximate alignment, not the first-plate-minimum rule or a
physical reset. The following is a proposed implementation order, not new
paper text and not changes already applied:

1. Replace the ill-conditioned fingertip-orientation proxy for this GT route
   with available MANO rotational FK, using explicit, independently checked
   fixed robot-frame calibration. Do not hide the defect by smoothing jumps
   or declaring all proxy orientations valid. Preserve the old reference.
2. Generate a separately named MINK reference and repeat the same native
   geometry diagnostics without adjusting its first-frame pose. This tests
   the actual collision impact of correcting the upstream target.
3. Independently resolve table alignment and finite support geometry; do not
   change heights merely to eliminate measured penetration. Return remaining
   calibration uncertainty explicitly before selecting an initializer.
4. Reassess collision coverage and formal initialization on the corrected
   input. Do not promote v3/v4, add unapproved posture compensation, or train
   RL against the current defective orientation reference.

## Verification

All 198 rows were checked for MANO joint reproduction, wrist Euler seed/FK
agreement, palm-height decomposition, and orientation-step evidence. The
maximum Euler seed rotation error is 1.08e-15 rad, and wrist rotational step
differences from MANO are below 4.76e-5 deg. No hand-order, metric-unit,
wrist-origin, or Euler-axis defect was found in these checked paths. This is
not an exhaustive proof that all coordinate/calibration code is correct.

Seven focused tests were added; the full suite passes 126 tests. These tests
confirm the diagnosis, including reproduction of the known defect, not that
the production target generator has been fixed. The production code and
original reference artifacts remain untouched.

The machine-readable final report is
`runs/taco_pour_input_logic_v1/final_report.json`; `report.json` is the earlier
diagnostic pass before adding the native left-human/plate comparison.
Reproduction command:

```bash
PYTHONPATH=src:external/mink/src OMP_NUM_THREADS=4 \
  /data_all/zzx/deximit_isolated/env-py311-cu118/bin/python \
  scripts/audit_taco_pour_input_logic.py --output <new-report-path.json>
```
