# Pour: Common Fingertip Geometry Diagnostic

Input: all 198 frames at 30 Hz of TACO Pour `(pour in some, bowl, plate)/20230927_017`,
the preserved `taco_pour_bimanual_mano_fk_v1` human/robot references, released
MANO poses/shapes and MANO v1.2 surfaces, the isolated XHand scene/native meshes,
and the source XHand XML/URDF. All paired results below are **right, left**.

## Outcome

The point and frame definitions are now explicit and mechanically checked in
one isolated candidate. **The centimeter tracking error is not repaired.**
This candidate is a local correspondence convention, not a recovered author
calibration, accepted reference, formal initializer, or completed RL result.
No production retargeter, original scene/reference, collision geometry, contact
alias, GT position, wrist target, object pose, cost, or reset rule was changed.

The complete 198-frame run gives mean position errors of **11.499/11.191 mm**,
versus 11.810/12.266 mm under the previous measuring convention. That small
difference does not justify adopting the candidate. Pinky error remains
23.779/22.153 mm, and the constrained full-orientation targets still conflict.

## Paper And Source Evidence

Section 3.2.1, Eq. (1), specifies tracking five fingertip positions and
orientations plus wrist orientation, subject to joint and self-collision
constraints. It does not specify the MANO vertex correspondence, the XHand
local tracking point, or a nail/pad/axial-roll calibration. Appendix C defines
the trajectory refinement, rewards and feasibility criteria, not those missing
geometric conventions. The palm-to-tip teleoperation description in Appendix
A.2 is not an additional action-branch calibration specification.

The SPIDER XHand assets are clean at Git commit
`4bd2756720ea95b9da126d98da3bf414fe964849`. Its XML sites and the URDF links
named `tip` use different points. The source `retarget_config.yaml` names the
URDF links, but source `spider/preprocess/ik.py` at Git HEAD and `ik_fast.py`
also use the XML sites for IK. Thus the XML sites cannot be dismissed as
contact-only markers or a proved accidental substitution. These are upstream
SPIDER assets, not published EgoEngine calibration code. No YAML scaling or
filtering settings were imported as paper requirements.

### Human Point

The released GT tips are MANO surface vertices, in thumb/index/middle/ring/pinky
order: right `(745,317,444,556,673)`, left `(745,317,445,556,673)`. Both hands'
reconstructed tip vertices match the GT used by the reference. In neutral
shape they lie at or near the distal cap, at most 1.42 mm behind the maximum
axial support among vertices dominated by the corresponding distal joint.
They are not inferred pressure centers or per-task contact points. The distinct
left/right middle vertex IDs are retained from TACO, not silently unified.

Both released shape vectors for this episode contain ten zeros. Replacing
mean-shape calibration with episode-shape calibration therefore changes no
axes here. The tip-to-DIP vector in the distal frame still varies by up to
2.55 mm over the motion because MANO skinning is not a rigid tip marker. This
does not authorize editing GT positions to force a rigid model.

### Robot Point

| Native distal-link coordinates | Inherited XML site | URDF named endpoint |
| --- | --- | --- |
| Right nonthumb, typical | `(0,.004,-.027)` m | `(0,0,-.0425)` m |
| Left nonthumb | `(0,-.004,-.027)` m | `(0,0,-.0425)` m |
| Right thumb | `(.04,0,-.012)` m | `(.050402455,0,0)` m |

The right middle endpoint has Z=-.042107 m; left thumb X=.050401732 m.
Inherited sites are about 16 mm from the URDF endpoints. Neither choice is
exactly on the native surface: nearest-surface distances are 3.45-4.04 mm for
the inherited sites and 1.07-1.46 mm for the endpoints. These are **unsigned**
distances; open finger meshes do not support an inside/outside interpretation.

## Explicit Local Contract

The isolated candidate uses this common convention:

| Component | Human | Robot |
| --- | --- | --- |
| Origin | Released MANO tip vertex | Nearest native mesh triangle point to the URDF named endpoint |
| +Z, longitudinal | Neutral DIP-to-GT-tip direction | Distal-link origin to URDF named endpoint |
| +X, roll reference | Neutral palm normal projected perpendicular to +Z | Neutral robot palm normal projected perpendicular to +Z |
| +Y | `cross(Z,X)` | `cross(Z,X)` |
| Motion | Transport neutral axes by distal MANO FK | Transport local axes by robot FK |

Points are compared in meters in the same fixed simulator world frame. This
does not mean human and robot coordinates should numerically match in their
different local link frames. The robot projection is deterministic, with no
offset fitted to this episode or its errors. It is a geometric candidate for
a distal surface landmark, not proof of equivalent functional contact.

For each side/finger, let `C_old` and `C_new` be the old and candidate robot
anatomical frames expressed in the site frame. The target change is:

```text
R_target_new = R_target_old @ C_old @ C_new.T
R_target_new @ C_new = R_target_old @ C_old = R_human_anatomical
```

Only the robot target frame calibration changes. Human GT point positions and
MANO motion remain unchanged. All frames are right-handed and orthonormal;
rotation-step magnitudes change by at most 1.81e-16 rad. The longitudinal frame
changes by 8.427 degrees for nonthumb fingers and 16.735 degrees for thumbs.
The full coordinates and frame matrices are in `geometry_contract.json`.

Crucially, a point and a longitudinal axis do not determine axial twist. A
rounded cap's surface normal is not automatically a nail or pad normal. The
projected-palm roll reference remains **our local convention**, not author
information or an independently measured human/robot correspondence.

## Controlled Results

The same production `retarget()` runs sequentially over all frames: position
cost 10, tip orientation cost 1, wrist orientation cost 3, 8 iterations/frame,
80 at frame zero, the same joint bounds, 174 declared self pairs and fixed
previous-frame speed envelope. No weight sweep, altered first-frame rule,
physics step, MPC or RL is introduced.

Four-way measurement avoids confusing a new pose with just a new ruler:

| Measured point | Original qpos: right/left mean | Candidate qpos: right/left mean |
| --- | --- | --- |
| Inherited virtual sites | 11.810 / 12.266 mm | 18.817 / 18.661 mm |
| Candidate distal surface sites | 18.714 / 19.093 mm | 11.499 / 11.191 mm |

Each optimizer brings its own requested points closer. It does not demonstrate
that the alternative convention is the author's intended one. Candidate tip
orientation means are 20.106/21.024 degrees; wrist means are 5.321/3.964 degrees.

Middle/ring/pinky still share a fixed robot X axis. Their desired middle-pinky
X-axis separation averages 44.123/56.611 degrees. The previously derived
three-finger RMS orientation lower bound remains 18.942/24.278 degrees on
average. The local frame change does not remove this incompatibility. This
is a bound for these orientation targets, not a global position-reachability
bound and not proof that every possible calibration must fail.

In plain terms: agreeing where to put the measuring mark does not give the
robot the human hand's missing spreading/twisting motions. Requiring all five
points and all five complete orientations can demand mutually inconsistent
poses. Earlier position-only ablations show that lower position errors are
possible, but with an orientation tradeoff. We cannot call lowering orientation
weights or replacing full poses with axes an exact paper-calibration repair.

### Physical Checks

Declared joint/self-pair/speed checks pass. Coverage remains incomplete.
At frame zero the native palms are above the assumed table by 13.748/2.654 mm,
but the whole hands still extend below it by **11.286/4.486 mm**, at the thumbs.
The left pinky has 24/256 sampled points inside the closed plate, with maximum
sampled depth **1.394 mm**. Sampling is not a full-surface bound. All table
numbers are relative to the same independently uncalibrated Z=.72 m plane.

The full-trajectory collision-shell audit also records hand/object, hand/table
and omitted self-pair intersections. Shell distances are not native-surface
depths. No collision coverage problem is declared resolved by moving sites.

## Artifacts And Verification

- `scripts/audit_taco_tip_semantics.py`: read-only GT/native-mesh semantics audit.
- `scripts/diagnose_taco_tip_contract.py`: isolated point/frame candidate and comparison.
- `runs/taco_pour_tip_semantics_v1/report.json`: geometric evidence and hashes.
- `runs/taco_pour_tip_contract_v1/geometry_contract.json`: exact mapping, source/code hashes.
- `runs/taco_pour_tip_contract_v1/comparison.json`: full four-way per-frame measurements.
- `runs/taco_pour_tip_contract_v1/collision_audit.json`: full-trajectory shell checks.
- `runs/taco_pour_tip_contract_v1/initial_native_contacts.json`: native initial-state checks.
- `tests/test_taco_tip_contract.py`: geometry, frame transport, asset immutability,
  no-overwrite, unmodified contact aliases, and independent body-local FK measurements.

The complete scene XML is checked to differ only in ten canonical site
positions and the resolved compiler mesh directory. Compiled physical model
arrays and all source mesh hashes are identical. Source/reference/code hashes
are rechecked after the run. Original outputs are preserved. The diagnostic
scene/reference must not be substituted into Replay/RL merely because the
limited kinematic gate passed.

Verification on 2026-09-06: the complete test suite passed, **185 tests**,
with 10 existing MANO-pickle NumPy/SciPy deprecation warnings. The independent
body-local FK test reproduces every frame of all four reported comparisons.

The remaining missing information is the author's exact MANO/XHand point and
axial-roll mapping. This experiment clarifies and tests one convention but
does not recover that information or establish millimeter tracking.
