# Paper-First Audit Results

> Historical early audit. Current readiness and full-horizon results are in
> `docs/rl_reproduction_status.md`.

Input: the four approved TACO sequences' original left/right hand GT, tool/target
world-pose GT, metric meshes, RGB/depth and camera parameters; the preserved brush
v4 MINK reference and isolated two-XHand/two-passive-object scene. Human GT is a
retargeting input, not PPO actions. No perception reconstruction or MPC was run.

## Execution Status

1. Input audit executed for all four approved episodes. All four have structurally
   usable world-space GT. This is not proof of physical task compatibility.
2. Existing brush alignment and mesh scaling audited against A.1/A.3. The applied
   transforms are internally consistent; the local vertical anchor is not an
   independently validated table calibration.
3. Collision-pair and native-geometry audits executed. No copy/scale/pair mismatch
   was found in the current declared model. Complete self-collision coverage is
   unresolved; no geometry or collision pairs were changed.
4. Initial overlaps and analytic translation alternatives measured. The reset
   procedure is not published, and no alternative was selected or applied.
5. Existing exact reward and two-chunk tests rerun. At that audit time, real PPO
   integration/training was not performed. The later MJWP/PPO smoke integration
   is tracked in `docs/rl_reproduction_status.md`; it does not resolve the
   physical reset or establish task SR/Step/Reward.

## Four Inputs

| Task / episode | Hand/tool/target/camera GT rows | RGB decoded frames | Depth decoded frames | Finding |
| --- | --- | --- | --- | --- |
| brush / 20230927_027 | 209 each | 209 | 209 | Structural checks pass; brush reset remains invalid/unvalidated. |
| cut / 20230917_020 | 361 each | 358 | 356 | Media-to-GT alignment unresolved; no rows trimmed. |
| skim / 20230926_004 | 400 each | 400 | 400 | Released camera maps all hand GT behind the camera. |
| smear / 20231103_071 | 90 each | 90 | 90 | Structural checks pass; no robot-scene feasibility claim yet. |

The left/right PKL wrist translations agree with the respective joint arrays;
frame keys are contiguous and one-based. Object and camera transforms pass
float32-tolerance rigidity checks. All selected object meshes are available and
scaled from centimeters once. Decoded frame counts match container counts;
matching counts alone do not prove temporal correspondence.

Original depth containers are 15 fps, while RGB/GT and resized depth are 30 fps.
Do not infer GT timing from the original depth container playback rate. Cut's
original depth is 1024x768 versus 1920x1080 RGB; it cannot simply inherit the RGB
intrinsic matrix without an established mapping. The other three original depth
videos are 1920x1080. No synchronization or depth calibration was changed.

For skim, nominal camera-space hand depth is -0.778 to -0.528 m. Inverting the
extrinsics puts the hands in front, but at 1.839 to 2.067 m; front-facing points
alone do not validate that inverse. The camera was not inverted or replaced by
the copied WiLoR/PnP proxy. This camera issue does not automatically invalidate
the world-space GT for the explicitly approved oracle-input RL setting.

Visual inspection of the existing six-frame RGB previews supports bimanual tool
and target handling in all four examples. The visible motions are brush/bowl,
spatula/tray, spatula/perforated plate, and eraser/box. The task names do not
establish a requirement to simulate material removal, liquid or compliant
bristles for C.3 object-pose scoring. No new phase boundaries or reward terms
were inferred from the sparse previews.

## Alignment And Scale

For the preserved brush v4 artifacts:

- Exported hand positions and object transforms reproduce the applied shared
  transform exactly; robot object qpos matches the transformed GT exactly.
- Maximum relative hand/object and camera/object transform discrepancy is below
  4e-16. These are algebra checks, not independent calibration measurements.
- The model table is at z=0.72 m; the first object-pair center is at x=0.6,y=0.
- Independently transformed native asset vertices and compiled MuJoCo visual
  bounds agree to below 3e-9 m, ruling out double scaling of these visual assets.
- The first-frame target-bottom vertical anchor is still a local convention.
  No depth-based realignment or per-frame world shift was applied.

## Collision Coverage

The current model has 174 explicit hand pairs and 1,967 total physical pairs.
MINK and runtime hand pairs match, and hand collision-shell geometry matches the
pinned source template. Independent joint-limit checks find zero violating rows.
These results apply only to the declared model, not a full CAD certificate.

All 102 omitted intra-hand shell pairs were enumerated: 20 are welded/adjacent
assembly candidates, 82 are nonadjacent. Of the 82, 25 penetrate somewhere in the
209-row reference. Several already overlap in the source zero configuration;
therefore blindly enabling every pair would also constrain source assembly or
oversized-shell artifacts. It is equally unjustified to discard every omitted
pair as an artifact.

At the worst frame of each penetrating nonadjacent pair, deterministic native
surface samples were tested against watertight counterpart meshes. Some palm/
proximal pairs have positive containment evidence. Many finger meshes are open,
so no signed-containment claim was made for them. A 256-point sample with no
detected overlap is not proof of nonintersection. A complete triangle-surface
cross-check remains pending: obtaining the optional python-fcl library failed
with the configured unavailable proxy and timed out on a bounded direct PyPI
check. No package was installed and no shared environment was changed.

## Initial State

| Geometry | Initial support clearance |
| --- | --- |
| Right hand collision shells, worst | -11.686 mm |
| Left hand collision shells, worst | -13.399 mm |
| Right native palm mesh | -7.186 mm |
| Left native palm mesh | -8.665 mm |
| Brush collision pieces | -2.273 mm |
| Brush native mesh | -1.311 mm |
| Bowl collision pieces | +0.000950 mm |

Thus replacing coarse hand shells alone cannot remove the initial native-mesh
table intersection. Initial hand/tool and hand/target shell clearances are
positive (33.243 and 34.739 mm respectively); initial acquisition of object
contact is a later control problem, distinct from the initial table overlap.

Translation-only alternatives were measured, not applied:

- Independent upward shifts need at least 11.686 mm for the right hand,
  13.399 mm for the left hand, and 2.273 mm for the brush to clear the current
  table plane. These shifts alter initial relative poses and do not establish
  self-collision feasibility, trajectory continuity, or successful execution.
- A common 13.399 mm upward shift preserves hand/object relative poses but lifts
  the initially supported bowl off the table. It is not a calibration repair.
- Current qvel is a finite difference of the MINK/GT reference. Neither retaining
  that velocity nor zeroing it has been established as the authors' reset rule.

Section 3.2.1 does not specify environment-constrained initial-state projection;
A.3 does not describe physical reset or settling. A new reset implementation
therefore needs an explicit, disclosed engineering decision. Unchanged GT goals,
fixed evaluation thresholds and honest failure reporting remain mandatory.

## Artifacts And Verification

- [Input report](/data_all/zzx/3.2RL/runs/paper_first_audit_v1/inputs.json)
- [Collision/initialization report](/data_all/zzx/3.2RL/runs/paper_first_audit_v1/initialization.json)
- [Input audit command](/data_all/zzx/3.2RL/scripts/audit_taco_paper_inputs.py)
- [Initialization audit command](/data_all/zzx/3.2RL/scripts/audit_taco_initialization.py)
- [Regression tests](/data_all/zzx/3.2RL/tests/test_taco_paper_audit.py)

48 local tests passed: 40 existing tests plus 8 added audit checks. This includes
the paper reward formulas, mock-backend scheduler, renderer mapping, preserved
source hashes and no-reset/no-pass diagnostic assertions. It is not a real PPO
or physical task validation. No simulation rollout or new video was generated.

The reports contain source hashes and refuse to overwrite an existing output.
The approved renderer and its gate are unchanged; training_ready remains false.
