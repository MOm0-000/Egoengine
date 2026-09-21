# Pour Sample Status

> Historical sample-acquisition checkpoint. Its old `training_ready=false`
> statement is superseded; current status is in `docs/rl_reproduction_status.md`.

Input: `(pour in some, bowl, plate)/20230927_017`, with 198 rows of left/right
hand GT, bowl/plate pose GT and camera transforms, 198 original RGB frames and
198 original uint16 depth frames. RGB/GT are 30 Hz, approximately 6.6 seconds. Human GT
will feed MINK; it is not an executable robot reference.

## Acquisition And Selection

The user requested Pour/Bowl/Plate as the next priority and authorized acquiring
a new sample. This supersedes the earlier smear priority and four-only acquisition
restriction for this task. Existing brush/cut/skim/smear data and results remain
unchanged; other task processing is deferred.

There are 27 matching metadata entries, of which 15 are marked complete with good
calibration. The first in sequence-ID order, `20230927_017`, was selected before
testing any rollout or measuring its rotation. This is a disclosed local input
selection, not a recovery of the authors' episode ID.

Required RGB, hand GT, object GT, meshes and camera files already existed in local
release ZIPs. Thirteen files (39,817,482 bytes) were copied into an independent
bundle under `/data_all/zzx/3.2RL/data/taco_v1/pour_bowl_plate/`. There are no links
back to the source projects. Each copied member was CRC-checked and SHA-256 hashed;
the two independent hand-joint archives also agree exactly. No network download
or source-project modification was needed.

The original `egocentric_depth.avi` has now been acquired separately from the
official HuggingFace release. It is FFV1 `gray16le`, 1920x1080, and decodes all
198 frames without error. Its SHA-256 is recorded in
`depth_original/original_depth_manifest.json`; conversion uses
`depth_m = raw_uint16 / 4000.0`. The obsolete lossy resized copy was removed.
The resulting table/registration audit is documented in
[the raw-depth report](/data_all/zzx/3.2RL/docs/pour_raw_depth_table_audit.md).

## Input Findings

- Both hand PKLs have 198 contiguous source-frame IDs and wrist translations
  exactly equal to the corresponding GT joint array. No hands were synthesized.
- Both object pose arrays and the camera transform array have 198 valid rigid
  transforms. All hand GT is in front of the nominal camera and within its image.
  These checks do not prove pixel-perfect calibration or temporal alignment.
- Tool `022` is the bowl; target `135` is the plate/tray. Both released meshes are
  watertight after normal mesh loading, with dimensions about 11.8x11.7x7.0 cm
  and 31.8x18.4x3.7 cm respectively. The acquisition itself produced no collision
  decomposition; the later scene milestone below now has independent parts.
- Raw RGB frames 0, 79 and 150 show the initial layout, right-hand bowl lift/tilt
  with left-hand plate support, and the returned objects. No robot video was
  rendered and no rendering script was added or changed.
- Bowl maximum rotation relative to its initial pose is 0.700781 rad; maximum
  height increase is 0.167669 m. This is not the paper's approximately 1.13 rad
  example, and it must not be described as the exact author demonstration.
- Under the existing diagnostic first-target-bottom alignment, the bowl starts
  1.488 mm above the table. The plate starts at the table but goes up to 1.230 mm
  below it in later GT frames. This is a reference/support diagnostic, not a new
  success criterion or a reason to alter GT. The later robot audit found initial
  hand/table and hand/target penetrations; no reset was repaired or accepted.

## Protocol

Appendix C.2 (PDF page 22) gives the Pour/Bowl/Plate task example a 0.12 m position
threshold and 1.5 rad rotation threshold. They are recorded in the active protocol
as task-level example values. Tracking weights lambda_p/lambda_R and the mapping
to the combined boundary C remain unspecified; the example values do not resolve
those omissions. No coefficients were inferred, and training_ready remains false.

The approved bimanual, two-passive-object architecture, Replay -> residual PPO,
20-step chunks, two-chunk lookahead, tool-only and tool+target reporting, and exact
copied renderer remain unchanged. The 13.40 mm hand/table penetration measured
for brush is historical evidence, not a measurement of the new Pour scene.

The next scene milestone is now executed: an isolated Pour scene, complete
198-frame MINK reference, collision coverage and initialization audits exist.
See [Pour scene feasibility](/data_all/zzx/3.2RL/docs/pour_scene_feasibility.md).
The declared kinematic constraints pass, but the unchanged baseline's initial
native geometry penetrates the table/plate. A later user-authorized hand-only
diagnostic produced a separate declared-feasible candidate; see
[the diagnostic](/data_all/zzx/3.2RL/docs/pour_initial_hand_diagnostic.md).
No candidate was adopted as a reset; no GT correction, physical rollout or RL
training has occurred. No brush object parts, reference or reset shifts were used.

## Verification

The acquisition milestone passed 53 tests. The later scene milestone has 64
passing local tests, adding isolated assets, exact object scaling/inertias,
collision-pair coverage, full GT preservation, native-contact diagnostics and
the existing renderer's mapping. These are implementation checks, not physics
or RL task success. The subsequent initial-hand diagnostic milestone passed
74 tests while preserving the original inputs and baseline.
The later v4 first-frame diagnostic removes one verified native thumb overlap
and compares initial velocity/control inputs without selecting a reset or
executing physics. See [the follow-up](/data_all/zzx/3.2RL/docs/pour_thumb_and_reset_inputs.md).

## Files

- [Raw human video](/data_all/zzx/3.2RL/data/taco_v1/pour_bowl_plate/rgb/taco_pour_bowl_plate_20230927_017.mp4)
- [Selection record](/data_all/zzx/3.2RL/data/taco_v1/pour_bowl_plate/selection.json)
- [Acquisition manifest](/data_all/zzx/3.2RL/data/taco_v1/pour_bowl_plate/acquisition_manifest.json)
- [Input audit](/data_all/zzx/3.2RL/data/taco_v1/pour_bowl_plate/input_audit.json)
- [Original depth manifest](/data_all/zzx/3.2RL/data/taco_v1/pour_bowl_plate/depth_original/original_depth_manifest.json)
- [Raw-depth table audit](/data_all/zzx/3.2RL/docs/pour_raw_depth_table_audit.md)
- [Reproduction command](/data_all/zzx/3.2RL/scripts/prepare_taco_pour_sample.py)
- [Active protocol](/data_all/zzx/3.2RL/configs/replay_rl_protocol.yaml)
