# TACO source alignment — official projection v2

**Decision:** `CAMERA_DEPTH_REGISTRATION_UNRESOLVED`

The rigid-object control is dense on the tool but the minimum target valid-depth fraction is only 0.107; RGB/depth container timing metadata also disagree. Pixel-registered egocentric depth is not established, so the depth channel is prohibited from judging the released hand pose.

## Fixed evidence chain

- Projection is exclusively the pinned official TACO PyTorch3D implementation.
- Endpoint 0 is source frame 0 at 0 seconds; ±1 is diagnostic only.
- Rigid-object depth control: `CAMERA_DEPTH_REGISTRATION_UNRESOLVED`.
- Hand/depth comparison: `PROHIBITED_BY_CAMERA_DEPTH_GATE`.
- The 12-view automatic masks are secondary conflict checks, not independent truth.
- Physics, control, planning, retarget candidate generation, RL, promotion and chunk commit: all zero.

## Key numbers

- Object median-of-medians absolute depth error: 12.155 mm.
- Tool valid-depth fraction (minimum across frames): 0.999.
- Target valid-depth fraction (minimum across frames): 0.107.
- Timing diagnostic zero-offset best fraction: 0.250; offsets were not selected.
- RGB/depth container-rate mismatch observed: `True` (RGB 30/1, 6.600000 s; depth 15/1, 13.200000 s). Both decode to 198 frames and no resampling was applied.
- Allocentric label mapping: `{'right_hand': 1, 'left_hand': 2, 'tool': 3, 'target': 4}`; margin 1.380321.

## Interpretation boundary

Automatic masks were initialized from the released 3D meshes and refined by image models. Agreement cannot independently prove the 3D pose correct; systematic multi-camera disagreement can only add conflict evidence.
