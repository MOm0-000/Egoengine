# Retired Issue #14 MINK candidate attempts

These implementation-only attempts were removed rather than retained beside
the formal evidence:

- `failed_render`: MINK completed, but evidence rendering requested 720 pixels
  against the immutable 640-pixel MuJoCo framebuffer.
- `failed_glfw`: MINK completed, but the headless server had not yet selected
  EGL before importing MuJoCo.
- `superseded_proxy_only_label`: MINK and rendering completed, but the initial
  summary treated collision-proxy proximity as sufficient for a positive
  pickup-alignment label. It was superseded by the native-mesh gate in
  commit `bb0bd2d`.

All three generated the same deterministic 209-frame MINK trajectory as the
formal run. Their bulky duplicate artifacts were deleted after the final run
reproduced the trajectory and wrote the corrected classification.
