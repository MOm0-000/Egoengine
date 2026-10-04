# Endpoint-0 source RGB reconciliation

- Classification: `RGB_3D_SOURCE_ALIGNMENT_UNRESOLVED`.
- Endpoint 0 maps to RGB/3D/camera row 0 at `0.000000 s`.
- All RGB, hand-3D, camera, and reference streams have `198` rows at `30 fps`.
- Released extrinsics are used exactly as `T_camera_world`; no inverse, offset, rotation, or frame shift was fitted.
- All projected wrist/ring/pinky points for frames [0, 14, 15, 16, 17, 20] lie inside the 1920×1080 image: `True`.
- Pixel reprojection error is deliberately `null`: the release has no independent 2D landmark truth.
- Overlays are diagnostic evidence, not a numeric source-alignment certificate.
