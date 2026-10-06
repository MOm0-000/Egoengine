# TACO RGB-D timebase handoff

The active candidate depth-time contract is evidence-bound by
`runs/taco_rgbd_timebase_registration_repair_v1/`.

- The TACO paper describes an Intel RealSense L515 egocentric sensor and a
  camera/mocap system operating at 30 Hz, with 1920x1080 egocentric images.
- The official projection path indexes RGB, object pose, hand annotation and
  egocentric extrinsic with the same row index.
- A TACO maintainer documents the FFV1 depth AVI encoder with an input rate of
  15 fps, while the official README applies FFmpeg's `fps=30` filter when
  decoding. That filter duplicates/drops frames; it is not a metadata-only
  rewrite.
- TACO metric depth is `uint16 raw / 4000` metres. The old README value 1000
  was an upstream documentation error corrected by official commit
  `c06f82cec2c79b9e443f19b9f9cdb83031908559`.
- TACO maintainers acknowledge that some sequences have hardware-related
  egocentric modality mismatch and publish an available-sequence list. No
  sample-specific offset or time warp may be inferred for such data.
- Brush `20230927_027`, Pour `20230927_017`, Skim `20230926_004`, and Smear
  `20231103_071` are all present in that official available list.

For all four local release files, RGB, native depth, object poses, hand rows and
camera extrinsics have equal counts. Running the official `fps=30` filter on
the 15-fps containers doubles those counts, while exact native-frame hashes do
not show an already-expanded pair pattern. Mapping annotations by container
PTS uses only about the first half of native depth. The candidate contract is
therefore:

```text
native depth frame i <-> annotation row i
logical timestamp = i / 30 seconds
container PTS is not annotation logical time
do not expand the native file again
depth metres = raw uint16 / 4000
```

This is a local evidence-backed contract, not a newly recovered author file
format specification. It does not modify the AVI or active support surface.

Depth coverage and spatial registration must remain separate classifications.
Across uniformly selected frames, high-coverage rigid anchors in Brush, Pour
and Smear pass the frozen 20-mm median-depth consistency gate. Skim has no
high-coverage anchor in the fixed set and is classified as an observability
limitation, not evidence of camera misregistration. The current decision is
`RGBD_REGISTRATION_RESOLVED_DEPTH_OBSERVABILITY_LIMITED`.

A later, separately authorized support-surface estimator may consume this
contract. It must exclude projected foreground and must not use a bowl/target
bottom, expected table height, or a task residual as an estimator input.
