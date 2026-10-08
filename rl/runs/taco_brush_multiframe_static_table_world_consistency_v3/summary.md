# Brush multiframe raw-Depth table planes in world coordinates

Twelve frames spanning the full video were independently selected in RGB and their full-resolution overlays were reviewed before reading Depth. No pixel polygon set is shared across frames. Each frame uses all valid raw-depth pixels in its own RGB-approved polygons. Fits are unconstrained orthogonal SVD planes in world coordinates; there is no robust loss, RANSAC, horizontal prior, distance rejection, truncation, subsampling, bowl plane, or calibration.

## Cross-frame consistency

- Plane offset span: `3.849 mm`.
- Pairwise normal angle median / max: `0.588 / 1.573 deg`.
- Signed plane separation at the frame-0 plane centroid min / max / span: `-2.774 / 1.273 / 4.047 mm`.
- Per-frame own-plane residual median range: `3.772` to `5.119 mm`; P95 range: `17.396` to `21.221 mm`.

## Per-frame planes

| frame | points | normal | tilt to +Z (deg; descriptive) | offset (m) | own residual median/P95/max (mm) | angle to f0 (deg) | separation at f0 centroid (mm) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 94615 | `[0.028775, -0.021056, 0.999364]` | 2.043 | 0.568361 | 5.119/17.973/92.366 | 0.000 | 0.000 |
| 19 | 43828 | `[0.032151, 0.001679, 0.999482]` | 1.845 | 0.567591 | 4.733/19.169/55.512 | 1.317 | -0.041 |
| 38 | 51077 | `[0.030149, -0.013137, 0.999459]` | 1.885 | 0.566831 | 3.772/17.880/62.730 | 0.461 | 1.273 |
| 57 | 78131 | `[0.040420, -0.018060, 0.999020]` | 2.537 | 0.569810 | 4.308/17.396/98.675 | 0.689 | -2.073 |
| 76 | 78729 | `[0.033383, -0.024726, 0.999137]` | 2.381 | 0.570189 | 4.801/19.730/599.635 | 0.338 | -1.959 |
| 95 | 75967 | `[0.039233, -0.023087, 0.998963]` | 2.609 | 0.569564 | 4.711/17.714/441.877 | 0.611 | -1.654 |
| 114 | 80446 | `[0.036343, -0.014581, 0.999233]` | 2.244 | 0.569151 | 4.893/18.381/118.803 | 0.571 | -1.299 |
| 133 | 80999 | `[0.039979, -0.024632, 0.998897]` | 2.691 | 0.570680 | 4.684/21.120/81.320 | 0.674 | -2.774 |
| 152 | 77452 | `[0.039950, -0.017943, 0.999041]` | 2.510 | 0.569338 | 4.258/19.758/97.118 | 0.665 | -1.580 |
| 171 | 74798 | `[0.031342, -0.021463, 0.999278]` | 2.177 | 0.570562 | 4.761/20.520/553.230 | 0.149 | -2.307 |
| 190 | 72233 | `[0.035839, -0.009775, 0.999310]` | 2.129 | 0.568880 | 4.727/21.221/246.341 | 0.763 | -1.136 |
| 208 | 73392 | `[0.042179, -0.009029, 0.999069]` | 2.472 | 0.567915 | 5.018/20.795/839.977 | 1.032 | -0.511 |

## Interpretation

**Direct numerical answer: the twelve fitted world planes are not identical.** Their maximum pairwise normal angle is `1.573 deg`, and their signed separation span at the frame-0 plane centroid is `4.047 mm`. No pass/fail stability tolerance was introduced after seeing the result.

This v3 run uses only the independently reviewed, conservative per-frame selections. Interpretation must be based on the numerical table above; no archived v1/v2 plane estimate is used as evidence or as a comparison baseline.

The plane-to-plane statistics describe whether the selected raw-Depth tabletop regions map to one stable world plane under the official camera parameters. They do not by themselves assign any observed inconsistency to depth noise, RGB/depth registration, timebase, or camera extrinsics.

Frame 0 is only the comparison reference; it is not treated as a standard-answer plane.
