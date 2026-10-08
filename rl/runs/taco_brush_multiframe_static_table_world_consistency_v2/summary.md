# Brush multiframe raw-Depth table planes in world coordinates

Twelve frames spanning the full video were independently selected in RGB and their full-resolution overlays were reviewed before reading Depth. No pixel polygon set is shared across frames. Each frame uses all valid raw-depth pixels in its own RGB-approved polygons. Fits are unconstrained orthogonal SVD planes in world coordinates; there is no robust loss, RANSAC, horizontal prior, distance rejection, truncation, subsampling, bowl plane, or calibration.

## Cross-frame consistency

- Plane offset span: `3.749 mm`.
- Pairwise normal angle median / max: `0.670 / 6.020 deg`.
- Signed plane separation at the frame-0 plane centroid min / max / span: `-4.911 / 0.000 / 4.911 mm`.
- Per-frame own-plane residual median range: `4.258` to `11.472 mm`; P95 range: `17.396` to `23.144 mm`.

## Per-frame planes

| frame | points | normal | tilt to +Z (deg; descriptive) | offset (m) | own residual median/P95/max (mm) | angle to f0 (deg) | separation at f0 centroid (mm) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 94615 | `[0.028775, -0.021056, 0.999364]` | 2.043 | 0.568361 | 5.119/17.973/92.366 | 0.000 | 0.000 |
| 19 | 60426 | `[0.008976, 0.015102, 0.999846]` | 1.007 | 0.569293 | 5.917/19.564/980.690 | 2.362 | -1.351 |
| 38 | 74569 | `[-0.030573, 0.053164, 0.998118]` | 3.516 | 0.571665 | 11.472/23.144/2232.808 | 5.447 | -4.911 |
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

**Direct answer under the frozen all-point contract: the twelve Depth-derived world planes are not one stable numerical plane.** Ten of twelve frames (including frame 0) are within about 1.04 degrees of the frame-0 normal, while frames 19 and 38 deviate by 2.36 and 5.45 degrees. All frame-0-centroid separations remain within 4.92 mm.

The corrected per-frame selections do not reproduce the large drift reported by the invalid fixed-polygon v1 experiment. Most fitted world planes form a much tighter cluster, but the twelve raw-Depth fits are still not one stable numerical plane: frames 19 and especially 38 have larger normal deviations, and the all-point clouds have sizeable residuals to their own unconstrained fits. Under the mandated no-rejection contract, this is evidence of remaining cross-frame inconsistency and non-planar/raw-depth contamination, but it does not by itself assign the cause to depth noise, RGB/depth registration, timebase, or camera extrinsics.

Frame 0 is only the comparison reference; it is not treated as a standard-answer plane.
