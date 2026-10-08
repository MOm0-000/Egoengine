# Brush multiframe raw-Depth table planes in world coordinates

Twelve frames spanning the full video were fixed before reading their plane results. Each frame uses all valid raw-depth pixels in the RGB-approved polygons. Fits are unconstrained orthogonal SVD planes in world coordinates; there is no robust loss, RANSAC, horizontal prior, distance rejection, truncation, subsampling, bowl plane, or calibration.

## Cross-frame consistency

- Plane offset span: `464.727 mm`.
- Pairwise normal angle median / max: `36.085 / 88.723 deg`.
- Signed plane separation at the frame-0 plane centroid min / max / span: `-15.897 / 22.474 / 38.371 mm`.
- Per-frame own-plane residual median range: `7.725` to `60.684 mm`; P95 range: `23.304` to `125.296 mm`.

## Per-frame planes

| frame | points | normal | tilt to +Z (deg; descriptive) | offset (m) | own residual median/P95/max (mm) | angle to f0 (deg) | separation at f0 centroid (mm) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 25594 | `[-0.734614, 0.559478, 0.383831]` | 67.429 | 0.348231 | 19.413/83.613/510.875 | 0.000 | 0.000 |
| 19 | 14217 | `[-0.657735, 0.526186, 0.538993]` | 57.385 | 0.424853 | 33.862/96.854/300.239 | 10.116 | -3.231 |
| 38 | 33700 | `[-0.681750, 0.551716, 0.480444]` | 61.286 | 0.387343 | 41.976/101.918/590.578 | 6.329 | 6.201 |
| 57 | 56135 | `[-0.614687, 0.769043, 0.175309]` | 79.903 | 0.209436 | 57.244/102.012/784.480 | 18.358 | 9.967 |
| 76 | 52722 | `[-0.532720, 0.846181, 0.013631]` | 89.219 | 0.115720 | 46.361/125.296/326.570 | 29.542 | 2.863 |
| 95 | 52174 | `[-0.566685, 0.817036, 0.106397]` | 83.892 | 0.167718 | 56.403/109.336/166.626 | 23.901 | 7.116 |
| 114 | 55670 | `[-0.592369, 0.742591, 0.312502]` | 71.790 | 0.269150 | 60.684/102.634/144.624 | 13.934 | 22.474 |
| 133 | 56637 | `[-0.029363, 0.027595, 0.999188]` | 2.309 | 0.579206 | 7.725/23.765/549.771 | 65.132 | -15.897 |
| 152 | 55597 | `[-0.001209, 0.009467, 0.999954]` | 0.547 | 0.574344 | 7.836/23.304/540.836 | 67.046 | -15.638 |
| 171 | 47326 | `[-0.067608, 0.059416, 0.995941]` | 5.164 | 0.580446 | 9.536/27.020/537.056 | 62.278 | -11.892 |
| 190 | 42749 | `[-0.453749, 0.513151, 0.728552]` | 43.235 | 0.495874 | 40.861/83.866/773.836 | 25.833 | -1.160 |
| 208 | 41243 | `[-0.436476, 0.484679, 0.758007]` | 40.711 | 0.515908 | 35.309/75.898/1159.350 | 28.023 | -8.382 |

## Interpretation

The fitted world planes are not the same plane. Their orientations and positions vary far beyond the within-mesh bowl support-plane residuals from the preceding geometry audit. Moreover, many selected raw-depth clouds have large residuals to their own unconstrained fit, so the evidence is not merely a clean plane undergoing rigid extrinsic drift: under the mandated no-rejection contract, several frames are not tightly planar at all. This rules out cross-frame world-plane consistency but does not by itself assign the problem to depth noise, RGB/depth registration, timebase, or camera extrinsics.

Frame 0 is only the comparison reference; it is not treated as a standard-answer plane.
