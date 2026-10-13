# Brush bowl contact-ring planes across 12 table-contact frames

This geometry-only audit applies each selected frame's official bowl pose to the same 216 mesh vertices previously validated as the bowl's annular lower support region. It does not read Depth, estimate a table, run calibration, or consume the active support plane as geometry input.

The planes are contact-implied table planes: they describe where the table would have to be if the official pose places the full bowl ring in contact. Mesh and pose alone do not independently prove physical contact.

## Cross-frame comparison

- Pairwise normal angle median / max: `0.132168 / 0.274688 deg`.
- Maximum normal angle to frame 0: `0.179525 deg`.
- Ring-centroid signed distance to the frame-0 plane min / max / span: `-0.124858 / 0.646700 / 0.771559 mm`.
- Ring-centroid signed distance to the 12-frame consensus plane min / max / span: `-0.364714 / 0.337346 / 0.702060 mm`.
- Plane offset span: `0.932173 mm`.
- Early six-frame within-phase centroid span / max pairwise normal angle: `0.124858 mm / 0.120227 deg`.
- Late six-frame within-phase centroid span / max pairwise normal angle: `0.046910 mm / 0.091608 deg`.
- Late-minus-early mean ring-plane height along the frame-0 normal: `0.697819 mm`; cross-phase normal-angle median: `0.194650 deg`.

**Interpretation:** the twelve contact-implied planes are numerically very close, with two internally tighter temporal clusters. They are not bitwise identical: the late cluster is about 0.698 mm above the early cluster along the frame-0 normal and has a roughly 0.195-degree median cross-phase normal difference.

## Per-frame planes

| frame | normal | offset (m) | angle to f0 (deg) | ring centroid to f0 plane (mm) | ring centroid to consensus (mm) | ring residual median/P95/max (mm) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `[0.00352986, 0.04436893, 0.99900898]` | 0.549819919 | 0.000000 | 0.000000 | -0.239065 | 0.067261/0.157130/0.272399 |
| 2 | `[0.00403925, 0.04366937, 0.99903787]` | 0.549693756 | 0.049609 | -0.124858 | -0.364714 | 0.067261/0.157130/0.272399 |
| 4 | `[0.00430100, 0.04241912, 0.99909065]` | 0.549771687 | 0.120227 | -0.091213 | -0.330739 | 0.067261/0.157130/0.272399 |
| 6 | `[0.00439415, 0.04276039, 0.99907569]` | 0.549775926 | 0.104693 | -0.062851 | -0.302786 | 0.067261/0.157130/0.272399 |
| 8 | `[0.00414632, 0.04240431, 0.99909193]` | 0.549782025 | 0.118071 | -0.092729 | -0.332698 | 0.067261/0.157130/0.272399 |
| 9 | `[0.00434931, 0.04332998, 0.99905135]` | 0.549738916 | 0.075854 | -0.074443 | -0.314146 | 0.067261/0.157130/0.272399 |
| 152 | `[0.00113077, 0.04401656, 0.99903016]` | 0.550625930 | 0.138938 | 0.618039 | 0.307158 | 0.067261/0.157130/0.272399 |
| 163 | `[0.00131382, 0.04484401, 0.99899314]` | 0.550573062 | 0.129858 | 0.609089 | 0.299926 | 0.067261/0.157130/0.272399 |
| 174 | `[0.00196641, 0.04511686, 0.99897978]` | 0.550535396 | 0.099316 | 0.628856 | 0.319845 | 0.067261/0.157130/0.272399 |
| 185 | `[0.00213478, 0.04504003, 0.99898290]` | 0.550535539 | 0.088713 | 0.638344 | 0.329072 | 0.067261/0.157130/0.272399 |
| 196 | `[0.00058993, 0.04545176, 0.99896636]` | 0.550593262 | 0.179525 | 0.599790 | 0.290800 | 0.067261/0.157130/0.272399 |
| 208 | `[0.00115791, 0.04501695, 0.99898555]` | 0.550615500 | 0.140890 | 0.646700 | 0.337346 | 0.067261/0.157130/0.272399 |

No equality tolerance is introduced after seeing the result. The numerical spans above are the audit outcome; interpretation must retain the contact assumption and cannot be presented as a Depth-derived table measurement.
