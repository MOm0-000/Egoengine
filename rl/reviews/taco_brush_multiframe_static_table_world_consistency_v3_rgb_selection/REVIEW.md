# v3 RGB-only table selection review

Status: `FROZEN_BEFORE_DEPTH`

The twelve RGB frames were reviewed independently at full resolution. Every
polygon lies visibly inside the turquoise tabletop and avoids the bowl, brush,
both hands, sleeves, cast-shadow boundaries, the moving table boundary, and
uncertain narrow gaps.

Frames 19 and 38 were specifically corrected after the v2 review was found
unsafe:

- the upper-left green polygons were reduced and moved down/right, away from
  the upper-left table boundary;
- the left-mid purple polygons were reduced and moved right, leaving clear
  visible tabletop margin from the left boundary.

No Depth, camera array, fitted plane, residual, or cross-frame statistic was
consulted while defining or reviewing these polygons. The coordinates in the
v3 config are frozen before the Depth audit is allowed to run.
