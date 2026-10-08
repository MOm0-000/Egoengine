# RGB-only per-frame table selection review

This directory freezes the visual selection used by `taco_brush_multiframe_static_table_world_consistency_v2` before any selected-frame Depth was read.

- Frames reviewed independently: `0, 19, 38, 57, 76, 95, 114, 133, 152, 171, 190, 208`.
- Each overlay was inspected at full resolution.
- Every polygon is inside the visible turquoise tabletop for that frame and avoids the visible bowl, brush, hands, sleeves, table boundary, and uncertain narrow gaps.
- No image-space polygon set is reused between frames.
- Frame 0 no longer contains the invalid old top-right/purple fixed region.
- Frame 208's upper regions were moved inward with the table and do not include the old green region's off-table upper-left area.

These overlays are selection evidence only. They do not contain or depend on Depth-plane results.
