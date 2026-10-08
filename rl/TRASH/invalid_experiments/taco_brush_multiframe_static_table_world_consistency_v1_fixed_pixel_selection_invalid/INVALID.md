# Invalid evidence: fixed pixel polygons reused across moving-camera frames

The v1 experiment is retired and must not be used as algorithmic or geometric evidence.

Its configuration claimed that every selected RGB frame had been visually approved, but the implementation reused one set of image-space polygons across all 12 frames. The table moves in image coordinates with the egocentric camera. Consequently, some selected pixels did not depict the table (notably the purple/top-right region in frame 0 and the upper-left part of the green region in frame 208).

All v1 fitted planes, cross-frame drift values, plots, and summaries are therefore ineligible for evidence. The source data were not modified. The corrected experiment must use independently frozen polygons for every frame, produce a per-frame RGB overlay before reading Depth, and reject any shared-polygon configuration.
