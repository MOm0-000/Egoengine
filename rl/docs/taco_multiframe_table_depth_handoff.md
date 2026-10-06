# TACO multiframe table-depth handoff

The RGB-D timebase and spatial-registration questions are closed and must not
be reopened as explanations for the table result:

- native depth row `i` maps to annotation/RGB/camera row `i`;
- logical time is `i / 30`, not the 15-fps AVI PTS;
- metric depth is `raw_uint16 / 4000` metres;
- spatial registration passes on high-coverage rigid anchors;
- some object surfaces remain depth-observability limited.

The evidence in `runs/taco_multiframe_table_depth_estimation_v1/` independently
estimates the Brush table from measured background depth. It excludes official
projected left hand, right hand, tool and target masks, uses no colour mask or
sample rectangle, and never reads a bowl/brush bottom during fitting.

The measured-depth plane is internally stable:

```text
normal = [0.0246967242, -0.0284477308, 0.9992901473]
offset = 0.5702517944 m
fit-frame offset MAD = 0.0009024847 m
validation table-support points = 292232
validation signed residual p05..p95 = [-0.0081655112, 0.0080524246] m
```

The same frozen algorithm finds stable planes for Brush, Pour and Smear. Skim
has empty official foreground projection in all local frames, so the required
interaction ROI cannot be formed and it is correctly classified as
`INSUFFICIENT_TABLE_DEPTH_EVIDENCE` rather than replaced with a hand-written
rectangle.

Only after the Brush consensus JSON was written and SHA256-pinned were object
bottoms read. The bowl bottom is `-22.316846 mm` and the brush bottom is
`-22.275507 mm` relative to the estimated plane. Both are outside the
independently frozen validation uncertainty. The formal result is therefore:

```text
BRUSH_TABLE_HOLDOUT_MISMATCH
```

The historical brush `-1.311 mm` value is superseded as an estimator claim; it
was not used to tune or select this plane. No candidate support contract was
created, and the active target-bottom `SupportSurfaceContract` remains
unchanged. The next task must explain the measured-depth/table versus released
object-geometry discrepancy; it must not revisit 15/30-Hz mapping or silently
move the table/object to make the holdout pass.
