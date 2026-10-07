# TACO multiframe table-depth handoff

The RGB-D timebase and spatial-registration evidence remains unchanged:

- native depth row `i` maps to annotation/RGB/camera row `i`;
- logical time is `i / 30`, not the 15-fps AVI PTS;
- metric depth is `raw_uint16 / 4000` metres;
- spatial registration passes on high-coverage rigid anchors;
- some object surfaces remain depth-observability limited.

The historical run in `runs/taco_multiframe_table_depth_estimation_v1/`
estimated a Brush plane from measured background depth after excluding official
projected hands, tool and target. Its frozen candidate was:

```text
normal = [0.0246967242, -0.0284477308, 0.9992901473]
offset = 0.5702517944 m
fit-frame offset MAD = 0.0009024847 m
```

The old report additionally selected validation points by requiring them to be
within the candidate plane's 10 mm inlier distance, then reported the selected
residual p05..p95 as approximately `[-8.17, +8.05] mm`. That procedure is
candidate-conditioned and the background region is not an independently
labelled table region. It therefore does **not** establish an independent
table uncertainty interval or absolute position accuracy. The former
`BRUSH_TABLE_HOLDOUT_MISMATCH` conclusion and the claim that the bowl bottom
was outside an independently verified uncertainty interval are withdrawn.

The bowl (`-22.316846 mm`) and brush (`-22.275507 mm`) bottom distances remain
valid geometric distances to that frozen candidate plane. They cannot by
themselves identify camera calibration as the cause, nor prove that the plane
is reality-grounded. Independent table validation is not completed and
absolute position precision is not verified.

The same estimator produced internally stable candidates for Brush, Pour and
Smear; Skim lacked a usable projected interaction ROI. Internal stability is
not independent accuracy. No candidate support contract was created and the
active target-bottom `SupportSurfaceContract` remains unchanged.

The measurement-code repair also establishes that historical Brush
post-correction penetration reports sometimes compared the original table to
a newly refitted near-horizontal plane. New audits must instead transform the
exact original plane (full normal and offset) with the saved rigid correction.
Those old severe penetration values require same-plane re-audit; they are not
to be used as proof that the underlying approximately two-centimetre mismatch
was fixed or explained.

Current status:

```text
MULTIFRAME_TABLE_PLANE_CANDIDATE_UNVALIDATED
REAL_GEOMETRIC_ERROR_CAUSE_UNRESOLVED
```

Future work must preserve the timebase findings, distinguish filtered fitting
residuals from independent validation, and must not move the table or objects
to make the holdout pass.
