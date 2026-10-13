# Brush frame-0 raw table depth vs fixed bowl support plane

No reference/table plane was refitted. No horizontal prior, calibration, distance rejection,
signed-distance truncation, or spatial subsampling was used.

## Selection

- Manual RGB-only table regions: `6`.
- Valid raw depth points reported: `129036` / `146161` selected pixels.
- Invalid raw-depth pixels (no 3-D measurement): `17125`.
- Points rejected by distance: `0`.

## Signed distance to the fixed bowl-bottom support plane

- Normal: `[0.003529859428565972, 0.04436892570337393, 0.9990089782000677]`; offset: `0.549819918539 m`.
- All points min / P05 / median / P95 / max: `-237.206 / 0.628 / 17.997 / 41.623 / 67.479 mm`.
- Mean / standard deviation: `18.951 / 13.329 mm`.
- Negative / zero / positive: `5305 / 0 / 123731`.
- Positive fraction: `95.889%`.
- Per-region median range: `5.992` to `30.076 mm` (span `24.084 mm`).

## Per-region medians

- `A_top_left`: n=`21873`, median=`30.076 mm`, P05/P95=`11.762/49.841 mm`.
- `B_top_mid`: n=`24905`, median=`24.753 mm`, P05/P95=`5.706/44.069 mm`.
- `C_top_right`: n=`9516`, median=`24.378 mm`, P05/P95=`4.662/46.334 mm`.
- `D_left_mid`: n=`22273`, median=`19.278 mm`, P05/P95=`5.662/33.153 mm`.
- `F_lower_mid`: n=`33971`, median=`5.992 mm`, P05/P95=`-1.557/14.173 mm`.
- `G_right_mid`: n=`16498`, median=`17.483 mm`, P05/P95=`1.518/33.585 mm`.

## Interpretation

The raw table points do not coincide with the fixed bowl-bottom support plane. Most points lie on the positive-normal side by roughly centimetres, and the regional medians change systematically across the image. This is strong evidence of a geometric inconsistency between the Depth/camera chain and the object-pose/mesh chain; this audit alone does not assign the error to either side.

These are direct measurements against the previously frozen bowl geometry plane, not a replacement table estimate.
