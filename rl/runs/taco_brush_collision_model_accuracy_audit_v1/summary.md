# Brush collision-model accuracy audit

- Classification: `NO_EXISTING_DROP_IN_PROXY_UNIFORMLY_FIXES_NATIVE_GEOMETRY`
- MINK optimization: `NOT RUN`
- Geometry generation / CoACD rerun: `NOT RUN`

| frame | state | object | native / proxy-min hand body | native (mm) | current proxy (mm) | visual-hull proxy (mm) | current / hull abs error (mm) | hand local over / missed | object local over / missed |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 40 | baseline | brush | right_hand_index_rota_link1 / right_hand_index_rota_link2 | -2.417 | -7.352 | -7.268 | 4.935 / 4.851 | 1.000 / 0.000 | 0.920 / 0.062 |
| 40 | baseline | bowl | left_hand_index_rota_link2 / left_hand_index_rota_link2 | 18.882 | 19.365 | 18.885 | 0.483 / 0.002 | 0.268 / 0.697 | 0.447 / 0.061 |
| 40 | local_candidate | brush | right_hand_index_rota_link2 / right_hand_index_rota_link2 | 1.161 | -0.001 | -0.641 | 1.162 / 1.802 | 0.430 / 0.508 | 0.934 / 0.055 |
| 40 | local_candidate | bowl | left_hand_index_rota_link2 / left_hand_index_rota_link2 | 19.018 | 19.555 | 19.020 | 0.537 / 0.002 | 0.268 / 0.697 | 0.443 / 0.061 |
| 80 | baseline | brush | right_hand_thumb_rota_link2 / right_hand_index_rota_link2 | -3.253 | -14.730 | -13.125 | 11.477 / 9.872 | 0.967 / 0.020 | 0.975 / 0.021 |
| 80 | baseline | bowl | left_hand_index_rota_link2 / left_hand_index_rota_link2 | -1.054 | -11.035 | -10.561 | 9.980 / 9.507 | 0.729 / 0.254 | 0.455 / 0.090 |
| 80 | local_candidate | brush | right_hand_mid_link2 / right_hand_thumb_rota_link2 | 3.710 | 4.816 | 0.000 | 1.105 / 3.710 | 0.139 / 0.850 | 0.885 / 0.090 |
| 80 | local_candidate | bowl | left_hand_index_rota_link2 / left_hand_index_rota_link1 | -1.202 | -0.001 | -1.701 | 1.201 / 0.499 | 0.400 / 0.572 | 0.475 / 0.066 |
| 120 | baseline | brush | right_hand_ring_link2 / right_hand_index_rota_link1 | -2.177 | -11.046 | -9.662 | 8.869 / 7.485 | 0.615 / 0.348 | 0.887 / 0.098 |
| 120 | baseline | bowl | left_hand_index_rota_link2 / left_hand_index_rota_link2 | -1.036 | -6.907 | -7.874 | 5.871 / 6.838 | 0.504 / 0.479 | 0.479 / 0.080 |
| 120 | local_candidate | brush | right_hand_mid_link2 / right_hand_mid_link2 | 3.949 | 5.358 | 4.670 | 1.410 / 0.722 | 0.422 / 0.553 | 0.760 / 0.221 |
| 120 | local_candidate | bowl | left_hand_index_rota_link2 / left_hand_index_rota_link2 | -0.645 | -0.000 | -1.047 | 0.645 / 0.402 | 0.438 / 0.535 | 0.504 / 0.072 |

## Existing-model decision

- Current proxy/native sign agreement: 9/12.
- Native-hand visual convex-hull/native sign agreement: 10/12.
- Visual-hull alternative has lower absolute distance error in 9/12 rows; it is not a uniform fix.
- The current proxy's minimum-distance hand body matches the native minimum body in 7/12 rows.
- Bowl 146's existing 32-part `convex_m` decomposition is already active in every query; no unused finer CoACD model was found.
- Brush 071 has 17 active convex parts but no checked-in CoACD provenance or alternate `convex_m` candidate, so it cannot be claimed or promoted as a CoACD repair.
- No collision representation is changed and no new MINK solve is authorized by this audit.

Positive signed coverage means the proxy extends across the native surface (overcoverage); negative means the proxy does not reach the native surface (missed coverage). The unchanged 50 µm native-material tolerance is used for counts.
