# Brush MANO/XHand hand-object penetration attribution

This is a read-only six-frame geometry audit. It does not modify or rerun MINK.

| frame | pair | official MANO/native object (mm) | XHand/native object (mm) | MuJoCo proxy (mm) | attribution |
|---:|---|---:|---:|---:|---|
| 0 | right/brush | 38.349012 | 32.689860 | 33.107896 | `NO_MATERIAL_PENETRATION` |
| 0 | left/bowl | 44.421213 | 25.286616 | 28.078569 | `NO_MATERIAL_PENETRATION` |
| 40 | right/brush | 0.166782 | -2.416511 | -7.351590 | `RETARGET_ADDED_PENETRATION` |
| 40 | left/bowl | 34.259927 | 18.882395 | 19.365215 | `NO_MATERIAL_PENETRATION` |
| 80 | right/brush | -2.656176 | -3.253140 | -14.730486 | `OFFICIAL_GEOMETRY_PENETRATION_RETAINED_AFTER_RETARGET` |
| 80 | left/bowl | -1.369326 | -1.054436 | -11.034761 | `OFFICIAL_GEOMETRY_PENETRATION_RETAINED_AFTER_RETARGET` |
| 120 | right/brush | -3.485528 | -2.176761 | -11.045610 | `OFFICIAL_GEOMETRY_PENETRATION_RETAINED_AFTER_RETARGET` |
| 120 | left/bowl | -1.704208 | -1.036054 | -6.907007 | `OFFICIAL_GEOMETRY_PENETRATION_RETAINED_AFTER_RETARGET` |
| 160 | right/brush | 27.498697 | 23.157931 | 23.413894 | `NO_MATERIAL_PENETRATION` |
| 160 | left/bowl | 36.968985 | 22.030559 | 24.930309 | `NO_MATERIAL_PENETRATION` |
| 208 | right/brush | 28.166251 | 20.578151 | 21.884714 | `NO_MATERIAL_PENETRATION` |
| 208 | left/bowl | 39.379444 | 25.504504 | 28.213046 | `NO_MATERIAL_PENETRATION` |

## Result

- Origin counts: `{"NO_MATERIAL_PENETRATION": 7, "OFFICIAL_GEOMETRY_PENETRATION_RETAINED_AFTER_RETARGET": 4, "RETARGET_ADDED_PENETRATION": 1}`
- Proxy/native sign agreement: `12/12` on the fixed sample.
- Proxy/native maximum absolute distance error: `11.477346 mm`.
- Penetration-depth exaggeration range: `3.042x` to `10.465x`.
- Active MINK hand-object collision limit: `ABSENT`.
- Drop-in proxy verdict: `NOT_RELIABLE_AS_DROP_IN_NATIVE_HAND_OBJECT_GATE`.

The word inherited is used only at frame/interaction-pair level. MANO and XHand do not have vertex-identical topology, so this audit does not claim one-to-one contact-point inheritance.
