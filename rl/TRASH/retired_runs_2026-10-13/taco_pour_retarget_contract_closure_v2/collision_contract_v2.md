# Collision contract v2

## A1 CSV recount

- Native-stricter rows: `25`.
- Rows containing `left_hand_link_visual ↔ left_thumb_rota1_visual`: `25`.
- Left hand-object findings: `0`.
- Left hand-table findings: `2` across `1` row.
- Other omitted self pairs: `0`; unknown omitted pairs: `0`.

## Palm–thumb adjudication

- Classification: `TRUE_SELF_COLLISION`.
- Legal FK grid: `77` samples; native intersections: `7`.
- Samples whose contact extends beyond the entire rota1→rota2 joint span: `1`.
- Maximum native penetration: `0.008628776 m`.
- Consequently this pair is not added to the structural-overlap allowlist.

## Recomputed state library

- States: `140` = `64` trajectory states + `76` probes.
- Unknown semantics: `0`.
- Native true-self findings missed by runtime proxy: `29`.
- States with left hand-object/table native material findings: `90`.
- Collision semantics certified: `True`.
- Runtime collision representation complete: `False`.
- Uncategorized mismatches: `0`.
- Native material findings in deliberately bad/probe states are evidence, not by themselves a semantic-contract failure.
- Structural overlap never explains hand-object or hand-table findings.
