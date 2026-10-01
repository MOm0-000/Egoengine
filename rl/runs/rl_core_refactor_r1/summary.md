# RL core refactor R1 verification

Status: `STRUCTURAL_REFACTOR_VERIFIED`

- offline actor parity: `True`
- six closed-loop anchors: `True`
- two-epoch collector: `True`
- complete one-epoch optimizer smoke: `True` (`4` critic + `1` actor update)
- control intervals: `720`
- training enabled: `false`
- chunk commit enabled: `false`

The saved isolation fixture starts after the historical external-critic passes,
so offline bitwise update parity is actor-only. The separate real epoch smoke
demonstrates that the direct critic path executes, but does not mislabel it as
historical critic parity.
