# Virtual object assist v1

Status: `COMPLETED_NO_STRICT_WINDOW_SUCCESS`

This local, non-paper-faithful experiment changed training dynamics with a bounded tool wrench. All official evaluations used the original unassisted dynamics from the committed endpoint-40 boundary.

- completed epoch: `250`
- assisted tail coverage at epochs 41--50: `40/40`
- official unassisted fixed evaluations: `e0=20/40, e50=19/40, e100=14/40, e150=19/40, e200=20/40, e250=19/40`
- assisted diagnostics (not success eligible): `e50@alpha=1=40/40, e150@alpha=0.333333=23/40`
- all-in physics steps: `408041`
- actor / critic optimizer steps: `250 / 1000`
- strict chunk commit: `false`
- automatic follow-on: `false`
