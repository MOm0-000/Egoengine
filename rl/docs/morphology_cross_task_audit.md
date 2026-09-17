# MANO-XHand Morphology Cross-Task Check

The Pour diagnostic suggested scaling each wrist-relative fingertip vector by
the measured XHand/MANO finger-chain ratio. That candidate is stored only in
the Pour report and was not promoted.

The same diagnostic was run on the existing complete Brush/Brush/Bowl reference
using the same XHand MJCF and MANO models. Reports:

* Pour: [`runs/taco_pour_morphology_audit_v3/report.json`](/data_all/zzx/3.2RL/runs/taco_pour_morphology_audit_v3/report.json)
* Brush: [`runs/taco_brush_morphology_audit_v1/report.json`](/data_all/zzx/3.2RL/runs/taco_brush_morphology_audit_v1/report.json)

The neutral morphology ratios are robot properties and agree between tasks:
global chain scale is about `1.186`, while the per-finger ratios are `1.267`,
`1.150`, `1.09`, `1.156`, and `1.359` for thumb through pinky. The human
landmark ratios remain bounded over the posture strata, but that does not imply
that changing a task's target points preserves its contacts.

With the inherited full-pose local probe, Pour changed from `11.52 mm` to
`7.52 mm` mean fingertip position error under the per-finger candidate. Brush
changed from `26.18 mm` to `29.76 mm`; its mean fingertip orientation error also
remained about `51 deg`. The global candidate was worse on Brush as well
(`31.49 mm`). Wrist-position residuals can improve while fingertip/contact
geometry gets worse, so they cannot be used as the sole acceptance criterion.

This is the required counterexample against freezing a Pour-fitted morphology
mapping. The measured morphology is real and useful for diagnostics, but the
target scaling is task-dependent and remains outside Replay/RL. A formal
mapping needs cross-task contact preservation and exact-triptych visual review
before any weight search.

The next candidate is documented in
[Contact-Geometry Mapping Plan](contact_geometry_mapping_plan.md). It maps
object-local contact patches with bounded feasibility slack instead of scaling
world-space fingertip vectors; it remains read-only until cross-task validation
passes.
