# TACO Pour hand-object visual audit v1

Status: `READONLY_STATIC_AUDIT_COMPLETE`

This was an offline audit of saved artifacts. It performed no physics
integration, control interval, action sampling, actor/critic/vision forward,
optimizer update, reference regeneration or model mutation.

Full immutable server evidence:

`/data_all/zzx/3.2RL/runs/taco_pour_hand_object_visual_audit_v1`

## Result

- The prepared human object transforms and the robot reference object qpos
  agree over all 198 frames (`max_abs=9.282e-8`). The apparent early mismatch
  is therefore not a tool/target role swap or a human-to-reference object
  transform discrepancy. The RGB panel remains an independent camera view and
  is not pixel-registered to the simulation panels.
- A saved, committed Replay prefix provides actual endpoints `0--20`. Its two
  historical variants are byte-identical, and endpoint 20 matches the formal
  committed snapshot bitwise for qpos/qvel. Endpoint 0 is effectively the
  reference state; the bowl(tool)-to-tray(target) relative-position error is
  `0.000 mm` there.
- Relative drift is initially small (`1.69 mm` at endpoint 5, `4.86 mm` at 10,
  `10.56 mm` at 15), then changes sharply on transition `15->16`: `24.11 mm`
  at endpoint 16, `37.40 mm` at 17, `54.06 mm` at 18, `65.64 mm` at 19 and
  `72.67 mm` at 20. At endpoint 20 its xyz components are
  `[+46.49,+55.78,+2.92] mm` relative to the reference object pair.
- At endpoint 20, the bowl position error is `48.84 mm` and the tray position
  error is `25.28 mm`; both objects contribute to the relative-position
  discrepancy. The frozen tool-only local ellipse score rises from `0.022586`
  at endpoint 15 to `0.117437` at 16 and `0.425967` at 20.
- The clearer tail change starts at endpoint 45--46. Saved right-tool contacts
  go from one index contact at 45 to none at 46--60. Tool z error grows from
  `-6.54 mm` at 45 to `-58.34 mm` at 50 and `-109.54 mm` at 60. The formal
  tracking failure is only at endpoint 60 (`score=1.004819`).
- Saved left-target contact flags persist through endpoint 60, so this is not
  a complete loss of every hand-object relation.
- The evidence therefore has two observable stages: an early executed
  object-pair divergence beginning around `15->16`, and a later right-tool
  contact loss around `45->46`. It does not prove either transition is
  mathematically irrecoverable or establish their exact physical causes.

Important limitations: the endpoint `0--19` prefix did not save contact flags
or forces; its score is recomputed offline from saved qpos (with endpoint 20
regressed against the snapshot). There are no RGB camera extrinsics, distances
to visual meshes are unsigned, robot tip sites are not finger-pad surfaces,
and the main trajectory is a verified composite of existing saved segments
rather than a new rollout.
