# TACO Pour hand-object visual audit v1

Status: `READONLY_STATIC_AUDIT_COMPLETE`

This was an offline audit of saved artifacts at source commit
`3635de5ddfa575780f1a4a9c30dfafd33a350f9d`. It performed no physics
integration, control interval, action sampling, actor/critic/vision forward,
optimizer update, reference regeneration or model mutation.

Full immutable server evidence:

`/data_all/zzx/3.2RL/runs/taco_pour_hand_object_visual_audit_v1`

## Result

- RGB shows a right-hand bowl-rim pinch/support grasp and a left-hand
  tray-edge/underside support. Prepared human GT and the MINK kinematic
  reference preserve those broad hand/object regions. Across endpoints 20--60,
  human-target to reference fingertip error medians are `11.8 mm` (right) and
  `10.1 mm` (left). Exact digit placement differs, especially for pinkies.
- The first recorded actual state, endpoint 20, already differs from the
  same-endpoint reference: tool position error is `48.84 mm` and the mean
  right fingertip marker difference is `30.5 mm`. Actual endpoints 0--19 are
  absent, so the first divergence cannot be located earlier.
- The clearer tail change starts at endpoint 45--46. Saved right-tool contacts
  go from one index contact at 45 to none at 46--60. Tool z error grows from
  `-6.54 mm` at 45 to `-58.34 mm` at 50 and `-109.54 mm` at 60. The formal
  tracking failure is only at endpoint 60 (`score=1.004819`).
- Saved left-target contact flags persist through endpoint 60, so this is not
  a complete loss of every hand-object relation.
- The evidence supports an accumulated right-hand/bowl discrepancy before the
  formal tracking crossing. It does not prove that endpoint 46 is
  mathematically irrecoverable or that contact loss alone caused failure.

Important limitations: no RGB camera extrinsics, no historical contact force,
unsigned visual-mesh distances only, robot tip sites are not finger-pad
surfaces, and the main trajectory is a verified composite of existing saved
segments rather than a new rollout.
