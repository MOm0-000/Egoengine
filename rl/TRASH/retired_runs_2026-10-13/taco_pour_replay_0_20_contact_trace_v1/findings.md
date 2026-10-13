# Replay 0→20 contact trace v1

## Outcome

The concern visible in the earlier rendering is confirmed: the simulated bowl/tool and tray/target do not preserve the reference pair relation. This is not a reference-transform mismatch. The bitwise-reproduced physical Replay itself drifts.

This diagnostic does **not** modify the scene, reference, controls, reward, action mapping, or physics. It does not train, commit a chunk, or claim a repair.

## Replay identity and evidence level

- All 21 endpoint `qpos`, `qvel`, and `ctrl` arrays from plain Replay A are bitwise equal to the historical committed prefix.
- Observer-on Replay B is bitwise equal to A at every endpoint, and its complete s20 is bitwise equal to A across all fields.
- Against historical full s20, 361/362 fields are bitwise equal. The sole difference is `contact.geomcollisionid` in inactive slots after `nacon=52`; every active row is equal. Installed MJWarp 3.13 allocates that field with `wp.empty` (its source contains `TODO(team): set values`). The mismatch is recorded rather than hidden; it is undefined padding, not a contact or solver difference.
- No historical full s0 existed. The complete s0 used here was reconstructed with the exact historical path from the accepted initialization state, then serialized before Replay. Therefore the evidence is a full-state reconstruction anchored by endpoint and s20 identity—not a claim that a historical s0 file was restored.
- Historical runs did not store substep contact forces. The force trace here is newly measured on the endpoint-bitwise-reproduced trajectory.

See `replay_parity.json`, `input_manifest.json`, and `attempts/attempt_1/` for the fail-closed first attempt and the one authorized reconstruction fix.

## What is directly measured

### The object-pair error begins before 15→16

There is no new success threshold here. Numerically, the relation starts departing at endpoint 1. The target/tray accumulates most of the early displacement while the tool/bowl remains near a roughly 2 mm position error:

| endpoint | tool error | target error | tool-target relative error |
|---:|---:|---:|---:|
| 5 | 1.993 mm | 2.530 mm | 1.687 mm |
| 10 | 1.956 mm | 4.960 mm | 4.863 mm |
| 14 | 1.984 mm | 6.828 mm | 6.940 mm |
| 15 | 1.938 mm | 9.669 mm | 10.560 mm |
| 16 | 12.003 mm | 13.886 mm | 24.111 mm |
| 20 | 48.835 mm | 25.285 mm | 72.675 mm |

At endpoint 20 the pair-error vector is `[+46.493, +55.781, +2.919] mm`. Thus the visible error is not solely the bowl moving: both objects deviate, with the bowl dominating after endpoint 15.

### The sharp dynamics turn starts late in 14→15

The first recorded right-hand→tool contact is global physics substep 148 (`source14`, substep 7), not at endpoint 16. All 160 right-hand/tool contact rows in endpoints 0→20 are right-index/tool contacts; there is no thumb/tool contact in this prefix.

- Substeps 141–147: the tool remains floor-supported. Its accumulated contact impulse is approximately `[0.000002, 0.000000, +0.053820] N·s`, almost purely vertical support.
- Substeps 148–150: right-index/tool contact appears and, together with floor interaction, delivers about `[+0.033060, +0.021607, +0.050167] N·s` to the tool. Tool velocity changes from approximately zero at substep 147 to `[0.1483, 0.0927, 0.1154] m/s` after substep 150.
- Substeps 151–153: index/tool interaction continues and adds about `[+0.031452, +0.027008, +0.027846] N·s`.
- At substep 154 (`15→16`, substep 3), right-index/tool contact disappears. From 154 through 159 the tool has no hand, floor, target, or other recorded contact. Its x/y velocity remains near `0.284/0.206 m/s`, while z velocity decreases by about `0.03278 m/s` per 3.33 ms step—consistent with ballistic gravity during the contact-free interval.
- At substep 160 the tool contacts the floor again, receiving about `[-0.031292, -0.019378, +0.050671] N·s`.

The target simultaneously undergoes intermittent left ring/pinky and floor contacts with large, varying impulses. No bowl↔tray contact, right-hand↔tray contact, or left-hand↔bowl contact was recorded in this prefix. The relative-pose error is therefore produced by two separately driven object motions, not by a direct bowl–tray collision.

The reference control norm does not show a unique new spike at 15→16: total control-row changes are larger at several earlier transitions, and the mean actuator-force norm falls from about 111 N-equivalent at source 14 to 97.5 at source 15. The direct evidence points more strongly to a contact-mode transition than to a single abrupt command discontinuity.

## Evidence-supported explanation

The principal supported explanation is:

> The right index begins pushing the bowl/tool late in 14→15, gives it substantial +x/+y momentum, and loses transmission at 15→16 substep 3. The bowl then travels without contact for six substeps before hitting the floor again. At the same time, the tray/target is being moved by intermittent left-hand and floor interactions. These two motions compound into the 10.56→24.11 mm pair-error jump.

This is a time-correlated physical explanation from the actual saved solve outputs and state changes. It does **not** prove that friction, collision geometry, the reference, or the low-level controller is the unique root cause; that would require a separately authorized intervention.

## Visual review

I inspected both fixed-camera endpoint videos for all endpoints 0…20, the endpoint sheets, both camera views for all eleven 15→16 states, and the collision/visual side-by-side frames for endpoints 0, 10, and 12…20.

- The actual and reference object pair are visually aligned at endpoint 0.
- A small relative drift is present before 15; it becomes unambiguous by 15 and opens sharply at 16.
- The top view confirms that the divergence is primarily in the horizontal plane, consistent with the measured +x/+y pair-error components.
- The 15→16 solve-aligned sequence shows the bowl continuing away from the right index after the early substeps; the original collision-geometries view does not reveal a direct bowl–tray collision.
- Contact points/force arrows can overlap and very small forces are hard to see at the chosen scale. The numerical CSV/NPZ, not arrow length, is authoritative.

Key views:

- `review_sheets/endpoints_0_20_top_sheet.png`
- `review_sheets/substeps_15_16_top_sheet.png`
- `review_sheets/collision_12_20_oblique_sheet.png`
- `curves/object_errors.png`
- `curves/substep_contacts_motion.png`
- `index.html`

## Status labels

- **historical_tracking_criterion:** below the old tool-only normalized-ellipse boundary through endpoint 20 (`score=0.425967` at 20).
- **replay_identity:** established at all endpoints; full-state evidence has the explicitly adjudicated inactive-padding exception described above; B is fully bitwise equal to A.
- **task_relation_review:** concern confirmed. The bowl–tray relation is not preserved.
- **downstream_viability:** not established. The old committed s20 remains historical evidence but must not automatically be treated as a reliable task-quality start state.

Why the old gate did not reject this: it tracked only the tool against a very broad local position/rotation ellipse. The 48.8 mm endpoint-20 tool error remains below its 120 mm position scale, the target is omitted, and the 72.7 mm object-pair error is not part of that criterion.
