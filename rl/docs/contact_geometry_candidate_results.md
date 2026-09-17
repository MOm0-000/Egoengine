# Contact-Geometry Mapping Candidate Results

## Scope

This is a read-only diagnostic on the existing bimanual TACO references. It
does not change either formal reference and it does not run Replay -> RL.
The candidate projects each MANO fingertip target onto the held object's
physical collision surface, then applies a bounded robot-only correction using
the projected point and inward surface normal.

The references already contain collision violations. Therefore this run uses
preserve_baseline_violations=True: a trial may inherit the incoming violation,
but the explicit self/floor/object collision family used by the solver must not
become worse than that frame's baseline. This is a comparison mode, not a
physics-validity waiver.

Contact evidence is defined as a human fingertip target within 5 mm of the
object collision surface. The contact proxy counts an XHand fingertip as
physically contacting when its measured hand-object gap is at most 1 mm.
These are engineering labels, not TACO tactile ground truth.

## Results

All lengths are mean values; position errors are in mm and angles in radians.
The arrow is baseline -> candidate.

| Task / hand | fingertip target | contact patch | approach angle | wrist position | wrist orientation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pour / right | 14.924 -> 14.936 | 19.548 -> 19.568 | 1.3254 -> 1.3245 | 48.362 -> 48.362 | 0.10736 -> 0.10722 |
| Pour / left | 19.411 -> 19.411 | 17.626 -> 17.626 | 1.3376 -> 1.3376 | 56.942 -> 56.942 | 0.23721 -> 0.23721 |
| Brush / right | 32.202 -> 32.212 | 31.698 -> 31.647 | 1.4694 -> 1.4617 | 51.836 -> 51.835 | 0.56087 -> 0.56093 |
| Brush / left | 32.235 -> 32.234 | 55.611 -> 55.611 | 1.4130 -> 1.4130 | 67.254 -> 67.254 | 0.38311 -> 0.38311 |

The full-trajectory contact proxy changed as follows:

| Task | precision | recall |
| --- | ---: | ---: |
| Pour | 0.31365 -> 0.31324 | 0.87546 -> 0.87546 |
| Brush | 0.36053 -> 0.36148 | 0.51311 -> 0.51311 |

The candidate correction was nonzero only for a subset of the right-hand
frames. Its maximum absolute qpos change was 0.02239 in Pour and 0.08366 in
Brush; qpos contains both metres and radians, so these are reported as mixed
units rather than incorrectly converted to one angle unit. The left hand was
rejected or unchanged for the useful contact objectives.

## Collision and trajectory checks

The object qpos was unchanged bit-for-bit. Joint-limit violations remained
zero. The explicit collision families checked by the solver were never made
worse than their corresponding baseline frame.

The independent full audit exposes the remaining coverage problem. The
non-adjacent hand-shell family is not in the source MuJoCo pair list:

| Task | non-adjacent-shell mean delta | worst-frame delta | frames worse than baseline |
| --- | ---: | ---: | ---: |
| Pour | -0.0113 mm | -0.0419 mm | 56 |
| Brush | -0.0094 mm | -0.0419 mm | 47 |

The right-hand correction also increased the mean actuated-qpos acceleration
norm from 130.93 to 131.47 in Pour and from 143.61 to 144.65 in Brush.
These values are small relative to the existing trajectory variation, but they
are not improvements.

## Decision

This first object-local contact candidate is not better overall:

* Pour is effectively unchanged or slightly worse. The small right-hand angle
  improvement is outweighed by a 0.020 mm patch-error increase and a small
  precision drop.
* Brush has a small right-hand improvement (0.051 mm patch error and
  0.0077 rad approach angle), but fingertip error and temporal smoothness do
  not improve, and the left hand is unchanged.
* The omitted collision family can still worsen, so the result is not a legal
  seed for Replay -> RL.

The mapping idea remains useful as a diagnostic, but it should not be frozen
or connected to RL yet. The next required change is complete collision
coverage in the candidate safety check, followed by a new comparison; only a
candidate that improves across tasks while satisfying that full check can
advance.
