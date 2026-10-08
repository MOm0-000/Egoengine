# Brush first visual divergence v1

## Result

There are two distinct first-error boundaries:

- **Human RGB vs MINK reference:** a semantic hand-pose/morphology mismatch is already visible at frame 0. This is a reference/retarget mismatch, not a physics-execution error. Exact pixel agreement is not expected between MANO/human hands and XHand, so this finding is qualitative.
- **MINK reference vs controlled Replay:** endpoint 0 is bitwise-identical in every reported hand/object group. The first exact execution divergence is endpoint 1. It becomes unambiguously visible in the registered comparison by endpoint 3.

Early exact Replay deviations are:

| endpoint | right wrist translation RMS | right wrist rotation RMS | right finger RMS | brush position / rotation | bowl position / rotation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 mm | 0.000 deg | 0.000 deg | 0.000 mm / 0.000 deg | 0.000 mm / 0.000 deg |
| 1 | 4.130 mm | 4.085 deg | 4.516 deg | 2.454 mm / 3.158 deg | 4.428 mm / 0.666 deg |
| 2 | 5.651 mm | 6.864 deg | 8.577 deg | 6.239 mm / 9.139 deg | 4.940 mm / 2.460 deg |
| 3 | 17.335 mm | 7.377 deg | 12.301 deg | 5.724 mm / 8.331 deg | 4.618 mm / 2.617 deg |

Therefore the later grasp/contact failures are downstream. The current evidence orders the defects as:

1. the XHand reference is already an imperfect semantic match to the human hands at frame 0;
2. physics/control adds a new error immediately on transition 0 to 1;
3. that execution error is plainly visible by endpoint 3 and grows before the previously studied contact-failure interval.

The official brush and bowl meshes visibly align with the RGB objects in the registered panels. The simulator/world translation was derived independently from both official object pose streams over all frames; its maximum translation residual is `3.89e-15 m`. This validates the camera registration path, but is not a pixel-mask accuracy metric for the hands.

## Method and limits

Synchronized read-only comparison of human RGB, the current MINK reference, and the controlled physics Replay. The registered panels use the official per-frame egocentric camera; the fixed top/front panels remain semantic views. No MINK optimization, physics stepping, or training was run. See `frame_metrics.csv` for exact MINK-to-Replay state differences and `first_divergence.json` for the reviewed boundary classification.
