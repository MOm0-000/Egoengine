# Offline visual evidence standard

This standard applies to saved TACO rollout evidence. It is an evidence and
review contract, not an authorization to execute physics, a controller, a
policy, retargeting, or an optimizer.

## Timeline and provenance

Every endpoint image is resolved through the saved trajectory `endpoint`
field and the reference's explicit `frame_indices`/`timestamps_s`. The frame
identifier is checked against sequentially decoded RGB and independently
probed presentation timestamps. Matching lengths or nominal FPS alone are not
accepted as alignment proof. `frame_map.csv` records the reference row, source
frame, decoded frame, PTS, error, method, status, and provenance.

Real RGB remains an independent camera and is never described as
pixel-registered to a simulator camera. No mirroring, reframing, synthesized
view, or interpolated RGB is permitted. Substep pages use only saved qpos; RGB
and endpoint-only reference panels are explicitly marked as not substep
synchronized.

## Required endpoint evidence

For every saved endpoint and each fixed `top`/`oblique` view, preserve one
independently openable comparison:

1. time-aligned real RGB, subject to its distribution policy;
2. kinematic robot reference;
3. each saved actual condition.

An actual condition must say `INITIAL`, `NOT REACHED`, `NOT RECORDED`, or the
saved legacy tracking status. Tracking below the old boundary is neutral
evidence and must not be labelled task-valid. Tool/target absolute errors,
world-pair translation error, target-frame relative pose error, and tracking
score retain separate labels and units.

Every output binds its source trajectories, models, RGB, renderer revision,
configuration, cameras, and dimensions in a hash manifest. Static rendering
may set qpos and call `mj_kinematics`; calls that advance simulation time are
forbidden.

## Playback, navigation, and events

The realtime video contains each endpoint once at the verified reference
cadence. Slow playback repeats every endpoint uniformly and records each
repeat in `video_frame_map.csv`. Selective unlabelled pauses are forbidden.

Contact sheets are navigation aids, not substitutes for individual images.
The standalone HTML and Markdown index link every endpoint in both views.
Automatic events only navigate to positive error increments, speed peaks,
semantic contact transitions, declared boundaries, and saved terminations;
they never declare a new task failure.

Saved substeps may be expanded without resimulation. Contact rows are grouped
by semantic roles rather than unstable row IDs. A historical force may be
drawn only against a verified solve-time geometry state; otherwise report the
force numerically and mark the overlay unavailable.

## Review and publication

Rendering completeness and visual-review completion are independent. A new
package starts `PENDING` even if an older review exists. Completion requires a
named reviewer, the exact image-manifest hash, viewed coverage, observations,
limitations, earliest suspicious intervals, and the smallest next evidence.
Observations, data-supported interpretations, and hypotheses are distinct.

The repository intentionally excludes raw TACO RGB. Public Git previews
therefore use an explicit controlled-data placeholder in the RGB panel, while
the authorized server package retains the complete four-column comparison.
The omission is recorded in completeness and publication manifests; no frame
may be silently skipped. A clean checkout must contain decodable public
previews, contact sheets, selected keyframes, indexes, review metadata, and a
lightweight handoff archive.

## Accounting

Every run reports input hashes before and after, decoded frames, static FK and
render calls, encoded frames, file/byte counts, and elapsed time. It must state
zero physics steps, control intervals, action candidates, MINK solves,
actor/critic or vision-estimation forwards, optimizer steps, chunk commits,
and promotions.
