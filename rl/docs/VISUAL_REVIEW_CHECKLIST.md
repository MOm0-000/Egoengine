# Visual review checklist

Use this small checklist alongside numerical acceptance; it is not a paper metric and does not change reward or termination.

- For every new reference, initialization, scene, or action mapping, render and inspect an untrained short prefix before starting a long run.
- At fixed milestones, render the complete saved prefix from consistent cameras. Diagnose failures from the beginning, not only the last frame.
- Before committing a boundary, require both the numerical contract and an actually inspected relation/contact view. If the latter is absent, do not call task quality validated.
- If several methods fail after the same boundary, review the source prefix instead of assuming the committed state is sound.
- Always retain both objects and their relative-pose metric. A tool-only score must not hide the target in the evidence or UI.
- Expand labels such as `VALID_OFFLINE_SCORE`: state the exact old numerical criterion and never imply stable grasp, task equivalence, or downstream viability.

## Early-contact origin review (`taco_pour_early_contact_origin_v1`)

- [x] Original RGB endpoints 0, 10, 14, 15, 16 and 20 inspected alongside
  human GT, MINK reference and Replay actual; RGB is marked non-registered.
- [x] Every frame of right thumb/index endpoints 12--20 and left support
  endpoints 0--15 inspected through fixed-camera contact sheets.
- [x] RR/AR/RA/AA panels at endpoints 14--16 use one camera/scale and are
  labelled as offline, non-executable geometry combinations.
- [x] Bowl, tray and bowl--tray relative-position curves retained; the early
  tray drift is not hidden by the historical tool-only score.
- [x] Target-fit sites, visual-triangle distances, queryable collision-pair
  distances and saved solver contact rows remain separate observables.
- [x] Contact timeline includes hand--object, hand--floor and object--floor
  roles; `left_hand:other` / `right_hand:other` are reported rather than
  silently assigned.
- [x] Visual review does not claim force closure, collision truth from visual
  overlap, dynamic feasibility of a static combination, or task success.
