# Visual review checklist

Use this small checklist alongside numerical acceptance; it is not a paper metric and does not change reward or termination.

- For every new reference, initialization, scene, or action mapping, render and inspect an untrained short prefix before starting a long run.
- At fixed milestones, render the complete saved prefix from consistent cameras. Diagnose failures from the beginning, not only the last frame.
- Before committing a boundary, require both the numerical contract and an actually inspected relation/contact view. If the latter is absent, do not call task quality validated.
- If several methods fail after the same boundary, review the source prefix instead of assuming the committed state is sound.
- Always retain both objects and their relative-pose metric. A tool-only score must not hide the target in the evidence or UI.
- Expand labels such as `VALID_OFFLINE_SCORE`: state the exact old numerical criterion and never imply stable grasp, task equivalence, or downstream viability.
