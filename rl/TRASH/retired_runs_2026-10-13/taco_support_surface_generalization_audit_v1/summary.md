# TACO support-surface generalization audit v1

**Decision:** `TACO_RGBD_REGISTRATION_BLOCKER` at Phase C. No estimator, MINK, physics, Replay, MPC, RL, promotion, or chunk commit ran.

## Required answers

1. **Current active support plane:** the Brush scene builder passes target id 146 to `taco_project_sample_support_contract`, which takes the target frame-0 minimum along world +Z and maps it to simulator `z=0.72 m`. The infra audit and v2 retarget/static audit check or consume this same contract.
2. **Why it is sample-specific:** target 146 is the bowl, so this construction uses the bowl bottom as the source plane. It is a project sample contract, not an observation-derived estimator.
3. **RGB/depth/camera registration:** not yet trustworthy for support inference. RGB has 209 frames at 30 Hz while depth has 209 frames at 15 Hz (durations 6.967 s versus 13.933 s), so the official fps=30 decode instruction has no unique 209-row alignment here. Independently, the frozen official-projection gate failed: target minimum valid fraction is `0.212748` and target median rigid-depth error is `20.950 mm`.
4. **Did a new estimator use bowl/brush bottoms?** No. No estimator executed because Phase C failed; bowl/brush bottom values were never read as estimator inputs.
5. **Independent plane versus bowl bottom:** not evaluated; no consensus plane was permitted.
6. **Brush -1.311 mm under an independent plane:** not evaluated; retaining the old number only as historical context, not as an input or new result.
7. **Other-sample stability:** not evaluated. The local inventory contains only three complete 1920x1080/equal-count sequences; the fourth local dev4 sequence is incomplete, so Phase F would also lack the required four complete samples if reached.
8. **Eligible for scene-builder integration?** No. The active SupportSurfaceContract remains unchanged.

## Stop discipline

The registration blocker prevents background point-cloud fitting and hold-out validation. Timing offsets `-1/0/+1` are diagnostics only; no offset was selected to improve results. Existing v2 and collision artifacts were not modified.
