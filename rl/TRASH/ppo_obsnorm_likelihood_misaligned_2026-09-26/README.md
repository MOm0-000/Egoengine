# PPO checkpoints with mismatched observation normalization

These uncompressed checkpoint duplicates were removed from the active run
directories on 2026-09-26. They were trained while PPO rollout likelihoods and
recomputed likelihoods used different observation-normalization states.

They may be used only to reproduce frozen behavioral audits. They must not be
used for warm start, task-performance comparison or post-fix algorithm claims.
The formal fail-closed classification and hashes are in:

`configs/taco_pour_ppo_checkpoint_eligibility_v1.yaml`

The compressed, hash-bound checkpoint required by existing read-only audits
remains under the corresponding evidence run; it has the same eligibility
restriction.
