# Quarantine

This directory is for obsolete or rejected project artifacts that have been
explicitly confirmed as unused. Moving an artifact here does not make it part
of the formal Replay-to-RL input chain. Do not place active models, references,
GT, reports, or reproducibility evidence here without first recording why they
were superseded.

`rejected_candidates/2026-09-20_collision_semantics_repair_v2_self_guard`
contains the rejected sphere, convex and hybrid left palm/thumb guard fits and
their local CoACD asset. They are retained only as negative audit evidence:
each failed an independent offset holdout and none belongs to the formal scene.

The following training-trace quarantine directories are server-local and are
deliberately not uploaded to Git; the checkpoint directories alone contain
about 322 MB of automatically emitted, unused weights.

`taco_pour_training_trace_transparency_pre_poststep_capture_20260923` is the
superseded first transparency run. The action recorder was subsequently moved
after the official action preprocessing/physics call so it introduces no extra
pre-step GPU synchronization. It is not formal evidence.

`taco_pour_training_trace_transparency_pre_failure_contract_20260923` is the
next superseded small report. The active version adds explicit failure-path and
mixed metre/radian coordinate-unit metadata.

`taco_pour_training_trace_transparency_v1_superseded_terminology` passed the
same CPU transparency gate, but called the stochastic PPO sample a generic
`policy_output_prelimit`. Schema v2 replaces it with the unambiguous
`sampled_action_preclamp` and explicitly states that actor `mu`/variance are not
recorded. The v1 report is not part of the active evidence chain.

`taco_pour_training_trace_transparency_v1_generated_checkpoints` and its
`_final` counterpart contain the 84 MB one-epoch checkpoints and TensorBoard
event files automatically emitted by the official PPO trainer during the
logging-equivalence audits. They are not used by the comparison: the audit
compared the live model, both optimizers, critic, complete simulator state,
recurrent state and RNG states bitwise, and keeps only the small report and
visitation artifacts active.

`taco_pour_training_trace_transparency_v2_generated_checkpoints` is the same
automatic output from the active terminology-corrected audit. It is likewise
excluded from the formal evidence and Git; the active v2 report and visitation
files remain under `runs/`.
