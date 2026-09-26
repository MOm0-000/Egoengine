# Post-fix fresh PPO uncompressed duplicate

The uncompressed trainer checkpoint was moved here after its gzip payload was
verified. The active evidence artifact is:

`runs/taco_pour_postfix_fresh_ppo_credit_instrumented_v1/checkpoint_artifacts/last_ep_8_rew__5.789554_.pth.gz`

Both payload hashes are recorded in the run's `checkpoint_transport.json`.
This checkpoint is valid post-fix algorithm evidence but is not authorized for
warm start, and the diagnostic run committed no Replay→RL chunk.

The TensorBoard event file is also stored here because the hash-bound JSON/NPZ
credit evidence is the formal record and the event stream is not consumed by
any audit or test.
