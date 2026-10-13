# TACO Pour Algorithmic Reproduction Training Benchmark v2

The residual-identity repair and both mandatory infrastructure gates passed:

- Formal Replay: `30/40`, first failure endpoint `51`.
- Candidate B at zero training steps: exact Replay identity, `30/40`, first failure endpoint `51`.
- Fresh checkpoint roundtrip: actor, critic, both optimizers, normalization, RNN, environment and RNG all exact.

Candidate A then halted fail-closed at epoch 4, before that epoch's optimizer step. Its first differentiable likelihood recomputation had `max|ratio-1| = 5.7220459e-6`, above the frozen `5e-6` gate. An autograd warmup did not eliminate the discrepancy, so that workaround was removed and the incomplete raw run was moved to `TRASH`.

Only three actor updates completed (`4,800` physics steps), far below the first `100k` milestone. No checkpoint was promoted, Candidate B training did not start, no conditional seeds ran, and no chunk was committed. Algorithmic A/B conclusions remain unset.
