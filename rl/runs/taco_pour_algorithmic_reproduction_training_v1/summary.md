# TACO Pour Algorithmic Reproduction Training Benchmark v1

Candidate B's mandatory zero-step gate was evaluated before any optimizer update.

- Formal Replay: `30/40`, first failure endpoint `51`.
- Candidate B at zero physics training steps: `21/40`, first failure endpoint `42`.
- Maximum command difference: `1.1920929e-07`.
- Maximum qpos difference: `0.183517218`.

The B0 gate failed. Per the frozen contract, Candidate A and B training did not start, no checkpoint was produced, and no chunk was committed.
