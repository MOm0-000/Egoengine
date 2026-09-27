# TACO Pour Algorithmic Reproduction Training Benchmark v3

The mandatory fresh Candidate-B zero-step gate passed exactly: Replay and
Candidate B were both `30/40`, first failure endpoint `51`, with bitwise-equal
command, qpos, qvel, and tracking score plus equal termination/contact flags.

Candidate A then started from fresh actor, critic, optimizers, and RMS. Epochs
1–31 completed legally. Before the epoch-32 optimizer update, actor parameters,
actor RMS, and normalization version were still exact and the likelihood ratio
error was only `3.8146973e-6`; however, `max |mu_old-mu_new|` was
`1.1920929e-6` (`10 * float32 epsilon`), exceeding the predeclared
`8 * epsilon` gate (`9.5367432e-7`). The run therefore failed closed with zero
optimizer updates in the failed epoch.

- Completed Candidate-A actor updates: `31`.
- Physics steps before failure: `51,200`.
- 100k milestone reached: `false`.
- Candidate-B training started: `false`.
- Dynamic gate relaxation: `false`.
- Chunk committed: `false`.

The incomplete Candidate-A runtime directory was moved to `TRASH`; the formal
run retains only this failure report and immutable artifact hashes.
