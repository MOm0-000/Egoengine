# Reward-aligned Replay→RL implementation

The formal runner is `scripts/run_taco_replay_rl.py`. It has two active modes:

1. `--replay-only`: starts from accepted initialization at endpoint 0, commits
   only CPU-validated 20-step prefixes, and stops at the first Replay failure.
2. Corrected PPO: requires the new endpoint-20 boundary, the reward-alignment
   gate, the corrected 4-world zero-optimizer gate and an exact hash-bound
   one-run authorization contract.

For transition `t→t+1`:

```text
actor goal/command       ref[t+1]
actor command preview    ref[t+2]
reward target            ref[t+1]
returned next goal       ref[t+2]
```

The reward target is computed independently of the next observation. Each
validation/training row records `command_reference_endpoint`,
`reward_reference_endpoint` and `next_observation_goal_reference_endpoint`.

GPU MuJoCo-Warp is used only for policy optimization. Replay and trained-policy
acceptance are rerun on CPU from the same complete boundary. A successful
40-step lookahead commits the CPU state at step 20. GPU state is never accepted
or committed.

The current PPO distribution samples directly from the actuator-feasible
truncated Gaussian. The state-dependent bounds are stored in the rollout buffer
and the same truncated likelihood is used during PPO recomputation. Without a
tail curriculum, recurrent resets use zero hidden state; the archived 3+1
curriculum is not part of the corrected experiment.
