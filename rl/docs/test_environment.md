# Test environment

## Active reward-aligned baseline

The active Replay→RL chain uses:

- Python from `/data_all/zzx/egoengine/spider/.venv/bin/python`;
- the local `.env_mjwp313_overlay` containing MuJoCo 3.13,
  MuJoCo-Warp 3.13 and Warp 1.15;
- project sources in `src` and the checked-in MINK source in
  `external/mink/src`.

Run the complete suite from the project root with:

```bash
PYTHONPATH="$PWD/.env_mjwp313_overlay:$PWD/src:$PWD/external/mink/src" \
  OMP_NUM_THREADS=4 \
  /data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q
```

On 2026-09-25 this command passed **613 tests and 57 subtests** in 105.07 s.
The 17 warnings are known upstream/diagnostic warnings: capsule-mesh MULTICCD
capacity, PyTorch AMP deprecations, two Trimesh degenerate-volume warnings and
seven SciPy pickle deprecations.

The active runtime contract is reward-aligned:

```text
state t + command ref[t+1]
        -> physical state t+1
reward/termination against ref[t+1]
returned next observation goal ref[t+2]
```

The no-training temporal gate, corrected Replay rebase, CPU repeatability
gate, dual-backend transfer gate and the first corrected PPO run are under:

```text
runs/taco_pour_reward_alignment_gate_v1/
runs/taco_pour_corrected_replay_rebase_v1/
runs/taco_pour_cpu_backend_repeatability_v1/
runs/taco_pour_dual_backend_contract_v1/
runs/taco_pour_corrected_ppo_gate_v1/
runs/taco_pour_corrected_first_ppo_v1/
runs/corrected_endpoint44_48_failure_attribution_v1/
runs/taco_pour_corrected_local_controllability_v1/
```

The first corrected PPO authorization is consumed. Replay passed the first
40-step lookahead and committed the CPU endpoint-20 boundary. In the second
lookahead Replay passed 30 intervals and PPO passed 27; neither passed 40/40,
so no new boundary was committed and formal training is closed pending a new
algorithm decision.

The endpoint 44--48 attribution is a no-training replay of the frozen actor.
It requires exact equality with both saved formal CPU traces before reporting
new contact flags, action bounds, actor distribution values or training-v5
coverage statistics.

The local-controllability run is also no-training. It restores hash-bound
complete physics plus RNN states at endpoints 44--47, checks every zero
perturbation branch against formal endpoint 48, and marks all actions outside
the formal residual support as diagnostic-only.

## Archived invalid performance evidence

Runs and diagnostics produced by the former off-by-one reward path are not
active evidence. They are isolated under:

```text
TRASH/historical_reward_misaligned_2026-09-25/
```

That archive must not be used to resume a boundary, warm-start an actor, or
compare task performance. Its README records the exact reason and scope. The
old files remain recoverable only as bug history.

## Other environments

The CPU-only collision and mesh audit environment remains:

```text
/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python
```

It uses Python 3.11, NumPy 1.26.4 and MuJoCo 3.12.0. Native triangle checks
require `python-fcl==0.7.0.11`; closed-solid and distance audits use the
already-installed Manifold/Open3D tools. These diagnostics do not replace the
MuJoCo-Warp 3.13 runtime used by Replay→RL.

The legacy Spider environment without the local overlay carries a different
MuJoCo/MuJoCo-Warp version and is not accepted as task-performance evidence.
Every runtime report records its backend contract and physics-contract hash so
results from the two stacks cannot be silently mixed.
