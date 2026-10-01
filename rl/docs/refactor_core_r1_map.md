# RL core refactor R1 map

## Scope

The active CPU Pour path is extracted from the historical Candidate-G/FULL
implementation without changing its task, network, distribution, optimizer,
reward, observation, action, reference timing or MuJoCo-Warp physics contract.
The implementation baseline is `a1622a4`, so the B1/B2/B3 fixes reviewed after
`96e8c03` remain active. The older commit is used only as a numerical oracle.

## Before

```text
CandidateGWarmStartPpoAgent
  -> SupportAnchoredBoundedMeanPpoAgent
  -> AlgorithmicBenchmarkPpoAgent
  -> StateFeasibleTruncatedGaussianPpoAgent
  -> external PpoAgent
```

Runtime construction additionally crossed three historical Candidate runners.
Likelihood recomputation, logging and epoch sequencing were distributed across
method overrides.

## After

| Stable module | Responsibility | Historical source extracted |
| --- | --- | --- |
| `core/env.py` | actor-free CPU MJWP world(s), fixed boundary | `mjwp_env.py`, runner assembly |
| `core/state_io.py` | hashes, full snapshots, RNG, checkpoint schemas | Candidate runner helpers |
| `core/policy.py` | pinned actor/critic construction, one RNN evaluator, prefix burn-in | `run_mjwp_ppo.py`, Candidate G/D evaluators |
| `core/distribution.py` | support, bounded mean, truncated Normal math | v6 and state-feasible Candidate modules |
| `core/rollout.py` | fixed-boundary collector, GAE, world-major layout | external agent and curriculum overrides |
| `core/ppo.py` | direct critic/actor update and explicit epoch order | five-level trainer inheritance |
| `core/audit.py` | detached pure summaries/hashes | optimizer overrides and callbacks |
| `core/runner.py` | `inspect` and bounded `verify`; no training/commit | Candidate G runner |

The new core reuses only pinned bottom-level dependencies: Spider/MuJoCo-Warp,
`MJWPVectorEnv`, H2S2R network/RMS/asymmetric-critic building blocks, and the
candidate-independent task contracts. It never constructs a Candidate agent
or calls the legacy `PpoAgent` training orchestration.

## Not migrated

GPU training, long training, chunk scheduling/commit, randomized/tail
curriculum, other tasks, and historical experiment reporting are deliberately
outside R1. Historical artifacts remain immutable evidence.

After bounded verification, the retired implementation and every script/test
in its import closure were moved (not copied) to
`TRASH/rl_core_refactor_r1_legacy_2026-10-01/`. This intentionally overrides
the original draft's suggestion to leave old source beside the new core: the
user requested a single unambiguous active implementation. Git history and the
dated archive preserve recovery.
