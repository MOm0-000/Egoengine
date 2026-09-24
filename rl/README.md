# EgoEngine 3.2 RL reproduction

This directory is the code and audit snapshot for reproducing the action-generation
and RL portions of EgoEngine Section 3.2 and Appendix C. The active experiment is
the bimanual TACO Pour/Bowl/Plate episode `20230927_017`.

The implementation follows **Replay -> residual PPO**. MPC is deliberately omitted
because the relevant EgoEngine component is closed source. This branch therefore
does not claim an exact reproduction of the paper's three-mode cost comparison.

## Current status

- The two-hand/two-passive-object model, MINK reference path, paper reward equations,
  20-step chunks, 40-step lookahead and 20-step commit contract are implemented.
- The real MJWP/Human2Sim2Robot PPO adapter, complete-state rollback, independent
  four-world training, and GPU-training/CPU-acceptance contracts are implemented.
- A collision-checked initialization is accepted for the current local physics
  contract. This does not make the MINK motion prior itself physically executable.
- The current local normalized-ellipse experiment still fails the strict first-window
  `40/40` CPU gate. The latest controlled 3+1 curriculum run reached `32/40`; no
  full-horizon normalized-ellipse success is claimed.
- The paper does not publish the exact objective coefficients or actor observation
  encoding. Local choices remain explicitly separated from paper-recovered facts.

Start with [`docs/pour_discussion_handoff.md`](docs/pour_discussion_handoff.md) for
the concise technical handoff, then see [`docs/replay_rl_implementation.md`](docs/replay_rl_implementation.md)
and [`configs/replay_rl_protocol.yaml`](configs/replay_rl_protocol.yaml) for the
implementation contract.

## What is included

- `src/`, `scripts/`, `tests/`, `configs/`, `diagnostics/`: implementation and checks.
- `models/`: the local XHand/MuJoCo model snapshot used by the audits.
- `runs/`: compact references, geometry candidates and audit reports. PPO checkpoints
  and smoke-training weights are excluded.
- `external/mink/`: the MINK runtime and unit-test subset based on commit
  `ab45779fea46933832dee1c240f94103633347a1`, including the three local compatibility
  and explicit-pair changes used by this project. Upstream documentation examples
  and their collection-only test are omitted.
- `external/human2sim2robot/`: the PPO source subset used here, copied from commit
  `c468b751041c721ff48146b54a891b1ec99c2e2b`. Unused deployment, data-processing
  and original simulator task code is intentionally not duplicated.
- `external/spider_compat/`: the minimal importable SPIDER subset used by the formal
  runner, based on `facebookresearch/spider` commit
  `4bd2756720ea95b9da126d98da3bf414fe964849` and adapted for the pinned MuJoCo-Warp
  3.13 state layout. Unused preprocessing, deployment and simulator code is omitted.

The third-party directories retain their upstream licenses. Local changes are
engineering adaptations and are not presented as published EgoEngine settings.

## Data intentionally not uploaded

The raw TACO archive, RGB/depth videos, MANO PKLs, GT arrays, the paper PDF, Python
environments and PPO weights are not committed. The local project is about 26 GB,
including a 25 GB source archive; placing that material in ordinary Git would be
neither practical nor a clean dependency boundary. Small selection, acquisition and
checksum manifests are kept under `data/taco_v1/pour_bowl_plate/`.

Restore the TACO files at the paths recorded in
`configs/replay_rl_protocol.yaml` and the acquisition manifests. Original depth must
remain 1920x1080 `gray16le` and is decoded as `raw_uint16 / 4000.0` metres. Do not use
the obsolete resized depth as metric input.

## External setup

The required MINK runtime, Human2Sim2Robot PPO source, and minimal importable SPIDER
compatibility subset are included as pinned code snapshots. The formal entry point
defaults to `external/spider_compat`; `SPIDER_ROOT` may still override it explicitly.

CPU audit invocation used in the source workspace:

```bash
PYTHONPATH="$PWD/src:$PWD/external/mink/src" OMP_NUM_THREADS=4 \
  python -m pytest -q
```

The last complete local-data run passed **637 tests and 57 subtests**.
The separately executed GPU adapter suite had 9 passing tests. A clone without the
omitted TACO inputs cannot run the data-dependent tests until those inputs are restored.

## Important boundaries

- `z=0.72 m` is stated in EgoEngine Appendix A.1 for the authors' robot table. The
  complete TACO-to-simulation vertical alignment recipe is not published.
- Collision approximations, local reward coefficients and reset candidates are clearly
  marked as local settings where the paper does not publish values.
- No collision pair, source GT frame, object pose or task threshold may be silently
  removed to make a rollout pass.
- The only permitted project renderer is the copied
  `diagnostics/render_exact_deximit_triptych.py`.
