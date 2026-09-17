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
- The real MJWP/Human2Sim2Robot PPO adapter has passed GPU interface, rollback and
  chunk-boundary tests. These tests are not a successful Pour rollout.
- The current Pour first frame is not a legal physical reset. Native right/left hand
  meshes enter the present simulated table by about 20.79/9.65 mm. A legal reset has
  not been selected, so formal Pour Replay -> RL has not been run.
- Palm/index shell overlap at the audited initial pose is classified as a collision-
  shape false positive, not native CAD penetration. The same body pair has confirmed
  native interference at other joint angles and cannot be globally exempted.

Start with [`docs/pour_discussion_handoff.md`](docs/pour_discussion_handoff.md) for
the concise technical handoff, then see [`docs/replay_rl_implementation.md`](docs/replay_rl_implementation.md)
and [`configs/replay_rl_protocol.yaml`](configs/replay_rl_protocol.yaml) for the
implementation contract.

## What is included

- `src/`, `scripts/`, `tests/`, `configs/`, `diagnostics/`: implementation and checks.
- `models/`: the local XHand/MuJoCo model snapshot used by the audits.
- `runs/`: compact references, geometry candidates and audit reports. PPO checkpoints
  and smoke-training weights are excluded.
- `external/mink/`: MINK source based on commit
  `ab45779fea46933832dee1c240f94103633347a1`, including the three local compatibility
  and explicit-pair changes used by this project.
- `external/human2sim2robot/`: the PPO source subset used here, copied from commit
  `c468b751041c721ff48146b54a891b1ec99c2e2b`. Unused deployment, data-processing
  and original simulator task code is intentionally not duplicated.
- `external/spider_overlay/`: the local SPIDER files used by this work, based on
  `facebookresearch/spider` commit `4bd2756720ea95b9da126d98da3bf414fe964849`.

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

MINK and the required Human2Sim2Robot PPO source are included for an exact code snapshot. SPIDER must
be cloned separately, checked out at the commit above, and overlaid with the saved
files:

```bash
git clone https://github.com/facebookresearch/spider.git external/spider
git -C external/spider checkout 4bd2756720ea95b9da126d98da3bf414fe964849
rsync -a external/spider_overlay/ external/spider/
export SPIDER_ROOT="$PWD/external/spider"
```

CPU audit invocation used in the source workspace:

```bash
PYTHONPATH="$PWD/src:$PWD/external/mink/src" OMP_NUM_THREADS=4 \
  python -m pytest -q
```

The last complete local-data run passed **522 tests, 1 skipped, and 57 subtests**.
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
