# DexImit 3.2 backup

This directory is a diagnostic-only backup of the DexImit bridge for the TACO
eraser episode `taco_smear_eraser_box_20231103_071` (`smear071`, object 078).
It is not part of the formal renderer or the 3.3 pipeline.

## Frozen result

The checked-in artifact is the historical successful candidate 21 replay:

- active fingers: thumb, index, middle, ring
- BODex depth: 0, original per-depth top-120 ranking
- original rank: 54
- pregrasp / grasp / motion end rows: 18 / 30 / 39
- SAPIEN: pass
- exact MuJoCo replay: strict gate pass
- candidate pose policy: `legacy`
- seed: `20260829`

`legacy` is intentional. This pool was generated with the old identity-pose
BODex contract. Changing it to `strict` changes the experiment and must not be
used to claim reproduction of this frozen result.

The fixture contains only the inputs, candidate pools, exact scene, traces,
calibration evidence, and the fixed-camera success video needed for this
sample. The large original `runs/` tree is not required.

## Reproduce

The upstream DexImit checkout and its patched working tree are external
dependencies. The backup records the upstream commit and the current working
tree patch:

```bash
git clone https://github.com/mujc2021/DexImit-Open.git /tmp/DexImit-Open-c5749809
cd /tmp/DexImit-Open-c5749809
git checkout c5749809fa425525a9bf7aa392dbb3f6268ae28f
git apply /path/to/Egoengine/deximit/vendor/DexImit-Open-c5749809-worktree.patch
```

From the root of this repository, run:

```bash
export DEXIMIT_ROOT=/tmp/DexImit-Open-c5749809
export DEXIMIT_PYTHON=/path/to/deximit-python
export MUJOCO_PYTHON=/path/to/mujoco-python
./deximit/reproduce_smear071.sh
```

The script creates a fresh output directory under `/tmp` (or
`$OUTPUT_ROOT`), runs the unchanged DexImit `rollout_and_select_grasp` path in
SAPIEN, exports candidate 21's physics-step trace, and replays that trace in
the exact MuJoCo scene. It exits non-zero unless both SAPIEN and the MuJoCo
strict physical gate pass. It never overwrites the checked-in evidence.

Required runtime components are the DexImit/BODex environment with CUDA,
SAPIEN 3.0.1, cuRobo 0.7.8, and a MuJoCo Python installation compatible with
the checked-in diagnostic scripts.

## Frozen evidence

- SAPIEN summary: `fixture/smear071/sapien/summary.json`
- SAPIEN trace: `fixture/smear071/sapien/candidate21_trace.npz`
- MuJoCo scene: `fixture/smear071/mujoco/scene.xml`
- MuJoCo replay report: `fixture/smear071/mujoco/report.json`
- fixed-camera video: `fixture/smear071/evidence/real_ref_sim_passed_fixed_third_person.mp4`

All artifact links used by the runnable contract are relative to the repository
root. The prompt stores the same MANO numerical data as the source artifact but
uses fixture-relative provenance paths so a fresh clone does not depend on the
original workstation layout.
