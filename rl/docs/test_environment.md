# Test Environment

## Four-world training checkpoint

On 2026-09-23, the four-world engineering gate restored all **342** Warp state
fields bitwise in four independent GPU worlds and proved that a complete reset
of one world leaves the other three snapshots unchanged. The single authorized
eight-epoch experiment recorded 1,280 samples and completed GPU actor to CPU
actor transfer. CPU Replay validated `29/40`; PPO validated `31/40` and failed
at endpoint 52, so no state was committed. The first attempted run completed
training but exposed a missing official checkpoint method before CPU
validation; that incomplete run is isolated under `TRASH/incomplete_runs/`
and is not used as algorithm evidence.

After adding the independent-world checkpoint, formal evidence audit, and
regression tests, the complete suite passed **612 tests and 57 subtests** in
110.48 s. The 17 warnings are existing MuJoCo-Warp contact-capacity,
PyTorch AMP deprecation, Trimesh degenerate-volume, and SciPy pickle warnings;
there were no failures or skipped tests.

## Prospective training-coverage checkpoint

On 2026-09-23, one predeclared eight-epoch prospective coverage run used the
accepted version-2 visitation logger. An independent audit hash-checked the
formal report, manifest, and all eight raw/summary epoch pairs, reconstructed
episodes across epoch boundaries, and verified 320 samples. Outcome endpoints
46--50 were visited `6, 6, 4, 4, 4` times. The deterministic CPU result was
Replay `29/40` and PPO `28/40`; neither mode passed and no state was committed.
After adding the portable audit and three static evidence-contract tests, the
complete isolated suite passed **607 tests and 57 subtests** in 112.79 s. The
GPU training run itself is experimental evidence, not a unit test.

## PPO training-visitation logging checkpoint

On 2026-09-23, after adding epoch-level PPO visitation records and the paired
logging-transparency audit, the complete isolated suite passed **592 tests, 1
skipped test, and 57 subtests** in 89.94 s. The skip is the real CUDA adapter
module in the CPU test environment. The separate real CPU MuJoCo-Warp audit
ran two one-epoch/four-step PPO trials and found every compared training,
simulator, recurrent, and RNG state bitwise equal with logging off versus on.
The logged pre-clamp value is the stochastic PPO action sample, not actor mean
`mu`; the visitation schema is version 2 after making that terminology explicit.
This audit changes no task-performance result and does not authorize a new
training-budget or seed search.

## Policy diagnostic checkpoint

On 2026-09-22, after adding the read-only normalized-policy diagnostic, the
isolated CPU suite passed **571 tests, 1 skipped test, and 57 subtests** in
88.32 s. The four new tests cover checkpoint normalization/RNN state transfer,
failure-tail decomposition, reward reconstruction, action saturation, and the
non-causal status of the independent 8/16-epoch comparison. The diagnostic
itself performs no physics rollout or training.

The older Spider environment still carries MuJoCo/MJWP 3.7, so its two expected
MJWP contract failures must not be mixed with the 3.13 contract. Conversely,
placing the 3.13 packages in front of that unadapted 3.7 Spider checkout fails
at the known `xfrc_applied` vector-layout boundary. Neither mixed environment is
used as evidence for the policy diagnostic.

## Current normalized-ellipse checkpoint

On 2026-09-22, the complete MuJoCo 3.13 / mujoco-warp 3.13 / Warp 1.15 GPU
stack passed **569 tests and 57 subtests** in 101.13 s. At the preceding
checkpoint, the isolated CPU stack passed **559 tests and 57 subtests**, with
the CUDA/MJWP module skipped. A
targeted run passed all 9 real-GPU adapter tests plus 9 objective/observation
contract tests. The remaining warnings are the
recorded capsule-mesh MULTICCD limitation, upstream AMP deprecations, two
Trimesh diagnostics and seven SciPy pickle deprecations.

## Current initialization and raw-depth audit

Latest implementation check (2026-09-20): **558 passed, 57 subtests**
in 105.04 s. The current checks include the frozen first-40 four-way collision
attribution, the training-blocked collision-semantics v2 candidate, its isolated
object/floor contact roles and rejected palm/thumb guards, the skipped-release
`not evaluated` schema, the bilateral index-root semantic-guard contract,
TRASH relocation integrity, pose-specific false-alarm classification,
contained/open CAD limitations, invalid volumes/distances, and separating a
50-micrometre reporting cut from actual runtime contact generation. Historical
raw measurements are retained; physics shapes and reset validity are unchanged.
The separately executed GPU adapter suite has **9 passing tests**
(11.37 s),
including real official-PPO fallback from an incoming boundary, non-autoreset
validation, and 40 physical control intervals committing only interval 20 in a
static test fixture. This is interface evidence, not a Pour task-success run.
Local MINK now has opt-in preservation of requested explicit parent-child pairs.
Original GT, scene and archived diagnostic provenance hashes remain unchanged.
The archived fingertip-contract regression now rebuilds and compares the complete
geometry definition and independently remeasures its saved trajectories instead
of requiring the historical generation solver's entire file to remain unedited.
The three newest tests certify empty-cavity subtraction on a known solid,
reject cutting actual native material, and reject invalid CoACD output before
creating a partial asset directory. Local repairs remain rejected for training
because the real palm/index shape screen still reports false overlaps.

The 2026-09-17 audit uses the already isolated environment:

```bash
PYTHONPATH=src:external/mink/src OMP_NUM_THREADS=4 \
  /data_all/zzx/deximit_isolated/env-py311-cu118/bin/python -m pytest -q
```

Its versions are Python 3.11, NumPy 1.26.4 and MuJoCo 3.12.0. Native triangle
checks additionally require `python-fcl==0.7.0.11` (installed for this audit;
its Cython dependency is 3.3.0). FCL is a diagnostic dependency, not a change
to MuJoCo's physical collision solver. Closed-native distance audits now use
the existing Open3D 0.18.0, with independent float64 solid-angle checks for
extreme points; ambiguous Trimesh ray signs failed real-bowl regressions.
Closed intersection volumes still use Manifold. Collision experiments installed
`coacd==1.0.12`, matching the original decomposition version. No main robot or
simulator dependency was upgraded during this audit.

After the collision-shape refinement and contact-capacity checks, the full suite
passed **510 tests and 57 subtests**, with **1 skipped** and 7 existing SciPy
deprecation warnings (112.37 s). The previous checkpoint had 504 passing tests.
The latest six checks cover true solid overlap vs surface crossing, open meshes,
explicitly disabling implicit hole repair, actual distal geometry invariance,
URDF joint comparison and overwrite protection. The final targeted contact and
two-chunk scheduler test run passed 18 tests (1.95 s).
The new tests cover allocation overflow (including broadphase and per-world
constraints), physical-vs-visual mesh selection, clipped-piece containment, and
avoiding contact dynamics during pure mesh measurement. A further real clipped
finger regression exposed a false 0.176 mm gap in metre-scale Trimesh proximity
queries: Open3D, a millimetre-scaled query and independent convex projection
agree on about 2.8 nm at that point. Current object/finger missing-coverage
measurements use unsigned Open3D distance; no mesh or simulator was changed by
this numerical fix. Open-surface and scale-invariance tests are included.
The earlier seven preflight tests cover native-triangle intersections,
transform updates, open surfaces, the contained-solid limitation, metric object
overfill, overwrite protection, and corrected-reference selection. Tests passing
does not mean the measured scene passes physical initialization checks.

## Actual GPU capacity validation

The isolated CPU environment has no `mujoco_warp`; the skipped adapter module was
separately tested in the existing SPIDER environment, **9 passed** (11.37 s).
Actual capacity measurements use that same environment, MuJoCo **3.7.0**,
`mujoco-warp==3.7.0.1`, Warp **1.12.1**, on an A100. No environment was upgraded.
CPU 3.12 and GPU-environment CPU 3.7 contact counts differ; retain the version
beside every reported count instead of treating them as interchangeable.

```bash
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 \
  /data_all/zzx/egoengine/spider/.venv/bin/python \
  scripts/audit_taco_contact_capacity.py \
  --scene runs/taco_pour_collision_repair/scene_refined.xml \
  --gpu --worlds 16 --nconmax 1024 --njmax 2048 --steps 30 \
  --spider-config configs/taco_pour_bimanual_ppo.yaml \
  --output runs/taco_pour_collision_repair/capacity_refined_16env.json
```

The output already exists and the script refuses to overwrite it. For a rerun,
first decide whether the previous result is still a needed comparison; do not
silently replace it or proliferate versions. The 4- and 16-world runs completed
198 synchronized reference snapshots and 0.1 s held-control stresses from rows
0 and 111 without overflow or nonfinite states. This is capacity validation,
not successful Replay, a legal reset, or PPO training. Formal PPO settings and
the active source scene remain unchanged.

## Earlier test environment (historical)

The earlier reproducible environment used for this checkout is:

```text
/data_all/zzx/egoengine/spider/.venv/bin/python
```

It contains NumPy `2.4.4`, MuJoCo `3.7.0`, PyTorch `2.11.0+cu130`,
`smplx`, `pytest`, and `robot_descriptions`, GitPython and its
`gitdb`/`smmap` dependencies. The local MINK checkout is installed in editable
mode, so imports resolve to `external/mink/src/mink`.

Run the complete project suite from the repository root with:

```bash
PYTHONPATH="$PWD/src:$PWD/external/mink/src" \
  /data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q
```

On 2026-09-09 this command passed **481 tests and 57 subtests**. Seven MANO
pickle NumPy/SciPy deprecation warnings remain; they do not change numerical
results.

MuJoCo 3.12 no longer exposes `MjData.qM`, and its compiled enum fields are
NumPy integers. The vendored MINK compatibility change uses the current
`mj_fullM(model, data, dense)` call and falls back to the historical form;
project comparisons cast enum values to integers. This only fixes binding/API
compatibility and does not change the robot model or controller behavior.
