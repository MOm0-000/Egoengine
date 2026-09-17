# EgoEngine 3.1/3.2 reproduction sidecar

`egoengine_repro` consumes frozen `video_to_spider` artifacts and writes every
new result to a separate experiment directory. It does not change the existing
pipeline defaults or write into a source run.

## Profiles

- `auto_current`: freezes the current SAM3, WiLoR, Depth Anything, SAM3D,
  FoundationPose, smoothing, scale, contact and QC behavior.
- `paper_faithful`: declares dataset calibration, oracle EgoDex hand poses,
  known object mesh, FoundationStereo and FoundationPose with that known mesh.
  GT hand poses are allowed only through the isolated oracle adapter.

The paper profile is an oracle upper-bound experiment until a known object mesh
and its FoundationPose trajectory are supplied. Reusing the current SAM3D mesh
or contact-calibrated object trajectory is a hybrid smoke test, not a
paper-faithful result.

## Experiment layout

```text
repro_runs/<episode>/<profile>/
  experiment_manifest.json
  evaluation/offline_3_1_metrics.json
  retarget/human_reference.npz
  retarget/reference_robot_trajectory.npz
  retarget/reference_robot_trajectory.json
  replay/replay_chunk_report.json
  replay/replay_accepted_actions.npz
```

`experiment_manifest.json` stores the resolved configuration and hash, Git
revisions, seeds, recursively hashed source artifacts, and stage-level input and
output hashes. Pass `--experiment-manifest` to each later command to append its
stage record.

## Auto-current run

Run data preparation and 3.1 evaluation in `v2s-core`:

```bash
cd /data_all/intern02/egoengine/video_to_spider
RUN=runs/egodex_vertical_pick_place_111_full
EXP="$PWD/repro_runs/111/auto_current"
GT="$PWD/repro_runs/111/ground_truth"

conda run -n v2s-core egoengine-repro init-experiment \
  --config auto_current --source-run "$RUN" --output-dir "$EXP" \
  --episode-id 111 --seeds 0 1 2 3 4
conda run -n v2s-core egoengine-repro make-egodex-gt \
  --source-run "$RUN" --output-dir "$GT" \
  --experiment-manifest "$EXP/experiment_manifest.json"
conda run -n v2s-core egoengine-repro evaluate-3.1 \
  --config auto_current --source-run "$RUN" \
  --ground-truth-manifest "$GT/ground_truth_manifest.json" \
  --output-dir "$EXP/evaluation" \
  --experiment-manifest "$EXP/experiment_manifest.json"
conda run -n v2s-core egoengine-repro prepare-human-reference \
  --source-run "$RUN" --output "$EXP/retarget/human_reference.npz" \
  --experiment-manifest "$EXP/experiment_manifest.json"
```

Run MINK and MuJoCo Replay in the SPIDER environment:

```bash
SPIDER=/data_all/intern02/egoengine/spider
MODEL="$PWD/$RUN/spider_export/dataset/processed/video_to_spider_egodex/xhand/right/vertical_pick_place/scene.xml"
PYTHONPATH="$PWD" "$SPIDER/.venv/bin/python" -m egoengine_repro.cli retarget-mink \
  --config auto_current --human-reference "$EXP/retarget/human_reference.npz" \
  --model "$MODEL" --output "$EXP/retarget/reference_robot_trajectory.npz" \
  --experiment-manifest "$EXP/experiment_manifest.json"
PYTHONPATH="$PWD" "$SPIDER/.venv/bin/python" -m egoengine_repro.cli evaluate-replay \
  --config auto_current --reference "$EXP/retarget/reference_robot_trajectory.npz" \
  --model "$MODEL" --output-dir "$EXP/replay" \
  --experiment-manifest "$EXP/experiment_manifest.json"
```

Replay uses `H=20` and evaluates the current plus next chunk. A current chunk is
accepted only when its full lookahead window remains below the object-tracking
feasibility boundary. The stored final state is the current-chunk boundary,
not the end of the lookahead rollout.

The bundled Aria-style reward normalizes the documented `0.08 m` translation
and `2.5 rad` rotation boundaries into Eq. 2 with `C=1`. Per-task studies can
override those values while keeping both A/B profiles identical. Reports expose
the paper's normalized Step and Reward ratios, control-level simulation Cost,
and MuJoCo physics substeps as a separate implementation diagnostic.

## Paired SPIDER comparison

Convert legacy SPIDER IK, the new MINK output, and saved MJWP controls into one
reference/replay protocol, then execute all three from the same state:

```bash
PYTHONPATH="$PWD" "$SPIDER/.venv/bin/python" -m egoengine_repro.cli compare-spider \
  --config auto_current --model "$MODEL" \
  --keypoints /path/to/trajectory_keypoints.npz \
  --old-ik /path/to/trajectory_kinematic.npz \
  --new-mink "$EXP/retarget/reference_robot_trajectory.npz" \
  --spider-mjwp /path/to/trajectory_mjwp.npz \
  --spider-config /path/to/config.yaml \
  --output-dir "$EXP/paired_spider_seed0" --seed 0 \
  --experiment-manifest "$EXP/experiment_manifest.json"
```

The adapter infers the valid-filter phase of legacy IK from the unchanged
object track, applies the translation-only SPIDER support alignment to MINK,
and resamples all commands to the common 10 ms overlap. Every method uses the
legacy IK state at the overlap boundary, the keypoint artifact's object and
human wrist targets, one MuJoCo model, one reward profile, and one seed. MJWP is
replayed from its saved `ctrl`; its sampling cost is reported separately as
`sum(opt_steps) * num_samples * horizon_steps`. Source artifacts are read-only,
and all converted references and reports are written below the comparison
directory. The command requires the replay `common_dt` to match saved MJWP and
records the saved optimizer seed separately from the common fresh replay seed.
The exact historical generation-model hash
cannot be verified when that config points to a now-inaccessible source path;
the report records this caveat while guaranteeing one model for all fresh
replays.

## Live Replay/MPC scheduler

Run paper-style Replay first and invoke live SPIDER MJWP only for a failed
two-chunk window:

```bash
PYTHONPATH="$SPIDER:$PWD" "$SPIDER/.venv/bin/python" -m egoengine_repro.cli \
  evaluate-replay-mpc --config auto_current \
  --reference "$EXP/retarget/reference_robot_trajectory.npz" \
  --model "$MODEL" --spider-config /path/to/frozen/spider_config.yaml \
  --output-dir "$EXP/replay_mpc_seed0" --seed 0 --device cuda:0 \
  --experiment-manifest "$EXP/experiment_manifest.json"
```

`MPCMode.solve()` synchronizes the failed chunk's CPU MuJoCo state into every
MJWarp sample world, optimizes the current plus next `H=20` chunk with the same
paper reward, and validates the optimized controls by fresh CPU replay. Only
the first chunk state is committed. Reports separate optimizer rollout cost,
validation cost, wall time, seed, and accepted Replay/MPC mode per chunk.

After running the old-IK and MINK references with the same seed, build the
four-group Replay/Replay-to-MPC report:

```bash
PYTHONPATH="$PWD" "$SPIDER/.venv/bin/python" -m egoengine_repro.cli \
  summarize-action-comparison \
  --paired-comparison "$EXP/paired_spider_seed0/comparison_report.json" \
  --old-replay-mpc "$EXP/live_old_seed0/replay_mpc_report.json" \
  --mink-replay-mpc "$EXP/live_mink_seed0/replay_mpc_report.json" \
  --output "$EXP/action_comparison_seed0.json" \
  --experiment-manifest "$EXP/experiment_manifest.json"
```

The summary rejects mismatched profiles, seeds, timelines, or method identities.
It reports rollout progress (`Step`) separately from committed complete chunks.

## Frozen four-episode development set

The bundled `oakinkv2_xhand_right_dev4` set freezes four local SPIDER examples
with distinct task categories: board wiping, beaker stirring, spoon/bowl
scooping, and tube pouring. Each episode includes MANO keypoints, legacy XHand
IK, a MuJoCo scene, saved SPIDER output, and a frozen optimizer config. This is
a 3.2 engineering set; it is not a substitute for the later EgoDex/TACO 3.1
ground-truth evaluation.

Run the complete four-episode, five-seed paired experiment:

```bash
PYTHONPATH="$SPIDER:$PWD" "$SPIDER/.venv/bin/python" -m egoengine_repro.cli \
  run-devset --devset oakinkv2_xhand_right_dev4 --config auto_current \
  --output-dir "$PWD/repro_runs/devset_oakinkv2_xhand_right_4" \
  --seeds 0 1 2 3 4 --device cuda:0 \
  --num-samples 256 --max-iterations 8
```

The runner first converts SPIDER MANO keypoints to the MINK input contract,
retargets each episode, and creates same-model old-IK/MINK references. It then
runs every live evaluation in an isolated child process so MJWarp resources are
released between groups. Existing reports are reused only when seed and input
artifact hashes match. `frozen_devset_manifest.json` locks the selected inputs;
`devset_summary.json` reports per-task and aggregate SR, Step, Reward, Cost,
standard deviation, and paired deltas.

### Complete MANO direction input

The frozen SPIDER keypoint files retain only the wrist and five fingertips.
Recover the complete OakInk-v2 21-joint MANO skeleton from the raw MANO
parameters in the WiLoR environment:

```bash
PYTHONPATH="$PWD" conda run --no-capture-output -n v2s-wilor \
  python -m egoengine_repro.cli recover-devset-mano21 \
  --devset oakinkv2_xhand_right_dev4 \
  --output-dir "$PWD/repro_runs/mano21_oakinkv2_dev4"
```

Run the isolated direction-constraint profile in the SPIDER environment:

```bash
PYTHONPATH="$SPIDER:$PWD" "$SPIDER/.venv/bin/python" -m egoengine_repro.cli \
  run-devset --devset oakinkv2_xhand_right_dev4 \
  --config auto_current_mano_direction \
  --mano21-input-dir "$PWD/repro_runs/mano21_oakinkv2_dev4" \
  --output-dir "$PWD/repro_runs/devset_oakinkv2_xhand_right_4_mano_direction" \
  --seeds 0 1 2 3 4 --device cuda:0 --num-samples 256 --max-iterations 8
```

The adapter preserves the original five fingertip positions and uses each
MANO `tip - distal` vector only to orient the corresponding XHand fingertip
site. `auto_current_mano_direction` sets the direction cost to `1.0`, matching
the paper profile, while the original `auto_current` remains unchanged. Use
`compare-devset-mink-variants` to create a hash-checked paired A/B report from
current-code control and candidate directories.

## MINK failure diagnostics and reward ablation

Run the frozen one-factor reward study on board wiping, spoon/bowl scooping, and
beaker stirring. The baseline and three variants share the MINK references,
MuJoCo models, seeds, and MPC budget:

```bash
PYTHONPATH="$SPIDER:$PWD" "$SPIDER/.venv/bin/python" -m egoengine_repro.cli \
  run-reward-ablation --ablation mink_reward_ablation_dev3 \
  --base-config auto_current \
  --base-devset-output "$PWD/repro_runs/devset_oakinkv2_xhand_right_4" \
  --output-dir "$PWD/repro_runs/reward_ablation_mink_dev3" \
  --seeds 0 1 2 3 4 --device cuda:0 \
  --num-samples 256 --max-iterations 8
```

The four variants are baseline, contact off, human-mimic off, and smoothness
off. `reward_ablation_summary.json` contains paired SR/Step/object-reward
deltas, the initial action jump, and object/human/contact/smoothness reward
breakdowns for every Replay-failed chunk and its MPC attempt. A positive-contact
counter refers to the weighted contact reward, not an independent physical
contact detector. Generated variant profiles and all rollouts stay below the
ablation output directory; `auto_current` and the source artifacts are unchanged.

## Oracle hand input

The oracle adapter consumes the evaluation-only hand artifact and an explicit
world-to-simulation transform. Select only hands present in the robot model:

```bash
conda run -n v2s-core egoengine-repro prepare-oracle-human-reference \
  --hand-ground-truth "$GT/hand_ground_truth.npz" \
  --t-sim-world /path/to/fixed_T_sim_world.npy \
  --object-reference /path/to/known_mesh_foundationpose_reference.npz \
  --hand right --object-side right \
  --output "$PWD/repro_runs/111/paper_faithful/retarget/human_reference.npz"
```

`--t-sim-world` accepts a `(4,4)` `.npy`, an `.npz` containing
`T_sim_world`, or a JSON report containing that key. For strict A/B evaluation,
freeze this transform independently of the automatic contact-scale and
per-frame floor-shift heuristics.

## 3.1 ground-truth contract

The evaluator reads only artifacts named in `ground_truth_manifest.json`.
Missing modalities are reported as `unavailable`; FoundationPose predictions
are never silently reused as GT.

| Artifact | Required content |
| --- | --- |
| `hand` | NPZ: `frame_indices`, `timestamps_s`, `hand_order`, `T_world_joint` `(T,H,21,4,4)`, `confidence` |
| `camera` | NPZ: `frame_indices`, `intrinsics`, `T_world_camera` |
| `segmentation` | NPZ: `frame_indices`, `masks` `(T,H,W)`, optional `valid` |
| `depth` | NPZ: `frame_indices`, `depth_m`, optional pixel `valid` |
| `mesh` | Metric canonical mesh readable by trimesh |
| `object_trajectory` | NPZ: `frame_indices`, `T_sim_object` or `T_camera_object`, optional `valid` |
| `contact` | NPZ: binary `contact` matching `(T,H,5)`, optional broadcastable `valid` |

Reported metrics cover hand wrist/fingertip/all-joint errors, root-relative
MPJPE, wrist rotation, reprojection and identity switches; mask IoU, validity
and temporal stability; depth AbsRel, scale and temporal warp residual; mesh
Chamfer/F-score/scale/completeness; object translation, rotation, ADD/ADD-S,
jumps and validity; fingertip-object distance, contact precision/recall/F1 and
object-local slip.

Freeze the 20-episode EgoDex hand/camera set independently of any predictions:

```bash
conda run -n v2s-core egoengine-repro materialize-evaluation-set \
  --evaluation-set egodex_hand_camera_20 \
  --output-dir "$PWD/repro_runs/formal_3_1_gt/egodex_hand_camera_20"
```

Materialize the preregistered 16-sequence TACO object set directly from the
downloaded release ZIPs:

```bash
conda run -n v2s-core egoengine-repro materialize-taco-set \
  --taco-set taco_object_gt_16 \
  --output-dir "$PWD/repro_runs/formal_3_1_gt/taco_object_gt_16"
```

The TACO bundles contain world-frame MANO21 positions, egocentric camera
calibration, metric tool mesh, world/camera tool pose, and contact derived from
a frozen 5 mm GT-geometry threshold. TACO V1 masks in this release are
allocentric at 6 FPS. They are registered as auxiliary provenance and are not
accepted as GT for the egocentric SAM3 output. Egocentric mask IoU therefore
still requires manual masks or a compatible first-person annotation source.

## Independent episode gate

Use `oakink_xhand_bimanual_dev7` for the locally available bimanual development
set:

```bash
PYTHONPATH="$SPIDER:$PWD" "$SPIDER/.venv/bin/python" -m egoengine_repro.cli \
  run-devset --devset oakink_xhand_bimanual_dev7 --config auto_current \
  --output-dir "$PWD/repro_runs/devset_oakink_xhand_bimanual_7" --prepare-only
```

The 35 OakInk `data_id` directories contain only eight unique keypoint
trajectories; most IDs are saved SPIDER optimizer seeds for the same
demonstration. The loader hashes keypoint artifacts and records the independent
trajectory count. `oakink_xhand_bimanual_seed_replicates20` is retained only as
a seed-replicate preflight and must not be reported as a formal 20-episode A/B.
The formal `20 episodes x 5 seeds` run remains gated until at least 20 unique
TACO/EgoDex trajectories have been exported to the same XHand/MuJoCo contract.

## TACO first-person four-profile A/B

The frozen development subset uses the full intervals of brush/bowl,
cut/spatula/plate, skim/spatula/plate, and smear/eraser/box. Extract only those
members from the 24.7 GB archive, then ingest RGB and camera calibration without
copying hand/object/contact GT into the automatic runs:

```bash
conda run --no-capture-output -n v2s-core python -m egoengine_repro.cli \
  extract-taco-rgb --taco-set taco_first_person_dev4 \
  --output-dir "$PWD/datasets/taco_v1/selected_rgb_dev4"

conda run --no-capture-output -n v2s-core python -m egoengine_repro.cli \
  ingest-taco-devset --taco-set taco_first_person_dev4 \
  --video-dir "$PWD/datasets/taco_v1/selected_rgb_dev4" \
  --output-dir "$PWD/repro_runs/taco_first_person_dev4_ingest"
```

`taco_perception_ablation_dev4.yaml` freezes four profiles:

| Profile | Hand | Mask | Depth | Mesh | Alignment |
| --- | --- | --- | --- | --- | --- |
| `auto_current` | WiLoR | SAM3 | Depth Anything | SAM3D | current auto |
| `known_mesh` | WiLoR | shared SAM3 | shared | TACO mesh | current auto |
| `oracle_hand_mesh` | TACO MANO21 | shared SAM3 | shared | shared known-mesh tracking | current auto |
| `paper_faithful` | TACO MANO21 | SAM2 | shared | TACO mesh | fixed TACO pseudo-base |

Shared artifacts are read-only symlinks, so the paired comparison changes one
input family at a time without rerunning deterministic upstream inference. The
paper profile places the tool/target center 0.6 m along simulator +X, centers Y,
and shifts the target support surface to the 0.72 m table height. The paper does
not publish the complete axis convention, so that local +X convention is
recorded explicitly in every alignment report.

After freezing one first-frame object point per episode in the ablation YAML:

```bash
conda run --no-capture-output -n v2s-core python -m egoengine_repro.cli \
  run-taco-ablation \
  --ingest-manifest "$PWD/repro_runs/taco_first_person_dev4_ingest/taco_first_person_ingest_manifest.json" \
  --ground-truth-set-manifest "$PWD/repro_runs/formal_3_1_gt/taco_object_gt_16/frozen_taco_evaluation_set_manifest.json" \
  --output-dir "$PWD/repro_runs/taco_perception_ablation_dev4" --gpu 6
```

SAM2.1 Large runs from the `v2s-sam3` environment with the official SAM2
package installed alongside SAM3. Object masks use one first-frame positive
point and video-memory propagation; hand masks use projected MANO21 keypoints
on each frame and are labeled `oracle_hand`. TACO has one egocentric color
stream rather than rectified stereo pairs, so the first A/B shares Depth
Anything. FoundationStereo remains reserved for actual rectified Aria stereo;
native TACO depth is a later, separately labeled oracle-depth ablation.

## TACO oracle metric-depth ablation

The P1 ablation renders object-only camera-z depth and an object geometry mask
from the TACO mesh, object pose, and camera parameters.  These artifacts are
always labeled `rendered_gt_proxy`; they are neither manual dense-depth labels
nor complete-scene sensor ground truth.  The strict pair reuses the frozen
known-mesh mask, hand output, and mesh while replacing only FoundationPose's
depth input.  The SAM2/oracle-hand profile reuses the exact same rendered depth
directory so the extra mask/prompt change remains explicit.

```bash
conda run --no-capture-output -n v2s-core python -m egoengine_repro.cli \
  run-taco-oracle-depth-ablation \
  --source-ablation-manifest \
  repro_runs/taco_perception_ablation_dev4/taco_ablation_manifest.json \
  --output-dir repro_runs/taco_oracle_depth_ablation_dev4 --gpu 6
```

The skim release calibration puts the interval's MANO/object points behind the
camera.  Its oracle-hand branch uses a per-frame MANO3D-to-WiLoR-2D PnP result
labeled `derived_calibration_proxy` and `independent_ground_truth=false`.
See `docs/egoengine_repro/06_P1_ORACLE_METRIC_DEPTH.md` for the disclosure,
acceptance gates, and strict-pair coverage policy.

The TACO-specific action profile also follows the appendix by disabling domain
randomization, human-mimic reward, and action-smoothness reward. The general
Aria-oriented `paper_faithful.yaml` is unchanged.
