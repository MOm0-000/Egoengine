# Reproducibility and release checklist

This document defines the source, environment, model, and validation boundary for the GitHub release. It is intentionally stricter than “the current working directory runs”: a release candidate must be reconstructable from tracked files plus separately distributed environment/model assets.

## Current automatic safety baseline

- Pipeline: mono ingest → SAM3 → WiLoR → DA3METRIC-LARGE → UniDepthV2 reject-only gate, or calibrated rectified stereo ingest → unmodified FoundationStereo → native metric gate with global and automatic-object-mask coverage checks; then SAM 3D static metric scale fit → FoundationPose full-track gate → active-hand sequence optimization → MINK q_ref gate → Replay → SPIDER/MJWP.
- Inference/gates use no hand or object ground truth and no per-video thresholds.
- `vertical_pick_place/111`: tracking gate rejects the 2.112 rad rotation jump; later stages must not run.
- `basic_pick_place/0`: depth, tracking and sequence gates pass. Exact discrete MINK projection satisfies joint limits and all four MuJoCo geometry groups. After removing contact-target rewriting, consuming full landmark-derived fingertip SO(3), and mapping Eq.(1) coefficients through square-root MINK residual scales, position/wrist P95 improve to 16.92 mm/0.00134 rad; full fingertip orientation remains 2.236 rad and the q_ref is rejected. The same residual tradeoff repeats on oracle111. Replay/MPC must not run.
- Tests: `134 passed` in `video_to_spider`; eleven MINK unit tests pass in SPIDER's venv.
- Stereo status: the integration and hard gates are implemented, but the pre-registered HOT3D/ZED promotion experiment is not yet complete. Until the numeric gate passes, DA3 remains the documented production default.
- Evidence: `experiments/pipeline_phase1_regression_20260811/`, `experiments/pipeline_phase2_mink_landmark_smoke_20260811/`, `experiments/pipeline_phase3_discrete_projection_20260811/`, and `experiments/pipeline_phase4_fidelity_ablation_20260811/`.

## Historical structural baseline

- Dataset: EgoDex `vertical_pick_place/111`, raw MP4 and HDF5, 54 frames at 30 FPS.
- Verified code commit: `007c909af2bd80c66658f0e06696b9972a5fd569` (the subsequent documentation-only commit does not change runtime code).
- Isolation: a fresh RUN_DIR; no previous episode result cache was resumed.
- GPU: physical GPU 7 only, exposed to each model process as the single logical device `cuda:0`.
- Pipeline: ingest → SAM3 → WiLoR → Depth Anything V2 → SAM 3D Objects → FoundationPose → sequence/contact optimization → SPIDER export → contact/XML/native IK → MJWP → visualization and unified report.
- Structural result: 12/12 stages available, all five SPIDER subprocesses returned 0, MJWP video generated.
- Tests: 67 passed.
- Quality boundary: MJWP mean position error was 0.0701 m and passed the 0.1 m threshold; mean rotation error was 0.9262 rad and did not pass the 0.5 rad paper threshold. The release is therefore engineering-complete, not a claim that all paper metrics were reproduced.

## Repository layout

The large upstream projects are intentionally not vendored into this Git repository. Clone them at the exact revisions below:

| Component | Repository | Revision |
|---|---|---|
| SAM3 | `https://github.com/facebookresearch/sam3.git` | `6dbb02bd38288df755dfa1378000a861e65b84f6` |
| SAM 3D Objects | `https://github.com/facebookresearch/sam-3d-objects.git` | `f91db411c50efee93d8db7aeb323885650f6f722` |
| FoundationPose | `https://github.com/NVlabs/FoundationPose.git` | `a1b694b83e633c2cb6115b9063d940a687759392` |
| WiLoR | `https://github.com/rolpotamias/WiLoR.git` | `fcb911312a38fa8badd30d9656a167485d61b8f9` |
| Depth Anything 3 | `https://github.com/ByteDance-Seed/Depth-Anything-3.git` | `3d835ec` (validated experiment revision) |
| UniDepth | `https://github.com/lpiccinelli-eth/UniDepth.git` | `8d8cfe4` (validated experiment revision) |
| FoundationStereo | `https://github.com/NVlabs/FoundationStereo.git` | `6e8806816b533e4d13ddbb95ffa907b797060a62` |
| Depth Anything V2 | `https://github.com/DepthAnything/Depth-Anything-V2.git` | `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` (legacy ablation only) |
| SPIDER base | `https://github.com/facebookresearch/spider.git` | `71238456bf97a7eeb3d0471aa31974e2d404d4ae` |

Example source setup, assuming this repository is at `$EGOENGINE_ROOT/video_to_spider`:

```bash
export EGOENGINE_ROOT=/path/to/egoengine
export REPO_ROOT="$EGOENGINE_ROOT/video_to_spider"
mkdir -p "$REPO_ROOT/third_party"

git clone https://github.com/facebookresearch/sam3.git "$REPO_ROOT/third_party/sam3"
git -C "$REPO_ROOT/third_party/sam3" checkout --detach 6dbb02bd38288df755dfa1378000a861e65b84f6

git clone https://github.com/facebookresearch/sam-3d-objects.git "$REPO_ROOT/third_party/sam-3d-objects"
git -C "$REPO_ROOT/third_party/sam-3d-objects" checkout --detach f91db411c50efee93d8db7aeb323885650f6f722

git clone https://github.com/NVlabs/FoundationPose.git "$REPO_ROOT/third_party/FoundationPose"
git -C "$REPO_ROOT/third_party/FoundationPose" checkout --detach a1b694b83e633c2cb6115b9063d940a687759392

git clone https://github.com/rolpotamias/WiLoR.git "$REPO_ROOT/third_party/WiLoR"
git -C "$REPO_ROOT/third_party/WiLoR" checkout --detach fcb911312a38fa8badd30d9656a167485d61b8f9

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$REPO_ROOT/third_party/Depth-Anything-3"
git -C "$REPO_ROOT/third_party/Depth-Anything-3" checkout --detach 3d835ec

git clone https://github.com/lpiccinelli-eth/UniDepth.git "$REPO_ROOT/third_party/UniDepth"
git -C "$REPO_ROOT/third_party/UniDepth" checkout --detach 8d8cfe4

git clone https://github.com/NVlabs/FoundationStereo.git "$REPO_ROOT/third_party/FoundationStereo"
git -C "$REPO_ROOT/third_party/FoundationStereo" checkout --detach 6e8806816b533e4d13ddbb95ffa907b797060a62

# Legacy depth ablation only:
git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$REPO_ROOT/third_party/Depth-Anything-V2"
git -C "$REPO_ROOT/third_party/Depth-Anything-V2" checkout --detach a561b849ebae10a6f5ef49e26c83cbbcd36c71bf

git clone https://github.com/facebookresearch/spider.git "$EGOENGINE_ROOT/spider"
git -C "$EGOENGINE_ROOT/spider" checkout --detach 71238456bf97a7eeb3d0471aa31974e2d404d4ae
git -C "$EGOENGINE_ROOT/spider" am \
  "$REPO_ROOT/patches/0001-Make-frozen-SPIDER-lock-reproducible-from-public-PyP.patch"
```

The SPIDER patch changes inaccessible internal CodeArtifact URLs to public PyPI files while preserving package names and versions. After applying it:

```text
uv.lock SHA256: 23b98373fd8e0d0664871e4bcbaa9389009c06276f7aacb5d56afc946c9e5b85
```

## Why the environments are separate

The upstream CUDA stacks have incompatible binary requirements. The verified matrix is:

| Environment | Purpose | Key verified versions |
|---|---|---|
| `v2s-core` | ingest, orchestration, export, reports, tests | Python 3.11.15; NumPy 1.26.4; no PyTorch |
| `v2s-sam3` | SAM 3.1 segmentation/tracking | Python 3.12.13; PyTorch 2.10.0+cu128; torchvision 0.25; NumPy 1.26.4 |
| `v2s-wilor` | hand detection and MANO reconstruction | Python 3.10.20; PyTorch 2.0.0+cu117; torchvision 0.15.1; Ultralytics 8.1.34 |
| `v2s-depth` | metric Depth Anything V2 | Python 3.10.20; PyTorch 2.5.1+cu124; NumPy 2.2.6 |
| dedicated DA3 env | primary DA3 metric depth | official DA3 dependencies; package separately before release |
| dedicated UniDepth env | reject-only depth cross-check | official UniDepth dependencies; package separately before release |
| dedicated FoundationStereo env | calibrated stereo metric depth | official dependencies; pinned checkout is read-only and called through our adapter |
| `v2s-sam3d` | SAM 3D mesh proposals | Python 3.11; PyTorch 2.5.1+cu121; xFormers 0.0.28.post3; FlashAttention 2.8.3; spconv-cu121 2.3.8 |
| `v2s-foundationpose` | 6D registration and tracking | Python 3.11.15; PyTorch 2.5.1+cu124; Warp 1.15; nvdiffrast 0.4 |
| `v2s-opt` | smoothing, scale, contact, QC | Python 3.11.15; PyTorch 2.5.1+cu121; NumPy 2.4.4 |
| SPIDER `.venv` | xHand IK, MuJoCo, MJWP | Python 3.12.13; PyTorch 2.11.0+cu130; MuJoCo 3.7.0; mujoco-warp 3.7.0.1; Warp 1.12.1 |

Do not install these packages into base. Use `conda run -n <environment>` for every Conda stage and SPIDER's own uv environment for IK/MJWP.

## Restoring Conda environments

The validated internal deployment has six conda-pack archives in `$EGOENGINE_ROOT/env`:

```text
v2s-core.tar.gz
v2s-depth.tar.gz
v2s-foundationpose.tar.gz
v2s-opt.tar.gz
v2s-sam3d.tar.gz
v2s-wilor.tar.gz
```

Each archive must be extracted into its own empty Conda prefix, never into base or the source repository. Example:

```bash
conda create -y -n v2s-core
tar -xzf "$EGOENGINE_ROOT/env/v2s-core.tar.gz" \
  -C /path/to/miniconda/envs/v2s-core
conda run -n v2s-core conda-unpack
```

Repeat with the matching environment/archive names. These multi-GB archives are deployment assets and are not suitable for normal GitHub storage; publish them separately with a SHA256 manifest or rebuild from the upstream projects.

SAM3 follows the official repository instructions rather than the six archived environments:

```bash
conda create -y -n v2s-sam3 python=3.12
conda run -n v2s-sam3 python -m pip install \
  torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
conda run -n v2s-sam3 python -m pip install -e "$REPO_ROOT/third_party/sam3"
```

WiLoR uses PyTorch 2.0 and must not be launched with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Restoring SPIDER

Use uv 0.12.3 or a compatible newer uv after applying the tracked patch:

```bash
cd "$EGOENGINE_ROOT/spider"
/path/to/uv sync --frozen
/path/to/uv lock --check
```

Runtime commands use `uv run --frozen --no-sync`, so they neither re-resolve nor change the environment.

## Required model assets

Model weights are intentionally excluded from Git. Official sources and download commands are recorded in [模型权重下载来源.md](../模型权重下载来源.md). At minimum, provide these files through the official upstream download procedures or an access-controlled model store:

```text
third_party/sam3/checkpoints/sam3.1_multiplex.pt
third_party/WiLoR/pretrained_models/wilor_final.ckpt
third_party/WiLoR/pretrained_models/model_config.yaml
third_party/WiLoR/pretrained_models/detector.pt
third_party/WiLoR/mano_data/MANO_RIGHT.pkl
third_party/WiLoR/mano_data/mano_mean_params.npz
third_party/depth-checkpoints/DA3METRIC-LARGE/config.json
third_party/depth-checkpoints/DA3METRIC-LARGE/model.safetensors
third_party/depth-checkpoints/unidepth-v2-vitl14/config.json
third_party/depth-checkpoints/unidepth-v2-vitl14/model.safetensors
third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml
third_party/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth
third_party/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth
```

Two automatically cached SAM3D dependencies were explicitly checked:

| Model | Size | SHA256 |
|---|---:|---|
| MoGe ViT-L | 1,256,823,446 bytes | `da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f` |
| DINOv2 ViT-L/14 reg4 | 1,217,607,321 bytes | `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51` |

Do not commit credentials, `.env` files, Hugging Face tokens, datasets, model weights, Conda archives, or generated runs to a public repository.

## Release verification

Run the read-only preflight from the repository root:

```bash
EGOENGINE_ROOT=/path/to/egoengine GPU=7 bash scripts/check_release_preflight.sh
```

Run core tests without creating Python/test caches:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 conda run -n v2s-core \
  python -m pytest -q -p no:cacheprovider
```

Then follow the stage-by-stage commands in the main README with a new RUN_DIR. A release is accepted only when:

1. the repository and all dependency revisions match this document;
2. preflight and core tests pass from a clean checkout;
3. the run starts from a nonexistent RUN_DIR and uses one specified physical GPU;
4. every downstream stage has a valid accepted gate; a rejected gate is a valid terminal outcome and must not be bypassed;
5. only an accepted MINK q_ref may reach Replay/MJWP, and any generated MJWP video must be decodable;
6. metric quality and rejection reasons are reported separately from structural completion.

The final fresh validation took approximately 38 minutes for 54 frames with warm model/compile caches; FoundationPose alone took about 23 minutes. Budget 35–60 minutes for a similar 54–90 frame episode. First-time downloads and environment restoration are not included.
