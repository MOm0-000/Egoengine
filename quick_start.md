# Quick Start：从 EgoDex 生成最终 MJWP 视频

本页只保留已经在 `vertical_pick_place/111` 上完整验证过的命令。先完成 [快速配置环境.md](./快速配置环境.md)，并确保模型权重和 EgoDex 数据已经就位。

下面所有模型阶段严格串行，只使用一张指定 GPU。不要复用旧 `RUN_DIR`。

## 1. 设置路径

```bash
set -euo pipefail

export EGOENGINE_ROOT=/path/to/egoengine
export REPO_ROOT="$EGOENGINE_ROOT/video_to_spider"
export SPIDER_ROOT="$EGOENGINE_ROOT/spider"
export DATASET_ROOT=/path/to/egodex
export TASK=vertical_pick_place
export EPISODE_ID=111
export GPU=7
export RUN_DIR="$REPO_ROOT/runs/egodex_${TASK}_${EPISODE_ID}_$(date +%Y%m%d_%H%M%S)"

cd "$REPO_ROOT"
test ! -e "$RUN_DIR"
```

`DATASET_ROOT` 中应能找到该 episode 的 MP4/HDF5，例如 `part5/vertical_pick_place/111.mp4` 和 `111.hdf5`。

## 2. 运行 preflight

```bash
EGOENGINE_ROOT="$EGOENGINE_ROOT" GPU="$GPU" \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  bash scripts/check_release_preflight.sh
```

只有看到 `release preflight passed` 才继续。

## 3. 从原始 MP4/HDF5 串行运行

```bash
PYTHONNOUSERSITE=1 conda run -n v2s-core \
  python -m video_to_spider.cli ingest \
  --root "$DATASET_ROOT" \
  --task "$TASK" \
  --episode-id "$EPISODE_ID" \
  --output-dir "$RUN_DIR"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 \
  ./scripts/run_model_adapter.sh v2s-sam3 video_to_spider.adapters.sam3 \
  --run-dir "$RUN_DIR" \
  --checkpoint "$REPO_ROOT/third_party/sam3/checkpoints/sam3.1_multiplex.pt" \
  --max-candidates 12 \
  --max-instances 2

CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 \
  ./scripts/run_model_adapter.sh v2s-wilor video_to_spider.adapters.wilor \
  --run-dir "$RUN_DIR" \
  --checkpoint "$REPO_ROOT/third_party/WiLoR/pretrained_models/wilor_final.ckpt" \
  --model-config "$REPO_ROOT/third_party/WiLoR/pretrained_models/model_config.yaml" \
  --detector-checkpoint "$REPO_ROOT/third_party/WiLoR/pretrained_models/detector.pt"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  ./scripts/run_model_adapter.sh v2s-depth video_to_spider.adapters.depth_anything \
  --run-dir "$RUN_DIR" \
  --checkpoint "$REPO_ROOT/third_party/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth" \
  --encoder vitl \
  --input-size 518 \
  --max-depth 20

CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 \
  ./scripts/run_model_adapter.sh v2s-sam3d video_to_spider.adapters.sam3d_objects \
  --run-dir "$RUN_DIR" \
  --config-path "$REPO_ROOT/third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml" \
  --seeds 45 46 47 \
  --max-keyframes 1 \
  --max-proposals 3 \
  --low-vram \
  --moge-resolution-level 4 \
  --max-slat-coords 20000

CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 \
  ./scripts/run_model_adapter.sh v2s-foundationpose video_to_spider.adapters.foundationpose \
  --run-dir "$RUN_DIR" \
  --foundationpose-root "$REPO_ROOT/third_party/FoundationPose" \
  --max-candidates 3 \
  --screening-radius 1 \
  --register-iter 2 \
  --track-iter 1 \
  --max-input-side 640

CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 \
  conda run -n v2s-opt python -m video_to_spider.cli optimize \
  --run-dir "$RUN_DIR" \
  --allow-unvalidated-contact-scale \
  --contact-enter-distance-m 0.020 \
  --max-contact-slip-p95-m-s 0.45
```

上述三个优化参数是已验证 episode `vertical_pick_place/111` 的灵敏度配置。换数据时应先使用 README 中的严格默认值。

## 4. 生成诊断、SPIDER、IK 和 MJWP

```bash
PYTHONNOUSERSITE=1 conda run -n v2s-core \
  python -m video_to_spider.cli oracle-validate-extrinsics \
  --run-dir "$RUN_DIR" \
  --uses-ground-truth

PYTHONNOUSERSITE=1 conda run -n v2s-core \
  python -m video_to_spider.cli evaluate-wilor \
  --run-dir "$RUN_DIR" \
  --uses-ground-truth

PYTHONNOUSERSITE=1 conda run -n v2s-core \
  python -m video_to_spider.cli export-spider \
  --run-dir "$RUN_DIR" \
  --spider-package-root "$SPIDER_ROOT/spider" \
  --task "$TASK" \
  --data-id "$EPISODE_ID" \
  --embodiment-type right \
  --hand-sides right \
  --robot-type xhand

PYTHONNOUSERSITE=1 conda run -n v2s-core \
  python -m video_to_spider.cli run-spider \
  --dataset-root "$RUN_DIR/spider_export/dataset" \
  --spider-root "$SPIDER_ROOT" \
  --task "$TASK" \
  --data-id "$EPISODE_ID" \
  --embodiment-type right \
  --robot-type xhand \
  --gpu "$GPU"

PYTHONNOUSERSITE=1 conda run -n v2s-core \
  python -m video_to_spider.cli visualize-run \
  --run-dir "$RUN_DIR" \
  --overwrite

PYTHONNOUSERSITE=1 conda run -n v2s-core \
  python -m video_to_spider.cli evaluate-run \
  --run-dir "$RUN_DIR"
```

`visualize-run` 必须带 `--overwrite`，因为 SAM3D/FoundationPose 已在同一次新运行中生成部分诊断视频；这不是复用旧缓存。

## 5. 验收并取得最终视频

```bash
export REPORT="$RUN_DIR/evaluation/unified_run_report.json"
export MJWP_VIDEO="$RUN_DIR/spider_export/dataset/processed/video_to_spider_egodex/xhand/right/$TASK/$EPISODE_ID/visualization_mjwp.mp4"

test -s "$REPORT"
test -s "$MJWP_VIDEO"

PYTHONNOUSERSITE=1 conda run -n v2s-core python -c '
import json, sys
d = json.load(open(sys.argv[1]))
c = d["completion"]
assert c["available_stage_count"] == c["total_stage_count"] == 12
assert c["missing_stages"] == [] and c["invalid_stages"] == []
assert c["spider_chain_complete"]
assert c["simulation_video_complete"]
assert c["m4_complete"]
print(json.dumps(c, indent=2))
' "$REPORT"

env -u DEBUG -u debug ffmpeg -nostdin -v error \
  -i "$MJWP_VIDEO" -map 0:v:0 -f null -

printf '最终视频：%s\n' "$MJWP_VIDEO"
```

已验证的 54 帧 episode 在模型、环境和编译缓存就绪时约需 38 分钟；类似 54–90 帧 episode 建议预留 35–60 分钟。
