# Quick Start：从 EgoDex 运行带硬 gate 的自动流程

本页给出当前默认路径。它不会保证每个视频都生成 MJWP：任一 depth、tracking、sequence 或 MINK gate 拒绝时必须停止。`vertical_pick_place/111` 的新跟踪 gate 已拒绝；当前无 GT 上游通过样本是 `basic_pick_place/0`，其 MINK 离散碰撞/关节可行性已通过，但 Eq.(1) 对齐修复后的 q_ref 仍会因指尖位置和完整 SO(3) 姿态保真度不足被拒绝。

下面所有模型阶段严格串行，只使用一张指定 GPU。不要复用旧 `RUN_DIR`。

## 1. 设置路径

```bash
set -euo pipefail

export EGOENGINE_ROOT=/path/to/egoengine
export REPO_ROOT="$EGOENGINE_ROOT/video_to_spider"
export SPIDER_ROOT="$EGOENGINE_ROOT/spider"
export DATASET_ROOT=/path/to/egodex
export TASK=basic_pick_place
export EPISODE_ID=0
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
  /path/to/da3-env/python -m video_to_spider.adapters.depth_da3 \
  --run-dir "$RUN_DIR" \
  --model-root "$REPO_ROOT/third_party/Depth-Anything-3" \
  --checkpoint "$REPO_ROOT/third_party/depth-checkpoints/DA3METRIC-LARGE"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 \
  /path/to/unidepth-env/python -m video_to_spider.adapters.depth_unidepth \
  --run-dir "$RUN_DIR" \
  --model-root "$REPO_ROOT/third_party/UniDepth" \
  --checkpoint "$REPO_ROOT/third_party/depth-checkpoints/unidepth-v2-vitl14"

PYTHONNOUSERSITE=1 conda run -n v2s-core \
  python -m video_to_spider.adapters.depth_gate --run-dir "$RUN_DIR"

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
  --run-dir "$RUN_DIR"
```

全部 episode 使用相同默认参数；禁止按 data_id 放宽 gate。

### 已标定双目替换单目深度阶段

双目路线目前已接通但仍处于预注册 HOT3D/ZED promotion 实验；未通过全部数值门槛前，
上面的 DA3 路径仍是默认。合法双目输入先用官方设备工具完成同步、去畸变和极线校正，再运行：

```bash
PYTHONNOUSERSITE=1 conda run -n v2s-core \
  python -m video_to_spider.cli ingest-stereo \
  --left-dir "$LEFT_RECT" --right-dir "$RIGHT_RECT" \
  --intrinsics "$K_RECT_LEFT" --right-intrinsics "$K_RECT_RIGHT" \
  --common-valid-mask "$STEREO_COMMON_VALID" \
  --baseline-m "$BASELINE_M" \
  --camera-poses "$T_WORLD_CAMERA" --timestamps "$TIMESTAMPS" \
  --frame-indices "$FRAME_INDICES" --fps "$FPS" \
  --task "$TASK" --episode-id "$EPISODE_ID" \
  --instruction "$INSTRUCTION" --output-dir "$RUN_DIR"

# 先运行上方同一条 SAM 3 命令，生成自动物体/手 mask；双目 gate 必须检查物体区域 coverage。

CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 \
  /path/to/foundationstereo-env/python \
  -m video_to_spider.adapters.depth_foundationstereo \
  --run-dir "$RUN_DIR" --repository /path/to/official/FoundationStereo \
  --checkpoint /path/to/23-51-11/model_best_bp2.pth \
  --config /path/to/23-51-11/cfg.yaml \
  --physical-gpu-index "$GPU" --gpu-uuid "$GPU_UUID"

PYTHONNOUSERSITE=1 conda run -n v2s-core \
  python -m video_to_spider.adapters.depth_gate --run-dir "$RUN_DIR"
```

双目 adapter 只调用固定提交的未修改官方模型；发现官方 tracked 源码被改、权重哈希不符、
标定不一致或目标 GPU 已有进程就拒绝启动。左右图还必须分别运行 WiLoR，并用
`python -m video_to_spider.adapters.hand_stereo --run-dir "$RUN_DIR"` 生成独立标定三角化手部；
优化时传入 `--hand-artifact "$RUN_DIR/hands/wilor_stereo_raw.npz"`。后续 `optimize` 的默认
`auto` 会自动冻结双目 metric 几何并只验证接触，不再做 contact similarity target rewrite。
若只提供单目 WiLoR 深度，双目 sequence gate 会拒绝导出。

## 4. 生成诊断、SPIDER、IK 和 MJWP

```bash
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

## 5. 验收

```bash
export REPORT="$RUN_DIR/evaluation/unified_run_report.json"
export MJWP_VIDEO="$RUN_DIR/spider_export/dataset/processed/video_to_spider_egodex/xhand/right/$TASK/$EPISODE_ID/visualization_mjwp.mp4"

test -s "$REPORT"
if test -s "$MJWP_VIDEO"; then
  env -u DEBUG -u debug ffmpeg -nostdin -v error \
    -i "$MJWP_VIDEO" -map 0:v:0 -f null -
  printf '最终视频：%s\n' "$MJWP_VIDEO"
else
  find "$RUN_DIR" \( -name '*gate.json' -o -name '*_metrics.json' \) -print
  printf '流程被 gate 拒绝；检查上面的 JSON，禁止绕过。\n'
fi
```

只有 MINK gate 与后续 Replay/MPC 均成功时，`MJWP_VIDEO` 才应存在。gate 拒绝时应保留对应 JSON/NPZ 诊断，不要通过删 gate、换原生 IK 或放宽阈值继续生成视频。
