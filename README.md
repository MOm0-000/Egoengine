# Video-to-SPIDER

将单目或已标定双目 ego RGB episode 转换为 SPIDER/xHand 可执行轨迹的 artifact-oriented pipeline。

当前流程读取 RGB、相机内参、逐帧相机外参和任务文本；双目输入还读取同步的右图和已标定基线。随后依次完成目标分割、手部重建、深度估计、物体网格重建、6D 位姿跟踪、手物联合优化、SPIDER 导出、IK 和 MJWP 仿真。手部 Ground Truth 只允许用于最后的显式诊断，不进入推理链路。

> 当前仓库没有 `run-all` 命令。完整流程必须按本文的 artifact 边界逐步执行。每一步都保存数值 artifact、metadata 和可视化，便于在进入下一步前验证。

当前安全基线不再把“命令跑完”视为复现成功。单目使用 DA3/UniDepth gate；双目使用标定、极线和 FoundationStereo 原生 metric gate。FoundationPose、sequence optimization 和 MINK 也有固定 reject gate；任一 gate 失败时不得继续 Replay/MPC。双目 promotion 的 HOT3D/ZED 对比仍在进行，尚未因“代码已接通”而宣布替换单目默认值。`vertical_pick_place/111` 的新 FoundationPose 轨迹因旋转跳变被拒绝；`basic_pick_place/0` 已通过无 GT 上游 gate，但 Eq.(1) 对齐修复后的 MINK q_ref 仍因指尖位置和完整 SO(3) 姿态保真度不足被拒绝。当前核心 video-to-SPIDER 测试为 `134 passed`，另有 11 项 MINK 单元测试通过；这表示 gate 和 artifact 接线可用，不表示抓取问题已经解决。

首次使用建议按以下顺序阅读：

1. [快速配置环境.md](./快速配置环境.md)：逐行配置源码、Conda 环境、SPIDER 和模型资产。
2. [quick_start.md](./quick_start.md)：从原始 MP4/HDF5 串行生成最终 MJWP 视频的最短命令清单。
3. 本 README：逐阶段质量检查、重跑和故障定位。

## 流程总览

```text
Ego RGB + calibration + camera trajectory
  -> monocular ingest OR synchronized rectified stereo ingest
  -> SAM 3 object/hand segmentation
  -> WiLoR hand reconstruction
  -> mono: DA3METRIC-LARGE + UniDepthV2 reject-only gate
     OR stereo: unmodified FoundationStereo + calibrated raw metric gate
  -> SAM 3D Objects mesh proposals + static metric scale refit
  -> FoundationPose proposal selection, tracking and full-track gate
  -> sequence optimization and contact inference
  -> SPIDER export -> paper-style MINK q_ref gate
  -> deterministic Replay -> failed Replay windows escalate to SPIDER/MJWP MPC
  -> visualization and unified evaluation
```

主要输出：

```text
runs/<run-id>/
  input/                         输入来源和任务文本
  frames/                        RGB 帧和统一时间轴
  calibration/                   K 和 T_world_camera
  segmentation/                  SAM 3 mask、候选指标和 overlay
  hands/                         WiLoR MANO/joints 和 overlay
  depth/                         metric depth Zarr 和视频
  mesh_proposals/                SAM 3D 网格候选和排名
  object_tracking/               FoundationPose 轨迹和选中网格
  optimization/                  对齐轨迹、接触和手角色
  spider_export/                 SPIDER dataset、IK、MJWP 和仿真视频
  visualization/                 三类自动诊断视频
  evaluation/                    GT 诊断和统一报告
  manifest.json                  各阶段命令、状态、输出和质量指标
```

## 环境和路径

以下命令假设在仓库根目录执行：

```bash
export EGOENGINE_ROOT=/path/to/egoengine
export REPO_ROOT="$EGOENGINE_ROOT/video_to_spider"
export SPIDER_ROOT="$EGOENGINE_ROOT/spider"
export SPIDER_PACKAGE_ROOT="$SPIDER_ROOT/spider"
cd "$REPO_ROOT"
```

当前机器使用彼此隔离的 Conda 环境：

| 阶段 | 环境 |
|---|---|
| ingest、导出、评估、可视化 | `v2s-core` |
| SAM 3 | `v2s-sam3` |
| WiLoR | `v2s-wilor` |
| DA3METRIC-LARGE 主深度 | 独立 DA3 环境 |
| UniDepthV2 reject-only 复核 | 独立 UniDepth 环境 |
| FoundationStereo 双目深度 | 官方 FoundationStereo 独立环境；仓库必须保持只读 |
| SAM 3D Objects | `v2s-sam3d` |
| FoundationPose | `v2s-foundationpose` |
| 序列优化 | `v2s-opt` |
| SPIDER/MJWP | `$SPIDER_ROOT/.venv`，由 SPIDER 的 `pyproject.toml` 与修正后的 `uv.lock` 管理 |

不同模型需要互不兼容的 PyTorch/CUDA/NumPy 组合，不能可靠地压进一个环境。完整恢复方式、已有 conda-pack 归档与 SAM3 官方安装方式见 [docs/REPRODUCIBILITY.md](./docs/REPRODUCIBILITY.md)。运行项目命令时只使用 `conda run -n <env>`，不要向 base 安装依赖。

先做只读 preflight：

```bash
conda env list

test -s third_party/sam3/checkpoints/sam3.1_multiplex.pt
test -s third_party/WiLoR/pretrained_models/wilor_final.ckpt
test -s third_party/WiLoR/pretrained_models/model_config.yaml
test -s third_party/WiLoR/pretrained_models/detector.pt
test -d third_party/Depth-Anything-3
test -d third_party/UniDepth
test -d third_party/FoundationStereo
test -s third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml
test -s third_party/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth
test -s third_party/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth
test -d "$SPIDER_ROOT/.venv"
command -v ffmpeg
command -v ffprobe

nvidia-smi -i "${GPU:-7}"
```

核心环境 import 和测试：

```bash
PYTHONNOUSERSITE=1 conda run -n v2s-core python -c "import cv2,h5py,numpy,scipy,trimesh,zarr; print('v2s-core ok')"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 conda run -n v2s-core \
  python -m pytest -q -p no:cacheprovider
```

仓库生成的诊断 MP4 统一通过系统 `ffmpeg` 编码为 H.264 (`libx264`)、
`yuv420p`，并启用 faststart，便于在浏览器和常用播放器中直接查看。

## 完整执行示例

下面使用已验证的 `vertical_pick_place/111` 完整 episode。每次正式验收必须使用此前不存在的新 RUN_DIR：

```bash
DATASET_ROOT=/data_all/share/datasets/egodex
TASK=vertical_pick_place
EPISODE_ID=111
RUN_DIR=runs/egodex_vertical_pick_place_111_release_<唯一后缀>
GPU=7
test ! -e "$RUN_DIR"
```

如果使用其他数据，只需要修改 `TASK`、`EPISODE_ID`、`RUN_DIR` 和 `GPU`。不要复用已有 `RUN_DIR`，除非明确希望用 `--overwrite` 重跑。

### 1. 扫描和导入 EgoDex

确认数据根目录可被扫描：

```bash
conda run -n v2s-core python -m video_to_spider.cli scan-egodex \
  --root "$DATASET_ROOT"
```

导入完整视频：

```bash
conda run -n v2s-core python -m video_to_spider.cli ingest \
  --root "$DATASET_ROOT" \
  --task "$TASK" \
  --episode-id "$EPISODE_ID" \
  --output-dir "$RUN_DIR" \
  --overwrite
```

只跑一个帧区间时使用半开区间 `[start, end)`：

```bash
conda run -n v2s-core python -m video_to_spider.cli ingest \
  --root "$DATASET_ROOT" \
  --task "$TASK" \
  --episode-id "$EPISODE_ID" \
  --output-dir "$RUN_DIR" \
  --start-frame 0 \
  --end-frame 90 \
  --overwrite
```

检查任务文本、候选词、帧数和相机输入：

```bash
jq '{task_directory,episode_id,instruction_text,object_keyword_candidates,video,selected_frame_interval}' \
  "$RUN_DIR/input/source.json"

jq '.stages.ingest' "$RUN_DIR/manifest.json"
```

成功输出：

```text
input/source.json
frames/frame_index.json
frames/rgb/*.jpg
calibration/intrinsics.npy
calibration/T_world_camera.npy
manifest.json
```

#### 已标定双目输入（实验接入，promotion 尚未完成）

双目路径要求上游官方相机工具已经完成去畸变、同步和极线校正；通用 pipeline 不修改设备模型，
也不把原始 fisheye 图像伪装成 pinhole 双目。移动的 ego 相机必须提供逐帧
`T_world_camera`，静态测试台才允许显式使用 `--static-camera`：

```bash
conda run -n v2s-core python -m video_to_spider.cli ingest-stereo \
  --left-dir /path/to/rectified/left \
  --right-dir /path/to/rectified/right \
  --intrinsics /path/to/K_rect.npy \
  --right-intrinsics /path/to/K_rect_right.npy \
  --common-valid-mask /path/to/official_rectified_common_valid.npy \
  --baseline-m 0.0636 \
  --camera-poses /path/to/T_world_camera.npy \
  --timestamps /path/to/timestamps_s.json \
  --frame-indices /path/to/frame_indices.json \
  --task "$TASK" --episode-id "$EPISODE_ID" \
  --instruction "pick up the object" --fps 30 \
  --output-dir "$RUN_DIR"
```

导入阶段会用左右各自 K 的归一化光线抽样审计分辨率、正视差比例和垂直视差；还要求官方相机
工具导出的共同有效域 mask。只有能确认校正图没有黑边/裁剪无效区时，才可改用显式
`--full-image-common-valid`；不通过时拒绝创建可用 run。
`calibration/stereo.json` 记录标定与 gate，左图继续作为 SAM/手部/物体跟踪的参考相机。

先按下文 SAM 3 阶段生成自动物体 mask；双目 gate 会把它作为必须的、带哈希的覆盖率证据。
然后 FoundationStereo 通过自有薄适配层调用固定提交的官方仓库。适配层会拒绝修改过的官方 tracked
源码、错误权重哈希、标定不一致以及启动时已有进程的目标 GPU；不会结束或抢占已有进程：

```bash
CUDA_VISIBLE_DEVICES="$GPU" /path/to/foundationstereo-env/python \
  -m video_to_spider.adapters.depth_foundationstereo \
  --run-dir "$RUN_DIR" \
  --repository /path/to/official/FoundationStereo \
  --checkpoint /path/to/23-51-11/model_best_bp2.pth \
  --config /path/to/23-51-11/cfg.yaml \
  --physical-gpu-index "$GPU" \
  --gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

conda run -n v2s-core python -m video_to_spider.adapters.depth_gate \
  --run-dir "$RUN_DIR"
```

FoundationStereo 要求官方相机工具把左右图重投影到同一个 rectified K；若两个校正 K 不同，
适配器会拒绝，而不会把普通像素差误当成视差。深度直接使用
`fx_rect × baseline / disparity`，禁止 GT scale/shift 和逐视频补偿。双目 gate
同时检查全图和自动物体 mask 内的有效深度覆盖；这样操作物体落在左图无视差边界时，即使全图
coverage 很高也会拒绝。gate 通过后，还需在同一组校正左右图分别运行未修改的 WiLoR，再用
标定几何独立三角化手部：

```bash
# 左图：默认输出 hands/wilor_raw.npz
CUDA_VISIBLE_DEVICES="$GPU" ./scripts/run_model_adapter.sh \
  v2s-wilor video_to_spider.adapters.wilor \
  --run-dir "$RUN_DIR" --camera-view left \
  --checkpoint third_party/WiLoR/pretrained_models/wilor_final.ckpt \
  --model-config third_party/WiLoR/pretrained_models/model_config.yaml \
  --detector-checkpoint third_party/WiLoR/pretrained_models/detector.pt

# 右图：默认输出 hands_right/wilor_raw.npz
CUDA_VISIBLE_DEVICES="$GPU" ./scripts/run_model_adapter.sh \
  v2s-wilor video_to_spider.adapters.wilor \
  --run-dir "$RUN_DIR" --camera-view right \
  --checkpoint third_party/WiLoR/pretrained_models/wilor_final.ckpt \
  --model-config third_party/WiLoR/pretrained_models/model_config.yaml \
  --detector-checkpoint third_party/WiLoR/pretrained_models/detector.pt

conda run -n v2s-core python -m video_to_spider.adapters.hand_stereo \
  --run-dir "$RUN_DIR"
```

该手部适配层只使用左右 2D joint rays、K 和 baseline，不读取物体、接触或 GT；固定检查正视差、
垂直误差、重投影和覆盖率。优化时显式传入
`--hand-artifact "$RUN_DIR/hands/wilor_stereo_raw.npz"`。双目
`optimize --contact-similarity-mode auto` 自动解析为 `validate_only`：接触仅评分/验收，不得再用
物体深度或接触相似变换重写人手/物体 metric 轨迹。没有通过双目手 gate 的 WiLoR 单目深度会被
明确拒绝，不能悄悄进入 MINK。最终 MuJoCo 物理验收仍必须保留。

### 2. SAM 3 目标和手分割

使用任务文本派生出的全部候选词。`keyword_candidates` 最多产生 12 个词，因此正式运行建议 `--max-candidates 12`；较小值只适合 smoke test，可能漏掉真正物体词。

```bash
CUDA_VISIBLE_DEVICES="$GPU" ./scripts/run_model_adapter.sh \
  v2s-sam3 video_to_spider.adapters.sam3 \
  --run-dir "$RUN_DIR" \
  --checkpoint third_party/sam3/checkpoints/sam3.1_multiplex.pt \
  --max-candidates 12 \
  --max-instances 2 \
  --overwrite
```

检查选中的文本和所有候选指标：

```bash
jq '{selected_prompt,hand_metrics,candidates:[.candidates[]|{prompt,target_object_id,metrics}]}' \
  "$RUN_DIR/segmentation/metadata.json"
```

进入 SAM 3D 前必须检查：

- `selected_prompt` 是被搬运物体，而不是目标容器。例如应搬运 `block` 时，不能接受 `bowl`。
- 选中候选的 `metrics.valid_rate` 应大于 0，最好接近 1。
- `segmentation/perception_overlay.mp4` 中绿色目标 mask 应跟随正确物体，红色区域应覆盖画面中的手。

当前自动排名偏好大且稳定的 mask，可能把 `bowl`、`dustpan` 等放置容器排在真正被搬运物体之前。语义不正确时应停止该 episode；后续数值成功不能修复错误物体选择。

成功输出：

```text
segmentation/object_masks.npz
segmentation/hand_masks.npz
segmentation/metadata.json
segmentation/perception_overlay.mp4
```

### 3. WiLoR 双手重建

WiLoR 固定保存 `[left, right]` 两个 observation 槽位；没有检测到的手会保留为 invalid，优化阶段再决定是否降级为单手。

```bash
CUDA_VISIBLE_DEVICES="$GPU" ./scripts/run_model_adapter.sh \
  v2s-wilor video_to_spider.adapters.wilor \
  --run-dir "$RUN_DIR" \
  --checkpoint third_party/WiLoR/pretrained_models/wilor_final.ckpt \
  --model-config third_party/WiLoR/pretrained_models/model_config.yaml \
  --detector-checkpoint third_party/WiLoR/pretrained_models/detector.pt \
  --overwrite
```

检查左右手有效率和 overlay：

```bash
jq '{frame_count,valid_rate_left,valid_rate_right,detections_per_frame}' \
  "$RUN_DIR/hands/metadata.json"
```

成功输出：

```text
hands/wilor_raw.npz
hands/metadata.json
hands/wilor_overlay.mp4
```

### 4. 单目 DA3 主深度、UniDepth 独立复核和 gate

主深度固定使用 `DA3METRIC-LARGE`；UniDepthV2 只做 reject-only 复核，不允许重标定或混合 DA3。两个模型分别在官方依赖环境中运行：

```bash
CUDA_VISIBLE_DEVICES="$GPU" python -m video_to_spider.adapters.depth_da3 \
  --run-dir "$RUN_DIR" \
  --model-root third_party/Depth-Anything-3 \
  --checkpoint third_party/depth-checkpoints/DA3METRIC-LARGE \
  --overwrite

CUDA_VISIBLE_DEVICES="$GPU" python -m video_to_spider.adapters.depth_unidepth \
  --run-dir "$RUN_DIR" \
  --model-root third_party/UniDepth \
  --checkpoint third_party/depth-checkpoints/unidepth-v2-vitl14 \
  --overwrite

conda run -n v2s-core python -m video_to_spider.adapters.depth_gate \
  --run-dir "$RUN_DIR"
```

检查 metadata：

```bash
jq '{accepted,checks,metrics,limits}' "$RUN_DIR/depth/depth_gate.json"
```

成功输出：

```text
depth/metric_depth.zarr/
depth/metadata.json
depth/metric_depth.mp4
depth_crosscheck/unidepth_depth.zarr/
depth/depth_gate.json
```

缺失或拒绝的 `depth_gate.json` 会阻止 SAM3D 尺度拟合和后续跟踪。DA3 是唯一主深度；UniDepth 不写回其尺度。

### 5. SAM 3D Objects 网格候选

下面是共享 40 GB A100 上验证过的 low-VRAM 配置。3 个 seed 会生成 3 个真实网格候选：

```bash
CUDA_VISIBLE_DEVICES="$GPU" ./scripts/run_model_adapter.sh \
  v2s-sam3d video_to_spider.adapters.sam3d_objects \
  --run-dir "$RUN_DIR" \
  --config-path third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml \
  --seeds 45 46 47 \
  --max-keyframes 1 \
  --max-proposals 3 \
  --low-vram \
  --moge-resolution-level 4 \
  --max-slat-coords 20000 \
  --overwrite
```

检查候选是否合格：

```bash
jq '{success,qualified_count,failure_reason,keyframes,proposals:[.proposals[]|{proposal_id,qualified,rank,static_score}]}' \
  "$RUN_DIR/mesh_proposals/mesh_ranking.json"
```

成功输出：

```text
mesh_proposals/mesh_ranking.json
mesh_proposals/proposal_*/raw.glb
mesh_proposals/proposal_*/visual.obj
mesh_proposals/proposal_*/collision_source.obj
visualization/05_mesh_proposals.mp4
```

如果报错 `mesh_failed: no valid SAM object keyframe`，先检查 SAM 3 选中候选的 `valid_rate`。所有候选都是 0 时应停止，不要通过降低 SAM 3D 阈值绕过无效输入。

### 6. FoundationPose 候选筛选和 6D 跟踪

FoundationPose 会尝试最多 3 个网格候选，并根据 mask IoU、相对深度、跳变和跟踪有效率选择最终网格：

```bash
CUDA_VISIBLE_DEVICES="$GPU" ./scripts/run_model_adapter.sh \
  v2s-foundationpose video_to_spider.adapters.foundationpose \
  --run-dir "$RUN_DIR" \
  --foundationpose-root third_party/FoundationPose \
  --max-candidates 3 \
  --screening-radius 1 \
  --register-iter 2 \
  --track-iter 1 \
  --max-input-side 640 \
  --overwrite
```

检查跟踪质量：

```bash
jq '.' "$RUN_DIR/object_tracking/tracking_metrics.json"
jq '.' "$RUN_DIR/object_tracking/selected_mesh.json"
```

重点字段：

```text
metrics.valid_rate
metrics.mean_mask_iou
metrics.translation_jump_p95_m
metrics.rotation_jump_p95_rad
metrics.registration_count
metrics.tracking_score
```

成功输出：

```text
object_tracking/foundationpose_raw.npz
object_tracking/selected_mesh.json
object_tracking/tracking_metrics.json
visualization/06_foundationpose.mp4
```

### 7. 序列优化、接触和单/双手判定

优化器完成物体尺度、轨迹平滑、可观测手腕/末节方向、手物接触和手角色判定：

```bash
conda run -n v2s-opt python -m video_to_spider.cli optimize \
  --run-dir "$RUN_DIR" \
  --overwrite
```

正式自动流程使用固定默认值，不允许按 data_id 放宽。CLI 默认只导出具有持续接触证据的 active 手；`--include-passive-hands` 只用于可视化/诊断。

检查优化指标和最终保留的手：

```bash
jq '{hands,simulation_floor,anchor,optimization,raw_vs_aligned,contact,quality_control}' \
  "$RUN_DIR/optimization/optimization_metrics.json"
```

优化先以 FoundationPose/DA3 metric depth 作为物体深度证据，并以物体 mask 优化轮廓尺度；
如果连续多帧存在可靠的二维指尖/物体接触候选，则同步缩放物体网格和相机平移，搜索与
MANO 指尖最一致的三维表面尺度。同步缩放不会改变物体二维投影，但结果的绝对公制尺度会标记为
`contact_calibrated_not_externally_validated`，需要真实物体尺寸才能进一步验证。
WiLoR 的弱透视平移只有在完整手部投影仍处于二维信任域内时才允许校准。
只有 `quality_control.export_ready == true` 的结果允许进入 `export-spider`，避免二维轮廓改善但
手部二维对齐或 manipulation 接触退化的轨迹被导出。默认要求至少一只 active 手具有连续接触；
仅处理非操作视频时可显式增加 `--allow-no-contact`。

EgoDex 手部 GT 只允许作为显式 oracle 消融，不能静默混入普通 RGB 流程：

```bash
conda run -n v2s-opt python -m video_to_spider.cli optimize \
  --run-dir "$RUN_DIR" \
  --hand-source egodex_gt \
  --uses-ground-truth \
  --active-hands-only \
  --overwrite
```

缺少 `--uses-ground-truth` 时程序会拒绝运行该分支；metrics 和 manifest 均记录
`hand_source=egodex_gt` 与 `uses_ground_truth=true`。

角色规则：

- `active`：可靠可见，并具有持续二维接触证据，参与手物约束。
- `passive`：可靠可见但没有持续接触证据；默认不进入动作导出。
- `invalid`：重建有效率不足，不进入导出 artifact。
- mesh 的归一化缩放系数不是物体半径，不能用于“靠近即 active”的判定。

读取 SPIDER 应使用的手顺序：

```bash
jq -r '.hands.artifact_hand_order | join(" ")' \
  "$RUN_DIR/optimization/optimization_metrics.json"
```

可能结果及对应 embodiment：

| `artifact_hand_order` | `embodiment_type` | `hand-sides` |
|---|---|---|
| `left right` | `bimanual` | `left right` |
| `right` | `right` | `right` |
| `left` | `left` | `left` |

成功输出：

```text
optimization/aligned_trajectory.npz
optimization/contact.npz
optimization/optimization_metrics.json
visualization/07_raw_vs_aligned_contact.mp4
```

### 8. 根据手顺序设置 SPIDER embodiment

下面的 shell 片段自动读取优化结果，避免把单手 episode 错导出为双手：

```bash
ARTIFACT_HAND_ORDER=$(jq -r '.hands.artifact_hand_order | join(" ")' \
  "$RUN_DIR/optimization/optimization_metrics.json")

case "$ARTIFACT_HAND_ORDER" in
  "left right")
    EMBODIMENT_TYPE=bimanual
    HAND_SIDES=(left right)
    ;;
  "right")
    EMBODIMENT_TYPE=right
    HAND_SIDES=(right)
    ;;
  "left")
    EMBODIMENT_TYPE=left
    HAND_SIDES=(left)
    ;;
  *)
    echo "unsupported artifact hand order: $ARTIFACT_HAND_ORDER" >&2
    exit 1
    ;;
esac

printf 'embodiment=%s hands=%s\n' "$EMBODIMENT_TYPE" "${HAND_SIDES[*]}"
```

后续 export 和 run-spider 必须复用同一个 `EMBODIMENT_TYPE`。
`export-spider` 未显式传入 `--embodiment-type/--hand-sides` 时，也会根据
`artifact_hand_order` 自动推断；上面的变量仍应继续用于 `run-spider`。

### 9. 导出 SPIDER dataset

```bash
conda run -n v2s-core python -m video_to_spider.cli export-spider \
  --run-dir "$RUN_DIR" \
  --spider-package-root "$SPIDER_PACKAGE_ROOT" \
  --task "$TASK" \
  --data-id "$EPISODE_ID" \
  --embodiment-type "$EMBODIMENT_TYPE" \
  --hand-sides "${HAND_SIDES[@]}" \
  --robot-type xhand
```

导出前会检查：

- 请求的手不能是 `invalid`。
- 物体轨迹不能穿透地面。
- 手腕和指尖必须高于 xHand 地面安全高度。
- 手腕到指尖距离必须合理。
- manipulation 优化必须具有 active 手和连续有效接触。

检查导出 manifest：

```bash
find "$RUN_DIR/spider_export/dataset/processed/video_to_spider_egodex" \
  -name export_manifest.json -print
```

主要输出位于：

```text
spider_export/dataset/processed/video_to_spider_egodex/
  mano/<embodiment>/<task>/<id>/trajectory_keypoints.npz
  mano/<embodiment>/<task>/<id>/export_manifest.json
  assets/objects/<run-id>/
```

### 10. 运行 SPIDER、IK 和 MJWP

`run-spider` 内部会在 `$SPIDER_ROOT` 中使用现有 `uv` 环境，设置 EGL headless rendering，并依次运行以下阶段：

```text
01 decompose_fast
02 detect_contact
03 generate_xml
04 默认 MINK q_ref（五指位置、完整 SO(3) 姿态代理、腕向、限位和碰撞）+ 确定性 Replay
05 可选的实验性 contact-aware preshape（默认关闭）
06 Replay 失败时交给带真实 MuJoCo 接触/力闭合代价的 MJWP
07 CPU MuJoCo 独立复算接触、法向力、法向对置和穿透
```

默认 `--ik-backend mink`。MINK 使用五指观测位置、由 DIP-to-tip 轴与掌面法向构造的完整 SO(3) 指尖姿态代理，以及完整腕向；每个 xHand tip site 的 XML 局部轴先在中性位姿中标定，再映射到统一的几何指尖帧。接触标签不会改写 Eq.(1) 的人手指尖位置；旧的物体表面点重写仅保留为显式 `--use-object-contact-position-targets` 非论文诊断开关。`--lambda-w` 直接表示论文中的二次损失系数，代码向 MINK 传入其平方根，以抵消 MINK 对 `Task.cost` 的再次平方。完整指尖旋转的默认二次系数为 `1e-4`（等价于 `0.01 m/rad` 的残差尺度），`lambda_w=1`；这是米制位置与弧度姿态的量纲归一，不是按抓取模式设置的特例。每个插值目标提交前都会复算精确 MuJoCo signed distance；速度阻尼无法推出已有穿透时，固定预算的 recovery task 做离散投影，仍不可行就写 `mink_projection_audit.json` 并拒绝。`mink_qref_gate.json` 要求位置/完整指尖姿态/腕向、关节限位、自碰、手地、非末节手物碰撞和末节穿透全部通过；拒绝时只保存 `trajectory_mink_rejected.npz`，不会回退到原生 IK 后继续 MPC。

前一轮 fidelity 消融表明：非末节 `1 mm` clearance、接触末节 `2.5 mm` 穿透上限和每目标 4 次 IK 迭代都不是 basic0 位置失败的主因。随后完成的 Eq.(1) 对齐修复在 basic0/oracle111 上把位置 P95 从 21.87/20.74 mm 降到 16.92/15.38 mm，把腕向 P95 从 0.300/0.256 rad 降到 0.00134/0.00664 rad；但完整指尖姿态 P95 仍为 2.236/1.681 rad，严格 gate 因而继续拒绝。提高完整姿态系数会改善 DIP 方向却恶化位置，表明当前 landmark 姿态代理与 xHand 低维形态存在不可由单一权重消除的冲突。完整证据分别位于 `../experiments/pipeline_phase4_fidelity_ablation_20260811/` 和 `../experiments/pipeline_phase5_paper_objective_alignment_20260811/`。

默认固定 IK seed、关闭 Replay 噪声，并按 20 帧块和两块前瞻检查 Replay；所有窗口通过则跳过 MJWP。当前控制器仍是“任一窗口失败则整段升级 MJWP”，失败 chunk 单独切换与轨迹拼接尚未实现，报告会明确记录这一差距。MJWP 会保留命令行显式传入的
`data_path/model_path/output_dir`，不会再把预抓取轨迹静默改回 `trajectory_kinematic.npz`。
MJWP 不再只依赖指尖到参考点的运动学距离：它直接读取每个采样世界的 MuJoCo
`contact.geom/dist/frame`，并由 `contact.efc_address → efc.force` 还原法向接触力。在参考轨迹要求
抓握时，拇指和至少一根指定其他手指必须同时承载最小法向力、手到物体法向必须相对；手掌及
手指的过深物体穿透另行受罚。约束只惩罚违约且在阈值后饱和，不会通过无限增大握力刷奖励。
这是采样 MPC 的软约束；若当前动作分布内没有可行的持续接触 rollout，它会在指标中明确记录
违约，但不能凭空生成可行动作。因此最终仍必须以第 07 阶段的 CPU MuJoCo 复算为准。
导出前会将 episode 初始物体的支撑面对齐到仿真地面，同时保持手物相对几何关系。
优化器只有在手部公制深度校准通过后，才允许使用接触证据修正物体尺度。自动生产流程不得使用 `--allow-unvalidated-contact-scale`；该开关仅保留给有明确标注的消融实验。

执行完整链路并保存 IK/MJWP 视频：

```bash
conda run -n v2s-core python -m video_to_spider.cli run-spider \
  --dataset-root "$RUN_DIR/spider_export/dataset" \
  --spider-root "$SPIDER_ROOT" \
  --task "$TASK" \
  --data-id "$EPISODE_ID" \
  --embodiment-type "$EMBODIMENT_TYPE" \
  --robot-type xhand \
  --gpu "$GPU"
```

若要做原生 SPIDER IK 对照（不属于默认论文式流程），附加：

```bash
--ik-backend spider-native
```

实验性预抓取必须显式增加 `--contact-aware-preshape`；默认关闭，以避免复现 v120 的动作改写和按抓取模式过拟合。生产 MINK 当前只实现项目实际使用的 `xhand/right`，其他 robot/embodiment 会明确拒绝。

只验证到 IK、不运行 MJWP：

```bash
conda run -n v2s-core python -m video_to_spider.cli run-spider \
  --dataset-root "$RUN_DIR/spider_export/dataset" \
  --spider-root "$SPIDER_ROOT" \
  --task "$TASK" \
  --data-id "$EPISODE_ID" \
  --embodiment-type "$EMBODIMENT_TYPE" \
  --robot-type xhand \
  --gpu "$GPU" \
  --no-mjwp
```

检查阶段返回码、Replay/MPC 选择、MJWP 指标和真实接触：

```bash
SPIDER_REPORT="$RUN_DIR/spider_export/dataset/processed/video_to_spider_egodex/xhand/$EMBODIMENT_TYPE/$TASK/$EPISODE_ID/spider_run_report.json"

jq '{embodiment_type,commands:[.commands[]|{returncode,runtime_s,log}],mode_selection,mjwp_metrics,physical_interaction,demonstration_success,artifacts}' \
  "$SPIDER_REPORT"
```

当前质量阈值：

```text
mean object position error < 0.1 m
mean object rotation error < 0.5 rad
至少 3 个真实物理力闭合帧满足：
  拇指与至少一根其他手指法向力均 >= 0.2 N
  两侧手到物体接触法向 cosine <= -0.2
  最大手物穿透 <= 0.003 m
```

论文式慢配置的默认值为 horizon 1.6 s、ctrl_dt 0.08 s、2048 samples、16 iterations；
接触项包含逐指目标、运动学拇指对指、真实 MuJoCo 力闭合违约、穿透和单向抬升项。
这里的“力闭合”是适配两指对置抓取的摩擦接触代理判据，并非完整 6D grasp-wrench-space 证明。
RL fallback 尚未实现。所有阈值均可由 `run-spider --help` 中的
`--force-closure-*` 参数显式调整，正式批量评估应固定同一配置，不能按单个视频调参。

五个命令都返回 0 只代表链路完成，不代表目标语义正确，也不代表 MJWP 达到质量阈值。

### 11. 生成三类自动可视化

以下命令只读取已保存的 artifact，在 CPU 上重建三类诊断视频：

```bash
conda run -n v2s-core python -m video_to_spider.cli visualize-run \
  --run-dir "$RUN_DIR" \
  --overwrite
```

输出：

```text
visualization/05_mesh_proposals.mp4
visualization/06_foundationpose.mp4
visualization/07_raw_vs_aligned_contact.mp4
visualization/visualization_manifest.json
```

验证视频可解码：

```bash
for VIDEO in \
  "$RUN_DIR/visualization/05_mesh_proposals.mp4" \
  "$RUN_DIR/visualization/06_foundationpose.mp4" \
  "$RUN_DIR/visualization/07_raw_vs_aligned_contact.mp4"
do
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames \
    -of default=noprint_wrappers=1 "$VIDEO"
done
```

### 12. 可选 EgoDex Ground Truth 诊断

这两个命令显式读取 EgoDex GT，只用于诊断，不能作为推理输入。

外参方向 oracle：

```bash
conda run -n v2s-core python -m video_to_spider.cli oracle-validate-extrinsics \
  --run-dir "$RUN_DIR" \
  --uses-ground-truth
```

WiLoR GT 误差：

```bash
conda run -n v2s-core python -m video_to_spider.cli evaluate-wilor \
  --run-dir "$RUN_DIR" \
  --uses-ground-truth
```

oracle 可能因为正反两种外参解释都能投影到画面而报告 `camera direction ambiguous`。这是诊断失败，不应修改已经导入的外参，也不影响普通推理 artifact。

### 13. 生成统一报告

所有可运行阶段结束后执行：

```bash
conda run -n v2s-core python -m video_to_spider.cli evaluate-run \
  --run-dir "$RUN_DIR"
```

报告路径：

```text
evaluation/unified_run_report.json
```

查看完成状态、关键上游指标和 MJWP：

```bash
jq '{
  completion,
  segmentation: {
    selected_prompt: .stages.segmentation.data.selected_prompt
  },
  wilor: {
    left: .stages.wilor.data.valid_rate_left,
    right: .stages.wilor.data.valid_rate_right
  },
  sam3d_qualified: .stages.mesh_proposals.data.qualified_count,
  foundationpose: .stages.object_tracking.data.metrics,
  hand_roles: .stages.optimization.data.hands,
  spider_returncodes: [.stages.spider.data.commands[]?.returncode],
  mjwp: .stages.spider.data.mjwp_metrics,
  visualizations: .diagnostics.visualization.data.outputs
}' "$RUN_DIR/evaluation/unified_run_report.json"
```

完整链路的结构性完成条件：

```text
completion.spider_chain_complete == true
completion.simulation_video_complete == true
completion.m4_complete == true
```

质量验收还必须独立确认：

1. SAM 3 选择的是被搬运物体。
2. 三类自动可视化与原视频语义一致。
3. FoundationPose 跟踪连续且 mask IoU 可接受。
4. SPIDER 各阶段成功；预抓取返回码 2 只能在报告明确记录受控回退时接受。
5. MJWP position/rotation 同时达到阈值。

## 阶段重跑和故障定位

所有正式命令都按阶段写 artifact。失败后只重跑失败阶段及其下游，不需要重新 ingest。

### 输出已存在

模型阶段默认拒绝覆盖已有 artifact。确认需要重跑后添加 `--overwrite`。旧失败证据会尽可能保留在对应的 failure history 或日志中。

### SAM 3 没有有效目标

```bash
jq '[.candidates[]|{prompt,valid_rate:.metrics.valid_rate,score:.metrics.selection_score}]' \
  "$RUN_DIR/segmentation/metadata.json"
```

如果目标词的有效率为 0，SAM 3D 会报：

```text
mesh_failed: no valid SAM object keyframe
```

此时应停止该 episode。当前正式 pipeline 不接受人工点、框、mask 或额外 prompt 回退。

### SAM 3 选中了放置容器

对比：

```bash
jq '{instruction_text,object_keyword_candidates}' "$RUN_DIR/input/source.json"
jq '.selected_prompt' "$RUN_DIR/segmentation/metadata.json"
```

例如指令为 `pick up block ... place it on bowl` 时，被搬运物体是 `block`，不能因为 `bowl` mask 更稳定就继续处理 `bowl`。

### 单手和双手不匹配

不要根据任务文本中的 `right hand` 直接强制单手。以优化结果为准：

```bash
jq '.hands' "$RUN_DIR/optimization/optimization_metrics.json"
```

可靠可见的 passive hand 也必须保留。只有角色为 `invalid` 的手才从 artifact 中移除。

### GPU 显存不足

先检查共享任务：

```bash
nvidia-smi
```

同一个 episode 的 GPU 模型阶段建议串行执行。SAM 3D 保持 `--low-vram --moge-resolution-level 4 --max-slat-coords 20000`。不要终止其他用户的进程。

### SPIDER 阶段失败

每个 SPIDER 子命令都有独立日志：

```bash
find "$RUN_DIR/spider_export/dataset/processed/video_to_spider_egodex/xhand" \
  -path '*/pipeline_logs/*.log' -print
```

按 `01_decompose_fast.log` 到 `07_physics_contact_metrics.log` 的顺序定位第一个非零返回码。
接触搜索完成但未通过验收时，`05_grasp_preshape.log` 的返回码 2 是受控回退，不是崩溃；
详细原因以同目录的 `grasp_preshape_report.json` 为准。

## 批量测试建议

每个 episode 使用独立 `RUN_DIR`，逐条串行执行本文 1-13 步。记录随机种子、任务、episode ID、帧区间和 GPU。例如：

```text
release baseline: vertical_pick_place/111, 54 frames, physical GPU 7
additional episodes: choose independent fresh RUN_DIR values and run serially
```

批量汇总至少记录：

```text
最后成功阶段
SAM 3 selected prompt 和 valid rate
目标语义是否正确
WiLoR left/right valid rate
SAM 3D qualified proposal count
FoundationPose IoU、tracking score、rotation jump
hand roles 和 artifact_hand_order
SPIDER 各阶段返回码和预抓取 accepted/rejected 状态
MJWP position/rotation error
三类可视化路径
失败原因和日志路径
```

本仓库的一次已验证批次汇总示例位于：

```text
runs/batch_20260730_pipeline_summary.json
```

## 坐标和单位约定

- 列向量约定：`T_A_B` 将 B 坐标中的点变换到 A 坐标。
- Camera frame：OpenCV，`+X` 向右、`+Y` 向下、`+Z` 向前。
- EgoDex world 通过一个 episode 级固定 `T_sim_world` 转换到 MuJoCo `+Z` 向上坐标系。
- 平移、深度、网格顶点和接触位置统一使用米。
- 内部旋转使用旋转矩阵；SPIDER 导出四元数为 `wxyz`。
- 推理和优化使用原视频时间轴；SPIDER 导出重采样到 50 Hz。
- Ground Truth 只能由带 `--uses-ground-truth` 的诊断命令读取。

更详细的设计约束和历史实现记录见 [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) 与 [PROGRESS.md](./PROGRESS.md)。
