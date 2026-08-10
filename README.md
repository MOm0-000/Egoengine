# Video-to-SPIDER

将 EgoDex 单目 RGB episode 转换为 SPIDER/xHand 可执行轨迹的 artifact-oriented pipeline。

当前流程读取 EgoDex 的 RGB、相机内参、逐帧相机外参和任务文本，依次完成目标分割、手部重建、深度估计、物体网格重建、6D 位姿跟踪、手物联合优化、SPIDER 导出、IK 和 MJWP 仿真。手部 Ground Truth 只允许用于最后的显式诊断，不进入推理链路。

> 当前仓库没有 `run-all` 命令。完整流程必须按本文的 artifact 边界逐步执行。每一步都保存数值 artifact、metadata 和可视化，便于在进入下一步前验证。

本仓库的已验证发布基线使用 EgoDex `vertical_pick_place/111`（54 帧）和单张物理 GPU 7，从全新 RUN_DIR 串行生成了 MJWP 视频与 12/12 阶段统一报告；核心测试为 `67 passed`。这证明工程链路完整，但该样本的 MJWP 旋转误差仍未达到论文阈值。环境恢复、第三方 revision、权重和验收边界见 [docs/REPRODUCIBILITY.md](./docs/REPRODUCIBILITY.md)。

首次使用建议按以下顺序阅读：

1. [快速配置环境.md](./快速配置环境.md)：逐行配置源码、Conda 环境、SPIDER 和模型资产。
2. [quick_start.md](./quick_start.md)：从原始 MP4/HDF5 串行生成最终 MJWP 视频的最短命令清单。
3. 本 README：逐阶段质量检查、重跑和故障定位。

## 流程总览

```text
EgoDex MP4 + HDF5
  -> ingest
  -> SAM 3 object/hand segmentation
  -> WiLoR hand reconstruction
  -> Depth Anything V2 metric depth
  -> SAM 3D Objects mesh proposals
  -> FoundationPose proposal selection and tracking
  -> sequence optimization and contact inference
  -> SPIDER export
  -> decomposition -> contact cross-check -> XML -> IK -> MJWP
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
| Depth Anything V2 | `v2s-depth` |
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
test -s third_party/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth
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

### 4. Depth Anything V2 metric depth

深度阶段在原始像素分辨率保存 Zarr。当前使用 Hypersim ViT-L metric checkpoint：

```bash
CUDA_VISIBLE_DEVICES="$GPU" ./scripts/run_model_adapter.sh \
  v2s-depth video_to_spider.adapters.depth_anything \
  --run-dir "$RUN_DIR" \
  --checkpoint third_party/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
  --encoder vitl \
  --overwrite
```

检查 metadata：

```bash
jq '{model,encoder,original_resolution,valid_ratio,median_depth_m,outputs}' \
  "$RUN_DIR/depth/metadata.json"
```

成功输出：

```text
depth/metric_depth.zarr/
depth/metadata.json
depth/metric_depth.mp4
```

Depth Anything 的绝对尺度只作为低权重先验，不能覆盖 EgoDex 相机外参和 WiLoR 的米制证据。

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

优化器完成物体尺度、轨迹平滑、手腕姿态缺口插值、手物接触和仿真地面约束：

```bash
conda run -n v2s-opt python -m video_to_spider.cli optimize \
  --run-dir "$RUN_DIR" \
  --allow-unvalidated-contact-scale \
  --contact-enter-distance-m 0.020 \
  --max-contact-slip-p95-m-s 0.45 \
  --overwrite
```

上述三个显式放宽参数是 `vertical_pick_place/111` 的已记录灵敏度配置，适用于柔软、近对称毛绒物体；代码中的严格默认值仍是 12 mm 与 0.30 m/s。新 episode 应先尝试严格默认配置，只有在保留失败诊断并核查 2D/3D 接触证据后才能放宽。

检查优化指标和最终保留的手：

```bash
jq '{hands,simulation_floor,anchor,optimization,raw_vs_aligned,contact,quality_control}' \
  "$RUN_DIR/optimization/optimization_metrics.json"
```

优化先以 FoundationPose/Depth Anything 作为物体深度先验，并以物体 mask 优化轮廓尺度；
如果连续多帧存在可靠的二维指尖/物体接触候选，则同步缩放物体网格和相机平移，搜索与
MANO 指尖最一致的三维表面尺度。同步缩放不会改变物体二维投影，但结果的绝对公制尺度会标记为
`contact_calibrated_not_externally_validated`，需要真实物体尺寸才能进一步验证。
WiLoR 的弱透视平移只有在完整手部投影仍处于二维信任域内时才允许校准。
只有 `quality_control.export_ready == true` 的结果允许进入 `export-spider`，避免二维轮廓改善但
手部二维对齐或 manipulation 接触退化的轨迹被导出。默认要求至少一只 active 手具有连续接触；
仅处理非操作视频时可显式增加 `--allow-no-contact`。

角色规则：

- `active`：可靠可见并靠近物体，参与手物约束。
- `passive`：可靠可见但不操作物体，仍保留其动作并渲染。
- `invalid`：重建有效率不足，不进入导出 artifact。
- 画面中双手都可靠时，无论是否只有一只手实际操作，`artifact_hand_order` 都保留双手。
- 某只手全程不可见或不可靠时，才自动降级为 `left` 或 `right` 单手 artifact。
- passive 手如果需要超过 5 cm 的整体地面修正，会被视为绝对平移不可靠并降级为 invalid。

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

`run-spider` 内部会在 `$SPIDER_ROOT` 中使用现有 `uv` 环境，设置 EGL headless rendering，并依次运行 5 个阶段：

```text
01 decompose_fast
02 detect_contact
03 generate_xml
04 ik (contact-aware)
05 MJWP
```

IK 阶段会保留原生 `contact` 和 `contact_pos` 数组，供 MJWP 的接触目标使用。
导出前会将 episode 初始物体的支撑面对齐到仿真地面，同时保持手物相对几何关系。
优化器只有在手部公制深度校准通过后，才允许使用接触证据修正物体尺度；如需实验性
放宽限制，可显式使用 `--allow-unvalidated-contact-scale`。

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

检查五阶段返回码和 MJWP 指标：

```bash
SPIDER_REPORT="$RUN_DIR/spider_export/dataset/processed/video_to_spider_egodex/xhand/$EMBODIMENT_TYPE/$TASK/$EPISODE_ID/spider_run_report.json"

jq '{embodiment_type,commands:[.commands[]|{returncode,runtime_s,log}],mjwp_metrics,artifacts}' \
  "$SPIDER_REPORT"
```

当前质量阈值：

```text
mean object position error < 0.1 m
mean object rotation error < 0.5 rad
```

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
4. SPIDER 五阶段返回码都是 0。
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

按 `01_decompose_fast.log` 到 `05_mjwp.log` 的顺序定位第一个非零返回码。

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
SPIDER 五阶段返回码
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
