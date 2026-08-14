# Video-to-SPIDER V1 实现计划（EgoDex / 已标定双目 + 已知相机外参）

> 2026-08-14 补充：本文下方仍保留早期 V1 的 EgoDex 计划与 WP 拆分；当前代码本体已扩展到
> 已标定双目输入、`Replay→MPC→RL` 模式切换，以及复用 H2S2R `PpoAgent` 的 RL 残差策略适配层。
> 最新状态以 `README.md`、`PROGRESS.md` 和 `docs/REPRODUCIBILITY.md` 为准。

## 1. 文档目的

本文档用于指导多个 agent 并行实现 `video_to_spider` 第一版，并作为接口、范围、验收标准和集成顺序的唯一基准。

第一版以 `/data_all/share/datasets/egodex` 中的 EgoDex 为测试对象，输入为单目 RGB 视频、相机内参、逐帧相机外参和 HDF5 中的任务文字指令。相机轨迹不通过 SLAM 估计。EgoDex 的手部标注只允许用于离线指标和显式 oracle 排障，不允许进入最终视觉推理路径。

论文对应实现位于：

- `../reference/arXiv-2511.09484v2/body/exp.tex`：单 RGB 相机实验说明。
- `../reference/arXiv-2511.09484v2/appendix/implementation.tex`：手部 IK、50 Hz 重采样和 10 Hz 低通设置。
- `../spider/`：下游 SPIDER 代码。

## 2. V1 目标与非目标

### 2.1 目标

给定一个 EgoDex episode。数据根目录固定为：

```text
/data_all/share/datasets/egodex/
  part*/{task}/{id}.mp4
  part*/{task}/{id}.hdf5
```

同一 `{id}` 的 MP4/HDF5 构成一个 episode。pipeline 必须递归扫描 `part*`，不允许把某个分区或某个任务硬编码为唯一输入。

```text
episode.mp4
episode.hdf5
HDF5 task instruction（llm_description 或 llm_description2）
```

仅从以下允许输入产生完整 SPIDER 参考轨迹：

```text
RGB 帧
视频时间/FPS
camera/intrinsic
transforms/camera（逐帧相机外参）
HDF5 task instruction（仅用于自动提取 SAM 3 文本关键词）
```

最终输出包括：

1. 统一米制、统一仿真世界坐标系的左右手腕、五指指尖和物体 6D 轨迹。
2. 规范坐标下的物体 visual mesh 和凸分解 collision mesh。
3. 手物接触状态及接触点。
4. SPIDER 所需的 `trajectory_keypoints.npz`、`task_info.json`、场景 XML、IK 轨迹和 MJWP 轨迹。
5. 每个阶段的缓存、质量指标、可视化和可复现实验 manifest。

### 2.2 V1 支持范围

- 单个刚性目标物体。
- 单手或双手操作，但首批自动筛选片段优先选择单手。
- 3 到 10 秒的 1080p、30 FPS 片段。
- 不透明、非强反光、非严重旋转对称的物体。
- SAM 3 只使用从 HDF5 task instruction 自动提取的文本关键词；不接受点、框、mask 或额外 prompt。
- 离线处理，不要求实时。

### 2.3 V1 非目标

- 不实现 SLAM、视觉里程计或未知相机轨迹恢复。
- 不支持透明物体、镜面物体、软体、流体和可变形物体。
- 不支持剪刀、盒盖、钳子等需要多个刚体关节的 articulated object。
- 不保证从任意互联网视频恢复绝对尺度。
- 不修改或统一安装所有 `third_party` 项目的环境。
- 不在第一版训练或微调基础模型。
- 不做真机部署；第一版终点为 MuJoCo/SPIDER 中的完整仿真轨迹。

## 3. 已冻结的核心决策

1. **相机位姿来源**：直接读取 EgoDex `transforms/camera[t]`，V1 不使用 SLAM。
2. **相机内参来源**：直接读取 EgoDex `camera/intrinsic`，禁止使用 WiLoR 默认焦距替代。
3. **坐标转换**：所有模型先输出 camera frame 观测，再通过逐帧外参转换到 EgoDex world frame，最后通过一个 episode 级固定变换进入 SPIDER sim frame。
4. **尺度来源**：EgoDex 外参平移是米制锚点；Depth Anything V2 metric 和 MANO 手尺寸仅作为鲁棒先验。SAM 3D Objects 的网格尺度必须单独优化。
5. **网格与位姿解耦**：SAM 3D Objects 负责产生并静态排序 canonical mesh proposals；FoundationPose 按排序逐个注册，用跟踪质量选出最终 proposal 并产生逐帧位姿先验，不修改候选网格几何。
6. **时序处理**：禁止把逐帧模型输出直接送入 SPIDER。必须经过序列级平滑与手物联合优化。
7. **SPIDER 文件名**：使用实际代码读取的 `trajectory_keypoints.npz`，不是部分文档中的 `trajectory_keypoint.npz`。
8. **四元数格式**：内部使用旋转矩阵或 SO(3)；写入 SPIDER 时统一为 `wxyz`。
9. **第三方隔离**：SAM 3、SAM 3D Objects、WiLoR、Depth Anything V2 和 FoundationPose 分别使用独立环境，通过文件 artifact 交换数据。
10. **GT 隔离**：EgoDex 手和身体 transforms 只能由 evaluator/oracle 命令读取；普通 pipeline 入口必须拒绝或忽略这些字段。
11. **目标提示来源**：读取 `which_llm_description` 指定的 HDF5 文本，优先使用 `llm_description`/`llm_description2`；通过确定性的关键词规则生成 SAM 3 文本 prompt。
12. **自动失败处理**：关键词候选全部失败时，记录 `segmentation_failed` 并终止该 episode；不得退回额外 prompt 或复制上一帧 mask。

### 3.1 已固定执行环境

除 SPIDER 外，pipeline 使用以下 Conda 环境；adapter 必须通过 artifact 文件交换数据，不得在一个环境中跨项目 import：

| 范围 | 环境/运行方式 | 2026-07-29 preflight |
|---|---|---|
| schema、ingest、manifest、通用评测/编排 | `v2s-core` | `h5py/cv2/numpy/scipy/trimesh/zarr` import 通过 |
| SAM 3 | `v2s-sam3` | `torch/sam3` import 通过，checkpoint 已存在 |
| SAM 3D Objects | `v2s-sam3d` | Conda 环境已创建，但缺少 `torch` 和推理依赖，尚不可运行 |
| WiLoR | `v2s-wilor` | `torch/wilor` import 通过，checkpoint/MANO 已存在 |
| Depth Anything V2 | `v2s-depth` | metric model class import 通过，Hypersim ViT-L checkpoint 已存在 |
| FoundationPose | `v2s-foundationpose` | `torch/estimater` import 通过，scorer/refiner 权重已存在 |
| 序列优化、接触、mesh 数值处理 | `v2s-opt` | `torch/numpy/scipy/trimesh/zarr` import 通过 |
| SPIDER/MJWP | `../spider/.venv`，由 `uv` 管理 | `spider/torch/mujoco/warp` import 通过 |

SPIDER 命令必须在 `../spider` 中通过 `uv` 执行，不创建 Conda 环境。已有 `.venv` 的离线/受限网络执行命令为：

```bash
UV_CACHE_DIR=/tmp/video-to-spider-uv-cache uv run --frozen --no-sync python ...
```

正常联网环境可先运行 `uv sync --frozen`，再去掉 `--no-sync`。`--no-sync` 只允许在 preflight 已确认现有 `.venv` 可用时使用。上述 import preflight 不等于 GPU 推理通过；所有 CUDA 模型仍必须在可见 GPU 节点完成 checkpoint load 和 3 到 10 帧 smoke test。

## 4. 总体数据流

```text
EgoDex MP4 + HDF5
        |
        +--> ingest: 帧、时间戳、K、T_world_camera[t]
        |
        +--> instruction parser: object noun phrases / keyword candidates
                         |
        +--> SAM 3 text prompts: object masklet + hand masks + 置信度
        |
        +--> WiLoR: MANO、21 joints、手腕/手平移先验
        |
        +--> Depth Anything V2: 逐帧 metric depth 先验
        |
        +--> keyframe selector
                 |
                 +--> SAM 3D Objects: ranked canonical mesh proposals
                                   |
                                   +--> mesh repair / scale initialization
                                                    |
                                                    +--> FoundationPose candidate screening
                                                         selected canonical mesh
                                                         T_camera_object_raw[t]
        |
        +------------------------------------------------------+
                                                               |
                          sequence hand-object optimizer <------+
                                                               |
                             T_sim_hand[t], T_sim_object[t]
                                           |
                                  contact inference
                                           |
                                     SPIDER exporter
                                           |
                      decompose -> detect_contact -> XML -> IK -> MJWP
```

## 5. 坐标、尺度和时间约定

### 5.1 变换命名

统一采用列向量，`T_A_B` 表示把 B 坐标中的点变换到 A 坐标：

```text
p_A = T_A_B @ p_B
```

必须使用以下字段名，不允许使用语义不清的 `pose`、`extrinsic`：

```text
T_world_camera[t]
T_camera_hand[t]
T_camera_object[t]
T_world_hand[t]   = T_world_camera[t] @ T_camera_hand[t]
T_world_object[t] = T_world_camera[t] @ T_camera_object[t]
T_sim_world
T_sim_hand[t]     = T_sim_world @ T_world_hand[t]
T_sim_object[t]   = T_sim_world @ T_world_object[t]
```

EgoDex `transforms/camera` 在进入 pipeline 时必须通过数值和投影测试确认是 `T_world_camera`。不得仅根据字段名猜测方向。

### 5.2 Camera frame

模型适配器输出统一使用 OpenCV camera frame：

```text
+X: image right
+Y: image down
+Z: camera forward
```

如果第三方项目使用 OpenGL/PyTorch3D convention，转换只能在该 adapter 内完成，并在 metadata 中记录原始和目标 convention。

### 5.3 Sim frame

SPIDER/MuJoCo 使用右手系、`+Z` 向上。`T_sim_world` 为 episode 级固定变换：

- 利用 EgoDex world 的重力/竖直方向建立 `+Z`。
- 将初始物体最低点放在 `z = 0` 附近。
- 将初始物体中心的 XY 平移到工作区中心。
- 保持整段轨迹只使用同一个 `T_sim_world`，禁止逐帧归一化。
- `T_sim_world` 必须写入 manifest，保证能够逆变换回 EgoDex world。

### 5.4 尺度

- 所有平移、深度、mesh vertex 和接触位置统一使用米。
- 相机外参提供 world trajectory 的米制基准。
- SAM 3D Objects mesh 有待估尺度 `s_object`。
- `s_object` 的初始化来自 mask 内深度点云与 mesh 可见包围盒拟合。
- 序列优化中允许优化一个全局 `s_object`，V1 不允许每帧独立缩放。
- MANO `betas` 每只手整段共享，禁止每帧独立手尺寸。

### 5.5 时间

- ingest 保留原始 frame index、PTS、FPS 和时间戳。
- 所有模型 artifact 必须携带相同的 `frame_indices` 和 `timestamps_s`。
- 优化在原始 30 Hz 时间轴进行。
- 导出前重采样到 50 Hz。
- 平移使用零相位低通；旋转在 SO(3) log space 平滑；MANO pose 使用关节空间平滑。
- 论文设置为 10 Hz 低通，作为默认值但必须配置化。
- 二值 contact 不做低通，而使用阈值滞回和最短持续帧数。

## 6. 工程目录与所有权

目标目录：

```text
video_to_spider/
  IMPLEMENTATION_PLAN.md
  PROGRESS.md
  pyproject.toml
  configs/
    egodex_v1.yaml
    egodex_v1_clips.yaml
    object_keyword_rules.yaml
  video_to_spider/
    __init__.py
    schemas.py
    manifest.py
    coordinates.py
    ingest/
      egodex.py
      egodex_ground_truth.py
    adapters/
      sam3.py
      wilor.py
      depth_anything.py
      sam3d_objects.py
      foundationpose.py
    optimization/
      sequence.py
      losses.py
      contact.py
      smoothing.py
    export/
      spider.py
    eval/
      egodex.py
      metrics.py
    visualization/
      overlays.py
      scene.py
    cli.py
  scripts/
    run_model_adapter.sh
  tests/
    fixtures/
    test_schemas.py
    test_coordinates.py
    test_spider_export.py
    test_tiny_pipeline.py
```

约束：

- 除非对应工作包明确授权，agent 不得修改 `third_party/`。
- `schemas.py`、artifact 版本和公共坐标工具只由基础设施工作包维护。
- 每个 adapter 只拥有自己的文件和测试。
- SPIDER exporter 工作包可以读取 `../spider`，但第一版不应改动 SPIDER；确需修改时必须单独记录原因和兼容影响。
- `PROGRESS.md` 是多 agent 共享的唯一进度账本。任何 agent 开始工作、状态变化、产生交付物或发现/解除阻塞时，都必须在同一次工作增量中更新该文件；不得只在对话中报告状态。

## 7. Artifact 协议

### 7.1 Run 目录

```text
runs/{run_id}/
  manifest.json
  input/
    source.json
  frames/
    frame_index.json
    rgb/000000.jpg
  calibration/
    intrinsics.npy
    T_world_camera.npy
    T_sim_world.npy
  segmentation/
    object_masks.npz
    hand_masks.npz
    metadata.json
  hands/
    wilor_raw.npz
    metadata.json
  depth/
    metric_depth.zarr
    metadata.json
  object/
    keyframes.json
    proposals/
      {proposal_id}/
        visual.obj
        collision_source.obj
        metadata.json
    mesh_ranking.json
    selected_mesh.json
    foundationpose_candidates/
      {proposal_id}/
        tracking_metrics.json
    foundationpose_raw.npz
  alignment/
    aligned_trajectory.npz
    metrics.json
  contact/
    contact.npz
    metrics.json
  spider/
    export_manifest.json
  visualization/
    perception_overlay.mp4
    alignment_overlay.mp4
    simulation.mp4
```

### 7.2 Manifest 必需字段

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "source_episode": "...",
  "config_sha256": "...",
  "git_revisions": {},
  "model_revisions": {},
  "allowed_inputs": ["rgb", "intrinsics", "camera_extrinsics", "instruction_text", "object_keyword_candidates"],
  "frame_count": 0,
  "fps": 30.0,
  "stages": {},
  "coordinate_conventions": {},
  "units": "meter-second-radian"
}
```

每个 stage 必须记录：输入 hash、输出路径、运行环境、命令、开始/结束时间、成功状态、警告和质量指标。缓存命中必须基于输入 hash 与配置，而不是只判断文件是否存在。

### 7.3 核心 artifact schema

`mesh_ranking.json` 至少包含：

```text
schema_version
proposal_ids[]
proposals[].visual_mesh_path
proposals[].collision_source_mesh_path
proposals[].canonical_transform
proposals[].scale_initial_m
proposals[].integrity_metrics
proposals[].silhouette_residual
proposals[].depth_residual
proposals[].static_score
proposals[].static_rank
proposals[].eligible
proposals[].rejection_reasons[]
```

该文件只允许包含 Stage 5 可计算的静态指标，不得写入 FoundationPose tracking score。

`selected_mesh.json` 至少包含：

```text
schema_version
selected_proposal_id
visual_mesh_path
collision_source_mesh_path
canonical_transform
scale_initial_m
selection_mode             first_all_gates | best_safe_score
tracking_score
tracking_metrics
attempts[].proposal_id
attempts[].status
attempts[].tracking_score
attempts[].rejection_reasons[]
```

其中 mesh 路径必须引用 Stage 5 proposal，禁止复制后丢失来源；`attempts` 顺序必须与实际运行顺序一致。

`wilor_raw.npz`：

```text
frame_indices           [T]
timestamps_s            [T]
side                     [T, H]，0=left, 1=right
valid                    [T, H]
score                    [T, H]
mano_global_orient       [T, H, 3, 3]
mano_hand_pose           [T, H, 15, 3, 3]
mano_betas               [T, H, 10]
joints_camera_rootrel    [T, H, 21, 3]
vertices_camera_rootrel  [T, H, 778, 3]
translation_camera       [T, H, 3]
```

`foundationpose_raw.npz`：

```text
frame_indices       [T]
timestamps_s        [T]
T_camera_object     [T, 4, 4]
valid               [T]
confidence           [T]
registration_frame  [T]
depth_residual       [T]
mask_iou             [T]
```

`aligned_trajectory.npz`：

```text
frame_indices       [T]
timestamps_s        [T]
T_sim_object        [T, O, 4, 4]
T_sim_wrist         [T, H, 4, 4]
fingertips_sim      [T, H, 5, 3]
mano_pose           [T, H, 15, 3, 3]
mano_betas          [H, 10]
object_scale_to_m   [O]，canonical unit 到 meter 的固定缩放
valid_object        [T, O]
valid_hand          [T, H]
confidence_object   [T, O]
confidence_hand     [T, H]
```

所有 schema loader 必须校验 shape、dtype、有限值、单位、旋转正交性、`det(R) > 0`、齐次矩阵最后一行和严格递增时间戳。

## 8. 各阶段实现要求

### Stage 0：Preflight

检查并生成机器可读报告：

- EgoDex episode 的 MP4/HDF5 是否成对存在。
- 视频和 HDF5 帧数、FPS、分辨率是否一致。
- `camera/intrinsic` 和 `transforms/camera` 是否存在且有效。
- GPU、driver、CUDA 和各模型环境是否可用。
- 所需 checkpoint 和 MANO 文件是否存在。
- 目标输出空间是否足够。

当前已知状态（2026-07-29）：

- `/data_all/share/datasets/egodex` 约 336G，当前包含 1 个 `part`、26 个任务目录；递归扫描确认 46,234 组同目录同 stem 的 MP4/HDF5 episode，缺失 MP4 或 HDF5 对端的数量均为 0。
- `v2s-core` 已包含 `h5py`、OpenCV、NumPy、SciPy、trimesh 和 zarr，ingest 的 Python 依赖 import preflight 已通过。
- SAM 3 的 `sam3.1_multiplex.pt`、SAM 3D Objects 的主要生成器/编码器/mesh decoder checkpoints、WiLoR detector/model、`MANO_RIGHT.pkl`、Depth Anything Hypersim ViT-L metric checkpoint、FoundationPose scorer/refiner 权重均已存在。
- Depth Anything relative-depth 路径下的 `checkpoints/depth_anything_v2_vitl.pth` 仍为 0 字节，但 V1 使用的 `metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth` 有效存在；preflight 必须检查实际配置引用 metric 文件，不能误用 0 字节文件。
- `v2s-core`、`v2s-sam3`、`v2s-wilor`、`v2s-depth`、`v2s-foundationpose` 和 `v2s-opt` 的核心 import 已通过。`v2s-sam3d` 虽已创建，但当前缺少 `torch`，仍是 Stage 5 阻塞项。
- SPIDER 使用 `../spider/.venv` 和 `uv`；`uv run --frozen --no-sync` 的核心 import preflight 已通过。
- 当前执行节点无法访问 NVIDIA driver，所有模型的 checkpoint load/GPU inference 和 MJWP smoke test 尚未完成。

验收：缺少资源时给出明确错误和修复建议，不产生半成品 success manifest。

### Stage 1：EgoDex ingest

职责：

- 递归扫描 `/data_all/share/datasets/egodex/part*/*`，按同名 stem 配对 MP4/HDF5。
- 按 PTS 解码视频，生成稳定 frame index。
- 加载 `K` 和逐帧 `T_world_camera`。
- 读取 HDF5 `llm_description`、`llm_description2`、`which_llm_description` 和任务目录名。
- 生成确定性的 `instruction_text` 和 `object_keyword_candidates`，保存到 input manifest。
- 检查外参方向、连续性、单位和帧同步。
- 生成 RGB frame cache、calibration artifact 和 source manifest。
- evaluator 单独加载 GT 手 transforms/confidences。

外参方向验证：

1. 分别测试 `T_world_camera` 和其逆矩阵。
2. 将 EgoDex GT 手关节转换到候选 camera frame。
3. 使用 `K` 投影并与 RGB 上的手位置进行 overlay。
4. 用有效深度比例、GT 手投影的 robust reprojection residual 和旋转正交误差自动选择候选方向，并将结论固化为测试。

注意 EgoDex RGB 可能由多相机合成，GT 重投影存在系统偏差。该测试只用于排除轴向/求逆错误，不把亚像素误差作为质量门槛。

验收：对每个 episode 自动检查 MP4/HDF5 stem、帧数、FPS、K、外参 shape 和 instruction 字段；不满足条件的 episode 写入 `preflight_failed.json`，不进入后续阶段。

### Stage 2：instruction parser + SAM 3 mask tracking

职责：

- 从 HDF5 任务指令抽取目标物体名词短语和同义词候选。解析顺序为：
  1. 选择 `which_llm_description` 对应文本；
  2. 清理颜色、数量、动作和位置修饰词，保留名词短语；
  3. 使用 `configs/object_keyword_rules.yaml` 进行词形归一和同义词扩展；
  4. 按原短语、去颜色短语、任务目录名生成去重后的候选列表。
- 逐个候选只使用 SAM 3 text prompt 运行，不接受点、框、mask 或额外 prompt。
- 使用程序化分数选择候选：检测置信度、mask 面积合理性、时序稳定性、前后向传播一致性、与 hand exclusion mask 的重叠率。
- 输出目标物体 masklet、置信度和 object instance ID。
- 同时产生 hand/person exclusion mask，供深度和 FoundationPose 使用。
- 对每个关键词候选保存独立结果，最终只保留自动评分最高者。

质量逻辑：

- 检测 mask 面积突变、消失、分裂、越界和与手 mask 的异常重叠。
- 从自动选出的最高质量 anchor frame 向前和向后传播。
- 对失败区间标记 invalid，不得静默复制上一帧。

验收：自动报告候选 prompt、选择分数、mask 有效率、面积/边界时序稳定性、前后向一致性、丢失率和 ID switch。所有候选均失败时返回 `segmentation_failed`，不要求外部介入。

### Stage 3：WiLoR hand reconstruction

职责：

- 使用 EgoDex `K` 取代 demo 的默认焦距假设。
- 保存 `pred_mano_params`、21 joints、vertices 和相机平移，不只导出 OBJ。
- 通过 handedness、2D 跟踪和运动连续性维持左右手 ID。
- 每只手整段聚合一个初始 `beta`，低置信度帧不参与聚合。
- 缺帧只做短区间插值，并保留 `valid=false`。

禁止在 adapter 内做最终世界坐标平滑；adapter 只输出原始观测和置信度。

验收：在 EgoDex GT 上报告绝对 MPJPE、root-relative MPJPE、五指指尖 MPJPE、腕部旋转误差、ID switch 和 acceleration jitter。

### Stage 4：Depth Anything V2

职责：

- 使用 indoor metric checkpoint 产生米制深度先验。
- 保持原分辨率映射信息，禁止 resize 后直接当作原图深度。
- 保存有效 mask、置信度/不确定性代理和模型版本。
- 对手和目标物体分别计算 mask 内深度统计。
- 可选加入 Video Depth Anything，但不作为 V1 阻塞依赖。

深度只作为观测先验，不能覆盖外参、MANO 或网格的一致性约束。

验收：相邻帧静态背景 warp 后深度变化、mask 内空洞率和异常值比例有报告；不能仅保存彩色深度图。

### Stage 5：关键帧与 SAM 3D Objects mesh

关键帧候选评分考虑：

- 目标 mask 面积和边界完整度。
- 与手 mask 的遮挡比例。
- 图像清晰度。
- 视角多样性。
- 深度有效率。

从前 K 个候选生成 3 到 5 个 mesh proposal。必须启用 mesh postprocess，输出三角网格而非仅 Gaussian splat。Stage 5 只生成候选和不依赖 FoundationPose 的静态排序，不在本阶段宣称选出最终 mesh。

Mesh 后处理：

- 合并或选择主要 connected component。
- 修复法向、退化面、自交和小孔洞。
- 记录 canonical origin、轴、包围盒和所有变换。
- 用 mask 内点云估计初始尺度和位置。
- 用多个验证帧的轮廓和深度拟合对 proposals 进行确定性静态排序，写入 `mesh_ranking.json`。
- 保留原始 proposal，禁止覆盖后无法追溯。

验收：至少一个 proposal 能被 `trimesh` 和 MuJoCo 读取；所有合格 proposal 无 NaN、零面积主轴或错误数量级尺度；每个 proposal 都有完整的 canonical transform、程序化 mesh integrity、渲染 silhouette residual、depth residual 和确定性排名。Stage 5 的验收不运行或依赖 FoundationPose。

### Stage 6：FoundationPose object tracking

使用 model-based 模式，输入：

```text
ranked canonical mesh proposals
K
RGB[t]
metric depth prior[t]
SAM 3 object mask[t]
```

实现要求：

- 按 `mesh_ranking.json` 顺序逐个尝试合格 proposal，先在固定验证帧上执行 registration 和短程双向 tracking。
- 用 mask IoU、深度残差、渲染残差、有效帧率和姿态连续性生成确定性 tracking score；首个达到全部门槛的 proposal 即被选中，否则选择总分最高且达到最低安全门槛的 proposal。
- 所有候选都不合格时返回 `object_tracking_failed`，不得把静态排名第一的 proposal 强行标记成功。
- 将最终选择及理由写入 `selected_mesh.json`，并保留每个候选的 `foundationpose_candidates/{proposal_id}/tracking_metrics.json`。
- 对选中的 proposal 从最佳 anchor frame 执行完整 registration。
- 从 anchor 双向跟踪到序列首尾。
- 利用 SAM mask 去除手遮挡像素。
- 根据 mask IoU、深度残差、姿态速度和渲染残差检测漂移。
- 漂移后在候选帧重新 registration，并记录 segment ID。
- 对称物体必须支持 symmetry transforms；评测旋转时按对称群取最小误差。
- 输出 raw pose 和置信度，不在 adapter 内隐藏式平滑。

验收：`selected_mesh.json` 引用的 proposal、尺度和 canonical transform 可追溯到 Stage 5；最终轨迹无非有限位姿；所有尝试候选及其拒绝原因有记录；有效帧比例、重注册次数、平移/旋转跳变和渲染残差有报告。

### Stage 7：手物序列联合优化

优化变量：

```text
s_object                         一个全局物体尺度
delta_T_world_object[t]          FoundationPose 位姿修正
delta_T_world_wrist[h,t]         WiLoR 手腕修正
theta_mano[h,t]                  MANO pose
beta_mano[h]                     每只手共享 shape
```

`T_world_camera[t]` 在 V1 中固定，不参与优化。`T_sim_world` 在进入优化前固定。

损失：

```text
E_hand_2d       MANO joints/vertices 重投影
E_object_mask   渲染轮廓与 SAM 3 mask
E_depth         可见 mesh/hand 与深度先验
E_wilor         WiLoR MANO 和手腕先验
E_pose          FoundationPose 先验
E_shape         共享 beta、骨长和合理手尺寸
E_temporal      SE(3) 和关节速度/加速度
E_contact       接触时指尖到物体 SDF 距离及低滑移
E_penetration   手物非穿透
E_support       物体与工作平面关系
E_scale         mesh 尺度先验
```

全部观测项使用置信度和鲁棒核。旋转残差使用 SO(3) geodesic，不对 quaternion 分量直接做 L2。

推荐分阶段求解：

1. 固定手，优化 mesh scale 和 object pose。
2. 固定物体，优化手腕、MANO pose 和共享 beta。
3. 开启轮廓、深度与时序项联合优化。
4. 从 SDF 距离、相对速度和持续时间生成软接触。
5. 开启 contact/penetration/support 项联合细化。
6. 重新推断 contact，短窗口细化一次。

采用 2 到 4 秒滑动窗口，0.5 秒重叠；重叠区在 Lie algebra 中融合。所有阶段保存 loss breakdown 和 before/after overlay。

验收：总 loss 下降；无尺度漂移；物体轨迹无未解释跳变；手物穿透 P95 小于 5 mm；接触区局部滑移目标小于 10 mm。

### Stage 8：接触推断

每个手指的接触概率综合：

- 指尖到物体 SDF 的有符号距离。
- 指尖与物体表面相对速度。
- 2D 遮挡/接近证据。
- 连续帧持续性。
- 优化器置信度。

二值化采用进入/退出双阈值和最短持续帧数。`contact_pos` 使用物体 canonical/local frame，不能保存成 world frame。

验收：`contact.npz` 通过 schema、有限值和坐标系校验；接触进入/退出满足配置化距离、相对速度、滞回和最短持续帧阈值；非接触区远距离虚假点率及稳定抓取中的 local contact 跳变均由批量 metrics 按固定阈值判定。可视化只作为自动生成的诊断 artifact，不作为通过条件。

### Stage 9：SPIDER 导出与运行

导出目录遵循 `../spider/docs/usage/data-structure.md`，但字段 shape 以 SPIDER 实际处理代码为准。

`trajectory_keypoints.npz`：

```text
qpos_wrist_right [T50, 7]      xyz + quat_wxyz
qpos_finger_right[T50, 5, 7]   五指指尖；quat 可为 identity
qpos_obj_right   [T50, 7]
qpos_wrist_left  [T50, 7]
qpos_finger_left [T50, 5, 7]
qpos_obj_left    [T50, 7]
contact_right    [T50, 5]
contact_pos_right[5, 3]
contact_left     [T50, 5]
contact_pos_left [5, 3]
```

单物体双手操作遵循现有 GigaHand/SPIDER 约定：真实物体和轨迹只写入 `right_object`，两只手都可与该物体接触；`left_object` 保持无 mesh 的 dummy pose。不得把同一个物体复制成两个可碰撞动态物体。单手模式仍输出完整 schema，未使用一侧的位置填零、四元数填 identity，并在 metadata 中标记 invalid。

`task_info.json` 至少包含：

```text
task
dataset_name = video_to_spider_egodex
embodiment_type
data_id
ref_dt = 0.02
right_object_mesh_dir / left_object_mesh_dir
right_object_convex_dir / left_object_convex_dir
source_run_id
```

导出后依次运行：

1. `spider/preprocess/decompose_fast.py`
2. `spider/preprocess/detect_contact.py`，作为视觉接触的交叉检查
3. `spider/preprocess/generate_xml.py`
4. `spider/preprocess/ik.py`
5. `examples/run_mjwp.py`

需要保留视觉 contact 和 SPIDER detect_contact 两套结果及差异指标，不允许后者静默覆盖前者。

为复现实验论文标准，结果报告同时给出：

- 论文成功阈值：位置误差 `< 0.1 m`，旋转误差 `< 0.5 rad`。
- 当前 SPIDER config 实际使用的阈值。

验收：导出文件可被 SPIDER loader、scene generator 和 IK 无修改读取；MJWP 至少完成一个选定片段并生成视频和指标。

## 9. EgoDex 评测与必要诊断

### 9.1 数据选择

由 `ingest/egodex.py` 按固定 seed 和规则自动生成 `configs/egodex_v1_clips.yaml`，再冻结文件内容。初始目标为 20 个 episode/clip：

- 10 个开发片段，用于调参。
- 5 个验证片段，用于选择超参数。
- 5 个冻结测试片段，仅用于最终报告。

选择条件：

- 单刚体、目标明确。
- 至少一个低遮挡网格关键帧。
- 包含接触前、接触中和接触后阶段。
- 首批以 pick/place、移动、倾倒刚性容器为主。
- 自动排除明显透明、反光、articulated 或视频/HDF5 不同步的样本。

每个 clip 记录 episode 路径、起止 frame、HDF5 instruction、自动抽取的关键词、筛选规则版本和排除原因代码。冻结测试片段只能由配置文件版本控制，禁止按最终结果替换。

### 9.2 必要诊断对照

V1 不进行 A/B/C/D 形式的正式消融，也不要求为同一 clip 重复运行多套完整感知 pipeline。主结果只来自一条固定流程：EgoDex 已知相机外参、WiLoR 手部观测、视觉物体 mesh/pose、序列联合优化和 SPIDER。

只保留以下一项必需对照，用于判断自定义序列优化器是否实际改善输入：

| 对照 | 手与物体输入 | 输出 | 目的 |
|---|---|---|---|
| raw vs aligned | 完全相同的 WiLoR、selected mesh、FoundationPose、mask、depth 和相机外参 | 优化前 raw trajectory 与优化后 aligned trajectory | 检查 jitter、重投影、silhouette/depth residual、penetration 和 contact slip 是否改善 |

该对照复用同一次感知结果，只读取已保存的 raw/aligned artifacts，不重新运行 SAM 3、WiLoR、Depth Anything、SAM 3D Objects 或 FoundationPose。V1 报告必须包含优化前后指标及逐项变化；若某项因不可观测而无法计算，标记 `not_observable`，不得填零。

EgoDex GT 手只用于离线指标。当前对应命令为 `optimize --hand-source egodex_gt --uses-ground-truth`，仅在定位手部观测是否为失败根因时运行；必须在 manifest 标记 `uses_ground_truth=true`，其结果不进入主结果、完成率或 V1 验收。旧的 `--oracle-hand` 名称已不再存在于 CLI。SLAM 对照不属于 V1。

### 9.3 指标

输入与相机：

- frame/PTS 对齐误差。
- 外参旋转正交误差和轨迹连续性。
- GT 手投影 residual、有效深度比例和外参连续性。

手部：

- camera/world absolute MPJPE。
- root-relative MPJPE。
- fingertip MPJPE。
- wrist translation 和 geodesic rotation error。
- 左右手 ID switch、有效帧率和 acceleration jitter。

物体：

- SAM 3 前后向传播一致性、mask 面积/边界时序稳定性。
- 渲染 mesh 与 SAM 3 mask 的轮廓一致性。
- 可见表面 depth residual。
- 跟踪有效率、重注册数和轨迹跳变。
- EgoDex 无 object CAD/6D GT，因此不报告伪造的 ADD/ADD-S；后续用 HOT3D 补充物体 6D 定量评测。

手物对齐：

- penetration depth mean/P95/max。
- contact local slip。
- 接触持续性、阈值滞回次数和接触状态时序一致性。

SPIDER：

- IK fingertip/wrist error。
- MJWP object position/rotation tracking error。
- 论文阈值下 success rate。
- pipeline 完成率、失败阶段分布和每阶段耗时。

### 9.4 V1 完成标准

功能门槛：

- 20 个固定片段中至少 16 个能完成 perception、alignment 和 SPIDER export。
- 至少 10 个能完成 MJWP，不因格式、NaN、坐标或尺度错误退出。
- 所有失败都有明确 stage、错误原因和可复现命令。

质量目标：

- WiLoR fingertip MPJPE 中位数目标 `< 35 mm`。
- SAM 3 前后向 mask 一致性目标 `> 0.70`，并满足 mask 有效率和面积稳定性阈值。
- 手物 penetration P95 目标 `< 5 mm`。
- 稳定接触 local slip 目标 `< 10 mm`。
- 完成 MJWP 的片段中，至少 60% 达到 `< 0.1 m / < 0.5 rad`。

这些是 V1 工程目标，不是模型能力保证。未达标时必须保留分阶段指标，以定位是 segmentation、mesh、depth、pose、alignment 还是 SPIDER 失败。

## 10. 多 Agent 工作包

### WP0：基础设施与 schema

**所有权**：`pyproject.toml`、`schemas.py`、`manifest.py`、`coordinates.py`、公共 schema/coordinate tests。

**输出**：可安装基础包、artifact dataclass/loader、manifest、坐标转换与验证工具。

**依赖**：无。

**验收**：合成 SE(3) round-trip、NPZ schema 校验和 manifest cache key 测试通过。

### WP1：EgoDex ingest 与 GT evaluator

**所有权**：`ingest/egodex.py`、`ingest/egodex_ground_truth.py`、ingest tests。WP1 只提供 GT reader，不在 `eval/` 中实现指标。

**输入**：WP0 schema。

**输出**：RGB/calibration artifacts、外参方向验证 overlay、独立 GT reader。

**依赖**：WP0。

**验收**：一个选定 episode 能生成严格同步的输入 artifact；普通 pipeline artifact 不含 GT 手字段。

### WP2：SAM 3 adapter 与关键帧初选

**所有权**：`adapters/sam3.py`、mask QC、相关 tests。

**输入**：WP0 schema、WP1 frames。

**输出**：object/hand masks、置信度、失败区间、初始关键帧排名。

**依赖**：WP0；可使用 WP1 的已生成 fixture 并行开发。

**验收**：一个 clip 完成双向 mask propagation 并生成 overlay/QC metrics。

### WP3：WiLoR adapter

**所有权**：`adapters/wilor.py`、手 ID/置信度处理和 adapter tests。

**输入**：WP0 schema、WP1 frames/K。

**输出**：完整 `wilor_raw.npz` 和 overlay。

**依赖**：WP0；与 WP2/WP4 并行。

**验收**：保存 MANO params 而非仅 mesh；能通过 EgoDex evaluator 计算 MPJPE。

### WP4：Depth Anything adapter

**所有权**：`adapters/depth_anything.py`、depth QC 和 tests。

**输入**：WP0 schema、WP1 frames/K、可选 WP2 masks。

**输出**：metric depth、valid mask、统计和可视化。

**依赖**：WP0；mask 内统计依赖 WP2，但基础推理可并行。

**验收**：深度保持原图像素映射、单位明确、无只保存渲染图的问题。

### WP5：SAM 3D Objects mesh adapter

**所有权**：`adapters/sam3d_objects.py`、mesh proposal/repair/selection tests。

**输入**：WP2 masks/keyframes、WP4 depth。

**输出**：canonical mesh proposals、每个 proposal 的 visual/collision-source OBJ 与 metadata、`mesh_ranking.json`。WP5 不输出最终 mesh。

**依赖**：WP2，推荐等待 WP4。

**验收**：至少一个合格 proposal 可被 trimesh/MuJoCo 加载；静态排序只依赖 mesh integrity、silhouette 和 depth residual；不调用 FoundationPose。

### WP6：FoundationPose adapter

**所有权**：`adapters/foundationpose.py`、双向跟踪、重注册和 tests。

**输入**：WP1 frames/K、WP2 masks、WP4 depth、WP5 ranked mesh proposals。

**输出**：`selected_mesh.json`、`foundationpose_raw.npz`、所有候选的跟踪 QC、overlay。

**依赖**：WP1、WP2、WP4、WP5。

**验收**：按静态排名尝试 proposals，以固定 tracking score 选出最终 mesh；一个 clip 从 anchor 双向输出完整轨迹，所有候选结果、漂移帧和重注册记录可见。

### WP7：序列优化与接触

**所有权**：`optimization/` 全目录及 optimizer tests。

**输入**：WP0 schema、WP1 外参、WP2 masks、WP3 hands、WP4 depth、WP6 selected mesh/object pose。

**输出**：aligned trajectory、contact artifact、loss/quality metrics。

**依赖**：WP0、WP1、WP3、WP6；可先用合成 artifact 开发。

**验收**：合成数据可恢复已知变换；真实 clip 的 loss、penetration 和 jitter 相比 raw 输入下降。

### WP8：SPIDER exporter 与下游集成

**所有权**：`export/spider.py`、`export/spider_runner.py`、export tests。公共 CLI 只由 WP10 维护。

**输入**：WP0 schema、WP6 selected mesh、WP7 aligned/contact。

**输出**：标准 dataset tree、keypoints NPZ、task info、下游运行记录。

**依赖**：WP0；可先用合成 aligned trajectory 开发，最终依赖 WP6 和 WP7。

**验收**：合成和真实 clip 均能通过 SPIDER scene/IK loader；至少一个真实 clip 完成 MJWP。

### WP9：评测与可视化

**所有权**：`eval/`、`visualization/` 和相关 tests，不拥有各 adapter。

**输入**：所有 stage artifact、WP1 GT reader。

**输出**：统一 metrics JSON/CSV、raw-vs-aligned 必要诊断、主流程批量报告、overlay 和 3D scene 视频。

**依赖**：WP0、WP1；按 artifact schema 可与模型 adapter 并行开发。

**验收**：同一命令能汇总一批 clips，缺少某阶段时明确标记 missing 而非填零。

### WP10：CLI、编排与端到端测试

**所有权**：`cli.py`、configs、shell environment launcher、tiny pipeline test、用户文档。

**输入**：WP0 定义的公共接口及各工作包 stage entrypoint。

**输出**：`preflight/run-stage/run-all/evaluate/export-spider` 命令。

**依赖**：WP0 后即可搭骨架，最终集成依赖 WP1-WP9。

**验收**：阶段可独立重跑、缓存可复用、失败可恢复；端到端命令不会读取未授权 GT 字段。

## 11. 并行执行波次

```text
Wave 0:
  WP0  schema / manifest / coordinates

Wave 1（并行）:
  WP1  EgoDex ingest
  WP2  SAM 3
  WP3  WiLoR
  WP4  Depth Anything
  WP8  SPIDER exporter（使用合成 artifact）
  WP9  evaluator 骨架
  WP10 CLI 骨架

Wave 2:
  WP5  SAM 3D Objects ranked mesh proposals

Wave 3:
  WP6  FoundationPose proposal selection + tracking
  WP7  optimizer（先合成，后接真实 artifact）

Wave 4:
  WP8  真实 SPIDER 集成
  WP9  主流程评测与 raw-vs-aligned 必要诊断
  WP10  end-to-end 编排和回归测试
```

WP0 的 schema 合入前，其他 agent 只做探索性代码，不得各自发明永久 artifact 格式。

## 12. Agent 协作规则

每个 agent 开始前必须：

1. 阅读本文档及自己工作包的输入/输出协议。
2. 检查共享工作区已有修改，避免覆盖其他 agent 工作。
3. 只编辑工作包拥有的文件；公共接口变更先提交提案给 WP0/主 agent。
4. 使用 adapter 和 artifact 边界，不跨环境直接 import 第三方项目内部对象。
5. 在 `PROGRESS.md` 将对应工作包更新为 `in_progress`，并按其中的 Agent 更新格式记录本次范围；若依赖未满足则更新为 `blocked` 并记录阻塞条件。

每个 agent 交付时必须报告：

- 修改文件。
- artifact schema/version。
- 运行命令和环境名。
- 通过/失败的测试。
- 已知失败模式和未完成项。
- 一个最小真实或合成样例的输出路径。

上述交付信息必须同步写入 `PROGRESS.md`：工作包表、执行波次、资源阻塞和变更日志按实际情况一起更新。状态或交付物发生变化但未更新 `PROGRESS.md`，视为该增量尚未交付。

禁止行为：

- 静默修改坐标 convention、单位或 quaternion 顺序。
- 用 GT 手 pose 修补普通 pipeline 输出。
- 覆盖 raw artifact，只保留平滑后结果。
- 在 stage 失败时复制上一帧并仍标记 valid。
- 直接修改 third-party demo 形成不可追踪的本地 fork。
- 为了让下游运行而手工改 NPZ 数值但不记录 manifest。

## 13. 测试策略

### 13.1 不依赖模型的必需测试

- SE(3) compose/inverse/round-trip。
- OpenCV camera 到 sim frame 的轴向测试。
- quaternion `wxyz <-> matrix` round-trip。
- 30 Hz 到 50 Hz 的 translation/rotation resampling。
- schema invalid shape、NaN、非正交旋转和错误单位拒绝测试。
- synthetic hand/object/contact trajectory 导出 SPIDER 测试。
- manifest hash 和缓存失效测试。

### 13.2 Adapter smoke tests

每个 adapter 至少支持：

```text
--start-frame
--end-frame
--device
--output-dir
--overwrite
--dry-run
```

模型 smoke test 使用 3 到 10 帧，完整质量测试使用固定开发 clip。无 checkpoint/GPU 时应 skip 并说明原因，不能伪装通过。

### 13.3 端到端回归

- Tiny synthetic pipeline：CI/无 GPU 可运行。
- EgoDex smoke clip：需要模型环境和 GPU。
- EgoDex frozen clips：只在里程碑或发布候选运行。
- SPIDER MJWP：单独标记为 GPU/长时测试。

## 14. 风险与降级路径

| 风险 | 检测 | V1 降级 |
|---|---|---|
| EgoDex archive 不完整 | preflight/unzip test | 只准备固定 clips，不解压整包 |
| SAM 3 丢失目标 | mask QC | 自动尝试剩余关键词候选；全部失败则标记 episode failed |
| SAM 3D mesh 幻觉 | 多候选渲染评分 | 自动选择最低 residual proposal；没有合格 proposal 则标记 mesh_failed |
| 物体严重对称 | symmetry-aware score | 旋转按对称群评测；必要时只跟踪可观测轴 |
| Depth metric 偏差 | 静态背景/手尺寸/轮廓残差 | 降低 depth loss，依赖 sequence scale 优化 |
| FoundationPose 漂移 | mask/depth/速度 residual | 分段重注册并保留 invalid gap |
| WiLoR 遮挡抖动 | confidence/GT metrics | 短 gap 插值，长 gap 标 invalid |
| Mesh 不适合 MuJoCo | trimesh/MuJoCo load test | repair + convex decomposition，视觉/碰撞 mesh 分离 |
| 优化陷入局部极值 | loss breakdown/多初始化 | 分阶段求解、滑窗、多尺度初始化 |
| SPIDER 初始接触失败 | IK/contact overlay | 截取接触前起始帧，或启用 open-hand 初始化 |
| 第三方依赖冲突 | environment preflight | 独立 env + subprocess artifact bridge |

## 15. 里程碑

### M0：基础闭环

- WP0 完成。
- 合成 aligned trajectory 可导出到 SPIDER 并通过 loader/scene/IK。

### M1：感知 artifact 闭环

- 一个 EgoDex clip 完成 ingest、SAM 3、WiLoR、depth。
- 所有结果在同一 frame/K/extrinsics 约定下可视化。

### M2：物体轨迹闭环

- 一个 clip 产生 ranked mesh proposals，并由 FoundationPose tracking score 选出可用 canonical mesh。
- FoundationPose 完成双向跟踪并转换到 world/sim frame。

### M3：联合优化闭环

- 一个 clip 输出 aligned hand/object/contact。
- 相比 raw 结果，jitter、penetration 和渲染 residual 有定量改善。

### M4：SPIDER 完整闭环

- 一个真实 EgoDex clip 完成 mesh decomposition、scene、IK 和 MJWP。
- 生成 simulation video、tracking metrics 和可复现 manifest。

### M5：V1 验收

- 固定 20 clips 各运行一次主流程，并完成 raw-vs-aligned 必要诊断；不运行正式消融矩阵。
- 达到第 9.4 节功能门槛，或明确给出每个未达标指标的主要上游原因。
- 文档、环境检查、命令和失败恢复流程完整。

## 16. V1 之后

V1 完成后按以下顺序扩展：

1. 使用 HOT3D 的 CAD 和 object pose GT 做物体 6D 定量验证。
2. 引入 Video Depth Anything，提高深度时间一致性。
3. 扩展 instruction keyword 词典和自动文本目标识别覆盖率。
4. 增加 `camera_pose_source=slam`，支持只有 RGB 的普通移动相机视频。
5. 支持多物体及物体-物体接触。
6. 为 articulated object 引入多刚体拓扑与关节状态估计。
