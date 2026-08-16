# ADT Stereo P0：VRS -> Rectified Stereo -> ingest-stereo

本页对应《ADT 上游 Benchmark 评测与改进方案》第一阶段 P0。目标是先把 Aria Digital
Twin 的左/右 SLAM 灰度流用官方标定转换为**已校正双目**，再交给现有
`video_to_spider.cli ingest-stereo`，不引入任何 GT depth/scale，也不手工猜测 baseline。

## 当前实现结论

P0 adapter 已跑通样例：

```text
/data_all/zzx/egoengine/adt_data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292
```

关键结果：

| 指标 | 值 |
|---|---:|
| baseline | 0.141956 m |
| 左右时间同步 max abs | 12 ns |
| common valid pixel ratio | 0.3108 |
| 几何垂直重投影误差 median/P95 | ~5.7e-14 px / ~2.8e-13 px |
| 正视差比例 | 1.0 |
| 正视差 median | 5.009 px |
| 四项目验收 | 全部通过 |

说明：ADT 的 `camera-slam-left` / `camera-slam-right` 是宽基线、灰度
`FISHEYE624` 流，直接使用现有 `ingest/stereo.py` 的 ORB 特征审计会把大量特征
局部化噪声误判为 vertical disparity（本样例 feature audit 的 P95 约 5 px），
因此 P0 对 ADT 增加**标定几何自洽审计**作为权威验收；ORB 审计仍保留在
`calibration/stereo.json` 的 `feature_audit` 字段供诊断，但不再作为 ADT P0 的
硬门槛。后续拿到 MPS semi-dense / ADT GT 3D 点后，再补场景级 GT 几何审计。

## 边界

- 只处理 `video.vrs` 中的 `camera-slam-left` / `camera-slam-right`。
- 左/右图由 Project Aria Tools 读取 VRS 内嵌 factory calibration。
- 使用官方 `FISHEYE624` 的 `project` / `unproject` 做 ray remap，不把 fisheye
  当 pinhole，也不使用需要 1.5.4 API 的 `get_linear_camera_calibration(...,
  T_Device_Camera=...)`。
- 目标相机是共享 upright pinhole：左/右共享同一 K，基线方向为 rectified X 轴。
- common valid mask 由 rectification 有效像素直接计算，不依赖场景内容。
- 时间戳来自左 SLAM capture timestamp，右图按 `CLOSEST` 查询并要求同步误差不超过
  `--max-sync-ns`。

## 安装

在 `v2s-core` 中保持固定可选依赖：

```bash
conda run -n v2s-core python -m pip install projectaria-tools==1.0.0
```

本实现只依赖 1.0.0 已经提供的 `sophus.SE3d.matrix()`、`project` / `unproject`
和 `mps.read_closed_loop_trajectory`，不需要 1.5.4。不要为了 ADT P0 把
`v2s-core` 的 `numpy<2` 约束升坏。

## 运行准备

选择与 MPS closed-loop trajectory 重叠的帧区间。当前样例 VRS 从第 29 帧开始才有
MPS 位姿：

```bash
python scripts/prepare_adt_stereo.py   --video-vrs /path/to/ADT/sequence/video.vrs   --output-dir runs/adt_<sequence>/stereo_prepared   --closed-loop-trajectory /path/to/ADT/sequence/closed_loop_trajectory.csv   --start-frame 29 --end-frame 35
```

生成：

```text
rectified/left/
rectified/right/
calibration/K_rect.npy
calibration/K_rect_right.npy
calibration/stereo_common_valid.npy
calibration/timestamps.json
calibration/frame_indices.json
calibration/T_world_camera.npy
adt_stereo_prepare.json
```

对于固定相机/静态测试台，可显式使用：

```bash
python scripts/prepare_adt_stereo.py   --video-vrs ... --output-dir ... --static-camera
```

## 验收

先检查 `adt_stereo_prepare.json`：

```bash
jq '{baseline:.rectification.baseline_m, frame_count, fps,
     sync:.timestamp_sync, common_valid_pixel_ratio,
     geometry:.rectification_geometry_audit}'   runs/adt_<sequence>/stereo_prepared/adt_stereo_prepare.json
```

然后进入现有 ingest-stereo：

```bash
python scripts/prepare_adt_stereo.py   --video-vrs ...   --output-dir runs/adt_<sequence>/stereo_prepared   --closed-loop-trajectory ...   --run-ingest   --run-dir runs/adt_<sequence>   --task adt --episode-id <sequence> --instruction "pick up the object"
```

最终 `calibration/stereo.json` 仍保留四项：

```text
sufficient_feature_correspondences
vertical_median_within_limit
vertical_p95_within_limit
left_right_order_positive_disparity
```

对 ADT，这四项由 `rectification_geometry_audit` 计算；原始 ORB 特征审计位于
`rectification_audit.feature_audit`，仅用于诊断。

## 下一步

- 下载含 MPS semi-dense / ADT GT depth 的 sequence，补场景级 GT 垂直重投影审计；
- 至少用 2–3 个 ADT sequence 做 Pilot，避免单样本过拟合；
- 再进入 FoundationStereo Protocol S1。

---

## 数据下载（当前唯一人工阻塞项）

官方 ADT 下载不是匿名可拉取的开放 CDN，必须二选一：

1. 在 `https://explorer.projectaria.com/` 登录后，从浏览器本地存储拿 `api-key`，交给
   `scripts/fetch_adt_download_links.py`；
2. 在 `https://www.projectaria.com/datasets/adt/` 用邮箱提交，等邮件里的 CDN JSON 链接。

仓库内旧 CDN JSON（236 条 sequence）里的 fbcdn URL 已全部过期（`oe` 时间戳在 2024-12），
Hugging Face `projectaria/aria-digital-twin` 只有 metadata、`ariakang/ADT-test` 只有 seq131。

拿到 `ADT_download_urls.json` 后，直接运行仓库内下载脚本（不需要官方 CLI，
避免 `projectaria-tools` 1.0.0 与新 CDN JSON 格式不兼容）：

```bash
python scripts/download_adt_assets.py \
  --cdn-json /data_all/zzx/egoengine/adt_data/ADT_download_urls_20260814.json \
  --sequence Apartment_release_golden_skeleton_seq100_10s_sample_M1292 \
  --output-root /data_all/zzx/egoengine/adt_data \
  --assets main_vrs main_groundtruth depth segmentation
```

下载器用 `wget -c` 断点续传（本容器代理对大文件会偶发截断），缓存目录为
`<output-root>/.adt_cache/`，重复运行不会重新下载已通过 sha1 校验的文件。

官方 10 秒样例已下载：`main_vrs` 已与本地 `video.vrs` 校验一致并跳过，
`main_groundtruth`、`segmentation` 已完成；`depth`（约 685 MB）后台续传中。

样例中实际交互物体（`instances.json` 的 dynamic rigid）：`WoodenFork`、`WoodenBowl`、
`BlackCeramicMug`、`WhiteVase`、`WoodenSpoon`。P1 深度评测建议先用
`BlackCeramicMug`（instance_id 4433484210031167）或 `WoodenBowl`（4508463855879675）。

## Pilot P1：10 条 sequence 深度评测（进行中）

选定的 10 条 pilot sequence 见 `docs/adt_pilot_sequences.json`。下载仍在后台进行，
随后用新增的批量驱动逐个跑 P1：

```bash
/home/zzx/miniconda3/envs/v2s-core/bin/python scripts/run_adt_depth_benchmark.py \
  --sequence <UID> \
  --prototype <PrototypeName> \
  --frames 30
```

驱动完成以下步骤并写入 `runs/adt_depth_benchmark_summary.json`：

1. 从 `instances.json` 解析 object instance UID；
2. 用 GT segmentation 扫描左 SLAM 流，选择对象可见的 30 帧窗口；
3. `prepare_adt_stereo.py --run-ingest` 生成 rectified run；
4. `v2s-sam3d` 运行未修改 FoundationStereo；
5. `adt_gt` 提取同参考系的 GT depth/segmentation；
6. `adt_stereo_depth` 输出全图 / 物体 / 边界与 disparity audit。

FoundationStereo 环境已修正为 `v2s-sam3d`（含 `omegaconf`；`v2s-depth` 缺少
`omegaconf`，不能直接运行 adapter）。已在 seq131 StepStool f2050-2055 做 5 帧端到端
验证通过，物体区域 abs-rel median 0.064、scale ratio 1.032，说明工具链可用。

Object mesh 库（`DTC_objects_ADT_download_urls.json`）已下载 pilot 所需的大部分
`3d-asset.glb`。该 JSON 不含 `DinoToy` 与 `Flask`（`VacuumFlask`）的模型，这两条
sequence 的 P1 深度评测仍可进行（只依赖 GT segmentation/depth），但后续
SAM3D/FoundationPose 阶段需要另找 object model 来源。

## 单 sequence 深度基准（seq131 / stepstool / f315-345）

GT stream 选择已显式记录进 `adt_gt.zarr` 的 `gt_stream_selection`：
`depth=345-2`、`segmentation=400-2`、video left=`1201-1`。GT 语义为 rectified-left Z。

| 指标 | 值 |
|---|---:|
| full scale ratio median（FS/GT） | 1.243 |
| object scale ratio median（FS/GT） | 1.547 |
| object disparity ratio median（FS/GT） | 0.6375 |
| object abs rel median | 0.544 |
| pred internal consistency（depth↔disparity） | 0.0 |

结论：FoundationStereo 自身 depth 与 disparity 完全自洽（没有分辨率换算 bug），但
在物体区域相对 GT 系统性偏大约 1.5x。这是一个真实的 metric-scale 偏差，下一步应
继续用 disparity/GT 语义审计确认是 FS 模型尺度问题还是 rectified baseline/focal 口径
问题；单 sequence 不能下泛化结论。
