# ADT Stereo P0：VRS -> Rectified Stereo -> ingest-stereo

本页对应《ADT 上游 Benchmark 评测与改进方案》第一阶段 P0。目标是先把 Aria Digital
Twin 的左/右 SLAM 灰度流用官方标定转换为**已校正双目**，再交给现有
`video_to_spider.cli ingest-stereo`，不引入任何 GT depth/scale，也不手工猜测 baseline。

## 边界

- 只处理 `video.vrs` 中的 `camera-slam-left` / `camera-slam-right`。
- 左/右图由 Project Aria Tools 读取 VRS 内嵌 factory calibration。
- 目标相机是共享 upright linear/pinhole 模型；左相机为单位外参，右相机外参为
  `[+baseline, 0, 0]`，从而保证正确正视差。
- common valid mask 由全白源图通过同一官方 rectify 映射生成，不依赖场景内容。
- 时间戳来自左 SLAM capture timestamp，右图按 `CLOSEST` 查询并要求同步误差不超过
  `--max-sync-ns`。

## 安装

在 `v2s-core` 中安装固定的可选 ADT 依赖：

```bash
conda run -n v2s-core python -m pip install 'video-to-spider[adt] @ .'
```

或者直接安装：

```bash
conda run -n v2s-core python -m pip install projectaria-tools==1.0.0
```

不要安装较新的 `projectaria-tools` 版本；它们会拉入 `rerun-sdk>=0.20`，与当前
`numpy<2` 约束冲突。

## 运行准备

```bash
python scripts/prepare_adt_stereo.py \
  --video-vrs /path/to/ADT/sequence/video.vrs \
  --output-dir runs/adt_<sequence>/stereo_prepared \
  --closed-loop-trajectory /path/to/ADT/sequence/MPS/slam/closed_loop_trajectory.csv
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
python scripts/prepare_adt_stereo.py \
  --video-vrs ... --output-dir ... --static-camera
```

## 验收

先检查 `adt_stereo_prepare.json`：

```bash
jq '{baseline:.rectification.baseline_m, frame_count, fps,
     sync:.timestamp_sync, common_valid_pixel_ratio}' \
  runs/adt_<sequence>/stereo_prepared/adt_stereo_prepare.json
```

然后进入现有 ingest-stereo：

```bash
python scripts/prepare_adt_stereo.py \
  --video-vrs ... \
  --output-dir runs/adt_<sequence>/stereo_prepared \
  --closed-loop-trajectory ... \
  --run-ingest \
  --run-dir runs/adt_<sequence> \
  --task adt --episode-id <sequence> --instruction "pick up the object"
```

最终验收仍是 `calibration/stereo.json` 中的四项：

```text
sufficient_feature_correspondences
vertical_median_within_limit
vertical_p95_within_limit
left_right_order_positive_disparity
```

任一失败都说明 P0 未通过，不得进入 FoundationStereo。
