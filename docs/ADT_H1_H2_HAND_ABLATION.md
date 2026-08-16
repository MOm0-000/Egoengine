# ADT Phase A Hand Ablation：H0 / H1 / H2

生成时间：2026-08-15

本页对应《EgoEngine_下一阶段上游改进与消融实验计划.md》Phase A。
固定同一批 5 条已有 `hands/wilor_stereo_raw.npz` 的 ADT 样本，保持深度、物体、sequence 模块不变，只替换 hand 单目模型 / stereo 融合。

## 1. 运行方式

H1（HaMeR 单目 + 原 stereo triangulation）：

```bash
cd /data_all/zzx/egoengine/video_to_spider
PYTHONPATH=. /home/zzx/miniconda3/envs/v2s-core/bin/python \
  scripts/run_adt_hamer_hand_benchmark.py --infer --overwrite --gpu 6 --max-runs 5
```

H2（HaMeR + 轻量 calibrated stereo bundle adjustment）：

```bash
cd /data_all/zzx/egoengine/video_to_spider
PYTHONPATH=. /home/zzx/miniconda3/envs/v2s-core/bin/python \
  scripts/run_adt_hamer_hand_ba_benchmark.py --overwrite --max-runs 5
```

产物：

- H1：`runs/adt_hamer_hand_benchmark_summary.json`、`runs/hamer_hand_benchmark_matrix.csv`
- H2：`runs/adt_hamer_hand_ba_benchmark_summary.json`、`runs/hamer_hand_ba_benchmark_matrix.csv`
- 各 run 的 `hands_hamer/`、`hands_right_hamer/`、`hands_hamer_ba/`

## 2. 结果对比

### 2.1 H0 Baseline（当前 WiLoR + triangulation）

| Run | hand | accepted | joint_valid_rate | required_frame_rate |
|---|---:|---:|---:|---:|
| WhiteLiddedTrashBin | left | False | 0.0460 | 0.0000 |
| meal_seq131/WoodenBowl | left | False | 0.0000 | 0.0000 |
| work_seq107/BookDeepLearning | left | False | 0.3762 | 0.0000 |
| work_seq107/BookDeepLearning | right | False | 0.6952 | 0.0667 |
| recognition_seq138/BookDeepLearning | left | False | 0.1413 | 0.0000 |
| BlackCeramicMug | left | False | 0.0175 | 0.0000 |

### 2.2 H1 HaMeR + 原 stereo fusion

| Run | hand | accepted | joint_valid_rate | required_frame_rate |
|---|---:|---:|---:|---:|
| WhiteLiddedTrashBin | left | False | 0.0397 | 0.0000 |
| meal_seq131/WoodenBowl | left | False | 0.0000 | 0.0000 |
| work_seq107/BookDeepLearning | left | False | 0.4270 | 0.0667 |
| work_seq107/BookDeepLearning | right | False | 0.5905 | 0.0667 |
| recognition_seq138/BookDeepLearning | left | False | 0.1159 | 0.0000 |
| BlackCeramicMug | left | False | 0.0143 | 0.0000 |

### 2.3 H2 HaMeR + calibrated stereo BA

| Run | hand | accepted | joint_valid_rate | required_frame_rate |
|---|---:|---:|---:|---:|
| WhiteLiddedTrashBin | left | False | 0.0603 | 0.0000 |
| meal_seq131/WoodenBowl | left | False | 0.0000 | 0.0000 |
| work_seq107/BookDeepLearning | left | False | 0.7095 | 0.4667 |
| work_seq107/BookDeepLearning | right | **True** | 0.9159 | 0.6333 |
| recognition_seq138/BookDeepLearning | left | False | 0.2286 | 0.0000 |
| BlackCeramicMug | left | False | 0.0254 | 0.0000 |

## 3. 结论

1. **H1 单换 HaMeR 没有整体改善**，work_seq107 右手 joint_valid_rate 还略低于 WiLoR baseline。因此不能把失败简单归因为“WiLoR 单目网络太弱”。

2. **H2 在检测充分的样本上显著改善**：work_seq107 右手从 rejected 变为 accepted，`joint_valid_rate` 0.916、`required_frame_rate` 0.633。这说明当左右视图都有可靠 2D 观测时，cross-view + metric bundle adjustment 是有效升级。

3. **当前 5 条 ADT 的主要阻塞是右视图 / 部分左视图手检测缺失**：
   - `meal_seq131`：HaMeR 左/右 detection 几乎全 0。
   - `BlackCeramicMug`：左/右只有 1/30 帧检出。
   - `WhiteLiddedTrashBin` / `recognition_seq138`：右视图 `valid_rate_right=0.0`。
   这些不是 stereo BA 能修复的，因为固定 P7 gate 要求左右视图都有观测。

4. Phase A 的 `P7 accepted 0/5 -> >=3/5` 目标未达到。根因与之前 `ADT_STEREO_P0` 报告一致：ADT 是灰度 wide-baseline SLAM pair，手检测域外退化，适合做几何 stress test，不适合作为最终 hand benchmark 的通过门槛。

## 4. 建议下一步

- 保留 H2 作为候选 hand branch：`hands_hamer_ba/hamer_stereo_ba_raw.npz`。
- 不再在 ADT 上继续调 MINK/P7 阈值；改为：
  1. 在 HOT3D 上验证 H2，得到真实双目 RGB ego-hand 的接受率；
  2. 或为 ADT 增加专门的手检测器适配（更低的检测阈值、灰度增强、body-keypoint 引导），但这是 detector 模块，不是本计划 H1/H2 的“单目模型 + stereo fusion”范畴；
  3. 按方案转入 Object Reconstruction（O0/O1）时，使用 H2 的 work_seq107 结果作为 hand 分支 baseline，不再让 hand 阻塞 object 上游评测。

## 5. 已新增文件

- `video_to_spider/adapters/hamer.py`
- `scripts/run_adt_hamer_hand_benchmark.py`
- `scripts/run_adt_hamer_hand_ba_benchmark.py`
- `docs/ADT_H1_H2_HAND_ABLATION.md`
- `runs/adt_hamer_hand_benchmark_summary.json`
- `runs/hamer_hand_benchmark_matrix.csv`
- `runs/adt_hamer_hand_ba_benchmark_summary.json`
- `runs/hamer_hand_ba_benchmark_matrix.csv`
