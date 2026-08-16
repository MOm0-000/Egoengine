# Hand 2D Observation 瓶颈定位：协议修订版

生成时间：2026-08-16

本协议是对《Hand_2D_Observation_瓶颈定位_下一步实验方案》的修订，不改变原方案总体方向，只补齐执行时容易引入不可比误差的定义。

上一轮结果基线：

- `docs/下一阶段Hand瓶颈定位_消融结果汇总.md`
- `runs/hot3d_hand_diagnosis/hand_oracle_summary.json`
- `runs/adt_hand_observation_audit_summary.json`

---

## 1. 目标

在固定数据域和固定几何协议下，把当前 Hand 上游误差分解为：

```text
2D observation coverage
2D landmark localization
cross-view correspondence
absolute root depth / metric scale
stereo reconstruction / BA
```

最终必须明确回答：

> 当前应优先优化哪一个模块。

---

## 2. 固定原则

1. 一次只替换一个误差来源。
2. GT 只用于评测和 oracle，不进入最终推理。
3. 所有实验使用相同 HOT3D clip、相同帧区间、相同手 side。
4. 所有评测先按 GT-visible 域过滤，不能把所有帧混在一起。
5. 所有 Pred 2D 均来自 RGB-only 模型；GT correspondence 只允许修正 identity，不允许替换 2D 坐标。
6. 不通过 Sequence 阶段重写 metric scale 来吸收 Hand 误差。
7. 每个 oracle 输出必须记录 `config.json`、指标、失败原因和可视化。

---

## 3. 固定数据

### 3.1 HOT3D 固定 8 clip

清单：

`runs/hot3d_hand_diagnosis/subset_manifest.json`

固定设置：

- `train_aria`
- 每条 30 帧
- 每条固定选定手 side 与主要目标物体
- 左右目使用 `1201-1` 与 `1201-2`

本协议沿用以下 8 条：

| Clip | 选定手 | 目标物体 |
|---|---:|---:|
| `clip-001852.tar` | left | 29 |
| `clip-001856.tar` | left | 24 |
| `clip-001891.tar` | right | 4 |
| `clip-002273.tar` | right | 22 |
| `clip-002316.tar` | right | 29 |
| `clip-002850.tar` | right | 17 |
| `clip-002921.tar` | right | 19 |
| `clip-002974.tar` | left | 10 |

### 3.2 ADT 固定 10 条

清单：

`runs/adt_depth_benchmark_summary.json`

用途：

- 迁移门槛验证；
- 2D observation 压力测试；
- strict Sequence / MINK-ready 回归。

---

## 4. 固定几何与坐标协议

### 4.1 HOT3D 相机

HOT3D-Clips 的 SLAM left/right 为 fisheye。当前实验采用以下 benchmark-only 转换：

```text
source fisheye camera
→ same-size PinholePlaneCameraModel
→ 使用 source focal length / principal point
→ warp image 到该 pinhole
```

该转换不是生产 ADT rectification，必须写入 `config.json`：

```json
{
  "camera_protocol": "same_size_pinhole_from_source_fisheye",
  "note": "benchmark-only; not production rectification"
}
```

### 4.2 GT Hand 坐标

GT Hand 统一使用 MANO 模型输出 21 个标准 landmark：

```text
HOT3D hands.json
→ mano_theta / wrist_xform / mano_beta
→ MANOHandModel.forward
→ world 21 landmarks
→ T_world_camera inverse
→ left/right camera landmarks
→ pinhole K projection
→ GT 2D
```

landmark 顺序必须与 pipeline 内 `joints_camera_rootrel` 的 21 点顺序完全一致。

### 4.3 Pred Hand 坐标

使用现有标准 artifact：

```text
WiLoR:
  hands/wilor_raw.npz
  hands_right/wilor_raw.npz

HaMeR:
  hands_hamer/hamer_raw.npz
  hands_right_hamer/hamer_raw.npz
```

绝对相机坐标：

```text
joints_camera_rootrel + translation_camera
```

2D 投影：

```text
uv = project(K, joints_camera_absolute)
```

---

## 5. 实验 0：评测域冻结

每个 view 的 `GT-visible` 定义为同时满足：

1. `hands.json[selected_side]["boxes_amodal"][stream_id]` 存在；
2. GT hand 的 wrist 3D 投影在该 camera 下 `z > 0`；
3. GT hand 的 required landmarks 至少有 1 个落在图像范围内。

固定输出：

```text
runs/hot3d_hand_diagnosis/eval_domain.json
```

结构：

```json
{
  "clip": "clip-001852.tar",
  "selected_side": "left",
  "frames": 30,
  "gt_visible_left": [true, ...],
  "gt_visible_right": [true, ...],
  "gt_visible_both": [true, ...]
}
```

后续所有 detection rate、2D error、correspondence 指标，只在对应 GT-visible 域内计算。

---

## 6. 实验 1：2D Detection / Localization 诊断

### 6.1 输入

```text
WiLoR left/right
HaMeR left/right
HOT3D GT 2D projection
```

### 6.2 定义

对选定手 side `s`：

```text
P_left[f]  = model valid[f,s] and score[f,s] >= 0.2
P_right[f] = model valid[f,s] and score[f,s] >= 0.2

required_available_left[f] = all required joints finite and z > 0
required_available_right[f] = same
```

### 6.3 指标

仅在 GT-visible 域内统计：

```text
left detection rate
right detection rate
both detection rate
left required-landmark rate
right required-landmark rate
median confidence
```

2D accuracy 只在 detection 成功帧统计：

```text
mean / median / p95 2D keypoint error
wrist 2D error
fingertip 2D error
2D PCK @ 5px / 10px / 20px
```

分维度输出：

```text
left-only
right-only
both-view frames
```

### 6.4 判定

```text
detection rate 低
→ observation coverage 优先

detection rate 高但 2D PCK 差
→ 2D landmark localization 优先
```

### 6.5 输出

```text
runs/hot3d_hand_diagnosis/exp1_2d_localization_summary.json
runs/hot3d_hand_diagnosis/exp1_2d_localization_matrix.csv
```

---

## 7. 实验 2：Cross-view Correspondence 诊断

### 7.1 数据域

只使用：

```text
GT-visible both
AND
P_left and P_right
```

### 7.2 指标

```text
Sampson epipolar distance
left→right reprojection error
right→left reprojection error
triangulated positive-depth ratio
triangulation reprojection error
```

### 7.3 C0 / C1 定义

#### C0：Pred 2D + 当前 correspondence

```text
left model output side s
+ right model output side s
+ 当前 joint index
→ triangulation
```

不做任何 GT 修正。

#### C1：Pred 2D + GT correspondence

只修正 identity，不修正 2D 坐标：

```text
若某 view 有多个 hand detection，
按 GT hand side s 选择对应实例；
若 side label 不可用或冲突，选择与另一 view 几何最一致的实例；
仍然使用该实例原始的 predicted 2D landmarks。
```

禁止：

- 使用 GT 2D 坐标；
- 使用 GT 3D 重投影替换 predicted 2D；
- 使用 GT box 重新裁剪模型输出。

### 7.4 比较

分别对 WiLoR / HaMeR 计算 C0 与 C1 的：

```text
MPJPE
wrist error
fingertip error
reprojection error
```

判定：

```text
C1 >> C0
→ cross-view correspondence 是独立瓶颈

C1 ≈ C0 且 3D 仍差
→ 问题在 2D localization 或 absolute depth/scale
```

### 7.5 输出

```text
runs/hot3d_hand_diagnosis/exp2_correspondence_summary.json
runs/hot3d_hand_diagnosis/exp2_correspondence_matrix.csv
```

---

## 8. 实验 3：Absolute Depth / Scale Oracle

### 8.1 基础定义

对每一帧，得到选定手：

```text
Pred joints in left camera
GT joints in left camera
```

定义以下 4 个变体：

```text
D0 = Pred raw

D1 = D0 平移，使 pred wrist 与 GT wrist 重合

D2 = D0 对 root-relative joints 做全局 metric scale，
     使 pred wrist-to-fingertip bone scale 与 GT 一致

D3 = D0 先 D2 再 D1
```

这里的 D1/D2/D3 是 oracle 诊断，不进入生产推理。

### 8.2 指标

```text
MPJPE
root-aligned MPJPE
PA-MPJPE
wrist position error
fingertip position error
hand scale error
root depth error
```

### 8.3 判定

```text
D1/D3 的 absolute MPJPE 大幅下降
但 PA-MPJPE 变化小
→ absolute root depth 是主要瓶颈

D2/D3 的 root-aligned MPJPE 明显下降
→ metric hand scale 是主要瓶颈

D1/D2/D3 都仍差
→ 2D localization / articulation 仍占主导
```

### 8.4 可选增强

如果简单对齐后仍不明确，再使用 constrained reconstruction：

```text
Pred 2D reprojection
+ GT root depth anchor
+ GT hand scale anchor
+ MANO bone prior
```

但增强版必须作为单独 `oracle_mode=constrained` 记录，不与 D0–D3 混用。

### 8.5 输出

```text
runs/hot3d_hand_diagnosis/exp3_depth_scale_summary.json
runs/hot3d_hand_diagnosis/exp3_depth_scale_matrix.csv
```

---

## 9. ADT 10 条迁移验证

HOT3D 上确定主要瓶颈后，再在 ADT 固定 10 条上验证对应修复。

### 9.1 固定配置

```text
Object Perception  = Phase 0 final mask branch
Stereo Depth       = FoundationStereo
Object Reconstruction = O5 hybrid candidate pool
Sequence           = strict validate_only
```

只替换 Hand 2D observation 或对应的深度/尺度修复。

### 9.2 迁移门槛

至少满足以下一项才可替换当前 Hand：

```text
1. accepted_any_hand 从 1/5 提升
2. work_seq107 稳定通过 shared_metric_hand_depth_verified
3. ADT joint/required frame valid rate 有显著且稳定提升
4. 不降低 Object / Depth / Sequence 其他指标
```

### 9.3 最终回归

```text
ADT 10 条
→ accepted hand rate
→ object pose / reconstruction
→ stereo depth
→ hand-object relative pose
→ strict Sequence
→ metric_scale_rewritten == 0
```

输出：

```text
runs/hot3d_hand_diagnosis/adt_migration_summary.json
runs/hot3d_hand_diagnosis/adt_migration_matrix.csv
```

---

## 10. 联合判断表

| 现象 | 主要瓶颈 |
|---|---|
| both-view detection rate 很低 | 2D observation coverage |
| detection 高但 2D PCK 差 | 2D landmark localization |
| 2D accuracy 尚可但 C1 显著优于 C0 | cross-view correspondence |
| D1 显著改善 absolute MPJPE | absolute root depth |
| D2 显著改善 root-aligned MPJPE | metric hand scale |
| 上述均较好但 3D 仍差 | stereo reconstruction / BA |

---

## 11. 实验顺序

```text
0. 评测域冻结
1. Detection / Localization
2. Cross-view Correspondence
3. Absolute Depth / Scale Oracle
4. 确定主瓶颈
5. 只针对主瓶颈做最小修复
6. ADT 10 条迁移验证
7. work_seq107 shared_metric_hand_depth_verified
8. strict Sequence / MINK-ready
```

---

## 12. 最终交付

统一诊断表字段：

```text
Clip
Method
Detection coverage
2D PCK / pixel error
Epipolar error
Correspondence quality
Root depth error
Scale error
MPJPE
PA-MPJPE
Wrist error
Fingertip error
```

最终结论必须明确写出：

> 当前 Hand 上游主要应优化：coverage / localization / correspondence / depth / scale。

---

## 13. 风险与回退

- 若 HOT3D joint-level GT visibility 无法可靠获得，只使用 hand-level visibility，并明确记录该限制。
- 若 C1 无法无歧义实现，改为报告“correspondence oracle 不可构造”，不强行近似。
- 若 ADT 缺少 Hand GT，则不做 ADT 的 D1–D3 absolute oracle，只用 ADT accepted rate 与 Sequence gate。
- 若修复后 ADT 无迁移收益，结论应转为 domain gap / observation coverage，而不是继续替换 Hand backbone。

---

## 14. 环境隔离

- GT MANO / oracle：`v2s-hamer`
- WiLoR：`v2s-wilor`
- HaMeR：`v2s-hamer`
- ADT audit / Sequence：`v2s-core`

每个外部模型继续独立环境，不向 base 安装依赖。
