# 双目 Ego 单手动态物体：下一阶段实验方案（优化版）

## 1. 目标

在固定上游配置下，把“当前上游最大瓶颈”从经验判断推进到可量化定位：

1. 分清 Hand 瓶颈主要来自 **2D observation** 还是 **3D reconstruction**；
2. 分清 Hand、Object、Depth 各自对最终 Sequence / MINK-ready 的影响；
3. 在 **最终产品输入为双目 RGB 视频** 的约束下，选出真正可迁移的 Hand 方案。

原则：

- 最终推理只允许使用 RGB 输入和在线可计算几何，例如 calibrated stereo + FoundationStereo depth；
- GT、HOT3D 专用标注只用于评测和 oracle 诊断；
- 不通过 sequence optimizer 重写 object scale 或吸收上游误差；
- HOT3D 只负责“快速定位 Hand 问题”，ADT 负责“最终迁移和回归”；
- 一次只替换一个模块。

---

## 2. 数据与固定样本

### 2.1 ADT：最终回归集

继续使用：

- 固定 10 条 ADT 样本；
- 每条 30 帧窗口；
- 最终 mask branch；
- FoundationStereo metric depth；
- O5 hybrid object candidate pool；
- strict `validate_only` Sequence。

用途：

- 最终 Hand 迁移测试；
- Hand-Object 耦合诊断；
- Sequence / MINK-ready 回归；
- 判断“HOT3D 上最优”是否能真正改善最终 RGB 产品链路。

### 2.2 HOT3D：Hand 诊断集

选择 5–10 条单手样本，优先级：

1. 物体明显运动；
2. 手部存在遮挡、快速运动或运动模糊；
3. 手部在画面中占比中等或偏小，接近 ego 条件；
4. 左右目 / multi-view 可观测；
5. GT hand 3D 完整。

**数据准备前置条件：**

- 固定 `sequence_id + clip_id + frame range`；
- 确认 RGB、相机标定、MANO GT、object GT 可下载且许可证允许使用；
- 下载失败或标注缺失的样本要提前剔除，不能中途替换。

---

## 3. 固定上游配置

```text
Object Perception
→ Phase 0 final mask branch

Stereo Depth
→ FoundationStereo

Object Reconstruction
→ O5 hybrid candidate pool

Sequence
→ strict validate_only
```

所有后续实验只替换 Hand，或只替换 Object Oracle 中的单个输入来源。

---

## 4. Hand Oracle 分解

目标：把 Hand 问题拆成 2D 与 3D，判断重建上限。

### 4.1 H-Oracle-2D

输入：

```text
HOT3D GT 2D Hand
+ 真实 stereo calibration
+ 当前 Stereo BA / MANO
```

输出：

- 3D Hand；
- 重建误差。

指标：

- MPJPE；
- PA-MPJPE；
- PCK@10/20/30 mm；
- wrist position error；
- fingertip position error；
- wrist / fingertip orientation error；
- bone-length error。

判断：

- 如果 `GT 2D → 3D` 本身误差就大，说明瓶颈在 stereo calibration、MANO shape 或 BA；
- 如果误差很小，说明当前 pipeline 的瓶颈主要不在 3D reconstruction。

### 4.2 H-Oracle-3D

输入：

```text
预测 2D Hand
+ GT stereo calibration / camera geometry
```

与 H-Oracle-2D 使用相同指标。

判断：

```text
GT 2D → 3D
vs
Pred 2D → 3D
```

- 若两者接近且都好：3D 模块 OK，问题在 2D observation；
- 若两者接近且都差：3D/BA 是共同瓶颈；
- 若 Pred 明显差于 GT：2D detection / correspondence 是主要瓶颈。

### 4.3 H-Oracle-Downstream

输入：

```text
HOT3D GT 3D Hand
+ Pred Object
+ Pred Depth
+ Sequence
```

输出：

- hand-object relative pose；
- object pose error；
- Sequence export；
- MINK-ready。

用于回答：

> 如果 Hand 完全正确，当前上游还剩多少问题？

### 4.4 Oracle 执行规范

所有 GT hand 必须先转换为 pipeline 内部统一表示，并记录转换配置：

- 坐标系：`camera / world / MANO root-relative`；
- 左/右手和左右目对应关系；
- `translation` 与 `root-relative joints` 的拆分；
- scale 是否冻结；
- GT 2D 投影所用的 intrinsic/distortion。

缺少坐标说明的 oracle 结果不参与结论。

---

## 5. 2D Observation Audit

除 HOT3D Oracle 外，在 ADT 上做轻量 2D 诊断：

- 左右目是否都检测到手；
- 检测 confidence；
- required landmark 是否存在；
- 左右目对应 landmark 的 epipolar error；
- hand box 大小、离图像边缘距离；
- 运动模糊 / 遮挡比例。

目的：

- 避免只因为 HOT3D 条件好而高估某个 Hand 方法；
- 量化 ADT 中“没有 2D observation”这一已知失败模式。

---

## 6. Hand 方法横向比较

在 HOT3D 固定子集上比较：

```text
H0  WiLoR + stereo triangulation
H1  HaMeR + stereo fusion
H2  HaMeR + calibrated stereo BA
H3  UmeTrack
H4  POEM / POEM-v2
H5  Best 2D proposal + Stereo MANO BA
```

### 6.1 模块指标

3D Hand：

- MPJPE；
- PA-MPJPE；
- PCK；
- fingertip error；
- wrist error；
- orientation error。

Stereo / multi-view：

- reprojection error；
- epipolar error；
- triangulation error；
- depth consistency。

Temporal：

- per-joint velocity / acceleration error；
- jitter；
- trajectory smoothness。

Hand-Object：

- relative translation error；
- relative rotation error；
- fingertip-to-object distance；
- contact localization error。

### 6.2 初筛规则

HOT3D 上保留候选方法，必须满足：

```text
PA-MPJPE 或 PCK 处于合理范围
AND
跨 sequence 失败模式可解释
```

不允许只按平均 MPJPE 排序。

---

## 7. RGB 产品输入约束下的迁移测试

HOT3D 上排名高不代表最终可用。最终产品输入是 **RGB 视频**，所以：

- 任何候选 Hand 方法必须能在没有 HOT3D GT box / mask / MANO 先验的情况下运行；
- 只允许使用 RGB、相机标定和 FoundationStereo depth；
- 最终选择必须在 ADT 上产生真实下游收益。

迁移测试流程：

```text
HOT3D top-K Hand
→ 接入 ADT 10 条
→ 与 H2 / H5 baseline 比较
→ 检查 accepted hand rate + downstream
```

只有满足以下至少一项，才替换当前 Hand：

1. `accepted_any_hand` 从 `1/5` 提升；
2. `work_seq107` 稳定通过 `shared_metric_hand_depth_verified`；
3. ADT 上 joint/required frame valid rate 有显著且稳定的提升；
4. 不降低 Object / Depth / Sequence 其他指标。

如果 HOT3D 最优方法在 ADT 上没有收益，则结论应为：

> 瓶颈不在模型选择，而在 ADT 的 2D observation / domain gap。

---

## 8. Hand–Object Oracle 分解

在 ADT 固定 10 条上，对当前最优 Hand + O5 Object 做：

### O-Oracle

```text
Pred Hand + GT Object
```

### H-Oracle

```text
GT Hand + Pred Object
```

### Full Oracle

```text
GT Hand + GT Object
```

指标：

- object ADD / ADD-S；
- translation error；
- rotation error；
- mesh Chamfer / F-score（可用时）；
- hand-object relative translation / rotation error；
- Sequence export / MINK-ready。

判断：

- `O-Oracle` 差 → Object 是独立瓶颈；
- `H-Oracle` 差 → Hand 对下游有实质影响；
- `Full Oracle` 仍差 → Depth / calibration / Sequence 还有未解决问题。

### 数据前置条件

- 确认 ADT GT object mesh 是否可用；
- 确认 GT object trajectory 与当前窗口对齐；
- 确认 GT hand 在 ADT 中是否可用；
- 缺少 GT 的样本只做 Pred-Pred 基线，不强行做 Oracle。

---

## 9. 模块优化与最终收益比较

完成 Oracle 定位后，只优化影响最大的模块。

例如确定 Hand 是主要瓶颈：

```text
Current Hand
vs H2
vs H3
vs H4
vs H5
```

同时报告：

- 模块指标；
- ADT 下游指标；
- HOT3D 指标；
- 跨 sequence 方差和失败模式。

---

## 10. 最终完整回归

将选出的 Hand 与固定配置组合：

```text
Phase 0 final Object Perception
+ FoundationStereo
+ O5 Object Reconstruction
+ Best Hand
+ strict Sequence
```

重新运行 ADT 10 条，报告：

- Hand 3D accuracy；
- Object 6D pose / reconstruction accuracy；
- stereo depth accuracy；
- hand-object relative pose accuracy；
- Sequence export；
- MINK-ready；
- `metric_scale_rewritten` 必须保持为 0。

---

## 11. 实验组织与执行清单

所有实验固定：

- ADT 10 条；
- HOT3D 固定 5–10 条；
- 相同 frame range；
- 每次只改一个模块；
- GT 只用于评测 / oracle；
- 输出统一保留 `config.json / module_metrics.json / downstream_metrics.json / failure_reason.json / visualization/`。

启动前必须确认：

- [ ] HOT3D 样本可下载、可加载；
- [ ] Hand GT 坐标与 pipeline schema 的转换已定义；
- [ ] Object GT mesh / trajectory 可用性已确认；
- [ ] 当前 H2/H5 baseline 在 ADT 上的结果已复现；
- [ ] 所有 Hand 候选能在 RGB-only 条件下运行；
- [ ] 明确“最佳 Hand”的 ADT 迁移门槛。

---

## 12. 风险与回退

- 若 HOT3D 下载 / 许可证阻塞：先使用 ADT 2D observation audit，并继续用现有 H0–H5 做 ADT 回归。
- 若 HOT3D 最优模型无法迁移：不强行替换，结论转为“当前瓶颈是 domain gap / 2D observation”。
- 若 Object GT 不完整：保留 Pred-Pred、H-Oracle-2D/3D 和 sequence 结果，O-Oracle 仅做可用子集。
- 若最终仍卡在 `shared_metric_hand_depth_verified`：优先修 Hand 的 metric depth 一致性，而不是放松 sequence gate。

