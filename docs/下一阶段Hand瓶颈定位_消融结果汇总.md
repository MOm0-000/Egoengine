# 下一阶段 Hand 瓶颈定位：消融实验结果汇总

生成时间：2026-08-16

对应协议：`docs/Hand_2D_Observation_瓶颈定位_协议修订版.md`

结论先行：

- 本次已修复 HOT3D 标定读取中的 `quaternion_wxyz` 顺序 bug，并重新生成 8 条 clip 的 pinhole benchmark 标定。
- 在冻结的 `GT-visible` 评测域上，`GT 2D -> 3D` oracle 仍接近零误差，说明固定相机几何下的三角化/重建不是当前主瓶颈。
- WiLoR 与 HaMeR 的 2D observation coverage 是最大瓶颈：8 条中没有任何一条在 `GT-visible both` 域上同时完成左右目 required-landmark 观测。
- 2D localization 在“检测成功”的帧里也明显偏弱：可见视图的中位 2D 关键点误差约 `42–106 px`，`PCK@5px` 普遍低于 `3%`。
- 可评测的单目深度/尺度 oracle 显示，`D1`（wrist 对齐）比 `D2`（global hand scale）更有效；absolute root depth 和 2D/articulation 同时是主要残差，不是单纯 metric hand scale。
- ADT 10 条迁移回归保持：H0/H1/H3/H4 `0/5`，H2/H5 `1/5`；strict Sequence `0/10`；`metric_scale_rewritten` 违规数仍为 `0`。

---

## 1. 执行环境与固定数据

### 1.1 HOT3D 固定 8 clip

清单：`runs/hot3d_hand_diagnosis/subset_manifest.json`

数据源：`bop-benchmark/hot3d`，匿名可下载，未触发真人登录/注册。

每条 clip 取前 30 帧，固定选定手 side，左右目使用 `1201-1` 与 `1201-2`。

### 1.2 环境隔离

- GT MANO / oracle：`v2s-hamer`
- WiLoR：`v2s-wilor`
- HaMeR：`v2s-hamer`
- ADT audit / Sequence：`v2s-core`

未向 base 环境安装依赖。

### 1.3 已修复的协议问题

`scripts/prepare_hot3d_hand_runs.py` 原先将 HOT3D 的 `quaternion_wxyz` 误按 `xyzw` 解析，导致 `T_world_camera` 旋转矩阵错误。已改为 `w,x,y,z -> Rotation.from_quat([x,y,z,w])`，并重新生成标定。

修复前旧 `hand_oracle_summary.json` 的接近零误差属于自洽但几何错误的数值；修复后以本文件列出的新协议产物为准。

---

## 2. 实验 0：冻结 GT-visible 评测域

结果文件：

- `runs/hot3d_hand_diagnosis/eval_domain.json`
- `runs/hot3d_hand_diagnosis/exp0_config.json`
- `runs/hot3d_hand_diagnosis/exp0_domain_vis.png`

GT-visible 定义：

1. `hands.json[selected_side]["boxes_amodal"][stream_id]` 存在；
2. GT hand wrist 3D 投影在该 camera 下 `z > 0`；
3. required joints `[0,4,8,12,16,20]` 至少有 1 个落在 pinhole 图像范围内。

| Clip | Side | GT-visible left | GT-visible right | GT-visible both | GT 2D oracle both 域 |
|---|---|---:|---:|---:|---|
| `clip-001852.tar` | left | 30 | 0 | 0 | 无候选 |
| `clip-001856.tar` | left | 30 | 7 | 7 | 7/7，MPJPE ≈ 0 |
| `clip-001891.tar` | right | 0 | 30 | 0 | 无候选 |
| `clip-002273.tar` | right | 0 | 30 | 0 | 无候选 |
| `clip-002316.tar` | right | 4 | 30 | 4 | 4/4，MPJPE ≈ 0 |
| `clip-002850.tar` | right | 0 | 30 | 0 | 无候选 |
| `clip-002921.tar` | right | 0 | 30 | 0 | 无候选 |
| `clip-002974.tar` | left | 30 | 0 | 0 | 无候选 |

结论：

> 在真正可评测的 `GT-visible both` 帧上，`GT 2D -> 3D` oracle 仍为机器零误差。固定相机几何下的三角化/重建不是主瓶颈。

---

## 3. 实验 1：2D Detection / Localization

结果文件：

- `runs/hot3d_hand_diagnosis/exp1_2d_localization_summary.json`
- `runs/hot3d_hand_diagnosis/exp1_2d_localization_matrix.csv`
- `runs/hot3d_hand_diagnosis/exp1_config.json`
- `runs/hot3d_hand_diagnosis/exp1_2d_localization_vis.png`

Detection 定义：`model valid` 且 `score >= 0.2` 且所有 21 个 joint 有限且 `z > 0`。所有 rate 均在 GT-visible 域内计算。

| Clip | Side | Model | L det | R det | Both det | 可见视图 2D median |
|---|---|---|---:|---:|---:|---:|
| `clip-001852.tar` | left | WiLoR | 0.00 | 0.00 | 0.00 | N/A |
| `clip-001852.tar` | left | HaMeR | 0.00 | 0.00 | 0.00 | N/A |
| `clip-001856.tar` | left | WiLoR | 0.83 | 0.00 | 0.00 | 44.0 px |
| `clip-001856.tar` | left | HaMeR | 0.83 | 0.00 | 0.00 | 47.8 px |
| `clip-001891.tar` | right | WiLoR | 0.00 | 0.20 | 0.00 | 43.1 px |
| `clip-001891.tar` | right | HaMeR | 0.00 | 0.20 | 0.00 | 43.9 px |
| `clip-002273.tar` | right | WiLoR | 0.00 | 0.73 | 0.00 | 52.2 px |
| `clip-002273.tar` | right | HaMeR | 0.00 | 0.73 | 0.00 | 51.5 px |
| `clip-002316.tar` | right | WiLoR | 0.00 | 1.00 | 0.00 | 42.3 px |
| `clip-002316.tar` | right | HaMeR | 0.00 | 1.00 | 0.00 | 43.3 px |
| `clip-002850.tar` | right | WiLoR | 0.00 | 1.00 | 0.00 | 74.4 px |
| `clip-002850.tar` | right | HaMeR | 0.00 | 1.00 | 0.00 | 73.4 px |
| `clip-002921.tar` | right | WiLoR | 0.00 | 0.53 | 0.00 | 51.5 px |
| `clip-002921.tar` | right | HaMeR | 0.00 | 0.53 | 0.00 | 50.9 px |
| `clip-002974.tar` | left | WiLoR | 1.00 | 0.00 | 0.00 | 106.3 px |
| `clip-002974.tar` | left | HaMeR | 1.00 | 0.00 | 0.00 | 103.3 px |

关键现象：

- 8 条中没有任何一条达到 `both detection > 0`。
- 选定手侧在 GT-visible 视图中常出现单侧完全漏检：`clip-001852` 左侧 GT-visible 30 帧，但 WiLoR/HaMeR 均为 0。
- 检测成功帧的 2D accuracy 仍不强：可见视图中位误差 `42–106 px`，`PCK@5px` 普遍低于 `3%`。
- WiLoR 与 HaMeR 差异很小，换 backbone 不改变 coverage/localization 瓶颈。

判定：

> 2D observation coverage 是第一优先级；检测成功后 2D landmark localization 仍需继续优化。

---

## 4. 实验 2：Cross-view Correspondence

结果文件：

- `runs/hot3d_hand_diagnosis/exp2_correspondence_summary.json`
- `runs/hot3d_hand_diagnosis/exp2_correspondence_matrix.csv`
- `runs/hot3d_hand_diagnosis/exp2_config.json`

数据域要求：

```text
GT-visible both
AND
P_left and P_right
```

结果：

- `16 / 16` 个 model×clip 组合均为 `no_both_view_frames`。
- C0 与 C1 都无法在当前固定域上构造，因此不能把 cross-view correspondence 独立拆出来作为当前首要瓶颈。
- 这本身说明覆盖问题是 correspondence 的上游阻断项。

判定：

> 在 2D coverage 未解决前，C0/C1 correspondence oracle 不可构造；应优先修复 coverage，再回到 correspondence。

---

## 5. 实验 3：Absolute Depth / Scale Oracle

结果文件：

- `runs/hot3d_hand_diagnosis/exp3_depth_scale_summary.json`
- `runs/hot3d_hand_diagnosis/exp3_depth_scale_matrix.csv`
- `runs/hot3d_hand_diagnosis/exp3_config.json`
- `runs/hot3d_hand_diagnosis/exp3_depth_scale_vis.png`

参考视图：left camera；只在 left GT-visible 且模型 required-available 的帧上计算。

可评测组合：`clip-001856`（25 帧）、`clip-002974`（30 帧）。

| Clip | Model | Variant | MPJPE | Root-aligned | PA-MPJPE |
|---|---|---|---:|---:|---:|
| `clip-001856.tar` | WiLoR | D0 | 90.6 | 74.3 | 65.9 |
| `clip-001856.tar` | WiLoR | D1 | 74.3 | 74.3 | 65.9 |
| `clip-001856.tar` | WiLoR | D2 | 86.6 | 71.0 | 60.2 |
| `clip-001856.tar` | WiLoR | D3 | 71.0 | 71.0 | 60.2 |
| `clip-001856.tar` | HaMeR | D0 | 92.1 | 74.9 | 66.0 |
| `clip-001856.tar` | HaMeR | D1 | 74.9 | 74.9 | 66.0 |
| `clip-001856.tar` | HaMeR | D2 | 88.2 | 71.4 | 60.3 |
| `clip-001856.tar` | HaMeR | D3 | 71.4 | 71.4 | 60.3 |
| `clip-002974.tar` | WiLoR | D0 | 107.8 | 78.9 | 59.4 |
| `clip-002974.tar` | WiLoR | D1 | 78.9 | 78.9 | 59.4 |
| `clip-002974.tar` | WiLoR | D2 | 106.9 | 77.1 | 55.9 |
| `clip-002974.tar` | WiLoR | D3 | 77.1 | 77.1 | 55.9 |
| `clip-002974.tar` | HaMeR | D0 | 107.9 | 79.0 | 59.7 |
| `clip-002974.tar` | HaMeR | D1 | 79.0 | 79.0 | 59.7 |
| `clip-002974.tar` | HaMeR | D2 | 106.9 | 77.5 | 56.2 |
| `clip-002974.tar` | HaMeR | D3 | 77.5 | 77.5 | 56.2 |

定义：

```text
D0 = Pred raw
D1 = D0 平移，使 pred wrist 与 GT wrist 重合
D2 = D0 对 root-relative joints 做全局 metric scale
D3 = D2 后再 D1
```

关键现象：

- `D1` 将 absolute MPJPE 降低约 `16–29 mm`，说明 absolute root depth/wrist translation 是明显残差。
- `D2` 只额外改善 root-aligned MPJPE 约 `1–4 mm`，说明 metric hand scale 不是主要残差。
- 即使 D3 后，PA-MPJPE 仍约 `56–60 mm`，root-aligned MPJPE 仍约 `71–79 mm`，说明 articulation / 2D localization 残差仍占主导。

判定：

> 在已有观测的帧中，瓶颈排序为：`2D/articulation localization > absolute root depth > metric hand scale`。

---

## 6. ADT 10 条迁移验证

结果文件：

- `runs/hot3d_hand_diagnosis/adt_migration_summary.json`
- `runs/hot3d_hand_diagnosis/adt_migration_matrix.csv`
- `runs/adt_hand_observation_audit_summary.json`
- `runs/adt_sequence_benchmark_all_summary.json`

固定配置：

```text
Object Perception  = Phase 0 final mask branch
Stereo Depth       = FoundationStereo
Object Reconstruction = O5 hybrid candidate pool
Sequence           = strict validate_only
```

### 6.1 Hand 方法 H0–H5

| 实验 | 方法 | 可评测 | accepted_any_hand |
|---|---|---:|---:|
| H0 | WiLoR + stereo triangulation | 5 | 0/5 |
| H1 | HaMeR + stereo fusion | 5 | 0/5 |
| H2 | HaMeR + calibrated stereo BA | 5 | 1/5 |
| H3 | UmeTrack | 5 | 0/5 |
| H4 | POEM / POEM-v2 | 5 | 0/5 |
| H5 | Best proposal + stereo MANO/depth/bone BA | 5 | 1/5 |

唯一 accepted 仍是：

```text
work_seq107 / BookDeepLearning 右手
```

### 6.2 ADT 2D observation audit

复现同一模式：

- 只有 `work_seq107` 右手达到左右目同时高观测。
- 其他样本普遍缺失一侧或双侧 Hand observation。
- WiLoR / HaMeR 在 ADT 上同样无法通过换 backbone 解决 coverage。

### 6.3 strict Sequence 回归

- strict 通过：`0/10`
- `work_seq107` 只缺 `shared_metric_hand_depth_verified`
- `work_seq107` 的 `scale_delta=-2.65e-09`
- `metric_scale_rewritten` 违规数：`0`

结论：

> Sequence 当前未通过不是 object metric scale 被 rewrite 造成的；`work_seq107` 剩余失败仍是 Hand 与 metric depth 的一致性 gate。

---

## 7. 联合判断

| 现象 | 主要瓶颈 |
|---|---|
| both-view detection rate 全为 0 | 2D observation coverage |
| 单视图检测成功但 `PCK@5px` 普遍 < 3% | 2D landmark localization |
| C0/C1 无法构造 | correspondence 的上游 coverage 阻断 |
| D1 显著改善 absolute MPJPE | absolute root depth |
| D2 改善有限 | metric hand scale 非主要 |
| D3 后 PA/root-aligned 仍高 | articulation / 2D localization |

最终判断：

> 当前 Hand 上游应优先优化 `2D observation coverage`，其次优化 `2D landmark localization / articulation`，然后才是 `absolute root depth`。不建议继续把主要精力放在 stereo 三角化、metric hand scale 或替换 hand backbone。

---

## 8. 交付物清单

HOT3D 修订版诊断：

- `runs/hot3d_hand_diagnosis/eval_domain.json`
- `runs/hot3d_hand_diagnosis/exp1_2d_localization_summary.json`
- `runs/hot3d_hand_diagnosis/exp1_2d_localization_matrix.csv`
- `runs/hot3d_hand_diagnosis/exp2_correspondence_summary.json`
- `runs/hot3d_hand_diagnosis/exp2_correspondence_matrix.csv`
- `runs/hot3d_hand_diagnosis/exp3_depth_scale_summary.json`
- `runs/hot3d_hand_diagnosis/exp3_depth_scale_matrix.csv`

ADT 迁移回归：

- `runs/hot3d_hand_diagnosis/adt_migration_summary.json`
- `runs/hot3d_hand_diagnosis/adt_migration_matrix.csv`
- `runs/adt_hand_observation_audit_summary.json`
- `runs/adt_sequence_benchmark_all_summary.json`

可视化：

- `runs/hot3d_hand_diagnosis/exp0_domain_vis.png`
- `runs/hot3d_hand_diagnosis/exp1_2d_localization_vis.png`
- `runs/hot3d_hand_diagnosis/exp3_depth_scale_vis.png`

脚本：

- `scripts/run_hot3d_hand_diagnosis_experiments.py`
- `scripts/summarize_hand_bottleneck_migration.py`
- `scripts/visualize_hot3d_hand_diagnosis.py`
