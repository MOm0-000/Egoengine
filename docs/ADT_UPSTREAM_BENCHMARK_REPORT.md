# ADT 上游 Benchmark 阶段性报告

> 最新 Hand/Object/Depth 三阶段 paired 消融汇总见 [`ADT_上游改进与消融实验总结.md`](./ADT_上游改进与消融实验总结.md)。本文件保留较早阶段记录。

本报告对应《ADT_上游_Benchmark_评测与改进方案.md》的当前执行结果。  
范围截止到 MINK 之前，目标是定位双目上游中真正阻塞 EgoEngine 3.1/3.2 的模块，而不是提高平均分。

## 关键产物

| 阶段 | 文件 |
|---|---|
| P1 深度评测 | `runs/adt_depth_benchmark_summary.json` |
| P2/P4/P7 审计 | `runs/adt_upstream_audit_summary_current.json` |
| P5 GT Replacement | `runs/adt_p5_gt_replacement_summary.json` |
| P5 表格 / 排序 | `runs/gt_replacement_matrix.csv`、`runs/bottleneck_ranking.csv` |
| P6 Sequence | `runs/adt_sequence_benchmark_summary.json`、`runs/sequence_benchmark_matrix.csv` |
| P7 Stereo Hand | `runs/adt_stereo_hand_benchmark_summary.json`、`runs/stereo_hand_benchmark_matrix.csv` |
| P7 失败拆解 | 各 run 的 `hands/hand_stereo_failure_analysis.json` / `.npz` |

## 1. P5 GT Replacement 初步结论

P5 已补齐失败样本记录，并把 G1 calibration rescue 纳入统一 ranking。当前有效 base run 为 4 条，仍是 preliminary，不作为最终结论。

| Run | f0 centroid median (mm) | depth_gt rescue (mm) | mesh_gt rescue (mm) | mask_gt rescue (mm) |
|---|---:|---:|---:|---:|
| seq138/BookDeepLearning f2177-2207 | 33.51 | 24.35 | 24.23 | failed |
| Lite_seq032/WoodenBowl f1771-1801 | 59.06 | failed | failed | failed |
| meal_seq131/WoodenBowl f2692-2722 | 42.82 | 27.73 | 20.40 | 5.83 |
| work_seq107/BookDeepLearning f1579-1609 | 12.05 | -2.73 | -2.35 | -12.57 |

汇总脚本给出的当前排序：

| 模块 | available | failure | median centroid rescue (mm) | positive fraction |
|---|---:|---:|---:|---:|
| depth_gt | 4 | 1 | 24.35 | 2/3 |
| mesh_gt | 4 | 1 | 20.40 | 2/3 |
| mask_gt | 4 | 2 | 5.83 | 1/2 |
| calibration_gt | 2 | 1 | -0.05 | 0/1 |

解释：

- `depth_gt` 在 seq138 与 meal_seq131 上显著改善物体质心误差，说明当前 FoundationStereo/投影后的物体深度仍是关键误差源之一；但在 Lite_seq032/WoodenBowl 上 GT depth 使 FoundationPose tracking gate 拒绝，说明该样本还受深度投影/尺度一致性问题影响。
- `mesh_gt` 在 seq138 上把 rotation error 从约 175.7° 降到约 1.6°，说明对旋转对称/细长物体，SAM3D mesh shape/scale 会极大影响旋转估计；Lite_seq032/WoodenBowl 的 GT mesh 也未通过 tracking，说明不能只把原因归到 mesh shape。
- `mask_gt` 不是主要救回项，且失败数更高；不应作为本轮优先修复对象。
- `calibration_gt` 在 full-upstream stereo branch 上 rescue 约 -0.05 mm，当前不是第一瓶颈。

## 2. P6 Sequence：严格 `validate_only`

P6 已把 object sequence 与 failing stereo-hand auxiliary 解耦：克隆 run 时不再携带 `hands/wilor_stereo_raw.npz`，但保留单目 `wilor_raw.npz`。这不是放宽手部门禁，而是让 object 轨迹 benchmark 不被 H Protocol 的辅助手部 artifact 阻塞；`strict_shared_metric` 仍会把单目 WiLoR 手深度标记为未验证。

当前只有 2 条 source run 具备进入 sequence 所需的上游 artifacts。

| Run | Sequence 状态 | raw centroid median (mm) | aligned centroid median (mm) | delta (mm) | scale_delta | 关键 violation |
|---|---:|---:|---:|---:|---:|---|
| meal_seq131/WoodenBowl f2692-2722 | failed_before_export: no_active_manipulating_hand_at_frozen_shared_metric_scale | 86.84 | — | — | — | sequence_failed_before_export |
| work_seq107/BookDeepLearning f1579-1609 | export_qc_failed: shared_metric_hand_depth_verified | 24.38 | 28.26 | +3.88 | +0.0901 | sequence_export_qc_failed; metric_scale_rewritten_+0.0901 |

关键结论：

- `meal_seq131/WoodenBowl` 在严格 metric hand depth 下没有 active hand，sequence 未产出 aligned。
- `work_seq107/BookDeepLearning` 产出了 aligned trajectory，但最终 export QC 因为手部 metric depth 未独立验证而失败；同时 sequence 仍把 object scale 从 0.1868 m 改到 0.2036 m（+9.01%），违反 stereo 阶段 `validate_only` 不应改写 metric scale 的协议。
- 对该样本，sequence 之后的 centroid translation error 从 24.38 mm 升到 28.26 mm，ADD-S 从 16.31 mm 升到 16.73 mm，旋转误差基本不变。也就是说，当前 sequence 的 silhouette scale rewrite 没有改善 object GT 误差，反而使 translation 变差。

这比上一版“全卡在 stereo_hand_gate_rejected”更进一步：现在能看到 sequence 在严格双目协议下真正做了什么，而不是没进优化。

## 2.5 P5 G1：GT calibration rescue（full-upstream stereo branch）

因为 RGB object branch 的 calibration 已经使用 GT camera trajectory，G1 若放在该分支会退化为恒等控制；因此按方案语义，把 G1 放到 full-upstream stereo branch 上，只替换 `calibration/T_world_camera.npy` 为 ADT `aria_trajectory.csv` 的 rectified-left 相机轨迹，再跑 `validate_only` sequence。

产物：

- `runs/adt_calibration_rescue_summary.json`
- `runs/calibration_rescue_matrix.csv`

| Run | Sequence 状态 | raw centroid median (mm) | aligned centroid median (mm) | delta (mm) | scale_delta |
|---|---:|---:|---:|---:|---:|
| meal_seq131/WoodenBowl | failed_before_sequence_artifacts | 86.84 | — | — | — |
| work_seq107/BookDeepLearning | export_qc_failed | 24.38 | 24.42 | +0.05 | +0.0901 |

结论：GT camera trajectory 替换后，`work_seq107` 的 aligned centroid 只比 raw 差 0.05 mm，基本不变；scale 仍被 silhouette 改 +9.01%。这说明当前 object sequence 的主要问题不是 camera trajectory 标定，而是 sequence 自身的 metric scale rewrite 与 hand depth export QC。

## 3. P7 Stereo Hand Auxiliary 结论

当前 10 条 pilot 中只有 5 条已有 stereo hand artifact，全部 `accepted_any_hand=false`。  
已新增 `hand_stereo_failure_analysis.json` 和 `.npz`，保留 per-joint 的失败原因，不改任何固定阈值。

| Run | 表现较好侧 | joint_valid_rate | required landmarks frame valid rate |
|---|---:|---:|---:|
| work_seq107 | right | 0.695 | 0.067 |
| work_seq107 | left | 0.376 | 0.0 |
| seq138 | left | 0.141 | 0.0 |
| clean_seq140 | left | 0.046 | 0.0 |
| BlackCeramicMug | left | 0.017 | 0.0 |

失败拆解后的主因：

1. **上游 WiLoR 输入缺失/无效是首要原因**。meal、BlackCeramicMug、clean_seq140 等样本在某一侧 view 的 `valid` 几乎全为 0，导致 `input_invalid` 覆盖 27–30 帧，joint_valid 自然极低。
2. **hand identity 基本正确**。对 `work_seq107` 的左右 `side` 顺序一致，左右视图 hand 0/1 没有明显交换；P7 早期怀疑的 hand identity 不是主要问题。
3. **指尖 joint 的跨视角 vertical/reprojection 不一致是第二原因**。`work_seq107` 右手的 wrist 每帧有效，但指尖 joints 被 `vertical_above_max_px` 大量过滤（DIP/tip 关节最高 27/30 帧），左手也有类似现象。
4. 因此 P7 不是“双目几何崩了”，而是 **WiLoR 在 ADT 灰度 SLAM pair 上域外退化 + 左右视图独立推理不约束 epipolar/一致性**。这与方案 Protocol H 的定位一致：ADT 只适合做 hand 三角化数学/一致性的 stress test，最终 hand benchmark 应由 HOT3D 补齐。

当前不修改 `MIN_REQUIRED_JOINT_RATE` / `MIN_REQUIRED_FRAME_RATE`，也不放宽 vertical/reprojection 门禁。

## 4. 当前建议的推进顺序

按方案逻辑，下一步应是：

1. P5 的 `G1 = GT calibration rescue` 已补入统一 ranking；GT replacement 已扩到 4 条有效 object base run，并记录失败 variant。下一步应优先修 object-region depth/scale 一致性，而不是继续只加样本。
2. 对 sequence 的 `metric_scale_rewritten_+9.01%` 做根因处理：在 `contact_similarity_mode=validate_only` 且 `strict_shared_metric=true` 时，确认 silhouette scale rewrite 是否应该被冻结；如果冻结，这属于 sequence adapter 的 protocol bug，而不是 ADT 数据问题。
3. 对 hand 的最终处理，按方案转 HOT3D，或引入真正的跨视角 MANO/WiLoR joint refinement；不再只调 ADT gate 阈值。
4. P6 目前只有 2 条，且一条失败、一条 export QC 失败，不能下最终 sequence 结论；需要先扩大 P6 候选样本集（完整 ADT samples）。

## 5. 尚未补齐的项

- `G1 = GT calibration rescue`：已在 stereo full-upstream branch 上实现，并纳入 `bottleneck_ranking.csv`；RGB object branch 本身已是 GT calibration，因此不重复做恒等控制。
- P6 aligned 评估：目前 1/2 条产出 aligned，但仍无严格通过 export QC 的样本；raw/aligned delta 和 scale rewrite 表尚未具备统计意义。
- 完整 FP replacement matrix：当前 rescue 已覆盖 `calibration_gt/depth_gt/mask_gt/mesh_gt`，但尚未做全部 FP0–FP7 组合；FoundationPose oracle 仍待补。
- 真正的双目手部一致性修复：尚未实现；需要 HOT3D 或跨视角 joint refinement。
