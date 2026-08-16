# ADT 上游改进与消融实验总结

生成时间：2026-08-15

对应文档：《EgoEngine_下一阶段上游改进与消融实验计划.md》。

本报告汇总 Phase A（Hand）、Phase B（Object Reconstruction）、Phase C（Object-ROI Depth Refinement）的 paired 消融结果。所有实验均遵守以下规则：

- 同一批 ADT 样本、同一时间窗口；
- 一次只替换一个模块；
- GT 只用于 oracle replacement / 误差上界分析，不进入最终推理链；
- 同时观察“模块自身指标”和“下游 FoundationPose / gate 指标”。

当前工作目录：`/data_all/zzx/egoengine/video_to_spider`。

---

## 1. 样本清单与掩码可用性

现有 10 条 ADT benchmark 中，已生成 8 条 `*_rgbobject` 分支。SAM3 对象掩码可用性如下：

| 样本 | SAM3 object mask | O0/SAM3D | O1 可运行 |
|---|---:|---|---|
| WhiteLiddedTrashBin | 30/30 | 3 个合格 proposal | 是 |
| StepStool | 0/30 | 无 ranking | 否（mask 缺失） |
| BookDeepLearning seq138 | 30/30 | 3 个合格 proposal | 是 |
| WoodenSpoon | 30/30 | 3 个合格 proposal | 是 |
| BlackCeramicMug | 0/30 | 无 ranking | 否（mask 缺失） |
| DinoToy | 30/30 | 0 个合格 proposal | 是 |
| Flask | 0/30 | 无 ranking | 否（mask 缺失） |
| WoodenBowl seq032 | 30/30 | 3 个合格 proposal | 是 |

结论：O1 无法自动运行在 3 条 mask 全空的样本（StepStool / BlackCeramicMug / Flask）。这是 SAM3 对象掩码上游失败，不是 O1 几何重建自身失败。

---

## 2. Phase A：Stereo Hand

详细结果见 `docs/ADT_H1_H2_HAND_ABLATION.md`。摘要：

| 实验 | 修改内容 | 结论 |
|---|---|---|
| H0 | WiLoR + 当前 stereo triangulation | 5 条样本 accepted 0/5 |
| H1 | HaMeR + 原 stereo fusion | 无整体改善，说明主因不是 WiLoR 单目网络 |
| H2 | HaMeR + calibrated stereo BA | 仅 `work_seq107` 右手从 rejected → accepted（joint_valid_rate 0.916） |

Phase A 的主要阻塞是 ADT 灰度 wide-baseline SLAM pair 的左右手检测缺失，H2 的 BA 无法修复检测缺失。

Phase A `P7 accepted 0/5 -> >=3/5` 目标未达到。

---

## 3. Phase B：Object Reconstruction

### 3.1 O0 — SAM3D baseline

| 样本 | SAM3D qualified | FoundationPose 最终 gate |
|---|---:|---|
| DinoToy | 0 | 失败（无 mesh） |
| WoodenBowl seq032 | 3 | 失败（rel depth 0.206 > 0.20） |
| BookDeepLearning seq138 | 3 | 通过（tracking_score 0.858） |
| WhiteLiddedTrashBin | 3 | 失败（所有 proposal 未过安全阈值） |
| WoodenSpoon | 3 | 失败（register valid too small） |

### 3.2 O1 — masked multi-frame metric fusion（新增）

实现：`video_to_spider/adapters/object_mvg.py`。使用 SAM3 mask、calibrated metric depth 和 per-frame `T_world_camera`，将各帧 object-visible 前景点变换到参考相机帧融合，并生成 alpha-shape mesh。`selected_scale_m=1.0`，不重写 metric scale。

| 样本 | O1 qualified | O1 FoundationPose 结果 |
|---|---:|---|
| DinoToy | 1 | **通过**（valid 1.0, IoU 0.388, rel 0.100, tracking 0.729） |
| WoodenBowl seq032 | 1 | 失败（IoU 0.852 但 rel 0.255、trans 0.125） |
| BookDeepLearning seq138 | 1 | 失败（valid_rate 0.0） |
| WhiteLiddedTrashBin | 1 | 失败（IoU 0.096、trans 1.804、rot 1.816） |
| WoodenSpoon | 0 | 失败（object ROI 无可融合深度点） |

### 3.3 Phase B 结论

- O1 只在 DinoToy 上把 O0 的“无 mesh”提升为 FoundationPose gate 通过。
- O1 没有普遍优于 SAM3D：在 BookDeepLearning / WoodenBowl 上，SAM3D 的下游表现更好。
- 因此不能把 O1 作为通用替代；它更适合作为 SAM3D 失败时的几何 fallback，而不是默认主路线。
- 未达到方案中“SAM3D 失败样本至少恢复 2 条”的目标。

---

## 4. Phase C：Object-ROI Depth Refinement

实现：`scripts/run_adt_object_depth_refine.py`。

D1 = mask 腐蚀 + TELEA invalid fill + bilateral；D2 = D1 + 5 帧时域中值。下表为 object ROI 内相对 ADT RGB GT depth 的指标，越低越好（delta1 越高越好）。

| 样本 | D0 AbsRel | D1 AbsRel | D2 AbsRel | D0 RMSE | D1 RMSE | D2 RMSE | D0 δ1 | D1 δ1 | D2 δ1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DinoToy | 0.829 | 1.042 | 0.772 | 1.553 | 1.389 | 1.402 | 0.771 | 0.538 | 0.518 |
| WoodenBowl seq032 | 0.148 | 0.833 | 0.504 | 0.478 | 0.663 | 0.619 | 0.929 | 0.527 | 0.571 |
| BookDeepLearning seq138 | 0.258 | 0.964 | 0.625 | 0.484 | 0.662 | 0.489 | 0.950 | 0.522 | 0.563 |
| WhiteLiddedTrashBin | 0.091 | 1.075 | 0.549 | 0.403 | 0.634 | 0.430 | 0.988 | 0.558 | 0.658 |
| WoodenSpoon | 0.106 | 0.149 | 0.217 | 0.180 | 0.350 | 0.524 | 0.998 | 0.915 | 0.797 |

### Phase C 结论

- D1/D2 确实降低了 object ROI invalid rate，但以明显牺牲 AbsRel / RMSE / δ1 为代价。
- 当前 FoundationStereo 投影 depth 在有效像素上已比较准确，主要问题是“稀疏”。单纯用 inpainting / 时域中值填补大空洞会引入大误差，不应直接作为默认模块。
- 因此 D1/D2 的 naive 版本不被采纳；若继续做 depth refinement，应先做小洞填充 + confidence weighting，而不是对大片 invalid 区域做无监督外推。

---

## 5. Phase D / E 状态

- Phase D：FoundationPose 保持为固定下游评测器，没有替换。符合方案要求。
- Phase E：strict sequence `validate_only` 尚未进行完整 10 条回归，因为上游 Hand/Object 的通过率仍不足；按方案，不应让 sequence 去“修回”上游误差。

---

## 6. 总体结论

1. Hand：H2 是对的方向，但受 ADT 手检测缺失限制，需要在真实双目 RGB ego-hand 数据（HOT3D）上再验证。
2. Object Reconstruction：O1 是有效的几何 fallback，但只能救回 DinoToy 这一类，不能替代 SAM3D；建议 SAM3D 与 O1 组成候选池，由 FoundationPose gate 选择。
3. Object Depth：D1/D2 naive invalid fill 有负面影响；需要更保守的填充策略。
4. 当前最上游的硬阻塞是“SAM3 mask 在 StepStool / BlackCeramicMug / Flask 全空”，导致 O1 和 D1/D2 无法在这三条上自动运行。

---

## 7. 推荐下一步

- 优先修复 SAM3 object mask：对 StepStool / BlackCeramicMug / Flask 调整 text prompt、anchor、多实例/近手 ranking 或增加点 prompt 融合；但不要因单一样本过拟合。
- 在 HOT3D 双目 RGB 上验证 H2 hand BA 的真实接受率。
- 若继续 depth refinement：实现“仅填充小空洞 + boundary erosion + 置信度加权”，而不是对大片 invalid 做无监督 fill。
- 对 O1 做更稳的 mesh 后处理：当前 alpha-shape 常产生多个连通分量；可先做 largest-component 选择 / 粗 TSDF / 生成式 shape completion（O2）。

---

## 8. 新增文件

- `video_to_spider/adapters/object_mvg.py`：O1 masked multi-frame metric fusion。
- `scripts/run_adt_object_depth_refine.py`：D1/D2 object-ROI depth refinement。
- `scripts/summarize_adt_phase_bc_ablation.py`：Phase B/C 指标聚合。
- `runs/adt_phase_bc_ablation_summary.json`：Phase B/C 聚合结果。
- `docs/ADT_上游改进与消融实验总结.md`：本报告。
