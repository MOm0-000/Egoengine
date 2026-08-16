# ADT 10 样本全流程消融实验结果汇总

生成时间：2026-08-15

对应方案：《双目 EgoEngine 下一阶段详细实验方案 v3 紧凑版》

固定样本：`runs/adt_depth_benchmark_summary.json` 中 10 条 ADT 样本，每样本 30 帧窗口。

原则：

- 一次只替换一个模块；
- GT 只用于评测或 oracle 诊断，不进入最终推理；
- 不重写 metric scale，不在 sequence 阶段“修复”上游误差；
- 所有 mask 提升先落到独立 clone run，不覆盖原 S0/S1 artifact。

---

## 1. 总体结论

当前已把可执行的上游消融推到 10 条固定 ADT 样本：

- **Phase 0 Object Perception 已达标**：最终 mask coverage `298/300 = 99.33%`。
- **Phase 1 Stereo Hand 未达标**：H0/H1 均为 `0/5 accepted_any_hand`，H2 为 `1/5`；H3 UmeTrack 与 H4 POEM-v2 已执行，但受 ADT 双目灰度/wide-baseline 与 HaMeR 输入缺失限制，仍为 `0/5 accepted_any_hand`；H5 Best-proposal + stereo MANO/depth/bone BA 已执行，仍为 `1/5`。
- **Phase 2 Object Reconstruction 已补做 O2/O3/O4/O5**：O1 在 `DinoToy` 之外又救回 `BlackCeramicMug`；O2 BundleSDF 与 O3 FoundationPose model-free 均达到模块级 `9/10` 成功，仅 `WoodenSpoon` 因 ADT 物体区域无有效 metric depth 失败；O4 VGGT/MASt3R/MUSt3R 已全部跑完并输出几何先验；O5 已汇总 O0/O1/O2/O3 候选并选择 best candidate。
- **Phase 3 Depth Refinement 否定 naive fill**：D1/D2-naive 降低 invalid rate，但明显恶化 AbsRel/RMSE/δ1；新增 D2-object-centric 使用 BundleSDF object pose 做 canonical fusion，指标与 D0 基本一致但未整体优于 D0；D3 已执行 DEFOM-Stereo 与 Stereo Anywhere，二者均不能整体替换 FoundationStereo。
- **Phase 4 strict Sequence 未通过**：当前 2 条候选均未通过 export QC，但 `work_seq107/BookDeepLearning` 的 `metric_scale_rewritten` 已修复为 `scale_delta≈0`，剩余失败仅为 `shared_metric_hand_depth_verified`。

因此，当前最前端 mask 阻塞已解除；Object Reconstruction 已有 O2/O3 两个 9/10 的 whole-block 候选并完成 hybrid gate；完整 Digital Twin 上游仍被 Hand 卡住。

---

## 2. Phase 0：Object Perception / SAM3 泛化

结果文件：

- `runs/adt_phase0_final_10_summary.json`
- `runs/adt_phase0_final_10_matrix.csv`
- `runs/adt_phase0_sam3_s2_multianchor_summary.json`

| Sample | Window | S0 valid | S0 IoU | S1 valid | S1 IoU | S2 valid | S2 IoU | Final branch | Final valid | Final IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| WhiteLiddedTrashBin | 1216-1246 | 1.000 | 0.761 | 1.000 | 0.761 | - | - | S1 | 1.000 | 0.761 |
| WoodenBowl / meal_seq131 | 2692-2722 | 1.000 | 0.888 | 1.000 | 0.888 | - | - | S1 | 1.000 | 0.888 |
| BookDeepLearning / work_seq107 | 1579-1609 | 1.000 | 0.875 | 1.000 | 0.875 | - | - | S1 | 1.000 | 0.875 |
| BookDeepLearning / recognition_seq138 | 2177-2207 | 1.000 | 0.926 | 1.000 | 0.925 | - | - | S1 | 1.000 | 0.925 |
| BlackCeramicMug | 1703-1733 | 0.000 | 0.000 | 0.000 | 0.000 | 0.967 | 0.657 | S2 multi-anchor | 0.967 | 0.657 |
| WoodenSpoon | 2152-2182 | 1.000 | 0.603 | 1.000 | 0.603 | - | - | S1 | 1.000 | 0.603 |
| WoodenBowl / Lite_seq032 | 1771-1801 | 1.000 | 0.904 | 1.000 | 0.904 | - | - | S1 | 1.000 | 0.904 |
| DinoToy | 1547-1577 | 1.000 | 0.653 | 1.000 | 0.654 | - | - | S1 | 1.000 | 0.654 |
| Flask | 1537-1567 | 0.000 | 0.000 | 0.000 | 0.000 | 0.967 | 0.364 | S2 multi-anchor | 0.967 | 0.364 |
| StepStool | 611-641 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.940 | S2 multi-anchor | 1.000 | 0.940 |

聚合：

- `macro_valid_rate = 0.9933`
- `frames_with_mask = 298/300`
- `macro_mean_iou = 0.7571`

结论：

- S1 prompt ensemble 对原有成功样本基本不退化。
- Grounding DINO + SAM3 box prompt 的 multi-anchor fallback 对 `BlackCeramicMug / Flask / StepStool` 三类文本-only 失败样本有效。
- 方案成功标准 `>=90% coverage` 与 `3 条失败样本中至少 2/3 恢复到 >=27/30` 已满足。

### 2.1 S3 stereo-aware selection 已执行

结果文件：`runs/adt_phase0_s3_summary.json`

- 对已有 S1/S2 候选计算 depth support、positive disparity、hand proximity、temporal centroid jump。
- S1 有效的 7 条均继续选择 S1；`BlackCeramicMug / Flask / StepStool` 选择 S2 multi-anchor，与最终 Phase 0 选择一致。
- S3 未改变最终 mask 决策，但补上了方案要求的 stereo-aware gate 记录。

---

## 3. Phase 1：Stereo Hand Reconstruction

结果文件：

- `runs/adt_stereo_hand_benchmark_summary.json`（H0）
- `runs/adt_hamer_hand_benchmark_summary.json`（H1）
- `runs/adt_hamer_hand_ba_benchmark_summary.json`（H2）

### 3.1 H0/H1/H2 已执行

| 实验 | 修改 | 可评测样本 | accepted_any_hand | 结论 |
|---|---:|---:|---|
| H0 | WiLoR + 当前 stereo triangulation | 5 | 0/5 | 未通过 |
| H1 | HaMeR + 原 stereo fusion | 5 | 0/5 | 单换 backbone 无整体收益 |
| H2 | HaMeR + calibrated stereo BA | 5 | 1/5 | 仅 `work_seq107` 右手 accepted |

H2 关键提升：

- `work_seq107/BookDeepLearning` 右手：
  - `joint_valid_rate = 0.916`
  - `required_landmarks_frame_valid_rate = 0.633`
  - `accepted = true`

H2 未解决问题：

- `meal_seq131/WoodenBowl`、`BlackCeramicMug` 等样本左右 view 检测缺失；
- stereo BA 无法恢复不存在的 2D 观测。

### 3.2 H3/H4/H5 状态

| 实验 | 状态 | 原因 |
|---|---|---|
| H3 UmeTrack | 已执行 | 对可评测 HaMeR 输入跑完；`accepted_any_hand=false`，立体 reprojection gate 未通过 |
| H4 POEM / POEM-v2 | 已执行 | 对可评测 HaMeR 输入跑完；`accepted_any_hand=false` |
| H5 Best proposal + Stereo MANO BA | 已执行 | 以最佳可用 proposal（HaMeR）加上 MANO/depth/bone 残差重跑；`accepted_any_hand=true` 仅 `work_seq107` |

结果文件：

- `runs/adt_umetrack_hand_benchmark_summary.json`
- `runs/adt_poem_hand_benchmark_summary.json`
- `runs/adt_hamer_hand_h5_benchmark_summary.json`

说明：

- H3/H4 均未改变 ADT hand 结论；现有 5 条可评测样本的 stereo 2D observation 缺失仍是上游阻塞。
- 固定 10 条样本中 `WoodenSpoon / WoodenBowl-Lite / DinoToy / Flask / StepStool` 缺少 HaMeR artifact，H3/H4 只能对已有 HaMeR 的 5 条运行。
- H5 与 H2 结论一致：只在已有右目观测的 `work_seq107` 上 accepted，无法弥补其他样本缺失的 2D 观测。

---

## 4. Phase 2：Unknown-object Reconstruction + Tracking

结果文件：

- `runs/adt_phase_bc_ablation_summary.json`
- 最终 mask clone：`runs/*_phase_final/mesh_proposals/omvg_mesh_ranking.json`
- 新补 SAM3D：`runs/*_phase_final/mesh_proposals/mesh_ranking.json`
- 新补 FP：`runs/*_phase_final/object_tracking*/`

### 4.1 O0/O1 下游概览

| Sample | Final mask valid | O0 SAM3D | O0 FP accepted | O1 qualified | O1 static IoU | O1 static rel depth | O1 FP accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| WhiteLiddedTrashBin | 1.000 | 3 | false | 1 | 0.163 | 0.575 | false |
| WoodenBowl / meal_seq131 | 1.000 | 3 | true | 1 | 0.264 | 0.258 | 未单独跑 FP |
| BookDeepLearning / work_seq107 | 1.000 | 3 | true | 1 | 0.185 | 0.086 | 未单独跑 FP |
| BookDeepLearning / recognition_seq138 | 1.000 | 3 | true | 1 | 0.734 | 0.129 | false |
| BlackCeramicMug | 0.967 | 3 | false | 1 | 0.518 | 0.075 | **true** |
| WoodenSpoon | 1.000 | 3 | false | 0 | - | - | false |
| WoodenBowl / Lite_seq032 | 1.000 | 3 | false | 1 | 0.820 | 0.209 | false |
| DinoToy | 1.000 | 0 | - | 1 | 0.451 | 0.084 | **true** |
| Flask | 0.967 | 0 | false | 1 | 0.327 | 0.188 | false |
| StepStool | 1.000 | 3 | false | 1 | 0.621 | 1.062 | false |

说明：

- `O0 SAM3D` 列为 qualified proposal 数；`O0 FP accepted` 表示该分支最终 FoundationPose gate。
- `O1 qualified` 列为 masked multi-frame metric fusion 是否生成合格 mesh。
- 新补的 `BlackCeramicMug` 在 S2 mask + O1 后，FoundationPose gate 首次变为 `accepted=true`。

### 4.2 O2-O5 状态

| 实验 | 状态 | 说明 |
|---|---|---|
| O2 BundleSDF | 已执行 | C++ `my_cpp` 已修复并构建；10 条中 9 条成功输出 `ob_in_cam/` 与 mesh，`WoodenSpoon` 因 ADT 物体区域无有效 metric depth 失败 |
| O3 FoundationPose model-free | 已执行 | 8 个 reference views + VGGT 相机先验训练 Neural Object Field；10 条中 9 条输出 metric mesh，`WoodenSpoon` 同样因无有效物体深度失败 |
| O4 VGGT / VGGT-Ω / MASt3R / MUSt3R | 已执行 | VGGT-1B、MASt3R、MUSt3R 均跑完 10 条，输出 depth/pointmap/camera 先验；VGGT-Ω 沿用 VGGT-1B checkpoint |
| O5 Hybrid Candidate Pool | 已执行 | 汇总 O0/O1/O2/O3 候选并做 static mask/depth fit；结果文件 `runs/adt_o5_hybrid_summary.json` |

O2 结果：`runs/adt_bundlesdf/adt_bundlesdf_summary.json`

- 成功 9/10：除 `WoodenSpoon` 外均 `returncode=0`、`pose_count=30`、`mesh_count=2`。
- `WoodenSpoon` 失败原因：Phase 0 mask 在 30 帧窗口内与 FoundationStereo valid depth 无交集，BundleSDF 首帧无法初始化 object point cloud。

O3 结果：`runs/adt_foundationpose_modelfree/adt_foundationpose_modelfree_summary.json`

- 成功 9/10，输出 `runs/adt_foundationpose_modelfree/<run>/model_free_mesh.obj`。
- `WoodenSpoon` 失败原因同 O2。

O4 结果：

- VGGT：`runs/adt_vggt/adt_vggt_summary.json`，10/10 成功。
- MASt3R：`runs/adt_mast3r/adt_mast3r_summary.json`，10/10 成功。
- MUSt3R：`runs/adt_must3r/adt_must3r_summary.json`，10/10 成功。
- 这些输出仅作为 multi-view correspondence / point tracks / camera-object motion 初始化与几何 prior，最终 metric scale 仍锚定 calibrated stereo + FoundationStereo。

O5 结果：`runs/adt_o5_hybrid_summary.json`

- 对每个样本汇总 `o0_o1_fp / o2_bundlesdf / o3_model_free` 候选。
- `WoodenSpoon` 无可用候选；其余 9 条至少有一个 whole-block 候选。
- 统一 gate 当前仍以 O0/O1 FoundationPose accepted 为最强候选；O3 model-free 在 `WhiteLiddedTrashBin / BlackCeramicMug / Flask / StepStool` 等样本提供可评测 fallback。

---

## 5. Phase 3：Stereo Depth Coverage Refinement

结果文件：

- `runs/adt_depth_benchmark_summary.json`（D0 基础深度）
- `runs/*_phase_final/depth_roi_refined/depth_metrics.json`（D1/D2）

| Sample | D0 AbsRel | D0 RMSE | D0 δ1 | D0 invalid | D1 AbsRel | D1 RMSE | D1 δ1 | D1 invalid | D2 AbsRel | D2 RMSE | D2 δ1 | D2 invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WhiteLiddedTrashBin | 0.091 | 0.403 | 0.988 | 0.657 | 1.075 | 0.634 | 0.558 | 0.337 | 0.549 | 0.430 | 0.658 | 0.231 |
| StepStool | 0.062 | 0.189 | 0.978 | 0.585 | 0.889 | 0.546 | 0.599 | 0.260 | 0.499 | 0.373 | 0.659 | 0.169 |
| BookDeepLearning / seq138 | 0.254 | 0.479 | 0.952 | 0.682 | 0.963 | 0.661 | 0.523 | 0.356 | 0.627 | 0.487 | 0.561 | 0.271 |
| WoodenSpoon | 0.106 | 0.180 | 0.998 | 0.620 | 0.149 | 0.350 | 0.915 | 0.566 | 0.217 | 0.524 | 0.797 | 0.478 |
| BlackCeramicMug | 0.248 | 0.567 | 0.925 | 0.616 | 0.683 | 0.752 | 0.564 | 0.266 | 0.484 | 0.751 | 0.578 | 0.149 |
| DinoToy | 0.831 | 1.554 | 0.771 | 0.607 | 1.040 | 1.387 | 0.536 | 0.389 | 0.772 | 1.402 | 0.514 | 0.224 |
| Flask | 0.271 | 0.609 | 0.938 | 0.807 | 0.690 | 1.221 | 0.356 | 0.366 | 0.571 | 1.248 | 0.374 | 0.296 |
| WoodenBowl / Lite_seq032 | 0.148 | 0.478 | 0.929 | 0.646 | 0.833 | 0.663 | 0.527 | 0.300 | 0.504 | 0.619 | 0.571 | 0.172 |
| WoodenBowl / meal_seq131 | 0.044 | 0.045 | 0.977 | 0.570 | 0.723 | 0.505 | 0.612 | 0.248 | 0.442 | 0.324 | 0.645 | 0.149 |
| BookDeepLearning / work_seq107 | 0.083 | 0.202 | 0.972 | 0.645 | 0.754 | 0.576 | 0.557 | 0.313 | 0.470 | 0.389 | 0.600 | 0.200 |

结论：

- D1/D2 明显降低 object ROI invalid rate；
- 但 D1/D2 在大多数样本上恶化了 AbsRel/RMSE/δ1；
- 当前不采纳 naive TELEA/temporal median 作为默认深度模块。

### 5.0 D2 object-centric temporal fusion 已执行

结果文件：`runs/adt_d2_objectcentric_summary.json`

该方法使用 O2 BundleSDF 的 `ob_in_cam` 轨迹，将 object-ROI depth 反投影到 canonical object frame，做 5 mm voxel 融合后再投影回每帧，作为方案中的 object-centric temporal fusion 候选。

| Sample | D0 AbsRel | D0 δ1 | D2-OC AbsRel | D2-OC RMSE | D2-OC δ1 |
|---|---:|---:|---:|---:|---:|
| WhiteLiddedTrashBin | 0.091 | 0.988 | 0.091 | 0.403 | 0.988 |
| WoodenBowl / meal_seq131 | 0.044 | 0.977 | 0.044 | 0.045 | 0.977 |
| BookDeepLearning / work_seq107 | 0.083 | 0.972 | 0.083 | 0.202 | 0.972 |
| BookDeepLearning / seq138 | 0.254 | 0.952 | 0.254 | 0.479 | 0.952 |
| BlackCeramicMug | 0.248 | 0.925 | 0.248 | 0.567 | 0.925 |
| WoodenSpoon | 0.106 | 0.998 | - | - | - |
| WoodenBowl / Lite_seq032 | 0.148 | 0.929 | 0.148 | 0.478 | 0.929 |
| DinoToy | 0.831 | 0.771 | 0.831 | 1.554 | 0.771 |
| Flask | 0.271 | 0.938 | 0.271 | 0.609 | 0.938 |
| StepStool | 0.062 | 0.978 | 0.062 | 0.189 | 0.978 |

结论：D2 object-centric 与 D0 指标基本一致，没有因为 temporal fusion 带来明显提升；`WoodenSpoon` 因 O2 无有效 pose 无法执行。默认仍保留 D0。

### 5.1 D3 状态

已执行 DEFOM-Stereo backbone 对比，结果文件：`runs/adt_defom_stereo_summary.json`。

| Sample | D0 obj AbsRel | D0 obj δ1 | D3 obj AbsRel | D3 obj RMSE | D3 obj δ1 | D3 full AbsRel | D3 full δ1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| WhiteLiddedTrashBin | 0.102 | 0.876 | 2.248 | 485.791 | 0.012 | 0.462 | 0.353 |
| WoodenBowl / meal_seq131 | 0.070 | 0.954 | - | - | - | 0.636 | 0.338 |
| BookDeepLearning / work_seq107 | 0.088 | 0.978 | 0.494 | 2.161 | 0.119 | 0.419 | 0.391 |
| BookDeepLearning / seq138 | 0.103 | 0.983 | 0.364 | 1.093 | 0.113 | 0.433 | 0.295 |
| BlackCeramicMug | 0.118 | 0.970 | 0.064 | 0.477 | 0.940 | 0.227 | 0.664 |
| WoodenSpoon | 0.107 | 1.000 | 0.328 | 1.074 | 0.263 | 0.384 | 0.373 |
| WoodenBowl / Lite_seq032 | 0.179 | 0.958 | 0.110 | 0.536 | 0.969 | 0.169 | 0.771 |
| DinoToy | 0.798 | 0.885 | 0.125 | 0.674 | 0.944 | 0.186 | 0.655 |
| Flask | 0.227 | 0.958 | 0.198 | 0.726 | 0.671 | 0.232 | 0.644 |
| StepStool | 0.038 | 0.993 | 0.539 | 2.034 | 0.132 | 0.368 | 0.505 |

结论：

- DEFOM-Stereo 在 `DinoToy / BlackCeramicMug / WoodenBowl-Lite` 上 object ROI 优于 FoundationStereo D0；
- 但在 `WhiteLiddedTrashBin / StepStool / Book-work` 上出现明显 outlier/scale 失败，不能整体替换 FoundationStereo；
- Stereo Anywhere 已接入并跑完 10 条：coverage 较高，但 full/object depth error 明显差于 FoundationStereo，不能整体替换；
- D3 结论维持：保留 FoundationStereo 作为 metric depth 主线。

Stereo Anywhere 结果：`runs/adt_stereoanywhere_summary.json`

| Sample | D0 obj AbsRel | D0 obj δ1 | SA obj AbsRel | SA obj δ1 | SA full AbsRel | SA full δ1 |
|---|---:|---:|---:|---:|---:|---:|
| WhiteLiddedTrashBin | 0.102 | 0.876 | 0.748 | 0.236 | 0.466 | 0.288 |
| WoodenBowl / meal_seq131 | 0.070 | 0.954 | - | - | 0.920 | 0.376 |
| BookDeepLearning / work_seq107 | 0.088 | 0.978 | 0.998 | 0.173 | 0.443 | 0.364 |
| BookDeepLearning / seq138 | 0.103 | 0.983 | 0.581 | 0.148 | 0.511 | 0.228 |
| BlackCeramicMug | 0.118 | 0.970 | 0.129 | 0.876 | 0.197 | 0.704 |
| WoodenSpoon | 0.107 | 1.000 | 0.377 | 0.343 | 0.321 | 0.466 |
| WoodenBowl / Lite_seq032 | 0.179 | 0.958 | 0.218 | 0.742 | 0.319 | 0.531 |
| DinoToy | 0.798 | 0.885 | 0.168 | 0.950 | 0.143 | 0.826 |
| Flask | 0.227 | 0.958 | 0.113 | 0.929 | 0.199 | 0.700 |
| StepStool | 0.038 | 0.993 | 0.603 | 0.127 | 0.311 | 0.537 |

---

## 6. Phase 4：Strict Sequence → MINK-ready

结果文件：

- `runs/adt_sequence_benchmark_summary.json`（已用修复后代码重跑）
- `runs/sequence_benchmark_matrix.csv`
- `video_to_spider/optimization/sequence.py`（strict scale freeze 修复）

当前候选：

| Sample | Sequence status | Raw centroid median | Aligned centroid median | Scale delta | Key violation |
|---|---:|---:|---:|---:|---|
| WoodenBowl / meal_seq131 | failed_before_export | 86.84 mm | - | - | no active manipulating hand |
| BookDeepLearning / work_seq107 | export_qc_failed | 24.38 mm | 28.24 mm | ~0.0 | shared_metric_hand_depth_verified |

结论：

- 当前 0/2 条严格通过 sequence export QC；
- `work_seq107` 的 object scale 已在 `validate_only` 下冻结，`scale_delta≈0`；当前失败仅剩 `shared_metric_hand_depth_verified`；
- 在 Hand/Object 上游达标前，不应继续放宽 sequence gate。

### 6.1 全 10 条 strict Sequence 回归已执行

结果文件：`runs/adt_sequence_benchmark_all_summary.json`

对 10 条固定 ADT 样本全部尝试 strict `validate_only` 入口，未放宽任何 gate：

- `work_seq107` 达到 `export_qc_failed`，仅缺 `shared_metric_hand_depth_verified`。
- `meal_seq131` 在 strict shared-metric 下仍为 `no_active_manipulating_hand`。
- 其余样本主要失败在 `depth_gate_rejected: object_valid_coverage` 或缺少 `object_tracking/foundationpose_raw.npz`。
- 严格通过数仍为 `0/10`，没有 metric scale rewrite。

---

## 7. 方案成功标准检查

| 指标 | 目标 | 当前结果 | 状态 |
|---|---:|---:|---|
| Object mask coverage | >=90% | 99.33% | 通过 |
| 原成功样本不明显退化 | 无退化 | 7/7 保持 `valid_rate=1.0` | 通过 |
| Stereo hand accepted | >=3/5 可评测样本 | H2/H5 均为 1/5；H3/H4 为 0/5 | 未通过 |
| 完整 object run | >=7/10 | O2 BundleSDF 9/10；O3 model-free 9/10；FP gate 5/10 | 模块级通过，统一 gate 仍待接 |
| Sequence export QC | >=2 条严格通过 | 全 10 条尝试中 0 条严格通过 | 未通过 |
| object metric scale rewrite | 0 | 0 条；scale_delta≈0 | 通过 |
| MINK-ready upstream | >=1 条严格通过 | 0 | 未通过 |

---

## 8. 当前新增/修改产物

- `scripts/prepare_adt_phase_final_runs.py`
- `scripts/summarize_adt_phase0_final.py`（已加入 S2 merged 产物 fallback）
- `scripts/run_adt_defom_stereo.py`
- `scripts/summarize_adt_defom_stereo.py`
- `scripts/run_adt_stereoanywhere.py`
- `scripts/summarize_adt_stereoanywhere.py`
- `scripts/run_adt_umetrack_hand_benchmark.py`
- `scripts/run_adt_poem_hand_benchmark.py`
- `scripts/run_adt_hamer_hand_h5_benchmark.py`
- `scripts/run_adt_bundlesdf.py`
- `scripts/run_adt_foundationpose_modelfree.py`
- `scripts/run_adt_vggt.py`
- `scripts/run_adt_mast3r.py`
- `scripts/run_adt_must3r.py`
- `scripts/run_adt_phase0_s3.py`
- `scripts/run_adt_d2_objectcentric.py`
- `scripts/run_adt_o5_hybrid.py`
- `scripts/run_adt_sequence_benchmark_all.py`
- `third_party/DEFOM-Stereo/`、`third_party/stereoanywhere/`
- `third_party/BundleSDF/`（C++ `my_cpp` 已修复构建）
- `third_party/FoundationPose/bundlesdf/mycuda`（model-free NeRF CUDA ops 已构建）
- `third_party/vggt/`、`third_party/mast3r/`、`third_party/must3r/`（模型权重已下载）
- `runs/adt_defom_stereo_summary.json`
- `runs/adt_stereoanywhere_summary.json`
- `runs/adt_umetrack_hand_benchmark_summary.json`
- `runs/adt_poem_hand_benchmark_summary.json`
- `runs/adt_hamer_hand_h5_benchmark_summary.json`
- `runs/adt_phase0_s3_summary.json`
- `runs/adt_d2_objectcentric_summary.json`
- `runs/adt_o5_hybrid_summary.json`
- `runs/adt_sequence_benchmark_all_summary.json`
- `runs/adt_bundlesdf/adt_bundlesdf_summary.json`
- `runs/adt_foundationpose_modelfree/adt_foundationpose_modelfree_summary.json`
- `runs/adt_vggt/adt_vggt_summary.json`
- `runs/adt_mast3r/adt_mast3r_summary.json`
- `runs/adt_must3r/adt_must3r_summary.json`
- `runs/*_phase_final/`：使用最终 mask 的独立下游 clone
- `runs/*_phase_final/depth_roi_refined/depth_metrics.json`
- `runs/*_phase_final/mesh_proposals/omvg_mesh_ranking.json`
- `runs/*_phase_final/mesh_proposals/mesh_ranking.json`
- `runs/*_phase_final/object_tracking/` 或 `object_tracking_omvg/`

---

## 9. 下一步建议

1. Hand：H3/H4 已排除；H5 不应继续在 ADT 灰度 SLAM pair 上盲跑，优先在 HOT3D 双目 RGB 上验证 H2/Best-proposal + Stereo MANO BA。
2. Object：O2/O3 已验证 unknown-object reconstruction 可到 9/10；下一步统一 O2/O3 mesh 与 FoundationPose gate，重点处理 `WoodenSpoon` 这类无有效 object depth 的失败样本。
3. Depth：保留 FoundationStereo；Stereo Anywhere/DEFOM 均不作为默认 backbone；若继续做 D3，应只比较可安装 backbone，不做大范围 inpainting。
4. Sequence：`metric_scale_rewritten` 已修复并回归验证；下一步继续解决 hand depth verification，不通过 optimizer 吸收上游误差。
5. 数据：ADT 继续作为 stress test，不用于 MINK/MPC 通过门槛；最终 hand/object 达标需加入 HOT3D 或自采双目 RGB。
