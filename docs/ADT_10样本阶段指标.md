# ADT 10 样本阶段指标

> 最新 Hand/Object/Depth 三阶段 paired 消融汇总见 [`ADT_上游改进与消融实验总结.md`](./ADT_上游改进与消融实验总结.md)。本文件保留较早阶段记录。

生成时间：2026-08-15

说明：下表是当前 `video_to_spider` 仓库中 ADT 上游 benchmark 的只读汇总，不修改任何评测结果。除特别注明外，深度指标均为中位数。

主要来源：

- `runs/adt_upstream_audit_summary_current.json`
- `runs/adt_p4_p5_batch_state.json`
- `runs/adt_p5_gt_replacement_summary.json`
- `runs/adt_sequence_benchmark_summary.json`
- `runs/adt_stereo_hand_benchmark_summary.json`

## 0. 样本与数据状态

| # | Sequence | Object | Window | 文件布局 |
|---:|---|---:|---:|---|
| 1 | Apartment_release_clean_seq140_M1292 | WhiteLiddedTrashBin | f1216_1246 | vrs_files / GT / MPS 存在 |
| 2 | Apartment_release_meal_seq131_M1292 | WoodenBowl | f2692_2722 | vrs_files / GT / MPS 存在 |
| 3 | Apartment_release_work_seq107_M1292 | BookDeepLearning | f1579_1609 | vrs_files / GT / MPS 存在 |
| 4 | Apartment_release_recognition_seq138_M1292 | BookDeepLearning | f2177_2207 | vrs_files / GT / MPS 存在 |
| 5 | Lite_release_recognition_BlackCeramicMug_seq030_61283 | BlackCeramicMug | f1703_1733 | vrs_files / GT / MPS 存在 |
| 6 | Apartment_release_recognition_seq140_M1292 | WoodenSpoon | f2152_2182 | vrs_files / GT / MPS 存在 |
| 7 | Lite_release_recognition_WoodenBowl_seq032_61283 | WoodenBowl | f1771_1801 | vrs_files / GT / MPS 存在 |
| 8 | Lite_release_recognition_DinoToy_seq030_61283 | DinoToy | f1547_1577 | vrs_files / GT / MPS 存在 |
| 9 | Lite_release_recognition_Flask_seq030_61283 | Flask | f1537_1567 | vrs_files / GT / MPS 存在 |
| 10 | Apartment_release_decoration_seq131_M1292 | StepStool | f611_641 | vrs_files / GT / MPS 存在 |

说明：目录层面 10 条均已具备 `vrs_files/`、GT JSON/CSV 和 MPS closed-loop trajectory。本次汇总没有重新逐条打开每个 VRS 做帧完整性校验。

## 1. P1 FoundationStereo vs GT depth

| # | Sample | full AbsRel | full RMSE m | full δ1 | full scale | obj AbsRel | obj RMSE m | obj δ1 | obj scale | obj invalid | boundary AbsRel | depth gate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | WhiteLiddedTrashBin | 0.6249 | 5.5702 | 0.3295 | 1.3833 | 0.1021 | 0.2031 | 0.8762 | 1.0053 | 0.0000 | 0.3342 | False |
| 2 | WoodenBowl / meal_seq131 | 0.4160 | 2.1783 | 0.4254 | 1.3289 | 0.0703 | 0.0513 | 0.9543 | 1.0291 | 0.0000 | 0.2156 | True |
| 3 | BookDeepLearning / work_seq107 | 0.3927 | 1.4309 | 0.3697 | 1.2260 | 0.0882 | 0.2276 | 0.9775 | 1.0399 | 0.0000 | 0.2750 | True |
| 4 | BookDeepLearning / recognition_seq138 | 0.4568 | 1.9177 | 0.2728 | 1.3700 | 0.1027 | 0.3181 | 0.9826 | 1.0289 | 0.0000 | 0.3938 | False |
| 5 | BlackCeramicMug | 0.1125 | 0.3650 | 0.8905 | 1.0118 | 0.1184 | 0.2896 | 0.9699 | 1.0000 | 0.0000 | 0.3205 | False |
| 6 | WoodenSpoon | 0.2082 | 0.7058 | 0.5199 | 1.2434 | 0.1072 | 0.1717 | 1.0000 | 1.1120 | 0.0000 | 0.1044 | None |
| 7 | WoodenBowl / Lite_seq032 | 0.1181 | 0.4963 | 0.8956 | 1.0141 | 0.1791 | 0.4044 | 0.9578 | 1.0343 | 0.0000 | 0.4832 | None |
| 8 | DinoToy | 0.1090 | 0.5852 | 0.9150 | 0.9871 | 0.7982 | 1.0387 | 0.8855 | 0.9981 | 0.0000 | 0.7074 | None |
| 9 | Flask | 0.1060 | 0.4621 | 0.9140 | 1.0224 | 0.2273 | 0.4069 | 0.9580 | 1.0557 | 0.0000 | 0.5645 | None |
| 10 | StepStool | 0.1974 | 1.0568 | 0.7326 | 1.0784 | 0.0384 | 0.0547 | 0.9933 | 1.0256 | 0.0000 | 0.2792 | None |

## 2. P4 RGB Object Digital Twin 批处理状态

| # | Sample | P1 depth gate | prepare | SAM3 | SAM3D | FoundationPose | batch status |
|---:|---|---:|---|---|---|---|---|
| 1 | WhiteLiddedTrashBin | False | ok | ok | ok | fail | error |
| 2 | WoodenBowl / meal_seq131 | True | existing | existing | existing | existing | ok |
| 3 | BookDeepLearning / work_seq107 | True | existing | existing | existing | existing | ok |
| 4 | BookDeepLearning / recognition_seq138 | False | existing | existing | existing | existing | ok，但 `mask_gt` variant failed |
| 5 | BlackCeramicMug | False | ok | ok | fail | not reached | error |
| 6 | WoodenSpoon | None | ok | ok | ok | fail | error |
| 7 | WoodenBowl / Lite_seq032 | None | ok | ok | ok | fail | error |
| 8 | DinoToy | None | ok | ok | fail | not reached | error |
| 9 | Flask | None | ok | ok | fail | not reached | error |
| 10 | StepStool | None | ok | ok | fail | not reached | error |

说明：

- `existing` 表示该 sample 的 RGB object run 已存在，本轮批处理直接复用，不再产生新的步骤记录。
- `None` 表示这些样本的 P1 depth gate 尚未形成统一通过/失败结论。
- `Lite_seq032/WoodenBowl` 的 `foundationpose_raw.npz` 与 P5 `f0` 指标文件实际存在，但批处理日志里 FoundationPose 步骤因 `median_relative_depth_residual` gate 被拒绝，因此状态表中仍记为 fail/error。

## 3. P4 已形成完整 Object run 的 4 条

| Sample | SAM3 valid | SAM3D | FP valid | FP mean mask IoU | FP median depth residual | FP tracking score |
|---|---:|---:|---:|---:|---:|---:|
| recognition_seq138 / BookDeepLearning | 1.0000 | ok | 1.0000 | 0.6868 | 0.1892 | 0.8581 |
| meal_seq131 / WoodenBowl | 1.0000 | ok | 1.0000 | 0.7657 | 0.1272 | 0.8870 |
| work_seq107 / BookDeepLearning | 1.0000 | ok | 1.0000 | 0.7719 | 0.0755 | 0.8889 |
| Lite_seq032 / WoodenBowl | 1.0000 | ok | 1.0000 | 0.4804 | 0.2064 | 0.7691 |

## 4. P5 GT Replacement

当前有完整 `f0` 的 base run 共 4 条。

| Sample | F0 centroid median mm | depth_gt | mask_gt | mesh_gt |
|---|---:|---|---|---|
| recognition_seq138 / BookDeepLearning | 33.51 | ok 9.16 mm | failed | ok 9.28 mm |
| Lite_seq032 / WoodenBowl | 59.06 | failed | failed | failed |
| meal_seq131 / WoodenBowl | 42.82 | ok 15.09 mm | ok 36.99 mm | ok 22.42 mm |
| work_seq107 / BookDeepLearning | 12.05 | ok 14.78 mm | ok 24.62 mm | ok 14.40 mm |

说明：`ok X mm` 表示该 GT replacement variant 通过，`X` 为该 variant 的 centroid translation error median。

## 5. P6 Sequence

当前只有 2 条 source run 具备进入 sequence 的 artifacts。

| Sample | status | raw centroid mm | aligned centroid mm | centroid delta mm | scale delta | violation |
|---|---:|---:|---:|---:|---:|---|
| meal_seq131 / WoodenBowl | failed_before_export | 86.84 | — | — | — | sequence_failed_before_export |
| work_seq107 / BookDeepLearning | export_qc_failed | 24.38 | 28.26 | 0.004 | 0.0901 | sequence_export_qc_failed；metric_scale_rewritten_+0.0901 |

结论：严格 `validate_only` 下，目前没有任何 sequence 通过 export QC。

## 6. P7 Stereo Hand Auxiliary

| # | Sample | status | accepted_any_hand | best joint_valid_rate | best required_landmarks_frame_valid_rate |
|---:|---|---:|---:|---:|---:|
| 1 | WhiteLiddedTrashBin | evaluated | False | 0.0460 | 0.0000 |
| 2 | WoodenBowl / meal_seq131 | evaluated | False | 0.0000 | 0.0000 |
| 3 | BookDeepLearning / work_seq107 | evaluated | False | 0.6952 | 0.0667 |
| 4 | BookDeepLearning / recognition_seq138 | evaluated | False | 0.1413 | 0.0000 |
| 5 | BlackCeramicMug | evaluated | False | 0.0175 | 0.0000 |
| 6 | WoodenSpoon | stereo_artifact_missing | — | — | — |
| 7 | WoodenBowl / Lite_seq032 | stereo_artifact_missing | — | — | — |
| 8 | DinoToy | stereo_artifact_missing | — | — | — |
| 9 | Flask | stereo_artifact_missing | — | — | — |
| 10 | StepStool | stereo_artifact_missing | — | — | — |

## 7. 全流程结论

严格按当前 ADT benchmark 的 P0–P7 看，目前没有任何一个样本完整通过上游全流程。

如果只按“从 ADT 输入走到 SPIDER/MINK 尝试”来算，最接近完整的是：

- `work_seq107 / BookDeepLearning`

它在旧的 `adt_upstream_stage_summary_20260814.json` 中曾经到达 SPIDER export，并进行了 MINK 检查，但 MINK 因 `fingertip_orientation_fidelity` 和 `wrist_orientation_fidelity` 被拒绝。

在当前更严格的 P6 `validate_only` sequence benchmark 中，它虽然生成了 aligned trajectory，但 export QC 仍失败，原因是 `shared_metric_hand_depth_verified` 不通过，同时 object scale 被改写约 `+9.01%`。

所以准确回答是：

- 不是“只有一个样本跑完了全流程”；
- 而是“目前没有任何样本通过全流程验收，最接近的是 work_seq107，但也停在 sequence export QC / MINK 失败”。

