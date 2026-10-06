# TACO 多帧桌面深度估计与验证 v1

**正式分类：`BRUSH_TABLE_HOLDOUT_MISMATCH`。** active `SupportSurfaceContract` 没有修改。

1. **有没有用 bowl bottom 估桌面？** 没有。估计器只读取真实 metric Depth、相机、官方前景投影和 world +Z；consensus 文件写入并 SHA256 冻结后才读取 bowl/brush mesh。
2. **使用多少帧？** 四个样本共处理 `897` 帧 native Depth；Brush 的 fit/holdout 划分严格由 `frame % 5` 决定。
3. **去前景后候选点：** 空间 stride=8 后四样本累计 `5276593` 个真实背景深度点；没有插值或单目补洞。
4. **逐帧稳定性：** Brush 状态为 `STABLE_TABLE_PLANE`；fit offset MAD 为 `0.0009024846910391515` m。
5. **最终桌面：** normal=`[0.024696724175695732, -0.028447730842748646, 0.9992901472669928]`，world-plane offset/高度=`0.570251794 m`。
6. **真实 Depth holdout：** `{"all_background": {"point_count": 447102, "signed_residual_median_m": 0.0008400722652949777, "signed_residual_mad_m": 0.006253146116463437, "signed_residual_p05_m": -0.01835839107226243, "signed_residual_p50_m": 0.0008400722652949777, "signed_residual_p95_m": 0.1620621580473009, "absolute_residual_median_m": 0.006187353606198265, "absolute_residual_p95_m": 0.1725729963658208}, "table_support": {"point_count": 292232, "signed_residual_median_m": -0.00012710816195476982, "signed_residual_mad_m": 0.003513489020145699, "signed_residual_p05_m": -0.008165511244383744, "signed_residual_p50_m": -0.00012710816195476982, "signed_residual_p95_m": 0.00805242464734248, "absolute_residual_median_m": 0.0035140168422862494, "absolute_residual_p95_m": 0.008986256407107597}, "table_support_point_count": 292232, "table_support_fraction": 0.653613716780511, "selection": "absolute consensus-plane distance <= frozen RANSAC inlier distance"}`
7. **冻结后 bowl bottom 距离：** `-22.317 mm`；状态 `OUTSIDE_ESTIMATOR_UNCERTAINTY`。
8. **brush bottom 距离：** `-22.276 mm`。
9. **旧 -1.311 mm：** 没有作为真值或输入延续；应由上面的新独立测量取代。
10. **其他样本：** `{"brush_brush_bowl_20230927_027": "STABLE_TABLE_PLANE", "pour_bowl_plate_20230927_017": "STABLE_TABLE_PLANE", "skim_spatula_plate_20230926_004": "INSUFFICIENT_TABLE_DEPTH_EVIDENCE", "smear_eraser_box_20231103_071": "STABLE_TABLE_PLANE"}`，全部使用同一算法和参数。
11. **能否下一任务替换 active contract？** `不能；当前分类未通过候选验证`。

本轮 MINK、physics、Replay、MPC、RL、promotion、chunk commit 和 active support replacement 均为 0。
