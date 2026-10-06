# TACO Depth / Mocap 世界坐标校准残差裁决 v1

**正式分类：`CALIBRATION_RESIDUAL_UNRESOLVED`。** 未修改 Depth、object pose、egocentric extrinsic 或 active `SupportSurfaceContract`。

1. **TACO 官方是不是认为相机和物体在同一个 world？** 是。设计目标是把 mocap 物体、手和头戴 L515 连接到同一个 mocap world；同日 sequence 共用 world。
2. **官方相机外参怎么标定？** 12 个场景 marker 的 mocap world 3-D 点（论文报告误差小于 1 mm）与人工 RGB 像素做 PnP，目标语义为 world→RGB camera。
3. **官方公开 Depth→RGB 内部外参了吗？** 没有找到清晰发布。`Depth pixel + RGB intrinsic` 是本项目的 `LOCAL_EVIDENCE_BACKED_INTERPRETATION`。
4. **Brush 与 Pour 同日 Depth table 是否一致？** 是；offset 最大差 `0.834 mm`，两者都来自同一冻结估计器。
5. **17 ms 能解释约 20 mm 吗？** `TEMPORAL_SYNC_UNLIKELY_TO_EXPLAIN_20MM`；按 1 m 杠杆臂计入旋转的保守最大值仍小于 20 mm。
6. **无 correction 的刚体表面误差？** `{"brush_brush_bowl_20230927_027": {"signed_median_m": -0.0053835774306207895, "absolute_median_m": 0.005397253204137087, "absolute_p90_m": 0.02903428375720979, "absolute_p95_m": 0.057968976721167374}, "pour_bowl_plate_20230927_017": {"signed_median_m": -0.003347837133333087, "absolute_median_m": 0.0042108953930437565, "absolute_p90_m": 0.0122331565245986, "absolute_p95_m": 0.018991116806864723}}`（单位为米，低覆盖 target 未参与 fit）。
7. **world-fixed 能跨 sequence 吗？** `True`。
8. **camera-local 能跨 sequence 吗？** `True`。
9. **哪种解释更符合数据？** `两种简单固定模型都没有得到唯一、可跨 sequence 的支持`。
10. **完全不看桌面拟合后，能自动修复桌面 mismatch 吗？** `NOT_RUN_NO_CROSS_SEQUENCE_VALIDATED_CORRECTION`。
11. **冻结后 bowl / brush bottom 还差多少？** `{"schema": "taco_brush_object_bottom_final_holdout_v1", "status": "NOT_READ_NO_FROZEN_CORRECTED_TABLE", "used_for_calibration_fit": false}`。
12. **下一步有资格正式接入 correction 吗？** `否；未满足全部独立验证条件`。

本轮只使用高 Depth coverage 的 tool 刚体表面拟合；table、bowl bottom、brush bottom 与 simulator 高度均未进入 correction objective。issue #18/#20 只作为官方 tracker 中的社区线索，不视为作者确认。
