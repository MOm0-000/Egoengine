# 校准算法已知答案测试与遮挡修复 v1

**状态：开发与公开练习完成；正式保密考试未执行。**

- 本轮直接比较了正确 4×4 恢复变换与算法输出；没有用表面距离冒充真值误差。
- 已证实并最小修复两个实现错误：float32 距离场与默认有限差分尺度错配；目标单独轮廓未排除名义手/其他物体遮挡。
- 原版公开可比题：translation median/max = `3.259283/25.058554 mm`；rotation median/max = `0.241727/1.860642 deg`。
- 修复版：translation median/max = `0.000101/8.631833 mm`；rotation median/max = `0.000005/1.085808 deg`。
- 遮挡题误选 occluder 点总数：原版 `511`，修复版 `18`。完整逐题误选、保留与错删见 `occlusion_comparison.csv`；空选择不会被记成零污染。
- 人工形状与现有 TACO mesh 结果分别保留在 CSV，不合并伪装成单一精度。
- 无公共修正/退化输入暴露了当前方法缺少一般可辨识性证书；未更换目标、增加多起点或扩展模型。
- bubblewrap 假秘密演示：exit `0`，secret visible `False`，network available `False`。这不等于正式环境已经隔离。
- 真实 TACO 数据没有重拟合或修改，上一轮 `CALIBRATION_RESIDUAL_UNRESOLVED` 结论保持不变。
