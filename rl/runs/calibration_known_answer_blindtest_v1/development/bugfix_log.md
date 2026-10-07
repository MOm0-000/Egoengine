# 已证实并修复的程序错误

1. **数值差分精度错配**：Open3D signed distance 为 float32，旧 SciPy 默认差分可被舍入吞掉。保留 `diff_step=null` 的原版入口；修复版固定 `diff_step=0.001`，不改变目标、loss 或求解器。
2. **目标单独轮廓忽略遮挡**：旧选择器在目标投影内直接读取 measured depth。修复版同时使用名义目标、双手和其他物体的深度顺序，近似平局保守排除。

公开练习汇总：`{"baseline": {"comparable_outputs": 22, "translation_error_mm_median": 3.259283340505771, "translation_error_mm_max": 25.05855449193272, "rotation_error_deg_median": 0.24172702009205754, "rotation_error_deg_max": 1.8606423242297132, "selected_occluder_points": 511}, "fixed": {"comparable_outputs": 28, "translation_error_mm_median": 0.00010120437889128826, "translation_error_mm_max": 8.631832707752054, "rotation_error_deg_median": 4.829673078902935e-06, "rotation_error_deg_max": 1.0858083246887549, "selected_occluder_points": 18}}`。
