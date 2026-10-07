# 用户运行正式保密考试

开发与公开核对已完成。下一批 96 题必须由用户在私有外层 namespace 中生成；Codex 不应在预测封存前读取题面、私有 seed、答案或中间日志。

```bash
CURRENT_PY=/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python
OPEN3D_PY=/data_all/zzx/calibration_open3d_v0200/env/bin/python
TOOL=rl/scripts/calibration_open3d_comparison_v1.py

$CURRENT_PY $TOOL generate --secret-seed-file /secure/private/seed.txt --public-dir /secure/public/input --private-dir /secure/private/answer
$CURRENT_PY $TOOL prepare --input-dir /secure/public/input --prepared-dir /secure/public/prepared
$CURRENT_PY $TOOL run-bwrap --prepared-dir /secure/public/prepared --output-dir /secure/output/current_fixed --method CURRENT_FIXED
$OPEN3D_PY $TOOL run-bwrap --prepared-dir /secure/public/prepared --output-dir /secure/output/open3d_point_to_plane --method OPEN3D_POINT_TO_PLANE
$OPEN3D_PY $TOOL run-bwrap --prepared-dir /secure/public/prepared --output-dir /secure/output/open3d_robust_point_to_plane --method OPEN3D_ROBUST_POINT_TO_PLANE
$CURRENT_PY $TOOL score --input-dir /secure/public/input --private-dir /secure/private/answer --prepared-dir /secure/public/prepared   --current-dir /secure/output/current_fixed --point-dir /secure/output/open3d_point_to_plane   --robust-dir /secure/output/open3d_robust_point_to_plane --output-dir /secure/score
$CURRENT_PY $TOOL verify --input-dir /secure/public/input --private-dir /secure/private/answer --prepared-dir /secure/public/prepared
```

所有三份预测完成并哈希封存后才能评分。正式环境禁网，solver 不得挂载私有答案或其他方法输出。若接口失败，本批考试作废；修复后必须换新私有 seed。
