# Brush 单帧校准临时修正与桌面穿透检查 v1

**最终结论：`PENETRATION_REDUCED_BUT_REMAINS`。**

本轮逐帧独立使用当前修复版自创算法；未修改原始 Depth、相机参数、物体姿态、手轨迹或 active `SupportSurfaceContract`。世界坐标与相机坐标结果均完整保留，未按穿透结果挑选模型。桌面基线来自同一帧未修正原始深度，未使用旧的 bowl-bottom support 定义。

- 穿透门槛：`-0.050 mm`（更负才计为穿透）
- 样本帧数：`209`
- MINK / physics / Replay / MPC / RL：全部 `0`

| 坐标模型 | 修正输出帧 | 同帧可比较 | 未检查 | 修正前任一实体穿透帧 | 修正后任一实体穿透帧 | 最深修正后穿透 | 分类 |
|---|---:|---:|---:|---:|---:|---:|---|
| WORLD_FIXED | 116 | 115 | 94 | 106 | 104 | -55.505 mm @ frame 184 (left_hand) | PENETRATION_REDUCED_BUT_REMAINS |
| CAMERA_LOCAL | 117 | 117 | 92 | 108 | 105 | -52.801 mm @ frame 157 (left_hand) | PENETRATION_REDUCED_BUT_REMAINS |

## 各实体穿透帧数（相同有效帧）

### WORLD_FIXED

| 实体 | 修正前穿透帧 | 修正后穿透帧 | 修正前最小距离 mm | 修正后最小距离 mm |
|---|---:|---:|---:|---:|
| brush | 97 | 95 | -25.584 | -13.303 |
| bowl | 104 | 94 | -25.703 | -43.061 |
| left_hand | 102 | 92 | -23.416 | -55.505 |
| right_hand | 58 | 58 | -22.604 | -18.195 |
| hand | 102 | 94 | -23.416 | -55.505 |

第 0 帧：
- brush -20.123→-8.471 mm, bowl -19.136→-1.038 mm, left_hand -17.634→0.719 mm, right_hand -16.138→-12.247 mm

桌面跨帧 offset 波动（相同有效帧）：
- 修正前 range `6.568 mm`，MAD `0.819 mm`。
- 修正后 range `51.969 mm`，MAD `2.959 mm`。
### CAMERA_LOCAL

| 实体 | 修正前穿透帧 | 修正后穿透帧 | 修正前最小距离 mm | 修正后最小距离 mm |
|---|---:|---:|---:|---:|
| brush | 100 | 95 | -25.584 | -12.701 |
| bowl | 107 | 95 | -25.703 | -39.519 |
| left_hand | 104 | 94 | -23.416 | -52.801 |
| right_hand | 58 | 58 | -22.604 | -17.206 |
| hand | 104 | 95 | -23.416 | -52.801 |

第 0 帧：
- brush -20.123→-8.601 mm, bowl -19.136→-1.655 mm, left_hand -17.634→-0.662 mm, right_hand -16.138→-12.777 mm

桌面跨帧 offset 波动（相同有效帧）：
- 修正前 range `6.568 mm`，MAD `0.851 mm`。
- 修正后 range `53.435 mm`，MAD `3.083 mm`。

## 解释边界

- 部分帧改善不能视为全程通过。
- 穿透减轻只说明该临时点云解释改变了估计桌面，不证明标定正确。
- correction objective 只使用可见刷子表面深度；桌面、碗底和刷子底部距离均未参与拟合。碗只作事后验证。
- 本轮没有把任何 correction 或桌面写入 active runtime。
