# 左掌—左拇指碰撞边界拓扑 v1

## 这一步回答什么

这一步不修模型，只先回答一个简单问题：左拇指两个关节在什么组合下，原始
CAD 会真正穿进左掌；当前被拒的 Hybrid 又把这条界线画在了哪里。

输入保持不变：正式 scene、198 帧 robot reference、原始 palm/thumb CAD，及
被拒 Hybrid 的 8 对局部凸块和 2 对球。没有修改 XML，没有重新 MINK，没有
跑物理 reset、capacity 或 PPO。

## 怎么检查

- `thumb_bend` 在 `[0, 1.83] rad` 取 257 个切片；
- 每个切片把 `thumb_rota1` 在 `[-1.05, 1.57] rad` 取 513 个粗点；
- 不预设只有一次跳变，逐一寻找所有 clear↔collision 变化；
- 每个变化再二分，直到数值括号不宽于 `1e-7 rad`；
- native CAD 和冻结的 Hybrid 完全独立地提取边界；
- 原 Hybrid 最终留出集中的 1 个漏检点和 1 个误报点单独复测。

`1e-7 rad` 只表示 FCL 数值定位括号，不代表 CAD 制造精度。有限的 513 点粗扫
也可能漏掉比一个粗网格单元还窄的碰撞岛；报告明确保留这个限制。

## 结果

| 项目 | native CAD | 被拒 Hybrid |
| --- | ---: | ---: |
| 无碰撞 bend 切片 | 104 | 104 |
| 单一碰撞区间切片 | 153 | 153 |
| 多碰撞区间切片 | 0 | 0 |
| 与对方拓扑不一致切片 | — | 0 |

每个发生碰撞的切片都只有一个连续区间，而且都延伸到 `rota1=1.57 rad`。但从
二维平面看，当前关节域内有两段分开的采样 bend 带：

- `bend=0 ... 0.900703125 rad`；
- `bend=1.6512890625 ... 1.83 rad`。

带端点只是 257 个 bend 采样点的首尾，没有在 bend 方向二分，不应误读成
精确 onset。

Hybrid 没有画错“有几块碰撞区域”，只是边界略有偏移。在 153 个可比切片上：

- 边界 RMSE：`0.0002126817 rad`；
- 中位绝对误差：`0.0001779491 rad`；
- P95 绝对误差：`0.0004215027 rad`；
- 最大绝对误差：`0.0005000377 rad = 0.02865°`；
- 32 个切片偏早，会造成窄带误报；
- 121 个切片偏晚，会造成窄带漏检。

原先最终留出集的 FN 和 FP 都被新审计器原样复现，所以不是旧报告的统计误会。

## 下一步

问题已经适合按两条边界分支做 CEGIS：把实测 FN/FP 当约束，允许删除或缩紧
造成 FP 的旧组件，再增加只覆盖 FN 局部的组件。不能只加一个球，因为 pair
判定是 OR；只加碰撞体只能扩大碰撞集合，救 FN 的同时不会自动消掉 FP。

新 self guard 在完整二维域通过独立留出验证后，顺序固定为：合并 external/
floor candidate；只因 self pair 变化而重跑 198 帧 MINK；在新 reference 上重跑
0–197 全时域 external/floor attribution；复测 A/B t0；再跑 capacity、重建
physics contract 和 `initialization_protocol_v2`。至少一个 reset 通过 t0 与被动
释放后，才冻结 observation 并打开第一段 Replay→RL。

机器可读结果：
`runs/taco_pour_left_palm_thumb_boundary_topology_v1/report.json`。
