# Pour first-40 碰撞归因 v1

这是一轮只读诊断，不是新的碰撞模型，也不是 Replay/RL 结果。输入被哈希固定，
检查 reference endpoint 0–40，并额外检查初始化协议 v1 的 Candidate A 首帧。
没有修改场景、参考轨迹或 GT，也没有执行物理步进。

## 检查办法

对 first-40 中每一组“原始 CAD 表面发生交叉、且不是已确认固定装配缝”的
部件，分别检查四种组合：

1. 原始 CAD ↔ 原始 CAD；
2. 当前 collision proxy ↔ 当前 collision proxy；
3. 原始第一侧 ↔ 第二侧 collision proxy；
4. 第一侧 collision proxy ↔ 原始第二侧。

这样可以把“碰撞 pair 根本没声明”“手侧壳缺材料”“物体侧壳缺材料”和
“两侧组合后才出现的误差”分开。`50 µm` 只用作原始闭合实体的采样报告档位，
不是 MuJoCo 的接触激活阈值，也不是论文阈值。

## Candidate A 的直接结论

| 原始干涉 | 当前模型为什么漏掉 | 下一步最小修复方向 |
| --- | --- | --- |
| 左中指末节 ↔ 盘 | 手指 capsule 缺材料；原始中指与当前盘 proxy 会碰，但当前中指 proxy 与原始盘不碰 | 只给左中指末节—盘这类外部接触增加局部 semantic guard；不改盘 |
| 左小指末节 ↔ 盘 | 同样是手指侧 underfill | 只给左小指末节—盘增加局部 semantic guard；不改盘 |
| 左掌 ↔ 拇指近节 | 两个 body 之间没有 runtime pair | 不能直接连接现有 palm box 和 thumb capsule；需单独拟合 pair-specific guard |

中指的当前 runtime 间隙约 `+2.00 µm`，但原始 CAD 采样最大进入盘约
`1.761 mm`。小指的 runtime 间隙约 `+0.187 mm`，原始 CAD 采样最大进入盘约
`0.357 mm`。两个 mixed test 都证明盘的当前 proxy 在这两个局部不是主要漏项，
因此不应先改盘的 32 块碰撞分解。

左掌—拇指近节在 Candidate A 中有约 `0.457/0.357 mm` 的双向采样实体证据，
但当前没有任何声明的 runtime geom pair。现有 palm box 与 thumb capsule 在
endpoint 0–40 的 41 个状态中始终相交，其中有 22 帧原始 CAD 是 clear。
所以“把现有两块直接连成 pair”会制造 22/41 的明显误报。该 pair 相对运动由
拇指 bend 与第一旋转关节共同决定，需要二维姿态采样，不能套用食指根部的
单关节一维球拟合。

## first-40 结果

自动选中 15 组非固定装配缝 pair，共 41 个离散 endpoint：

- 原始 CAD surface crossing：157 个 frame-pair；
- 其中有超过 50 µm 的闭合实体采样证据：155 个；
- 当前 runtime proxy 已检测到：132 个；
- runtime 漏掉：25 个，其中 23 个有上述实体证据；
- 漏项归因：19 个来自左掌—拇指 pair 缺失，6 个来自第一侧（手侧）proxy underfill；
- 同一批 pair/endpoint 上另有 23 个 runtime 有碰撞而原始 CAD 表面不交叉的情况，
  后续拟合必须把这些 false-positive 样本一起保留，不能只追求零漏检。

另一个更重要的结果是：132/157 个原始 crossing 当前 runtime 已经能看到。
因此 first-40 的主要问题不能全部归咎于 collision proxy。当前双手 MINK 正式
reference 只把已声明的手内/双手碰撞作为硬约束，没有把手—碗、手—盘的
非穿透作为该 bilateral retarget 的硬约束。于是参考轨迹本身可以进入物体，
即使 runtime 接触模型能够检测它。

## 下一步建议

1. 先拟合左中指末节和左小指末节的 pair-specific external guards。guard 只参与
   对盘的 pair，不应顺带改变手—桌、双手或其他手指的接触。
2. 对左掌—拇指近节做两关节 CAD sweep，再拟合独立 semantic guards；不能直接
   启用现有大 palm box/capsule pair。
3. 在 Candidate A、endpoint 0–40、错位 holdout 和完整 198 帧上同时统计漏检与
   误报。只有这一步通过，才更新正式 scene。
4. 更新 scene 后重新重定向。新的 bilateral retarget 必须显式决定如何实施
   “手—物体不穿透但允许接触”；这属于本项目局部扩展，不能说是论文公开参数。
5. 新 reference 通过同一归因审计后，再创建
   `taco_pour_initialization_protocol_v2`。v1 保留为失败的负实验。

完整逐 pair、逐 endpoint 证据见
`runs/taco_pour_first40_collision_attribution_v1/report.json`。
