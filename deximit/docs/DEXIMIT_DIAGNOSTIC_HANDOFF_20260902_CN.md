# DexImit 隔离诊断与跨样本测试交接（2026-09-03 更新）

## 1. 一句话结论

DexImit 到当前项目的隔离接入已经能完整运行“BODex 生成抓姿 → 原版 DexImit 排序和 SAPIEN 筛选 → 等价 MuJoCo 复验 → 固定第三人称三联视频”。目前严格双仿真通过的仍只有橡皮涂抹样本的候选 21；刷子、两段铲刀动作和锤子样本都在 SAPIEN 阶段失败，因此没有进入 MuJoCo，也没有生成容易误导的成功视频。

本阶段始终是独立诊断，不得写入正式 renderer 或论文 3.3 主链。

## 2. 不可改变的约束

- 永久禁止跟随镜头。所有可交付三联视频必须使用固定世界坐标第三人称视角。
- 三联视频列名固定为 `real | reference | sim`；第三列虽由 MuJoCo 产生，也只显示为 `sim`。
- 只有 SAPIEN 与 MuJoCo 都通过严格门槛，才允许生成三联成功视频。
- 失败样本只保存指标和原因，不通过调参反复“救样本”，也不渲染成看似成功的视频。
- C、D 分阶段接触控制只可用于诊断对照，绝不进入正式 renderer/3.3。
- 新样本使用同一套已冻结参数，不允许逐样本修改质量、摩擦、力限或接触门槛。

## 3. 已完成结果表

| 物品 | 任务与序列 | SAPIEN | MuJoCo | 最终判定 | 视频 |
|---|---|---:|---:|---|---|
| 橡皮（078） | smear，`20231103_071` | 候选 21 通过 | 严格门槛通过 | 成功，但目前只有这一个严格双通过样本 | `runs/deximit_full_bridge/smear071/triptych_candidate21_v4_reference_style_sim/real_ref_sim_passed_fixed_third_person.mp4` |
| 刷子（071） | brush，`20230927_027` | 0/120 | 按门槛不运行 | 失败于 SAPIEN | 无成功视频 |
| 铲刀（033） | cut，`20230917_020` | 0/120 | 按门槛不运行 | 失败于 SAPIEN | 无成功视频 |
| 铲刀（033） | skim off，`20230926_004` | 0/120 | 按门槛不运行 | 失败于 SAPIEN | 无成功视频 |
| 锤子（139） | hit，`20231102_051` | 0/120 | 按门槛不运行 | 失败于 SAPIEN | 无成功视频 |

不能根据这一结果声称已经具备跨物体泛化性：严格双通过率在已完成的五段测试中为 1/5，而且四个新样本都没有跨过第一套仿真。

## 4. 新增样本的细分指标

统一设置：每个物体生成四个抓取深度（0、1、2、3），每个深度 500 个 BODex 候选；先由 cuRobo 检查可达性，再按 DexImit 原版的真人手方向误差排序，实际测试前 120 个；SAPIEN 的原版物体平均顶点运动误差门槛为 20 mm。

| 序列 | 初筛可达数 / 2000 | 120 个中完整规划数 | 有机器手接触的候选数 | 物体误差：最小 / 中位 / 最大 | 主要失败层级 |
|---|---:|---:|---:|---:|---|
| brush027 | 640 | 99 | 29 | 62.7 / 81.2 / 114.3 mm | 21 个路径阶段失败；其余多数未把刷子带离桌面 |
| cut020 | 1846 | 120 | 69 | 146.9 / 158.4 / 188.3 mm | 路径都能执行，但抓姿没有让铲刀跟随目标运动 |
| skim004 | 995 | 4 | 0 | 158.8 / 158.8 / 158.8 mm | 95 个在示范运动规划失败，20 个预抓失败，1 个抓取失败；没有机器手承载接触 |
| hit051 | 1240 | 91 | 23 | 66.6 / 92.3 / 112.0 mm | 28 个预抓阶段失败，1 个抓取阶段失败；其余候选未达到 20 mm 物体误差门槛 |

原始报告：

- `runs/deximit_generalization_v1/brush027/sapien_combined_top120_v1/summary.json`
- `runs/deximit_generalization_v1/cut020/sapien_combined_top120_v1/summary.json`
- `runs/deximit_generalization_v1/skim004/sapien_combined_top120_v1/summary.json`
- `runs/deximit_generalization_v1/hit051/sapien_combined_top120_v1/summary.json`
- 机器可读汇总：`runs/deximit_generalization_v1/generalization_summary_v1.json`

## 5. 如何理解失败原因

这三次失败不能归因于 MuJoCo，因为它们在进入 MuJoCo 前就已被 SAPIEN 拒绝。

最重要的共同问题是：BODex 生成的是“从物体几何上看可能稳定”的抓姿，而当前排序主要比较机器手朝向与真人手朝向，没有直接评价这个抓姿是否适合后续任务运动，也没有先把整段机械臂路径的可执行性纳入排序。因此会出现以下情况：

1. 排名靠前，但机械臂在预抓或后续运动中到不了。
2. 手指碰到了物体，但接触位置或对向夹持不足，物体仍留在桌上。
3. 细长工具绕世界坐标旋转时，目标表面顶点位移很大；只比较抓取瞬间方向，不能保证后续跟随。

这说明下一步研究重点应放在“任务条件抓姿排序”和“整段路径可行性”，而不是继续调整 MuJoCo 摩擦。当前停止规则是正确的：某样本 0/120 后记录失败并换样本。

## 6. 橡皮候选 21 的严格通过证据

SAPIEN 报告：

- `runs/deximit_full_bridge/smear071/sapien_candidate21_joint_forces_v13/summary.json`
- 明确筛选的候选 21 为 1/1 通过。

MuJoCo 报告：

- `runs/deximit_full_bridge/smear071/mujoco_candidate21_exact_equiv_v9_strong_friction/report.json`
- 抓稳后的最低抬升：64.69 mm。
- 抓稳末端抬升：65.39 mm。
- 稳定段竖直波动：0.70 mm。
- 最大允许物体穿透门槛：2 mm；实际最小手物间隙约为 -0.097 mm。
- MuJoCo 相对 SAPIEN 的物体位置平均误差：0.86 mm，最大误差：2.30 mm。
- 严格门槛字段 `strict_gate.passed` 为 `true`。

固定第三人称三联视频：

- `runs/deximit_full_bridge/smear071/triptych_candidate21_v4_reference_style_sim/real_ref_sim_passed_fixed_third_person.mp4`

## 7. 本轮代码修复

### 7.1 接触标注生成

`scripts/build_taco_surface_contact.py`

- 新增 `--query-workers`，把精确最近三角面查询分块并行。
- 输出顺序保持不变；串行与并行结果已有逐元素一致性测试。
- `mesh.contains` 前增加包围盒预筛，只减少无意义计算，不改变几何定义。

### 7.2 BODex 候选生成

`diagnostics/generate_bodex_candidates.py`

- 修复候选坐标契约：生成器现在必须显式接收 warmup 后的 SAPIEN 物体姿态，先按上游 `gen_traj.py` 的真实姿态调用 BODex，再把 BODex 返回的世界坐标转换回源物体坐标后保存。此前固定用单位姿态求解，再在筛选器中旋回；对非对称细长工具这不是等价变换。
- 原有 NPZ 字段和筛选器的 `object_initial @ local_pose` 路径保持不变，因此已有 identity 生成的橡皮候选 21 仍可作为旧协议回归基准；新池不能覆盖旧池。
- 调用时用 `--object-pose-wxyz X Y Z W QX QY QZ` 传入对应 SAPIEN 报告中的 `object_initial_pose_sapien`，不能省略或回退到单位姿态。仅需复现旧池时才显式使用 `--legacy-identity-pose`。
- 临时 URDF 中的机器人名称由物体网格名生成，不再写死为橡皮样本名。
- 第三方原生扩展在 Python 正常退出时会破坏堆内存。现在只在候选文件已原子写入、校验和已完成且成功信息已刷新后调用 `os._exit(0)`，绕开有缺陷的第三方清理函数；前面的真实异常仍保持非零退出。

### 7.3 SAPIEN 筛选

`diagnostics/run_deximit_sapien_screen.py`

- 活跃手指从写死的 4 指改为经过校验的 2–5 指，支持五指标注。
- 物体名称来自网格文件名，不再写死为 `eraser`。
- 候选池必须同时匹配序列编号、源物体网格哈希、手指数；防止跨序列或跨网格串线。
- MANO 提示必须绑定同一份第三版接触标注及其哈希。
- 修复无视频的批量筛选仍强制初始化光线追踪降噪器、从而在本机停死的问题。现在 `--render-mode raster` 可用于纯筛选，只替换图像初始化后端，SAPIEN 物理、相机和候选均不变。
- 未使用的两套图像侧场景在普通光栅或轨迹导出时不会创建，避免它们覆盖固定 Front 相机的渲染器。

### 7.4 MuJoCo 与视频

`diagnostics/build_exact_deximit_mujoco_scene.py`

- `--candidate` 改为必填，删除隐藏的候选 21 默认值，避免新样本悄悄复用橡皮候选编号。

`diagnostics/render_exact_deximit_triptych.py`

- 真人阶段行号改为必填，删除隐藏的橡皮默认行号。
- 程序在内存中强制移除跟随相机，只允许固定世界第三人称相机。
- MuJoCo 列统一显示为 `sim`，并复刻正式 reference 的视觉场景。

### 7.5 验证

命令：

```bash
spider/.venv/bin/python -m pytest -q tests/test_bodex_diagnostic_contract.py
```

结果：`34 passed`。覆盖 BODex 世界姿态到物体局部姿态的往返、显式 SAPIEN 生成姿态、2–5 指契约、跨序列/跨网格候选拒绝、接触查询串并行一致性、固定相机与诊断隔离等检查。

### 7.6 姿态契约 A/B

- 使用锤子 `hit051`、四指、depth=1、seed `20260827`，分别比较单位姿态池和按 warmup 后 SAPIEN 姿态求解的池；两者不是刚体等价，候选关节和 wrist orientation 均发生明显变化。
- 新姿态池的原版 rotation top-120 对照仍为 `0/120`，因此这次修复只确认并恢复了上游候选生成语义，不能把它解释为锤子样本已经成功。
- 对照输出保存在 `runs/deximit_bodex_probe/hit051/bodex_right_f4_d1_posefix_v1.npz` 和 `runs/deximit_generalization_v1/hit051/sapien_actual_pose_ab_v2_top120/`，均为诊断文件，不覆盖冻结结果。

### 7.7 候选池姿态绑定修复（2026-09-03）

继续审计发现，候选池 loader 之前只检查网格、序列和手指数，未检查 BODex 生成候选时使用的物体姿态。这样旧的 identity-pose 池可以无提示地被放到 warmup 后的旋转物体上；对非对称细长工具，这不是刚体等价变换。

- `diagnostics/run_deximit_sapien_screen.py` 现在默认使用 `--candidate-pose-policy strict`：每个池必须声明 `generation_object_pose_wxyz`（可从 NPZ 或 provenance 读取），并在 SAPIEN warmup 后同时满足位置误差 `<=20 um`、旋转误差 `<=20 urad`。
- 旧池仍可用 `--candidate-pose-policy legacy` 显式复现，但运行配置和摘要会标记为 `legacy_unverified`；不会覆盖旧报告，也不能作为严格上游等价证据。
- loader 同时拒绝有效候选行中的 NaN、无穷值和非法四元数，避免把损坏输入误报为 cuRobo/SAPIEN 规划失败。
- 橡皮 candidate 21 使用显式 legacy 兼容模式回归仍通过 SAPIEN：`1/1`，物体平均顶点误差 `17.75 mm`；原成功目录保持不变。
- 契约测试为 `37 passed`，全量测试为 `319 passed`。本轮没有重新宣称刷子、铲刀或锤子成功，也没有修改正式 renderer/3.3 主链。

## 8. 环境与运行要点

DexImit 源码：

```text
/data_all/zzx/deximit_isolated/DexImit-Open-c5749809
```

DexImit/SAPIEN/BODex 环境：

```text
/data_all/zzx/deximit_isolated/env-py311-cu118
```

运行 BODex 或 SAPIEN 前必须使用：

```bash
LD_LIBRARY_PATH=/data_all/zzx/deximit_isolated/env-py311-cu118/lib:/usr/local/cuda/lib64
PYTHONPATH=/data_all/zzx/egoengine_new
```

否则系统旧版 `libstdc++` 会缺少 `GLIBCXX_3.4.31`。SAPIEN 批量筛选还必须加 `--render-mode raster`，否则本机的光线降噪器可能打印非法显存访问后停在图形事件等待。

MuJoCo 与测试使用：

```text
/data_all/zzx/egoengine_new/spider/.venv/bin/python
```

## 9. 锤子样本的完成记录

已完成：

- 官方 MANO 文件已解包到 `data/taco_v1/hand_poses_v1/extract/Hand_Poses/(hit, hammer, toy)/20231102_051/`。
- 真人视频为 `data/taco_v1/object_task3_rgb/taco_hit_hammer_toy_20231102_051/color.mp4`。
- 原项目参考数据位于 `runs/taco_action_taco_hit_hammer_toy_20231102_051/`。

已完成的诊断输入与阶段：

- 保守接触标注：`runs/contact_gt_v3/hit051/contact_geom_v3.npz`。
- 人工阶段标签：预抓第 25 行、抓取第 52 行、动作截断第 60 行；活跃手指为拇指、食指、中指、无名指，记录在 `runs/deximit_generalization_v1/hit051/manual_pickup_subactions_v1.json`。
- 四个深度各生成 500 个 BODex 候选，并按原版 DexImit MANO 方向排序。
- SAPIEN 冻结筛选已完成 `120/120`：`1240/2000` 个候选通过初始 IK 可达性，前 120 个中 `91` 个完成四段规划，`23` 个有机器手接触，但 `0/120` 通过严格物体误差门槛。

因此按停止规则，锤子样本不运行 MuJoCo，不导出轨迹，也不生成三联视频。完整机器记录位于 `runs/deximit_generalization_v1/hit051/sapien_combined_top120_v1/summary.json`；这次是 SAPIEN 阶段物理筛选失败，不再是此前“尚未完成”的状态。

## 10. 可恢复的失败运行

以下目录只是被移入回收区，没有永久删除：

- `.trash/deximit_generalization_failed/brush027_work_f4_d0_failed_glibcxx_20260902`：第一次 BODex 因旧 `libstdc++` 失败。
- `.trash/deximit_generalization_failed/brush027_sapien_rt_stall_20260902`：光线降噪器停死，只含运行配置。

它们不应作为有效实验输入；如需追查仍可恢复。

## 11. 给下一位 AI 的最短决策规则

1. 先读本文件和每个样本的 `summary.json`，不要重跑已完成的四个失败样本。
2. 锤子冻结流程已经完成；除非明确开展新的实验，不要把本次 `0/120` 重新跑成另一条结果。
3. SAPIEN 失败就停止；SAPIEN 通过才运行 MuJoCo。
4. MuJoCo 必须使用与橡皮候选 21 相同的等价转换方法，但强摩擦保持审计必须针对新的通过候选重新验证，不能直接把候选 21 的候选专用报告冒充新证据。
5. 只有两套仿真都通过才输出固定第三人称 `real | reference | sim` 视频。
6. 所有结果继续保持 `diagnostic_only=true`、`formal_renderer_3_3_eligible=false`。
