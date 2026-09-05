# DexImit 当前 MuJoCo 诊断链与开源代码差异审计

## 1. 审计范围

本报告区分两层差异：

1. DexImit-Open 官方提交与当前实际使用的 DexImit/SAPIEN 隔离副本。
2. 当前实际使用的 SAPIEN 流程与新增的 MuJoCo 诊断流程。

开源代码副本：

```text
/data_all/zzx/deximit_isolated/DexImit-Open-c5749809
```

记录的基线提交：`c5749809fa425525a9bf7aa392dbb3f6268ae28f`。

当前隔离副本还有 4 个未提交修改文件：

- `third_party/any2dex/any2dex/env/base_env.py`
- `third_party/any2dex/any2dex/env/base_env_for_render.py`
- `third_party/any2dex/any2dex/util/bodex_util.py`
- `third_party/any2dex/any2dex/util/util.py`

因此，下面的“原始 SAPIEN”指当前实际运行的 DexImit 副本；涉及官方提交时会明确写成“上游提交”。

## 2. 隐藏问题检查结果

### 已确认并修复的问题

| 位置 | 问题 | 修复效果 |
|---|---|---|
| `env/base_env.py`、`env/base_env_for_render.py` | 新版 `trimesh` 的 `mesh.centroid` 可能是只读数组，但原代码会修改其 Z 分量。 | 复制成可写数组，避免初始化物体时因依赖版本直接报错；几何数值不变。 |
| `util/bodex_util.py` | BODex 的 squeeze 关节由抓取姿态外推后可能超过 URDF 关节限位。 | 在生成候选时按同一套 URDF 限位裁剪 squeeze，并记录哪些候选被裁剪；不再把非法手指目标送入物理仿真。 |
| `util/util.py` | 尝试不同候选时原代码只恢复机器人位置，不恢复速度和 drive target。上一候选的残余动态状态可能污染下一候选。 | 每次尝试恢复 qpos、qvel、drive target、物体速度，并清理 cuRobo world cache，使候选从相同状态开始。 |
| `util/util.py` | 运行时对抓取候选的手指目标减小/增加角度后没有再次检查限位。 | 对原始手目标、放松后的 pregrasp/grasp 和 squeeze 偏移统一做限位检查。 |
| `diagnostics/audit_deximit_loaded_hold.py` | 保持审计只检查“拇指 + 中指/无名指”，漏掉合法的食指对向接触。 | 改为“拇指 + 任意其他命名手指”，与项目通用抓取契约一致。 |

这些修复没有把掌部接触当作手指抓取，也没有放宽 SAPIEN 的 20 mm 物体误差、MuJoCo 的 2 mm 穿透或连续接触门槛。

### 当前没有确认的剩余阻断 bug

修复后橡皮 candidate 21 的 3 个强摩擦 profile 仍通过，说明候选状态恢复和食指判定修复没有破坏既有成功证据。契约测试 `40 passed`，全量测试 `326 passed`。

刷子、两段铲刀和锤子均在 SAPIEN 的前 120 个候选中为 `0/120`，所以它们没有进入 MuJoCo。这个事实排除了“MuJoCo 保持审计把它们判失败”这一解释；当前证据更支持抓姿/路径/任务运动本身不足，而不是 MuJoCo 隐藏 bug。

## 3. 从 BODex 到 SAPIEN：哪些仍来自 DexImit

### 上游流程

官方 `pipeline/gen_traj.py` 的单手抓取流程是：

1. 根据当前子动作选择一个 `grasp_depth` 和手指数。
2. 用 `GraspSynthesizer.synthesize_grasp` 生成 BODex 候选。
3. 用 cuRobo 批量做 IK 初筛。
4. 按真人手提示的姿态距离排序，最多尝试 120 个候选。
5. 对每个候选执行 `pregrasp -> grasp -> squeeze -> lift`。
6. 手指闭合，保持 20 个仿真步，再用物体顶点平均运动误差判定成功。

对应代码是 `pipeline/gen_traj.py:342-447` 和 `third_party/any2dex/any2dex/util/util.py:415-687`。

### 当前诊断链复用的部分

`diagnostics/run_deximit_sapien_screen.py:1543-1562` 直接调用隔离副本中的 `rollout_and_select_grasp`，没有重写 SAPIEN 的抓取成功规则。以下行为仍是 DexImit 逻辑：

- 右手 BODex 姿态和 SAPIEN 关节顺序的映射。
- pregrasp 沿手坐标方向后移 10 cm，并复制 grasp 姿态。
- pregrasp/grasp 手指放松 0.2 rad。
- squeeze 保持候选给出的闭合关节。
- cuRobo 的四段规划和 SAPIEN `step_headless` 物理执行。
- 物体平均顶点运动误差不超过 0.02 m 才算 SAPIEN pass。

当前诊断额外记录了每个物理步的 qpos、qvel、drive target、关节力、接触点、法向/切向冲量和 link 级接触，但这些记录不替换原始 pass 条件。

## 4. 当前 SAPIEN 与上游提交的工程差异

除了第 2 节的 bug 修复，当前诊断 runner 还有以下有意的诊断隔离：

- 不生成正式 renderer 或论文 3.3 数据，只保存诊断报告。
- 为了在本机无光线追踪降噪器时继续筛选，可使用 `--render-mode raster`；这只改变图像初始化后端，不改变 SAPIEN 物理参数、相机位置、候选和控制。
- 禁止构造未使用的 SyntheticPC 辅助场景，避免其覆盖 Front 相机的渲染器；筛选和轨迹导出不依赖这些图像场景。
- 新候选池记录 BODex 生成时的物体姿态、网格哈希、DexImit 工作区哈希。严格模式要求它们与本次 SAPIEN warmup 后的状态一致，防止把单位姿态生成的非对称工具抓姿悄悄放到旋转后的物体上。
- 对四个泛化样本使用了诊断用的人工阶段标签时，阶段标签文件会被哈希绑定；这不是改动 DexImit 的执行器，而是为没有正式任务标签的样本补充可审计输入。

还有一个需要明确的实验范围差异：上游 `gen_traj.py` 按当前子动作的一个深度运行；当前四个泛化样本的 `sapien_combined_top120_v1` 默认把深度 0、1、2、3 的有效候选合并后，再按真人姿态误差取全局前 120 个。代码和报告把它标记为“combined four-depth diagnostic ranking”。若要与某一个官方子动作深度逐候选对齐，应使用 `--pool-depth 0/1/2/3` 分别运行；现有 combined 结果不能称为某个单独深度的逐位复现。

## 5. SAPIEN 与 MuJoCo 的物理差异

| 项目 | DexImit/SAPIEN | 当前 MuJoCo 诊断 |
|---|---|---|
| 机器人模型 | 直接加载 DexImit 的 XHand+UR5e URDF。 | 从同一份 URDF 转换；关节树、轴、限位和 link 惯性尽量保留。 |
| 机器人重力 | 每个 SAPIEN 机器人 link `disable_gravity=True`。 | 每个机器人 body 使用 `gravcomp=1`。 |
| 物理步 | `1/240 s`；一个控制目标执行 12 个物理步。 | 源时间仍为 `1/240 s`；每个源步用 25 个 `1/240/25 s` 内部步模拟 TGS 位置迭代。 |
| 驱动器 | PhysX TGS force drive，刚度 1000、阻尼 100、力限 `1e10`。 | 禁用原生 MuJoCo actuator，`sapien_equiv.py` 按 PhysX 隐式 drive 公式和耦合逆质量近似施加冲量，并在每个源步固定 drive target。 |
| 物体碰撞 | `add_convex_collision_from_file`，由 PhysX cooking 生成凸碰撞体。 | 使用从 SAPIEN 导出的 PhysX cooked convex 网格作为显式 MuJoCo collision mesh；原始高分辨率网格只用于可视化。 |
| 物体视觉 | 原始物体网格。 | 使用同一几何的 MuJoCo-readable 网格作为 visual，碰撞和视觉分开。 |
| 物体质量 | SAPIEN `init_object_flexible_with_pose` 根据网格体积、density 和 `min` 规则设置质量，并设置线/角阻尼 20。 | 从 SAPIEN 运行记录读取质量、质心和惯性；无接触速度衰减按 SAPIEN 记录校准。 |
| 桌面 | 半尺寸 `[0.6, 0.8, 0.03]`，顶面 Z=`0.714`，静/动摩擦均为 1。 | 同尺寸和顶面；表面与物体的有效摩擦使用 PhysX 默认材质组合后的校准值。 |
| 接触 | PhysX PCM、TGS、25 次位置迭代、每次迭代更新摩擦，shape contact offset `0.02`、rest offset `0`。 | MuJoCo pair contact 显式设置接触距离、`solref/solimp`、摩擦锥和 noslip；参数来自候选无关的接触校准。 |
| 静/动摩擦 | PhysX 原生接触补丁会维护接触历史。 | 普通模式按切向速度在静/动系数间切换；强摩擦模式额外使用每个手物凸片的一条保留切向 anchor，作为诊断近似。 |
| 成功定义 | 原始 DexImit 的物体平均顶点运动误差。 | 先保留 SAPIEN pass，再检查最大穿透、抓取后对向手指接触、保持段连续性和 MuJoCo/SAPIEN 轨迹误差。 |

## 6. 哪些是“等价复现”，哪些是新增方案

### 近似保持不变的部分

- BODex 候选的来源和 raw seed 语义。
- SAPIEN 中实际执行的 DexImit 抓取契约。
- XHand/UR5e 关节名称、顺序映射和限位。
- 物体初始姿态、质量、质心、惯性和 SAPIEN cooked convex 碰撞几何。
- `1/240 s` 源时间尺度、25 次 PhysX TGS 位置迭代对应的 MuJoCo 内部时间划分。

### 不是 DexImit-Open 原功能的部分

- `build_exact_deximit_mujoco_scene.py`、`replay_exact_deximit_mujoco.py` 和 `sapien_equiv.py` 是当前项目新增的跨引擎诊断代码。
- PhysX TGS drive 的 Python 冲量转换是根据 PhysX 行为和公开实现线索写的等价器，不是 DexImit-Open 中的 MuJoCo 实现。
- MuJoCo 的 `solref/solimp` 接触响应校准、速度相关摩擦切换和 strong-anchor 摩擦历史都是跨引擎诊断方案，不是论文中已经提供的抓取优化模块。
- “连续多指对向接触”“最大 2 mm 穿透”“保持段轨迹一致性”等是当前项目为了防止把掌部支撑或短暂碰撞误判成抓稳而增加的审计门槛，不是原版 SAPIEN 的 pass 条件。

这些新增检查默认只产生 `diagnostic_only=true` 的证据，`formal_renderer_3_3_eligible=false`；它们不能反过来修改或替代 DexImit 的正式生成链。

## 7. 对四个失败样本的结论

刷子抓取、铲刀切、铲刀撇取和锤子样本的共同结果是 SAPIEN 前 120 个候选 `0/120`。当前证据显示：

- 有些候选能完成规划，但没有形成持续、对向的多指承载。
- 有些候选只产生掌部推动或单指接触。
- 有些候选在 pregrasp、grasp 或后续任务运动阶段规划失败。
- 这些样本在 SAPIEN 阶段就被拒绝，因此 MuJoCo 的接触近似不是它们当前失败的直接原因。

细长柄物体确实更容易暴露 BODex 只按抓取瞬间几何/真人姿态排序的局限，但目前不能只凭“细长”下结论。切、撇、敲击还要求抓姿在后续动态运动中产生合适的力矩和接触方向；这是任务条件抓姿选择问题，和纯粹的深度输入错误是不同层次的问题。

## 8. 最终状态

- 已确认的审计逻辑 bug 已修复。
- 既有橡皮成功证据回归通过。
- 四个失败样本没有被参数放宽或 MuJoCo 诊断强行救回。
- 当前 MuJoCo 链是面向 SAPIEN 的隔离诊断，不应描述成 DexImit-Open 官方 MuJoCo 实现。
- 本报告对应的测试结果：

```text
spider/.venv/bin/python -m pytest -q tests/test_bodex_diagnostic_contract.py
40 passed

spider/.venv/bin/python -m pytest -q
326 passed
```
