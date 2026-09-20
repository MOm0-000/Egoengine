# Collision Coverage: Bug Review, Findings, Then Proposed Work

## 误报清理后的当前结论（2026-09-17）

**当前 Pour 首帧仍不合法。清理错误分类不等于修好了物理模型。**
可直接交给外部讨论的自包含摘要见
[当前讨论材料](/data_all/zzx/3.2RL/docs/pour_discussion_handoff.md)。

- 删除“首帧手掌—食指根部发生原始实体穿透”的判断：左右两侧该姿态的原始
  闭合实体交集为 0，属于碰撞壳误报。它仍是碰撞形状缺陷，不能交给求解器当真障碍。
- 删除“第 93、177 行接触漏检”的判断：两处都有运行时接触，只是低于 50 微米
  的报告统计档位。输出代码已移除 `missed_at_50um` / `missed_source_rows`
  这类误导命名，另行统计真正没有生成接触的记录；后者也需要复核，不能直接
  把三角面贴合等同于漏检实体穿透。
- 掌部—食指审计现在按姿态分列“确认实体重叠／确认外壳误报／证据不足”。
  没有原始实体证据的角度不能自动清零；同一部件在其他角度的真实重叠仍保留。
- 首帧原始右/左拇指末节进入当前仿真桌面 **20.79 / 9.65 mm**、左掌与拇指
  原始实体交集 **9.25 mm³** 仍存在。它们不是上面两类误报。

本次没有删除物理碰撞对、缩小原始网格、改变阈值、参考、GT 或初始状态。
原始测量报告保留用于复核；其中旧字段是历史统计，不再作为活动漏检结论。
下面的局部形状对照表只测了中立姿态的一对部件，不能代表整个 Pour 首帧。
此次全项目回归 **522 passed、1 skipped、57 subtests passed**（88.13 s）；
正式场景、细化候选和当前真人/机器人参考的哈希均与清理前一致。没有重跑 GPU
任务训练，已有 GPU 接线测试也不被重新解释为任务成功。

## 三步推进：碰撞规则与真实 Replay→PPO 接线（当前）

本轮保持原始 GT、198 帧参考、桌高、关节限位和正式 PPO 场景不变。

1. **MINK 的相邻关节过滤已修正，但新碰撞对尚未启用。** 以前即使 XML
   显式声明了手掌—食指根部，MINK 仍会因父子关节关系删除它。现增加默认关闭
   的选项，只保留调用方请求且 XML 显式声明的对，不把所有相邻关节一起打开。
   双手重定向、首帧诊断和规则核对已使用该选项；合成关节回归测试通过。
2. **发现启用前必须解决的外壳误报。** 原始 CAD 在中立姿态不相交，实体交集
   为 0，但当前 32 块手掌与单凸壳食指根部的重叠为右 3.895、左 3.868 mm。
   将食指根部单独细分为 32 块后仍为右 3.893、左 3.867 mm，几乎没有改善；
   所测关节角中没有一个外壳无碰撞姿态。不能把这样的约束直接交给 MINK。
   结果见 `index_root_baseline.json`、`index_root_screen.json`，位于
   `runs/taco_pour_collision_repair/`。均仅编译这两个部件对来隔离形状问题，
   不是整个场景的物理验证。
3. **真实 Replay→PPO 接线已测试通过。** GPU 上实际运行了官方 PPO 的训练、
   边界恢复及验证；静态接口测试完成 40 个控制间隔，只提交第 20 个间隔的
   原样快照。该测试不代表 Pour 任务完成。接触计数恢复、旧缓冲区接触误计、
   失效缓冲区方向导致的 NaN 传播也已修正，细节见
   [实现记录](/data_all/zzx/3.2RL/docs/replay_rl_implementation.md)。

进一步的实际对照如下。表内是**中立关节姿态下的外壳误报**，不是原始手掌
实体穿入食指；该姿态的原始实体交集仍为 0。

| 碰撞形状候选 | 右手误报深度 | 左手误报深度 | 判定 |
| --- | ---: | ---: | --- |
| 32 块手掌＋单块食指根部 | 3.895 mm | 3.868 mm | 无法启用该相邻对 |
| 32 块手掌＋32 块食指根部 | 3.893 mm | 3.867 mm | 几乎无变化 |
| 全手掌 128 块＋32 块食指根部 | 生成失败 | 1.996 mm | 仍不合格 |
| 去掉两处已证实为空的关节槽填充＋32 块食指根部 | 2.254 mm | 1.460 mm | 改善，但仍不合格 |

最后一项只在碰撞壳中去除原始 CAD 已证实为空的两个小长方体区域：主槽和
轴孔内的局部空隙。每个区域都先与闭合原始 CAD 做实体交集，交集无三角面才
允许切除；用平面切割保持每个输出块为凸形。原始 CAD 未缩小、未填洞。
这套局部处理是**本地几何修正**，不是 EgoEngine 公开方法。数值仍不达标，
没有导出新的正式场景、重跑正式 MINK 参考或选择 reset。

原始关节在 +0.175 rad 的真实 CAD 干涉仍被这些候选检出，未以删碰撞对或改
关节范围将其隐藏。当前只完成这两个部件对的局部检查，没有为未选中的形状
重复整个 GPU 容量测试。结果见 `index_refined_left_screen.json`、
`index_cavities_screen.json`；局部切割分块分别为右 81、左 89 块。
另用右 30,977、左 31,091 个原始网格顶点及三角面中心复查：原先被碰撞体
覆盖的点，没有新增超过 0.1 微米的遗漏（凸半空间检查）。这支持“没有靠挖掉
真实手掌来改善”的判断，但仍是有限采样，不是全表面数值精度的证明。

全手右掌分解在第 64 块检测到无效几何，未被任何模型加载。已删除其 64 个
不完整 `.obj` 和空目录，保留 `index_assembly_failure.json` 的生成参数和
失败原因。生成器改为先验证全部分块再写目录，防止半套输出被误用。
其余实测候选仍作为形状对照保留，没有新增重复版本的重定向或渲染脚本。

**两种初始化尚未正式比较，也未选择新 reset。** 先把关节槽的假障碍排除，再
用同一模型比较“无碰撞操作前姿态”和“固定物体下的物理修正”。当前阻碍是
上述仍未消除的假障碍，不是等待 PPO 代码或要求 Replay 先成功。前者方法依据是
[Human2Sim2Robot 附录 B](https://arxiv.org/html/2504.12609v1#A2)，后者是
[DexMachina 附录 A.2](https://arxiv.org/html/2505.24853v1#A1.SS2)。这些论文并没有
提供当前 XHand/TACO 模型的具体碰撞分块、初始化时长或正式验收数值，不冒充
EgoEngine 已公开配置。

## 上轮：剩余接触问题分类与 Replay→RL 入口检查

输入：未改动的 `scene_refined.xml`、原始机器人/盘子网格、完整 198 帧参考、
复制保存的双手 URDF，以及原始表面检查结果。执行脚本为
`scripts/audit_taco_remaining_contacts.py`，结果为
`runs/taco_pour_collision_repair/remaining_contacts.json`。
没有修改模型、关节范围、桌高、GT、参考或正式训练配置，也没有推进物理时间。

### 1. 盘边：局部形状近似偏差，不是单位/坐标错误

最大采样向外偏差仍为 **1.615 mm**，最严重位置在盘沿下侧，另有约 1.589 mm
的点在盘沿内壁。最严重凸块是 `left_object_177`，最近原始表面法线主要朝下
（z 分量约 -0.946）；独立立体角检查确认这些点在原始实体外。原始盘子尺寸
为 317.7×184.1×36.8 mm，没有发现本轮误差来自尺度重复换算或左右手坐标混用。
凸分解把局部曲面/凹处填厚是与数据一致的解释，不把请求分解阈值当实测误差。

本次 198 帧共检查 **8773 个活动碰撞对的最近点**，其中 601 个属于含 >1 mm
全局采样偏差的凸块，但所测实际接触最近点的偏差最大为 **0.998 mm**，没有
超过 1 mm 的点。最严重者是第 109 行左食指末节与盘子凸块 117 的接触。
对最严重的 20 点另查其确实落在对应凸块表面，最大偏离仅 4.27e-8 m。
这不是“盘边误差无影响”的证明：每个活动对只取一个最近点，不覆盖所有接触
流形点或未来 RL 姿态；1 mm 只是统计档位，不是论文容许误差。

判断：已有足够依据把它记录为已定位的采样近似误差，不再盲目增加整个盘子的分块数。
可以保留当前候选用于隔离初始化对照，后续围绕真实抓握位置检查接触是否被这
一层外壳提前触发；尚不把近 1 mm 的实际接触偏差说成零或正式验收通过。

### 2. 两条差异：已经检测到接触，不是漏检

| 源行（零起始） | 部件 | 实际接触数 | 最大外壳重叠 | 原始表面采样穿入 |
| --- | --- | ---: | ---: | ---: |
| 93 | 左食指末节—左中指近节 | 3 | 0.03915 mm | 0.00593 mm |
| 177 | 左食指末节—左中指末节 | 4 | 0.02425 mm | 0.01273 mm |

所有需要的凸块对均存在，独立 `mj_geomDistance` 与运行接触距离一致。之前
`missed_at_50um_frame_pairs=2` 的意思仅是“没有达到 0.05 mm 的统计档位”，
并非没有生成物理接触，也不是 MuJoCo 的接触激活阈值。**这两条不再作为漏检
阻挡项。** 第 177 行辅助闭合主部件的实体交集约 0.00374 mm³，不能把体积
误当深度；第 93 行近节 CAD 不闭合，只使用其表面进入另一闭合部件的证据。

### 3. 关节装配：24 对、147 个角度快照，不能一律豁免

每对检查了零角度、首帧角度、关节范围内五个等距角度（重复值合并）。24 个
关节的父子关系、平移、朝向、轴和范围都与复制的源 URDF 一致；没有发现这
部分由移植时的轴符号或关节位置错误造成。结果可分成以下四组：

| 类别 | 对数 | 检查结论 |
| --- | ---: | --- |
| 掌部—拇指根部转轴 | 2 | 接触集中在轴附近，采样穿入不到 0.00005 mm，属于数值精度量级的装配贴合证据。 |
| 掌部—食指根部侧摆连接件 | 2 | 零角度和首帧没有交集，部分允许角度出现明显实体相交，必须与轴承贴合区别处理。 |
| 食指根部—近节，以及四指近节—末节 | 10 | 主要是轴孔附近薄层交叉，常见约 0.02–0.055 mm；部分极限角度向外延伸，不能据此认证所有姿态安全。 |
| 掌部—中/无名/小指近节，以及拇指的两级连接 | 10 | 原始 CAD 在零角度就存在约 0.6–2.5 mm 局部穿入，部分极限角更大；属于源装配模型重叠/运动干涉待区分，不是新增碰撞外壳造成。 |

最明确的真实干涉是第二组：侧摆到 **+0.175 rad** 时，两侧掌部与食指连接件
的实体交集约 **196.1 mm³**，采样穿入最大 **2.972 mm**。交叉点可离转轴
约 **34.6 mm**，不只在轴承接合处。当前真实参考的右手第 **45** 行、左手第
**13** 行已经到达这一角度；原始表面扫描分别在 **52 / 130 帧**发现交叉。
因此这不是仅在未来极端姿态下才可能出现的问题，也不是放大碰撞壳造成的误报。

[MuJoCo 官方碰撞筛选说明](https://mujoco.readthedocs.io/en/stable/computation/index.html#selection)
解释了默认忽略父子体是为了避免关节内部持续接触；显式碰撞对可以绕过该筛选。
这支持“不能把所有相邻接触直接当动作失败”，却不支持“所有父子体相交都合法”。
这里应优先处理已证实的两对远离转轴的干涉；其余装配关系要有明确的局部豁免
说明，不能为了检查完整而把所有相邻凸包直接设成互撞，也不能继续笼统忽略。
对于没有闭合实体的 CAD，已记录表面/包含证据和限制，没有偷偷补洞后声称认证。

新测试发现 Trimesh 的 `split` 默认可能补小孔，当前和原手指辅助实体审计均已
显式设 `repair=False`。两根真实末节的 7908 个主部件三角面与改前逐项完全一致，
此前这两根手指的数值没有因此改变；这是通用检查逻辑的防错修正，不是发现
原始 Pour 数据被补过洞。没有新增平行旧脚本或错误报告。

### 什么时候能进入 Replay→RL

**不要求所有误差归零，也不要求 Replay 已经成功抓住物体。** 论文 3.2.2/C.1
本来就用 RL 修正 Replay 的接触与跟踪失败。真正的入口条件和当前缺口是：

1. **冻结同一个可解释的碰撞模型。** 把两对已证实的掌部—食指干涉纳入局部
   碰撞处理，明确装配豁免边界，并同步 MINK/物理引擎的规则。当前参考仍来自
   旧碰撞模型，不能用它在旧模型上通过的结果替代新模型的自碰撞可行性检查。
   盘边保留已测误差及接触检查，不无止境追求零近似误差。
2. **得到一个合法、可重复的初始状态。** 按已批准的两种初始化候选做比较，
   固定 qpos、qvel、首条控制和参考起点，检查释放两个被动物体后没有初始化
   冲击/异常跳变。已知首帧手/桌穿透并未被本轮修正。
3. **接上真实两块调度与 PPO。** 代码审查发现 `solve_chunk` 目前只有模拟后端
   的测试调用；`run_mjwp_ppo.py` 是独立 PPO smoke 入口，并不调用它。真实
   MJWP 环境虽有保存/恢复接口，但还需直接接通“失败后从同一块边界状态训练、
   验证 40 步、只提交 20 步”。训练环境也不能偷偷重置回参考首帧。
   运行时同时采用已验证容量，并显式记录临时奖励系数及阈值到 C 的换算是
   本地配置，不能冒充未公开的论文数值。

前两项通过，即可开始**首个最多 40 控制步的诊断 Replay**，不需要先完成整段
轨迹。第三项接通并通过真实状态回退测试后，就能在该窗口的 Replay 失败处
进入 PPO。完整轨迹成功率、稳定抓握和 RL 提升是进入后要测的结果，不应再
反过来作为入口门槛。现在没有足够依据保证只剩几小时或一个固定训练时长。

此前“已经集成”的描述已在 `docs/replay_rl_implementation.md` 中更正为
“各部件已实现/测试，端到端切换尚未连接”；本轮没有以 smoke 测试冒充完整
流程。本轮全套测试 **510 passed、1 skipped、57 subtests passed**（112.37 s），
最终针对本轮检查和两块调度的测试另为 **18 passed**。核验本轮报告的 674 项
输入/依赖和脚本哈希。测试通过不改变 `training_ready=false`。

## 已完成的形状细化与实际 GPU 容量检查

输入：上一轮 `scene_budgeted_palms.xml`、原始 CAD/物体模型，以及现行 Pour
198 帧 `robot_reference.npz`。原始模型、参考、桌高、质量、关节和控制器不变。
输出为未正式采用的 `scene_refined.xml`；具体替换部件记录于 `refinements.json`。

### 形状策略和已确认测量

- 碗复用已有的 288 块候选，不重复生成；其采样最大向外误差为 0.785 mm。
- 盘子用 CoACD 官方 `real_metric=True`，请求 0.0005 m，并测试无块数上限和
  256 块限制。前者产生 1121 块，编译后采样最大偏差仍为 1.481 mm；后者为
  256 块，原始凸块采样最大偏差约 1.615 mm。请求阈值不能写成实测最大误差。
  这些是本地数值实验，不是 EgoEngine 公布的碰撞参数。
- 左食指/中指末节各分成 16 块，其他手部形状沿用上一轮。两份末节 CAD 的
  7910 个三角形包含一个 7908 面的闭合主部件和两个反向重复的内部三角形；
  原网格并不是简单的外壳缺洞。没有删除原始面或修改视觉模型，CoACD 在候选
  生成时使用自动预处理。检测对原始全部顶点/面中心查漏覆盖，并另用闭合
  主部件作辅助距离对照；不能将它冒充完整 CAD 的严格实体认证。
- 预处理后的凸块曾伸出原始凸包最多 0.278 mm，使额外手内碰撞记录反而增加。
  随后用 [Trimesh/Manifold 实体交集](https://trimesh.org/trimesh.boolean.html)
  将各块限制在原始凸包以内，不修改或整体缩小原始网格。所有结果顶点的
  半空间检查最大越界小于 2e-9 m。这是本地修正，不是论文指定步骤。
  修正前仅保留 `preclip_screen.json` 说明失败原因；场景和完整报告在原位置更新。
- 完整场景编译得到 701 个几何体、143889 对显式碰撞。编译成本明显增加，
  审计代码已避免同一进程重复编译同一模型。没有通过删除碰撞对降低成本。

无上限盘子相比 256 块只再降低约 0.133 mm 的采样最大偏差，未安装进物理场景。
其 **1121 个 OBJ 和 1 份分解清单已删除**；来源、参数、测量和重建方法保留在
`uncapped_metric_screen.json`。可以重新生成，原始数据未删除。

### 最终候选与上一轮对照

以下均来自当前 `refinement_comparison.json`，不是修正前的中间结果。
手部统计覆盖同一份参考的 198 帧、301 对非相邻原始部件。

| 项目 | 上一轮分块掌候选 | 本轮候选 |
| --- | ---: | ---: |
| 碗外壳采样最大向外误差 | 1.148 mm | 0.785 mm |
| 盘外壳采样最大向外误差 | 2.862 mm | 1.615 mm |
| 原始三角表面未交叉、外壳却重叠的“帧×部件对” | 72 | 58 |
| 上述额外记录的最大外壳重叠 | 1.826 mm | 0.716 mm |
| 原始表面交叉、外壳重叠未达 50 µm 的“帧×部件对” | 2 | 2 |
| 左食指/中指末节采样最大向外误差，辅助闭合主部件对照 | 5.220 / 5.220 mm | 1.817 / 1.817 mm |
| 左食指/中指末节原始表面采样最大欠覆盖 | 0 / 0 mm | 约 0.0054 / 0.0054 mm |

盘与上一轮已有的 135 块精细候选相比，最大向外误差为 **2.089→1.615 mm**；
不是把上一轮最好结果隐去。碗的 0.785 mm 候选是复用已有结果。
两根末节各检查 11857 个原始顶点/面中心，均未发现欠覆盖超过 50 µm 的点。
并非严格零欠覆盖，不能把有限采样结果说成所有接触位置都已证明准确。
当前碗/盘各 4096 个原始表面采样点最大欠覆盖分别为 0.0059 / 0.0092 mm。
所有这些都是采样值，不是全局误差上界；统计档位不是论文验收标准。

**本轮又修正了一项测量误差。** 原来的 `Trimesh.closest_point` 在米单位的
细小三角面附近，把一处约 2.8e-9 m 的距离算成了 0.176 mm；将同一几何放大
到毫米后重算、Open3D 无符号距离、独立双精度凸体投影三者一致，支持前者为
数值误报。因此撤回中间结果的手指最大欠覆盖 0.176 / 0.174 mm 和 113 点超
50 µm 的说法，盘面中间结果 0.105 mm 也不再采用。
现已用 Open3D 无符号距离替换当前审计中的这两处最近表面查询，并加入真实
反例、独立投影、开放三角面、尺度变化、批次顺序和非法输入测试。修正只影响
测量，没有改动模型或接触求解，GPU 容量结果仍对应同一场景哈希。原报告直接
在原位置重算，不保留并行错误版本。历史报告中使用旧最近表面查询的欠覆盖
细项没有逐个重算，不能声称它们都通过了此次修正；有符号向外误差函数未改动。

尚未解释的两条记录仍是左食指末节—左中指近节第 93 行、左食指末节—左中指
末节第 177 行（均为零起始源行）。不能把“未达 50 µm”直接称作没有接触。
剩余额外重叠最大值出现在左食指近节—左中指末节，而非两根末节之间。
无原始表面交叉仍不能排除实体完整包含，相邻关节装配处也尚未完成认证。

### 容量不能只看最终接触数

[MuJoCo Warp 官方说明](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#batch-sizes)
区分跨环境共享的接触空间和每个环境单独的约束空间。在项目实际使用的
MuJoCo 3.7.0 / mujoco-warp 3.7.0.1 / Warp 1.12.1 实现中，还要检查
`ncollision`：它是粗筛阶段的候选对数量，也占用接触缓冲区；溢出时会漏算，
即使最终 `nacon` 没超限也不能判通过。

上一轮模型的真实负对照：4 个相同环境、每环境 128 / 每环境约束 512，在第
44 帧出现 832 个候选对，总容量只有 512。报告立即判失败，不采用被截断后的
接触数。扩为 512 / 2048 后，198 帧同帧同步测试无溢出：候选对最大总数848，
接触最大总数460，单环境约束最大460。结果在 `capacity_budgeted_original.json`
和 `capacity_budgeted_roomy.json`，仅代表原 XML 选项下的静态检查。

新模型测试使用 SPIDER 现有 `setup_mj_model` 原函数读取 PPO 配置，保持其
时间步长/积分器/求解器设置，只有本次候选模型路径和测试容量另行指定。
分别检查 4 / 16 环境中的全部 198 帧，并从首帧及接触、候选对、约束峰值帧
进行短时固定控制压力测试。这个测试不修正初始状态，不证明已经抓住物体，
也不是成功 Replay 或 PPO 训练。结果文件为 `capacity_refined.json` 和
`capacity_refined_16env.json`。

最终两组都完成全部 198 帧；压力测试起点为源行 0 和 111，各推进 30 步，
步长 1/300 s，即每段 0.1 s。控制量固定为起点参考值，不追踪整段参考。
测试容量为 `nconmax=1024`（按环境数形成共享总池）、`njmax=2048`（每环境）。

| 测试 | 接触池总容量 | 最大候选碰撞对总数 | 最大接触总数 | 单环境最大约束数 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 环境 | 4096 | 1744 | 744 | 745 | 无溢出，状态有限 |
| 16 环境 | 16384 | 6976 | 2976 | 745 | 无溢出，状态有限 |

候选对池峰值占 42.6%，约束峰值占 36.4%；这只是已测范围内的余量，不保证
未见姿态或长时间 RL 的容量。两组分别实际推进 240 / 960 个“环境×物理步”。
CPU 3.7 同参考检查最大接触 158、约束 633，无 MuJoCo 警告。所有容量数均读取
未截断计数，负对照证明了不能只查看最终生成的接触数。

CPU 几何审计使用 MuJoCo 3.12.0；GPU 测试环境是 3.7.0。接触生成数存在版本
差异，不能把两个版本的接触数直接混作同一测量。没有升级任何一个环境。
3.12 本轮候选最大接触数为 253；旧候选为 135，同版本对照保存在几何报告。

### 本轮结论和边界

形状误差有实测改善，GPU 容量在上述测试范围内通过，但仍不是无误差或已正式
接受的碰撞模型。新模型有 701 个几何体、143889 对显式碰撞，XML 编译约需数
分钟；CPU 3.12 最终重测单帧 `forward` 中位耗时约 0.71→1.91 ms，是明确的性能代价。
下一步针对残留的盘边误差、手指欠覆盖和两条接触差异做局部核实，避免无上限
加块；通过后才在同一模型上比较两种初始化。本轮未修改 GT、桌高、参考、正式
PPO 配置或 reset，`training_ready=false`。

纯形状审计已改用 `mj_kinematics`，不再对默认零姿态执行接触/约束求解，避免
与测量无关的 arena 内存警告；真实参考姿态的接触容量另由上面的完整流程检查。
最终全套回归 **504 passed、1 skipped、57 subtests passed**（92.37 s，7 条原有
SciPy 弃用警告）。隔离 CPU 环境没有 mujoco-warp 导致一处模块跳过；实际 GPU
环境补跑对应适配测试 **5 passed**，并完成上述真实容量测试。另核验当前模型、
几何对照及两组容量报告关联的 **999 项唯一现存输入/依赖哈希**；历史生成代码
哈希明确保留为历史记录，不冒充当前代码。测试通过不等于物理初始化通过。

## 上一轮：碰撞模型改进实测与距离误判修正（2026-09-17）

**当前进度：碰撞模型候选比较完成，尚未比较两种初始化，也没有执行 Replay/PPO。**
所有候选位于 `runs/taco_pour_collision_repair/`，没有替换正式场景，没有改原始
物体 GT、198 帧参考、桌面、关节、质量或控制器。它们仍是未接受的实验候选。

### 先纠正一次真实的测量错误

原距离检测会把个别碗内的实体点判到物体外。在无块数上限、关闭预处理、更细
分解三组实验中，先前的碗最大误差 `5.665 / 5.932 / 5.513 mm` 均由这种误判
产生，现撤回。不是改变了模型才使这些值消失。

已将六个审计脚本的原始网格有符号距离改为 Open3D 11 射线判断，保持项目原有
“实体内为正”的符号，并对每组最严重的五个点用双精度三角形立体角求和独立
复核。真实反例、单点/批量/逆序一致性、解析立方体、坐标变换、开放网格拒绝
均加入 `tests/test_mesh_distance.py`。不为开放机器人 CAD 强行补洞或声称有
可靠的内部。此方法也不构成任意自交网格的严格证明。

[Trimesh 源码](https://github.com/mikedh/trimesh/blob/main/trimesh/ray/ray_util.py)
可见歧义射线的随机重试；本环境真实查询中观测到相同点的符号变化。
[Open3D 官方说明](https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html)
要求闭合网格，并建议用多条射线降低边/顶点交点歧义。这些是数值检测修正，
不是 EgoEngine 奖励参数或新的任务成功阈值。

在原文件位置重新生成了 `comparison.json`、`object_parameter_screen.json`，
更新了现行 preflight 的物体采样部分，以及现行参考的初始接触/初始化报告。
没有保留并行的错误版本。历史报告仅保留其历史/回归价值，其中依赖旧有符号
距离的数值不能直接视为经过此次修正验证；历史的“33 组重算一致”也不是对
检测方法正确性的证明。生产侧未启用的旧接触投影仍含 `contains`，此次未将它们
混入当前 MINK 链路；当前 Pour 参考的 `surface_projection=false`。

未被推翻的结果：原碰撞碗/盘的采样最大向外偏差仍约 `5.677 / 5.001 mm`；
手进入当前桌面 `20.79 / 9.65 mm` 来自直接顶点高度，不依赖物体内外判断。
当前首帧左无名指/小指进入盘子实体的采样最大深度重算为 `1.493 / 1.768 mm`，
内部样本数仍为 `11 / 21`；此处修正前后几乎无变化。左掌/拇指 `9.25 mm³`
来自另一种闭合实体布尔计算，也不依赖这次修正的距离函数。

### 物体模型：分块改进有效，但还不是无误差模型

| 碰撞分解 | 碗/盘块数 | 碗采样最大向外偏差 | 盘采样最大向外偏差 |
| --- | ---: | ---: | ---: |
| 原模型，最多 32 块 | 32 / 32 | 5.677 mm | 5.001 mm |
| 同参数，取消块数上限 | 164 / 86 | 1.148 mm | 2.862 mm |
| 再关闭预处理 | 165 / 85 | 1.148 mm | 2.862 mm |
| 更细分解，归一化阈值 .02→.01 | 288 / 135 | 0.785 mm | 2.089 mm |

这是各组全部编译凸块顶点和面中心的采样最大值，**不是全局误差上界**。
另外，每个对照在相同的 8192 个平衡采样点上比较，取消块数上限后，碗多占
空间超过 1 mm 的点数 `35→4`、盘 `318→168`。不同参数对照的采样点集合不同，
不能直接跨对照比较计数。原始表面的 4096 点检查未发现超过 1 mm 的漏覆盖，
不等于整张表面已经被证明覆盖。1 mm/50 µm 都只是统计档位。

取消块数上限有依据：[CoACD 官方文档](https://github.com/SarahWeiii/CoACD)
说明强行合并至指定块数可能牺牲近似精度。这里的 `.02/.01` 是 CoACD 的归一化
参数，不是米，也不是论文公布的碰撞误差。版本固定 `coacd==1.0.12`。

### 手部模型：补齐漏检，不能简单把整只手改成凸包

- 六个闭合部件的局部修正补上了双侧掌/拇指碰撞及食指根部。全 198 帧，掌/拇指
  原始表面记录右 44 帧、左 23 帧，与修正后的 >50 µm 碰撞记录一致；其余四类
  已知手指漏检在这个局部候选里仍未修复。
- 全部 26 个原始手部网格分别用单个凸包，虽能检查所有 301 对非相邻手部件，
  却会填满手掌凹处：出现约 `13.5 mm` 的掌/食指碰撞壳重叠，原始三角表面未
  检出对应交叉。这个简单替换方案不能直接采用。
- 只将每个手掌分成 32 块，其余指节仍单凸包，额外的“部件对×帧”记录从
  `433→72`，最大额外重叠从 `13.50→1.826 mm`。两种候选均有 2 个原始表面
  交叉记录未达到 50 µm 的壳重叠档位；不能据此自动断言漏检，也不能不查原因。
- FCL 表面交叉不是实体穿透深度；无交叉也不能排除完整包含。因此“额外记录”
  不全部等同已证明的误报。相邻装配处和开放 CAD 的剩余误差还需要解释。

结果：`hand_geometry_screen.json`。分块掌候选的全部帧最大接触数 **135**，
超过现行 PPO 的 `nconmax_per_env=128`；在采用新模型前必须重测 GPU 接触/约束
容量，不可原样套用旧容量。当前仅执行 CPU `mj_forward`，没有物理时间推进。

### 清理与下一步

已经删除无块数上限的六个手部件分解产生的 **1807 个生成文件**：仅手间和
掌/拇指就超过 110 万对组合，不适合继续安装到场景。保留了小型
`hand_decomposition_cost_screen.json`，包含来源、参数、块数和重建方法；原始
模型未删除。当前各碰撞候选仍承担独立对照，暂不把它们当重复旧版本删除。

下一步收敛盘边缘和指节的局部碰撞形状，核实剩余记录及装配处，检查容量。
然后在**同一候选模型**上比较下文两种初始化；先做隔离诊断，不自动升级成
跨任务正式方案。不能因为改进明显就把 `training_ready` 改成 true。

本轮最终整套测试：**494 passed、1 skipped、57 subtests passed**，7 条原有
SciPy 弃用警告，138.27 s。另核验 987 个现存候选凸块文件及源输入哈希。
测试通过不是场景物理可行性通过。

## Previous checkpoint: corrected MANO reference, before the collision candidates

The rest of this document is the historical orientation-bug baseline/v4 review.
Its 204.29 mm³ thumb intersection is **not** the current first-frame value.
The current input is `runs/taco_pour_bimanual_mano_fk_right_guard_v1`; both the protocol and
PPO config now point to it. The old reference is retained solely for the
orientation-fix comparison and regression tests, not as an active training input.

New script: `scripts/audit_taco_initialization_preflight.py`.
Current results: `runs/taco_pour_initialization_preflight_right_guard_v1/report.json` and
`native_checks.npz`. No new renderer, scene or reference was created. No
simulation stepping, initialization-candidate comparison or PPO was performed.

### 坐标与碰撞检查实测结果

- **统一坐标变换的导出计算一致。** 手部坐标、物体变换与原始 GT 的重新计算一致；
  相对坐标不变量误差小于 `5e-16`。物体四元数分量差异最大 `1.78e-8`，来自
  float32 旋转与单位四元数转换的数值差异；平移分量没有差异。原始视觉网格与
  MuJoCo 编译后顶点集合的最大距离为 `1.43e-8 m`。这些结果检查计算链，不是对
  深度/RGB 原始时空标定的认证。
- **当前初始姿态仍进入桌面。** 原始机器人网格的最低点，右手低 `20.79 mm`、
  左手低 `9.65 mm`，均在拇指末节；对应碰撞外壳最严重为 `21.91 mm`。
  这里是相对于当前水平无限桌面模型的测量，不能拿来证明真实桌面标定已经正确。
- **手内漏检有实际证据。** 用 FCL 检查所有 28 个原始网格之间的 378 对组合、
  全部 198 帧，包含没有独立碰撞体的食指根部。7 对非相邻手部件存在表面接触/
  交叉记录，但没有对应的物理碰撞对。布尔实体计算另外确认左手首帧掌部与拇指
  根部有 `9.25 mm³` 交集。体积单位不是穿透深度，不能写成 `9.25 mm`。
- **关节装配处不能误报成动作失败。** 首帧 23 对手内表面记录中，22 对是相邻
  部件，在手指零角度和全部 198 帧都存在记录。它们先归为“装配关系待解释”，
  既不自动判非法，也不据此自动加入碰撞豁免。FCL 布尔检测也不能区分接触和
  穿透深度，更不能排除一个封闭物体被完整包在另一个内部。
- **物体碰撞形状确有毫米级向外偏大证据。** 在实际编译的各凸块顶点和三角面
  中心采样，碗 `47,374` 点、盘 `13,150` 点，测到相对于闭合原始物体网格的
  最大向外距离分别为 `5.677 / 5.001 mm`。本轮修正符号后，碗有 `461` 个点、盘有 `733` 个点
  超出 `1 mm`。这是采样找到的误差，不是全局最大误差；1 mm 只是统计档位，
  不是新增成功阈值。原 preflight 未查漏覆盖，本轮有限采样结果见上方。

遗漏且出现表面记录的非相邻手部件：

| 部件对 | 有记录的帧数 / 198 |
| --- | ---: |
| 右掌—右拇指根部 | 44 |
| 右食指近节—右中指末节 | 3 |
| 左掌—左拇指根部 | 23 |
| 左拇指根部—左食指根部 | 4 |
| 左食指近节—左中指近节 | 76 |
| 左食指近节—左中指末节 | 12 |
| 左食指末节—左中指近节 | 80 |

### 接下来的顺序

当前检查没有通过，不能先用存在漏检的物理模型给初始化候选排名：碰撞没有被
模型发现时，物理修正也不会自动修好它。下一步优先处理有证据的掌部/拇指和
食指根部区域，并核实上表其余部件；同时检查碗内壁、盘面、边缘的碰撞块多占/
少占空间。不能通过一律缩小外壳、改桌高或专门限制某个样本的关节角来宣称修复。

模型检查通过后，保持同一个模型、原始物体 GT 和全部源帧，比较：

1. **无碰撞操作前姿态**：借鉴 [Human2Sim2Robot 附录 B](https://arxiv.org/html/2504.12609v1#A2)，
   让机器人从物体附近、没有相交的姿态开始。它不等于“已经抓住物体”；还需检查
   接上参考动作时是否突然跳动。本项目不照搬其手臂/手型偏移，也不因此删掉源帧。
2. **物理修正重定向**：借鉴 [DexMachina 附录 A.2](https://arxiv.org/html/2505.24853v1#A1.SS2)，
   在离线修正时固定物体，将原重定向姿态作为柔性控制目标，记录仿真实际达到的
   手姿态，而不是把目标值直接当成已达到的状态。固定物体只限预处理；后续比较
   和 Replay→RL 都必须恢复两个物体的被动动力学。这是 MuJoCo 上的方法借鉴，
   不是直接复用 Genesis 代码或恢复 EgoEngine 未公开的 reset。

比较至少报告：原始网格/碰撞体的相交、指尖位置与朝向、腕部误差、接上动作时
的跳变、接触力、释放物体后的位移，以及 Replay 的实际失败位置。只有通过的
候选才能接残差 RL，不能用 PPO 的奖励上涨反推碰撞模型正确。

### 清理

删除了 `runs/taco_pour_raw_depth_table_audit_v1/report.json` 及其空目录：没有引用，
全部字段和值已包含在当前 `v2/report.json` 中，后者还增加了逐帧稳定性检查。
没有删除原始数据、修正前后对照轨迹或现有回归测试所依赖的历史结果。

完整测试：486 passed、1 skipped、57 subtests passed；7 条原有 SciPy 弃用警告。
另已重新核验本次报告中的 100 项源文件/网格哈希，原始输入未变化。测试通过
证明检查代码和既有功能的回归状态，不代表这个场景已经满足物理初始化要求。

## Historical review (orientation-bug reference and temporary v4)

Input: the unchanged Pour two-hand/two-passive-object scene, its full 198-frame
robot reference, the preserved temporary v4 posture, original native robot/object
meshes, the 64 object collision pieces, copied robot URDFs and earlier diagnostic
reports. This is a code review and model-coverage diagnosis, not a physical
rollout, model replacement or formal initialization method.

The user limits first-frame hand-pose adjustments to temporary single-sample
tests. They are not part of the formal Replay -> RL pipeline; no generalization,
automatic new-sample use or default reset is inferred from their feasibility.

## Confirmed Bugs And Fixes

Interpretation correction: the malformed states, 10 mm origin shift and
mismatching hashes below were injected test faults, not observed corruptions
of the real Pour files. A subsequent full historical revalidation recomputes
33 groups of metrics with no discrepancies. See
[post-fix impact and corrections](/data_all/zzx/3.2RL/docs/pour_bug_impact_corrections.md).
The proposal later in this document has not been applied or advanced during
that revalidation.

1. **Invalid states could pass feasibility.** A NaN/Inf object coordinate or
   zero/nonunit free-object quaternion could reach MuJoCo and be reported as
   feasible. In the NaN reproduction, a hand/object distance query returned the
   0.05 m detection cap, giving zero reported violations. NaN distance outputs
   also compare false against a negative-penetration threshold. The audit now
   rejects malformed/nonfinite qpos, nonunit quaternions and nonfinite geometry
   or distances. It rejects invalid numerical tolerances and AABB inputs too;
   no automatic normalization or data repair is performed.
2. **Source-model agreement was a fixed sentence.** Changing a thumb joint
   origin by 10 mm still produced the text saying the URDF and model agree.
   Parent/child structure was not compared either. Agreement is now derived
   from measured origin/axis/limit differences, parent/child/type checks, joint
   local anchors and reference offsets. The local comparison tolerances are
   reported; they are not paper collision or task-success thresholds.
3. **Old native reports could be reused with different meshes.** The saved
   candidate auditor checked XML/NPZ hashes, but ignored the native report's
   external-mesh hashes. It now verifies those dependencies before loading
   geometry, checks that the native manifest covers the current visual inputs,
   and snapshots all 92 scene mesh dependencies for new diagnostics. A native
   report's 28 mesh entries alone do not cover the 64 CoACD collision pieces.
   Legacy reports remain unchanged and are explicitly identified as lacking
   the new complete snapshot. The new review independently verifies current
   collision-piece hashes against their existing provenance.

Ten targeted failure-injection cases failed before the fixes and passed after
them. Additional guards and inventory tests bring the full suite to 107 passing
tests. This is scoped to the collision/initial-state diagnostic code and its
contracts, not a claim that the entire repository or pending PPO adapter is
bug-free.

The preserved real Pour inputs are finite with valid unit quaternions. Rechecking
them leaves the old conclusions unchanged: original row 0 violates the declared
environment geometry; temporary v4 passes the declared checks only. Both source
thumb kinematic comparisons pass the new conditional check. No original asset,
reference, candidate, old report, reward parameter or renderer was modified.

## What Collision Coverage Means Here

Coverage has three separate requirements: relevant physical surfaces must have
collision geometry, relevant body pairs must be checked, and the geometry must
approximate the surfaces closely enough to distinguish contact from penetration.
Agreement between MINK and runtime pair lists establishes consistency of those
lists, not completeness of the modeled hand.

### Existing Pairs

All runtime geom masks are zero, so the explicit pair table is the mechanism
enabling these contacts. There are 2,822 declared physical pairs:

| Family | Count | What that establishes |
| --- | --- | --- |
| Same-hand shells | 30 | Selected self-contact pairs, not complete self coverage |
| Cross-hand shells | 144 | All 12-by-12 existing shell combinations |
| Hands/bowl | 768 | All 24-by-32 existing shell/piece combinations |
| Hands/plate | 768 | All 24-by-32 existing shell/piece combinations |
| Bowl/plate | 1,024 | All existing 32-by-32 piece combinations |
| Hands/table | 24 | All existing hand shells against the table |
| Objects/table | 64 | All existing object pieces against the table |

Each hand has 12 collision shells, yielding 132 possible same-hand shell pairs
across both hands. Only 30 are declared; 102 are omitted. Of those 102, 82 are
nonadjacent and 20 adjacent under the current body topology. The latter are
assembly candidates requiring explanation, not automatic collision exemptions.

### Proven Omission Versus Shell False Positives

The omitted left palm/proximal-thumb pair has a full closed-native-mesh
intersection of 204.288967 mm^3 at original row 0 and temporary v3. The joint
parameters match the copied source, and the zero joint pose is clear. This is
positive evidence of a pose-dependent native intersection which the existing
declared self-collision checks do not constrain.

Temporary v4 removes this particular Boolean intersection, but only by changing
one first-frame angle. It does not repair runtime coverage: a subsequent command
can bring the same omitted bodies back together without that pair generating a
constraint. Therefore the v4 result cannot substitute for a collision-model fix.

Conversely, 17 omitted nonadjacent shell pairs still penetrate in v4. Both
palm/proximal-thumb native pairs are clear at Boolean numerical precision, yet
their broad shell distances remain -8.601 mm. Enabling all omitted old shells
would therefore impose some unsupported constraints. The other 15 overlaps are
not classified as either genuine or false solely from absent sampled containment.

### Bodies Outside The Shell Inventory

The scene has 26 native hand meshes but only 24 hand shells. The two unrepresented
same-body meshes are `right_index_bend_visual` and `left_index_bend_visual`, both
on articulated index-root bodies. They were absent from a pair enumeration based
only on `collision_hand_*` names. The independent native hand/object AABB check
did include them, but that diagnostic does not add them to physical collision
handling.

This means 156 same-hand native-body combinations exist, versus 132 combinations
of existing shells. Only 30 native-body combinations are directly represented
by declared same-hand shell pairs. Similarly, 169 cross-hand native-body
combinations exist, versus 144 directly represented by current shell pairs.
These are inventory counts, not counts of actual collisions. A neighboring shell
may partly cover a root component; its extent has not been certified. An absent
same-body shell alone is not proof of a specific surface gap.

### Native Checks Are Incomplete

Seventeen of the 26 native hand meshes are open, leaving nine closed positive
volumes. Open surfaces do not support the signed-volume containment or Boolean
solid tests used for the closed palm/thumb pair. A few hundred surface samples
cannot exclude narrow crossings, especially when both meshes are open. Thus
"no sampled inside points" is not a no-intersection certificate.

### Object Approximation Remains Uncertified

The bowl and plate each use 32 metric CoACD parts. Both reached the existing
part cap. The earlier console observations reported maximum concavity around
0.07549 and 0.02831, respectively, above requested 0.02. These are algorithmic
concavity diagnostics, not meter-valued penetration depths. Current provenance
explicitly records `requested_threshold_certified=false`.

Complete enumeration of these pieces cannot establish that their union preserves
the bowl interior, rim, plate surface or inter-object gaps. It also does not prove
that the approximations are unusable; those task-relevant surfaces need geometric
comparison. Shape accuracy and pair coverage are separate questions.

## Paper-First Basis For The Proposal

The supplied PDF was searched again for collision, convex decomposition, pair
settings and contact-model information, then the relevant sections were reread.

| Evidence | Requirement or limitation |
| --- | --- |
| Section 3.2.1, Eq. (1), PDF pp.3-4 | MINK fingertip/wrist fitting must respect joint limits and self collisions. |
| Section 3.2.2, p.4 | Replay/contact discrepancies are expected; object-centric refinement handles execution failures. |
| Appendix A.1/A.3, pp.15-16 | Shared scene alignment, 0.6 m offset and 0.72 m table; no collision-shape or reset recipe. |
| Appendix C.1-C.2, pp.20-22 | Chunked refinement and rewards; no TACO collision-shape/pair implementation. The margin randomization values belong to Aria. |
| Section 6 | Contact-model error is a limitation, not permission to ignore demonstrated rigid-body intersections. |

The paper does not identify collision shapes, a self-pair table, CoACD settings,
per-pair exemptions or an automatic first-frame feasibility projection. The
following implementation choices are consequently local proposals, not recovered
author methods. They do not alter the approved Replay -> RL choice or fill in
the still-unspecified reward coefficients.

## Proposed Work, Not Applied Model Changes

1. **Complete the evidence table before changing pairs.** Start from all 26 native
   hand bodies, not only existing shells. Record whether each has a collision
   representation, each pair's runtime/MINK inclusion, its assembly relation,
   and the actual evidence supporting exclusion. Treat missing geometry,
   omitted pairs, coarse-shell false positives and unknown cases separately.
   The inventory in this review is the first completed part of that table.
2. **Classify geometry using appropriate established tests.** Retain full Boolean
   intersection for valid closed solids. For open meshes, use an established
   triangle-mesh collision/distance library, such as FCL, to detect surface
   crossings independently of solid containment. A surface-distance result on
   an open mesh does not define its material interior. Do not automatically fill
   holes, infer a solid, or treat a negative sample result as an exemption. If
   the physical shape remains ambiguous, report that missing information.
3. **Fix demonstrated coverage defects locally in separate derived assets.**
   Prioritize the palm/proximal-thumb region with known positive evidence and
   investigate the two index-root bodies. Where a valid closed native solid is
   available, evaluate a local convex decomposition that preserves its shape,
   then enable the justified nonadjacent pair. Do not merely enable the current
   over-broad palm/thumb shell, globally shrink the hand, or introduce a special
   joint-angle cap learned from this Pour frame. Any new geometry belongs to its
   robot link and must be evaluated across configurations, not fitted to a
   single successful hand pose. Inertial parameters must remain fixed while
   collision geometry is compared.
4. **Validate the bowl/plate contact surfaces separately.** Compare derived
   collision pieces with the native bowl interior/rim and plate surfaces using
   geometry and contact-location errors, independently of RL success. If a
   deficiency is demonstrated, create a separate decomposition with a disclosed
   complexity budget and compare it against the original. Raising the 32-part
   cap is a candidate experiment, not an automatic fix; do not select a budget
   or acceptance distance silently. Millimeter-valued geometric tolerances and
   approximation/runtime tradeoffs are still choices to be specified.
5. **Use one validated geometry/pair specification in MINK and runtime.** Generate
   both from the same per-link/per-pair source and test their enumeration. Add
   regression poses with known native intersection and known separation,
   including the original/v3/v4 examples, neutral poses, and coverage of the
   joint range. Separate future physical contact tests must verify that included
   contacts actually generate responses and that unsupported shell constraints
   are not introduced. No such physics integration was run in this review.

Before adopting any derived model, test it on the other approved samples while
holding model settings fixed, and report fitting degradation, residual overlaps
and failures without discarding frames. This evaluates the robot collision
model; it does not promote the temporary first-frame posture adjustment into a
formal or generalizable initializer.

Later reference hand/object overlaps and failed Replay are not themselves new
paper rejection criteria. A corrected self-collision model and a valid,
explicitly specified initialization are distinct from requiring every reference
frame to already be a successful physical trajectory. Formal initialization
remains a separate unresolved design question.

No model/pair replacement, new reset, MPC, physical rollout, training, altered GT
or new rendering was performed. The strict renderer gate remains unchanged.

## Evidence And Code

- [Current coverage review](/data_all/zzx/3.2RL/runs/taco_pour_collision_review_v1/report.json)
- [Inventory/recheck runner](/data_all/zzx/3.2RL/scripts/audit_taco_collision_coverage.py)
- [Fault-injection regressions](/data_all/zzx/3.2RL/tests/test_collision_review_guards.py)
- [Prior temporary thumb/control results](/data_all/zzx/3.2RL/docs/pour_thumb_and_reset_inputs.md)
