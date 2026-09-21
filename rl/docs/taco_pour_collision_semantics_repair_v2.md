# Pour collision semantics repair v2

> Historical candidate-stage report. Its blocker status is superseded: the
> later combined collision model, high-resolution object SDF, floor-contact
> calibration, capacity audit, and initialization protocol v2 have passed.
> Current state is in `docs/rl_reproduction_status.md`. The measurements below
> are retained as provenance for why the final model was built.

## 输入

- 当前正式 Pour scene（只读）；
- 当前 198 帧双手 MINK reference，其中本轮只验 endpoints 0–40；
- `first40_collision_attribution_v1` 冻结的 15 组 CAD/runtime 归因；
- Candidate A 的 t0 状态；
- 原始 XHand CAD、现有 bowl/plate 32 块 collision decomposition；
- 原论文 §3.2.1 的边界：MINK 公开约束为 joint limits 与 self-collision，
  没有公开 hand-object nonpenetration hard constraint。

## 已通过的候选修补

本轮没有修改正式 scene。生成的
`runs/taco_pour_collision_semantics_repair_v2/candidate_runtime_semantics_scene.xml`
是 training-blocked 候选。

对 hand-object，增加 9 个局部椭球。每个椭球只与指定的 bowl 或 plate 物体
pair 相连；需要修 overfill 的旧 pair 被替换，需要补 underfill 的旧 pair 被
保留并增加局部 pair。没有全局放大 finger capsule，也没有把这些 pair 加进
MINK。结果：

| 检查 | v1 | v2 candidate |
| --- | ---: | ---: |
| first-40 hand-object material FN | 4 | 0 |
| first-40 hand-object CAD-clear runtime FP | 23（全部 selected pair） | 0（hand-object） |
| Candidate A middle2/pinky2 外部漏检 | 2 | 0 |

9 个新 geom 全部映射到正确 side/finger channel；288 条新 external pair 都只连
指定物体。MINK self group 仍为 178，新增 external geom 进入该 group 的数量为
0。

对 hand-floor，旧 capsule/box 在 1066 个 frame-link 检查上有 16 个
`>50 µm` 漏检和 10 个 CAD-clear 误报。桌面是平面，因此原 CAD mesh 的凸包
与原 mesh 具有相同的最低支撑点。候选用 26 个只与 floor 配对的 mesh geom
替换 24 条旧 floor pair；重新检查得到 FN=0、FP=0。这些 mesh 不与物体、
另一只手或手内 link 配对。

## 仍未通过：左掌—左拇指近节

原始 CAD 显示 first-40 中 19 个实体穿入漏检。直接连接现有 palm box 与 thumb
capsule 会在 22 个 CAD-clear endpoint 上误报，因此不能采用。

本轮依次测试了三种只服务这一 assembly pair 的候选：

- sphere：校准集与 first-40 为 0/0，但最终错位网格仍有 3 个边界误报；
- 128×128 局部 convex parts：有 1 个校准碰撞点无法由任何零误报 part pair
  覆盖，而且 CoACD 明确报告 128 块上限下未达到请求精度；
- hybrid（8 个 convex pair + 2 个 sphere pair）：校准集与 first-40 为 0/0，
  两张全新最终留出网格分别留下 1 FN 和 1 FP。

这三个候选均未写入正式 XML，保存在
`TRASH/rejected_candidates/2026-09-20_collision_semantics_repair_v2_self_guard/`
作为负实验，不属于正式链路。

后续只读的
`taco_pour_left_palm_thumb_boundary_topology_v1` 没有继续加 guard，而是先把
native CAD 与被拒 Hybrid 在完整关节域内的边界画清。257 个 bend 切片上，
native 有 104 个切片完全不碰、153 个切片只有一个且贴着 rota1 上限的连续
碰撞区间，没有多区间或孤立碰撞岛。碰撞只出现在两个采样 bend 带：
`[0, 0.900703125]` 与 `[1.6512890625, 1.83] rad`。Hybrid 的区间拓扑完全一致，
但边界最大偏移约 `0.00050004 rad (0.02865°)`；原先最终留出集的 1 FN/1 FP
均被独立复现。因此下一步是分两条边界做 CEGIS，允许删减或缩紧造成误报的
旧局部 guard，再补只覆盖漏检带的局部几何；不是继续增加全局球。

## 当时的 gate（已被后续工作取代）

外部 hand-object、hand-floor 与 contact-role contract 已通过候选验收；左掌—
拇指 self-collision 仍是 blocker。因此：

- 正式 scene/reference 未变；
- 不重新 MINK retarget；
- 不运行 capacity；
- 不启动 `initialization_protocol_v2`；
- 不进入 Replay→RL。

正式依据是 `runs/taco_pour_collision_semantics_repair_v2/audit_report.json`。
边界拓扑依据是
`runs/taco_pour_left_palm_thumb_boundary_topology_v1/report.json`。
