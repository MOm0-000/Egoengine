# 水壶样本保持审计记录（2026-09-04）

## 范围

本记录只针对 `taco_pour_kettle_cup_20230917_036`，属于诊断实验。所有输入仍来自原版 DexImit 的 SAPIEN 筛选；只有 SAPIEN 通过的候选才进入 MuJoCo。结果不进入正式 renderer 或论文 3.3 主链，也没有修改摩擦、质量、穿透阈值或成功门槛。

MuJoCo 场景使用 SAPIEN 导出的 PhysX cooked convex collision，场景文件为：

`runs/deximit_generalization_v1/pour_kettle036/mujoco_sapien_exact_scene_candidate230_prestrong_v1/scene.xml`

保持审计从记录的第 1500 个物理步初始化，保持 0.5 秒，并测试 3 个强摩擦时间常数。所有实验串行完成。

## 候选结果

| 候选 | 深度 | SAPIEN 保持阶段 | MuJoCo 强摩擦保持阶段 | 结论 |
|---:|---:|---|---|---|
| 85 | 1 | 食指承载；拇指仅检测到接触 | 只有食指有法向力 | 无拇指对向抓取 |
| 109 | 2 | 掌部 + 中指承载 | 主要只有中指有法向力 | 无拇指对向抓取 |
| 116 | 1 | 拇指承载；食指只是接近 | 拇指和食指偶尔同时有力，但拇指在末段不连续 | 不能通过连续保持门槛 |
| 230 | 1 | 只有掌部承载；中指只是接近 | 没有稳定手指承载 | 掌部推动/支撑，不是多指抓取 |
| 307 | 3 | 只有掌部承载 | 没有稳定手指承载 | 掌部接触，不是多指抓取 |

对应 MuJoCo 报告：

- `runs/deximit_generalization_v1/pour_kettle036/mujoco_loaded_hold_audit_candidate085_v1_strong/report.json`
- `runs/deximit_generalization_v1/pour_kettle036/mujoco_loaded_hold_audit_candidate109_v1_strong/report.json`
- `runs/deximit_generalization_v1/pour_kettle036/mujoco_loaded_hold_audit_candidate116_v2_any_finger_strong/report.json`
- `runs/deximit_generalization_v1/pour_kettle036/mujoco_loaded_hold_audit_candidate230_v1_strong/report.json`
- `runs/deximit_generalization_v1/pour_kettle036/mujoco_loaded_hold_audit_candidate307_v1_strong/report.json`

候选 112、151、324 的 SAPIEN 结果为失败，因此没有送入 MuJoCo，遵守“先 SAPIEN、后 MuJoCo”的串行门槛。

## 发现的代码问题

`diagnostics/audit_deximit_loaded_hold.py` 原来把“对向接触”写成了“拇指 + 中指/无名指”，漏掉食指。项目的通用抓取门槛定义是“拇指 + 任意其他命名手指”，因此该保持审计的判断条件不一致。

现在保持审计改为拇指与 `index`、`mid`、`ring` 或 `pinky` 中任意一根同时有有效法向接触。这里只修正了审计逻辑，没有改变物理模型或放宽门槛。

修复后的候选 116 仍然失败：即使将食指纳入判定，末段 24 个采样步中拇指接触不连续。橡皮 candidate 21 回归仍为 3/3 强摩擦 profile 通过，既有成功证据没有被破坏。

## 结论

水壶存在 SAPIEN 通过候选，但这些候选主要是掌部支撑、单指承载，或只有短暂的两指同时接触。目前没有候选满足严格的连续多指跨 MuJoCo 保持门槛。这个结果不能归因于水壶的“静态几何没有接触”：某些姿态在静态扫描中距离很近，但动态求解后并没有持续的有效法向接触；真正需要的是能在整个保持段持续承载的对向接触。

因此本轮不继续调摩擦或穿透参数，也不把水壶标为成功样本。下一步若继续研究，应从任务阶段的抓姿排序和 SAPIEN 中真实承载接触的候选选择入手，而不是修改 MuJoCo 成功判定。

## 验证

```text
spider/.venv/bin/python -m pytest -q tests/test_bodex_diagnostic_contract.py
40 passed
```

橡皮回归报告：

`runs/deximit_full_bridge/smear071/mujoco_loaded_hold_audit_v9_any_finger_regression/report.json`

该报告仍为诊断性保持审计，不替换橡皮原有正式成功证据。
