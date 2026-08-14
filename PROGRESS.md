# Video-to-SPIDER V1 实施进度

最后更新：2026-08-14

本文档是实现期间的滚动状态记录。每个工作包开始、状态变化、产生交付物、发现/解除阻塞或改变接口时，必须更新对应行和变更日志。状态只允许使用：`pending`、`in_progress`、`blocked`、`done`、`failed`。

## 2026-08-13 当前快照

以下内容覆盖最新代码状态；下方较早的 WP/变更日志仍保留为历史记录。

| 项目 | 状态 | 结果 |
|---|---|---|
| 核心测试 | done | `v2s-core` 下 `160 passed`（`python -m pytest -q -p no:cacheprovider`） |
| RL 训练核心 | done | 复用克隆的 H2S2R `PpoAgent`，未自行重写 PPO |
| RL 环境适配器 | done | `video_to_spider/rl/mjwp_env.py` 已通过 GPU 冒烟 `reset/step/state roundtrip` |
| RL 最小训练入口 | done | `scripts/run_mjwp_ppo.py` 已在单卡 GPU2 跑通最小训练循环并保存 checkpoint |
| RL solver 注入 | done | `run_mjwp_modeswitch.py` 已支持 `+use_rl_reward=true` 并通过 `MJWP_RL_CHECKPOINT` / `MJWP_RL_TRAIN_METADATA` 注入训练出的残差策略；该入口独立于 `run-spider`，自动主链路尚未默认接入 |
| 冗余清理 | done | 已删除无引用的 `rl/networks.py`、旧实验局部 `.venv*` 和 Python cache；未动克隆上游仓库 |

## 当前基线

| 项目 | 状态 | 结果 |
|---|---|---|
| 实现计划 | done | [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) 已固定 EgoDex 外参、自动文字关键词、mesh proposal 两阶段选择和最小必要诊断 |
| Git 仓库 | done | `video_to_spider/.git` 当前备份分支为 `zxx`，`origin` 为 `https://github.com/MOm0-000/Egoengine.git`；`third_party/`、权重和运行产物由 `.gitignore` 排除 |
| EgoDex 数据根目录 | done | `/data_all/share/datasets/egodex` |
| EgoDex 结构 | done | 当前为 1 个 `part`、26 个任务目录；结构为 `part*/{task}/{id}.mp4` 与同名 `.hdf5` |
| EgoDex 文件配对 | done | 已确认 46,234 组同目录同 stem 配对；缺失 HDF5 为 0，缺失 MP4 为 0 |
| EgoDex 总体积 | done | 约 336G |
| 核心/ingest 环境 | done | `v2s-core` 的 `h5py/cv2/numpy/scipy/trimesh/zarr` import preflight 通过 |
| WiLoR 权重 | done | detector、WiLoR checkpoint、`MANO_RIGHT.pkl` 已存在 |
| SAM 3.1 权重 | done | `sam3.1_multiplex.pt` 已存在；`v2s-sam3` import preflight 通过 |
| SAM 3D Objects 权重 | done | generator/encoder/mesh decoder checkpoints 已存在并完成真实 held-coin proposal inference |
| Depth Anything metric 权重 | done | Hypersim ViT-L metric checkpoint 已存在；0 字节 relative-depth 文件明确禁止使用 |
| FoundationPose 权重 | done | scorer/refiner 两套权重已存在；`v2s-foundationpose` import preflight 通过 |
| SPIDER uv 环境 | done | `../spider/.venv` 已存在；`uv run --frozen --no-sync` 可 import `spider/torch/mujoco/warp` |
| 代码实现 | in_progress | 早期 WP0/WP1/WP2/WP5/WP6/WP7/WP8/WP9/WP10 记录见下文；当前核心代码快照见本文件顶部 `2026-08-13 当前快照` |

## 执行环境

| 范围 | 环境 | 状态 | 验证结果 |
|---|---|---|---|
| core/ingest/eval/orchestration | `v2s-core` | done | Python 3.11；核心数据、图像、mesh 包 import 通过 |
| SAM 3 | `v2s-sam3` | done | `torch 2.10.0+cu128`、SAM 3.1 import/checkpoint load 与 90-frame text-only GPU inference 通过；object 89/90、hand 90/90 valid |
| SAM 3D Objects | `v2s-sam3d` | done | Python 3.11；无需 `LIDRA_SKIP_INIT` 即可 plain import `torch/pytorch3d/kaolin/sam3d_objects`；PyTorch3D 0.7.8、Kaolin 0.17.0、`nvdiffrast 0.4.0` 与 inference extras CUDA smoke 通过；9 个 checkpoint 与 DINOv2/MoGe 均可离线完成 CPU pipeline 初始化；`zarr 2.18.3` 可只读打开真实 depth artifact |
| WiLoR | `v2s-wilor` | done | `torch 2.0.0+cu117`；真实 90-frame detector/model inference 与 schema/evaluator 通过 |
| Depth Anything V2 | `v2s-depth` | done | `torch 2.5.1+cu124`；Hypersim ViT-L 真实 90-frame GPU inference 与 Zarr QC 通过 |
| FoundationPose | `v2s-foundationpose` | done | `torch 2.5.1+cu124`、`estimater` import 通过 |
| optimizer/contact | `v2s-opt` | done | `torch/numpy/scipy/trimesh/zarr` import 通过 |
| SPIDER/MJWP | `../spider/.venv` (`uv`) | done | `torch 2.11.0+cu130`、MuJoCo 3.7.0、Warp 1.12.1；synthetic full MJWP GPU smoke 通过 |

环境 `done` 与工作包 `done` 含义分开记录。当前节点可见 8 张 A100 40GB；SAM 3.1、WiLoR、metric depth、SAM 3D Objects、FoundationPose 和 synthetic MJWP 已完成真实或 GPU smoke，但 GPU 正被其他任务高负载共享。SAM 3D Objects 已通过 low-VRAM stage-wise offload 生成 3 个真实 proposal，不再是环境或 WP5 阻塞项。

## 工作包状态

| 工作包 | 所有权 | 状态 | 依赖 | 当前交付物 | 下一步 |
|---|---|---|---|---|---|
| WP0 | schema、manifest、coordinates | done | 无 | `pyproject.toml`、`schemas.py`、`manifest.py`、`coordinates.py`；schema version 1.0 | 作为其他 WP 的冻结公共接口 |
| WP1 | EgoDex ingest、GT reader | done | WP0 | `ingest/egodex.py`、显式 GT reader/oracle 外参诊断；真实 90 帧 artifact | 后续 adapter 复用 `runs/egodex_flip_coin_0_f000_090` |
| WP2 | instruction parser、SAM 3 adapter | done | WP0/WP1 | 真实 90-frame text-only coin/hand masks；自动恢复 frames 0–11，frame 30 因完全遮挡显式 invalid；object valid `0.9889`、hand valid `1.0`；旧 artifact/OOM 证据均保留 | 作为 WP6 完整 tracking 输入 |
| WP3 | WiLoR adapter | done | WP0/WP1 | 真实 90-frame 完整 MANO/joints/vertices/camera translation、overlay、GT metrics | 作为 WP7 的 hand observation 输入 |
| WP4 | Depth Anything adapter | done | WP0/WP1 | 真实 90-frame 原分辨率 metric Zarr、uncertainty、warp QC 和视频 | 作为 WP5/WP7 的鲁棒先验；不得覆盖相机/手尺度 |
| WP5 | SAM 3D Objects mesh | done | WP2/WP4 | low-VRAM stage-wise CUDA offload 真实生成 3/3 合格 held-coin proposals；raw GLB、canonical visual/collision mesh、静态 ranking、自动转台对比视频与历史失败证据均保留 | WP6 对 proposals 作最终 tracking selection |
| WP6 | FoundationPose adapter | done | WP1/WP2/WP4/WP5 | 完整 90-frame masks 上筛选 3 个真实 proposals，仍选 seed 42；86/90 valid、17 次自动重注册、mean mask IoU `0.3574`、tracking score `0.5710`；自动保存 mask/mesh/invalid/register overlay | 作为 WP7 raw pose 输入；invalid gaps 保留 |
| WP7 | sequence optimizer、contact | done | WP1/WP3/WP6 | 真实 90-frame aligned/contact；episode-level `s_object 0.0890 -> 0.02269 m`，jitter `16.63 -> 0.946 m/s²`、reprojection `11.57 -> 4.65 px`、silhouette IoU `0.3452 -> 0.3503`；自动保存 raw-vs-aligned/contact 视频；depth scale 冲突与不可观测项显式保留 | 作为 WP8 SPIDER export 输入 |
| WP8 | SPIDER exporter、runner | done | WP0/WP6/WP7 | 真实 bimanual xhand dataset；decomposition、contact cross-check、scene、148-frame IK 与 MJWP 全部返回 0；EGL headless 生成 148-frame IK 与 150-frame ref-vs-sim MJWP 视频 | 当前真实 rollout 完成；质量误差单独保留 |
| WP9 | metrics、visualization | done | WP0/WP1 | 真实 mesh proposal turntable、FoundationPose overlay、raw-vs-aligned/contact、SPIDER/MJWP 视频与 artifact-only unified report；诊断 manifest 不参与 M4 通过条件 | 扩展到批量 episode 时复用 |
| WP10 | CLI、编排、端到端测试 | in_progress | WP0-WP9 | 已有 scan/ingest/oracle/evaluate/optimize/export-spider/run-spider/visualize-run CLI 与 synthetic M0 fixture | 增加跨 Conda 环境 run-stage/run-all 与缓存恢复 |

## 执行波次

| 波次 | 工作包 | 状态 | 进入条件 | 退出条件 |
|---|---|---|---|---|
| Wave 0 | WP0 | done | 计划文档已冻结 | 22 个当前 CPU 单测中的 schema/坐标/manifest 测试通过 |
| Wave 1 | WP1、WP2、WP3、WP4、WP8、WP9、WP10 | in_progress | WP0 done | WP1/WP2/WP3/WP4 done；WP8/WP10 已有合成/真实基础 artifact，集成待推进 |
| Wave 2 | WP5 | done | held-coin 3-frame WP2 与 WP4 artifact 可用 | 3 个真实 checkpoint proposals 合格、排序且 raw GLB/失败历史保留 |
| Wave 3 | WP6、WP7 | done | WP5 done | 真实 90-frame object pose、aligned trajectory 与 contact artifact 完成 |
| Wave 4 | WP8、WP9、WP10 集成 | done | WP6、WP7 done | 一个真实 episode 完成 decomposition/contact/scene/IK/MJWP 与统一报告 |

## 里程碑状态

| 里程碑 | 状态 | 证据 | 下一门槛 |
|---|---|---|---|
| M0 基础闭环 | done | WP0 tests；`runs/synthetic_runner_smoke` 通过 decomposition、contact cross-check、scene、full xhand IK 和 MJWP；MJWP error `0.0054 m / 0.0066 rad` | none |
| M1 感知 artifact 闭环 | done | 同一真实 clip 已有 90-frame ingest/WiLoR/depth/SAM 3；held coin object valid `89/90`、hand valid `90/90`，唯一 gap 为完全遮挡 frame 30 | none |
| M2 物体轨迹闭环 | done | 真实 seed-42 canonical mesh 与 FoundationPose 90-frame timeline；86/90 valid，gap 与重注册显式保留 | none |
| M3 联合优化闭环 | done | 真实 aligned trajectory/contact 通过 schema；全局 scale、jitter、mask reprojection 与 silhouette 定量改善；metric-depth residual 恶化及 penetration/raw-slip 不可观测项显式报告 | none |
| M4 SPIDER 完整闭环 | done | 使用优化后的 `0.02269 m` coin scale，真实 `flip_coin/0 [0,90)` 完成 5 个 SPIDER 命令及 IK/MJWP simulation videos；统一报告 `12/12`、`simulation_video_complete=true`、`m4_complete=true` | MJWP position `0.0864 m` 达标，rotation `1.1971 rad` 未达 `0.5 rad`，作为质量结果保留而非链路失败 |

## 自动化流程状态

```text
EgoDex recursive scan                 done
  -> instruction extraction           done
  -> SAM 3 text keyword candidates    done (90-frame text-only artifact)
  -> automatic candidate ranking      done (held coin visually confirmed; invalid gap preserved)
  -> WiLoR hand observations          done (90 frames)
  -> Depth Anything metric depth      done (90 frames)
  -> SAM 3D ranked mesh proposals     done (3/3 real checkpoint proposals qualified)
  -> FoundationPose mesh selection/tracking done (real 90-frame artifact; seed 42)
  -> hand-object optimization         done (real aligned trajectory)
  -> contact inference                done (real hysteresis/local-frame artifact)
  -> SPIDER export                    done (real bimanual xhand dataset)
  -> MJWP rollout                     done (15 records; full command success)
  -> diagnostic visualization         done (mesh/tracking/raw-vs-aligned-contact)
  -> batch metrics/report              done for target clip (12/12; M4 complete)
```

## 固定自动规则

- 数据根目录：`/data_all/share/datasets/egodex`。
- episode 配对：相同目录、相同 stem 的 `.mp4` 和 `.hdf5`。
- 任务文本：按 `which_llm_description` 选择 `llm_description` 或 `llm_description2`，缺失时回退到可用字段和任务目录名。
- SAM 3：只允许 text prompt；候选来自名词短语、去修饰词短语和 `object_keyword_rules.yaml` 同义词扩展。
- SAM 3 选择：检测分数、mask 有效率、前后向传播一致性、面积稳定性、手 mask 重叠率的加权分数。
- 关键帧选择：清晰度、目标可见面积、遮挡率、深度有效率和视角变化的确定性评分。
- mesh proposal 静态排序：WP5 只使用 MuJoCo/trimesh integrity、mask silhouette residual 和 depth residual。
- 最终 mesh 选择：WP6 按静态排名尝试 proposals，使用 FoundationPose 有效帧率、mask IoU、depth/render residual 和姿态连续性选择；所有尝试及拒绝原因必须保留。程序化 mesh 只允许 debug smoke，禁止作为真实 WP5/WP6 结果。
- 必要诊断：只比较同一次感知结果的 raw trajectory 与 aligned trajectory；不运行 A/B/C/D 正式消融矩阵。
- 失败处理：候选或 proposal 全部不合格时写入失败状态和原因码，不复制上一帧、不要求额外输入。
- 验收方式：仅使用 schema 校验、数值阈值、渲染 residual、仿真返回值和可复现 metrics。

## 资源与环境阻塞

| 阻塞项 | 影响 | 解除方式 | 负责人 |
|---|---|---|---|
| 共享 GPU 显存竞争 | 已通过 CPU video/tracker-state offload、single-frame grounding、自动 invalid-span resume 和稳定窗口 admission 完成 SAM 3；FoundationPose/MJWP 仍需共享窗口 | 保持只读监控和稳定窗口 admission，不终止或修改其他用户进程 | 主代理/环境维护者 |

SPIDER 在当前受限执行环境中必须使用可写缓存目录并复用已同步的 `.venv`：

```bash
cd /data_all/liyunhao/egoengine/spider
UV_CACHE_DIR=/tmp/video-to-spider-uv-cache uv run --frozen --no-sync python ...
```

## 变更日志

### 2026-07-30

- WP9 可视化补齐：新增 CPU-only `visualize-run`，从已保存 artifact 自动生成 120-frame mesh proposal turntable、90-frame FoundationPose mask/mesh/invalid/register overlay、90-frame raw-vs-aligned/contact 对比视频及 `visualization_manifest.json`。三个阶段结束时自动尝试生成；诊断编码失败只写 warning，不会推翻有效数值 artifact。真实视频分别为 `1280x720`、`960x540`、`1280x438`，均经 ffprobe 和抽帧检查；统一报告在独立 `diagnostics.visualization` 字段暴露状态，不改变 M4 的 `12/12` 验收边界。
- M4 完成：真实 `egodex_flip_coin_0_f000_090` 导出 149-frame/50 Hz bimanual xhand dataset。SPIDER `decompose_fast`、contact cross-check、`generate_xml`、148-frame `ik_fast`、256-sample/8-iteration MJWP 全部返回 0；IK/MJWP NPZ 全部有限。无 DISPLAY 的首次 GLFW 失败日志保留；runner 固定 `MUJOCO_GL=egl`/`PYOPENGL_PLATFORM=egl` 后，正式生成 148-frame IK video 与 150-frame ref-vs-sim MJWP simulation video。
- 统一报告 verifier 已强化为必须同时存在非空 MJWP trajectory 与 simulation video；当前达到 `12/12 stages available`、`spider_chain_complete=true`、`simulation_video_complete=true`、`m4_complete=true`，且 `ground_truth_consumed_by_inference=false`。全局 scale 修正后的 MJWP object error 为 `0.0864 m / 1.1971 rad`：position 达到 `0.1 m` 门槛，rotation 未达到 `0.5 rad`；链路完成与质量门槛分开记录。
- WP6 完成：在完整 90-frame SAM masks 上按固定策略筛选 3 个真实 proposals，仍选 `proposal_00_f000000_s42`。最终 FoundationPose valid `86/90`、mean mask IoU `0.3574`、tracking score `0.5710`，17 次自动重注册，所有 invalid gaps 保留。
- WP7 完成：修复 `smooth_second_difference` 对 advanced-indexed non-contiguous fingertip 数组写入临时 reshape copy 的问题并新增回归测试。补齐计划要求的 episode-level 全局 `s_object` 优化，以 18 个自动采样帧的 observed/rendered mask area ratio 将 scale 从 `0.0890` 调整到 `0.02269 m`；真实 coin 导出尺寸约 `22.7 x 4.6 x 21.5 mm`。object acceleration jitter 从 `16.63` 降至 `0.946 m/s²`、mask centroid reprojection 从 `11.57` 降至 `4.65 px`、30-frame silhouette IoU 从 `0.3452` 升至 `0.3503`。Depth Anything relative residual 从 `0.0162` 恶化到 `0.7227`，与已记录的 metric-depth/hand scale 冲突一致；penetration 因 aligned full hand surface 未物化、raw slip 因无 raw contact 均标为 `not_observable`。
- WP2 完成：SAM 3 全 90 帧首次传播得到 78/90 valid；新增 suppression sentinel 归一化与完全自动的 invalid-span resume，在不使用点/框/mask/手工帧选择的前提下恢复 frames 0–11。最终 held-coin object valid `89/90`、hand valid `90/90`；frame 30 为完全遮挡，按计划保留 explicit invalid gap。前三帧与 validated smoke mask IoU 为 `0.9949/1.0000/0.9990`。原三帧 artifact、首次完整传播和 OOM failure history 均单独保留。
- WP0 完成：新增可安装包、schema 1.0 严格校验、SE(3)/`wxyz`/30-to-50 Hz 工具、内容哈希 manifest；合成 round-trip、非法 NPZ 和 cache invalidation 测试通过。
- WP1 完成：实现 EgoDex `part*/*` 递归配对、HDF5 root attribute 指令选择、GT 隔离的 RGB/K/`T_world_camera` ingest，以及必须显式声明 `--uses-ground-truth` 的外参方向 oracle 命令。
- 真实 `flip_coin/0` 的 `[0,90)` 共 90 帧生成 `runs/egodex_flip_coin_0_f000_090`。普通 ingest manifest 只声明 RGB、K、camera extrinsics、instruction text 和 keyword candidates；不包含手/身体 GT。
- 外参 oracle 在 frame 45 验证 `transforms/camera` 为 `T_world_camera`：正确方向 12 个腕/指尖投影的 in-frame ratio 为 1.0，逆向解释为 0.0；报告与 overlay 位于 `calibration/oracle_camera_direction/`，manifest 标记 `uses_ground_truth=true`。
- WP8/WP10 启动：实现单物体 SPIDER exporter、inactive side 的 identity quaternion 约定、50 Hz 重采样和 synthetic M0 fixture。
- `runs/synthetic_m0` 已通过 SPIDER `decompose_fast`（4 convex parts）、`generate_xml`（67 contact pairs）和 `ik_fast` 10-frame smoke；`scene.xml` 可由 MuJoCo 加载，`trajectory_kinematic.npz` 的 `qpos[9,25]`/`qvel[9,24]` 全部有限。
- `v2s-core` 已以 editable 方式安装本包；依赖约束固定为 NumPy `<2` 和 OpenCV `<5`，恢复并验证 `numpy 1.26.4`、`opencv 4.11.0`、`h5py 3.12.1`。
- WP2 启动：新增 SAM 3.1 multiplex text-only artifact adapter 和独立环境 launcher。真实 3-frame `coin`/`hand` checkpoint inference 通过，object/hand valid rate 均为 1.0，并生成 1920x1080 masks/overlay。
- 可视化发现原始最高置信度实例选择偏向桌面静止圆片；已改为模型置信度与 hand-mask 距离联合评分。共享任务随后把 GPU 4–7 剩余显存从约 21GB 压至约 8GB，新规则重验在峰值 OOM，旧 smoke artifact 保留且 manifest 明确警告，未伪装为新规则已通过。
- WP3 完成：真实 90-frame WiLoR 左右手有效率均为 1.0，完整 schema 通过；使用 EgoDex `K` 转换 camera translation，z 的 P5/median/P95 为 `0.296/0.312/0.383 m`，保存每侧共享 beta 初值。
- WiLoR 显式 GT evaluator：右手 fingertip MPJPE `24.9 mm`、root-relative `21.5 mm`、wrist translation `37.0 mm`；左手分别为 `63.8/23.1/82.2 mm`。左手 wrist rotation 约 `3.01 rad`，记录为 handedness/轴约定待优化项，不隐藏该失败指标。
- WP4 完成：安装 `zarr 2.18.3`/`numcodecs 0.13.1` 到 `v2s-depth`；真实 90-frame Hypersim ViT-L metric inference 输出 1920x1080 `depth_m/valid/uncertainty_proxy` Zarr（约 622MB），valid ratio 1.0，并保存 89 对外参 warp residual。
- metric depth 全局中值约 `1.45 m`，与 WiLoR/GT 手深约 `0.3-0.5 m` 存在明显尺度差；已作为 WP7 的鲁棒先验风险记录，禁止用它覆盖已知相机外参与手尺度。

日期：2026-07-30
工作包：WP0
状态：done
修改：`pyproject.toml`、`video_to_spider/schemas.py`、`manifest.py`、`coordinates.py`、`tests/test_schemas.py`、`test_coordinates.py`、`test_manifest.py`
验证：`conda run -n v2s-core python -m pytest -q` -> `12 passed`
artifact：schema version `1.0`；合成验证由 tests 临时目录生成
阻塞：none
下一步：冻结公共 schema，后续变更必须保持向后兼容或显式升版

日期：2026-07-30
工作包：WP1
状态：done
修改：`video_to_spider/ingest/egodex.py`、`egodex_ground_truth.py`、`cli.py`、`tests/test_ingest.py`
验证：`conda run -n v2s-core python -m video_to_spider.cli ingest --task flip_coin --episode-id 0 --output-dir runs/egodex_flip_coin_0_f000_090 --start-frame 0 --end-frame 90 --overwrite`；oracle direction margin `1.0`
artifact：`runs/egodex_flip_coin_0_f000_090/manifest.json`，schema version `1.0`
阻塞：none
下一步：WP2/WP3/WP4 消费同一真实 frame/K/extrinsics artifact

日期：2026-07-30
工作包：WP8
状态：in_progress
修改：`video_to_spider/export/spider.py`、`scripts/make_synthetic_m0.py`、`tests/test_spider_export.py`
验证：SPIDER `decompose_fast`、`generate_xml --no-show-viewer`、`ik_fast --end-idx 10 --no-show-viewer --no-save-video` 均 exit 0；MuJoCo scene load 与 IK finite check 通过
artifact：`runs/synthetic_m0/dataset/processed/video_to_spider_egodex/`，export schema version `1.0`
阻塞：真实 WP6 selected mesh 与 WP7 aligned/contact 尚未产生；synthetic MJWP 已验证，真实仍待输入
下一步：模型与优化 artifact 可用后导出真实 EgoDex clip 并运行 detect_contact/MJWP

日期：2026-07-30
工作包：WP10
状态：in_progress
修改：`video_to_spider/cli.py`、`scripts/make_synthetic_m0.py`
验证：真实 ingest、oracle 命令和 synthetic fixture 命令均 exit 0
artifact：同 WP1/WP8
阻塞：其余 stage entrypoint 尚未实现
下一步：补齐 preflight/run-stage/run-all/evaluate/export-spider 编排与缓存恢复

日期：2026-07-30
工作包：WP2
状态：in_progress
修改：`video_to_spider/adapters/sam3.py`、`scripts/run_model_adapter.sh`
验证：`CUDA_VISIBLE_DEVICES=4 ... sam3 --end-frame 3 --max-candidates 1` checkpoint/text propagation exit 0；mask valid rate 1.0；新 hand-proximity 规则重验因共享 GPU 峰值 OOM 尚未通过
artifact：`runs/egodex_flip_coin_0_f000_090/segmentation/`，schema version `1.0`
阻塞：非独占 GPU 显存暂不足以重验新规则/完整 90 帧；代码和旧 smoke artifact 可继续被其他 WP 使用，但不能宣称 WP2 done
下一步：资源回落后先重验手持 coin 实例，再运行完整 90 帧和所有 instruction candidates

日期：2026-07-30
工作包：WP3
状态：done
修改：`video_to_spider/adapters/wilor.py`、`video_to_spider/eval/egodex.py`、`video_to_spider/cli.py`
验证：90-frame GPU inference exit 0；`validate_npz(..., 'wilor_raw')` 通过；`evaluate-wilor --uses-ground-truth` 输出左右手指标
artifact：`runs/egodex_flip_coin_0_f000_090/hands/wilor_raw.npz` 与 `evaluation/wilor_hand_metrics.json`，schema version `1.0`
阻塞：none；左手绝对 pose/rotation 质量较弱，作为 WP7 低权重观测处理
下一步：WP7 消费 raw artifact，保留置信度并比较 aligned 改善

日期：2026-07-30
工作包：WP4
状态：done
修改：`video_to_spider/adapters/depth_anything.py`
验证：90-frame Hypersim ViT-L GPU inference exit 0；Zarr arrays shape `(90,1080,1920)`、finite/valid ratio 1.0；89 对 background warp metrics 已生成
artifact：`runs/egodex_flip_coin_0_f000_090/depth/metric_depth.zarr`，schema version `1.0`
阻塞：none；绝对尺度与手/相机证据不一致，必须使用鲁棒低权重
下一步：WP5 用 mask 内深度初始化 mesh scale，WP7 做联合尺度诊断

日期：2026-07-30
工作包：WP9
状态：in_progress
修改：`video_to_spider/eval/egodex.py`、SAM/WiLoR/depth 自动 overlay
验证：外参 oracle、WiLoR GT metrics 和各 perception 视频均生成；12 个 CPU 单测通过
artifact：`runs/egodex_flip_coin_0_f000_090/evaluation/` 与各 stage overlay
阻塞：WP6/WP7 尚无 raw/aligned object trajectory，不能完成统一报告
下一步：补批量汇总与 raw-vs-aligned 必需诊断

日期：2026-07-30
工作包：WP2
状态：in_progress
修改：`video_to_spider/adapters/sam3.py`（hand-proximity 选择与 video CPU offload）；旧 artifact 移至 `segmentation_legacy_stationary/`
验证：`CUDA_VISIBLE_DEVICES=7 ... --end-frame 3 --max-candidates 1` exit 0；overlay 程序化/视觉检查确认 `target_object_id=1` 为右拇指上的 coin，anchor hand distance `0 px`；90-frame full run 在 hand propagation frame 4 OOM
artifact：`runs/egodex_flip_coin_0_f000_090/segmentation/`（validated 3-frame schema 1.0）；`segmentation_full/` 保留失败目录
阻塞：完整 90-frame SAM 3.1 实测本进程约占 20.7GB，需要单卡至少约 22GB 空闲显存
下一步：资源满足后运行 90-frame full artifact，不能用 3-frame结果伪装完成 M1

日期：2026-07-30
工作包：WP5 环境
状态：done
修改：完成 `v2s-sam3d` Conda 依赖安装与 Hydra 官方 patch；补齐 `third_party/sam-3d-objects/checkpoints/hf/` 顶层 checkpoint 布局（复用已有文件 hardlink）；补充 artifact 读取栈 `zarr 2.18.3`、`numcodecs 0.13.1`、`asciitree 0.3.3`；缓存官方 DINOv2 repo 与 `dinov2_vitl14_reg` 预训练权重；从 NVLabs 官方 commit `253ac4fcea7de5f396371124af597e6cc957bfae` 安装 `nvdiffrast 0.4.0`；补充上游公开发行版缺失的无副作用 `sam3d_objects/init.py`，使标准 import 不再依赖 `LIDRA_SKIP_INIT`，未修改模型或 pipeline 语义
验证：`env -u LIDRA_SKIP_INIT conda run -n v2s-sam3d python -c "import torch,pytorch3d,kaolin,sam3d_objects"` exit 0，版本为 `torch 2.5.1+cu121`、PyTorch3D 0.7.8、Kaolin 0.17.0；同样不设置 `LIDRA_SKIP_INIT` 时，9 个 checkpoint、MoGe 与两个 condition embedder 在禁止网络下载条件下完成完整 CPU `InferencePipelinePointMap` 初始化，全部 model/depth/embedder device 为 CPU；FlashAttention 2.8.3、gsplat 1.5.3 CUDA smoke 通过；`nvdiffrast` 在 A100 创建 `RasterizeCudaContext` 并完成 16×16 单三角形 CUDA rasterization，`rast/rast_db` shape 均为 `(1,16,16,4)`、72 个覆盖像素且 finite；Hydra `utils.py` SHA256 为 `b5799ea99626593e7650cd6ab7e15639e32d495d1a16f71d627811f86a2a6fba`；`zarr.open_group(..., mode="r")` 对真实 metric-depth group 返回 `read_only=True`，5 个数组可读且元数据哈希前后不变；GPU 初始化仍因共享显存 OOM
artifact：`/data_all/liyunhao/miniconda3/envs/v2s-sam3d`；`third_party/sam-3d-objects/sam3d_objects/init.py`（公开版 no-op initialization compatibility module）；`nvdiffrast-0.4.0.dist-info/direct_url.json` 固定 NVLabs commit `253ac4fcea7de5f396371124af597e6cc957bfae`；`third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml` 与 9 个已验证权重；`~/.cache/torch/hub/facebookresearch_dinov2_main`（main commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`，`hubconf.py` SHA256 `c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64`）；`~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`（1217607321 bytes，SHA256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`）；`runs/egodex_flip_coin_0_f000_090/depth/metric_depth.zarr` 只读验证，schema version `1.0`
阻塞：环境与 CPU checkpoint runtime smoke 已完成；8 张 A100 当前均被高负载共享，真实 GPU proposals 仍待足够空闲显存
下一步：获得显著多于当前 8.2GB 的单卡空闲显存后复用本地 cache 运行 GPU `Inference`，再生成 held-coin 的 3 到 5 个真实 mesh proposals

日期：2026-07-30
工作包：WP5
状态：in_progress
修改：`video_to_spider/adapters/sam3d_objects.py`、`tests/test_sam3d_objects.py`、`pyproject.toml`
验证：held-coin masks 的关键帧 dry-run exit 0；frame 0 排名第一；repair/canonical/static-fit CPU tests 通过
artifact：`runs/egodex_flip_coin_0_f000_090/mesh_proposals/dry_run.json`，schema version `1.0`
阻塞：`v2s-sam3d` inference extras/checkpoint smoke 尚未完成
下一步：环境交接后对 held-coin keyframe 生成 3 到 5 个真实 mesh proposals

日期：2026-07-30
工作包：WP6
状态：in_progress
修改：`video_to_spider/adapters/foundationpose.py`、`tests/test_foundationpose.py`
验证：固定 tracking score、候选统计、registration/jump tests 通过；`v2s-foundationpose` checkpoint/import preflight 通过
artifact：真实 output 待 WP5；目标 schema 为 `object_tracking/foundationpose_raw.npz` + `selected_mesh.json`
阻塞：缺少真实 WP5 mesh proposals 与完整 90-frame masks
下一步：按静态排名运行候选 screening，再对 selected mesh 双向 tracking

日期：2026-07-30
工作包：WP7
状态：in_progress
修改：`video_to_spider/optimization/{sequence,smoothing,contact}.py`、`tests/test_optimization.py`
验证：二阶平滑降低 jitter、SO(3) 保持正交、接触滞回/local position tests 通过
artifact：真实 aligned/contact 待 WP6；schema version `1.0`
阻塞：缺少完整 WP6 object pose timeline
下一步：真实运行时保留左手低置信度与 depth scale conflict，并输出 raw-vs-aligned metrics

日期：2026-07-30
工作包：WP8/WP10
状态：in_progress
修改：`export/spider.py`、`export/spider_runner.py`、`cli.py`、相关 tests
验证：`conda run -n v2s-core python -m pytest -q` -> `20 passed`；fresh synthetic runner 的 decomposition/detect_contact/scene/full IK/MJWP 均 exit 0；MJWP `0.0054 m / 0.0066 rad`，论文阈值 success；visual contact 在 detect_contact 后 byte-identical 恢复
artifact：`runs/synthetic_runner_smoke/.../spider_run_report.json`；visual/SPIDER 两套 contact、scene、IK/MJWP NPZ 与逐命令日志
阻塞：真实 WP6/WP7 artifact 未产生；synthetic detect_contact 单手仍返回 10 列，runner 已记录 shape mismatch 并只比较共同 5 指
下一步：真实 aligned/contact 后执行 decomposition → detect_contact → scene → IK → MJWP

日期：2026-07-30
工作包：WP9
状态：in_progress
修改：`video_to_spider/eval/metrics.py`、`tests/test_metrics.py`、`cli.py`
验证：`conda run -n v2s-core python -m pytest -q` -> `21 passed`；`evaluate-run` 对当前真实 run 明确报告 5/12 stages available、7 stages missing、`m4_complete=false`
artifact：`runs/egodex_flip_coin_0_f000_090/evaluation/unified_run_report.json`，schema version `1.0`
阻塞：WP5/WP6/WP7/真实 SPIDER artifacts 尚未产生
下一步：下游 artifact 到齐后重跑同一命令并要求 `m4_complete=true`

日期：2026-07-30
工作包：WP6
状态：in_progress
修改：`video_to_spider/adapters/foundationpose.py`、`tests/test_foundationpose.py`
验证：先以 1920x1080 输入复现 scorer 阶段 5.84GiB 瞬时分配 OOM；加入 `--max-input-side 640` 及 K 同步缩放后，在同一共享 A100 上完成 checkpoint、252-view register、双向 track；FoundationPose 单测 `3 passed`
artifact：`runs/foundationpose_adapter_smoke/object_tracking/foundationpose_raw.npz`、`selected_mesh.json`、`tracking_metrics.json`，schema version `1.0`；3 帧 valid rate `1.0`、mean mask IoU `0.0578`、tracking score `0.7031`
阻塞：该运行使用明确标注 `debug_only` 的程序化 coin，只验证 WP6 runtime，不能替代真实 WP5 mesh；完整真实 WP6 仍缺 WP5 proposals 与 90-frame masks
下一步：真实 WP5/WP2 artifact 到齐后用相同显存上界运行完整候选筛选和 90-frame 双向跟踪

日期：2026-07-30
工作包：WP5
状态：in_progress
修改：`video_to_spider/adapters/sam3d_objects.py` 增加模型初始化/提案异常时的 manifest 失败落盘
验证：在 GPU 6 短暂约 22.4GiB 空闲窗口启动真实 held-coin WP5；关键帧排序前置检查发现 `v2s-sam3d` 缺少 `zarr`，以 `ModuleNotFoundError` 退出且真实 manifest `sam3d_objects.success=false`
artifact：`runs/egodex_flip_coin_0_f000_090/manifest.json` 保存失败类型、命令、环境和时间；未生成或伪造 mesh proposal
阻塞：Zarr 只读依赖已补齐并由 adapter dry-run 验证；共享 GPU 恢复到每卡约 5.8-8.3GiB 空闲，仍不足完整 pipeline
下一步：显存再次回落后原命令重跑真实 proposals

日期：2026-07-30
工作包：WP5
状态：done
修改：`video_to_spider/adapters/sam3d_objects.py` 增加 low-VRAM 分阶段 CUDA offload、独立 DINO condition token、MoGe resolution 配置、失败归档与 overwrite 历史保留
验证：`CUDA_VISIBLE_DEVICES=6 ... v2s-sam3d ... --seeds 42 43 44 --max-keyframes 2 --max-proposals 3 --low-vram --moge-resolution-level 6 --overwrite` exit 0；3/3 proposals qualified；seed 44 静态 rank 1 且 watertight
artifact：`runs/egodex_flip_coin_0_f000_090/mesh_proposals/mesh_ranking.json`、`proposal_00_f000000_s42/` 至 `proposal_02_f000000_s44/`、`failure_history/`；schema version `1.0`
阻塞：none；metric depth 与 hand scale 冲突导致静态 silhouette/depth score 为 0，已按设计交由 WP6 tracking 作最终选择
下一步：WP6 在完整 90-frame masks 上复用真实 proposals 和最终 selected mesh

日期：2026-07-30
工作包：WP6
状态：in_progress
修改：`video_to_spider/adapters/foundationpose.py` 加入 K 一致输入缩放、`--max-input-side`、显式 failure manifest 与 debug-only mesh warning
验证：`CUDA_VISIBLE_DEVICES=7 ... v2s-foundationpose ... --max-candidates 3 --screening-radius 1 --register-iter 2 --track-iter 1 --max-input-side 640 --overwrite` exit 0；真实 3-frame proposal screening 选中 `proposal_00_f000000_s42`；最终 valid rate `1.0`、mean mask IoU `0.56090`、relative depth residual median `0.01371`、tracking score `0.83388`
artifact：`runs/egodex_flip_coin_0_f000_090/object_tracking/foundationpose_raw.npz`、`selected_mesh.json`、`tracking_metrics.json` 与 `foundationpose_candidates/`；schema version `1.0`
阻塞：当前 object/mask timeline 仅 3 帧，不能宣称 M2；等待完整 90-frame SAM 3 masks
下一步：保持 selected proposal 自动策略不变，重跑 90-frame bidirectional tracking

日期：2026-07-30
工作包：WP2
状态：in_progress
修改：`video_to_spider/adapters/sam3.py` 在 anchor 自动排名后移除未选 object，只传播 selected coin；hand session cap 为 2、selected object propagation cap 为 1
验证：`conda run -n v2s-core python -m py_compile video_to_spider/adapters/sam3.py`、`python -m pytest -q` -> `22 passed`、`git diff --check`；单次 28.4GB admission window 启动后共享 Ray worker 立即回收约 15GB，anchor hand grounding OOM，证明必须使用连续采样 admission gate
artifact：现有 validated `segmentation/` 保持不变；失败证据位于 `segmentation_lowmem_smoke/failure.json`，完整目标为 `segmentation_full/`
阻塞：GPU 4–7 的共享 Ray workload 当前使空闲显存在约 8–28GB 间快速变化，尚无持续安全窗口
下一步：先在独立目录完成 low-memory 3-frame regression，再运行并验证 90-frame artifact 后安全 promote

日期：2026-07-30
工作包：WP9
状态：in_progress
修改：同步统一报告对真实 WP5 与 3-frame WP6 artifacts 的可用性
验证：`evaluate-run` 当前报告 8/12 stages available、缺少 optimization/aligned/contact/spider、`m4_complete=false`；`conda run -n v2s-core python -m pytest -q` -> `22 passed`
artifact：`runs/egodex_flip_coin_0_f000_090/evaluation/unified_run_report.json`，schema version `1.0`
阻塞：完整 WP2/WP6 timeline 与 WP7/真实 SPIDER artifacts 尚未产生
下一步：M4 全链完成后重跑并要求 `m4_complete=true`

### 2026-07-29

- 新增 `IMPLEMENTATION_PLAN.md` 的固定数据根目录：`/data_all/share/datasets/egodex`。
- 确认当前数据目录约 336G，包含 1 个 `part`、26 个任务目录和 46,234 组同 stem MP4/HDF5；两类缺失对端均为 0。
- 移除人工 prompt、人工点/框/mask、人工选帧、人工 mesh 选择和人工抽检要求。
- 将 SAM 3 目标分割固定为 HDF5 任务文字指令关键词驱动的 text-only 流程。
- 新增自动 instruction parser、关键词候选评分、失败原因码和程序化验收规则。
- 新增本进度文档，初始所有实现工作包为 `pending`。
- 修复 WP5/WP6 依赖环：WP5 输出静态排序 proposals，WP6 用 tracking score 选择最终 mesh；WP8 改为消费 WP6 selected mesh。
- 移除 A/B/C/D 正式消融，只保留不重复感知推理的 raw-vs-aligned 序列优化诊断；oracle hand 仅作为显式失败排障，不进入主结果或验收。
- 在 `video_to_spider/` 初始化独立 Git 仓库和 `main` 分支；新增 `.gitignore`，排除独立管理的 `third_party/`、模型权重、运行输出和本地缓存。
- 更新实际环境映射：SPIDER 固定使用 `uv` 管理的 `../spider/.venv`，其余阶段使用 `v2s-*` Conda 环境。
- 核验 SAM 3、SAM 3D Objects、WiLoR、Depth Anything metric 和 FoundationPose 权重主文件均已存在；移除旧的权重下载阻塞。
- `v2s-core`、`v2s-sam3`、`v2s-wilor`、`v2s-depth`、`v2s-foundationpose`、`v2s-opt` 和 SPIDER 完成核心 import preflight；发现 `v2s-sam3d` 缺少 `torch`，保留为环境阻塞。
- 当前节点不可见 NVIDIA driver，checkpoint load、GPU inference 和 MJWP 仍待 GPU 节点验证。

## Agent 更新格式

每个 agent 开始工作、状态变化、产生交付物或发现/解除阻塞时，必须同时更新上方对应工作包/波次/阻塞表，并在本文件中追加一条记录。不得把聊天消息或最终答复当作进度更新的替代品。

```text
日期：YYYY-MM-DD
工作包：WPx
状态：pending | in_progress | blocked | done | failed
修改：文件路径列表
验证：命令及结果
artifact：最小输出路径和 schema version
阻塞：没有则写 none
下一步：一句话
```

不得只更新状态而不记录验证命令和 artifact 路径。
