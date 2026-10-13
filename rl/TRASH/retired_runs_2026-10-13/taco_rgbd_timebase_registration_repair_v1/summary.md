# TACO RGB-D 时间轴与注册修复 v1

**正式分类：`RGBD_REGISTRATION_RESOLVED_DEPTH_OBSERVABILITY_LIMITED`。** 本轮没有运行桌面估计、MINK、physics、Replay、MPC、RL、promotion 或 chunk commit，也没有修改 active `SupportSurfaceContract`。

## 必答问题

1. **论文系统频率：** 相机系统和动捕系统都是 30 Hz；头戴设备是 Intel RealSense L515，egocentric 分辨率为 1920×1080。
2. **Depth AVI 为什么是 15 fps：** 官方 maintainer 公开的 FFV1 编码命令把输入 framerate 设置为 15；因此容器的 15 Hz 是官方编码流程留下的事实，不是本项目偶然改坏的 metadata。
3. **为什么 README 又写 fps=30：** README 用 FFmpeg `fps=30` filter 生成 30 Hz 输出；该 filter 会复制或删除真实帧，不只是改 metadata。
4. **本地文件来源：** Brush 本地文件是官方 release source 的重命名字节一致副本；四个预声明样本均有 release manifest/source 证据且 SHA/字节一致。
5. **Brush 官方 fps30 输出数：** `418` 帧，native 是 `209` 帧。
6. **复制关系：** exact raw-pixel hash 显示基本为 `native i -> output 2i, 2i+1`；完整逐帧候选映射见 `official_decode_frame_map.json`。
7. **native 209 帧是否已含重复 pair：** even-pair exact duplicate fraction 为 `0.000000`，不支持“当前文件已经先扩帧”的 H3。
8. **Brush 最可信 mapping：** `INDEX_ALIGNED`：native depth frame i 对 annotation row i；逻辑时间是 `i/30`，AVI PTS 不作为 annotation 逻辑时间，也不把 209 帧再次扩成 418 后硬配 pose。
9. **其他官方样本：** Pour、Skim、Smear 与 Brush 一样，RGB、annotation 和 native depth 计数逐行一致；official fps30 会约翻倍，container timestamp 只使用约前半 native data。四者都在官方 egocentric available list。
10. **低 coverage 的 bowl：** 被单独归类为 depth observability；只有高覆盖刚体 anchor 决定空间注册，低 coverage 不再单独判 registration fail。
11. **active scale1000 bug：** 未发现。active TACO metric-depth 路径是 `uint16 raw / 4000`；检出的 `/1000` 是非深度几何单位换算，历史 artifact/TRASH 未篡改。
12. **是否可重新开始桌面估计：** `可以在下一项独立任务中重新启动（本轮仍未执行）`。

## 长期合同

容器 fps 和 annotation logical fps 不是同一个概念。Depth valid coverage 与 RGB-D spatial registration 也不是同一个概念。后续代码必须使用候选时间合同，不得重写 AVI header、重复做 15→30 expansion、搜索 sample-specific offset，或用 bowl/brush bottom 修正深度。
