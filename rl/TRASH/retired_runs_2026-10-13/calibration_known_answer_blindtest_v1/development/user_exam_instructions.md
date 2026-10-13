# 用户主持的正式保密考试

正式考试**尚未执行**。在 Codex 无权访问的独立账号或机器上，固定仓库提交后执行：

```bash
PY=/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python
TOOL=rl/scripts/calibration_known_answer_blindtest_v1.py

# 1. 检查隔离（只使用假秘密）
$PY $TOOL isolation-demo --output /secure/audit/isolation_demo.json

# 2. 用户在私有环境生成题面和答案；seed 文件不得挂给 solver
$PY $TOOL generate --secret-seed-file /secure/private/seed.txt \
  --public-dir /secure/public/input --private-dir /secure/private/answer

# 3. 分别运行两个冻结版本；run-bwrap 只挂载只读题面、冻结代码和单独输出目录
$PY $TOOL run-bwrap --input-dir /secure/public/input --output-dir /secure/output/baseline --version baseline
$PY $TOOL run-bwrap --input-dir /secure/public/input --output-dir /secure/output/fixed --version fixed

# 4. 两份原始预测写完并改为只读后，用户才运行评分
$PY $TOOL score --input-dir /secure/public/input --private-dir /secure/private/answer \
  --baseline-dir /secure/output/baseline --fixed-dir /secure/output/fixed \
  --output-dir /secure/score

# 5. 校验公开输入、私有答案和原始输出中的哈希
$PY $TOOL verify --input-dir /secure/public/input --private-dir /secure/private/answer
```

不要把 `/secure/private`、用户主目录、开发工作区或网络挂入 solver。成绩公开前不要向 Codex提供题目、日志或中间分数。
