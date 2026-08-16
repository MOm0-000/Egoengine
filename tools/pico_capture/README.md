# PICO Capture Bridge

采集端与 Python EgoEngine 解耦。设备端应输出：

```text
recording/
├── manifest.json
├── calibration.json
├── timestamps.csv
├── left/
├── right/
├── head_pose.csv
└── hand_tracking.csv
```

所有 PICO camera / HMD / hand 能力当前为 `UNVERIFIED`。
