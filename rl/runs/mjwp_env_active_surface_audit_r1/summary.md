# MJWP environment active-surface audit R1

Status: `ACTIVE_PHYSICS_PATH_VERIFIED`

- `mjwp_env.py`: 1,465 -> 1,245 lines.
- Default active tests: `31 passed`.
- Offline actor parity: passed.
- Six real closed-loop anchors: bitwise equal.
- Two-epoch/four-independent-world collector: passed.
- Bounded cost: 560 control intervals / 5,600 physics steps.
- Training and chunk commit: disabled.

The removed code was orchestration/trace compatibility, not physics. Complete
snapshots, contacts, constraints, reward/observation construction and the
MuJoCo-Warp step sequence remain active.
