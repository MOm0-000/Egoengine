# RL core known issues

| ID | Status | Current location / meaning |
| --- | --- | --- |
| B1 | closed at `a1622a4` | `core.ppo.prepare_batch` uses one value-RMS update/transform for old values and returns. |
| B2 | closed at `a1622a4` | `core.state_io.validate_physics_snapshot` and `MJWPVectorEnv.set_env_state` reject missing declared fields before mutation. |
| B3 | closed at `a1622a4` | independent critic freeze is authoritative in addition to any outer freeze. |
| S1 | open semantic decision | actor-internal clipped value loss remains centered on the external critic rollout value. R1 names and preserves this current local contract. |

R1 does not claim that S1 is paper-faithful, nor that bounded structural parity
establishes Pour task success. Training and chunk commit remain disabled.
