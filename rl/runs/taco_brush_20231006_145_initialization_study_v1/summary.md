# 20231006_145 initialization feasibility study

- Classification: `STATIC_INITIALIZATION_FEASIBLE_EARLY_PHYSICS_EXECUTION_INVALID_REPLAY_FAILED_CLOSED`
- Static initialization: `FEASIBLE WITH SAFE ENTRY`
- Controlled Replay: `FAILED CLOSED`
- Original data / active support modified: `NO / NO`
- RL / MPC / promotion: `NOT RUN / NOT RUN / FORBIDDEN`

## Ordered results

| gate | result | decisive evidence |
|---|---|---|
| 1. MANO penetration scope | local finger issue | frame 0: right thumb/middle/ring/pinky and left pinky; no palm/wrist region |
| 2. legal source start | frame 11 exists, but cropped | deletes frames 0–10 (0.367 s); cannot be called full reproduction |
| 3. DexTrack-style safe entry | PASS | 30 mm retreat, exact join at frame 11; safe-entry MINK `MINK_PASS` |
| 4. XHand constraints | initialization PASS | table minimum `-0.013 mm`; later selected interaction penetrations `29` |
| 5. controlled Replay | FAIL | brush pickup `False`, bowl pickup `True`, first tracking failure endpoint `35` |

## Key physical finding

The safe-entry **reference** starts legally, but the physical execution first violates the XHand/table native-geometry gate at endpoint `5`; the deepest sampled hand/table penetration is `-3.371 mm`. The full bounded Replay reached endpoint 121 without a MuJoCo hard stop, but reaching the horizon is not a pass.

The brush lifted only `4.087 mm` versus `108.933 mm` in the reference and never satisfied the stable hand-contact/no-floor pickup gate. The bowl did satisfy that local pickup gate for `6` consecutive intervals.

Therefore the source-frame initialization defect can be repaired kinematically without cropping, but the resulting low-level physical execution is not valid: early hand/table collision, brush tracking failure, missing brush pickup, native mesh blockers, and proxy/native mismatch remain. No RL is authorized from this evidence.
