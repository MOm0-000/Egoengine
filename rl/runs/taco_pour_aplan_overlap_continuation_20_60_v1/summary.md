# A_PLAN overlap continuation 20→60 v1

Classification: **COMPLETE_TO_60_WITH_OVERLAP_QUALITY_TRADEOFF**.

- The archived 93.75% projection statistic mixed float32 roundoff with real support clipping. Exact reconstruction of all 1,024 parent candidates gives noisy-slot support-clip fractions of `3.329950%` for A_PLAN and `2.979167%` for L_PLAN; candidate bytes and winner order are unchanged.
- Parent 0→40 replay parity: `True`.
- Parent A_PLAN source0–19 actions and the uninterrupted physical prefix through endpoint20 are bitwise preserved; the captured full s20 is a new isolated candidate boundary, not a committed state.
- Frozen-suffix baseline: terminal endpoint `60`, first failure `None`.
- OVERLAP_PLAN: terminal endpoint `60`, completed to 60 `True`.
- 20→40 quality preserved under frozen guards: `False`. Relative to parent A_PLAN, 20→40 RMS improves target position/rotation and pair translation, but worsens tool position by `0.008707 m`, tool rotation by `0.042170 rad`, and pair rotation by `0.019187 rad`.
- At endpoint60 the formal tracking score is `0.988889` for the frozen-suffix baseline and `0.951798` for OVERLAP_PLAN. OVERLAP_PLAN improves both object positions, target rotation, and pair translation there, but worsens tool rotation by `0.046458 rad` and pair rotation by `0.013921 rad`.
- Cold replay bitwise parity: `True`.
- Physics substeps: `163762` / `169000`.
- Visual review: complete. Both trajectories are coherent to endpoint60 with no obvious explosion or floor penetration, but OVERLAP_PLAN shows a tool-rotation/angular-speed trade-off (maximum tool angular speed `6.178 rad/s`); promotion/chunk commit/RL remain unauthorized.
- One initial setup invocation used a nonexistent repository-relative overlay and stopped before world construction with zero physics steps. The retained retry record identifies the corrected pinned server overlay; the single physical attempt then completed normally.

Endpoint 60 is the hard stop. This result does not certify endpoint 61, endpoint 80, or the full 198-frame task.
