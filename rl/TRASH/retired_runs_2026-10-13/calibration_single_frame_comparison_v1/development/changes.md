# Active calibration changes

| Change | Status | Reason |
|---|---|---|
| Remove project grouped ICP loop and deterministic face-index sampler | removed from active code | Single-frame Open3D must call official `registration_icp` once |
| Add `fit_single_surface_correction` | active | Allows one observation while preserving the frozen SciPy residual and solver settings |
| Keep historical run evidence | retained under `rl/runs/calibration_open3d_comparison_v1` | Historical facts remain immutable |
| Retire old comparison runner/config/tests | recoverable from commit `4e9ad4a8cfb593e6584b332f151939778416510e` | Prevent normal tooling from loading deleted grouped code |

Implementation and public evidence freeze: `0fe4e0b3cabd284563da418555761b6ddd0ca844`.
