"""Exact command-target semantics for residual actions and actuator ranges."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def control_target_residuals(
    model: mujoco.MjModel,
    reference_ctrl: np.ndarray,
    requested_ctrl: np.ndarray,
) -> dict[str, Any]:
    """Separate requested, range-effective, and range-lost target offsets.

    ``effective`` describes the control target after MuJoCo's actuator-range
    clamp.  It is not realized qpos motion after servos, contact, and dynamics.
    """
    reference = np.asarray(reference_ctrl, dtype=np.float64)
    requested = np.asarray(requested_ctrl, dtype=np.float64)
    if reference.shape != requested.shape or reference.shape[-1] != model.nu:
        raise ValueError("control arrays must share a final dimension equal to model.nu")
    if not np.isfinite(reference).all() or not np.isfinite(requested).all():
        raise ValueError("control arrays must be finite")

    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    clamp_enabled = not bool(
        int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    )
    reference_after = reference.copy()
    requested_after = requested.copy()
    if clamp_enabled and limited.any():
        lower = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64)
        upper = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64)
        reference_after[..., limited] = np.clip(
            reference_after[..., limited], lower[limited], upper[limited]
        )
        requested_after[..., limited] = np.clip(
            requested_after[..., limited], lower[limited], upper[limited]
        )

    requested_residual = requested - reference
    effective_residual = requested_after - reference_after
    lost = requested_residual - effective_residual
    if not np.allclose(
        requested_residual,
        effective_residual + lost,
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    ):
        raise RuntimeError("residual decomposition identity failed")
    return {
        "requested_residual": requested_residual,
        "effective_residual_after_ctrlrange": effective_residual,
        "residual_lost_to_ctrlrange": lost,
        "reference_ctrl_after_ctrlrange": reference_after,
        "ctrl_after_ctrlrange": requested_after,
        "control_clamping_enabled": clamp_enabled,
        "ctrllimited": limited,
    }


def ctrlrange_contract(model: mujoco.MjModel, indices: tuple[int, ...]) -> dict[str, Any]:
    """Serialize the model fields that define control-target range semantics."""
    if any(index < 0 or index >= model.nu for index in indices):
        raise ValueError("actuator index is outside model.nu")
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)[list(indices)]
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)[list(indices)]
    clamp_enabled = not bool(
        int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    )
    return {
        "control_clamping_enabled": clamp_enabled,
        "model_disableflags": int(model.opt.disableflags),
        "ctrllimited": limited.tolist(),
        "ctrlrange": ranges.tolist(),
    }


def residual_from_trace_step(step: dict[str, Any], *, effective: bool = False) -> np.ndarray:
    """Read v3 trace semantics while preserving immutable v2 evidence."""
    if effective:
        if "effective_residual_after_ctrlrange" not in step:
            raise ValueError("legacy trace has no after-ctrlrange residual")
        return np.asarray(step["effective_residual_after_ctrlrange"], dtype=np.float64)
    if "requested_residual" in step:
        return np.asarray(step["requested_residual"], dtype=np.float64)
    if "applied_residual" in step:
        return np.asarray(step["applied_residual"], dtype=np.float64)
    raise ValueError("trace step contains no requested residual field")
