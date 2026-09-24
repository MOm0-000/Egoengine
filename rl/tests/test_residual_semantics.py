"""Unit tests for requested versus actuator-range-effective residuals."""

from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_spider.rl.residual_semantics import (
    control_target_residuals,
    ctrlrange_contract,
    residual_from_trace_step,
)


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <joint name="limited_joint" type="slide"/>
              <joint name="unlimited_joint" type="hinge"/>
              <geom type="sphere" size="0.01" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <position joint="limited_joint" ctrllimited="true" ctrlrange="-1 1"/>
            <position joint="unlimited_joint" ctrllimited="false"/>
          </actuator>
        </mujoco>
        """
    )


def test_control_target_residuals_respect_enabled_ctrlrange():
    model = _model()
    reference = np.array([[0.9, 2.0], [-2.0, 3.0]], np.float64)
    requested_ctrl = np.array([[1.2, 2.5], [-1.7, 2.0]], np.float64)
    terms = control_target_residuals(model, reference, requested_ctrl)

    np.testing.assert_allclose(
        terms["requested_residual"], [[0.3, 0.5], [0.3, -1.0]]
    )
    np.testing.assert_allclose(
        terms["effective_residual_after_ctrlrange"], [[0.1, 0.5], [0.0, -1.0]]
    )
    np.testing.assert_allclose(
        terms["requested_residual"],
        terms["effective_residual_after_ctrlrange"]
        + terms["residual_lost_to_ctrlrange"],
        rtol=0,
        atol=np.finfo(np.float64).eps,
    )
    assert terms["ctrllimited"].tolist() == [True, False]


def test_control_target_residuals_do_not_clamp_when_model_disables_it():
    model = _model()
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    reference = np.array([0.9, 2.0])
    requested_ctrl = np.array([1.2, 2.5])
    terms = control_target_residuals(model, reference, requested_ctrl)

    np.testing.assert_array_equal(
        terms["effective_residual_after_ctrlrange"], terms["requested_residual"]
    )
    np.testing.assert_array_equal(terms["residual_lost_to_ctrlrange"], [0.0, 0.0])
    contract = ctrlrange_contract(model, (0, 1))
    assert contract["control_clamping_enabled"] is False
    assert contract["ctrllimited"] == [True, False]


def test_trace_reader_preserves_legacy_v2_meaning():
    current = {
        "requested_residual": [0.1, -0.2],
        "effective_residual_after_ctrlrange": [0.05, -0.2],
    }
    legacy = {"applied_residual": [0.1, -0.2]}
    np.testing.assert_array_equal(residual_from_trace_step(current), [0.1, -0.2])
    np.testing.assert_array_equal(residual_from_trace_step(legacy), [0.1, -0.2])
    np.testing.assert_array_equal(
        residual_from_trace_step(current, effective=True), [0.05, -0.2]
    )
    with pytest.raises(ValueError, match="legacy trace"):
        residual_from_trace_step(legacy, effective=True)
