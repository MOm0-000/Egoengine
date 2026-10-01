from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_taco_pour_depth_registration import residual_record


def test_depth_residual_sign_is_raw_minus_rendered():
    rendered = np.full((20, 20), 0.6)
    observed = rendered + 0.005
    record, residual = residual_record(observed, rendered)
    assert np.median(residual) == pytest.approx(0.005)
    assert record["raw_minus_rendered_depth_mm_percentiles"][2] == pytest.approx(5.0)
