import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_taco_replay_rl import load_accepted_initialization


def test_old_posture_diagnostic_cannot_be_loaded_as_accepted_reset(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(dict(candidate_declared_state_feasible=True, accepted_as_reset=False)))
    with pytest.raises(ValueError, match="not passed"):
        load_accepted_initialization(path, ROOT / "configs/taco_pour_bimanual_ppo.yaml")
