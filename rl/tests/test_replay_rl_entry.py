import json
from pathlib import Path
import sys

import pytest
import numpy as np
import yaml
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_taco_replay_rl import load_accepted_initialization


def test_old_posture_diagnostic_cannot_be_loaded_as_accepted_reset(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(dict(candidate_declared_state_feasible=True, accepted_as_reset=False)))
    with pytest.raises(ValueError, match="not passed"):
        load_accepted_initialization(path, ROOT / "configs/taco_pour_bimanual_ppo.yaml")


def _accepted_fixture(tmp_path, *, object_qvel_provenance="reference_finite_difference"):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>")
    qpos = np.arange(100, dtype=np.float32).reshape(2, 50) / 100
    qvel = np.arange(96, dtype=np.float32).reshape(2, 48) / 1000
    ctrl = qpos[:, :36].copy()
    reference = tmp_path / "reference.npz"
    np.savez(reference, qpos=qpos, qvel=qvel, ctrl=ctrl)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model_path": str(scene), "data_path": str(reference)}))
    contract = {
        "state_contract_version": "egoengine_replay_rl_initial_state_v1",
        "reference_index": 0,
        "hand_qpos_provenance": "local_procedure:collision_aware_retarget_v1",
        "hand_qvel_provenance": "reference_finite_difference",
        "object_qpos_provenance": "reference_endpoint_0",
        "object_qvel_provenance": object_qvel_provenance,
        "ctrl_provenance": "reference_endpoint_0",
        "first_command_reference_index": 1,
        "first_command_semantics": "reference_endpoint_target_plus_residual",
        "object_hold_method": "offline_fixed_object_then_release",
        "object_constraints_released": True,
        "release_timing": "before_accepted_snapshot",
        "post_release_validation_steps": 10,
    }
    initial = tmp_path / "initial.npz"
    np.savez(initial, qpos=qpos[0], qvel=qvel[0], ctrl=ctrl[0], **contract)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "accepted_for_replay_rl": True,
        "scene": {"path": str(scene.resolve()), "sha256": digest(scene)},
        "reference": {"path": str(reference.resolve()), "sha256": digest(reference)},
        "initial_state": {"path": str(initial.resolve()), "sha256": digest(initial)},
        "state_contract": contract,
        "release_validation": {
            "passed": True,
            "object_constraints_active_after_release": False,
            "steps": 10,
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    return report_path, config, initial


def test_complete_initial_state_contract_is_accepted(tmp_path):
    report, config, _ = _accepted_fixture(tmp_path)
    initial, provenance = load_accepted_initialization(report, config)
    assert initial["qpos"].shape == (50,)
    assert provenance["validated_state_contract"]["object_qvel_provenance"] == "reference_finite_difference"


def test_object_qvel_provenance_is_checked_against_values(tmp_path):
    report, config, initial_path = _accepted_fixture(tmp_path, object_qvel_provenance="zero")
    with pytest.raises(ValueError, match="provenance says zero"):
        load_accepted_initialization(report, config)


def test_formal_scene_cannot_retain_hold_constraints(tmp_path):
    report, config, _ = _accepted_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text())
    scene = Path(config_data["model_path"])
    scene.write_text("""<mujoco><worldbody><body name='right_object'><freejoint
        name='right_object_joint'/></body></worldbody><equality><weld
        body1='right_object'/></equality></mujoco>""")
    payload = json.loads(report.read_text())
    payload["scene"]["sha256"] = hashlib.sha256(scene.read_bytes()).hexdigest()
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="object hold constraint"):
        load_accepted_initialization(report, config)
