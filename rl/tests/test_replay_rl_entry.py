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
from video_to_spider.rl.physics_contract import (
    build_physics_contract,
    compile_mujoco_model,
)


def test_old_posture_diagnostic_cannot_be_loaded_as_accepted_reset(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(dict(candidate_declared_state_feasible=True, accepted_as_reset=False)))
    with pytest.raises(ValueError, match="not passed"):
        load_accepted_initialization(path, ROOT / "configs/taco_pour_bimanual_ppo.yaml")


def _accepted_fixture(tmp_path, *, object_qvel_provenance="reference_finite_difference"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mesh = tmp_path / "tetra.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco><compiler meshdir='.'/><asset><mesh name='tetra' file='tetra.obj'/></asset></mujoco>")
    qpos = np.arange(100, dtype=np.float32).reshape(2, 50) / 100
    qvel = np.arange(96, dtype=np.float32).reshape(2, 48) / 1000
    ctrl = qpos[:, :36].copy()
    reference = tmp_path / "reference.npz"
    np.savez(reference, qpos=qpos, qvel=qvel, ctrl=ctrl)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "model_path": str(scene), "data_path": str(reference), "simulator": "mjwp",
        "embodiment_type": "bimanual", "sim_dt": 1 / 300, "ctrl_dt": 1 / 30,
        "ref_dt": 1 / 30, "nconmax_per_env": 128, "njmax_per_env": 512,
    }))
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
    physics = build_physics_contract(config)
    report = {
        "accepted_for_replay_rl": True,
        "scene": {"path": str(scene.resolve()), "sha256": digest(scene)},
        "reference": {"path": str(reference.resolve()), "sha256": digest(reference)},
        "initial_state": {"path": str(initial.resolve()), "sha256": digest(initial)},
        "state_contract": contract,
        "physics_contract": physics,
        "release_validation": {
            "passed": True,
            "object_constraints_active_after_release": False,
            "requested_control_intervals": 10,
            "executed_control_intervals": 10,
            "physics_contract_sha256": physics["physics_contract_sha256"],
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


def test_release_validation_is_bound_to_config_and_external_assets(tmp_path):
    report, config, _ = _accepted_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text())
    config_data["sim_dt"] = 1 / 600
    config.write_text(yaml.safe_dump(config_data))
    with pytest.raises(ValueError, match="another physics contract"):
        load_accepted_initialization(report, config)

    report, config, _ = _accepted_fixture(tmp_path / "asset_case")
    mesh = config.parent / "tetra.obj"
    mesh.write_text(mesh.read_text() + "# same geometry, changed artifact\n")
    with pytest.raises(ValueError, match="another physics contract"):
        load_accepted_initialization(report, config)


def test_formal_scene_cannot_retain_hold_constraints(tmp_path):
    report, config, _ = _accepted_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text())
    scene = Path(config_data["model_path"])
    scene.write_text("""<mujoco><worldbody><body name='right_object'><freejoint
        name='right_object_joint'/><geom type='sphere' size='.01' mass='.01'/></body></worldbody><equality><weld
        body1='right_object'/></equality></mujoco>""")
    payload = json.loads(report.read_text())
    payload["scene"]["sha256"] = hashlib.sha256(scene.read_bytes()).hexdigest()
    physics = build_physics_contract(config)
    payload["physics_contract"] = physics
    payload["release_validation"]["physics_contract_sha256"] = physics["physics_contract_sha256"]
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="object hold constraint"):
        load_accepted_initialization(report, config)


def test_invalid_sdf_octree_depth_fails_closed_before_compilation(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>")
    with pytest.raises(ValueError, match="positive integers"):
        compile_mujoco_model(scene, {"object_mesh": 0})
