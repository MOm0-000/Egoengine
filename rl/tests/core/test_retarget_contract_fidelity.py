from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from egoengine_repro.retarget.contract_fidelity import (
    FINGERS,
    HUMAN_CHAINS,
    ROBOT_CHAINS,
    deterministic_probe_arrays,
    direct_directions,
    finite_difference_jacobian,
    finger_probes,
    human_chain,
    semantic_landmark_candidates,
    wrist_z_probes,
)


ROOT = Path(__file__).resolve().parents[2]


def _hand_model() -> mujoco.MjModel:
    bodies = []
    for offset, (finger, names) in enumerate(ROBOT_CHAINS.items()):
        joint1, joint2, body1, body2, tip = names
        bodies.append(f"""
          <body name="{body1}" pos="0 {0.01 * offset:.3f} 0">
            <joint name="{joint1}" axis="0 1 0" range="-1 1"/>
            <geom type="capsule" size=".002 .01" pos=".01 0 0" euler="0 1.5708 0"/>
            <body name="{body2}" pos=".02 0 0">
              <joint name="{joint2}" axis="0 0 1" range="-1 1"/>
              <geom type="capsule" size=".002 .01" pos=".01 0 0" euler="0 1.5708 0"/>
              <site name="{tip}" pos=".02 0 0"/>
            </body>
          </body>""")
    xml = "".join(bodies)
    return mujoco.MjModel.from_xml_string(f"""
      <mujoco>
        <worldbody>
          <body name="left_hand_link">
            <site name="left_palm"/>
            {xml}
          </body>
        </worldbody>
      </mujoco>
    """)


def test_contract_is_strictly_zero_runtime_and_has_one_final_classification_set():
    value = yaml.safe_load(
        (ROOT / "configs/taco_pour_retarget_contract_fidelity_audit_v1.yaml").read_text()
    )
    assert value["schema"] == "taco_pour_retarget_contract_fidelity_audit_v1"
    assert not any(value["authorization"].values())
    assert value["landmarks"]["preferred"] == "joint_anchor"
    assert value["frames"]["semantic_debug"] == [0, 14, 15, 16, 17, 20]
    assert set(value["final_classifications"]) == {
        "OBJECTIVE_AND_GEOMETRY_CONTRACT_CERTIFIED",
        "SEMANTIC_LANDMARK_BLOCKER",
        "OBJECTIVE_FIDELITY_BLOCKER",
        "COLLISION_PROXY_SEMANTICS_BLOCKER",
        "MULTIPLE_UPSTREAM_BLOCKERS",
        "FUNCTIONAL_ERROR",
    }


def test_full_human_chain_retains_dip_and_direct_segments():
    joints = np.arange(63, dtype=np.float64).reshape(21, 3)
    for finger in FINGERS:
        chain = human_chain(joints, finger)
        assert chain.shape == (4, 3)
        assert np.array_equal(chain, joints[list(HUMAN_CHAINS[finger])])
    proximal, distal = direct_directions(
        np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0]], dtype=float)
    )
    assert np.array_equal(proximal, [1.0, 0.0, 0.0])
    assert np.array_equal(distal, [0.0, 1.0, 0.0])


def test_joint_anchor_map_and_finite_difference_are_finger_local():
    model = _hand_model()
    mapping = semantic_landmark_candidates(model)
    assert mapping["fingers"]["ring"]["human_full_chain_indices"] == [13, 14, 15, 16]
    assert mapping["fingers"]["pinky"]["human_full_chain_indices"] == [17, 18, 19, 20]
    jacobian = finite_difference_jacobian(
        model, np.zeros(model.nq), "left_hand_ring_joint1", 1e-4
    )
    assert np.linalg.norm(jacobian["ring"]) > 1e-3
    assert np.linalg.norm(jacobian["pinky"]) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(jacobian["wrist"]) == pytest.approx(0.0, abs=1e-12)


def test_deterministic_probe_library_has_frozen_cardinality_and_coordinates():
    model = _hand_model()
    old = np.zeros((21, model.nq), dtype=np.float64)
    probes = finger_probes(model, old, [15, 16, 17], [-0.05, -0.025, 0, 0.025, 0.05])
    assert len(probes) == 60
    for probe in probes:
        changed = np.flatnonzero(probe.qpos != old[probe.endpoint])
        assert len(changed) <= 1
        assert probe.metadata["clipped"] is False

    accepted = old.copy()
    accepted[0, 0] = 0.2
    z_probes = wrist_z_probes(old, accepted, [0, 0.25, 0.5, 0.75, 1.0], address=0)
    assert len(z_probes) == 10
    packed = deterministic_probe_arrays(probes + z_probes)
    assert packed["qpos"].shape == (70, model.nq)
    assert len(np.unique(packed["probe_id"])) == 70


def test_audit_runner_contains_no_simulator_step_or_optimizer_call():
    source = (ROOT / "scripts/audit_taco_pour_retarget_contract_fidelity.py").read_text()
    assert "mj_step(" not in source
    assert ".step(" not in source
    assert "scipy.optimize" not in source
    assert "minimize(" not in source
    assert "optimizer_generated_candidates\": 0" in source
    assert "stop_after_semantic_blocker" in source
