from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from egoengine_repro.retarget.support_plane_limit import (
    build_native_support_geoms,
    geom_world_vertices,
    native_hand_visual_geom_ids,
)
from egoengine_repro.retarget.trajectory_refinement import (
    apply_delta,
    differentiate_trajectory,
    file_sha256,
    immutable_clearance_blockers,
    interpolate_qpos,
    interpolated_support_linearization,
    path_minimum_clearance,
    resolve_variables,
    validate_scalar_joint_ranges,
)


REPO = Path(__file__).resolve().parents[3]
SCENE = REPO / "rl/runs/taco_brush_20231006_145_safe_entry_mink_v1/generated_scene.xml"
REFERENCE = (
    REPO
    / "rl/runs/taco_brush_20231006_145_safe_entry_mink_v1/safe_entry/robot_reference.npz"
)
CONFIG = (
    REPO
    / "rl/configs/taco_brush_20231006_145_continuous_reference_refinement_v1.yaml"
)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE))


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


@pytest.fixture(scope="module")
def geoms(model):
    return build_native_support_geoms(
        model, mujoco, native_hand_visual_geom_ids(model, mujoco),
    )


def m1_variables(model):
    names = ("left_hand_pinky_joint1", "left_hand_pinky_joint2")
    return resolve_variables(
        model,
        endpoints=(5, 6, 7, 8),
        joint_names=names,
        budgets={name: 0.05 for name in names},
    )


def test_resolved_variables_are_only_authorized(model):
    variables = m1_variables(model)
    assert len(variables) == 8
    assert {item.endpoint for item in variables} == {5, 6, 7, 8}
    assert {item.joint_name for item in variables} == {
        "left_hand_pinky_joint1", "left_hand_pinky_joint2",
    }
    assert all(item.qpos_address < 36 for item in variables)


def test_duplicate_candidate_definition_is_rejected(model):
    with pytest.raises(ValueError, match="unique"):
        resolve_variables(
            model,
            endpoints=(5, 5),
            joint_names=("left_hand_pinky_joint1",),
            budgets={"left_hand_pinky_joint1": 0.05},
        )


def test_apply_delta_preserves_fixed_boundaries_and_objects(model, reference):
    variables = m1_variables(model)
    delta = np.linspace(-0.01, 0.01, len(variables))
    result = apply_delta(reference["qpos"], variables, delta)
    assert np.array_equal(result[:5], reference["qpos"][:5])
    assert np.array_equal(result[9:], reference["qpos"][9:])
    assert np.array_equal(result[:, 36:], reference["qpos"][:, 36:])
    changed = np.argwhere(result != reference["qpos"])
    allowed = {(item.endpoint, item.qpos_address) for item in variables}
    assert {tuple(value) for value in changed} <= allowed


def test_modification_budget_is_hard(model, reference):
    variables = m1_variables(model)
    delta = np.zeros(len(variables))
    delta[0] = 0.0500001
    with pytest.raises(ValueError, match="budget"):
        apply_delta(reference["qpos"], variables, delta)


def test_mujoco_interpolation_preserves_frozen_object_channel(model, reference):
    q = interpolate_qpos(
        model, mujoco, reference["qpos"], 5, 1.0, freeze_dofs_from=36,
    )
    assert np.allclose(q[:36], reference["qpos"][6, :36], atol=1e-12, rtol=0.0)
    assert np.array_equal(q[36:], reference["qpos"][5, 36:])


def test_interpolated_distance_jacobian_matches_finite_difference(
    model, reference, geoms,
):
    variables = m1_variables(model)
    geom = next(item for item in geoms if item.name == "left_pinky_link2_visual")
    distance, gradient, _ = interpolated_support_linearization(
        model, mujoco, reference["qpos"], variables, geom,
        source_endpoint=6, alpha=0.4, plane_z_m=0.72,
    )
    index = next(
        i for i, item in enumerate(variables)
        if item.endpoint == 6 and item.joint_name == "left_hand_pinky_joint1"
    )
    epsilon = 1e-7
    delta = np.zeros(len(variables))
    delta[index] = epsilon
    perturbed = apply_delta(reference["qpos"], variables, delta)
    moved, _, _ = interpolated_support_linearization(
        model, mujoco, perturbed, variables, geom,
        source_endpoint=6, alpha=0.4, plane_z_m=0.72,
    )
    finite_difference = (moved - distance) / epsilon
    assert finite_difference == pytest.approx(gradient[index], abs=2e-6)


def test_nonlinear_distance_is_recomputed_after_update(model, reference, geoms):
    variables = m1_variables(model)
    geom = next(item for item in geoms if item.name == "left_pinky_link2_visual")
    baseline, _, _ = interpolated_support_linearization(
        model, mujoco, reference["qpos"], variables, geom,
        source_endpoint=6, alpha=0.5, plane_z_m=0.72,
    )
    delta = np.zeros(len(variables))
    for index, variable in enumerate(variables):
        if variable.endpoint in (6, 7) and variable.joint_name == "left_hand_pinky_joint1":
            delta[index] = -0.01
    changed = apply_delta(reference["qpos"], variables, delta)
    actual, _, _ = interpolated_support_linearization(
        model, mujoco, changed, variables, geom,
        source_endpoint=6, alpha=0.5, plane_z_m=0.72,
    )
    assert actual > baseline


def test_joint_range_violation_is_rejected(model, reference):
    variables = m1_variables(model)
    invalid = reference["qpos"].copy()
    variable = variables[0]
    invalid[variable.endpoint, variable.qpos_address] = (
        float(model.jnt_range[variable.joint_id, 1]) + 0.01
    )
    with pytest.raises(ValueError, match="range"):
        validate_scalar_joint_ranges(model, invalid, variables)


def test_qvel_derivation_uses_mujoco_position_difference(model, reference):
    derived = differentiate_trajectory(
        model, mujoco, reference["qpos"], reference["timestamps_s"],
    )
    assert derived.shape == (122, 48)
    assert np.isfinite(derived).all()
    assert np.allclose(derived, reference["qvel"], atol=1e-12, rtol=0.0)


def test_nonmonotonic_timestamps_are_rejected(model, reference):
    timestamps = reference["timestamps_s"].copy()
    timestamps[6] = timestamps[5]
    with pytest.raises(ValueError, match="strictly increasing"):
        differentiate_trajectory(model, mujoco, reference["qpos"], timestamps)


def test_m1_has_exact_immutable_right_hand_blockers(model, reference, geoms):
    blockers = immutable_clearance_blockers(
        model, mujoco, reference["qpos"], m1_variables(model), geoms,
        endpoints=(5, 6, 7, 8), plane_z_m=0.72,
        minimum_clearance_m=0.0005,
    )
    right = [row for row in blockers if row["geom"].startswith("right_")]
    assert {(row["endpoint"], row["body"]) for row in right} == {
        (7, "right_hand_pinky_link2"),
        (8, "right_hand_pinky_link2"),
    }
    assert all(row["candidate_influencing_dofs"] == [] for row in right)


def test_already_safe_clearance_has_no_immutable_blocker(model, reference, geoms):
    blockers = immutable_clearance_blockers(
        model, mujoco, reference["qpos"], m1_variables(model), geoms,
        endpoints=(5, 6, 7, 8), plane_z_m=0.72,
        minimum_clearance_m=-1.0,
    )
    assert blockers == []


def test_each_frozen_candidate_is_blocked_before_qp(model, reference, geoms):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for candidate in cfg["candidate_order"]:
        budgets = {}
        for name in candidate["authorized_joints"]:
            if name.startswith("L_forearm_t"):
                budgets[name] = 0.002
            elif name.startswith("L_forearm_"):
                budgets[name] = np.deg2rad(1.0)
            else:
                budgets[name] = 0.05
        variables = resolve_variables(
            model,
            endpoints=cfg["modified_endpoints"],
            joint_names=candidate["authorized_joints"],
            budgets=budgets,
        )
        blockers = immutable_clearance_blockers(
            model, mujoco, reference["qpos"], variables, geoms,
            endpoints=cfg["geometry_contract"]["core_endpoints"],
            plane_z_m=cfg["geometry_contract"]["table_z_m"],
            minimum_clearance_m=cfg["geometry_contract"]["core_minimum_clearance_m"],
        )
        assert any(row["geom"].startswith("right_") for row in blockers)


def test_dense_checker_detects_safe_endpoints_and_unsafe_midpoint():
    xml = """
    <mujoco>
      <asset>
        <mesh name="tetra" vertex="0 0 -1  .01 0 -.99  -.01 0 -.99  0 .01 -.99"/>
      </asset>
      <worldbody>
        <body name="left_test_body" pos="0 0 1">
          <joint name="hinge" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom name="left_test_visual" type="mesh" mesh="tetra"/>
        </body>
      </worldbody>
    </mujoco>
    """
    synthetic = mujoco.MjModel.from_xml_string(xml)
    synthetic_geoms = build_native_support_geoms(
        synthetic, mujoco, native_hand_visual_geom_ids(synthetic, mujoco),
    )
    qpos = np.asarray([[-1.0], [1.0]])
    data = mujoco.MjData(synthetic)
    endpoint_distances = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(synthetic, data)
        endpoint_distances.append(min(
            float(geom_world_vertices(data, geom, full=True)[:, 2].min())
            for geom in synthetic_geoms
        ))
    assert min(endpoint_distances) > 0.4
    minimum = path_minimum_clearance(
        synthetic, mujoco, qpos, synthetic_geoms,
        intervals=(0,), alphas=np.linspace(0.0, 1.0, 101), plane_z_m=0.0,
        freeze_dofs_from=None,
    )
    assert minimum["alpha"] == pytest.approx(0.5)
    assert minimum["distance_m"] < 1e-9


def test_existing_dense_path_minimum_is_negative(model, reference, geoms):
    minimum = path_minimum_clearance(
        model, mujoco, reference["qpos"], geoms,
        intervals=(6,), alphas=np.linspace(0.0, 1.0, 101), plane_z_m=0.72,
    )
    assert minimum["distance_m"] < -0.0001
    assert minimum["body"] == "left_hand_pinky_link2"


def test_original_reference_hash_is_not_modified():
    before = file_sha256(REFERENCE)
    with np.load(REFERENCE, allow_pickle=False) as archive:
        _ = np.asarray(archive["qpos"]).copy()
    assert file_sha256(REFERENCE) == before
    assert before == "9216e8e72ff824c5ddf5bfdf530212d214273e0a23df4e9433a89b38a25c39ea"
