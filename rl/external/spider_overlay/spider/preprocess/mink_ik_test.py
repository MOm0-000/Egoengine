import mujoco
import mink
import numpy as np

from spider.preprocess.mink_ik import (
    ExactSignedDistanceLimit,
    GeometryConstraint,
    SignedDistanceRecoveryTask,
    _exact_joint_limit_violations,
    _geometric_frame,
    _initialize_floor_clearance,
    _mink_residual_scale,
    _site_local_fingertip_frames,
    interpolate_pose7,
    moving_average_configurations,
    evaluate_qref_gate,
    project_configuration_collisions,
    qref_refinement_eligibility,
)


def test_geometric_frame_uses_normal_as_x_and_distal_direction_as_z():
    frame = _geometric_frame(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 2.0]),
    )

    np.testing.assert_allclose(frame, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(frame), 1.0, atol=1e-8)


def test_floor_clearance_initialization_repairs_penetrating_free_body():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom name='floor' type='plane' size='1 1 .1'/>"
        "<body pos='0 0 -.05'><freejoint/><geom name='collision_hand_right_palm_0' "
        "size='.02'/></body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)

    lift = _initialize_floor_clearance(model, data, 0.001)

    assert lift > 0.07
    distance = mujoco.mj_geomDistance(model, data, 1, 0, 1.0, None)
    assert distance >= 0.001


def test_site_local_fingertip_frames_recover_neutral_geometric_frames():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><site name='right_palm'/>"
        + "".join(
            f"<body name='{finger}'><geom size='.01'/><site "
            f"name='right_{finger}_tip' pos='0 0 .1'/></body>"
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        )
        + "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    local_frames = _site_local_fingertip_frames(model)

    np.testing.assert_allclose(
        np.swapaxes(local_frames, -1, -2) @ local_frames,
        np.broadcast_to(np.eye(3), local_frames.shape),
        atol=1e-8,
    )
    np.testing.assert_allclose(np.linalg.det(local_frames), 1.0, atol=1e-8)


def test_objective_coefficient_maps_to_sqrt_mink_residual_scale():
    assert _mink_residual_scale(0.1, "lambda_w") == np.sqrt(0.1)
    assert _mink_residual_scale(1.0, "tip") == 1.0


def test_paper_tip_pose_uses_position_and_orientation_without_suppressing_rotation():
    import inspect

    signature = inspect.signature(__import__(
        "spider.preprocess.mink_ik", fromlist=["main"]
    ).main)

    assert signature.parameters["finger_position_cost"].default == 1.0
    assert signature.parameters["finger_orientation_cost"].default == 1e-3
    assert signature.parameters["lambda_w"].default == 7.5e-3


def test_signed_distance_recovery_jacobian_matches_penetration_finite_difference():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom name='floor' type='plane' size='1 1 .1'/>"
        "<body pos='0 0 .015'><freejoint/><geom name='ball' size='.02'/>"
        "</body></worldbody></mujoco>"
    )
    configuration = mink.Configuration(model)
    task = SignedDistanceRecoveryTask(0, 1, 0.001, 1.0)
    error = task.compute_error(configuration)
    analytic = task.compute_jacobian(configuration)[:, 2]
    epsilon = 1e-7
    velocity = np.zeros(model.nv)
    velocity[2] = 1.0
    mujoco.mj_integratePos(model, configuration.data.qpos, velocity, epsilon)
    configuration.update()
    numeric = (task.compute_error(configuration) - error) / epsilon

    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_discrete_projection_recovers_existing_floor_penetration():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom name='floor' type='plane' size='1 1 .1'/>"
        "<body pos='0 0 .015'><freejoint/><geom name='ball' size='.02'/>"
        "</body></worldbody></mujoco>"
    )
    configuration = mink.Configuration(model)
    distance_limit = mink.CollisionAvoidanceLimit(
        model,
        [(["floor"], ["ball"])],
        gain=0.2,
        minimum_distance_from_collisions=0.001,
        collision_detection_distance=0.1,
        bound_relaxation=0.0,
    )

    result = project_configuration_collisions(
        configuration,
        [GeometryConstraint("hand_floor", 0, 1, 0.001)],
        mink.DampingTask(model, cost=1e-3),
        [mink.ConfigurationLimit(model), distance_limit],
        [],
        sim_dt=0.005,
        solver="daqp",
        damping=1e-5,
        maximum_iterations=30,
        tolerance_m=1e-6,
        safety_margin_m=5e-5,
        recovery_cost=10.0,
        recovery_gain=0.5,
    )

    assert result["accepted"]
    assert result["initial_violation_count"] == 1
    assert result["iterations"] > 0
    distance = mujoco.mj_geomDistance(model, configuration.data, 0, 1, 1.0, None)
    assert distance >= 0.001 - 1e-6


def test_exact_distance_limit_forbids_worsening_existing_penetration():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom name='floor' type='plane' size='1 1 .1'/>"
        "<body pos='0 0 .015'><freejoint/><geom name='ball' size='.02'/>"
        "</body></worldbody></mujoco>"
    )
    configuration = mink.Configuration(model)
    limit = ExactSignedDistanceLimit(
        [GeometryConstraint("hand_floor", 0, 1, 0.001)], gain=0.2,
    )

    inequality = limit.compute_qp_inequalities(configuration, 0.005)

    assert inequality.G is not None and inequality.h is not None
    np.testing.assert_allclose(inequality.G[0, 2], -1.0, atol=1e-8)
    np.testing.assert_allclose(inequality.h[0], 0.0, atol=1e-8)


def test_exact_joint_limit_validation_reports_numerical_drift():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><joint name='hinge' range='-1 1'/>"
        "<geom size='.01'/></body></worldbody></mujoco>"
    )
    qpos = np.asarray([1.0 + 2e-8])

    violations = _exact_joint_limit_violations(model, qpos, tolerance=1e-8)

    assert len(violations) == 1
    assert violations[0]["joint"] == "hinge"


def test_moving_average_renormalizes_free_joint_quaternion():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint name='object'/><geom size='0.1'/></body>"
        "</worldbody></mujoco>"
    )
    values = np.zeros((3, model.nq))
    values[:, 3] = [1.0, 0.8, 0.6]
    values[:, 4] = [0.0, 0.2, 0.4]

    filtered = moving_average_configurations(model, values, 3)

    np.testing.assert_allclose(np.linalg.norm(filtered[:, 3:7], axis=1), 1.0)


def test_interpolate_pose7_uses_shortest_normalized_quaternion_arc():
    first = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    second = np.array([2.0, 0.0, 0.0, -0.70710678, 0.0, 0.0, -0.70710678])

    middle = interpolate_pose7(first, second, 0.5)

    np.testing.assert_allclose(middle[:3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(middle[3:]), 1.0)
    assert middle[3] > 0.0


def test_qref_gate_rejects_non_distal_object_penetration():
    trajectory = {
        "fingertip_position_error_m": {"p95": 0.005},
        "fingertip_orientation_error_rad": {"p95": 0.2},
        # Direction is a diagnostic subset; the full SO(3) metric owns the gate.
        "fingertip_direction_error_rad": {"p95": 2.0},
        "wrist_orientation_error_rad": {"p95": 0.05},
    }
    geometry = {
        "joint_limits": {"violation_frame_count": 0},
        "self_collision": {"minimum_signed_distance_m": 0.0},
        "hand_floor": {"minimum_signed_distance_m": 0.0},
        "non_distal_object": {"minimum_signed_distance_m": -0.01},
        "distal_object": {"minimum_signed_distance_m": -0.001},
    }

    gate = evaluate_qref_gate(
        trajectory, geometry,
        non_distal_object_clearance_m=0.001,
        distal_object_max_penetration_m=0.0025,
        floor_clearance_m=0.0,
        paper_only_constraints=False,
    )

    assert not gate["accepted"]
    assert gate["failed_checks"] == ["non_distal_object_collision"]
    eligibility = qref_refinement_eligibility(gate)
    assert not eligibility["eligible"]
    assert eligibility["hard_failures"] == ["non_distal_object_collision"]


def test_paper_qref_gate_keeps_scene_geometry_diagnostic_only():
    trajectory = {
        "fingertip_position_error_m": {"p95": 0.005},
        "fingertip_orientation_error_rad": {"p95": 0.2},
        "wrist_orientation_error_rad": {"p95": 0.05},
    }
    geometry = {
        "joint_limits": {"violation_frame_count": 0},
        "self_collision": {"minimum_signed_distance_m": 0.0},
        "hand_floor": {"minimum_signed_distance_m": -0.02},
        "non_distal_object": {"minimum_signed_distance_m": -0.01},
        "distal_object": {"minimum_signed_distance_m": -0.01},
    }

    gate = evaluate_qref_gate(
        trajectory, geometry,
        non_distal_object_clearance_m=0.001,
        distal_object_max_penetration_m=0.0025,
        floor_clearance_m=0.0,
    )

    assert gate["accepted"]
    assert set(gate["checks"]) == {
        "fingertip_position_fidelity", "fingertip_orientation_fidelity",
        "wrist_orientation_fidelity", "joint_limits", "self_collision",
    }
    assert gate["scene_geometry_diagnostics"]["floor_collision"] is False


def test_fidelity_failure_can_remain_refinement_eligible():
    gate = {
        "checks": {
            "fingertip_position_fidelity": False,
            "fingertip_orientation_fidelity": False,
            "wrist_orientation_fidelity": True,
            "joint_limits": True,
            "self_collision": True,
            "floor_collision": True,
            "non_distal_object_collision": True,
            "distal_object_penetration": True,
        }
    }

    eligibility = qref_refinement_eligibility(gate)

    assert eligibility["eligible"]
    assert eligibility["hard_failures"] == []
    assert eligibility["fidelity_warnings"] == [
        "fingertip_position_fidelity",
        "fingertip_orientation_fidelity",
    ]
