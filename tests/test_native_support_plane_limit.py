from pathlib import Path
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
mink = pytest.importorskip("mink")
qpsolvers = pytest.importorskip("qpsolvers")
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl/src"))

from egoengine_repro.retarget.support_plane_limit import (
    NativeSupportPlaneLimit,
    build_native_support_geoms,
    independent_native_floor_rows,
    native_hand_visual_geom_ids,
)
from egoengine_repro.scene.support_surface import Plane


SCENE = Path(
    "/data_all/zzx/3.2RL/models/taco_xhand/xhand/bimanual/"
    "taco_brush_brush_bowl_20230927_027/scene_source_contacts_mass.xml"
)
BASELINE = Path(
    "rl/runs/taco_brush_brush_bowl_paper_baseline_v1/robot_reference.npz"
)


def model_and_qpos():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    with np.load(BASELINE, allow_pickle=False) as archive:
        qpos = archive["qpos"].copy()
    return model, qpos


def test_native_support_hull_matches_full_mesh_under_random_rigid_poses() -> None:
    model, _ = model_and_qpos()
    geoms = build_native_support_geoms(
        model, mujoco, native_hand_visual_geom_ids(model, mujoco)
    )
    rng = np.random.default_rng(17)
    for geom in geoms:
        for _ in range(4):
            rotation = Rotation.random(random_state=rng).as_matrix()
            translation = rng.normal(size=3)
            normal = rng.normal(size=3)
            normal /= np.linalg.norm(normal)
            full = (geom.full_vertices @ rotation.T + translation) @ normal
            hull = (geom.support_vertices @ rotation.T + translation) @ normal
            assert np.isclose(full.min(), hull.min(), atol=1e-12)


def test_native_support_normal_jacobian_matches_finite_difference() -> None:
    model, qpos = model_and_qpos()
    plane = Plane(normal=[0.19, -0.37, 0.91], offset=0.68, frame="simulator")
    ids = native_hand_visual_geom_ids(model, mujoco)
    limit = NativeSupportPlaneLimit(model, mujoco, plane, ids)
    rng = np.random.default_rng(23)
    for frame in rng.choice(len(qpos), size=4, replace=False):
        configuration = mink.Configuration(model, q=qpos[frame])
        rows = limit.rows(configuration)
        for row in rows[::7]:
            velocity = rng.normal(size=model.nv)
            velocity[36:] = 0.0
            velocity /= np.linalg.norm(velocity)
            analytic = float(row["normal_jacobian"] @ velocity)
            eps = 1e-7
            values = []
            for sign in (-1.0, 1.0):
                q = qpos[frame].copy()
                mujoco.mj_integratePos(model, q, velocity, sign * eps)
                probe = mink.Configuration(model, q=q)
                match = next(
                    candidate for candidate in limit.rows(probe)
                    if candidate["geom"].geom_id == row["geom"].geom_id
                )
                values.append(match["distance_m"])
            finite = (values[1] - values[0]) / (2.0 * eps)
            assert np.isclose(analytic, finite, atol=2e-6, rtol=2e-5)


def test_native_support_limit_accepts_tilted_plane() -> None:
    model, qpos = model_and_qpos()
    normal = Rotation.from_euler("xy", [0.31, -0.22]).apply([0.0, 0.0, 1.0])
    plane = Plane(normal=normal, offset=0.5, frame="tilted")
    limit = NativeSupportPlaneLimit(
        model, mujoco, plane, native_hand_visual_geom_ids(model, mujoco)
    )
    configuration = mink.Configuration(model, q=qpos[17])
    constraint = limit.compute_qp_inequalities(configuration, 1.0 / 240.0)
    assert constraint.G.shape == (26, model.nv)
    assert constraint.h.shape == (26,)
    assert np.isfinite(constraint.G).all() and np.isfinite(constraint.h).all()


def test_native_support_limit_actively_depenetrates() -> None:
    model, qpos = model_and_qpos()
    plane = Plane(normal=[0.0, 0.0, 1.0], offset=0.72, frame="simulator")
    limit = NativeSupportPlaneLimit(
        model, mujoco, plane, native_hand_visual_geom_ids(model, mujoco),
        minimum_clearance_m=0.0, gain=0.85, depenetration_step_m=0.002,
    )
    configuration = mink.Configuration(model, q=qpos[0])
    initial = limit.minimum_distance(configuration)
    assert initial < -0.018
    for _ in range(16):
        constraint = limit.compute_qp_inequalities(configuration, 1.0 / 240.0)
        delta = qpsolvers.solve_qp(
            np.eye(model.nv), np.zeros(model.nv), constraint.G, constraint.h,
            solver="daqp", primal_tol=1e-9, dual_tol=1e-9,
        )
        assert delta is not None
        configuration.integrate_inplace(delta, 1.0)
    final = limit.minimum_distance(configuration)
    assert final > initial + 0.018
    assert final >= -1e-8


def test_brush_baseline_frame0_native_limit_regression() -> None:
    model, qpos = model_and_qpos()
    plane = Plane(normal=[0.0, 0.0, 1.0], offset=0.72, frame="simulator")
    ids = native_hand_visual_geom_ids(model, mujoco)
    rows = independent_native_floor_rows(model, mujoco, plane, qpos[:1], ids)
    by_side = {
        side: min(row["minimum_signed_distance_m"] for row in rows
                  if row["side"] == side)
        for side in ("left", "right")
    }
    assert np.isclose(by_side["left"], -0.020878211733495577, atol=1e-12)
    assert np.isclose(by_side["right"], -0.01890884276181959, atol=1e-12)
