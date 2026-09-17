"""Regression probes are real Pour bowl points, not injected scene corruption."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from egoengine_repro.retarget.mesh_distance import (
    closed_mesh_signed_distance, mesh_surface_distance, solid_angle_winding,
)

ROOT = Path(__file__).resolve().parents[1]
PROBES = np.array([
    [.04342943660565967, -.005247433200225069, .008946142943865674],
    [-.02066273060940849, .008361496706464635, -.022597869751828602],
    [.01249252617858973, -.02136127091606624, -.022062839331967506],
    [.026707019546947975, -.02237269974360954, -.015230516581010193],
])


def test_real_bowl_false_outside_regression_and_batch_invariance():
    mesh = trimesh.load_mesh(ROOT / "models/taco_xhand/assets/objects/022/visual.obj", process=True)
    mesh.apply_scale(.01)
    np.testing.assert_allclose(solid_angle_winding(mesh, PROBES), [0, 1, 1, 1], atol=1e-10)
    signed = closed_mesh_signed_distance(mesh, PROBES)
    np.testing.assert_allclose(signed, [-.00567693, .00566523, .00593204, .00551311], atol=1e-6)
    single = np.concatenate([closed_mesh_signed_distance(mesh, p[None]) for p in PROBES])
    reversed_batch = closed_mesh_signed_distance(mesh, PROBES[::-1])[::-1]
    duplicated = closed_mesh_signed_distance(mesh, np.tile(PROBES, (17, 1))).reshape(17, 4)
    np.testing.assert_array_equal(single, signed)
    np.testing.assert_array_equal(reversed_batch, signed)
    np.testing.assert_array_equal(duplicated, np.tile(signed, (17, 1)))


def test_box_analytic_distance_empty_and_rigid_transform():
    mesh = trimesh.creation.box()
    points = np.array([[0, 0, 0], [.4, 0, 0], [.7, 0, 0], [.6, .6, .6]])
    expected = [.5, .1, -.2, -np.sqrt(3) * .1]
    np.testing.assert_allclose(closed_mesh_signed_distance(mesh, points), expected, atol=1e-7)
    assert closed_mesh_signed_distance(mesh, np.empty((0, 3))).shape == (0,)
    transform = trimesh.transformations.rotation_matrix(.8, [1, 2, 3])
    transform[:3, 3] = [.2, -.1, .72]
    mesh.apply_transform(transform)
    moved = points @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(closed_mesh_signed_distance(mesh, moved), expected, atol=2e-7)
    np.testing.assert_allclose(solid_angle_winding(mesh, moved), [1, 1, 0, 0], atol=1e-12)


def test_invalid_and_open_meshes_rejected():
    box = trimesh.creation.box()
    with pytest.raises(ValueError, match="finite"):
        closed_mesh_signed_distance(box, [[np.nan, 0, 0]])
    with pytest.raises(ValueError, match="finite"):
        closed_mesh_signed_distance(box, [0, 0, 0])
    opened = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1])
    for query in (closed_mesh_signed_distance, solid_angle_winding):
        with pytest.raises(ValueError, match="closed"):
            query(opened, [[0, 0, 0]])


def test_real_clipped_finger_nanometre_gap_not_false_0p176mm():
    from scipy.optimize import minimize

    mesh = trimesh.load_mesh(ROOT / "runs/taco_pour_collision_repair/assets/left_index_distal_clipped/6.obj",
                             process=True).convex_hull
    point = np.array([.0036550034613658986, .007961600398023924, -.03941897302865982])
    normal = mesh.face_normals
    offsets = np.einsum("ij,ij->i", normal, mesh.triangles_center) * 1000
    # Independent double-precision convex projection, solving in millimetres.
    target = point * 1000
    optimum = minimize(lambda x: np.dot(x - target, x - target), target,
        jac=lambda x: 2 * (x - target), method="SLSQP",
        constraints=[dict(type="ineq", fun=lambda x: offsets - normal @ x,
                          jac=lambda x: -normal)], options=dict(ftol=1e-15, maxiter=100))
    assert optimum.success
    expected = np.linalg.norm(optimum.x / 1000 - point)
    assert expected < 5e-9
    np.testing.assert_allclose(mesh_surface_distance(mesh, point[None]), [expected], atol=5e-9)
    scaled = mesh.copy()
    scaled.apply_scale(1000)
    independent = trimesh.proximity.closest_point(scaled, target[None])[1] / 1000
    np.testing.assert_allclose(independent, [expected], atol=1e-10)


def test_unsigned_open_surface_distance_and_validation():
    triangle = trimesh.Trimesh(vertices=[[0, 0, 0], [.01, 0, 0], [0, .01, 0]],
                               faces=[[0, 1, 2]], process=False)
    points = np.array([[.001, .001, .002], [.001, .001, -.002], [0, 0, 0]])
    expected = [.002, .002, 0]
    np.testing.assert_allclose(mesh_surface_distance(triangle, points), expected, atol=1e-9)
    np.testing.assert_array_equal(mesh_surface_distance(triangle, points)[::-1],
                                  mesh_surface_distance(triangle, points[::-1]))
    scaled = triangle.copy()
    scaled.apply_scale(1000)
    np.testing.assert_allclose(mesh_surface_distance(scaled, points * 1000) / 1000, expected, atol=1e-9)
    assert mesh_surface_distance(triangle, np.empty((0, 3))).shape == (0,)
    for invalid in ([0, 0, 0], [[np.nan, 0, 0]]):
        with pytest.raises(ValueError, match="finite"):
            mesh_surface_distance(triangle, invalid)
