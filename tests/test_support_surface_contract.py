from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl/src"))

from egoengine_repro.scene.support_surface import (
    Plane,
    ResolvedSupportSurface,
    build_taco_scene_alignment,
    make_taco_support_contract,
    resolve_support_surface,
)


RUN_INPUTS = Path("rl/runs/taco_brush_camera_table_infra_repair_v1/inputs")


def rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def test_plane_signed_distance_and_rigid_transform() -> None:
    normal = np.array([1.0, 2.0, 3.0])
    normal /= np.linalg.norm(normal)
    plane = Plane(normal=normal, offset=0.4, frame="source")
    on_plane = normal * 0.4
    points = np.stack([on_plane, on_plane + normal * 0.03, on_plane - normal * 0.02])
    assert np.allclose(plane.signed_distance(points), [0.0, 0.03, -0.02])

    transform = np.eye(4)
    transform[:3, :3] = rotation_x(0.61)
    transform[:3, 3] = [0.2, -0.3, 0.5]
    mapped = plane.transform(transform, target_frame="target")
    mapped_points = points @ transform[:3, :3].T + transform[:3, 3]
    assert np.allclose(mapped.signed_distance(mapped_points), plane.signed_distance(points))


def test_brush_exact_support_and_source_to_sim_mapping() -> None:
    mesh_path = Path("rl/models/taco_xhand/assets/objects/146/visual.obj")
    pose_path = RUN_INPUTS / "target_146.npy"
    assert mesh_path.is_file() and pose_path.is_file()
    mesh = trimesh.load_mesh(mesh_path, process=False)
    mesh.apply_scale(0.01)
    poses = np.load(pose_path, allow_pickle=False)
    contract = make_taco_support_contract(
        target_id="146", simulator_offset_m=0.72,
        provenance="PROJECT_SAMPLE_CONTRACT",
    )
    support = resolve_support_surface(contract, {
        "target": {"vertices": mesh.vertices, "poses": poses},
    })
    assert np.isclose(support.source.offset, 0.5501410086395977, atol=1e-12)
    world = mesh.vertices @ poses[0, :3, :3].T + poses[0, :3, 3]
    assert np.isclose(support.signed_distance(world).min(), 0.0, atol=1e-12)
    assert np.isclose(
        support.assert_entity_supported(mesh, poses[0], tolerance_m=1e-12),
        0.0,
        atol=1e-12,
    )

    center = poses[0, :3, 3]
    desired = center.copy()
    desired[:2] = [0.6, 0.0]
    transform = build_taco_scene_alignment(
        support, source_scene_center=center, desired_scene_center_sim=desired,
    )
    mapped_plane = support.transform(transform)
    mapped_world = world @ transform[:3, :3].T + transform[:3, 3]
    assert np.allclose(mapped_plane.normal, support.simulator.normal)
    assert np.isclose(mapped_plane.offset, support.simulator.offset, atol=1e-12)
    assert np.isclose(support.simulator.signed_distance(mapped_world).min(), 0.0,
                      atol=1e-12)


def test_common_translation_preserves_clearance() -> None:
    plane = Plane(normal=[0.2, -0.3, 0.9], offset=0.7, frame="world")
    points = np.array([[0.1, 0.2, 0.8], [-0.2, 0.5, 1.1]])
    transform = np.eye(4)
    transform[:3, 3] = [1.3, -0.7, 0.4]
    moved = plane.transform(transform, target_frame="moved")
    moved_points = points + transform[:3, 3]
    assert np.allclose(moved.signed_distance(moved_points), plane.signed_distance(points))


def test_tilted_support_alignment() -> None:
    source_plane = Plane(normal=[0.0, 0.0, 1.0], offset=0.2, frame="source")
    simulator_plane = source_plane.transform(
        np.block([[rotation_x(0.4), np.array([[0.1], [0.2], [0.3]])],
                  [np.zeros((1, 3)), np.ones((1, 1))]]),
        target_frame="simulator",
    )
    contract = make_taco_support_contract(target_id="synthetic", simulator_offset_m=0.0)
    support = ResolvedSupportSurface(
        contract=contract,
        source=source_plane,
        simulator=simulator_plane,
        support_entity_minimum_distance_m=0.0,
    )
    rotation = rotation_x(0.4)
    center = np.array([0.3, 0.4, 0.5])
    desired = rotation @ center + np.array([0.1, 0.2, 0.3])
    transform = build_taco_scene_alignment(
        support, source_scene_center=center, desired_scene_center_sim=desired,
        rotation_sim_world=rotation,
    )
    mapped = source_plane.transform(transform, target_frame="simulator")
    assert np.allclose(mapped.normal, simulator_plane.normal)
    assert np.isclose(mapped.offset, simulator_plane.offset)
    tangent = np.array([1.0, 0.0, 0.0])
    supported_points = np.stack([
        source_plane.normal * source_plane.offset,
        source_plane.normal * source_plane.offset + tangent,
        source_plane.normal * (source_plane.offset + 0.03),
    ])
    assert np.isclose(source_plane.minimum_signed_distance(supported_points), 0.0)
    assert np.isclose(
        source_plane.assert_entity_supported(supported_points, tolerance_m=1e-12),
        0.0,
    )
