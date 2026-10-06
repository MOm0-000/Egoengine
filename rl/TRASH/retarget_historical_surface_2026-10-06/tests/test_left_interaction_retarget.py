from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from egoengine_repro.retarget.interaction_aware import (
    FINGERS,
    bone_error,
    deterministic_surface_samples,
    human_semantic_points,
    interaction_error,
    interaction_topology,
)


ROOT = Path(__file__).resolve().parents[2]


def test_human_semantic_subset_has_only_the_declared_16_landmarks():
    joints = np.arange(63, dtype=np.float64).reshape(21, 3)
    points = human_semantic_points(joints)
    expected = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
    assert points.shape == (16, 3)
    assert np.array_equal(points, joints[expected])


def test_surface_samples_are_deterministic_and_lie_in_selected_triangles():
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]])
    first = deterministic_surface_samples(vertices, faces, 50, 0)
    second = deterministic_surface_samples(vertices, faces, 50, 0)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    points, face_index, barycentric = first
    assert np.all(barycentric >= 0.0)
    assert np.allclose(barycentric.sum(axis=1), 1.0)
    reconstructed = np.einsum("ni,nij->nj", barycentric, vertices[faces[face_index]])
    assert np.allclose(points, reconstructed)


def test_bone_and_interaction_errors_are_zero_for_identical_geometry():
    directions = {
        finger: (
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
        )
        for finger in FINGERS
    }
    total, per_finger, angles = bone_error(directions, directions)
    assert total == pytest.approx(0.0)
    assert all(value == pytest.approx(0.0) for value in per_finger.values())
    assert all(
        row[key] == pytest.approx(0.0)
        for row in angles.values()
        for key in ("proximal_deg", "distal_deg")
    )

    rng = np.random.default_rng(5)
    points = rng.normal(size=(20, 3))
    _, neighbors = interaction_topology(points)
    assert interaction_error(points, points, neighbors, 30.0) == pytest.approx(0.0)


def test_contract_freezes_the_single_candidate_and_paper_seeded_weights():
    contract = yaml.safe_load(
        (ROOT / "configs/taco_pour_left_interaction_retarget_v1.yaml").read_text()
    )
    assert contract["solver"]["candidate_count"] == 1
    assert contract["solver"]["multi_start"] is False
    assert contract["solver"]["parameter_sweep"] is False
    assert contract["solver"]["numerical_feasibility_tolerance"] == pytest.approx(2e-7)
    assert contract["semantic_keypoints"]["count"] == 16
    assert contract["interaction_mesh"] == {
        "object_role": "target",
        "object_visual_geom": "left_object_visual",
        "object_surface_samples": 50,
        "sample_seed": 0,
        "kappa": 30.0,
    }
    assert contract["warm_start"] == {"lambda_bone": 1.0, "lambda_smooth": 2.5}
    assert contract["refinement"] == {
        "lambda_interaction_mesh": 500.0,
        "lambda_bone": 0.1,
        "lambda_temporal": 2.5,
        "lambda_base_translation_delta": 100.0,
        "lambda_base_rotation_delta": 1.0,
    }
    assert not any(contract["authorization"].values())
