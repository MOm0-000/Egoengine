import numpy as np
from scipy.spatial.transform import Rotation

from video_to_spider.coordinates import (
    compose_transforms, invert_transform, matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix, resample_transforms, transform_points,
)


def _transform(rotation, translation):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    value[:3, 3] = translation
    return value


def test_se3_round_trip_and_points():
    transform = _transform([0.2, -0.1, 0.3], [1.0, 2.0, -0.5])
    inverse = invert_transform(transform)
    np.testing.assert_allclose(compose_transforms(transform, inverse), np.eye(4), atol=1e-8)
    point = np.array([0.1, -0.4, 2.0])
    np.testing.assert_allclose(transform_points(inverse, transform_points(transform, point)), point, atol=1e-8)


def test_wxyz_round_trip():
    matrices = Rotation.from_rotvec([[0.1, 0.2, 0.3], [-0.4, 0.2, 0.1]]).as_matrix()
    quaternions = matrix_to_quaternion_wxyz(matrices)
    assert np.all(quaternions[:, 0] >= 0)
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(quaternions), matrices, atol=1e-8)


def test_resample_transform_translation_and_rotation():
    source_t = np.array([0.0, 1.0])
    transforms = np.stack([_transform([0, 0, 0], [0, 0, 0]), _transform([0, 0, np.pi], [1, 2, 3])])
    result = resample_transforms(source_t, transforms, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(result[1, :3, 3], [0.5, 1.0, 1.5])
    midpoint = Rotation.from_matrix(result[1, :3, :3]).magnitude()
    np.testing.assert_allclose(midpoint, np.pi / 2, atol=1e-8)

