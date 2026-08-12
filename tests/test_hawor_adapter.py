import numpy as np
import pytest

from video_to_spider.adapters.hawor import interpolate_detection_boxes


def test_hawor_box_interpolation_preserves_detections_and_fills_gaps():
    boxes = np.full((5, 4), np.nan)
    boxes[1] = [10, 20, 30, 50]
    boxes[3] = [20, 30, 50, 70]
    detected = np.array([False, True, False, True, False])

    result = interpolate_detection_boxes(boxes, detected, (100, 80))

    np.testing.assert_allclose(result[1], boxes[1])
    np.testing.assert_allclose(result[3], boxes[3])
    np.testing.assert_allclose(result[2], [15, 25, 40, 60])
    np.testing.assert_allclose(result[0], boxes[1])
    np.testing.assert_allclose(result[4], boxes[3])


def test_hawor_box_interpolation_rejects_single_detection():
    boxes = np.full((3, 4), np.nan)
    boxes[1] = [10, 20, 30, 50]
    with pytest.raises(ValueError, match="at least two"):
        interpolate_detection_boxes(
            boxes, np.array([False, True, False]), (100, 80),
        )
