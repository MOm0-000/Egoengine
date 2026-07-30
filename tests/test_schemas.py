import numpy as np
import pytest

from video_to_spider.schemas import ArtifactValidationError, validate_foundationpose_raw


def _artifact():
    t = 3
    return {
        "frame_indices": np.arange(t, dtype=np.int64),
        "timestamps_s": np.arange(t, dtype=np.float64) / 30,
        "T_camera_object": np.repeat(np.eye(4)[None], t, axis=0),
        "valid": np.ones(t, dtype=bool),
        "confidence": np.ones(t),
        "registration_frame": np.zeros(t, dtype=np.int64),
        "depth_residual": np.zeros(t),
        "mask_iou": np.ones(t),
    }


def test_foundationpose_schema_accepts_valid_artifact():
    validate_foundationpose_raw(_artifact())


@pytest.mark.parametrize("mutation", ["nan", "last_row", "reflection", "timestamps"])
def test_foundationpose_schema_rejects_invalid_artifact(mutation):
    artifact = _artifact()
    if mutation == "nan":
        artifact["confidence"][1] = np.nan
    elif mutation == "last_row":
        artifact["T_camera_object"][0, 3, 0] = 1
    elif mutation == "reflection":
        artifact["T_camera_object"][0, 0, 0] = -1
    else:
        artifact["timestamps_s"][2] = artifact["timestamps_s"][1]
    with pytest.raises(ArtifactValidationError):
        validate_foundationpose_raw(artifact)

