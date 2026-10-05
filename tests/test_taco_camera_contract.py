import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl/src"))

from egoengine_repro.taco.camera_contract import (
    CameraModelContract,
    load_allocentric_camera_contracts,
    rectify_mask,
    rectify_rgb,
)


RUN_INPUTS = Path("rl/runs/taco_brush_camera_table_infra_repair_v1/inputs")


def contracts():
    return load_allocentric_camera_contracts(
        RUN_INPUTS / "calibration.json",
        image_domain="UNDISTORTED_PINHOLE",
        certification="PINNED_OFFICIAL_CODE_BEHAVIOR",
    )[0]


def test_real_twelve_camera_records_and_video_sizes() -> None:
    values = contracts()
    metadata = json.loads((RUN_INPUTS / "allocentric_video_metadata.json").read_text())
    assert len(values) == 12 == len(metadata)
    for camera_id, camera in values.items():
        assert camera.K.shape == (3, 3)
        assert camera.R.shape == (3, 3)
        assert camera.T.shape == (3,)
        assert camera.image_size == tuple(metadata[camera_id][key] for key in ("width", "height"))
        assert camera.distortion_field_name == "distCoeff"
        assert camera.distortion_field_name in camera.raw_record_keys
        assert camera.distortion_coefficients is not None
        assert camera.image_domain == "UNDISTORTED_PINHOLE"


def test_distortion_projection_round_trip() -> None:
    camera = contracts()["21218078"]
    points = np.array([
        [-0.03, -0.02, 1.0], [0.0, 0.0, 1.2], [0.02, -0.01, 1.4],
        [-0.015, 0.025, 1.1], [0.03, 0.02, 1.5],
    ], dtype=np.float64)
    distorted, _ = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3), camera.K,
        camera.distortion_coefficients,
    )
    recovered = cv2.undistortPoints(
        distorted, camera.K, camera.distortion_coefficients, P=camera.K,
    ).reshape(-1, 2)
    pinhole = points[:, :2] / points[:, 2:]
    pinhole = pinhole @ camera.K[:2, :2].T + camera.K[:2, 2]
    assert np.max(np.abs(recovered - pinhole)) < 1e-7


def zero_distortion_contract() -> CameraModelContract:
    return CameraModelContract.from_record(
        "synthetic",
        {"imgSize": [24, 18], "K": [20, 0, 12, 0, 20, 9, 0, 0, 1],
         "R": np.eye(3).reshape(-1).tolist(), "T": [0, 0, 0],
         "D": [0, 0, 0, 0, 0]},
        image_domain="RAW_DISTORTED", certification="SYNTHETIC_TEST",
    )


def test_zero_distortion_remap_is_identity() -> None:
    camera = zero_distortion_contract()
    rgb = np.arange(18 * 24 * 3, dtype=np.uint8).reshape(18, 24, 3)
    mask = (np.arange(18 * 24).reshape(18, 24) % 5).astype(np.uint8)
    rectified_rgb, k_rgb = rectify_rgb(rgb, camera)
    rectified_mask, k_mask = rectify_mask(mask, camera)
    assert np.array_equal(rectified_rgb, rgb)
    assert np.array_equal(rectified_mask, mask)
    assert np.array_equal(k_rgb, camera.K)
    assert np.array_equal(k_mask, camera.K)


def test_mask_rectification_preserves_dtype_and_labels() -> None:
    source = contracts()["21218078"]
    record = source.to_dict()
    raw = {
        "imgSize": record["image_size"], "K": np.asarray(record["K"]).reshape(-1).tolist(),
        "R": np.asarray(record["R"]).reshape(-1).tolist(), "T": record["T"],
        "distortion_coefficients": record["distortion_coefficients"],
        "rectifyAlpha": 0.0,
    }
    camera = CameraModelContract.from_record(
        "raw-test", raw, image_domain="RAW_DISTORTED",
        certification="SYNTHETIC_DOMAIN_TEST",
    )
    width, height = camera.image_size
    mask = np.tile(np.arange(width, dtype=np.uint8) % 5, (height, 1))
    rectified, _ = rectify_mask(mask, camera)
    assert rectified.dtype == mask.dtype
    assert set(np.unique(rectified)) == {0, 1, 2, 3, 4}
