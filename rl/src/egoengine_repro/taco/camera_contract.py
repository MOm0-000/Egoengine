"""Lossless camera calibration parser and explicit image-domain contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


IMAGE_DOMAINS = {"RAW_DISTORTED", "UNDISTORTED_PINHOLE", "UNKNOWN_IMAGE_DOMAIN"}
_DISTORTION_NAMES = (
    "distCoeff", "distCoeffs", "distortion_coefficients", "distortion", "D"
)
_KNOWN_FIELDS = {"imgSize", "K", "R", "T", "rectifyAlpha", *_DISTORTION_NAMES}


def _raw_hash(record: Mapping[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array(record: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(record[key], dtype=np.float64).reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"camera {key} is nonfinite")
    return value


@dataclass(frozen=True)
class CameraModelContract:
    camera_id: str
    raw_record_sha256: str
    raw_record_keys: tuple[str, ...]
    image_size: tuple[int, int]
    K: np.ndarray
    R: np.ndarray
    T: np.ndarray
    distortion_field_name: str | None
    distortion_coefficients: np.ndarray | None
    distortion_model: str
    image_domain: str
    projection_domain: str
    certification: str
    rectify_alpha: float | None
    unrecognized_fields: Mapping[str, Any]

    @classmethod
    def from_record(cls, camera_id: str, record: Mapping[str, Any], *,
                    image_domain: str, certification: str) -> "CameraModelContract":
        if image_domain not in IMAGE_DOMAINS:
            raise ValueError(f"unknown image domain: {image_domain}")
        required = {"imgSize", "K", "R", "T"}
        missing = required - set(record)
        if missing:
            raise ValueError(f"camera {camera_id} missing {sorted(missing)}")
        size_values = np.asarray(record["imgSize"], dtype=np.int64)
        if size_values.shape != (2,) or (size_values <= 0).any():
            raise ValueError("imgSize must be positive [width,height]")
        fields = [name for name in _DISTORTION_NAMES if name in record]
        if len(fields) > 1:
            raise ValueError(f"ambiguous distortion fields: {fields}")
        field = fields[0] if fields else None
        distortion = None if field is None else np.asarray(record[field], dtype=np.float64)
        if distortion is not None and (distortion.ndim != 1 or not np.isfinite(distortion).all()):
            raise ValueError("distortion coefficients must be a finite vector")
        if distortion is None:
            model = "NONE_PUBLISHED"
        elif len(distortion) in {4, 5, 8, 12, 14}:
            model = f"OPENCV_BROWN_CONRADY_{len(distortion)}"
        else:
            model = f"PUBLISHED_UNCLASSIFIED_{len(distortion)}"
        projection = ("PINHOLE_K_RECT" if image_domain == "RAW_DISTORTED"
                      else "PINHOLE_K" if image_domain == "UNDISTORTED_PINHOLE"
                      else "QUALITATIVE_ONLY_UNCERTIFIED")
        alpha = None if "rectifyAlpha" not in record else float(record["rectifyAlpha"])
        unknown = {key: record[key] for key in record if key not in _KNOWN_FIELDS}
        return cls(
            camera_id=str(camera_id), raw_record_sha256=_raw_hash(record),
            raw_record_keys=tuple(sorted(record)),
            image_size=(int(size_values[0]), int(size_values[1])),
            K=_array(record, "K", (3, 3)), R=_array(record, "R", (3, 3)),
            T=_array(record, "T", (3,)), distortion_field_name=field,
            distortion_coefficients=distortion, distortion_model=model,
            image_domain=image_domain, projection_domain=projection,
            certification=certification, rectify_alpha=alpha,
            unrecognized_fields=unknown,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "raw_record_sha256": self.raw_record_sha256,
            "raw_record_keys": list(self.raw_record_keys),
            "image_size": list(self.image_size),
            "K": self.K.tolist(), "R": self.R.tolist(), "T": self.T.tolist(),
            "distortion_field_name": self.distortion_field_name,
            "distortion_coefficients": (None if self.distortion_coefficients is None
                                         else self.distortion_coefficients.tolist()),
            "distortion_model": self.distortion_model,
            "image_domain": self.image_domain,
            "projection_domain": self.projection_domain,
            "certification": self.certification,
            "rectify_alpha": self.rectify_alpha,
            "unrecognized_fields": dict(self.unrecognized_fields),
        }


def load_allocentric_camera_contracts(
    calibration_path: str | Path, *, image_domain: str, certification: str,
) -> tuple[dict[str, CameraModelContract], dict[str, Any]]:
    path = Path(calibration_path).resolve(strict=True)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError("calibration root must be a nonempty camera mapping")
    contracts = {
        str(camera_id): CameraModelContract.from_record(
            str(camera_id), record, image_domain=image_domain,
            certification=certification,
        )
        for camera_id, record in raw.items()
    }
    return contracts, raw


def rectification_parameters(contract: CameraModelContract) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distortion = contract.distortion_coefficients
    if distortion is None:
        distortion = np.zeros(5, dtype=np.float64)
    if np.all(distortion == 0):
        return contract.K.copy(), distortion.copy(), contract.K.copy()
    alpha = 0.0 if contract.rectify_alpha is None else contract.rectify_alpha
    k_rect, _ = cv2.getOptimalNewCameraMatrix(
        contract.K, distortion, contract.image_size, alpha, contract.image_size,
        centerPrincipalPoint=False,
    )
    return contract.K.copy(), distortion.copy(), k_rect


def _rectify(image: np.ndarray, contract: CameraModelContract, *, interpolation: int) -> tuple[np.ndarray, np.ndarray]:
    if tuple(image.shape[1::-1]) != contract.image_size:
        raise ValueError("image size differs from camera contract")
    if contract.image_domain == "UNDISTORTED_PINHOLE":
        return image.copy(), contract.K.copy()
    if contract.image_domain != "RAW_DISTORTED":
        raise ValueError("cannot rectify an uncertified image domain")
    k, distortion, k_rect = rectification_parameters(contract)
    if np.all(distortion == 0):
        return image.copy(), k_rect
    map_x, map_y = cv2.initUndistortRectifyMap(
        k, distortion, np.eye(3), k_rect, contract.image_size, cv2.CV_32FC1,
    )
    return cv2.remap(image, map_x, map_y, interpolation=interpolation,
                     borderMode=cv2.BORDER_CONSTANT), k_rect


def rectify_rgb(image: np.ndarray, contract: CameraModelContract) -> tuple[np.ndarray, np.ndarray]:
    return _rectify(image, contract, interpolation=cv2.INTER_LINEAR)


def rectify_mask(mask: np.ndarray, contract: CameraModelContract) -> tuple[np.ndarray, np.ndarray]:
    before = set(np.unique(mask).tolist())
    result, k_rect = _rectify(mask, contract, interpolation=cv2.INTER_NEAREST)
    after = set(np.unique(result).tolist())
    if result.dtype != mask.dtype:
        raise AssertionError("mask rectification changed dtype")
    if not after.issubset(before | {0}):
        raise AssertionError("nearest-neighbor rectification invented mask labels")
    return result, k_rect
