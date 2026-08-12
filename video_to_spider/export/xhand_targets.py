"""Embodiment-scale normalization for exported xHand MINK targets.

EgoEngine aligns human and robot fingertip poses directly, but does not state a
morphology-normalization rule.  This optional adapter applies exactly one
isotropic hand-size ratio in the observed palm frame.  It never substitutes
xHand's neutral open-hand pose for the observed human pose, never uses contact
labels or object ground truth, and never fits a scale per frame.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# Measured once from the bundled xHand right.xml neutral configuration in the
# semantic right_palm site frame.  Finger order: thumb, index, middle, ring,
# pinky.  This is robot-model calibration, not episode data.
XHAND_NEUTRAL_TIP_VECTORS_PALM_M = np.asarray(
    [
        [0.024798587296476354, 0.1464937070178007, 0.04031111339572416],
        [0.0105, 0.0265, 0.1905],
        [0.0105, 0.0040, 0.1910],
        [0.0105, -0.0160, 0.1880],
        [0.0105, -0.0360, 0.1850],
    ],
    dtype=np.float64,
)


def isotropic_hand_size_scale(
    source_neutral: np.ndarray,
    target_neutral: np.ndarray = XHAND_NEUTRAL_TIP_VECTORS_PALM_M,
) -> float:
    source = np.asarray(source_neutral, dtype=np.float64)
    target = np.asarray(target_neutral, dtype=np.float64)
    if source.shape != (5, 3) or target.shape != (5, 3):
        raise ValueError("neutral fingertip vectors must have shape (5, 3)")
    denominator = float(np.sum(source * source))
    if denominator <= 1e-12:
        raise ValueError("source neutral hand has zero radial size")
    return float(np.sqrt(np.sum(target * target) / denominator))


def normalize_palm_relative_positions(
    wrist_position: np.ndarray,
    palm_rotation: np.ndarray,
    fingertip_position: np.ndarray,
    scale: float,
) -> np.ndarray:
    wrist = np.asarray(wrist_position, dtype=np.float64)
    rotation = np.asarray(palm_rotation, dtype=np.float64)
    fingertips = np.asarray(fingertip_position, dtype=np.float64)
    if wrist.ndim != 2 or wrist.shape[1] != 3:
        raise ValueError("wrist_position must have shape (T, 3)")
    if rotation.shape != (len(wrist), 3, 3):
        raise ValueError("palm_rotation must have shape (T, 3, 3)")
    if fingertips.shape != (len(wrist), 5, 3):
        raise ValueError("fingertip_position must have shape (T, 5, 3)")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    local = np.einsum(
        "tji,tfj->tfi", rotation, fingertips - wrist[:, None],
    )
    return wrist[:, None] + np.einsum(
        "tij,tfj->tfi", rotation, float(scale) * local,
    )


def normalize_exported_arrays(
    values: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {key: np.asarray(value).copy() for key, value in values.items()}
    if bool(np.asarray(arrays.get("morphology_normalization_applied", False))):
        return arrays, {"applied": False, "status": "already_applied"}
    required = {
        "qpos_wrist_right", "qpos_finger_right",
        "human_palm_orientation_right",
        "human_neutral_fingertip_vectors_right",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(
            "palm morphology normalization requires calibrated source geometry: "
            + ", ".join(missing)
        )
    wrist = np.asarray(arrays["qpos_wrist_right"], dtype=np.float64)
    fingers = np.asarray(arrays["qpos_finger_right"], dtype=np.float64)
    scale = isotropic_hand_size_scale(
        arrays["human_neutral_fingertip_vectors_right"],
    )
    normalized = normalize_palm_relative_positions(
        wrist[:, :3], arrays["human_palm_orientation_right"],
        fingers[:, :, :3], scale,
    )
    correction = np.linalg.norm(normalized - fingers[:, :, :3], axis=-1)
    arrays["morphology_original_fingertip_positions_right"] = (
        fingers[:, :, :3].copy()
    )
    arrays["qpos_finger_right"] = fingers.copy()
    arrays["qpos_finger_right"][:, :, :3] = normalized
    arrays["xhand_neutral_fingertip_vectors_right"] = (
        XHAND_NEUTRAL_TIP_VECTORS_PALM_M.astype(np.float32)
    )
    arrays["morphology_isotropic_scale"] = np.asarray(scale)
    arrays["morphology_normalization_applied"] = np.asarray(True)
    arrays["morphology_normalization_source"] = np.asarray(
        "one global neutral-geometry isotropic scale in the observed palm frame"
    )
    return arrays, {
        "applied": True,
        "status": "normalized",
        "paper_status": (
            "EgoEngine does not specify morphology normalization; optional "
            "implementation addition"
        ),
        "policy": (
            "one global hand-size scale in the observed palm frame; no "
            "per-finger neutral-pose replacement"
        ),
        "global_isotropic_scale": scale,
        "correction_m": {
            "mean": float(correction.mean()),
            "p95": float(np.percentile(correction, 95)),
            "max": float(correction.max()),
        },
        "uses_robot_output": False,
        "uses_object_ground_truth": False,
        "uses_contact_labels": False,
        "per_frame_fitting": False,
    }


def normalize_exported_keypoints(
    path: str | Path, *, report_path: str | Path | None = None,
) -> dict[str, Any]:
    target = Path(path).resolve()
    with np.load(target, allow_pickle=False) as artifact:
        source = {key: np.asarray(artifact[key]) for key in artifact.files}
    arrays, report = normalize_exported_arrays(source)
    if report["status"] == "normalized":
        payload = io.BytesIO()
        np.savez_compressed(payload, **arrays)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=target.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload.getvalue())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
