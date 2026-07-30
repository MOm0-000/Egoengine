"""Explicit evaluator/oracle-only reader for EgoDex hand ground truth."""

from __future__ import annotations

from pathlib import Path
import json

import cv2
import h5py
import numpy as np

from ..coordinates import invert_transform
from ..manifest import RunManifest, stage_cache_key
from ..schemas import validate_transforms

FINGERTIP_NAMES = ("ThumbTip", "IndexFingerTip", "MiddleFingerTip", "RingFingerTip", "LittleFingerTip")


def load_hand_ground_truth(path: str | Path, side: str) -> dict[str, np.ndarray]:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    names = [f"{side}Hand", *(f"{side}{suffix}" for suffix in FINGERTIP_NAMES)]
    with h5py.File(path, "r") as handle:
        transforms = np.stack([np.asarray(handle[f"transforms/{name}"], dtype=np.float64) for name in names], axis=1)
        confidence = np.stack([np.asarray(handle[f"confidences/{name}"], dtype=np.float64) for name in names], axis=1)
    validate_transforms(f"{side}_hand_ground_truth", transforms)
    return {"names": np.asarray(names), "T_world_joint": transforms, "confidence": confidence}


def _project(K: np.ndarray, points_camera: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict[str, float]]:
    homogeneous = (K @ points_camera[..., None])[..., 0]
    uv = homogeneous[..., :2] / homogeneous[..., 2:3]
    positive = points_camera[..., 2] > 1e-6
    inside = positive & (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
    metrics = {
        "positive_depth_ratio": float(np.mean(positive)),
        "in_frame_ratio": float(np.mean(inside)),
        "median_depth_m": float(np.median(points_camera[..., 2])),
    }
    return uv, metrics


def validate_camera_direction_oracle(run_dir: str | Path, *, frame_index: int | None = None) -> Path:
    """Explicit GT-only diagnostic; never called by the ordinary ingest path."""
    root = Path(run_dir)
    manifest = RunManifest.load(root / "manifest.json")
    source = json.loads((root / "input/source.json").read_text(encoding="utf-8"))
    hdf5_path, mp4_path = Path(source["hdf5_path"]), Path(source["mp4_path"])
    start, end = source["selected_frame_interval"]
    selected = start + (end - start) // 2 if frame_index is None else frame_index
    if not start <= selected < end:
        raise ValueError(f"frame_index {selected} outside selected interval [{start}, {end})")
    names = [
        "leftHand", *(f"left{suffix}" for suffix in FINGERTIP_NAMES),
        "rightHand", *(f"right{suffix}" for suffix in FINGERTIP_NAMES),
    ]
    with h5py.File(hdf5_path, "r") as handle:
        K = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        raw_camera = np.asarray(handle["transforms/camera"], dtype=np.float64)
        points_world = np.stack([np.asarray(handle[f"transforms/{name}"][:, :3, 3]) for name in names], axis=1)
        confidence = np.stack([np.asarray(handle[f"confidences/{name}"]) for name in names], axis=1)
    ones = np.ones(points_world.shape[:-1] + (1,), dtype=points_world.dtype)
    points_h = np.concatenate([points_world, ones], axis=-1)
    interpretations = {
        "provided_is_T_world_camera": invert_transform(raw_camera),
        "provided_is_T_camera_world": raw_camera,
    }
    capture = cv2.VideoCapture(str(mp4_path))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, selected)
    ok, rgb = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode diagnostic frame {selected}")
    results: dict[str, dict[str, float]] = {}
    projected: dict[str, np.ndarray] = {}
    for label, T_camera_world in interpretations.items():
        points_camera = np.einsum("tij,tkj->tki", T_camera_world, points_h)[..., :3]
        uv, metrics = _project(K, points_camera, width, height)
        confident = confidence > 0
        metrics["confident_in_frame_ratio"] = float(np.mean(
            confident & (points_camera[..., 2] > 1e-6)
            & (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
        ))
        results[label] = metrics
        projected[label] = uv
    selected_label = max(results, key=lambda label: (results[label]["in_frame_ratio"], results[label]["positive_depth_ratio"]))
    margin = results[selected_label]["in_frame_ratio"] - min(value["in_frame_ratio"] for value in results.values())
    if margin < 0.25:
        raise RuntimeError(f"camera direction ambiguous: {results}")
    diagnostic_dir = root / "calibration/oracle_camera_direction"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    colors = {"provided_is_T_world_camera": (0, 255, 0), "provided_is_T_camera_world": (0, 0, 255)}
    overlay = rgb.copy()
    for label, uv in projected.items():
        for point in uv[selected]:
            if np.all(np.isfinite(point)):
                cv2.circle(overlay, tuple(np.rint(point).astype(int)), 7, colors[label], -1, lineType=cv2.LINE_AA)
    overlay_path = diagnostic_dir / f"frame_{selected:06d}.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    report = {
        "schema_version": "1.0", "uses_ground_truth": True,
        "purpose": "oracle coordinate-direction diagnostic only; excluded from the inference path and V1 metrics",
        "frame_index": selected, "selected_interpretation": selected_label,
        "selection_margin_in_frame_ratio": margin, "interpretations": results,
        "overlay": str(overlay_path.relative_to(root)),
    }
    report_path = diagnostic_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cache_key = stage_cache_key("oracle_camera_direction", {"frame_index": selected}, [hdf5_path, mp4_path])
    manifest.start_stage(
        "oracle_camera_direction", cache_key=cache_key,
        command=["oracle-validate-extrinsics", str(root)], environment="v2s-core",
    )
    manifest.data["stages"]["oracle_camera_direction"]["uses_ground_truth"] = True
    manifest.finish_stage(
        "oracle_camera_direction", success=True,
        outputs=[str(report_path.relative_to(root)), str(overlay_path.relative_to(root))],
        quality_metrics={"selected_interpretation": selected_label, "selection_margin_in_frame_ratio": margin},
    )
    return report_path
