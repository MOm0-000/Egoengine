"""Ground-truth bundle creation. These functions are evaluation-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from .. import SCHEMA_VERSION
from ..artifacts import artifact_record


FINGER_NAMES = ("Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")
JOINT_SUFFIXES = ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")


def _hand_names(side: str) -> list[str]:
    return [f"{side}Hand", *(f"{side}{finger}{suffix}" for finger in FINGER_NAMES for suffix in JOINT_SUFFIXES)]


def _canonical_joint_names() -> list[str]:
    return ["Wrist", *(f"{finger}{suffix}" for finger in FINGER_NAMES for suffix in JOINT_SUFFIXES)]


def _confidence(handle: h5py.File, names: list[str], count: int) -> tuple[np.ndarray, str]:
    paths = [f"confidences/{name}" for name in names]
    if all(path in handle for path in paths):
        return np.stack([np.asarray(handle[path], dtype=np.float64) for path in paths], axis=1), "recorded"
    return np.ones((count, len(names)), dtype=np.float64), "missing_assumed_valid"


def create_egodex_ground_truth_from_hdf5(
    hdf5_path: str | Path, output_dir: str | Path, *, episode_id: str,
    frame_indices: Sequence[int] | np.ndarray | None = None,
    timestamps_s: Sequence[float] | np.ndarray | None = None,
    fps: float = 30.0,
) -> Path:
    """Materialize an evaluation-only EgoDex hand/camera bundle.

    This dataset-level entry point does not depend on an inference run. It is
    therefore suitable for freezing a test set before predictions are made.
    """
    source_path = Path(hdf5_path).resolve()
    root = Path(output_dir).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if fps <= 0:
        raise ValueError("fps must be positive")
    root.mkdir(parents=True, exist_ok=True)
    hand_transforms, hand_confidence, confidence_sources = [], [], []
    with h5py.File(source_path, "r") as handle:
        frame_count = len(handle["transforms/camera"])
        selected = (
            np.arange(frame_count, dtype=np.int64)
            if frame_indices is None else np.asarray(frame_indices, dtype=np.int64)
        )
        if selected.ndim != 1 or not len(selected):
            raise ValueError("frame_indices must be a non-empty one-dimensional array")
        if np.any(selected < 0) or np.any(selected >= frame_count):
            raise ValueError("frame_indices fall outside the EgoDex episode")
        if np.any(np.diff(selected) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        for side in ("left", "right"):
            names = _hand_names(side)
            values = np.stack([np.asarray(handle[f"transforms/{name}"], dtype=np.float64) for name in names], axis=1)
            confidence, confidence_source = _confidence(handle, names, len(values))
            hand_transforms.append(values[selected])
            hand_confidence.append(confidence[selected])
            confidence_sources.append(confidence_source)
        K = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        camera = np.asarray(handle["transforms/camera"], dtype=np.float64)[selected]
    timestamps = (
        selected.astype(np.float64) / float(fps)
        if timestamps_s is None else np.asarray(timestamps_s, dtype=np.float64)
    )
    if timestamps.shape != selected.shape or not np.isfinite(timestamps).all():
        raise ValueError("timestamps_s must be finite and match frame_indices")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_s must be strictly increasing")
    hand_path = root / "hand_ground_truth.npz"
    np.savez_compressed(
        hand_path, frame_indices=selected, timestamps_s=timestamps,
        hand_order=np.asarray(["left", "right"]),
        joint_names=np.asarray(_canonical_joint_names()),
        T_world_joint=np.stack(hand_transforms, axis=1),
        confidence=np.stack(hand_confidence, axis=1),
    )
    camera_path = root / "camera_ground_truth.npz"
    np.savez_compressed(
        camera_path, frame_indices=selected, timestamps_s=timestamps,
        intrinsics=K, T_world_camera=camera,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset": "EgoDex", "episode_id": str(episode_id),
        "uses_ground_truth": True, "scope": "evaluation_only",
        "confidence_sources": dict(zip(("left", "right"), confidence_sources)),
        "camera_source": "egodex_hdf5",
        "source": artifact_record(source_path),
        "frame_interval": [int(selected[0]), int(selected[-1]) + 1],
        "frame_count": int(len(selected)),
        "fps": float(fps),
        "quality_labels": {
            "hand": "dataset_ground_truth",
            "camera": "dataset_calibration_ground_truth",
        },
        "artifacts": {
            "hand": artifact_record(hand_path), "camera": artifact_record(camera_path),
        },
        "unavailable_modalities": ["segmentation", "depth", "mesh", "object_trajectory", "contact"],
    }
    manifest_path = root / "ground_truth_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def create_egodex_ground_truth_bundle(run_dir: str | Path, output_dir: str | Path) -> Path:
    """Materialize EgoDex hand/camera GT matching one frozen inference run."""
    source_run = Path(run_dir).resolve()
    root = Path(output_dir).resolve()
    if source_run == root or source_run in root.parents:
        raise ValueError("ground-truth bundle must be outside the source run")
    source = json.loads((source_run / "input/source.json").read_text(encoding="utf-8"))
    with np.load(source_run / "hands/wilor_raw.npz", allow_pickle=False) as artifact:
        frame_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
        timestamps_s = np.asarray(artifact["timestamps_s"], dtype=np.float64)
    return create_egodex_ground_truth_from_hdf5(
        source["hdf5_path"], root, episode_id=str(source["episode_id"]),
        frame_indices=frame_indices, timestamps_s=timestamps_s,
    )
