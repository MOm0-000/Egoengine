"""Frozen dataset protocols and ground-truth registration for section 3.1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh
import yaml

from .. import SCHEMA_VERSION
from ..artifacts import artifact_record
from .ground_truth import create_egodex_ground_truth_from_hdf5


MODALITIES = (
    "hand", "camera", "segmentation", "depth", "mesh", "object_trajectory", "contact",
)
ALLOWED_GT_PROVENANCE = {
    "dataset_ground_truth",
    "dataset_calibration_ground_truth",
    "dataset_automatic_annotation",
    "manual_annotation",
    "derived_from_ground_truth_geometry",
}


@dataclass(frozen=True)
class EvaluationSetSpec:
    path: Path
    data: dict[str, Any]
    dataset_root: Path
    selection_sha256: str

    @property
    def name(self) -> str:
        return str(self.data["name"])


def bundled_evaluation_set_path(name: str) -> Path:
    bundled = {
        "egodex_hand_camera_20": "formal_3_1_egodex_hand_camera_20.yaml",
    }
    if name not in bundled:
        raise ValueError(f"unknown bundled evaluation set: {name}")
    return Path(str(files("egoengine_repro").joinpath("configs", bundled[name])))


def load_evaluation_set(path_or_name: str | Path) -> EvaluationSetSpec:
    candidate = Path(path_or_name)
    path = candidate if candidate.exists() else bundled_evaluation_set_path(str(path_or_name))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or str(data.get("schema_version")) != "1.0":
        raise ValueError("evaluation set must use schema version 1.0")
    root = Path(data["dataset_root"]).expanduser().resolve()
    episodes = data.get("episodes")
    expected_count = int(data.get("episode_count", 0))
    if not isinstance(episodes, list) or expected_count <= 0 or len(episodes) != expected_count:
        raise ValueError("evaluation set episode_count must match its episode list")
    identifiers = [str(episode.get("episode_id", "")) for episode in episodes]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("evaluation episode IDs must be non-empty and unique")
    for episode in episodes:
        if str(episode.get("dataset")) != "EgoDex":
            raise ValueError("the current materializer supports EgoDex evaluation episodes")
        source = (root / str(episode["hdf5"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        interval = episode.get("frame_interval")
        if (
            not isinstance(interval, list) or len(interval) != 2
            or int(interval[0]) < 0 or int(interval[1]) <= int(interval[0])
        ):
            raise ValueError(f"invalid frame_interval for {episode['episode_id']}")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return EvaluationSetSpec(path.resolve(), data, root, digest)


def materialize_evaluation_set(
    spec: EvaluationSetSpec, output_dir: str | Path, *, force: bool = False,
) -> Path:
    """Freeze and materialize all evaluation-only GT bundles in a protocol."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for episode in spec.data["episodes"]:
        episode_id = str(episode["episode_id"])
        source = (spec.dataset_root / str(episode["hdf5"])).resolve()
        start, end = (int(value) for value in episode["frame_interval"])
        destination = output / "episodes" / episode_id
        manifest_path = destination / "ground_truth_manifest.json"
        if force or not manifest_path.is_file():
            manifest_path = create_egodex_ground_truth_from_hdf5(
                source, destination, episode_id=episode_id,
                frame_indices=np.arange(start, end, dtype=np.int64),
                fps=float(episode.get("fps", spec.data.get("fps", 30.0))),
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("scope") != "evaluation_only":
            raise ValueError(f"GT bundle is not evaluation-only: {manifest_path}")
        episodes.append({
            "episode_id": episode_id,
            "task": str(episode["task"]),
            "dataset": str(episode["dataset"]),
            "benchmark_role": str(spec.data["benchmark_role"]),
            "frame_interval": [start, end],
            "frame_count": end - start,
            "source": artifact_record(source),
            "ground_truth_manifest": artifact_record(manifest_path),
            "available_modalities": sorted(manifest["artifacts"]),
        })
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "name": spec.name,
        "dataset": str(spec.data["dataset"]),
        "benchmark_role": str(spec.data["benchmark_role"]),
        "exact_paper_test_set": bool(spec.data.get("exact_paper_test_set", False)),
        "selection_sha256": spec.selection_sha256,
        "episode_count": len(episodes),
        "spec": artifact_record(spec.path),
        "episodes": episodes,
        "gt_policy": {
            "scope": "evaluation_only",
            "may_enter_automatic_inference": False,
        },
    }
    destination = output / "frozen_evaluation_set_manifest.json"
    if destination.is_file() and not force:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        for key in ("selection_sha256", "episode_count", "episodes"):
            if existing.get(key) != frozen.get(key):
                raise ValueError("frozen evaluation inputs changed; use a new output directory")
        return destination
    destination.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return destination


def _npz_keys(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as artifact:
        return set(artifact.files)


def _validate_standard_artifact(modality: str, path: Path) -> None:
    required = {
        "hand": {"frame_indices", "T_world_joint", "confidence"},
        "camera": {"frame_indices", "intrinsics", "T_world_camera"},
        "segmentation": {"frame_indices", "masks"},
        "depth": {"frame_indices", "depth_m"},
        "object_trajectory": {"frame_indices", "valid"},
        "contact": {"frame_indices", "contact"},
    }
    if modality == "mesh":
        mesh = trimesh.load_mesh(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"mesh GT is empty or unsupported: {path}")
        return
    keys = _npz_keys(path)
    missing = required[modality] - keys
    if missing:
        raise ValueError(f"{modality} GT misses arrays {sorted(missing)}: {path}")
    if modality == "object_trajectory" and not ({"T_world_object", "T_camera_object", "T_sim_object", "T_object"} & keys):
        raise ValueError("object trajectory GT requires a named object transform array")


def register_ground_truth_bundle(spec_path: str | Path, output_dir: str | Path) -> Path:
    """Register standardized TACO/manual artifacts without copying or transforming them."""
    source_spec = Path(spec_path).resolve()
    spec = yaml.safe_load(source_spec.read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping) or str(spec.get("schema_version")) != "1.0":
        raise ValueError("ground-truth registration spec must use schema version 1.0")
    artifacts_spec = spec.get("artifacts", {})
    provenance = spec.get("provenance", {})
    if not isinstance(artifacts_spec, Mapping) or not artifacts_spec:
        raise ValueError("ground-truth registration requires at least one artifact")
    unknown = set(artifacts_spec) - set(MODALITIES)
    if unknown:
        raise ValueError(f"unknown GT modalities: {sorted(unknown)}")
    records: dict[str, Any] = {}
    labels: dict[str, str] = {}
    for modality, raw_path in artifacts_spec.items():
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = source_spec.parent / path
        path = path.resolve()
        label = str(provenance.get(modality, ""))
        if label not in ALLOWED_GT_PROVENANCE:
            raise ValueError(f"invalid or missing GT provenance for {modality}: {label}")
        _validate_standard_artifact(str(modality), path)
        records[str(modality)] = artifact_record(path)
        labels[str(modality)] = label
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(spec["dataset"]),
        "episode_id": str(spec["episode_id"]),
        "uses_ground_truth": True,
        "scope": "evaluation_only",
        "benchmark_role": str(spec.get("benchmark_role", "local_gt_evaluation")),
        "exact_paper_test_set": bool(spec.get("exact_paper_test_set", False)),
        "quality_labels": labels,
        "artifacts": records,
        "unavailable_modalities": sorted(set(MODALITIES) - set(records)),
        "registration_spec": artifact_record(source_spec),
    }
    destination = output / "ground_truth_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination
