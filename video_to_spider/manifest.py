"""Reproducible run manifests and content-based stage cache keys."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schemas import SCHEMA_VERSION, UNITS


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stage_cache_key(stage: str, config: Any, inputs: Iterable[str | Path]) -> str:
    records = []
    for raw in sorted((Path(p) for p in inputs), key=lambda p: str(p)):
        stat = raw.stat()
        records.append({"path": str(raw.resolve()), "size": stat.st_size, "sha256": sha256_file(raw)})
    payload = {"schema_version": SCHEMA_VERSION, "stage": stage, "config": config, "inputs": records}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class RunManifest:
    def __init__(self, path: str | Path, data: dict[str, Any]):
        self.path = Path(path)
        self.data = data

    @classmethod
    def create(
        cls, path: str | Path, *, run_id: str, source_episode: str, config: Any,
        frame_count: int, fps: float,
    ) -> "RunManifest":
        destination = Path(path)
        config_hash = hashlib.sha256(_canonical_json(config)).hexdigest()
        root = Path(__file__).resolve().parents[1]
        data = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "source_episode": source_episode,
            "config_sha256": config_hash,
            "git_revisions": {"video_to_spider": _git_revision(root)},
            "model_revisions": {},
            "allowed_inputs": [
                "rgb", "intrinsics", "camera_extrinsics", "instruction_text",
                "object_keyword_candidates",
            ],
            "frame_count": int(frame_count),
            "fps": float(fps),
            "stages": {},
            "coordinate_conventions": {
                "transform": "column-vector T_A_B maps B to A",
                "camera": "OpenCV +X right +Y down +Z forward",
                "quaternion_export": "wxyz",
            },
            "units": UNITS,
            "host": {"hostname": platform.node(), "pid": os.getpid()},
        }
        manifest = cls(destination, data)
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        source = Path(path)
        return cls(source, json.loads(source.read_text(encoding="utf-8")))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def start_stage(self, name: str, *, cache_key: str, command: list[str], environment: str) -> None:
        self.data["stages"][name] = {
            "cache_key": cache_key, "command": command, "environment": environment,
            "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
            "success": False, "outputs": [], "warnings": [], "quality_metrics": {},
        }
        self.save()

    def finish_stage(
        self, name: str, *, success: bool, outputs: list[str],
        quality_metrics: dict[str, Any], warnings: list[str] | None = None,
    ) -> None:
        stage = self.data["stages"][name]
        stage.update({
            "finished_at": datetime.now(timezone.utc).isoformat(), "success": bool(success),
            "outputs": outputs, "quality_metrics": quality_metrics, "warnings": warnings or [],
        })
        self.save()

