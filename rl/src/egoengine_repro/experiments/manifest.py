"""Reproducible experiment manifest with source-artifact provenance."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .. import SCHEMA_VERSION
from ..artifacts import collect_artifacts, ensure_isolated_output
from ..config import ReproConfig


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class ExperimentManifest:
    path: Path
    data: dict[str, Any]

    @classmethod
    def create(
        cls, *, output_dir: str | Path, source_run: str | Path, config: ReproConfig,
        episode_id: str, seeds: Iterable[int], source_artifacts: Iterable[str | Path],
        repository_roots: Iterable[str | Path] = (),
    ) -> "ExperimentManifest":
        source = Path(source_run).resolve()
        root = ensure_isolated_output(output_dir, source)
        revisions = {
            str(Path(repo).resolve()): _git_revision(Path(repo).resolve())
            for repo in repository_roots
        }
        data = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": root.name,
            "episode_id": str(episode_id),
            "profile": config.name,
            "config": {
                "path": str(config.path), "sha256": config.sha256, "resolved": config.data,
            },
            "source_run": str(source),
            "source_artifacts": collect_artifacts(source_artifacts),
            "seeds": [int(seed) for seed in seeds],
            "ground_truth_policy": {
                "uses_ground_truth": config.uses_ground_truth,
                "scope": config.data["evaluation"].get("gt_scope", "evaluation_only"),
                "inference_outputs_may_not_be_written_to_source_run": True,
            },
            "git_revisions": revisions,
            "host": {
                "hostname": socket.gethostname(), "python": platform.python_version(),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stages": {},
        }
        manifest = cls(path=root / "experiment_manifest.json", data=data)
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentManifest":
        value = Path(path).resolve()
        return cls(value, json.loads(value.read_text(encoding="utf-8")))

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

    def record_stage(
        self, name: str, *, outputs: Iterable[str | Path], inputs: Iterable[str | Path] = (),
        metrics: dict[str, Any] | None = None, uses_ground_truth: bool = False,
    ) -> None:
        output_records = collect_artifacts(outputs)
        self.data["stages"][name] = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "uses_ground_truth": bool(uses_ground_truth),
            "inputs": collect_artifacts(inputs),
            "outputs": output_records,
            "metrics": metrics or {},
        }
        self.save()
