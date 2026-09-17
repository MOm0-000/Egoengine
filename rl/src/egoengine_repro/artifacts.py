"""Helpers for immutable, isolated experiment artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path) -> dict[str, object]:
    value = Path(path).resolve()
    if value.is_file():
        return {
            "path": str(value), "type": "file", "size_bytes": value.stat().st_size,
            "sha256": sha256_file(value),
        }
    if value.is_dir():
        files = sorted((item for item in value.rglob("*") if item.is_file()), key=str)
        digest = hashlib.sha256()
        total_size = 0
        for item in files:
            relative = item.relative_to(value).as_posix()
            file_digest = sha256_file(item)
            size = item.stat().st_size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_digest.encode("ascii"))
            digest.update(b"\0")
            total_size += size
        return {
            "path": str(value), "type": "directory", "file_count": len(files),
            "size_bytes": total_size, "sha256": digest.hexdigest(),
        }
    if not value.exists():
        raise FileNotFoundError(value)
    raise ValueError(f"artifact must be a regular file or directory: {value}")


def collect_artifacts(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    return [artifact_record(path) for path in sorted((Path(item).resolve() for item in paths), key=str)]


def ensure_isolated_output(output_dir: str | Path, source_run: str | Path) -> Path:
    output = Path(output_dir).resolve()
    source = Path(source_run).resolve()
    if output == source or source in output.parents:
        raise ValueError("reproduction output must not be inside the source run")
    output.mkdir(parents=True, exist_ok=True)
    return output
