"""Pure contracts for offline visual-evidence publication.

This module deliberately contains no MuJoCo, environment, policy, or renderer
imports.  The rendering entry point consumes these contracts after all saved
input timelines have been validated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FRAME_MAP_FIELDS = (
    "endpoint",
    "reference_row",
    "reference_time_s",
    "source_frame_id",
    "decoded_video_frame_index",
    "rgb_pts_s",
    "alignment_method",
    "alignment_offset_s",
    "alignment_error_s",
    "rgb_available",
    "alignment_status",
    "provenance",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_manifest(records: Iterable[Mapping[str, Any]]) -> str:
    """Bind review status to sorted path/hash pairs, independent of JSON layout."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: str(row["path"])):
        digest.update(f"{record['path']}\0{record['sha256']}\n".encode())
    return digest.hexdigest()


def resolve_frame_map(
    frame_indices: np.ndarray,
    timestamps_s: np.ndarray,
    decoded_pts_s: Sequence[float | None],
    *,
    endpoint_start: int,
    endpoint_stop: int,
    provenance: str,
) -> list[dict[str, Any]]:
    """Resolve reference rows to explicitly named decoded video frames.

    ``frame_indices`` is required to contain decoded-video frame identifiers;
    the caller must establish that provenance before using this function.
    Non-zero offsets are supported and duplicate/non-monotonic mappings fail.
    """
    frames = np.asarray(frame_indices)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if frames.ndim != 1 or timestamps.shape != frames.shape:
        raise ValueError("frame indices and timestamps must be aligned 1-D arrays")
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("frame indices must be integers")
    if not len(frames) or (np.diff(frames) <= 0).any():
        raise ValueError("frame indices must be strictly increasing without duplicates")
    if not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0).any():
        raise ValueError("reference timestamps must be finite and strictly increasing")
    if endpoint_start < 0 or endpoint_stop < endpoint_start or endpoint_stop >= len(frames):
        raise ValueError("requested endpoint range is outside the reference")
    local_intervals = np.diff(timestamps)
    rows: list[dict[str, Any]] = []
    for endpoint in range(endpoint_start, endpoint_stop + 1):
        decoded = int(frames[endpoint])
        available = 0 <= decoded < len(decoded_pts_s) and decoded_pts_s[decoded] is not None
        pts = float(decoded_pts_s[decoded]) if available else None
        reference_time = float(timestamps[endpoint])
        error = abs(pts - reference_time) if pts is not None else None
        if endpoint == 0:
            local_interval = float(local_intervals[0])
        elif endpoint == len(timestamps) - 1:
            local_interval = float(local_intervals[-1])
        else:
            local_interval = float(min(local_intervals[endpoint - 1], local_intervals[endpoint]))
        aligned = available and error is not None and error <= 0.5 * local_interval + 1.0e-9
        rows.append(
            {
                "endpoint": endpoint,
                "reference_row": endpoint,
                "reference_time_s": reference_time,
                "source_frame_id": decoded,
                "decoded_video_frame_index": decoded if available else "",
                "rgb_pts_s": pts if pts is not None else "",
                "alignment_method": "explicit_frame_map",
                "alignment_offset_s": 0.0,
                "alignment_error_s": error if error is not None else "",
                "rgb_available": bool(available),
                "alignment_status": "aligned" if aligned else "unaligned",
                "provenance": provenance,
            }
        )
    return rows


def endpoint_rows(endpoint_array: np.ndarray) -> dict[int, int]:
    endpoints = np.asarray(endpoint_array)
    if endpoints.ndim != 1 or not np.issubdtype(endpoints.dtype, np.integer):
        raise ValueError("trajectory endpoint field must be one-dimensional integers")
    if len(np.unique(endpoints)) != len(endpoints):
        raise ValueError("trajectory endpoint field contains duplicates")
    return {int(endpoint): row for row, endpoint in enumerate(endpoints)}


def endpoint_status(
    endpoint: int,
    available: Mapping[int, int],
    *,
    first_failure_endpoint: int | None,
) -> str:
    if endpoint == 0 and endpoint in available:
        return "INITIAL"
    if endpoint not in available:
        if available and endpoint > max(available):
            return "NOT REACHED"
        return "NOT RECORDED"
    if first_failure_endpoint is not None and endpoint >= first_failure_endpoint:
        return "OLD TRACKING TERMINATED" if endpoint == first_failure_endpoint else "POST TERMINATION RECORDED"
    return "LEGACY TRACKING BELOW BOUNDARY; TASK RELATION NOT CERTIFIED"


def video_frame_map(
    frame_rows: Sequence[Mapping[str, Any]], *, slowdown_factor: int
) -> list[dict[str, Any]]:
    if slowdown_factor <= 0:
        raise ValueError("slowdown factor must be positive")
    output = []
    index = 0
    for row in frame_rows:
        for repeat in range(slowdown_factor):
            output.append(
                {
                    "output_frame_index": index,
                    "endpoint": int(row["endpoint"]),
                    "reference_time_s": float(row["reference_time_s"]),
                    "source_frame_id": int(row["source_frame_id"]),
                    "uniform_repeat_index": repeat,
                    "slowdown_factor": slowdown_factor,
                }
            )
            index += 1
    return output


def top_positive_increments(values: Mapping[int, float], count: int) -> list[tuple[int, float]]:
    ordered = sorted((int(endpoint), float(value)) for endpoint, value in values.items())
    increments = [
        (endpoint, value - previous)
        for (previous_endpoint, previous), (endpoint, value) in zip(ordered, ordered[1:])
        if endpoint == previous_endpoint + 1 and value - previous > 0.0
    ]
    return sorted(increments, key=lambda item: (-item[1], item[0]))[:count]


def semantic_contact_sets(
    source_endpoint: np.ndarray,
    group: np.ndarray,
    role1: np.ndarray,
    role2: np.ndarray,
    *,
    meaningful_groups: set[str],
) -> dict[int, set[tuple[str, str, str]]]:
    result: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    for source, group_name, first, second in zip(source_endpoint, group, role1, role2):
        group_text = str(group_name)
        if group_text in meaningful_groups:
            result[int(source)].add((group_text, str(first), str(second)))
    return dict(result)


def contact_transitions(
    states: Mapping[int, set[tuple[str, str, str]]], *, start: int, stop: int
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous: set[tuple[str, str, str]] = set()
    for source in range(start, stop):
        current = set(states.get(source, set()))
        for identity in sorted(current - previous):
            events.append({"endpoint": source + 1, "transition": "appeared", "semantic_identity": "|".join(identity)})
        for identity in sorted(previous - current):
            events.append({"endpoint": source + 1, "transition": "disappeared", "semantic_identity": "|".join(identity)})
        previous = current
    return events


def validate_publication(
    root: Path,
    *,
    endpoint_count: int,
    views: Sequence[str],
    expected_sheet_count: int,
) -> dict[str, Any]:
    root = Path(root)
    missing = []
    previews = []
    for view in views:
        for endpoint in range(endpoint_count):
            path = root / "previews" / view / f"endpoint_{endpoint:06d}.jpg"
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path.relative_to(root)))
            else:
                previews.append(path)
    sheets = sorted((root / "sheets").glob("*/*.jpg"))
    for relative in ("VISUAL_INDEX.md", "index.html", "frame_map.csv", "events.csv", "visual_review.json"):
        if not (root / relative).is_file():
            missing.append(relative)
    return {
        "valid": not missing and len(sheets) == expected_sheet_count,
        "preview_count": len(previews),
        "expected_preview_count": endpoint_count * len(views),
        "sheet_count": len(sheets),
        "expected_sheet_count": expected_sheet_count,
        "missing": missing,
    }


@dataclass(frozen=True)
class ReviewBinding:
    status: str
    image_manifest_sha256: str


def pending_review(image_manifest_sha256: str) -> ReviewBinding:
    return ReviewBinding(status="PENDING", image_manifest_sha256=image_manifest_sha256)
