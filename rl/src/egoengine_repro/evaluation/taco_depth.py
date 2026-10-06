"""Fail-closed helpers for TACO metric-depth provenance and timebase audits.

The TACO release stores metric depth in lossless 16-bit AVI containers whose
container rate is not, by itself, the logical annotation rate.  This module
keeps decoding, scale conversion, duplicate analysis, and explicit frame maps
separate.  It deliberately knows nothing about any particular TACO sequence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class DepthVideoSpec:
    width: int
    height: int
    frame_count: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.frame_count <= 0:
            raise ValueError("depth video dimensions and frame_count must be positive")

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * np.dtype("<u2").itemsize


def raw_depth_to_metres(raw: np.ndarray, *, scale: float = 4000.0) -> np.ndarray:
    """Convert TACO uint16 depth to metres using the release scale contract."""
    if raw.dtype != np.uint16:
        raise TypeError(f"TACO depth must be uint16, got {raw.dtype}")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth scale must be finite and positive")
    return raw.astype(np.float32) / np.float32(scale)


def ffprobe_video(path: str | Path, *, count_frames: bool = True) -> dict:
    path = Path(path).resolve(strict=True)
    command = ["ffprobe", "-v", "error"]
    if count_frames:
        command.append("-count_frames")
    command += ["-select_streams", "v:0", "-show_streams", "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream: {path}")
    return streams[0]


def stream_frame_count(stream: dict) -> int:
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value not in (None, "N/A"):
            count = int(value)
            if count > 0:
                return count
    raise ValueError("video stream does not expose a positive frame count")


def frame_rate(value: str) -> Fraction:
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"invalid frame rate: {value!r}") from error
    if result <= 0:
        raise ValueError(f"frame rate must be positive: {value!r}")
    return result


def iter_depth_frames(
    path: str | Path,
    spec: DepthVideoSpec,
    *,
    fps: float | None = None,
) -> Iterator[np.ndarray]:
    """Stream exact uint16 decoded frames, optionally through FFmpeg's fps filter."""
    path = Path(path).resolve(strict=True)
    command = ["ffmpeg", "-v", "error", "-i", str(path)]
    if fps is not None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        command += ["-vf", f"fps=fps={fps:g}"]
    command += ["-f", "rawvideo", "-pix_fmt", "gray16le", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    assert process.stderr is not None
    seen = 0
    try:
        while True:
            payload = process.stdout.read(spec.frame_bytes)
            if not payload:
                break
            if len(payload) != spec.frame_bytes:
                raise RuntimeError(
                    f"truncated depth frame {seen}: {len(payload)} != {spec.frame_bytes}"
                )
            seen += 1
            yield np.frombuffer(payload, dtype="<u2").reshape(spec.height, spec.width).copy()
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        process.stderr.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg depth decode failed ({return_code}): {stderr[-2000:]}")
    if fps is None and seen != spec.frame_count:
        raise RuntimeError(f"decoded {seen} native frames, expected {spec.frame_count}")


def frame_sha256(raw: np.ndarray) -> str:
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise TypeError("frame hash requires a 2-D uint16 array")
    return hashlib.sha256(np.ascontiguousarray(raw.astype("<u2", copy=False)).tobytes()).hexdigest()


def frame_record(index: int, raw: np.ndarray, *, scale: float = 4000.0) -> dict:
    metres = raw_depth_to_metres(raw, scale=scale)
    valid = raw > 0
    values = metres[valid]
    return {
        "frame": int(index),
        "raw_sha256": frame_sha256(raw),
        "valid_pixel_fraction": float(np.mean(valid)),
        "valid_depth_m": None if values.size == 0 else {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        },
    }


def consecutive_difference(previous: np.ndarray, current: np.ndarray) -> dict:
    if previous.dtype != np.uint16 or current.dtype != np.uint16:
        raise TypeError("duplicate analysis requires uint16 frames")
    if previous.shape != current.shape:
        raise ValueError("duplicate analysis requires equal frame shapes")
    previous_valid = previous > 0
    current_valid = current > 0
    both = previous_valid & current_valid
    absolute = np.abs(previous.astype(np.int32) - current.astype(np.int32))
    return {
        "exact_equal": bool(np.array_equal(previous, current)),
        "valid_mask_equal": bool(np.array_equal(previous_valid, current_valid)),
        "changed_pixel_fraction": float(np.mean(previous != current)),
        "median_absolute_difference_raw_on_joint_valid": (
            None if not np.any(both) else float(np.median(absolute[both]))
        ),
    }


def duplicate_summary(hashes: Sequence[str], links: Sequence[dict]) -> dict:
    if not hashes:
        raise ValueError("cannot analyze an empty depth sequence")
    if len(links) != max(len(hashes) - 1, 0):
        raise ValueError("consecutive link count does not match frame hashes")
    even_pairs = [(i, i + 1) for i in range(0, len(hashes) - 1, 2)]
    odd_pairs = [(i, i + 1) for i in range(1, len(hashes) - 1, 2)]

    def fraction(pairs: list[tuple[int, int]]) -> float:
        return float(np.mean([hashes[a] == hashes[b] for a, b in pairs])) if pairs else 0.0

    runs: list[dict] = []
    start = 0
    for index in range(1, len(hashes) + 1):
        if index == len(hashes) or hashes[index] != hashes[start]:
            if index - start > 1:
                runs.append({"start": start, "end_inclusive": index - 1, "length": index - start})
            start = index
    return {
        "frame_count": len(hashes),
        "unique_frame_hashes": len(set(hashes)),
        "consecutive_exact_duplicate_count": sum(bool(row["exact_equal"]) for row in links),
        "consecutive_exact_duplicate_fraction": (
            sum(bool(row["exact_equal"]) for row in links) / len(links) if links else 0.0
        ),
        "pattern_0_eq_1_2_eq_3_fraction": fraction(even_pairs),
        "pattern_1_eq_2_3_eq_4_fraction": fraction(odd_pairs),
        "duplicate_runs": runs,
    }


def exact_output_to_native_map(
    native_hashes: Sequence[str], output_hashes: Sequence[str], *,
    allow_ambiguous_native_duplicates: bool = False,
) -> list[dict]:
    if not native_hashes or not output_hashes:
        raise ValueError("native and output sequences must be non-empty")
    lookup: dict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(native_hashes):
        lookup[digest].append(index)
    rows = []
    previous = -1
    for output_index, digest in enumerate(output_hashes):
        candidates = lookup.get(digest, [])
        if not candidates:
            raise ValueError(f"official output {output_index} has no exact native source")
        if len(candidates) > 1 and not allow_ambiguous_native_duplicates:
            raise ValueError(
                f"official output {output_index} matches multiple native frames: {candidates}"
            )
        monotonic = [candidate for candidate in candidates if candidate >= previous]
        if not monotonic:
            raise ValueError(f"official output {output_index} has no monotonic native source")
        # Exact duplicate native frames are intrinsically ambiguous. Preserve every
        # candidate and choose the earliest monotonic index only as a display aid.
        selected = monotonic[0]
        previous = selected
        rows.append({
            "output_frame": output_index,
            "raw_sha256": digest,
            "native_source_candidates": candidates,
            "selected_earliest_monotonic_native": selected,
            "source_is_unique": len(candidates) == 1,
        })
    return rows


def container_timestamp_mapping(
    *, annotation_count: int, logical_rate_hz: float,
    native_count: int, native_rate_hz: float,
) -> list[int]:
    if annotation_count <= 0 or native_count <= 0:
        raise ValueError("mapping frame counts must be positive")
    values = np.asarray([i / logical_rate_hz for i in range(annotation_count)], dtype=np.float64)
    if not np.all(np.isfinite(values)) or logical_rate_hz <= 0 or native_rate_hz <= 0:
        raise ValueError("mapping rates and times must be finite and positive")
    mapping = np.rint(values * native_rate_hz).astype(np.int64)
    mapping = np.clip(mapping, 0, native_count - 1)
    if mapping.size != annotation_count or np.any(np.diff(mapping) < 0):
        raise ValueError("container timestamp mapping is not complete and monotonic")
    return mapping.tolist()
