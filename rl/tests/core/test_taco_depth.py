from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from egoengine_repro.evaluation.taco_depth import (
    DepthVideoSpec,
    consecutive_difference,
    container_timestamp_mapping,
    duplicate_summary,
    exact_output_to_native_map,
    frame_sha256,
    iter_depth_frames,
    raw_depth_to_metres,
)


def _encode_ffv1(path: Path, frames: list[np.ndarray], *, width: int, height: int) -> None:
    payload = b"".join(frame.astype("<u2", copy=False).tobytes() for frame in frames)
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "gray16le",
        "-video_size", f"{width}x{height}", "-framerate", "15", "-i", "-",
        "-c:v", "ffv1", "-level", "1", "-pix_fmt", "gray16le", str(path),
    ], input=payload, check=True)


def test_official_fps30_duplicates_unique_15hz_frames(tmp_path: Path) -> None:
    frames = [np.full((3, 4), value, dtype=np.uint16) for value in (101, 202, 303)]
    video = tmp_path / "depth.avi"
    _encode_ffv1(video, frames, width=4, height=3)
    spec = DepthVideoSpec(width=4, height=3, frame_count=3)
    native = list(iter_depth_frames(video, spec))
    official = list(iter_depth_frames(video, spec, fps=30.0))
    native_hashes = [frame_sha256(frame) for frame in native]
    official_hashes = [frame_sha256(frame) for frame in official]
    mapping = exact_output_to_native_map(native_hashes, official_hashes)
    assert len(official) == 6
    assert [row["selected_earliest_monotonic_native"] for row in mapping] == [0, 0, 1, 1, 2, 2]


def test_duplicate_detector_recognizes_already_expanded_sequence() -> None:
    unique = [np.full((2, 2), value, dtype=np.uint16) for value in (1, 2, 3)]
    frames = [frame for frame in unique for _ in range(2)]
    hashes = [frame_sha256(frame) for frame in frames]
    links = [consecutive_difference(a, b) for a, b in zip(frames, frames[1:])]
    report = duplicate_summary(hashes, links)
    assert report["pattern_0_eq_1_2_eq_3_fraction"] == 1.0
    assert report["pattern_1_eq_2_3_eq_4_fraction"] == 0.0
    assert report["unique_frame_hashes"] == 3


def test_taco_scale_is_4000_raw_units_per_metre() -> None:
    result = raw_depth_to_metres(np.asarray([[4000]], dtype=np.uint16))
    assert result.item() == pytest.approx(1.0)


def test_invalid_inputs_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        raw_depth_to_metres(np.asarray([[4000]], dtype=np.int32))
    with pytest.raises(ValueError):
        raw_depth_to_metres(np.asarray([[4000]], dtype=np.uint16), scale=float("nan"))
    with pytest.raises(ValueError):
        DepthVideoSpec(width=0, height=2, frame_count=1)
    with pytest.raises(ValueError):
        exact_output_to_native_map(["a"], ["b"])
    with pytest.raises(ValueError):
        exact_output_to_native_map(["a", "a"], ["a"])
    with pytest.raises(ValueError):
        container_timestamp_mapping(
            annotation_count=3, logical_rate_hz=float("nan"), native_count=3, native_rate_hz=15,
        )
    with pytest.raises(ValueError):
        container_timestamp_mapping(
            annotation_count=0, logical_rate_hz=30, native_count=3, native_rate_hz=15,
        )
    truncated = tmp_path / "truncated.raw"
    truncated.write_bytes(b"\x00" * 7)
    process = subprocess.run([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "gray16le",
        "-video_size", "2x2", "-framerate", "15", "-i", str(truncated),
        "-c:v", "ffv1", str(tmp_path / "bad.avi"),
    ])
    assert process.returncode != 0


def test_wrong_resolution_decode_fails_closed(tmp_path: Path) -> None:
    frames = [np.full((3, 4), value, dtype=np.uint16) for value in (11, 22, 33)]
    video = tmp_path / "depth.avi"
    _encode_ffv1(video, frames, width=4, height=3)
    wrong = DepthVideoSpec(width=5, height=3, frame_count=3)
    with pytest.raises(RuntimeError):
        list(iter_depth_frames(video, wrong))
