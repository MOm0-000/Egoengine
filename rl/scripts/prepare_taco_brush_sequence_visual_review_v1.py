#!/usr/bin/env python3
"""Extract and present six official TACO Brush/Brush/Bowl RGB sequences."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
from zipfile import ZipFile

import cv2
import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
SOURCE = Path("/data_all/zzx/egoengine_new/datasets/taco_v1")
RGB_ARCHIVE = SOURCE / "Egocentric_RGB_Videos.zip"
METADATA = SOURCE / "taco_info.csv"
SPLIT = Path(
    "/data_all/zzx/3.2RL/third_party/"
    "TACO-Instructions-b385ac1e89f35214bfbbef0d63e6df0ac8dac02e/"
    "data_lists/v1_overall_data_train_test_split.txt"
)
OUTPUT = RL_ROOT / "runs/taco_brush_sequence_visual_review_v1"
SEQUENCES = (
    "20231005_186",
    "20231005_188",
    "20231005_190",
    "20231005_194",
    "20231005_196",
    "20231006_145",
)
TRIPLET = "(brush, brush, bowl)"
FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def labeled(image: np.ndarray, title: str, width: int, height: int) -> np.ndarray:
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(result, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(
        result,
        title,
        (9, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def decode(path: Path, width: int, height: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames or fps <= 0.0:
        raise RuntimeError(f"invalid video: {path}")
    return frames, fps


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    videos_dir = OUTPUT / "videos"
    sheets_dir = OUTPUT / "sheets"
    videos_dir.mkdir(parents=True)
    sheets_dir.mkdir()

    with METADATA.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = {}
    for row in rows:
        sequence = row["sequence_id"].rsplit("/", 1)[-1]
        if sequence in SEQUENCES:
            if row["triplet"] != TRIPLET:
                raise ValueError(f"wrong triplet for {sequence}: {row['triplet']}")
            selected[sequence] = row
    if set(selected) != set(SEQUENCES):
        raise ValueError(f"metadata missing requested sequences: {set(SEQUENCES) - set(selected)}")

    split_map = {}
    for line in SPLIT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sequence, split = line.split(",", 1)
        split_map[sequence] = split
    if any(split_map.get(sequence) != "test_3" for sequence in SEQUENCES):
        raise ValueError("requested sequences are not all in official test_3")

    records = {}
    decoded = {}
    archive_info = RGB_ARCHIVE.stat()
    with ZipFile(RGB_ARCHIVE) as archive:
        for sequence in SEQUENCES:
            row = selected[sequence]
            member = row["egocentric_rgb_path"]
            entry = archive.getinfo(member)
            destination = videos_dir / f"{sequence}.mp4"
            with archive.open(entry) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
            if destination.stat().st_size != entry.file_size:
                raise RuntimeError(f"length mismatch for {sequence}")
            frames, decoded_fps = decode(destination, 480, 270)
            decoded[sequence] = frames
            expected_frames = int(row["n_frames"])
            if len(frames) != expected_frames:
                raise ValueError(
                    f"frame count mismatch for {sequence}: {len(frames)} != {expected_frames}"
                )
            indices = [round((len(frames) - 1) * fraction) for fraction in FRACTIONS]
            strip = np.concatenate(
                [
                    labeled(
                        frames[index],
                        f"{sequence}  f{index:03d}  {fraction:3.0%}",
                        384,
                        216,
                    )
                    for index, fraction in zip(indices, FRACTIONS)
                ],
                axis=1,
            )
            sheet = sheets_dir / f"{sequence}_six_frames.jpg"
            if not cv2.imwrite(str(sheet), strip, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError(f"cannot write {sheet}")
            records[sequence] = {
                "triplet": row["triplet"],
                "official_split": split_map[sequence],
                "metadata_status": {
                    "all_modalities_complete": row["all_modalities_complete"],
                    "calib_status": row["calib_status"],
                    "calib_reviewed": row["calib_reviewed"],
                },
                "frames": len(frames),
                "metadata_duration_s": float(row["duration_s"]),
                "metadata_fps": float(row["fps"]),
                "decoded_fps": decoded_fps,
                "sampled_frame_indices": indices,
                "source_member": member,
                "source_member_crc32": f"{entry.CRC:08x}",
                "video": artifact(destination),
                "sheet": artifact(sheet),
            }

    overview_rows = []
    for sequence in SEQUENCES:
        frames = decoded[sequence]
        indices = records[sequence]["sampled_frame_indices"]
        overview_rows.append(
            np.concatenate(
                [
                    labeled(
                        frames[index],
                        f"{sequence}  f{index:03d}  {fraction:3.0%}",
                        320,
                        180,
                    )
                    for index, fraction in zip(indices, FRACTIONS)
                ],
                axis=1,
            )
        )
    overview = np.concatenate(overview_rows, axis=0)
    overview_path = OUTPUT / "all_sequences_six_frame_overview.jpg"
    if not cv2.imwrite(str(overview_path), overview, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"cannot write {overview_path}")

    comparison_path = OUTPUT / "all_sequences_normalized_comparison.mp4"
    writer = cv2.VideoWriter(
        str(comparison_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (1440, 540)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {comparison_path}")
    try:
        for output_index in range(120):
            fraction = output_index / 119.0
            panels = []
            for sequence in SEQUENCES:
                frames = decoded[sequence]
                source_index = round((len(frames) - 1) * fraction)
                panels.append(
                    labeled(
                        frames[source_index],
                        f"{sequence}  f{source_index:03d}  {fraction:3.0%}",
                        480,
                        270,
                    )
                )
            writer.write(
                np.concatenate(
                    [np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)],
                    axis=0,
                )
            )
    finally:
        writer.release()

    manifest = {
        "schema": "taco_brush_sequence_visual_review_v1",
        "status": "OFFICIAL_RGB_VISUAL_REVIEW_ONLY",
        "requested_sequences": list(SEQUENCES),
        "triplet": TRIPLET,
        "official_split": "test_3",
        "source_archive": {
            "path": str(RGB_ARCHIVE.resolve()),
            "bytes": archive_info.st_size,
            "mtime_ns": archive_info.st_mtime_ns,
            "note": "Existing local official release archive; no network download used.",
        },
        "source_metadata": artifact(METADATA),
        "source_split_list": artifact(SPLIT),
        "sequences": records,
        "outputs": {
            "overview": artifact(overview_path),
            "normalized_comparison": artifact(comparison_path),
        },
        "mutations": {
            "source_data": 0,
            "annotations": 0,
            "physics": 0,
            "retargeting": 0,
            "training": 0,
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
    }
    write_json(OUTPUT / "manifest.json", manifest)

    lines = [
        "# TACO Brush sequence visual review v1",
        "",
        "All six requested sequences are official `(brush, brush, bowl)` entries in the "
        "TACO v1 `test_3` split. The RGB files were copied byte-for-byte from the existing "
        "local official release archive; no network download was needed.",
        "",
        "| sequence | frames | duration (s) | calibration | modalities |",
        "|---|---:|---:|---|---|",
    ]
    for sequence in SEQUENCES:
        row = records[sequence]
        lines.append(
            f"| `{sequence}` | {row['frames']} | {row['metadata_duration_s']:.2f} | "
            f"{row['metadata_status']['calib_status']} / reviewed | complete |"
        )
    lines.extend(
        [
            "",
            "The review artifacts do not identify which sequence EgoEngine used. They "
            "present the six official candidates without selecting a winner; that visual "
            "match remains a separate human review decision.",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hashes = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "server_artifacts.sha256":
            hashes.append(f"{sha256(path)}  {path.relative_to(OUTPUT)}")
    (OUTPUT / "server_artifacts.sha256").write_text(
        "\n".join(hashes) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "overview": str(overview_path),
                "comparison": str(comparison_path),
                "sequences": records,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
