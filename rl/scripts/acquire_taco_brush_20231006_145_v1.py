#!/usr/bin/env python3
"""Acquire every sequence-specific TACO artifact for Brush 20231006_145."""

from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
from typing import Any, BinaryIO
from zipfile import ZipFile, ZipInfo

from huggingface_hub import HfFileSystem, hf_hub_download


SEQUENCE = "20231006_145"
TRIPLET = "(brush, brush, bowl)"
SEQUENCE_PATH = f"{TRIPLET}/{SEQUENCE}"
CAMERAS = (
    "21218078", "22070938", "22139905", "22139906",
    "22139908", "22139909", "22139910", "22139911",
    "22139913", "22139914", "22139916", "22139946",
)
REPO_ID = "mzhobro/taco_dataset"
HF_PREFIX = f"datasets/{REPO_ID}/"
LOCAL_RELEASE = Path("/data_all/zzx/egoengine_new/datasets/taco_v1")
LOCAL_HANDS = Path("/data_all/zzx/3.2RL/data/taco_v1/hand_poses_v1/Hand_Poses.zip")
LOCAL_DEPTH_RESIZED = Path(
    "/data_all/zzx/egoengine_new/data/taco_v1/depth_resized/"
    "Egocentric_Depth_Videos.zip"
)
LOCAL_MANO = Path("/data_all/zzx/3.2RL/data/taco_v1/hand_poses_v1/mano_v1_2")
OUTPUT = Path(
    "/data_all/zzx/3.2RL/data/taco_v1/brush_brush_bowl_20231006_145"
)


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


class RemoteConcatFile(io.RawIOBase):
    """Seekable concatenation of remote multipart archive files."""

    def __init__(self, fs: HfFileSystem, paths: list[str]) -> None:
        self.fs = fs
        self.paths = paths
        self.infos = [fs.info(path) for path in paths]
        self.sizes = [int(info["size"]) for info in self.infos]
        self.ends: list[int] = []
        total = 0
        for size in self.sizes:
            total += size
            self.ends.append(total)
        self.position = 0
        self.handles: list[BinaryIO | None] = [None] * len(paths)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.ends[-1] + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.ends[-1] - self.position
        chunks = []
        while size > 0 and self.position < self.ends[-1]:
            index = bisect.bisect_right(self.ends, self.position)
            part_start = 0 if index == 0 else self.ends[index - 1]
            count = min(size, self.ends[index] - self.position)
            if self.handles[index] is None:
                self.handles[index] = self.fs.open(
                    self.paths[index],
                    "rb",
                    block_size=8 * 1024 * 1024,
                    cache_type="readahead",
                )
            handle = self.handles[index]
            assert handle is not None
            handle.seek(self.position - part_start)
            block = handle.read(count)
            if not block:
                break
            chunks.append(block)
            self.position += len(block)
            size -= len(block)
        return b"".join(chunks)

    def close(self) -> None:
        for handle in self.handles:
            if handle is not None:
                handle.close()
        super().close()


def output_path(official_member: str) -> Path:
    return OUTPUT / official_member


def extract_entry(
    archive: ZipFile,
    entry: ZipInfo,
    destination: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    if entry.is_dir():
        raise ValueError(f"expected file, got directory: {entry.filename}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(entry) as stream, destination.open("xb") as output:
        shutil.copyfileobj(stream, output, length=1 << 20)
    if destination.stat().st_size != entry.file_size:
        raise RuntimeError(f"length mismatch: {destination}")
    record = {
        "source": source,
        "member": entry.filename,
        "member_bytes": entry.file_size,
        "member_compressed_bytes": entry.compress_size,
        "member_crc32": f"{entry.CRC:08x}",
        "zip_crc_verified_during_extract": True,
        "output": artifact(destination),
    }
    print(
        f"EXTRACTED {destination.relative_to(OUTPUT)} "
        f"({destination.stat().st_size / 1_000_000:.2f} MB)",
        flush=True,
    )
    return record


def one_entry(archive: ZipFile, predicate, label: str) -> ZipInfo:
    matches = [entry for entry in archive.infolist() if predicate(entry)]
    if len(matches) != 1:
        raise ValueError(f"expected one {label}, got {len(matches)}")
    return matches[0]


def extract_local(
    records: list[dict[str, Any]],
    archive_path: Path,
    official_member: str,
    destination_member: str | None = None,
) -> None:
    with ZipFile(archive_path) as archive:
        entry = archive.getinfo(official_member)
        records.append(
            extract_entry(
                archive,
                entry,
                output_path(destination_member or official_member),
                {
                    "kind": "existing_local_official_archive",
                    "path": str(archive_path.resolve()),
                    "bytes": archive_path.stat().st_size,
                },
            )
        )


def remote_paths_for_camera(kind: str, camera: str) -> list[str]:
    base = f"{HF_PREFIX}{kind}/{camera}.zip"
    if camera in {"22139906", "22139914"}:
        return [base + ".00.part", base + ".01.part"]
    return [base]


def remote_source(fs: HfFileSystem, paths: list[str]) -> dict[str, Any]:
    parts = []
    for path in paths:
        info = fs.info(path)
        lfs = info.get("lfs")
        parts.append(
            {
                "path": path,
                "bytes": int(info["size"]),
                "lfs_sha256": getattr(lfs, "sha256", None),
            }
        )
    return {
        "kind": "official_huggingface_remote_zip_range_extract",
        "repo_id": REPO_ID,
        "parts": parts,
        "full_archives_downloaded": False,
    }


def extract_remote_camera(
    fs: HfFileSystem,
    records: list[dict[str, Any]],
    kind: str,
    camera: str,
) -> None:
    paths = remote_paths_for_camera(kind, camera)
    with RemoteConcatFile(fs, paths) as remote:
        with ZipFile(remote) as archive:
            entry = one_entry(
                archive,
                lambda item: (
                    not item.is_dir()
                    and SEQUENCE in item.filename
                    and item.filename.endswith(f"/{camera}.mp4")
                ),
                f"{kind} {camera} {SEQUENCE}",
            )
            official_member = f"{kind}/{SEQUENCE_PATH}/{camera}.mp4"
            records.append(
                extract_entry(
                    archive,
                    entry,
                    output_path(official_member),
                    remote_source(fs, paths),
                )
            )


def extract_remote_depth(fs: HfFileSystem, records: list[dict[str, Any]]) -> None:
    paths = [
        HF_PREFIX + "Egocentric_Depth_Videos.zip.00.part",
        HF_PREFIX + "Egocentric_Depth_Videos.zip.01.part",
    ]
    with RemoteConcatFile(fs, paths) as remote:
        with ZipFile(remote) as archive:
            entry = one_entry(
                archive,
                lambda item: (
                    not item.is_dir()
                    and SEQUENCE in item.filename
                    and item.filename.endswith("/egocentric_depth.avi")
                ),
                f"original egocentric depth {SEQUENCE}",
            )
            official_member = f"Egocentric_Depth_Videos/{SEQUENCE_PATH}/egocentric_depth.avi"
            records.append(
                extract_entry(
                    archive,
                    entry,
                    output_path(official_member),
                    remote_source(fs, paths),
                )
            )


def extract_remote_small_zip(
    fs: HfFileSystem,
    records: list[dict[str, Any]],
    repo_file: str,
    suffix: str,
    destination_member: str,
) -> None:
    path = HF_PREFIX + repo_file
    with fs.open(path, "rb", block_size=8 * 1024 * 1024, cache_type="readahead") as remote:
        with ZipFile(remote) as archive:
            entry = one_entry(
                archive,
                lambda item: (
                    not item.is_dir()
                    and SEQUENCE in item.filename
                    and item.filename.endswith(suffix)
                ),
                f"{repo_file} {SEQUENCE}",
            )
            records.append(
                extract_entry(
                    archive,
                    entry,
                    output_path(destination_member),
                    remote_source(fs, [path]),
                )
            )


def copy_regular(source: Path, destination: Path, label: str) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1 << 20)
    print(f"COPIED {destination.relative_to(OUTPUT)}", flush=True)
    return {
        "source": {"kind": label, "path": str(source.resolve())},
        "output": artifact(destination),
    }


def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    records: list[dict[str, Any]] = []

    metadata_path = LOCAL_RELEASE / "taco_info.csv"
    with metadata_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matching = [row for row in rows if row["sequence_id"] == SEQUENCE_PATH]
    if len(matching) != 1:
        raise ValueError(f"expected one metadata row, got {len(matching)}")
    metadata = matching[0]
    if metadata["all_modalities_complete"] != "True" or metadata["calib_status"] != "good":
        raise ValueError("sequence does not have complete, good-calibration metadata")

    extract_local(
        records,
        LOCAL_RELEASE / "Egocentric_RGB_Videos.zip",
        f"Egocentric_RGB_Videos/{SEQUENCE_PATH}/color.mp4",
    )
    extract_local(
        records,
        LOCAL_DEPTH_RESIZED,
        f"Egocentric_Depth_Videos/{SEQUENCE_PATH}/egocentric_depth.avi",
        f"Egocentric_Depth_Videos_resized/{SEQUENCE_PATH}/egocentric_depth.avi",
    )
    for filename in (
        "left_hand.pkl",
        "right_hand.pkl",
        "left_hand_shape.pkl",
        "right_hand_shape.pkl",
        "hand_joints.npy",
    ):
        extract_local(
            records,
            LOCAL_HANDS,
            f"Hand_Poses/{SEQUENCE_PATH}/{filename}",
        )
    extract_local(
        records,
        LOCAL_RELEASE / "Hand_Poses_3D.zip",
        f"Hand_Poses_3D/{SEQUENCE_PATH}/hand_joints.npy",
    )
    for filename in ("tool_171.npy", "target_180.npy"):
        extract_local(
            records,
            LOCAL_RELEASE / "Object_Poses.zip",
            f"Object_Poses/{SEQUENCE_PATH}/{filename}",
        )
    for object_id in ("171", "180"):
        extract_local(
            records,
            LOCAL_RELEASE / "Object_Models.zip",
            f"object_models_released/{object_id}_cm.obj",
        )
    for filename in ("egocentric_frame_extrinsic.npy", "egocentric_intrinsic.txt"):
        extract_local(
            records,
            LOCAL_RELEASE / "Egocentric_Camera_Parameters.zip",
            f"Egocentric_Camera_Parameters/{SEQUENCE_PATH}/{filename}",
        )
    for camera in CAMERAS:
        extract_local(
            records,
            LOCAL_RELEASE / "2D_Segmentation.zip",
            f"2D_Segmentation/{SEQUENCE_PATH}/{camera}_masks.npy",
        )

    for filename in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
        records.append(
            copy_regular(
                LOCAL_MANO / "models" / filename,
                output_path(f"mano_v1_2/models/{filename}"),
                "existing_local_official_mano_model",
            )
        )
    for filename in ("LICENSE.txt", "models/LICENSE.txt"):
        source = LOCAL_MANO / filename
        if source.exists():
            records.append(
                copy_regular(
                    source,
                    output_path(f"mano_v1_2/{filename}"),
                    "existing_local_official_mano_license",
                )
            )

    fs = HfFileSystem()
    extract_remote_depth(fs, records)
    extract_remote_small_zip(
        fs,
        records,
        "Allocentric_Camera_Parameters.zip",
        "/calibration.json",
        f"Allocentric_Camera_Parameters/{SEQUENCE_PATH}/calibration.json",
    )
    for kind in (
        "Allocentric_RGB_Videos",
        "Marker_Removed_Allocentric_RGB_Videos",
    ):
        for camera in CAMERAS:
            extract_remote_camera(fs, records, kind, camera)

    overlay_cache = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename="taco_analysis/overlays/brush_brush_bowl__20231006_145.jpg",
        )
    )
    records.append(
        copy_regular(
            overlay_cache,
            output_path("official_derived_preview/brush_brush_bowl__20231006_145.jpg"),
            "official_huggingface_derived_preview",
        )
    )

    with (OUTPUT / "taco_info.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata))
        writer.writeheader()
        writer.writerow(metadata)

    manifest = {
        "schema": "taco_brush_20231006_145_complete_acquisition_v1",
        "status": "COMPLETE_SEQUENCE_SPECIFIC_MODALITY_ACQUISITION",
        "sequence": SEQUENCE_PATH,
        "object_ids": {"tool_brush": "171", "target_bowl": "180"},
        "metadata": metadata,
        "camera_ids": list(CAMERAS),
        "records": records,
        "record_count": len(records),
        "output_bytes": sum(
            int(record["output"]["bytes"])
            for record in records
        ),
        "source_data_modified": False,
        "full_remote_archives_downloaded": False,
        "remote_access_method": (
            "seekable HTTP range extraction of exact ZIP members, including multipart ZIPs"
        ),
        "notes": [
            "Both original and marker-removed 12-camera allocentric RGB videos are included.",
            "Both original official depth and the existing locally resized depth are included.",
            "The official derived overlay is included for convenience but is not raw annotation.",
        ],
    }
    write_json(OUTPUT / "acquisition_manifest.json", manifest)

    summary = f"""# TACO Brush/Brush/Bowl 20231006_145 acquisition

Status: **COMPLETE_SEQUENCE_SPECIFIC_MODALITY_ACQUISITION**

- sequence: `{SEQUENCE_PATH}`
- frames: `{metadata['n_frames']}` at `{metadata['fps']}` fps
- tool/object: brush `171`, bowl `180`
- calibration: `{metadata['calib_status']}`, reviewed `{metadata['calib_reviewed']}`
- files acquired: `{len(records)}`
- bytes acquired: `{manifest['output_bytes']}`
- full remote archive downloads: `false`

Included: egocentric RGB, original and resized egocentric depth, full MANO hand
parameters and shapes, independent 3D hand joints, object poses and meshes,
egocentric and allocentric calibration, all 12 segmentation arrays, all 12 original
allocentric videos, all 12 marker-removed allocentric videos, MANO model files and
the official derived preview.

Remote video/depth artifacts were read as exact members of the official Hugging Face
ZIP archives via HTTP range requests. Every extracted ZIP member passed CRC checking;
every output has a SHA256 in `acquisition_manifest.json`.
"""
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")

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
                "records": len(records),
                "bytes": manifest["output_bytes"],
                "status": manifest["status"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
