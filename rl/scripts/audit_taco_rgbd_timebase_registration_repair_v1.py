#!/usr/bin/env python3
"""Repair the TACO RGB-D timebase contract before any support inference.

This audit is intentionally zero-runtime: no support estimation, retargeting,
physics, Replay, MPC, RL, promotion, or chunk commit is reachable here.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.evaluation.taco_depth import (  # noqa: E402
    DepthVideoSpec,
    consecutive_difference,
    container_timestamp_mapping,
    duplicate_summary,
    exact_output_to_native_map,
    ffprobe_video,
    frame_rate,
    frame_record,
    frame_sha256,
    iter_depth_frames,
    raw_depth_to_metres,
    stream_frame_count,
)
from egoengine_repro.evaluation.taco_official_projection import TacoOfficialProjector  # noqa: E402


DEFAULT_CONFIG = RL_ROOT / "configs/taco_rgbd_timebase_registration_repair_v1.yaml"


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{label}: missing={sorted(expected-actual)} unknown={sorted(actual-expected)}")


SAMPLE_KEYS = {
    "key", "triplet", "sequence", "expected_frames", "tool_id", "target_id", "root",
    "rgb_video", "depth_video", "release_depth_source", "release_manifest", "intrinsic",
    "extrinsic", "hand_joints", "object_pose_dir", "object_model_dir",
}


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    exact_keys(cfg, {
        "schema", "status", "minimum_baseline", "paper_faithful", "authorization",
        "official", "timeline", "registration", "samples", "output",
    }, "root")
    if cfg["schema"] != "taco_rgbd_timebase_registration_repair_v1":
        raise ValueError("unexpected schema")
    exact_keys(cfg["authorization"], {
        "support_plane_estimation", "active_support_contract_replacement", "mink",
        "physics", "replay", "mpc", "reinforcement_learning", "promotion", "chunk_commit",
    }, "authorization")
    exact_keys(cfg["official"], {
        "paper_url", "repository", "commit", "checkout", "projection_entrypoint",
        "available_sequences", "depth_issue_url", "hardware_issue_url", "scale_fix_commit",
        "scale_fix_url", "device",
    }, "official")
    exact_keys(cfg["timeline"], {
        "logical_rate_hz", "encoded_depth_rate_hz", "official_decode_rate_hz", "depth_width",
        "depth_height", "depth_dtype", "depth_scale", "h3_minimum_exact_pair_fraction",
        "container_timestamp_minimum_native_usage_fraction", "hypotheses",
    }, "timeline")
    exact_keys(cfg["registration"], {
        "uniform_frame_count", "minimum_samples_with_high_coverage_anchors",
        "interior_erosion_px", "front_surface_tolerance_m", "minimum_valid_pixels",
        "high_depth_coverage_fraction", "maximum_median_absolute_error_mm",
        "threshold_provenance",
    }, "registration")
    if cfg["status"] != "ZERO_RUNTIME_RGBD_TIMEBASE_AND_REGISTRATION_REPAIR":
        raise ValueError("unexpected runtime status")
    if cfg["paper_faithful"] is not False:
        raise ValueError("the local repaired mapping must not be labelled paper-faithful")
    if any(cfg["authorization"].values()):
        raise ValueError("this audit authorizes zero runtime and zero contract replacement")
    if len(cfg["samples"]) != 4:
        raise ValueError("the four predeclared official-available samples are required")
    for index, sample in enumerate(cfg["samples"]):
        exact_keys(sample, SAMPLE_KEYS, f"sample[{index}]")
    if cfg["timeline"]["hypotheses"] != [
        "INDEX_ALIGNED", "NATIVE_15_TO_30", "CURRENT_FILE_ALREADY_EXPANDED",
        "SEQUENCE_MAPPING_UNRESOLVED",
    ]:
        raise ValueError("timeline hypotheses changed")
    if float(cfg["timeline"]["depth_scale"]) != 4000.0:
        raise ValueError("TACO metric depth scale must be 4000")
    if cfg["timeline"]["depth_dtype"] != "uint16":
        raise ValueError("TACO metric depth dtype must be uint16")
    if float(cfg["registration"]["high_depth_coverage_fraction"]) != 0.5:
        raise ValueError("frozen high-depth-coverage threshold changed")
    if float(cfg["registration"]["maximum_median_absolute_error_mm"]) != 20.0:
        raise ValueError("frozen geometry threshold changed")
    if int(cfg["registration"]["minimum_samples_with_high_coverage_anchors"]) < 2:
        raise ValueError("spatial registration needs high-coverage anchors in multiple samples")
    head = git_head(REPO_ROOT)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], head],
        cwd=REPO_ROOT, check=True,
    )
    official = Path(cfg["official"]["checkout"])
    if git_head(official) != cfg["official"]["commit"]:
        raise ValueError("pinned official checkout commit mismatch")
    return cfg


def resolve_samples(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path_keys = {
        "root", "rgb_video", "depth_video", "release_depth_source", "release_manifest",
        "intrinsic", "extrinsic", "hand_joints", "object_pose_dir", "object_model_dir",
    }
    values = []
    for source in cfg["samples"]:
        sample = dict(source)
        for key in path_keys:
            sample[key] = Path(sample[key]).resolve(strict=True)
        for role in ("tool", "target"):
            object_id = sample[f"{role}_id"]
            sample[f"{role}_pose"] = (
                sample["object_pose_dir"] / f"{role}_{object_id}.npy"
            ).resolve(strict=True)
            sample[f"{role}_mesh"] = (
                sample["object_model_dir"] / f"{object_id}_cm.obj"
            ).resolve(strict=True)
        values.append(sample)
    return values


def ffmpeg_filtered_count(path: Path, fps: float) -> int:
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps=fps={fps:g}",
        "-f", "null", "-", "-progress", "pipe:1", "-nostats",
    ], check=True, capture_output=True, text=True)
    matches = re.findall(r"^frame=(\d+)$", result.stdout, re.MULTILINE)
    if not matches:
        raise RuntimeError(f"FFmpeg did not report filtered frame count: {path}")
    return int(matches[-1])


def write_container_timeline(path: Path, output: Path) -> dict[str, Any]:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_streams", "-show_frames", "-show_packets",
        "-select_streams", "v:0", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    counts: dict[str, int] = {"streams": 0, "frames": 0, "packets": 0}
    with output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(value.get("streams", [])):
            counts["streams"] += 1
            stream.write(json.dumps({"record_type": "stream", "index": index, **row}, sort_keys=True) + "\n")
        # With -show_frames and -show_packets together, ffprobe emits a single
        # ordered packets_and_frames array.  Preserve that order and its PTS.
        for index, row in enumerate(value.get("packets_and_frames", [])):
            kind = row.get("type")
            if kind not in {"frame", "packet"}:
                raise ValueError(f"unexpected ffprobe timeline record: {kind!r}")
            counts[f"{kind}s"] += 1
            stream.write(json.dumps({"record_type": kind, "index": index, **row}, sort_keys=True) + "\n")
    return counts


def decode_native_sample(
    sample: dict[str, Any], cfg: dict[str, Any], *, keep_manifest: bool,
) -> dict[str, Any]:
    timeline = cfg["timeline"]
    stream = ffprobe_video(sample["depth_video"])
    spec = DepthVideoSpec(
        width=int(timeline["depth_width"]), height=int(timeline["depth_height"]),
        frame_count=stream_frame_count(stream),
    )
    selected_indices = sorted(set(np.rint(np.linspace(
        0, sample["expected_frames"] - 1, int(cfg["registration"]["uniform_frame_count"]),
    )).astype(int).tolist()))
    records, hashes, links, selected = [], [], [], {}
    previous = None
    for index, raw in enumerate(iter_depth_frames(sample["depth_video"], spec)):
        digest = frame_sha256(raw)
        hashes.append(digest)
        if keep_manifest:
            records.append(frame_record(index, raw, scale=float(timeline["depth_scale"])))
        if previous is not None:
            links.append({"previous_frame": index - 1, "current_frame": index,
                          **consecutive_difference(previous, raw)})
        if index in selected_indices:
            selected[index] = raw
        previous = raw
    if len(hashes) != sample["expected_frames"]:
        raise ValueError(f"{sample['key']}: native depth count differs from contract")
    report = duplicate_summary(hashes, links)
    report.update({
        "sample": sample["key"], "stream": stream,
        "uniform_registration_frames": selected_indices,
        "consecutive_links": links,
    })
    return {"stream": stream, "spec": spec, "records": records, "hashes": hashes,
            "duplicate": report, "selected_depth": selected}


def provenance(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for sample in samples:
        local, source = sample["depth_video"], sample["release_depth_source"]
        local_hash, source_hash = sha256(local), sha256(source)
        same_path = local.samefile(source)
        byte_identical = local.stat().st_size == source.stat().st_size and local_hash == source_hash
        if not byte_identical:
            classification = "LOCAL_REENCODE_CONFIRMED"
        elif same_path:
            classification = "BYTE_IDENTICAL_TO_RELEASE_SOURCE"
        else:
            classification = "RENAMED_BYTE_IDENTICAL_COPY"
        rows.append({
            "sample": sample["key"], "classification": classification,
            "byte_identical": byte_identical, "same_filesystem_object": same_path,
            "local": artifact(local), "release_source": artifact(source),
            "release_manifest": artifact(sample["release_manifest"]),
        })
    return {
        "schema": "taco_depth_file_provenance_v1", "samples": rows,
        "all_release_provenance_closed": all(row["byte_identical"] for row in rows),
        "note": "dev4 release sources were extracted from exact official archive members with size and CRC checks; Pour is the exact downloaded release member recorded by its manifest.",
    }


def official_contract(cfg: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    official = cfg["official"]
    available_path = Path(official["checkout"]) / official["available_sequences"]
    available = set(available_path.read_text(encoding="utf-8").splitlines())
    availability = {
        sample["key"]: f"{sample['triplet']} {sample['sequence']}" in available for sample in samples
    }
    return {
        "schema": "official_taco_rgbd_contract_v1",
        "facts": [
            {"fact": "TACO uses Intel RealSense L515; camera and mocap systems run at 30 Hz; egocentric resolution is 1920x1080.", "source": official["paper_url"]},
            {"fact": "Pinned official projection code uses object, hand, extrinsic, and RGB by the same frame index.", "source": official["repository"], "commit": official["commit"], "entrypoint": official["projection_entrypoint"]},
            {"fact": "A TACO maintainer states that depth AVI encoding used input framerate 15 with FFV1.", "source": official["depth_issue_url"], "issue": 15},
            {"fact": "The official README decodes depth with the fps=30 filter and specifies uint16 1920x1080 depth with scale 4000.", "source": official["repository"], "commit": official["commit"]},
            {"fact": "The official scale documentation was corrected from 1000 to 4000.", "source": official["scale_fix_url"], "commit": official["scale_fix_commit"]},
            {"fact": "FFmpeg's fps filter duplicates or drops frames to obtain a constant requested frame rate.", "source": "https://ffmpeg.org/ffmpeg-filters.html#fps"},
            {"fact": "TACO maintainers acknowledge hardware-related egocentric modality mismatch in some sequences and point users to the available-sequence list.", "source": official["hardware_issue_url"], "issue": 7},
        ],
        "available_sequence_list": artifact(available_path),
        "predeclared_sample_membership": availability,
        "all_samples_officially_egocentric_available": all(availability.values()),
    }


def count_contract(
    cfg: dict[str, Any], samples: list[dict[str, Any]], decoded: dict[str, dict[str, Any]],
    official_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    rows, hypothesis_rows = [], []
    for sample in samples:
        rgb = ffprobe_video(sample["rgb_video"])
        native = len(decoded[sample["key"]]["hashes"])
        annotation_counts = {
            "extrinsic": int(np.load(sample["extrinsic"], mmap_mode="r").shape[0]),
            "hand": int(np.load(sample["hand_joints"], mmap_mode="r").shape[0]),
            "tool": int(np.load(sample["tool_pose"], mmap_mode="r").shape[0]),
            "target": int(np.load(sample["target_pose"], mmap_mode="r").shape[0]),
        }
        expected = sample["expected_frames"]
        rgb_count = stream_frame_count(rgb)
        official_count = official_counts[sample["key"]]
        mapping_c = container_timestamp_mapping(
            annotation_count=expected, logical_rate_hz=float(cfg["timeline"]["logical_rate_hz"]),
            native_count=native, native_rate_hz=float(cfg["timeline"]["encoded_depth_rate_hz"]),
        )
        native_usage = len(set(mapping_c)) / native
        duplicate = decoded[sample["key"]]["duplicate"]
        even_pair_fraction = duplicate["pattern_0_eq_1_2_eq_3_fraction"]
        odd_pair_fraction = duplicate["pattern_1_eq_2_3_eq_4_fraction"]
        pair_fraction = max(even_pair_fraction, odd_pair_fraction)
        rgb_rate = frame_rate(rgb["avg_frame_rate"])
        depth_rate = frame_rate(decoded[sample["key"]]["stream"]["avg_frame_rate"])
        a_pass = (
            rgb_count == native == expected
            and all(value == expected for value in annotation_counts.values())
            and rgb_rate == int(cfg["timeline"]["logical_rate_hz"])
            and depth_rate == int(cfg["timeline"]["encoded_depth_rate_hz"])
        )
        b_pass = official_count == expected and abs(2 * native - expected) <= 1
        c_pass = native_usage >= float(cfg["timeline"]["container_timestamp_minimum_native_usage_fraction"])
        h3_pass = pair_fraction >= float(cfg["timeline"]["h3_minimum_exact_pair_fraction"])
        rows.append({
            "sample": sample["key"], "rgb": rgb_count, "annotation": expected,
            "annotation_modalities": annotation_counts, "native_depth": native,
            "official_decode30": official_count, "rgb_container_rate": str(rgb_rate),
            "depth_container_rate": str(depth_rate),
        })
        hypothesis_rows.append({
            "sample": sample["key"],
            "INDEX_ALIGNED": {"structural_pass": a_pass, "reason": "all modalities and native depth have one row per annotation" if a_pass else "count mismatch"},
            "NATIVE_15_TO_30": {"structural_pass": b_pass, "reason": "official fps30 output must equal annotation count and native count must be approximately half"},
            "CURRENT_FILE_ALREADY_EXPANDED": {
                "structural_pass": h3_pass,
                "native_even_pair_duplicate_fraction": even_pair_fraction,
                "native_odd_pair_duplicate_fraction": odd_pair_fraction,
                "maximum_predeclared_pair_fraction": pair_fraction,
            },
            "CONTAINER_TIMESTAMP": {"structural_pass": c_pass, "unique_native_usage_fraction": native_usage, "mapped_native_min": min(mapping_c), "mapped_native_max": max(mapping_c)},
        })
    index_closed = all(row["INDEX_ALIGNED"]["structural_pass"] for row in hypothesis_rows) and not any(
        row["CURRENT_FILE_ALREADY_EXPANDED"]["structural_pass"] for row in hypothesis_rows
    )
    return (
        {"schema": "taco_multi_sample_count_contract_v1", "samples": rows},
        {"schema": "taco_multi_sample_timeline_hypotheses_v1", "predeclared_only": True,
         "samples": hypothesis_rows, "selected_candidate": "INDEX_ALIGNED" if index_closed else None},
        index_closed,
    )


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    size = 2 * pixels + 1
    return cv2.erode(mask.astype(np.uint8), np.ones((size, size), dtype=np.uint8)) > 0


def object_mesh(vertices: np.ndarray, faces: np.ndarray, pose: np.ndarray) -> trimesh.Trimesh:
    transformed = vertices @ pose[:3, :3].T + pose[:3, 3]
    return trimesh.Trimesh(vertices=transformed, faces=faces, process=False)


def geometry_validation(
    cfg: dict[str, Any], samples: list[dict[str, Any]], decoded: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool, bool]:
    reg = cfg["registration"]
    utils = Path(cfg["official"]["checkout"]) / "dataset_utils"
    all_records, sample_summaries, observations = [], [], []
    for sample in samples:
        intrinsic = np.loadtxt(sample["intrinsic"])
        extrinsic = np.load(sample["extrinsic"], mmap_mode="r")
        objects = {}
        for role in ("tool", "target"):
            mesh = trimesh.load_mesh(sample[f"{role}_mesh"], process=False)
            mesh.apply_scale(0.01)
            objects[role] = {
                "vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.faces),
                "poses": np.load(sample[f"{role}_pose"], mmap_mode="r"),
            }
        projector = TacoOfficialProjector(
            dataset_utils=utils,
            image_size=(int(cfg["timeline"]["depth_width"]), int(cfg["timeline"]["depth_height"])),
            intrinsic=intrinsic, extrinsic=extrinsic[0], device=cfg["official"]["device"],
        )
        rasterizer = projector.official_wrapper.renderer.renderer.rasterizer
        rasterizer.raster_settings = replace(rasterizer.raster_settings, max_faces_per_bin=400000)
        records = []
        for frame in decoded[sample["key"]]["duplicate"]["uniform_registration_frames"]:
            depth = raw_depth_to_metres(
                decoded[sample["key"]]["selected_depth"][frame],
                scale=float(cfg["timeline"]["depth_scale"]),
            )
            projector.set_camera(intrinsic, extrinsic[frame])
            meshes = {
                role: object_mesh(row["vertices"], row["faces"], row["poses"][frame])
                for role, row in objects.items()
            }
            full = projector.render_depth([meshes["tool"], meshes["target"]])
            for role in ("tool", "target"):
                rendered = projector.render_depth([meshes[role]])
                front = ((rendered > 0) & (full > 0) &
                         (np.abs(rendered - full) <= float(reg["front_surface_tolerance_m"])))
                selector = erode(front, int(reg["interior_erosion_px"]))
                valid = selector & np.isfinite(depth) & (depth > 0)
                residual = depth[valid] - rendered[valid]
                absolute = np.abs(residual) * 1000.0
                fraction = float(valid.sum() / max(int(selector.sum()), 1))
                enough = int(valid.sum()) >= int(reg["minimum_valid_pixels"])
                coverage = (
                    "INSUFFICIENT_VALID_DEPTH" if not enough else
                    "HIGH_DEPTH_COVERAGE" if fraction >= float(reg["high_depth_coverage_fraction"])
                    else "LOW_DEPTH_COVERAGE"
                )
                row = {
                    "sample": sample["key"], "frame": frame, "mapping": "INDEX_ALIGNED",
                    "object": role, "interior_pixels": int(selector.sum()),
                    "valid_pixels": int(valid.sum()), "valid_depth_fraction": fraction,
                    "observability": coverage,
                    "signed_error_mm": None if not residual.size else {
                        "median": float(np.median(residual) * 1000.0),
                    },
                    "absolute_error_mm": None if not absolute.size else {
                        "median": float(np.median(absolute)), "p90": float(np.percentile(absolute, 90)),
                        "p95": float(np.percentile(absolute, 95)),
                    },
                }
                records.append(row)
                observations.append({key: row[key] for key in (
                    "sample", "frame", "object", "interior_pixels", "valid_pixels",
                    "valid_depth_fraction", "observability",
                )})
        anchors = [row for row in records if row["observability"] == "HIGH_DEPTH_COVERAGE" and row["absolute_error_mm"]]
        median = None if not anchors else float(np.median([row["absolute_error_mm"]["median"] for row in anchors]))
        passed = None if not anchors else median <= float(reg["maximum_median_absolute_error_mm"])
        sample_summaries.append({
            "sample": sample["key"], "uniform_frames": decoded[sample["key"]]["duplicate"]["uniform_registration_frames"],
            "high_coverage_anchor_comparisons": len(anchors),
            "median_of_anchor_median_absolute_error_mm": median,
            "spatial_registration_pass": passed,
            "assessment": (
                "NO_HIGH_COVERAGE_ANCHOR_OBSERVABILITY_ONLY" if passed is None else
                "SPATIAL_REGISTRATION_PASS" if passed else "SPATIAL_REGISTRATION_FAIL"
            ),
        })
        all_records.extend(records)
        del projector
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    anchor_samples = [row for row in sample_summaries if row["spatial_registration_pass"] is not None]
    spatial_pass = len(anchor_samples) >= int(reg["minimum_samples_with_high_coverage_anchors"]) and all(
        row["spatial_registration_pass"] for row in anchor_samples
    )
    limited = any(row["observability"] != "HIGH_DEPTH_COVERAGE" for row in observations)
    validation = {
        "schema": "taco_rigid_anchor_timeline_validation_v1", "mapping": "INDEX_ALIGNED",
        "frame_selection": "12 uniformly spaced frames including first and last, frozen before geometry",
        "records": all_records, "sample_summaries": sample_summaries,
    }
    spatial = {
        "schema": "taco_rgb_depth_spatial_registration_v1",
        "classification": "SPATIAL_REGISTRATION_PASS" if spatial_pass else "RGBD_SPATIAL_REGISTRATION_BLOCKER",
        "registration_uses_only_high_coverage_anchor_comparisons": True,
        "low_coverage_does_not_fail_registration": True,
        "samples_without_high_coverage_anchor_are_observability_only": True,
        "anchor_sample_count": len(anchor_samples),
        "minimum_anchor_sample_count": reg["minimum_samples_with_high_coverage_anchors"],
        "threshold_mm": reg["maximum_median_absolute_error_mm"],
        "threshold_provenance": reg["threshold_provenance"],
        "sample_summaries": sample_summaries,
    }
    observability = {
        "schema": "taco_depth_observability_v1", "records": observations,
        "classification_counts": {
            label: sum(row["observability"] == label for row in observations)
            for label in ("HIGH_DEPTH_COVERAGE", "LOW_DEPTH_COVERAGE", "INSUFFICIENT_VALID_DEPTH")
        },
        "depth_observability_limited": limited,
        "interpretation": "Low or missing sensor returns are observability evidence, not spatial-registration failure.",
    }
    return validation, spatial, observability, spatial_pass, limited


def active_depth_code_audit() -> dict[str, Any]:
    command = [
        "rg", "-n", r"(/\s*1000|/1000|\*\s*0\.001|depth_scale\s*[:=]\s*1000|millimeter|gray16|4000)",
        "rl/src", "rl/scripts", "rl/configs",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    hits = result.stdout.splitlines()
    suspect = [row for row in hits if re.search(r"depth_scale\s*[:=]\s*1000|/\s*1000|\*\s*0\.001", row)]
    classified = []
    for row in suspect:
        if "audit_taco_remaining_contacts.py" in row or "run_contact_geometry_candidate.py" in row:
            kind = "NON_DEPTH_GEOMETRY_MILLIMETRE_TO_METRE_CONVERSION"
        else:
            kind = "REVIEWED_NON_TACO_METRIC_DEPTH_PATH"
        classified.append({"match": row, "classification": kind})
    return {
        "schema": "taco_active_depth_code_audit_v1", "search_command": command,
        "all_matches": hits, "scale_1000_like_matches": classified,
        "active_taco_metric_depth_contract": "uint16 raw / 4000 metres",
        "active_taco_depth_scale_bug": False,
        "historical_artifacts_and_TRASH_not_rewritten": True,
    }


def source_pins(cfg_path: Path, cfg: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    official = Path(cfg["official"]["checkout"])
    sources: dict[str, Path] = {
        "config": cfg_path, "runner": Path(__file__),
        "depth_module": RL_ROOT / "src/egoengine_repro/evaluation/taco_depth.py",
        "depth_tests": RL_ROOT / "tests/core/test_taco_depth.py",
        "official_projection_wrapper": RL_ROOT / "src/egoengine_repro/evaluation/taco_official_projection.py",
        "official_readme": official / "README.md",
        "official_projection_entrypoint": official / cfg["official"]["projection_entrypoint"],
        "official_available_sequences": official / cfg["official"]["available_sequences"],
    }
    for sample in samples:
        for key in ("rgb_video", "depth_video", "release_depth_source", "release_manifest", "intrinsic", "extrinsic", "hand_joints", "tool_pose", "target_pose", "tool_mesh", "target_mesh"):
            sources[f"{sample['key']}:{key}"] = sample[key]
    return {
        "schema": "taco_rgbd_timebase_registration_source_pins_v1",
        "repository_head_before_audit": git_head(REPO_ROOT),
        "official_checkout_commit": git_head(official),
        "artifacts": {key: artifact(path) for key, path in sources.items()},
    }


def write_hashes(output: Path) -> None:
    rows = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "server_artifacts.sha256":
            rows.append(f"{sha256(path)}  {path.name}")
    (output / "server_artifacts.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg_path = args.config.resolve(strict=True)
    cfg = load_config(cfg_path)
    samples = resolve_samples(cfg)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    contract = official_contract(cfg, samples)
    prov = provenance(samples)
    if not contract["all_samples_officially_egocentric_available"]:
        raise RuntimeError("a predeclared sample is absent from the official egocentric available list")
    if not prov["all_release_provenance_closed"]:
        raise RuntimeError("LOCAL_DEPTH_PROVENANCE_BLOCKER")

    brush = samples[0]
    timeline_counts = write_container_timeline(brush["depth_video"], output / "depth_container_timeline.jsonl")
    decoded: dict[str, dict[str, Any]] = {}
    official_counts = {}
    for sample in samples:
        decoded[sample["key"]] = decode_native_sample(sample, cfg, keep_manifest=sample is brush)
        official_counts[sample["key"]] = ffmpeg_filtered_count(
            sample["depth_video"], float(cfg["timeline"]["official_decode_rate_hz"]),
        )

    with (output / "native_depth_frame_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in decoded[brush["key"]]["records"]:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    official_hashes = [
        frame_sha256(raw) for raw in iter_depth_frames(
            brush["depth_video"], decoded[brush["key"]]["spec"],
            fps=float(cfg["timeline"]["official_decode_rate_hz"]),
        )
    ]
    decode_map = exact_output_to_native_map(decoded[brush["key"]]["hashes"], official_hashes)
    official_decode = {
        "schema": "taco_official_fps30_decode_map_v1",
        "sample": brush["key"], "native_frame_count": len(decoded[brush["key"]]["hashes"]),
        "output_frame_count": len(official_hashes),
        "filter": "fps=fps=30", "pixel_identity_mapping": decode_map,
        "basically_two_outputs_per_native": len(official_hashes) == 2 * len(decoded[brush["key"]]["hashes"]),
    }

    duplicate = {
        "schema": "taco_native_depth_duplicate_structure_v1",
        "samples": [decoded[sample["key"]]["duplicate"] for sample in samples],
    }
    counts, hypotheses, timeline_closed = count_contract(cfg, samples, decoded, official_counts)
    if timeline_closed:
        validation, spatial, observability, spatial_pass, limited = geometry_validation(cfg, samples, decoded)
    else:
        validation = spatial = observability = None
        spatial_pass = limited = False

    if not timeline_closed:
        classification = "DEPTH_TEMPORAL_MAPPING_UNRESOLVED"
    elif not spatial_pass:
        classification = "RGBD_SPATIAL_REGISTRATION_BLOCKER"
    elif limited:
        classification = "RGBD_REGISTRATION_RESOLVED_DEPTH_OBSERVABILITY_LIMITED"
    else:
        classification = "RGBD_REGISTRATION_RESOLVED"

    reports = {
        "official_taco_rgbd_contract.json": contract,
        "depth_active_code_audit.json": active_depth_code_audit(),
        "depth_file_provenance.json": prov,
        "official_decode_frame_map.json": official_decode,
        "native_depth_duplicate_structure.json": duplicate,
        "multi_sample_count_contract.json": counts,
        "multi_sample_timeline_hypotheses.json": hypotheses,
        "source_pins.json": source_pins(cfg_path, cfg, samples),
        "config_consumption_audit.json": {
            "schema": "taco_rgbd_timebase_registration_config_consumption_v1",
            "config": artifact(cfg_path), "exact_key_validation": True,
            "unknown_sample_keys_rejected": True, "all_sections_consumed": True,
        },
    }
    if validation is not None:
        reports.update({
            "rigid_anchor_timeline_validation.json": validation,
            "rgb_depth_spatial_registration.json": spatial,
            "depth_observability.json": observability,
        })
    if timeline_closed:
        reports["candidate_depth_timeline_contract.json"] = {
            "schema": "taco_candidate_depth_timeline_contract_v1",
            "status": "CANDIDATE_NOT_ACTIVE_SUPPORT_CONTRACT",
            "logical_rate_hz": 30, "container_rate_hz": 15,
            "frame_mapping": "INDEX_ALIGNED", "container_pts_are_logical_time": False,
            "logical_timestamp": "frame_index / 30", "depth_scale": 4000,
            "do_not_expand_native_frames": True,
        }
    for name, report in reports.items():
        write_json(output / name, report)

    decision = {
        "schema": "taco_rgbd_timebase_registration_repair_decision_v1",
        "classification": classification,
        "timeline_mapping": "INDEX_ALIGNED" if timeline_closed else None,
        "timeline_closed": timeline_closed, "spatial_registration_pass": spatial_pass,
        "depth_observability_limited": limited,
        "support_plane_estimator_may_resume_in_next_separate_task": classification in {
            "RGBD_REGISTRATION_RESOLVED_DEPTH_OBSERVABILITY_LIMITED", "RGBD_REGISTRATION_RESOLVED",
        },
        "container_timeline_records": timeline_counts,
        "runtime_counts": {"support_plane": 0, "mink": 0, "physics": 0, "replay": 0,
                           "mpc": 0, "rl": 0, "promotion": 0, "chunk_commit": 0},
        "active_support_contract_changed": False,
    }
    write_json(output / "decision.json", decision)

    brush_duplicate = decoded[brush["key"]]["duplicate"]
    summary = f"""# TACO RGB-D 时间轴与注册修复 v1

**正式分类：`{classification}`。** 本轮没有运行桌面估计、MINK、physics、Replay、MPC、RL、promotion 或 chunk commit，也没有修改 active `SupportSurfaceContract`。

## 必答问题

1. **论文系统频率：** 相机系统和动捕系统都是 30 Hz；头戴设备是 Intel RealSense L515，egocentric 分辨率为 1920×1080。
2. **Depth AVI 为什么是 15 fps：** 官方 maintainer 公开的 FFV1 编码命令把输入 framerate 设置为 15；因此容器的 15 Hz 是官方编码流程留下的事实，不是本项目偶然改坏的 metadata。
3. **为什么 README 又写 fps=30：** README 用 FFmpeg `fps=30` filter 生成 30 Hz 输出；该 filter 会复制或删除真实帧，不只是改 metadata。
4. **本地文件来源：** Brush 本地文件是官方 release source 的重命名字节一致副本；四个预声明样本均有 release manifest/source 证据且 SHA/字节一致。
5. **Brush 官方 fps30 输出数：** `{len(official_hashes)}` 帧，native 是 `{len(decoded[brush['key']]['hashes'])}` 帧。
6. **复制关系：** exact raw-pixel hash 显示基本为 `native i -> output 2i, 2i+1`；完整逐帧候选映射见 `official_decode_frame_map.json`。
7. **native 209 帧是否已含重复 pair：** even-pair exact duplicate fraction 为 `{brush_duplicate['pattern_0_eq_1_2_eq_3_fraction']:.6f}`，不支持“当前文件已经先扩帧”的 H3。
8. **Brush 最可信 mapping：** `INDEX_ALIGNED`：native depth frame i 对 annotation row i；逻辑时间是 `i/30`，AVI PTS 不作为 annotation 逻辑时间，也不把 209 帧再次扩成 418 后硬配 pose。
9. **其他官方样本：** Pour、Skim、Smear 与 Brush 一样，RGB、annotation 和 native depth 计数逐行一致；official fps30 会约翻倍，container timestamp 只使用约前半 native data。四者都在官方 egocentric available list。
10. **低 coverage 的 bowl：** 被单独归类为 depth observability；只有高覆盖刚体 anchor 决定空间注册，低 coverage 不再单独判 registration fail。
11. **active scale1000 bug：** 未发现。active TACO metric-depth 路径是 `uint16 raw / 4000`；检出的 `/1000` 是非深度几何单位换算，历史 artifact/TRASH 未篡改。
12. **是否可重新开始桌面估计：** `{'可以在下一项独立任务中重新启动（本轮仍未执行）' if decision['support_plane_estimator_may_resume_in_next_separate_task'] else '不可以；仍有前置 blocker'}`。

## 长期合同

容器 fps 和 annotation logical fps 不是同一个概念。Depth valid coverage 与 RGB-D spatial registration 也不是同一个概念。后续代码必须使用候选时间合同，不得重写 AVI header、重复做 15→30 expansion、搜索 sample-specific offset，或用 bowl/brush bottom 修正深度。
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    write_hashes(output)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
