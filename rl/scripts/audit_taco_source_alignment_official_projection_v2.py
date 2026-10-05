#!/usr/bin/env python3
"""Audit TACO source alignment using only the pinned official projection path.

No physics, control, planning, retargeting or reinforcement learning is run.
The official TACO renderer owns every world-to-camera and camera-to-pixel
operation; this file only loads inputs and analyzes its rendered products.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from zipfile import ZipFile

import cv2
import numpy as np
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoengine_repro.evaluation.taco_official_projection import (  # noqa: E402
    TacoOfficialProjector,
    load_official_hand_sequence,
    official_overlay,
)


DEFAULT_CONFIG = ROOT / "configs/taco_source_alignment_official_projection_v2.yaml"
CLASS_ORDER = ("right_hand", "left_hand", "tool", "target")
CLASS_COLORS = {
    "right_hand": (255, 60, 60),
    "left_hand": (60, 255, 60),
    "tool": (60, 110, 255),
    "target": (255, 60, 255),
}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def member_artifact(archive: Path, member: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with ZipFile(archive) as zip_file, zip_file.open(member) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        info = zip_file.getinfo(member)
    return {
        "archive": str(archive.resolve()),
        "member": member,
        "bytes": info.file_size,
        "compressed_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}",
        "sha256": digest.hexdigest(),
    }


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True,
    ).strip()


def video_metadata(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,"
            "nb_frames,nb_read_frames,duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream: {path}")
    return streams[0]


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if value.get("schema") != "taco_source_alignment_official_projection_v2":
        raise ValueError("unexpected source-alignment contract")
    if any(value["authorization"].values()):
        raise ValueError("source-alignment audit must authorize zero runtime work")
    expected = value["expected_baseline"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected, git_head(ROOT)],
        cwd=ROOT, check=True,
    )
    official = Path(value["official"]["checkout"])
    if git_head(official) != value["official"]["commit"]:
        raise ValueError("official TACO checkout does not match the frozen commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=official, text=True,
    ).strip():
        raise ValueError("tracked official TACO source is dirty")
    return value


def video_frames(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    frame = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if frame in indices:
            result[frame] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame += 1
    capture.release()
    if set(result) != indices:
        raise RuntimeError(f"missing RGB frames in {path}: {sorted(indices - set(result))}")
    return result


def depth_frames(
    path: Path, indices: set[int], *, count: int, height: int, width: int,
    scale: float,
) -> dict[int, np.ndarray]:
    process = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "gray16le", "-"], stdout=subprocess.PIPE,
    )
    result: dict[int, np.ndarray] = {}
    frame_bytes = height * width * 2
    assert process.stdout is not None
    for frame in range(count):
        payload = process.stdout.read(frame_bytes)
        if len(payload) != frame_bytes:
            raise RuntimeError(f"depth frame {frame} is truncated")
        if frame in indices:
            result[frame] = (
                np.frombuffer(payload, dtype="<u2").reshape(height, width).copy()
                .astype(np.float32) / float(scale)
            )
    process.stdout.close()
    if process.wait() != 0 or set(result) != indices:
        raise RuntimeError("depth decode failed")
    return result


def mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def load_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = {key: Path(value) for key, value in cfg["inputs"].items()
             if key not in {"segmentation_prefix"}}
    sequence = Path(cfg["sequence"]["triplet"]) / cfg["sequence"]["name"]
    official_utils = Path(cfg["official"]["checkout"]) / "dataset_utils"
    hands: dict[str, Any] = {}
    for side in ("right", "left"):
        pose = paths["hand_pose_dir"] / f"{side}_hand.pkl"
        shape = paths["hand_pose_dir"] / f"{side}_hand_shape.pkl"
        vertices, joints, faces, weights = load_official_hand_sequence(
            dataset_utils=official_utils, pose_path=pose, shape_path=shape,
            side=side, device=cfg["official"]["device"],
        )
        hands[side] = {
            "vertices": vertices, "joints": joints, "faces": faces,
            "weights": weights, "pose_path": pose, "shape_path": shape,
        }
    objects: dict[str, Any] = {}
    for role in ("tool", "target"):
        object_id = cfg["objects"][role]["id"]
        model_path = paths["object_model_dir"] / f"{object_id}_cm.obj"
        pose_path = paths["object_pose_dir"] / f"{role}_{object_id}.npy"
        value = trimesh.load_mesh(model_path, process=False)
        value.apply_scale(0.01)
        objects[role] = {
            "vertices": np.asarray(value.vertices), "faces": np.asarray(value.faces),
            "poses": np.load(pose_path, allow_pickle=False),
            "model_path": model_path, "pose_path": pose_path,
        }
    return {
        "paths": paths,
        "sequence": sequence,
        "official_utils": official_utils,
        "hands": hands,
        "objects": objects,
        "intrinsic": np.loadtxt(paths["egocentric_intrinsic"]),
        "extrinsics": np.load(paths["egocentric_extrinsic"], allow_pickle=False),
        "video_metadata": {
            "rgb": video_metadata(paths["rgb_video"]),
            "depth": video_metadata(paths["depth_video"]),
        },
    }


def frame_meshes(data: dict[str, Any], frame: int) -> dict[str, trimesh.Trimesh]:
    result = {
        "right_hand": mesh(data["hands"]["right"]["vertices"][frame],
                           data["hands"]["right"]["faces"]),
        "left_hand": mesh(data["hands"]["left"]["vertices"][frame],
                          data["hands"]["left"]["faces"]),
    }
    for role in ("tool", "target"):
        row = data["objects"][role]
        pose = row["poses"][frame]
        vertices = row["vertices"] @ pose[:3, :3].T + pose[:3, 3]
        result[role] = mesh(vertices, row["faces"])
    return result


def official_projection_baseline(
    cfg: dict[str, Any], data: dict[str, Any], output: Path,
    rgbs: dict[int, np.ndarray], projector: TacoOfficialProjector,
) -> dict[str, Any]:
    rows = []
    for frame in cfg["sequence"]["focus_frames"]:
        projector.set_camera(data["intrinsic"], data["extrinsics"][frame])
        values = frame_meshes(data, frame)
        groups = {
            "hands": [values["right_hand"], values["left_hand"]],
            "objects": [values["tool"], values["target"]],
            "scene": [values[name] for name in CLASS_ORDER],
        }
        raw_path = output / "official_egocentric/raw" / f"frame_{frame:03d}.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(raw_path), cv2.cvtColor(rgbs[frame], cv2.COLOR_RGB2BGR))
        row = {"frame": frame, "raw_rgb": str(raw_path)}
        for name, meshes in groups.items():
            rendered = np.clip(projector.render_rgb(meshes) * 255.0, 0, 255).astype(np.uint8)
            overlaid = official_overlay(
                dataset_utils=data["official_utils"], rgb=rgbs[frame], render=rendered,
            )
            render_path = output / "official_egocentric/render" / f"frame_{frame:03d}_{name}.png"
            overlay_path = output / "official_egocentric/overlay" / f"frame_{frame:03d}_{name}.png"
            render_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(render_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
            row[f"{name}_render"] = str(render_path)
            row[f"{name}_overlay"] = str(overlay_path)
        rows.append(row)
    return {
        "schema": "taco_official_egocentric_projection_manifest_v2",
        "official_entrypoint": str(
            data["official_utils"] / "project_pose_to_egocentric_view.py"
        ),
        "projection_implementation": str(
            data["official_utils"] / "pyt3d_wrapper.py"
        ),
        "manual_pixel_offset": False,
        "alternate_camera_convention_tested": False,
        "frames": rows,
    }


def eroded(mask_value: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask_value.astype(bool)
    size = 2 * pixels + 1
    return cv2.erode(mask_value.astype(np.uint8), np.ones((size, size), np.uint8)) > 0


def residual_stats(
    observed: np.ndarray, rendered: np.ndarray, selector: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    interior_count = int(selector.sum())
    valid = selector & np.isfinite(observed) & (observed > 0) & (rendered > 0)
    values = observed[valid] - rendered[valid]
    if not len(values):
        stats = {
            "interior_pixels": interior_count, "valid_pixels": 0,
            "valid_pixel_fraction": 0.0, "signed_error_median_mm": None,
            "absolute_error_mm": None,
        }
    else:
        absolute = np.abs(values) * 1000.0
        stats = {
            "interior_pixels": interior_count,
            "valid_pixels": int(len(values)),
            "valid_pixel_fraction": float(len(values) / max(interior_count, 1)),
            "signed_error_median_mm": float(np.median(values) * 1000.0),
            "absolute_error_mm": {
                "median": float(np.median(absolute)),
                "p90": float(np.percentile(absolute, 90)),
                "p95": float(np.percentile(absolute, 95)),
            },
        }
    residual_image = np.full(observed.shape, np.nan, np.float32)
    residual_image[valid] = observed[valid] - rendered[valid]
    return stats, valid, residual_image


def depth_color(values: np.ndarray, *, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((values - lo) / max(hi - lo, 1e-8), 0, 1)
    image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~np.isfinite(values) | (values <= 0)] = 0
    return image


def error_color(values: np.ndarray, limit_m: float, absolute: bool = False) -> np.ndarray:
    if absolute:
        scaled = np.clip(np.abs(values) / limit_m, 0, 1)
        image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    else:
        scaled = np.clip(values / limit_m, -1, 1)
        image = cv2.applyColorMap(((scaled + 1) * 127.5).astype(np.uint8), cv2.COLORMAP_JET)
    image[~np.isfinite(values)] = 0
    return image


def object_depth_control(
    cfg: dict[str, Any], data: dict[str, Any], output: Path,
    depths: dict[int, np.ndarray], projector: TacoOfficialProjector,
) -> dict[str, Any]:
    erosion = int(cfg["analysis"]["interior_erosion_px"])
    tolerance = float(cfg["analysis"]["front_surface_tolerance_m"])
    records = []
    for observed_frame in cfg["sequence"]["focus_frames"]:
        for offset in cfg["sequence"]["timing_offsets"]:
            annotation_frame = observed_frame + offset
            if not 0 <= annotation_frame < cfg["sequence"]["frame_count"]:
                continue
            projector.set_camera(data["intrinsic"], data["extrinsics"][annotation_frame])
            values = frame_meshes(data, annotation_frame)
            full = projector.render_depth([values[name] for name in CLASS_ORDER])
            object_depths = {
                role: projector.render_depth([values[role]]) for role in ("tool", "target")
            }
            for role, rendered in object_depths.items():
                front = (rendered > 0) & (full > 0) & (np.abs(rendered - full) <= tolerance)
                selector = eroded(front, erosion)
                stats, valid, error = residual_stats(depths[observed_frame], rendered, selector)
                record = {
                    "observed_depth_frame": observed_frame,
                    "annotation_frame": annotation_frame,
                    "timing_offset": offset,
                    "object": role,
                    **stats,
                }
                records.append(record)
                if offset == 0:
                    directory = output / "object_depth" / f"frame_{observed_frame:03d}" / role
                    directory.mkdir(parents=True, exist_ok=True)
                    visible_render = rendered.copy()
                    visible_render[~selector] = np.nan
                    cv2.imwrite(str(directory / "observed_depth.png"),
                                depth_color(depths[observed_frame], lo=0.3, hi=1.2))
                    cv2.imwrite(str(directory / "rendered_depth.png"),
                                depth_color(visible_render, lo=0.3, hi=1.2))
                    cv2.imwrite(str(directory / "signed_error.png"),
                                error_color(error, 0.05, absolute=False))
                    cv2.imwrite(str(directory / "absolute_error.png"),
                                error_color(error, 0.05, absolute=True))
                    cv2.imwrite(str(directory / "valid_pixels.png"), valid.astype(np.uint8) * 255)

    current = [row for row in records if row["timing_offset"] == 0]
    usable = [row for row in current if row["absolute_error_mm"] is not None]
    by_offset = {}
    for offset in cfg["sequence"]["timing_offsets"]:
        rows = [row for row in records if row["timing_offset"] == offset
                and row["absolute_error_mm"] is not None]
        by_offset[str(offset)] = {
            "comparisons": len(rows),
            "median_of_median_absolute_error_mm": float(np.median([
                row["absolute_error_mm"]["median"] for row in rows
            ])) if rows else None,
            "median_valid_pixel_fraction": float(np.median([
                row["valid_pixel_fraction"] for row in rows
            ])) if rows else None,
        }
    per_observation_best = []
    for frame in cfg["sequence"]["focus_frames"]:
        for role in ("tool", "target"):
            rows = [row for row in records if row["observed_depth_frame"] == frame
                    and row["object"] == role and row["absolute_error_mm"] is not None]
            if rows:
                best = min(rows, key=lambda row: row["absolute_error_mm"]["median"])
                per_observation_best.append({
                    "frame": frame, "object": role,
                    "best_offset": best["timing_offset"],
                    "best_median_absolute_error_mm": best["absolute_error_mm"]["median"],
                })
    median_error = float(np.median([
        row["absolute_error_mm"]["median"] for row in usable
    ])) if usable else float("inf")
    median_coverage = float(np.median([
        row["valid_pixel_fraction"] for row in usable
    ])) if usable else 0.0
    minimum_coverage = float(min(
        (row["valid_pixel_fraction"] for row in usable), default=0.0,
    ))
    zero_best_fraction = float(np.mean([
        row["best_offset"] == 0 for row in per_observation_best
    ])) if per_observation_best else 0.0
    per_object = {}
    for role in ("tool", "target"):
        rows = [row for row in usable if row["object"] == role]
        per_object[role] = {
            "comparisons": len(rows),
            "minimum_valid_pixel_fraction": float(min(
                (row["valid_pixel_fraction"] for row in rows), default=0.0,
            )),
            "median_valid_pixel_fraction": float(np.median([
                row["valid_pixel_fraction"] for row in rows
            ])) if rows else None,
            "median_of_median_absolute_error_mm": float(np.median([
                row["absolute_error_mm"]["median"] for row in rows
            ])) if rows else None,
        }
    # Registration control thresholds apply only to the rigid-object gate.  The
    # later hand decision is relative to these measured object errors.
    minimum_pixels = int(cfg["analysis"]["minimum_valid_pixels"])
    minimum_fraction = float(
        cfg["analysis"]["minimum_valid_pixel_fraction_per_rigid_pair"]
    )
    maximum_error = float(
        cfg["analysis"]["maximum_median_of_medians_absolute_error_mm"]
    )
    passed = (
        len(usable) == 2 * len(cfg["sequence"]["focus_frames"])
        and all(row["valid_pixels"] >= minimum_pixels for row in usable)
        and minimum_coverage >= minimum_fraction
        and median_error <= maximum_error
    )
    return {
        "schema": "taco_object_depth_control_v2",
        "classification": (
            "DEPTH_CAMERA_CONTROL_PASS" if passed
            else "CAMERA_DEPTH_REGISTRATION_UNRESOLVED"
        ),
        "depth_decode": "ffmpeg gray16le; uint16 / 4000 metres",
        "video_stream_metadata": data["video_metadata"],
        "frame_index_policy": (
            "endpoint 0 = RGB frame 0 = depth frame 0; no rate-based resampling or "
            "automatic timing-offset selection"
        ),
        "container_rate_mismatch_observed": (
            data["video_metadata"]["rgb"].get("avg_frame_rate")
            != data["video_metadata"]["depth"].get("avg_frame_rate")
        ),
        "depth_rgb_registration_assumed_before_gate": False,
        "comparison": "raw depth minus official-rendered frontmost rigid-object depth",
        "interior_erosion_px": erosion,
        "records": records,
        "timing_summary": by_offset,
        "timing_best_by_frame_object": per_observation_best,
        "per_object_aggregate": per_object,
        "aggregate": {
            "median_of_median_absolute_error_mm": median_error,
            "median_valid_pixel_fraction": median_coverage,
            "minimum_pair_valid_pixel_fraction": minimum_coverage,
            "zero_offset_best_fraction": zero_best_fraction,
        },
        "gate_contract": {
            "all_frame_object_pairs_required": True,
            "minimum_valid_pixels_per_pair": minimum_pixels,
            "minimum_valid_pixel_fraction_per_pair": minimum_fraction,
            "median_of_median_absolute_error_mm_max": maximum_error,
            "timing_offsets_are_diagnostic_not_selected": True,
        },
    }


def hand_vertex_regions(data: dict[str, Any]) -> tuple[np.ndarray, dict[int, str]]:
    hand = data["hands"]["left"]
    dominant = np.argmax(hand["weights"], axis=1)
    labels = np.full(len(dominant), -1, np.int16)
    names = {
        0: "palm_wrist",
        1: "ring_proximal",
        2: "ring_middle_distal",
        3: "ring_tip",
        4: "pinky_proximal",
        5: "pinky_middle_distal",
        6: "pinky_tip",
    }
    labels[dominant == 0] = 0
    vertices = hand["vertices"][0]
    for proximal, middle, distal, tip_index, codes in (
        (10, 11, 12, 556, (1, 2, 3)),
        (7, 8, 9, 673, (4, 5, 6)),
    ):
        labels[dominant == proximal] = codes[0]
        labels[(dominant == middle) | (dominant == distal)] = codes[1]
        candidates = np.flatnonzero(dominant == distal)
        order = candidates[np.argsort(np.linalg.norm(
            vertices[candidates] - vertices[tip_index], axis=1,
        ))]
        labels[order[: min(60, len(order))]] = codes[2]
    return labels, names


def region_mesh(
    vertices: np.ndarray, faces: np.ndarray, vertex_labels: np.ndarray, code: int,
) -> trimesh.Trimesh | None:
    face_labels = vertex_labels[faces]
    selected = np.sum(face_labels == code, axis=1) >= 2
    if not np.any(selected):
        return None
    return mesh(vertices, faces[selected])


def hand_depth_consistency(
    cfg: dict[str, Any], data: dict[str, Any], output: Path,
    depths: dict[int, np.ndarray], projector: TacoOfficialProjector,
    object_report: dict[str, Any], rgbs: dict[int, np.ndarray],
) -> dict[str, Any]:
    if object_report["classification"] != "DEPTH_CAMERA_CONTROL_PASS":
        return {
            "schema": "taco_hand_depth_consistency_v2",
            "executed": False,
            "reason": "rigid-object depth-camera control did not pass",
            "classification": "PROHIBITED_BY_CAMERA_DEPTH_GATE",
        }
    vertex_labels, names = hand_vertex_regions(data)
    erosion = int(cfg["analysis"]["interior_erosion_px"])
    tolerance = float(cfg["analysis"]["front_surface_tolerance_m"])
    records = []
    for frame in cfg["sequence"]["focus_frames"]:
        projector.set_camera(data["intrinsic"], data["extrinsics"][frame])
        values = frame_meshes(data, frame)
        full = projector.render_depth([values[name] for name in CLASS_ORDER])
        left = projector.render_depth([values["left_hand"]])
        left_front = (left > 0) & (full > 0) & (np.abs(left - full) <= tolerance)
        frame_errors = np.full(left.shape, np.nan, np.float32)
        for code, name in names.items():
            region = region_mesh(
                data["hands"]["left"]["vertices"][frame],
                data["hands"]["left"]["faces"], vertex_labels, code,
            )
            if region is None:
                continue
            rendered = projector.render_depth([region])
            selector = eroded((rendered > 0) & left_front, erosion)
            stats, valid, error = residual_stats(depths[frame], rendered, selector)
            frame_errors[valid] = error[valid]
            records.append({"frame": frame, "region": name, **stats})
        directory = output / "hand_depth" / f"frame_{frame:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "rgb.png"), cv2.cvtColor(rgbs[frame], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(directory / "left_hand_theoretical_depth.png"),
                    depth_color(left, lo=0.3, hi=1.2))
        cv2.imwrite(str(directory / "observed_depth.png"),
                    depth_color(depths[frame], lo=0.3, hi=1.2))
        cv2.imwrite(str(directory / "signed_error.png"),
                    error_color(frame_errors, 0.05, absolute=False))
        cv2.imwrite(str(directory / "absolute_error.png"),
                    error_color(frame_errors, 0.05, absolute=True))

    control_by_frame = {}
    for frame in cfg["sequence"]["focus_frames"]:
        rows = [row for row in object_report["records"]
                if row["timing_offset"] == 0 and row["observed_depth_frame"] == frame
                and row["absolute_error_mm"] is not None]
        control_by_frame[frame] = max(
            (row["absolute_error_mm"]["p90"] for row in rows), default=float("inf")
        )
    target_names = {
        "ring_proximal", "ring_middle_distal", "ring_tip",
        "pinky_proximal", "pinky_middle_distal", "pinky_tip",
    }
    comparisons = []
    for row in records:
        if row["region"] not in target_names or row["absolute_error_mm"] is None:
            continue
        threshold = control_by_frame[row["frame"]]
        comparisons.append({
            "frame": row["frame"], "region": row["region"],
            "hand_median_absolute_error_mm": row["absolute_error_mm"]["median"],
            "same_frame_rigid_object_p90_absolute_error_mm": threshold,
            "exceeds_rigid_object_p90": row["absolute_error_mm"]["median"] > threshold,
            "signed_error_median_mm": row["signed_error_median_mm"],
        })
    by_region = {}
    for name in sorted(target_names):
        rows = [row for row in comparisons if row["region"] == name]
        signs = [np.sign(row["signed_error_median_mm"]) for row in rows
                 if row["signed_error_median_mm"] not in (None, 0)]
        by_region[name] = {
            "visible_frames": len(rows),
            "frames_exceeding_rigid_object_p90": sum(
                row["exceeds_rigid_object_p90"] for row in rows
            ),
            "consistent_signed_direction": bool(
                signs and abs(sum(signs)) == len(signs)
            ),
        }
    conflict_regions = [name for name, row in by_region.items()
                        if row["visible_frames"] >= 3
                        and row["frames_exceeding_rigid_object_p90"] >= 3
                        and row["consistent_signed_direction"]]
    consistent = bool(comparisons) and not conflict_regions
    return {
        "schema": "taco_hand_depth_consistency_v2",
        "executed": True,
        "comparison_basis": "hand regional median absolute error versus same-frame maximum rigid-object p90",
        "fixed_global_hand_error_threshold_used": False,
        "records": records,
        "relative_comparisons": comparisons,
        "region_summary": by_region,
        "conflict_regions": conflict_regions,
        "classification": (
            "HAND_DEPTH_CONSISTENT_WITH_RIGID_CONTROL" if consistent
            else "HAND_DEPTH_SYSTEMATIC_EXCESS_FOUND" if conflict_regions
            else "HAND_DEPTH_EVIDENCE_INSUFFICIENT"
        ),
    }


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def allocentric_secondary(
    cfg: dict[str, Any], data: dict[str, Any], output: Path,
) -> dict[str, Any]:
    calibration = json.loads(data["paths"]["allocentric_calibration"].read_text())
    videos = sorted(data["paths"]["allocentric_video_dir"].glob("*.mp4"))
    if set(path.stem for path in videos) != set(calibration):
        raise ValueError("allocentric video/calibration camera sets disagree")
    wanted = set(cfg["sequence"]["allocentric_frames"])
    rgb_by_camera = {path.stem: video_frames(path, wanted) for path in videos}
    archive = data["paths"]["segmentation_archive"]
    prefix = cfg["inputs"]["segmentation_prefix"]
    masks_by_camera = {}
    member_records = []
    with ZipFile(archive) as zip_file:
        for camera in sorted(calibration):
            member = f"{prefix}/{camera}_masks.npy"
            import io
            payload = zip_file.read(member)
            masks_by_camera[camera] = np.load(io.BytesIO(payload), allow_pickle=False)
            member_records.append(member_artifact(archive, member))

    samples = []
    projector = None
    for camera in sorted(calibration):
        row = calibration[camera]
        intrinsic = np.asarray(row["K"], dtype=np.float64).reshape(3, 3)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = np.asarray(row["R"]).reshape(3, 3)
        extrinsic[:3, 3] = np.asarray(row["T"])
        size = tuple(int(x) for x in row["imgSize"])
        if projector is None:
            projector = TacoOfficialProjector(
                dataset_utils=data["official_utils"], image_size=size,
                intrinsic=intrinsic, extrinsic=extrinsic,
                device=cfg["official"]["device"],
            )
        for frame in sorted(wanted):
            projector.set_camera(intrinsic, extrinsic)
            values = frame_meshes(data, frame)
            rendered = {
                name: projector.render_mask([values[name]]) for name in CLASS_ORDER
            }
            seg_index = round(frame * cfg["analysis"]["allocentric_mask_rate_hz"]
                              / cfg["sequence"]["fps"])
            published = masks_by_camera[camera][seg_index]
            if published.shape != rendered[CLASS_ORDER[0]].shape:
                published = cv2.resize(
                    published, size, interpolation=cv2.INTER_NEAREST,
                )
            samples.append({
                "camera": camera, "frame": frame, "segmentation_index": seg_index,
                "rgb": rgb_by_camera[camera][frame],
                "published": published, "rendered": rendered,
            })

    score_matrix = np.zeros((4, 4), dtype=np.float64)
    labels = (1, 2, 3, 4)
    for class_index, name in enumerate(CLASS_ORDER):
        for label_index, label in enumerate(labels):
            score_matrix[class_index, label_index] = np.mean([
                intersection_over_union(sample["rendered"][name],
                                        sample["published"] == label)
                for sample in samples
            ])
    assignments = []
    for permutation in itertools.permutations(labels):
        score = float(sum(score_matrix[i, label - 1]
                          for i, label in enumerate(permutation)))
        assignments.append((score, permutation))
    assignments.sort(reverse=True)
    mapping = {name: int(label) for name, label in zip(CLASS_ORDER, assignments[0][1])}

    details = []
    page_by_frame: dict[int, list[np.ndarray]] = {frame: [] for frame in wanted}
    for sample in samples:
        rgb = cv2.cvtColor(sample["rgb"], cv2.COLOR_RGB2BGR)
        display = rgb.copy()
        class_rows = []
        for name in CLASS_ORDER:
            published = sample["published"] == mapping[name]
            rendered = sample["rendered"][name]
            contours_pub, _ = cv2.findContours(
                published.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            contours_render, _ = cv2.findContours(
                rendered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            color = CLASS_COLORS[name]
            cv2.drawContours(display, contours_pub, -1, color, 2)
            cv2.drawContours(display, contours_render, -1, (255, 255, 255), 1)
            class_rows.append({
                "class": name, "published_label": mapping[name],
                "iou": intersection_over_union(published, rendered),
            })
        cv2.putText(display, f"cam {sample['camera']} frame {sample['frame']}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(display, "color=published; white=official projection",
                    (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        page_by_frame[sample["frame"]].append(display)
        details.append({
            "camera": sample["camera"], "frame": sample["frame"],
            "segmentation_index": sample["segmentation_index"],
            "classes": class_rows,
        })
    visuals = []
    for frame, images in sorted(page_by_frame.items()):
        rows = [np.concatenate(images[i:i + 3], axis=1) for i in range(0, 12, 3)]
        page = np.concatenate(rows, axis=0)
        path = output / "allocentric_masks" / f"frame_{frame:03d}_12views.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), page)
        visuals.append(str(path))
    per_class = {
        name: {
            "mean_iou": float(np.mean([
                row["iou"] for detail in details for row in detail["classes"]
                if row["class"] == name
            ])),
            "median_iou": float(np.median([
                row["iou"] for detail in details for row in detail["classes"]
                if row["class"] == name
            ])),
        }
        for name in CLASS_ORDER
    }
    return {
        "schema": "taco_allocentric_mask_secondary_check_v2",
        "role": "secondary_conflict_detection_only",
        "not_independent_ground_truth": True,
        "mask_generation_note": "3D mesh silhouette initialization followed by SAM/Track-Anything refinement",
        "label_mapping_method": "exhaustive 4! assignment maximizing mean IoU over 12 cameras and frames 0/15/20",
        "label_mapping": mapping,
        "mapping_score": assignments[0][0],
        "runner_up_mapping_score": assignments[1][0],
        "mapping_margin": assignments[0][0] - assignments[1][0],
        "iou_matrix_rows_classes_columns_labels_1_to_4": score_matrix.tolist(),
        "per_class": per_class,
        "details": details,
        "visuals": visuals,
        "mask_members": member_records,
    }


def input_manifest(cfg: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    paths = data["paths"]
    files = [
        paths["rgb_video"], paths["depth_video"], paths["egocentric_intrinsic"],
        paths["egocentric_extrinsic"], paths["allocentric_calibration"],
    ]
    files.extend(sorted(paths["allocentric_video_dir"].glob("*.mp4")))
    for side in ("left", "right"):
        files.extend([
            paths["hand_pose_dir"] / f"{side}_hand.pkl",
            paths["hand_pose_dir"] / f"{side}_hand_shape.pkl",
            paths["mano_model_dir"] / f"MANO_{side.upper()}.pkl",
        ])
    for role in ("tool", "target"):
        files.extend([data["objects"][role]["model_path"], data["objects"][role]["pose_path"]])
    return {
        "schema": "taco_source_alignment_input_manifest_v2",
        "sequence": f"{cfg['sequence']['triplet']}/{cfg['sequence']['name']}",
        "endpoint_zero_mapping": "endpoint 0 = source frame 0 = 0 seconds",
        "video_stream_metadata": data["video_metadata"],
        "timing_note": (
            "RGB and depth both decode to 198 frames, but their container frame-rate/duration "
            "metadata disagree. The contract keeps identity-by-index and reports +/-1 only as "
            "a diagnostic; it does not silently resample either stream."
        ),
        "files": [artifact(path) for path in files],
        "segmentation_archive": artifact(paths["segmentation_archive"]),
        "allocentric_acquisition_manifest": artifact(
            paths["allocentric_video_dir"].parents[2] / "acquisition_manifest.json"
        ),
    }


def source_pins(cfg: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    checkout = Path(cfg["official"]["checkout"])
    official_files = [
        checkout / "dataset_utils/project_pose_to_egocentric_view.py",
        checkout / "dataset_utils/pyt3d_wrapper.py",
        checkout / "dataset_utils/hand_pose_loader.py",
        checkout / "dataset_utils/video_utils.py",
    ]
    return {
        "schema": "taco_source_pins_v2",
        "repository": cfg["official"]["repository"],
        "commit": cfg["official"]["commit"],
        "checkout": str(checkout),
        "entrypoint": str(checkout / cfg["official"]["entrypoint"]),
        "tracked_source_clean": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=checkout, text=True,
        ).strip() == "",
        "files": [artifact(path) for path in official_files],
        "local_wrapper": artifact(
            ROOT / "src/egoengine_repro/evaluation/taco_official_projection.py"
        ),
        "wrapper_contains_projection_mathematics": False,
    }


def projection_cleanup(
    cfg: dict[str, Any], pins: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "taco_projection_code_cleanup_v2",
        "removed_or_disabled": [
            "egoengine_repro.retarget.contract_closure.project_world_points",
            "egoengine_repro.retarget.paper_audit.project_world",
            "scripts/audit_taco_paper_inputs.py local released/inverse projection diagnostic",
            "scripts/audit_taco_pour_retarget_contract_closure_v2.py local RGB overlay projection",
            "scripts/audit_taco_pour_depth_registration.py local camera/raycast projection",
            "scripts/audit_taco_pour_table_calibration.py local camera projection",
            "scripts/audit_taco_pour_raw_depth_table.py local inverse camera projection",
        ],
        "removed_files": [
            "scripts/audit_taco_pour_depth_registration.py",
            "scripts/audit_taco_pour_table_calibration.py",
            "scripts/audit_taco_pour_raw_depth_table.py",
        ],
        "remaining_taco_projection_entrypoint": (
            "egoengine_repro.evaluation.taco_official_projection.TacoOfficialProjector"
        ),
        "official_code_entrypoint": (
            "dataset_utils/project_pose_to_egocentric_view.py -> Pyt3DWrapper"
        ),
        "official_repository": cfg["official"]["repository"],
        "official_commit": cfg["official"]["commit"],
        "official_files": pins["files"],
        "local_wrapper": pins["local_wrapper"],
        "wrapper_contains_projection_mathematics": False,
        "generic_non_taco_projection_code_untouched": True,
    }


def decision(
    object_report: dict[str, Any], hand_report: dict[str, Any],
    allocentric_report: dict[str, Any],
) -> dict[str, Any]:
    if object_report["classification"] != "DEPTH_CAMERA_CONTROL_PASS":
        classification = "CAMERA_DEPTH_REGISTRATION_UNRESOLVED"
        rationale = (
            "The rigid-object control is dense on the tool but the minimum target valid-depth "
            f"fraction is only {object_report['per_object_aggregate']['target']['minimum_valid_pixel_fraction']:.3f}; "
            "RGB/depth container timing metadata also disagree. Pixel-registered egocentric "
            "depth is not established, so the depth channel is prohibited from judging the "
            "released hand pose."
        )
    elif hand_report["classification"] == "HAND_DEPTH_SYSTEMATIC_EXCESS_FOUND":
        classification = "SOURCE_3D_CONFLICT"
        rationale = (
            "Rigid-object depth control passes, while multiple ring/pinky regions show "
            "same-direction errors beyond the same-frame rigid-object p90 control."
        )
    elif hand_report["classification"] == "HAND_DEPTH_CONSISTENT_WITH_RIGID_CONTROL":
        classification = "SOURCE_RGBD_CONSISTENT"
        rationale = (
            "Official egocentric projection and rigid-object depth control pass; visible "
            "left ring/pinky errors remain within the measured rigid-object control scale."
        )
    else:
        classification = "SOURCE_ALIGNMENT_UNRESOLVED"
        rationale = "Visible hand/depth support is insufficient for a relative control decision."
    return {
        "schema": "taco_source_alignment_decision_v2",
        "classification": classification,
        "rationale": rationale,
        "object_depth_control": object_report["classification"],
        "hand_depth_consistency": hand_report["classification"],
        "allocentric_masks_role": allocentric_report["role"],
        "manual_annotation_required": classification == "SOURCE_ALIGNMENT_UNRESOLVED",
        "runtime_counts": {
            "physics_steps": 0, "control_steps": 0,
            "new_robot_retarget_candidates": 0, "planner_calls": 0,
            "reinforcement_learning_steps": 0, "reference_promotions": 0,
            "chunk_commits": 0,
        },
    }


def write_summary(
    output: Path, decision_report: dict[str, Any], object_report: dict[str, Any],
    hand_report: dict[str, Any], allocentric_report: dict[str, Any],
    projection_report: dict[str, Any],
) -> None:
    lines = [
        "# TACO source alignment — official projection v2",
        "",
        f"**Decision:** `{decision_report['classification']}`",
        "",
        decision_report["rationale"],
        "",
        "## Fixed evidence chain",
        "",
        "- Projection is exclusively the pinned official TACO PyTorch3D implementation.",
        "- Endpoint 0 is source frame 0 at 0 seconds; ±1 is diagnostic only.",
        f"- Rigid-object depth control: `{object_report['classification']}`.",
        f"- Hand/depth comparison: `{hand_report['classification']}`.",
        "- The 12-view automatic masks are secondary conflict checks, not independent truth.",
        "- Physics, control, planning, retarget candidate generation, RL, promotion and chunk commit: all zero.",
        "",
        "## Key numbers",
        "",
        f"- Object median-of-medians absolute depth error: {object_report['aggregate']['median_of_median_absolute_error_mm']:.3f} mm.",
        f"- Tool valid-depth fraction (minimum across frames): {object_report['per_object_aggregate']['tool']['minimum_valid_pixel_fraction']:.3f}.",
        f"- Target valid-depth fraction (minimum across frames): {object_report['per_object_aggregate']['target']['minimum_valid_pixel_fraction']:.3f}.",
        f"- Timing diagnostic zero-offset best fraction: {object_report['aggregate']['zero_offset_best_fraction']:.3f}; offsets were not selected.",
        f"- RGB/depth container-rate mismatch observed: `{object_report['container_rate_mismatch_observed']}` "
        f"(RGB {object_report['video_stream_metadata']['rgb']['avg_frame_rate']}, "
        f"{object_report['video_stream_metadata']['rgb']['duration']} s; depth "
        f"{object_report['video_stream_metadata']['depth']['avg_frame_rate']}, "
        f"{object_report['video_stream_metadata']['depth']['duration']} s). Both decode to 198 "
        "frames and no resampling was applied.",
        f"- Allocentric label mapping: `{allocentric_report['label_mapping']}`; margin {allocentric_report['mapping_margin']:.6f}.",
        "",
        "## Interpretation boundary",
        "",
        "Automatic masks were initialized from the released 3D meshes and refined by image models. "
        "Agreement cannot independently prove the 3D pose correct; systematic multi-camera disagreement can only add conflict evidence.",
    ]
    if hand_report.get("conflict_regions"):
        lines.extend(["", "Conflict regions: " + ", ".join(hand_report["conflict_regions"]) + "."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    visual_lines = ["# Visual index", "", "## Official egocentric overlays", ""]
    for row in projection_report["frames"]:
        visual_lines.append(
            f"- frame {row['frame']}: [raw]({Path(row['raw_rgb']).relative_to(output)}) · "
            f"[hands]({Path(row['hands_overlay']).relative_to(output)}) · "
            f"[objects]({Path(row['objects_overlay']).relative_to(output)}) · "
            f"[scene]({Path(row['scene_overlay']).relative_to(output)})"
        )
    visual_lines.extend(["", "## Rigid-object depth controls", ""])
    for frame in sorted({row["observed_depth_frame"] for row in object_report["records"]}):
        for role in ("tool", "target"):
            base = Path("object_depth") / f"frame_{frame:03d}" / role
            visual_lines.append(
                f"- frame {frame} {role}: [rendered]({base/'rendered_depth.png'}) · "
                f"[signed]({base/'signed_error.png'}) · [absolute]({base/'absolute_error.png'})"
            )
    if hand_report.get("executed"):
        visual_lines.extend(["", "## Left-hand regional depth", ""])
        for frame in sorted({row["frame"] for row in hand_report["records"]}):
            base = Path("hand_depth") / f"frame_{frame:03d}"
            visual_lines.append(
                f"- frame {frame}: [RGB]({base/'rgb.png'}) · [theory]({base/'left_hand_theoretical_depth.png'}) · "
                f"[signed]({base/'signed_error.png'}) · [absolute]({base/'absolute_error.png'})"
            )
    visual_lines.extend(["", "## 12-view automatic-mask secondary checks", ""])
    for value in allocentric_report["visuals"]:
        path = Path(value)
        visual_lines.append(f"- [{path.stem}]({path.relative_to(output)})")
    (output / "VISUAL_INDEX.md").write_text("\n".join(visual_lines) + "\n", encoding="utf-8")


def hash_tree(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "server_artifacts.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text("\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    data = load_inputs(cfg)
    focus = set(cfg["sequence"]["focus_frames"])
    depth_indices = set(focus)
    rgbs = video_frames(data["paths"]["rgb_video"], focus)
    depths = depth_frames(
        data["paths"]["depth_video"], depth_indices,
        count=cfg["sequence"]["frame_count"], height=1080, width=1920,
        scale=cfg["analysis"]["depth_scale"],
    )
    projector = TacoOfficialProjector(
        dataset_utils=data["official_utils"], image_size=(1920, 1080),
        intrinsic=data["intrinsic"], extrinsic=data["extrinsics"][0],
        device=cfg["official"]["device"],
    )
    pins = source_pins(cfg, data)
    inputs = input_manifest(cfg, data)
    projection = official_projection_baseline(cfg, data, output, rgbs, projector)
    objects = object_depth_control(cfg, data, output, depths, projector)
    hands = hand_depth_consistency(cfg, data, output, depths, projector, objects, rgbs)
    allocentric = allocentric_secondary(cfg, data, output)
    final = decision(objects, hands, allocentric)
    reports = {
        "source_pins.json": pins,
        "input_manifest.json": inputs,
        "projection_code_cleanup.json": projection_cleanup(cfg, pins),
        "official_egocentric_projection_manifest.json": projection,
        "object_depth_control.json": objects,
        "hand_depth_consistency.json": hands,
        "allocentric_mask_secondary_check.json": allocentric,
        "source_alignment_decision.json": final,
    }
    for name, value in reports.items():
        write_json(output / name, value)
    write_summary(output, final, objects, hands, allocentric, projection)
    hash_tree(output)
    print(json.dumps({
        "output": str(output), "classification": final["classification"],
        "object_control": objects["classification"],
        "hand": hands["classification"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
