#!/usr/bin/env python3
"""Fit unconstrained world planes to frozen RGB-selected raw table depth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.evaluation.taco_depth import (  # noqa: E402
    DepthVideoSpec,
    ffprobe_video,
    iter_depth_frames,
    raw_depth_to_metres,
    stream_frame_count,
)
from egoengine_repro.scene.support_surface_estimation import (  # noqa: E402
    backproject_metric_depth,
    camera_points_to_world,
)


DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_multiframe_static_table_world_consistency_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def polygon_union(polygons: dict[str, list[list[int]]], height: int, width: int) -> np.ndarray:
    union = np.zeros((height, width), dtype=np.uint8)
    occupied = np.zeros_like(union)
    for name, points in polygons.items():
        vertices = np.asarray(points, dtype=np.int32)
        local = np.zeros_like(union)
        cv2.fillPoly(local, [vertices], 1)
        if np.any((occupied > 0) & (local > 0)):
            raise ValueError(f"manual RGB polygons overlap: {name}")
        occupied |= local
        union |= local
    return union.astype(bool)


def fit_all_points_plane(points_world: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("plane fit requires at least three world points")
    centroid = np.mean(points, axis=0)
    _, singular, axes = np.linalg.svd(points - centroid, full_matrices=False)
    normal = axes[-1]
    if normal[2] < 0:
        normal = -normal
    offset = float(normal @ centroid)
    signed = points @ normal - offset
    absolute = np.abs(signed)
    in_plane = (points - centroid) @ axes[:2].T
    return {
        "normal": normal,
        "offset_m": offset,
        "centroid_world_m": centroid,
        "signed_residual_m": signed,
        "singular_values_m": singular,
        "in_plane_principal_range_m": np.sort(np.ptp(in_plane, axis=0))[::-1],
        "residual_median_m": float(np.median(absolute)),
        "residual_p95_m": float(np.percentile(absolute, 95)),
        "residual_max_m": float(np.max(absolute)),
    }


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def decode_rgb_frames(path: Path, selected: list[int]) -> dict[int, np.ndarray]:
    wanted = set(selected)
    result: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            result[index] = frame.copy()
        index += 1
    capture.release()
    if sorted(result) != selected:
        raise RuntimeError(f"missing RGB frames: {sorted(wanted - set(result))}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["schema"] != "taco_brush_multiframe_static_table_world_consistency_v1":
        raise ValueError("unexpected schema")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], "HEAD"],
        cwd=REPO_ROOT, check=True,
    )
    selected = [int(value) for value in cfg["sample"]["selected_frames"]]
    if selected != sorted(set(selected)) or selected != [int(v) for v in cfg["manual_rgb_selection"]["visually_approved_frames"]]:
        raise ValueError("selected frames are not exactly the visually approved frames")
    frozen_fit = {
        "points": "ALL_VALID_RAW_DEPTH_PIXELS_INSIDE_PRESELECTED_RGB_POLYGONS",
        "method": "UNCONSTRAINED_ORTHOGONAL_SVD_IN_WORLD_FRAME",
        "normal_sign_only": "POSITIVE_WORLD_Z",
        "robust_loss": False,
        "ransac": False,
        "horizontal_prior": False,
        "distance_based_point_rejection": False,
        "residual_truncation": False,
        "spatial_subsampling": False,
        "calibration_algorithm": False,
        "comparison_reference": "FRAME0_FITTED_PLANE_ONLY_NOT_A_STANDARD_ANSWER",
    }
    if cfg["fit_contract"] != frozen_fit:
        raise ValueError("fit contract changed")
    forbidden = (
        "modify_source_data", "modify_active_support_contract", "use_bowl_support_plane",
        "calibration", "mink", "physics", "replay", "mpc", "reinforcement_learning",
    )
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("forbidden operation was authorized")

    paths = {key: Path(value).resolve(strict=True) for key, value in cfg["inputs"].items()}
    output = Path(cfg["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    width, height = int(cfg["depth"]["width"]), int(cfg["depth"]["height"])
    selector = polygon_union(cfg["manual_rgb_selection"]["polygons_xy_px"], height, width)
    rgb = decode_rgb_frames(paths["rgb_video"], selected)
    intrinsic = np.loadtxt(paths["intrinsic"])
    extrinsic = np.load(paths["extrinsic"], allow_pickle=False)
    if extrinsic.shape != (cfg["sample"]["total_frames"], 4, 4):
        raise ValueError("official extrinsic frame count mismatch")

    stream = ffprobe_video(paths["depth_video"])
    spec = DepthVideoSpec(width, height, stream_frame_count(stream))
    if spec.frame_count != cfg["sample"]["total_frames"]:
        raise ValueError("depth frame count mismatch")
    selected_set = set(selected)
    frame_data: list[dict[str, Any]] = []
    point_arrays: dict[str, np.ndarray] = {}
    for frame, raw in enumerate(iter_depth_frames(paths["depth_video"], spec)):
        if frame not in selected_set:
            continue
        depth_m = raw_depth_to_metres(raw, scale=float(cfg["depth"]["scale_raw_units_per_metre"]))
        valid_selected = selector & (raw != int(cfg["depth"]["invalid_raw_value"]))
        camera, pixels = backproject_metric_depth(
            depth_m, intrinsic, selector=valid_selected, spatial_stride_px=1,
        )
        world = camera_points_to_world(camera, extrinsic[frame])
        if len(world) != int(valid_selected.sum()):
            raise AssertionError("not every valid preselected point reached the plane fit")
        fit = fit_all_points_plane(world)
        prefix = f"frame_{frame:03d}"
        point_arrays[f"{prefix}_pixel_uv"] = pixels.astype(np.int32)
        point_arrays[f"{prefix}_raw_depth"] = raw[pixels[:, 1], pixels[:, 0]]
        point_arrays[f"{prefix}_point_world_m"] = world
        point_arrays[f"{prefix}_signed_residual_m"] = fit["signed_residual_m"]
        frame_data.append({
            "frame": frame,
            "selected_polygon_pixel_count": int(selector.sum()),
            "valid_depth_point_count": int(len(world)),
            "invalid_depth_pixel_count": int(selector.sum() - len(world)),
            "points_rejected_by_distance": 0,
            "normal": fit["normal"].tolist(),
            "offset_m": fit["offset_m"],
            "centroid_world_m": fit["centroid_world_m"].tolist(),
            "singular_values_m": fit["singular_values_m"].tolist(),
            "in_plane_principal_range_m": fit["in_plane_principal_range_m"].tolist(),
            "point_to_own_plane_absolute_distance_m": {
                "median": fit["residual_median_m"],
                "p95": fit["residual_p95_m"],
                "maximum": fit["residual_max_m"],
            },
        })
    if [row["frame"] for row in frame_data] != selected:
        raise RuntimeError("selected depth frames were not all processed")

    reference = frame_data[0]
    reference_normal = np.asarray(reference["normal"])
    reference_offset = float(reference["offset_m"])
    reference_centroid = np.asarray(reference["centroid_world_m"])
    normals = np.asarray([row["normal"] for row in frame_data])
    offsets = np.asarray([row["offset_m"] for row in frame_data])
    for row in frame_data:
        normal = np.asarray(row["normal"])
        row["comparison_to_frame0"] = {
            "normal_angle_deg": angle_deg(normal, reference_normal),
            "offset_delta_mm": (float(row["offset_m"]) - reference_offset) * 1000.0,
            "signed_separation_at_frame0_centroid_mm": (
                float(normal @ reference_centroid - float(row["offset_m"])) * 1000.0
            ),
        }
    pairwise_angles = np.array([
        angle_deg(normals[i], normals[j])
        for i in range(len(normals)) for j in range(i + 1, len(normals))
    ])
    frame0_separations = np.asarray([
        row["comparison_to_frame0"]["signed_separation_at_frame0_centroid_mm"]
        for row in frame_data
    ])
    results = {
        "schema": cfg["schema"],
        "sample": cfg["sample"],
        "inputs": {key: artifact(path) for key, path in paths.items()},
        "config": artifact(config_path),
        "selection": {
            "basis": cfg["manual_rgb_selection"]["basis"],
            "policy": cfg["manual_rgb_selection"]["policy"],
            "polygons_xy_px": cfg["manual_rgb_selection"]["polygons_xy_px"],
            "selected_polygon_pixel_count_per_frame": int(selector.sum()),
            "selected_frames": selected,
        },
        "fit_contract": cfg["fit_contract"],
        "per_frame": frame_data,
        "world_plane_consistency": {
            "comparison_reference": "frame 0 fitted plane; descriptive only",
            "offset_min_m": float(offsets.min()),
            "offset_max_m": float(offsets.max()),
            "offset_span_mm": float(np.ptp(offsets) * 1000.0),
            "maximum_pairwise_normal_angle_deg": float(pairwise_angles.max()),
            "median_pairwise_normal_angle_deg": float(np.median(pairwise_angles)),
            "frame0_centroid_separation_min_mm": float(frame0_separations.min()),
            "frame0_centroid_separation_max_mm": float(frame0_separations.max()),
            "frame0_centroid_separation_span_mm": float(np.ptp(frame0_separations)),
        },
        "integrity": {
            "source_data_modified": False,
            "points_rejected_by_distance_total": 0,
            "bowl_support_plane_used": False,
            "calibration_run": False,
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    write_json(output / "results.json", results)
    np.savez_compressed(output / "all_selected_frame_points.npz", **point_arrays)

    with (output / "per_frame_planes.csv").open("w", newline="", encoding="utf-8") as stream_out:
        writer = csv.writer(stream_out)
        writer.writerow([
            "frame", "valid_points", "normal_x", "normal_y", "normal_z", "offset_m",
            "residual_median_mm", "residual_p95_mm", "residual_max_mm",
            "angle_to_frame0_deg", "offset_delta_to_frame0_mm",
            "separation_at_frame0_centroid_mm",
        ])
        for row in frame_data:
            residual = row["point_to_own_plane_absolute_distance_m"]
            compare = row["comparison_to_frame0"]
            writer.writerow([
                row["frame"], row["valid_depth_point_count"], *row["normal"], row["offset_m"],
                residual["median"] * 1000.0, residual["p95"] * 1000.0,
                residual["maximum"] * 1000.0, compare["normal_angle_deg"],
                compare["offset_delta_mm"], compare["signed_separation_at_frame0_centroid_mm"],
            ])

    palette = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    thumbnails = []
    for frame in selected:
        image = rgb[frame].copy()
        tint = image.copy()
        for index, (name, points) in enumerate(cfg["manual_rgb_selection"]["polygons_xy_px"].items()):
            vertices = np.asarray(points, dtype=np.int32)
            color = palette[index % len(palette)]
            cv2.fillPoly(tint, [vertices], color)
            cv2.polylines(image, [vertices], True, color, 5)
            cv2.putText(image, name, tuple(vertices[0] + [0, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        image = cv2.addWeighted(image, 0.72, tint, 0.28, 0)
        image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (175, 40), (0, 0, 0), -1)
        cv2.putText(image, f"frame {frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        thumbnails.append(image)
    montage = np.vstack([np.hstack(thumbnails[i:i + 3]) for i in range(0, len(thumbnails), 3)])
    cv2.imwrite(str(output / "rgb_selection_montage.png"), montage)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    frames = np.asarray(selected)
    axes[0].plot(frames, [row["comparison_to_frame0"]["normal_angle_deg"] for row in frame_data], "o-", label="normal angle to frame 0")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("World-plane normal drift")
    axes[0].grid(alpha=0.3)
    axes[1].plot(frames, frame0_separations, "o-", label="separation at frame-0 centroid")
    axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Signed separation (mm)")
    axes[1].set_title("World-plane positional drift")
    axes[1].grid(alpha=0.3)
    figure.savefig(output / "world_plane_consistency.png", dpi=160)
    plt.close(figure)

    consistency = results["world_plane_consistency"]
    lines = [
        "# Brush multiframe raw-Depth table planes in world coordinates",
        "",
        "Twelve frames spanning the full video were fixed before reading their plane results. Each frame uses all valid raw-depth pixels in the RGB-approved polygons. Fits are unconstrained orthogonal SVD planes in world coordinates; there is no robust loss, RANSAC, horizontal prior, distance rejection, truncation, subsampling, bowl plane, or calibration.",
        "",
        "## Cross-frame consistency",
        "",
        f"- Plane offset span: `{consistency['offset_span_mm']:.3f} mm`.",
        f"- Pairwise normal angle median / max: `{consistency['median_pairwise_normal_angle_deg']:.3f} / {consistency['maximum_pairwise_normal_angle_deg']:.3f} deg`.",
        f"- Signed plane separation at the frame-0 plane centroid min / max / span: `{consistency['frame0_centroid_separation_min_mm']:.3f} / {consistency['frame0_centroid_separation_max_mm']:.3f} / {consistency['frame0_centroid_separation_span_mm']:.3f} mm`.",
        "",
        "## Per-frame planes",
        "",
        "| frame | points | normal | offset (m) | own residual median/P95/max (mm) | angle to f0 (deg) | separation at f0 centroid (mm) |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in frame_data:
        residual = row["point_to_own_plane_absolute_distance_m"]
        compare = row["comparison_to_frame0"]
        lines.append(
            f"| {row['frame']} | {row['valid_depth_point_count']} | `[{row['normal'][0]:.6f}, {row['normal'][1]:.6f}, {row['normal'][2]:.6f}]` | {row['offset_m']:.6f} | {residual['median']*1000:.3f}/{residual['p95']*1000:.3f}/{residual['maximum']*1000:.3f} | {compare['normal_angle_deg']:.3f} | {compare['signed_separation_at_frame0_centroid_mm']:.3f} |"
        )
    lines += [
        "",
        "Frame 0 is only the comparison reference; it is not treated as a standard-answer plane.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(consistency, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
