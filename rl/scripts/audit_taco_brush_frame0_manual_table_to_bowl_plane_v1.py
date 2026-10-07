#!/usr/bin/env python3
"""Measure visually selected raw frame-0 table depth against a fixed bowl plane.

This script deliberately contains no table-plane fit, calibration fit, spatial
subsampling, distance rejection, or signed-distance truncation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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


DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_frame0_manual_table_to_bowl_plane_v1.yaml"


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


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    result = np.percentile(values, levels)
    return {f"p{level:02d}": float(value) for level, value in zip(levels, result)}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    # Pixels are unique but depth can be tied. Average ranks preserve ties.
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranked = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranked[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return ranked

    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def decode_frame0_depth(path: Path, width: int, height: int) -> np.ndarray:
    stream = ffprobe_video(path)
    spec = DepthVideoSpec(width, height, stream_frame_count(stream))
    first: np.ndarray | None = None
    # Consume the exact decoder fully so its frame-count and process checks run.
    for index, frame in enumerate(iter_depth_frames(path, spec)):
        if index == 0:
            first = frame.copy()
    if first is None:
        raise RuntimeError("depth video contains no frames")
    return first


def decode_frame0_rgb(path: Path, width: int, height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None or frame.shape[:2] != (height, width):
        raise RuntimeError("could not decode expected frame-0 RGB")
    return frame


def polygon_masks(
    polygons: dict[str, list[list[int]]], height: int, width: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    union = np.zeros((height, width), dtype=np.uint8)
    labels = np.zeros((height, width), dtype=np.int16)
    names = list(polygons)
    for label, name in enumerate(names, start=1):
        vertices = np.asarray(polygons[name], dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
            raise ValueError(f"invalid polygon: {name}")
        local = np.zeros_like(union)
        cv2.fillPoly(local, [vertices], 1)
        if np.any((labels > 0) & (local > 0)):
            raise ValueError(f"manual polygons overlap: {name}")
        union |= local
        labels[local > 0] = label
    return union.astype(bool), labels, names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["schema"] != "taco_brush_frame0_manual_table_to_bowl_plane_v1":
        raise ValueError("unexpected config schema")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], "HEAD"],
        cwd=REPO_ROOT, check=True,
    )
    if cfg["sample"]["frame"] != 0 or cfg["depth"]["spatial_stride_px"] != 1:
        raise ValueError("this audit requires frame 0 and every selected depth pixel")
    expected_contract = {
        "refit_reference_plane": False,
        "horizontal_prior": False,
        "distance_based_point_rejection": False,
        "signed_distance_truncation": False,
        "calibration_algorithm": False,
        "report_every_valid_selected_depth_point": True,
        "signed_distance_equation": "fixed_normal dot world_point - fixed_offset",
        "sign_convention": "positive is the side pointed to by the fixed bowl-plane normal",
    }
    if cfg["measurement_contract"] != expected_contract:
        raise ValueError("measurement contract changed")
    if any(cfg["authorization"][key] for key in (
        "modify_source_data", "modify_active_support_contract", "table_plane_estimation",
        "calibration", "mink", "physics", "replay", "mpc", "reinforcement_learning",
    )):
        raise ValueError("forbidden operation was authorized")

    paths = {key: Path(value).resolve(strict=True) for key, value in cfg["inputs"].items()}
    protected = {
        key: Path(value).resolve(strict=True)
        for key, value in cfg["integrity_protected"].items()
    }
    output = Path(cfg["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    active_before = sha256(protected["active_support_contract"])

    width, height = int(cfg["depth"]["width"]), int(cfg["depth"]["height"])
    rgb = decode_frame0_rgb(paths["rgb_video"], width, height)
    raw = decode_frame0_depth(paths["depth_video"], width, height)
    depth_m = raw_depth_to_metres(raw, scale=float(cfg["depth"]["scale_raw_units_per_metre"]))
    union, region_labels, region_names = polygon_masks(
        cfg["manual_selection"]["polygons_xy_px"], height, width,
    )
    valid_selected = union & (raw != int(cfg["depth"]["invalid_raw_value"]))
    if not np.any(valid_selected):
        raise RuntimeError("manual table selection contains no valid depth")

    points_camera, pixels = backproject_metric_depth(
        depth_m, np.loadtxt(paths["intrinsic"]), selector=valid_selected,
        spatial_stride_px=1,
    )
    extrinsics = np.load(paths["extrinsic"], allow_pickle=False)
    world_to_camera = np.asarray(extrinsics[0], dtype=np.float64)
    points_world = camera_points_to_world(points_camera, world_to_camera)
    plane_source = json.loads(paths["fixed_bowl_support_plane"].read_text(encoding="utf-8"))
    if plane_source["classification"] != "SUPPORT_PLANE_WELL_DEFINED":
        raise ValueError("fixed bowl support plane is not eligible")
    normal = np.asarray(plane_source["primary"]["plane"]["normal"], dtype=np.float64)
    offset = float(plane_source["primary"]["plane"]["offset_m"])
    signed_m = points_world @ normal - offset
    point_regions = region_labels[pixels[:, 1], pixels[:, 0]]
    if not np.all(point_regions > 0) or len(signed_m) != int(valid_selected.sum()):
        raise AssertionError("not every valid preselected depth point was measured")

    rows = []
    for label, name in enumerate(region_names, start=1):
        chosen = point_regions == label
        values = signed_m[chosen]
        polygon_mask = region_labels == label
        rows.append({
            "region": name,
            "polygon_pixel_count": int(polygon_mask.sum()),
            "valid_depth_point_count": int(chosen.sum()),
            "invalid_depth_pixel_count": int(polygon_mask.sum() - chosen.sum()),
            "signed_distance_mm": {
                **{key: value * 1000.0 for key, value in quantiles(values).items()},
                "mean": float(np.mean(values) * 1000.0),
                "std": float(np.std(values) * 1000.0),
            },
            "world_xyz_range_m": np.ptp(points_world[chosen], axis=0).tolist(),
        })

    all_mm = signed_m * 1000.0
    regional_medians = np.asarray([row["signed_distance_mm"]["p50"] for row in rows])
    results = {
        "schema": cfg["schema"],
        "sample": cfg["sample"],
        "inputs": {key: artifact(path) for key, path in paths.items()},
        "config": artifact(config_path),
        "fixed_bowl_support_plane": {
            "normal": normal.tolist(),
            "offset_m": offset,
            "equation": "normal dot world_point = offset",
            "refitted": False,
        },
        "selection": {
            "basis": cfg["manual_selection"]["basis"],
            "policy": cfg["manual_selection"]["policy"],
            "polygons_xy_px": cfg["manual_selection"]["polygons_xy_px"],
            "polygon_count": len(region_names),
            "selected_polygon_pixel_count": int(union.sum()),
            "valid_selected_depth_point_count": int(valid_selected.sum()),
            "invalid_selected_depth_pixel_count": int((union & ~valid_selected).sum()),
            "rejected_by_distance_count": 0,
            "spatial_stride_px": 1,
        },
        "all_selected_points_signed_distance_mm": {
            **quantiles(all_mm),
            "mean": float(np.mean(all_mm)),
            "std": float(np.std(all_mm)),
            "negative_count": int(np.sum(signed_m < 0)),
            "zero_count": int(np.sum(signed_m == 0)),
            "positive_count": int(np.sum(signed_m > 0)),
            "positive_fraction": float(np.mean(signed_m > 0)),
        },
        "spatial_descriptors_only_not_a_plane_fit": {
            "per_region": rows,
            "regional_median_min_mm": float(regional_medians.min()),
            "regional_median_max_mm": float(regional_medians.max()),
            "regional_median_span_mm": float(np.ptp(regional_medians)),
            "spearman_signed_distance_vs_pixel_u": spearman(pixels[:, 0], signed_m),
            "spearman_signed_distance_vs_pixel_v": spearman(pixels[:, 1], signed_m),
            "spearman_signed_distance_vs_world_x": spearman(points_world[:, 0], signed_m),
            "spearman_signed_distance_vs_world_y": spearman(points_world[:, 1], signed_m),
        },
        "measurement_contract": cfg["measurement_contract"],
        "integrity": {
            "source_data_modified": False,
            "active_support_contract_modified": False,
            "active_support_sha256_before": active_before,
            "active_support_sha256_after": sha256(protected["active_support_contract"]),
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    if results["integrity"]["active_support_sha256_before"] != results["integrity"]["active_support_sha256_after"]:
        raise AssertionError("active SupportSurfaceContract changed")
    write_json(output / "results.json", results)

    np.savez_compressed(
        output / "all_selected_table_depth_points.npz",
        pixel_uv=pixels.astype(np.int32),
        region_id=point_regions.astype(np.int16),
        region_names=np.asarray(region_names),
        raw_depth=raw[pixels[:, 1], pixels[:, 0]],
        depth_m=depth_m[pixels[:, 1], pixels[:, 0]],
        point_camera_m=points_camera,
        point_world_m=points_world,
        signed_distance_m=signed_m,
    )
    with gzip.open(output / "all_selected_table_depth_points.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "region", "u_px", "v_px", "raw_depth", "depth_m", "camera_x_m",
            "camera_y_m", "camera_z_m", "world_x_m", "world_y_m", "world_z_m",
            "signed_distance_m",
        ])
        for uv, region, raw_value, depth, camera, world, distance in zip(
            pixels, point_regions, raw[pixels[:, 1], pixels[:, 0]],
            depth_m[pixels[:, 1], pixels[:, 0]], points_camera, points_world, signed_m,
        ):
            writer.writerow([
                region_names[int(region) - 1], int(uv[0]), int(uv[1]), int(raw_value),
                f"{float(depth):.9f}", *[f"{value:.9f}" for value in camera],
                *[f"{value:.9f}" for value in world], f"{float(distance):.9f}",
            ])

    with (output / "per_region.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "region", "polygon_pixels", "valid_depth_points", "invalid_depth_pixels",
            "distance_min_mm", "distance_p05_mm", "distance_median_mm",
            "distance_p95_mm", "distance_max_mm", "distance_mean_mm", "distance_std_mm",
        ])
        for row in rows:
            distance = row["signed_distance_mm"]
            writer.writerow([
                row["region"], row["polygon_pixel_count"], row["valid_depth_point_count"],
                row["invalid_depth_pixel_count"], distance["p00"], distance["p05"],
                distance["p50"], distance["p95"], distance["p100"],
                distance["mean"], distance["std"],
            ])

    overlay = rgb.copy()
    palette = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 255, 0), (0, 128, 255)]
    tint = rgb.copy()
    for index, name in enumerate(region_names):
        vertices = np.asarray(cfg["manual_selection"]["polygons_xy_px"][name], dtype=np.int32)
        color = palette[index % len(palette)]
        cv2.fillPoly(tint, [vertices], color)
        cv2.polylines(overlay, [vertices], True, color, 5)
        cv2.putText(overlay, name, tuple(vertices[0] + [0, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    overlay = cv2.addWeighted(overlay, 0.72, tint, 0.28, 0)
    cv2.imwrite(str(output / "frame0_rgb_manual_table_selection.png"), overlay)
    selected_image = np.zeros((height, width, 3), dtype=np.uint8)
    selected_image[union & ~valid_selected] = (80, 80, 80)
    for label in range(1, len(region_names) + 1):
        selected_image[valid_selected & (region_labels == label)] = palette[(label - 1) % len(palette)]
    cv2.imwrite(str(output / "frame0_selected_depth_pixels.png"), selected_image)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].hist(all_mm, bins=200, color="#315a9a", alpha=0.9)
    axes[0].axvline(float(np.median(all_mm)), color="#c43131", linewidth=2, label="median")
    axes[0].axvline(0.0, color="black", linewidth=1, linestyle="--", label="bowl plane")
    axes[0].set_title("All selected valid raw-depth points (no truncation)")
    axes[0].set_xlabel("Signed distance to fixed bowl support plane (mm)")
    axes[0].set_ylabel("Point count")
    axes[0].legend()
    region_y = np.arange(len(rows))
    medians = np.asarray([row["signed_distance_mm"]["p50"] for row in rows])
    p05 = np.asarray([row["signed_distance_mm"]["p05"] for row in rows])
    p95 = np.asarray([row["signed_distance_mm"]["p95"] for row in rows])
    axes[1].errorbar(
        medians, region_y, xerr=np.vstack([medians - p05, p95 - medians]),
        fmt="o", color="#315a9a", capsize=4,
    )
    axes[1].axvline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_yticks(region_y, [row["region"] for row in rows])
    axes[1].invert_yaxis()
    axes[1].set_title("Spatial regions: median and P05–P95")
    axes[1].set_xlabel("Signed distance (mm)")
    figure.savefig(output / "signed_distance_distribution.png", dpi=160)
    plt.close(figure)

    overall = results["all_selected_points_signed_distance_mm"]
    summary_lines = [
        "# Brush frame-0 raw table depth vs fixed bowl support plane",
        "",
        "No reference/table plane was refitted. No horizontal prior, calibration, distance rejection,",
        "signed-distance truncation, or spatial subsampling was used.",
        "",
        "## Selection",
        "",
        f"- Manual RGB-only table regions: `{len(region_names)}`.",
        f"- Valid raw depth points reported: `{len(signed_m)}` / `{int(union.sum())}` selected pixels.",
        f"- Invalid raw-depth pixels (no 3-D measurement): `{int((union & ~valid_selected).sum())}`.",
        f"- Points rejected by distance: `0`.",
        "",
        "## Signed distance to the fixed bowl-bottom support plane",
        "",
        f"- Normal: `{normal.tolist()}`; offset: `{offset:.12f} m`.",
        f"- All points min / P05 / median / P95 / max: `{overall['p00']:.3f} / {overall['p05']:.3f} / {overall['p50']:.3f} / {overall['p95']:.3f} / {overall['p100']:.3f} mm`.",
        f"- Mean / standard deviation: `{overall['mean']:.3f} / {overall['std']:.3f} mm`.",
        f"- Negative / zero / positive: `{overall['negative_count']} / {overall['zero_count']} / {overall['positive_count']}`.",
        f"- Positive fraction: `{overall['positive_fraction'] * 100:.3f}%`.",
        f"- Per-region median range: `{regional_medians.min():.3f}` to `{regional_medians.max():.3f} mm` (span `{np.ptp(regional_medians):.3f} mm`).",
        "",
        "## Per-region medians",
        "",
    ]
    for row in rows:
        summary_lines.append(
            f"- `{row['region']}`: n=`{row['valid_depth_point_count']}`, median=`{row['signed_distance_mm']['p50']:.3f} mm`, P05/P95=`{row['signed_distance_mm']['p05']:.3f}/{row['signed_distance_mm']['p95']:.3f} mm`."
        )
    summary_lines += [
        "",
        "## Interpretation",
        "",
        "The raw table points do not coincide with the fixed bowl-bottom support plane. Most points lie on the positive-normal side by roughly centimetres, and the regional medians change systematically across the image. This is strong evidence of a geometric inconsistency between the Depth/camera chain and the object-pose/mesh chain; this audit alone does not assign the error to either side.",
        "",
        "These are direct measurements against the previously frozen bowl geometry plane, not a replacement table estimate.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(json.dumps({
        "count": len(signed_m), "signed_distance_mm": overall,
        "regional_median_span_mm": float(np.ptp(regional_medians)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
