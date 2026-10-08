#!/usr/bin/env python3
"""Compare table planes implied by the fixed bowl-bottom contact ring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_bowl_contact_ring_multiframe_plane_consistency_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def fit_plane(points: np.ndarray) -> dict[str, Any]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError("plane fit requires finite Nx3 points")
    if not np.isfinite(values).all():
        raise ValueError("plane points are not finite")
    centroid = values.mean(axis=0)
    _, singular, axes = np.linalg.svd(values - centroid, full_matrices=False)
    normal = axes[-1]
    if normal[2] < 0:
        normal = -normal
    offset = float(normal @ centroid)
    signed = values @ normal - offset
    absolute = np.abs(signed)
    in_plane = (values - centroid) @ axes[:2].T
    radial = np.linalg.norm(in_plane, axis=1)
    return {
        "normal": normal,
        "offset_m": offset,
        "centroid_world_m": centroid,
        "signed_distance_m": signed,
        "singular_values_m": singular,
        "in_plane_range_m": np.sort(np.ptp(in_plane, axis=0))[::-1],
        "radial_p05_m": float(np.percentile(radial, 5)),
        "radial_p95_m": float(np.percentile(radial, 95)),
        "absolute_distance_m": {
            "median": float(np.median(absolute)),
            "p95": float(np.percentile(absolute, 95)),
            "maximum": float(absolute.max()),
        },
    }


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(first @ second), -1.0, 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["schema"] != "taco_brush_bowl_contact_ring_multiframe_plane_consistency_v1":
        raise ValueError("unexpected schema")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], "HEAD"],
        cwd=REPO_ROOT, check=True,
    )
    selected = [int(frame) for frame in cfg["sample"]["selected_frames"]]
    if selected != [0, 2, 4, 6, 8, 9, 152, 163, 174, 185, 196, 208]:
        raise ValueError("selected frames changed from the frozen balanced set")
    if selected != sorted(set(selected)):
        raise ValueError("selected frames must be sorted and unique")
    eligible = set(range(0, 10)) | set(range(152, 209))
    if not set(selected).issubset(eligible):
        raise ValueError("selected frame is outside the user-authorized contact intervals")
    frozen_fit = {
        "method": "UNCONSTRAINED_ORTHOGONAL_SVD",
        "normal_sign_only": "POSITIVE_WORLD_Z",
        "horizontal_prior": False,
        "robust_loss": False,
        "point_rejection": False,
        "depth_used": False,
        "historical_table_plane_used": False,
        "calibration_used": False,
        "comparison_reference": "FRAME0_CONTACT_RING_PLANE_DESCRIPTIVE_ONLY",
        "consensus_plane": "ALL_12_TRANSFORMED_CONTACT_RING_POINT_SETS",
    }
    if cfg["fit_contract"] != frozen_fit:
        raise ValueError("fit contract changed")
    if any(cfg["authorization"].values()):
        raise ValueError("a forbidden operation was authorized")

    paths = {key: Path(value).resolve(strict=True) for key, value in cfg["inputs"].items()}
    output = Path(cfg["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    audit = json.loads(paths["frozen_support_audit"].read_text(encoding="utf-8"))
    ring_archive = np.load(paths["frozen_support_points"], allow_pickle=False)
    ring_contract = cfg["contact_ring_contract"]
    if audit["classification"] != ring_contract["expected_classification"]:
        raise ValueError("frozen support classification changed")
    if audit["support_region_geometry"] != ring_contract["expected_geometry"]:
        raise ValueError("frozen support geometry changed")
    ring_indices = np.asarray(ring_archive["vertex_indices"], dtype=np.int64)
    if len(ring_indices) != ring_contract["expected_vertex_count"] or len(np.unique(ring_indices)) != len(ring_indices):
        raise ValueError("frozen ring vertex contract changed")

    mesh = trimesh.load_mesh(paths["mesh"], process=False)
    local_vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(cfg["mesh"]["scale_to_metres"])
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if ring_indices.min() < 0 or ring_indices.max() >= len(local_vertices):
        raise ValueError("frozen ring index is outside mesh")
    local_ring = local_vertices[ring_indices]
    poses = np.load(paths["official_pose"], allow_pickle=False).astype(np.float64)
    if poses.shape != (cfg["sample"]["total_frames"], 4, 4) or not np.isfinite(poses).all():
        raise ValueError("official pose array contract changed")
    active_before = sha256(paths["active_support_contract"])

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"contact_ring_vertex_indices": ring_indices}
    all_points: list[np.ndarray] = []
    for frame in selected:
        pose = poses[frame]
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-6):
            raise ValueError(f"frame {frame} pose rotation is not rigid")
        world = local_ring @ rotation.T + pose[:3, 3]
        fit = fit_plane(world)
        all_points.append(world)
        arrays[f"frame_{frame:03d}_ring_world_m"] = world
        arrays[f"frame_{frame:03d}_signed_distance_m"] = fit["signed_distance_m"]
        rows.append({
            "frame": frame,
            "pose_matrix": pose.tolist(),
            "support_point_count": int(len(world)),
            "normal": fit["normal"].tolist(),
            "offset_m": fit["offset_m"],
            "centroid_world_m": fit["centroid_world_m"].tolist(),
            "in_plane_range_m": fit["in_plane_range_m"].tolist(),
            "radial_p05_m": fit["radial_p05_m"],
            "radial_p95_m": fit["radial_p95_m"],
            "absolute_distance_m": fit["absolute_distance_m"],
        })

    reference_normal = np.asarray(rows[0]["normal"])
    reference_offset = float(rows[0]["offset_m"])
    reference_centroid = np.asarray(rows[0]["centroid_world_m"])
    consensus = fit_plane(np.concatenate(all_points, axis=0))
    consensus_normal = consensus["normal"]
    consensus_offset = float(consensus["offset_m"])
    for row in rows:
        normal = np.asarray(row["normal"])
        centroid = np.asarray(row["centroid_world_m"])
        row["comparison_to_frame0"] = {
            "normal_angle_deg": angle_deg(normal, reference_normal),
            "offset_delta_mm": (float(row["offset_m"]) - reference_offset) * 1000.0,
            "signed_plane_separation_at_frame0_centroid_mm": (
                float(normal @ reference_centroid - float(row["offset_m"])) * 1000.0
            ),
            "ring_centroid_signed_distance_to_frame0_plane_mm": (
                float(reference_normal @ centroid - reference_offset) * 1000.0
            ),
        }
        row["comparison_to_consensus"] = {
            "normal_angle_deg": angle_deg(normal, consensus_normal),
            "ring_centroid_signed_distance_mm": (
                float(consensus_normal @ centroid - consensus_offset) * 1000.0
            ),
        }

    normals = np.asarray([row["normal"] for row in rows])
    pair_angles = np.asarray([
        angle_deg(normals[first], normals[second])
        for first in range(len(normals)) for second in range(first + 1, len(normals))
    ])
    centroid_to_frame0 = np.asarray([
        row["comparison_to_frame0"]["ring_centroid_signed_distance_to_frame0_plane_mm"]
        for row in rows
    ])
    centroid_to_consensus = np.asarray([
        row["comparison_to_consensus"]["ring_centroid_signed_distance_mm"] for row in rows
    ])
    angle_to_frame0 = np.asarray([
        row["comparison_to_frame0"]["normal_angle_deg"] for row in rows
    ])
    result = {
        "schema": cfg["schema"],
        "sample": cfg["sample"],
        "inputs": {key: artifact(path) for key, path in paths.items()},
        "config": artifact(config_path),
        "mesh": {
            "vertex_count": int(len(local_vertices)),
            "face_count": int(len(faces)),
            "watertight": bool(mesh.is_watertight),
            "scale_to_metres": float(cfg["mesh"]["scale_to_metres"]),
        },
        "contact_ring_contract": ring_contract,
        "fit_contract": cfg["fit_contract"],
        "per_frame": rows,
        "consensus_plane": {
            "normal": consensus_normal.tolist(),
            "offset_m": consensus_offset,
            "absolute_distance_m": consensus["absolute_distance_m"],
        },
        "cross_frame_comparison": {
            "maximum_pairwise_normal_angle_deg": float(pair_angles.max()),
            "median_pairwise_normal_angle_deg": float(np.median(pair_angles)),
            "maximum_normal_angle_to_frame0_deg": float(angle_to_frame0.max()),
            "frame0_plane_ring_centroid_signed_distance_min_mm": float(centroid_to_frame0.min()),
            "frame0_plane_ring_centroid_signed_distance_max_mm": float(centroid_to_frame0.max()),
            "frame0_plane_ring_centroid_signed_distance_span_mm": float(np.ptp(centroid_to_frame0)),
            "consensus_ring_centroid_signed_distance_min_mm": float(centroid_to_consensus.min()),
            "consensus_ring_centroid_signed_distance_max_mm": float(centroid_to_consensus.max()),
            "consensus_ring_centroid_signed_distance_span_mm": float(np.ptp(centroid_to_consensus)),
            "plane_offset_span_mm": float(np.ptp([row["offset_m"] for row in rows]) * 1000.0),
        },
        "integrity": {
            "depth_used": False,
            "historical_table_plane_used": False,
            "calibration_used": False,
            "point_rejection_count": 0,
            "same_ring_indices_all_frames": True,
            "source_data_modified": False,
            "active_support_sha256_before": active_before,
            "active_support_sha256_after": sha256(paths["active_support_contract"]),
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    if result["integrity"]["active_support_sha256_before"] != result["integrity"]["active_support_sha256_after"]:
        raise AssertionError("active SupportSurfaceContract changed")
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    np.savez_compressed(output / "contact_ring_planes.npz", **arrays)

    with (output / "per_frame_planes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "frame", "support_points", "normal_x", "normal_y", "normal_z", "offset_m",
            "angle_to_frame0_deg", "centroid_distance_to_frame0_plane_mm",
            "angle_to_consensus_deg", "centroid_distance_to_consensus_mm",
            "ring_residual_median_mm", "ring_residual_p95_mm", "ring_residual_max_mm",
        ])
        for row in rows:
            residual = row["absolute_distance_m"]
            writer.writerow([
                row["frame"], row["support_point_count"], *row["normal"], row["offset_m"],
                row["comparison_to_frame0"]["normal_angle_deg"],
                row["comparison_to_frame0"]["ring_centroid_signed_distance_to_frame0_plane_mm"],
                row["comparison_to_consensus"]["normal_angle_deg"],
                row["comparison_to_consensus"]["ring_centroid_signed_distance_mm"],
                residual["median"] * 1000.0, residual["p95"] * 1000.0,
                residual["maximum"] * 1000.0,
            ])

    frames = np.asarray(selected)
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(frames, angle_to_frame0, "o-")
    axes[0].set_ylabel("Normal angle to frame 0 (deg)")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Bowl contact-ring plane orientation")
    axes[1].plot(frames, centroid_to_frame0, "o-", label="distance to frame-0 plane")
    axes[1].plot(frames, centroid_to_consensus, "s--", label="distance to consensus plane")
    axes[1].axhline(0.0, color="black", linewidth=1, linestyle=":")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Ring centroid signed distance (mm)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.savefig(output / "contact_ring_plane_consistency.png", dpi=160)
    plt.close(figure)

    comparison = result["cross_frame_comparison"]
    lines = [
        "# Brush bowl contact-ring planes across 12 table-contact frames",
        "",
        "This geometry-only audit applies each selected frame's official bowl pose to the same 216 mesh vertices previously validated as the bowl's annular lower support region. It does not read Depth, estimate a table, run calibration, or consume the active support plane as geometry input.",
        "",
        "The planes are contact-implied table planes: they describe where the table would have to be if the official pose places the full bowl ring in contact. Mesh and pose alone do not independently prove physical contact.",
        "",
        "## Cross-frame comparison",
        "",
        f"- Pairwise normal angle median / max: `{comparison['median_pairwise_normal_angle_deg']:.6f} / {comparison['maximum_pairwise_normal_angle_deg']:.6f} deg`.",
        f"- Maximum normal angle to frame 0: `{comparison['maximum_normal_angle_to_frame0_deg']:.6f} deg`.",
        f"- Ring-centroid signed distance to the frame-0 plane min / max / span: `{comparison['frame0_plane_ring_centroid_signed_distance_min_mm']:.6f} / {comparison['frame0_plane_ring_centroid_signed_distance_max_mm']:.6f} / {comparison['frame0_plane_ring_centroid_signed_distance_span_mm']:.6f} mm`.",
        f"- Ring-centroid signed distance to the 12-frame consensus plane min / max / span: `{comparison['consensus_ring_centroid_signed_distance_min_mm']:.6f} / {comparison['consensus_ring_centroid_signed_distance_max_mm']:.6f} / {comparison['consensus_ring_centroid_signed_distance_span_mm']:.6f} mm`.",
        f"- Plane offset span: `{comparison['plane_offset_span_mm']:.6f} mm`.",
        "",
        "## Per-frame planes",
        "",
        "| frame | normal | offset (m) | angle to f0 (deg) | ring centroid to f0 plane (mm) | ring centroid to consensus (mm) | ring residual median/P95/max (mm) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        residual = row["absolute_distance_m"]
        lines.append(
            f"| {row['frame']} | `[{row['normal'][0]:.8f}, {row['normal'][1]:.8f}, {row['normal'][2]:.8f}]` | {row['offset_m']:.9f} | {row['comparison_to_frame0']['normal_angle_deg']:.6f} | {row['comparison_to_frame0']['ring_centroid_signed_distance_to_frame0_plane_mm']:.6f} | {row['comparison_to_consensus']['ring_centroid_signed_distance_mm']:.6f} | {residual['median']*1000:.6f}/{residual['p95']*1000:.6f}/{residual['maximum']*1000:.6f} |"
        )
    lines += [
        "",
        "No equality tolerance is introduced after seeing the result. The numerical spans above are the audit outcome; interpretation must retain the contact assumption and cannot be presented as a Depth-derived table measurement.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
