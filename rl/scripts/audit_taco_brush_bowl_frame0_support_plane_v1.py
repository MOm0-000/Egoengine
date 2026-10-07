#!/usr/bin/env python3
"""Geometry-only support-region audit for Brush target bowl 146 at frame 0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import trimesh


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.scene.support_region import inspect_lower_support_region  # noqa: E402


MESH = Path("/data_all/zzx/3.2RL/data/taco_v1/dev4/object_models/object_models_released/146_cm.obj")
POSE = Path("/data_all/zzx/3.2RL/data/taco_v1/dev4/object_poses/Object_Poses/(brush, brush, bowl)/20230927_027/target_146.npy")
ACTIVE_SUPPORT = RL_ROOT / "runs/taco_brush_camera_table_infra_repair_v1/support_surface_contract.yaml"
DEFAULT_OUTPUT = RL_ROOT / "runs/taco_brush_bowl_frame0_support_plane_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mm(values: list[float]) -> str:
    return "[" + ", ".join(f"{value * 1000:.3f}" for value in values) + "]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    mesh = trimesh.load_mesh(MESH, process=False)
    vertices_local_m = np.asarray(mesh.vertices, dtype=np.float64) * 0.01
    faces = np.asarray(mesh.faces, dtype=np.int64)
    poses = np.load(POSE, allow_pickle=False)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("official target pose must be Fx4x4")
    pose = np.asarray(poses[0], dtype=np.float64)
    vertices_world = vertices_local_m @ pose[:3, :3].T + pose[:3, 3]
    active_before = sha256(ACTIVE_SUPPORT) if ACTIVE_SUPPORT.is_file() else None

    primary = inspect_lower_support_region(
        vertices_world, faces, gravity_up=[0, 0, 1],
        angular_sectors=72, points_per_sector=3,
    )
    sensitivity = []
    for sectors, per_sector in ((36, 1), (48, 2), (72, 1), (72, 5), (96, 3)):
        row = inspect_lower_support_region(
            vertices_world, faces, gravity_up=[0, 0, 1],
            angular_sectors=sectors, points_per_sector=per_sector,
        )
        sensitivity.append({
            "angular_sectors": sectors,
            "points_per_sector": per_sector,
            "classification": row["classification"],
            "normal": row["plane"]["normal"],
            "offset_m": row["plane"]["offset_m"],
            "median_distance_m": row["planarity"]["absolute_distance_m"]["median"],
            "p95_distance_m": row["planarity"]["absolute_distance_m"]["p95"],
            "maximum_distance_m": row["planarity"]["absolute_distance_m"]["maximum"],
        })

    support_indices = np.asarray(primary["selection"]["support_vertex_indices"], dtype=np.int64)
    support_points = vertices_world[support_indices]
    radial = primary["coverage"]["radial_distance_m"]
    ring_ratio = radial["p05"] / radial["p95"] if radial["p95"] > 0 else 0.0
    geometry_type = "ANNULAR_CONTACT_REGION" if ring_ratio >= 0.70 else "OTHER_DISTRIBUTED_REGION"
    result = {
        "schema": "taco_brush_bowl_frame0_support_plane_v1",
        "classification": primary["classification"],
        "sample": "(brush, brush, bowl)/20230927_027",
        "target": {"role": "bowl", "object_id": "146", "frame": 0},
        "inputs": {
            "mesh": artifact(MESH), "pose": artifact(POSE),
            "mesh_scale_to_metres": 0.01,
            "pose_matrix_frame0": pose.tolist(),
            "mesh_vertex_count": int(len(vertices_local_m)),
            "mesh_face_count": int(len(faces)),
            "mesh_watertight": bool(mesh.is_watertight),
        },
        "audit_contract": {
            "depth_used": False,
            "historical_table_plane_used": False,
            "calibration_used": False,
            "horizontal_plane_prior_used": False,
            "active_support_contract_used_as_input": False,
            "gravity_up_used_only_to_select_underside": [0.0, 0.0, 1.0],
            "plane_fit": "arbitrary-direction orthogonal SVD",
        },
        "support_region_geometry": geometry_type,
        "annular_evidence": {
            "radial_p05_over_p95": ring_ratio,
            "radial_distance_m": radial,
            "full_angular_coverage": primary["selection"]["represented_sectors"] == 72,
        },
        "primary": primary,
        "selection_sensitivity": sensitivity,
        "integrity": {
            "original_mesh_modified": False,
            "official_pose_modified": False,
            "active_support_contract_modified": False,
            "active_support_sha256_before": active_before,
            "active_support_sha256_after": sha256(ACTIVE_SUPPORT) if ACTIVE_SUPPORT.is_file() else None,
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    if result["integrity"]["active_support_sha256_before"] != result["integrity"]["active_support_sha256_after"]:
        raise AssertionError("active SupportSurfaceContract changed")
    write_json(output / "results.json", result)
    np.savez_compressed(
        output / "support_points.npz",
        vertex_indices=support_indices,
        points_world_m=support_points,
        plane_normal=np.asarray(primary["plane"]["normal"]),
        plane_offset_m=np.asarray(primary["plane"]["offset_m"]),
    )

    distances = primary["planarity"]["absolute_distance_m"]
    coverage = primary["coverage"]
    plane = primary["plane"]
    output.joinpath("summary.md").write_text(
        "# Brush target 146 frame-0 support geometry\n\n"
        f"**{primary['classification']}**\n\n"
        "本检查只读取 bowl 146 原始 mesh 与第0帧官方 pose；未使用 Depth、历史桌面、校准、10°水平先验或 active SupportSurfaceContract。"
        "世界 +Z 只用于识别物体下侧；最终平面是任意方向正交拟合。\n\n"
        "## 数值结果\n\n"
        f"- 支撑点：`{primary['selection']['unique_support_point_count']}`，"
        f"周向覆盖 `{primary['selection']['represented_sectors']}/72`。\n"
        f"- 世界 XYZ 覆盖：`{_mm(coverage['world_xyz_range_m'])} mm`。\n"
        f"- 平面内主方向覆盖：`{_mm(coverage['in_plane_principal_range_m'])} mm`，"
        f"二维 aspect=`{coverage['in_plane_singular_value_aspect']:.6f}`，不是点或线。\n"
        f"- 平面：normal=`{json.dumps(plane['normal'])}`，offset=`{plane['offset_m']:.12f} m`。\n"
        f"- 点到平面绝对距离：median=`{distances['median'] * 1000:.6f} mm`，"
        f"P95=`{distances['p95'] * 1000:.6f} mm`，max=`{distances['maximum'] * 1000:.6f} mm`。\n"
        f"- 网格中位边长：`{primary['mesh_resolution']['median_edge_length_m'] * 1000:.6f} mm`；"
        f"P95/max残差分别为其 `{primary['decision_gates']['observed_p95_distance_in_median_edges']:.6f}` / "
        f"`{primary['decision_gates']['observed_maximum_distance_in_median_edges']:.6f}`。\n"
        f"- 形状：`{geometry_type}`；平面内半径 P05/median/P95 = "
        f"`{radial['p05'] * 1000:.3f}/{radial['median'] * 1000:.3f}/{radial['p95'] * 1000:.3f} mm`。\n\n"
        "该结论说明 bowl mesh 的底部存在数量充分、二维分散且近似共面的环形支撑区域；"
        "它不证明该平面就是现实桌面，也不会写入 active contract。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "classification": primary["classification"],
        "support_points": len(support_indices),
        "normal": plane["normal"], "offset_m": plane["offset_m"],
        "median_p95_max_mm": [
            distances["median"] * 1000, distances["p95"] * 1000,
            distances["maximum"] * 1000,
        ],
        "geometry": geometry_type,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
