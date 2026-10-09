#!/usr/bin/env python3
"""Locate official MANO penetration and the first wholly legal source start."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_pair_measurement,
    triangle_object,
)
from egoengine_repro.evaluation.taco_surface import (  # noqa: E402
    MANO_TIP_VERTICES,
    SURFACE_REGION_ORDER,
    reconstruct_taco_mano,
)
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance  # noqa: E402


SCHEMA = "taco_brush_20231006_145_initialization_feasibility_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_20231006_145_initialization_feasibility_v1.yaml"


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
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fcl_entry(name: str, mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return {
        "geom": name,
        "vertices_sim_m": vertices,
        "faces": faces,
        "mesh": mesh,
        "fcl": triangle_object(vertices, faces),
    }


def material_pair(first: trimesh.Trimesh, second: trimesh.Trimesh, tolerance: float,
                  first_name: str, second_name: str) -> dict[str, Any]:
    measured = native_pair_measurement(
        fcl_entry(first_name, first), fcl_entry(second_name, second), tolerance,
    )
    distance = float(measured["distance_m"])
    return {
        "distance_m": distance,
        "material_penetration": bool(distance < -tolerance),
        "surface_intersection": bool(measured["surface_intersection"]),
        "reported_penetration_depth_m": float(measured["reported_penetration_depth_m"]),
    }


def load_object(data_root: Path, sequence: str, spec: dict[str, str]) -> dict[str, Any]:
    mesh_path = data_root / "object_models_released" / f"{spec['id']}_cm.obj"
    pose_path = data_root / "Object_Poses" / sequence / f"{spec['role']}_{spec['id']}.npy"
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"not one triangle mesh: {mesh_path}")
    mesh.apply_scale(0.01)
    poses = np.asarray(np.load(pose_path, allow_pickle=False), dtype=np.float64)
    return {"spec": spec, "mesh": mesh, "poses": poses, "mesh_path": mesh_path, "pose_path": pose_path}


def world_object(entry: dict[str, Any], frame: int) -> trimesh.Trimesh:
    mesh = entry["mesh"].copy()
    mesh.apply_transform(entry["poses"][frame])
    return mesh


def hand_mesh(hand: dict[str, Any], frame: int) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=hand["vertices"][frame], faces=hand["faces"], process=False)


def consecutive_start(flags: list[bool], length: int) -> int | None:
    for start in range(0, len(flags) - length + 1):
        if all(flags[start:start + length]):
            return start
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["schema"] != SCHEMA:
        raise ValueError("schema mismatch")
    forbidden = (
        "construct_safe_entry_candidate_after_geometry_gate", "run_mink", "physics", "replay",
        "mpc", "reinforcement_learning", "modify_original_data",
        "modify_active_support_contract", "promotion",
    )
    for key in forbidden:
        if cfg["authorization"][key]:
            raise ValueError(f"read-only phase forbids {key}")

    data_root = Path(cfg["paths"]["data_root"])
    output = Path(cfg["paths"]["output"])
    output.mkdir(parents=True, exist_ok=False)
    sequence = cfg["sample"]["sequence"]
    frames = int(cfg["sample"]["frames"])
    tolerance = float(cfg["geometry_contract"]["material_penetration_tolerance_m"])
    issue14_path = Path(cfg["geometry_contract"]["support_results"])
    issue14 = json.loads(issue14_path.read_text(encoding="utf-8"))
    table_z = float(issue14["candidate_support"]["table_world_z_m"])

    objects = {
        spec["name"]: load_object(data_root, sequence, spec)
        for spec in cfg["sample"]["objects"]
    }
    if any(entry["poses"].shape != (frames, 4, 4) for entry in objects.values()):
        raise ValueError("object timeline mismatch")

    hands: dict[str, dict[str, Any]] = {}
    for side in ("right", "left"):
        vertices, joints, faces, regions, keys = reconstruct_taco_mano(
            data_root / "Hand_Poses" / sequence / f"{side}_hand.pkl",
            data_root / "Hand_Poses" / sequence / f"{side}_hand_shape.pkl",
            data_root / "mano_v1_2" / "models" / f"MANO_{side.upper()}.pkl",
            side=side,
        )
        expected = tuple(f"{index:05d}" for index in range(1, frames + 1))
        if keys != expected:
            raise ValueError(f"{side} source frame keys mismatch")
        hands[side] = {
            "vertices": np.asarray(vertices, dtype=np.float64),
            "joints": np.asarray(joints, dtype=np.float64),
            "faces": np.asarray(faces, dtype=np.int32),
            "regions": np.asarray(regions, dtype=np.int8),
        }

    frame_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    legal_flags: list[bool] = []
    for frame in range(frames):
        world_objects = {name: world_object(entry, frame) for name, entry in objects.items()}
        world_hands = {side: hand_mesh(entry, frame) for side, entry in hands.items()}
        object_table = {
            name: float(np.min(np.asarray(mesh.vertices)[:, 2] - table_z))
            for name, mesh in world_objects.items()
        }
        table_legal = all(value >= -tolerance for value in object_table.values())
        object_pair = material_pair(
            world_objects["brush"], world_objects["bowl"], tolerance, "brush", "bowl",
        )
        pair_rows.append({
            "frame": frame, "pair": "brush-bowl", **object_pair,
        })

        hand_table_min: dict[str, float] = {}
        hand_object_penetration = False
        hand_table_legal = True
        for side, hand in hands.items():
            vertices = hand["vertices"][frame]
            signed_table = vertices[:, 2] - table_z
            hand_table_min[side] = float(np.min(signed_table))
            hand_table_legal &= hand_table_min[side] >= -tolerance
            tip_indices = MANO_TIP_VERTICES[side]
            tip_points = vertices[tip_indices]
            for region_id, region_name in enumerate(SURFACE_REGION_ORDER):
                selected = hand["regions"] == region_id
                values = signed_table[selected]
                region_vertices = vertices[selected]
                tip_distance = np.linalg.norm(
                    region_vertices[:, None, :] - tip_points[None, :, :], axis=2,
                ).min(axis=1)
                region_rows.append({
                    "frame": frame,
                    "side": side,
                    "region": region_name,
                    "minimum_table_signed_distance_m": float(np.min(values)),
                    "vertices_below_tolerance": int(np.sum(values < -tolerance)),
                    "penetrating_vertices_near_any_fingertip": int(np.sum(
                        (values < -tolerance)
                        & (tip_distance <= float(cfg["geometry_contract"]["fingertip_proximity_radius_m"]))
                    )),
                })
            for object_name, object_mesh in world_objects.items():
                measured = material_pair(
                    world_hands[side], object_mesh, tolerance,
                    f"mano_{side}", object_name,
                )
                hand_object_penetration |= measured["material_penetration"]
                signed_object = closed_mesh_signed_distance(object_mesh, vertices)
                inside = signed_object > tolerance
                inside_regions = sorted({
                    SURFACE_REGION_ORDER[int(value)] for value in hand["regions"][inside]
                })
                pair_rows.append({
                    "frame": frame,
                    "pair": f"mano_{side}-{object_name}",
                    **measured,
                    "inside_hand_vertex_count": int(np.sum(inside)),
                    "inside_hand_regions": ";".join(inside_regions),
                    "maximum_inside_hand_vertex_depth_m": float(max(0.0, np.max(signed_object))),
                })

        hand_pair = material_pair(
            world_hands["right"], world_hands["left"], tolerance,
            "mano_right", "mano_left",
        )
        pair_rows.append({"frame": frame, "pair": "mano_right-mano_left", **hand_pair})
        legal = bool(
            table_legal and hand_table_legal
            and not object_pair["material_penetration"]
            and not hand_object_penetration
            and not hand_pair["material_penetration"]
        )
        legal_flags.append(legal)
        frame_rows.append({
            "frame": frame,
            "time_s": frame / float(cfg["sample"]["fps"]),
            "right_mano_table_min_m": hand_table_min["right"],
            "left_mano_table_min_m": hand_table_min["left"],
            "brush_table_min_m": object_table["brush"],
            "bowl_table_min_m": object_table["bowl"],
            "hand_table_legal": hand_table_legal,
            "object_table_legal": table_legal,
            "hand_object_material_penetration": hand_object_penetration,
            "hand_hand_material_penetration": hand_pair["material_penetration"],
            "object_object_material_penetration": object_pair["material_penetration"],
            "fully_geometry_legal": legal,
        })

    write_csv(output / "frame_legality.csv", frame_rows)
    write_csv(output / "mano_table_regions.csv", region_rows)
    write_csv(output / "native_pair_distances.csv", pair_rows)

    stable_count = int(cfg["geometry_contract"]["legal_start_consecutive_frames"])
    first_legal = next((index for index, value in enumerate(legal_flags) if value), None)
    first_stable = consecutive_start(legal_flags, stable_count)
    initial_last = int(cfg["sample"]["initial_action_last_frame_inclusive"])
    initial_region = [row for row in region_rows if row["frame"] <= initial_last]
    penetrating_regions: dict[str, list[str]] = {}
    for side in ("right", "left"):
        names = sorted({
            row["region"] for row in initial_region
            if row["side"] == side and row["vertices_below_tolerance"] > 0
        })
        penetrating_regions[side] = names
    result = {
        "schema": SCHEMA,
        "classification": (
            "DIRECT_SOURCE_START_AVAILABLE" if first_stable is not None
            else "NO_DIRECT_FULLY_LEGAL_SOURCE_START"
        ),
        "support": {"world_z_m": table_z, "source": cfg["geometry_contract"]["support_source"]},
        "frame0": frame_rows[0],
        "initial_action": {
            "frames": [0, initial_last],
            "right_table_penetrating_frames": [
                row["frame"] for row in frame_rows[:initial_last + 1]
                if row["right_mano_table_min_m"] < -tolerance
            ],
            "left_table_penetrating_frames": [
                row["frame"] for row in frame_rows[:initial_last + 1]
                if row["left_mano_table_min_m"] < -tolerance
            ],
            "penetrating_surface_regions": penetrating_regions,
            "fully_legal_frames": [
                row["frame"] for row in frame_rows[:initial_last + 1]
                if row["fully_geometry_legal"]
            ],
        },
        "direct_start": {
            "first_single_legal_frame": first_legal,
            "required_consecutive_legal_frames": stable_count,
            "first_stable_legal_frame": first_stable,
            "discarded_original_frames_if_used": None if first_stable is None else list(range(first_stable)),
            "discarded_duration_s_if_used": None if first_stable is None else first_stable / float(cfg["sample"]["fps"]),
        },
        "counts": {
            "total_frames": frames,
            "fully_legal_frames": int(sum(legal_flags)),
            "right_table_penetrating_frames": int(sum(
                row["right_mano_table_min_m"] < -tolerance for row in frame_rows
            )),
            "left_table_penetrating_frames": int(sum(
                row["left_mano_table_min_m"] < -tolerance for row in frame_rows
            )),
            "hand_object_penetrating_frames": int(sum(
                row["hand_object_material_penetration"] for row in frame_rows
            )),
        },
        "next_gate": (
            "EVALUATE_DIRECT_START_AND_SAFE_ENTRY_COMPARATOR"
            if first_stable is not None else "CONSTRUCT_DEXTRACK_STYLE_SAFE_ENTRY_ONLY"
        ),
        "forbidden_runtime_counts": {"mink": 0, "physics": 0, "replay": 0, "mpc": 0, "rl": 0},
        "source_artifacts": {
            "config": artifact(config_path),
            "issue14_support": artifact(issue14_path),
            **{
                f"{name}_mesh": artifact(entry["mesh_path"])
                for name, entry in objects.items()
            },
            **{
                f"{name}_poses": artifact(entry["pose_path"])
                for name, entry in objects.items()
            },
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    write_json(output / "results.json", result)
    frame0_regions = [row for row in region_rows if row["frame"] == 0 and row["vertices_below_tolerance"]]
    lines = [
        "# 20231006_145 initialization feasibility: source geometry", "",
        f"- Classification: `{result['classification']}`",
        f"- Issue #14 table: world `z={table_z:.12f} m`",
        f"- First single fully legal source frame: `{first_legal}`",
        f"- First {stable_count}-frame legal start: `{first_stable}`",
        "- MINK / physics / Replay / MPC / RL: `NOT RUN`", "",
        "## Frame 0", "",
        "| entity | minimum table distance (mm) |",
        "|---|---:|",
        f"| official MANO right | {frame_rows[0]['right_mano_table_min_m'] * 1000:.6f} |",
        f"| official MANO left | {frame_rows[0]['left_mano_table_min_m'] * 1000:.6f} |",
        f"| brush | {frame_rows[0]['brush_table_min_m'] * 1000:.6f} |",
        f"| bowl | {frame_rows[0]['bowl_table_min_m'] * 1000:.6f} |", "",
        "## Frame-0 MANO regions materially below the table", "",
        "| side | region | minimum (mm) | vertices | near fingertip vertices |",
        "|---|---|---:|---:|---:|",
    ]
    for row in frame0_regions:
        lines.append(
            f"| {row['side']} | {row['region']} | "
            f"{row['minimum_table_signed_distance_m'] * 1000:.6f} | "
            f"{row['vertices_below_tolerance']} | "
            f"{row['penetrating_vertices_near_any_fingertip']} |"
        )
    lines += ["", "## Direct-start contract", ""]
    if first_stable is None:
        lines.append("No source frame begins the required consecutive fully legal interval; direct source cropping is unavailable.")
    else:
        lines += [
            f"Starting at frame {first_stable} would delete original frames `0..{first_stable - 1}` "
            f"({result['direct_start']['discarded_duration_s_if_used']:.3f} s).",
            "That cropped sequence must not be described as a complete reproduction.",
        ]
    lines += ["", "This phase only measures released MANO/object geometry. It does not repair the source, create a robot trajectory, or authorize physics.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "classification": result["classification"],
        "first_single_legal_frame": first_legal,
        "first_stable_legal_frame": first_stable,
        "counts": result["counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
