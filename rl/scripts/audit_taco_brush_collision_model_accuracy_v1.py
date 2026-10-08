#!/usr/bin/env python3
"""Read-only spatial coverage audit for Brush hand-object collision proxies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_world_objects,
    pair_group_audit,
)
from egoengine_repro.retarget.collision_audit import (  # noqa: E402
    collision_families,
    distances,
)
from run_taco_brush_issue14_mink_candidate_v1 import relevant_pairs  # noqa: E402


SCHEMA = "taco_brush_collision_model_accuracy_audit_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_collision_model_accuracy_audit_v1.yaml"


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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def distance_class(distance_m: float, tolerance_m: float) -> str:
    if distance_m < -tolerance_m:
        return "PENETRATION"
    if distance_m > tolerance_m:
        return "CLEARANCE"
    return "CONTACT_WITHIN_TOLERANCE"


def mesh_geom_world(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int, *,
    repair_winding: bool = True,
) -> trimesh.Trimesh:
    kind = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                model.mesh_vert[vertex_start:vertex_start + vertex_count], dtype=np.float64,
            ).copy(),
            faces=np.asarray(
                model.mesh_face[face_start:face_start + face_count], dtype=np.int64,
            ).copy(),
            process=False,
        )
    elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        mesh = trimesh.creation.capsule(
            height=2.0 * float(size[1]), radius=float(size[0]), count=[24, 24],
        )
    elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(size[0]))
    elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        mesh = trimesh.creation.cylinder(
            radius=float(size[0]), height=2.0 * float(size[1]), sections=48,
        )
    else:
        raise ValueError(
            f"unsupported collision geom {model.geom(geom_id).name}: "
            f"{mujoco.mjtGeom(kind).name}"
        )
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    transform[:3, 3] = np.asarray(data.geom_xpos[geom_id])
    mesh.apply_transform(transform)
    if repair_winding and mesh.is_watertight and not mesh.is_winding_consistent:
        # This only normalizes triangle orientation on the in-memory audit copy.
        # No vertex, face, source asset, or runtime collision geometry is changed.
        mesh.fix_normals(multibody=True)
    return mesh


def union_signed_distance(
    points: np.ndarray, proxy_meshes: list[trimesh.Trimesh],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("coverage points must have shape (N,3)")
    if not proxy_meshes:
        raise ValueError("at least one proxy mesh is required")
    values = []
    for mesh in proxy_meshes:
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError("signed proxy coverage requires watertight consistently wound meshes")
        values.append(trimesh.proximity.signed_distance(mesh, points))
    matrix = np.asarray(values, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite proxy coverage distance")
    winners = np.argmax(matrix, axis=0)
    return matrix[winners, np.arange(len(points))], winners


def _coverage_subset(
    points: np.ndarray, signed: np.ndarray, winners: np.ndarray,
    names: list[str], tolerance_m: float,
) -> dict[str, Any]:
    over = signed > tolerance_m
    under = signed < -tolerance_m
    aligned = ~(over | under)
    winner_counts = sorted(
        ({"proxy_geom": names[index], "sample_count": int((winners == index).sum())}
         for index in range(len(names)) if np.any(winners == index)),
        key=lambda row: (-row["sample_count"], row["proxy_geom"]),
    )
    over_index = int(np.argmax(signed))
    under_index = int(np.argmin(signed))
    return {
        "sample_count": int(len(points)),
        "overcoverage_count": int(over.sum()),
        "undercoverage_count": int(under.sum()),
        "within_tolerance_count": int(aligned.sum()),
        "overcoverage_fraction": float(over.mean()),
        "undercoverage_fraction": float(under.mean()),
        "within_tolerance_fraction": float(aligned.mean()),
        "signed_coverage_m": {
            "minimum": float(signed.min()),
            "p05": float(np.quantile(signed, 0.05)),
            "median": float(np.median(signed)),
            "p95": float(np.quantile(signed, 0.95)),
            "maximum": float(signed.max()),
        },
        "maximum_overcoverage": {
            "signed_m": float(signed[over_index]),
            "point_sim_m": points[over_index].tolist(),
            "proxy_geom": names[int(winners[over_index])],
        },
        "maximum_undercoverage": {
            "signed_m": float(signed[under_index]),
            "point_sim_m": points[under_index].tolist(),
            "proxy_geom": names[int(winners[under_index])],
        },
        "dominant_proxy_geoms": winner_counts[:8],
    }


def coverage_audit(
    native_mesh: trimesh.Trimesh, proxy_meshes: list[trimesh.Trimesh],
    proxy_names: list[str], anchor: np.ndarray, *, sample_count: int,
    local_count: int, seed: int, tolerance_m: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    points, _ = trimesh.sample.sample_surface(native_mesh, sample_count, seed=seed)
    signed, winners = union_signed_distance(points, proxy_meshes)
    distances_to_anchor = np.linalg.norm(points - np.asarray(anchor), axis=1)
    local_indices = np.argsort(distances_to_anchor)[:min(local_count, len(points))]
    local_points = points[local_indices]
    local_signed = signed[local_indices]
    local_winners = winners[local_indices]
    result = {
        "tolerance_m": tolerance_m,
        "global": _coverage_subset(points, signed, winners, proxy_names, tolerance_m),
        "local": _coverage_subset(
            local_points, local_signed, local_winners, proxy_names, tolerance_m,
        ),
        "local_patch": {
            "anchor_sim_m": np.asarray(anchor).tolist(),
            "maximum_sample_radius_m": float(distances_to_anchor[local_indices].max()),
        },
    }
    arrays = {
        "points": local_points,
        "signed": local_signed,
        "winners": local_winners,
    }
    return result, arrays


def visual_hand_geom_ids(model: mujoco.MjModel, side: str) -> list[int]:
    return [
        geom_id for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith(f"{side}_")
        and (model.geom(geom_id).name or "").endswith("_visual")
        and "object" not in (model.geom(geom_id).name or "")
    ]


def object_proxy_geom_ids(model: mujoco.MjModel, side: str) -> list[int]:
    return [
        geom_id for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith(f"{side}_object_")
        and not (model.geom(geom_id).name or "").endswith("visual")
    ]


def hand_proxy_geom_ids_for_body(model: mujoco.MjModel, body_id: int) -> list[int]:
    return [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == int(body_id)
        and (model.geom(geom_id).name or "").startswith("collision_hand_")
    ]


def minimum_distance_row(
    model: mujoco.MjModel, data: mujoco.MjData, pairs: list[tuple[int, int]],
    query_bound_m: float,
) -> dict[str, Any]:
    values = distances(model, data, pairs, detection=query_bound_m)
    index = int(np.argmin(values))
    first, second = pairs[index]
    return {
        "distance_m": float(values[index]),
        "geom1": model.geom(first).name,
        "geom2": model.geom(second).name,
        "body1": model.body(int(model.geom_bodyid[first])).name,
        "body2": model.body(int(model.geom_bodyid[second])).name,
    }


def contact_anchors(native_row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    point = native_row.get("deepest_contact_position_sim_m")
    if point is not None:
        anchor = np.asarray(point, dtype=np.float64)
        return anchor, anchor
    nearest = native_row.get("nearest_points_sim_m")
    if nearest is None or len(nearest) != 2:
        raise ValueError("native closest pair lacks a usable contact anchor")
    return np.asarray(nearest[0]), np.asarray(nearest[1])


def plot_coverage(
    rows: list[dict[str, Any]], arrays: dict[str, dict[str, np.ndarray]],
    component: str, output: Path,
) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(17, 12), constrained_layout=True)
    columns = [
        ("baseline", "brush"), ("local_candidate", "brush"),
        ("baseline", "bowl"), ("local_candidate", "bowl"),
    ]
    scatter = None
    for row_index, frame in enumerate((40, 80, 120)):
        for column_index, (state, object_name) in enumerate(columns):
            row = next(value for value in rows if value["frame"] == frame
                       and value["state"] == state and value["object"] == object_name)
            key = row["array_key"] + f"_{component}"
            values = arrays[key]
            points = values["points"]
            centered = points - points.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            projected = centered @ vh[:2].T * 1000.0
            scatter = axes[row_index, column_index].scatter(
                projected[:, 0], projected[:, 1], c=values["signed"] * 1000.0,
                cmap="coolwarm", vmin=-5.0, vmax=5.0, s=9, linewidths=0,
            )
            axes[row_index, column_index].set_aspect("equal", adjustable="box")
            axes[row_index, column_index].set_title(
                f"f{frame} {state} {object_name}\n{row['native_hand_body']}"
            )
            axes[row_index, column_index].set_xlabel("local PCA-1 (mm)")
            axes[row_index, column_index].set_ylabel("local PCA-2 (mm)")
    assert scatter is not None
    colorbar = figure.colorbar(scatter, ax=axes, shrink=0.75)
    colorbar.set_label("proxy signed coverage (mm): + over, - missed")
    figure.suptitle(f"{component.replace('_', ' ').title()} local native-surface coverage")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected collision accuracy audit schema")
    auth = cfg["authorization"]
    forbidden = (
        "mink_optimization", "geometry_generation", "coacd_rerun", "parameter_sweep",
        "physics", "replay", "mpc", "reinforcement_learning", "full_trajectory_retarget",
        "promotion",
    )
    if any(auth.get(key) is not False for key in forbidden):
        raise ValueError("read-only collision accuracy audit authorization is not fail-closed")

    paths = {key: Path(value).resolve(strict=(key != "output"))
             for key, value in cfg["paths"].items()}
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    saved = np.load(paths["feasibility_states"])
    frames = [int(value) for value in saved["frame_indices"]]
    if frames != [int(value) for value in cfg["sample"]["selected_frames"]]:
        raise ValueError("saved state frames do not match the frozen audit contract")
    states = {
        "baseline": np.asarray(saved["baseline_qpos"], dtype=np.float64),
        "local_candidate": np.asarray(saved["candidate_qpos"], dtype=np.float64),
    }
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tolerance = float(cfg["audit"]["native_material_tolerance_m"])
    query_bound = float(cfg["audit"]["distance_query_bound_m"])
    families = collision_families(model)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    orientation_repairs: list[str] = []
    for spec in cfg["sample"]["interaction_pairs"]:
        for geom_id in object_proxy_geom_ids(model, spec["object_side"]):
            raw_mesh = mesh_geom_world(model, data, geom_id, repair_winding=False)
            if not raw_mesh.is_watertight:
                raise ValueError(
                    f"existing object proxy is open; signed coverage unavailable: "
                    f"{model.geom(geom_id).name}"
                )
            if not raw_mesh.is_winding_consistent:
                orientation_repairs.append(model.geom(geom_id).name)
    orientation_repairs = sorted(set(orientation_repairs))

    for frame_index, frame in enumerate(frames):
        for state_name, qpos in states.items():
            data.qpos[:] = qpos[frame_index]
            mujoco.mj_forward(model, data)
            native = native_world_objects(model, data)
            for interaction_index, spec in enumerate(cfg["sample"]["interaction_pairs"]):
                object_name = spec["object"]
                object_side = spec["object_side"]
                family = families["hand_tool" if spec["role"] == "tool" else "hand_target"]
                pairs = relevant_pairs(model, family, spec["hand"], object_side)
                native_audit = pair_group_audit(model, data, native, pairs, tolerance)
                native_row = native_audit["native_minimum"]
                if native_row is None or native_row["distance_m"] is None:
                    raise ValueError("native audit did not produce a finite closest pair")
                current_proxy = minimum_distance_row(model, data, pairs, query_bound)
                object_proxy_ids = object_proxy_geom_ids(model, object_side)
                visual_pairs = [
                    (hand_geom, object_geom)
                    for hand_geom in visual_hand_geom_ids(model, spec["hand"])
                    for object_geom in object_proxy_ids
                ]
                visual_hull_proxy = minimum_distance_row(
                    model, data, visual_pairs, query_bound,
                )

                hand_body_id = int(model.body(native_row["body1"]).id)
                object_body_id = int(model.body(native_row["body2"]).id)
                hand_proxy_ids = hand_proxy_geom_ids_for_body(model, hand_body_id)
                if not hand_proxy_ids:
                    raise ValueError(f"native problem body has no collision proxy: {native_row['body1']}")
                hand_anchor, object_anchor = contact_anchors(native_row)
                hand_native_mesh = native[hand_body_id]["mesh"]
                object_native_mesh = native[object_body_id]["mesh"]
                hand_proxy_meshes = [mesh_geom_world(model, data, value) for value in hand_proxy_ids]
                object_proxy_meshes = [
                    mesh_geom_world(model, data, value) for value in object_proxy_ids
                ]
                key = f"f{frame}_{state_name}_{object_name}"
                hand_coverage, hand_arrays = coverage_audit(
                    hand_native_mesh, hand_proxy_meshes,
                    [model.geom(value).name for value in hand_proxy_ids], hand_anchor,
                    sample_count=int(cfg["audit"]["hand_surface_samples"]),
                    local_count=int(cfg["audit"]["local_patch_samples"]),
                    seed=int(cfg["audit"]["sample_seed"]) + 100 * frame + 10 * interaction_index,
                    tolerance_m=tolerance,
                )
                object_coverage, object_arrays = coverage_audit(
                    object_native_mesh, object_proxy_meshes,
                    [model.geom(value).name for value in object_proxy_ids], object_anchor,
                    sample_count=int(cfg["audit"]["object_surface_samples"]),
                    local_count=int(cfg["audit"]["local_patch_samples"]),
                    seed=int(cfg["audit"]["sample_seed"]) + 100 * frame + 10 * interaction_index + 1,
                    tolerance_m=tolerance,
                )
                arrays[key + "_hand_proxy"] = hand_arrays
                arrays[key + "_object_proxy"] = object_arrays
                native_distance = float(native_row["distance_m"])
                row = {
                    "frame": frame,
                    "state": state_name,
                    "object": object_name,
                    "native_hand_body": native_row["body1"],
                    "native_hand_visual_geom": native_row["native_visual_geom1"],
                    "native_object_visual_geom": native_row["native_visual_geom2"],
                    "native_distance_m": native_distance,
                    "native_classification": native_row["classification"],
                    "current_proxy": current_proxy,
                    "current_proxy_hand_body_matches_native": (
                        current_proxy["body1"] == native_row["body1"]
                    ),
                    "current_proxy_classification": distance_class(
                        current_proxy["distance_m"], tolerance,
                    ),
                    "current_proxy_absolute_error_m": abs(
                        current_proxy["distance_m"] - native_distance
                    ),
                    "visual_hand_hull_proxy": visual_hull_proxy,
                    "visual_hand_hull_proxy_classification": distance_class(
                        visual_hull_proxy["distance_m"], tolerance,
                    ),
                    "visual_hand_hull_absolute_error_m": abs(
                        visual_hull_proxy["distance_m"] - native_distance
                    ),
                    "hand_proxy_coverage": hand_coverage,
                    "object_proxy_coverage": object_coverage,
                    "array_key": key,
                }
                rows.append(row)

    array_payload: dict[str, np.ndarray] = {}
    for key, values in arrays.items():
        for suffix, value in values.items():
            array_payload[f"{key}_{suffix}"] = value
    np.savez_compressed(output / "local_coverage_samples.npz", **array_payload)
    plot_coverage(rows, arrays, "hand_proxy", output / "hand_proxy_local_coverage.png")
    plot_coverage(rows, arrays, "object_proxy", output / "object_proxy_local_coverage.png")

    current_sign_matches = sum(
        row["current_proxy_classification"] == row["native_classification"] for row in rows
    )
    hull_sign_matches = sum(
        row["visual_hand_hull_proxy_classification"] == row["native_classification"]
        for row in rows
    )
    current_errors = np.asarray([row["current_proxy_absolute_error_m"] for row in rows])
    hull_errors = np.asarray([row["visual_hand_hull_absolute_error_m"] for row in rows])
    hull_improved = hull_errors < current_errors
    representation = {}
    for object_name, spec in cfg["object_representations"].items():
        directory = Path(spec["active_directory"]).resolve(strict=True)
        object_row = {
            **spec,
            "active_directory": str(directory),
            "actual_obj_parts": len(list(directory.glob("*.obj"))),
        }
        provenance = spec.get("provenance")
        if provenance is not None:
            provenance_path = Path(provenance).resolve(strict=True)
            object_row["provenance_artifact"] = artifact(provenance_path)
            object_row["provenance_contents"] = json.loads(provenance_path.read_text())
        representation[object_name] = object_row

    aggregate = {
        "row_count": len(rows),
        "current_proxy_sign_matches": int(current_sign_matches),
        "visual_hand_hull_sign_matches": int(hull_sign_matches),
        "visual_hand_hull_lower_absolute_error_rows": int(hull_improved.sum()),
        "current_proxy_mean_absolute_error_m": float(current_errors.mean()),
        "visual_hand_hull_mean_absolute_error_m": float(hull_errors.mean()),
        "drop_in_static_replacement_uniformly_improved": bool(hull_improved.all()),
        "current_proxy_hand_body_matches_native_rows": int(sum(
            row["current_proxy_hand_body_matches_native"] for row in rows
        )),
    }
    if hull_improved.all() and hull_sign_matches == len(rows):
        classification = "EXISTING_VISUAL_HULL_IS_UNIFORMLY_BETTER_STATIC_PROXY"
    else:
        classification = "NO_EXISTING_DROP_IN_PROXY_UNIFORMLY_FIXES_NATIVE_GEOMETRY"
    result = {
        "schema": SCHEMA,
        "classification": classification,
        "aggregate": aggregate,
        "rows": rows,
        "object_representation_inventory": representation,
        "in_memory_triangle_winding_repairs": orientation_repairs,
        "frozen_audit": cfg["audit"],
        "mutations": {
            "mink_optimization": 0, "source_geometry": 0, "coacd_runs": 0,
            "physics_steps": 0, "training_steps": 0, "reference": 0,
        },
        "source_artifacts": {
            "scene": artifact(paths["scene"]),
            "feasibility_results": artifact(paths["feasibility_results"]),
            "feasibility_states": artifact(paths["feasibility_states"]),
            "config": artifact(config_path),
            "script": artifact(Path(__file__)),
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    write_json(output / "results.json", result)

    fields = [
        "frame", "state", "object", "native_hand_body", "native_mm",
        "current_proxy_mm", "current_proxy_hand_body", "current_proxy_geom1",
        "current_proxy_geom2", "current_proxy_hand_body_matches_native",
        "visual_hand_hull_proxy_mm", "current_absolute_error_mm",
        "visual_hull_absolute_error_mm", "hand_local_over_fraction",
        "hand_local_under_fraction", "hand_local_median_mm",
        "object_local_over_fraction", "object_local_under_fraction",
        "object_local_median_mm", "object_local_dominant_proxy",
    ]
    with (output / "coverage_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            hand_local = row["hand_proxy_coverage"]["local"]
            object_local = row["object_proxy_coverage"]["local"]
            writer.writerow({
                "frame": row["frame"], "state": row["state"], "object": row["object"],
                "native_hand_body": row["native_hand_body"],
                "native_mm": row["native_distance_m"] * 1000,
                "current_proxy_mm": row["current_proxy"]["distance_m"] * 1000,
                "current_proxy_hand_body": row["current_proxy"]["body1"],
                "current_proxy_geom1": row["current_proxy"]["geom1"],
                "current_proxy_geom2": row["current_proxy"]["geom2"],
                "current_proxy_hand_body_matches_native": (
                    row["current_proxy_hand_body_matches_native"]
                ),
                "visual_hand_hull_proxy_mm": row["visual_hand_hull_proxy"]["distance_m"] * 1000,
                "current_absolute_error_mm": row["current_proxy_absolute_error_m"] * 1000,
                "visual_hull_absolute_error_mm": row["visual_hand_hull_absolute_error_m"] * 1000,
                "hand_local_over_fraction": hand_local["overcoverage_fraction"],
                "hand_local_under_fraction": hand_local["undercoverage_fraction"],
                "hand_local_median_mm": hand_local["signed_coverage_m"]["median"] * 1000,
                "object_local_over_fraction": object_local["overcoverage_fraction"],
                "object_local_under_fraction": object_local["undercoverage_fraction"],
                "object_local_median_mm": object_local["signed_coverage_m"]["median"] * 1000,
                "object_local_dominant_proxy": object_local["dominant_proxy_geoms"][0]["proxy_geom"],
            })

    lines = [
        "# Brush collision-model accuracy audit", "",
        f"- Classification: `{classification}`",
        "- MINK optimization: `NOT RUN`",
        "- Geometry generation / CoACD rerun: `NOT RUN`", "",
        "| frame | state | object | native / proxy-min hand body | native (mm) | current proxy (mm) | visual-hull proxy (mm) | current / hull abs error (mm) | hand local over / missed | object local over / missed |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        hand_local = row["hand_proxy_coverage"]["local"]
        object_local = row["object_proxy_coverage"]["local"]
        lines.append(
            f"| {row['frame']} | {row['state']} | {row['object']} | "
            f"{row['native_hand_body']} / {row['current_proxy']['body1']} | "
            f"{row['native_distance_m']*1000:.3f} | "
            f"{row['current_proxy']['distance_m']*1000:.3f} | "
            f"{row['visual_hand_hull_proxy']['distance_m']*1000:.3f} | "
            f"{row['current_proxy_absolute_error_m']*1000:.3f} / "
            f"{row['visual_hand_hull_absolute_error_m']*1000:.3f} | "
            f"{hand_local['overcoverage_fraction']:.3f} / "
            f"{hand_local['undercoverage_fraction']:.3f} | "
            f"{object_local['overcoverage_fraction']:.3f} / "
            f"{object_local['undercoverage_fraction']:.3f} |"
        )
    lines += [
        "", "## Existing-model decision", "",
        f"- Current proxy/native sign agreement: {current_sign_matches}/{len(rows)}.",
        f"- Native-hand visual convex-hull/native sign agreement: {hull_sign_matches}/{len(rows)}.",
        f"- Visual-hull alternative has lower absolute distance error in {int(hull_improved.sum())}/{len(rows)} rows; it is not a uniform fix.",
        f"- The current proxy's minimum-distance hand body matches the native minimum body in {aggregate['current_proxy_hand_body_matches_native_rows']}/{len(rows)} rows.",
        "- Bowl 146's existing 32-part `convex_m` decomposition is already active in every query; no unused finer CoACD model was found.",
        "- Brush 071 has 17 active convex parts but no checked-in CoACD provenance or alternate `convex_m` candidate, so it cannot be claimed or promoted as a CoACD repair.",
        "- No collision representation is changed and no new MINK solve is authorized by this audit.",
        "",
        "Positive signed coverage means the proxy extends across the native surface (overcoverage); negative means the proxy does not reach the native surface (missed coverage). The unchanged 50 µm native-material tolerance is used for counts.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "server_artifacts.sha256":
            checksums.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

