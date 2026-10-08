#!/usr/bin/env python3
"""Attribute fixed-frame Brush hand-object penetrations without mutation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import mujoco
import numpy as np
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_pair_measurement,
    native_world_objects,
    pair_group_audit,
    triangle_object,
)
from egoengine_repro.evaluation.taco_surface import reconstruct_taco_mano  # noqa: E402
from egoengine_repro.retarget.collision_audit import (  # noqa: E402
    collision_families,
    distances,
)
from run_taco_brush_issue14_mink_candidate_v1 import relevant_pairs  # noqa: E402


SCHEMA = "taco_brush_mano_xhand_object_penetration_attribution_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_mano_xhand_object_penetration_attribution_v1.yaml"


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


def penetration_status(distance_m: float, tolerance_m: float) -> str:
    if distance_m < -tolerance_m:
        return "PENETRATION"
    if distance_m <= tolerance_m:
        return "CONTACT_WITHIN_TOLERANCE"
    return "CLEARANCE"


def origin_classification(
    official_distance_m: float, robot_distance_m: float, tolerance_m: float,
) -> str:
    official = penetration_status(official_distance_m, tolerance_m) == "PENETRATION"
    robot = penetration_status(robot_distance_m, tolerance_m) == "PENETRATION"
    if official and robot:
        return "OFFICIAL_GEOMETRY_PENETRATION_RETAINED_AFTER_RETARGET"
    if not official and robot:
        return "RETARGET_ADDED_PENETRATION"
    if official and not robot:
        return "RETARGET_REMOVED_OFFICIAL_PENETRATION"
    return "NO_MATERIAL_PENETRATION"


def mesh_entry(name: str, mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return {
        "geom": name,
        "vertices_sim_m": vertices,
        "faces": faces,
        "mesh": mesh,
        "fcl": triangle_object(vertices, faces),
    }


def official_geometry(cfg: dict[str, Any], data_root: Path, mano_root: Path) -> tuple[dict, dict]:
    sequence = cfg["sample"]["sequence"]
    hands_root = data_root / "hand_poses/Hand_Poses" / sequence
    objects_root = data_root / "object_poses/Object_Poses" / sequence
    mesh_root = data_root / "object_models/object_models_released"
    hands: dict[str, dict[str, Any]] = {}
    for side in ("right", "left"):
        vertices, joints, faces, _, keys = reconstruct_taco_mano(
            hands_root / f"{side}_hand.pkl",
            hands_root / f"{side}_hand_shape.pkl",
            mano_root / f"MANO_{side.upper()}.pkl",
            side=side,
        )
        expected_keys = tuple(f"{value:05d}" for value in range(1, cfg["sample"]["frames"] + 1))
        if keys != expected_keys:
            raise ValueError(f"official {side} MANO frame keys do not match 1..209")
        if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
            raise ValueError("official MANO reconstruction contains nonfinite values")
        hands[side] = {
            "vertices": vertices.astype(np.float64),
            "faces": np.asarray(faces, dtype=np.int32),
            "pose": hands_root / f"{side}_hand.pkl",
            "shape": hands_root / f"{side}_hand_shape.pkl",
            "model": mano_root / f"MANO_{side.upper()}.pkl",
        }
    objects: dict[str, dict[str, Any]] = {}
    for spec in cfg["sample"]["interaction_pairs"]:
        role, object_id, name = spec["role"], spec["object_id"], spec["object"]
        mesh_path = mesh_root / f"{object_id}_cm.obj"
        pose_path = objects_root / f"{role}_{object_id}.npy"
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"official {name} asset is not one triangle mesh")
        mesh.apply_scale(0.01)
        poses = np.load(pose_path, allow_pickle=False).astype(np.float64)
        if poses.shape != (cfg["sample"]["frames"], 4, 4):
            raise ValueError(f"official {name} poses do not match the timeline")
        objects[name] = {
            "mesh": mesh,
            "poses": poses,
            "mesh_path": mesh_path,
            "pose_path": pose_path,
        }
    return hands, objects


def official_measurement(
    hand: dict[str, Any], obj: dict[str, Any], frame: int, tolerance_m: float,
    *, hand_name: str, object_name: str,
) -> dict[str, Any]:
    hand_mesh = trimesh.Trimesh(
        vertices=hand["vertices"][frame], faces=hand["faces"], process=False,
    )
    object_mesh = obj["mesh"].copy()
    object_mesh.apply_transform(obj["poses"][frame])
    measured = native_pair_measurement(
        mesh_entry(f"official_mano_{hand_name}", hand_mesh),
        mesh_entry(f"official_{object_name}", object_mesh),
        tolerance_m,
    )
    distance = measured["distance_m"]
    if distance is None or not np.isfinite(distance):
        raise ValueError("official triangle-mesh distance is not finite")
    return {
        "distance_m": float(distance),
        "status": penetration_status(float(distance), tolerance_m),
        "surface_intersection": bool(measured["surface_intersection"]),
        "reported_penetration_depth_m": measured["reported_penetration_depth_m"],
        "contact_count": measured["contact_count"],
        "hand_mesh_watertight": bool(hand_mesh.is_watertight),
        "object_mesh_watertight": bool(object_mesh.is_watertight),
    }


def load_prior_proxy(path: Path) -> dict[tuple[int, str], float]:
    result = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frame = int(row["frame"])
            result[(frame, "brush")] = float(row["brush_right_hand_proxy_distance_m"])
            result[(frame, "bowl")] = float(row["bowl_left_hand_proxy_distance_m"])
    return result


def load_prior_native(path: Path) -> dict[tuple[int, str], float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    key = {"brush": "brush_right_hand", "bowl": "bowl_left_hand"}
    return {
        (int(row["frame"]), name): float(
            row["objects"][group]["minimum_native_distance_m"]
        )
        for row in raw["rows"] for name, group in key.items()
    }


def proxy_summary(rows: list[dict[str, Any]], tolerance_m: float) -> dict[str, Any]:
    matches = [row["proxy_native_sign_agreement"] for row in rows]
    errors = [abs(row["proxy_distance_m"] - row["xhand_native_distance_m"]) for row in rows]
    penetrating = [row for row in rows if row["xhand_native_status"] == "PENETRATION"]
    ratios = [
        abs(row["proxy_distance_m"]) / abs(row["xhand_native_distance_m"])
        for row in penetrating
    ]
    return {
        "observation_count": len(rows),
        "sign_agreement_count": int(sum(matches)),
        "sign_disagreement_count": int(len(matches) - sum(matches)),
        "sign_agreement_fraction": float(np.mean(matches)),
        "maximum_absolute_distance_error_m": float(max(errors)),
        "mean_absolute_distance_error_m": float(np.mean(errors)),
        "penetrating_observation_count": len(penetrating),
        "proxy_to_native_penetration_depth_ratio_min": float(min(ratios)) if ratios else None,
        "proxy_to_native_penetration_depth_ratio_max": float(max(ratios)) if ratios else None,
        "tolerance_m": tolerance_m,
        "scope": "SIX_PREDECLARED_FRAMES_ONLY",
    }


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "frame", "hand", "object", "official_mano_distance_mm", "official_status",
        "xhand_native_distance_mm", "xhand_native_status", "origin_classification",
        "proxy_distance_mm", "proxy_status", "proxy_minus_native_mm",
        "proxy_native_sign_agreement",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "frame": row["frame"], "hand": row["hand"], "object": row["object"],
                "official_mano_distance_mm": row["official_mano_distance_m"] * 1000,
                "official_status": row["official_mano_status"],
                "xhand_native_distance_mm": row["xhand_native_distance_m"] * 1000,
                "xhand_native_status": row["xhand_native_status"],
                "origin_classification": row["origin_classification"],
                "proxy_distance_mm": row["proxy_distance_m"] * 1000,
                "proxy_status": row["proxy_status"],
                "proxy_minus_native_mm": (
                    row["proxy_distance_m"] - row["xhand_native_distance_m"]
                ) * 1000,
                "proxy_native_sign_agreement": row["proxy_native_sign_agreement"],
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected attribution config schema")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], "HEAD"],
        cwd=REPO_ROOT, check=True,
    )
    forbidden = (
        "modify_table", "modify_object_pose", "modify_mink", "modify_source_trajectory",
        "retarget", "physics", "replay", "mpc", "reinforcement_learning",
        "training_search", "promotion", "chunk_commit",
    )
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("read-only attribution config authorizes a forbidden operation")
    paths = {
        key: Path(value).resolve(strict=True)
        for key, value in cfg["paths"].items() if key != "output"
    }
    output = Path(cfg["paths"]["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tolerance = float(cfg["geometry_contract"]["material_penetration_tolerance_m"])
    frames = [int(value) for value in cfg["sample"]["selected_frames"]]

    hands, objects = official_geometry(cfg, paths["data_root"], paths["mano_model_dir"])
    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if qpos.shape != (cfg["sample"]["frames"], 50):
        raise ValueError("robot reference does not match the frozen timeline")
    prior_proxy = load_prior_proxy(paths["prior_proxy_trace"])
    prior_native = load_prior_native(paths["prior_native_audit"])
    effective = json.loads(paths["effective_mink_contract"].read_text(encoding="utf-8"))

    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    data = mujoco.MjData(model)
    families = collision_families(model)
    specs = cfg["sample"]["interaction_pairs"]
    shell_pairs = {
        spec["object"]: relevant_pairs(
            model,
            families["hand_tool" if spec["role"] == "tool" else "hand_target"],
            spec["hand"], "right" if spec["role"] == "tool" else "left",
        ) for spec in specs
    }
    rows = []
    for frame in frames:
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        native = native_world_objects(model, data)
        for spec in specs:
            hand_name, object_name = spec["hand"], spec["object"]
            official = official_measurement(
                hands[hand_name], objects[object_name], frame, tolerance,
                hand_name=hand_name, object_name=object_name,
            )
            native_report = pair_group_audit(
                model, data, native, shell_pairs[object_name], tolerance)
            native_distance = float(native_report["native_minimum"]["distance_m"])
            proxy_distance = float(distances(
                model, data, shell_pairs[object_name], detection=0.1,
            ).min())
            if not np.isclose(
                native_distance, prior_native[(frame, object_name)], atol=1e-15, rtol=0,
            ):
                raise ValueError("XHand native distance did not reproduce prior evidence")
            if not np.isclose(
                proxy_distance, prior_proxy[(frame, object_name)], atol=1e-15, rtol=0,
            ):
                raise ValueError("MuJoCo proxy distance did not reproduce prior evidence")
            native_status = penetration_status(native_distance, tolerance)
            proxy_status = penetration_status(proxy_distance, tolerance)
            rows.append({
                "frame": frame,
                "hand": hand_name,
                "object": object_name,
                "official_mano_distance_m": official["distance_m"],
                "official_mano_status": official["status"],
                "official_measurement": official,
                "xhand_native_distance_m": native_distance,
                "xhand_native_status": native_status,
                "xhand_native_minimum_pair": native_report["native_minimum"],
                "origin_classification": origin_classification(
                    official["distance_m"], native_distance, tolerance),
                "proxy_distance_m": proxy_distance,
                "proxy_status": proxy_status,
                "proxy_native_sign_agreement": proxy_status == native_status,
            })

    counts: dict[str, int] = {}
    for row in rows:
        key = row["origin_classification"]
        counts[key] = counts.get(key, 0) + 1
    proxy = proxy_summary(rows, tolerance)
    hard_limits = effective.get("hard_limits", {})
    hand_object_constraint_present = any(
        key in hard_limits for key in ("object_collision", "hand_object_collision")
    )
    mink_assessment = {
        "effective_contract_has_hand_object_collision_limit": hand_object_constraint_present,
        "effective_contract_has_self_collision_limit": bool(
            hard_limits.get("self_collision", {}).get("present", False)),
        "effective_contract_has_native_support_limit": bool(
            hard_limits.get("native_support", {}).get("present", False)),
        "proxy_fixed_frame_sign_assessment": (
            "MATCHES_NATIVE_ON_ALL_SIX_FIXED_FRAMES"
            if proxy["sign_disagreement_count"] == 0
            else "PROXY_NATIVE_SIGN_DISAGREEMENT_PRESENT"
        ),
        "drop_in_reliability_verdict": "NOT_RELIABLE_AS_DROP_IN_NATIVE_HAND_OBJECT_GATE",
        "reasons": [
            "The active taco_bimanual MINK contract contains no hand-object collision limit.",
            "The proxy agrees with native penetration sign on this fixed sample but materially exaggerates penetration depth.",
            "Official MANO geometry itself penetrates the official objects at some frames, so a blanket zero-penetration proxy constraint would also suppress source-geometry interactions.",
            "Six frames do not establish whole-trajectory proxy/native equivalence.",
        ],
    }
    result = {
        "schema": SCHEMA,
        "classification": "READ_ONLY_ATTRIBUTION_COMPLETE",
        "selected_frames": frames,
        "material_penetration_tolerance_m": tolerance,
        "origin_classification_counts": counts,
        "rows": rows,
        "proxy_vs_native_summary": proxy,
        "mink_collision_avoidance_assessment": mink_assessment,
        "mutations": {
            "table": 0, "object_pose": 0, "mink": 0, "source_trajectory": 0,
            "retarget": 0, "physics_steps": 0, "training_steps": 0,
        },
        "source_artifacts": {
            key: artifact(path) for key, path in paths.items() if path.is_file()
        } | {
            "config": artifact(config_path),
            "script": artifact(Path(__file__)),
            "official_right_pose": artifact(hands["right"]["pose"]),
            "official_right_shape": artifact(hands["right"]["shape"]),
            "official_right_mano_model": artifact(hands["right"]["model"]),
            "official_left_pose": artifact(hands["left"]["pose"]),
            "official_left_shape": artifact(hands["left"]["shape"]),
            "official_left_mano_model": artifact(hands["left"]["model"]),
            "official_brush_mesh": artifact(objects["brush"]["mesh_path"]),
            "official_brush_pose": artifact(objects["brush"]["pose_path"]),
            "official_bowl_mesh": artifact(objects["bowl"]["mesh_path"]),
            "official_bowl_pose": artifact(objects["bowl"]["pose_path"]),
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    write_json(output / "results.json", result)
    write_table(output / "frame_pair_attribution.csv", rows)
    summary = [
        "# Brush MANO/XHand hand-object penetration attribution", "",
        "This is a read-only six-frame geometry audit. It does not modify or rerun MINK.", "",
        "| frame | pair | official MANO/native object (mm) | XHand/native object (mm) | MuJoCo proxy (mm) | attribution |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        summary.append(
            f"| {row['frame']} | {row['hand']}/{row['object']} | "
            f"{row['official_mano_distance_m']*1000:.6f} | "
            f"{row['xhand_native_distance_m']*1000:.6f} | "
            f"{row['proxy_distance_m']*1000:.6f} | "
            f"`{row['origin_classification']}` |"
        )
    summary += [
        "", "## Result", "",
        f"- Origin counts: `{json.dumps(counts, sort_keys=True)}`",
        f"- Proxy/native sign agreement: `{proxy['sign_agreement_count']}/{proxy['observation_count']}` on the fixed sample.",
        f"- Proxy/native maximum absolute distance error: `{proxy['maximum_absolute_distance_error_m']*1000:.6f} mm`.",
        f"- Penetration-depth exaggeration range: `{proxy['proxy_to_native_penetration_depth_ratio_min']:.3f}x` to `{proxy['proxy_to_native_penetration_depth_ratio_max']:.3f}x`.",
        "- Active MINK hand-object collision limit: `ABSENT`.",
        "- Drop-in proxy verdict: `NOT_RELIABLE_AS_DROP_IN_NATIVE_HAND_OBJECT_GATE`.", "",
        "The word inherited is used only at frame/interaction-pair level. MANO and XHand do not have vertex-identical topology, so this audit does not claim one-to-one contact-point inheritance.", "",
    ]
    (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "server_artifacts.sha256":
            checksums.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "classification": result["classification"],
        "origin_classification_counts": counts,
        "proxy_vs_native_summary": proxy,
        "mink_collision_avoidance_assessment": mink_assessment,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
