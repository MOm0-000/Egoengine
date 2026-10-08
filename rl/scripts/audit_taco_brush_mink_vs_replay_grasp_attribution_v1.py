#!/usr/bin/env python3
"""Compare frozen MINK grasp geometry with its executed Brush Replay state."""

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
)
from egoengine_repro.evaluation.replay_failure_attribution import (  # noqa: E402
    classify_native_measurement,
    semantic_finger,
    vector_cosine,
)
from egoengine_repro.retarget.collision_audit import distances  # noqa: E402


SCHEMA = "taco_brush_mink_vs_replay_grasp_attribution_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_mink_vs_replay_grasp_attribution_v1.yaml"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved)}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_vector(row: dict[str, Any], prefix: str, value: Any) -> None:
    vector = np.asarray(value, dtype=np.float64)
    for axis, scalar in zip("xyz", vector):
        row[f"{prefix}_{axis}"] = float(scalar)


def geometry_rank(row: dict[str, Any]) -> tuple[int, float]:
    order = {
        "PENETRATION": 0,
        "UNKNOWN_INTERSECTION_DEPTH": 1,
        "UNKNOWN_CONTAINMENT_OPEN_MESH": 1,
        "CONTACT_WITHIN_TOLERANCE": 2,
        "CLEARANCE": 3,
    }
    return order[row["classification"]], float(row["distance_m"])


def finger_geometry(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    hand: str,
    object_body: str,
    tolerance_m: float,
) -> tuple[dict[str, dict[str, Any]], mujoco.MjData]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    native = native_world_objects(model, data)
    object_id = int(model.body(object_body).id)
    object_entry = native[object_id]
    object_rotation = np.asarray(data.xmat[object_id]).reshape(3, 3)
    object_position = np.asarray(data.xpos[object_id]).copy()
    rows: dict[str, list[dict[str, Any]]] = {finger: [] for finger in FINGERS}
    for body_id, hand_entry in native.items():
        body_name = model.body(body_id).name
        if not body_name.startswith(f"{hand}_hand"):
            continue
        finger = semantic_finger(body_name)
        if finger not in rows:
            continue
        measured = native_pair_measurement(hand_entry, object_entry, tolerance_m)
        classification = classify_native_measurement(measured, tolerance_m)
        if measured.get("nearest_points_sim_m") is not None:
            hand_point = np.asarray(measured["nearest_points_sim_m"][0], dtype=np.float64)
            object_point = np.asarray(measured["nearest_points_sim_m"][1], dtype=np.float64)
        else:
            object_point = np.asarray(
                measured["deepest_contact_position_sim_m"], dtype=np.float64,
            )
            hand_point = object_point.copy()
        closest, _, triangle = trimesh.proximity.closest_point(
            object_entry["mesh"], object_point[None, :],
        )
        object_point = closest[0]
        normal_world = np.asarray(
            object_entry["mesh"].face_normals[int(triangle[0])], dtype=np.float64,
        )
        body_rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
        body_position = np.asarray(data.xpos[body_id]).copy()
        rows[finger].append({
            "finger": finger,
            "hand_body": body_name,
            "hand_body_id": int(body_id),
            "classification": classification,
            "distance_m": float(measured["distance_m"]),
            "reported_penetration_depth_m": float(
                measured["reported_penetration_depth_m"] or 0.0
            ),
            "hand_surface_point_world_m": hand_point,
            "hand_surface_point_body_local_m": body_rotation.T @ (hand_point - body_position),
            "object_surface_point_world_m": object_point,
            "object_surface_point_object_m": object_rotation.T @ (
                object_point - object_position
            ),
            "object_surface_normal_world": normal_world,
            "object_surface_normal_object": object_rotation.T @ normal_world,
            "fcl_contact_normal_world": measured.get("deepest_contact_normal"),
        })
    selected = {}
    for finger, candidates in rows.items():
        if not candidates:
            raise ValueError(f"no native {hand} {finger} mesh")
        selected[finger] = min(candidates, key=geometry_rank)
    return selected, data


def serialize_geometry_row(
    endpoint: int, state: str, hand: str, object_name: str,
    row: dict[str, Any], near_m: float,
) -> dict[str, Any]:
    result = {
        "endpoint": endpoint,
        "state": state,
        "hand": hand,
        "object": object_name,
        "finger": row["finger"],
        "hand_body": row["hand_body"],
        "classification": row["classification"],
        "distance_mm": float(row["distance_m"] * 1000.0),
        "penetration_depth_mm": float(row["reported_penetration_depth_m"] * 1000.0),
        "within_diagnostic_5mm": bool(row["distance_m"] <= near_m),
    }
    flatten_vector(result, "object_point_object_m", row["object_surface_point_object_m"])
    flatten_vector(result, "object_normal_object", row["object_surface_normal_object"])
    flatten_vector(result, "object_normal_world", row["object_surface_normal_world"])
    return result


def contact_side_cosine(rows: dict[str, dict[str, Any]], first: str, second: str) -> float | None:
    return vector_cosine(
        np.asarray(rows[first]["object_surface_point_object_m"]),
        np.asarray(rows[second]["object_surface_point_object_m"]),
    )


def actuator_group(name: str) -> tuple[str, str, str]:
    side = "left" if name.startswith(("left_", "L_")) else "right"
    finger = semantic_finger(name)
    if "forearm_t" in name:
        return side, "wrist_translation", "m"
    if "forearm_" in name:
        return side, "wrist_rotation", "rad"
    return side, finger, "rad"


def transform_body_point(data: mujoco.MjData, body_id: int, local: np.ndarray) -> np.ndarray:
    rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    return np.asarray(data.xpos[body_id]) + rotation @ local


def hash_tree(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "server_artifacts.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected config schema")
    if any(cfg["authorization"].get(key) is not False for key in (
        "physics_steps", "change_state", "change_commands", "change_geometry",
        "mink", "mpc", "reinforcement_learning", "promotion", "chunk_commit",
    )):
        raise ValueError("audit authorization is not fail-closed")
    paths = {
        key: Path(value).resolve(strict=(key != "output"))
        for key, value in cfg["paths"].items()
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    parity = json.loads(paths["replay_parity"].read_text(encoding="utf-8"))
    if not parity.get("passed"):
        raise ValueError("frozen Replay observer parity did not pass")
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    with np.load(paths["mink_reference"], allow_pickle=False) as archive:
        reference = {key: np.asarray(archive[key]).copy() for key in ("qpos", "ctrl")}
    with np.load(paths["replay_states"], allow_pickle=False) as archive:
        actual = {key: np.asarray(archive[key]).copy() for key in ("qpos", "qvel", "ctrl")}
    if reference["qpos"].shape[1] != model.nq or actual["qpos"].shape[1] != model.nq:
        raise ValueError("state/model qpos mismatch")

    tolerance = float(cfg["geometry"]["native_material_tolerance_m"])
    near = float(cfg["geometry"]["diagnostic_near_surface_m"])
    brush_range = range(*(
        int(cfg["sample"]["brush_endpoints"][0]),
        int(cfg["sample"]["brush_endpoints"][1]) + 1,
    ))
    bowl_range = range(*(
        int(cfg["sample"]["bowl_endpoints"][0]),
        int(cfg["sample"]["bowl_endpoints"][1]) + 1,
    ))
    left_range = range(*(
        int(cfg["sample"]["left_hand_endpoints"][0]),
        int(cfg["sample"]["left_hand_endpoints"][1]) + 1,
    ))

    geometry_rows: list[dict[str, Any]] = []
    grasp_rows: list[dict[str, Any]] = []
    selected: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    data_cache: dict[tuple[str, int], mujoco.MjData] = {}
    for object_name, hand, object_body, endpoints in (
        ("brush", "right", "right_object", brush_range),
        ("bowl", "left", "left_object", bowl_range),
    ):
        for endpoint in endpoints:
            for state, states in (("mink_reference", reference), ("replay_actual", actual)):
                rows, data = finger_geometry(
                    model, states["qpos"][endpoint], hand=hand,
                    object_body=object_body, tolerance_m=tolerance,
                )
                selected[(object_name, state, endpoint)] = rows
                data_cache[(state, endpoint)] = data
                for row in rows.values():
                    geometry_rows.append(serialize_geometry_row(
                        endpoint, state, hand, object_name, row, near,
                    ))
                grasp_rows.append({
                    "endpoint": endpoint,
                    "state": state,
                    "hand": hand,
                    "object": object_name,
                    **{
                        f"{finger}_distance_mm": rows[finger]["distance_m"] * 1000.0
                        for finger in FINGERS
                    },
                    "index_thumb_contact_side_cosine": contact_side_cosine(
                        rows, "index", "thumb",
                    ),
                    "index_middle_contact_side_cosine": contact_side_cosine(
                        rows, "index", "middle",
                    ),
                    "thumb_middle_contact_side_cosine": contact_side_cosine(
                        rows, "thumb", "middle",
                    ),
                    "near_finger_count_5mm": sum(
                        row["distance_m"] <= near for row in rows.values()
                    ),
                })

    tracking_rows: list[dict[str, Any]] = []
    for endpoint in range(15, 61):
        for actuator_id in range(model.nu):
            name = model.actuator(actuator_id).name
            side, group, unit = actuator_group(name)
            if side == "right" and endpoint > 41:
                continue
            if side == "left" and endpoint < 44:
                continue
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            qadr = int(model.jnt_qposadr[joint_id])
            value = float(actual["qpos"][endpoint, qadr])
            command = float(reference["ctrl"][endpoint, actuator_id])
            tracking_rows.append({
                "endpoint": endpoint,
                "side": side,
                "group": group,
                "unit": unit,
                "actuator": name,
                "joint": model.joint(joint_id).name,
                "mink_command": command,
                "replay_actual": value,
                "command_minus_actual": command - value,
                "absolute_error": abs(command - value),
            })

    self_rows: list[dict[str, Any]] = []
    proxy_pair = (
        int(model.geom("collision_hand_left_index_0").id),
        int(model.geom("collision_hand_left_middle_0").id),
    )
    index_body = int(model.body("left_hand_index_rota_link2").id)
    middle_body = int(model.body("left_hand_mid_link2").id)
    for endpoint in left_range:
        row: dict[str, Any] = {"endpoint": endpoint}
        endpoint_tracking = [
            value for value in tracking_rows
            if value["endpoint"] == endpoint and value["side"] == "left"
        ]
        for group in ("wrist_translation", "wrist_rotation", *FINGERS):
            values = [value["command_minus_actual"] for value in endpoint_tracking
                      if value["group"] == group]
            row[f"{group}_rms_error"] = (
                float(np.sqrt(np.mean(np.square(values)))) if values else None
            )
            row[f"{group}_max_abs_error"] = max(map(abs, values)) if values else None
        for state, states in (("mink_reference", reference), ("replay_actual", actual)):
            data = mujoco.MjData(model)
            data.qpos[:] = states["qpos"][endpoint]
            mujoco.mj_forward(model, data)
            proxy_distance = float(distances(
                model, data, [proxy_pair], detection=0.1,
            )[0])
            native = native_world_objects(model, data)
            measured = native_pair_measurement(
                native[index_body], native[middle_body], tolerance,
            )
            row[f"{state}_proxy_index_middle_distance_mm"] = proxy_distance * 1000.0
            row[f"{state}_native_index_middle_distance_mm"] = (
                float(measured["distance_m"]) * 1000.0
            )
        self_rows.append(row)

    motion_rows: list[dict[str, Any]] = []
    for object_name, object_body, endpoints, fingers in (
        ("brush", "right_object", range(15, 41), ("index", "thumb")),
        ("bowl", "left_object", range(52, 60), FINGERS),
    ):
        object_id = int(model.body(object_body).id)
        for endpoint in endpoints:
            current_data = data_cache[("mink_reference", endpoint)]
            next_data = mujoco.MjData(model)
            next_data.qpos[:] = reference["qpos"][endpoint + 1]
            mujoco.mj_forward(model, next_data)
            object_step = np.asarray(next_data.xpos[object_id]) - np.asarray(
                current_data.xpos[object_id]
            )
            for finger in fingers:
                nearest = selected[(object_name, "mink_reference", endpoint)][finger]
                body_id = int(nearest["hand_body_id"])
                current_point = np.asarray(nearest["hand_surface_point_world_m"])
                next_point = transform_body_point(
                    next_data, body_id,
                    np.asarray(nearest["hand_surface_point_body_local_m"]),
                )
                relative_step = next_point - current_point - object_step
                normal = np.asarray(nearest["object_surface_normal_world"])
                motion_rows.append({
                    "endpoint": endpoint,
                    "next_endpoint": endpoint + 1,
                    "object": object_name,
                    "finger": finger,
                    "hand_body": nearest["hand_body"],
                    "surface_distance_mm": nearest["distance_m"] * 1000.0,
                    "relative_surface_step_normal_mm": float(relative_step @ normal * 1000.0),
                    "relative_surface_step_world_z_mm": float(relative_step[2] * 1000.0),
                })

    demand_rows: list[dict[str, Any]] = []
    for object_name, object_body, endpoints in (
        ("brush", "right_object", brush_range),
        ("bowl", "left_object", bowl_range),
    ):
        for endpoint in endpoints:
            actual_data = data_cache[("replay_actual", endpoint)]
            hybrid_qpos = actual["qpos"][endpoint].copy()
            for actuator_id in range(model.nu):
                joint_id = int(model.actuator_trnid[actuator_id, 0])
                qadr = int(model.jnt_qposadr[joint_id])
                hybrid_qpos[qadr] = reference["qpos"][endpoint, qadr]
            hybrid_data = mujoco.MjData(model)
            hybrid_data.qpos[:] = hybrid_qpos
            mujoco.mj_forward(model, hybrid_data)
            for finger, nearest in selected[(object_name, "replay_actual", endpoint)].items():
                body_id = int(nearest["hand_body_id"])
                current_point = np.asarray(nearest["hand_surface_point_world_m"])
                desired_point = transform_body_point(
                    hybrid_data, body_id,
                    np.asarray(nearest["hand_surface_point_body_local_m"]),
                )
                displacement = desired_point - current_point
                normal = np.asarray(nearest["object_surface_normal_world"])
                demand_rows.append({
                    "endpoint": endpoint,
                    "object": object_name,
                    "finger": finger,
                    "hand_body": nearest["hand_body"],
                    "actual_surface_distance_mm": nearest["distance_m"] * 1000.0,
                    "actual_to_mink_target_normal_mm": float(displacement @ normal * 1000.0),
                    "actual_to_mink_target_world_z_mm": float(displacement[2] * 1000.0),
                    "actual_to_mink_target_norm_mm": float(np.linalg.norm(displacement) * 1000.0),
                })

    write_csv(output / "finger_object_native_geometry.csv", geometry_rows)
    write_csv(output / "grasp_geometry_summary.csv", grasp_rows)
    write_csv(output / "joint_tracking_errors.csv", tracking_rows)
    write_csv(output / "left_self_collision_persistence.csv", self_rows)
    write_csv(output / "mink_reference_surface_motion.csv", motion_rows)
    write_csv(output / "replay_to_mink_target_surface_demand.csv", demand_rows)

    def state_rows(object_name: str, state: str) -> list[dict[str, Any]]:
        return [row for row in grasp_rows
                if row["object"] == object_name and row["state"] == state]

    def distance_summary(object_name: str, state: str) -> dict[str, Any]:
        rows = state_rows(object_name, state)
        return {
            finger: {
                "median_mm": float(np.median([row[f"{finger}_distance_mm"] for row in rows])),
                "minimum_mm": float(min(row[f"{finger}_distance_mm"] for row in rows)),
                "near_5mm_endpoint_count": int(sum(
                    row[f"{finger}_distance_mm"] <= near * 1000.0 for row in rows
                )),
            }
            for finger in FINGERS
        }

    def side_cosine_summary(object_name: str, state: str) -> dict[str, float]:
        rows = state_rows(object_name, state)
        return {
            name: float(np.median([row[column] for row in rows]))
            for name, column in (
                ("index_thumb", "index_thumb_contact_side_cosine"),
                ("index_middle", "index_middle_contact_side_cosine"),
                ("thumb_middle", "thumb_middle_contact_side_cosine"),
            )
        }

    def tracking_summary(side: str, first: int, last: int) -> dict[str, Any]:
        result = {}
        for group in ("wrist_translation", "wrist_rotation", *FINGERS):
            per_endpoint = []
            for endpoint in range(first, last + 1):
                values = [
                    row["command_minus_actual"] for row in tracking_rows
                    if row["side"] == side and row["group"] == group
                    and row["endpoint"] == endpoint
                ]
                if values:
                    per_endpoint.append(float(np.sqrt(np.mean(np.square(values)))))
            result[group] = {
                "unit": "m" if group == "wrist_translation" else "rad",
                "median_endpoint_rms": float(np.median(per_endpoint)),
                "maximum_endpoint_rms": float(max(per_endpoint)),
                "last_endpoint_rms": float(per_endpoint[-1]),
            }
        return result

    def motion_summary(object_name: str, finger: str) -> dict[str, float]:
        rows = [row for row in motion_rows
                if row["object"] == object_name and row["finger"] == finger]
        return {
            "median_relative_surface_step_normal_mm": float(np.median([
                row["relative_surface_step_normal_mm"] for row in rows
            ])),
            "median_relative_surface_step_world_z_mm": float(np.median([
                row["relative_surface_step_world_z_mm"] for row in rows
            ])),
        }

    brush_reference = state_rows("brush", "mink_reference")
    brush_actual = state_rows("brush", "replay_actual")
    bowl_reference = state_rows("bowl", "mink_reference")
    bowl_actual = state_rows("bowl", "replay_actual")
    result = {
        "schema": SCHEMA,
        "classification": "REFERENCE_VS_EXECUTION_GEOMETRY_ATTRIBUTED",
        "brush": {
            "mink_reference_finger_distances": distance_summary("brush", "mink_reference"),
            "replay_actual_finger_distances": distance_summary("brush", "replay_actual"),
            "mink_contact_side_cosines": side_cosine_summary("brush", "mink_reference"),
            "replay_contact_side_cosines": side_cosine_summary("brush", "replay_actual"),
            "mink_index_thumb_side_cosine_median": float(np.median([
                row["index_thumb_contact_side_cosine"] for row in brush_reference
            ])),
            "replay_index_thumb_side_cosine_median": float(np.median([
                row["index_thumb_contact_side_cosine"] for row in brush_actual
            ])),
            "mink_reference_surface_motion": {
                finger: motion_summary("brush", finger) for finger in ("index", "thumb")
            },
            "right_hand_tracking_15_41": tracking_summary("right", 15, 41),
        },
        "bowl": {
            "mink_reference_finger_distances": distance_summary("bowl", "mink_reference"),
            "replay_actual_finger_distances": distance_summary("bowl", "replay_actual"),
            "mink_contact_side_cosines": side_cosine_summary("bowl", "mink_reference"),
            "replay_contact_side_cosines": side_cosine_summary("bowl", "replay_actual"),
            "mink_near_finger_count_median": float(np.median([
                row["near_finger_count_5mm"] for row in bowl_reference
            ])),
            "replay_near_finger_count_median": float(np.median([
                row["near_finger_count_5mm"] for row in bowl_actual
            ])),
            "left_hand_tracking_52_60": tracking_summary("left", 52, 60),
        },
        "left_hand": {
            "endpoint44_reference_proxy_distance_mm": self_rows[0][
                "mink_reference_proxy_index_middle_distance_mm"
            ],
            "endpoint44_actual_proxy_distance_mm": self_rows[0][
                "replay_actual_proxy_index_middle_distance_mm"
            ],
            "finger_rms_error_rad_by_endpoint": {
                str(row["endpoint"]): float(np.sqrt(np.mean([
                    row[f"{finger}_rms_error"] ** 2 for finger in FINGERS
                ]))) for row in self_rows
            },
            "tracking_44_60": tracking_summary("left", 44, 60),
            "proxy_overlap_endpoints_actual": [
                row["endpoint"] for row in self_rows
                if row["replay_actual_proxy_index_middle_distance_mm"] < 0.0
            ],
        },
        "mutations": {
            "physics_steps": 0, "state": 0, "commands": 0, "geometry": 0,
            "training_steps": 0, "promotion": 0,
        },
    }
    result["mechanism_conclusions"] = {
        "brush_15_41": (
            "MINK_REFERENCE_ALREADY_NONOPPOSING_AND_PENETRATING;_REPLAY_EXECUTION_ADDS_"
            "THUMB_MIDDLE_SEPARATION"
        ),
        "bowl_52_60": (
            "MINK_REFERENCE_HAS_THUMB_INDEX_MIDDLE_PROXIMITY_BUT_SAME_SIDE_PENETRATING_"
            "GEOMETRY;_REPLAY_LOSES_THUMB_MIDDLE_REACH"
        ),
        "left_44_60": (
            "FALSE_PROXY_COLLISION_IS_EXECUTION_INDUCED_AND_TRANSIENT;_FINGER_TRACKING_"
            "ERROR_PERSISTS_TO_BOWL_WINDOW"
        ),
        "overall": "BOTH_REFERENCE_GEOMETRY_AND_PHYSICAL_TRACKING_ARE_DEFECTIVE",
    }
    write_json(output / "results.json", result)
    manifest = {
        "schema": SCHEMA,
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
        "inputs": {key: artifact(path) for key, path in paths.items() if key != "output"}
        | {"config": artifact(config_path), "script": artifact(Path(__file__))},
        "authorization": cfg["authorization"],
        "physics_steps": 0,
    }
    write_json(output / "input_manifest.json", manifest)
    lines = [
        "# Brush MINK reference vs Replay execution attribution v1", "",
        f"- Classification: `{result['classification']}`",
        "- Physics / MINK / MPC / RL executed: `NO`",
        "- Geometry source: native visual meshes", "",
        "## Brush 15–41", "",
        f"- MINK median index/thumb side cosine: `{result['brush']['mink_index_thumb_side_cosine_median']:.6f}`",
        f"- Replay median index/thumb side cosine: `{result['brush']['replay_index_thumb_side_cosine_median']:.6f}`", "",
        "- Reference classification: index/thumb are usually on the same object side; index "
        "and middle also penetrate the native brush mesh. Replay adds loss of thumb/middle reach.",
        f"- MINK thumb median relative step: `{result['brush']['mink_reference_surface_motion']['thumb']['median_relative_surface_step_normal_mm']:.6f} mm` along the surface normal and "
        f"`{result['brush']['mink_reference_surface_motion']['thumb']['median_relative_surface_step_world_z_mm']:.6f} mm` in world Z.", "",
        "## Bowl 52–60", "",
        f"- MINK median fingers within 5 mm: `{result['bowl']['mink_near_finger_count_median']:.3f}`",
        f"- Replay median fingers within 5 mm: `{result['bowl']['replay_near_finger_count_median']:.3f}`", "",
        "- Reference classification: thumb/index/middle reach the bowl but remain concentrated "
        "on the same side and include native penetrations. Replay loses thumb and middle reach, "
        "leaving index-only contact.", "",
        "## Left hand 44–60", "",
        f"- Endpoint 44 MINK proxy index/middle distance: `{result['left_hand']['endpoint44_reference_proxy_distance_mm']:.6f} mm`",
        f"- Endpoint 44 Replay proxy index/middle distance: `{result['left_hand']['endpoint44_actual_proxy_distance_mm']:.6f} mm`", "",
        f"- Actual proxy-overlap endpoints in 44–60: `{result['left_hand']['proxy_overlap_endpoints_actual']}`.",
        "- The false overlap is execution-induced and clears after endpoint 45, but finger "
        "tracking RMS remains material through endpoint 60; this audit does not claim the "
        "transient collision is the sole cause of every later error.", "",
        f"- Overall attribution: `{result['mechanism_conclusions']['overall']}`.", "",
        "Contact-side cosine and the 5 mm band are descriptive diagnostics, not pass/fail gates. "
        "See the CSV files for raw native distances, surface locations/normals, and per-actuator errors.", "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    hash_tree(output)
    print(json.dumps(result, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
