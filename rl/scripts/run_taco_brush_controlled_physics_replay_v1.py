#!/usr/bin/env python3
"""Bounded, fail-closed CPU physics Replay for the Brush sample.

This diagnostic never runs MINK, MPC, or RL.  It replays the frozen MINK-v2
position targets from endpoint 0 through endpoint 80 in the unmodified source
MuJoCo dynamics, except for an experiment-local copy of the already audited
Issue #14 floor height.  Runtime proxy contacts/forces are aligned with sampled
native visual-mesh distances; the latter never participate in simulation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_world_objects,
    pair_group_audit,
    table_audit,
)
from egoengine_repro.evaluation.controlled_physics_replay import (  # noqa: E402
    control_substep_schedule,
    longest_true_run,
    proxy_native_geometry_relation,
    proxy_native_mismatch,
    quaternion_distance_rad,
)
from egoengine_repro.retarget.collision_audit import collision_families  # noqa: E402
from egoengine_repro.scene.support_surface import Plane  # noqa: E402
from run_taco_brush_issue14_mink_candidate_v1 import relevant_pairs  # noqa: E402


SCHEMA = "taco_brush_controlled_physics_replay_v1"
SUPPORTED_SCHEMAS = {
    SCHEMA,
    "taco_brush_20231006_145_initialization_controlled_replay_v1",
}
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_controlled_physics_replay_v1.yaml"
CONTACT_GROUPS = (
    "right_hand_brush", "left_hand_bowl", "cross_hand_object",
    "brush_floor", "bowl_floor", "hand_floor", "brush_bowl",
    "intrahand", "interhand", "other",
)


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


def build_local_scene(source: Path, destination: Path, floor_z_m: float) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("source scene has no compiler")
    for key in ("meshdir", "texturedir"):
        raw = compiler.attrib.get(key)
        if raw:
            compiler.set(key, str((source.parent / raw).resolve(strict=True)))
    floor = root.find("worldbody/geom[@name='floor']")
    if floor is None:
        raise ValueError("source scene has no floor geom")
    position = np.fromstring(floor.attrib.get("pos", "0 0 0"), sep=" ")
    if position.shape != (3,):
        raise ValueError("floor position is malformed")
    position[2] = floor_z_m
    floor.set("pos", " ".join(f"{value:.17g}" for value in position))
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def is_descendant(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == int(root_id):
            return True
        current = int(model.body_parentid[current])
    return False


def geom_role(model: mujoco.MjModel, geom_id: int) -> str:
    if int(geom_id) == int(model.geom("floor").id):
        return "floor"
    body = int(model.geom_bodyid[geom_id])
    if body == int(model.body("right_object").id):
        return "brush"
    if body == int(model.body("left_object").id):
        return "bowl"
    if is_descendant(model, body, int(model.body("R_forearm_ty_link").id)):
        return "right_hand"
    if is_descendant(model, body, int(model.body("L_forearm_ty_link").id)):
        return "left_hand"
    return "other"


def contact_group(first: str, second: str) -> str:
    roles = {first, second}
    if roles == {"right_hand", "brush"}:
        return "right_hand_brush"
    if roles == {"left_hand", "bowl"}:
        return "left_hand_bowl"
    if (("right_hand" in roles or "left_hand" in roles)
            and ("brush" in roles or "bowl" in roles)):
        return "cross_hand_object"
    if roles == {"brush", "floor"}:
        return "brush_floor"
    if roles == {"bowl", "floor"}:
        return "bowl_floor"
    if "floor" in roles and ("right_hand" in roles or "left_hand" in roles):
        return "hand_floor"
    if roles == {"brush", "bowl"}:
        return "brush_bowl"
    if first == second and first in {"right_hand", "left_hand"}:
        return "intrahand"
    if roles == {"right_hand", "left_hand"}:
        return "interhand"
    return "other"


def empty_interval() -> dict[str, dict[str, Any]]:
    return {
        group: {
            "normal_impulse_ns": 0.0,
            "max_force_n": 0.0,
            "max_proxy_penetration_m": 0.0,
            "contact_substeps": 0,
            "max_force_pair": None,
        }
        for group in CONTACT_GROUPS
    }


def sample_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, dt: float,
    interval: dict[str, dict[str, Any]], *, source: int, substep: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    maxima = {"force_n": 0.0, "proxy_penetration_m": 0.0}
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        role1, role2 = geom_role(model, first), geom_role(model, second)
        group = contact_group(role1, role2)
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal = max(0.0, float(wrench[0]))
        force = float(np.linalg.norm(wrench[:3]))
        penetration = max(0.0, -float(contact.dist))
        record = interval[group]
        record["normal_impulse_ns"] += normal * dt
        if force > record["max_force_n"]:
            record["max_force_n"] = force
            record["max_force_pair"] = [model.geom(first).name, model.geom(second).name]
        record["max_proxy_penetration_m"] = max(
            record["max_proxy_penetration_m"], penetration,
        )
        seen_groups.add(group)
        maxima["force_n"] = max(maxima["force_n"], force)
        maxima["proxy_penetration_m"] = max(
            maxima["proxy_penetration_m"], penetration,
        )
        rows.append({
            "source_endpoint": source,
            "outcome_endpoint": source + 1,
            "substep": substep,
            "time_s": float(data.time),
            "contact_index": index,
            "group": group,
            "geom1": model.geom(first).name,
            "geom2": model.geom(second).name,
            "distance_m": float(contact.dist),
            "normal_force_n": normal,
            "force_norm_n": force,
            "efc_address": int(contact.efc_address),
        })
    for group in seen_groups:
        interval[group]["contact_substeps"] += 1
    return rows, maxima


def object_state(
    model: mujoco.MjModel, data: mujoco.MjData, joint_name: str,
) -> dict[str, Any]:
    joint = model.joint(joint_name)
    qadr, dadr = int(joint.qposadr[0]), int(joint.dofadr[0])
    return {
        "position_m": np.asarray(data.qpos[qadr:qadr + 3]).copy(),
        "quaternion_wxyz": np.asarray(data.qpos[qadr + 3:qadr + 7]).copy(),
        "linear_velocity_m_s": np.asarray(data.qvel[dadr:dadr + 3]).copy(),
        "angular_velocity_rad_s": np.asarray(data.qvel[dadr + 3:dadr + 6]).copy(),
    }


def native_minimum_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "body1": row.get("body1"), "body2": row.get("body2"),
        "native_visual_geom1": row.get("native_visual_geom1"),
        "native_visual_geom2": row.get("native_visual_geom2"),
        "classification": row.get("classification"),
        "distance_m": row.get("distance_m"),
        "reported_penetration_depth_m": row.get("reported_penetration_depth_m"),
        "deepest_contact_position_sim_m": row.get("deepest_contact_position_sim_m"),
        "nearest_points_sim_m": row.get("nearest_points_sim_m"),
    }


def native_audit(
    model: mujoco.MjModel, data: mujoco.MjData, plane: Plane,
    specs: list[dict[str, Any]], tolerance: float,
) -> dict[str, Any]:
    native = native_world_objects(model, data)
    families = collision_families(model)
    interactions = {}
    for spec in specs:
        family_name = "hand_tool" if spec["role"] == "tool" else "hand_target"
        pairs = relevant_pairs(
            model, families[family_name], spec["hand"], spec["object_side"],
        )
        audited = pair_group_audit(model, data, native, pairs, tolerance)
        minimum = native_minimum_summary(audited["native_minimum"])
        interactions[spec["object"]] = {
            "native_minimum": minimum,
            "proxy_minimum": audited["shell_minimum"],
            "native_penetrating_pair_count": len(audited["native_penetrating_pairs"]),
            "unknown_pair_count": len(audited["unknown_pairs"]),
        }
    table = table_audit(native, plane, tolerance)
    return {
        "interactions": interactions,
        "table": {
            name: {
                "minimum_signed_distance_m": row["minimum_signed_distance_m"],
                "classification": row["classification"],
                "worst_location": row["worst_location"],
            }
            for name, row in table["entities"].items()
        },
    }


def controls_within_range(model: mujoco.MjModel, controls: np.ndarray) -> bool:
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    return bool(
        np.all(controls[:, limited] >= low[limited] - 1.0e-12)
        and np.all(controls[:, limited] <= high[limited] + 1.0e-12)
    )


def warning_counts(data: mujoco.MjData) -> list[int]:
    return [int(row.number) for row in data.warning]


def hash_tree(output: Path) -> None:
    target = output / "server_artifacts.sha256"
    rows = []
    for path in sorted(value for value in output.rglob("*") if value.is_file()
                       and value != target):
        rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def settled_object_qpos(
    *, csv_path: Path, variants: dict[str, str], transform: np.ndarray,
    expected_final_time_s: float,
) -> dict[str, np.ndarray]:
    """Load the final natural-seating poses and map world coordinates to sim."""
    selected: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for name, variant in variants.items():
                if row["object_variant"] == f"{name}__{variant}":
                    selected[name] = row
    missing = sorted(set(variants) - set(selected))
    if missing:
        raise ValueError(f"missing settled object rows: {missing}")
    if any(not np.isclose(float(row["time_s"]), expected_final_time_s,
                          rtol=0.0, atol=1.0e-9)
           for row in selected.values()):
        raise ValueError("natural-seating timeseries does not end at frozen time")
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    transform_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(transform_quat, rotation.reshape(-1))
    result = {}
    for name, row in selected.items():
        world_position = np.asarray([
            row["position_x_m"], row["position_y_m"], row["position_z_m"],
        ], dtype=np.float64)
        world_quaternion = np.asarray([
            row["quaternion_w"], row["quaternion_x"],
            row["quaternion_y"], row["quaternion_z"],
        ], dtype=np.float64)
        simulator_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(simulator_quaternion, transform_quat, world_quaternion)
        simulator_quaternion /= np.linalg.norm(simulator_quaternion)
        result[name] = np.concatenate((rotation @ world_position + translation,
                                       simulator_quaternion))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError("unexpected controlled Replay schema")
    forbidden = (
        "modify_source_scene", "modify_reference", "modify_active_support_contract",
        "modify_collision_geometry", "mink", "full_209_frame_replay", "mpc",
        "reinforcement_learning", "promotion", "chunk_commit",
    )
    if any(cfg["authorization"].get(key) is not False for key in forbidden):
        raise ValueError("authorization is not fail-closed")
    source_endpoint = int(cfg["sample"]["replay_source_endpoint"])
    outcome_endpoint = int(cfg["sample"]["replay_outcome_endpoint"])
    if source_endpoint != 0 or outcome_endpoint <= source_endpoint:
        raise ValueError("controlled Replay requires a positive endpoint-0 window")

    paths = {
        key: Path(value).resolve(strict=(key != "output"))
        for key, value in cfg["paths"].items()
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    local_scene = output / "experiment_local_scene.xml"
    build_local_scene(paths["scene"], local_scene, float(cfg["support"]["simulator_z_m"]))
    model = mujoco.MjModel.from_xml_path(str(local_scene))
    if not np.isclose(model.geom("floor").pos[2], cfg["support"]["simulator_z_m"],
                      rtol=0.0, atol=0.0):
        raise ValueError("experiment-local floor height did not compile exactly")

    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        reference = {name: np.asarray(archive[name]).copy()
                     for name in ("qpos", "qvel", "ctrl")}
        frequency = float(np.asarray(archive["frequency"]))
    reference_frames = int(cfg["sample"]["reference_frames"])
    if reference["qpos"].shape != (reference_frames, model.nq) \
            or reference["qvel"].shape != (reference_frames, model.nv) \
            or reference["ctrl"].shape != (reference_frames, model.nu):
        raise ValueError("reference dimensions changed")
    if frequency != float(cfg["sample"]["control_hz"]):
        raise ValueError("reference frequency changed")
    controls = reference["ctrl"][:outcome_endpoint + 1]
    if not controls_within_range(model, controls):
        raise ValueError("Replay reference contains out-of-range controls")

    schedule = control_substep_schedule(
        control_endpoints=outcome_endpoint, control_hz=frequency,
        simulation_dt_s=float(model.opt.timestep),
    )
    if int(schedule.sum()) > int(cfg["hard_stop"]["maximum_physics_steps"]):
        raise ValueError("frozen physics budget is insufficient")
    support_cfg = yaml.safe_load(paths["support_contract"].read_text(encoding="utf-8"))
    simulator = support_cfg["simulator"]
    plane = Plane(
        normal=np.asarray(simulator["normal"], dtype=np.float64),
        offset=float(simulator["offset_m"]), frame=str(simulator["frame"]),
    )
    if plane.offset != float(cfg["support"]["simulator_z_m"]):
        raise ValueError("support contract and Replay contract disagree")

    manifest = {
        "schema": cfg["schema"],
        "status": "FROZEN_BEFORE_PHYSICS",
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
        "inputs": {
            "config": artifact(config_path), "source_scene": artifact(paths["scene"]),
            "local_scene": artifact(local_scene),
            "robot_reference": artifact(paths["robot_reference"]),
            "support_contract": artifact(paths["support_contract"]),
            "prior_collision_audit": artifact(paths["prior_collision_audit"]),
            "script": artifact(Path(__file__)),
        },
        "physics": {
            "backend": "mujoco_cpu", "mujoco_version": mujoco.__version__,
            "authored_timestep_s": float(model.opt.timestep),
            "control_hz": frequency, "substeps_per_interval": schedule.tolist(),
            "total_planned_physics_steps": int(schedule.sum()),
            "initial_qvel": "ZERO", "command_timing": "ctrl[outcome_endpoint]",
        },
        "hard_stop": cfg["hard_stop"],
        "strict_pickup_gate": cfg["strict_pickup_gate"],
        "authorization": cfg["authorization"],
    }
    write_json(output / "input_manifest.json", manifest)

    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"][0]
    initial_state_provenance: dict[str, Any] = {
        "robot": "robot_reference_endpoint_0",
        "objects": "robot_reference_endpoint_0",
        "object_velocity": "ZERO",
    }
    if "initial_state" in cfg:
        initialization = cfg["initial_state"]
        robot_results_path = Path(initialization["robot_results"]).resolve(strict=True)
        transform_report = json.loads(robot_results_path.read_text(encoding="utf-8"))
        candidate_name = str(initialization["robot_candidate"])
        candidate = transform_report["mink_candidates"][candidate_name]
        if candidate["status"] != "MINK_PASS":
            raise ValueError("selected robot initialization did not pass MINK gate")
        if candidate["robot_reference"]["sha256"] != sha256(paths["robot_reference"]):
            raise ValueError("robot reference does not match selected passing candidate")
        seating_results_path = Path(
            initialization["natural_seating_results"]
        ).resolve(strict=True)
        seating_results = json.loads(seating_results_path.read_text(encoding="utf-8"))
        if not seating_results["decision"]["object_settling_passed_all_variants"]:
            raise ValueError("natural-seating gate did not pass all frozen variants")
        transform = np.asarray(transform_report["T_sim_world"], dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("initial-state transform must be 4x4")
        settled = settled_object_qpos(
            csv_path=Path(initialization["natural_seating_timeseries"]).resolve(strict=True),
            variants=dict(initialization["object_variants"]), transform=transform,
            expected_final_time_s=float(initialization["expected_final_time_s"]),
        )
        for name, joint_name in (("brush", "right_object_joint"),
                                 ("bowl", "left_object_joint")):
            joint = model.joint(joint_name)
            qadr = int(joint.qposadr[0])
            data.qpos[qadr:qadr + 7] = settled[name]
        initial_state_provenance = {
            "robot": initialization["robot"],
            "objects": initialization["objects"],
            "object_velocity": "ZERO",
            "natural_seating_timeseries": artifact(
                Path(initialization["natural_seating_timeseries"])
            ),
            "natural_seating_results": artifact(seating_results_path),
            "robot_results": artifact(robot_results_path),
            "robot_candidate": candidate_name,
            "object_variants": initialization["object_variants"],
            "settled_object_qpos_simulator": settled,
        }
    data.qvel[:] = 0.0
    data.ctrl[:] = reference["ctrl"][0]
    mujoco.mj_forward(model, data)
    initial_warning = warning_counts(data)
    if any(initial_warning):
        raise RuntimeError(f"MuJoCo warning during initialization: {initial_warning}")
    manifest["initial_state_provenance"] = initial_state_provenance
    write_json(output / "input_manifest.json", manifest)

    object_specs = {
        "brush": {"joint": "right_object_joint", "contact": "right_hand_brush",
                  "floor": "brush_floor"},
        "bowl": {"joint": "left_object_joint", "contact": "left_hand_bowl",
                 "floor": "bowl_floor"},
    }
    initial_objects = {
        name: object_state(model, data, spec["joint"])
        for name, spec in object_specs.items()
    }
    audit_endpoints = set(int(value) for value in cfg["native_geometry_audit"]["endpoints"])
    native_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    hard_stop: dict[str, Any] | None = None

    def append_endpoint(endpoint: int, interval: dict[str, dict[str, Any]]) -> None:
        row: dict[str, Any] = {"endpoint": endpoint, "time_s": float(data.time)}
        for name, spec in object_specs.items():
            state = object_state(model, data, spec["joint"])
            joint = model.joint(spec["joint"])
            qadr = int(joint.qposadr[0])
            ref_pos = reference["qpos"][endpoint, qadr:qadr + 3]
            ref_quat = reference["qpos"][endpoint, qadr + 3:qadr + 7]
            row.update({
                f"{name}_position_m": state["position_m"],
                f"{name}_quaternion_wxyz": state["quaternion_wxyz"],
                f"{name}_linear_velocity_m_s": state["linear_velocity_m_s"],
                f"{name}_angular_velocity_rad_s": state["angular_velocity_rad_s"],
                f"{name}_actual_lift_m": float(
                    state["position_m"][2] - initial_objects[name]["position_m"][2]
                ),
                f"{name}_reference_lift_m": float(
                    ref_pos[2] - reference["qpos"][0, qadr + 2]
                ),
                f"{name}_position_error_m": float(np.linalg.norm(state["position_m"] - ref_pos)),
                f"{name}_rotation_error_rad": quaternion_distance_rad(
                    state["quaternion_wxyz"], ref_quat,
                ),
                f"{name}_hand_normal_impulse_ns": interval[spec["contact"]]["normal_impulse_ns"],
                f"{name}_floor_normal_impulse_ns": interval[spec["floor"]]["normal_impulse_ns"],
                f"{name}_hand_max_force_n": interval[spec["contact"]]["max_force_n"],
                f"{name}_floor_max_force_n": interval[spec["floor"]]["max_force_n"],
                f"{name}_hand_contact_substeps": interval[spec["contact"]]["contact_substeps"],
                f"{name}_floor_contact_substeps": interval[spec["floor"]]["contact_substeps"],
            })
            if "strict_tracking_gate" in cfg:
                position_scale = float(cfg["strict_tracking_gate"]["position_scale_m"])
                rotation_scale = float(cfg["strict_tracking_gate"]["rotation_scale_rad"])
                row[f"{name}_tracking_score"] = float(np.sqrt(
                    (row[f"{name}_position_error_m"] / position_scale) ** 2
                    + (row[f"{name}_rotation_error_rad"] / rotation_scale) ** 2
                ))
        row["interval_groups"] = copy.deepcopy(interval)
        endpoint_rows.append(row)
        if endpoint in audit_endpoints:
            audited = native_audit(
                model, data, plane,
                list(cfg["native_geometry_audit"]["interaction_pairs"]),
                float(cfg["native_geometry_audit"]["hand_object_and_hand_table_tolerance_m"]),
            )
            native_row = {"endpoint": endpoint, **audited}
            for name, spec in object_specs.items():
                minimum = audited["interactions"][name]["native_minimum"]
                if minimum is None or minimum["distance_m"] is None:
                    mismatch = "UNKNOWN_NATIVE_DISTANCE"
                else:
                    mismatch = proxy_native_mismatch(
                        proxy_contact=interval[spec["contact"]]["normal_impulse_ns"] > 0.0,
                        native_distance_m=float(minimum["distance_m"]),
                        tolerance_m=float(
                            cfg["native_geometry_audit"]["hand_object_and_hand_table_tolerance_m"]
                        ),
                    )
                proxy_minimum = native_row["interactions"][name]["proxy_minimum"]
                if minimum is None or minimum["distance_m"] is None or proxy_minimum is None:
                    geometry = "UNKNOWN_GEOMETRY_DISTANCE"
                else:
                    geometry = proxy_native_geometry_relation(
                        proxy_distance_m=float(proxy_minimum["distance_m"]),
                        native_distance_m=float(minimum["distance_m"]),
                        tolerance_m=float(
                            cfg["native_geometry_audit"]["hand_object_and_hand_table_tolerance_m"]
                        ),
                    )
                native_row["interactions"][name]["force_native_exposure_classification"] = mismatch
                native_row["interactions"][name]["proxy_native_geometry_relation"] = geometry
                native_row["interactions"][name]["preceding_interval_normal_impulse_ns"] = (
                    interval[spec["contact"]]["normal_impulse_ns"]
                )
                native_row["interactions"][name]["preceding_interval_object_displacement_m"] = (
                    0.0 if endpoint == 0 else float(np.linalg.norm(
                        row[f"{name}_position_m"]
                        - endpoint_rows[-2][f"{name}_position_m"]
                    ))
                )
            native_rows.append(native_row)

    append_endpoint(0, empty_interval())
    previous_objects = copy.deepcopy(initial_objects)
    physics_steps = 0
    limits = cfg["hard_stop"]
    for source, substeps in enumerate(schedule):
        outcome = source + 1
        interval = empty_interval()
        data.ctrl[:] = reference["ctrl"][outcome]
        for substep in range(int(substeps)):
            mujoco.mj_step(model, data)
            physics_steps += 1
            rows, maxima = sample_contacts(
                model, data, float(model.opt.timestep), interval,
                source=source, substep=substep,
            )
            contact_rows.extend(rows)
            states = {
                name: object_state(model, data, spec["joint"])
                for name, spec in object_specs.items()
            }
            failures = []
            arrays = [data.qpos, data.qvel, data.ctrl, data.qacc]
            if not all(np.isfinite(value).all() for value in arrays):
                failures.append("NONFINITE_STATE")
            warnings = warning_counts(data)
            if limits["fail_on_any_mujoco_warning"] and any(warnings):
                failures.append("MUJOCO_WARNING")
            if maxima["force_n"] > float(limits["maximum_contact_force_n"]):
                failures.append("CONTACT_FORCE_LIMIT")
            if maxima["proxy_penetration_m"] > float(limits["maximum_proxy_penetration_m"]):
                failures.append("PROXY_PENETRATION_LIMIT")
            for name, state in states.items():
                linear_speed = float(np.linalg.norm(state["linear_velocity_m_s"]))
                angular_speed = float(np.linalg.norm(state["angular_velocity_rad_s"]))
                translation = float(np.linalg.norm(
                    state["position_m"] - previous_objects[name]["position_m"]
                ))
                rotation = quaternion_distance_rad(
                    state["quaternion_wxyz"], previous_objects[name]["quaternion_wxyz"],
                )
                if linear_speed > float(limits["maximum_object_linear_speed_m_s"]):
                    failures.append(f"{name.upper()}_LINEAR_SPEED_LIMIT")
                if angular_speed > float(limits["maximum_object_angular_speed_rad_s"]):
                    failures.append(f"{name.upper()}_ANGULAR_SPEED_LIMIT")
                if translation > float(limits["maximum_object_translation_per_substep_m"]):
                    failures.append(f"{name.upper()}_SUBSTEP_TRANSLATION_LIMIT")
                if rotation > float(limits["maximum_object_rotation_per_substep_rad"]):
                    failures.append(f"{name.upper()}_SUBSTEP_ROTATION_LIMIT")
            previous_objects = states
            if failures:
                hard_stop = {
                    "reasons": sorted(set(failures)), "source_endpoint": source,
                    "planned_outcome_endpoint": outcome, "substep": substep,
                    "physics_step": physics_steps, "time_s": float(data.time),
                    "warning_counts": warnings, "maximum_contact_force_n": maxima["force_n"],
                    "maximum_proxy_penetration_m": maxima["proxy_penetration_m"],
                    "object_states": states,
                }
                break
        if hard_stop is not None:
            break
        append_endpoint(outcome, interval)

    # Flatten endpoint records for portable CSV/NPZ evidence.
    scalar_fields = [
        "endpoint", "time_s",
        *[f"{name}_{suffix}" for name in object_specs for suffix in (
            "actual_lift_m", "reference_lift_m", "position_error_m", "rotation_error_rad",
            "hand_normal_impulse_ns", "floor_normal_impulse_ns", "hand_max_force_n",
            "floor_max_force_n", "hand_contact_substeps", "floor_contact_substeps",
        )],
    ]
    if "strict_tracking_gate" in cfg:
        scalar_fields.extend(
            f"{name}_tracking_score" for name in object_specs
        )
    with (output / "endpoint_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        for row in endpoint_rows:
            writer.writerow({key: row[key] for key in scalar_fields})
    with gzip.open(output / "contact_trace.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        fields = [
            "source_endpoint", "outcome_endpoint", "substep", "time_s", "contact_index",
            "group", "geom1", "geom2", "distance_m", "normal_force_n", "force_norm_n",
            "efc_address",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(contact_rows)
    np.savez_compressed(
        output / "endpoint_states.npz",
        endpoint=np.asarray([row["endpoint"] for row in endpoint_rows], dtype=np.int64),
        **{
            f"{name}_{field}": np.asarray([row[f"{name}_{field}"] for row in endpoint_rows])
            for name in object_specs
            for field in (
                "position_m", "quaternion_wxyz", "linear_velocity_m_s",
                "angular_velocity_rad_s",
            )
        },
    )
    write_json(output / "native_geometry_trace.json", native_rows)
    if hard_stop is not None:
        write_json(output / "hard_stop.json", hard_stop)

    pickup = {}
    stable_needed = int(cfg["strict_pickup_gate"]["stable_consecutive_control_intervals"])
    lift_needed = float(cfg["strict_pickup_gate"]["object_lift_delta_m"])
    for name, spec in object_specs.items():
        masks = [
            row[f"{name}_actual_lift_m"] >= lift_needed
            and row[f"{name}_hand_normal_impulse_ns"] > 0.0
            and row[f"{name}_floor_normal_impulse_ns"] == 0.0
            for row in endpoint_rows[1:]
        ]
        run = longest_true_run(masks)
        pickup[name] = {
            "maximum_actual_lift_m": max(row[f"{name}_actual_lift_m"] for row in endpoint_rows),
            "maximum_reference_lift_m": max(row[f"{name}_reference_lift_m"] for row in endpoint_rows),
            "lifted_hand_contact_no_floor_longest_run": run,
            "stable_pickup_pass": run >= stable_needed,
        }

    force_exposure_mismatches = []
    geometry_mismatches = []
    native_blockers = []
    total_impulse = {"brush": 0.0, "bowl": 0.0}
    mismatch_impulse = {"brush": 0.0, "bowl": 0.0}
    mismatch_displacement = {"brush": 0.0, "bowl": 0.0}
    for audited in native_rows:
        endpoint = int(audited["endpoint"])
        for name in object_specs:
            interaction = audited["interactions"][name]
            impulse = float(interaction["preceding_interval_normal_impulse_ns"])
            total_impulse[name] += impulse
            mismatch = interaction["force_native_exposure_classification"]
            if not mismatch.startswith("AGREEMENT_"):
                mismatch_row = {
                    "endpoint": endpoint, "object": name, "classification": mismatch,
                    "normal_impulse_ns": impulse,
                    "object_displacement_m": interaction["preceding_interval_object_displacement_m"],
                    "native_distance_m": (
                        None if interaction["native_minimum"] is None
                        else interaction["native_minimum"]["distance_m"]
                    ),
                    "proxy_distance_m": (
                        None if interaction["proxy_minimum"] is None
                        else interaction["proxy_minimum"]["distance_m"]
                    ),
                }
                force_exposure_mismatches.append(mismatch_row)
                mismatch_impulse[name] += impulse
                mismatch_displacement[name] += float(
                    interaction["preceding_interval_object_displacement_m"]
                )
            geometry = interaction["proxy_native_geometry_relation"]
            if geometry.startswith("PROXY_") or geometry.startswith("UNKNOWN_"):
                geometry_mismatches.append({
                    "endpoint": endpoint, "object": name, "classification": geometry,
                    "native_distance_m": (
                        None if interaction["native_minimum"] is None
                        else interaction["native_minimum"]["distance_m"]
                    ),
                    "proxy_distance_m": (
                        None if interaction["proxy_minimum"] is None
                        else interaction["proxy_minimum"]["distance_m"]
                    ),
                    "preceding_interval_normal_impulse_ns": impulse,
                    "preceding_interval_object_displacement_m": (
                        interaction["preceding_interval_object_displacement_m"]
                    ),
                })
            minimum = interaction["native_minimum"]
            if minimum is None or minimum["classification"] in {
                "PENETRATION", "UNKNOWN_INTERSECTION_DEPTH", "UNKNOWN_CONTAINMENT_OPEN_MESH",
            }:
                native_blockers.append({
                    "endpoint": endpoint, "entity": f"hand_{name}",
                    "classification": None if minimum is None else minimum["classification"],
                })
        for entity, table_row in audited["table"].items():
            limit = (
                float(cfg["native_geometry_audit"]["dynamic_object_floor_penetration_limit_m"])
                if entity in {"brush", "bowl"}
                else float(
                    cfg["native_geometry_audit"]["hand_object_and_hand_table_tolerance_m"]
                )
            )
            if float(table_row["minimum_signed_distance_m"]) < -limit:
                native_blockers.append({
                    "endpoint": endpoint, "entity": f"{entity}_table",
                    "classification": table_row["classification"],
                    "distance_m": table_row["minimum_signed_distance_m"],
                    "limit_m": limit,
                })

    tracking_failures = []
    if "strict_tracking_gate" in cfg:
        for row in endpoint_rows:
            for name in object_specs:
                score = float(row[f"{name}_tracking_score"])
                if score > float(cfg["strict_tracking_gate"]["maximum_score"]):
                    tracking_failures.append({
                        "endpoint": int(row["endpoint"]), "object": name,
                        "score": score,
                        "position_error_m": float(row[f"{name}_position_error_m"]),
                        "rotation_error_rad": float(row[f"{name}_rotation_error_rad"]),
                    })

    strict_pass = bool(
        hard_stop is None
        and all(row["stable_pickup_pass"] for row in pickup.values())
        and not native_blockers
        and not force_exposure_mismatches
        and not geometry_mismatches
        and not tracking_failures
        and endpoint_rows[-1]["endpoint"] == outcome_endpoint
    )
    if hard_stop is not None:
        classification = "CONTROLLED_REPLAY_HARD_STOP"
    elif strict_pass:
        classification = "CONTROLLED_REPLAY_STRICT_PASS_NO_PROMOTION"
    else:
        classification = "CONTROLLED_REPLAY_FAILED_CLOSED"

    proxy_impact = {
        name: {
            "sampled_hand_object_normal_impulse_ns": total_impulse[name],
            "mismatch_normal_impulse_ns": mismatch_impulse[name],
            "mismatch_impulse_fraction": (
                mismatch_impulse[name] / total_impulse[name] if total_impulse[name] else None
            ),
            "sum_object_displacement_during_mismatch_intervals_m": mismatch_displacement[name],
            "causal_interpretation": (
                "DYNAMIC_EXPOSURE_NOT_A_TRUE_GEOMETRY_COUNTERFACTUAL"
            ),
        }
        for name in object_specs
    }
    result = {
        "schema": cfg["schema"], "classification": classification,
        "strict_gate_passed": strict_pass, "promotion_authorized": False,
        "chunk_commit_allowed": False, "reinforcement_learning_runs": 0,
        "completed_outcome_endpoint": int(endpoint_rows[-1]["endpoint"]),
        "physics_steps": physics_steps, "hard_stop": hard_stop,
        "pickup": pickup, "native_blockers": native_blockers,
        "tracking_failures": tracking_failures,
        "proxy_native_force_exposure_mismatches": force_exposure_mismatches,
        "proxy_native_geometry_mismatches": geometry_mismatches,
        "proxy_error_dynamic_exposure": proxy_impact,
        "maximums": {
            "contact_force_n": max(
                (row["force_norm_n"] for row in contact_rows), default=0.0,
            ),
            "proxy_penetration_m": max(
                (max(0.0, -row["distance_m"]) for row in contact_rows), default=0.0,
            ),
            **{
                f"{name}_position_error_m": max(
                    row[f"{name}_position_error_m"] for row in endpoint_rows
                ) for name in object_specs
            },
        },
        "source_artifacts": manifest["inputs"],
        "repository_head": manifest["repository_head"],
    }
    write_json(output / "results.json", result)

    endpoints = np.asarray([row["endpoint"] for row in endpoint_rows])
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for name, color in (("brush", "tab:blue"), ("bowl", "tab:orange")):
        axes[0].plot(endpoints, [row[f"{name}_actual_lift_m"] * 1000 for row in endpoint_rows],
                     color=color, label=f"{name} actual")
        axes[0].plot(endpoints, [row[f"{name}_reference_lift_m"] * 1000 for row in endpoint_rows],
                     color=color, linestyle="--", label=f"{name} reference")
        axes[1].plot(endpoints, [row[f"{name}_position_error_m"] * 1000 for row in endpoint_rows],
                     color=color, label=name)
        axes[2].plot(endpoints, [row[f"{name}_hand_normal_impulse_ns"] for row in endpoint_rows],
                     color=color, label=f"{name} hand")
        axes[2].plot(endpoints, [row[f"{name}_floor_normal_impulse_ns"] for row in endpoint_rows],
                     color=color, linestyle=":", label=f"{name} floor")
    axes[0].axhline(lift_needed * 1000, color="black", linewidth=1, linestyle=":")
    axes[0].set_ylabel("body-z lift (mm)"); axes[0].legend(ncol=2)
    axes[1].set_ylabel("reference position error (mm)"); axes[1].legend()
    axes[2].set_ylabel("interval normal impulse (N s)"); axes[2].set_xlabel("endpoint")
    axes[2].legend(ncol=2)
    figure.savefig(output / "replay_dynamics.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Brush controlled physics Replay v1", "",
        f"- Classification: `{classification}`",
        f"- Strict gate: `{'PASS' if strict_pass else 'FAIL'}`",
        f"- Completed: endpoint `0→{endpoint_rows[-1]['endpoint']}` / planned `0→{outcome_endpoint}`",
        f"- CPU MuJoCo physics steps: `{physics_steps}`",
        f"- Hard stop: `{hard_stop['reasons'] if hard_stop else 'NONE'}`",
        "- RL / MPC / MINK: `NOT RUN`",
        "- Promotion / chunk commit: `FORBIDDEN`", "",
        "## Physical pickup gate", "",
        "| object | max actual/reference lift (mm) | longest lifted + hand-force + no-floor run | result |",
        "|---|---:|---:|---|",
    ]
    for name, row in pickup.items():
        lines.append(
            f"| {name} | {row['maximum_actual_lift_m']*1000:.3f} / "
            f"{row['maximum_reference_lift_m']*1000:.3f} | "
            f"{row['lifted_hand_contact_no_floor_longest_run']} | "
            f"{'PASS' if row['stable_pickup_pass'] else 'FAIL'} |"
        )
    lines += [
        "", "## Tracking gate", "",
        f"- Tracking failures: `{len(tracking_failures)}`",
        f"- First tracking failure: `{tracking_failures[0] if tracking_failures else 'NONE'}`",
        "", "## Collision/proxy evidence", "",
        f"- Native blocking observations: `{len(native_blockers)}`",
        f"- Endpoint proxy/native geometry mismatch observations: `{len(geometry_mismatches)}`",
        f"- Preceding-interval proxy-force/native-clear exposure observations: `{len(force_exposure_mismatches)}`",
        f"- Maximum runtime contact force: `{result['maximums']['contact_force_n']:.6f} N`",
        f"- Maximum runtime proxy penetration: `{result['maximums']['proxy_penetration_m']*1000:.6f} mm`",
    ]
    for name, row in proxy_impact.items():
        fraction = row["mismatch_impulse_fraction"]
        lines.append(
            f"- {name}: sampled hand-object impulse `{row['sampled_hand_object_normal_impulse_ns']:.9f} N·s`; "
            f"mismatch impulse `{row['mismatch_normal_impulse_ns']:.9f} N·s` "
            f"(`{'N/A' if fraction is None else f'{fraction:.3%}'}`); mismatch-interval "
            f"object displacement sum `{row['sum_object_displacement_during_mismatch_intervals_m']*1000:.6f} mm`."
        )
    lines += [
        "", "These impulse/displacement values measure exposure along the proxy-driven Replay. "
        "They are not a counterfactual simulation with exact native-mesh contact and therefore do not prove how the true-geometry trajectory would evolve.",
        "", "A strict PASS is never inferred merely because the bounded Replay reached its final endpoint.", "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    hash_tree(output)
    print(json.dumps({
        "classification": classification, "strict_gate_passed": strict_pass,
        "completed_outcome_endpoint": endpoint_rows[-1]["endpoint"],
        "physics_steps": physics_steps, "hard_stop": hard_stop,
        "pickup": pickup, "native_blockers": len(native_blockers),
        "tracking_failures": len(tracking_failures),
        "proxy_native_geometry_mismatches": len(geometry_mismatches),
        "proxy_native_force_exposure_mismatches": len(force_exposure_mismatches),
    }, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
