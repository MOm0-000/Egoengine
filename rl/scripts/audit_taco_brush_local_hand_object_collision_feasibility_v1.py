#!/usr/bin/env python3
"""Isolated fixed-frame MINK hand-object collision feasibility comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_world_objects,
    pair_group_audit,
)
from audit_taco_brush_mano_xhand_object_penetration_attribution_v1 import (  # noqa: E402
    penetration_status,
)
from egoengine_repro.retarget.collision_audit import (  # noqa: E402
    collision_families,
    distances,
    explicit_hand_pairs,
)
from egoengine_repro.retarget.kinematic_limits import (  # noqa: E402
    FrameDisplacementLimit,
    StrictCollisionLimit,
    enable_planning_collision_masks,
    explicit_collision_groups,
    joint_velocity_limits,
)
from egoengine_repro.retarget.schema import validate_human_reference  # noqa: E402
from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    NativeSupportPlaneLimit,
    native_hand_visual_geom_ids,
)
from egoengine_repro.retarget.taco_bimanual import FINGERS, SIDES  # noqa: E402
from egoengine_repro.retarget.taco_bimanual_settings import (  # noqa: E402
    load_taco_bimanual_settings,
)
from egoengine_repro.scene.support_surface import Plane  # noqa: E402
from run_taco_brush_issue14_mink_candidate_v1 import relevant_pairs  # noqa: E402


SCHEMA = "taco_brush_local_hand_object_collision_feasibility_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_local_hand_object_collision_feasibility_v1.yaml"


@dataclass
class LocalRuntime:
    model: mujoco.MjModel
    configuration: mink.Configuration
    tasks: list[Any]
    limits: list[Any]
    locks: list[Any]
    object_limit: StrictCollisionLimit
    self_limit: StrictCollisionLimit
    support_limit: NativeSupportPlaneLimit
    displacement_limit: FrameDisplacementLimit
    object_pairs: list[tuple[int, int]]
    self_pairs: list[tuple[int, int]]
    relevant_pairs: dict[str, list[tuple[int, int]]]
    velocity_addresses: np.ndarray
    velocity_ranges: np.ndarray


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


def load_support(path: Path) -> Plane:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("promotion_authorized") is not False:
        raise ValueError("local audit requires the unpromoted support candidate")
    value = raw["simulator"]
    return Plane(normal=value["normal"], offset=value["offset_m"], frame=value["frame"])


def hand_geom_ids(model: mujoco.MjModel, side: str | None = None) -> list[int]:
    prefix = "collision_hand_" if side is None else f"collision_hand_{side}_"
    return [
        geom for geom in range(model.ngeom)
        if (model.geom(geom).name or "").startswith(prefix)
    ]


def object_geom_ids(model: mujoco.MjModel, side: str) -> list[int]:
    return [
        geom for geom in range(model.ngeom)
        if (model.geom(geom).name or "").startswith(f"{side}_object_")
        and not (model.geom(geom).name or "").endswith("visual")
    ]


def pair_groups(model: mujoco.MjModel, pairs: list[tuple[int, int]]) -> list[tuple[list[str], list[str]]]:
    return [
        ([model.geom(first).name], [model.geom(second).name])
        for first, second in pairs
    ]


def build_runtime(
    cfg: dict[str, Any], scene: Path, settings: Any, support_plane: Plane,
    qpos: np.ndarray, previous_qpos: np.ndarray, human: dict[str, np.ndarray], frame: int,
) -> LocalRuntime:
    model = mujoco.MjModel.from_xml_path(str(scene))
    configuration = mink.Configuration(model, q=qpos.copy())
    tasks: list[Any] = []
    for hand, side in enumerate(SIDES):
        wrist = mink.FrameTask(
            f"{side}_hand_link", "body", position_cost=0.0,
            orientation_cost=settings.wrist_orientation_cost, lm_damping=1e-3,
        )
        wrist.set_target(mink.SE3.from_matrix(human["T_sim_wrist_target"][frame, hand]))
        tasks.append(wrist)
        for finger_index, finger in enumerate(FINGERS):
            task = mink.FrameTask(
                f"{side}_{finger}_tip", "site",
                position_cost=settings.fingertip_position_cost,
                orientation_cost=settings.fingertip_orientation_cost,
                lm_damping=1e-3,
            )
            task.set_target(mink.SE3.from_matrix(
                human["T_sim_fingertip_target"][frame, hand, finger_index]
            ))
            tasks.append(task)

    all_hand_ids = hand_geom_ids(model)
    selected_object_pairs: list[tuple[int, int]] = []
    relevant: dict[str, list[tuple[int, int]]] = {}
    families = collision_families(model)
    for spec in cfg["sample"]["interaction_pairs"]:
        object_name = spec["object"]
        family = families["hand_tool" if spec["role"] == "tool" else "hand_target"]
        pairs = relevant_pairs(
            model, family, spec["hand"], spec["object_side"],
        )
        relevant[object_name] = pairs
        selected_object_pairs.extend(pairs)
    selected_object_pairs = sorted(set(selected_object_pairs))
    all_object_ids = sorted({value for pair in selected_object_pairs for value in pair
                             if value not in all_hand_ids})
    enable_planning_collision_masks(model, all_hand_ids, all_object_ids)

    self_groups = explicit_collision_groups(
        model, mujoco, hand_geom_ids=set(all_hand_ids),
    )
    self_inner = mink.CollisionAvoidanceLimit(
        model, self_groups,
        minimum_distance_from_collisions=settings.planning_collision_buffer_m,
        collision_detection_distance=0.02,
        include_explicit_pairs=True,
    )
    self_limit = StrictCollisionLimit(
        self_inner, mujoco,
        minimum_distance=settings.planning_collision_buffer_m,
        depenetration_step=cfg["local_solver"]["collision_depenetration_step_m"],
        gain=cfg["local_solver"]["collision_gain"],
    )
    self_limit.enabled = True
    if set(self_limit.geom_id_pairs) != set(explicit_hand_pairs(model)):
        raise ValueError("local self-collision pairs differ from the active contract")

    object_inner = mink.CollisionAvoidanceLimit(
        model, pair_groups(model, selected_object_pairs),
        minimum_distance_from_collisions=cfg["local_solver"]["object_collision_minimum_distance_m"],
        collision_detection_distance=cfg["local_solver"]["object_collision_detection_distance_m"],
        include_explicit_pairs=True,
    )
    object_limit = StrictCollisionLimit(
        object_inner, mujoco,
        minimum_distance=cfg["local_solver"]["object_collision_minimum_distance_m"],
        depenetration_step=cfg["local_solver"]["collision_depenetration_step_m"],
        gain=cfg["local_solver"]["collision_gain"],
    )
    object_limit.enabled = True
    if set(object_limit.geom_id_pairs) != set(selected_object_pairs):
        raise ValueError("MINK filtered a requested relevant hand-object pair")

    velocity_map = joint_velocity_limits(model, mujoco, settings.velocity_limits)
    displacement = FrameDisplacementLimit(model, mujoco, velocity_map, mink.Constraint)
    dt = float(np.diff(human["timestamps_s"])[0])
    displacement.set_previous(previous_qpos, dt)
    support = cfg["native_support"]
    support_limit = NativeSupportPlaneLimit(
        model=model, mujoco=mujoco, plane=support_plane,
        visual_geom_ids=native_hand_visual_geom_ids(model, mujoco),
        minimum_clearance_m=support["minimum_clearance_m"],
        activation_distance_m=support["activation_distance_m"],
        gain=support["gain"],
        depenetration_step_m=support["depenetration_step_m"],
        constraint_type=mink.Constraint,
    )
    limits: list[Any] = [
        mink.ConfigurationLimit(model), self_limit, object_limit, displacement, support_limit,
    ]
    locks = [mink.DofFreezingTask(model, list(range(36, 48)))]
    addresses = np.asarray([
        int(model.joint(name).qposadr[0]) for name in velocity_map
    ], dtype=np.int64)
    ranges = np.asarray([model.joint(name).range for name in velocity_map], dtype=np.float64)
    return LocalRuntime(
        model=model, configuration=configuration, tasks=tasks, limits=limits, locks=locks,
        object_limit=object_limit, self_limit=self_limit, support_limit=support_limit,
        displacement_limit=displacement, object_pairs=selected_object_pairs,
        self_pairs=list(self_limit.geom_id_pairs), relevant_pairs=relevant,
        velocity_addresses=addresses, velocity_ranges=ranges,
    )


def metrics(
    runtime: LocalRuntime, human: dict[str, np.ndarray], frame: int,
    support_tolerance_m: float, material_tolerance_m: float,
) -> dict[str, Any]:
    model, configuration = runtime.model, runtime.configuration
    data = configuration.data
    fingertip = np.empty((2, 5), dtype=np.float64)
    wrist = np.empty(2, dtype=np.float64)
    for hand, side in enumerate(SIDES):
        for index, finger in enumerate(FINGERS):
            fingertip[hand, index] = np.linalg.norm(
                data.site_xpos[model.site(f"{side}_{finger}_tip").id]
                - human["T_sim_fingertip_target"][frame, hand, index, :3, 3]
            )
        body = model.body(f"{side}_hand_link").id
        relative = (
            data.xmat[body].reshape(3, 3).T
            @ human["T_sim_wrist_target"][frame, hand, :3, :3]
        )
        wrist[hand] = Rotation.from_matrix(relative).magnitude()
    joint_values = configuration.q[runtime.velocity_addresses]
    joint_margin = float(np.minimum(
        joint_values - runtime.velocity_ranges[:, 0],
        runtime.velocity_ranges[:, 1] - joint_values,
    ).min())
    proxy_by_object = {
        name: float(distances(model, data, pairs, detection=0.1).min())
        for name, pairs in runtime.relevant_pairs.items()
    }
    native = native_world_objects(model, data)
    native_by_object = {}
    for name, pairs in runtime.relevant_pairs.items():
        report = pair_group_audit(model, data, native, pairs, material_tolerance_m)
        distance = float(report["native_minimum"]["distance_m"])
        native_by_object[name] = {
            "distance_m": distance,
            "status": penetration_status(distance, material_tolerance_m),
            "minimum_pair": report["native_minimum"],
            "penetrating_pair_count": len(report["native_penetrating_pairs"]),
            "unknown_pair_count": len(report["unknown_pairs"]),
        }
    support_minimum = runtime.support_limit.full_mesh_minimum(data)
    self_minimum = float(distances(model, data, runtime.self_pairs).min())
    object_minimum = float(min(proxy_by_object.values()))
    return {
        "fingertip_error_m": fingertip.tolist(),
        "fingertip_mean_error_m": float(fingertip.mean()),
        "fingertip_max_error_m": float(fingertip.max()),
        "wrist_orientation_error_rad": wrist.tolist(),
        "joint_limit_minimum_margin": joint_margin,
        "joint_limits_pass": joint_margin >= -1e-6,
        "frame_displacement_minimum_margin": runtime.displacement_limit.minimum_margin(configuration),
        "frame_displacement_pass": runtime.displacement_limit.minimum_margin(configuration) >= -1e-6,
        "self_collision_minimum_distance_m": self_minimum,
        "self_collision_pass": self_minimum >= -1e-6,
        "native_support_minimum_distance_m": support_minimum,
        "native_support_pass": support_minimum >= -support_tolerance_m,
        "object_proxy_minimum_distance_m": object_minimum,
        "object_proxy_by_object_m": proxy_by_object,
        "native_hand_object_by_object": native_by_object,
    }


def solve_candidate(
    runtime: LocalRuntime, cfg: dict[str, Any], dt: float,
) -> dict[str, Any]:
    integration_dt = dt / int(cfg["local_solver"]["tracking_iterations"])
    error = None
    tracking_completed = 0
    closure_completed = 0
    try:
        for _ in range(int(cfg["local_solver"]["tracking_iterations"])):
            velocity = mink.solve_ik(
                runtime.configuration, runtime.tasks, integration_dt,
                solver=cfg["local_solver"]["solver"],
                damping=cfg["local_solver"]["damping"],
                limits=runtime.limits, constraints=runtime.locks,
                primal_tol=1e-6, dual_tol=1e-6,
            )
            runtime.configuration.integrate_inplace(velocity, integration_dt)
            tracking_completed += 1
        threshold = float(cfg["local_solver"]["object_collision_minimum_distance_m"])
        tolerance = float(cfg["local_solver"]["collision_validation_tolerance_m"])
        maximum = int(cfg["local_solver"]["maximum_feasibility_iterations"])
        for _ in range(maximum):
            minimum = float(distances(
                runtime.model, runtime.configuration.data,
                runtime.object_pairs, detection=0.1,
            ).min())
            if minimum >= threshold - tolerance:
                break
            velocity = mink.solve_ik(
                runtime.configuration, (), integration_dt,
                solver=cfg["local_solver"]["solver"],
                damping=cfg["local_solver"]["damping"],
                limits=runtime.limits, constraints=runtime.locks,
                primal_tol=1e-6, dual_tol=1e-6,
            )
            runtime.configuration.integrate_inplace(velocity, integration_dt)
            closure_completed += 1
        else:
            error = "OBJECT_PROXY_FEASIBILITY_CLOSURE_EXHAUSTED"
    except Exception as caught:  # fail closed but preserve the local diagnostic
        error = f"{type(caught).__name__}: {caught}"
    return {
        "tracking_iterations_completed": tracking_completed,
        "closure_iterations_completed": closure_completed,
        "solver_error": error,
    }


def classify_frame(
    baseline: dict[str, Any], candidate: dict[str, Any], solver_error: str | None,
    *, gap_reference_m: float,
) -> str:
    if solver_error is not None:
        return "LOCAL_MINK_SOLVE_FAILED"
    hard = (
        candidate["joint_limits_pass"]
        and candidate["frame_displacement_pass"]
        and candidate["self_collision_pass"]
        and candidate["native_support_pass"]
        and candidate["object_proxy_minimum_distance_m"] >= -1e-6
    )
    if not hard:
        return "LOCAL_HARD_CONTRACT_FAIL"
    native = candidate["native_hand_object_by_object"]
    if any(value["status"] == "PENETRATION" for value in native.values()):
        return "PROXY_CLEAR_BUT_NATIVE_PENETRATION_REMAINS"
    baseline_native = baseline["native_hand_object_by_object"]
    affected = [
        name for name, value in baseline_native.items()
        if value["status"] == "PENETRATION"
    ]
    if any(native[name]["distance_m"] > gap_reference_m for name in affected):
        return "NATIVE_PENETRATION_CLEARED_WITH_GAP_OVER_2MM"
    return "NATIVE_PENETRATION_CLEARED_WITHOUT_GAP_OVER_2MM"


def render_comparison(
    model: mujoco.MjModel, baseline: np.ndarray, candidate: np.ndarray,
    frames: list[int], output: Path, table_z: float,
) -> None:
    model.geom_pos[model.geom("floor").id, 2] = table_z
    renderer = mujoco.Renderer(model, height=360, width=480)
    data = mujoco.MjData(model)
    visual = output / "visuals"
    visual.mkdir()
    for row, frame in enumerate(frames):
        panels = []
        for state in (baseline[row], candidate[row]):
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            for elevation in (-18, -90):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0.60, 0.0, 0.76]
                camera.distance = 0.70
                camera.azimuth = 90
                camera.elevation = elevation
                renderer.update_scene(data, camera=camera)
                panels.append(renderer.render().copy())
        image = np.concatenate(panels, axis=1)
        path = visual / f"frame_{frame:03d}_baseline_candidate_front_top.jpg"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write {path}")
    renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected local feasibility schema")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], "HEAD"],
        cwd=REPO_ROOT, check=True,
    )
    forbidden = (
        "modify_table_contract", "modify_object_pose", "modify_task_weights",
        "modify_reference", "full_trajectory_retarget", "physics", "replay", "mpc",
        "reinforcement_learning", "training_search", "promotion", "chunk_commit",
    )
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("local read-only experiment authorizes a forbidden operation")
    paths = {
        key: Path(value).resolve(strict=True)
        for key, value in cfg["paths"].items() if key != "output"
    }
    output = Path(cfg["paths"]["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    settings, consumption = load_taco_bimanual_settings(paths["retarget_settings"])
    if consumption["unknown_key_count"] or consumption["unused_key_count"]:
        raise ValueError("retarget settings are not exactly consumed")
    support_plane = load_support(paths["candidate_support"])
    if support_plane.offset != cfg["frozen_contract"]["table_simulator_z_m"]:
        raise ValueError("candidate support changed")
    with np.load(paths["human_reference"], allow_pickle=False) as archive:
        human = dict(archive)
    validate_human_reference(human)
    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if qpos.shape != (cfg["sample"]["frames"], 50):
        raise ValueError("robot reference changed")
    prior = json.loads(paths["prior_attribution"].read_text(encoding="utf-8"))
    official = {
        (row["frame"], row["object"]): row["official_mano_distance_m"]
        for row in prior["rows"]
    }
    frames = [int(value) for value in cfg["sample"]["selected_frames"]]
    dt = float(np.diff(human["timestamps_s"])[0])
    rows = []
    baseline_states, candidate_states = [], []
    for frame in frames:
        runtime = build_runtime(
            cfg, paths["scene"], settings, support_plane,
            qpos[frame], qpos[frame - 1], human, frame,
        )
        baseline = metrics(
            runtime, human, frame,
            cfg["native_support"]["validation_tolerance_m"],
            cfg["local_solver"]["native_material_penetration_tolerance_m"],
        )
        solve = solve_candidate(runtime, cfg, dt)
        candidate = metrics(
            runtime, human, frame,
            cfg["native_support"]["validation_tolerance_m"],
            cfg["local_solver"]["native_material_penetration_tolerance_m"],
        )
        object_change = float(np.max(np.abs(runtime.configuration.q[36:50] - qpos[frame, 36:50])))
        if object_change > 1e-12:
            raise ValueError("object pose changed despite the frozen-DOF constraint")
        gap_reference = float(cfg["local_solver"]["unreasonable_native_gap_diagnostic_m"])
        row = {
            "frame": frame,
            "baseline": baseline,
            "candidate": candidate,
            "solver": solve,
            "object_qpos_maximum_absolute_change": object_change,
            "classification": classify_frame(
                baseline, candidate, solve["solver_error"], gap_reference_m=gap_reference,
            ),
            "per_object_change": {},
        }
        for spec in cfg["sample"]["interaction_pairs"]:
            name = spec["object"]
            before = baseline["native_hand_object_by_object"][name]["distance_m"]
            after = candidate["native_hand_object_by_object"][name]["distance_m"]
            row["per_object_change"][name] = {
                "official_mano_native_distance_m": official[(frame, name)],
                "baseline_xhand_native_distance_m": before,
                "candidate_xhand_native_distance_m": after,
                "native_distance_change_m": after - before,
                "baseline_was_penetrating": (
                    baseline["native_hand_object_by_object"][name]["status"]
                    == "PENETRATION"
                ),
                "candidate_gap_over_2mm_reference": after > gap_reference,
                "candidate_gap_over_2mm_after_baseline_penetration": (
                    baseline["native_hand_object_by_object"][name]["status"]
                    == "PENETRATION" and after > gap_reference
                ),
                "candidate_excess_distance_over_official_mano_m": after - official[(frame, name)],
            }
        rows.append(row)
        baseline_states.append(qpos[frame].copy())
        candidate_states.append(runtime.configuration.q.copy())

    baseline_array = np.stack(baseline_states)
    candidate_array = np.stack(candidate_states)
    np.savez_compressed(
        output / "local_candidates.npz", frame_indices=np.asarray(frames),
        baseline_qpos=baseline_array, candidate_qpos=candidate_array,
    )
    render_model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    render_comparison(
        render_model, baseline_array, candidate_array, frames, output, support_plane.offset,
    )
    classifications = {row["classification"] for row in rows}
    if classifications == {"NATIVE_PENETRATION_CLEARED_WITHOUT_GAP_OVER_2MM"}:
        overall = "LOCAL_OBJECT_COLLISION_CONSTRAINT_FEASIBLE_WITHOUT_LARGE_NATIVE_GAP"
    elif all(row["solver"]["solver_error"] is None for row in rows):
        overall = "LOCAL_OBJECT_COLLISION_CONSTRAINT_HAS_GEOMETRIC_SIDE_EFFECTS"
    else:
        overall = "LOCAL_OBJECT_COLLISION_CONSTRAINT_NOT_FEASIBLE"
    result = {
        "schema": SCHEMA,
        "classification": overall,
        "selected_frames": frames,
        "local_solver_contract": cfg["local_solver"],
        "frozen_contract": cfg["frozen_contract"],
        "rows": rows,
        "mutations": {
            "source_files": 0, "table_contract": 0, "object_pose": 0,
            "reference": 0, "full_trajectory_retarget": 0,
            "physics_steps": 0, "training_steps": 0,
        },
        "source_artifacts": {
            key: artifact(path) for key, path in paths.items() if path.is_file()
        } | {"config": artifact(config_path), "script": artifact(Path(__file__))},
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    write_json(output / "results.json", result)
    with (output / "frame_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "frame", "object", "classification", "solver_error",
            "baseline_native_mm", "candidate_native_mm", "candidate_proxy_mm",
            "official_mano_native_mm", "candidate_gap_over_2mm_reference",
            "candidate_gap_over_2mm_after_baseline_penetration",
            "baseline_tip_mean_mm", "candidate_tip_mean_mm",
            "candidate_joint_margin", "candidate_self_mm", "candidate_support_mm",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for spec in cfg["sample"]["interaction_pairs"]:
                name = spec["object"]
                change = row["per_object_change"][name]
                writer.writerow({
                    "frame": row["frame"], "object": name,
                    "classification": row["classification"],
                    "solver_error": row["solver"]["solver_error"],
                    "baseline_native_mm": change["baseline_xhand_native_distance_m"] * 1000,
                    "candidate_native_mm": change["candidate_xhand_native_distance_m"] * 1000,
                    "candidate_proxy_mm": row["candidate"]["object_proxy_by_object_m"][name] * 1000,
                    "official_mano_native_mm": change["official_mano_native_distance_m"] * 1000,
                    "candidate_gap_over_2mm_reference": change["candidate_gap_over_2mm_reference"],
                    "candidate_gap_over_2mm_after_baseline_penetration": (
                        change["candidate_gap_over_2mm_after_baseline_penetration"]
                    ),
                    "baseline_tip_mean_mm": row["baseline"]["fingertip_mean_error_m"] * 1000,
                    "candidate_tip_mean_mm": row["candidate"]["fingertip_mean_error_m"] * 1000,
                    "candidate_joint_margin": row["candidate"]["joint_limit_minimum_margin"],
                    "candidate_self_mm": row["candidate"]["self_collision_minimum_distance_m"] * 1000,
                    "candidate_support_mm": row["candidate"]["native_support_minimum_distance_m"] * 1000,
                })
    lines = [
        "# Brush local hand-object collision feasibility", "",
        f"- Classification: `{overall}`",
        "- Full 209-frame retarget: `NOT RUN`",
        "- Physics/training: `NOT RUN`", "",
        "| frame | object | baseline native (mm) | candidate native (mm) | candidate proxy (mm) | tip mean before/after (mm) | caused >2mm gap while clearing penetration | frame result |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        for spec in cfg["sample"]["interaction_pairs"]:
            name = spec["object"]
            change = row["per_object_change"][name]
            lines.append(
                f"| {row['frame']} | {name} | "
                f"{change['baseline_xhand_native_distance_m']*1000:.6f} | "
                f"{change['candidate_xhand_native_distance_m']*1000:.6f} | "
                f"{row['candidate']['object_proxy_by_object_m'][name]*1000:.6f} | "
                f"{row['baseline']['fingertip_mean_error_m']*1000:.6f} / "
                f"{row['candidate']['fingertip_mean_error_m']*1000:.6f} | "
                f"{change['candidate_gap_over_2mm_after_baseline_penetration']} | "
                f"`{row['classification']}` |"
            )
    lines += [
        "", "## Preserved constraints", "",
        "| frame | tip max before/after (mm) | joint limits | frame displacement | self collision | table support | object pose max change |",
        "|---:|---:|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['frame']} | "
            f"{row['baseline']['fingertip_max_error_m']*1000:.6f} / "
            f"{row['candidate']['fingertip_max_error_m']*1000:.6f} | "
            f"{row['candidate']['joint_limits_pass']} | "
            f"{row['candidate']['frame_displacement_pass']} | "
            f"{row['candidate']['self_collision_pass']} | "
            f"{row['candidate']['native_support_pass']} | "
            f"{row['object_qpos_maximum_absolute_change']:.3e} |"
        )
    proxy_native_mismatches = [
        str(row["frame"]) for row in rows
        if row["classification"] == "PROXY_CLEAR_BUT_NATIVE_PENETRATION_REMAINS"
    ]
    introduced_large_gaps = [
        f"frame {row['frame']} {name}"
        for row in rows
        for name, change in row["per_object_change"].items()
        if change["candidate_gap_over_2mm_after_baseline_penetration"]
    ]
    lines += [
        "", "## Decision", "",
        (
            "- Proxy/native mismatch frames: "
            + (", ".join(proxy_native_mismatches) if proxy_native_mismatches else "none")
            + "."
        ),
        (
            "- Newly introduced gaps over 2 mm while clearing a baseline penetration: "
            + (", ".join(introduced_large_gaps) if introduced_large_gaps else "none")
            + "."
        ),
        "- The existing MINK collision proxies are therefore not accepted as a reliable native-mesh hand-object correction for these three frames.",
        "- Candidate promotion and the full 209-frame retarget remain forbidden.",
        "",
        "The 2 mm value is a predeclared diagnostic reference only; it did not alter the zero-clearance solve. No threshold was changed after observing the result.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "server_artifacts.sha256":
            checksums.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": overall,
        "frames": [{
            "frame": row["frame"], "classification": row["classification"],
            "solver": row["solver"], "per_object_change": row["per_object_change"],
        } for row in rows],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
