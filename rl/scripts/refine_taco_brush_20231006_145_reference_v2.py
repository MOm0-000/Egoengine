#!/usr/bin/env python3
"""Generate the authorized M4 local continuous-reference candidate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.retarget.collision_audit import (  # noqa: E402
    distances,
    explicit_hand_pairs,
)
from egoengine_repro.retarget.kinematic_limits import (  # noqa: E402
    joint_velocity_limits,
)
from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    build_native_support_geoms,
    native_hand_visual_geom_ids,
)
from egoengine_repro.retarget.taco_bimanual_settings import (  # noqa: E402
    load_taco_bimanual_settings,
)
from egoengine_repro.retarget.trajectory_refinement import (  # noqa: E402
    apply_delta,
    differentiate_trajectory,
    file_sha256,
    fingertip_positions,
    full_mesh_support_witnesses,
    immutable_clearance_blockers,
    refine_native_support_path,
    resolve_variables,
    validate_scalar_joint_ranges,
)


DEFAULT_CONFIG = (
    RL_ROOT
    / "configs/taco_brush_20231006_145_continuous_reference_refinement_v2.yaml"
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("right", "left")
TIP_SITES = tuple(f"{side}_{finger}_tip" for side in SIDES for finger in FINGERS)


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
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_interval_summary(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {int(row["source_endpoint"]): row for row in rows}


def joint_budget(name: str, cfg: dict[str, Any]) -> float:
    budgets = cfg["modification_budgets"]
    if name.startswith(("L_forearm_t", "R_forearm_t")):
        return float(budgets["wrist_translation_m"])
    if name.startswith(("L_forearm_", "R_forearm_")):
        return float(budgets["wrist_rotation_rad"])
    return float(budgets["finger_joint_rad"])


def endpoint_native_distances(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    geoms: tuple[Any, ...],
    plane_z_m: float,
) -> np.ndarray:
    result = np.empty(len(qpos), dtype=np.float64)
    data = mujoco.MjData(model)
    for endpoint, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        result[endpoint] = min(
            row.distance_m
            for row in full_mesh_support_witnesses(
                model,
                data,
                geoms,
                plane_z_m=plane_z_m,
                endpoint=endpoint,
            )
        )
    return result


def derived_reference(
    model: mujoco.MjModel,
    original: dict[str, np.ndarray],
    human: dict[str, np.ndarray],
    qpos: np.ndarray,
    geoms: tuple[Any, ...],
    settings_path: Path,
    plane_z_m: float,
) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value).copy() for name, value in original.items()}
    result["qpos"] = qpos
    result["ctrl"] = qpos[:, : int(model.nu)].copy()
    result["qvel"] = differentiate_trajectory(
        model,
        mujoco,
        qpos,
        np.asarray(result["timestamps_s"], dtype=np.float64),
    )

    data = mujoco.MjData(model)
    tip_error = np.empty((len(qpos), 2, 5), dtype=np.float64)
    wrist_error = np.empty((len(qpos), 2), dtype=np.float64)
    collision_pairs = explicit_hand_pairs(model)
    self_distance = np.empty(len(qpos), dtype=np.float64)
    for endpoint, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side_index, side in enumerate(SIDES):
            for finger_index, finger in enumerate(FINGERS):
                site = int(model.site(f"{side}_{finger}_tip").id)
                tip_error[endpoint, side_index, finger_index] = np.linalg.norm(
                    data.site_xpos[site]
                    - human["T_sim_fingertip_target"][
                        endpoint, side_index, finger_index, :3, 3
                    ]
                )
            body = int(model.body(f"{side}_hand_link").id)
            relative = (
                np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3).T
                @ human["T_sim_wrist_target"][endpoint, side_index, :3, :3]
            )
            wrist_error[endpoint, side_index] = Rotation.from_matrix(relative).magnitude()
        self_distance[endpoint] = float(
            distances(model, data, collision_pairs, detection=0.05).min()
        )

    settings, consumption = load_taco_bimanual_settings(settings_path)
    if consumption["unknown_key_count"] or consumption["unused_key_count"]:
        raise ValueError("retarget settings are not exactly consumed")
    limits = joint_velocity_limits(model, mujoco, settings.velocity_limits)
    names = tuple(limits)
    addresses = np.asarray(
        [int(model.joint(name).qposadr[0]) for name in names], dtype=np.int64
    )
    speeds = np.asarray([limits[name] for name in names], dtype=np.float64)
    ranges = np.asarray(
        [model.joint(name).range for name in names], dtype=np.float64
    )
    dt_s = 1.0 / float(np.asarray(result["frequency"]))
    result["fingertip_position_error_m"] = tip_error
    result["wrist_orientation_error_rad"] = wrist_error
    result["self_collision_distance_m"] = self_distance
    result["joint_limit_min_margin"] = np.minimum(
        qpos[:, addresses] - ranges[:, 0], ranges[:, 1] - qpos[:, addresses]
    ).min(axis=1)
    result["frame_velocity_max_ratio"] = (
        np.abs(np.diff(qpos[:, addresses], axis=0)) / (dt_s * speeds)
    ).max(axis=1)
    result["native_support_min_distance_m"] = endpoint_native_distances(
        model, qpos, geoms, plane_z_m
    )
    return result


def fingertip_rows(
    model: mujoco.MjModel,
    original_qpos: np.ndarray,
    refined_qpos: np.ndarray,
    endpoints: list[int],
) -> list[dict[str, Any]]:
    original = fingertip_positions(
        model,
        mujoco,
        original_qpos,
        endpoints=endpoints,
        site_names=TIP_SITES,
    )
    refined = fingertip_positions(
        model,
        mujoco,
        refined_qpos,
        endpoints=endpoints,
        site_names=TIP_SITES,
    )
    rows: list[dict[str, Any]] = []
    for endpoint_index, endpoint in enumerate(endpoints):
        for site_index, site_name in enumerate(TIP_SITES):
            before = original[endpoint_index, site_index]
            after = refined[endpoint_index, site_index]
            rows.append({
                "endpoint": endpoint,
                "site": site_name,
                "original_x_m": f"{before[0]:.17g}",
                "original_y_m": f"{before[1]:.17g}",
                "original_z_m": f"{before[2]:.17g}",
                "refined_x_m": f"{after[0]:.17g}",
                "refined_y_m": f"{after[1]:.17g}",
                "refined_z_m": f"{after[2]:.17g}",
                "position_change_m": f"{np.linalg.norm(after - before):.17g}",
                "contact_status": "UNKNOWN_NATIVE_DISTANCE_AUDITED_SEPARATELY",
            })
    return rows


def exact_objective(
    variables: tuple[Any, ...],
    delta: np.ndarray,
    tips: list[dict[str, Any]],
    endpoints: list[int],
    weights: dict[str, float],
    fingertip_scale_m: float,
) -> dict[str, float]:
    index = {(item.endpoint, item.joint_name): i for i, item in enumerate(variables)}
    reference = sum((delta[i] / item.budget) ** 2 for i, item in enumerate(variables))
    smooth = 0.0
    for joint_name in dict.fromkeys(item.joint_name for item in variables):
        budget = next(item.budget for item in variables if item.joint_name == joint_name)
        for endpoint in endpoints:
            value = 0.0
            for neighbor, coefficient in (
                (endpoint - 1, 1.0), (endpoint, -2.0), (endpoint + 1, 1.0),
            ):
                item_index = index.get((neighbor, joint_name))
                if item_index is not None:
                    value += coefficient * delta[item_index]
            smooth += (value / budget) ** 2
    tip = sum(
        (float(row["position_change_m"]) / fingertip_scale_m) ** 2
        for row in tips
    )
    terms = {
        "reference_fidelity": float(weights["reference_fidelity"]) * reference,
        "second_difference_smoothness": float(
            weights["second_difference_smoothness"]
        ) * smooth,
        "fingertip_preservation": float(weights["fingertip_preservation"]) * tip,
        "object_relative_fingertip_preservation": float(
            weights["object_relative_fingertip_preservation"]
        ) * tip,
    }
    terms["total"] = sum(terms.values())
    return terms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != "taco_brush_20231006_145_continuous_reference_refinement_v2":
        raise ValueError("unexpected refinement config schema")
    paths = {
        name: Path(value).resolve(strict=(name != "output"))
        for name, value in cfg["paths"].items()
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    expected = cfg["expected_inputs"]
    for key, path_key in (
        ("robot_reference_sha256", "robot_reference"),
        ("human_reference_sha256", "human_reference"),
        ("scene_sha256", "scene"),
        ("retarget_settings_sha256", "retarget_settings"),
        ("finger_dynamics_contract_sha256", "finger_dynamics_contract"),
    ):
        if file_sha256(paths[path_key]) != expected[key]:
            raise ValueError(f"frozen input hash changed: {path_key}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["base_commit"], "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )

    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        reference = {name: np.asarray(archive[name]).copy() for name in archive.files}
    with np.load(paths["human_reference"], allow_pickle=False) as archive:
        human = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if reference["qpos"].shape != (122, 50) or reference["ctrl"].shape != (122, 36):
        raise ValueError("original reference dimensions changed")
    if not np.array_equal(reference["ctrl"], reference["qpos"][:, :36]):
        raise ValueError("original reference control mapping changed")
    if float(reference["frequency"]) != float(cfg["sample"]["original_frequency_hz"]):
        raise ValueError("original reference frequency changed")

    geometry = cfg["geometry_contract"]
    plane_z_m = float(geometry["table_z_m"])
    endpoints = [int(value) for value in cfg["modified_endpoints"]]
    geoms = build_native_support_geoms(
        model, mujoco, native_hand_visual_geom_ids(model, mujoco)
    )
    candidate_records: list[dict[str, Any]] = []
    selected_cfg = None
    selected_variables = None
    for candidate_cfg in cfg["candidate_order"]:
        names = [str(value) for value in candidate_cfg["authorized_joints"]]
        variables = resolve_variables(
            model,
            endpoints=endpoints,
            joint_names=names,
            budgets={name: joint_budget(name, cfg) for name in names},
            robot_qpos_stop=int(cfg["sample"]["robot_qpos_stop"]),
        )
        blockers = immutable_clearance_blockers(
            model,
            mujoco,
            reference["qpos"],
            variables,
            geoms,
            endpoints=geometry["core_endpoints"],
            plane_z_m=plane_z_m,
            minimum_clearance_m=float(geometry["core_minimum_clearance_m"]),
        )
        candidate_records.append({
            "candidate": candidate_cfg["id"],
            "authorized_joints": names,
            "variable_count": len(variables),
            "immutable_clearance_blockers": blockers,
            "status": "PRE_QP_INFEASIBLE" if blockers else "READY_FOR_SQP",
        })
        if not blockers:
            selected_cfg = candidate_cfg
            selected_variables = variables
            break
    if selected_cfg is None or selected_variables is None:
        raise RuntimeError("no authorized candidate passed the immutable geometry gate")
    if selected_cfg["id"] != "M4":
        raise RuntimeError("frozen M1-M3 failure sequence changed")

    prior = read_interval_summary(paths["prior_dense_interval_summary"])
    transition_floor = float(geometry["transition_floor_m"])
    allowance = float(geometry["transition_nonworsening_allowance_m"])
    thresholds = {
        interval: float(geometry["core_minimum_clearance_m"])
        for interval in geometry["core_intervals"]
    }
    for interval in geometry["transition_intervals"]:
        thresholds[int(interval)] = max(
            transition_floor,
            float(prior[int(interval)]["minimum_distance_m"]) - allowance,
        )
    intervals = sorted(thresholds)
    velocity_cfg = cfg["kinematic_contract"]["velocity_limits"]
    velocity_limits = {}
    for name in selected_cfg["authorized_joints"]:
        if name.startswith(("L_forearm_t", "R_forearm_t")):
            velocity_limits[name] = float(velocity_cfg["base_translation_m_s"])
        elif name.startswith(("L_forearm_", "R_forearm_")):
            velocity_limits[name] = float(velocity_cfg["base_rotation_rad_s"])
        else:
            velocity_limits[name] = float(velocity_cfg["finger_rad_s"])

    optimizer = cfg["optimizer"]
    solution = refine_native_support_path(
        model,
        mujoco,
        reference["qpos"],
        selected_variables,
        geoms,
        intervals=intervals,
        thresholds_m=thresholds,
        plane_z_m=plane_z_m,
        modified_endpoints=endpoints,
        site_names=TIP_SITES,
        velocity_limits=velocity_limits,
        dt_s=1.0 / float(reference["frequency"]),
        optimization_samples_per_interval=int(
            geometry["optimization_samples_per_interval"]
        ),
        nonlinear_check_samples_per_interval=int(
            geometry["nonlinear_check_samples_per_interval"]
        ),
        activation_margin_m=float(optimizer["linearization_activation_margin_m"]),
        maximum_iterations=int(optimizer["maximum_iterations"]),
        feasibility_tolerance_m=float(optimizer["feasibility_tolerance_m"]),
        initial_trust_region_fraction=float(
            optimizer["initial_trust_region_fraction_of_budget"]
        ),
        minimum_trust_region_fraction=float(
            optimizer["minimum_trust_region_fraction_of_budget"]
        ),
        maximum_trust_region_fraction=float(
            optimizer["maximum_trust_region_fraction_of_budget"]
        ),
        trust_region_shrink=float(optimizer["trust_region_shrink"]),
        trust_region_grow=float(optimizer["trust_region_grow"]),
        weights=optimizer["weights"],
        fingertip_scale_m=float(optimizer["normalization"]["fingertip_position_m"]),
        qp_solver=str(optimizer["qp_solver"]),
        freeze_dofs_from=int(cfg["sample"]["object_qpos_start"]),
    )
    if solution["status"] != "SOLVED":
        write_json(output / "optimization_failure.json", solution)
        return 2

    delta = np.asarray(solution["delta"], dtype=np.float64)
    refined_qpos = apply_delta(reference["qpos"], selected_variables, delta)
    validate_scalar_joint_ranges(model, refined_qpos, selected_variables)
    if not np.array_equal(refined_qpos[:, 36:], reference["qpos"][:, 36:]):
        raise ValueError("optimizer changed frozen object qpos")
    if not np.array_equal(refined_qpos[:5], reference["qpos"][:5]):
        raise ValueError("optimizer changed endpoint 0-4")
    if not np.array_equal(refined_qpos[9:], reference["qpos"][9:]):
        raise ValueError("optimizer changed endpoint 9 or later")

    tip_rows = fingertip_rows(model, reference["qpos"], refined_qpos, endpoints)
    maximum_tip_change = max(float(row["position_change_m"]) for row in tip_rows)
    if maximum_tip_change > float(cfg["preservation_contract"]["fingertip_position_max_change_m"]):
        raise RuntimeError("candidate exceeds the frozen fingertip displacement budget")
    refined = derived_reference(
        model,
        reference,
        human,
        refined_qpos,
        geoms,
        paths["retarget_settings"],
        plane_z_m,
    )
    candidate_path = output / "robot_reference_refined_v2.npz"
    np.savez_compressed(candidate_path, **refined)

    trace_rows: list[dict[str, Any]] = []
    for entry in solution["trace"]:
        for minimum in entry["interval_minima"]:
            trace_rows.append({
                "iteration": entry["iteration"],
                "stage": entry["stage"],
                "status": entry["status"],
                "source_endpoint": minimum["source_endpoint"],
                "outcome_endpoint": minimum["outcome_endpoint"],
                "alpha": f"{minimum['alpha']:.17g}",
                "minimum_distance_m": f"{minimum['distance_m']:.17g}",
                "required_distance_m": f"{thresholds[minimum['source_endpoint']]:.17g}",
                "maximum_violation_m": f"{entry['maximum_violation_m']:.17g}",
                "geom": minimum["geom"],
                "body": minimum["body"],
                "vertex_index": minimum["vertex_index"],
                "trust_region_fraction": f"{entry['trust_region_fraction']:.17g}",
                "qp_step_maximum_budget_fraction": (
                    "" if entry["qp_step_maximum_budget_fraction"] is None
                    else f"{entry['qp_step_maximum_budget_fraction']:.17g}"
                ),
            })
    write_csv(output / "optimization_trace.csv", trace_rows)
    delta_rows = [{
        "candidate": "M4",
        "endpoint": variable.endpoint,
        "joint": variable.joint_name,
        "qpos_address": variable.qpos_address,
        "budget": f"{variable.budget:.17g}",
        "applied_delta": f"{delta[index]:.17g}",
        "budget_fraction": f"{abs(delta[index]) / variable.budget:.17g}",
    } for index, variable in enumerate(selected_variables)]
    write_csv(output / "refinement_delta.csv", delta_rows)
    write_csv(output / "fingertip_contact_comparison.csv", tip_rows)

    interval_rows = []
    final_minima = {row["source_endpoint"]: row for row in solution["interval_minima"]}
    for interval in intervals:
        after = final_minima[interval]
        before = prior[interval]
        interval_rows.append({
            "source_endpoint": interval,
            "outcome_endpoint": interval + 1,
            "required_minimum_distance_m": f"{thresholds[interval]:.17g}",
            "original_minimum_distance_m": before["minimum_distance_m"],
            "original_minimum_alpha": before["minimum_alpha"],
            "refined_internal_minimum_distance_m": f"{after['distance_m']:.17g}",
            "refined_internal_minimum_alpha": f"{after['alpha']:.17g}",
            "refined_internal_geom": after["geom"],
            "audit_status": "AWAITING_INDEPENDENT_10001_SAMPLE_AUDIT",
        })
    write_csv(output / "interval_clearance_comparison.csv", interval_rows)

    objective = exact_objective(
        selected_variables,
        delta,
        tip_rows,
        endpoints,
        optimizer["weights"],
        float(optimizer["normalization"]["fingertip_position_m"]),
    )
    manifest = {
        "schema": "taco_brush_20231006_145_continuous_reference_refinement_v2_input_manifest",
        "status": "FROZEN_INPUTS_VERIFIED_CANDIDATE_EMITTED",
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "base_commit": cfg["base_commit"],
        "random_seed": int(cfg["random_seed"]),
        "software": {
            "python": sys.version,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "qp_solver": optimizer["qp_solver"],
        },
        "inputs": {
            "config": artifact(config_path),
            "script": artifact(Path(__file__)),
            **{
                name: artifact(path)
                for name, path in paths.items()
                if name != "output" and path.is_file()
            },
        },
        "compiled_model": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "timestep_s": float(model.opt.timestep),
            "native_hand_geom_count": len(geoms),
        },
        "authorization": cfg["authorization"],
        "selected_candidate": "M4",
        "candidate_reference": artifact(candidate_path),
    }
    write_json(output / "input_manifest.json", manifest)
    results = {
        "schema": "taco_brush_20231006_145_continuous_reference_refinement_v2",
        "classification": "CANDIDATE_EMITTED_AWAITING_INDEPENDENT_AUDIT",
        "selected_candidate": "M4",
        "candidate_sequence": candidate_records,
        "optimizer_status": solution["status"],
        "optimizer_iterations": solution["iterations"],
        "objective": objective,
        "maximum_fingertip_change_m": maximum_tip_change,
        "maximum_joint_budget_fraction": float(
            np.max(np.abs(delta) / np.asarray([item.budget for item in selected_variables]))
        ),
        "internal_interval_minima": solution["interval_minima"],
        "refined_reference": artifact(candidate_path),
        "original_reference_unchanged": (
            file_sha256(paths["robot_reference"])
            == expected["robot_reference_sha256"]
        ),
        "independent_geometry_audit": "NOT_RUN",
        "candidate_physics_replay": "NOT_RUN",
    }
    write_json(output / "results.json", results)
    (output / "reproduction_commands.txt").write_text(
        "PYTHONPATH=/tmp/egoengine_mujoco_370.3FmVPI:rl/src "
        "/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python "
        "rl/scripts/refine_taco_brush_20231006_145_reference_v2.py "
        "--config rl/configs/taco_brush_20231006_145_continuous_reference_refinement_v2.yaml\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "classification": results["classification"],
        "selected_candidate": "M4",
        "maximum_fingertip_change_m": maximum_tip_change,
        "internal_interval_minima": solution["interval_minima"],
        "candidate_reference": str(candidate_path),
    }, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
