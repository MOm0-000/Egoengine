#!/usr/bin/env python3
"""Independently audit the M4 refined reference after serialization."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_world_objects,
    pair_group_audit,
)
from egoengine_repro.retarget.collision_audit import (  # noqa: E402
    collision_families,
    distances,
    explicit_hand_pairs,
    hand_ids,
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
    differentiate_trajectory,
    file_sha256,
    fingertip_positions,
    interval_clearance_minima,
)
from run_taco_brush_issue14_mink_candidate_v1 import (  # noqa: E402
    relevant_pairs,
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def interval_thresholds(cfg: dict[str, Any], summary_path: Path) -> dict[int, float]:
    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        prior = {
            int(row["source_endpoint"]): row for row in csv.DictReader(stream)
        }
    geometry = cfg["geometry_contract"]
    result = {
        int(interval): float(geometry["core_minimum_clearance_m"])
        for interval in geometry["core_intervals"]
    }
    for interval in geometry["transition_intervals"]:
        interval = int(interval)
        result[interval] = max(
            float(geometry["transition_floor_m"]),
            float(prior[interval]["minimum_distance_m"])
            - float(geometry["transition_nonworsening_allowance_m"]),
        )
    return result


def wrist_changes(
    model: mujoco.MjModel,
    original: np.ndarray,
    refined: np.ndarray,
    endpoints: list[int],
) -> list[dict[str, Any]]:
    before = mujoco.MjData(model)
    after = mujoco.MjData(model)
    rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        before.qpos[:] = original[endpoint]
        after.qpos[:] = refined[endpoint]
        mujoco.mj_forward(model, before)
        mujoco.mj_forward(model, after)
        for side in SIDES:
            body = int(model.body(f"{side}_hand_link").id)
            old_rotation = np.asarray(before.xmat[body], dtype=np.float64).reshape(3, 3)
            new_rotation = np.asarray(after.xmat[body], dtype=np.float64).reshape(3, 3)
            rows.append({
                "endpoint": endpoint,
                "side": side,
                "translation_change_m": float(
                    np.linalg.norm(after.xpos[body] - before.xpos[body])
                ),
                "rotation_change_rad": float(
                    Rotation.from_matrix(old_rotation.T @ new_rotation).magnitude()
                ),
            })
    return rows


def self_collision_comparison(
    model: mujoco.MjModel,
    original: np.ndarray,
    refined: np.ndarray,
    endpoints: list[int],
    tolerance_m: float,
) -> dict[str, Any]:
    explicit = set(explicit_hand_pairs(model))
    pairs = []
    classifications = []
    for first, second in combinations(hand_ids(model), 2):
        first_name = model.geom(first).name
        second_name = model.geom(second).name
        if first_name.split("_")[2] != second_name.split("_")[2]:
            continue
        pairs.append((first, second))
        classifications.append("declared" if (first, second) in explicit else "omitted")
    before_data = mujoco.MjData(model)
    after_data = mujoco.MjData(model)
    rows = []
    blockers = []
    for endpoint in endpoints:
        before_data.qpos[:] = original[endpoint]
        after_data.qpos[:] = refined[endpoint]
        mujoco.mj_forward(model, before_data)
        mujoco.mj_forward(model, after_data)
        before_values = distances(model, before_data, pairs, detection=0.05)
        after_values = distances(model, after_data, pairs, detection=0.05)
        for pair_index, (first, second) in enumerate(pairs):
            before_value = float(before_values[pair_index])
            after_value = float(after_values[pair_index])
            newly_beyond_tolerance = (
                before_value >= -tolerance_m and after_value < -tolerance_m
            )
            worsened_over_allowance = after_value < before_value - tolerance_m
            row = {
                "endpoint": endpoint,
                "geom1": model.geom(first).name,
                "geom2": model.geom(second).name,
                "classification": classifications[pair_index],
                "original_distance_m": before_value,
                "refined_distance_m": after_value,
                "distance_change_m": after_value - before_value,
                "newly_beyond_tolerance": newly_beyond_tolerance,
                "worsened_over_allowance": worsened_over_allowance,
            }
            rows.append(row)
            if newly_beyond_tolerance or worsened_over_allowance:
                blockers.append(row)
    return {
        "status": "PASS" if not blockers else "FAIL",
        "pair_endpoint_count": len(rows),
        "blockers": blockers,
        "minimum_original_distance_m": min(row["original_distance_m"] for row in rows),
        "minimum_refined_distance_m": min(row["refined_distance_m"] for row in rows),
    }


def hand_object_comparison(
    model: mujoco.MjModel,
    original: np.ndarray,
    refined: np.ndarray,
    endpoints: list[int],
    tolerance_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = collision_families(model)
    groups = {
        "brush_right_hand": relevant_pairs(
            model, families["hand_tool"], "right", "right"
        ),
        "bowl_left_hand": relevant_pairs(
            model, families["hand_target"], "left", "left"
        ),
    }
    data = mujoco.MjData(model)
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for endpoint in endpoints:
        measurements: dict[tuple[str, str], dict[str, Any]] = {}
        for variant, state in (("original", original), ("refined", refined)):
            data.qpos[:] = state[endpoint]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            native = native_world_objects(model, data)
            for group, pairs in groups.items():
                measurements[(variant, group)] = pair_group_audit(
                    model, data, native, pairs, tolerance_m
                )
        for group in groups:
            before = float(
                measurements[("original", group)]["native_minimum"]["distance_m"]
            )
            after = float(
                measurements[("refined", group)]["native_minimum"]["distance_m"]
            )
            row = {
                "endpoint": endpoint,
                "group": group,
                "original_minimum_distance_m": before,
                "refined_minimum_distance_m": after,
                "distance_change_m": after - before,
                "worsened_over_allowance": after < before - tolerance_m,
                "original_unknown_pair_count": len(
                    measurements[("original", group)]["unknown_pairs"]
                ),
                "refined_unknown_pair_count": len(
                    measurements[("refined", group)]["unknown_pairs"]
                ),
            }
            rows.append(row)
            if row["worsened_over_allowance"] or row["refined_unknown_pair_count"]:
                blockers.append(row)
    return rows, blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != "taco_brush_20231006_145_continuous_reference_refinement_v2":
        raise ValueError("unexpected refinement config schema")
    paths = {name: Path(value).resolve() for name, value in cfg["paths"].items()}
    output = paths["output"]
    candidate_path = (output / "robot_reference_refined_v2.npz").resolve(strict=True)
    if (output / "native_geometry_audit.json").exists():
        raise FileExistsError(output / "native_geometry_audit.json")
    if file_sha256(paths["robot_reference"]) != cfg["expected_inputs"]["robot_reference_sha256"]:
        raise ValueError("original robot reference hash changed")
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        original = {name: np.asarray(archive[name]).copy() for name in archive.files}
    with np.load(candidate_path, allow_pickle=False) as archive:
        refined = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if set(original) != set(refined):
        raise ValueError("refined reference field set changed")

    qpos_original = original["qpos"]
    qpos_refined = refined["qpos"]
    endpoints = [int(value) for value in cfg["modified_endpoints"]]
    m4 = next(row for row in cfg["candidate_order"] if row["id"] == "M4")
    allowed_addresses = {
        int(model.joint(name).qposadr[0]) for name in m4["authorized_joints"]
    }
    changed = np.argwhere(qpos_refined != qpos_original)
    unauthorized_changes = [
        [int(endpoint), int(address)]
        for endpoint, address in changed
        if int(endpoint) not in endpoints or int(address) not in allowed_addresses
    ]
    object_exact = bool(np.array_equal(qpos_refined[:, 36:], qpos_original[:, 36:]))
    timestamps_exact = bool(np.array_equal(refined["timestamps_s"], original["timestamps_s"]))
    frequency_exact = bool(np.array_equal(refined["frequency"], original["frequency"]))
    ctrl_exact = bool(np.array_equal(refined["ctrl"], qpos_refined[:, :36]))
    derived_qvel = differentiate_trajectory(
        model, mujoco, qpos_refined, refined["timestamps_s"]
    )
    qvel_consistent = bool(
        np.allclose(refined["qvel"], derived_qvel, atol=1e-12, rtol=0.0)
    )

    thresholds = interval_thresholds(cfg, paths["prior_dense_interval_summary"])
    intervals = sorted(thresholds)
    geometry = cfg["geometry_contract"]
    geoms = build_native_support_geoms(
        model, mujoco, native_hand_visual_geom_ids(model, mujoco)
    )
    sample_count = int(geometry["independent_audit_samples_per_interval"])
    phase_count = int(geometry["independent_phase_offset_samples_per_interval"])
    primary = interval_clearance_minima(
        model,
        mujoco,
        qpos_refined,
        geoms,
        intervals=intervals,
        alphas=np.linspace(0.0, 1.0, sample_count),
        plane_z_m=float(geometry["table_z_m"]),
    )
    phase = interval_clearance_minima(
        model,
        mujoco,
        qpos_refined,
        geoms,
        intervals=intervals,
        alphas=(np.arange(phase_count, dtype=np.float64) + 0.5) / phase_count,
        plane_z_m=float(geometry["table_z_m"]),
    )
    clearance_rows: list[dict[str, Any]] = []
    clearance_blockers: list[dict[str, Any]] = []
    for grid_name, minima in (("primary", primary), ("phase_offset", phase)):
        for row in minima:
            threshold = thresholds[int(row["source_endpoint"])]
            audit_row = {
                "grid": grid_name,
                "source_endpoint": row["source_endpoint"],
                "outcome_endpoint": row["outcome_endpoint"],
                "alpha": row["alpha"],
                "minimum_distance_m": row["distance_m"],
                "required_distance_m": threshold,
                "margin_m": row["distance_m"] - threshold,
                "geom": row["geom"],
                "body": row["body"],
                "vertex_index": row["vertex_index"],
                "pass": row["distance_m"] >= threshold - 1e-12,
            }
            clearance_rows.append(audit_row)
            if not audit_row["pass"]:
                clearance_blockers.append(audit_row)
    write_csv(output / "independent_interval_clearance.csv", clearance_rows)

    tips_before = fingertip_positions(
        model, mujoco, qpos_original, endpoints=endpoints, site_names=TIP_SITES
    )
    tips_after = fingertip_positions(
        model, mujoco, qpos_refined, endpoints=endpoints, site_names=TIP_SITES
    )
    maximum_tip_change = float(
        np.linalg.norm(tips_after - tips_before, axis=-1).max()
    )
    wrists = wrist_changes(model, qpos_original, qpos_refined, endpoints)
    maximum_wrist_translation = max(row["translation_change_m"] for row in wrists)
    maximum_wrist_rotation = max(row["rotation_change_rad"] for row in wrists)

    settings, consumption = load_taco_bimanual_settings(paths["retarget_settings"])
    if consumption["unknown_key_count"] or consumption["unused_key_count"]:
        raise ValueError("retarget settings are not exactly consumed")
    velocity_limits = joint_velocity_limits(model, mujoco, settings.velocity_limits)
    names = tuple(velocity_limits)
    addresses = np.asarray(
        [int(model.joint(name).qposadr[0]) for name in names], dtype=np.int64
    )
    speeds = np.asarray([velocity_limits[name] for name in names], dtype=np.float64)
    ranges = np.asarray([model.joint(name).range for name in names], dtype=np.float64)
    dt_s = 1.0 / float(refined["frequency"])
    joint_margin = float(np.minimum(
        qpos_refined[:, addresses] - ranges[:, 0],
        ranges[:, 1] - qpos_refined[:, addresses],
    ).min())
    velocity_ratio = np.abs(np.diff(qpos_refined[:, addresses], axis=0)) / (dt_s * speeds)
    maximum_velocity_ratio = float(velocity_ratio.max())

    self_collision = self_collision_comparison(
        model,
        qpos_original,
        qpos_refined,
        endpoints,
        float(cfg["kinematic_contract"]["independent_self_collision_tolerance_m"]),
    )
    hand_object_rows, hand_object_blockers = hand_object_comparison(
        model,
        qpos_original,
        qpos_refined,
        endpoints,
        float(
            cfg["modification_budgets"][
                "hand_object_native_penetration_nonworsening_m"
            ]
        ),
    )
    write_csv(output / "hand_object_native_comparison.csv", hand_object_rows)

    checks = {
        "authorized_qpos_changes_only": not unauthorized_changes,
        "object_qpos_exactly_frozen": object_exact,
        "timestamps_exactly_frozen": timestamps_exact,
        "frequency_exactly_frozen": frequency_exact,
        "control_matches_robot_qpos": ctrl_exact,
        "qvel_recomputed_from_qpos": qvel_consistent,
        "two_dense_clearance_grids_pass": not clearance_blockers,
        "fingertip_budget_pass": maximum_tip_change
        <= float(cfg["preservation_contract"]["fingertip_position_max_change_m"]),
        "wrist_translation_budget_pass": maximum_wrist_translation
        <= float(cfg["preservation_contract"]["wrist_translation_max_change_m"]),
        "wrist_rotation_budget_pass": maximum_wrist_rotation
        <= float(cfg["preservation_contract"]["wrist_rotation_max_change_rad"]),
        "joint_limits_pass": joint_margin >= -1e-12,
        "original_30hz_velocity_limits_pass": maximum_velocity_ratio <= 1.0 + 1e-6,
        "self_collision_nonworsening_pass": self_collision["status"] == "PASS",
        "hand_object_native_nonworsening_pass": not hand_object_blockers,
    }
    passed = all(checks.values())
    audit = {
        "schema": "taco_brush_20231006_145_refined_reference_independent_audit_v2",
        "status": "PASS" if passed else "FAIL",
        "candidate_reference": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
        },
        "method": {
            "geometry": "full native hand visual mesh vertices",
            "primary_samples_per_interval": sample_count,
            "phase_offset_samples_per_interval": phase_count,
            "primary_grid_includes_endpoints": True,
            "phase_grid": "(index + 0.5) / sample_count",
            "finite_sampling_claim_only": True,
        },
        "checks": checks,
        "unauthorized_qpos_changes": unauthorized_changes,
        "clearance_blockers": clearance_blockers,
        "clearance_rows": clearance_rows,
        "maximum_fingertip_change_m": maximum_tip_change,
        "wrist_changes": wrists,
        "maximum_wrist_translation_m": maximum_wrist_translation,
        "maximum_wrist_rotation_rad": maximum_wrist_rotation,
        "joint_limit_minimum_margin": joint_margin,
        "maximum_original_30hz_velocity_ratio": maximum_velocity_ratio,
        "self_collision": self_collision,
        "hand_object_blockers": hand_object_blockers,
        "hand_object_rows": hand_object_rows,
        "original_reference_sha256_after_audit": file_sha256(paths["robot_reference"]),
    }
    write_json(output / "native_geometry_audit.json", audit)
    print(json.dumps({
        "status": audit["status"],
        "checks": checks,
        "minimum_clearance_m": min(row["minimum_distance_m"] for row in clearance_rows),
        "maximum_fingertip_change_m": maximum_tip_change,
        "maximum_velocity_ratio": maximum_velocity_ratio,
        "hand_object_blocker_count": len(hand_object_blockers),
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
