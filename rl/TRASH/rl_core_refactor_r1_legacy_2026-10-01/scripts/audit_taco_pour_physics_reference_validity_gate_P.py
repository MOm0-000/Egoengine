#!/usr/bin/env python3
"""Finite read-only Gate P for Pour physics, reference and actuation validity."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]

from audit_taco_pour_action_feasibility_minimum_rho import (  # noqa: E402
    compact_source_state,
)
from audit_taco_pour_binary_translation_last_off_reversal import (  # noqa: E402
    reproduce_baseline,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
)
from audit_taco_pour_source45_state_entry import state_record  # noqa: E402
from audit_taco_pour_source57_temporal_hold_gate import _assert_scores_exact  # noqa: E402
from audit_taco_pour_tail_semantic_suppression_oracle import (  # noqa: E402
    evaluate_candidates,
    load_contract as load_tail_contract,
    restore_semantic_choice,
)
from audit_taco_pour_low_level_contact_controllability_gate_L import (  # noqa: E402
    actuator_force_law,
    build_env,
    urdf_effort_limits,
)


HAND_QPOS = slice(0, 36)
HAND_QVEL = slice(0, 36)
TOOL_QPOS = slice(36, 43)
TOOL_QVEL = slice(36, 42)
P0_SOURCES = (45, 50, 57)
P2_SOURCES = (56, 57, 58)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_physics_reference_validity_gate_P_v1":
        raise ValueError("unsupported Gate P contract")
    if contract.get("status") != "authorized_finite_read_only_gate_P":
        raise ValueError("Gate P is not authorized")
    runtime = contract["runtime"]
    stopping = contract["stopping_rule"]
    if (
        contract.get("paper_faithful") is not False
        or runtime != {
            "training_allowed": False,
            "optimizer_updates_policy": False,
            "source57_action_or_controller_counterfactual_allowed": False,
            "model_parameter_mutation_allowed": False,
            "mass_friction_effort_sweep_allowed": False,
            "reward_or_objective_change_allowed": False,
            "chunk_acceptance_allowed": False,
            "chunk_commit_allowed": False,
            "order": ["P1", "P2", "P0", "P3", "P4"],
        }
        or stopping != {
            "Gate_P_closes_after_P1_P2_P0_P3_P4": True,
            "further_source57_action_controller_gain_force_axis_frame_or_per_source_search_allowed": False,
            "Gate_B_allowed_only_if_P0_through_P4_show_no_structural_issue": True,
            "Gate_C_allowed_only_if_P0_through_P4_show_no_structural_issue": True,
            "PPO_retraining_allowed": False,
            "actor_learning_rate_5e_minus_5_allowed": False,
            "learned_gate_allowed": False,
            "reward_change_allowed": False,
            "chunk_commit_allowed": False,
        }
        or contract["P5_objective_contract"]["status"]
        != "external_author_information_required"
        or contract["P5_objective_contract"][
            "local_parameter_recovery_experiment_allowed"
        ] is not False
    ):
        raise ValueError("Gate P definition changed")
    paths: dict[str, Path] = {}
    for name, row in contract["inputs"].items():
        artifact = Path(row["path"])
        if sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
        paths[name] = artifact
    return contract, paths, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    import mujoco

    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    value = np.clip((np.trace(first @ second.T) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(value))


def _tensor(data, field: str) -> np.ndarray:
    import warp as wp

    return wp.to_torch(getattr(data, field)).detach().cpu().numpy().copy()


def _geom_name(model, geom: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or f"geom#{geom}"


def _is_right_hand_tool_pair(names: tuple[str, str]) -> bool:
    hand = any(name.startswith("collision_hand_right") for name in names)
    tool = any(name.startswith("right_object_") for name in names)
    return hand and tool


def mjwp_contacts(env, *, only_right_tool: bool = False) -> list[dict]:
    """Decode live MJWP contacts without treating constraint-row sums as normal force."""
    import mujoco
    import warp as wp

    contact = env.env.data_wp.contact
    efc = env.env.data_wp.efc
    nacon = int(_tensor(env.env.data_wp, "nacon")[0])
    geom = wp.to_torch(contact.geom).detach().cpu().numpy().astype(np.int64)
    pos = wp.to_torch(contact.pos).detach().cpu().numpy().astype(np.float64)
    frame = wp.to_torch(contact.frame).detach().cpu().numpy().astype(np.float64)
    dist = wp.to_torch(contact.dist).detach().cpu().numpy().astype(np.float64)
    dim = wp.to_torch(contact.dim).detach().cpu().numpy().astype(np.int64)
    friction = wp.to_torch(contact.friction).detach().cpu().numpy().astype(np.float64)
    addresses = wp.to_torch(contact.efc_address).detach().cpu().numpy().astype(np.int64)
    world = wp.to_torch(contact.worldid).detach().cpu().numpy().astype(np.int64)
    efc_force = wp.to_torch(efc.force).detach().cpu().numpy().astype(np.float64)
    rows = []
    for index in range(min(nacon, len(geom))):
        if world[index] != 0 or dist[index] > 0.0:
            continue
        pair = (int(geom[index, 0]), int(geom[index, 1]))
        names = (_geom_name(env.env.model_cpu, pair[0]), _geom_name(env.env.model_cpu, pair[1]))
        if only_right_tool and not _is_right_hand_tool_pair(names):
            continue
        valid = addresses[index][
            (addresses[index] >= 0) & (addresses[index] < efc_force.shape[1])
        ]
        dimension = int(dim[index])
        decoded = np.zeros(max(dimension, 1), dtype=np.float64)
        raw = efc_force[0, valid] if len(valid) else np.zeros(0, dtype=np.float64)
        if dimension == 1 and len(raw):
            decoded[0] = raw[0]
        elif dimension > 1 and len(raw) == 2 * (dimension - 1):
            mujoco.mju_decodePyramid(
                decoded,
                np.asarray(raw, dtype=np.float64),
                np.asarray(friction[index, : dimension - 1], dtype=np.float64),
            )
        elif len(raw):
            decoded[: min(len(decoded), len(raw))] = raw[: len(decoded)]
        rotation = np.asarray(frame[index], dtype=np.float64).reshape(3, 3)
        vector_world = rotation[:dimension].T @ decoded[:dimension]
        rows.append({
            "contact_index": index,
            "geom_ids": list(pair),
            "geom_names": list(names),
            "position_world_m": pos[index].tolist(),
            "frame_world": rotation.tolist(),
            "normal_geom1_to_geom2_world": rotation[0].tolist(),
            "penetration_m": float(-dist[index]),
            "distance_m": float(dist[index]),
            "condim": dimension,
            "raw_constraint_forces": raw.tolist(),
            "contact_force_frame": decoded.tolist(),
            "normal_force_n": float(decoded[0]),
            "tangent_force_norm_n": float(np.linalg.norm(decoded[1:])),
            "contact_force_world_n": vector_world.tolist(),
        })
    return rows


def native_contacts(model, data, *, only_right_tool: bool = False) -> list[dict]:
    import mujoco

    rows = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if float(contact.dist) > 0.0:
            continue
        pair = (int(contact.geom1), int(contact.geom2))
        names = (_geom_name(model, pair[0]), _geom_name(model, pair[1]))
        if only_right_tool and not _is_right_hand_tool_pair(names):
            continue
        force6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force6)
        rotation = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        vector_world = rotation.T @ force6[:3]
        rows.append({
            "contact_index": index,
            "geom_ids": list(pair),
            "geom_names": list(names),
            "position_world_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
            "frame_world": rotation.tolist(),
            "normal_geom1_to_geom2_world": rotation[0].tolist(),
            "penetration_m": float(-contact.dist),
            "distance_m": float(contact.dist),
            "condim": int(contact.dim),
            "contact_force_frame": force6.tolist(),
            "normal_force_n": float(force6[0]),
            "tangent_force_norm_n": float(np.linalg.norm(force6[1:3])),
            "contact_force_world_n": vector_world.tolist(),
        })
    return rows


def policy_action(policy, env) -> tuple[np.ndarray, object]:
    packed = policy.obs_to_tensors(env.current_observation())
    result = policy.get_deterministic_action_values(packed)
    action = policy.preprocess_actions(result["deterministic_actions"])
    return np.asarray(action, np.float32), clone_hidden(result["rnn_states"])


def action_to_ctrl(env, action: np.ndarray) -> np.ndarray:
    reference = env._reference_ctrls(env.time_indices, offset=1)
    if env._state_feasible_action_contract is not None:
        reference, _ = env._snap_state_feasible_reference(reference)
    delta = torch.as_tensor(action, dtype=torch.float32, device=str(env.ego_cfg.device))
    return env._apply_residual(reference, delta)[0].detach().cpu().numpy().copy()


def capture_gate_P_states(*, backend, policy, boundary, parent_paths, tail_report,
                          reference_qpos, reference_qvel, objective) -> dict[int, dict]:
    captures, _ = reproduce_baseline(
        backend=backend,
        policy=policy,
        boundary=boundary,
        reference_qpos=reference_qpos,
        reference_qvel=reference_qvel,
        objective=objective,
        expected_arrays=parent_paths["baseline_oracle_arrays"],
        expected_off_sources=json.loads(
            parent_paths["baseline_oracle_report"].read_text()
        )["oracle_result"]["OFF_sources"],
        capture_sources=(44,),
    )
    backend.restore(captures[44]["snapshot"])
    backend.verify_restored_snapshot(captures[44]["snapshot"])
    policy.rnn_states = clone_hidden(captures[44]["pre_hidden"])
    result: dict[int, dict] = {}
    for source in range(44, 58):
        if source <= 56:
            evaluated = evaluate_candidates(
                backend=backend,
                policy=policy,
                source=source,
                reference_qpos=reference_qpos,
                reference_qvel=reference_qvel,
                objective=objective,
            )
            expected = (
                tail_report["tail_decisions"][source - 44]
                if source <= 55 else tail_report["source56_fork"]
            )
            _assert_scores_exact(evaluated, expected)
            expected_selected = (
                expected["selected"] if source <= 55
                else expected["selected_for_continuation"]
            )
            if evaluated["selected"]["name"] != expected_selected:
                raise RuntimeError(f"tail selected mode changed at source {source}")
            full_action = np.asarray(
                evaluated["actor"]["complete_deterministic_action"], np.float32
            )[None]
            result[source] = {
                "snapshot": evaluated["pre_snapshot"],
                "pre_hidden": evaluated["pre_hidden"],
                "post_hidden": evaluated["post_hidden"],
                "full_action": full_action,
                "full_ctrl": action_to_ctrl(backend.env, full_action),
                "source_state": evaluated["source_state"],
                "contacts": mjwp_contacts(backend.env, only_right_tool=True),
            }
            restore_semantic_choice(
                backend=backend,
                policy=policy,
                evaluated=evaluated,
                candidate=evaluated["selected"],
            )
        else:
            snapshot = backend.snapshot()
            pre_hidden = clone_hidden(policy.rnn_states)
            full_action, post_hidden = policy_action(policy, backend.env)
            result[source] = {
                "snapshot": snapshot,
                "pre_hidden": pre_hidden,
                "post_hidden": post_hidden,
                "full_action": full_action,
                "full_ctrl": action_to_ctrl(backend.env, full_action),
                "source_state": compact_source_state(
                    state_record(backend.env), source,
                    reference_qpos, reference_qvel, objective,
                ),
                "contacts": mjwp_contacts(backend.env, only_right_tool=True),
            }
            # The tail-oracle continuation selected ON at source57.  Reproduce
            # that exact branch only to capture the endpoint58 contact state.
            backend.step(full_action, source)
            policy.rnn_states = clone_hidden(post_hidden)
            result[58] = {
                "snapshot": backend.snapshot(),
                "source_state": compact_source_state(
                    state_record(backend.env), 58,
                    reference_qpos, reference_qvel, objective,
                ),
                "contacts": mjwp_contacts(backend.env, only_right_tool=True),
            }
    return result


def actuator_groups(names: list[str]) -> dict[str, list[int]]:
    groups = {
        "right_wrist_translation": [], "right_wrist_rotation": [],
        "right_thumb": [], "right_index": [], "right_middle": [],
        "right_ring": [], "right_pinky": [],
        "left_wrist_translation": [], "left_wrist_rotation": [],
        "left_thumb": [], "left_index": [], "left_middle": [],
        "left_ring": [], "left_pinky": [],
    }
    for index, name in enumerate(names):
        lower = name.lower()
        side = "right" if lower.startswith("r_") or lower.startswith("right_") else "left"
        if "forearm_t" in lower:
            groups[f"{side}_wrist_translation"].append(index)
        elif "forearm_" in lower:
            groups[f"{side}_wrist_rotation"].append(index)
        else:
            finger = next((finger for finger in FINGERS if finger in lower), None)
            if finger is None:
                raise ValueError(f"cannot group actuator {name}")
            groups[f"{side}_{finger}"].append(index)
    if sorted(index for values in groups.values() for index in values) != list(range(36)):
        raise ValueError("actuator groups do not partition the 36 controls")
    return groups


def _maximum_normal_force(*, model, data, contact_rows: list[dict],
                          qvel_addresses: np.ndarray, limits: np.ndarray,
                          selected_actuators: np.ndarray) -> dict:
    import mujoco
    from scipy.optimize import linprog

    columns = []
    per_contact = []
    selected_dofs = qvel_addresses[selected_actuators]
    selected_limits = limits[selected_actuators]
    for row in contact_rows:
        g0, g1 = row["geom_ids"]
        n0, n1 = row["geom_names"]
        hand_geom = g0 if n0.startswith("collision_hand_right") else g1
        hand_body = int(model.geom_bodyid[hand_geom])
        point = np.asarray(row["position_world_m"], dtype=np.float64)
        normal = np.asarray(row["normal_geom1_to_geom2_world"], dtype=np.float64)
        # Sign does not affect a symmetric effort envelope.
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp, jacr, point, hand_body)
        coefficients = jacp[:, selected_dofs].T @ normal
        ratios = np.divide(
            selected_limits, np.abs(coefficients),
            out=np.full_like(selected_limits, np.inf),
            where=np.abs(coefficients) > 1.0e-12,
        )
        maximum = float(np.min(ratios))
        columns.append(coefficients)
        per_contact.append({
            "geom_names": row["geom_names"],
            "position_world_m": row["position_world_m"],
            "normal_world": normal.tolist(),
            "maximum_single_contact_normal_force_n": maximum,
            "limiting_actuator": int(selected_actuators[int(np.argmin(ratios))]),
        })
    matrix = np.stack(columns, axis=1)
    # Maximize the sum of nonnegative normal forces at the two observed
    # constraints under one shared effort envelope.
    result = linprog(
        c=-np.ones(matrix.shape[1], dtype=np.float64),
        A_ub=np.vstack([matrix, -matrix]),
        b_ub=np.concatenate([selected_limits, selected_limits]),
        bounds=[(0.0, None)] * matrix.shape[1],
        method="highs",
    )
    return {
        "selected_actuator_indices": selected_actuators.tolist(),
        "individual_contacts": per_contact,
        "maximum_aggregate_normal_force_n": (
            float(np.sum(result.x)) if result.success else None
        ),
        "aggregate_LP_success": bool(result.success),
        "aggregate_contact_force_solution_n": (
            np.asarray(result.x, dtype=np.float64).tolist() if result.success else None
        ),
        "status": result.message,
    }


def audit_P1(*, paths: dict[str, Path], gate_L_report: dict, law: dict,
             model, source57: dict) -> tuple[dict, dict[str, np.ndarray]]:
    import mujoco

    arrays = np.load(paths["Gate_L_rollouts"])
    names = list(gate_L_report["diagnostic_model"]["actuator_names"])
    limits = np.asarray(gate_L_report["diagnostic_model"]["URDF_effort_limits"], np.float64)
    groups = actuator_groups(names)
    report_rows = {}
    saved: dict[str, np.ndarray] = {"effort_limit": limits}
    for anchor in ("formal_frozen_PPO", "gate_A1_best_position_velocity_feedback"):
        requested = np.asarray(arrays[f"L0_{anchor}_requested_effort"], np.float64)
        ratio = np.abs(requested) / limits[None]
        saturated = ratio > 1.0 + 1.0e-12
        if requested.shape != (30, 36):
            raise ValueError("Gate L requested-effort trace shape changed")
        grouped = {}
        for name, indices in groups.items():
            selection = saturated[:, indices]
            grouped[name] = {
                "actuator_indices": indices,
                "saturated_requests": int(selection.sum()),
                "total_requests": int(selection.size),
                "saturated_fraction": float(selection.mean()),
                "maximum_requested_to_limit_ratio": float(np.max(ratio[:, indices])),
            }
        report_rows[anchor] = {
            "saturated_requests": int(saturated.sum()),
            "total_requests": int(saturated.size),
            "saturated_fraction": float(saturated.mean()),
            "per_actuator_saturated_requests": saturated.sum(axis=0).astype(int).tolist(),
            "per_actuator_maximum_requested_to_limit_ratio": ratio.max(axis=0).tolist(),
            "groups": grouped,
        }
        saved[f"{anchor}_requested_effort"] = requested
        saved[f"{anchor}_requested_to_limit_ratio"] = ratio
        saved[f"{anchor}_saturated"] = saturated

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(source57["snapshot"]["qpos"])[0]
    data.qvel[:] = np.asarray(source57["snapshot"]["qvel"])[0]
    mujoco.mj_forward(model, data)
    contact_rows = source57["contacts"]
    pinky_rows = [
        row for row in contact_rows
        if "collision_hand_right_pinky_0" in row["geom_names"]
        and any(name in {"right_object_9", "right_object_23"} for name in row["geom_names"])
    ]
    if len(pinky_rows) != 2:
        raise RuntimeError("P1 requires the two exact source57 pinky contacts")
    qvel_addresses = np.asarray(law["qvel_address"], np.int64)
    finger_only = _maximum_normal_force(
        model=model, data=data, contact_rows=pinky_rows,
        qvel_addresses=qvel_addresses, limits=limits,
        selected_actuators=np.arange(6, 18, dtype=np.int64),
    )
    wrist_and_finger = _maximum_normal_force(
        model=model, data=data, contact_rows=pinky_rows,
        qvel_addresses=qvel_addresses, limits=limits,
        selected_actuators=np.arange(0, 18, dtype=np.int64),
    )
    measured_sum = float(sum(row["normal_force_n"] for row in pinky_rows))
    return {
        "anchors": report_rows,
        "all_observed_task_saturation_is_on_finger_actuators": bool(
            all(
                sum(row["per_actuator_saturated_requests"][:6]) == 0
                and sum(row["per_actuator_saturated_requests"][18:24]) == 0
                for row in report_rows.values()
            )
        ),
        "source57_measured_two_constraint_force_sum_n": measured_sum,
        "source57_normal_force_capacity": {
            "finger_only": finger_only,
            "right_wrist_plus_fingers": wrist_and_finger,
        },
        "measured_sum_exceeds_finger_only_aggregate_bound": bool(
            finger_only["maximum_aggregate_normal_force_n"] is not None
            and measured_sum > finger_only["maximum_aggregate_normal_force_n"]
        ),
    }, saved


def _load_bowl_meshes(paths: dict[str, Path]):
    import trimesh

    visual = trimesh.load(paths["bowl_visual_mesh"], force="mesh", process=True)
    visual.apply_scale(0.01)
    part9 = trimesh.load(paths["bowl_convex_part_9"], force="mesh", process=True)
    part23 = trimesh.load(paths["bowl_convex_part_23"], force="mesh", process=True)
    return visual, part9, part23


def _mesh_projection(mesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    nearest, distance, face = trimesh.proximity.closest_point(mesh, points)
    normals = np.asarray(mesh.face_normals[np.asarray(face, np.int64)], np.float64)
    return nearest, np.asarray(distance, np.float64), normals


def audit_P2(*, contract: dict, paths: dict[str, Path], states: dict[int, dict]) -> tuple[dict, dict[str, np.ndarray]]:
    import trimesh

    visual, part9, part23 = _load_bowl_meshes(paths)
    rows = {}
    saved: dict[str, np.ndarray] = {}
    for source in P2_SOURCES:
        state = states[source]
        qpos = np.asarray(state["snapshot"]["qpos"])[0]
        position = qpos[36:39]
        rotation = _rotation_matrix_wxyz(qpos[39:43])
        contacts = [
            row for row in state["contacts"]
            if "collision_hand_right_pinky_0" in row["geom_names"]
            and any(name in {"right_object_9", "right_object_23"} for name in row["geom_names"])
        ]
        contact_rows = []
        local_points = []
        outward_normals = []
        for row in contacts:
            point_world = np.asarray(row["position_world_m"], np.float64)
            point_local = rotation.T @ (point_world - position)
            normal = np.asarray(row["normal_geom1_to_geom2_world"], np.float64)
            object_is_first = row["geom_names"][0].startswith("right_object_")
            outward_world = normal if object_is_first else -normal
            outward_local = rotation.T @ outward_world
            local_points.append(point_local)
            outward_normals.append(outward_local)
            contact_rows.append(row | {
                "position_object_local_m": point_local.tolist(),
                "object_outward_normal_local": outward_local.tolist(),
            })
        if local_points:
            local = np.asarray(local_points, np.float64)
            nearest, visual_distance, visual_normal = _mesh_projection(visual, local)
            _, distance9, _ = _mesh_projection(part9, local)
            _, distance23, _ = _mesh_projection(part23, local)
        else:
            local = np.empty((0, 3), np.float64)
            nearest = np.empty((0, 3), np.float64)
            visual_distance = distance9 = distance23 = np.empty(0, np.float64)
            visual_normal = np.empty((0, 3), np.float64)
        for index, row in enumerate(contact_rows):
            row["visual_nearest_point_object_local_m"] = nearest[index].tolist()
            row["visual_projection_distance_m"] = float(visual_distance[index])
            row["visual_outward_normal_object_local"] = visual_normal[index].tolist()
            row["distance_to_part9_surface_m"] = float(distance9[index])
            row["distance_to_part23_surface_m"] = float(distance23[index])
        pair_metrics = None
        same_patch = False
        if len(contact_rows) == 2:
            separation = float(np.linalg.norm(local[0] - local[1]))
            dot = float(np.clip(np.dot(outward_normals[0], outward_normals[1]), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(dot)))
            thresholds = contract["P2_contact_manifold"]["same_surface_patch_diagnostic"]
            same_patch = bool(
                separation <= thresholds["maximum_contact_point_separation_m"]
                and angle <= thresholds["maximum_contact_normal_angle_deg"]
                and float(np.max(visual_distance)) <= thresholds["maximum_visual_projection_distance_m"]
                and max(float(distance9.max()), float(distance23.max()))
                <= thresholds["maximum_cross_part_surface_distance_m"]
            )
            pair_metrics = {
                "contact_point_separation_m": separation,
                "contact_normal_angle_deg": angle,
                "maximum_visual_projection_distance_m": float(np.max(visual_distance)),
                "maximum_cross_part_surface_distance_m": max(
                    float(distance9.max()), float(distance23.max())
                ),
            }
        rows[str(source)] = {
            "contacts": contact_rows,
            "pair_metrics": pair_metrics,
            "same_visual_surface_patch_under_predeclared_diagnostic": same_patch,
            "summed_normal_force_n": float(sum(row["normal_force_n"] for row in contacts)),
        }
        saved[f"source{source}_contact_points_object"] = local
        saved[f"source{source}_object_outward_normals"] = np.asarray(outward_normals, np.float64)
        saved[f"source{source}_visual_nearest"] = nearest

    intersection = trimesh.boolean.intersection([part9, part23], engine="manifold")
    intersection_volume = float(intersection.volume) if intersection is not None else 0.0
    intersection_faces = int(len(intersection.faces)) if intersection is not None else 0
    source57_same = rows["57"]["same_visual_surface_patch_under_predeclared_diagnostic"]
    return {
        "sources": rows,
        "part9_volume_m3": float(part9.volume),
        "part23_volume_m3": float(part23.volume),
        "part9_part23_intersection_volume_m3": intersection_volume,
        "part9_part23_intersection_face_count": intersection_faces,
        "parts_have_no_positive_volume_overlap": bool(intersection_volume <= 1.0e-12),
        "source57_constraints_resolve_to_same_visual_surface_patch": source57_same,
        "source57_summed_force_is_valid_unique_physical_force_target": not source57_same,
        "L1_force_target_status": (
            "invalid_as_a_unique_physical_force_target_due_to_decomposition_seam_duplicate_constraints"
            if source57_same else "retained_as_two_geometrically_distinct_surface_contacts"
        ),
    }, saved


def _snapshot_array(snapshot: dict, name: str) -> np.ndarray:
    value = snapshot[name]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _tracking_from_qpos(qpos: np.ndarray, endpoint: int,
                        reference_qpos: np.ndarray, objective) -> dict:
    position_delta = np.asarray(qpos[36:39], np.float64) - reference_qpos[endpoint, 36:39]
    position = float(np.linalg.norm(position_delta))
    rotation = _rotation_distance(
        _rotation_matrix_wxyz(np.asarray(qpos[39:43], np.float64)),
        _rotation_matrix_wxyz(reference_qpos[endpoint, 39:43]),
    )
    position_term = float(objective.tracking.lambda_p * position * position)
    rotation_term = float(objective.tracking.lambda_r * rotation * rotation)
    score = float(np.sqrt(position_term + rotation_term))
    return {
        "position_error_xyz_m": position_delta.tolist(),
        "position_error_m": position,
        "rotation_error_rad": rotation,
        "position_squared_contribution": position_term,
        "rotation_squared_contribution": rotation_term,
        "score": score,
        "feasible": bool(np.isfinite(score) and score < objective.tracking.boundary),
    }


def _pair_set(rows: list[dict]) -> list[list[str]]:
    return [list(pair) for pair in sorted({tuple(sorted(row["geom_names"])) for row in rows})]


def _native_from_snapshot(model, snapshot: dict):
    """Restore the common integrator state needed by the predeclared P0 parity test."""
    import mujoco

    data = mujoco.MjData(model)
    for name in ("qpos", "qvel", "act", "qacc_warmstart", "ctrl", "qfrc_applied",
                 "xfrc_applied", "mocap_pos", "mocap_quat"):
        if name not in snapshot or not hasattr(data, name):
            continue
        source = _snapshot_array(snapshot, name)
        if source.ndim and source.shape[0] == 1:
            source = source[0]
        target = getattr(data, name)
        if np.shape(target) == np.shape(source):
            target[...] = source
    if "time" in snapshot:
        data.time = float(_snapshot_array(snapshot, "time").reshape(-1)[0])
    mujoco.mj_forward(model, data)
    # mj_forward computes qacc_warmstart.  Restore the saved value after it so
    # the first native step starts from the same warmstart as MJWP.
    if "qacc_warmstart" in snapshot:
        warm = _snapshot_array(snapshot, "qacc_warmstart")
        if warm.ndim and warm.shape[0] == 1:
            warm = warm[0]
        if warm.shape == data.qacc_warmstart.shape:
            data.qacc_warmstart[:] = warm
    return data


def _mjwp_raw_step(env, ctrl: np.ndarray) -> None:
    env._mjwp.step_env(
        env.ego_cfg, env.env,
        torch.as_tensor(ctrl[None], dtype=torch.float32, device=str(env.ego_cfg.device)),
    )


def audit_P0(*, contract: dict, backend, model, states: dict[int, dict],
             reference_qpos: np.ndarray, objective) -> tuple[dict, dict[str, np.ndarray]]:
    import mujoco

    conditions = contract["P0_native_MJWP_parity"]["parity_conditions"]
    substeps = int(contract["P0_native_MJWP_parity"]["physics_substeps"])
    report: dict[str, dict] = {}
    saved: dict[str, np.ndarray] = {}
    for source in P0_SOURCES:
        state = states[source]
        snapshot = state["snapshot"]
        ctrl = np.asarray(state["full_ctrl"], np.float64)
        backend.restore(snapshot)
        backend.verify_restored_snapshot(snapshot)
        native = _native_from_snapshot(model, snapshot)
        native.ctrl[:] = ctrl
        mj_qpos, mj_qvel, mj_force, mj_constraint = [], [], [], []
        n_qpos, n_qvel, n_force, n_constraint = [], [], [], []
        pair_rows = []
        contact_details = []
        for step in range(substeps):
            _mjwp_raw_step(backend.env, ctrl)
            mujoco.mj_step(model, native)
            mq = _tensor(backend.env.env.data_wp, "qpos")[0]
            mv = _tensor(backend.env.env.data_wp, "qvel")[0]
            mf = _tensor(backend.env.env.data_wp, "actuator_force")[0]
            mc = _tensor(backend.env.env.data_wp, "qfrc_constraint")[0]
            mj_rows = mjwp_contacts(backend.env, only_right_tool=True)
            native_rows = native_contacts(model, native, only_right_tool=True)
            mj_qpos.append(mq); mj_qvel.append(mv); mj_force.append(mf); mj_constraint.append(mc)
            n_qpos.append(native.qpos.copy()); n_qvel.append(native.qvel.copy())
            n_force.append(native.actuator_force.copy()); n_constraint.append(native.qfrc_constraint.copy())
            mj_pairs, native_pairs = _pair_set(mj_rows), _pair_set(native_rows)
            pair_rows.append({
                "physics_substep": step + 1,
                "MJWP": mj_pairs,
                "native_MuJoCo": native_pairs,
                "exact_match": mj_pairs == native_pairs,
            })
            contact_details.append({
                "physics_substep": step + 1,
                "MJWP": mj_rows,
                "native_MuJoCo": native_rows,
            })
        mj_qpos = np.asarray(mj_qpos); mj_qvel = np.asarray(mj_qvel)
        n_qpos = np.asarray(n_qpos); n_qvel = np.asarray(n_qvel)
        endpoint = source + 1
        mj_tracking = _tracking_from_qpos(mj_qpos[-1], endpoint, reference_qpos, objective)
        native_tracking = _tracking_from_qpos(n_qpos[-1], endpoint, reference_qpos, objective)
        bowl_position = np.linalg.norm(mj_qpos[:, 36:39] - n_qpos[:, 36:39], axis=1)
        bowl_rotation = np.asarray([
            _rotation_distance(_rotation_matrix_wxyz(mj_qpos[i, 39:43]),
                               _rotation_matrix_wxyz(n_qpos[i, 39:43]))
            for i in range(substeps)
        ])
        hand_qpos = np.max(np.abs(mj_qpos[:, HAND_QPOS] - n_qpos[:, HAND_QPOS]), axis=1)
        qvel = np.max(np.abs(mj_qvel - n_qvel), axis=1)
        passed = bool(
            all(row["exact_match"] for row in pair_rows)
            and mj_tracking["feasible"] == native_tracking["feasible"]
            and float(bowl_position.max()) <= conditions["maximum_bowl_position_difference_m"]
            and float(bowl_rotation.max()) <= conditions["maximum_bowl_rotation_difference_rad"]
            and float(hand_qpos.max()) <= conditions["maximum_hand_qpos_difference"]
            and float(qvel.max()) <= conditions["maximum_qvel_difference"]
        )
        report[str(source)] = {
            "endpoint": endpoint,
            "passed": passed,
            "contact_pair_sets": pair_rows,
            "contact_details": contact_details,
            "endpoint_tracking": {"MJWP": mj_tracking, "native_MuJoCo": native_tracking},
            "maximum_differences": {
                "bowl_position_m": float(bowl_position.max()),
                "bowl_rotation_rad": float(bowl_rotation.max()),
                "hand_qpos": float(hand_qpos.max()),
                "qvel": float(qvel.max()),
                "actuator_force": float(np.max(np.abs(np.asarray(mj_force) - np.asarray(n_force)))),
                "constraint_force": float(np.max(np.abs(np.asarray(mj_constraint) - np.asarray(n_constraint)))),
            },
        }
        prefix = f"source{source}"
        saved[f"{prefix}_MJWP_qpos"] = mj_qpos
        saved[f"{prefix}_native_qpos"] = n_qpos
        saved[f"{prefix}_MJWP_qvel"] = mj_qvel
        saved[f"{prefix}_native_qvel"] = n_qvel
        saved[f"{prefix}_MJWP_actuator_force"] = np.asarray(mj_force)
        saved[f"{prefix}_native_actuator_force"] = np.asarray(n_force)
        saved[f"{prefix}_MJWP_qfrc_constraint"] = np.asarray(mj_constraint)
        saved[f"{prefix}_native_qfrc_constraint"] = np.asarray(n_constraint)
    return {
        "sources": report,
        "all_sources_passed": bool(all(row["passed"] for row in report.values())),
        "limited_claim": "finite ten-substep parity at the three predeclared exact states",
    }, saved


def _right_hand_surface_samples(model, data) -> list[dict]:
    """Finite, declared site and collision-primitive samples for P3/P4."""
    import mujoco

    rows = []
    site_names = {
        "right_palm": "palm", "right_thumb_tip": "thumb", "right_index_tip": "index",
        "right_middle_tip": "middle", "right_ring_tip": "ring", "right_pinky_tip": "pinky",
    }
    for name, role in site_names.items():
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site < 0:
            raise ValueError(f"missing P3 site {name}")
        rows.append({
            "sample": f"site:{name}", "role": role,
            "body_id": int(model.site_bodyid[site]),
            "point_world": np.asarray(data.site_xpos[site], np.float64).copy(),
        })
    for geom in range(model.ngeom):
        name = _geom_name(model, geom)
        if not name.startswith("collision_hand_right"):
            continue
        lower = name.lower()
        role = next((f for f in FINGERS if f in lower), "palm")
        center = np.asarray(data.geom_xpos[geom], np.float64)
        rotation = np.asarray(data.geom_xmat[geom], np.float64).reshape(3, 3)
        size = np.asarray(model.geom_size[geom], np.float64)
        geom_type = int(model.geom_type[geom])
        local = []
        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            local = [axis * sign * size[0] for axis in np.eye(3) for sign in (-1.0, 1.0)]
        elif geom_type in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            local = [np.array([sign * size[0], 0, 0]) for sign in (-1.0, 1.0)]
            local += [np.array([0, sign * size[0], 0]) for sign in (-1.0, 1.0)]
            local += [np.array([0, 0, sign * size[1]]) for sign in (-1.0, 1.0)]
        elif geom_type in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
            local = [np.eye(3)[axis] * sign * size[axis] for axis in range(3) for sign in (-1.0, 1.0)]
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                local += [np.asarray(signs) * size for signs in (
                    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                )]
        else:
            radius = float(model.geom_rbound[geom])
            local = [axis * sign * radius for axis in np.eye(3) for sign in (-1.0, 1.0)]
        for index, point in enumerate(local):
            rows.append({
                "sample": f"geom:{name}:{index}", "role": role,
                "body_id": int(model.geom_bodyid[geom]),
                "point_world": center + rotation @ point,
            })
    return rows


def audit_P3(*, model, reference_qpos: np.ndarray, reference_qvel: np.ndarray,
             reference_ctrl: np.ndarray, residual_scale: float,
             bowl_mesh) -> tuple[dict, dict[str, np.ndarray], dict[int, list[dict]]]:
    import mujoco

    data = mujoco.MjData(model)
    ctrlrange = np.asarray(model.actuator_ctrlrange, np.float64)
    limited = np.asarray(model.actuator_ctrllimited, bool)
    qvel_addresses = np.asarray([model.jnt_dofadr[int(j)] for j in model.actuator_trnid[:, 0]], np.int64)
    endpoints: dict[str, dict] = {}
    candidates: dict[int, list[dict]] = {}
    saved_points, saved_gaps, saved_authority, saved_endpoint, saved_role = [], [], [], [], []
    role_number = {role: index for index, role in enumerate(("palm",) + FINGERS)}
    for endpoint in range(20, 61):
        data.qpos[:] = reference_qpos[endpoint]
        data.qvel[:] = reference_qvel[endpoint]
        mujoco.mj_forward(model, data)
        object_position = np.asarray(data.xpos[37], np.float64)
        object_rotation = np.asarray(data.xmat[37], np.float64).reshape(3, 3)
        samples = _right_hand_surface_samples(model, data)
        points_world = np.asarray([row["point_world"] for row in samples], np.float64)
        points_local = (points_world - object_position) @ object_rotation
        nearest_local, gaps, normals_local = _mesh_projection(bowl_mesh, points_local)
        control = np.asarray(reference_ctrl[min(endpoint, len(reference_ctrl) - 1)], np.float64)
        low = np.full(model.nu, -1.0); high = np.full(model.nu, 1.0)
        low[limited] = np.maximum(low[limited], (ctrlrange[limited, 0] - control[limited]) / residual_scale)
        high[limited] = np.minimum(high[limited], (ctrlrange[limited, 1] - control[limited]) / residual_scale)
        if np.any(low > high + 2.0e-7):
            raise RuntimeError(f"empty feasible residual interval at reference endpoint {endpoint}")
        rows = []
        by_role = {}
        for index, sample in enumerate(samples):
            nearest_world = object_position + object_rotation @ nearest_local[index]
            toward_world = nearest_world - points_world[index]
            gap = float(gaps[index])
            direction = toward_world / gap if gap > 1.0e-12 else -object_rotation @ normals_local[index]
            jacp = np.zeros((3, model.nv), np.float64)
            jacr = np.zeros((3, model.nv), np.float64)
            mujoco.mj_jac(
                model, data, jacp, jacr, points_world[index], int(sample["body_id"]),
            )
            coefficients = direction @ jacp[:, qvel_addresses]
            displacement_low = residual_scale * low
            displacement_high = residual_scale * high
            maximum_closing = float(np.sum(np.maximum(
                coefficients * displacement_low, coefficients * displacement_high,
            )))
            reachable = bool(gap <= maximum_closing + 1.0e-6)
            row = {
                "sample": sample["sample"], "role": sample["role"],
                "body_id": int(sample["body_id"]),
                "point_world_m": points_world[index].tolist(),
                "point_object_local_m": points_local[index].tolist(),
                "nearest_bowl_surface_object_local_m": nearest_local[index].tolist(),
                "nearest_bowl_surface_world_m": nearest_world.tolist(),
                "visual_mesh_gap_m": gap,
                "visual_surface_normal_object_local": normals_local[index].tolist(),
                "maximum_first_order_closing_displacement_m": maximum_closing,
                "contact_reachable_under_first_order_box_support": reachable,
                "jacobian_actuator_coefficients": coefficients.tolist(),
            }
            rows.append(row)
            current = by_role.get(sample["role"])
            if current is None or gap < current["visual_mesh_gap_m"]:
                by_role[sample["role"]] = row
            saved_points.append(points_local[index]); saved_gaps.append(gap)
            saved_authority.append(maximum_closing); saved_endpoint.append(endpoint)
            saved_role.append(role_number[sample["role"]])
        finger_reachable = {finger: any(
            row["role"] == finger and row["contact_reachable_under_first_order_box_support"]
            for row in rows
        ) for finger in FINGERS}
        endpoints[str(endpoint)] = {
            "nearest_by_role": by_role,
            "sample_count": len(rows),
            "any_right_finger_contact_reachable": bool(any(finger_reachable.values())),
            "thumb_plus_non_thumb_contact_reachable": bool(
                finger_reachable["thumb"] and any(finger_reachable[f] for f in FINGERS[1:])
            ),
            "finger_reachable": finger_reachable,
        }
        candidates[endpoint] = [row for row in by_role.values()
                                if row["contact_reachable_under_first_order_box_support"]]
    return {
        "endpoints": endpoints,
        "finite_sampling_statement": (
            "Reachability is a first-order box-support diagnostic over the declared "
            "finite sites and collision-primitive surface samples, not a global IK proof."
        ),
        "endpoints_without_any_reachable_finger": [
            int(k) for k, row in endpoints.items() if not row["any_right_finger_contact_reachable"]
        ],
        "endpoints_without_reachable_thumb_plus_non_thumb": [
            int(k) for k, row in endpoints.items() if not row["thumb_plus_non_thumb_contact_reachable"]
        ],
    }, {
        "endpoint": np.asarray(saved_endpoint, np.int32),
        "role": np.asarray(saved_role, np.int8),
        "point_object_local_m": np.asarray(saved_points, np.float64),
        "visual_mesh_gap_m": np.asarray(saved_gaps, np.float64),
        "maximum_first_order_closing_displacement_m": np.asarray(saved_authority, np.float64),
    }, candidates


def _orthogonal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, seed); first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def _wrench_columns(candidates: list[dict], mu: float, sides: int,
                    object_position: np.ndarray, object_rotation: np.ndarray,
                    model, data, qvel_addresses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    wrench, effort = [], []
    for row in candidates:
        normal = object_rotation @ np.asarray(row["visual_surface_normal_object_local"], np.float64)
        normal /= np.linalg.norm(normal)
        tangent1, tangent2 = _orthogonal_basis(normal)
        point = np.asarray(row["nearest_bowl_surface_world_m"], np.float64)
        jacp = np.zeros((3, model.nv), np.float64)
        jacr = np.zeros((3, model.nv), np.float64)
        mujoco.mj_jac(model, data, jacp, jacr, point, int(row["body_id"]))
        for side in range(sides):
            theta = 2.0 * np.pi * side / sides
            # Force exerted by the hand on the object.  Alpha is the normal
            # force magnitude; the cone edge carries mu times that magnitude.
            force_world = -normal + mu * (np.cos(theta) * tangent1 + np.sin(theta) * tangent2)
            force_object = object_rotation.T @ force_world
            lever_object = object_rotation.T @ (point - object_position)
            wrench.append(np.concatenate([force_object, np.cross(lever_object, force_object)]))
            # Equal/opposite force acts on the hand.
            effort.append(-(jacp[:, qvel_addresses].T @ force_world))
    if not wrench:
        return np.zeros((6, 0), np.float64), np.zeros((len(qvel_addresses), 0), np.float64)
    return np.stack(wrench, axis=1), np.stack(effort, axis=1)


def _current_wrench_feasibility(wrench_matrix: np.ndarray, effort_matrix: np.ndarray,
                                target: np.ndarray, limits: np.ndarray) -> dict:
    from scipy.optimize import linprog

    count = wrench_matrix.shape[1]
    if count == 0:
        return {"success": False, "status": "no_reachable_contact_candidates", "normal_coefficients": None}
    result = linprog(
        c=np.zeros(count), A_ub=np.vstack([effort_matrix, -effort_matrix]),
        b_ub=np.concatenate([limits, limits]), A_eq=wrench_matrix, b_eq=target,
        bounds=[(0.0, None)] * count, method="highs",
    )
    return {
        "success": bool(result.success), "status": result.message,
        "normal_coefficients": result.x.tolist() if result.success else None,
    }


def _critical_effort_multiplier(wrench_matrix: np.ndarray, effort_matrix: np.ndarray,
                                target: np.ndarray, limits: np.ndarray) -> float | None:
    from scipy.optimize import linprog

    count = wrench_matrix.shape[1]
    if count == 0:
        return None
    objective = np.r_[np.zeros(count), 1.0]
    inequalities = np.block([
        [effort_matrix, -limits[:, None]],
        [-effort_matrix, -limits[:, None]],
    ])
    result = linprog(
        c=objective, A_ub=inequalities, b_ub=np.zeros(2 * len(limits)),
        A_eq=np.c_[wrench_matrix, np.zeros(6)], b_eq=target,
        bounds=[(0.0, None)] * count + [(0.0, None)], method="highs",
    )
    return float(result.x[-1]) if result.success else None


def _critical_mass_multiplier(wrench_matrix: np.ndarray, effort_matrix: np.ndarray,
                              target: np.ndarray, limits: np.ndarray) -> float | None:
    from scipy.optimize import linprog

    count = wrench_matrix.shape[1]
    if count == 0:
        return None
    # A alpha = multiplier * current-mass wrench.
    result = linprog(
        c=np.r_[np.zeros(count), -1.0],
        A_ub=np.block([[effort_matrix, np.zeros((len(limits), 1))],
                       [-effort_matrix, np.zeros((len(limits), 1))]]),
        b_ub=np.concatenate([limits, limits]),
        A_eq=np.c_[wrench_matrix, -target], b_eq=np.zeros(6),
        bounds=[(0.0, None)] * count + [(0.0, None)], method="highs",
    )
    return float(result.x[-1]) if result.success else None


def audit_P4(*, model, reference_qpos: np.ndarray, reference_qvel: np.ndarray,
             candidates: dict[int, list[dict]], effort_limits: np.ndarray,
             sides: int, dt: float) -> tuple[dict, dict[str, np.ndarray]]:
    import mujoco

    body = 37
    data = mujoco.MjData(model)
    positions, rotations = [], []
    for endpoint in range(len(reference_qpos)):
        data.qpos[:] = reference_qpos[endpoint]
        data.qvel[:] = reference_qvel[endpoint]
        mujoco.mj_forward(model, data)
        positions.append(np.asarray(data.xipos[body], np.float64).copy())
        rotations.append(np.asarray(data.xmat[body], np.float64).reshape(3, 3).copy())
    positions = np.asarray(positions); rotations = np.asarray(rotations)
    velocity = np.gradient(positions, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    omega = np.zeros_like(positions)
    for endpoint in range(1, len(reference_qpos) - 1):
        delta = rotations[endpoint + 1] @ rotations[endpoint - 1].T
        angle = _rotation_distance(delta, np.eye(3))
        vector = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
        if angle > 1.0e-12:
            vector *= angle / (2.0 * np.sin(angle))
        else:
            vector *= 0.5
        omega[endpoint] = vector / (2.0 * dt)
    omega[0] = omega[1]; omega[-1] = omega[-2]
    alpha = np.gradient(omega, dt, axis=0, edge_order=2)
    qvel_addresses = np.asarray([model.jnt_dofadr[int(j)] for j in model.actuator_trnid[:, 0]], np.int64)
    right_indices = np.arange(18, dtype=np.int64)
    right_limits = effort_limits[right_indices]
    right_dofs = qvel_addresses[right_indices]
    hand_frictions = [model.geom_friction[g, 0] for g in range(model.ngeom)
                      if _geom_name(model, g).startswith("collision_hand_right")]
    object_frictions = [model.geom_friction[g, 0] for g in range(model.ngeom)
                        if _geom_name(model, g).startswith("right_object_")]
    current_mu = float(max(np.median(hand_frictions), np.median(object_frictions)))
    rows = {}; saved_wrench, saved_endpoint = [], []
    gravity = np.asarray(model.opt.gravity, np.float64)
    mass = float(model.body_mass[body])
    inertia_body = np.diag(np.asarray(model.body_inertia[body], np.float64))
    for endpoint in range(44, 61):
        data.qpos[:] = reference_qpos[endpoint]
        data.qvel[:] = reference_qvel[endpoint]
        mujoco.mj_forward(model, data)
        rotation = rotations[endpoint]
        inertia_world = rotation @ inertia_body @ rotation.T
        force_world = mass * (acceleration[endpoint] - gravity)
        torque_world = inertia_world @ alpha[endpoint] + np.cross(
            omega[endpoint], inertia_world @ omega[endpoint]
        )
        target = np.concatenate([rotation.T @ force_world, rotation.T @ torque_world])
        contact_rows = candidates[endpoint]
        wrench_matrix, effort_matrix = _wrench_columns(
            contact_rows, current_mu, sides, positions[endpoint], rotation,
            model, data, right_dofs,
        )
        current = _current_wrench_feasibility(wrench_matrix, effort_matrix, target, right_limits)
        minimum_effort = _critical_effort_multiplier(wrench_matrix, effort_matrix, target, right_limits)
        maximum_mass = _critical_mass_multiplier(wrench_matrix, effort_matrix, target, right_limits)
        minimum_mu = None
        if contact_rows:
            high = 3.0
            high_w, high_e = _wrench_columns(contact_rows, high, sides, positions[endpoint], rotation,
                                              model, data, right_dofs)
            if _current_wrench_feasibility(high_w, high_e, target, right_limits)["success"]:
                low = 0.0
                for _ in range(24):
                    mid = 0.5 * (low + high)
                    mid_w, mid_e = _wrench_columns(contact_rows, mid, sides, positions[endpoint], rotation,
                                                   model, data, right_dofs)
                    if _current_wrench_feasibility(mid_w, mid_e, target, right_limits)["success"]:
                        high = mid
                    else:
                        low = mid
                minimum_mu = high
        rows[str(endpoint)] = {
            "reachable_contact_roles": [row["role"] for row in contact_rows],
            "candidate_count": len(contact_rows),
            "required_object_wrench_object_frame": target.tolist(),
            "current_friction_coefficient": current_mu,
            "current_model_feasible": current["success"],
            "LP_status": current["status"],
            "maximum_feasible_mass_multiplier": maximum_mass,
            "minimum_required_friction_coefficient_with_current_effort": minimum_mu,
            "minimum_required_effort_multiplier_with_current_friction": minimum_effort,
        }
        saved_wrench.append(target); saved_endpoint.append(endpoint)
    return {
        "endpoints": rows,
        "current_friction_combination_rule": (
            "MuJoCo equal-priority maximum; the audit uses max(median right-hand geom friction, "
            "median right-object geom friction), which is exact here because each family is uniform"
        ),
        "current_friction_coefficient": current_mu,
        "all_current_model_endpoints_feasible": bool(all(row["current_model_feasible"] for row in rows.values())),
        "infeasible_endpoints": [int(k) for k, row in rows.items() if not row["current_model_feasible"]],
        "finite_model_statement": (
            "This is a finite 16-sided friction-cone LP over P3 reachable samples, "
            "not a proof over all possible contact locations or forces."
        ),
    }, {
        "endpoint": np.asarray(saved_endpoint, np.int32),
        "required_wrench_object_frame": np.asarray(saved_wrench, np.float64),
        "reference_object_position_world_m": positions[44:61],
        "reference_object_linear_velocity_world_m_per_s": velocity[44:61],
        "reference_object_linear_acceleration_world_m_per_s2": acceleration[44:61],
        "reference_object_angular_velocity_world_rad_per_s": omega[44:61],
        "reference_object_angular_acceleration_world_rad_per_s2": alpha[44:61],
    }


def _write_summary(path: Path, report: dict) -> None:
    p1, p2, p0, p3, p4 = (report[name] for name in ("P1", "P2", "P0", "P3", "P4"))
    lines = [
        "# TACO Pour physics/reference validity Gate P v1", "",
        "Gate P was executed in the frozen order `P1 → P2 → P0 → P3 → P4`.",
        "No PPO training, controller tuning, task-parameter sweep, model mutation, chunk acceptance, or chunk commit occurred.", "",
        "## Results", "",
        f"- P1: {p1['anchors']['formal_frozen_PPO']['saturated_requests']}/1080 formal and "
        f"{p1['anchors']['gate_A1_best_position_velocity_feedback']['saturated_requests']}/1080 feedback effort requests exceeded declared limits.",
        f"- P2: source57 same visual-patch duplicate diagnostic = {p2['source57_constraints_resolve_to_same_visual_surface_patch']}.",
        f"- P0: all three finite native-MuJoCo/MJWP parity probes passed = {p0['all_sources_passed']}.",
        f"- P3: endpoints lacking any reachable right-finger sample = {p3['endpoints_without_any_reachable_finger']}.",
        f"- P3: endpoints lacking reachable thumb + non-thumb samples = {p3['endpoints_without_reachable_thumb_plus_non_thumb']}.",
        f"- P4: current-model finite wrench LP infeasible endpoints = {p4['infeasible_endpoints']}.", "",
        "## Decision", "",
        f"- Structural issues: `{report['decision']['structural_issues']}`",
        f"- Gate B/C remain blocked: `{not report['decision']['Gate_B_or_C_allowed']}`",
        "- P5 remains an external-author-information blocker; no local objective recovery was attempted.",
        "- Gate P is closed after these finite tests. The results are diagnostics, not global impossibility proofs.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_P1_heatmap(path: Path, arrays: dict[str, np.ndarray], names: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anchors = ("formal_frozen_PPO", "gate_A1_best_position_velocity_feedback")
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    for axis, anchor in zip(axes, anchors, strict=True):
        ratio = np.asarray(arrays[f"{anchor}_requested_to_limit_ratio"], np.float64).T
        image = axis.imshow(
            np.log10(np.maximum(ratio, 1.0e-3)), aspect="auto", origin="lower",
            cmap="magma", vmin=-3.0, vmax=float(np.log10(max(1.0, ratio.max()))),
        )
        axis.contour(ratio, levels=[1.0], colors="cyan", linewidths=0.7)
        axis.set_title(f"{anchor}: log10(|requested generalized effort| / declared limit)")
        axis.set_ylabel("actuator")
        axis.set_yticks(np.arange(len(names)))
        axis.set_yticklabels(names, fontsize=6)
        figure.colorbar(image, ax=axis, label="log10(request / limit)")
    axes[-1].set_xlabel("Gate-L physics substep (30 total)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/taco_pour_physics_reference_validity_gate_P_v1.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/taco_pour_physics_reference_validity_gate_P_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    gate_L_report = json.loads(paths["Gate_L_report"].read_text())
    if gate_L_report["decision"]["Gate_L_closed"] is not True:
        raise ValueError("Gate P requires closed Gate L")
    _, parent_paths, _ = load_tail_contract(paths["tail_contract"])
    tail_report = json.loads(paths["tail_report"].read_text())

    from run_mjwp_ppo import _build_network_config, _build_ppo_config, _load_ego_config, _load_reference
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import MJWPChunkBackend
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent, load_truncated_gaussian_profile,
    )

    objective = load_runtime_objective(parent_paths["protocol"], parent_paths["objective_profile"],
                                       tracking_variant="tool_only", require_run_ready=False)
    observation = load_runtime_observation(parent_paths["protocol"], parent_paths["observation_profile"],
                                           require_run_ready=False)
    residual, residual_artifact = load_residual_action_profile(parent_paths["action_profile"])
    distribution, _ = load_truncated_gaussian_profile(parent_paths["distribution_profile"])
    config = _load_ego_config(str(paths["formal_simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    reference_qpos = reference[0].detach().cpu().numpy().astype(np.float64)
    reference_qvel = reference[1].detach().cpu().numpy().astype(np.float64)
    reference_ctrl = np.load(paths["robot_reference"])["ctrl"].astype(np.float64)
    env = build_env(config, reference, objective, observation, residual)
    backend = MJWPChunkBackend(env)
    model = env.env.model_cpu
    effort_limits = urdf_effort_limits((paths["right_XHand_URDF"], paths["left_XHand_URDF"]))
    law = actuator_force_law(model, effort_limits)
    checkpoint = load_gzip_torch(parent_paths["checkpoint"])
    boundary = load_gzip_torch(parent_paths["boundary"])
    temporary = tempfile.TemporaryDirectory(prefix=".gate_P_", dir=ROOT / "runs")
    try:
        ppo_config = replace(_build_ppo_config(
            num_envs=1, horizon_length=40, seq_length=4, max_epochs=8,
            learning_rate=1e-4, device="cpu", asymmetric_critic=None,
        ), clip_actions=False)
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temporary.name) / "policy", ppo_config=ppo_config,
            network_config=_build_network_config(4), env=env, distribution_spec=distribution,
        )
        policy.model.load_state_dict(checkpoint["model"]); policy.set_eval()
        states = capture_gate_P_states(
            backend=backend, policy=policy, boundary=boundary, parent_paths=parent_paths,
            tail_report=tail_report, reference_qpos=reference_qpos,
            reference_qvel=reference_qvel, objective=objective,
        )
        policy.writer.close()
    finally:
        temporary.cleanup()

    # Results are computed before creating the formal output directory: an
    # exception cannot leave a partial run looking like official evidence.
    P1, P1_arrays = audit_P1(paths=paths, gate_L_report=gate_L_report, law=law,
                             model=model, source57=states[57])
    P2, P2_arrays = audit_P2(contract=contract, paths=paths, states=states)
    P0, P0_arrays = audit_P0(contract=contract, backend=backend, model=model, states=states,
                             reference_qpos=reference_qpos, objective=objective)
    bowl_mesh, _, _ = _load_bowl_meshes(paths)
    P3, P3_arrays, P3_candidates = audit_P3(
        model=model, reference_qpos=reference_qpos, reference_qvel=reference_qvel,
        reference_ctrl=reference_ctrl, residual_scale=float(residual.residual_scale), bowl_mesh=bowl_mesh,
    )
    P4, P4_arrays = audit_P4(
        model=model, reference_qpos=reference_qpos, reference_qvel=reference_qvel,
        candidates=P3_candidates,
        effort_limits=np.asarray(law["effort_limit"], np.float64),
        sides=int(contract["P4_reference_wrench_feasibility"]["friction_cone_sides"]),
        dt=1.0 / 30.0,
    )
    issues = []
    if any(row["saturated_requests"] for row in P1["anchors"].values()): issues.append("P1_declared_effort_saturation")
    if P1["measured_sum_exceeds_finger_only_aggregate_bound"]: issues.append("P1_measured_contact_force_exceeds_finger_only_Jacobian_bound")
    if P2["source57_constraints_resolve_to_same_visual_surface_patch"]: issues.append("P2_convex_decomposition_duplicate_contact_patch")
    if not P0["all_sources_passed"]: issues.append("P0_native_MJWP_backend_parity_failure")
    if P3["endpoints_without_any_reachable_finger"]: issues.append("P3_reference_lacks_reachable_right_finger_contact")
    if P3["endpoints_without_reachable_thumb_plus_non_thumb"]: issues.append("P3_reference_lacks_reachable_thumb_plus_non_thumb_contact")
    if not P4["all_current_model_endpoints_feasible"]: issues.append("P4_reference_wrench_infeasible_under_finite_contact_model")
    report = {
        "schema": "taco_pour_physics_reference_validity_gate_P_report_v1",
        "status": "completed_finite_read_only_gate_P_closed",
        "paper_faithful": False,
        "contract": contract_artifact,
        "runtime": contract["runtime"],
        "P1": P1, "P2": P2, "P0": P0, "P3": P3, "P4": P4,
        "P5": contract["P5_objective_contract"],
        "decision": {
            "Gate_P_closed": True,
            "structural_issues": issues,
            "Gate_B_or_C_allowed": len(issues) == 0,
            "PPO_retraining_allowed": False,
            "chunk_commit_allowed": False,
            "next_blocker": (
                "physics_reference_and_XHand_contact_actuator_model_review_required"
                if issues else "external_author_objective_and_low_level_controller_information_required"
            ),
        },
        "artifacts": {},
    }
    state_arrays = {}
    for source, state in states.items():
        for field in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
            if field in state["snapshot"]:
                state_arrays[f"source{source}_{field}"] = _snapshot_array(state["snapshot"], field)
        if "full_action" in state: state_arrays[f"source{source}_full_action"] = state["full_action"]
        if "full_ctrl" in state: state_arrays[f"source{source}_full_ctrl"] = state["full_ctrl"]
    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    artifact_data = {
        "exact_states_and_contacts": state_arrays, "P1_saturation": P1_arrays,
        "P2_geometry": P2_arrays, "P0_parity": P0_arrays,
        "P3_reachability": P3_arrays, "P4_wrench": P4_arrays,
    }
    for key, arrays in artifact_data.items():
        path = args.output / contract["artifacts"][key]
        np.savez_compressed(path, **arrays)
        report["artifacts"][key] = {"path": str(path.resolve()), "sha256": sha256(path)}
    heatmap_path = args.output / contract["artifacts"]["P1_saturation_heatmap"]
    _write_P1_heatmap(
        heatmap_path, P1_arrays,
        list(gate_L_report["diagnostic_model"]["actuator_names"]),
    )
    report["artifacts"]["P1_saturation_heatmap"] = {
        "path": str(heatmap_path.resolve()), "sha256": sha256(heatmap_path),
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({"report": str(report_path), "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
