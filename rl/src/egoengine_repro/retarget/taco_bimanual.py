"""Minimal active TACO bimanual MINK retarget runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .collision_audit import audit_intrahand_trajectory, distances, explicit_hand_pairs
from .kinematic_limits import (
    FrameDisplacementLimit,
    StrictCollisionLimit,
    enable_planning_collision_masks,
    explicit_collision_groups,
    joint_velocity_limits,
)
from .schema import validate_human_reference
from .support_plane_limit import NativeSupportPlaneLimit, native_hand_visual_geom_ids
from .taco_bimanual_settings import (
    SELF_ONLY_FINAL_FEASIBILITY,
    UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY,
    TacoBimanualRetargetSettings,
)
from ..scene.support_surface import Plane


SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose7(transform: np.ndarray) -> np.ndarray:
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.r_[transform[:3, 3], quaternion[[3, 0, 1, 2]]]


def differentiate(
    model: mujoco.MjModel, qpos: np.ndarray, dt: float,
) -> np.ndarray:
    if len(qpos) < 2 or dt <= 0.0:
        raise ValueError("differentiation requires at least two frames and positive dt")
    qvel = np.empty((len(qpos), model.nv), dtype=np.float64)
    for index in range(len(qpos)):
        first, last = max(0, index - 1), min(len(qpos) - 1, index + 1)
        mujoco.mj_differentiatePos(
            model, qvel[index], (last - first) * dt,
            qpos[first], qpos[last],
        )
    return qvel


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    ))


def _joint_state(
    model: Any, configuration: Any, addresses: np.ndarray, ranges: np.ndarray,
) -> tuple[float, bool]:
    current = np.asarray(configuration.q)[addresses]
    margin = float(np.minimum(
        current - ranges[:, 0], ranges[:, 1] - current,
    ).min())
    return margin, margin >= -1e-6


def _runtime_contract(
    *, settings: TacoBimanualRetargetSettings, dt: float,
    tasks: list[Any], configuration_limit: Any,
    displacement: FrameDisplacementLimit, collisions: list[StrictCollisionLimit],
    collision_pair_count: int, locks: list[Any],
    native_support: NativeSupportPlaneLimit | None,
) -> dict[str, Any]:
    wrist_tasks = [task for task in tasks if task.frame_name.endswith("_hand_link")]
    fingertip_tasks = [task for task in tasks if task.frame_name.endswith("_tip")]
    return {
        "schema": "taco_bimanual_effective_retarget_contract_v1",
        "algorithm": "MINK",
        "tasks": {
            "wrist": {
                "count": len(wrist_tasks),
                "position": "ABSENT",
                "orientation_cost": settings.wrist_orientation_cost,
            },
            "fingertips": {
                "count": len(fingertip_tasks),
                "count_per_hand": len(fingertip_tasks) // len(SIDES),
                "position_cost": settings.fingertip_position_cost,
                "orientation_cost": settings.fingertip_orientation_cost,
            },
            "posture": "ABSENT",
        },
        "hard_limits": {
            "configuration_limit": type(configuration_limit).__name__,
            "frame_displacement_limit": type(displacement).__name__,
            "self_collision": {
                "present": bool(collisions),
                "implementation": type(collisions[0]).__name__ if collisions else None,
                "explicit_pair_count": collision_pair_count,
                "minimum_accepted_distance_m": (
                    settings.accepted_min_self_collision_distance_m
                ),
                "acceptance_slack_m": (
                    settings.collision_acceptance_numerical_slack_m
                ),
            },
            "native_support": None if native_support is None else {
                "present": True,
                "implementation": type(native_support).__name__,
                "native_geom_count": len(native_support.geoms),
                **settings.native_support.to_dict(),
            },
        },
        "constraints": {
            "object_dofs_frozen": bool(locks),
            "frozen_dof_count": 12,
        },
        "timeline": {
            "source_dt_s": dt,
            "iterations_per_frame": settings.max_iterations_per_frame,
            "first_frame_iterations": max(80, settings.max_iterations_per_frame),
            "qp_integration_dt_s": dt / settings.max_iterations_per_frame,
        },
        "solver": {
            "name": "daqp",
            "damping": 1e-5,
            "primal_tolerance": settings.solver_primal_tolerance,
            "dual_tolerance": settings.solver_dual_tolerance,
        },
        "final_feasibility": {
            "algorithm": settings.final_feasibility_algorithm,
            "max_iterations": settings.max_feasibility_iterations,
            "tracking_tasks_during_closure": False,
            "direct_qpos_edit": False,
        },
        "action_optimization_config_dependencies": [],
    }


def retarget(
    scene: Path,
    human_path: Path,
    settings: TacoBimanualRetargetSettings,
    output: Path,
    *,
    expected_effective_contract: dict[str, Any] | None = None,
    frame_count: int | None = None,
) -> dict[str, Any]:
    """Run the typed, single-path TACO bimanual retargeter."""
    if not isinstance(settings, TacoBimanualRetargetSettings):
        raise TypeError("retarget requires TacoBimanualRetargetSettings")
    model = mujoco.MjModel.from_xml_path(str(scene))
    if (model.nq, model.nv, model.nu) != (50, 48, 36):
        raise ValueError("expected two 18-DoF floating hands and two passive objects")
    with np.load(human_path, allow_pickle=False) as archive:
        human = dict(archive)
    validate_human_reference(human)
    dt = float(np.diff(human["timestamps_s"])[0])
    total_frames = len(human["frame_indices"])
    count = total_frames if frame_count is None else int(frame_count)
    if not 2 <= count <= total_frames:
        raise ValueError("frame_count must be in [2, source frame count]")
    output.mkdir(parents=True, exist_ok=True)

    configuration = mink.Configuration(model)
    tasks: list[Any] = []
    for side in SIDES:
        tasks.append(mink.FrameTask(
            f"{side}_hand_link", "body", position_cost=0.0,
            orientation_cost=settings.wrist_orientation_cost, lm_damping=1e-3,
        ))
        tasks.extend(mink.FrameTask(
            f"{side}_{finger}_tip", "site",
            position_cost=settings.fingertip_position_cost,
            orientation_cost=settings.fingertip_orientation_cost,
            lm_damping=1e-3,
        ) for finger in FINGERS)

    hand_geoms = [
        geom for geom in range(model.ngeom)
        if (model.geom(geom).name or "").startswith("collision_hand_")
    ]
    runtime_pairs = explicit_hand_pairs(model)
    if not runtime_pairs:
        raise ValueError("an audited explicit self-collision contract is required")
    enable_planning_collision_masks(model, hand_geoms, [])
    groups = explicit_collision_groups(
        model, mujoco, hand_geom_ids=set(hand_geoms)
    )
    collisions: list[StrictCollisionLimit] = []
    if groups:
        inner = mink.CollisionAvoidanceLimit(
            model, groups,
            minimum_distance_from_collisions=settings.planning_collision_buffer_m,
            collision_detection_distance=0.02,
            include_explicit_pairs=True,
        )
        collision = StrictCollisionLimit(
            inner, mujoco,
            minimum_distance=settings.planning_collision_buffer_m,
            depenetration_step=0.002,
        )
        collision.enabled = True
        collisions.append(collision)
    constrained_pairs = set().union(*(
        set(limit.geom_id_pairs) for limit in collisions
    ))
    if constrained_pairs != set(runtime_pairs):
        raise ValueError("MINK filtered runtime self pairs")
    collision_pairs = tuple(sorted(constrained_pairs))

    velocity_map = joint_velocity_limits(
        model, mujoco, settings.velocity_limits
    )
    displacement = FrameDisplacementLimit(
        model, mujoco, velocity_map, mink.Constraint
    )
    configuration_limit = mink.ConfigurationLimit(model)
    native_support = None
    native_tolerance = None
    if settings.native_support is not None:
        support = settings.native_support
        native_tolerance = support.validation_tolerance_m
        native_support = NativeSupportPlaneLimit(
            model=model,
            mujoco=mujoco,
            plane=Plane(
                normal=support.normal, offset=support.offset_m,
                frame=support.frame,
            ),
            visual_geom_ids=native_hand_visual_geom_ids(model, mujoco),
            minimum_clearance_m=support.minimum_clearance_m,
            activation_distance_m=support.activation_distance_m,
            gain=support.gain,
            depenetration_step_m=support.depenetration_step_m,
            constraint_type=mink.Constraint,
        )
    limits: list[Any] = [configuration_limit, *collisions, displacement]
    if native_support is not None:
        limits.append(native_support)
    locks = [mink.DofFreezingTask(model, list(range(36, 48)))]
    effective_contract = _runtime_contract(
        settings=settings, dt=dt, tasks=tasks,
        configuration_limit=configuration_limit, displacement=displacement,
        collisions=collisions, collision_pair_count=len(collision_pairs),
        locks=locks, native_support=native_support,
    )
    _write_json(output / "effective_retarget_contract.json", effective_contract)
    if (
        expected_effective_contract is not None
        and expected_effective_contract != effective_contract
    ):
        _write_json(output / "effective_contract_mismatch.json", {
            "expected": expected_effective_contract,
            "constructed": effective_contract,
        })
        raise RuntimeError("effective retarget contract mismatch")

    qpos = np.empty((count, model.nq), dtype=np.float64)
    tip_errors = np.empty((count, 2, 5), dtype=np.float64)
    wrist_errors = np.empty((count, 2), dtype=np.float64)
    self_distances = np.empty(count, dtype=np.float64)
    native_distances = np.full(count, np.nan, dtype=np.float64)
    acceptance_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    previous = None
    substeps = settings.max_iterations_per_frame
    addresses = np.asarray([
        int(model.joint(name).qposadr[0]) for name in velocity_map
    ], dtype=np.int64)
    speeds = np.asarray(list(velocity_map.values()), dtype=np.float64)
    ranges = np.asarray([
        model.joint(name).range for name in velocity_map
    ], dtype=np.float64)

    def hard_state() -> dict[str, Any]:
        self_minimum = float(distances(
            model, configuration.data, collision_pairs
        ).min())
        joint_margin, joint_pass = _joint_state(
            model, configuration, addresses, ranges
        )
        displacement_margin = displacement.minimum_margin(configuration)
        displacement_pass = displacement_margin >= -1e-6
        if native_support is None:
            support_minimum = None
            support_geom = None
            support_vertex = None
            support_pass = True
        else:
            native_rows = native_support.full_mesh_rows(configuration.data)
            worst = min(native_rows, key=lambda row: row["distance_m"])
            support_minimum = float(worst["distance_m"])
            support_geom = worst["geom"].name
            support_vertex = int(worst["vertex_index"])
            support_pass = support_minimum >= -float(native_tolerance)
        self_pass = self_minimum >= (
            settings.accepted_min_self_collision_distance_m
            - settings.collision_acceptance_numerical_slack_m
        )
        return {
            "self_minimum_m": self_minimum,
            "self_pass": self_pass,
            "native_minimum_m": support_minimum,
            "native_geom": support_geom,
            "native_vertex_index": support_vertex,
            "native_pass": support_pass,
            "joint_minimum_margin": joint_margin,
            "joint_pass": joint_pass,
            "frame_displacement_minimum_margin": (
                None if np.isinf(displacement_margin) else displacement_margin
            ),
            "frame_displacement_pass": displacement_pass,
            "all_pass": self_pass and support_pass and joint_pass and displacement_pass,
        }

    def flush_traces() -> None:
        _write_jsonl(output / "per_frame_native_acceptance.jsonl", acceptance_rows)
        _write_jsonl(output / "feasibility_closure_trace.jsonl", closure_rows)

    def fail(frame: int, reason: str) -> None:
        np.savez_compressed(
            output / "failed_kinematic_prefix.npz",
            qpos=np.vstack([qpos[:frame], configuration.q]),
            frame_indices=human["frame_indices"][:frame + 1],
            failed_frame=frame,
        )
        flush_traces()
        state = hard_state()
        _write_json(output / "retarget_failure.json", {
            "status": "failed_kinematic_candidate",
            "failed_frame": frame,
            "reason": reason,
            "hard_state": state,
            "scene": str(scene.resolve()),
            "scene_sha256": sha256(scene),
            "strict_gate_passed": False,
            "rl_validation_completed": False,
        })
        raise RuntimeError(reason)

    def solve(frame: int, tracking_tasks: tuple[Any, ...] | list[Any]) -> np.ndarray:
        try:
            return mink.solve_ik(
                configuration, tracking_tasks, dt / substeps,
                solver="daqp", damping=1e-5, limits=limits,
                constraints=locks,
                primal_tol=settings.solver_primal_tolerance,
                dual_tol=settings.solver_dual_tolerance,
            )
        except Exception as error:
            fail(frame, f"QP failed at frame {frame}: {error}")
            raise AssertionError("unreachable")

    for frame in range(count):
        q = configuration.q.copy()
        for hand, side in enumerate(SIDES):
            object_address = int(
                model.joint(f"{side}_object_joint").qposadr[0]
            )
            q[object_address:object_address + 7] = pose7(
                human["T_sim_object_reference"][frame, hand]
            )
            wrist = human["T_sim_wrist_target"][frame, hand]
            if frame == 0:
                q[hand * 18:hand * 18 + 3] = wrist[:3, 3]
                angles = Rotation.from_matrix(wrist[:3, :3]).as_euler("ZXY")
                q[hand * 18 + 3:hand * 18 + 6] = angles * [1, 1, -1]
            targets = [wrist, *human["T_sim_fingertip_target"][frame, hand]]
            for index, target in enumerate(targets):
                tasks[hand * 6 + index].set_target(mink.SE3.from_matrix(target))
        configuration.update(q)
        displacement.set_previous(previous, dt if previous is not None else None)
        tracking_iterations = max(80, substeps) if frame == 0 else substeps
        for _ in range(tracking_iterations):
            velocity = solve(frame, tasks)
            configuration.integrate_inplace(velocity, dt / substeps)

        closure_count = 0
        for iteration in range(settings.max_feasibility_iterations + 1):
            before = hard_state()
            if settings.final_feasibility_algorithm == SELF_ONLY_FINAL_FEASIBILITY:
                done = before["self_pass"]
            else:
                done = before["all_pass"]
            if done:
                break
            if iteration == settings.max_feasibility_iterations:
                reason = (
                    "self-collision feasibility projection failed"
                    if settings.final_feasibility_algorithm
                    == SELF_ONLY_FINAL_FEASIBILITY
                    else "unified hard-feasibility closure exhausted"
                )
                acceptance_rows.append({"frame": frame, **before, "accepted": False})
                fail(frame, f"{reason} at frame {frame}")
            velocity = solve(frame, ())
            configuration.integrate_inplace(velocity, dt / substeps)
            after = hard_state()
            closure_rows.append({
                "frame": frame,
                "iteration": iteration,
                "algorithm": settings.final_feasibility_algorithm,
                "before": before,
                "after": after,
                "support_vertex_switched": (
                    before["native_geom"], before["native_vertex_index"]
                ) != (after["native_geom"], after["native_vertex_index"]),
            })
            closure_count += 1

        state = hard_state()
        acceptance_rows.append({
            "frame": frame,
            **state,
            "closure_iterations": closure_count,
            "accepted": state["all_pass"],
        })
        if not state["native_pass"]:
            fail(
                frame,
                f"native support-plane feasibility failed at frame {frame}: "
                f"{state['native_minimum_m']} m",
            )
        if settings.final_feasibility_algorithm == UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY:
            if not state["all_pass"]:
                fail(frame, f"unified hard-feasibility acceptance failed at frame {frame}")

        qpos[frame] = configuration.q
        previous = configuration.q.copy()
        native_distances[frame] = (
            np.nan if state["native_minimum_m"] is None
            else state["native_minimum_m"]
        )
        self_distances[frame] = state["self_minimum_m"]
        for hand, side in enumerate(SIDES):
            for finger_index, finger in enumerate(FINGERS):
                site = model.site(f"{side}_{finger}_tip").id
                tip_errors[frame, hand, finger_index] = np.linalg.norm(
                    configuration.data.site_xpos[site]
                    - human["T_sim_fingertip_target"][
                        frame, hand, finger_index, :3, 3
                    ]
                )
            body = model.body(f"{side}_hand_link").id
            relative = (
                configuration.data.xmat[body].reshape(3, 3).T
                @ human["T_sim_wrist_target"][frame, hand, :3, :3]
            )
            wrist_errors[frame, hand] = Rotation.from_matrix(relative).magnitude()
        if frame % 25 == 0:
            print(
                f"MINK {frame + 1}/{count}: tip mean "
                f"{tip_errors[frame].mean():.4f} m; self clearance "
                f"{self_distances[frame]:.6f} m",
                flush=True,
            )

    flush_traces()
    qvel = differentiate(model, qpos, dt)
    joint_margin = np.minimum(
        qpos[:, addresses] - ranges[:, 0],
        ranges[:, 1] - qpos[:, addresses],
    ).min(axis=1)
    velocity_ratio = np.abs(np.diff(qpos[:, addresses], axis=0)) / (dt * speeds)
    reference = {
        "qpos": qpos,
        "qvel": qvel,
        "ctrl": qpos[:, :36],
        "frequency": 1.0 / dt,
        "frame_indices": human["frame_indices"][:count],
        "timestamps_s": human["timestamps_s"][:count],
        "hand_order": np.asarray(SIDES),
        "object_roles": np.asarray(["tool", "target"]),
        "fingertip_position_error_m": tip_errors,
        "wrist_orientation_error_rad": wrist_errors,
        "self_collision_distance_m": self_distances,
        "joint_limit_min_margin": joint_margin,
        "frame_velocity_max_ratio": velocity_ratio.max(axis=1),
        "native_support_min_distance_m": native_distances,
    }
    np.savez_compressed(output / "robot_reference.npz", **reference)
    intrahand = audit_intrahand_trajectory(model, qpos)
    _write_json(output / "intrahand_collision_audit.json", {
        "scene": str(scene.resolve()),
        "scene_sha256": sha256(scene),
        "human_reference": str(human_path.resolve()),
        "human_reference_sha256": sha256(human_path),
        **intrahand,
    })
    report = {
        "status": "MINK_kinematic_candidate_not_physics_validated",
        "frames": count,
        "source_total_frames": total_frames,
        "scene": str(scene.resolve()),
        "scene_sha256": sha256(scene),
        "human_reference": str(human_path.resolve()),
        "human_reference_sha256": sha256(human_path),
        "effective_retarget_contract": effective_contract,
        "fingertip_mean_error_m": tip_errors.mean(axis=(0, 2)).tolist(),
        "fingertip_max_error_m": tip_errors.max(axis=(0, 2)).tolist(),
        "wrist_mean_error_rad": wrist_errors.mean(axis=0).tolist(),
        "self_collision_min_distance_m": float(self_distances.min()),
        "joint_limit_min_margin": float(joint_margin.min()),
        "joint_limit_violating_frames": int((joint_margin < -1e-6).sum()),
        "frame_velocity_max_ratio": float(velocity_ratio.max()),
        "frame_velocity_violating_intervals": int(
            (velocity_ratio.max(axis=1) > 1.0 + 1e-6).sum()
        ),
        "native_support_minimum_distance_m": float(np.nanmin(native_distances)),
        "strict_gate_passed": False,
        "rl_validation_completed": False,
    }
    _write_json(output / "retarget_report.json", report)
    return report
