"""User-approved first-frame hand-only feasibility diagnostics, never a reset."""

from __future__ import annotations

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .collision_audit import collision_families, distances, hand_ids, validate_qpos
from .mink import _StrictCollisionLimit, _enable_planning_collision_masks, _joint_velocity_limits


_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def scalar_hand_coordinates(model):
    """Resolve the two 18-DoF scalar hands from names and model addresses."""
    result = {}
    for side, prefixes in {
        "right": ("R_", "right_hand_"),
        "left": ("L_", "left_hand_"),
    }.items():
        rows = []
        for joint_id in range(model.njnt):
            name = model.joint(joint_id).name or ""
            if not name.startswith(prefixes):
                continue
            qpos_address = int(model.jnt_qposadr[joint_id])
            dof_address = int(model.jnt_dofadr[joint_id])
            qpos_width = (int(model.jnt_qposadr[joint_id + 1]) - qpos_address
                          if joint_id + 1 < model.njnt else model.nq - qpos_address)
            dof_width = (int(model.jnt_dofadr[joint_id + 1]) - dof_address
                         if joint_id + 1 < model.njnt else model.nv - dof_address)
            if qpos_width != 1 or dof_width != 1:
                raise ValueError(f"{side} hand joint {name} is not scalar")
            rows.append((qpos_address, dof_address, name))
        rows.sort()
        if len(rows) != 18:
            raise ValueError(f"expected 18 scalar {side} hand joints, found {len(rows)}")
        result[side] = {
            "qpos": np.asarray([row[0] for row in rows], dtype=np.int64),
            "dofs": np.asarray([row[1] for row in rows], dtype=np.int64),
            "joints": [row[2] for row in rows],
        }
    if set(result["right"]["qpos"]) & set(result["left"]["qpos"]):
        raise ValueError("left and right hand qpos mappings overlap")
    return result


def _frame_matrix(data, frame_id, frame_type):
    transform = np.eye(4)
    if frame_type == "site":
        transform[:3, :3] = data.site_xmat[frame_id].reshape(3, 3)
        transform[:3, 3] = data.site_xpos[frame_id]
    elif frame_type == "body":
        transform[:3, :3] = data.xmat[frame_id].reshape(3, 3)
        transform[:3, 3] = data.xpos[frame_id]
    else:
        raise ValueError(frame_type)
    return transform


def _left_reference_task_metrics(model, data, target_data, tasks):
    position, orientation = {}, {}
    for finger in _FINGERS:
        site = model.site(f"left_{finger}_tip").id
        position[finger] = float(np.linalg.norm(data.site_xpos[site] - target_data.site_xpos[site]))
        relative = data.site_xmat[site].reshape(3, 3).T @ target_data.site_xmat[site].reshape(3, 3)
        orientation[finger] = float(Rotation.from_matrix(relative).magnitude())
    wrist = model.body("left_hand_link").id
    wrist_relative = data.xmat[wrist].reshape(3, 3).T @ target_data.xmat[wrist].reshape(3, 3)
    errors = [np.asarray(task.compute_error(mink.Configuration(model, q=data.qpos.copy())))
              for task in tasks]
    # Use the installed MINK task cost exactly as the local weighted merit.
    merit = float(sum(np.square(np.asarray(task.cost) * error).sum()
                      for task, error in zip(tasks, errors)))
    return {
        "weighted_task_merit": merit,
        "tip_position_error_m": position,
        "tip_position_rms_m": float(np.sqrt(np.mean(np.square(list(position.values()))))),
        "tip_orientation_error_rad": orientation,
        "wrist_orientation_error_rad": float(Rotation.from_matrix(wrist_relative).magnitude()),
    }


def solve_left_reference_aligned_initial(
    model,
    seed_qpos,
    reference_qpos,
    velocity_settings,
    *,
    max_qp_calls=128,
    numerical_dt=1 / 240,
    planning_collision_buffer=2e-6,
    solver_primal_tolerance=1e-6,
    solver_dual_tolerance=1e-6,
    depenetration_step=0.002,
):
    """Run one deterministic left-only IK trajectory toward robot reference[0].

    The seed and target are deliberately separate.  This helper never selects a
    state using physics outcomes and never exits merely because the seed is
    already collision-feasible.
    """
    seed_qpos = validate_qpos(model, np.asarray(seed_qpos, dtype=float))
    reference_qpos = validate_qpos(model, np.asarray(reference_qpos, dtype=float))
    if (model.nq, model.nv, model.nu) != (50, 48, 36):
        raise ValueError("expected two scalar-coordinate hands and two free objects")
    if max_qp_calls < 1 or not np.isfinite(numerical_dt) or numerical_dt <= 0:
        raise ValueError("positive QP budget and numerical dt required")
    mapping = scalar_hand_coordinates(model)
    left_qpos = mapping["left"]["qpos"]
    left_dofs = mapping["left"]["dofs"]
    right_qpos = mapping["right"]["qpos"]
    right_dofs = mapping["right"]["dofs"]
    locked_dofs = np.asarray(sorted(set(range(model.nv)) - set(left_dofs)), dtype=np.int64)
    locked_qpos = np.asarray(sorted(set(range(model.nq)) - set(left_qpos)), dtype=np.int64)

    target_data = mujoco.MjData(model)
    target_data.qpos[:] = reference_qpos
    mujoco.mj_forward(model, target_data)
    tasks = []
    task_labels = []
    for finger in _FINGERS:
        name = f"left_{finger}_tip"
        task = mink.FrameTask(name, "site", position_cost=10.0,
                              orientation_cost=1.0, lm_damping=1e-3)
        task.set_target(mink.SE3.from_matrix(
            _frame_matrix(target_data, model.site(name).id, "site")
        ))
        tasks.append(task)
        task_labels.append(name)
    wrist_task = mink.FrameTask("left_hand_link", "body", position_cost=0.0,
                                orientation_cost=3.0, lm_damping=1e-3)
    wrist_task.set_target(mink.SE3.from_matrix(
        _frame_matrix(target_data, model.body("left_hand_link").id, "body")
    ))
    tasks.append(wrist_task)
    task_labels.append("left_hand_link_orientation")

    families = collision_families(model)
    constrained_names = ("self_explicit", "hand_tool", "hand_target", "hand_floor")
    pairs = sorted({tuple(sorted(pair)) for name in constrained_names for pair in families[name]})
    fixed_pairs = families["tool_target"] + families["tool_floor"] + families["target_floor"]
    config = mink.Configuration(model, q=seed_qpos)
    if (distances(model, config.data, fixed_pairs) < -1e-6).any():
        raise ValueError("fixed object geometry is illegal before left-only solve")
    hands = hand_ids(model)
    others = sorted({geom for pair in pairs for geom in pair} - set(hands))
    _enable_planning_collision_masks(model, hands, others)
    groups = [([model.geom(a).name], [model.geom(b).name]) for a, b in pairs]
    inner = mink.CollisionAvoidanceLimit(
        model,
        groups,
        minimum_distance_from_collisions=planning_collision_buffer,
        collision_detection_distance=0.02,
        include_explicit_pairs=True,
    )
    if set(inner.geom_id_pairs) != set(pairs):
        raise ValueError("MINK and declared initialization collision pairs differ")
    collision = _StrictCollisionLimit(
        inner, mujoco, minimum_distance=planning_collision_buffer,
        depenetration_step=depenetration_step,
    )
    collision.enabled = True
    speeds = _joint_velocity_limits(model, mujoco, velocity_settings)
    limits = [mink.ConfigurationLimit(model), collision, mink.VelocityLimit(model, speeds)]
    constraints = [mink.DofFreezingTask(model, locked_dofs.tolist())]

    states = []
    q_trace = []

    def record(iteration, *, step_norm=None, lock_residual=None):
        metrics = _left_reference_task_metrics(model, config.data, target_data, tasks)
        pair_distance = distances(model, config.data, pairs)
        states.append({
            "iteration": int(iteration),
            **metrics,
            "minimum_declared_distance_m": float(pair_distance.min()),
            "declared_penetrating_pair_count": int(np.count_nonzero(pair_distance < -1e-6)),
            "weighted_step_norm": step_norm,
            "maximum_locked_dof_velocity": lock_residual,
        })
        q_trace.append(config.q.copy())

    record(0)
    reason = "qp_budget_exhausted"
    qp_attempts = 0
    for call in range(max_qp_calls):
        qp_attempts += 1
        try:
            velocity = mink.solve_ik(
                config, tasks, numerical_dt, solver="daqp", damping=1e-5,
                limits=limits, constraints=constraints,
                primal_tol=solver_primal_tolerance, dual_tol=solver_dual_tolerance,
            )
        except Exception as error:
            reason = f"qp_failed:{type(error).__name__}:{error}"
            break
        if not np.isfinite(velocity).all():
            reason = "nonfinite_qp_velocity"
            break
        lock_residual = float(np.max(np.abs(velocity[locked_dofs])))
        if lock_residual > 1e-8:
            reason = "locked_dof_residual_exceeded"
            break
        velocity[locked_dofs] = 0.0
        candidate = config.q.copy()
        candidate[left_qpos] += velocity[left_dofs] * numerical_dt
        if not np.array_equal(candidate[locked_qpos], seed_qpos[locked_qpos]):
            raise RuntimeError("left-only numerical update changed a locked coordinate")
        step_norm = float(np.linalg.norm(candidate[left_qpos] - config.q[left_qpos]))
        config.update(candidate)
        record(call + 1, step_norm=step_norm, lock_residual=lock_residual)
        if step_norm < 1e-10:
            reason = "no_further_numerical_progress"
            break

    return np.asarray(q_trace), {
        "termination_reason": reason,
        "qp_calls": qp_attempts,
        "states": states,
        "task_labels": task_labels,
        "left_qpos_addresses": left_qpos.tolist(),
        "left_dof_addresses": left_dofs.tolist(),
        "right_qpos_addresses": right_qpos.tolist(),
        "right_dof_addresses": right_dofs.tolist(),
        "locked_qpos_addresses": locked_qpos.tolist(),
        "locked_dof_addresses": locked_dofs.tolist(),
        "locked_coordinates_byte_identical": bool(
            np.array_equal(
                np.asarray(q_trace)[:, locked_qpos],
                np.broadcast_to(seed_qpos[locked_qpos], (len(q_trace), len(locked_qpos))),
            )
        ),
        "seed_is_target": bool(np.array_equal(seed_qpos, reference_qpos)),
        "numerical_dt_s": float(numerical_dt),
        "planning_collision_buffer_m": float(planning_collision_buffer),
        "solver_primal_tolerance": float(solver_primal_tolerance),
        "solver_dual_tolerance": float(solver_dual_tolerance),
        "objective_provenance": "local left-only robot-reference[0] task-space alignment; not a published reset recipe",
    }


def state_summary(model, qpos, *, tolerance=1e-6):
    qpos = validate_qpos(model, qpos)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("finite nonnegative collision tolerance required")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    families = {}
    for name, pairs in collision_families(model).items():
        value = distances(model, data, pairs)
        worst = int(value.argmin())
        a, b = pairs[worst]
        families[name] = dict(pair_count=len(pairs), min_distance_m=float(value[worst]),
                              violations=int((value < -tolerance).sum()),
                              worst_pair=[model.geom(a).name, model.geom(b).name])
    limited = np.flatnonzero(model.jnt_limited)
    addresses = model.jnt_qposadr[limited]
    margin = np.minimum(qpos[addresses] - model.jnt_range[limited, 0],
                        model.jnt_range[limited, 1] - qpos[addresses])
    constrained = [key for key in families if key != "self_nonadjacent_shells"]
    return dict(families=families, joint_min_margin=float(margin.min()),
                joint_violations=int((margin < -tolerance).sum()),
                declared_state_feasible=bool(margin.min() >= -tolerance and
                    all(families[key]["violations"] == 0 for key in constrained)))


def above_fixed_geometry_seed(model, original):
    """Geometric search seed only: raise hand AABBs above fixed object AABBs."""
    original = validate_qpos(model, original)
    data = mujoco.MjData(model)
    data.qpos[:] = original
    mujoco.mj_forward(model, data)

    def z_bounds(g):
        rotation = data.geom_xmat[g].reshape(3, 3)
        center = data.geom_xpos[g] + rotation @ model.geom_aabb[g, :3]
        radius = np.abs(rotation) @ model.geom_aabb[g, 3:]
        return center[2] - radius[2], center[2] + radius[2]

    objects = {g for family in ("hand_tool", "hand_target")
               for _, g in collision_families(model)[family]}
    top = max(float(model.geom("floor").pos[2]), *(z_bounds(g)[1] for g in objects))
    candidate = original.copy()
    offsets = {}
    for side in ("right", "left"):
        geoms = [g for g in hand_ids(model) if model.geom(g).name.startswith(f"collision_hand_{side}_")]
        bottom = min(z_bounds(g)[0] for g in geoms)
        offset = max(0.0, top - bottom + 1e-6)
        prefix = "R_" if side == "right" else "L_"
        joints = [j for j in range(model.njnt) if (model.joint(j).name or "").startswith(prefix)
                  and "_tz_" in (model.joint(j).name or "")]
        if len(joints) != 1:
            raise ValueError("one Cartesian z-translation joint per hand required")
        joint = joints[0]
        if not np.allclose(data.xaxis[joint], [0, 0, 1]):
            raise ValueError("seed requires the inherited world-z hand translation convention")
        candidate[int(model.jnt_qposadr[joint])] += offset
        offsets[side] = offset
    return candidate, dict(method="object_top_minus_hand_bottom_from_current_geom_AABBs",
                           fixed_object_top_m=float(top), hand_z_offsets_m=offsets,
                           numerical_padding_m=1e-6, executed_as_motion=False)


def solve_initial_hands(model, original, velocity_settings, *, source_dt=1 / 30,
                        max_iterations=128, backtrack_separation=False, seed_mode="original",
                        planning_collision_buffer=0.0, accepted_min_distance=-1e-6,
                        solver_primal_tolerance=1e-9, solver_dual_tolerance=1e-9,
                        depenetration_step=0.002):
    """Find a nearby hand-only candidate under the unchanged declared geometry.

    Costs normalize scalar joint changes by the inherited one-frame velocity
    envelope. This is a local engineering posture objective, not Eq. (1), an RL
    reward or a globally minimum projection. Numerical iterations are not time.
    """
    original = np.asarray(original, dtype=float)
    if (model.nq, model.nv, model.nu) != (50, 48, 36):
        raise ValueError("expected two scalar-coordinate hands and two free objects")
    if original.shape != (model.nq,) or not np.isfinite(original).all():
        raise ValueError("expected one finite initial qpos")
    validate_qpos(model, original)
    if not np.isfinite(source_dt) or source_dt <= 0 or max_iterations < 1:
        raise ValueError("positive source dt and iteration budget required")
    numerical = (
        planning_collision_buffer, accepted_min_distance,
        solver_primal_tolerance, solver_dual_tolerance, depenetration_step,
    )
    if not np.isfinite(numerical).all():
        raise ValueError("finite collision and solver settings required")
    if (planning_collision_buffer < 0 or accepted_min_distance > planning_collision_buffer
            or solver_primal_tolerance <= 0 or solver_dual_tolerance <= 0
            or depenetration_step <= 0):
        raise ValueError("invalid collision or solver settings")
    families = collision_families(model)
    names = ("self_explicit", "hand_tool", "hand_target", "hand_floor")
    pairs = sorted({tuple(sorted(pair)) for name in names for pair in families[name]})
    fixed_pairs = families["tool_target"] + families["tool_floor"] + families["target_floor"]
    if seed_mode not in ("original", "above_fixed_geometry"):
        raise ValueError("unknown numerical seed mode")
    seed, seed_report = ((original.copy(), dict(method="original_first_frame")) if seed_mode == "original"
                         else above_fixed_geometry_seed(model, original))
    config = mink.Configuration(model, q=seed)
    if (distances(model, config.data, fixed_pairs) < -1e-6).any():
        raise ValueError("fixed objects already violate the declared geometry; hand-only solve cannot fix them")
    hands = hand_ids(model)
    others = sorted({g for pair in pairs for g in pair} - set(hands))
    _enable_planning_collision_masks(model, hands, others)
    groups = [([model.geom(a).name], [model.geom(b).name]) for a, b in pairs]
    inner = mink.CollisionAvoidanceLimit(
        model, groups, minimum_distance_from_collisions=planning_collision_buffer,
                                         collision_detection_distance=0.02, include_explicit_pairs=True)
    if set(inner.geom_id_pairs) != set(pairs):
        raise ValueError("MINK and declared runtime hand/environment pairs differ")
    collision = _StrictCollisionLimit(
        inner, mujoco, minimum_distance=planning_collision_buffer,
        depenetration_step=depenetration_step,
    )
    collision.enabled = True
    speeds = _joint_velocity_limits(model, mujoco, velocity_settings)
    if len(speeds) != 36:
        raise ValueError("all 36 scalar hand joints require inherited speed limits")
    costs = np.zeros(model.nv)
    addresses, dofs = [], []
    for name, speed in speeds.items():
        joint = model.joint(name)
        address, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        addresses.append(address)
        dofs.append(dof)
        costs[dof] = 1.0 / (source_dt * speed)
    object_dofs = sorted(set(range(model.nv)) - set(dofs))
    fixed_qpos = sorted(set(range(model.nq)) - set(addresses))
    posture = mink.PostureTask(model, cost=costs)
    posture.set_target(original)
    limits = [mink.ConfigurationLimit(model), collision, mink.VelocityLimit(model, speeds)]
    locks = [mink.DofFreezingTask(model, object_dofs)]
    integration_dt = source_dt / 8
    trace = [seed.copy()]
    history = []
    attempts = []
    reason = "iteration_budget_exhausted"
    best_q, best_cost, best_iteration = None, np.inf, None
    last_step_norm = np.inf
    for iteration in range(max_iterations + 1):
        value = distances(model, config.data, pairs)
        minimum = float(value.min())
        history.append(dict(iteration=iteration, minimum_declared_hand_distance_m=minimum,
                            penetrating_pairs=int((value < -1e-6).sum())))
        if minimum >= accepted_min_distance:
            cost = float(np.square((config.q[addresses] - original[addresses]) * costs[dofs]).sum())
            if cost < best_cost:
                best_q, best_cost, best_iteration = config.q.copy(), cost, iteration
            if seed_mode == "original" or last_step_norm < 1e-8:
                reason = "declared_hand_constraints_satisfied"
                break
        if iteration == max_iterations:
            break
        collision.depenetration_step = depenetration_step
        velocity = None
        last_error = None
        while collision.depenetration_step >= 1e-6:
            try:
                velocity = mink.solve_ik(config, [posture], integration_dt, solver="daqp", damping=1e-5,
                                         limits=limits, constraints=locks,
                                         primal_tol=solver_primal_tolerance,
                                         dual_tol=solver_dual_tolerance)
                attempts.append(dict(iteration=iteration, separation_step_m=collision.depenetration_step, solved=True))
                break
            except mink.NoSolutionFound as error:
                last_error = f"qp_failed: {type(error).__name__}: {error}"
                attempts.append(dict(iteration=iteration, separation_step_m=collision.depenetration_step, solved=False))
                if not backtrack_separation:
                    break
                collision.depenetration_step *= 0.5
        if velocity is None:
            reason = last_error or "separation_backtracking_exhausted"
            break
        if not np.isfinite(velocity).all() or np.abs(velocity[object_dofs]).max() > 1e-8:
            reason = "nonfinite_solution_or_object_velocity_lock_violation"
            break
        # Enforce exact zero only after verifying the equality residual. Integrate
        # no object DOF, including unit-quaternion renormalization roundoff.
        velocity[object_dofs] = 0.0
        candidate = config.q.copy()
        candidate[addresses] += velocity[dofs] * integration_dt
        if not np.array_equal(candidate[fixed_qpos], original[fixed_qpos]):
            raise RuntimeError("hand-only integration changed object coordinates")
        last_step_norm = float(np.linalg.norm((candidate[addresses] - config.q[addresses]) * costs[dofs]))
        config.update(candidate)
        trace.append(candidate.copy())
        if iteration % 10 == 0:
            print(f"initial-hand QP {iteration + 1}: prior minimum {minimum:.6f} m", flush=True)
    result = best_q if best_q is not None else config.q.copy()
    return result, dict(termination_reason=reason, iterations=len(trace) - 1,
        numerical_seed=seed_report, retained_feasible_iteration=best_iteration,
        retained_normalized_posture_cost=(best_cost if best_q is not None else None),
        globally_minimum_correction_proven=False,
        constrained_hand_pair_count=len(pairs), fixed_object_pair_count=len(fixed_pairs),
        solver_history=history, qp_attempts=attempts, separation_backtracking=backtrack_separation,
        hand_qpos_addresses=addresses, frozen_qpos_addresses=fixed_qpos,
        normalized_posture_cost=costs.tolist(), qp_integration_dt_s=integration_dt,
        object_coordinates_byte_identical=bool(np.array_equal(result[fixed_qpos], original[fixed_qpos])),
        step_limit_provenance="inherited speed limits times source_dt/8; numerical trust region, not elapsed time",
        objective_provenance="user-approved local normalized posture objective, not a published EgoEngine reset",
        minimum_distance_m=float(planning_collision_buffer),
        accepted_min_distance_m=float(accepted_min_distance),
        distance_acceptance_tolerance_m=float(max(0.0, -accepted_min_distance)),
        depenetration_step_m=float(depenetration_step),
        solver_primal_tolerance=float(solver_primal_tolerance),
        solver_dual_tolerance=float(solver_dual_tolerance)), np.asarray(trace)


def change_summary(model, before, after, human):
    data_before, data_after = mujoco.MjData(model), mujoco.MjData(model)
    data_before.qpos[:], data_after.qpos[:] = before, after
    mujoco.mj_forward(model, data_before)
    mujoco.mj_forward(model, data_after)
    result = {}
    for index, side in enumerate(("right", "left")):
        root = model.body(f"{side}_hand_link").id
        relative = data_before.xmat[root].reshape(3, 3).T @ data_after.xmat[root].reshape(3, 3)
        tips = [model.site(f"{side}_{finger}_tip").id for finger in ("thumb", "index", "middle", "ring", "pinky")]
        target = human["T_sim_fingertip_target"][0, index, :, :3, 3]
        result[side] = dict(wrist_translation_delta_m=(data_after.xpos[root] - data_before.xpos[root]).tolist(),
            wrist_translation_norm_m=float(np.linalg.norm(data_after.xpos[root] - data_before.xpos[root])),
            wrist_rotation_delta_rad=float(Rotation.from_matrix(relative).magnitude()),
            finger_joint_delta_max_rad=float(np.abs(after[index * 18 + 6:index * 18 + 18] -
                                                   before[index * 18 + 6:index * 18 + 18]).max()),
            baseline_tip_mean_error_m=float(np.linalg.norm(data_before.site_xpos[tips] - target, axis=1).mean()),
            candidate_tip_mean_error_m=float(np.linalg.norm(data_after.site_xpos[tips] - target, axis=1).mean()))
    return result


def next_reference_diagnostic(model, candidate, next_qpos, velocity_settings, source_dt):
    speeds = _joint_velocity_limits(model, mujoco, velocity_settings)
    ratios = []
    for name, speed in speeds.items():
        address = int(model.joint(name).qposadr[0])
        ratios.append(dict(joint=name, ratio=float(abs(next_qpos[address] - candidate[address]) / (source_dt * speed))))
    delta = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, delta, 1.0, candidate, next_qpos)
    samples = []
    for alpha in np.linspace(0, 1, 11):
        qpos = candidate.copy()
        mujoco.mj_integratePos(model, qpos, delta, float(alpha))
        state = state_summary(model, qpos)
        samples.append(dict(alpha=float(alpha), declared_state_feasible=state["declared_state_feasible"],
                            minimum_distances_m={key: value["min_distance_m"] for key, value in state["families"].items()}))
    return dict(source_interval=[0, 1], source_dt_s=source_dt, interpolation_samples=samples,
                max_speed_limit_ratio=max(r["ratio"] for r in ratios),
                speed_violations=[r for r in ratios if r["ratio"] > 1 + 1e-6],
                reference_row_1_modified=False, physical_transition_simulated=False,
                interpretation="diagnostic linear/manifold path, not Replay dynamics, a new reference or an RL entry gate")
