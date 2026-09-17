"""Isolated IK objective/constraint probes, never a replacement reference or reset."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
import mink
from audit_taco_initialization import visual_meshes
from egoengine_repro.retarget.collision_audit import distances, explicit_hand_pairs
from egoengine_repro.retarget.mink import (
    _FrameDisplacementLimit, _StrictCollisionLimit, _enable_planning_collision_masks,
    _explicit_collision_groups, _joint_velocity_limits,
)
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts
from egoengine_repro.retarget.taco_bimanual import FINGERS, SIDES, geometric_frame

RUN = ROOT / "runs/taco_pour_bimanual_mano_fk_v1"
UPSTREAM = Path("/data_all/zzx/egoengine/spider/spider/assets/robots/xhand")
TEMPLATE = ROOT / "models/taco_xhand/templates/xhand_bimanual_source.xml"
SAMPLED_ROWS = (0, 20, 50, 100, 150, 197)


class WorldPositionTask(mink.Task):
    """Euclidean site position only; no latent orientation in an SE(3) log."""

    def __init__(self, model, site_name, target, cost=10.):
        super().__init__(cost=np.full(3, cost), lm_damping=1e-3)
        self.site_id = model.site(site_name).id
        self.target = np.asarray(target, dtype=float).copy()
        if self.target.shape != (3,) or not np.isfinite(self.target).all():
            raise ValueError("finite three-dimensional target required")

    def compute_error(self, configuration):
        return configuration.data.site_xpos[self.site_id] - self.target

    def compute_jacobian(self, configuration):
        jac = np.zeros((3, configuration.model.nv))
        mujoco.mj_jacSite(configuration.model, configuration.data, jac, None, self.site_id)
        return jac


def fixed_axis_residual(targets, fixed_coordinates):
    """Best common translation on an invariant axis, with wrist rotation fixed."""
    offsets = np.asarray(targets) - np.asarray(fixed_coordinates)
    return offsets - offsets.mean(axis=-1, keepdims=True)


def urdf_fk(urdf, root_name, joint_values):
    transforms = {root_name: np.eye(4)}
    pending = list(urdf.findall("joint"))
    while pending:
        advanced = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            local = np.eye(4)
            local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            if joint.get("type") == "revolute":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                local[:3, :3] = local[:3, :3] @ Rotation.from_rotvec(axis * joint_values[joint.get("name")]).as_matrix()
            elif joint.get("type") != "fixed":
                raise ValueError("unexpected URDF joint type")
            transforms[joint.find("child").get("link")] = transforms[parent] @ local
            pending.remove(joint)
            advanced = True
        if not advanced:
            raise ValueError("URDF chain does not connect to expected root")
    return transforms


def load_inputs():
    report = json.loads((RUN / "retarget_report.json").read_text())
    scene = Path(report["scene"])
    if artifact(scene)["sha256"] != report["scene_sha256"]:
        raise ValueError("scene differs from preserved reference")
    with np.load(RUN / "human_reference.npz", allow_pickle=False) as src:
        human = dict(src)
    with np.load(RUN / "robot_reference.npz", allow_pickle=False) as src:
        robot = dict(src)
    inputs = [artifact(RUN / name) for name in (
        "human_reference.npz", "robot_reference.npz", "retarget_report.json", "input_audit.json")]
    inputs += [artifact(scene), *scene_mesh_artifacts(scene), artifact(TEMPLATE)]
    inputs += [artifact(UPSTREAM / f"{side}.xml") for side in SIDES]
    inputs += [artifact(UPSTREAM / f"xhand_{side}.urdf") for side in SIDES]
    return scene, report["inherited_settings"], human, robot, inputs


def semantic_audit(scene, human, robot):
    model = mujoco.MjModel.from_xml_path(str(scene))
    config = mink.Configuration(model)
    template = ET.parse(TEMPLATE).getroot()
    records = {}
    zero = config.data
    for hi, side in enumerate(SIDES):
        upstream = ET.parse(UPSTREAM / f"{side}.xml").getroot()
        urdf = ET.parse(UPSTREAM / f"xhand_{side}.urdf").getroot()
        root = model.body(f"{side}_hand_link").id
        palm = model.site(f"{side}_palm").id
        middle = model.body(f"{side}_hand_mid_link1").id
        robot_frame = geometric_frame(zero.site_xmat[palm].reshape(3, 3)[:, 0],
                                      zero.xpos[middle] - zero.xpos[root])
        sites, joint_checks = {}, []
        for finger in FINGERS:
            name = f"{side}_{finger}_tip"
            sid = model.site(name).id
            body = int(model.site_bodyid[sid])
            body_name = model.body(body).name
            point = model.site_pos[sid]
            for tree in (template, upstream):
                np.testing.assert_allclose(np.fromstring(tree.find(f".//site[@name='{name}']").get("pos"), sep=" "), point,
                                           atol=1e-12, rtol=0)
            tip_joint = next(j for j in urdf.findall("joint") if j.get("type") == "fixed"
                             and j.find("parent").get("link") == body_name
                             and "tip" in j.find("child").get("link"))
            endpoint = np.fromstring(tip_joint.find("origin").get("xyz"), sep=" ")
            np.testing.assert_allclose(np.fromstring(tip_joint.find("origin").get("rpy"), sep=" "), 0, atol=1e-12)
            site_frame = geometric_frame(robot_frame[:, 0], zero.site_xpos[sid] - zero.xpos[body])
            end_world = zero.xmat[body].reshape(3, 3) @ endpoint
            end_frame = geometric_frame(robot_frame[:, 0], end_world)
            angle = Rotation.from_matrix(site_frame.T @ end_frame).magnitude()
            sites[finger] = dict(site_name=name, body=body_name, inherited_site_m=point.tolist(),
                upstream_xml_site_identical=True, template_site_identical=True,
                urdf_fixed_tip_joint=tip_joint.get("name"), urdf_tip_m=endpoint.tolist(),
                site_to_urdf_tip_distance_m=float(np.linalg.norm(point - endpoint)),
                site_axis_vs_urdf_endpoint_axis_deg=float(np.rad2deg(angle)),
                identity_site_quaternion=model.site_quat[sid].tolist())
        for joint in urdf.findall("joint"):
            if joint.get("type") != "revolute":
                continue
            mj = model.joint(joint.get("name"))
            limit = joint.find("limit")
            expected = np.array([float(limit.get("lower")), float(limit.get("upper"))])
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            joint_checks.append(dict(joint=joint.get("name"),
                range_max_difference_rad=float(np.abs(expected - mj.range).max()),
                axis_max_difference=float(np.abs(axis - mj.axis).max())))
        records[side] = dict(sites=sites, urdf_joint_checks=joint_checks)

    fk_checks = {}
    for hi, side in enumerate(SIDES):
        urdf = ET.parse(UPSTREAM / f"xhand_{side}.urdf").getroot()
        root_name = f"{side}_hand_link"
        root = model.body(root_name).id
        position_error, angle_error = 0., 0.
        for q in robot["qpos"]:
            config.update(q)
            values = {j.get("name"): q[int(model.joint(j.get("name")).qposadr[0])]
                      for j in urdf.findall("joint") if j.get("type") == "revolute"}
            fk = urdf_fk(urdf, root_name, values)
            for body_name, local in fk.items():
                if body_name.endswith("tip") or body_name == f"{side}_hand_ee_link":
                    continue
                bid = model.body(body_name).id
                Rroot = config.data.xmat[root].reshape(3, 3)
                actual_p = Rroot.T @ (config.data.xpos[bid] - config.data.xpos[root])
                actual_R = Rroot.T @ config.data.xmat[bid].reshape(3, 3)
                position_error = max(position_error, float(np.linalg.norm(actual_p - local[:3, 3])))
                angle_error = max(angle_error, float(Rotation.from_matrix(actual_R.T @ local[:3, :3]).magnitude()))
        fk_checks[side] = dict(all_source_rows=len(robot["qpos"]), max_body_position_difference_m=position_error,
                               max_body_rotation_difference_rad=angle_error)

    # Middle/ring/pinky have only collinear +/-X revolute axes in the wrist
    # frame. Their X coordinates are invariant to both finger joint angles.
    invariant = {}
    for hi, side in enumerate(SIDES):
        body = model.body(f"{side}_hand_link").id
        ids = [model.site(f"{side}_{finger}_tip").id for finger in FINGERS[2:]]
        config.update(model.qpos0)
        R0 = config.data.xmat[body].reshape(3, 3)
        coordinates = ((config.data.site_xpos[ids] - config.data.xpos[body]) @ R0)[:, 0]
        measured_drift = 0.
        rng = np.random.default_rng(724 + hi)
        for _ in range(32):
            q = model.qpos0.copy()
            q[hi * 18 + 6:hi * 18 + 18] = rng.uniform(
                model.jnt_range[hi * 18 + 6:hi * 18 + 18, 0],
                model.jnt_range[hi * 18 + 6:hi * 18 + 18, 1])
            config.update(q)
            coords = ((config.data.site_xpos[ids] - config.data.xpos[body]) @ R0)[:, 0]
            measured_drift = max(measured_drift, float(np.abs(coords - coordinates).max()))
        if measured_drift > 1e-10:
            raise ValueError("invariant-axis assumption does not hold")
        bounds = {}
        for mode in ("target_wrist", "saved_wrist"):
            residuals = []
            for row, q in enumerate(robot["qpos"]):
                config.update(q)
                rot = (human["T_sim_wrist_target"][row, hi, :3, :3] if mode == "target_wrist"
                       else config.data.xmat[body].reshape(3, 3))
                target_x = human["T_sim_fingertip_target"][row, hi, 2:, :3, 3] @ rot[:, 0]
                residuals.append(fixed_axis_residual(target_x, coordinates))
            residuals = np.asarray(residuals)
            bounds[mode] = dict(centered_axis_residual_m=residuals.tolist(),
                rms_position_lower_bound_three_fingers_m=np.sqrt(np.mean(residuals ** 2, axis=1)).tolist(),
                mean_abs_centered_residual_by_finger_m=np.abs(residuals).mean(axis=0).tolist())
        invariant[side] = dict(fingers=list(FINGERS[2:]), wrist_local_fixed_x_m=coordinates.tolist(),
            maximum_random_joint_test_drift_m=measured_drift, rotation_conditioned_bounds=bounds,
            limitation="Lower bound on aggregate squared position loss with this wrist rotation fixed; not an individual-finger or free-wrist lower bound.")
    return dict(site_provenance=records, independent_urdf_fk_vs_mujoco=fk_checks, fixed_wrist_axis_diagnostic=invariant,
        author_point_and_axis_calibration_recovered=False,
        interpretation="SPIDER XML sites differ from URDF named endpoints; neither is proved the paper's intended MANO correspondence.")


class Experiment:
    def __init__(self, scene, settings, human, robot):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.human, self.robot, self.settings = human, robot, settings
        self.config = mink.Configuration(self.model)
        self.pairs = explicit_hand_pairs(self.model)
        geoms = [i for i in range(self.model.ngeom) if (self.model.geom(i).name or "").startswith("collision_hand_")]
        _enable_planning_collision_masks(self.model, geoms, [])
        groups = _explicit_collision_groups(self.model, mujoco, hand_geom_ids=set(geoms))
        inner = mink.CollisionAvoidanceLimit(self.model, groups,
            minimum_distance_from_collisions=0., collision_detection_distance=.02)
        if set(inner.geom_id_pairs) != set(self.pairs):
            raise ValueError("collision pair contract differs")
        self.collision = _StrictCollisionLimit(inner, mujoco, minimum_distance=0., depenetration_step=.002)
        self.collision.enabled = True
        self.velocity = _joint_velocity_limits(self.model, mujoco, settings["velocity_limits"])
        self.displacement = _FrameDisplacementLimit(self.model, mujoco, self.velocity, mink.Constraint)
        self.joint_limit = mink.ConfigurationLimit(self.model)
        self.locks = [mink.DofFreezingTask(self.model, list(range(36, 48)))]
        self.dt = float(np.diff(human["timestamps_s"])[0])
        self.qp_dt = self.dt / settings["max_iterations_per_frame"]
        self.addresses = np.array([int(self.model.joint(n).qposadr[0]) for n in self.velocity])
        self.speeds = np.array(list(self.velocity.values()))
        self.meshes, _ = visual_meshes(scene, self.model)

    def tasks(self, row, mode):
        tasks = []
        for hi, side in enumerate(SIDES):
            if mode != "position_only":
                task = mink.FrameTask(f"{side}_hand_link", "body", position_cost=0.,
                    orientation_cost=self.settings["wrist_orientation_cost"], lm_damping=1e-3)
                task.set_target(mink.SE3.from_matrix(self.human["T_sim_wrist_target"][row, hi]))
                tasks.append(task)
            for k, finger in enumerate(FINGERS):
                target = self.human["T_sim_fingertip_target"][row, hi, k]
                site = f"{side}_{finger}_tip"
                if mode in ("pose", "se3_no_tip_rotation"):
                    task = mink.FrameTask(site, "site", position_cost=self.settings["fingertip_position_cost"],
                        orientation_cost=self.settings["fingertip_orientation_cost"] if mode == "pose" else 0., lm_damping=1e-3)
                    task.set_target(mink.SE3.from_matrix(target))
                    tasks.append(task)
                else:
                    tasks.append(WorldPositionTask(self.model, site, target[:3, 3], self.settings["fingertip_position_cost"]))
                    if mode == "split_pose":
                        task = mink.FrameTask(site, "site", position_cost=0.,
                            orientation_cost=self.settings["fingertip_orientation_cost"], lm_damping=1e-3)
                        task.set_target(mink.SE3.from_matrix(target))
                        tasks.append(task)
        return tasks

    def metrics(self, row):
        position, orientation, wrist, table = [], [], [], []
        for hi, side in enumerate(SIDES):
            ids = [self.model.site(f"{side}_{finger}_tip").id for finger in FINGERS]
            targets = self.human["T_sim_fingertip_target"][row, hi]
            position.append(np.linalg.norm(self.config.data.site_xpos[ids] - targets[:, :3, 3], axis=1).tolist())
            actual = self.config.data.site_xmat[ids].reshape(5, 3, 3)
            orientation.append(np.rad2deg(Rotation.from_matrix(actual.swapaxes(-1, -2) @ targets[:, :3, :3]).magnitude()).tolist())
            root = self.model.body(f"{side}_hand_link").id
            wrist.append(float(np.rad2deg(Rotation.from_matrix(self.config.data.xmat[root].reshape(3, 3).T @
                self.human["T_sim_wrist_target"][row, hi, :3, :3]).magnitude())))
            clearances = []
            for gid, mesh in self.meshes.items():
                name = self.model.geom(gid).name
                if not name.startswith(side + "_") or name == side + "_object_visual":
                    continue
                body = self.model.geom_bodyid[gid]
                clearances.append(float((mesh.vertices @ self.config.data.xmat[body].reshape(3, 3)[2]).min()
                                        + self.config.data.xpos[body, 2] - .72))
            table.append(min(clearances))
        q = self.config.q
        joint_margin = float(np.minimum(q[:36] - self.model.jnt_range[:36, 0], self.model.jnt_range[:36, 1] - q[:36]).min())
        np.testing.assert_allclose(q[36:], self.robot["qpos"][row, 36:], atol=1e-12, rtol=0)
        speed = (float((np.abs(q[self.addresses] - self.robot["qpos"][row - 1, self.addresses]) /
                        (self.dt * self.speeds)).max()) if row else None)
        return dict(position_error_m=position, fingertip_orientation_error_deg=orientation,
                    wrist_orientation_error_deg=wrist, native_hand_table_min_clearance_m=table,
                    joint_min_margin=joint_margin, self_min_distance_m=float(distances(self.model, self.config.data, self.pairs).min()),
                    original_previous_frame_velocity_max_ratio=speed)

    def legal_endpoint(self, q, use_speed=True):
        margin = np.minimum(q[:36] - self.model.jnt_range[:36, 0], self.model.jnt_range[:36, 1] - q[:36]).min()
        speed_ok = (not use_speed or self.displacement.lower is None or
            (np.all(q[self.addresses] >= self.displacement.lower - 1e-9) and
             np.all(q[self.addresses] <= self.displacement.upper + 1e-9)))
        return bool(margin >= -1e-6 and speed_ok and distances(self.model, self.config.data, self.pairs).min() >= -1e-6)

    def solve(self, row, mode, max_iterations=160, use_speed=True, seed=None):
        start = self.robot["qpos"][row] if seed is None else seed
        self.config.update(start)
        self.displacement.set_previous(self.robot["qpos"][row - 1] if row and use_speed else None,
                                       self.dt if row and use_speed else None)
        tasks = self.tasks(row, mode)
        limits = [self.joint_limit, self.collision, self.displacement]
        history, converged, stalled, backtracks = [], False, False, 0
        if not self.legal_endpoint(self.config.q, use_speed):
            raise ValueError("diagnostic seed violates declared endpoint constraints")
        for iteration in range(max_iterations):
            dq = mink.solve_ik(self.config, tasks, self.qp_dt, solver="daqp", damping=1e-5,
                limits=limits, constraints=self.locks, primal_tol=1e-9, dual_tol=1e-9) * self.qp_dt
            if not np.isfinite(dq).all():
                raise ValueError("nonfinite IK step")
            max_step = float(np.abs(dq[:36]).max())
            loss = float(sum(np.sum((t.cost * t.compute_error(self.config)) ** 2) for t in tasks))
            if iteration in (0, 7, 19, 79, max_iterations - 1):
                history.append(dict(iteration=iteration, weighted_loss=loss,
                                    maximum_proposed_hand_increment=max_step))
            if max_step < 1e-8 and distances(self.model, self.config.data, self.pairs).min() >= -1e-6:
                converged = True
                break
            # This diagnostic backtracking is not used by the production
            # retargeter. Keep every accepted endpoint legal and non-increasing
            # in the measured nonlinear task loss; report stalls separately.
            before = self.config.q.copy()
            accepted = False
            for trial in range(21):
                self.config.update(before)
                self.config.integrate_inplace((.5 ** trial) * dq / self.qp_dt, self.qp_dt)
                new_loss = float(sum(np.sum((t.cost * t.compute_error(self.config)) ** 2) for t in tasks))
                if self.legal_endpoint(self.config.q, use_speed) and new_loss <= loss + 1e-14:
                    accepted = True
                    backtracks += trial
                    break
            if not accepted:
                self.config.update(before)
                stalled = True
                break
        metrics = self.metrics(row)
        valid = (metrics["joint_min_margin"] >= -1e-6 and metrics["self_min_distance_m"] >= -1e-6
                 and (not use_speed or not row or metrics["original_previous_frame_velocity_max_ratio"] <= 1 + 1e-6))
        return dict(source_row_zero_based=row, mode=mode, speed_envelope_enabled=use_speed,
            iterations=iteration + 1, converged=converged, line_search_stalled=stalled,
            diagnostic_backtracking_steps=backtracks,
            constraint_gate_passed=valid, last_proposed_increment=max_step, history=history, **metrics), self.config.q.copy()

    def relaxed_bound_probe(self, row, mode, q):
        self.config.update(q)
        tasks = self.tasks(row, mode)
        before = self.config.q.copy()
        records = []
        for label, limits in (("joint_and_self", [self.joint_limit, self.collision]),
                              ("self_only_joint_bounds_omitted", [self.collision])):
            dq = mink.solve_ik(self.config, tasks, self.qp_dt, solver="daqp", damping=1e-5,
                limits=limits, constraints=self.locks, primal_tol=1e-9, dual_tol=1e-9) * self.qp_dt
            predicted = float(sum(np.sum((t.cost * (t.compute_error(self.config) + t.compute_jacobian(self.config) @ dq)) ** 2) for t in tasks))
            current = float(sum(np.sum((t.cost * t.compute_error(self.config)) ** 2) for t in tasks))
            illegal = []
            for j in range(36):
                value = q[j] + dq[j]
                low, high = self.model.jnt_range[j]
                if value < low - 1e-6 or value > high + 1e-6:
                    illegal.append(dict(joint=self.model.joint(j).name, proposed_q=float(value), lower=float(low), upper=float(high)))
            records.append(dict(constraints=label, current_weighted_loss=current,
                linearized_next_weighted_loss=predicted, maximum_unapplied_hand_increment=float(np.abs(dq[:36]).max()),
                would_violate_joint_bounds=illegal, state_integrated=False))
        np.testing.assert_array_equal(self.config.q, before)
        return dict(source_row_zero_based=row, mode=mode, probes=records)

    def legal_seeds(self, row, count=3):
        rng = np.random.default_rng(9182 + row)
        source = self.robot["qpos"][row]
        seeds = []
        for _ in range(100):
            q = source.copy()
            for hi in range(2):
                q[hi * 18 + 3:hi * 18 + 6] += rng.uniform(-.12, .12, 3)
                adr = np.arange(hi * 18 + 6, hi * 18 + 18)
                q[adr] = np.clip(q[adr] + rng.uniform(-.4, .4, 12),
                    self.model.jnt_range[adr, 0] + 1e-6, self.model.jnt_range[adr, 1] - 1e-6)
            self.config.update(q)
            if distances(self.model, self.config.data, self.pairs).min() >= -1e-6:
                seeds.append(q)
                if len(seeds) == count:
                    return seeds
        raise ValueError("could not obtain declared-constraint-legal diagnostic seeds")


def summarize(rows):
    valid = [r for r in rows if r["constraint_gate_passed"]]
    errors = np.asarray([r["position_error_m"] for r in valid])
    return dict(rows=len(rows), valid_rows=len(valid), converged_rows=sum(r["converged"] for r in rows),
        stalled_rows=sum(r["line_search_stalled"] for r in rows),
        mean_position_m_by_hand=errors.mean(axis=(0, 2)).tolist() if len(errors) else None,
        mean_position_m_by_hand_finger=errors.mean(axis=0).tolist() if len(errors) else None,
        maximum_position_m_by_hand=errors.max(axis=(0, 2)).tolist() if len(errors) else None,
        mean_wrist_orientation_deg=np.mean([r["wrist_orientation_error_deg"] for r in valid], axis=0).tolist() if valid else None,
        subset_warning="Only declared-constraint-valid endpoints enter means; inspect counts and matched rows.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=160)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_iterations < 1:
        raise ValueError("positive iteration cap required")
    scene, settings, human, robot, inputs = load_inputs()
    code = [artifact(Path(__file__)), artifact(ROOT / "src/egoengine_repro/retarget/mink.py"),
            artifact(ROOT / "external/mink/src/mink/tasks/task.py"), artifact(ROOT / "external/mink/src/mink/tasks/frame_task.py")]
    semantics = semantic_audit(scene, human, robot)
    args.output.mkdir(parents=True)
    exp = Experiment(scene, settings, human, robot)
    chosen = SAMPLED_ROWS if args.sample_only else range(len(robot["qpos"]))
    all_rows, endpoints, summary = {}, {}, {}
    for mode in ("pose", "split_pose", "se3_no_tip_rotation", "position_wrist", "position_only"):
        rows, states = [], []
        for row in chosen:
            record, state = exp.solve(row, mode, args.max_iterations)
            rows.append(record)
            states.append(state)
            if row % 25 == 0 or row == chosen[-1]:
                print(f"{mode}: source row {row}; mean mm {np.mean(record['position_error_m'], axis=1) * 1000}; valid {record['constraint_gate_passed']}", flush=True)
        all_rows[mode], endpoints[mode] = rows, np.array(states)
        summary[mode] = summarize(rows)
        with (args.output / f"{mode}.json").open("x") as stream:
            json.dump(dict(summary=summary[mode], rows=rows), stream, indent=2, allow_nan=False)
    probes, multistart, extra = [], [], []
    for row in SAMPLED_ROWS:
        for mode in ("pose", "position_wrist", "position_only"):
            record, q = exp.solve(row, mode, args.max_iterations * 2, use_speed=False)
            extra.append(record)
            endpoints[f"no_speed_{mode}_{row}"] = q
            probes.append(exp.relaxed_bound_probe(row, mode, q))
        for si, seed in enumerate(exp.legal_seeds(row)):
            for mode in ("pose", "position_only"):
                record, q = exp.solve(row, mode, args.max_iterations * 2, use_speed=False, seed=seed)
                record["seed_index"] = si
                multistart.append(record)
                endpoints[f"multistart_{row}_{si}_{mode}"] = q
        print(f"constraint and multistart probes: row {row}", flush=True)
    verify_artifacts(inputs)
    np.savez_compressed(args.output / "diagnostic_endpoints_not_reference.npz", **endpoints)
    verify_artifacts(code)
    result = dict(status="isolated_kinematic_diagnosis_not_reference_or_initializer", inputs=inputs, code=code,
        semantics=semantics, frame_selection=list(chosen), summary=summary, no_speed_rows=extra,
        multistart_rows=multistart, unapplied_relaxed_bound_probes=probes,
        conditions=dict(original_joint_limits=True, original_self_pairs=len(exp.pairs),
            original_sites=True, original_orientation_targets=True, fixed_original_previous_frame=True,
            no_physics_integration=True, reference_overwritten=False, max_iterations=args.max_iterations,
            diagnostic_convergence_increment_tolerance=1e-8, costs_source="inherited local configuration, not author coefficients"),
        limitations=["Independent frame probes cannot be concatenated into a speed-feasible reference.",
            "Nonlinear feasible loss-nonincreasing backtracking is diagnostic-only, not the production integration rule; stalls are not convergence.",
            "No object/table avoidance or complete native self-coverage is claimed.",
            "Diagnostic objective removal, random seeds and no-speed probes are not adopted pipeline changes.",
            "Finite multistart search does not certify a global optimum or an irreducible morphology error."])
    with (args.output / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
