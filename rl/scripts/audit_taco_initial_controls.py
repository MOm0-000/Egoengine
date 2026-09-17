"""Compare instantaneous reset inputs with mj_forward only; no chosen reset."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from egoengine_repro.retarget.collision_audit import validate_qpos
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts


def pose_matched_ctrl(model, qpos):
    """Use actuator transmissions, not an assumed qpos slice; position servos only."""
    qpos = validate_qpos(model, qpos)
    ctrl = np.empty(model.nu)
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        if (int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT)
                or int(model.jnt_type[joint]) not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE))
                or int(model.actuator_dyntype[a]) != int(mujoco.mjtDyn.mjDYN_NONE)
                or int(model.actuator_gaintype[a]) != int(mujoco.mjtGain.mjGAIN_FIXED)
                or int(model.actuator_biastype[a]) != int(mujoco.mjtBias.mjBIAS_AFFINE)
                or model.actuator_gainprm[a, 0] <= 0
                or model.actuator_biasprm[a, 0] != 0
                or not np.isclose(model.actuator_biasprm[a, 1], -model.actuator_gainprm[a, 0])):
            raise ValueError("requires stateless scalar-joint position servos")
        ctrl[a] = model.actuator_gear[a, 0] * qpos[int(model.jnt_qposadr[joint])]
        if model.actuator_ctrllimited[a] and not model.actuator_ctrlrange[a, 0] - 1e-12 <= ctrl[a] <= model.actuator_ctrlrange[a, 1] + 1e-12:
            raise ValueError("pose-matched command exceeds inherited control limits")
    return ctrl


def contact_normal_velocity(model, data, contact):
    velocities = []
    for geom in (contact.geom1, contact.geom2):
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, contact.pos, int(model.geom_bodyid[geom]))
        velocities.append(jac @ data.qvel)
    # MuJoCo's contact normal points from geom1 to geom2; negative means closing.
    return float(np.dot(contact.frame[:3], velocities[1] - velocities[0]))


def forward_case(model, qpos, qvel, ctrl):
    qpos = validate_qpos(model, qpos)
    qvel, ctrl = np.asarray(qvel, dtype=float), np.asarray(ctrl, dtype=float)
    if qvel.shape != (model.nv,) or ctrl.shape != (model.nu,) or not np.isfinite(qvel).all() or not np.isfinite(ctrl).all():
        raise ValueError("finite correctly shaped qvel and ctrl required")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    if not np.array_equal(data.qpos, qpos) or not np.array_equal(data.qvel, qvel) or data.time != 0:
        raise RuntimeError("instantaneous audit changed the physical state or time")
    if not np.isfinite(data.qacc).all() or not np.isfinite(data.actuator_force).all():
        raise ValueError("nonfinite instantaneous dynamics")
    contacts = [dict(geom1=model.geom(c.geom1).name, geom2=model.geom(c.geom2).name,
                     distance_m=float(c.dist), active=bool(c.efc_address >= 0),
                     normal_velocity_m_s=contact_normal_velocity(model, data, c)) for c in data.contact]
    actuators = []
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        gear = float(model.actuator_gear[a, 0])
        dof = int(model.jnt_dofadr[joint])
        actuators.append(dict(actuator=model.actuator(a).name, joint=model.joint(joint).name,
            command=float(ctrl[a]), transmission_position=float(data.actuator_length[a]),
            command_error=float(ctrl[a] - data.actuator_length[a]),
            actuator_force=float(data.actuator_force[a]),
            generalized_force=float(gear * data.actuator_force[a]),
            generalized_force_unit=("N" if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_SLIDE) else "Nm"),
            qvel=float(qvel[dof]), qacc=float(data.qacc[dof])))
    objects = {}
    for side in ("right", "left"):
        dof = int(model.joint(f"{side}_object_joint").dofadr[0])
        objects[side] = dict(qvel_free_joint=qvel[dof:dof + 6].tolist(),
                             qacc_free_joint=data.qacc[dof:dof + 6].tolist())
    return dict(actuators=actuators, contacts=contacts, objects=objects,
        active_closing_contacts=sum(c["active"] and c["normal_velocity_m_s"] < -1e-9 for c in contacts),
        closing_velocity_numerical_tolerance_m_s=1e-9,
        qvel_input=qvel.tolist(), ctrl_input=ctrl.tolist(), time_s=float(data.time),
        physical_state_unchanged=True, simulation_steps_executed=0,
        mj_forward_evaluations=1, stability_or_task_success_claim=False)


def run(scene, baseline, candidate, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = [scene, baseline, candidate]
    preserved = [artifact(p) for p in inputs]
    mesh_snapshot = scene_mesh_artifacts(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(baseline, allow_pickle=False) as source:
        reference = dict(source)
    with np.load(candidate, allow_pickle=False) as source:
        if (source["qpos"].shape != (1, model.nq) or not bool(source["diagnostic_only"])
                or bool(source["accepted_as_reset"]) or "qvel" in source or "ctrl" in source):
            raise ValueError("expected a single unaccepted posture with no velocity or control")
        qpos = source["qpos"][0]
    if not np.array_equal(qpos[36:], reference["qpos"][0, 36:]):
        raise ValueError("object pose differs from original first frame")
    dt = float(reference["timestamps_s"][1] - reference["timestamps_s"][0])
    derivative = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, derivative, dt, reference["qpos"][0], reference["qpos"][1])
    if not np.allclose(derivative, reference["qvel"][0], rtol=1e-10, atol=1e-12):
        raise ValueError("reference first velocity is not the documented forward difference")
    velocities = dict(zero=np.zeros(model.nv), original_reference=reference["qvel"][0])
    controls = dict(original_reference=reference["ctrl"][0], candidate_pose=pose_matched_ctrl(model, qpos))
    cases = [dict(velocity=vn, command=cn, **forward_case(model, qpos, v, c))
             for vn, v in velocities.items() for cn, c in controls.items()]
    report = dict(status="instantaneous_velocity_control_comparison_not_reset_selection",
        input_qpos=artifact(candidate), source_row_zero_based=0, source_dt_s=dt,
        reference_velocity_provenance="unchanged baseline row0->row1 manifold forward difference, not measured robot velocity",
        reference_velocity_recomputed_from_candidate=False,
        comparison_provenance="local four-way diagnostic; paper does not publish TACO initial qvel or reset command",
        command_limit_roundoff_tolerance=1e-12,
        control_clipping="MuJoCo applies inherited limits internally; raw pose commands retained, including sub-1e-12 roundoff",
        cases=cases, accepted_as_reset=False, qvel_selected=False, ctrl_selected=False,
        physical_rollout_executed=False, simulation_steps_executed=0, mj_forward_evaluations=len(cases),
        preserved_artifacts=preserved, scene_meshes=mesh_snapshot, audit_code=[artifact(Path(__file__))],
        limitations=["same unvalidated candidate and incomplete collision model in all four cases",
                     "mj_forward is instantaneous dynamics, not settling or a stability test",
                     "pose-matched command is not gravity compensation or a modified Replay trajectory",
                     "no coefficients, model, reference, trainer or renderer changed"])
    if preserved != [artifact(p) for p in inputs]:
        raise RuntimeError("source inputs changed during audit")
    verify_artifacts(mesh_snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for case in cases:
        print(case["velocity"], case["command"], "active closing contacts:", case["active_closing_contacts"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.scene, args.baseline, args.candidate, args.output)


if __name__ == "__main__":
    main()
