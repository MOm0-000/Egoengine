"""Analytic orientation incompatibility and same-state tracking-point checks."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from diagnose_taco_retarget_objectives import FINGERS, SIDES, RUN, load_inputs, semantic_audit
from egoengine_repro.retarget.paper_audit import artifact, verify_artifacts


def angular_separation_degrees(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(axis=-1)))


def pairwise_orientation_rms_lower_bound(target_rotations):
    """Three actual X axes coincide: RMS geodesic error >= max pair angle/sqrt(6)."""
    axes = np.asarray(target_rotations)[..., :, :, 0]
    pair_angles = np.stack([angular_separation_degrees(axes[..., a, :], axes[..., b, :])
                            for a, b in combinations(range(3), 2)], axis=-1)
    return pair_angles, pair_angles.max(axis=-1) / np.sqrt(6.)


def inspect():
    scene, _, human, robot, inputs = load_inputs()
    semantics = semantic_audit(scene, human, robot)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    records = {}
    for hi, side in enumerate(SIDES):
        root = model.body(f"{side}_hand_link").id
        tips = [model.site(f"{side}_{f}_tip").id for f in FINGERS]
        bodies = model.site_bodyid[tips]
        offsets = np.array([semantics["site_provenance"][side]["sites"][f]["urdf_tip_m"] for f in FINGERS])
        site_error, endpoint_error, axis_drift, orientation_rms = [], [], [], []
        for row, q in enumerate(robot["qpos"]):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            body_R = data.xmat[bodies].reshape(5, 3, 3)
            end = np.einsum("kij,kj->ki", body_R, offsets) + data.xpos[bodies]
            targets = human["T_sim_fingertip_target"][row, hi]
            site_error.append(np.linalg.norm(data.site_xpos[tips] - targets[:, :3, 3], axis=-1))
            endpoint_error.append(np.linalg.norm(end - targets[:, :3, 3], axis=-1))
            actual_R = data.site_xmat[tips[2:]].reshape(3, 3, 3)
            axis_drift.append(np.abs(actual_R[:, :, 0] - data.xmat[root].reshape(3, 3)[:, 0]).max())
            orientation_rms.append(float(np.sqrt(np.mean(np.rad2deg(Rotation.from_matrix(
                actual_R.swapaxes(-1, -2) @ targets[2:, :3, :3]).magnitude()) ** 2))))
        if max(axis_drift) > 1e-12:
            raise ValueError("three robot X axes do not coincide")
        pair_angles, lower = pairwise_orientation_rms_lower_bound(human["T_sim_fingertip_target"][:, hi, 2:, :3, :3])
        if np.any(np.asarray(orientation_rms) < lower - 1e-8):
            raise ValueError("orientation lower bound exceeds measured feasible kinematic orientation error")
        np.testing.assert_allclose(site_error, robot["fingertip_position_error_m"][:, hi], atol=1e-12, rtol=0)
        records[side] = dict(
            axis_fingers=list(FINGERS[2:]), actual_shared_axis_drift=float(max(axis_drift)),
            desired_axis_pair_names=[list(p) for p in combinations(FINGERS[2:], 2)],
            desired_axis_pair_angles_deg=pair_angles.tolist(),
            desired_axis_pair_mean_deg=pair_angles.mean(axis=0).tolist(),
            desired_axis_pair_max_deg=pair_angles.max(axis=0).tolist(),
            rms_orientation_lower_bound_deg=lower.tolist(),
            mean_rms_orientation_lower_bound_deg=float(lower.mean()),
            max_rms_orientation_lower_bound_deg=float(lower.max()),
            same_qpos_site_position_mean_m=np.mean(site_error, axis=0).tolist(),
            same_qpos_urdf_endpoint_position_mean_m=np.mean(endpoint_error, axis=0).tolist())
    verify_artifacts(inputs)
    return dict(status="analytic_diagnostic_not_mapping_change", inputs=inputs,
        code=[artifact(Path(__file__)), artifact(ROOT / "scripts/diagnose_taco_retarget_objectives.py")],
        per_hand=records,
        lower_bound_derivation="For any common actual X axis u, geodesic orientation errors e_i >= angle(u,x_i). Triangle inequality gives e_i+e_j >= angle(x_i,x_j), hence sum of three e_i^2 >= max_pair_angle^2/2 and RMS >= max_pair_angle/sqrt(6).",
        scope="Current fixed orientation calibration and current XHand kinematics; shared-axis identity holds for every finger angle, not just joint bounds.",
        limitations=["This is not a paper-published diagnostic or an author acceptance threshold.",
            "It does not certify that current absolute human/robot frame calibration is appropriate.",
            "Same-qpos endpoint errors do not evaluate re-optimizing for those endpoints, and do not justify changing sites.",
            "No model, site, target, weight or reference state was modified."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({h: {k: v for k, v in r.items() if k.endswith("mean_deg") or k.endswith("max_deg") or k.startswith("same_qpos")}
                      for h, r in report["per_hand"].items()}, indent=2))


if __name__ == "__main__":
    main()
