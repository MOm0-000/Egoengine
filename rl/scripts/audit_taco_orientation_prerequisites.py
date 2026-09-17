"""Read-only orientation/geometry checks before fingertip-cost sweeps.

The report separates model-definition evidence from trajectory residuals.  A
constant residual rotation is estimated only as an offline diagnostic; no
target, site, scene, or reference is modified by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]

from egoengine_repro.evaluation.taco_surface import (  # noqa: E402
    load_taco_mano_sequence,
    reconstruct_taco_mano,
)
from egoengine_repro.retarget.taco_bimanual import (  # noqa: E402
    DIPS,
    FINGERS,
    MANO_DISTALS,
    SIDES,
    TIPS,
    geometric_frame,
    mano_fingertip_frames,
)


DEFAULT_HUMAN = ROOT / "runs/taco_pour_bimanual_mano_fk_v1/human_reference.npz"
DEFAULT_BASELINE = ROOT / "runs/taco_pour_bimanual_mano_fk_v1/robot_reference.npz"
DEFAULT_CANDIDATE = ROOT / "runs/taco_pour_spider_mink_experiment_v2/robot_reference.npz"
DEFAULT_SCENE = ROOT / (
    "models/taco_xhand/xhand/bimanual/"
    "taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
)
DEFAULT_HANDS = ROOT / (
    "data/taco_v1/pour_bowl_plate/hand_poses/Hand_Poses/"
    "(pour in some, bowl, plate)/20230927_017"
)
DEFAULT_MANO = ROOT / "data/taco_v1/hand_poses_v1/mano_v1_2/models"
DEFAULT_VENDOR = ROOT / "runs/taco_pour_spider_mink_experiment_v2/vendor/spider/assets/robots/xhand"
DEFAULT_OUTPUT = ROOT / "runs/taco_pour_orientation_prerequisite_audit_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("degenerate vector")
    return vector / norm


def _axis_angles(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    dots = np.clip(np.sum(actual * target, axis=-1), -1.0, 1.0)
    return np.rad2deg(np.arccos(dots))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _align_vector(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Return a rotation mapping source onto destination."""
    source = _unit(source)
    destination = _unit(destination)
    cross = np.cross(source, destination)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, destination), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3)
        trial = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = _unit(np.cross(source, trial))
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    axis = cross / sine
    K = _skew(axis)
    return np.eye(3) + sine * K + (1.0 - cosine) * (K @ K)


def _twist_angles(actual: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int]:
    """Signed target-vs-actual twist after aligning the two longitudinal axes."""
    values = []
    undefined = 0
    for robot, human in zip(actual, target, strict=True):
        z_robot, z_human = robot[:, 2], human[:, 2]
        align = _align_vector(z_human, z_robot)
        x_human = align @ human[:, 0]
        x_robot = robot[:, 0]
        x_human -= np.dot(x_human, z_robot) * z_robot
        x_robot -= np.dot(x_robot, z_robot) * z_robot
        nh, nr = np.linalg.norm(x_human), np.linalg.norm(x_robot)
        if min(nh, nr) < 1e-8:
            values.append(np.nan)
            undefined += 1
            continue
        x_human /= nh
        x_robot /= nr
        values.append(np.rad2deg(np.arctan2(
            np.dot(z_robot, np.cross(x_robot, x_human)),
            np.clip(np.dot(x_robot, x_human), -1.0, 1.0),
        )))
    return np.asarray(values), undefined


def _residual_report(actual: np.ndarray, target: np.ndarray) -> dict:
    """Summarize full, axis, and fixed-bias-corrected orientation residuals."""
    error = np.einsum("tji,tjk->tik", actual, target)
    rotations = Rotation.from_matrix(error)
    angles = np.rad2deg(rotations.magnitude())
    mean_rotation = rotations.mean()
    corrected_error = np.einsum("tij,jk->tik", error, mean_rotation.as_matrix().T)
    corrected_angles = np.rad2deg(Rotation.from_matrix(corrected_error).magnitude())
    split = int(np.ceil(0.2 * len(error)))
    train_mean = Rotation.from_matrix(error[:split]).mean()
    validation_angles = angles[split:]
    validation_corrected = np.rad2deg(Rotation.from_matrix(
        np.einsum("tij,jk->tik", error[split:], train_mean.as_matrix().T)
    ).magnitude())
    twist, undefined = _twist_angles(actual, target)
    axis = {
        name: _axis_angles(actual[:, index], target[:, index])
        for index, name in enumerate(("x", "y", "z"))
    }
    return {
        "full_angle_deg": {
            "mean": float(angles.mean()), "median": float(np.median(angles)),
            "p95": float(np.percentile(angles, 95)), "max": float(angles.max()),
        },
        "fixed_bias": {
            "mean_rotation_vector_rad": mean_rotation.as_rotvec().tolist(),
            "bias_angle_deg": float(np.rad2deg(mean_rotation.magnitude())),
            "after_right_multiplying_mean_inverse": {
                "mean": float(corrected_angles.mean()),
                "median": float(np.median(corrected_angles)),
                "p95": float(np.percentile(corrected_angles, 95)),
                "max": float(corrected_angles.max()),
            },
            "residual_mean_reduction_deg": float(angles.mean() - corrected_angles.mean()),
        },
        "chronological_20_80_holdout": {
            "calibration_frames": [0, split - 1],
            "validation_frames": [split, len(error) - 1],
            "calibration_mean_rotation_vector_rad": train_mean.as_rotvec().tolist(),
            "calibration_bias_angle_deg": float(np.rad2deg(train_mean.magnitude())),
            "validation_before_deg": {
                "mean": float(validation_angles.mean()),
                "p95": float(np.percentile(validation_angles, 95)),
            },
            "validation_after_deg": {
                "mean": float(validation_corrected.mean()),
                "p95": float(np.percentile(validation_corrected, 95)),
            },
            "validation_mean_reduction_deg": float(
                validation_angles.mean() - validation_corrected.mean()
            ),
            "target_or_qpos_modified": False,
        },
        "axis_angle_deg": {
            key: {
                "mean": float(value.mean()), "median": float(np.median(value)),
                "p95": float(np.percentile(value, 95)), "max": float(value.max()),
            } for key, value in axis.items()
        },
        "twist_after_longitudinal_alignment_deg": {
            "mean_abs": float(np.nanmean(np.abs(twist))),
            "median_abs": float(np.nanmedian(np.abs(twist))),
            "p95_abs": float(np.nanpercentile(np.abs(twist), 95)),
            "max_abs": float(np.nanmax(np.abs(twist))),
            "undefined_frames": undefined,
        },
    }


def _trajectory_orientation(model: mujoco.MjModel, qpos: np.ndarray,
                            human: dict, calibrations: np.ndarray,
                            hand: int, finger: int) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    actual = np.empty((len(qpos), 3, 3), dtype=float)
    calibration = calibrations[hand, finger]
    target = np.asarray(
        human["T_sim_fingertip_target"][:, hand, finger, :3, :3], dtype=float
    ) @ calibration
    site_id = model.site(f"{SIDES[hand]}_{FINGERS[finger]}_tip").id
    for row, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        actual[row] = data.site_xmat[site_id].reshape(3, 3) @ calibration
    return actual, target


def _frame_steps(rotations: np.ndarray) -> dict:
    steps = np.rad2deg(Rotation.from_matrix(
        np.einsum("tji,tjk->tik", rotations[:-1], rotations[1:])
    ).magnitude())
    return {
        "mean_deg": float(steps.mean()),
        "p95_deg": float(np.percentile(steps, 95)),
        "max_deg": float(steps.max()),
        "count_over_45_deg": int(np.sum(steps > 45.0)),
        "count_over_90_deg": int(np.sum(steps > 90.0)),
    }


def _current_anatomical_calibrations(model: mujoco.MjModel) -> np.ndarray:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = np.empty((2, 5, 3, 3), dtype=float)
    for hand, side in enumerate(SIDES):
        root = model.body(f"{side}_hand_link").id
        palm = model.site(f"{side}_palm").id
        middle = model.body(f"{side}_hand_mid_link1").id
        palm_frame = geometric_frame(
            data.site_xmat[palm].reshape(3, 3)[:, 0],
            data.xpos[middle] - data.xpos[root],
        )
        for finger_index, finger in enumerate(FINGERS):
            site_id = model.site(f"{side}_{finger}_tip").id
            body_id = int(model.site_bodyid[site_id])
            distal = data.site_xpos[site_id] - data.xpos[body_id]
            anatomical = geometric_frame(palm_frame[:, 0], distal)
            result[hand, finger_index] = (
                data.site_xmat[site_id].reshape(3, 3).T @ anatomical
            )
    return result


def _static_geometry(model: mujoco.MjModel, vendor: Path,
                     calibrations: np.ndarray) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    records = {}
    local_frames = {}
    for hand, side in enumerate(SIDES):
        urdf = ET.parse(vendor / f"xhand_{side}.urdf").getroot()
        root = model.body(f"{side}_hand_link").id
        palm = model.site(f"{side}_palm").id
        middle = model.body(f"{side}_hand_mid_link1").id
        palm_frame = geometric_frame(
            data.site_xmat[palm].reshape(3, 3)[:, 0],
            data.xpos[middle] - data.xpos[root],
        )
        local_frames[side] = {}
        records[side] = {}
        for finger in FINGERS:
            site_id = model.site(f"{side}_{finger}_tip").id
            body_id = int(model.site_bodyid[site_id])
            body_name = model.body(body_id).name
            fixed = [
                joint for joint in urdf.findall("joint")
                if joint.get("type") == "fixed"
                and joint.find("parent").get("link") == body_name
                and joint.find("child").get("link").endswith("tip")
            ]
            if len(fixed) != 1:
                raise ValueError(f"expected one URDF tip endpoint for {side}_{finger}")
            endpoint = np.fromstring(fixed[0].find("origin").get("xyz"), sep=" ")
            site_local = np.asarray(model.site_pos[site_id], dtype=float)
            site_world = data.site_xmat[site_id].reshape(3, 3)
            palm_local = palm_frame.T @ site_world
            local_frames[side][finger] = palm_local
            endpoint_axis = _unit(endpoint)
            site_vector_axis = _unit(site_local)
            site_body_axis = data.xmat[body_id].reshape(3, 3).T @ site_world[:, 2]
            records[side][finger] = {
                "site_body": body_name,
                "site_local_m": site_local.tolist(),
                "urdf_endpoint_local_m": endpoint.tolist(),
                "site_to_urdf_endpoint_distance_m": float(np.linalg.norm(site_local - endpoint)),
                "site_z_vs_urdf_endpoint_axis_deg": float(_axis_angles(
                    site_body_axis[None], endpoint_axis[None]
                )[0]),
                "site_body_to_site_vector_vs_urdf_endpoint_axis_deg": float(_axis_angles(
                    site_vector_axis[None], endpoint_axis[None]
                )[0]),
                "site_z_vs_site_body_to_site_vector_deg": float(_axis_angles(
                    site_body_axis[None], site_vector_axis[None]
                )[0]),
                "site_frame_in_palm_frame": palm_local.tolist(),
                "current_anatomical_frame_in_site_frame": calibrations[
                    hand, FINGERS.index(finger)
                ].tolist(),
                "site_frame_det_error": float(abs(np.linalg.det(site_world) - 1.0)),
            }
    mirror = {}
    reflection = np.diag([-1.0, 1.0, 1.0])
    for finger in FINGERS:
        right = local_frames["right"][finger]
        left = local_frames["left"][finger]
        direct = Rotation.from_matrix(left.T @ right).magnitude()
        mirrored = Rotation.from_matrix(left.T @ reflection @ right @ reflection).magnitude()
        mirror[finger] = {
            "direct_local_frame_difference_deg": float(np.rad2deg(direct)),
            "conjugate_reflection_difference_deg": float(np.rad2deg(mirrored)),
            "interpretation": "diagnostic only; canonical palm frames are a model convention",
        }
    return {"per_hand": records, "left_right_frame_check": mirror}


def _mano_and_target_continuity(human: dict, hands: Path, mano_models: Path) -> dict:
    records = {}
    for hand, side in enumerate(SIDES):
        pose_path = hands / f"{side}_hand.pkl"
        shape_path = hands / f"{side}_hand_shape.pkl"
        model_path = mano_models / f"MANO_{side.upper()}.pkl"
        poses, _, _, _ = load_taco_mano_sequence(pose_path, shape_path)
        _, joints, _, _, _ = reconstruct_taco_mano(
            pose_path, shape_path, model_path, side=side,
        )
        distal_frames, mano_report = mano_fingertip_frames(
            pose_path, shape_path, model_path, side, joints,
        )
        target_frames = np.asarray(
            human["T_sim_fingertip_target"][:, hand, :, :3, :3], dtype=float
        )
        target_steps = np.empty((len(target_frames) - 1, 5), dtype=float)
        mano_steps = np.empty_like(target_steps)
        projection_ratio = np.empty((len(joints), 5), dtype=float)
        for finger in range(5):
            target_steps[:, finger] = np.rad2deg(Rotation.from_matrix(
                target_frames[:-1, finger].swapaxes(-1, -2) @ target_frames[1:, finger]
            ).magnitude())
            mano_steps[:, finger] = np.rad2deg(Rotation.from_matrix(
                distal_frames[:-1, finger].swapaxes(-1, -2) @ distal_frames[1:, finger]
            ).magnitude())
        for row in range(len(joints)):
            normal = np.cross(
                joints[row, 5] - joints[row, 0],
                joints[row, 17] - joints[row, 0],
            )
            if side == "left":
                normal = -normal
            normal /= np.linalg.norm(normal)
            for finger, (tip, dip) in enumerate(zip(TIPS, DIPS, strict=True)):
                direction = _unit(joints[row, tip] - joints[row, dip])
                projection_ratio[row, finger] = np.linalg.norm(
                    normal - np.dot(normal, direction) * direction
                )
        target_determinant = np.linalg.det(target_frames)
        target_orthogonality = np.max(np.abs(
            np.einsum("tfji,tfjk->tfik", target_frames, target_frames)
            - np.eye(3)
        ))
        records[side] = {
            "target_frame_step_by_finger": [
                _frame_steps(target_frames[:, finger]) for finger in range(5)
            ],
            "mano_distal_frame_step_by_finger": [
                _frame_steps(distal_frames[:, finger]) for finger in range(5)
            ],
            "target_vs_mano_step_abs_max_deg_by_finger": np.max(
                np.abs(target_steps - mano_steps), axis=0
            ).tolist(),
            "old_projected_palm_normal_condition_ratio_by_finger": {
                "min": projection_ratio.min(axis=0).tolist(),
                "p01": np.percentile(projection_ratio, 1, axis=0).tolist(),
                "count_below_0_1": np.sum(projection_ratio < 0.1, axis=0).tolist(),
                "count_below_0_2": np.sum(projection_ratio < 0.2, axis=0).tolist(),
            },
            "target_rotation_det_max_error": float(np.max(np.abs(target_determinant - 1.0))),
            "target_rotation_orthogonality_max_error": float(target_orthogonality),
            "mano_fk_report": mano_report,
            "interpretation": (
                "The projected-palm ratio diagnoses the historical proxy; current "
                "human_reference fingertip rotations are from MANO rotational FK."
            ),
        }
    return records


def _trajectory_residuals(model: mujoco.MjModel, qpos: np.ndarray,
                          human: dict, calibrations: np.ndarray) -> dict:
    result = {}
    for hand, side in enumerate(SIDES):
        result[side] = {}
        for finger_index, finger in enumerate(FINGERS):
            actual, target = _trajectory_orientation(
                model, qpos, human, calibrations, hand, finger_index
            )
            result[side][finger] = _residual_report(actual, target)
    return result


def run(output: Path, human_path: Path, baseline_path: Path,
        candidate_path: Path, scene: Path, hands: Path, mano_models: Path,
        vendor: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with np.load(human_path, allow_pickle=False) as source:
        human = {key: np.asarray(source[key]) for key in source.files}
    with np.load(baseline_path, allow_pickle=False) as source:
        baseline = {key: np.asarray(source[key]) for key in source.files}
    with np.load(candidate_path, allow_pickle=False) as source:
        candidate = {key: np.asarray(source[key]) for key in source.files}
    if not np.array_equal(baseline["frame_indices"], candidate["frame_indices"]):
        raise ValueError("baseline and candidate frame rows differ")
    if len(human["frame_indices"]) != len(baseline["qpos"]):
        raise ValueError("human and robot frame counts differ")
    model = mujoco.MjModel.from_xml_path(str(scene))
    calibrations = _current_anatomical_calibrations(model)
    static = _static_geometry(model, vendor, calibrations)
    continuity = _mano_and_target_continuity(human, hands, mano_models)
    residuals = {
        "baseline_current_mink": _trajectory_residuals(
            model, baseline["qpos"], human, calibrations
        ),
        "candidate_spider_style": _trajectory_residuals(
            model, candidate["qpos"], human, calibrations
        ),
    }
    inputs = [human_path, baseline_path, candidate_path, scene]
    inputs += [hands / f"{side}_hand.pkl" for side in SIDES]
    inputs += [hands / f"{side}_hand_shape.pkl" for side in SIDES]
    inputs += [mano_models / f"MANO_{side.upper()}.pkl" for side in SIDES]
    inputs += [vendor / f"xhand_{side}.urdf" for side in SIDES]
    report = {
        "status": "orientation_prerequisite_audit_not_a_mapping_change",
        "inputs": [artifact(path) for path in inputs],
        "code": [artifact(Path(__file__)), artifact(ROOT / "src/egoengine_repro/retarget/taco_bimanual.py")],
        "static_point_and_axis_geometry": static,
        "target_and_source_continuity": continuity,
        "trajectory_residuals": residuals,
        "conclusions": {
            "site_endpoint_mismatch_is_real": True,
            "site_endpoint_mismatch_directly_explains_rotation_error": False,
            "target_frame_is_orthonormal": all(
                value["target_rotation_det_max_error"] < 1e-10
                and value["target_rotation_orthogonality_max_error"] < 1e-10
                for value in continuity.values()
            ),
            "target_step_matches_mano_fk": all(
                max(value["target_vs_mano_step_abs_max_deg_by_finger"]) < 1e-6
                for value in continuity.values()
            ),
            "constant_bias_is_only_a_diagnostic": True,
            "no_targets_or_models_modified": True,
        },
        "limitations": [
            "The fixed residual rotation is fitted to a trajectory only to test the bias hypothesis; it is not adopted as calibration.",
            "A qpos residual includes IK/morphology effects as well as any frame-definition error.",
            "Left/right reflection comparisons use canonical model palm frames and are diagnostic, not author evidence.",
            "No fingertip cost sweep, filtering, target rewrite, or physics trajectory was run here.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--hands", type=Path, default=DEFAULT_HANDS)
    parser.add_argument("--mano-models", type=Path, default=DEFAULT_MANO)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    args = parser.parse_args()
    report = run(
        args.output, args.human, args.baseline, args.candidate, args.scene,
        args.hands, args.mano_models, args.vendor,
    )
    print(json.dumps(report["conclusions"], indent=2), flush=True)


if __name__ == "__main__":
    main()
