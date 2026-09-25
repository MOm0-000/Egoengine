"""Audit mixed-unit residual scale against consecutive reference commands.

This is a read-only command-space audit.  It does not run physics or PPO and
does not claim that a reference command increment equals realized robot motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from egoengine_repro.retarget.paper_audit import artifact, verify_artifacts


EXPECTED_ACTUATORS = [
    *(f"R_forearm_{axis}_position" for axis in ("tx", "ty", "tz", "roll", "pitch", "yaw")),
    "right_thumb_bend_position", "right_thumb_rota1_position", "right_thumb_rota2_position",
    "right_index_bend_position", "right_index_joint1_position", "right_index_joint2_position",
    "right_middle_joint1_position", "right_middle_joint2_position",
    "right_ring_joint1_position", "right_ring_joint2_position",
    "right_pinky_joint1_position", "right_pinky_joint2_position",
    *(f"L_forearm_{axis}_position" for axis in ("tx", "ty", "tz", "roll", "pitch", "yaw")),
    "left_thumb_bend_position", "left_thumb_rota1_position", "left_thumb_rota2_position",
    "left_index_bend_position", "left_index_joint1_position", "left_index_joint2_position",
    "left_middle_joint1_position", "left_middle_joint2_position",
    "left_ring_joint1_position", "left_ring_joint2_position",
    "left_pinky_joint1_position", "left_pinky_joint2_position",
]

LOCAL_NAMES = [
    "wrist_translation_x", "wrist_translation_y", "wrist_translation_z",
    "wrist_rotation_roll", "wrist_rotation_pitch", "wrist_rotation_yaw",
    "thumb_bend", "thumb_rotation_1", "thumb_rotation_2",
    "index_bend", "index_joint_1", "index_joint_2",
    "middle_joint_1", "middle_joint_2", "ring_joint_1", "ring_joint_2",
    "pinky_joint_1", "pinky_joint_2",
]


def _absolute_stats(values: np.ndarray, residual_limit: float) -> dict:
    values = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("statistics require finite, nonempty values")
    quantiles = {
        name: float(value)
        for name, value in zip(
            ("p50", "p90", "p95", "p99"),
            np.quantile(values, (0.50, 0.90, 0.95, 0.99)),
        )
    }
    quantiles["max"] = float(values.max())
    ratios = {
        name: (None if value == 0.0 else float(residual_limit / value))
        for name, value in quantiles.items()
    }
    return {
        "sample_count": int(values.size),
        "absolute_increment": quantiles,
        "residual_limit_over_increment": ratios,
    }


def _actuator_records(model: mujoco.MjModel) -> list[dict]:
    records = []
    for index in range(model.nu):
        if int(model.actuator_trntype[index]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError(f"actuator {index} is not a scalar joint transmission")
        joint_id = int(model.actuator_trnid[index, 0])
        records.append({
            "index": index,
            "actuator": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index),
            "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
        })
    names = [row["actuator"] for row in records]
    if names != EXPECTED_ACTUATORS:
        raise ValueError("formal actuator order no longer matches the audited 18+18 mixed-unit contract")
    return records


def run(config_path: Path, action_profile_path: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    config = yaml.safe_load(config_path.read_text())
    action_profile = yaml.safe_load(action_profile_path.read_text())
    scene = Path(config["model_path"]).resolve(strict=True)
    reference = Path(config["data_path"]).resolve(strict=True)
    residual_scale = float(action_profile["mapping"]["residual_scale"])
    residual_clip = float(action_profile["mapping"]["residual_clip_rad"])
    policy_low = float(action_profile["policy_output"]["lower"])
    policy_high = float(action_profile["policy_output"]["upper"])
    if (policy_low, policy_high) != (-1.0, 1.0) or residual_scale != 0.05 or residual_clip != 0.05:
        raise ValueError("expected the frozen scaled-action profile with a 0.05 component limit")

    preserved = [artifact(path) for path in (config_path, action_profile_path, scene, reference)]
    model = mujoco.MjModel.from_xml_path(str(scene))
    actuators = _actuator_records(model)
    with np.load(reference, allow_pickle=False) as arrays:
        ctrl = np.asarray(arrays["ctrl"], dtype=np.float64)
        frequency_hz = float(np.asarray(arrays["frequency"]).reshape(()))
    if ctrl.shape != (198, 36) or model.nu != 36 or frequency_hz != 30.0:
        raise ValueError("expected the frozen 198-frame, 36-control, 30 Hz Pour reference")
    if not np.isfinite(ctrl).all():
        raise ValueError("reference controls are not finite")

    increments = np.diff(ctrl, axis=0)
    if np.max(np.abs(increments[:, [3, 4, 5, 21, 22, 23]])) >= np.pi:
        raise ValueError("wrist angle increments cross a wrap boundary; direct differences are invalid")

    hands = {}
    for side, offset in (("right", 0), ("left", 18)):
        coordinates = []
        for local_index, local_name in enumerate(LOCAL_NAMES):
            index = offset + local_index
            unit = "m" if local_index < 3 else "rad"
            coordinates.append({
                **actuators[index],
                "coordinate": local_name,
                "unit": unit,
                **_absolute_stats(increments[:, index], residual_clip),
            })
        groups = {}
        for name, start, stop, unit in (
            ("wrist_translation_components", 0, 3, "m"),
            ("wrist_rotation_components", 3, 6, "rad"),
            ("finger_joint_components", 6, 18, "rad"),
        ):
            groups[name] = {
                "unit": unit,
                "aggregation": "all absolute component increments flattened across coordinates and transitions",
                **_absolute_stats(increments[:, offset + start:offset + stop], residual_clip),
            }
        hands[side] = {"coordinates": coordinates, "groups": groups}

    report = {
        "schema": "taco_pour_reference_action_scale_audit_v1",
        "status": "mixed_unit_imbalance_quantified_no_scale_candidate_selected",
        "scope": {
            "training_executed": False,
            "physics_rollout_executed": False,
            "simulator_environments_created": 0,
            "reference_or_configuration_modified": False,
            "quantity_measured": "absolute consecutive reference control-target increments",
            "realized_robot_motion_measured": False,
            "author_recovered_residual_scale": False,
        },
        "reference": {
            "frames": int(ctrl.shape[0]),
            "transitions": int(increments.shape[0]),
            "frequency_hz": frequency_hz,
            "control_dimensions": int(ctrl.shape[1]),
        },
        "current_local_action_mapping": {
            "formula": "delta_i = clip(0.05 * clip(u_i, -1, 1), -0.05, 0.05)",
            "component_limit_numeric": residual_clip,
            "translation_component_limit_m": residual_clip,
            "rotation_and_finger_component_limit_rad": residual_clip,
            "same_numeric_scale_across_incompatible_units": True,
            "profile_field_name_residual_clip_rad_is_inaccurate_for_translation_dimensions": True,
        },
        "actuator_order_verified": True,
        "hands": hands,
        "interpretation": {
            "comparison_rule": "limit divided by each empirical absolute-increment quantile; values above 1 mean the residual limit exceeds that reference increment statistic",
            "primary_descriptive_quantile": "p95 is reported as a typical-high comparator, not a success threshold",
            "right_p95_ratios": {
                name: hands["right"]["groups"][name]["residual_limit_over_increment"]["p95"]
                for name in hands["right"]["groups"]
            },
            "left_p95_ratios": {
                name: hands["left"]["groups"][name]["residual_limit_over_increment"]["p95"]
                for name in hands["left"]["groups"]
            },
            "mixed_unit_scale_is_balanced_by_reference_dynamics": False,
            "basis": "the 0.05 m translation limit is multiple p95 reference translation increments, while 0.05 rad is below p95 wrist-rotation and aggregate finger increments",
            "split_scale_candidate_selected": False,
            "next_training_authorized": False,
        },
        "limitations": [
            "Reference command changes are not measured joint motion, object motion, controllability, or an optimal residual magnitude.",
            "Large reference changes can reflect retargeting variation and do not by themselves prescribe PPO action limits.",
            "This audit establishes a mixed-unit imbalance but does not choose translation, rotation, or finger scales.",
        ],
        "preserved_inputs": preserved,
        "audit_code": [artifact(Path(__file__))],
    }
    verify_artifacts(preserved)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "runs/taco_pour_floor_contact_v1/candidate_ppo_config.yaml",
    )
    parser.add_argument(
        "--action-profile",
        type=Path,
        default=ROOT / "configs/taco_pour_residual_action_scaled_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/taco_pour_reference_action_scale_audit_v1/report.json",
    )
    args = parser.parse_args()
    report = run(args.config.resolve(), args.action_profile.resolve(), args.output.resolve())
    for side in ("right", "left"):
        ratios = report["interpretation"][f"{side}_p95_ratios"]
        print(side, "p95 limit ratios", json.dumps(ratios, sort_keys=True))


if __name__ == "__main__":
    main()
