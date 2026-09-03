#!/usr/bin/env python3
"""Compare loaded SAPIEN drives with native and converted MuJoCo drives.

This audit contains no table, object, grasp candidate, or renderer.  It holds
each of seven robot configurations at a fixed target, applies deterministic
single-joint loads, and compares the following 0.1 s response with the released
SAPIEN probe.  Position is the primary metric because PhysX reports the last
TGS substep velocity rather than the full-timestep average velocity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = Path(__file__).resolve().parent
for index, path in enumerate((ROOT, DIAGNOSTICS)):
    if str(path) not in sys.path:
        sys.path.insert(index, str(path))

import mujoco
import numpy as np

from sapien_equiv import PhysxTgsForceDrive, advance_physx_tgs_microstep


SOURCE_DT_S = 1.0 / 240.0
TGS_SUBSTEPS = 25
PROBE_SCHEMA = (
    "deximit_sapien_joint_drive_probe_v9_clean_loaded_limits_diagnostic_only"
)
SCENE_SCHEMA = (
    "deximit_exact_urdf_mujoco_scene_v9_bound_friction_drive_limits_diagnostic_only"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).eps)
    return float(np.linalg.norm(actual - expected) / denominator)


def simulate(
    scene: Path, names: list[str], configurations: np.ndarray,
    load_amplitude: np.ndarray, load_sign: np.ndarray, steps: int, *,
    converted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    if converted:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    data = mujoco.MjData(model)
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    actuator_ids: list[int] = []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{name}",
        )
        if min(joint, actuator) < 0:
            raise ValueError(f"scene lacks the bound scalar drive {name!r}")
        qpos_addresses.append(int(model.jnt_qposadr[joint]))
        dof_addresses.append(int(model.jnt_dofadr[joint]))
        actuator_ids.append(actuator)
    drive = None
    if converted:
        drive = PhysxTgsForceDrive(
            model, mujoco, names, stiffness=1000.0, damping=100.0,
        )

    shape = (
        len(configurations), len(load_amplitude), len(names), steps, len(names),
    )
    qpos = np.empty(shape, dtype=np.float64)
    qvel = np.empty_like(qpos)
    for config_index, configuration in enumerate(configurations):
        for scale_index, amplitude in enumerate(load_amplitude):
            for load_axis in range(len(names)):
                mujoco.mj_resetData(model, data)
                data.qpos[qpos_addresses] = configuration
                data.qvel[:] = 0.0
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[dof_addresses[load_axis]] = (
                    load_sign[config_index, load_axis] * amplitude[load_axis]
                )
                if converted:
                    assert drive is not None
                    drive.set_target(configuration)
                else:
                    data.ctrl[actuator_ids] = configuration
                mujoco.mj_forward(model, data)
                for step in range(steps):
                    if converted:
                        assert drive is not None
                        drive.begin_frame(data)
                    for _ in range(TGS_SUBSTEPS):
                        if converted:
                            assert drive is not None
                            advance_physx_tgs_microstep(model, data, mujoco, drive)
                        else:
                            mujoco.mj_step(model, data)
                    qpos[config_index, scale_index, load_axis, step] = (
                        data.qpos[qpos_addresses]
                    )
                    qvel[config_index, scale_index, load_axis, step] = (
                        data.qvel[dof_addresses]
                    )
                if (
                    not np.isfinite(qpos[config_index, scale_index, load_axis]).all()
                    or not np.isfinite(qvel[config_index, scale_index, load_axis]).all()
                    or any(int(warning.number) for warning in data.warning)
                ):
                    mode = "converted" if converted else "native"
                    raise FloatingPointError(f"{mode} loaded-drive audit became unstable")
    return qpos, qvel


def response_metrics(
    qpos: np.ndarray, qvel: np.ndarray, source_qpos: np.ndarray,
    source_qvel: np.ndarray, configurations: np.ndarray,
) -> dict[str, float]:
    base = configurations[:, None, None, None, :]
    delta = qpos - base
    source_delta = source_qpos - base
    diagonal = np.arange(18)
    return {
        "position_response_relative_error": relative_error(delta, source_delta),
        "arm_position_response_relative_error": relative_error(
            delta[:, :, :6, :, :6], source_delta[:, :, :6, :, :6],
        ),
        "hand_position_response_relative_error": relative_error(
            delta[:, :, 6:, :, 6:], source_delta[:, :, 6:, :, 6:],
        ),
        "same_axis_position_response_relative_error": relative_error(
            delta[:, :, diagonal, :, diagonal],
            source_delta[:, :, diagonal, :, diagonal],
        ),
        "position_rmse_rad": float(np.sqrt(np.mean(np.square(qpos - source_qpos)))),
        "velocity_response_relative_error_secondary": relative_error(
            qvel, source_qvel,
        ),
        "velocity_rmse_rad_s_secondary": float(
            np.sqrt(np.mean(np.square(qvel - source_qvel))),
        ),
        "maximum_abs_velocity_rad_s": float(np.max(np.abs(qvel))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve(strict=True)
    probe_path = args.sapien_probe.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite loaded-drive audit {output}")

    provenance_path = scene.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    drive_binding = provenance.get("drive_conversion", {})
    if (
        provenance.get("schema") != SCENE_SCHEMA
        or provenance.get("diagnostic_only") is not True
        or provenance.get("formal_renderer_3_3_eligible") is not False
        or Path(str(drive_binding.get("source", ""))).resolve(strict=True)
        != probe_path
        or drive_binding.get("source_sha256") != sha256(probe_path)
        or Path(str(drive_binding.get("runtime_converter_implementation", ""))).resolve(
            strict=True,
        ) != Path(__file__).resolve().with_name("sapien_equiv.py")
        or drive_binding.get("runtime_converter_implementation_sha256")
        != sha256(Path(__file__).resolve().with_name("sapien_equiv.py"))
        or drive_binding.get("native_mujoco_actuators_disabled_at_runtime") is not True
        or drive_binding.get("native_mujoco_joint_limits_disabled_at_runtime") is not True
    ):
        raise ValueError("scene is not bound to this isolated loaded-drive probe")

    with np.load(probe_path, allow_pickle=False) as values:
        if (
            str(np.asarray(values["schema"]).item()) != PROBE_SCHEMA
            or not bool(np.asarray(values["diagnostic_only"]).item())
            or bool(np.asarray(values["formal_renderer_3_3_eligible"]).item())
            or not np.isclose(float(values["physics_dt_s"]), SOURCE_DT_S)
            or int(values["loaded_response_steps"]) != 24
            or not np.allclose(values["stiffness"], 1000.0)
            or not np.allclose(values["damping"], 100.0)
        ):
            raise ValueError("loaded SAPIEN drive probe violates its isolated contract")
        names = [str(value) for value in values["joint_names"]]
        configurations = np.asarray(values["configurations"], dtype=np.float64)
        amplitude = np.asarray(values["loaded_response_amplitude_nm"], dtype=np.float64)
        sign = np.asarray(values["loaded_response_sign"], dtype=np.float64)
        source_qpos = np.asarray(values["loaded_response_qpos"], dtype=np.float64)
        source_qvel = np.asarray(values["loaded_response_qvel"], dtype=np.float64)
        steps = int(values["loaded_response_steps"])
    expected_shape = (7, 3, 18, 24, 18)
    if (
        len(names) != 18 or len(set(names)) != 18
        or configurations.shape != (7, 18) or amplitude.shape != (3, 18)
        or sign.shape != (7, 18) or source_qpos.shape != expected_shape
        or source_qvel.shape != expected_shape
        or not all(np.isfinite(value).all() for value in (
            configurations, amplitude, sign, source_qpos, source_qvel,
        ))
    ):
        raise ValueError("loaded SAPIEN drive arrays are malformed")

    native_qpos, native_qvel = simulate(
        scene, names, configurations, amplitude, sign, steps, converted=False,
    )
    converted_qpos, converted_qvel = simulate(
        scene, names, configurations, amplitude, sign, steps, converted=True,
    )
    native = response_metrics(
        native_qpos, native_qvel, source_qpos, source_qvel, configurations,
    )
    converted = response_metrics(
        converted_qpos, converted_qvel, source_qpos, source_qvel, configurations,
    )
    primary_keys = (
        "position_response_relative_error",
        "arm_position_response_relative_error",
        "hand_position_response_relative_error",
    )
    passed = all(converted[key] < native[key] for key in primary_keys)
    improvement = {
        key: float(native[key] / max(converted[key], np.finfo(float).eps))
        for key in primary_keys
    }

    output.mkdir(parents=True)
    trace = output / "trace.npz"
    np.savez_compressed(
        trace,
        schema=np.asarray("deximit_loaded_drive_audit_v1_diagnostic_only"),
        diagnostic_only=np.asarray(True),
        formal_renderer_3_3_eligible=np.asarray(False),
        grasp_candidate_executed=np.asarray(False),
        source_probe_sha256=np.asarray(sha256(probe_path)),
        scene_sha256=np.asarray(sha256(scene)),
        source_qpos=source_qpos, source_qvel=source_qvel,
        native_qpos=native_qpos, native_qvel=native_qvel,
        converted_qpos=converted_qpos, converted_qvel=converted_qvel,
    )
    report = {
        "schema": "deximit_loaded_drive_audit_v1_diagnostic_only",
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "table_object_and_contact_present": False,
        "scene": str(scene), "scene_sha256": sha256(scene),
        "scene_provenance": str(provenance_path),
        "sapien_probe": str(probe_path), "sapien_probe_sha256": sha256(probe_path),
        "configuration_count": 7,
        "load_scale_count": 3,
        "loaded_axis_count": 18,
        "duration_s_per_trial": steps * SOURCE_DT_S,
        "position_is_primary_metric": True,
        "velocity_is_secondary_because": (
            "PhysX exposes the last TGS substep velocity rather than a full-step average"
        ),
        "native_mujoco_drive": native,
        "converted_physx_order_drive": converted,
        "primary_error_improvement_factor": improvement,
        "candidate_free_gate_passed": passed,
        "gate_rule": "converted position error must beat native for all, arm, and hand",
        "trace": str(trace), "trace_sha256": sha256(trace),
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
