#!/usr/bin/env python3
"""Audit native and converted MuJoCo joint limits against isolated SAPIEN."""

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


PROBE_SCHEMA = (
    "deximit_sapien_joint_drive_probe_v9_clean_loaded_limits_diagnostic_only"
)
SCENE_SCHEMA = (
    "deximit_exact_urdf_mujoco_scene_v9_bound_friction_drive_limits_diagnostic_only"
)
TGS_SUBSTEPS = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def simulate(
    scene: Path, names: list[str], start: np.ndarray, target: np.ndarray,
    steps: int, *, converted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    if converted:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    data = mujoco.MjData(model)
    drive = PhysxTgsForceDrive(model, mujoco, names)
    qpos_addresses = np.asarray(drive.qpos_addresses, dtype=int)
    dof_addresses = np.asarray(drive.dof_addresses, dtype=int)
    actuator_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{name}")
        for name in names
    ], dtype=int)
    if np.any(actuator_ids < 0):
        raise ValueError("scene lacks one or more bound drives")
    qpos = np.empty((18, 2, steps, 18), dtype=np.float64)
    qvel = np.empty_like(qpos)
    for axis in range(18):
        for side in range(2):
            mujoco.mj_resetData(model, data)
            data.qpos[qpos_addresses] = start[axis, side]
            if converted:
                drive.set_target(target[axis, side])
            else:
                data.ctrl[actuator_ids] = target[axis, side]
            mujoco.mj_forward(model, data)
            for step in range(steps):
                if converted:
                    drive.begin_frame(data)
                for _ in range(TGS_SUBSTEPS):
                    if converted:
                        advance_physx_tgs_microstep(model, data, mujoco, drive)
                    else:
                        mujoco.mj_step(model, data)
                qpos[axis, side, step] = data.qpos[qpos_addresses]
                qvel[axis, side, step] = data.qvel[dof_addresses]
            if (
                not np.isfinite(qpos[axis, side]).all()
                or not np.isfinite(qvel[axis, side]).all()
                or any(int(warning.number) for warning in data.warning)
            ):
                mode = "converted" if converted else "native"
                raise FloatingPointError(f"{mode} limit audit became unstable")
    return qpos, qvel


def driven(values: np.ndarray) -> np.ndarray:
    result = np.empty(values.shape[:3], dtype=np.float64)
    for axis in range(18):
        result[axis] = values[axis, :, :, axis]
    return result


def metrics(
    qpos: np.ndarray, qvel: np.ndarray, source_qpos: np.ndarray,
    source_qvel: np.ndarray, start: np.ndarray, limits: np.ndarray,
) -> dict[str, float]:
    actual_axis = driven(qpos)
    source_axis = driven(source_qpos)
    start_axis = np.empty((18, 2, 1), dtype=np.float64)
    for axis in range(18):
        start_axis[axis, :, 0] = start[axis, :, axis]
    response_norm = max(
        float(np.linalg.norm(source_axis - start_axis)), np.finfo(float).eps,
    )
    lower_violation = limits[:, 0, None] - actual_axis[:, 0]
    upper_violation = actual_axis[:, 1] - limits[:, 1, None]
    return {
        "driven_axis_position_response_relative_error": float(
            np.linalg.norm(actual_axis - source_axis) / response_norm
        ),
        "driven_axis_position_rmse_rad": float(
            np.sqrt(np.mean(np.square(actual_axis - source_axis)))
        ),
        "driven_axis_position_error_rad_max": float(
            np.max(np.abs(actual_axis - source_axis))
        ),
        "maximum_joint_limit_violation_rad": float(max(
            0.0, np.max(lower_violation), np.max(upper_violation),
        )),
        "all_joint_position_rmse_rad": float(
            np.sqrt(np.mean(np.square(qpos - source_qpos)))
        ),
        "all_joint_position_error_rad_max": float(np.max(np.abs(qpos - source_qpos))),
        "all_joint_velocity_rmse_rad_s_secondary": float(
            np.sqrt(np.mean(np.square(qvel - source_qvel)))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve(strict=True)
    probe = args.sapien_probe.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite joint-limit audit {output}")
    provenance_path = scene.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    binding = provenance.get("drive_conversion", {})
    if (
        provenance.get("schema") != SCENE_SCHEMA
        or Path(str(binding.get("source", ""))).resolve(strict=True) != probe
        or binding.get("source_sha256") != sha256(probe)
        or Path(str(binding.get("runtime_converter_implementation", ""))).resolve(
            strict=True,
        ) != Path(__file__).resolve().with_name("sapien_equiv.py")
        or binding.get("runtime_converter_implementation_sha256")
        != sha256(Path(__file__).resolve().with_name("sapien_equiv.py"))
        or binding.get("native_mujoco_joint_limits_disabled_at_runtime") is not True
    ):
        raise ValueError("scene is not bound to this isolated joint-limit probe")
    with np.load(probe, allow_pickle=False) as values:
        if (
            str(np.asarray(values["schema"]).item()) != PROBE_SCHEMA
            or not bool(values["diagnostic_only"])
            or bool(values["formal_renderer_3_3_eligible"])
            or int(values["limit_response_steps"]) != 24
        ):
            raise ValueError("SAPIEN joint-limit probe violates its isolated contract")
        names = [str(value) for value in values["joint_names"]]
        start = np.asarray(values["limit_response_start_qpos"], dtype=np.float64)
        target = np.asarray(values["limit_response_drive_target"], dtype=np.float64)
        source_qpos = np.asarray(values["limit_response_qpos"], dtype=np.float64)
        source_qvel = np.asarray(values["limit_response_qvel"], dtype=np.float64)
        limits = np.asarray(values["joint_limits"], dtype=np.float64)
        steps = int(values["limit_response_steps"])
    if (
        len(names) != 18 or start.shape != (18, 2, 18)
        or target.shape != start.shape or source_qpos.shape != (18, 2, 24, 18)
        or source_qvel.shape != source_qpos.shape or limits.shape != (18, 2)
    ):
        raise ValueError("SAPIEN joint-limit arrays are malformed")
    native_qpos, native_qvel = simulate(
        scene, names, start, target, steps, converted=False,
    )
    converted_qpos, converted_qvel = simulate(
        scene, names, start, target, steps, converted=True,
    )
    native = metrics(
        native_qpos, native_qvel, source_qpos, source_qvel, start, limits,
    )
    converted = metrics(
        converted_qpos, converted_qvel, source_qpos, source_qvel, start, limits,
    )
    source = metrics(
        source_qpos, source_qvel, source_qpos, source_qvel, start, limits,
    )
    passed = bool(
        converted["driven_axis_position_rmse_rad"]
        < native["driven_axis_position_rmse_rad"]
        and converted["all_joint_position_rmse_rad"]
        < native["all_joint_position_rmse_rad"]
        and converted["maximum_joint_limit_violation_rad"]
        <= source["maximum_joint_limit_violation_rad"] + 1.0e-7
    )
    output.mkdir(parents=True)
    trace = output / "trace.npz"
    np.savez_compressed(
        trace,
        schema=np.asarray("deximit_joint_limit_audit_v1_diagnostic_only"),
        diagnostic_only=np.asarray(True), formal_renderer_3_3_eligible=np.asarray(False),
        grasp_candidate_executed=np.asarray(False), source_probe_sha256=np.asarray(sha256(probe)),
        scene_sha256=np.asarray(sha256(scene)), source_qpos=source_qpos,
        source_qvel=source_qvel, native_qpos=native_qpos, native_qvel=native_qvel,
        converted_qpos=converted_qpos, converted_qvel=converted_qvel,
    )
    report = {
        "schema": "deximit_joint_limit_audit_v1_diagnostic_only",
        "diagnostic_only": True, "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "table_object_and_contact_present": False,
        "scene": str(scene), "scene_sha256": sha256(scene),
        "sapien_probe": str(probe), "sapien_probe_sha256": sha256(probe),
        "joint_count": 18, "sides_per_joint": 2,
        "source_sapien_limits": source,
        "native_mujoco_limits": native,
        "converted_physx_hard_limits": converted,
        "candidate_free_gate_passed": passed,
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
