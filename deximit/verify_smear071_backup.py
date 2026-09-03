#!/usr/bin/env python3
"""Validate the checked-in smear071 fixture without running DexImit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "deximit/fixture/smear071"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"missing fixture file: {path}")
    return path


def main() -> int:
    input_root = FIXTURE / "input"
    sapien_root = FIXTURE / "sapien"
    mujoco_root = FIXTURE / "mujoco"
    paths = {
        "mesh": input_root / "tool_078_meters.ply",
        "human": input_root / "human_reference_v2.npz",
        "contact": input_root / "contact_geom_v3.npz",
        "prompt": input_root / "mano_prompt_v1.npz",
        "manual": input_root / "manual_pickup_subactions_v1.json",
        "pool": input_root / "candidate_pools/bodex_right_f4_d0_seed20260827_v2.npz",
        "target": sapien_root / "candidate21_target.npy",
        "trace": sapien_root / "candidate21_trace.npz",
        "summary": sapien_root / "summary.json",
        "scene": mujoco_root / "scene.xml",
        "provenance": mujoco_root / "scene.provenance.json",
        "report": mujoco_root / "report.json",
        "home_probe": mujoco_root / "bindings/joint_drive_probe.npz",
    }
    for path in paths.values():
        require(path)

    for path in (paths["scene"], paths["provenance"], paths["summary"], paths["report"]):
        if "/data_all/" in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"stale workstation path remains in {path}")

    summary = load_json(paths["summary"])
    if (
        summary.get("schema") != "deximit_original_sapien_screen_v10_joint_force_metrics_diagnostic_only"
        or summary.get("diagnostic_only") is not True
        or summary.get("formal_renderer_3_3_eligible") is not False
        or summary.get("original_sapien_pass_count") != 1
    ):
        raise RuntimeError("SAPIEN summary does not describe the frozen diagnostic pass")
    rows = [
        row for row in summary.get("original_sapien_passes", [])
        if row.get("source_candidate_index") == 21 and row.get("depth") == 0
    ]
    if len(rows) != 1 or rows[0].get("original_sapien_pass") is not True:
        raise RuntimeError("frozen candidate 21 SAPIEN pass is missing")
    row = rows[0]
    if repo_path(str(row["exported_trajectory"])) != paths["trace"].resolve():
        raise RuntimeError("frozen summary is not bound to candidate21_trace.npz")
    if repo_path(str(row["selected_target"])) != paths["target"].resolve():
        raise RuntimeError("frozen summary is not bound to candidate21_target.npy")
    if row.get("exported_trajectory_sha256") != sha256(paths["trace"]):
        raise RuntimeError("SAPIEN trace hash does not match the frozen summary")
    if row.get("selected_target_sha256") != sha256(paths["target"]):
        raise RuntimeError("selected target hash does not match the frozen summary")

    with np.load(paths["prompt"], allow_pickle=False) as data:
        if (
            str(np.asarray(data["schema"]).item()) != "deximit_taco_mano_prompt_v1_diagnostic_only"
            or bool(np.asarray(data["diagnostic_only"]).item()) is not True
            or bool(np.asarray(data["formal_renderer_3_3_eligible"]).item()) is not False
            or repo_path(str(np.asarray(data["source_contact_v3"]).item())) != paths["contact"].resolve()
            or str(np.asarray(data["source_contact_v3_sha256"]).item()) != sha256(paths["contact"])
        ):
            raise RuntimeError("MANO prompt provenance is not fixture-bound")

    with np.load(paths["trace"], allow_pickle=False) as data:
        if (
            str(np.asarray(data["schema"]).item()) != "deximit_sapien_hand_trace_v8_joint_force_metrics_diagnostic_only"
            or int(np.asarray(data["candidate_index"]).item()) != 21
            or int(np.asarray(data["pool_depth"]).item()) != 0
            or np.asarray(data["right_robot_qpos_sapien"]).shape
            != (len(data["phase"]), 18)
        ):
            raise RuntimeError("frozen SAPIEN trace violates the replay contract")

    provenance = load_json(paths["provenance"])
    if (
        provenance.get("diagnostic_only") is not True
        or provenance.get("formal_renderer_3_3_eligible") is not False
        or repo_path(str(provenance["sapien_summary"])) != paths["summary"].resolve()
        or provenance.get("sapien_summary_sha256") != sha256(paths["summary"])
    ):
        raise RuntimeError("MuJoCo provenance is not bound to the frozen SAPIEN summary")
    for section in (
        "contact_calibration", "pinch_friction_calibration",
        "strong_friction_conversion", "drive_conversion", "hand_contact_calibration",
    ):
        binding = provenance.get(section, {})
        source = repo_path(str(binding.get("source", "")))
        if not source.is_file() or binding.get("source_sha256") != sha256(source):
            raise RuntimeError(f"invalid provenance source for {section}")

    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    if model.nq != 25 or model.nv != 24 or model.nu != 18 or model.ncam != 1:
        raise RuntimeError("portable MuJoCo scene dimensions differ from the frozen contract")
    report = load_json(paths["report"])
    gate = report.get("strict_gate", {})
    if (
        gate.get("passed") is not True
        or report.get("scene_sha256") != sha256(paths["scene"])
        or report.get("source_trace_sha256") != sha256(paths["trace"])
        or report.get("source_summary_sha256") != sha256(paths["summary"])
        or report.get("trace_sha256") != sha256(mujoco_root / "trace.npz")
    ):
        raise RuntimeError("frozen MuJoCo report is not bound to the fixture files")

    print("smear071 fixture OK: candidate 21 SAPIEN pass and MuJoCo strict pass")
    print(f"scene: {paths['scene']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"fixture verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
