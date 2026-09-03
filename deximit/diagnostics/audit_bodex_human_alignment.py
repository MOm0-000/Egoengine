#!/usr/bin/env python3
"""Measure BODex proposals against the observed human grasp.

The physics sweep answers whether a proposal can hold the free object.  This
program answers the separate question of whether that already-successful
proposal resembles the human hand.  All comparisons are made in the moving
object frame, so object motion cannot masquerade as hand-pose error.

This is a diagnostic-only selector.  It cannot produce an input for the formal
renderer, controller, or 3.3 pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egoengine_repro.action.replay import MujocoReplayBackend

from bodex_triptych import (
    load_candidate,
    load_object_pose,
    map_target,
    project_hand_joint_limits,
    require_file,
    require_scene,
)


SCHEMA = "xhand_bodex_human_alignment_v1_diagnostic_only"
SWEEP_SCHEMA = "xhand_bodex_candidate_sweep_v4_diagnostic_only"
PALM_SITE = "right_palm"
TIP_SITES = tuple(
    f"right_{name}_tip" for name in ("thumb", "index", "middle", "ring", "pinky")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--physics-report", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--object-reference", type=Path, required=True)
    parser.add_argument("--human-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-row", type=int, default=40)
    parser.add_argument("--window-start-row", type=int, default=40)
    parser.add_argument("--window-end-row", type=int, default=60)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform(position: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def object_transform(object_qpos: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(
        object_qpos[3:], scalar_first=True,
    ).as_matrix()
    result[:3, 3] = object_qpos[:3]
    return result


def relative(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return ``first^-1 @ second`` for one transform or a transform batch."""
    return np.linalg.inv(first) @ second


def rotation_error_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = np.swapaxes(first[..., :3, :3], -1, -2) @ second[..., :3, :3]
    return Rotation.from_matrix(delta).magnitude()


def site_transform(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site < 0:
        raise ValueError(f"diagnostic scene lacks required site {name!r}")
    return transform(data.site_xpos[site], data.site_xmat[site])


def candidate_relative_geometry(
    backend: MujocoReplayBackend,
    candidate_path: Path,
    candidate_index: int,
    object_qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    poses, joints, _ = load_candidate(candidate_path, candidate_index)
    grasp = map_target(
        backend.model, object_qpos, poses[1], joints[1], backend.object_qpos_address,
    )
    squeeze = map_target(
        backend.model, object_qpos, poses[2], joints[2], backend.object_qpos_address,
    )
    object_inverse = np.linalg.inv(object_transform(object_qpos))

    grasp_qpos, _ = project_hand_joint_limits(backend.model, grasp.qpos)
    squeeze_qpos, _ = project_hand_joint_limits(backend.model, squeeze.qpos)
    backend.data.qpos[:] = grasp_qpos
    backend.data.qvel[:] = 0.0
    backend.mujoco.mj_forward(backend.model, backend.data)
    palm = object_inverse @ site_transform(backend.model, backend.data, PALM_SITE)

    backend.data.qpos[:] = squeeze_qpos
    backend.data.qvel[:] = 0.0
    backend.mujoco.mj_forward(backend.model, backend.data)
    tips = np.stack([
        object_inverse @ site_transform(backend.model, backend.data, name)
        for name in TIP_SITES
    ])
    return palm, tips


def load_human(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "frame_indices", "hand_order", "T_sim_wrist_target",
            "T_sim_fingertip_target", "T_sim_object_reference", "valid_hand",
        }
        if required - set(data.files):
            raise ValueError("human reference lacks required hand/object transforms")
        result = {name: np.asarray(data[name]).copy() for name in required}
    order = tuple(str(value) for value in result["hand_order"].tolist())
    if order != ("right",):
        raise ValueError(f"expected one right-hand reference, got {order}")
    count = len(result["frame_indices"])
    if (
        result["T_sim_wrist_target"].shape != (count, 1, 4, 4)
        or result["T_sim_fingertip_target"].shape != (count, 1, 5, 4, 4)
        or result["T_sim_object_reference"].shape != (count, 1, 4, 4)
        or result["valid_hand"].shape != (count, 1)
    ):
        raise ValueError("human reference array shapes are inconsistent")
    for name in (
        "T_sim_wrist_target", "T_sim_fingertip_target", "T_sim_object_reference",
    ):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"human reference {name} contains non-finite values")
    return result


def load_passed_rows(
    report_path: Path, candidate_path: Path, scene_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != SWEEP_SCHEMA
        or report.get("diagnostic_only") is not True
        or report.get("formal_renderer_3_3_eligible") is not False
        or report.get("candidate_sha256") != sha256(candidate_path)
        or report.get("scene_sha256") != sha256(scene_path)
    ):
        raise ValueError("physics report does not match the diagnostic inputs")
    rows = [row for row in report["ranked_candidates"] if row["physics_gate_pass"]]
    if not rows:
        raise ValueError("physics report has no candidate that passed the strict gate")
    return report, rows


def metric_row(
    candidate_index: int,
    candidate_palm: np.ndarray,
    candidate_tips: np.ndarray,
    human_palm: np.ndarray,
    human_tips: np.ndarray,
    anchor_offset: int,
    physics: dict[str, object],
) -> dict[str, object]:
    palm_rotation = np.rad2deg(rotation_error_rad(candidate_palm, human_palm))
    palm_position = np.linalg.norm(
        candidate_palm[None, :3, 3] - human_palm[:, :3, 3], axis=-1,
    )
    tip_position = np.linalg.norm(
        candidate_tips[None, :, :3, 3] - human_tips[:, :, :3, 3], axis=-1,
    )
    return {
        "candidate_index": int(candidate_index),
        "anchor_palm_rotation_error_deg": float(palm_rotation[anchor_offset]),
        "anchor_palm_position_error_cm": float(100.0 * palm_position[anchor_offset]),
        "anchor_fingertip_position_error_cm": float(
            100.0 * tip_position[anchor_offset].mean()
        ),
        "window_palm_rotation_error_mean_deg": float(palm_rotation.mean()),
        "window_palm_position_error_mean_cm": float(100.0 * palm_position.mean()),
        "window_fingertip_position_error_mean_cm": float(100.0 * tip_position.mean()),
        "window_opposed_four_fingertip_error_mean_cm": float(
            100.0 * tip_position[:, :4].mean()
        ),
        "physics": {
            key: physics[key] for key in (
                "hold_min_lift_m", "hold_vertical_span_m",
                "minimum_sampled_hand_object_gap_m", "peak_normal_force_n",
            )
        },
    }


def main() -> int:
    args = parse_args()
    candidate_path = require_file(args.candidate, "candidate")
    report_path = require_file(args.physics_report, "physics report")
    scene_path = require_file(args.scene, "scene")
    object_reference = require_file(args.object_reference, "object reference")
    human_reference = require_file(args.human_reference, "human reference")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite alignment output: {output}")
    if not (
        0 <= args.window_start_row <= args.anchor_row <= args.window_end_row
    ):
        raise ValueError("anchor row must lie inside the inclusive comparison window")

    physics_report, passed = load_passed_rows(report_path, candidate_path, scene_path)
    human = load_human(human_reference)
    count = len(human["frame_indices"])
    if args.window_end_row >= count:
        raise ValueError(f"comparison row exceeds the {count}-row human reference")
    rows = np.arange(args.window_start_row, args.window_end_row + 1)
    if not np.asarray(human["valid_hand"])[rows, 0].all():
        raise ValueError("human hand is invalid inside the comparison window")

    human_object = np.asarray(human["T_sim_object_reference"])[rows, 0]
    human_palm = relative(
        human_object, np.asarray(human["T_sim_wrist_target"])[rows, 0],
    )
    human_tips = relative(
        human_object[:, None], np.asarray(human["T_sim_fingertip_target"])[rows, 0],
    )

    backend = MujocoReplayBackend(
        scene_path, object_joint_name="right_object_joint", hand_order=("right",),
        seed=int(physics_report["parameters"]["seed"]), finger_impedance=None,
    )
    require_scene(backend)
    object_qpos = load_object_pose(
        object_reference, expected_nq=backend.model.nq,
        object_address=backend.object_qpos_address,
    )
    metrics = []
    for physics in passed:
        index = int(physics["candidate_index"])
        palm, tips = candidate_relative_geometry(
            backend, candidate_path, index, object_qpos,
        )
        metrics.append(metric_row(
            index, palm, tips, human_palm, human_tips,
            args.anchor_row - args.window_start_row, physics,
        ))
    # DexImit's rotation prompt ranks the palm orientation first.  The window
    # average and positions are deterministic tie breakers, not an opaque sum.
    ranked = sorted(metrics, key=lambda row: (
        row["anchor_palm_rotation_error_deg"],
        row["window_palm_rotation_error_mean_deg"],
        row["anchor_palm_position_error_cm"],
        row["anchor_fingertip_position_error_cm"],
    ))
    result = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "selection_scope": "strict-physics-pass candidates only",
        "selection_rule": (
            "minimum anchor palm rotation error; then window palm rotation, "
            "anchor palm position, and anchor fingertip position"
        ),
        "coordinate_frame": "object local; human object motion removed per row",
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "physics_report": str(report_path),
        "physics_report_sha256": sha256(report_path),
        "scene": str(scene_path),
        "scene_sha256": sha256(scene_path),
        "object_reference": str(object_reference),
        "object_reference_sha256": sha256(object_reference),
        "human_reference": str(human_reference),
        "human_reference_sha256": sha256(human_reference),
        "anchor_row": args.anchor_row,
        "anchor_source_frame": int(human["frame_indices"][args.anchor_row]),
        "window_rows_inclusive": [args.window_start_row, args.window_end_row],
        "window_source_frames_inclusive": [
            int(human["frame_indices"][args.window_start_row]),
            int(human["frame_indices"][args.window_end_row]),
        ],
        "physics_pass_count": len(ranked),
        "selected_candidate_index": ranked[0]["candidate_index"],
        "ranked_physics_pass_candidates": ranked,
        "limitations": [
            "BODex provides a static grasp proposal, whereas the human window moves over time.",
            "This audit does not add DexImit's arm planner or source-timed motion transfer.",
            "The comparison covers the right hand; the video's left hand is outside this isolated probe.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "physics_pass_count": len(ranked),
        "selected_candidate_index": ranked[0]["candidate_index"],
        "selected_metrics": ranked[0],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
