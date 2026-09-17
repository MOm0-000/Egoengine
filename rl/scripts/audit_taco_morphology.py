#!/usr/bin/env python3
"""Read-only MANO/XHand neutral-morphology and scaled-target audit.

This script measures morphology in independent anatomical frames and evaluates
two diagnostic point-scaling hypotheses. It never edits the released GT,
production MJCF, retarget weights, or replay artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src"), str(ROOT / "scripts")]

from audit_taco_geometry_contract import (  # noqa: E402
    SAMPLED_ROWS,
    _load,
    _summary,
    _posture_strata,
    _probe_reachability,
)
from egoengine_repro.evaluation.taco_surface import (  # noqa: E402
    MANO21_SOURCE,
    MANO_TIP_VERTICES,
    _load_model_data,
    reconstruct_taco_mano,
)
from egoengine_repro.retarget.taco_bimanual import (  # noqa: E402
    DIPS,
    FINGERS,
    SIDES,
    TIPS,
    geometric_frame,
)


DEFAULT_RUN = ROOT / "runs/taco_pour_bimanual_mano_fk_v1"
DEFAULT_SCENE = ROOT / (
    "models/taco_xhand/xhand/bimanual/"
    "taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
)
DEFAULT_HANDS = ROOT / (
    "data/taco_v1/pour_bowl_plate/hand_poses/Hand_Poses/"
    "(pour in some, bowl, plate)/20230927_017"
)
DEFAULT_MANO = ROOT / "data/taco_v1/hand_poses_v1/mano_v1_2/models"
DEFAULT_OUTPUT = ROOT / "runs/taco_pour_morphology_audit_v3"

# MANO21 indices in the published wrist, thumb, index, middle, ring, pinky
# ordering. The robot uses the same four-link abstraction: wrist, MCP, PIP,
# DIP, endpoint (the first body origin is the MCP-like point).
HUMAN_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

ROBOT_CHAINS = {
    "thumb": ("{side}_hand_link", "{side}_hand_thumb_bend_link",
              "{side}_hand_thumb_rota_link1", "{side}_hand_thumb_rota_link2"),
    "index": ("{side}_hand_link", "{side}_hand_index_bend_link",
              "{side}_hand_index_rota_link1", "{side}_hand_index_rota_link2"),
    "middle": ("{side}_hand_link", "{side}_hand_mid_link1", "{side}_hand_mid_link2"),
    "ring": ("{side}_hand_link", "{side}_hand_ring_link1", "{side}_hand_ring_link2"),
    "pinky": ("{side}_hand_link", "{side}_hand_pinky_link1", "{side}_hand_pinky_link2"),
}


def _lengths(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return segments, float(segments.sum()), float(np.linalg.norm(points[-1] - points[0]))


def _human_neutral(side: str, model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return neutral MANO21 points and the neutral anatomical frame."""
    import smplx
    import torch
    from smplx.utils import Struct

    layer = smplx.MANO(
        "unused", data_struct=Struct(**_load_model_data(model_path)),
        is_rhand=side == "right", use_pca=False, flat_hand_mean=True,
        create_transl=False,
    )
    with torch.no_grad():
        neutral = layer(
            global_orient=torch.zeros(1, 3), hand_pose=torch.zeros(1, 45),
            betas=torch.zeros(1, 10), return_verts=True,
        )
    all_points = torch.cat(
        (neutral.joints, neutral.vertices[:, MANO_TIP_VERTICES[side]]), dim=1,
    ).numpy()[0]
    points = all_points[MANO21_SOURCE].astype(float)
    normal = np.cross(points[5] - points[0], points[17] - points[0])
    if side == "left":
        normal = -normal
    frame = geometric_frame(normal, points[9] - points[0])
    local = (points - points[0]) @ frame
    return local, frame


def _robot_neutral(model: mujoco.MjModel, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Return robot chain points in the same wrist-centered anatomical frame."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    root = model.body(f"{side}_hand_link").id
    palm = model.site(f"{side}_palm").id
    middle = model.body(f"{side}_hand_mid_link1").id
    frame = geometric_frame(
        data.site_xmat[palm].reshape(3, 3)[:, 0],
        data.xpos[middle] - data.xpos[root],
    )
    all_points: list[np.ndarray] = []
    for finger in FINGERS:
        chain = [data.xpos[model.body(name.format(side=side)).id] for name in ROBOT_CHAINS[finger]]
        site = model.site(f"{side}_{finger}_tip").id
        body = int(model.site_bodyid[site])
        chain.append(data.site_xpos[site])
        local = (np.asarray(chain) - data.xpos[root]) @ frame
        all_points.extend(local)
    # The first entry of each chain is repeated; reshape via a deterministic
    # per-finger dictionary in the caller instead of relying on this flat list.
    points_by_finger = {}
    offset = 0
    for finger in FINGERS:
        count = len(ROBOT_CHAINS[finger]) + 1
        points_by_finger[finger] = np.asarray(all_points[offset:offset + count])
        offset += count
    return points_by_finger, frame


def _human_geometry(side: str, local: np.ndarray) -> dict:
    records = {}
    chains = {}
    for finger, indices in HUMAN_CHAINS.items():
        points = local[list(indices)]
        segments, chain_length, tip_reach = _lengths(points)
        chains[finger] = points
        records[finger] = {
            "segment_lengths_mm": (segments * 1000).tolist(),
            "chain_length_mm": chain_length * 1000,
            "wrist_to_tip_mm": tip_reach * 1000,
            "neutral_tip_vector_in_palm_frame_mm": (points[-1] * 1000).tolist(),
        }
    return {
        "side": side,
        "palm_width_index_mcp_to_pinky_mcp_mm": float(np.linalg.norm(local[5] - local[17]) * 1000),
        "palm_length_wrist_to_middle_mcp_mm": float(np.linalg.norm(local[9]) * 1000),
        "fingers": records,
        "chain_points": {key: value.tolist() for key, value in chains.items()},
    }


def _robot_geometry(side: str, points: dict[str, np.ndarray]) -> dict:
    records = {}
    for finger, chain in points.items():
        segments, chain_length, tip_reach = _lengths(chain)
        records[finger] = {
            "segment_lengths_mm": (segments * 1000).tolist(),
            "chain_length_mm": chain_length * 1000,
            "wrist_to_tip_mm": tip_reach * 1000,
            "neutral_tip_vector_in_palm_frame_mm": (chain[-1] * 1000).tolist(),
        }
    return {
        "side": side,
        "palm_width_index_mcp_to_pinky_mcp_mm": float(np.linalg.norm(points["index"][1] - points["pinky"][1]) * 1000),
        "palm_length_wrist_to_middle_mcp_mm": float(np.linalg.norm(points["middle"][1]) * 1000),
        "fingers": records,
        "chain_points": {key: value.tolist() for key, value in points.items()},
    }


def _trajectory_morphology_stability(
    human_reference: dict, robot_reference: dict, model: mujoco.MjModel,
) -> dict:
    """Measure whether neutral ratios remain meaningful over posture strata."""
    strata = _posture_strata(human_reference)
    human_joints = np.asarray(human_reference["joint_positions_sim"], dtype=float)
    data = mujoco.MjData(model)
    result = {}
    for hi, side in enumerate(SIDES):
        chain_lengths = np.empty((len(human_joints), 5), dtype=float)
        palm_width = np.empty(len(human_joints), dtype=float)
        palm_length = np.empty(len(human_joints), dtype=float)
        for row, joints in enumerate(human_joints[:, hi]):
            palm_width[row] = np.linalg.norm(joints[5] - joints[17])
            palm_length[row] = np.linalg.norm(joints[9] - joints[0])
            for fi, finger in enumerate(FINGERS):
                chain = joints[list(HUMAN_CHAINS[finger])]
                chain_lengths[row, fi] = np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()
        robot_chain_lengths = np.empty_like(chain_lengths)
        robot_width = np.empty(len(human_joints), dtype=float)
        robot_length = np.empty(len(human_joints), dtype=float)
        for row, qpos in enumerate(robot_reference["qpos"]):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            body = {}
            for names in ROBOT_CHAINS.values():
                for template in names:
                    name = template.format(side=side)
                    body[name] = data.xpos[model.body(name).id]
            robot_width[row] = np.linalg.norm(body[f"{side}_hand_index_bend_link"] -
                                               body[f"{side}_hand_pinky_link1"])
            robot_length[row] = np.linalg.norm(body[f"{side}_hand_mid_link1"] -
                                                body[f"{side}_hand_link"])
            for fi, finger in enumerate(FINGERS):
                names = [name.format(side=side) for name in ROBOT_CHAINS[finger]]
                points = [body[name] for name in names]
                site = model.site(f"{side}_{finger}_tip").id
                points.append(data.site_xpos[site])
                robot_chain_lengths[row, fi] = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        posture_records = {}
        for posture, indices in strata[side]["strata_rows"].items():
            rows = np.asarray(indices, dtype=int)
            ratios = robot_chain_lengths[rows] / chain_lengths[rows]
            posture_records[posture] = {
                "rows": [int(v) for v in rows.tolist()],
                "human_chain_length_mm_by_finger": _summary(chain_lengths[rows] * 1000),
                "robot_chain_length_mm_by_finger": _summary(robot_chain_lengths[rows] * 1000),
                "robot_over_human_chain_ratio_by_finger": [
                    _summary(ratios[:, fi]) for fi in range(5)
                ],
                "human_palm_width_mm": _summary(palm_width[rows] * 1000),
                "human_palm_length_mm": _summary(palm_length[rows] * 1000),
                "ratio_p95_minus_p05_by_finger": (
                    np.percentile(ratios, 95, axis=0) - np.percentile(ratios, 5, axis=0)
                ).tolist(),
            }
        result[side] = {
            "posture_strata": posture_records,
            "all_rows_ratio_p95_minus_p05_by_finger": (
                np.percentile(robot_chain_lengths / chain_lengths, 95, axis=0) -
                np.percentile(robot_chain_lengths / chain_lengths, 5, axis=0)
            ).tolist(),
            "interpretation": "Robot rigid-link chain lengths are fixed; variation is driven by MANO joint/skin landmarks and tests whether neutral scale transfers across posture.",
        }
    return result


def _morphology_comparison(human: dict, robot: dict) -> dict:
    scales = {}
    human_lengths, robot_lengths = [], []
    for finger in FINGERS:
        h = human["fingers"][finger]
        r = robot["fingers"][finger]
        scale = r["chain_length_mm"] / h["chain_length_mm"]
        human_segments = np.asarray(h["segment_lengths_mm"])
        robot_segments = np.asarray(r["segment_lengths_mm"])
        segment_scale = None
        if human_segments.shape == robot_segments.shape:
            segment_scale = (robot_segments / human_segments).tolist()
        scales[finger] = {
            "chain_scale_robot_over_human": float(scale),
            "tip_reach_scale_robot_over_human": float(r["wrist_to_tip_mm"] / h["wrist_to_tip_mm"]),
            "segment_scale_robot_over_human": segment_scale,
            "human_segment_count": int(human_segments.size),
            "robot_segment_count": int(robot_segments.size),
            "segmentwise_comparison": "direct" if segment_scale is not None else "not_comparable_missing_robot_joint",
            "tip_vector_difference_mm": (np.asarray(r["neutral_tip_vector_in_palm_frame_mm"]) -
                                          np.asarray(h["neutral_tip_vector_in_palm_frame_mm"])).tolist(),
        }
        human_lengths.append(h["chain_length_mm"])
        robot_lengths.append(r["chain_length_mm"])
    human_lengths = np.asarray(human_lengths)
    robot_lengths = np.asarray(robot_lengths)
    global_scale = float(np.dot(human_lengths, robot_lengths) / np.dot(human_lengths, human_lengths))
    palm_width_scale = robot["palm_width_index_mcp_to_pinky_mcp_mm"] / human["palm_width_index_mcp_to_pinky_mcp_mm"]
    palm_length_scale = robot["palm_length_wrist_to_middle_mcp_mm"] / human["palm_length_wrist_to_middle_mcp_mm"]
    return {
        "global_chain_scale_robot_over_human": global_scale,
        "global_chain_scale_rmse_mm": float(np.sqrt(np.mean((global_scale * human_lengths - robot_lengths) ** 2))),
        "palm_width_scale_robot_over_human": float(palm_width_scale),
        "palm_length_scale_robot_over_human": float(palm_length_scale),
        "per_finger": scales,
        "spread_of_per_finger_chain_scales": float(max(v["chain_scale_robot_over_human"] for v in scales.values()) -
                                                    min(v["chain_scale_robot_over_human"] for v in scales.values())),
        "interpretation": "Scales are morphology measurements, not author calibration or production retarget settings.",
    }


def _scaled_human_targets(human_reference: dict, scales: np.ndarray, mode: str) -> dict:
    target = {key: np.array(value, copy=True) for key, value in human_reference.items()}
    wrist = human_reference["T_sim_wrist_target"][:, :, :3, 3]
    wrist_R = human_reference["T_sim_wrist_target"][:, :, :3, :3]
    points = human_reference["T_sim_fingertip_target"][:, :, :, :3, 3]
    local = np.einsum("thij,thfj->thfi", np.swapaxes(wrist_R, -1, -2), points - wrist[:, :, None])
    if mode == "global_chain":
        factors = np.broadcast_to(scales[:, None], (2, 5))
    elif mode == "per_finger_chain":
        factors = np.broadcast_to(scales, (2, 5))
    else:
        raise ValueError(f"unknown morphology scale mode {mode}")
    mapped = wrist[:, :, None] + np.einsum(
        "thij,thfj->thfi", wrist_R, local * factors[None, :, :, None]
    )
    target["T_sim_fingertip_target"][:, :, :, :3, 3] = mapped
    target["morphology_target_scaling_mode"] = np.asarray(mode)
    return target


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    report, human_reference, robot_reference = _load(args.run)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    if len(robot_reference["qpos"]) != len(human_reference["frame_indices"]):
        raise ValueError("human and robot frame counts differ")

    human_geometry, robot_geometry, comparison = {}, {}, {}
    neutral_frames = {}
    for side in SIDES:
        human_local, human_frame = _human_neutral(side, args.mano_models / f"MANO_{side.upper()}.pkl")
        robot_points, robot_frame = _robot_neutral(model, side)
        human_geometry[side] = _human_geometry(side, human_local)
        robot_geometry[side] = _robot_geometry(side, robot_points)
        comparison[side] = _morphology_comparison(human_geometry[side], robot_geometry[side])
        neutral_frames[side] = {
            "human_palm_frame_columns_world_neutral_model": human_frame.tolist(),
            "robot_palm_frame_columns_world_qpos0": robot_frame.tolist(),
            "frames_are_compared_only_after_localization": True,
        }

    stability = _trajectory_morphology_stability(human_reference, robot_reference, model)

    global_scales = np.asarray([comparison[side]["global_chain_scale_robot_over_human"] for side in SIDES])
    per_finger_scales = np.asarray([
        [comparison[side]["per_finger"][finger]["chain_scale_robot_over_human"] for finger in FINGERS]
        for side in SIDES
    ])
    settings = dict(report["inherited_settings"])
    probes = {"baseline": _probe_reachability(args.scene, human_reference, robot_reference, settings, SAMPLED_ROWS)}
    candidates = {}
    for mode, scales in (("global_chain", global_scales), ("per_finger_chain", per_finger_scales)):
        diagnostic_human = _scaled_human_targets(human_reference, scales, mode)
        candidates[mode] = {
            "scale_robot_over_human": scales.tolist(),
            "target_points_only_changed": True,
            "orientation_targets_unchanged": True,
            "wrist_targets_unchanged": True,
            "probes": _probe_reachability(args.scene, diagnostic_human, robot_reference, settings, SAMPLED_ROWS),
        }

    args.output.mkdir(parents=True)
    result = {
        "status": "neutral_morphology_audit_read_only",
        "weights_modified": False,
        "formal_reference_modified": False,
        "rl_validation_completed": False,
        "source_contract": {
            "human": "MANO v1.2 zero betas, flat-hand-mean identity pose, MANO21 joints and released tip vertices",
            "robot": "XHand MJCF qpos0, wrist body origin, native XML fingertip site endpoint",
            "anatomical_frame": "+X palm normal projected orthogonal to +Z; +Z wrist-to-middle-MCP; +Y=cross(+Z,+X)",
            "point_units": "meters internally, millimeters in reported measurements",
        },
        "neutral_frames": neutral_frames,
        "human_neutral_geometry": human_geometry,
        "robot_neutral_geometry": robot_geometry,
        "morphology_comparison": comparison,
        "trajectory_morphology_stability": stability,
        "empirical_scaled_target_probes": candidates,
        "baseline_probe": probes["baseline"],
        "conclusions": {
            "global_scale_consistent_between_hands": bool(np.max(global_scales) - np.min(global_scales) < 0.02),
            "per_finger_scale_spread": float(np.ptp(per_finger_scales, axis=1).max()),
            "morphology_candidate_is_formal_contract": False,
            "grid_search_allowed": False,
            "reason_grid_blocked": "A scale can reduce point mismatch while changing the human task geometry; the diagnostic probes do not establish author correspondence or cross-task stability.",
        },
        "limitations": [
            "Neutral morphology is a reference convention, not a measured subject-specific calibration.",
            "Scaling wrist-relative fingertip vectors does not reconstruct intermediate joint landmarks or contact geometry.",
            "The six-row probes are local constrained solves, not global reachable-set proofs.",
            "No scaled candidate is allowed into Replay or RL before cross-task and visual validation.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--hands", type=Path, default=DEFAULT_HANDS)
    parser.add_argument("--mano-models", type=Path, default=DEFAULT_MANO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result = run(parser.parse_args())
    print(json.dumps(result["conclusions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
