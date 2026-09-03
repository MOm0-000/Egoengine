#!/usr/bin/env python3
"""Run DexImit's original MANO ranking, cuRobo planning, and SAPIEN screen.

The program consumes already generated BODex pools but deliberately feeds the
*raw* seeds to DexImit's unchanged ``rollout_and_select_grasp`` function.  That
function applies the official pregrasp/relaxation contract exactly once.  All
outputs remain diagnostic-only and are checkpointed after every attempted
candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "deximit_original_sapien_screen_v10_joint_force_metrics_diagnostic_only"
TRACE_SCHEMA = "deximit_sapien_hand_trace_v8_joint_force_metrics_diagnostic_only"
CANDIDATE_SCHEMA = "xhand_bodex_grasp_candidates_v2_diagnostic_only"
TABLE_HEIGHT_M = 0.714
LOAD_BEARING_IMPULSE_EPS_NS = 1.0e-10
CONTACT_CHANNELS = (
    "table", "palm", "thumb", "index", "mid", "ring", "pinky", "other",
)
CONTACT_LINKS = (
    "table", "right_hand_link",
    "right_hand_thumb_bend_link", "right_hand_thumb_rota_link1",
    "right_hand_thumb_rota_link2", "right_hand_index_bend_link",
    "right_hand_index_rota_link1", "right_hand_index_rota_link2",
    "right_hand_mid_link1", "right_hand_mid_link2",
    "right_hand_ring_link1", "right_hand_ring_link2",
    "right_hand_pinky_link1", "right_hand_pinky_link2", "other",
)
CANDIDATE_POSE_POLICIES = ("strict", "legacy")
CANDIDATE_POSE_POSITION_TOLERANCE_M = 2.0e-5
CANDIDATE_POSE_ROTATION_TOLERANCE_RAD = 2.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deximit-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--human-reference", type=Path, required=True)
    parser.add_argument("--deximit-prompt", type=Path, required=True)
    parser.add_argument("--contact-v3", type=Path, required=True)
    parser.add_argument(
        "--manual-label", type=Path,
        help=(
            "Diagnostic-only, human-reviewed pregrasp/grasp/motion rows. When set, "
            "these rows replace the contact-v3-derived anchor and motion interval."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--render-dir", type=Path,
        help=(
            "Save the original DexImit SAPIEN replay for one explicitly selected "
            "candidate. Diagnostic-only; it uses BaseEnv's fixed Front third-person "
            "camera and never enables a following camera."
        ),
    )
    parser.add_argument(
        "--render-mode", choices=("rt", "raster"), default="rt",
        help=(
            "SAPIEN image initialization backend. 'rt' is DexImit's default "
            "ray-traced view; 'raster' keeps identical physics/camera settings but "
            "also supports headless screening on machines with a broken ray-tracing "
            "denoiser."
        ),
    )
    parser.add_argument(
        "--pool-depth", type=int, choices=(0, 1, 2, 3),
        help="Run one BODex depth exactly as DexImit's per-depth top-120 contract.",
    )
    parser.add_argument("--max-rollout", type=int, default=120)
    parser.add_argument("--error-threshold-m", type=float, default=0.02)
    parser.add_argument(
        "--candidate-pose-policy", choices=CANDIDATE_POSE_POLICIES,
        default="strict",
        help=(
            "strict requires every pool to declare the SAPIEN pose used during "
            "BODex generation and to match the post-warmup pose; legacy explicitly "
            "allows old pools without this binding for reproducibility only."
        ),
    )
    parser.add_argument(
        "--motion-mode", choices=("source", "vertical_lift"), default="source",
        help=(
            "source preserves the v3-derived demonstrated segment; vertical_lift is "
            "an attribution-only 50 mm grasp control and cannot enter the dual-sim gate."
        ),
    )
    parser.add_argument("--vertical-lift-m", type=float, default=0.05)
    parser.add_argument(
        "--source-candidate-index", type=int, action="append",
        help=(
            "Attribution controls only: after original top-N ranking, execute only "
            "these source candidate indices."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--export-trajectory", action="store_true",
        help=(
            "For exactly one explicitly selected candidate, save the original "
            "per-physics-step SAPIEN right-hand pose/joints for cross-engine replay."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_worktree_state(root: Path) -> dict[str, object]:
    """Record the DexImit revision and tracked working-tree changes in a run."""
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        check=True, capture_output=True,
    ).stdout
    state_hash = hashlib.sha256(status.encode("utf-8") + b"\0" + diff).hexdigest()
    return {
        "deximit_commit": commit,
        "deximit_worktree_dirty": bool(status),
        "deximit_worktree_status": status.splitlines(),
        "deximit_worktree_state_sha256": state_hash,
    }


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(quaternion_wxyz, scalar_first=True).as_matrix()
    result[:3, 3] = position
    return result


def pose7(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((
        matrix[:3, 3],
        Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True),
    ))


def contact_contract(path: Path) -> tuple[int, int, list[str], str]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        episode_id = str(np.asarray(data["episode_id"]).item())
        states = np.asarray(data["state"], dtype=np.int8)
        hands = [str(value) for value in data["hand_order"]]
        regions = [str(value) for value in data["region_order"]]
    if schema != "taco_mano_surface_contact_v3_conservative_geometric_evidence":
        raise ValueError("SAPIEN screening requires conservative surface-contact v3")
    state = states[:, hands.index("right")]
    fingers = [name for name in regions if name != "palm"]
    opposed = (
        (state[:, regions.index("thumb")] == 1)
        & np.any(state[:, [regions.index(name) for name in fingers if name != "thumb"]] == 1, axis=1)
    )
    rows = np.flatnonzero(opposed)
    if not len(rows):
        raise ValueError("v3 has no right-hand thumb-plus-other contact")
    start = int(rows[0])
    end = start
    while end + 1 < len(opposed) and opposed[end + 1]:
        end += 1
    active = [name for name in fingers if state[start, regions.index(name)] == 1]
    if len(active) < 2:
        raise ValueError("v3 opposed-contact row does not contain two named fingers")
    if not episode_id:
        raise ValueError("v3 contact artifact lacks an episode identity")
    return start, end, active, episode_id


def manual_contract(
    path: Path, *, expected_episode_id: str,
) -> tuple[int, int, int, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "deximit_manual_subactions_v1_diagnostic_only"
        or payload.get("diagnostic_only") is not True
        or payload.get("formal_renderer_3_3_eligible") is not False
        or payload.get("hand") != "right"
        or payload.get("episode_id") != expected_episode_id
    ):
        raise ValueError("manual label violates the isolated diagnostic contract")
    rows = payload.get("rows")
    active = payload.get("active_fingers")
    if not isinstance(rows, dict) or not isinstance(active, list):
        raise ValueError("manual label lacks rows or active_fingers")
    pregrasp = int(rows["pregrasp"])
    grasp = int(rows["grasp"])
    motion = int(rows["motion"])
    if not 0 <= pregrasp < grasp < motion:
        raise ValueError("manual rows must satisfy pregrasp < grasp < motion")
    allowed = {"thumb", "index", "middle", "ring", "pinky"}
    if len(active) < 2 or len(active) != len(set(active)) or not set(active) <= allowed:
        raise ValueError("manual active_fingers must contain distinct finger names")
    return pregrasp, grasp, motion, [str(name) for name in active]


def validate_active_fingers(active: list[str]) -> int:
    """Return the BODex finger count for one validated right-hand contract."""
    allowed = {"thumb", "index", "middle", "ring", "pinky"}
    if (
        len(active) not in (2, 3, 4, 5)
        or len(active) != len(set(active))
        or not set(active) <= allowed
    ):
        raise ValueError(
            "active_fingers must contain 2 to 5 distinct named fingers",
        )
    return len(active)


def deximit_prompt(
    prompt_path: Path, human_path: Path, contact_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(prompt_path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        diagnostic_only = bool(np.asarray(data["diagnostic_only"]).item())
        formal_eligible = bool(np.asarray(data["formal_renderer_3_3_eligible"]).item())
        prompt = np.asarray(data["T_sim_deximit_prompt"], dtype=np.float64)
        prompt_objects = np.asarray(data["T_sim_object_reference"], dtype=np.float64)
        source_human_hash = str(np.asarray(data["source_human_reference_sha256"]).item())
        source_contact = Path(str(np.asarray(data["source_contact_v3"]).item())).resolve()
        source_contact_hash = str(np.asarray(data["source_contact_v3_sha256"]).item())
    if (
        schema != "deximit_taco_mano_prompt_v1_diagnostic_only"
        or not diagnostic_only or formal_eligible
        or source_human_hash != sha256(human_path)
        or source_contact != contact_path
        or source_contact_hash != sha256(contact_path)
    ):
        raise ValueError("DexImit prompt violates its isolated diagnostic contract")
    with np.load(human_path, allow_pickle=False) as data:
        objects = np.asarray(data["T_sim_object_reference"], dtype=np.float64)[:, 0]
    if prompt.shape != objects.shape or not np.allclose(
        prompt_objects, objects, atol=2.0e-7, rtol=0.0,
    ):
        raise ValueError("DexImit prompt and human-reference object frames do not align")
    return prompt, objects


def normalized_pose7(value: Any, *, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{label} must contain seven finite values")
    quaternion_norm = float(np.linalg.norm(pose[3:]))
    if not 0.5 < quaternion_norm < 1.5:
        raise ValueError(f"{label} quaternion has an invalid norm")
    pose = pose.copy()
    pose[3:] /= quaternion_norm
    return pose


def pose_error(
    expected: np.ndarray, actual: np.ndarray,
) -> tuple[float, float]:
    expected_pose = normalized_pose7(expected, label="expected pose")
    actual_pose = normalized_pose7(actual, label="actual pose")
    position_error = float(np.linalg.norm(expected_pose[:3] - actual_pose[:3]))
    relative_rotation = Rotation.from_quat(
        expected_pose[3:], scalar_first=True,
    ).inv() * Rotation.from_quat(
        actual_pose[3:], scalar_first=True,
    )
    rotation_error = float(relative_rotation.magnitude())
    return position_error, rotation_error


def candidate_pool_pose_metadata(pool: dict[str, object]) -> dict[str, object]:
    pose = pool.get("generation_pose")
    return {
        "path": str(pool["path"]),
        "depth": int(pool["depth"]),
        "generation_pose_protocol": pool.get("generation_pose_protocol"),
        "generation_object_pose_wxyz": (
            None if pose is None else np.asarray(pose, dtype=np.float64).tolist()
        ),
        "deximit_worktree_state_sha256": pool.get(
            "deximit_worktree_state_sha256"
        ),
    }


def validate_candidate_pool_object_poses(
    pools: list[dict[str, object]], object_initial: np.ndarray, *, policy: str,
) -> list[dict[str, object]]:
    """Bind candidate generation to the exact object pose used by this run.

    BODex returns a world-frame grasp, so relocating an identity-pose pool onto
    a rotated asymmetric object is not equivalent to generating that pool in
    the rotated pose.  Legacy artifacts remain usable only when the caller
    opts into the old protocol explicitly.
    """
    if policy not in CANDIDATE_POSE_POLICIES:
        raise ValueError(f"unknown candidate pose policy: {policy}")
    expected = pose7(object_initial)
    metadata = []
    for pool in pools:
        generation_pose = pool.get("generation_pose")
        generation_protocol = pool.get("generation_pose_protocol")
        if policy == "strict" and generation_protocol == "legacy_identity_pose":
            raise ValueError(
                "strict candidate pose policy rejects a legacy identity-pose pool: "
                f"{pool['path']}"
            )
        if generation_pose is None:
            if policy == "strict":
                raise ValueError(
                    "strict candidate pose policy requires generation_object_pose_wxyz: "
                    f"{pool['path']}"
                )
            metadata.append({
                **candidate_pool_pose_metadata(pool),
                "binding": "legacy_unverified",
            })
            continue
        position_error, rotation_error = pose_error(expected, np.asarray(generation_pose))
        if policy == "strict" and (
            position_error > CANDIDATE_POSE_POSITION_TOLERANCE_M
            or rotation_error > CANDIDATE_POSE_ROTATION_TOLERANCE_RAD
        ):
            raise ValueError(
                "candidate pool generation pose does not match the post-warmup "
                f"SAPIEN object pose: {pool['path']} "
                f"(position_error={position_error:g} m, "
                f"rotation_error={rotation_error:g} rad)"
            )
        metadata.append({
            **candidate_pool_pose_metadata(pool),
            "binding": (
                "strict_post_warmup_match" if policy == "strict"
                else "explicit_legacy_override_with_pose_check_bypassed"
            ),
            "position_error_m": position_error,
            "rotation_error_rad": rotation_error,
        })
    return metadata


def load_pool(
    path: Path, expected_fingers: int, *, expected_episode_id: str,
    expected_mesh_sha256: str, candidate_pose_policy: str = "strict",
    expected_deximit_state_sha256: str | None = None,
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        raw_pose = np.asarray(data["raw_hand_pose_object_wxyz"], dtype=np.float64)
        raw_qpos = np.asarray(data["raw_qpos_sapien_order"], dtype=np.float64)
        valid = np.asarray(data["candidate_valid"], dtype=bool)
        archive_episode_id = str(np.asarray(data["episode_id"]).item())
        archive_mesh_sha256 = str(np.asarray(data["source_mesh_sha256"]).item())
        provenance = json.loads(str(np.asarray(data["provenance_json"]).item()))
        archive_generation_pose = (
            np.asarray(data["generation_object_pose_wxyz"], dtype=np.float64)
            if "generation_object_pose_wxyz" in data.files else None
        )
    if candidate_pose_policy not in CANDIDATE_POSE_POLICIES:
        raise ValueError(f"unknown candidate pose policy: {candidate_pose_policy}")
    pool_deximit_state = provenance.get("deximit_worktree_state_sha256")
    if candidate_pose_policy == "strict" and expected_deximit_state_sha256 is not None:
        if pool_deximit_state != expected_deximit_state_sha256:
            raise ValueError(
                "strict candidate pose policy requires a candidate pool generated "
                "with the current DexImit worktree state: "
                f"{path}"
            )
    provenance_generation_pose = provenance.get("generation_object_pose_wxyz")
    if archive_generation_pose is not None and provenance_generation_pose is not None:
        archive_pose = normalized_pose7(
            archive_generation_pose, label="candidate pool generation pose"
        )
        provenance_pose = normalized_pose7(
            provenance_generation_pose, label="candidate provenance generation pose"
        )
        position_error, rotation_error = pose_error(archive_pose, provenance_pose)
        if (
            position_error > 1.0e-7
            or rotation_error > 1.0e-7
        ):
            raise ValueError(
                "candidate pool generation pose disagrees with its provenance: "
                f"{path}"
            )
    generation_pose = (
        normalized_pose7(
            archive_generation_pose if archive_generation_pose is not None
            else provenance_generation_pose,
            label="candidate pool generation pose",
        )
        if archive_generation_pose is not None or provenance_generation_pose is not None
        else None
    )
    if (
        schema != CANDIDATE_SCHEMA
        or not bool(provenance.get("diagnostic_only"))
        or bool(provenance.get("formal_renderer_3_3_eligible"))
        or int(provenance.get("fingers", -1)) != expected_fingers
        or str(provenance.get("episode_id", "")) != expected_episode_id
        or str(provenance.get("source_mesh_sha256", "")) != expected_mesh_sha256
        or archive_episode_id != expected_episode_id
        or archive_mesh_sha256 != expected_mesh_sha256
        or valid.ndim != 1
        or raw_pose.shape != (len(valid), 3, 7)
        or raw_qpos.shape != (len(valid), 3, 12)
    ):
        raise ValueError(f"candidate pool violates the diagnostic contract: {path}")
    valid_rows = np.flatnonzero(valid)
    if (
        not np.isfinite(raw_pose[valid_rows]).all()
        or not np.isfinite(raw_qpos[valid_rows]).all()
        or not np.isfinite(np.linalg.norm(raw_pose[valid_rows, :, 3:], axis=-1)).all()
        or np.any(
            (np.linalg.norm(raw_pose[valid_rows, :, 3:], axis=-1) <= 0.5)
            | (np.linalg.norm(raw_pose[valid_rows, :, 3:], axis=-1) >= 1.5)
        )
    ):
        raise ValueError(f"candidate pool contains non-finite or invalid valid rows: {path}")
    if candidate_pose_policy == "strict" and generation_pose is None:
        raise ValueError(
            "strict candidate pose policy requires generation_object_pose_wxyz: "
            f"{path}"
        )
    return {
        "path": path,
        "sha256": sha256(path),
        "depth": int(provenance["grasp_depth"]),
        "pose": raw_pose,
        "qpos": raw_qpos,
        "valid": valid,
        "generation_pose": generation_pose,
        "generation_pose_protocol": provenance.get("generation_pose_protocol"),
        "deximit_worktree_state_sha256": pool_deximit_state,
    }


def world_candidates(pool: dict[str, object], object_pose: np.ndarray) -> np.ndarray:
    poses = np.asarray(pool["pose"], dtype=np.float64)
    qpos = np.asarray(pool["qpos"], dtype=np.float64)
    output = np.zeros((len(poses), 1, 3, 19), dtype=np.float32)
    for candidate in range(len(poses)):
        for stage in range(3):
            local = pose_matrix(poses[candidate, stage, :3], poses[candidate, stage, 3:])
            output[candidate, 0, stage, :7] = pose7(object_pose @ local)
            output[candidate, 0, stage, 7:] = qpos[candidate, stage]
    return output


def update_locked_bases(
    env: object, ik_solver: list[object], motion_gen: list[object],
    asset_root: Path, config: dict[str, object], load_yaml: object,
    RobotConfig: object,
) -> None:
    for hand_index, hand in enumerate(("left", "right")):
        matrix = env.robot_left_transformation if hand == "left" else env.robot_right_transformation
        rotation = Rotation.from_matrix(matrix[:3, :3]).as_euler("XYZ", degrees=False)
        offset = {
            "pos_x_joint": float(matrix[0, 3]),
            "pos_y_joint": float(matrix[1, 3]),
            "pos_z_joint": float(matrix[2, 3]),
            "rot_x_joint": float(rotation[0]),
            "rot_y_joint": float(rotation[1]),
            "rot_z_joint": float(rotation[2]),
        }
        entry = config["robot"][f"ur5e_with_{hand}_hand"]
        ik_dict = load_yaml(str(asset_root / entry["curobo_ik_solver_config_path"]))["robot_cfg"]
        ik_dict["kinematics"]["lock_joints"] = offset
        ik_cfg = RobotConfig.from_dict(ik_dict, ik_solver[hand_index].tensor_args)
        ik_solver[hand_index].kinematics.update_kinematics_config(
            ik_cfg.kinematics.kinematics_config,
        )
        motion_dict = load_yaml(str(asset_root / entry["curobo_motion_gen_config_path"]))["robot_cfg"]
        motion_dict["kinematics"]["lock_joints"] = offset
        motion_cfg = RobotConfig.from_dict(motion_dict, motion_gen[hand_index].tensor_args)
        motion_gen[hand_index].kinematics.update_kinematics_config(
            motion_cfg.kinematics.kinematics_config,
        )


class PhysicsRecorder:
    """Exact BaseEnv headless stepping plus per-physics-step read-only metrics."""

    def __init__(self, env: object, object_actor: object, right_urdf_path: Path):
        import sapien
        import torch
        import pytorch_kinematics as pk

        self.env = env
        self.object_actor = object_actor
        self.sapien = sapien
        self.torch = torch
        self.object_body = object_actor.find_component_by_type(
            sapien.physx.PhysxRigidBodyComponent,
        )
        hand_links = [
            link for link in env.robot_right.links
            if link.get_name() == "right_hand_link"
        ]
        if len(hand_links) != 1:
            raise ValueError(
                f"expected one SAPIEN right_hand_link, found {len(hand_links)}",
            )
        self.right_hand_link = hand_links[0]
        if not isinstance(
            self.right_hand_link, sapien.physx.PhysxArticulationLinkComponent,
        ):
            raise TypeError("SAPIEN right_hand_link is not an articulation body component")
        self.right_hand_body = self.right_hand_link
        self.right_robot_link_order = tuple(
            link.get_name() for link in env.robot_right.get_links()
        )
        if (
            len(self.right_robot_link_order) != len(set(self.right_robot_link_order))
            or not self.right_robot_link_order
        ):
            raise ValueError("SAPIEN right robot link order is malformed")
        self.pk_right = pk.build_chain_from_urdf(
            right_urdf_path.read_bytes(),
        ).to(dtype=torch.float32)
        sim_joint_names = [joint.get_name() for joint in env.active_joints_right]
        kinematic_joint_names = list(self.pk_right.get_joint_parameter_names())
        if set(sim_joint_names) != set(kinematic_joint_names):
            raise ValueError("SAPIEN and kinematic-chain right joint names differ")
        self.pk_from_sim = np.asarray(
            [sim_joint_names.index(name) for name in kinematic_joint_names],
            dtype=np.int64,
        )
        self.reset()

    def reset(self) -> None:
        self.object_pose: list[list[float]] = []
        self.right_hand_pose: list[list[float]] = []
        self.right_robot_qpos: list[list[float]] = []
        self.right_robot_qvel: list[list[float]] = []
        self.right_robot_qacc: list[list[float]] = []
        self.right_robot_qf: list[list[float]] = []
        self.right_robot_link_incoming_joint_force: list[np.ndarray] = []
        self.right_hand_linear_velocity: list[list[float]] = []
        self.right_hand_angular_velocity: list[list[float]] = []
        self.object_linear_velocity: list[list[float]] = []
        self.object_angular_velocity: list[list[float]] = []
        self.right_drive_target: list[list[float]] = []
        self.right_hand_drive_pose: list[list[float]] = []
        self.physics_sample_time_s: list[float] = []
        self.contact_detected: list[np.ndarray] = []
        self.contact_load_bearing: list[np.ndarray] = []
        self.contact_point_count: list[np.ndarray] = []
        self.contact_min_separation: list[np.ndarray] = []
        self.contact_impulse_norm_sum: list[np.ndarray] = []
        self.contact_impulse_net_on_object: list[np.ndarray] = []
        self.contact_normal_impulse_net_on_object: list[np.ndarray] = []
        self.contact_tangent_impulse_net_on_object: list[np.ndarray] = []
        self.contact_position_mean: list[np.ndarray] = []
        self.contact_point_spread_rms_radius: list[np.ndarray] = []
        self.contact_patch_count: list[np.ndarray] = []
        self.contact_patch_rms_radius_mean: list[np.ndarray] = []
        self.contact_patch_rms_radius_max: list[np.ndarray] = []
        self.contact_normal_toward_object_mean: list[np.ndarray] = []
        self.link_contact_detected: list[np.ndarray] = []
        self.link_contact_load_bearing: list[np.ndarray] = []
        self.link_contact_point_count: list[np.ndarray] = []
        self.link_contact_min_separation: list[np.ndarray] = []
        self.link_contact_impulse_net_on_object: list[np.ndarray] = []
        self.link_contact_position_mean: list[np.ndarray] = []
        self.link_contact_normal_toward_object_mean: list[np.ndarray] = []
        self.current_right_drive_target: np.ndarray | None = None
        self.current_right_hand_drive_pose: np.ndarray | None = None
        self.object_contact_steps = 0
        self.right_hand_contact_steps = 0
        self.object_detected_contact_steps = 0
        self.right_hand_detected_contact_steps = 0
        self.minimum_object_separation_m = np.inf
        self.minimum_right_hand_separation_m = np.inf
        self.maximum_object_impulse_ns = 0.0
        self.maximum_right_hand_impulse_ns = 0.0
        self.physics_steps = 0

    @staticmethod
    def body_name(body: object) -> str:
        entity = getattr(body, "entity", None)
        return str(getattr(entity, "name", ""))

    @staticmethod
    def contact_channel(name: str) -> int:
        if name == "table":
            return CONTACT_CHANNELS.index("table")
        if name == "right_hand_link":
            return CONTACT_CHANNELS.index("palm")
        for channel, token in (
            ("thumb", "thumb"), ("index", "index"), ("mid", "mid"),
            ("ring", "ring"), ("pinky", "pinky"),
        ):
            if name.startswith("right_hand_") and token in name:
                return CONTACT_CHANNELS.index(channel)
        return CONTACT_CHANNELS.index("other")

    def capture(self) -> None:
        pose = self.object_actor.get_pose()
        self.object_pose.append(np.concatenate((pose.p, pose.q)).astype(float).tolist())
        # SAPIEN 3 deprecates component.get_pose because its frame is
        # ambiguous.  An articulation link's exported world pose is the
        # owning entity pose, so request that API explicitly.
        hand_pose = self.right_hand_link.get_entity_pose()
        self.right_hand_pose.append(
            np.concatenate((hand_pose.p, hand_pose.q)).astype(float).tolist(),
        )
        self.right_robot_qpos.append(
            np.asarray(self.env.robot_right.get_qpos(), dtype=np.float64).tolist(),
        )
        self.right_robot_qvel.append(
            np.asarray(self.env.robot_right.get_qvel(), dtype=np.float64).tolist(),
        )
        self.right_robot_qacc.append(
            np.asarray(self.env.robot_right.get_qacc(), dtype=np.float64).tolist(),
        )
        self.right_robot_qf.append(
            np.asarray(self.env.robot_right.get_qf(), dtype=np.float64).tolist(),
        )
        incoming_joint_force = np.asarray(
            self.env.robot_right.get_link_incoming_joint_forces(),
            dtype=np.float64,
        )
        if incoming_joint_force.shape != (len(self.right_robot_link_order), 6):
            raise RuntimeError("SAPIEN incoming joint force shape changed")
        self.right_robot_link_incoming_joint_force.append(incoming_joint_force)
        self.right_hand_linear_velocity.append(
            np.asarray(self.right_hand_body.get_linear_velocity(), dtype=np.float64).tolist(),
        )
        self.right_hand_angular_velocity.append(
            np.asarray(self.right_hand_body.get_angular_velocity(), dtype=np.float64).tolist(),
        )
        self.object_linear_velocity.append(
            np.asarray(self.object_body.get_linear_velocity(), dtype=np.float64).tolist(),
        )
        self.object_angular_velocity.append(
            np.asarray(self.object_body.get_angular_velocity(), dtype=np.float64).tolist(),
        )
        if self.current_right_drive_target is None:
            raise RuntimeError("physics capture has no current right-hand drive target")
        self.right_drive_target.append(self.current_right_drive_target.tolist())
        if self.current_right_hand_drive_pose is None:
            raise RuntimeError("physics capture has no current right-hand drive pose")
        self.right_hand_drive_pose.append(self.current_right_hand_drive_pose.tolist())
        self.physics_sample_time_s.append(
            float((self.physics_steps + 1) * self.env.timestep),
        )
        object_detected = False
        right_detected = False
        object_load_bearing = False
        right_load_bearing = False
        channel_count = len(CONTACT_CHANNELS)
        detected = np.zeros(channel_count, dtype=bool)
        load_bearing = np.zeros(channel_count, dtype=bool)
        point_count = np.zeros(channel_count, dtype=np.int32)
        min_separation = np.full(channel_count, np.nan, dtype=np.float64)
        impulse_norm_sum = np.zeros(channel_count, dtype=np.float64)
        impulse_net = np.zeros((channel_count, 3), dtype=np.float64)
        normal_impulse_net = np.zeros((channel_count, 3), dtype=np.float64)
        tangent_impulse_net = np.zeros((channel_count, 3), dtype=np.float64)
        position_sum = np.zeros((channel_count, 3), dtype=np.float64)
        position_squared_norm_sum = np.zeros(channel_count, dtype=np.float64)
        patch_radii: list[list[float]] = [[] for _ in range(channel_count)]
        normal_sum = np.zeros((channel_count, 3), dtype=np.float64)
        link_count = len(CONTACT_LINKS)
        link_detected = np.zeros(link_count, dtype=bool)
        link_load_bearing = np.zeros(link_count, dtype=bool)
        link_point_count = np.zeros(link_count, dtype=np.int32)
        link_min_separation = np.full(link_count, np.nan, dtype=np.float64)
        link_impulse_net = np.zeros((link_count, 3), dtype=np.float64)
        link_position_sum = np.zeros((link_count, 3), dtype=np.float64)
        link_normal_sum = np.zeros((link_count, 3), dtype=np.float64)
        object_center = np.asarray(pose.p, dtype=np.float64)
        for contact in self.env.scene.get_contacts():
            if self.object_body not in contact.bodies:
                continue
            object_detected = True
            names = [self.body_name(body) for body in contact.bodies]
            is_right_hand = any(name.startswith("right_hand_") for name in names)
            right_detected |= is_right_hand
            counterpart_names = [
                name for body, name in zip(contact.bodies, names)
                if body != self.object_body
            ]
            if len(counterpart_names) != 1:
                raise RuntimeError("object contact does not have one counterpart body")
            channel = self.contact_channel(counterpart_names[0])
            detected[channel] = True
            counterpart = counterpart_names[0]
            link = CONTACT_LINKS.index(
                counterpart if counterpart in CONTACT_LINKS else "other",
            )
            link_detected[link] = True
            points = tuple(contact.points)
            if points:
                patch_positions = np.asarray(
                    [point.position for point in points], dtype=np.float64,
                )
                patch_center = np.mean(patch_positions, axis=0)
                patch_radii[channel].append(float(np.sqrt(np.mean(np.sum(
                    np.square(patch_positions - patch_center), axis=1,
                )))))
            for point in points:
                separation = float(point.separation)
                position = np.asarray(point.position, dtype=np.float64)
                normal = np.asarray(point.normal, dtype=np.float64)
                impulse_vector = np.asarray(point.impulse, dtype=np.float64)
                toward_object = object_center - position
                if float(normal @ toward_object) < 0.0:
                    normal = -normal
                if float(impulse_vector @ toward_object) < 0.0:
                    impulse_vector = -impulse_vector
                impulse = float(np.linalg.norm(impulse_vector))
                normal_impulse = max(0.0, float(impulse_vector @ normal)) * normal
                tangent_impulse = impulse_vector - normal_impulse
                point_count[channel] += 1
                position_sum[channel] += position
                position_squared_norm_sum[channel] += float(position @ position)
                normal_sum[channel] += normal
                impulse_norm_sum[channel] += impulse
                impulse_net[channel] += impulse_vector
                normal_impulse_net[channel] += normal_impulse
                tangent_impulse_net[channel] += tangent_impulse
                link_point_count[link] += 1
                link_position_sum[link] += position
                link_normal_sum[link] += normal
                link_impulse_net[link] += impulse_vector
                link_load_bearing[link] |= impulse > LOAD_BEARING_IMPULSE_EPS_NS
                if np.isnan(link_min_separation[link]):
                    link_min_separation[link] = separation
                else:
                    link_min_separation[link] = min(
                        link_min_separation[link], separation,
                    )
                if np.isnan(min_separation[channel]):
                    min_separation[channel] = separation
                else:
                    min_separation[channel] = min(
                        min_separation[channel], separation,
                    )
                load_bearing[channel] |= impulse > LOAD_BEARING_IMPULSE_EPS_NS
                self.minimum_object_separation_m = min(
                    self.minimum_object_separation_m, separation,
                )
                self.maximum_object_impulse_ns = max(
                    self.maximum_object_impulse_ns, impulse,
                )
                if impulse > LOAD_BEARING_IMPULSE_EPS_NS:
                    object_load_bearing = True
                if is_right_hand:
                    self.minimum_right_hand_separation_m = min(
                        self.minimum_right_hand_separation_m, separation,
                    )
                    self.maximum_right_hand_impulse_ns = max(
                        self.maximum_right_hand_impulse_ns, impulse,
                    )
                    if impulse > LOAD_BEARING_IMPULSE_EPS_NS:
                        right_load_bearing = True
        self.object_detected_contact_steps += int(object_detected)
        self.right_hand_detected_contact_steps += int(right_detected)
        self.object_contact_steps += int(object_load_bearing)
        self.right_hand_contact_steps += int(right_load_bearing)
        position_mean = np.full((channel_count, 3), np.nan, dtype=np.float64)
        point_spread_rms_radius = np.full(channel_count, np.nan, dtype=np.float64)
        patch_count = np.asarray(
            [len(values) for values in patch_radii], dtype=np.int32,
        )
        patch_rms_radius_mean = np.full(channel_count, np.nan, dtype=np.float64)
        patch_rms_radius_max = np.full(channel_count, np.nan, dtype=np.float64)
        normal_mean = np.full((channel_count, 3), np.nan, dtype=np.float64)
        populated = point_count > 0
        position_mean[populated] = position_sum[populated] / point_count[populated, None]
        mean_squared_radius = (
            position_squared_norm_sum[populated] / point_count[populated]
            - np.sum(np.square(position_mean[populated]), axis=1)
        )
        point_spread_rms_radius[populated] = np.sqrt(
            np.maximum(mean_squared_radius, 0.0),
        )
        for channel, radii in enumerate(patch_radii):
            if radii:
                patch_rms_radius_mean[channel] = float(np.mean(radii))
                patch_rms_radius_max[channel] = float(np.max(radii))
        normal_mean[populated] = normal_sum[populated] / point_count[populated, None]
        normal_norm = np.linalg.norm(normal_mean[populated], axis=1)
        nonzero = normal_norm > np.finfo(np.float64).eps
        populated_indices = np.flatnonzero(populated)
        normal_mean[populated_indices[nonzero]] /= normal_norm[nonzero, None]
        link_populated = link_point_count > 0
        link_position_mean = np.full((link_count, 3), np.nan, dtype=np.float64)
        link_normal_mean = np.full((link_count, 3), np.nan, dtype=np.float64)
        link_position_mean[link_populated] = (
            link_position_sum[link_populated]
            / link_point_count[link_populated, None]
        )
        link_normal_mean[link_populated] = (
            link_normal_sum[link_populated]
            / link_point_count[link_populated, None]
        )
        link_normal_norm = np.linalg.norm(link_normal_mean[link_populated], axis=1)
        link_nonzero = link_normal_norm > np.finfo(np.float64).eps
        link_indices = np.flatnonzero(link_populated)
        link_normal_mean[link_indices[link_nonzero]] /= (
            link_normal_norm[link_nonzero, None]
        )
        self.contact_detected.append(detected)
        self.contact_load_bearing.append(load_bearing)
        self.contact_point_count.append(point_count)
        self.contact_min_separation.append(min_separation)
        self.contact_impulse_norm_sum.append(impulse_norm_sum)
        self.contact_impulse_net_on_object.append(impulse_net)
        self.contact_normal_impulse_net_on_object.append(normal_impulse_net)
        self.contact_tangent_impulse_net_on_object.append(tangent_impulse_net)
        self.contact_position_mean.append(position_mean)
        self.contact_point_spread_rms_radius.append(point_spread_rms_radius)
        self.contact_patch_count.append(patch_count)
        self.contact_patch_rms_radius_mean.append(patch_rms_radius_mean)
        self.contact_patch_rms_radius_max.append(patch_rms_radius_max)
        self.contact_normal_toward_object_mean.append(normal_mean)
        self.link_contact_detected.append(link_detected)
        self.link_contact_load_bearing.append(link_load_bearing)
        self.link_contact_point_count.append(link_point_count)
        self.link_contact_min_separation.append(link_min_separation)
        self.link_contact_impulse_net_on_object.append(link_impulse_net)
        self.link_contact_position_mean.append(link_position_mean)
        self.link_contact_normal_toward_object_mean.append(link_normal_mean)
        self.physics_steps += 1

    def step_headless(self, action: np.ndarray, disable_gravity: bool = False) -> dict[str, object]:
        self.set_right_drive_target(action)
        self.env.apply_action(action)
        for actor in self.env.object_list:
            body = actor.find_component_by_type(self.sapien.physx.PhysxRigidBodyComponent)
            body.disable_gravity = disable_gravity
        for _ in range(self.env.frame_skip):
            self.env.scene.step()
            self.capture()
        return {}

    def step_rendered(
        self, action: np.ndarray, get_obs: bool = True, disable_gravity: bool = False,
    ) -> dict[str, object]:
        """Mirror BaseEnv.step and retain the existing read-only physics audit."""
        self.set_right_drive_target(action)
        self.env.apply_action(action)
        for actor in self.env.object_list:
            body = actor.find_component_by_type(self.sapien.physx.PhysxRigidBodyComponent)
            body.disable_gravity = disable_gravity
        for _ in range(self.env.frame_skip):
            self.env.scene.step()
            self.capture()
        self.env.scene.update_render()
        return self.env.get_obs() if get_obs else {}

    def set_right_drive_target(self, action: np.ndarray) -> None:
        target = np.asarray(action[-18:], dtype=np.float64).copy()
        ordered = self.torch.as_tensor(
            target[self.pk_from_sim], dtype=self.torch.float32,
        )
        transform = (
            self.pk_right.forward_kinematics(ordered)["right_hand_link"]
            .get_matrix().detach().cpu().numpy().squeeze()
        )
        world = np.asarray(self.env.robot_right_transformation) @ transform
        self.current_right_drive_target = target
        self.current_right_hand_drive_pose = pose7(world)

    def report(self, initial: np.ndarray, relative_target: np.ndarray, vertices: np.ndarray) -> dict[str, object]:
        if self.object_pose:
            final = pose_matrix(
                np.asarray(self.object_pose[-1][:3]), np.asarray(self.object_pose[-1][3:]),
            )
            computed = final @ np.linalg.inv(initial)
            target_vertices = vertices @ relative_target[:3, :3].T + relative_target[:3, 3]
            computed_vertices = vertices @ computed[:3, :3].T + computed[:3, 3]
            error = float(np.linalg.norm(target_vertices - computed_vertices, axis=1).mean())
        else:
            final = initial.copy()
            error = None
        cmass_pose = self.object_body.get_cmass_local_pose()
        shapes = list(self.object_body.get_collision_shapes())
        return {
            "physics_steps": self.physics_steps,
            "contact_step_definition": (
                "load-bearing object contact with point impulse greater than "
                f"{LOAD_BEARING_IMPULSE_EPS_NS:g} N*s"
            ),
            "object_contact_steps": self.object_contact_steps,
            "right_hand_contact_steps": self.right_hand_contact_steps,
            "object_detected_contact_steps": self.object_detected_contact_steps,
            "right_hand_detected_contact_steps": self.right_hand_detected_contact_steps,
            "minimum_object_contact_separation_m": (
                None if not np.isfinite(self.minimum_object_separation_m)
                else float(self.minimum_object_separation_m)
            ),
            "minimum_right_hand_contact_separation_m": (
                None if not np.isfinite(self.minimum_right_hand_separation_m)
                else float(self.minimum_right_hand_separation_m)
            ),
            "maximum_object_contact_impulse_ns": self.maximum_object_impulse_ns,
            "maximum_right_hand_contact_impulse_ns": self.maximum_right_hand_impulse_ns,
            "object_mass_kg": float(self.object_body.get_mass()),
            "object_inertia_kg_m2": np.asarray(
                self.object_body.get_inertia(), dtype=np.float64,
            ).tolist(),
            "object_cmass_local_pose_wxyz": np.concatenate((
                np.asarray(cmass_pose.p, dtype=np.float64),
                np.asarray(cmass_pose.q, dtype=np.float64),
            )).tolist(),
            "object_linear_damping": float(self.object_body.get_linear_damping()),
            "object_angular_damping": float(self.object_body.get_angular_damping()),
            "object_collision_shapes": [
                {
                    "contact_offset_m": float(shape.get_contact_offset()),
                    "rest_offset_m": float(shape.get_rest_offset()),
                    "static_friction": float(
                        shape.get_physical_material().get_static_friction()
                    ),
                    "dynamic_friction": float(
                        shape.get_physical_material().get_dynamic_friction()
                    ),
                }
                for shape in shapes
            ],
            "object_initial_pose": initial.tolist(),
            "object_last_sampled_pose": final.tolist(),
            "mean_target_vertex_error_m": error,
        }


class PlanningRecorder:
    """Read-only proxy that preserves which original cuRobo stage failed."""

    STAGES = ("pregrasp", "grasp", "squeeze", "demonstrated_object_motion")

    def __init__(self, planner: object):
        self.planner = planner
        self.reset()

    def reset(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self.planner, name)

    def plan_single(self, *args: object, **kwargs: object) -> object:
        result = self.planner.plan_single(*args, **kwargs)
        success = False
        status = None
        if result is not None:
            value = getattr(result, "success", False)
            if hasattr(value, "item"):
                value = value.item()
            success = bool(value)
            status = str(getattr(result, "status", "")) or None
        trajectory_control_steps = None
        if success:
            trajectory_control_steps = int(
                result.get_interpolated_plan().position.shape[0],
            )
        self.calls.append({
            "stage": self.STAGES[min(len(self.calls), len(self.STAGES) - 1)],
            "success": success,
            "status": status,
            "trajectory_control_steps": trajectory_control_steps,
        })
        return result


def trajectory_export_is_complete(
    planning_calls: list[dict[str, object]],
    physics_steps: int,
    frame_skip: int,
) -> bool:
    """Return whether a failed rollout still has a complete trace to export.

    The original pass gate also checks the object-motion error.  That gate must
    remain independent from trace export: a complete failed rollout is exactly
    the evidence needed to diagnose contact and drive behavior.
    """
    if physics_steps <= 0 or frame_skip <= 0 or len(planning_calls) != 4:
        return False
    if any(
        not bool(call.get("success"))
        or call.get("trajectory_control_steps") is None
        or int(call["trajectory_control_steps"]) <= 0
        for call in planning_calls
    ):
        return False
    planned_steps = sum(
        int(call["trajectory_control_steps"]) * frame_skip
        for call in planning_calls
    )
    return planned_steps <= physics_steps


def main() -> int:
    args = parse_args()
    if (
        args.max_rollout <= 0 or args.error_threshold_m <= 0.0
        or args.vertical_lift_m <= 0.0
    ):
        raise ValueError("rollout count and error threshold must be positive")
    if args.render_dir is not None:
        requested = args.source_candidate_index or []
        if len(set(requested)) != 1:
            raise ValueError(
                "--render-dir requires exactly one --source-candidate-index; "
                "rendering an unselected candidate is forbidden",
            )
    if args.export_trajectory:
        requested = args.source_candidate_index or []
        if len(set(requested)) != 1:
            raise ValueError(
                "--export-trajectory requires exactly one --source-candidate-index",
            )
    deximit = args.deximit_root.resolve(strict=True)
    package_root = deximit / "third_party/any2dex/any2dex"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    deximit_state = git_worktree_state(deximit)
    sys.path.insert(0, str(package_root))
    sys.path.insert(0, str(package_root / "third_party/BODex_api/src"))

    import sapien
    import torch
    import trimesh
    import yaml
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.util_file import load_yaml
    import env.base_env_for_render as render_env
    from env.base_env_for_render import BaseEnv
    from util.curobo_util import setup_curobo_utils
    from util.util import (
        calculate_pose_distance,
        rollout_and_select_grasp,
        sort_grasp_for_single_hand,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("DexImit SAPIEN screening requires CUDA")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    contact_path = args.contact_v3.resolve(strict=True)
    human_path = args.human_reference.resolve(strict=True)
    prompt_path = args.deximit_prompt.resolve(strict=True)
    mesh_path = args.mesh.resolve(strict=True)
    manual_path = args.manual_label.resolve(strict=True) if args.manual_label else None
    contact_anchor, contact_end, contact_active, episode_id = contact_contract(contact_path)
    if manual_path is None:
        pregrasp = None
        anchor, motion_end, active = contact_anchor, contact_end, contact_active
        label_source = "contact_v3_derived"
    else:
        pregrasp, anchor, motion_end, active = manual_contract(
            manual_path, expected_episode_id=episode_id,
        )
        label_source = "human_reviewed_video_and_geometry"
    expected_fingers = validate_active_fingers(active)
    mesh_hash = sha256(mesh_path)
    pools = [
        load_pool(
            path.resolve(strict=True), expected_fingers,
            expected_episode_id=episode_id, expected_mesh_sha256=mesh_hash,
            candidate_pose_policy=args.candidate_pose_policy,
            expected_deximit_state_sha256=str(
                deximit_state["deximit_worktree_state_sha256"]
            ),
        )
        for path in args.candidate
    ]
    depths = sorted(int(pool["depth"]) for pool in pools)
    if args.pool_depth is None:
        if depths != [0, 1, 2, 3]:
            raise ValueError(
                f"combined diagnostic sweep must contain exactly 0,1,2,3; got {depths}",
            )
        selection_scope = "combined four-depth diagnostic ranking"
    else:
        if depths != [args.pool_depth]:
            raise ValueError(
                f"per-depth run requires one depth-{args.pool_depth} pool; got {depths}",
            )
        selection_scope = f"original DexImit per-depth top-{args.max_rollout} ranking"

    prompt, object_reference = deximit_prompt(prompt_path, human_path, contact_path)
    shift = np.eye(4, dtype=np.float64)
    shift[2, 3] = TABLE_HEIGHT_M
    requested_object_world = shift[None] @ object_reference

    output = args.output_dir.resolve()
    config_path = output / "run_config.json"
    checkpoint_path = output / "checkpoint.json"
    summary_path = output / "summary.json"
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=args.resume)
    render_dir = args.render_dir.resolve() if args.render_dir is not None else None
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)
    pass_dir = output / "sapien_pass_targets"
    pass_dir.mkdir(exist_ok=True)
    declared = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "diagnostic_runner": str(Path(__file__).resolve()),
        "diagnostic_runner_sha256": sha256(Path(__file__).resolve()),
        "deximit_root": str(deximit),
        **deximit_state,
        "curobo_version": "0.7.8",
        "sapien_version": sapien.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "candidate_pools": [
            {
                "path": str(pool["path"]), "sha256": pool["sha256"],
                "depth": pool["depth"],
                "generation_pose_protocol": pool["generation_pose_protocol"],
                "has_generation_object_pose": pool["generation_pose"] is not None,
                "deximit_worktree_state_sha256": pool["deximit_worktree_state_sha256"],
            }
            for pool in pools
        ],
        "mesh": str(mesh_path),
        "mesh_sha256": sha256(mesh_path),
        "human_reference": str(human_path),
        "human_reference_sha256": sha256(human_path),
        "deximit_prompt": str(prompt_path),
        "deximit_prompt_sha256": sha256(prompt_path),
        "prompt_source": "released TACO MANO global root rotation in exact DexImit canonical convention",
        "contact_v3": str(contact_path),
        "contact_v3_sha256": sha256(contact_path),
        "label_source": label_source,
        "manual_label": str(manual_path) if manual_path else None,
        "manual_label_sha256": sha256(manual_path) if manual_path else None,
        "manual_pregrasp_row": pregrasp,
        "contact_anchor_row": anchor,
        "motion_end_row": motion_end,
        "active_fingers": active,
        "finger_count": expected_fingers,
        "depth_policy": "v3 does not observe grasp depth; preserve original BODex definition and sweep 0,1,2,3",
        "pool_depth": args.pool_depth,
        "candidate_pose_policy": args.candidate_pose_policy,
        "selection_scope": selection_scope,
        "time_alignment": "preserve grasp-row hand/object relative pose, then relocate both to the resting object",
        "candidate_isolation": "restore robot qpos/qvel/drive targets and object pose/velocities before every rollout",
        "prompt_type": "rotation",
        "max_rollout": args.max_rollout,
        "error_threshold_m": args.error_threshold_m,
        "motion_mode": args.motion_mode,
        "vertical_lift_m": args.vertical_lift_m,
        "dual_sim_gate_eligible": args.motion_mode == "source",
        "requested_source_candidate_indices": args.source_candidate_index,
        "diagnostic_render_dir": str(render_dir) if render_dir else None,
        "render_camera": "fixed Front third-person" if render_dir else None,
        "image_initialization_backend": args.render_mode,
        "render_mode": args.render_mode if render_dir else None,
        "camera_following_allowed": False,
        "seed": args.seed,
        "export_trajectory": args.export_trajectory,
        "trajectory_export_image_backend": (
            "raster initialization only; no camera frames requested"
            if args.export_trajectory and render_dir is None else None
        ),
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != declared:
            raise ValueError("resume configuration differs from the existing SAPIEN run")
    else:
        atomic_json(config_path, declared)

    config_file = package_root / "env/config/env_hand.yaml"
    with config_file.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    original_synthetic_pc = render_env.SyntheticPC
    disable_unused_image_scenes = (
        render_dir is not None or args.export_trajectory or args.render_mode == "raster"
    )
    if disable_unused_image_scenes:
        # BaseEnv builds two SyntheticPC side scenes even when use_synthetic_pc
        # is false.  They are unused by this grasp executor yet overwrite the
        # Front camera's renderer with OIDN, which crashes before the first
        # frame on this machine.  Do not construct those unused side scenes.
        class DisabledSyntheticPC:
            def __init__(self, *unused_args: object, **unused_kwargs: object) -> None:
                pass

        render_env.SyntheticPC = DisabledSyntheticPC
    use_raster_backend = args.render_mode == "raster" or args.export_trajectory
    if use_raster_backend:
        # BaseEnv selects its global shader before it creates the Front camera.
        # Keep the released scene/physics code unchanged and only replace its
        # failing denoiser-dependent image backend for this diagnostic replay.
        original_setup_render = BaseEnv.set_up_physics_and_render

        def setup_raster_render(self: object) -> None:
            # This is the released method with only its four ray-tracing image
            # configuration calls replaced.  Scene physics settings stay exact.
            sapien.physx.set_shape_config(contact_offset=0.02, rest_offset=0.0)
            sapien.physx.set_body_config(
                solver_position_iterations=25,
                solver_velocity_iterations=1,
                sleep_threshold=0.005,
            )
            sapien.physx.set_scene_config(
                gravity=np.array([0.0, 0.0, -9.81]),
                bounce_threshold=2.0,
                enable_pcm=True,
                enable_tgs=True,
                enable_ccd=False,
                enable_enhanced_determinism=False,
                enable_friction_every_iteration=True,
                cpu_workers=0,
            )
            sapien.physx.set_default_material(
                static_friction=0.7, dynamic_friction=0.5, restitution=0.0,
            )
            self.scene = sapien.Scene([
                sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(),
            ])
            self.scene.set_timestep(self.timestep)
            sapien.render.set_camera_shader_dir("default")
            sapien.render.set_viewer_shader_dir("default")

        BaseEnv.set_up_physics_and_render = setup_raster_render
        try:
            env = BaseEnv(config)
        finally:
            BaseEnv.set_up_physics_and_render = original_setup_render
            render_env.SyntheticPC = original_synthetic_pc
    else:
        try:
            env = BaseEnv(config)
        finally:
            render_env.SyntheticPC = original_synthetic_pc
    env.init_object_flexible_with_pose(
        str(mesh_path), 1.0, requested_object_world[0], mesh_path.stem,
        force_z_equals_0=False,
        collision_type="convex", static_friction=0.7, dynamic_friction=0.5,
        disable_gravity=False, density=None,
    )
    env.reset()
    object_actor = env.object_list[0]
    actor_pose = object_actor.get_pose()
    object_initial = pose_matrix(np.asarray(actor_pose.p), np.asarray(actor_pose.q))
    candidate_pose_binding = validate_candidate_pool_object_poses(
        pools, object_initial, policy=args.candidate_pose_policy,
    )
    requested_initial = requested_object_world[0]
    settled_position_offset = float(np.linalg.norm(
        object_initial[:3, 3] - requested_initial[:3, 3]
    ))
    settled_rotation_offset = float(Rotation.from_matrix(
        object_initial[:3, :3].T @ requested_initial[:3, :3]
    ).magnitude())
    # The object is physically initialized at its resting pose, while the MANO
    # prompt comes from the contact anchor.  Preserve the *hand relative to
    # object* pose at the anchor and relocate that pair to the resting object.
    # Applying a row-0 world transform directly would mix two source times.
    anchor_to_sapien = object_initial @ np.linalg.inv(object_reference[anchor])
    prompt_world = anchor_to_sapien[None] @ prompt
    if args.motion_mode == "source":
        relative_source = object_reference[motion_end] @ np.linalg.inv(object_reference[anchor])
        relative_target = anchor_to_sapien @ relative_source @ np.linalg.inv(anchor_to_sapien)
    else:
        relative_target = np.eye(4, dtype=np.float64)
        relative_target[2, 3] = args.vertical_lift_m
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("object input is not a single mesh")
    object_vertices_world = np.asarray(mesh.vertices) @ object_initial[:3, :3].T + object_initial[:3, 3]
    object_bottom_m = float(object_vertices_world[:, 2].min())
    if abs(object_bottom_m - TABLE_HEIGHT_M) > 3.0e-3:
        raise ValueError(
            f"reference object bottom is not on the SAPIEN table: {object_bottom_m:g} m",
        )
    robot_initial = [env.robot_left.get_qpos().copy(), env.robot_right.get_qpos().copy()]

    def restore_clean_initial_state() -> None:
        for robot, qpos in zip((env.robot_left, env.robot_right), robot_initial):
            robot.set_qpos(qpos)
            robot.set_qvel(np.zeros_like(robot.get_qvel()))
        env.apply_action(np.concatenate(robot_initial))
        object_actor.set_pose(sapien.Pose(object_initial[:3, 3], pose7(object_initial)[3:]))
        body = object_actor.find_component_by_type(sapien.physx.PhysxRigidBodyComponent)
        body.set_linear_velocity(np.zeros(3))
        body.set_angular_velocity(np.zeros(3))

    _, ik_solver, motion_gen, _ = setup_curobo_utils(
        is_bimanual=True,
        left_motion_gen_config_path=config["robot"]["ur5e_with_left_hand"]["curobo_motion_gen_config_path"],
        right_motion_gen_config_path=config["robot"]["ur5e_with_right_hand"]["curobo_motion_gen_config_path"],
        left_ik_solver_config_path=config["robot"]["ur5e_with_left_hand"]["curobo_ik_solver_config_path"],
        right_ik_solver_config_path=config["robot"]["ur5e_with_right_hand"]["curobo_ik_solver_config_path"],
    )
    update_locked_bases(
        env, ik_solver, motion_gen, package_root / "asset", config, load_yaml, RobotConfig,
    )
    recorded_motion_gen = [PlanningRecorder(planner) for planner in motion_gen]

    rank_rows: list[dict[str, object]] = []
    rank_poses: list[np.ndarray] = []
    candidates_by_pool: dict[str, np.ndarray] = {}
    ik_counts: dict[str, int] = {}
    for pool in pools:
        world = world_candidates(pool, object_initial)
        pool_key = str(pool["path"])
        candidates_by_pool[pool_key] = world
        finite = np.flatnonzero(np.asarray(pool["valid"], dtype=bool))
        goal = Pose.from_batch_list(world[finite, 0, 0, :7].tolist(), ik_solver[1].tensor_args)
        solved = ik_solver[1].solve_batch(goal)
        success = solved.success.squeeze(dim=-1).cpu().numpy().astype(bool, copy=False)
        ik_counts[str(pool["depth"])] = int(success.sum())
        for source_index in finite[success]:
            candidate_pose = world[source_index, 0, 1, :7]
            rank_poses.append(candidate_pose.copy())
            rank_rows.append({
                "pool": pool_key,
                "depth": int(pool["depth"]),
                "source_candidate_index": int(source_index),
            })
    if not rank_rows:
        raise RuntimeError("cuRobo rejected every candidate in all four depth pools")
    rank_pose_array = np.asarray(rank_poses, dtype=np.float64)
    prompt_pose = pose7(prompt_world[anchor])
    original_order = sort_grasp_for_single_hand(
        prompt_pose, rank_pose_array, sort_key="rotation",
    )
    original_errors = calculate_pose_distance(
        prompt_pose, rank_pose_array, sort_key="rotation",
    )
    ranked = []
    for combined_index in original_order[: args.max_rollout]:
        row = dict(rank_rows[int(combined_index)])
        row["rotation_prompt_error_rad"] = float(original_errors[int(combined_index)])
        ranked.append(row)
    for rank, row in enumerate(ranked):
        row["original_rank"] = rank
    if args.source_candidate_index is not None:
        requested_indices = list(dict.fromkeys(args.source_candidate_index))
        requested_set = set(requested_indices)
        available_set = {int(row["source_candidate_index"]) for row in ranked}
        missing = [index for index in requested_indices if index not in available_set]
        if missing:
            raise ValueError(
                f"requested source indices are absent from the original top-N ranking: {missing}",
            )
        ranked = [
            row for row in ranked
            if int(row["source_candidate_index"]) in requested_set
        ]

    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists() else {"attempts": []}
    )
    attempts: list[dict[str, object]] = list(checkpoint.get("attempts", []))
    attempted = {
        (str(row["pool"]), int(row["source_candidate_index"])) for row in attempts
    }
    recorder = PhysicsRecorder(
        env,
        object_actor,
        package_root / "asset" / config["robot"]["ur5e_with_right_hand"]["urdf_path"],
    )
    original_step_headless = env.step_headless
    original_step = env.step
    env.step_headless = recorder.step_headless
    if render_dir is not None:
        env.step = recorder.step_rendered
    started = time.monotonic()
    try:
        for row in ranked:
            key = (str(row["pool"]), int(row["source_candidate_index"]))
            if key in attempted:
                continue
            restore_clean_initial_state()
            recorder.reset()
            for planner in recorded_motion_gen:
                planner.reset()
            candidate = candidates_by_pool[key[0]][key[1] : key[1] + 1]
            passed = False
            error_text = None
            selected = None
            render_path = (
                render_dir / (
                    f"depth_{int(row['depth'])}_rank_{int(row['original_rank']):03d}_"
                    f"candidate_{int(row['source_candidate_index']):03d}.mp4"
                )
                if render_dir is not None else None
            )
            try:
                selected = rollout_and_select_grasp(
                    env=env,
                    data_all=candidate,
                    motion_gen=recorded_motion_gen,
                    relative_transformation=relative_target,
                    obj_verts=object_vertices_world,
                    hand_idx=1,
                    object_idx=0,
                    mano_prompt=None,
                    frame_index=None,
                    render_path=str(render_path) if render_path is not None else None,
                    headless=render_path is None,
                    prompt_type=None,
                    grasp_error_threshold=args.error_threshold_m,
                    squeeze_angle_offset=0.0,
                    pregrasp_distance=0.1,
                    disable_gravity=False,
                    grasp_prompt=None,
                    max_rollout=1,
                )
                passed = True
            except RuntimeError as error:
                error_text = str(error)
            metrics = recorder.report(object_initial, relative_target, object_vertices_world)
            attempt = {
                **row,
                "original_sapien_pass": passed,
                "failure": error_text,
                "right_curobo_stages": recorded_motion_gen[1].calls,
                "metrics": metrics,
            }
            if args.export_trajectory:
                if not trajectory_export_is_complete(
                    recorded_motion_gen[1].calls,
                    recorder.physics_steps,
                    int(env.frame_skip),
                ):
                    raise RuntimeError(
                        "cannot export a cross-engine trace from an incomplete SAPIEN rollout",
                    )
                trajectory_dir = output / "sapien_trajectories"
                trajectory_dir.mkdir(exist_ok=True)
                trajectory_path = trajectory_dir / (
                    f"rank_{int(row['original_rank']):03d}_depth_{int(row['depth'])}_"
                    f"candidate_{int(row['source_candidate_index']):03d}.npz"
                )
                if trajectory_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite SAPIEN trajectory: {trajectory_path}",
                    )
                stage_counts = [
                    int(call["trajectory_control_steps"]) * int(env.frame_skip)
                    for call in recorded_motion_gen[1].calls
                ]
                planned_steps = sum(stage_counts)
                if planned_steps > recorder.physics_steps:
                    raise RuntimeError("recorded cuRobo stages exceed captured physics steps")
                phase = np.concatenate([
                    *(
                        np.full(count, call["stage"], dtype="U32")
                        for call, count in zip(recorded_motion_gen[1].calls, stage_counts)
                    ),
                    np.full(
                        recorder.physics_steps - planned_steps,
                        "hold",
                        dtype="U32",
                    ),
                ])
                arrays = {
                    "object_pose_sapien_wxyz": np.asarray(recorder.object_pose, dtype=np.float64),
                    "right_hand_link_pose_sapien_wxyz": np.asarray(
                        recorder.right_hand_pose, dtype=np.float64,
                    ),
                    "right_robot_qpos_sapien": np.asarray(
                        recorder.right_robot_qpos, dtype=np.float64,
                    ),
                    "right_robot_qvel_sapien": np.asarray(
                        recorder.right_robot_qvel, dtype=np.float64,
                    ),
                    "right_robot_qacc_sapien": np.asarray(
                        recorder.right_robot_qacc, dtype=np.float64,
                    ),
                    "right_robot_qf_sapien": np.asarray(
                        recorder.right_robot_qf, dtype=np.float64,
                    ),
                    "right_robot_link_incoming_joint_force_child_frame": np.asarray(
                        recorder.right_robot_link_incoming_joint_force,
                        dtype=np.float64,
                    ),
                    "right_hand_link_linear_velocity_sapien": np.asarray(
                        recorder.right_hand_linear_velocity, dtype=np.float64,
                    ),
                    "right_hand_link_angular_velocity_sapien": np.asarray(
                        recorder.right_hand_angular_velocity, dtype=np.float64,
                    ),
                    "object_linear_velocity_sapien": np.asarray(
                        recorder.object_linear_velocity, dtype=np.float64,
                    ),
                    "object_angular_velocity_sapien": np.asarray(
                        recorder.object_angular_velocity, dtype=np.float64,
                    ),
                    "right_drive_target_sapien": np.asarray(
                        recorder.right_drive_target, dtype=np.float64,
                    ),
                    "right_hand_link_drive_pose_sapien_wxyz": np.asarray(
                        recorder.right_hand_drive_pose, dtype=np.float64,
                    ),
                    "physics_sample_time_s": np.asarray(
                        recorder.physics_sample_time_s, dtype=np.float64,
                    ),
                }
                contact_arrays = {
                    "contact_channel_detected": np.asarray(
                        recorder.contact_detected, dtype=bool,
                    ),
                    "contact_channel_load_bearing": np.asarray(
                        recorder.contact_load_bearing, dtype=bool,
                    ),
                    "contact_channel_point_count": np.asarray(
                        recorder.contact_point_count, dtype=np.int32,
                    ),
                    "contact_channel_min_separation_m": np.asarray(
                        recorder.contact_min_separation, dtype=np.float64,
                    ),
                    "contact_channel_impulse_norm_sum_ns": np.asarray(
                        recorder.contact_impulse_norm_sum, dtype=np.float64,
                    ),
                    "contact_channel_impulse_net_on_object_ns": np.asarray(
                        recorder.contact_impulse_net_on_object, dtype=np.float64,
                    ),
                    "contact_channel_normal_impulse_net_on_object_ns": np.asarray(
                        recorder.contact_normal_impulse_net_on_object,
                        dtype=np.float64,
                    ),
                    "contact_channel_tangent_impulse_net_on_object_ns": np.asarray(
                        recorder.contact_tangent_impulse_net_on_object,
                        dtype=np.float64,
                    ),
                    "contact_channel_position_mean_m": np.asarray(
                        recorder.contact_position_mean, dtype=np.float64,
                    ),
                    "contact_channel_point_spread_rms_radius_m": np.asarray(
                        recorder.contact_point_spread_rms_radius,
                        dtype=np.float64,
                    ),
                    "contact_channel_patch_count": np.asarray(
                        recorder.contact_patch_count, dtype=np.int32,
                    ),
                    "contact_channel_patch_rms_radius_m_mean": np.asarray(
                        recorder.contact_patch_rms_radius_mean,
                        dtype=np.float64,
                    ),
                    "contact_channel_patch_rms_radius_m_max": np.asarray(
                        recorder.contact_patch_rms_radius_max,
                        dtype=np.float64,
                    ),
                    "contact_channel_normal_toward_object_mean": np.asarray(
                        recorder.contact_normal_toward_object_mean, dtype=np.float64,
                    ),
                    "contact_link_detected": np.asarray(
                        recorder.link_contact_detected, dtype=bool,
                    ),
                    "contact_link_load_bearing": np.asarray(
                        recorder.link_contact_load_bearing, dtype=bool,
                    ),
                    "contact_link_point_count": np.asarray(
                        recorder.link_contact_point_count, dtype=np.int32,
                    ),
                    "contact_link_min_separation_m": np.asarray(
                        recorder.link_contact_min_separation, dtype=np.float64,
                    ),
                    "contact_link_impulse_net_on_object_ns": np.asarray(
                        recorder.link_contact_impulse_net_on_object,
                        dtype=np.float64,
                    ),
                    "contact_link_position_mean_m": np.asarray(
                        recorder.link_contact_position_mean, dtype=np.float64,
                    ),
                    "contact_link_normal_toward_object_mean": np.asarray(
                        recorder.link_contact_normal_toward_object_mean,
                        dtype=np.float64,
                    ),
                }
                expected_shapes = {
                    "object_pose_sapien_wxyz": (recorder.physics_steps, 7),
                    "right_hand_link_pose_sapien_wxyz": (recorder.physics_steps, 7),
                    "right_robot_qpos_sapien": (recorder.physics_steps, 18),
                    "right_robot_qvel_sapien": (recorder.physics_steps, 18),
                    "right_robot_qacc_sapien": (recorder.physics_steps, 18),
                    "right_robot_qf_sapien": (recorder.physics_steps, 18),
                    "right_robot_link_incoming_joint_force_child_frame": (
                        recorder.physics_steps,
                        len(recorder.right_robot_link_order), 6,
                    ),
                    "right_hand_link_linear_velocity_sapien": (recorder.physics_steps, 3),
                    "right_hand_link_angular_velocity_sapien": (recorder.physics_steps, 3),
                    "object_linear_velocity_sapien": (recorder.physics_steps, 3),
                    "object_angular_velocity_sapien": (recorder.physics_steps, 3),
                    "right_drive_target_sapien": (recorder.physics_steps, 18),
                    "right_hand_link_drive_pose_sapien_wxyz": (recorder.physics_steps, 7),
                    "physics_sample_time_s": (recorder.physics_steps,),
                }
                if any(
                    value.shape != expected_shapes[name]
                    or not np.isfinite(value).all()
                    for name, value in arrays.items()
                ):
                    raise RuntimeError("SAPIEN trajectory arrays are not physics-step aligned")
                contact_count = len(CONTACT_CHANNELS)
                contact_points = contact_arrays["contact_channel_point_count"]
                populated = contact_points > 0
                if (
                    contact_arrays["contact_channel_detected"].shape
                    != (recorder.physics_steps, contact_count)
                    or contact_arrays["contact_channel_load_bearing"].shape
                    != (recorder.physics_steps, contact_count)
                    or contact_points.shape != (recorder.physics_steps, contact_count)
                    or contact_arrays["contact_channel_min_separation_m"].shape
                    != (recorder.physics_steps, contact_count)
                    or contact_arrays["contact_channel_impulse_norm_sum_ns"].shape
                    != (recorder.physics_steps, contact_count)
                    or contact_arrays["contact_channel_impulse_net_on_object_ns"].shape
                    != (recorder.physics_steps, contact_count, 3)
                    or contact_arrays[
                        "contact_channel_normal_impulse_net_on_object_ns"
                    ].shape != (recorder.physics_steps, contact_count, 3)
                    or contact_arrays[
                        "contact_channel_tangent_impulse_net_on_object_ns"
                    ].shape != (recorder.physics_steps, contact_count, 3)
                    or contact_arrays["contact_channel_position_mean_m"].shape
                    != (recorder.physics_steps, contact_count, 3)
                    or contact_arrays[
                        "contact_channel_point_spread_rms_radius_m"
                    ].shape != (recorder.physics_steps, contact_count)
                    or contact_arrays["contact_channel_patch_count"].shape
                    != (recorder.physics_steps, contact_count)
                    or contact_arrays[
                        "contact_channel_patch_rms_radius_m_mean"
                    ].shape != (recorder.physics_steps, contact_count)
                    or contact_arrays[
                        "contact_channel_patch_rms_radius_m_max"
                    ].shape != (recorder.physics_steps, contact_count)
                    or contact_arrays["contact_channel_normal_toward_object_mean"].shape
                    != (recorder.physics_steps, contact_count, 3)
                    or not np.array_equal(
                        contact_arrays["contact_channel_detected"], populated,
                    )
                    or not np.isfinite(contact_arrays[
                        "contact_channel_impulse_norm_sum_ns"
                    ]).all()
                    or not np.isfinite(contact_arrays[
                        "contact_channel_impulse_net_on_object_ns"
                    ]).all()
                    or not np.isfinite(contact_arrays[
                        "contact_channel_normal_impulse_net_on_object_ns"
                    ]).all()
                    or not np.isfinite(contact_arrays[
                        "contact_channel_tangent_impulse_net_on_object_ns"
                    ]).all()
                    or not np.allclose(
                        contact_arrays[
                            "contact_channel_normal_impulse_net_on_object_ns"
                        ] + contact_arrays[
                            "contact_channel_tangent_impulse_net_on_object_ns"
                        ],
                        contact_arrays["contact_channel_impulse_net_on_object_ns"],
                        atol=1.0e-10,
                        rtol=1.0e-8,
                    )
                    or not np.isfinite(contact_arrays[
                        "contact_channel_min_separation_m"
                    ][populated]).all()
                    or not np.isnan(contact_arrays[
                        "contact_channel_min_separation_m"
                    ][~populated]).all()
                    or not np.isfinite(contact_arrays[
                        "contact_channel_position_mean_m"
                    ][populated]).all()
                    or not np.isnan(contact_arrays[
                        "contact_channel_position_mean_m"
                    ][~populated]).all()
                    or not np.isfinite(contact_arrays[
                        "contact_channel_point_spread_rms_radius_m"
                    ][populated]).all()
                    or not np.isnan(contact_arrays[
                        "contact_channel_point_spread_rms_radius_m"
                    ][~populated]).all()
                    or not np.array_equal(
                        contact_arrays["contact_channel_patch_count"] > 0,
                        populated,
                    )
                    or not np.isfinite(contact_arrays[
                        "contact_channel_patch_rms_radius_m_mean"
                    ][populated]).all()
                    or not np.isnan(contact_arrays[
                        "contact_channel_patch_rms_radius_m_mean"
                    ][~populated]).all()
                    or not np.isfinite(contact_arrays[
                        "contact_channel_patch_rms_radius_m_max"
                    ][populated]).all()
                    or not np.isnan(contact_arrays[
                        "contact_channel_patch_rms_radius_m_max"
                    ][~populated]).all()
                ):
                    raise RuntimeError("detailed SAPIEN contact arrays are malformed")
                link_count = len(CONTACT_LINKS)
                link_points = contact_arrays["contact_link_point_count"]
                link_populated = link_points > 0
                if (
                    contact_arrays["contact_link_detected"].shape
                    != (recorder.physics_steps, link_count)
                    or contact_arrays["contact_link_load_bearing"].shape
                    != (recorder.physics_steps, link_count)
                    or link_points.shape != (recorder.physics_steps, link_count)
                    or contact_arrays["contact_link_min_separation_m"].shape
                    != (recorder.physics_steps, link_count)
                    or contact_arrays[
                        "contact_link_impulse_net_on_object_ns"
                    ].shape != (recorder.physics_steps, link_count, 3)
                    or contact_arrays["contact_link_position_mean_m"].shape
                    != (recorder.physics_steps, link_count, 3)
                    or contact_arrays[
                        "contact_link_normal_toward_object_mean"
                    ].shape != (recorder.physics_steps, link_count, 3)
                    or not np.array_equal(
                        contact_arrays["contact_link_detected"], link_populated,
                    )
                    or not np.isfinite(contact_arrays[
                        "contact_link_impulse_net_on_object_ns"
                    ]).all()
                    or not np.isfinite(contact_arrays[
                        "contact_link_min_separation_m"
                    ][link_populated]).all()
                    or not np.isnan(contact_arrays[
                        "contact_link_min_separation_m"
                    ][~link_populated]).all()
                    or not np.isfinite(contact_arrays[
                        "contact_link_position_mean_m"
                    ][link_populated]).all()
                    or not np.isnan(contact_arrays[
                        "contact_link_position_mean_m"
                    ][~link_populated]).all()
                    or not np.isfinite(contact_arrays[
                        "contact_link_normal_toward_object_mean"
                    ][link_populated]).all()
                    or not np.isnan(contact_arrays[
                        "contact_link_normal_toward_object_mean"
                    ][~link_populated]).all()
                ):
                    raise RuntimeError("link-resolved SAPIEN contact arrays are malformed")
                if phase.shape != (recorder.physics_steps,) or not np.allclose(
                    arrays["physics_sample_time_s"],
                    (np.arange(recorder.physics_steps, dtype=np.float64) + 1.0)
                    * float(env.timestep),
                    atol=1.0e-10, rtol=0.0,
                ):
                    raise RuntimeError("SAPIEN trajectory phase or sample clock is malformed")
                with trajectory_path.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        schema=np.asarray(TRACE_SCHEMA),
                        diagnostic_only=np.asarray(True),
                        formal_renderer_3_3_eligible=np.asarray(False),
                        sample_semantics=np.asarray(
                            "post-step state under same-index drive target"
                        ),
                        original_sapien_pass=np.asarray(bool(passed)),
                        original_rollout_failure=np.asarray(error_text or ""),
                        candidate_index=np.asarray(int(row["source_candidate_index"])),
                        pool_depth=np.asarray(int(row["depth"])),
                        physics_dt_s=np.asarray(float(env.timestep)),
                        control_frame_skip=np.asarray(int(env.frame_skip)),
                        phase=phase,
                        contact_channel_order=np.asarray(CONTACT_CHANNELS),
                        contact_link_order=np.asarray(CONTACT_LINKS),
                        right_robot_link_order=np.asarray(
                            recorder.right_robot_link_order,
                        ),
                        incoming_joint_force_component_order=np.asarray((
                            "force_x", "force_y", "force_z",
                            "torque_x", "torque_y", "torque_z",
                        )),
                        incoming_joint_force_frame=np.asarray(
                            "child incoming-joint frame",
                        ),
                        **arrays,
                        **contact_arrays,
                    )
                attempt["exported_trajectory"] = str(trajectory_path)
                attempt["exported_trajectory_sha256"] = sha256(trajectory_path)
            if render_path is not None:
                attempt["render_requested_path"] = str(render_path)
                attempt["render_camera"] = "fixed Front third-person"
                attempt["render_mode"] = args.render_mode
                attempt["camera_following_allowed"] = False
                attempt["render_videos"] = [
                    str(path) for path in sorted(
                        render_dir.glob(f"{render_path.stem}grasp_*.mp4"),
                    )
                ]
            if passed:
                assert selected is not None
                target_path = pass_dir / (
                    f"rank_{int(row['original_rank']):03d}_depth_{int(row['depth'])}_"
                    f"candidate_{int(row['source_candidate_index']):03d}.npy"
                )
                np.save(target_path, np.asarray(selected, dtype=np.float32))
                attempt["selected_target"] = str(target_path)
                attempt["selected_target_sha256"] = sha256(target_path)
            attempts.append(attempt)
            attempted.add(key)
            atomic_json(checkpoint_path, {"attempts": attempts})
            print(json.dumps({
                "attempted": len(attempts),
                "total": len(ranked),
                "rank": row["original_rank"],
                "depth": row["depth"],
                "candidate": row["source_candidate_index"],
                "sapien_pass": passed,
                "passes_so_far": sum(bool(value["original_sapien_pass"]) for value in attempts),
            }, ensure_ascii=False), flush=True)
    finally:
        env.step_headless = original_step_headless
        env.step = original_step

    passes = [row for row in attempts if bool(row["original_sapien_pass"])]
    summary = {
        **declared,
        "object_initial_pose_sapien": object_initial.tolist(),
        "candidate_pool_pose_binding": candidate_pose_binding,
        "post_warmup_settled_position_offset_m": settled_position_offset,
        "post_warmup_settled_rotation_offset_rad": settled_rotation_offset,
        "object_bottom_height_m": object_bottom_m,
        "table_height_m": TABLE_HEIGHT_M,
        "anchor_reference_to_sapien": anchor_to_sapien.tolist(),
        "relative_object_motion_anchor_to_end": relative_target.tolist(),
        "ik_success_by_depth": ik_counts,
        "ik_success_total": sum(ik_counts.values()),
        "ranked_after_ik": len(ranked),
        "attempted": len(attempts),
        "original_sapien_pass_count": len(passes),
        "original_sapien_passes": passes,
        "all_attempts": attempts,
        "elapsed_s_this_process": time.monotonic() - started,
        "success_definition": (
            "unchanged DexImit rollout_and_select_grasp: cuRobo plans pregrasp, grasp, "
            "squeeze and then "
            + (
                "the v3-derived demonstrated relative object motion"
                if args.motion_mode == "source"
                else f"an attribution-only vertical {args.vertical_lift_m:g} m lift"
            )
            + "; pass when mean object-vertex motion error is at most "
            f"{args.error_threshold_m:g} m"
        ),
        "limitations": [
            "SAPIEN contact separation and impulse are sampled for audit but do not replace the original pass rule.",
            "Every SAPIEN pass still requires the independent current-MuJoCo 2 mm penetration gate.",
            "No result from this directory is eligible for the formal renderer or 3.3.",
            *(
                [
                    "The saved video is a DexImit/SAPIEN diagnostic replay only; it is not a real|reference|MuJoCo success video.",
                    "The diagnostic renderer uses the fixed Front third-person camera; camera following is forbidden.",
                    *(
                        ["The ray-tracing denoiser crashed on this machine, so this replay uses SAPIEN's raster image backend while preserving released DexImit physics, candidate, and camera pose."]
                        if args.render_mode == "raster" else []
                    ),
                ]
                if render_dir is not None else []
            ),
            *(
                ["The vertical-lift control cannot satisfy or bypass the source-motion SAPIEN gate."]
                if args.motion_mode == "vertical_lift" else []
            ),
        ],
    }
    atomic_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "ik_success_by_depth": ik_counts,
        "attempted": len(attempts),
        "sapien_passes": len(passes),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
