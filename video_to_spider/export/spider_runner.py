"""Run the SPIDER preprocessing/IK/MJWP chain while preserving contact variants."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .spider import DATASET_NAME

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def _resolve_uv(spider_root: Path) -> Path:
    """Find uv without requiring the calling Conda environment to own it."""
    candidates: list[Path] = []
    configured = os.environ.get("UV_EXECUTABLE")
    if configured:
        candidates.append(Path(configured).expanduser())
    discovered = shutil.which("uv")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend([
        spider_root.parent / ".tools/uv/bin/uv",
        spider_root / ".venv/bin/uv",
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "PATH"
    raise FileNotFoundError(f"uv executable not found; searched: {searched}")


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True, check=False
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    record = {
        "command": command, "returncode": result.returncode,
        "runtime_s": float(time.monotonic() - started), "log": str(log_path),
    }
    if result.returncode not in allowed_returncodes:
        raise RuntimeError(f"SPIDER command failed ({result.returncode}); see {log_path}")
    return record


def _contact_difference(visual_path: Path, spider_path: Path) -> dict[str, Any]:
    with np.load(visual_path, allow_pickle=False) as visual, np.load(spider_path, allow_pickle=False) as detected:
        report = {}
        for side in ("right", "left"):
            first = visual[f"contact_{side}"].astype(bool)
            second = detected[f"contact_{side}"].astype(bool)
            common_frames = min(first.shape[0], second.shape[0])
            common_fingers = min(first.shape[1], second.shape[1])
            first_common = first[:common_frames, :common_fingers]
            second_common = second[:common_frames, :common_fingers]
            first_position = visual[f"contact_pos_{side}"][:common_fingers]
            second_position = detected[f"contact_pos_{side}"][:common_fingers]
            report[side] = {
                "visual_shape": list(first.shape), "spider_shape": list(second.shape),
                "visual_contact_rate": float(first.mean()),
                "spider_contact_rate": float(second.mean()),
                "binary_disagreement_rate_common": float(np.not_equal(first_common, second_common).mean()),
                "visual_only_rate_common": float((first_common & ~second_common).mean()),
                "spider_only_rate_common": float((~first_common & second_common).mean()),
                "contact_position_l2_m_common": np.linalg.norm(
                    first_position - second_position, axis=-1
                ).tolist(),
                "shape_mismatch": bool(first.shape != second.shape),
            }
    return report


def _inspect_hand_floor_contacts(scene_paths: list[Path]) -> dict[str, Any]:
    """Verify that SPIDER materialized the requested hand-floor contacts."""
    pair_counts: dict[str, int] = {}
    for scene_path in scene_paths:
        if not scene_path.exists():
            raise RuntimeError(f"SPIDER did not generate expected scene: {scene_path}")
        tree = ET.parse(scene_path)
        pairs = [
            pair for pair in tree.getroot().findall("./contact/pair")
            if pair.get("name", "").startswith("collision_hand_")
            and pair.get("name", "").endswith("_floor")
        ]
        if not pairs:
            raise RuntimeError(f"SPIDER scene has no hand-floor contact pairs: {scene_path}")
        pair_counts[scene_path.name] = len(pairs)
    return {
        "hand_floor_collision_enabled": True,
        "pair_count_by_scene": pair_counts,
    }


def _configure_contact_reward(
    scene_path: Path, task_info_path: Path, embodiment_type: str,
) -> dict[str, Any]:
    """Resolve the hand tracking sites that MJWP uses for contact rewards."""
    sides = {
        "right": ("right",),
        "left": ("left",),
        "bimanual": ("right", "left"),
    }[embodiment_type]
    worldbody = ET.parse(scene_path).getroot().find("./worldbody")
    if worldbody is None:
        raise RuntimeError(f"SPIDER scene has no worldbody: {scene_path}")
    site_names = [
        site.get("name")
        for site in worldbody.iter("site")
    ]
    expected = [
        f"track_hand_{side}_{finger}_tip"
        for side in sides for finger in FINGER_ORDER
    ]
    missing = [name for name in expected if name not in site_names]
    if missing:
        raise RuntimeError(f"SPIDER scene lacks hand contact tracking sites: {missing}")
    site_ids = [site_names.index(name) for name in expected]
    task_info = json.loads(task_info_path.read_text(encoding="utf-8"))
    task_info["contact_site_ids"] = site_ids
    task_info["contact_site_names"] = expected
    task_info_path.write_text(json.dumps(task_info, indent=2) + "\n", encoding="utf-8")
    return {
        "site_ids": site_ids,
        "site_names": expected,
        "task_info": str(task_info_path),
    }


def _contact_position_channels(
    scene_path: Path, embodiment_type: str,
) -> tuple[list[int], list[str]]:
    """Map canonical fingertips to the object-side mocap targets saved by IK."""
    sides = {
        "right": ("right",),
        "left": ("left",),
        "bimanual": ("right", "left"),
    }[embodiment_type]
    mocap_names = [
        body.get("name")
        for body in ET.parse(scene_path).getroot().iter("body")
        if body.get("mocap", "false").lower() in {"true", "1"}
    ]
    expected = [
        f"ref_object_{side}_{finger}_tip"
        for side in sides for finger in FINGER_ORDER
    ]
    missing = [name for name in expected if name not in mocap_names]
    if missing:
        raise RuntimeError(f"SPIDER scene lacks object-side contact mocaps: {missing}")
    return [mocap_names.index(name) for name in expected], expected


def _normalize_kinematic_contact(
    trajectory_path: Path, expected_contacts: int, *,
    contact_position_channels: list[int] | None = None,
) -> dict[str, Any]:
    """Align native IK contact arrays to its filtered qpos timeline."""
    with np.load(trajectory_path, allow_pickle=False) as artifact:
        arrays = {key: np.asarray(artifact[key]) for key in artifact.files}
    if "contact" not in arrays or "contact_pos" not in arrays:
        raise RuntimeError(f"SPIDER IK omitted contact arrays: {trajectory_path}")
    count = len(arrays["qpos"])
    contact = arrays["contact"]
    contact_pos = arrays["contact_pos"]
    if contact.ndim != 2 or contact_pos.ndim != 3 or contact_pos.shape[-1] != 3:
        raise RuntimeError("SPIDER IK contact arrays have unsupported shapes")
    if len(contact) < count or len(contact_pos) < count:
        raise RuntimeError("SPIDER IK contact timeline is shorter than qpos")
    contact_start = (len(contact) - count) // 2
    position_start = (len(contact_pos) - count) // 2
    contact = contact[contact_start : contact_start + count, -expected_contacts:]
    position_channels = (
        list(range(contact_pos.shape[1] - expected_contacts, contact_pos.shape[1]))
        if contact_position_channels is None else list(contact_position_channels)
    )
    if len(position_channels) != expected_contacts or any(
        channel < 0 or channel >= contact_pos.shape[1] for channel in position_channels
    ):
        raise RuntimeError(
            f"invalid contact position channels {position_channels} for shape {contact_pos.shape}"
        )
    contact_pos = contact_pos[
        position_start : position_start + count, position_channels
    ]
    if contact.shape != (count, expected_contacts):
        raise RuntimeError(f"normalized contact has unexpected shape: {contact.shape}")
    if contact_pos.shape != (count, expected_contacts, 3):
        raise RuntimeError(
            f"normalized contact positions have unexpected shape: {contact_pos.shape}"
        )
    arrays["contact"] = contact.astype(np.float32)
    arrays["contact_pos"] = contact_pos.astype(np.float32)
    np.savez_compressed(trajectory_path, **arrays)
    active = contact >= 0.5
    return {
        "frame_count": count,
        "contact_shape": list(contact.shape),
        "contact_pos_shape": list(contact_pos.shape),
        "contact_position_channels": position_channels,
        "contact_rate": float(active.mean()),
        "per_channel_contact_rate": active.mean(axis=0).tolist(),
        "unique_contact_pattern_count": int(np.unique(active, axis=0).shape[0]),
        "all_channels_identical": bool(
            expected_contacts > 1 and np.all(active == active[:, :1])
        ),
        "active_contact_frame_count": int(active.any(axis=1).sum()),
    }


def _replay_metrics(
    trajectory_path: Path, rollout_path: Path, embodiment_type: str, *,
    chunk_steps: int = 20, lookahead_chunks: int = 2,
    position_threshold_m: float = 0.05, rotation_threshold_rad: float = 0.5,
    min_motion_transfer_ratio: float = 0.5, min_reference_motion_m: float = 0.01,
) -> dict[str, Any]:
    """Decide whether deterministic physics Replay is sufficient before MPC.

    Each decision window covers the current chunk plus the requested lookahead.
    The current implementation escalates the whole clip to MPC if any window
    fails; the per-window decisions are retained for a future mixed controller.
    """
    if chunk_steps <= 0 or lookahead_chunks <= 0:
        raise ValueError("chunk_steps and lookahead_chunks must be positive")
    if position_threshold_m <= 0.0 or rotation_threshold_rad <= 0.0:
        raise ValueError("Replay thresholds must be positive")
    if not 0.0 <= min_motion_transfer_ratio <= 1.0:
        raise ValueError("min_motion_transfer_ratio must be in [0, 1]")
    with np.load(trajectory_path, allow_pickle=False) as artifact:
        reference = np.asarray(artifact["qpos"], dtype=np.float64)
    with np.load(rollout_path, allow_pickle=False) as artifact:
        rollout = np.asarray(artifact["qpos"], dtype=np.float64)
    count = min(len(reference), len(rollout))
    if count < 2 or reference.ndim != 2 or rollout.ndim != 2:
        raise RuntimeError("Replay assessment requires two non-empty qpos trajectories")
    if reference.shape[1] != rollout.shape[1] or reference.shape[1] < 7:
        raise RuntimeError("Replay/reference qpos shapes are incompatible")
    # SPIDER stores right_object before left_object. V1 always materializes the
    # real object as right_object, including a bimanual scene.
    object_start = reference.shape[1] - (14 if embodiment_type == "bimanual" else 7)
    reference_object = reference[:count, object_start : object_start + 7]
    rollout_object = rollout[:count, object_start : object_start + 7]
    position_error = np.linalg.norm(
        reference_object[:, :3] - rollout_object[:, :3], axis=1
    )
    reference_quaternion = reference_object[:, 3:]
    rollout_quaternion = rollout_object[:, 3:]
    reference_quaternion /= np.maximum(
        np.linalg.norm(reference_quaternion, axis=1, keepdims=True), 1e-12
    )
    rollout_quaternion /= np.maximum(
        np.linalg.norm(rollout_quaternion, axis=1, keepdims=True), 1e-12
    )
    quaternion_dot = np.abs(np.sum(reference_quaternion * rollout_quaternion, axis=1))
    rotation_error = 2.0 * np.arccos(np.clip(quaternion_dot, -1.0, 1.0))

    windows: list[dict[str, Any]] = []
    decision_span = chunk_steps * lookahead_chunks
    for start in range(0, count, chunk_steps):
        end = min(count, start + decision_span)
        ref_motion = np.linalg.norm(
            reference_object[start:end, :3] - reference_object[start, :3], axis=1
        ).max()
        rollout_motion = np.linalg.norm(
            rollout_object[start:end, :3] - rollout_object[start, :3], axis=1
        ).max()
        transfer = (
            float(rollout_motion / ref_motion)
            if ref_motion >= min_reference_motion_m else None
        )
        position_p95 = float(np.percentile(position_error[start:end], 95))
        rotation_p95 = float(np.percentile(rotation_error[start:end], 95))
        feasible = bool(
            position_p95 <= position_threshold_m
            and rotation_p95 <= rotation_threshold_rad
            and (transfer is None or transfer >= min_motion_transfer_ratio)
        )
        windows.append({
            "start_step": start, "end_step_exclusive": end,
            "position_error_p95_m": position_p95,
            "rotation_error_p95_rad": rotation_p95,
            "reference_peak_motion_m": float(ref_motion),
            "replay_peak_motion_m": float(rollout_motion),
            "motion_transfer_ratio": transfer, "feasible": feasible,
        })
    feasible = bool(all(window["feasible"] for window in windows))
    return {
        "feasible": feasible,
        "decision": "Replay" if feasible else "MPC",
        "policy": "full-clip Replay only when every chunk/lookahead window passes; otherwise full-clip MPC",
        "chunk_steps": chunk_steps, "lookahead_chunks": lookahead_chunks,
        "thresholds": {
            "position_p95_m": position_threshold_m,
            "rotation_p95_rad": rotation_threshold_rad,
            "min_motion_transfer_ratio": min_motion_transfer_ratio,
            "min_reference_motion_m": min_reference_motion_m,
        },
        "position_error_m": {
            "mean": float(position_error.mean()),
            "p95": float(np.percentile(position_error, 95)),
            "max": float(position_error.max()),
            "final": float(position_error[-1]),
        },
        "rotation_error_rad": {
            "mean": float(rotation_error.mean()),
            "p95": float(np.percentile(rotation_error, 95)),
            "max": float(rotation_error.max()),
            "final": float(rotation_error[-1]),
        },
        "windows": windows,
    }


def _mjwp_metrics(path: Path, log_path: Path | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {key: np.asarray(artifact[key]) for key in artifact.files}
    metrics: dict[str, Any] = {"keys": sorted(arrays), "record_count": int(next(iter(arrays.values())).shape[0])}
    for key in ("obj_pos_dist", "obj_quat_dist", "pos_dist", "quat_dist"):
        if key in arrays:
            values = arrays[key].astype(np.float64)
            metrics[key] = {
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)), "max": float(np.max(values)),
            }
    position = metrics.get("obj_pos_dist", metrics.get("pos_dist"))
    rotation = metrics.get("obj_quat_dist", metrics.get("quat_dist"))
    if (position is None or rotation is None) and log_path is not None and log_path.exists():
        match = re.search(
            r"Final object tracking error: pos=([0-9.eE+-]+), quat=([0-9.eE+-]+)",
            log_path.read_text(encoding="utf-8"),
        )
        if match:
            position = {"mean": float(match.group(1)), "source": "SPIDER final log"}
            rotation = {"mean": float(match.group(2)), "source": "SPIDER final log"}
            metrics["obj_pos_dist"] = position
            metrics["obj_quat_dist"] = rotation
    metrics["paper_threshold_success"] = bool(
        position is not None and rotation is not None
        and position["mean"] < 0.1 and rotation["mean"] < 0.5
    )
    metrics["paper_thresholds"] = {"position_m": 0.1, "rotation_rad": 0.5}
    return metrics


def run_spider_chain(
    *, dataset_root: str | Path, spider_root: str | Path, task: str, data_id: int,
    embodiment_type: str, robot_type: str = "xhand", gpu: int = 0,
    run_mjwp: bool = True, save_video: bool = True, ik_end_idx: int = -1,
    mjwp_num_samples: int = 2048, mjwp_iterations: int = 16,
    mjwp_override: str = "gigahand_origin", mjwp_horizon: float = 1.6,
    mjwp_ctrl_dt: float = 0.08, mjwp_knot_dt: float = 0.2,
    contact_reward_scale: float = 0.025, contact_opposition_reward_scale: float = 0.0,
    force_closure_reward_scale: float = 0.010,
    force_closure_penetration_reward_scale: float = 1.0,
    force_closure_min_normal_force_n: float = 0.2,
    force_closure_min_opposition_cosine: float = 0.2,
    force_closure_max_penetration_m: float = 0.003,
    force_closure_minimum_frames: int = 3,
    lift_reward_scale: float = 0.025, sanity_check_seconds: float = 1.0,
    paper_objective: bool = True, action_smoothness_reward_scale: float = 0.8,
    ik_backend: str = "mink", collision_aware_ik: bool = True,
    mink_allow_fidelity_rejected_qref_for_refinement: bool = False,
    mink_collision_projection_max_iterations: int = 80,
    mink_controller_contact_target_policy: str = "qref_fingertip_site",
    mink_scene_collision_constraints: bool = False,
    mink_floor_clearance_m: float = 0.0,
    mink_non_distal_object_clearance_m: float = 0.0,
    mink_distal_object_max_penetration_m: float = 0.0025,
    replay_noise_scale: float = 0.0, ik_seed: int = 0,
    adaptive_mode_switching: bool = True, replay_chunk_steps: int = 20,
    replay_lookahead_chunks: int = 2, replay_position_threshold_m: float = 0.05,
    replay_rotation_threshold_rad: float = 0.5,
    replay_min_motion_transfer_ratio: float = 0.5,
    contact_aware_preshape: bool = False,
    preshape_approach_frames: int = 12,
    preshape_clearance_m: float = 0.03,
    preshape_collision_margin_m: float = 0.001,
    preshape_contact_penetration_m: float = 0.0025,
    preshape_solver_iterations: int = 180,
    preshape_solver_dt: float = 0.01,
) -> Path:
    if ik_backend not in {"mink", "spider-native"}:
        raise ValueError("ik_backend must be 'mink' or 'spider-native'")
    if ik_backend == "mink" and (robot_type != "xhand" or embodiment_type != "right"):
        raise ValueError("the production MINK backend currently supports xhand/right only")
    if mink_floor_clearance_m < 0.0 or mink_non_distal_object_clearance_m < 0.0:
        raise ValueError("MINK floor/non-distal clearances must be nonnegative")
    if mink_distal_object_max_penetration_m < 0.0:
        raise ValueError("MINK distal penetration allowance must be nonnegative")
    contact_auxiliary_scales = {
        "contact": contact_reward_scale,
        "thumb_other_opposition": contact_opposition_reward_scale,
        "force_closure_bonus": force_closure_reward_scale,
        "lift": lift_reward_scale,
    }
    if any(value < 0.0 for value in contact_auxiliary_scales.values()):
        raise ValueError("MPC/RL reward scales must be nonnegative")
    maximum_auxiliary_scale = max(contact_auxiliary_scales.values(), default=0.0)
    # The SPIDER objective weights both object position and human wrist mimic
    # at 1.0.  Keep every contact/grasp bonus strictly below those primaries;
    # penetration remains a separate safety penalty and is not a bonus.
    if maximum_auxiliary_scale >= 1.0:
        raise ValueError(
            "contact/opposition/force-closure/lift rewards must remain below "
            "the 1.0 object-trajectory and human-mimic primary weights; got "
            + json.dumps(contact_auxiliary_scales, sort_keys=True)
        )
    dataset = Path(dataset_root).resolve()
    spider = Path(spider_root).resolve()
    mano_dir = dataset / "processed" / DATASET_NAME / "mano" / embodiment_type / task / str(data_id)
    robot_dir = dataset / "processed" / DATASET_NAME / robot_type / embodiment_type / task / str(data_id)
    keypoints = mano_dir / "trajectory_keypoints.npz"
    if not keypoints.exists():
        raise FileNotFoundError(keypoints)
    logs = robot_dir / "pipeline_logs"
    visual_contact = mano_dir / "trajectory_keypoints_visual_contact.npz"
    detected_contact = mano_dir / "trajectory_keypoints_spider_contact.npz"
    shutil.copy2(keypoints, visual_contact)
    # A shared /tmp cache may belong to another Unix user on a multi-user node.
    # Keep the frozen, no-sync run isolated and make the temporary cache
    # self-cleaning when this function returns or unwinds with an exception.
    uv_cache = tempfile.TemporaryDirectory(prefix=f"video-to-spider-uv-{os.getuid()}-")
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "UV_CACHE_DIR": uv_cache.name,
        "PYTHONPATH": str(spider),
        # MuJoCo otherwise falls back to GLFW/X11 even when SPIDER is invoked
        # with --no-show-viewer. EGL provides deterministic offscreen rendering
        # for the M4 simulation videos on headless GPU nodes.
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
    })
    uv = str(_resolve_uv(spider))
    base = [uv, "run", "--frozen", "--no-sync", "python", "-m"]
    common = [
        "--dataset-dir", str(dataset), "--dataset-name", DATASET_NAME,
        "--embodiment-type", embodiment_type, "--task", task, "--data-id", str(data_id),
    ]
    commands = []
    commands.append(_run(
        base + ["spider.preprocess.decompose_fast"] + common + ["--robot-type", robot_type],
        cwd=spider, env=environment, log_path=logs / "01_decompose_fast.log",
    ))
    try:
        commands.append(_run(
            base + ["spider.preprocess.detect_contact"] + common + ["--no-show-viewer", "--no-save-video"],
            cwd=spider, env=environment, log_path=logs / "02_detect_contact.log",
        ))
        shutil.copy2(keypoints, detected_contact)
        contact_difference = _contact_difference(visual_contact, detected_contact)
    finally:
        # The visual contact is the primary V1 observation. Always restore it,
        # including when SPIDER emits a malformed or shape-mismatched result.
        shutil.copy2(visual_contact, keypoints)
    commands.append(_run(
        base + ["spider.preprocess.generate_xml"] + common
        + ["--robot-type", robot_type, "--hand-floor-collision", "--no-show-viewer"],
        cwd=spider, env=environment, log_path=logs / "03_generate_xml.log",
    ))
    scene = robot_dir.parent / "scene.xml"
    geometry_constraints = _inspect_hand_floor_contacts([
        scene, robot_dir.parent / "scene_eq.xml",
    ])
    if ik_backend == "mink":
        ik_command = (
            base + ["spider.preprocess.mink_ik"] + common
            + ["--robot-type", robot_type, "--end-idx", str(ik_end_idx),
               "--seed", str(ik_seed),
               "--collision-projection-max-iterations",
               str(mink_collision_projection_max_iterations),
               "--controller-contact-target-policy",
               mink_controller_contact_target_policy,
               "--floor-clearance-m", str(mink_floor_clearance_m),
               "--non-distal-object-clearance-m",
               str(mink_non_distal_object_clearance_m),
               "--distal-object-max-penetration-m",
               str(mink_distal_object_max_penetration_m)]
            + (
                ["--no-paper-only-constraints"]
                if mink_scene_collision_constraints
                else ["--paper-only-constraints"]
            )
            + (
                ["--allow-fidelity-rejected-qref-for-refinement"]
                if mink_allow_fidelity_rejected_qref_for_refinement else []
            )
            + (["--save-video"] if save_video else ["--no-save-video"])
        )
        ik_log = logs / "04_mink_ik.log"
    else:
        ik_command = (
            base + ["spider.preprocess.ik"] + common
            + ["--robot-type", robot_type, "--end-idx", str(ik_end_idx),
               "--no-show-viewer", "--no-aggregate-contact",
               "--replay-noise-scale", str(replay_noise_scale),
               "--seed", str(ik_seed)]
            + (["--enable-collision"] if collision_aware_ik else [])
            + (["--save-video"] if save_video else ["--no-save-video"])
        )
        ik_log = logs / "04_spider_native_ik.log"
    commands.append(_run(
        ik_command, cwd=spider, env=environment, log_path=ik_log,
    ))
    trajectory_kinematic = robot_dir / "trajectory_kinematic.npz"
    if not trajectory_kinematic.exists() or not scene.exists():
        raise RuntimeError("SPIDER IK did not produce scene.xml and trajectory_kinematic.npz")
    expected_contacts = 10 if embodiment_type == "bimanual" else 5
    contact_position_channels, contact_position_names = _contact_position_channels(
        scene, embodiment_type,
    )
    contact_artifact = _normalize_kinematic_contact(
        trajectory_kinematic, expected_contacts,
        contact_position_channels=(
            None if ik_backend == "mink" else contact_position_channels
        ),
    )
    contact_artifact["contact_position_names"] = contact_position_names
    task_info_path = robot_dir.parent / "task_info.json"
    contact_reward = _configure_contact_reward(
        scene, task_info_path, embodiment_type,
    )
    rollout_path = robot_dir / "trajectory_ikrollout.npz"
    if not rollout_path.exists():
        raise RuntimeError(f"SPIDER IK did not produce deterministic Replay: {rollout_path}")
    replay_assessment = _replay_metrics(
        trajectory_kinematic, rollout_path, embodiment_type,
        chunk_steps=replay_chunk_steps, lookahead_chunks=replay_lookahead_chunks,
        position_threshold_m=replay_position_threshold_m,
        rotation_threshold_rad=replay_rotation_threshold_rad,
        min_motion_transfer_ratio=replay_min_motion_transfer_ratio,
    )
    preshape_path = robot_dir / "trajectory_grasp_preshape.npz"
    preshape_report_path = robot_dir / "grasp_preshape_report.json"
    preshape_supported = robot_type == "xhand" and embodiment_type == "right"
    preshape_report: dict[str, Any] = {
        "enabled": contact_aware_preshape,
        "supported": preshape_supported,
        "accepted": False,
        "status": "disabled" if not contact_aware_preshape else "unsupported",
    }
    if contact_aware_preshape and preshape_supported:
        preshape_command = base + [
            "spider.preprocess.grasp_preshape",
            "--model-path", str(scene),
            "--trajectory-path", str(trajectory_kinematic),
            "--output-path", str(preshape_path),
            "--report-path", str(preshape_report_path),
            "--embodiment-type", embodiment_type,
            "--approach-frames", str(preshape_approach_frames),
            "--pregrasp-clearance-m", str(preshape_clearance_m),
            "--collision-margin-m", str(preshape_collision_margin_m),
            "--contact-penetration-m", str(preshape_contact_penetration_m),
            "--solver-iterations", str(preshape_solver_iterations),
            "--solver-dt", str(preshape_solver_dt),
        ]
        commands.append(_run(
            preshape_command,
            cwd=spider,
            env=environment,
            log_path=logs / "05_grasp_preshape.log",
            # Exit 2 means the generic search completed but failed its physical
            # acceptance checks.  It is an explicit, audited native-path fallback.
            allowed_returncodes=(0, 2),
        ))
        if not preshape_report_path.exists():
            raise RuntimeError(
                "contact-aware preshape completed without its validation report"
            )
        preshape_report = json.loads(preshape_report_path.read_text(encoding="utf-8"))
        preshape_report["enabled"] = True
        preshape_report["supported"] = True
        preshape_report["status"] = (
            "accepted" if preshape_report.get("accepted") else "rejected"
        )
        if preshape_report["accepted"] and not preshape_path.exists():
            raise RuntimeError(
                "accepted contact-aware preshape has no trajectory artifact"
            )
    preshape_accepted = bool(preshape_report.get("accepted"))
    trajectory_for_control = preshape_path if preshape_accepted else trajectory_kinematic
    mjwp_path = robot_dir / "trajectory_mjwp.npz"
    execute_mpc = bool(
        run_mjwp
        and (
            preshape_accepted
            or not adaptive_mode_switching
            or not replay_assessment["feasible"]
        )
    )
    if execute_mpc:
        mjwp_command = [
            uv, "run", "--frozen", "--no-sync", "python", "examples/run_mjwp.py",
            f"+override={mjwp_override}", f"dataset_dir={dataset}", f"dataset_name={DATASET_NAME}",
            f"robot_type={robot_type}", f"embodiment_type={embodiment_type}", f"task={task}",
            f"data_id={data_id}", f"model_path={scene}", f"data_path={trajectory_for_control}",
            f"output_dir={robot_dir}", "device=cuda:0", "show_viewer=false", "viewer=mujoco",
            f"save_video={'true' if save_video else 'false'}", "save_rerun=false", "save_viser=false",
            f"num_samples={mjwp_num_samples}", f"max_num_iterations={mjwp_iterations}",
            f"horizon={mjwp_horizon}", f"ctrl_dt={mjwp_ctrl_dt}", f"knot_dt={mjwp_knot_dt}",
            # EgoEngine Appendix C.7 uses live physical contact, not SPIDER's
            # upstream fingertip-target tracking reward.
            f"+physical_contact_rew_scale={contact_reward_scale}",
            "+contact_rew_scale=0.0",
            "+contact_opposition_rew_scale=0.0",
            f"+force_closure_rew_scale={force_closure_reward_scale}",
            (
                "+force_closure_penetration_rew_scale="
                f"{force_closure_penetration_reward_scale}"
            ),
            (
                "+force_closure_min_normal_force_n="
                f"{force_closure_min_normal_force_n}"
            ),
            (
                "+force_closure_min_opposition_cosine="
                f"{force_closure_min_opposition_cosine}"
            ),
            (
                "+force_closure_max_penetration_m="
                f"{force_closure_max_penetration_m}"
            ),
            f"+lift_rew_scale={lift_reward_scale}",
            f"+paper_objective={'true' if paper_objective else 'false'}",
            f"+action_smoothness_rew_scale={action_smoothness_reward_scale}",
            f"+sanity_check_seconds={sanity_check_seconds}",
        ]
        commands.append(_run(
            mjwp_command, cwd=spider, env=environment, log_path=logs / "06_mjwp.log"
        ))
        if not mjwp_path.exists():
            raise RuntimeError(f"MJWP completed without expected artifact: {mjwp_path}")
    ik_video = robot_dir / "visualization_ik.mp4"
    mjwp_video = robot_dir / "visualization_mjwp.mp4"
    if save_video and not ik_video.exists():
        raise RuntimeError(f"SPIDER IK completed without expected simulation video: {ik_video}")
    if execute_mpc and save_video and not mjwp_video.exists():
        raise RuntimeError(f"MJWP completed without expected simulation video: {mjwp_video}")
    selected_trajectory = (
        mjwp_path if execute_mpc
        else preshape_path if preshape_accepted
        else rollout_path
    )
    physics_metrics_path = robot_dir / "physics_contact_metrics.json"
    commands.append(_run(
        base + ["spider.postprocess.physics_contact_metrics",
                "--model-path", str(scene),
                "--trajectory-path", str(selected_trajectory),
                "--reference-trajectory-path", str(trajectory_for_control),
                "--ref-dt", "0.02",
                "--embodiment-type", embodiment_type,
                "--min-normal-force-n", str(force_closure_min_normal_force_n),
                "--min-opposition-cosine", str(force_closure_min_opposition_cosine),
                "--max-penetration-m", str(force_closure_max_penetration_m),
                "--minimum-force-closure-frames", str(force_closure_minimum_frames),
                "--output-path", str(physics_metrics_path)],
        cwd=spider, env=environment,
        log_path=logs / "07_physics_contact_metrics.log",
    ))
    physics_metrics = json.loads(physics_metrics_path.read_text(encoding="utf-8"))
    uv_cache.cleanup()
    physical_acceptance = physics_metrics["physical_acceptance"]
    demonstration_success = bool(physical_acceptance["accepted"])
    selected_mode = (
        "MPC (contact-aware preshape handoff)" if execute_mpc and preshape_accepted
        else "MPC" if execute_mpc
        else "Contact-aware preshape; MPC disabled" if preshape_accepted
        else "Replay" if replay_assessment["feasible"]
        else "Replay failed; MPC disabled"
    )
    if not demonstration_success:
        selected_mode += "; rejected by MuJoCo, RL fallback required"
    report = {
        "schema_version": "1.0", "dataset_root": str(dataset), "dataset_name": DATASET_NAME,
        "task": task, "data_id": data_id, "embodiment_type": embodiment_type,
        "robot_type": robot_type, "gpu_physical_index": gpu, "commands": commands,
        "simulation_geometry": geometry_constraints,
        "mode_selection": {
            "selected_mode": selected_mode,
            "adaptive_mode_switching": adaptive_mode_switching,
            "replay": replay_assessment,
            "contact_aware_preshape": preshape_report,
            "rl_fallback": (
                "required_but_not_implemented"
                if not demonstration_success else "not_required"
            ),
            "terminal_status": (
                "accepted" if demonstration_success
                else "failed_no_executable_demonstration"
            ),
        },
        "physical_interaction": physics_metrics,
        "contact_pipeline": {
            "kinematic_artifact": contact_artifact,
            "reward_configuration": contact_reward,
            "contact_reward_scale": contact_reward_scale,
            "contact_opposition_reward_scale": contact_opposition_reward_scale,
            "paper_contact_reward": {
                "equation": "C.7",
                "type": "live MuJoCo binary thumb + any non-thumb contact bonus",
                "upstream_contact_targets_used": False,
                "reward_scale": contact_reward_scale,
            },
            "legacy_spider_contact_point_tracking": {
                "enabled": False,
                "reward_scale": 0.0,
                "reason": "not present in EgoEngine Appendix C.7",
            },
            "force_closure_constraint": {
                "reward_scale": force_closure_reward_scale,
                "penetration_reward_scale": force_closure_penetration_reward_scale,
                "min_normal_force_n": force_closure_min_normal_force_n,
                "min_opposition_cosine": force_closure_min_opposition_cosine,
                "max_penetration_m": force_closure_max_penetration_m,
                "minimum_validation_frames": force_closure_minimum_frames,
                "source": "live MJWarp contact pairs/normals/constraint forces",
                "type": "soft two-sided frictional stability bonus",
                "acceptance_authority": "diagnostic_only",
            },
            "lift_reward_scale": lift_reward_scale,
        },
        "ik_configuration": {
            "backend": ik_backend,
            "paper_equation_1_mink": ik_backend == "mink",
            "fingertip_orientation_source": (
                "calibrated MANO distal-link FK full SO(3)"
                if ik_backend == "mink" else "not consumed by native SPIDER IK"
            ),
            "fidelity_rejected_qref_continuation": (
                mink_allow_fidelity_rejected_qref_for_refinement
                if ik_backend == "mink" else False
            ),
            "collision_projection_max_iterations": (
                mink_collision_projection_max_iterations
                if ik_backend == "mink" else None
            ),
            "controller_contact_target_policy": (
                mink_controller_contact_target_policy
                if ik_backend == "mink" else None
            ),
            "scene_collision_constraints": (
                mink_scene_collision_constraints if ik_backend == "mink" else None
            ),
            "scene_collision_parameters_m": (
                {
                    "floor_clearance": mink_floor_clearance_m,
                    "non_distal_object_clearance": (
                        mink_non_distal_object_clearance_m
                    ),
                    "distal_object_max_penetration": (
                        mink_distal_object_max_penetration_m
                    ),
                }
                if ik_backend == "mink" else None
            ),
            "collision_aware": collision_aware_ik,
            "aggregate_contact": False,
            "replay_noise_scale": replay_noise_scale,
            "seed": ik_seed,
        },
        "mpc_configuration": {
            "executed": execute_mpc, "override": mjwp_override,
            "num_samples": mjwp_num_samples, "iterations": mjwp_iterations,
            "horizon_s": mjwp_horizon, "ctrl_dt_s": mjwp_ctrl_dt,
            "knot_dt_s": mjwp_knot_dt,
            "paper_objective": paper_objective,
            "action_smoothness_reward_scale": action_smoothness_reward_scale,
            "sanity_check_seconds": sanity_check_seconds,
            "input_trajectory": str(trajectory_for_control),
        },
        "artifacts": {
            "visual_contact": str(visual_contact), "spider_detected_contact": str(detected_contact),
            "scene": str(scene), "trajectory_kinematic": str(trajectory_kinematic),
            "trajectory_replay": str(rollout_path),
            "trajectory_grasp_preshape": (
                str(preshape_path) if preshape_accepted else None
            ),
            "grasp_preshape_report": (
                str(preshape_report_path) if preshape_report_path.exists() else None
            ),
            "trajectory_for_control": str(trajectory_for_control),
            "physics_contact_metrics": str(physics_metrics_path),
            "ik_video": str(ik_video) if save_video else None,
            "trajectory_mjwp": str(mjwp_path) if execute_mpc else None,
            # A rendered MPC rollout remains useful for diagnosis even when
            # final MuJoCo acceptance rejects it.  Only final_simulation_video
            # is eligible for downstream delivery as a converted result.
            "diagnostic_mjwp_video": (
                str(mjwp_video) if execute_mpc and save_video else None
            ),
            "final_simulation_video": (
                str(mjwp_video)
                if execute_mpc and save_video and demonstration_success else None
            ),
            "mjwp_video": str(mjwp_video) if execute_mpc and save_video else None,
        },
        "contact_crosscheck": contact_difference,
        "mjwp_metrics": _mjwp_metrics(mjwp_path, logs / "06_mjwp.log") if execute_mpc else None,
        "current_spider_thresholds": {"position_m": 0.1, "rotation_rad": 0.5},
    }
    tracking_success = bool(
        report["mjwp_metrics"] is not None
        and report["mjwp_metrics"]["paper_threshold_success"]
    )
    report["demonstration_success"] = demonstration_success
    report["demonstration_success_criteria"] = {
        "object_tracking_thresholds_passed": bool(
            physics_metrics["object_tracking"]["passed"]
        ),
        "bounded_penetration_passed": bool(
            physics_metrics["contact_quality"]["penetration_passed"]
        ),
        "reasonable_slip_passed": bool(
            physics_metrics["contact_quality"]["slip_passed"]
        ),
        "support_conditioned_force_explanation_passed": bool(
            physics_metrics["support_conditioned_force_explanation"]["passed"]
        ),
        "physical_thumb_opposition_bonus": physics_metrics["grasp_opposition_success"],
        "physical_force_closure_bonus": physics_metrics["force_closure_success"],
        "opposition_and_force_closure_are_hard_gates": False,
        "rl_fallback_required_on_failure": True,
        "rl_fallback_implemented": False,
    }
    report_path = robot_dir / "spider_run_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-id", type=int, required=True)
    parser.add_argument("--embodiment-type", choices=["right", "left", "bimanual"], required=True)
    parser.add_argument("--robot-type", default="xhand")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-mjwp", action="store_true")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--ik-end-idx", type=int, default=-1)
    parser.add_argument("--mjwp-num-samples", type=int, default=2048)
    parser.add_argument("--mjwp-iterations", type=int, default=16)
    parser.add_argument("--mjwp-override", default="gigahand_origin")
    parser.add_argument("--mjwp-horizon", type=float, default=1.6)
    parser.add_argument("--mjwp-ctrl-dt", type=float, default=0.08)
    parser.add_argument("--mjwp-knot-dt", type=float, default=0.2)
    parser.add_argument("--contact-reward-scale", type=float, default=0.025)
    parser.add_argument("--contact-opposition-reward-scale", type=float, default=0.0)
    parser.add_argument("--force-closure-reward-scale", type=float, default=0.010)
    parser.add_argument(
        "--force-closure-penetration-reward-scale", type=float, default=1.0
    )
    parser.add_argument("--force-closure-min-normal-force-n", type=float, default=0.2)
    parser.add_argument(
        "--force-closure-min-opposition-cosine", type=float, default=0.2
    )
    parser.add_argument("--force-closure-max-penetration-m", type=float, default=0.003)
    parser.add_argument("--force-closure-minimum-frames", type=int, default=3)
    parser.add_argument("--lift-reward-scale", type=float, default=0.025)
    parser.add_argument(
        "--action-smoothness-reward-scale", type=float, default=0.8
    )
    parser.add_argument(
        "--no-paper-objective", dest="paper_objective", action="store_false"
    )
    parser.set_defaults(paper_objective=True)
    parser.add_argument("--sanity-check-seconds", type=float, default=1.0)
    parser.add_argument("--no-collision-aware-ik", action="store_true")
    parser.add_argument(
        "--ik-backend", choices=["mink", "spider-native"], default="mink"
    )
    parser.add_argument(
        "--mink-fidelity-continuation",
        dest="mink_allow_fidelity_rejected_qref_for_refinement",
        action="store_true",
        help=(
            "Explicitly allow a q_ref that failed fingertip/wrist fidelity "
            "thresholds to continue to MPC refinement. Off by default."
        ),
    )
    # Backward-compatible spelling; it only reiterates the safe default.
    parser.add_argument(
        "--no-mink-fidelity-continuation",
        dest="mink_allow_fidelity_rejected_qref_for_refinement",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(mink_allow_fidelity_rejected_qref_for_refinement=False)
    parser.add_argument(
        "--mink-collision-projection-max-iterations", type=int, default=80
    )
    parser.add_argument(
        "--mink-controller-contact-target-policy",
        choices=[
            "object_surface", "fingertip_collision_center",
            "qref_fingertip_site",
        ],
        default="qref_fingertip_site",
    )
    parser.add_argument(
        "--mink-scene-collision-constraints", action="store_true",
        help=(
            "Add digital-twin hand-floor and hand-object constraints to the "
            "paper's joint-limit/self-collision MINK feasible set. Fingertip "
            "targets remain the observed human targets."
        ),
    )
    parser.add_argument("--mink-floor-clearance-m", type=float, default=0.0)
    parser.add_argument(
        "--mink-non-distal-object-clearance-m", type=float, default=0.0,
    )
    parser.add_argument(
        "--mink-distal-object-max-penetration-m", type=float, default=0.0025,
    )
    parser.add_argument("--replay-noise-scale", type=float, default=0.0)
    parser.add_argument("--ik-seed", type=int, default=0)
    parser.add_argument("--no-adaptive-mode-switching", action="store_true")
    parser.add_argument("--replay-chunk-steps", type=int, default=20)
    parser.add_argument("--replay-lookahead-chunks", type=int, default=2)
    parser.add_argument("--replay-position-threshold-m", type=float, default=0.05)
    parser.add_argument("--replay-rotation-threshold-rad", type=float, default=0.5)
    parser.add_argument("--replay-min-motion-transfer-ratio", type=float, default=0.5)
    preshape = parser.add_mutually_exclusive_group()
    preshape.add_argument(
        "--contact-aware-preshape", dest="contact_aware_preshape",
        action="store_true",
    )
    preshape.add_argument(
        "--no-contact-aware-preshape", dest="contact_aware_preshape",
        action="store_false",
    )
    parser.set_defaults(contact_aware_preshape=False)
    parser.add_argument("--preshape-approach-frames", type=int, default=12)
    parser.add_argument("--preshape-clearance-m", type=float, default=0.03)
    parser.add_argument("--preshape-collision-margin-m", type=float, default=0.001)
    parser.add_argument("--preshape-contact-penetration-m", type=float, default=0.0025)
    parser.add_argument("--preshape-solver-iterations", type=int, default=180)
    parser.add_argument("--preshape-solver-dt", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(run_spider_chain(
        dataset_root=args.dataset_root, spider_root=args.spider_root, task=args.task,
        data_id=args.data_id, embodiment_type=args.embodiment_type, robot_type=args.robot_type,
        gpu=args.gpu, run_mjwp=not args.no_mjwp, save_video=not args.no_save_video,
        ik_end_idx=args.ik_end_idx, mjwp_num_samples=args.mjwp_num_samples,
        mjwp_iterations=args.mjwp_iterations, mjwp_override=args.mjwp_override,
        mjwp_horizon=args.mjwp_horizon, mjwp_ctrl_dt=args.mjwp_ctrl_dt,
        mjwp_knot_dt=args.mjwp_knot_dt, contact_reward_scale=args.contact_reward_scale,
        contact_opposition_reward_scale=args.contact_opposition_reward_scale,
        force_closure_reward_scale=args.force_closure_reward_scale,
        force_closure_penetration_reward_scale=(
            args.force_closure_penetration_reward_scale
        ),
        force_closure_min_normal_force_n=args.force_closure_min_normal_force_n,
        force_closure_min_opposition_cosine=(
            args.force_closure_min_opposition_cosine
        ),
        force_closure_max_penetration_m=args.force_closure_max_penetration_m,
        force_closure_minimum_frames=args.force_closure_minimum_frames,
        lift_reward_scale=args.lift_reward_scale,
        paper_objective=args.paper_objective,
        action_smoothness_reward_scale=args.action_smoothness_reward_scale,
        sanity_check_seconds=args.sanity_check_seconds,
        collision_aware_ik=not args.no_collision_aware_ik,
        ik_backend=args.ik_backend,
        mink_allow_fidelity_rejected_qref_for_refinement=(
            args.mink_allow_fidelity_rejected_qref_for_refinement
        ),
        mink_collision_projection_max_iterations=(
            args.mink_collision_projection_max_iterations
        ),
        mink_controller_contact_target_policy=(
            args.mink_controller_contact_target_policy
        ),
        mink_scene_collision_constraints=args.mink_scene_collision_constraints,
        mink_floor_clearance_m=args.mink_floor_clearance_m,
        mink_non_distal_object_clearance_m=(
            args.mink_non_distal_object_clearance_m
        ),
        mink_distal_object_max_penetration_m=(
            args.mink_distal_object_max_penetration_m
        ),
        replay_noise_scale=args.replay_noise_scale,
        ik_seed=args.ik_seed,
        adaptive_mode_switching=not args.no_adaptive_mode_switching,
        replay_chunk_steps=args.replay_chunk_steps,
        replay_lookahead_chunks=args.replay_lookahead_chunks,
        replay_position_threshold_m=args.replay_position_threshold_m,
        replay_rotation_threshold_rad=args.replay_rotation_threshold_rad,
        replay_min_motion_transfer_ratio=args.replay_min_motion_transfer_ratio,
        contact_aware_preshape=args.contact_aware_preshape,
        preshape_approach_frames=args.preshape_approach_frames,
        preshape_clearance_m=args.preshape_clearance_m,
        preshape_collision_margin_m=args.preshape_collision_margin_m,
        preshape_contact_penetration_m=args.preshape_contact_penetration_m,
        preshape_solver_iterations=args.preshape_solver_iterations,
        preshape_solver_dt=args.preshape_solver_dt,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
