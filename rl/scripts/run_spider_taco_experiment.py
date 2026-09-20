"""Isolated SPIDER-style MINK experiment on the prepared TACO Pour GT.

This script is deliberately a candidate generator, not a replacement for the
paper-faithful retargeter.  It ports only the kinematic choices visible in
SPIDER's upstream ``ik_fast.py``: wrist position/orientation tracking,
position-only fingertip tracking, a posture regularizer, and two-stage frame-0
initialization.  The local collision, joint-limit, and frame-velocity checks
remain enabled so a lower position residual cannot be attributed to silently
removing feasibility constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]

import mujoco
import mink
import numpy as np
from scipy.spatial.transform import Rotation

from egoengine_repro.retarget.collision_audit import (  # noqa: E402
    audit_trajectory,
    audit_intrahand_trajectory,
    distances,
    explicit_hand_pairs,
    validate_qpos,
)
from egoengine_repro.retarget.mink import (  # noqa: E402
    _FrameDisplacementLimit,
    _StrictCollisionLimit,
    _enable_planning_collision_masks,
    _explicit_collision_groups,
    _joint_velocity_limits,
)
from egoengine_repro.retarget.taco_bimanual import differentiate, pose7  # noqa: E402


SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SOURCE_COMMIT = "71238456bf97a7eeb3d0471aa31974e2d404d4ae"
MINK_COMMIT = "ab45779fea46933832dee1c240f94103633347a1"
DEFAULT_RUN = ROOT / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1"
DEFAULT_OUTPUT = ROOT / "runs/taco_pour_spider_mink_experiment_v1"
DEFAULT_SCENE = ROOT / (
    "models/taco_xhand/xhand/bimanual/"
    "taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
)
VENDOR = DEFAULT_OUTPUT / "vendor/spider"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _set_object_pose(model: mujoco.MjModel, q: np.ndarray, human: dict,
                     frame: int) -> None:
    for hand, side in enumerate(SIDES):
        address = int(model.joint(f"{side}_object_joint").qposadr[0])
        q[address:address + 7] = pose7(human["T_sim_object_reference"][frame, hand])


def _build_tasks(model: mujoco.MjModel, human: dict, frame: int,
                 posture: mink.PostureTask, wrist_tasks: list[mink.FrameTask],
                 finger_tasks: list[mink.FrameTask]) -> None:
    for hand, side in enumerate(SIDES):
        wrist_tasks[hand].set_target(
            mink.SE3.from_matrix(human["T_sim_wrist_target"][frame, hand])
        )
        for finger_index in range(5):
            target = human["T_sim_fingertip_target"][frame, hand, finger_index]
            finger_tasks[hand * 5 + finger_index].set_target(
                mink.SE3.from_translation(target[:3, 3])
            )


def _make_collision_limits(model: mujoco.MjModel, settings: dict):
    hand_geoms = [
        geom for geom in range(model.ngeom)
        if (model.geom(geom).name or "").startswith("collision_hand_")
    ]
    pairs = explicit_hand_pairs(model)
    if not pairs:
        raise ValueError("SPIDER experiment requires an audited hand-pair contract")
    _enable_planning_collision_masks(model, hand_geoms, [])
    groups = _explicit_collision_groups(model, mujoco, hand_geom_ids=set(hand_geoms))
    inner = mink.CollisionAvoidanceLimit(
        model,
        groups,
        minimum_distance_from_collisions=0.0,
        collision_detection_distance=0.02,
    )
    if set(inner.geom_id_pairs) != set(pairs):
        raise ValueError("MINK filtered out runtime self pairs")
    collision = _StrictCollisionLimit(
        inner, mujoco, minimum_distance=0.0, depenetration_step=0.002
    )
    collision.enabled = True
    velocity = _joint_velocity_limits(model, mujoco, settings["velocity_limits"])
    displacement = _FrameDisplacementLimit(model, mujoco, velocity, mink.Constraint)
    return pairs, collision, velocity, displacement


def _metrics(model: mujoco.MjModel, qpos: np.ndarray, human: dict,
             velocity: dict[str, float], scene: Path) -> dict:
    validate_qpos(model, qpos, trajectory=True)
    data = mujoco.MjData(model)
    tip_position = np.empty((len(qpos), 2, 5), dtype=float)
    tip_orientation = np.empty((len(qpos), 2, 5), dtype=float)
    wrist_position = np.empty((len(qpos), 2), dtype=float)
    wrist_orientation = np.empty((len(qpos), 2), dtype=float)
    explicit_self = np.empty(len(qpos), dtype=float)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for hand, side in enumerate(SIDES):
            body_id = model.body(f"{side}_hand_link").id
            body_position = data.xpos[body_id]
            body_rotation = data.xmat[body_id].reshape(3, 3)
            wrist_target = human["T_sim_wrist_target"][frame, hand]
            wrist_position[frame, hand] = np.linalg.norm(
                body_position - wrist_target[:3, 3]
            )
            wrist_orientation[frame, hand] = np.rad2deg(
                Rotation.from_matrix(
                    body_rotation.T @ wrist_target[:3, :3]
                ).magnitude()
            )
            for finger_index, finger in enumerate(FINGERS):
                site_id = model.site(f"{side}_{finger}_tip").id
                site_rotation = data.site_xmat[site_id].reshape(3, 3)
                target = human["T_sim_fingertip_target"][frame, hand, finger_index]
                tip_position[frame, hand, finger_index] = np.linalg.norm(
                    data.site_xpos[site_id] - target[:3, 3]
                )
                tip_orientation[frame, hand, finger_index] = np.rad2deg(
                    Rotation.from_matrix(
                        site_rotation.T @ target[:3, :3]
                    ).magnitude()
                )
        explicit_self[frame] = float(distances(model, data, explicit_hand_pairs(model)).min())

    addresses = np.asarray(
        [int(model.joint(name).qposadr[0]) for name in velocity], dtype=int
    )
    speeds = np.asarray(list(velocity.values()), dtype=float)
    ranges = np.asarray([model.joint(name).range for name in velocity], dtype=float)
    margins = np.minimum(
        qpos[:, addresses] - ranges[:, 0], ranges[:, 1] - qpos[:, addresses]
    ).min(axis=1)
    dt = float(np.diff(human["timestamps_s"])[0])
    velocity_ratio = np.abs(np.diff(qpos[:, addresses], axis=0)) / (dt * speeds)
    collisions = audit_trajectory(model, qpos)
    object_errors = []
    for hand, side in enumerate(SIDES):
        address = int(model.joint(f"{side}_object_joint").qposadr[0])
        expected = np.asarray(
            [pose7(pose) for pose in human["T_sim_object_reference"][:, hand]]
        )
        object_errors.append(float(np.abs(qpos[:, address:address + 7] - expected).max()))
    return {
        "fingertip_position_error_m": tip_position,
        "fingertip_orientation_error_deg": tip_orientation,
        "wrist_position_error_m": wrist_position,
        "wrist_orientation_error_deg": wrist_orientation,
        "self_collision_distance_m": explicit_self,
        "joint_limit_min_margin": margins,
        "frame_velocity_max_ratio": velocity_ratio.max(axis=1),
        "collision_families": collisions,
        "object_reference_max_abs_error": object_errors,
        "scene": artifact(scene),
    }


def _summary(metrics: dict) -> dict:
    tips = metrics["fingertip_position_error_m"]
    wrists = metrics["wrist_position_error_m"]
    tip_ori = metrics["fingertip_orientation_error_deg"]
    wrist_ori = metrics["wrist_orientation_error_deg"]
    families = metrics["collision_families"]
    physical_collision_ok = all(
        info["min_distance_m"] is None or info["min_distance_m"] >= -1e-6
        for info in families.values() if isinstance(info, dict) and "min_distance_m" in info
    )
    joint_ok = bool(np.all(metrics["joint_limit_min_margin"] >= -1e-6))
    speed_ok = bool(np.all(metrics["frame_velocity_max_ratio"] <= 1.0 + 1e-6))
    self_ok = bool(np.all(metrics["self_collision_distance_m"] >= -1e-6))
    return {
        "fingertip_mean_error_m_by_hand": tips.mean(axis=(0, 2)).tolist(),
        "fingertip_max_error_m_by_hand": tips.max(axis=(0, 2)).tolist(),
        "fingertip_mean_error_mm_by_hand": (tips.mean(axis=(0, 2)) * 1000).tolist(),
        "fingertip_mean_error_mm_by_hand_finger": (tips.mean(axis=0) * 1000).tolist(),
        "fingertip_orientation_mean_deg_by_hand": tip_ori.mean(axis=(0, 2)).tolist(),
        "wrist_position_mean_mm_by_hand": (wrists.mean(axis=0) * 1000).tolist(),
        "wrist_orientation_mean_deg_by_hand": wrist_ori.mean(axis=0).tolist(),
        "self_collision_min_distance_m": float(metrics["self_collision_distance_m"].min()),
        "joint_limit_min_margin": float(metrics["joint_limit_min_margin"].min()),
        "frame_velocity_max_ratio": float(metrics["frame_velocity_max_ratio"].max()),
        "collision_family_min_distance_m": {
            key: info.get("min_distance_m")
            for key, info in families.items()
            if isinstance(info, dict) and "min_distance_m" in info
        },
        "self_collision_constraint_ok": self_ok,
        "joint_limit_ok": joint_ok,
        "frame_velocity_ok": speed_ok,
        "all_collision_families_clear": physical_collision_ok,
        "declared_kinematic_gate_passed": bool(self_ok and joint_ok and speed_ok),
        "strict_gate_passed": bool(self_ok and joint_ok and speed_ok and physical_collision_ok),
    }


def _candidate_summary(path: Path) -> dict | None:
    report = path / "retarget_report.json"
    if not report.exists():
        return None
    value = json.loads(report.read_text())
    return value.get("summary")


def run(output: Path, human_path: Path, scene: Path, baseline: Path,
        settings: dict, wrist_init_steps: int, finger_init_steps: int) -> dict:
    if output.is_symlink():
        raise FileExistsError(output)
    if output.exists():
        # The vendor snapshot is intentionally staged before the run.  It is
        # the only pre-existing entry permitted; every result artifact must
        # be new so a rerun cannot silently replace evidence.
        unexpected = [
            path for path in output.iterdir() if path.name != "vendor"
        ]
        if unexpected:
            raise FileExistsError(
                f"output already contains result files: {', '.join(map(str, unexpected))}"
            )
    with np.load(human_path, allow_pickle=False) as source:
        human = {key: np.asarray(source[key]) for key in source.files}
    q0 = np.asarray(human["T_sim_object_reference"])
    if q0.shape[0] < 2:
        raise ValueError("TACO reference must contain at least two frames")
    model = mujoco.MjModel.from_xml_path(str(scene))
    if (model.nq, model.nv, model.nu) != (50, 48, 36):
        raise ValueError("expected the two 18-DoF XHand TACO scene")
    pairs, collision, velocity, displacement = _make_collision_limits(model, settings)
    configuration = mink.Configuration(model)
    posture = mink.PostureTask(model, cost=settings["posture_cost"], lm_damping=1e-3)
    posture.set_target(configuration.q.copy())
    wrist_tasks = [
        mink.FrameTask(
            f"{side}_hand_link", "body",
            position_cost=settings["wrist_position_cost"],
            orientation_cost=settings["wrist_orientation_cost"],
            lm_damping=1e-3,
        ) for side in SIDES
    ]
    finger_tasks = [
        mink.FrameTask(
            f"{side}_{finger}_tip", "site",
            position_cost=settings["fingertip_position_cost"],
            orientation_cost=0.0,
            lm_damping=1e-3,
        ) for side in SIDES for finger in FINGERS
    ]
    locks = [mink.DofFreezingTask(model, list(range(36, 48)))]
    dt = float(np.diff(human["timestamps_s"])[0])
    substeps = int(settings["max_iterations_per_frame"])
    if substeps < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid frame or solver timing")
    qp_dt = dt / substeps
    qpos = np.empty((len(human["frame_indices"]), model.nq), dtype=float)
    phase_counts = {"wrist": 0, "fingertip": 0}
    solver_failures = []
    previous = None

    def solve(tasks, frame):
        try:
            return mink.solve_ik(
                configuration, tasks, qp_dt, solver="daqp", damping=1e-5,
                limits=[mink.ConfigurationLimit(model), collision, displacement],
                constraints=locks, primal_tol=1e-9, dual_tol=1e-9,
            )
        except Exception as error:  # retain frame context in the artifact
            solver_failures.append({"frame": frame, "error": repr(error)})
            raise

    def project_collision(frame):
        for _ in range(128):
            if distances(model, configuration.data, pairs).min() >= -1e-6:
                return
            velocity_step = solve([], frame)
            configuration.integrate_inplace(velocity_step, qp_dt)
        raise RuntimeError(f"self-collision feasibility projection failed at frame {frame}")

    for frame in range(len(qpos)):
        if frame == 0:
            q = configuration.q.copy()
            _set_object_pose(model, q, human, frame)
            # SPIDER starts from its model's home pose.  This TACO scene's
            # home pose places both floating hands at the origin, which is
            # already deeply self-colliding under the local strict contract.
            # Use the same explicit wrist seed as the existing TACO adapter;
            # the subsequent wrist-only and fingertip phases remain intact.
            for hand in range(2):
                wrist = human["T_sim_wrist_target"][frame, hand]
                q[hand * 18:hand * 18 + 3] = wrist[:3, 3]
                q[hand * 18 + 3:hand * 18 + 6] = (
                    Rotation.from_matrix(wrist[:3, :3]).as_euler("ZXY")
                    * np.asarray([1.0, 1.0, -1.0])
                )
            configuration.update(q)
            _build_tasks(model, human, frame, posture, wrist_tasks, finger_tasks)
            for _ in range(wrist_init_steps):
                configuration.integrate_inplace(solve([posture, *wrist_tasks], frame), qp_dt)
            phase_counts["wrist"] = wrist_init_steps
            configuration.update()
            for _ in range(finger_init_steps):
                configuration.integrate_inplace(
                    solve([posture, *wrist_tasks, *finger_tasks], frame), qp_dt
                )
            phase_counts["fingertip"] = finger_init_steps
        else:
            q = configuration.q.copy()
            _set_object_pose(model, q, human, frame)
            configuration.update(q)
            _build_tasks(model, human, frame, posture, wrist_tasks, finger_tasks)
            displacement.set_previous(previous, dt)
            for _ in range(substeps):
                configuration.integrate_inplace(
                    solve([posture, *wrist_tasks, *finger_tasks], frame), qp_dt
                )
        project_collision(frame)
        qpos[frame] = configuration.q
        previous = configuration.q.copy()
        if frame % 25 == 0 or frame == len(qpos) - 1:
            print(
                f"SPIDER-style {frame + 1}/{len(qpos)}; self clearance "
                f"{distances(model, configuration.data, pairs).min():.6f} m",
                flush=True,
            )

    qvel = differentiate(model, qpos, dt)
    metrics = _metrics(model, qpos, human, velocity, scene)
    intrahand_audit = audit_intrahand_trajectory(model, qpos)
    summary = _summary(metrics)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "robot_reference.npz",
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[:, :36],
        frequency=1.0 / dt,
        frame_indices=human["frame_indices"],
        timestamps_s=human["timestamps_s"],
        hand_order=np.asarray(SIDES),
        object_roles=np.asarray(["tool", "target"]),
        fingertip_position_error_m=metrics["fingertip_position_error_m"],
        fingertip_orientation_error_deg=metrics["fingertip_orientation_error_deg"],
        wrist_position_error_m=metrics["wrist_position_error_m"],
        wrist_orientation_error_deg=metrics["wrist_orientation_error_deg"],
        self_collision_distance_m=metrics["self_collision_distance_m"],
        joint_limit_min_margin=metrics["joint_limit_min_margin"],
        frame_velocity_max_ratio=metrics["frame_velocity_max_ratio"],
    )
    np.savez_compressed(output / "diagnostic_metrics.npz", **{
        key: value for key, value in metrics.items()
        if isinstance(value, np.ndarray)
    })
    intrahand_path = output / "intrahand_collision_audit.json"
    intrahand_path.write_text(json.dumps({
        "scene": str(scene.resolve()),
        "human_reference": str(human_path.resolve()),
        **intrahand_audit,
    }, indent=2) + "\n")
    baseline_report = _candidate_summary(baseline)
    vendor_root = output / "vendor/spider"
    vendor_files = sorted(p for p in vendor_root.rglob("*") if p.is_file())
    report = {
        "status": "isolated_spider_style_kinematic_candidate",
        "formal_replacement": False,
        "rl_validation_completed": False,
        "source_commit": SOURCE_COMMIT,
        "mink_source_commit": MINK_COMMIT,
        "source_repo": "/data_all/zzx/egoengine/spider",
        "mink_module": str(Path(mink.__file__).resolve()),
        "mink_source": artifact(ROOT / "external/mink/src/mink/__init__.py"),
        "inputs": [artifact(human_path), artifact(scene)],
        "vendor_files": [artifact(path) for path in vendor_files],
        "baseline_run": str(baseline.resolve()),
        "baseline_summary": baseline_report,
        "settings": settings,
        "effective_strategy": {
            "wrist_position_cost": settings["wrist_position_cost"],
            "wrist_orientation_cost": settings["wrist_orientation_cost"],
            "fingertip_position_cost": settings["fingertip_position_cost"],
            "fingertip_orientation_cost": 0.0,
            "posture_cost": settings["posture_cost"],
            "wrist_init_steps": wrist_init_steps,
            "fingertip_init_steps": finger_init_steps,
            "sequential_hot_start": True,
            "moving_average_filter": False,
            "objects": "GT pose copied each frame and object DoFs locked",
        },
        "constraints": {
            "explicit_self_collision_pairs": len(pairs),
            "joint_position_limits": True,
            "frame_velocity_limits": True,
            "collision_projection": "strict local separating QP, up to 128 steps/frame",
        },
        "phase_counts": phase_counts,
        "solver_failures": solver_failures,
        "summary": summary,
        "intrahand_collision_audit_path": str(intrahand_path.resolve()),
        "intrahand_collision_summary": {
            "pair_count": intrahand_audit["pair_count"],
            "counts": intrahand_audit["counts"],
            "by_classification": intrahand_audit["by_classification"],
            "min_distance_m": intrahand_audit["min_distance_m"],
            "penetrating_frames": intrahand_audit["penetrating_frames"],
            "tolerance_m": intrahand_audit["tolerance_m"],
        },
        "object_reference_max_abs_error": metrics["object_reference_max_abs_error"],
        "interpretation": (
            "Compare the summary against baseline_summary only on matched frames. "
            "A fingertip reduction is a diagnostic improvement only when wrist, "
            "orientation, collision, joint, and velocity gates remain acceptable."
        ),
    }
    (output / "retarget_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--human", type=Path, default=DEFAULT_RUN / "human_reference.npz")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--wrist-init-steps", type=int, default=200)
    parser.add_argument("--fingertip-init-steps", type=int, default=300)
    args = parser.parse_args()
    if args.wrist_init_steps < 1 or args.fingertip_init_steps < 1:
        raise ValueError("initialization step counts must be positive")
    settings = json.loads(
        (ROOT / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/retarget_report.json").read_text()
    )["inherited_settings"]
    report = run(
        args.output, args.human, args.scene, args.baseline, settings,
        args.wrist_init_steps, args.fingertip_init_steps,
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
