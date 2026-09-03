#!/usr/bin/env python3
"""Rank isolated BODex proposals by free-object physics before any rendering.

Every finite proposal is checked geometrically.  Only proposals that are
already unsafe in a static target are excluded from the dynamic rollout.  All
remaining proposals execute the same hand-only, free-object motion used by
``bodex_triptych.py``.  This creates a diagnostic ranking, never an input to
the formal controller, renderer, or 3.3 pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egoengine_repro.action.contracts import XHAND_SELF_FLOOR_TOLERANCE_M
from egoengine_repro.action.geometry import explicit_collision_pairs, minimum_pair_distance
from egoengine_repro.action.replay import MujocoReplayBackend

from bodex_triptych import (
    DEFAULT_MAX_OBJECT_PENETRATION_M,
    FINGERS,
    STAGE_NAMES,
    Target,
    apply_physics_profile,
    checked_target,
    compensated_target,
    contact_row,
    dynamic_rollout_targets,
    load_candidate,
    load_object_pose,
    map_target,
    project_hand_joint_limits,
    require_file,
    require_scene,
    smooth_target,
    strict_grasp_gate,
    target_gaps,
)


SCHEMA = "xhand_bodex_candidate_sweep_v5_diagnostic_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--object-reference", type=Path, required=True)
    parser.add_argument("--human-reference", type=Path)
    parser.add_argument("--manual-label", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--physics-profile",
        choices=("current", "deximit_drive", "deximit_approx", "deximit_full"),
        default="current",
        help=(
            "current uses bounded XHand effort; deximit_drive reproduces DexImit's "
            "stiff finger position drives; deximit_approx also uses its default object mass; "
            "deximit_full uses the isolated single-hull scene and object damping too."
        ),
    )
    parser.add_argument(
        "--deximit-object-mass", type=float, default=0.05885494234682545,
        help="Object mass for deximit_approx; smear-eraser default is mesh volume * 300.",
    )
    parser.add_argument("--finger-kp", type=float, default=2.0)
    parser.add_argument("--finger-kv", type=float, default=0.002)
    parser.add_argument("--finger-cap", type=float, default=1.1)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--approach-s", type=float, default=1.0)
    parser.add_argument("--squeeze-s", type=float, default=0.75)
    parser.add_argument("--lift-s", type=float, default=1.0)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--lift-m", type=float, default=0.05)
    parser.add_argument(
        "--motion-mode", choices=("vertical_lift", "source"), default="vertical_lift",
        help=(
            "vertical_lift uses the existing 50 mm control; source applies the "
            "human-reviewed object's relative 6DoF motion to the hand root."
        ),
    )
    parser.add_argument(
        "--max-object-penetration-m", type=float,
        default=DEFAULT_MAX_OBJECT_PENETRATION_M,
        help="Maximum sampled hand/object collision overlap allowed by the strict gate.",
    )
    parser.add_argument(
        "--closure-stage", choices=("grasp", "squeeze"), default="squeeze",
        help="BODex target held during lift; use only as a separately labelled schedule ablation.",
    )
    parser.add_argument(
        "--gap-stride", type=int, default=10,
        help="Measure dynamic collision distances every N physics steps while screening.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="For a timing smoke test only: evaluate the first N finite proposals (0 means all).",
    )
    parser.add_argument(
        "--candidate-index", type=int, action="append",
        help=(
            "Evaluate only these source indices. Repeat the option for multiple "
            "SAPIEN pass-through candidates."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_matrix(pose_wxyz: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("pose must be a finite xyz+wxyz vector")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def source_motion_contract(
    human_reference: Path, manual_label: Path, object_qpos: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    label = json.loads(manual_label.read_text(encoding="utf-8"))
    if (
        label.get("schema") != "deximit_manual_subactions_v1_diagnostic_only"
        or label.get("diagnostic_only") is not True
        or label.get("formal_renderer_3_3_eligible") is not False
        or label.get("hand") != "right"
    ):
        raise ValueError("manual label violates the isolated diagnostic contract")
    rows = label.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("manual label lacks action rows")
    grasp, motion = int(rows["grasp"]), int(rows["motion"])
    if not 0 <= grasp < motion:
        raise ValueError("manual grasp/motion rows are not ordered")
    with np.load(human_reference, allow_pickle=False) as values:
        reference = np.asarray(values["T_sim_object_reference"], dtype=np.float64)
    if reference.ndim == 4 and reference.shape[1] == 1:
        reference = reference[:, 0]
    if reference.ndim != 3 or reference.shape[1:] != (4, 4) or motion >= len(reference):
        raise ValueError("human reference lacks the labelled object poses")
    source_relative = reference[motion] @ np.linalg.inv(reference[grasp])
    initial = pose_matrix(object_qpos)
    relocate = initial @ np.linalg.inv(reference[grasp])
    relative_world = relocate @ source_relative @ np.linalg.inv(relocate)
    return relative_world, {
        "label_source": "human_reviewed_video_and_geometry",
        "pregrasp_row": int(rows["pregrasp"]),
        "grasp_row": grasp,
        "motion_row": motion,
        "translation_m": relative_world[:3, 3].tolist(),
        "translation_norm_m": float(np.linalg.norm(relative_world[:3, 3])),
        "rotation_deg": float(np.degrees(Rotation.from_matrix(relative_world[:3, :3]).magnitude())),
    }


def phase_metrics(
    backend: MujocoReplayBackend, inverse_data: mujoco.MjData,
    inverse_force: np.ndarray, *, name: str, first: np.ndarray, second: np.ndarray,
    duration_s: float, finger_kp: float, finger_kv: float,
    pairs: dict[str, tuple[tuple[int, int], ...]], gap_stride: int,
) -> dict[str, object]:
    """Run one phase without retaining a render trace.

    Object height and contact are sampled at every physical step.  Expensive
    signed-distance checks are sampled densely for screening; the selected
    winner is then rerun by ``bodex_triptych.py`` with a full, replay-checked
    per-step trace before a video can be made.
    """
    model = backend.model
    steps = max(1, int(round(duration_s / float(model.opt.timestep))))
    object_z: list[float] = []
    contact_seen = np.zeros(len(FINGERS), dtype=bool)
    opposed_contact_seen = False
    peak_force = np.zeros(len(FINGERS), dtype=np.float64)
    minima = {family: float("inf") for family in pairs}
    max_target_error = 0.0
    robot_addresses = backend.actuator_qpos_addresses
    for step in range(1, steps + 1):
        desired, velocity, acceleration = smooth_target(
            first, second, step / steps, duration_s, robot_addresses,
        )
        desired, command = compensated_target(
            backend, inverse_data, inverse_force, desired, velocity, acceleration,
            finger_kp, finger_kv,
        )
        backend.step(backend.reference_action(command), float(model.opt.timestep))
        if (
            not np.isfinite(backend.data.qpos).all()
            or not np.isfinite(backend.data.qvel).all()
            or not np.isfinite(backend.data.qacc).all()
            or any(int(item.number) > 0 for item in backend.data.warning)
        ):
            raise FloatingPointError(
                f"numerically unstable rollout at phase {name}, step {step}/{steps}",
            )
        touched, normal = contact_row(backend)
        contact_seen |= touched
        opposed_contact_seen |= bool(touched[0] and touched[1:].any())
        peak_force = np.maximum(peak_force, normal)
        object_z.append(float(backend.data.xpos[backend.object_body_id, 2]))
        max_target_error = max(
            max_target_error,
            float(np.linalg.norm(
                backend.data.qpos[robot_addresses] - desired[robot_addresses],
            )),
        )
        if step == 1 or step == steps or step % gap_stride == 0:
            for family, values in pairs.items():
                minima[family] = min(
                    minima[family],
                    minimum_pair_distance(model, backend.data, backend.mujoco, values),
                )
    return {
        "name": name,
        "steps": steps,
        "object_z_start_m": object_z[0],
        "object_z_end_m": object_z[-1],
        "object_z_min_m": min(object_z),
        "object_z_peak_m": max(object_z),
        "contacted_digits": [
            finger for number, finger in enumerate(FINGERS) if contact_seen[number]
        ],
        "simultaneous_thumb_and_other_observed": opposed_contact_seen,
        "peak_normal_force_n": {
            finger: float(peak_force[number]) for number, finger in enumerate(FINGERS)
        },
        "minimum_sampled_gaps_m": minima,
        "maximum_target_error_l2": max_target_error,
    }


def evaluate_one(
    backend: MujocoReplayBackend, *, candidate_path: Path, index: int,
    object_qpos: np.ndarray, pairs: dict[str, tuple[tuple[int, int], ...]],
    motion_transform: np.ndarray | None, args: argparse.Namespace,
) -> dict[str, object]:
    pose, joints, _ = load_candidate(candidate_path, index)
    targets = [
        map_target(
            backend.model, object_qpos, pose[stage], joints[stage],
            backend.object_qpos_address,
        )
        for stage in range(3)
    ]
    for target in targets:
        checked_target(backend, target, object_qpos)
    projected = []
    limit_violations = []
    for target in targets:
        qpos, violations = project_hand_joint_limits(backend.model, target.qpos)
        projected.append(Target(
            qpos, target.hand_position_world_m, target.hand_rotation_world,
        ))
        limit_violations.append(violations)
    static = [target_gaps(backend, target, pairs) for target in projected]
    static_hard_legal = all(
        values["self"] >= -XHAND_SELF_FLOOR_TOLERANCE_M
        and values["floor"] >= -XHAND_SELF_FLOOR_TOLERANCE_M
        for values in static
    )
    result: dict[str, object] = {
        "candidate_index": index,
        "static_projected_target_gaps_m": dict(zip(("pregrasp", "grasp", "squeeze"), static)),
        "target_joint_limit_projection_rad": dict(zip(
            ("pregrasp", "grasp", "squeeze"), limit_violations,
        )),
        "maximum_target_joint_limit_projection_rad": max(
            max(values.values()) for values in limit_violations
        ),
        "static_hard_legal": static_hard_legal,
    }
    if not static_hard_legal:
        result.update({
            "dynamic_rollout": "skipped: static self or floor safety violation",
            "physics_gate_pass": False,
            "rank_score": -1_000_000_000.0,
        })
        return result

    if args.motion_mode == "source" and motion_transform is None:
        raise ValueError("source motion requires a relative transform")
    pregrasp, grasp, closure, move = dynamic_rollout_targets(
        backend, targets, projected[0].qpos,
        closure_stage=args.closure_stage, motion_transform=motion_transform,
        object_qpos=object_qpos, vertical_lift_m=args.lift_m,
    )
    backend.initialize(pregrasp, np.zeros(backend.model.nv, dtype=np.float64))
    inverse_data = mujoco.MjData(backend.model)
    inverse_force = np.zeros(backend.model.nv, dtype=np.float64)
    try:
        phases = [
            phase_metrics(
                backend, inverse_data, inverse_force, name=name, first=first, second=second,
                duration_s=duration, finger_kp=args.finger_kp, finger_kv=args.finger_kv,
                pairs=pairs, gap_stride=args.gap_stride,
            )
            for name, first, second, duration in (
                (STAGE_NAMES[0], pregrasp, pregrasp, args.settle_s),
                (STAGE_NAMES[1], pregrasp, grasp, args.approach_s),
                (STAGE_NAMES[2], grasp, closure, args.squeeze_s),
                (STAGE_NAMES[3], closure, move, args.lift_s),
                (STAGE_NAMES[4], move, move, args.hold_s),
            )
        ]
    except FloatingPointError as error:
        result.update({
            "dynamic_rollout": f"failed: {error}",
            "physics_gate_pass": False,
            "rank_score": -500_000_000.0,
        })
        return result
    rest_z = float(phases[0]["object_z_end_m"])
    hold = phases[-1]
    dynamic_self = min(float(item["minimum_sampled_gaps_m"]["self"]) for item in phases)
    dynamic_floor = min(float(item["minimum_sampled_gaps_m"]["floor"]) for item in phases)
    dynamic_hard_legal = bool(
        dynamic_self >= -XHAND_SELF_FLOOR_TOLERANCE_M
        and dynamic_floor >= -XHAND_SELF_FLOOR_TOLERANCE_M
    )
    contact_names = {
        name for phase in phases for name in phase["contacted_digits"]
    }
    opposed_contact = any(
        bool(phase["simultaneous_thumb_and_other_observed"])
        for phase in phases[2:]
    )
    hold_end_lift_m = float(hold["object_z_end_m"]) - rest_z
    hold_min_lift_m = float(hold["object_z_min_m"]) - rest_z
    held_lift = hold_end_lift_m >= 0.02 and hold_min_lift_m >= 0.019
    hold_opposed_contact = bool(hold["simultaneous_thumb_and_other_observed"])
    hold_vertical_span_m = float(hold["object_z_peak_m"]) - float(hold["object_z_min_m"])
    stable_hold = hold_vertical_span_m <= 0.01
    minimum_object_gap_m = min(
        float(item["minimum_sampled_gaps_m"]["object"]) for item in phases
    )
    object_penetration_safe = bool(
        minimum_object_gap_m >= -args.max_object_penetration_m
    )
    peak_normal_force_n = max(
        float(value)
        for phase in phases for value in phase["peak_normal_force_n"].values()
    )
    gate = strict_grasp_gate(
        hard_legal=dynamic_hard_legal,
        opposed_contact=opposed_contact,
        hold_opposed_contact=hold_opposed_contact,
        held_lift=held_lift,
        stable_hold=stable_hold,
        minimum_object_gap_m=minimum_object_gap_m,
        maximum_object_penetration_m=args.max_object_penetration_m,
    )
    # A full pass remains dominant.  Within the same gate class, prefer stable
    # retained lift with shallower object overlap and lower contact force; this
    # avoids ranking a violent upward squeeze as the best grasp solely because
    # it produced the greatest height.
    rank_score = (
        (10_000_000.0 if gate else 0.0)
        + (1_000_000.0 if dynamic_hard_legal else 0.0)
        + (100_000.0 if opposed_contact else 0.0)
        + (50_000.0 if hold_opposed_contact else 0.0)
        + (25_000.0 if stable_hold else 0.0)
        + 10_000.0 * float(np.clip(hold_min_lift_m, -0.10, 0.10))
        - 100_000.0 * max(0.0, -minimum_object_gap_m)
        - peak_normal_force_n
        + float(len(contact_names))
    )
    result.update({
        "dynamic_rollout": "completed",
        "phases": phases,
        "rest_object_z_m": rest_z,
        "minimum_sampled_hand_self_gap_m": dynamic_self,
        "minimum_sampled_hand_floor_gap_m": dynamic_floor,
        "dynamic_hard_legal": dynamic_hard_legal,
        "contacted_digits": sorted(contact_names),
        "simultaneous_thumb_and_other_observed": opposed_contact,
        "hold_simultaneous_thumb_and_other_observed": hold_opposed_contact,
        "hold_end_lift_m": hold_end_lift_m,
        "hold_min_lift_m": hold_min_lift_m,
        "held_lift_gate": held_lift,
        "hold_vertical_span_m": hold_vertical_span_m,
        "stable_hold_gate": stable_hold,
        "minimum_sampled_hand_object_gap_m": minimum_object_gap_m,
        "object_penetration_gate": object_penetration_safe,
        "peak_normal_force_n": peak_normal_force_n,
        "physics_gate_pass": gate,
        "rank_score": rank_score,
    })
    return result


def json_value(value: object) -> object:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main() -> int:
    args = parse_args()
    if min(
        args.finger_kp, args.finger_kv, args.finger_cap, args.settle_s,
        args.approach_s, args.squeeze_s, args.lift_s, args.hold_s, args.lift_m,
        args.max_object_penetration_m,
    ) <= 0.0 or args.gap_stride <= 0 or args.limit < 0:
        raise ValueError("all controller, duration, lift and screening values must be positive")
    candidate_path = require_file(args.candidate, "candidate")
    scene_path = require_file(args.scene, "scene")
    object_reference = require_file(args.object_reference, "object reference")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sweep output: {output}")

    with np.load(candidate_path, allow_pickle=False) as values:
        candidate_valid = np.asarray(values["candidate_valid"], dtype=bool)
    indices = np.flatnonzero(candidate_valid).astype(np.int64).tolist()
    if args.candidate_index is not None:
        if args.limit:
            raise ValueError("--limit and --candidate-index are mutually exclusive")
        requested = list(dict.fromkeys(args.candidate_index))
        finite = set(indices)
        missing = [index for index in requested if index not in finite]
        if missing:
            raise ValueError(f"requested candidate indices are not finite: {missing}")
        indices = requested
    elif args.limit:
        indices = indices[:args.limit]
    if not indices:
        raise ValueError("candidate file has no finite proposal to audit")

    impedance = (
        (args.finger_kp, args.finger_kv, args.finger_cap)
        if args.physics_profile == "current" else None
    )
    backend = MujocoReplayBackend(
        scene_path, object_joint_name="right_object_joint", hand_order=("right",),
        seed=args.seed,
        finger_impedance=impedance,
    )
    require_scene(backend)
    physics_profile = apply_physics_profile(
        backend, args.physics_profile, args.deximit_object_mass,
    )
    object_qpos = load_object_pose(
        object_reference, expected_nq=backend.model.nq,
        object_address=backend.object_qpos_address,
    )
    human_reference = None
    manual_label = None
    motion_transform = None
    motion_contract = {
        "label_source": "diagnostic_vertical_lift",
        "vertical_lift_m": args.lift_m,
    }
    if args.motion_mode == "source":
        if args.human_reference is None or args.manual_label is None:
            raise ValueError("source motion requires --human-reference and --manual-label")
        human_reference = require_file(args.human_reference, "human reference")
        manual_label = require_file(args.manual_label, "manual label")
        motion_transform, motion_contract = source_motion_contract(
            human_reference, manual_label, object_qpos,
        )
    elif args.human_reference is not None or args.manual_label is not None:
        raise ValueError("human reference and manual label are only valid for source motion")
    pairs = explicit_collision_pairs(
        backend.model, backend.mujoco, object_geom_ids=backend.object_geom_ids,
        hand_sides=("right",),
    )
    started = time.monotonic()
    rows: list[dict[str, object]] = []
    for number, index in enumerate(indices, start=1):
        row = evaluate_one(
            backend, candidate_path=candidate_path, index=index, object_qpos=object_qpos,
            pairs=pairs, motion_transform=motion_transform, args=args,
        )
        rows.append(row)
        if number == 1 or number % 10 == 0 or number == len(indices):
            passed = sum(bool(item["physics_gate_pass"]) for item in rows)
            print(
                f"checked {number}/{len(indices)}; physics gates passed={passed}; "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    ranked = sorted(rows, key=lambda item: float(item["rank_score"]), reverse=True)
    passed = [row for row in ranked if bool(row["physics_gate_pass"])]
    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "scene": str(scene_path),
        "scene_sha256": sha256(scene_path),
        "object_reference": str(object_reference),
        "object_reference_sha256": sha256(object_reference),
        "human_reference": str(human_reference) if human_reference else None,
        "human_reference_sha256": sha256(human_reference) if human_reference else None,
        "manual_label": str(manual_label) if manual_label else None,
        "manual_label_sha256": sha256(manual_label) if manual_label else None,
        "motion_contract": motion_contract,
        "selection_contract": {
            "render_before_selection": False,
            "source_candidate_indices": indices,
            "static_all_finite_candidates_checked": len(indices),
            "static_failures_skip_dynamic_rollout": True,
            "dynamic_rollout": "hand actuators only; free object; no object control",
            "gate": (
                "static and sampled dynamic self/floor safety; simultaneous thumb-plus-other "
                "contact during closure/lift and during hold; at least 20 mm lift retained; "
                "hold vertical span at most 10 mm; sampled hand/object penetration at most "
                f"{1000.0 * args.max_object_penetration_m:g} mm"
            ),
            "winner_verification": (
                "the top candidate must be rerun with full per-step trace and replay "
                "validation before it can be rendered"
            ),
            "kinematic_mapping": (
                "named BODex-to-MuJoCo joints; negate index abduction because the "
                "published URDF axes have opposite signs; project static targets and "
                "initial qpos into current MuJoCo joint limits"
            ),
        },
        "parameters": {
            key: getattr(args, key) for key in (
                "seed", "finger_kp", "finger_kv", "finger_cap", "settle_s",
                "approach_s", "squeeze_s", "lift_s", "hold_s", "lift_m", "motion_mode", "gap_stride",
                "closure_stage", "max_object_penetration_m",
            )
        },
        "physics_profile": physics_profile,
        "counts": {
            "finite_evaluated": len(rows),
            "static_hard_legal": sum(bool(row["static_hard_legal"]) for row in rows),
            "dynamic_completed": sum(row["dynamic_rollout"] == "completed" for row in rows),
            "physics_gate_passed": len(passed),
        },
        "selected_candidate_index": int(ranked[0]["candidate_index"]),
        "selected_candidate_passes_gate": bool(ranked[0]["physics_gate_pass"]),
        "ranked_candidates": ranked,
        "elapsed_s": time.monotonic() - started,
        "limitations": [
            "This is a BODex-only diagnostic ranking, not a formal controller or renderer result.",
            "Dynamic signed distances are densely sampled for screening; the selected winner requires a full per-step replay-checked rerun.",
            "The scene camera is intentionally absent from this program: no rendering occurs before selection.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "counts": report["counts"],
        "selected_candidate_index": report["selected_candidate_index"],
        "selected_candidate_passes_gate": report["selected_candidate_passes_gate"],
        "elapsed_s": report["elapsed_s"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
