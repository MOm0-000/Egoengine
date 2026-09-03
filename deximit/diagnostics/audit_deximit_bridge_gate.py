#!/usr/bin/env python3
"""Aggregate the isolated DexImit SAPIEN gate before MuJoCo or rendering."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "deximit_dual_sim_gate_v1_diagnostic_only"
SAPIEN_SCHEMA = "deximit_original_sapien_screen_v4_physical_contact_metrics_diagnostic_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sapien-summary", type=Path, action="append", required=True)
    parser.add_argument("--prompt-audit", type=Path, required=True)
    parser.add_argument("--contact-v3", type=Path, required=True)
    parser.add_argument("--human-reference", type=Path, required=True)
    parser.add_argument("--formal-audit", type=Path, required=True)
    parser.add_argument("--formal-ref", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def q(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    result = np.quantile(np.asarray(values, dtype=np.float64), (0.0, 0.5, 0.95, 1.0))
    return dict(zip(("min", "median", "p95", "max"), map(float, result), strict=True))


def main() -> int:
    args = parse_args()
    summary_paths = [path.resolve(strict=True) for path in args.sapien_summary]
    if len(summary_paths) != 4:
        raise ValueError("exactly four per-depth SAPIEN summaries are required")
    prompt_audit_path = args.prompt_audit.resolve(strict=True)
    contact_path = args.contact_v3.resolve(strict=True)
    human_path = args.human_reference.resolve(strict=True)
    formal_audit = args.formal_audit.resolve(strict=True)
    formal_ref = args.formal_ref.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    with prompt_audit_path.open(encoding="utf-8") as handle:
        prompt_audit = json.load(handle)
    if prompt_audit.get("decision", {}).get("previous_sapien_ranking_valid") is not False:
        raise ValueError("prompt audit does not explicitly invalidate the old ranking")

    by_depth: dict[int, dict[str, object]] = {}
    pass_queue: list[dict[str, object]] = []
    for path in summary_paths:
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        depth = int(summary["pool_depth"])
        if (
            summary.get("schema") != SAPIEN_SCHEMA
            or not bool(summary.get("diagnostic_only"))
            or bool(summary.get("formal_renderer_3_3_eligible"))
            or int(summary.get("attempted", -1)) != 120
            or int(summary.get("ranked_after_ik", -1)) != 120
            or depth in by_depth
        ):
            raise ValueError(f"invalid or incomplete SAPIEN summary: {path}")
        attempts = list(summary["all_attempts"])
        stage_failures: collections.Counter[str] = collections.Counter()
        errors: list[float] = []
        contacts: list[int] = []
        for row in attempts:
            calls = row["right_curobo_stages"]
            failed = next((call["stage"] for call in calls if not call["success"]), None)
            stage_failures[failed or "all_plans_completed"] += 1
            metrics = row["metrics"]
            if not str(metrics.get("contact_step_definition", "")).startswith(
                "load-bearing object contact"
            ):
                raise ValueError("SAPIEN contact counts are not load-bearing contact metrics")
            value = metrics.get("mean_target_vertex_error_m")
            if value is not None:
                errors.append(float(value))
            contacts.append(int(metrics["right_hand_contact_steps"]))
        passes = list(summary["original_sapien_passes"])
        for row in passes:
            target = Path(row["selected_target"]).resolve(strict=True)
            if sha256(target) != row["selected_target_sha256"]:
                raise ValueError(f"SAPIEN pass target hash mismatch: {target}")
            pass_queue.append({
                "depth": depth,
                "source_candidate_index": int(row["source_candidate_index"]),
                "original_rank": int(row["original_rank"]),
                "selected_target": str(target),
                "selected_target_sha256": sha256(target),
                "candidate_pool": str(row["pool"]),
            })
        by_depth[depth] = {
            "summary": str(path),
            "summary_sha256": sha256(path),
            "ik_success": int(summary["ik_success_by_depth"][str(depth)]),
            "attempted": len(attempts),
            "sapien_passed": len(passes),
            "planning_outcomes": dict(stage_failures),
            "target_vertex_error_m": q(errors),
            "candidates_with_right_hand_object_contact": sum(value > 0 for value in contacts),
            "maximum_right_hand_contact_steps": max(contacts),
            "settled_object_bottom_height_m": float(summary["object_bottom_height_m"]),
            "table_height_m": float(summary["table_height_m"]),
        }
    if sorted(by_depth) != [0, 1, 2, 3]:
        raise ValueError("SAPIEN summaries do not cover depths 0,1,2,3")

    with np.load(contact_path, allow_pickle=False) as data:
        states = np.asarray(data["state"], dtype=np.int8)
        hands = [str(value) for value in data["hand_order"]]
        regions = [str(value) for value in data["region_order"]]
    with np.load(human_path, allow_pickle=False) as data:
        object_pose = np.asarray(data["T_sim_object_reference"], dtype=np.float64)[:, 0]
    right = hands.index("right")
    state = states[:, right]
    opposed = (
        (state[:, regions.index("thumb")] == 1)
        & np.any(state[:, [regions.index(name) for name in regions if name not in {"palm", "thumb"}]] == 1, axis=1)
    )
    rows = np.flatnonzero(opposed)
    anchor = int(rows[0])
    end = anchor
    while end + 1 < len(opposed) and opposed[end + 1]:
        end += 1
    centre_motion = float(np.linalg.norm(
        object_pose[end, :3, 3] - object_pose[anchor, :3, 3]
    ))
    rotation_motion = float(Rotation.from_matrix(
        object_pose[end, :3, :3] @ object_pose[anchor, :3, :3].T
    ).magnitude())

    total_passes = len(pass_queue)
    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "input_corrections": {
            "exact_mano_prompt": str(prompt_audit_path),
            "exact_mano_prompt_sha256": sha256(prompt_audit_path),
            "previous_prompt_rotation_error_deg_median": float(
                prompt_audit["checks"]["correct_vs_previous_deximit_prompt_rotation_deg"]["median"]
            ),
        },
        "source_motion_contract": {
            "contact_anchor_row": anchor,
            "continuous_opposition_end_row": end,
            "object_centre_displacement_m": centre_motion,
            "object_rotation_deg": float(np.degrees(rotation_motion)),
            "warning": (
                "This is a v3-contact-derived surrogate because the TACO release does not "
                "provide DexImit's hand-authored grasp/motion subaction labels."
            ),
        },
        "sapien": {
            "definition": (
                "unchanged DexImit rollout: pregrasp, grasp, squeeze, then the source "
                "object motion; mean object-surface motion error <= 20 mm"
            ),
            "depths": {str(depth): by_depth[depth] for depth in sorted(by_depth)},
            "total_attempted": 480,
            "total_passed": total_passes,
        },
        "mujoco": {
            "input_queue": pass_queue,
            "attempted": 0,
            "passed": 0,
            "strict_maximum_hand_object_penetration_m": 0.002,
            "status": (
                "not_run_no_sapien_pass" if total_passes == 0
                else "pending_only_for_listed_sapien_passes"
            ),
        },
        "video": {
            "generated": False,
            "required_view": "fixed front third-person; camera following forbidden",
            "status": (
                "not_generated_no_dual_sim_pass" if total_passes == 0
                else "pending_only_after_mujoco_pass"
            ),
        },
        "formal_chain_identity": {
            "audit": str(formal_audit),
            "audit_sha256": sha256(formal_audit),
            "ref": str(formal_ref),
            "ref_sha256": sha256(formal_ref),
            "modified_by_this_diagnostic": False,
        },
        "decision": (
            "stop_at_sapien_gate" if total_passes == 0
            else "run_current_mujoco_only_on_input_queue"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps({
        "output": str(output),
        "sapien_attempted": 480,
        "sapien_passed": total_passes,
        "mujoco_attempted": 0,
        "video_generated": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
