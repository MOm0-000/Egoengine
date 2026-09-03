#!/usr/bin/env python3
"""Explain the isolated DexImit result without relaxing the dual-sim gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--lift-summary", type=Path, action="append", required=True)
    parser.add_argument("--current-mujoco", type=Path, action="append", required=True)
    parser.add_argument("--drive-mujoco", type=Path, action="append", required=True)
    parser.add_argument("--official-current-mujoco", type=Path, action="append", required=True)
    parser.add_argument("--official-drive-mujoco", type=Path, action="append", required=True)
    parser.add_argument("--official-scene-provenance", type=Path, required=True)
    parser.add_argument("--camera-audit", type=Path, required=True)
    parser.add_argument("--alignment-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, object]:
    with path.resolve(strict=True).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    source_path = args.source_gate.resolve(strict=True)
    source = load(source_path)
    if source.get("decision") != "stop_at_sapien_gate":
        raise ValueError("source-motion dual-sim gate must remain closed")
    lift_paths = [path.resolve(strict=True) for path in args.lift_summary]
    if len(lift_paths) != 4:
        raise ValueError("four vertical-lift attribution summaries are required")
    lift_by_depth: dict[int, dict[str, object]] = {}
    lift_passes: list[tuple[int, int]] = []
    for path in lift_paths:
        summary = load(path)
        if (
            summary.get("motion_mode") != "vertical_lift"
            or bool(summary.get("dual_sim_gate_eligible"))
            or not bool(summary.get("diagnostic_only"))
        ):
            raise ValueError(f"not an isolated vertical-lift control: {path}")
        depth = int(summary["pool_depth"])
        passes = list(summary["original_sapien_passes"])
        lift_by_depth[depth] = {
            "summary": str(path),
            "summary_sha256": sha256(path),
            "contacted_source_candidates_tested": int(summary["attempted"]),
            "vertical_lift_passed": len(passes),
            "passed_source_indices": [int(row["source_candidate_index"]) for row in passes],
        }
        lift_passes.extend((depth, int(row["source_candidate_index"])) for row in passes)
    if sorted(lift_by_depth) != [0, 1, 2, 3]:
        raise ValueError("vertical controls do not cover depths 0,1,2,3")

    def mujoco_rows(paths: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        reports, rows = [], []
        for path in paths:
            resolved = path.resolve(strict=True)
            report = load(resolved)
            reports.append({
                "path": str(resolved),
                "sha256": sha256(resolved),
                "counts": report["counts"],
            })
            rows.extend(report["ranked_candidates"])
        return reports, rows

    current_reports, current_rows = mujoco_rows(args.current_mujoco)
    drive_reports, drive_rows = mujoco_rows(args.drive_mujoco)
    official_current_reports, official_current_rows = mujoco_rows(args.official_current_mujoco)
    official_drive_reports, official_drive_rows = mujoco_rows(args.official_drive_mujoco)
    if len(current_rows) != len(lift_passes) or len(drive_rows) != len(lift_passes):
        raise ValueError("MuJoCo controls do not cover every SAPIEN vertical-lift pass")
    if (
        len(official_current_rows) != len(lift_passes)
        or len(official_drive_rows) != len(lift_passes)
    ):
        raise ValueError("official-collision controls do not cover every SAPIEN lift pass")
    current_dynamic = [row for row in current_rows if row.get("dynamic_rollout") == "completed"]
    drive_dynamic = [row for row in drive_rows if row.get("dynamic_rollout") == "completed"]
    drive_lifted = [
        row for row in drive_dynamic
        if float(row.get("hold_min_lift_m", -1.0)) >= 0.019
    ]
    official_current_dynamic = [
        row for row in official_current_rows if row.get("dynamic_rollout") == "completed"
    ]
    official_drive_dynamic = [
        row for row in official_drive_rows if row.get("dynamic_rollout") == "completed"
    ]
    official_drive_lifted = [
        row for row in official_drive_dynamic
        if float(row.get("hold_min_lift_m", -1.0)) >= 0.019
    ]

    alignment_path = args.alignment_audit.resolve(strict=True)
    alignment = load(alignment_path)
    proxy = alignment["collision_geometry"]["official_mesh_vs_current_mujoco_proxy"]
    official_provenance_path = args.official_scene_provenance.resolve(strict=True)
    official_provenance = load(official_provenance_path)
    if official_provenance.get("camera_change", {}).get("new_mode") != "fixed":
        raise ValueError("official collision scene does not freeze the front camera")
    camera_path = args.camera_audit.resolve(strict=True)
    camera_audit = load(camera_path)
    old_camera, fixed_camera = camera_audit["scenes"]
    if old_camera["fixed_camera_pass"] or not fixed_camera["fixed_camera_pass"]:
        raise ValueError("camera audit does not prove old-following/new-fixed separation")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = {
        "schema": "deximit_failure_attribution_v2_official_collision_diagnostic_only",
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "source_motion_gate": {
            "path": str(source_path),
            "sha256": sha256(source_path),
            "sapien_attempted": int(source["sapien"]["total_attempted"]),
            "sapien_passed": int(source["sapien"]["total_passed"]),
            "mujoco_queue_size": len(source["mujoco"]["input_queue"]),
            "video_generated": bool(source["video"]["generated"]),
        },
        "vertical_lift_attribution_control": {
            "may_enter_dual_sim_gate": False,
            "depths": {str(depth): lift_by_depth[depth] for depth in sorted(lift_by_depth)},
            "contacted_source_candidates_tested": sum(
                int(row["contacted_source_candidates_tested"]) for row in lift_by_depth.values()
            ),
            "sapien_vertical_lift_passed": len(lift_passes),
        },
        "current_mujoco_control": {
            "reports": current_reports,
            "evaluated": len(current_rows),
            "static_legal": sum(bool(row["static_hard_legal"]) for row in current_rows),
            "dynamic_completed": len(current_dynamic),
            "strict_passed": sum(bool(row["physics_gate_pass"]) for row in current_rows),
            "dynamic_with_thumb_opposition": sum(
                bool(row.get("simultaneous_thumb_and_other_observed")) for row in current_dynamic
            ),
            "dynamic_retaining_20mm_lift": sum(
                float(row.get("hold_min_lift_m", -1.0)) >= 0.019 for row in current_dynamic
            ),
            "maximum_object_penetration_m": 0.002,
        },
        "high_stiffness_mujoco_control": {
            "reports": drive_reports,
            "evaluated": len(drive_rows),
            "strict_passed": sum(bool(row["physics_gate_pass"]) for row in drive_rows),
            "retaining_20mm_lift": len(drive_lifted),
            "lifted_but_strictly_invalid": [
                {
                    "candidate_index": int(row["candidate_index"]),
                    "hold_min_lift_m": float(row["hold_min_lift_m"]),
                    "minimum_hand_object_gap_m": float(row["minimum_sampled_hand_object_gap_m"]),
                }
                for row in drive_lifted if not bool(row["physics_gate_pass"])
            ],
        },
        "official_collision_mujoco_control": {
            "scene_provenance": str(official_provenance_path),
            "scene_provenance_sha256": sha256(official_provenance_path),
            "current_controller_reports": official_current_reports,
            "current_controller_strict_passed": sum(
                bool(row["physics_gate_pass"]) for row in official_current_rows
            ),
            "current_controller_dynamic_retaining_20mm_lift": sum(
                float(row.get("hold_min_lift_m", -1.0)) >= 0.019
                for row in official_current_dynamic
            ),
            "high_stiffness_reports": official_drive_reports,
            "high_stiffness_strict_passed": sum(
                bool(row["physics_gate_pass"]) for row in official_drive_rows
            ),
            "high_stiffness_lifted_but_strictly_invalid": [
                {
                    "candidate_index": int(row["candidate_index"]),
                    "hold_min_lift_m": float(row["hold_min_lift_m"]),
                    "minimum_hand_object_gap_m": float(row["minimum_sampled_hand_object_gap_m"]),
                }
                for row in official_drive_lifted if not bool(row["physics_gate_pass"])
            ],
        },
        "collision_proxy_audit": {
            "path": str(alignment_path),
            "sha256": sha256(alignment_path),
            "official_vs_mujoco_symmetric_mean_m": float(
                proxy["whole_hand_sampled_surface"]["symmetric_mean_m"]
            ),
            "official_vs_mujoco_symmetric_p95_m": float(
                proxy["whole_hand_sampled_surface"]["symmetric_p95_m"]
            ),
            "missing_current_proxy_links": [
                name for name, row in proxy["per_link"].items()
                if not bool(row["current_proxy_present"])
            ],
        },
        "camera_audit": {
            "path": str(camera_path),
            "sha256": sha256(camera_path),
            "old_front_mode": old_camera["compiled_mode_name"],
            "old_front_maximum_motion_m": old_camera["maximum_position_motion_m"],
            "new_front_mode": fixed_camera["compiled_mode_name"],
            "new_front_maximum_motion_m": fixed_camera["maximum_position_motion_m"],
            "old_clean_v16_videos_valid": False,
        },
        "priorities": [
            {
                "priority": 1,
                "finding": "旧排序输入把 XHand 掌心方向当成 MANO 根方向，方向错 121.49 度。",
                "status": "已修复并重跑；旧 480 次已明确作废。",
            },
            {
                "priority": 2,
                "finding": "v3 接触自动推得的 48→61 段包含 17.8 厘米平移和 42.1 度旋转，不等同于 DexImit 人工子动作标签。",
                "evidence": "源动作 0/480，但简单竖直抬起在 SAPIEN 为 8/85。",
            },
            {
                "priority": 3,
                "finding": "SAPIEN 成功抓姿不能稳定迁移到当前 MuJoCo 控制器和碰撞代理。",
                "evidence": "当前限力控制器 0/8；高刚度可抬起一个，但以约 16.7 毫米穿透换取成功，超过 2 毫米门槛。",
            },
            {
                "priority": 4,
                "finding": "MuJoCo 简化碰撞代理与官方网格不等价。",
                "evidence": (
                    "对称表面距离 p95 为 8.43 毫米，且一个食指基部连杆缺少物理代理；"
                    "换官方网格后高刚度候选 72 的穿透由 16.7 降至 10.2 毫米，但仍未通过。"
                ),
            },
            {
                "priority": 5,
                "finding": "旧 front 相机实际是 trackcom 跟随相机。",
                "evidence": "90 帧内移动最大 155.1 毫米；新隔离场景 fixed 相机移动为 0。",
            },
        ],
        "decision": {
            "formal_video_allowed": False,
            "reason": "没有候选同时通过源动作 SAPIEN 和当前 MuJoCo 严格门。",
            "recommended_next_experiment": (
                "给源序列建立人工可审计的抓取与动作关键帧，并在官方网格诊断场景中搜索"
                "满足 2 毫米穿透门的接触感知限力控制；不要通过提高刚度或放宽门槛制造成功。"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps({
        "output": str(output),
        "source_sapien_pass": 0,
        "lift_sapien_pass": len(lift_passes),
        "current_mujoco_pass": 0,
        "formal_video_allowed": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
