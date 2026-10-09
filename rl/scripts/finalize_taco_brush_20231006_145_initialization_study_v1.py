#!/usr/bin/env python3
"""Finalize the ordered, read-only initialization feasibility study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
SCHEMA = "taco_brush_20231006_145_initialization_study_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_20231006_145_initialization_study_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected study schema")
    forbidden = (
        "modify_original_data", "modify_active_support_contract", "run_mink",
        "run_physics", "run_replay", "run_mpc", "run_rl", "promotion",
    )
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("summary stage is not fail-closed")

    paths = {key: Path(value).resolve(strict=(key != "output"))
             for key, value in cfg["paths"].items()}
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source = json.loads(paths["source_geometry"].read_text(encoding="utf-8"))
    xhand = json.loads(paths["xhand_kinematics"].read_text(encoding="utf-8"))
    seating = json.loads(paths["natural_seating"].read_text(encoding="utf-8"))
    replay = json.loads(paths["replay"].read_text(encoding="utf-8"))
    if source["schema"] != "taco_brush_20231006_145_initialization_feasibility_v1":
        raise ValueError("source geometry evidence schema changed")
    if xhand["schema"] != "taco_brush_20231006_145_safe_entry_mink_v1":
        raise ValueError("XHand evidence schema changed")
    if seating["schema"] != "taco_brush_20231006_145_natural_seating_extension_v1":
        raise ValueError("natural seating evidence schema changed")
    if replay["schema"] != "taco_brush_20231006_145_initialization_controlled_replay_v1":
        raise ValueError("Replay evidence schema changed")

    safe = xhand["mink_candidates"]["safe_entry"]
    direct = xhand["mink_candidates"]["direct_crop"]
    object_settling_passed = bool(
        seating["decision"]["object_settling_passed_all_variants"]
    )
    first_tracking = replay["tracking_failures"][0] if replay["tracking_failures"] else None
    table_blockers = [row for row in replay["native_blockers"]
                      if row["entity"].endswith("xhand_table")]
    earliest_table_blocker = min(
        (int(row["endpoint"]) for row in table_blockers), default=None,
    )
    deepest_table_blocker = min(
        table_blockers, key=lambda row: float(row.get("distance_m", 0.0)),
        default=None,
    )
    classification = (
        "STATIC_INITIALIZATION_FEASIBLE_EARLY_PHYSICS_EXECUTION_INVALID_"
        "REPLAY_FAILED_CLOSED"
    )
    result = {
        "schema": SCHEMA,
        "classification": classification,
        "sample": cfg["sample"],
        "ordered_gates": {
            "1_mano_penetration_scope": {
                "status": "LOCAL_FINGER_TABLE_PENETRATION_CONFIRMED",
                "frame0": source["frame0"],
                "initial_action": source["initial_action"],
                "interpretation": (
                    "Frame-0 penetration is on fingers, not palm/wrist; left clears "
                    "after frame 5 and right after frame 10."
                ),
            },
            "2_direct_legal_start": {
                "status": "SOURCE_FRAME_11_LEGAL_BUT_CROPPED_NOT_COMPLETE_REPRODUCTION",
                "contract": source["direct_start"],
                "fresh_mink_status": direct,
            },
            "3_safe_entry": {
                "status": "SAFE_ENTRY_REFERENCE_AND_MINK_PASS",
                "contract": xhand["safe_entry"],
                "candidate": safe,
            },
            "4_xhand_environment_constraints": {
                "status": "INITIALIZATION_PASS_WITH_LATER_INTERACTION_GEOMETRY_RISK",
                "native_initialization_pass": safe["initialization_native_geometry_pass"],
                "selected_interaction_observations": safe[
                    "interaction_native_hand_object_observations"
                ],
            },
            "5_physics_replay": {
                "status": replay["classification"],
                "object_settling_passed_all_variants": object_settling_passed,
                "completed_endpoint": replay["completed_outcome_endpoint"],
                "hard_stop": replay["hard_stop"],
                "pickup": replay["pickup"],
                "first_tracking_failure": first_tracking,
                "tracking_failure_count": len(replay["tracking_failures"]),
                "native_blocker_count": len(replay["native_blockers"]),
                "earliest_xhand_table_blocker_endpoint": earliest_table_blocker,
                "deepest_xhand_table_blocker": deepest_table_blocker,
                "maximums": replay["maximums"],
                "proxy_native_geometry_mismatch_count": len(
                    replay["proxy_native_geometry_mismatches"]
                ),
                "proxy_native_force_exposure_mismatch_count": len(
                    replay["proxy_native_force_exposure_mismatches"]
                ),
            },
        },
        "provenance_contract": {
            "support": "Issue #14 global official-object minimum; unpromoted candidate",
            "robot_initial_state": "safe-entry fresh MINK endpoint 0",
            "object_initial_state": (
                "10-second isolated natural seating, zero-initial-velocity variant"
            ),
            "reference_actions": (
                "safe-entry MINK controls; official per-frame object reference unmodified"
            ),
            "active_support_modified": False,
            "original_data_modified": False,
            "rl_runs": 0,
            "mpc_runs": 0,
            "promotion_authorized": False,
        },
        "decision": {
            "static_initialization_feasible": True,
            "controlled_replay_strict_pass": False,
            "brush_stable_pickup": replay["pickup"]["brush"]["stable_pickup_pass"],
            "bowl_stable_pickup": replay["pickup"]["bowl"]["stable_pickup_pass"],
            "replay_or_rl_promotion_authorized": False,
        },
        "artifacts": {key: artifact(path) for key, path in paths.items() if key != "output"},
        "config": artifact(config_path),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    write_json(output / "results.json", result)

    deepest_mm = (
        "n/a" if deepest_table_blocker is None
        else f"{1000 * float(deepest_table_blocker['distance_m']):.3f}"
    )
    lines = [
        "# 20231006_145 initialization feasibility study", "",
        f"- Classification: `{classification}`",
        "- Static initialization: `FEASIBLE WITH SAFE ENTRY`",
        "- Controlled Replay: `FAILED CLOSED`",
        "- Original data / active support modified: `NO / NO`",
        "- RL / MPC / promotion: `NOT RUN / NOT RUN / FORBIDDEN`", "",
        "## Ordered results", "",
        "| gate | result | decisive evidence |", "|---|---|---|",
        "| 1. MANO penetration scope | local finger issue | frame 0: right thumb/middle/ring/pinky and left pinky; no palm/wrist region |",
        "| 2. legal source start | frame 11 exists, but cropped | deletes frames 0–10 (0.367 s); cannot be called full reproduction |",
        f"| 3. DexTrack-style safe entry | PASS | 30 mm retreat, exact join at frame 11; safe-entry MINK `{safe['status']}` |",
        f"| 4. XHand constraints | initialization PASS | table minimum `{1000 * safe['native_hand_table_minimum_m']:.3f} mm`; later selected interaction penetrations `{safe['interaction_native_hand_object_observations']['penetrating_pair_observation_count']}` |",
        f"| 5. controlled Replay | FAIL | brush pickup `{replay['pickup']['brush']['stable_pickup_pass']}`, bowl pickup `{replay['pickup']['bowl']['stable_pickup_pass']}`, first tracking failure endpoint `{first_tracking['endpoint'] if first_tracking else 'none'}` |",
        "", "## Key physical finding", "",
        f"The safe-entry **reference** starts legally, but the physical execution first violates the XHand/table native-geometry gate at endpoint `{earliest_table_blocker}`; the deepest sampled hand/table penetration is `{deepest_mm} mm`. The full bounded Replay reached endpoint 121 without a MuJoCo hard stop, but reaching the horizon is not a pass.",
        "",
        f"The brush lifted only `{1000 * replay['pickup']['brush']['maximum_actual_lift_m']:.3f} mm` versus `{1000 * replay['pickup']['brush']['maximum_reference_lift_m']:.3f} mm` in the reference and never satisfied the stable hand-contact/no-floor pickup gate. The bowl did satisfy that local pickup gate for `{replay['pickup']['bowl']['lifted_hand_contact_no_floor_longest_run']}` consecutive intervals.",
        "",
        "Therefore the source-frame initialization defect can be repaired kinematically without cropping, but the resulting low-level physical execution is not valid: early hand/table collision, brush tracking failure, missing brush pickup, native mesh blockers, and proxy/native mismatch remain. No RL is authorized from this evidence.", "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "classification": classification,
        "static_initialization_feasible": True,
        "replay_strict_pass": False,
        "first_tracking_failure": first_tracking,
        "earliest_xhand_table_blocker_endpoint": earliest_table_blocker,
        "pickup": replay["pickup"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
