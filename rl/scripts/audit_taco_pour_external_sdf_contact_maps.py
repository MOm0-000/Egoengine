#!/usr/bin/env python3
"""Verify that the SDF contact geoms reach the intended RL contact channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "semantic_sdf_right_thumb_rota_link2": (0, 0),
    "semantic_sdf_right_index_rota_link2": (1, 0),
    "semantic_sdf_left_ring_link2": (8, 1),
    "semantic_sdf_left_pinky_link2": (9, 1),
}


def run(scene: Path, output: Path, spider_root: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    sys.path.insert(0, str(spider_root))
    from spider.config import build_force_closure_geom_maps

    model = mujoco.MjModel.from_xml_path(str(scene))
    finger_map, hand_map, object_map = build_force_closure_geom_maps(
        model, "bimanual"
    )
    records = []
    passed = True
    for pair_name, (expected_finger, expected_group) in EXPECTED.items():
        pair_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_PAIR, pair_name
        )
        if pair_id < 0:
            records.append({"pair": pair_name, "missing": True, "passed": False})
            passed = False
            continue
        geom_ids = (int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id]))
        hand_ids = [gid for gid in geom_ids if finger_map[gid] >= 0]
        object_ids = [gid for gid in geom_ids if object_map[gid] >= 0]
        row_passed = (
            len(hand_ids) == 1
            and len(object_ids) == 1
            and finger_map[hand_ids[0]] == expected_finger
            and hand_map[hand_ids[0]] == expected_group
            and object_map[object_ids[0]] == expected_group
        )
        passed &= row_passed
        records.append({
            "pair": pair_name,
            "hand_geom": model.geom(hand_ids[0]).name if len(hand_ids) == 1 else None,
            "object_geom": model.geom(object_ids[0]).name if len(object_ids) == 1 else None,
            "finger_channel": finger_map[hand_ids[0]] if len(hand_ids) == 1 else None,
            "hand_group": hand_map[hand_ids[0]] if len(hand_ids) == 1 else None,
            "object_group": object_map[object_ids[0]] if len(object_ids) == 1 else None,
            "expected_finger_channel": expected_finger,
            "expected_group": expected_group,
            "passed": row_passed,
        })
    report = {
        "schema": "taco_pour_external_sdf_contact_maps_v1",
        "scene": str(scene.resolve()),
        "spider_root": str(spider_root.resolve()),
        "pairs": records,
        "physics_contacts_visible_to_rl": passed,
        "passed": passed,
        "training_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spider-root", type=Path, default=ROOT / "external/spider_compat"
    )
    args = parser.parse_args()
    result = run(args.scene, args.output, args.spider_root)
    raise SystemExit(0 if result["passed"] else 1)
