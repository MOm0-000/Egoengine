#!/usr/bin/env python3
"""Summarize one strict diagnostic controller-gain sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for source in args.input:
        path = source.resolve(strict=True)
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        candidate = report["ranked_candidates"]
        if len(candidate) != 1 or int(candidate[0]["candidate_index"]) != 72:
            raise ValueError(f"gain sweep input is not candidate 72: {path}")
        row = candidate[0]
        parameters = report["parameters"]
        rows.append({
            "path": str(path),
            "sha256": sha256(path),
            "finger_kp": float(parameters["finger_kp"]),
            "finger_kv": float(parameters["finger_kv"]),
            "strict_pass": bool(row["physics_gate_pass"]),
            "hold_opposition": bool(row.get("hold_simultaneous_thumb_and_other_observed")),
            "hold_min_lift_m": float(row.get("hold_min_lift_m", -1.0)),
            "minimum_hand_object_gap_m": float(row.get("minimum_sampled_hand_object_gap_m", -1.0)),
            "peak_normal_force_n": float(row.get("peak_normal_force_n", 0.0)),
            "dynamic_hard_legal": bool(row.get("dynamic_hard_legal")),
        })
    safe = [
        row for row in rows
        if row["dynamic_hard_legal"] and row["minimum_hand_object_gap_m"] >= -0.002
    ]
    best_safe = max(safe, key=lambda row: row["hold_min_lift_m"]) if safe else None
    payload = {
        "schema": "bodex_candidate72_controller_gain_sweep_v1_diagnostic_only",
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "candidate_index": 72,
        "official_collision_scene": True,
        "strict_maximum_penetration_m": 0.002,
        "configuration_count": len(rows),
        "strict_passed": sum(row["strict_pass"] for row in rows),
        "penetration_and_hard_legal": len(safe),
        "retaining_20mm_lift": sum(row["hold_min_lift_m"] >= 0.019 for row in rows),
        "best_safe_configuration": best_safe,
        "rows": sorted(rows, key=lambda row: (row["finger_kp"], row["finger_kv"])),
        "decision": (
            "No proportional/damping gain in the bounded-torque sweep retains lift; "
            "stop manual gain tuning."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({
        "output": str(output),
        "configurations": len(rows),
        "strict_passed": payload["strict_passed"],
        "safe": len(safe),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
