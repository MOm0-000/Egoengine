"""Recompute matched baseline/candidate metrics for the isolated SPIDER run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]

from egoengine_repro.retarget.mink import _joint_velocity_limits  # noqa: E402
from run_spider_taco_experiment import _metrics, _summary, artifact  # noqa: E402


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path,
                        default=ROOT / "runs/taco_pour_spider_mink_experiment_v1")
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "runs/taco_pour_bimanual_mano_fk_v1")
    parser.add_argument("--scene", type=Path,
                        default=ROOT / (
                            "models/taco_xhand/xhand/bimanual/"
                            "taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
                        ))
    parser.add_argument("--human", type=Path,
                        default=ROOT / "runs/taco_pour_bimanual_mano_fk_v1/human_reference.npz")
    args = parser.parse_args()
    comparison_path = args.experiment / "comparison.json"
    if comparison_path.exists():
        raise FileExistsError(comparison_path)
    human = _load(args.human)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    baseline_report = json.loads((args.baseline / "retarget_report.json").read_text())
    velocity = _joint_velocity_limits(model, mujoco,
                                       baseline_report["inherited_settings"]["velocity_limits"])
    baseline = _load(args.baseline / "robot_reference.npz")
    candidate = _load(args.experiment / "robot_reference.npz")
    if not np.array_equal(baseline["frame_indices"], candidate["frame_indices"]):
        raise ValueError("baseline and candidate frame rows differ")
    baseline_metrics = _metrics(model, baseline["qpos"], human, velocity, args.scene)
    candidate_metrics = _metrics(model, candidate["qpos"], human, velocity, args.scene)
    baseline_summary = _summary(baseline_metrics)
    candidate_summary = _summary(candidate_metrics)
    fields = (
        "fingertip_mean_error_mm_by_hand",
        "fingertip_orientation_mean_deg_by_hand",
        "wrist_position_mean_mm_by_hand",
        "wrist_orientation_mean_deg_by_hand",
        "self_collision_min_distance_m",
        "joint_limit_min_margin",
        "frame_velocity_max_ratio",
        "strict_gate_passed",
    )
    delta = {}
    for field in fields:
        old, new = baseline_summary[field], candidate_summary[field]
        if isinstance(old, list):
            delta[field] = (np.asarray(new, dtype=float) - np.asarray(old, dtype=float)).tolist()
        elif isinstance(old, (int, float)) and isinstance(new, (int, float)):
            delta[field] = float(new - old)
        else:
            delta[field] = {"baseline": old, "candidate": new, "changed": old != new}
    result = {
        "status": "matched_recomputed_comparison",
        "baseline": {"run": str(args.baseline.resolve()), "summary": baseline_summary},
        "candidate": {"run": str(args.experiment.resolve()), "summary": candidate_summary},
        "candidate_minus_baseline": delta,
        "same_inputs": [artifact(args.human), artifact(args.scene)],
        "same_frame_rows": int(len(candidate["qpos"])),
        "interpretation": (
            "Negative fingertip position delta is a lower residual. It is not a "
            "replacement recommendation when orientation or physical collision "
            "gates regress. Full collision families are diagnostics; explicit "
            "self-collision, joint, and frame-velocity gates are reported separately."
        ),
    }
    comparison_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
