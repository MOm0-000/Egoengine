"""Write a non-rendering collision audit without modifying the scene or trace."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np
from egoengine_repro.retarget.collision_audit import (
    audit_intrahand_trajectory, audit_trajectory, source_topology_report,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    with np.load(args.reference, allow_pickle=False) as data:
        qpos = data["qpos"]
        report = audit_trajectory(model, qpos)
        report["intrahand"] = audit_intrahand_trajectory(model, qpos)
    report["source_topology"] = source_topology_report(
        ROOT / "models/taco_xhand/templates/xhand_bimanual_source.xml")
    report["inputs"] = [dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                        for p in (args.scene, args.reference)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("pair_count", "min_distance_m", "penetrating_frames")}
                      for k, v in report.items() if isinstance(v, dict) and "pair_count" in v}, indent=2))


if __name__ == "__main__":
    main()
