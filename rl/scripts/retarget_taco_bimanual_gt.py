"""Prepare explicitly selected TACO GT and run the isolated two-hand MINK adapter."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]

import yaml
from egoengine_repro.retarget.taco_bimanual import prepare, retarget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/taco_brush_bimanual_gt_v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/taco_v1/dev4")
    parser.add_argument("--sequence", default="(brush, brush, bowl)/20230927_027")
    parser.add_argument("--episode", default="taco_brush_brush_bowl_20230927_027")
    parser.add_argument("--mano-model-dir", type=Path,
                        default=ROOT / "data/taco_v1/hand_poses_v1/mano_v1_2/models")
    parser.add_argument("--scene", type=Path, default=ROOT / "models/taco_xhand/xhand/bimanual/taco_brush_brush_bowl_20230927_027/scene_source_contacts_mass.xml")
    args = parser.parse_args()
    scene = args.scene
    human = prepare(args.data_root, scene, args.sequence, args.episode, args.output,
                    mano_model_dir=args.mano_model_dir)
    if not args.prepare_only:
        settings = yaml.safe_load((ROOT / "src/egoengine_repro/configs/paper_faithful_taco.yaml").read_text())["retarget"]
        print(json.dumps(retarget(scene, human, settings, args.output), indent=2))


if __name__ == "__main__":
    main()
