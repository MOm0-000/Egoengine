"""Minimal stage CLI; later work packages extend this without bypassing artifact boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingest.egodex import DEFAULT_ROOT, find_episode, ingest_episode, scan_episodes
from .ingest.egodex_ground_truth import validate_camera_direction_oracle
from .eval.egodex import evaluate_wilor
from .eval.metrics import build_run_report
from .export.spider import export_spider_dataset
from .export.spider_runner import run_spider_chain
from .optimization.sequence import optimize_run
from .visualization import render_run_visualizations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-to-spider")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan-egodex")
    scan.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ingest.add_argument("--task", required=True)
    ingest.add_argument("--episode-id", required=True)
    ingest.add_argument("--output-dir", type=Path, required=True)
    ingest.add_argument("--start-frame", type=int, default=0)
    ingest.add_argument("--end-frame", type=int)
    ingest.add_argument("--overwrite", action="store_true")
    oracle = commands.add_parser("oracle-validate-extrinsics")
    oracle.add_argument("--run-dir", type=Path, required=True)
    oracle.add_argument("--frame-index", type=int)
    oracle.add_argument("--uses-ground-truth", action="store_true", required=True)
    evaluate = commands.add_parser("evaluate-wilor")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--confidence-threshold", type=float, default=0.0)
    evaluate.add_argument("--uses-ground-truth", action="store_true", required=True)
    optimize = commands.add_parser("optimize")
    optimize.add_argument("--run-dir", type=Path, required=True)
    optimize.add_argument("--smoothing-strength", type=float, default=18.0)
    optimize.add_argument("--hand-smoothing-strength", type=float, default=5.0)
    optimize.add_argument("--overwrite", action="store_true")
    export = commands.add_parser("export-spider")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--dataset-root", type=Path)
    export.add_argument("--spider-package-root", type=Path, required=True)
    export.add_argument("--task")
    export.add_argument("--data-id", type=int, default=0)
    export.add_argument("--embodiment-type", choices=["right", "left", "bimanual"], default="bimanual")
    export.add_argument("--hand-sides", nargs="+", choices=["left", "right"], default=["left", "right"])
    export.add_argument("--robot-type", default="xhand")
    spider = commands.add_parser("run-spider")
    spider.add_argument("--dataset-root", type=Path, required=True)
    spider.add_argument("--spider-root", type=Path, required=True)
    spider.add_argument("--task", required=True)
    spider.add_argument("--data-id", type=int, default=0)
    spider.add_argument("--embodiment-type", choices=["right", "left", "bimanual"], default="bimanual")
    spider.add_argument("--robot-type", default="xhand")
    spider.add_argument("--gpu", type=int, default=0)
    spider.add_argument("--no-mjwp", action="store_true")
    spider.add_argument("--no-save-video", action="store_true")
    spider.add_argument("--ik-end-idx", type=int, default=-1)
    report = commands.add_parser("evaluate-run")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--spider-report", type=Path)
    visualize = commands.add_parser("visualize-run")
    visualize.add_argument("--run-dir", type=Path, required=True)
    visualize.add_argument("--max-side", type=int, default=960)
    visualize.add_argument("--mesh-duration-s", type=float, default=4.0)
    visualize.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "scan-egodex":
        episodes = list(scan_episodes(args.root))
        print(json.dumps({"root": str(args.root), "episode_count": len(episodes)}, indent=2))
        return 0
    if args.command == "oracle-validate-extrinsics":
        print(validate_camera_direction_oracle(args.run_dir, frame_index=args.frame_index))
        return 0
    if args.command == "evaluate-wilor":
        print(evaluate_wilor(args.run_dir, confidence_threshold=args.confidence_threshold))
        return 0
    if args.command == "optimize":
        print(optimize_run(
            args.run_dir, smoothing_strength=args.smoothing_strength,
            hand_smoothing_strength=args.hand_smoothing_strength, overwrite=args.overwrite,
        ))
        return 0
    if args.command == "export-spider":
        run_dir = args.run_dir.resolve()
        source = json.loads((run_dir / "input/source.json").read_text(encoding="utf-8"))
        selected = json.loads((run_dir / "object_tracking/selected_mesh.json").read_text(encoding="utf-8"))
        optimization_metrics = json.loads(
            (run_dir / "optimization/optimization_metrics.json").read_text(encoding="utf-8")
        )
        hand_roles = optimization_metrics.get("hands", {}).get("roles")
        dataset_root = args.dataset_root or run_dir / "spider_export/dataset"
        print(export_spider_dataset(
            aligned_path=run_dir / "optimization/aligned_trajectory.npz",
            contact_path=run_dir / "optimization/contact.npz",
            visual_mesh_path=run_dir / selected["canonical_visual_mesh"],
            dataset_root=dataset_root, task=args.task or source["task_directory"],
            data_id=args.data_id, source_run_id=run_dir.name, hand_sides=args.hand_sides,
            spider_package_root=args.spider_package_root,
            hand_roles=hand_roles,
            embodiment_type=args.embodiment_type, robot_type=args.robot_type,
        ))
        return 0
    if args.command == "run-spider":
        print(run_spider_chain(
            dataset_root=args.dataset_root, spider_root=args.spider_root, task=args.task,
            data_id=args.data_id, embodiment_type=args.embodiment_type,
            robot_type=args.robot_type, gpu=args.gpu, run_mjwp=not args.no_mjwp,
            save_video=not args.no_save_video, ik_end_idx=args.ik_end_idx,
        ))
        return 0
    if args.command == "evaluate-run":
        print(build_run_report(args.run_dir, spider_report=args.spider_report))
        return 0
    if args.command == "visualize-run":
        print(render_run_visualizations(
            args.run_dir, overwrite=args.overwrite, max_side=args.max_side,
            mesh_duration_s=args.mesh_duration_s,
        ))
        return 0
    episode = find_episode(args.root, args.task, args.episode_id)
    path = ingest_episode(
        episode, args.output_dir, start_frame=args.start_frame, end_frame=args.end_frame,
        overwrite=args.overwrite,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
