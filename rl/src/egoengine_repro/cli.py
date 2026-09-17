"""Command-line entry points for the isolated EgoEngine reproduction package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .experiments import ExperimentManifest


def _manifest_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-manifest", type=Path)


def _record_stage(
    manifest_path: Path | None, name: str, *, inputs: list[Path], outputs: list[Path],
    metrics: dict | None = None, uses_ground_truth: bool = False,
) -> None:
    if manifest_path is None:
        return
    ExperimentManifest.load(manifest_path).record_stage(
        name, inputs=inputs, outputs=outputs, metrics=metrics,
        uses_ground_truth=uses_ground_truth,
    )


def _existing_source_artifacts(run_dir: Path) -> list[Path]:
    candidates = (
        "manifest.json", "input/source.json", "segmentation/object_masks.npz",
        "segmentation/metadata.json", "hands/wilor_raw.npz", "hands/metadata.json",
        "calibration/intrinsics.npy", "calibration/T_world_camera.npy",
        "depth/metadata.json", "depth/metric_depth.zarr", "mesh_proposals/mesh_ranking.json",
        "object_tracking/foundationpose_raw.npz", "object_tracking/selected_mesh.json",
        "optimization/aligned_trajectory.npz", "optimization/contact.npz",
        "optimization/optimization_metrics.json",
    )
    artifacts = [run_dir / relative for relative in candidates if (run_dir / relative).exists()]
    source_path = run_dir / "input/source.json"
    if source_path.is_file():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        artifacts.extend(
            Path(source[key]).resolve() for key in ("mp4_path", "hdf5_path")
            if source.get(key) and Path(source[key]).is_file()
        )
    selected_mesh_path = run_dir / "object_tracking/selected_mesh.json"
    if selected_mesh_path.is_file():
        selected_mesh = json.loads(selected_mesh_path.read_text(encoding="utf-8"))
        mesh = run_dir / selected_mesh.get("canonical_visual_mesh", "")
        if mesh.is_file():
            artifacts.append(mesh)
    return list(dict.fromkeys(path.resolve() for path in artifacts))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egoengine-repro")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-experiment")
    init.add_argument("--config", required=True)
    init.add_argument("--source-run", type=Path, required=True)
    init.add_argument("--output-dir", type=Path, required=True)
    init.add_argument("--episode-id", required=True)
    init.add_argument("--seeds", nargs="+", type=int, default=[0])
    init.add_argument("--source-artifact", action="append", type=Path, default=[])
    init.add_argument("--repo-root", action="append", type=Path, default=[])

    gt = commands.add_parser("make-egodex-gt")
    gt.add_argument("--source-run", type=Path, required=True)
    gt.add_argument("--output-dir", type=Path, required=True)
    _manifest_argument(gt)

    evaluation_set = commands.add_parser("materialize-evaluation-set")
    evaluation_set.add_argument("--evaluation-set", default="egodex_hand_camera_20")
    evaluation_set.add_argument("--output-dir", type=Path, required=True)
    evaluation_set.add_argument("--force", action="store_true")

    taco_set = commands.add_parser("materialize-taco-set")
    taco_set.add_argument("--taco-set", default="taco_object_gt_16")
    taco_set.add_argument("--output-dir", type=Path, required=True)
    taco_set.add_argument("--force", action="store_true")

    taco_rgb = commands.add_parser("extract-taco-rgb")
    taco_rgb.add_argument("--taco-set", default="taco_first_person_dev4")
    taco_rgb.add_argument("--output-dir", type=Path, required=True)
    taco_rgb.add_argument("--force", action="store_true")

    taco_ingest = commands.add_parser("ingest-taco-devset")
    taco_ingest.add_argument("--taco-set", default="taco_first_person_dev4")
    taco_ingest.add_argument("--video-dir", type=Path, required=True)
    taco_ingest.add_argument("--output-dir", type=Path, required=True)
    taco_ingest.add_argument("--force", action="store_true")

    taco_mesh = commands.add_parser("prepare-taco-known-mesh")
    taco_mesh.add_argument("--source-run", type=Path, required=True)
    taco_mesh.add_argument("--ground-truth-manifest", type=Path, required=True)
    taco_mesh.add_argument("--overwrite", action="store_true")
    _manifest_argument(taco_mesh)

    taco_base = commands.add_parser("freeze-taco-base-frame")
    taco_base.add_argument("--taco-set", default="taco_first_person_dev4")
    taco_base.add_argument("--episode-id", required=True)
    taco_base.add_argument("--output", type=Path, required=True)
    taco_base.add_argument("--overwrite", action="store_true")
    _manifest_argument(taco_base)

    sam2 = commands.add_parser("run-sam2-paper")
    sam2.add_argument("--source-run", type=Path, required=True)
    sam2.add_argument("--hand-ground-truth", type=Path, required=True)
    sam2.add_argument("--checkpoint", type=Path, required=True)
    sam2.add_argument(
        "--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    sam2.add_argument("--object-point", nargs=2, type=float, required=True, metavar=("X", "Y"))
    sam2.add_argument("--prompt-frame", type=int, default=0)
    sam2.add_argument("--output-dir", type=Path)
    sam2.add_argument("--confidence-threshold", type=float, default=0.0)
    sam2.add_argument("--overwrite", action="store_true")
    sam2.add_argument("--dry-run", action="store_true")
    _manifest_argument(sam2)

    taco_object_ref = commands.add_parser("prepare-taco-object-reference")
    taco_object_ref.add_argument("--source-run", type=Path, required=True)
    taco_object_ref.add_argument("--t-sim-world", type=Path, required=True)
    taco_object_ref.add_argument("--output", type=Path, required=True)
    taco_object_ref.add_argument("--overwrite", action="store_true")
    _manifest_argument(taco_object_ref)

    taco_ab = commands.add_parser("run-taco-ablation")
    taco_ab.add_argument("--ablation", default="taco_perception_ablation_dev4")
    taco_ab.add_argument("--ingest-manifest", type=Path, required=True)
    taco_ab.add_argument("--ground-truth-set-manifest", type=Path, required=True)
    taco_ab.add_argument("--output-dir", type=Path, required=True)
    taco_ab.add_argument("--profile", action="append")
    taco_ab.add_argument("--gpu", type=int, default=6)
    taco_ab.add_argument("--prepare-only", action="store_true")
    taco_ab.add_argument("--force", action="store_true")

    auto_seg = commands.add_parser("run-taco-auto-segmentation-v2")
    auto_seg.add_argument("--spec", type=Path)
    auto_seg.add_argument("--source-ablation-manifest", type=Path, required=True)
    auto_seg.add_argument("--output-dir", type=Path, required=True)
    auto_seg.add_argument("--gpu", type=int, default=6)
    auto_seg.add_argument("--prepare-only", action="store_true")
    auto_seg.add_argument("--force", action="store_true")

    depth_cal = commands.add_parser("run-taco-depth-calibration")
    depth_cal.add_argument("--spec", type=Path)
    depth_cal.add_argument("--segmentation-manifest", type=Path, required=True)
    depth_cal.add_argument("--output-dir", type=Path, required=True)
    depth_cal.add_argument("--prepare-only", action="store_true")
    depth_cal.add_argument("--gpu", type=int)

    oracle_depth = commands.add_parser("render-taco-oracle-depth")
    oracle_depth.add_argument("--source-run", type=Path, required=True)
    oracle_depth.add_argument("--ground-truth-manifest", type=Path, required=True)
    oracle_depth.add_argument("--overwrite", action="store_true")

    camera_proxy = commands.add_parser("derive-taco-camera-proxy")
    camera_proxy.add_argument("--ground-truth-manifest", type=Path, required=True)
    camera_proxy.add_argument("--wilor-artifact", type=Path, required=True)
    camera_proxy.add_argument("--intrinsics", type=Path, required=True)
    camera_proxy.add_argument("--output-dir", type=Path, required=True)
    camera_proxy.add_argument("--overwrite", action="store_true")

    oracle_ab = commands.add_parser("run-taco-oracle-depth-ablation")
    oracle_ab.add_argument("--spec", type=Path)
    oracle_ab.add_argument("--source-ablation-manifest", type=Path, required=True)
    oracle_ab.add_argument("--output-dir", type=Path, required=True)
    oracle_ab.add_argument("--gpu", type=int, default=6)
    oracle_ab.add_argument("--prepare-only", action="store_true")
    oracle_ab.add_argument("--force", action="store_true")

    reuse = commands.add_parser("reuse-run-artifacts")
    reuse.add_argument("--source-run", type=Path, required=True)
    reuse.add_argument("--target-run", type=Path, required=True)
    reuse.add_argument(
        "--artifact", action="append", required=True,
        choices=(
            "segmentation", "hands", "depth", "mesh_proposals",
            "object_tracking", "optimization", "rendered_gt_proxy",
        ),
    )
    reuse.add_argument("--overwrite", action="store_true")

    register_gt = commands.add_parser("register-ground-truth")
    register_gt.add_argument("--spec", type=Path, required=True)
    register_gt.add_argument("--output-dir", type=Path, required=True)

    evaluate = commands.add_parser("evaluate-3.1")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--source-run", type=Path, required=True)
    evaluate.add_argument("--ground-truth-manifest", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    _manifest_argument(evaluate)

    human = commands.add_parser("prepare-human-reference")
    human.add_argument("--source-run", type=Path, required=True)
    human.add_argument("--output", type=Path, required=True)
    _manifest_argument(human)

    spider_human = commands.add_parser("prepare-spider-human-reference")
    spider_human.add_argument("--keypoints", type=Path, required=True)
    spider_human.add_argument("--side", choices=("left", "right", "bimanual"), default="right")
    spider_human.add_argument("--ref-dt", type=float, default=0.02)
    spider_human.add_argument("--output", type=Path, required=True)
    _manifest_argument(spider_human)

    oracle = commands.add_parser("prepare-oracle-human-reference")
    oracle.add_argument("--hand-ground-truth", type=Path, required=True)
    oracle.add_argument("--t-sim-world", type=Path, required=True)
    oracle.add_argument("--object-reference", type=Path)
    oracle.add_argument("--object-side", choices=("left", "right"), default="right")
    oracle.add_argument("--hand", choices=("left", "right"), action="append")
    oracle.add_argument("--confidence-threshold", type=float, default=0.0)
    oracle.add_argument("--output", type=Path, required=True)
    _manifest_argument(oracle)

    retarget = commands.add_parser("retarget-mink")
    retarget.add_argument("--config", required=True)
    retarget.add_argument("--human-reference", type=Path, required=True)
    retarget.add_argument("--model", type=Path, required=True)
    retarget.add_argument("--output", type=Path, required=True)
    retarget.add_argument("--solver", default="daqp")
    retarget.add_argument("--contact-reference", type=Path, default=None,
                          help="GT contact mask npz (contact field, T x 5) for "
                               "contact-aware retargeting")
    _manifest_argument(retarget)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-experiment":
        config = load_config(args.config)
        source = args.source_run.resolve()
        artifacts = args.source_artifact or _existing_source_artifacts(source)
        repositories = args.repo_root or [Path(__file__).resolve().parents[1]]
        manifest = ExperimentManifest.create(
            output_dir=args.output_dir, source_run=source, config=config,
            episode_id=args.episode_id, seeds=args.seeds, source_artifacts=artifacts,
            repository_roots=repositories,
        )
        print(manifest.path)
        return 0
    if args.command == "make-egodex-gt":
        from .evaluation.ground_truth import create_egodex_ground_truth_bundle

        result = create_egodex_ground_truth_bundle(args.source_run, args.output_dir)
        gt_data = json.loads(result.read_text(encoding="utf-8"))
        outputs = [result, *(Path(record["path"]) for record in gt_data["artifacts"].values())]
        _record_stage(
            args.experiment_manifest, "ground_truth_bundle", inputs=[args.source_run / "input/source.json"],
            outputs=outputs, metrics={"unavailable_modalities": gt_data["unavailable_modalities"]},
            uses_ground_truth=True,
        )
        print(result)
        return 0
    if args.command == "materialize-evaluation-set":
        from .evaluation import load_evaluation_set, materialize_evaluation_set

        result = materialize_evaluation_set(
            load_evaluation_set(args.evaluation_set), args.output_dir, force=args.force,
        )
        print(result)
        return 0
    if args.command == "materialize-taco-set":
        from .evaluation import load_taco_set, materialize_taco_set

        result = materialize_taco_set(
            load_taco_set(args.taco_set), args.output_dir, force=args.force,
        )
        print(result)
        return 0
    if args.command == "extract-taco-rgb":
        from .ingest import extract_taco_rgb_videos, load_taco_first_person_set

        result = extract_taco_rgb_videos(
            load_taco_first_person_set(args.taco_set), args.output_dir, force=args.force,
        )
        print(result)
        return 0
    if args.command == "ingest-taco-devset":
        from .ingest import ingest_taco_devset, load_taco_first_person_set

        result = ingest_taco_devset(
            load_taco_first_person_set(args.taco_set), args.video_dir,
            args.output_dir, force=args.force,
        )
        print(result)
        return 0
    if args.command == "prepare-taco-known-mesh":
        from .ingest import materialize_known_mesh_proposal

        result = materialize_known_mesh_proposal(
            args.source_run, args.ground_truth_manifest, overwrite=args.overwrite,
        )
        _record_stage(
            args.experiment_manifest, "known_mesh_proposal",
            inputs=[args.ground_truth_manifest], outputs=[result],
            metrics={"mesh_source": "TACO released metric model"},
            uses_ground_truth=True,
        )
        print(result)
        return 0
    if args.command == "freeze-taco-base-frame":
        from .ingest import freeze_taco_base_frame, load_taco_first_person_set

        spec = load_taco_first_person_set(args.taco_set)
        result = freeze_taco_base_frame(
            spec, args.episode_id, args.output, overwrite=args.overwrite,
        )
        _record_stage(
            args.experiment_manifest, "taco_fixed_base_frame",
            inputs=[spec.path], outputs=[result, result.with_suffix(".json")],
            metrics={
                "workspace_offset_m": float(spec.data["base_frame"]["workspace_offset_m"]),
                "table_height_m": float(spec.data["base_frame"]["table_height_m"]),
            },
            uses_ground_truth=True,
        )
        print(result)
        return 0
    if args.command == "run-sam2-paper":
        from .perception import run_sam2_paper_masks

        result = run_sam2_paper_masks(
            args.source_run, args.hand_ground_truth, args.checkpoint,
            tuple(args.object_point), model_config=args.model_config,
            prompt_frame=args.prompt_frame, output_dir=args.output_dir,
            confidence_threshold=args.confidence_threshold,
            overwrite=args.overwrite, dry_run=args.dry_run,
        )
        outputs = [result]
        if not args.dry_run:
            output = result.parent
            outputs.extend([
                output / "object_masks.npz", output / "hand_masks.npz",
                output / "perception_overlay.mp4", output / "sam2_prompts.npz",
            ])
        _record_stage(
            args.experiment_manifest, "sam2_paper_masks",
            inputs=[args.hand_ground_truth, args.checkpoint], outputs=outputs,
            metrics={"hand_source": "oracle_hand", "paper_route": True},
            uses_ground_truth=True,
        )
        print(result)
        return 0
    if args.command == "prepare-taco-object-reference":
        from .ingest import materialize_foundationpose_object_reference

        result = materialize_foundationpose_object_reference(
            args.source_run, args.t_sim_world, args.output, overwrite=args.overwrite,
        )
        _record_stage(
            args.experiment_manifest, "taco_foundationpose_object_reference",
            inputs=[
                args.source_run / "object_tracking/foundationpose_raw.npz",
                args.t_sim_world,
            ], outputs=[result, result.with_suffix(".json")],
            metrics={"contact_scale_calibration": False, "per_frame_floor_shift": False},
        )
        print(result)
        return 0
    if args.command == "run-taco-ablation":
        from .experiments import (
            execute_taco_ablation, load_taco_ablation, prepare_taco_ablation,
        )

        manifest = prepare_taco_ablation(
            load_taco_ablation(args.ablation), args.ingest_manifest,
            args.ground_truth_set_manifest, args.output_dir,
            profiles=args.profile, force=args.force,
        )
        result = manifest if args.prepare_only else execute_taco_ablation(
            manifest, force=args.force, gpu=args.gpu,
            progress=lambda message: print(message, flush=True),
        )
        print(result)
        return 0
    if args.command == "render-taco-oracle-depth":
        from .perception import render_taco_oracle_metric_depth

        result = render_taco_oracle_metric_depth(
            args.source_run, args.ground_truth_manifest, overwrite=args.overwrite,
        )
        print(result)
        return 0
    if args.command == "run-taco-auto-segmentation-v2":
        from .experiments import (
            execute_taco_auto_segmentation_v2, load_taco_auto_segmentation_spec,
            prepare_taco_auto_segmentation_v2,
        )

        manifest = prepare_taco_auto_segmentation_v2(
            load_taco_auto_segmentation_spec(args.spec), args.source_ablation_manifest,
            args.output_dir, force=args.force,
        )
        result = manifest if args.prepare_only else execute_taco_auto_segmentation_v2(
            manifest, force=args.force, gpu=args.gpu,
            progress=lambda message: print(message, flush=True),
        )
        print(result)
        return 0
    if args.command == "run-taco-depth-calibration":
        from .experiments import (
            execute_taco_depth_calibration, load_taco_depth_calibration_spec,
            prepare_taco_depth_calibration,
        )
        manifest = prepare_taco_depth_calibration(
            load_taco_depth_calibration_spec(args.spec), args.segmentation_manifest,
            args.output_dir,
        )
        result = manifest if args.prepare_only else execute_taco_depth_calibration(manifest, gpu=args.gpu)
        print(result)
        return 0
    if args.command == "derive-taco-camera-proxy":
        from .evaluation import derive_taco_camera_calibration_proxy

        result = derive_taco_camera_calibration_proxy(
            args.ground_truth_manifest, args.wilor_artifact, args.intrinsics,
            args.output_dir, overwrite=args.overwrite,
        )
        print(result)
        return 0
    if args.command == "run-taco-oracle-depth-ablation":
        from .experiments import (
            execute_taco_oracle_depth_ablation, load_taco_oracle_depth_spec,
            prepare_taco_oracle_depth_ablation,
        )

        manifest = prepare_taco_oracle_depth_ablation(
            load_taco_oracle_depth_spec(args.spec), args.source_ablation_manifest,
            args.output_dir, force=args.force,
        )
        result = manifest if args.prepare_only else execute_taco_oracle_depth_ablation(
            manifest, force=args.force, gpu=args.gpu,
            progress=lambda message: print(message, flush=True),
        )
        print(result)
        return 0
    if args.command == "reuse-run-artifacts":
        from .ingest import reuse_run_artifacts

        result = reuse_run_artifacts(
            args.source_run, args.target_run, args.artifact, overwrite=args.overwrite,
        )
        print(result)
        return 0
    if args.command == "register-ground-truth":
        from .evaluation import register_ground_truth_bundle

        result = register_ground_truth_bundle(args.spec, args.output_dir)
        print(result)
        return 0
    if args.command == "evaluate-3.1":
        from .evaluation.evaluator import evaluate_run

        result = evaluate_run(
            args.source_run, args.ground_truth_manifest, args.output_dir, load_config(args.config),
        )
        report = json.loads(result.read_text(encoding="utf-8"))
        _record_stage(
            args.experiment_manifest, "offline_evaluation_3_1",
            inputs=[args.ground_truth_manifest], outputs=[result],
            metrics={name: value["status"] for name, value in report["modalities"].items()},
            uses_ground_truth=True,
        )
        print(result)
        return 0
    if args.command == "prepare-human-reference":
        from .retarget.input import human_reference_from_aligned

        result = human_reference_from_aligned(args.source_run, args.output)
        _record_stage(
            args.experiment_manifest, "prepare_auto_human_reference",
            inputs=[args.source_run / "optimization/aligned_trajectory.npz"], outputs=[result],
        )
        print(result)
        return 0
    if args.command == "prepare-spider-human-reference":
        from .retarget.input import human_reference_from_spider_keypoints

        result = human_reference_from_spider_keypoints(
            args.keypoints, args.output, side=args.side, ref_dt=args.ref_dt,
        )
        _record_stage(
            args.experiment_manifest, "prepare_spider_human_reference",
            inputs=[args.keypoints], outputs=[result],
            metrics={"hand_source": "spider_mano_keypoints", "side": args.side},
        )
        print(result)
        return 0
    if args.command == "prepare-oracle-human-reference":
        from .retarget.input import human_reference_from_oracle_hand

        result = human_reference_from_oracle_hand(
            args.hand_ground_truth, args.t_sim_world, args.output, hand_order=args.hand,
            object_reference_path=args.object_reference, object_side=args.object_side,
            confidence_threshold=args.confidence_threshold,
        )
        inputs = [args.hand_ground_truth, args.t_sim_world]
        if args.object_reference is not None:
            inputs.append(args.object_reference)
        _record_stage(
            args.experiment_manifest, "prepare_oracle_human_reference",
            inputs=inputs, outputs=[result], metrics={"hand_source": "oracle_hand"},
            uses_ground_truth=True,
        )
        print(result)
        return 0
    if args.command == "retarget-mink":
        from .retarget.mink import retarget_with_mink

        config = load_config(args.config)
        result = retarget_with_mink(
            args.human_reference, args.model, args.output, config, solver=args.solver,
            contact_reference_path=args.contact_reference,
        )
        report_path = result.with_suffix(".json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _record_stage(
            args.experiment_manifest, "mink_retarget", inputs=[args.human_reference, args.model],
            outputs=[result, report_path], metrics={
                key: value for key, value in report.items()
                if key.endswith("_count") or key.endswith("_mean") or "_mean_" in key
            }, uses_ground_truth=config.uses_ground_truth,
        )
        print(result)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
