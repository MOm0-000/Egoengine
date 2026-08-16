"""Minimal stage CLI; later work packages extend this without bypassing artifact boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval.egodex import evaluate_wilor
from .eval.metrics import build_run_report
from .export.spider import export_spider_dataset
from .export.spider_runner import run_spider_chain
from .ingest.egodex import DEFAULT_ROOT, find_episode, ingest_episode, scan_episodes
from .ingest.egodex_ground_truth import validate_camera_direction_oracle
from .ingest.stereo import ingest_rectified_stereo
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
    stereo = commands.add_parser("ingest-stereo")
    stereo.add_argument("--left-dir", type=Path, required=True)
    stereo.add_argument("--right-dir", type=Path, required=True)
    stereo.add_argument("--intrinsics", type=Path, required=True)
    stereo.add_argument(
        "--right-intrinsics", type=Path,
        help="rectified right K; defaults to the left K when the camera toolkit exports a shared K",
    )
    common_valid = stereo.add_mutually_exclusive_group(required=True)
    common_valid.add_argument("--common-valid-mask", type=Path)
    common_valid.add_argument("--full-image-common-valid", action="store_true")
    stereo.add_argument("--baseline-m", type=float, required=True)
    stereo.add_argument("--output-dir", type=Path, required=True)
    stereo.add_argument("--task", required=True)
    stereo.add_argument("--episode-id", required=True)
    stereo.add_argument("--instruction", required=True)
    stereo.add_argument("--fps", type=float, required=True)
    stereo.add_argument("--camera-poses", type=Path)
    stereo.add_argument("--timestamps", type=Path)
    stereo.add_argument("--frame-indices", type=Path)
    stereo.add_argument("--static-camera", action="store_true")
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
    optimize.add_argument("--allow-no-contact", action="store_true")
    optimize.add_argument("--allow-unvalidated-contact-scale", action="store_true")
    optimize.add_argument(
        "--hand-source", choices=["wilor", "hawor", "egodex_gt"], default="wilor",
    )
    optimize.add_argument(
        "--hand-artifact", type=Path,
        help=(
            "Use an explicit hand artifact produced by an audited upstream adapter; "
            "hand-source still declares its schema/semantics."
        ),
    )
    optimize.add_argument("--uses-ground-truth", action="store_true")
    hand_export = optimize.add_mutually_exclusive_group()
    hand_export.add_argument(
        "--active-hands-only", dest="active_hands_only", action="store_true",
        help="export only hands with persistent manipulation-contact evidence (default)",
    )
    hand_export.add_argument(
        "--include-passive-hands", dest="active_hands_only", action="store_false",
        help="also retain visible non-manipulating hands for visualization/analysis",
    )
    optimize.set_defaults(active_hands_only=True)
    optimize.add_argument("--contact-enter-distance-m", type=float, default=0.012)
    optimize.add_argument("--max-contact-slip-p95-m-s", type=float, default=0.30)
    optimize.add_argument(
        "--fingertip-orientation-source",
        choices=("mano_fk", "landmark_proxy"),
        default="mano_fk",
        help=(
            "axis-calibrated MANO rotational FK (default) or the legacy "
            "landmark proxy for an explicit ablation"
        ),
    )
    optimize.add_argument(
        "--stabilize-precontact-object-translation", action="store_true",
        help=(
            "diagnostic monocular ablation: hold object translation fixed until "
            "persistent 2-D hand/object proximity"
        ),
    )
    optimize.add_argument("--precontact-stabilization-lead-frames", type=int, default=2)
    optimize.add_argument(
        "--contact-similarity-mode", choices=("auto", "refine", "validate_only"),
        default="auto",
        help=(
            "refine may adjust the object similarity from contact; validate_only "
            "tests shared metric scale at ratio 1.0 without moving the object"
        ),
    )
    optimize.add_argument(
        "--object-artifact", type=Path,
        help="Use an automatically selected upstream object track instead of FoundationPose.",
    )
    optimize.add_argument(
        "--object-scale-to-m", type=float,
        help="Freeze an upstream calibrated mesh scale; disables silhouette/contact scale refit.",
    )
    optimize.add_argument(
        "--object-valid-key",
        help="Restrict the selected object artifact to its upstream valid window.",
    )
    optimize.add_argument("--overwrite", action="store_true")
    export = commands.add_parser("export-spider")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--dataset-root", type=Path)
    export.add_argument("--spider-package-root", type=Path, required=True)
    export.add_argument("--task")
    export.add_argument("--data-id", type=int, default=0)
    export.add_argument("--embodiment-type", choices=["right", "left", "bimanual"])
    export.add_argument("--hand-sides", nargs="+", choices=["left", "right"])
    export.add_argument("--robot-type", default="xhand")
    export.add_argument(
        "--normalize-xhand-morphology", action="store_true",
        help=(
            "Apply one static palm-frame hand-size scale when the aligned "
            "artifact provides calibrated neutral hand geometry."
        ),
    )
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
    spider.add_argument("--mjwp-num-samples", type=int, default=2048)
    spider.add_argument("--mjwp-iterations", type=int, default=16)
    spider.add_argument("--mjwp-override", default="gigahand_origin")
    spider.add_argument("--mjwp-horizon", type=float, default=1.6)
    spider.add_argument("--mjwp-ctrl-dt", type=float, default=0.08)
    spider.add_argument("--mjwp-knot-dt", type=float, default=0.2)
    spider.add_argument("--contact-reward-scale", type=float, default=2.0)
    spider.add_argument("--contact-opposition-reward-scale", type=float, default=0.0)
    spider.add_argument("--base-pos-rew-scale", type=float, default=0.2)
    spider.add_argument("--base-rot-rew-scale", type=float, default=1.0)
    spider.add_argument("--joint-rew-scale", type=float, default=0.0)
    spider.add_argument("--force-closure-reward-scale", type=float, default=0.010)
    spider.add_argument(
        "--force-closure-penetration-reward-scale", type=float, default=1.0
    )
    spider.add_argument("--force-closure-min-normal-force-n", type=float, default=0.2)
    spider.add_argument(
        "--force-closure-min-opposition-cosine", type=float, default=0.2
    )
    spider.add_argument("--force-closure-max-penetration-m", type=float, default=0.003)
    spider.add_argument("--force-closure-minimum-frames", type=int, default=3)
    spider.add_argument("--lift-reward-scale", type=float, default=0.025)
    spider.add_argument(
        "--action-smoothness-reward-scale", type=float, default=0.8
    )
    spider.add_argument(
        "--no-paper-objective", dest="paper_objective", action="store_false"
    )
    spider.set_defaults(paper_objective=True)
    spider.add_argument("--sanity-check-seconds", type=float, default=1.0)
    spider.add_argument("--no-collision-aware-ik", action="store_true")
    spider.add_argument(
        "--ik-backend", choices=["mink", "spider-native"], default="mink"
    )
    spider.add_argument(
        "--mink-fidelity-continuation",
        dest="mink_allow_fidelity_rejected_qref_for_refinement",
        action="store_true",
    )
    spider.set_defaults(mink_allow_fidelity_rejected_qref_for_refinement=False)
    spider.add_argument(
        "--mink-collision-projection-max-iterations", type=int, default=80,
    )
    spider.add_argument(
        "--mink-controller-contact-target-policy",
        choices=[
            "object_surface", "fingertip_collision_center",
            "qref_fingertip_site",
        ],
        default="fingertip_collision_center",
    )
    spider.add_argument("--mink-scene-collision-constraints", action="store_true")
    spider.add_argument("--mink-floor-clearance-m", type=float, default=0.0)
    spider.add_argument(
        "--mink-non-distal-object-clearance-m", type=float, default=0.0,
    )
    spider.add_argument(
        "--mink-distal-object-max-penetration-m", type=float, default=0.0025,
    )
    spider.add_argument("--replay-noise-scale", type=float, default=0.0)
    spider.add_argument("--ik-seed", type=int, default=0)
    spider.add_argument("--no-adaptive-mode-switching", action="store_true")
    spider.add_argument("--replay-chunk-steps", type=int, default=20)
    spider.add_argument("--replay-lookahead-chunks", type=int, default=2)
    spider.add_argument("--replay-position-threshold-m", type=float, default=0.05)
    spider.add_argument("--replay-rotation-threshold-rad", type=float, default=0.5)
    spider.add_argument("--replay-min-motion-transfer-ratio", type=float, default=0.5)
    preshape = spider.add_mutually_exclusive_group()
    preshape.add_argument(
        "--contact-aware-preshape", dest="contact_aware_preshape",
        action="store_true",
    )
    preshape.add_argument(
        "--no-contact-aware-preshape", dest="contact_aware_preshape",
        action="store_false",
    )
    spider.set_defaults(contact_aware_preshape=False)
    spider.add_argument("--preshape-approach-frames", type=int, default=12)
    spider.add_argument("--preshape-clearance-m", type=float, default=0.03)
    spider.add_argument("--preshape-collision-margin-m", type=float, default=0.001)
    spider.add_argument("--preshape-contact-penetration-m", type=float, default=0.0025)
    spider.add_argument("--preshape-solver-iterations", type=int, default=180)
    spider.add_argument("--preshape-solver-dt", type=float, default=0.01)
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
    if args.command == "ingest-stereo":
        print(
            ingest_rectified_stereo(
                left_dir=args.left_dir,
                right_dir=args.right_dir,
                intrinsics_path=args.intrinsics,
                right_intrinsics_path=args.right_intrinsics,
                common_valid_mask_path=args.common_valid_mask,
                full_image_common_valid=args.full_image_common_valid,
                baseline_m=args.baseline_m,
                output_dir=args.output_dir,
                task=args.task,
                episode_id=args.episode_id,
                instruction=args.instruction,
                fps=args.fps,
                camera_poses_path=args.camera_poses,
                timestamps_path=args.timestamps,
                frame_indices_path=args.frame_indices,
                static_camera=args.static_camera,
            )
        )
        return 0
    if args.command == "evaluate-wilor":
        print(evaluate_wilor(args.run_dir, confidence_threshold=args.confidence_threshold))
        return 0
    if args.command == "optimize":
        print(optimize_run(
            args.run_dir, smoothing_strength=args.smoothing_strength,
            hand_smoothing_strength=args.hand_smoothing_strength,
            require_contact=not args.allow_no_contact,
            allow_unvalidated_contact_scale=args.allow_unvalidated_contact_scale,
            hand_source=args.hand_source, uses_ground_truth=args.uses_ground_truth,
            active_hands_only=args.active_hands_only,
            contact_enter_distance_m=args.contact_enter_distance_m,
            max_contact_slip_p95_m_s=args.max_contact_slip_p95_m_s,
            fingertip_orientation_source=args.fingertip_orientation_source,
            stabilize_precontact_object_translation=(
                args.stabilize_precontact_object_translation
            ),
            precontact_stabilization_lead_frames=(
                args.precontact_stabilization_lead_frames
            ),
            contact_similarity_mode=args.contact_similarity_mode,
            hand_artifact_path=args.hand_artifact,
            object_artifact_path=args.object_artifact,
            object_scale_to_m=args.object_scale_to_m,
            object_valid_key=args.object_valid_key,
            overwrite=args.overwrite,
        ))
        return 0
    if args.command == "export-spider":
        run_dir = args.run_dir.resolve()
        source = json.loads((run_dir / "input/source.json").read_text(encoding="utf-8"))
        selected = json.loads((run_dir / "object_tracking/selected_mesh.json").read_text(encoding="utf-8"))
        optimization_metrics = json.loads(
            (run_dir / "optimization/optimization_metrics.json").read_text(encoding="utf-8")
        )
        quality_control = optimization_metrics.get("quality_control", {})
        if quality_control.get("export_ready") is not True:
            raise RuntimeError(
                "optimization artifact is not export-ready; rerun optimization and inspect quality_control"
            )
        hand_roles = optimization_metrics.get("hands", {}).get("roles")
        artifact_hand_order = optimization_metrics.get("hands", {}).get("artifact_hand_order")
        hand_sides = args.hand_sides or artifact_hand_order
        if not hand_sides:
            raise RuntimeError("optimization report does not define artifact_hand_order")
        inferred_embodiment = {
            ("right",): "right",
            ("left",): "left",
            ("left", "right"): "bimanual",
        }.get(tuple(hand_sides))
        embodiment_type = args.embodiment_type or inferred_embodiment
        if embodiment_type is None:
            raise RuntimeError(f"cannot infer SPIDER embodiment from hand order: {hand_sides}")
        dataset_root = args.dataset_root or run_dir / "spider_export/dataset"
        print(export_spider_dataset(
            aligned_path=run_dir / "optimization/aligned_trajectory.npz",
            contact_path=run_dir / "optimization/contact.npz",
            visual_mesh_path=run_dir / selected["canonical_visual_mesh"],
            dataset_root=dataset_root, task=args.task or source["task_directory"],
            data_id=args.data_id, source_run_id=run_dir.name, hand_sides=hand_sides,
            spider_package_root=args.spider_package_root,
            hand_roles=hand_roles,
            embodiment_type=embodiment_type, robot_type=args.robot_type,
            normalize_xhand_morphology=args.normalize_xhand_morphology,
        ))
        return 0
    if args.command == "run-spider":
        print(run_spider_chain(
            dataset_root=args.dataset_root, spider_root=args.spider_root, task=args.task,
            data_id=args.data_id, embodiment_type=args.embodiment_type,
            robot_type=args.robot_type, gpu=args.gpu, run_mjwp=not args.no_mjwp,
            save_video=not args.no_save_video, ik_end_idx=args.ik_end_idx,
            mjwp_num_samples=args.mjwp_num_samples, mjwp_iterations=args.mjwp_iterations,
            mjwp_override=args.mjwp_override, mjwp_horizon=args.mjwp_horizon,
            mjwp_ctrl_dt=args.mjwp_ctrl_dt, mjwp_knot_dt=args.mjwp_knot_dt,
            contact_reward_scale=args.contact_reward_scale,
            contact_opposition_reward_scale=args.contact_opposition_reward_scale,
            base_pos_rew_scale=args.base_pos_rew_scale,
            base_rot_rew_scale=args.base_rot_rew_scale,
            joint_rew_scale=args.joint_rew_scale,
            force_closure_reward_scale=args.force_closure_reward_scale,
            force_closure_penetration_reward_scale=(
                args.force_closure_penetration_reward_scale
            ),
            force_closure_min_normal_force_n=args.force_closure_min_normal_force_n,
            force_closure_min_opposition_cosine=(
                args.force_closure_min_opposition_cosine
            ),
            force_closure_max_penetration_m=args.force_closure_max_penetration_m,
            force_closure_minimum_frames=args.force_closure_minimum_frames,
            lift_reward_scale=args.lift_reward_scale,
            paper_objective=args.paper_objective,
            action_smoothness_reward_scale=args.action_smoothness_reward_scale,
            sanity_check_seconds=args.sanity_check_seconds,
            collision_aware_ik=not args.no_collision_aware_ik,
            ik_backend=args.ik_backend,
            mink_allow_fidelity_rejected_qref_for_refinement=(
                args.mink_allow_fidelity_rejected_qref_for_refinement
            ),
            mink_collision_projection_max_iterations=(
                args.mink_collision_projection_max_iterations
            ),
            mink_controller_contact_target_policy=(
                args.mink_controller_contact_target_policy
            ),
            mink_scene_collision_constraints=(
                args.mink_scene_collision_constraints
            ),
            mink_floor_clearance_m=args.mink_floor_clearance_m,
            mink_non_distal_object_clearance_m=(
                args.mink_non_distal_object_clearance_m
            ),
            mink_distal_object_max_penetration_m=(
                args.mink_distal_object_max_penetration_m
            ),
            replay_noise_scale=args.replay_noise_scale,
            ik_seed=args.ik_seed,
            adaptive_mode_switching=not args.no_adaptive_mode_switching,
            replay_chunk_steps=args.replay_chunk_steps,
            replay_lookahead_chunks=args.replay_lookahead_chunks,
            replay_position_threshold_m=args.replay_position_threshold_m,
            replay_rotation_threshold_rad=args.replay_rotation_threshold_rad,
            replay_min_motion_transfer_ratio=args.replay_min_motion_transfer_ratio,
            contact_aware_preshape=args.contact_aware_preshape,
            preshape_approach_frames=args.preshape_approach_frames,
            preshape_clearance_m=args.preshape_clearance_m,
            preshape_collision_margin_m=args.preshape_collision_margin_m,
            preshape_contact_penetration_m=args.preshape_contact_penetration_m,
            preshape_solver_iterations=args.preshape_solver_iterations,
            preshape_solver_dt=args.preshape_solver_dt,
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
