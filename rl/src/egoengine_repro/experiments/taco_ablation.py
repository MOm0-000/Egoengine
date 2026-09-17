"""Prepare and execute the frozen TACO perception A/B ladder."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from video_to_spider.manifest import RunManifest
from video_to_spider.schemas import SCHEMA_VERSION

from ..artifacts import artifact_record
from ..config import bundled_config_path
from ..evaluation import slice_taco_ground_truth_bundle
from ..ingest import freeze_taco_base_frame, load_taco_first_person_set


@dataclass(frozen=True)
class TacoAblationSpec:
    path: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.data["name"])


def bundled_taco_ablation_path(name: str) -> Path:
    if name != "taco_perception_ablation_dev4":
        raise ValueError(f"unknown bundled TACO ablation: {name}")
    return Path(str(files("egoengine_repro").joinpath(
        "configs", "taco_perception_ablation_dev4.yaml",
    )))


def load_taco_ablation(path_or_name: str | Path) -> TacoAblationSpec:
    candidate = Path(path_or_name)
    path = candidate if candidate.exists() else bundled_taco_ablation_path(str(path_or_name))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or str(data.get("schema_version")) != "1.0":
        raise ValueError("TACO ablation must use schema version 1.0")
    profiles = data.get("profiles")
    expected = {"auto_current", "known_mesh", "oracle_hand_mesh", "paper_faithful"}
    if not isinstance(profiles, dict) or set(profiles) != expected:
        raise ValueError("TACO ablation must define the frozen four-profile ladder")
    if not isinstance(data.get("episodes"), dict) or len(data["episodes"]) != 4:
        raise ValueError("TACO development ablation must define four episodes")
    return TacoAblationSpec(path=path.resolve(), data=data)


def _gt_lookup(path: Path) -> dict[str, Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["episode_id"]): Path(item["ground_truth_manifest"]["path"]).resolve()
        for item in data["episodes"]
    }


def _ingest_lookup(path: Path) -> dict[str, Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["episode_id"]): Path(item["source_run"]["path"]).resolve().parent
        for item in data["episodes"]
    }


def _branch_run(source: Path, destination: Path, profile: str, *, force: bool) -> None:
    if destination.exists():
        if not force:
            return
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in ("input", "frames", "calibration"):
        (destination / name).symlink_to(source / name, target_is_directory=True)
    shutil.copy2(source / "manifest.json", destination / "manifest.json")
    manifest = RunManifest.load(destination / "manifest.json")
    manifest.data["run_id"] = destination.parent.name
    manifest.data["repro_profile"] = profile
    manifest.data["branch_source_run"] = str(source)
    manifest.data["branch_inputs_are_read_only_symlinks"] = True
    manifest.save()


def _model_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _cmd(*values: str | Path | int | float) -> list[str]:
    return [str(value) for value in values]


def _stage(
    name: str, environment: str, command: list[str], outputs: Iterable[Path],
    *, blocked_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name, "environment": environment, "command": command,
        "outputs": [str(path.resolve()) for path in outputs],
        "status": "blocked_config" if blocked_reason else "pending",
        "blocked_reason": blocked_reason,
    }


def _segmentation_stage(
    spec: TacoAblationSpec, repo: Path, episode_id: str,
    profile_data: dict[str, Any], run_dir: Path, hand_gt: Path,
) -> dict[str, Any]:
    model, runtime = spec.data["model_inputs"], spec.data["runtime"]
    outputs = [run_dir / "segmentation/object_masks.npz", run_dir / "segmentation/hand_masks.npz"]
    if profile_data["object_segmentation"] == "sam3":
        return _stage("segmentation_sam3", "v2s-sam3", _cmd(
            repo / "scripts/run_model_adapter.sh", "v2s-sam3", "video_to_spider.adapters.sam3",
            "--run-dir", run_dir, "--checkpoint", _model_path(repo, model["sam3_checkpoint"]),
            "--max-candidates", runtime["sam3_max_candidates"],
            "--max-instances", runtime["sam3_max_instances"], "--overwrite",
        ), outputs)
    prompt_spec = spec.data["episodes"][episode_id]
    point = prompt_spec.get("object_point_xy")
    blocked = None if point is not None else "freeze object_point_xy after inspecting first RGB"
    point = point or [0.0, 0.0]
    return _stage("segmentation_sam2", "v2s-sam3+sam2", _cmd(
        "conda", "run", "--no-capture-output", "-n", "v2s-sam3",
        "python", "-m", "egoengine_repro.perception.sam2",
        "--run-dir", run_dir, "--hand-ground-truth", hand_gt,
        "--checkpoint", _model_path(repo, model["sam2_checkpoint"]),
        "--model-config", model["sam2_model_config"],
        "--object-point", point[0], point[1],
        "--prompt-frame", int(prompt_spec.get("object_prompt_frame", 0)),
        "--skip-overlay", "--overwrite",
    ), outputs, blocked_reason=blocked)


def _reuse_stage(source_run: Path, target_run: Path, artifacts: list[str]) -> dict[str, Any]:
    command: list[str | Path] = [
        "conda", "run", "--no-capture-output", "-n", "v2s-core",
        "python", "-m", "egoengine_repro.cli", "reuse-run-artifacts",
        "--source-run", source_run, "--target-run", target_run,
    ]
    for artifact in artifacts:
        command.extend(["--artifact", artifact])
    command.append("--overwrite")
    return _stage(
        "reuse_" + "_".join(artifacts), "v2s-core",
        _cmd(*command), [target_run / artifact for artifact in artifacts],
    )


def _plan_for_profile(
    spec: TacoAblationSpec, repo: Path, episode_id: str, profile: str,
    profile_data: dict[str, Any], available_profiles: set[str],
    run_dir: Path, profile_root: Path, gt_manifest: Path, fixed_base: Path | None,
) -> list[dict[str, Any]]:
    model, runtime = spec.data["model_inputs"], spec.data["runtime"]
    gt = json.loads(gt_manifest.read_text(encoding="utf-8"))
    hand_gt = Path(gt["artifacts"]["hand"]["path"]).resolve()
    runner = repo / "scripts/run_model_adapter.sh"
    stages: list[dict[str, Any]] = []
    auto_run = profile_root.parent / "auto_current/run"
    known_run = profile_root.parent / "known_mesh/run"
    reuse_auto_upstream = profile in {"known_mesh", "oracle_hand_mesh"} and (
        "auto_current" in available_profiles
    )
    if reuse_auto_upstream:
        reusable = ["segmentation", "depth"]
        if profile == "known_mesh":
            reusable.append("hands")
        stages.append(_reuse_stage(auto_run, run_dir, reusable))
    else:
        stages.append(_segmentation_stage(spec, repo, episode_id, profile_data, run_dir, hand_gt))
    if profile_data["hand_source"] == "wilor" and not reuse_auto_upstream:
        stages.append(_stage("hands_wilor", "v2s-wilor", _cmd(
            runner, "v2s-wilor", "video_to_spider.adapters.wilor", "--run-dir", run_dir,
            "--checkpoint", _model_path(repo, model["wilor_checkpoint"]),
            "--model-config", _model_path(repo, model["wilor_model_config"]),
            "--detector-checkpoint", _model_path(repo, model["wilor_detector_checkpoint"]),
            "--skip-overlay", "--overwrite",
        ), [run_dir / "hands/wilor_raw.npz"]))
    if profile == "paper_faithful" and "auto_current" in available_profiles:
        stages.append(_reuse_stage(auto_run, run_dir, ["depth"]))
    elif not reuse_auto_upstream:
        stages.append(_stage("depth_anything", "v2s-depth", _cmd(
            runner, "v2s-depth", "video_to_spider.adapters.depth_anything", "--run-dir", run_dir,
            "--checkpoint", _model_path(repo, model["depth_checkpoint"]),
            "--encoder", "vitl", "--skip-video", "--overwrite",
        ), [run_dir / "depth/metric_depth.zarr", run_dir / "depth/metadata.json"]))
    reuse_known_tracking = profile == "oracle_hand_mesh" and "known_mesh" in available_profiles
    if reuse_known_tracking:
        stages.append(_reuse_stage(
            known_run, run_dir, ["mesh_proposals", "object_tracking", "optimization"],
        ))
    elif profile_data["mesh"] == "sam3d":
        stages.append(_stage("mesh_sam3d", "v2s-sam3d", _cmd(
            runner, "v2s-sam3d", "video_to_spider.adapters.sam3d_objects",
            "--run-dir", run_dir, "--config-path", _model_path(repo, model["sam3d_config"]),
            "--seeds", *runtime["sam3d_seeds"], "--max-keyframes", 1,
            "--max-proposals", 3, "--low-vram", "--moge-resolution-level", 4,
            "--max-slat-coords", 20000, "--overwrite",
        ), [run_dir / "mesh_proposals/mesh_ranking.json"]))
    elif not reuse_known_tracking:
        stages.append(_stage("mesh_known", "v2s-core", _cmd(
            "conda", "run", "--no-capture-output", "-n", "v2s-core",
            "python", "-m", "egoengine_repro.cli", "prepare-taco-known-mesh",
            "--source-run", run_dir, "--ground-truth-manifest", gt_manifest, "--overwrite",
        ), [run_dir / "mesh_proposals/mesh_ranking.json"]))
    if not reuse_known_tracking:
        stages.append(_stage("object_foundationpose", "v2s-foundationpose", _cmd(
            runner, "v2s-foundationpose", "video_to_spider.adapters.foundationpose",
            "--run-dir", run_dir,
            "--foundationpose-root", _model_path(repo, model["foundationpose_root"]),
            "--max-candidates", runtime["foundationpose_max_candidates"],
            "--screening-radius", runtime["foundationpose_screening_radius"],
            "--register-iter", runtime["foundationpose_register_iter"],
            "--track-iter", runtime["foundationpose_track_iter"],
            "--max-input-side", runtime["foundationpose_max_input_side"],
            "--skip-visualizations", "--overwrite",
        ), [run_dir / "object_tracking/foundationpose_raw.npz"]))
    if profile in {"auto_current", "known_mesh"}:
        stages.append(_stage("auto_sequence_optimization", "v2s-opt", _cmd(
            "conda", "run", "--no-capture-output", "-n", "v2s-opt",
            "python", "-m", "video_to_spider.cli", "optimize",
            "--run-dir", run_dir, "--skip-visualization", "--overwrite",
        ), [
            run_dir / "optimization/aligned_trajectory.npz",
            run_dir / "optimization/optimization_metrics.json",
        ]))
        auto_human_ref = profile_root / "retarget/human_reference.npz"
        stages.append(_stage("auto_human_reference", "v2s-core", _cmd(
            "conda", "run", "--no-capture-output", "-n", "v2s-core",
            "python", "-m", "egoengine_repro.cli", "prepare-human-reference",
            "--source-run", run_dir, "--output", auto_human_ref,
        ), [auto_human_ref]))
    elif profile == "oracle_hand_mesh" and reuse_known_tracking:
        oracle_ref = profile_root / "retarget/human_reference.npz"
        stages.append(_stage("oracle_human_reference", "v2s-core", _cmd(
            "conda", "run", "--no-capture-output", "-n", "v2s-core",
            "python", "-m", "egoengine_repro.cli", "prepare-oracle-human-reference",
            "--hand-ground-truth", hand_gt,
            "--t-sim-world", known_run / "optimization/optimization_metrics.json",
            "--object-reference", known_run / "optimization/aligned_trajectory.npz",
            "--hand", "left", "--hand", "right", "--output", oracle_ref,
        ), [oracle_ref]))
    if fixed_base is not None:
        object_ref = profile_root / "retarget/object_reference.npz"
        human_ref = profile_root / "retarget/human_reference.npz"
        stages.append(_stage("object_reference", "v2s-core", _cmd(
            "conda", "run", "--no-capture-output", "-n", "v2s-core",
            "python", "-m", "egoengine_repro.cli", "prepare-taco-object-reference",
            "--source-run", run_dir, "--t-sim-world", fixed_base,
            "--output", object_ref, "--overwrite",
        ), [object_ref]))
        stages.append(_stage("oracle_human_reference", "v2s-core", _cmd(
            "conda", "run", "--no-capture-output", "-n", "v2s-core",
            "python", "-m", "egoengine_repro.cli", "prepare-oracle-human-reference",
            "--hand-ground-truth", hand_gt, "--t-sim-world", fixed_base,
            "--object-reference", object_ref, "--hand", "left", "--hand", "right",
            "--output", human_ref,
        ), [human_ref]))
    config_path = bundled_config_path(str(profile_data["config"]))
    evaluation = profile_root / "evaluation_3_1/offline_3_1_metrics.json"
    stages.append(_stage("offline_evaluation_3_1", "v2s-core", _cmd(
        "conda", "run", "--no-capture-output", "-n", "v2s-core",
        "python", "-m", "egoengine_repro.cli", "evaluate-3.1",
        "--config", config_path, "--source-run", run_dir,
        "--ground-truth-manifest", gt_manifest,
        "--output-dir", profile_root / "evaluation_3_1",
    ), [evaluation]))
    return stages


def prepare_taco_ablation(
    spec: TacoAblationSpec, ingest_manifest: str | Path,
    ground_truth_set_manifest: str | Path, output_dir: str | Path,
    *, profiles: Iterable[str] | None = None, force: bool = False,
) -> Path:
    repo = Path(__file__).resolve().parents[2]
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_profiles = list(profiles or spec.data["profiles"])
    unknown = set(selected_profiles) - set(spec.data["profiles"])
    if unknown:
        raise ValueError(f"unknown TACO profiles: {sorted(unknown)}")
    ingest_path = Path(ingest_manifest).resolve()
    gt_set_path = Path(ground_truth_set_manifest).resolve()
    ingest, gt = _ingest_lookup(ingest_path), _gt_lookup(gt_set_path)
    taco_set = load_taco_first_person_set(str(spec.data["dataset_set"]))
    episode_records = []
    for episode_id in spec.data["episodes"]:
        if episode_id not in ingest or episode_id not in gt:
            raise ValueError(f"TACO episode is missing ingest or GT: {episode_id}")
        frame_payload = json.loads(
            (ingest[episode_id] / "frames/frame_index.json").read_text(encoding="utf-8")
        )
        selected_frames = [row["source_frame_index"] for row in frame_payload["frames"]]
        gt[episode_id] = slice_taco_ground_truth_bundle(
            gt[episode_id], selected_frames,
            output / "ground_truth_intervals" / episode_id, force=force,
        )
        profile_records = []
        for profile in selected_profiles:
            profile_root = output / "episodes" / episode_id / "profiles" / profile
            run_dir = profile_root / "run"
            _branch_run(ingest[episode_id], run_dir, profile, force=force)
            profile_data = spec.data["profiles"][profile]
            fixed_base = None
            if profile_data["alignment"] == "taco_fixed_base":
                fixed_base = profile_root / "inputs/fixed_T_sim_world.npz"
                if force or not fixed_base.is_file():
                    freeze_taco_base_frame(taco_set, episode_id, fixed_base, overwrite=force)
            plan_path = profile_root / "execution_plan.json"
            stages = _plan_for_profile(
                spec, repo, episode_id, profile, profile_data, set(selected_profiles),
                run_dir, profile_root,
                gt[episode_id], fixed_base,
            )
            plan = {
                "schema_version": SCHEMA_VERSION, "ablation": spec.name,
                "episode_id": episode_id, "profile": profile,
                "source_run": str(ingest[episode_id]), "run_dir": str(run_dir),
                "ground_truth_manifest": str(gt[episode_id]),
                "same_interval_contract": artifact_record(run_dir / "frames/frame_index.json"),
                "uses_ground_truth_in_inference": bool(
                    profile_data["hand_source"] == "taco_oracle_hand"
                    or profile_data["mesh"] == "taco_known_mesh" or fixed_base is not None
                ),
                "stages": stages,
            }
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            profile_records.append({
                "profile": profile, "run_dir": str(run_dir),
                "execution_plan": artifact_record(plan_path),
                "blocked_stage_count": sum(stage["status"] == "blocked_config" for stage in stages),
            })
        episode_records.append({"episode_id": episode_id, "profiles": profile_records})
    manifest_path = output / "taco_ablation_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "name": spec.name,
        "spec": artifact_record(spec.path), "ingest_manifest": artifact_record(ingest_path),
        "ground_truth_set_manifest": artifact_record(gt_set_path),
        "profiles": selected_profiles, "episode_count": len(episode_records),
        "episodes": episode_records,
    }, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _flatten_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_metrics(child, name))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def summarize_taco_ablation(manifest_path: str | Path, output_path: str | Path) -> Path:
    manifest = json.loads(Path(manifest_path).resolve().read_text(encoding="utf-8"))
    by_episode: dict[str, dict[str, dict[str, float]]] = {}
    reports = []
    for episode in manifest["episodes"]:
        profile_metrics = {}
        for profile in episode["profiles"]:
            plan = json.loads(Path(profile["execution_plan"]["path"]).read_text(encoding="utf-8"))
            report_path = Path(plan["run_dir"]).parent / "evaluation_3_1/offline_3_1_metrics.json"
            if not report_path.is_file():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            metrics = _flatten_metrics({
                name: modality.get("metrics", {})
                for name, modality in report["modalities"].items()
                if modality.get("status") == "available"
            })
            profile_metrics[profile["profile"]] = metrics
            reports.append(artifact_record(report_path))
        by_episode[episode["episode_id"]] = profile_metrics
    profile_values: dict[str, dict[str, list[float]]] = {}
    paired_values: dict[str, dict[str, list[float]]] = {}
    for profile_metrics in by_episode.values():
        baseline = profile_metrics.get("auto_current", {})
        for profile, metrics in profile_metrics.items():
            destination = profile_values.setdefault(profile, {})
            for metric, value in metrics.items():
                destination.setdefault(metric, []).append(value)
            if profile == "auto_current":
                continue
            deltas = paired_values.setdefault(profile, {})
            for metric in set(baseline) & set(metrics):
                deltas.setdefault(metric, []).append(metrics[metric] - baseline[metric])
    aggregate = {
        profile: {
            metric: {"count": len(values), "mean": sum(values) / len(values)}
            for metric, values in metrics.items()
        }
        for profile, metrics in profile_values.items()
    }
    paired = {
        profile: {
            metric: {"count": len(values), "mean_candidate_minus_auto": sum(values) / len(values)}
            for metric, values in metrics.items()
        }
        for profile, metrics in paired_values.items()
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "source_manifest": artifact_record(Path(manifest_path).resolve()),
        "report_count": len(reports), "reports": reports,
        "metric_direction_policy": "raw values and signed deltas only; interpret per metric",
        "aggregate": aggregate, "paired_delta_vs_auto_current": paired,
    }, indent=2) + "\n", encoding="utf-8")
    return destination


def execute_taco_ablation(
    manifest_path: str | Path, *, force: bool = False,
    gpu: int | None = 6,
    progress: Callable[[str], None] | None = None,
) -> Path:
    source = Path(manifest_path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parents[2]
    callback = progress or (lambda _: None)
    results = []
    for episode in manifest["episodes"]:
        for profile_record in episode["profiles"]:
            plan_path = Path(profile_record["execution_plan"]["path"])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            profile_failed = False
            dependency_failed = False
            for stage in plan["stages"]:
                outputs = [Path(path) for path in stage["outputs"]]
                is_evaluation = stage["name"] == "offline_evaluation_3_1"
                if dependency_failed and not is_evaluation:
                    stage["status"] = "skipped_dependency"
                    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
                    continue
                if stage["status"] == "blocked_config":
                    callback(f"{plan['episode_id']} {plan['profile']} {stage['name']}: blocked")
                    profile_failed = True
                    dependency_failed = True
                    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
                    continue
                if not force and outputs and all(path.exists() for path in outputs):
                    stage["status"] = "skipped_existing"
                    continue
                callback(f"{plan['episode_id']} {plan['profile']} {stage['name']}: running")
                log_path = plan_path.parent / "logs" / f"{stage['name']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
                    str(repo), environment.get("PYTHONPATH", ""),
                ]))
                if gpu is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                started = time.monotonic()
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        stage["command"], cwd=repo, env=environment,
                        stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
                    )
                stage.update({
                    "status": "completed" if completed.returncode == 0 else "failed",
                    "return_code": completed.returncode,
                    "wall_time_s": float(time.monotonic() - started), "log": str(log_path),
                })
                plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
                if completed.returncode != 0:
                    callback(f"{plan['episode_id']} {plan['profile']} {stage['name']}: failed")
                    profile_failed = True
                    dependency_failed = True
            results.append({
                "episode_id": plan["episode_id"], "profile": plan["profile"],
                "success": not profile_failed and all(
                    stage["status"] in {"completed", "skipped_existing"}
                    for stage in plan["stages"]
                ),
                "execution_plan": artifact_record(plan_path),
            })
    result_path = source.parent / "taco_ablation_execution_summary.json"
    comparison_path = summarize_taco_ablation(
        source, source.parent / "taco_ablation_3_1_comparison.json",
    )
    result_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "source_manifest": artifact_record(source),
        "run_count": len(results), "success_count": sum(item["success"] for item in results),
        "cuda_visible_device": gpu,
        "comparison_3_1": artifact_record(comparison_path),
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    return result_path
