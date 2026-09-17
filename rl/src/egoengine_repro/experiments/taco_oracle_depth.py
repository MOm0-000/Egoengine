"""Controlled TACO oracle metric-depth ablation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from video_to_spider.schemas import SCHEMA_VERSION

from ..artifacts import artifact_record
from ..config import bundled_config_path
from ..evaluation import (
    derive_taco_camera_calibration_proxy, install_taco_camera_calibration_proxy,
)
from ..ingest import load_taco_first_person_set
from .taco_ablation import (
    _branch_run, _cmd, _reuse_stage, _segmentation_stage, _stage,
    execute_taco_ablation,
)


PROFILE_ORDER = (
    "auto_current", "known_mesh_mono_depth", "known_mesh_oracle_depth",
    "known_mesh_oracle_depth_sam2_oracle_hand",
)


@dataclass(frozen=True)
class TacoOracleDepthSpec:
    path: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.data["name"])


def bundled_taco_oracle_depth_path() -> Path:
    return Path(str(files("egoengine_repro").joinpath(
        "configs", "taco_oracle_depth_ablation_dev4.yaml",
    )))


def load_taco_oracle_depth_spec(path: str | Path | None = None) -> TacoOracleDepthSpec:
    candidate = Path(path) if path is not None else bundled_taco_oracle_depth_path()
    data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or str(data.get("schema_version")) != "1.0":
        raise ValueError("TACO oracle-depth spec must use schema version 1.0")
    if tuple(data.get("profiles", {}).keys()) != PROFILE_ORDER:
        raise ValueError("TACO oracle-depth spec must preserve the frozen four-profile order")
    return TacoOracleDepthSpec(candidate.resolve(), data)


def _source_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        str(episode["episode_id"]): {
            str(profile["profile"]): {
                **profile,
                "plan": json.loads(Path(
                    profile["execution_plan"]["path"],
                ).read_text(encoding="utf-8")),
            }
            for profile in episode["profiles"]
        }
        for episode in manifest["episodes"]
    }


def _foundationpose_stage(
    repo: Path, spec: dict[str, Any], run_dir: Path,
) -> dict[str, Any]:
    model, runtime = spec["model_inputs"], spec["runtime"]
    return _stage("object_foundationpose", "v2s-foundationpose", _cmd(
        repo / "scripts/run_model_adapter.sh", "v2s-foundationpose",
        "video_to_spider.adapters.foundationpose", "--run-dir", run_dir,
        "--foundationpose-root", repo / model["foundationpose_root"],
        "--max-candidates", runtime["foundationpose_max_candidates"],
        "--screening-radius", runtime["foundationpose_screening_radius"],
        "--register-iter", runtime["foundationpose_register_iter"],
        "--track-iter", runtime["foundationpose_track_iter"],
        "--max-input-side", runtime["foundationpose_max_input_side"],
        "--skip-visualizations", "--overwrite",
    ), [
        run_dir / "object_tracking/foundationpose_raw.npz",
        run_dir / "object_tracking/selected_mesh.json",
    ])


def _oracle_depth_stage(run_dir: Path, gt_manifest: Path) -> dict[str, Any]:
    return _stage("render_oracle_metric_depth", "v2s-foundationpose", _cmd(
        "conda", "run", "--no-capture-output", "-n", "v2s-foundationpose",
        "python", "-m", "egoengine_repro.perception.oracle_depth",
        "--run-dir", run_dir, "--ground-truth-manifest", gt_manifest, "--overwrite",
    ), [
        run_dir / "rendered_gt_proxy/metadata.json",
        run_dir / "rendered_gt_proxy/object_metric_depth.zarr",
        run_dir / "depth/metric_depth.zarr",
    ])


def _evaluation_stage(run_dir: Path, profile_root: Path, gt_manifest: Path) -> dict[str, Any]:
    output = profile_root / "evaluation_3_1/offline_3_1_metrics.json"
    return _stage("offline_evaluation_3_1", "v2s-core", _cmd(
        "conda", "run", "--no-capture-output", "-n", "v2s-core",
        "python", "-m", "egoengine_repro.cli", "evaluate-3.1",
        "--config", bundled_config_path("auto_current"), "--source-run", run_dir,
        "--ground-truth-manifest", gt_manifest,
        "--output-dir", profile_root / "evaluation_3_1",
    ), [output])


def prepare_taco_oracle_depth_ablation(
    spec: TacoOracleDepthSpec, source_ablation_manifest: str | Path,
    output_dir: str | Path, *, force: bool = False,
) -> Path:
    repo = Path(__file__).resolve().parents[2]
    source_path = Path(source_ablation_manifest).resolve()
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source = _source_lookup(source_manifest)
    taco_set = load_taco_first_person_set(str(spec.data["dataset_set"]))
    original_spec = yaml.safe_load(Path(source_manifest["spec"]["path"]).read_text(encoding="utf-8"))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    episode_records = []
    for episode in taco_set.data["episodes"]:
        episode_id = str(episode["episode_id"])
        profiles = source[episode_id]
        auto_plan = profiles["auto_current"]["plan"]
        known_plan = profiles["known_mesh"]["plan"]
        paper_plan = profiles["paper_faithful"]["plan"]
        auto_run = Path(auto_plan["run_dir"]).resolve()
        known_run = Path(known_plan["run_dir"]).resolve()
        paper_run = Path(paper_plan["run_dir"]).resolve()
        gt_manifest = Path(auto_plan["ground_truth_manifest"]).resolve()
        camera_proxy = None
        if episode.get("camera_calibration_proxy") == "wilor_mano_pnp":
            camera_proxy = derive_taco_camera_calibration_proxy(
                gt_manifest, auto_run / "hands/wilor_raw.npz",
                auto_run / "calibration/intrinsics.npy",
                output / "ground_truth_proxies" / episode_id, overwrite=force,
            )
            gt_manifest = camera_proxy
        profile_records = []
        for profile in PROFILE_ORDER:
            profile_root = output / "episodes" / episode_id / "profiles" / profile
            profile_root.mkdir(parents=True, exist_ok=True)
            if profile == "auto_current":
                run_dir = auto_run
                stages = [_evaluation_stage(run_dir, profile_root, gt_manifest)]
                inputs = {"inference_run_reused_without_changes": artifact_record(run_dir / "manifest.json")}
            elif profile == "known_mesh_mono_depth":
                run_dir = known_run
                stages = [_evaluation_stage(run_dir, profile_root, gt_manifest)]
                inputs = {"inference_run_reused_without_changes": artifact_record(run_dir / "manifest.json")}
            else:
                run_dir = profile_root / "run"
                _branch_run(Path(auto_plan["source_run"]), run_dir, profile, force=force)
                if camera_proxy is not None:
                    install_taco_camera_calibration_proxy(run_dir, camera_proxy)
                if profile == "known_mesh_oracle_depth":
                    source_tracking_available = (
                        known_run / "object_tracking/foundationpose_raw.npz"
                    ).is_file()
                    if source_tracking_available:
                        reusable = ["segmentation", "hands", "mesh_proposals"]
                    else:
                        reusable = ["segmentation", "hands"]
                    stages = [
                        _reuse_stage(known_run, run_dir, reusable),
                        _oracle_depth_stage(run_dir, gt_manifest),
                    ]
                    if not source_tracking_available:
                        stages.append(_stage(
                            "mesh_known", "v2s-core", [],
                            [run_dir / "mesh_proposals/mesh_ranking.json"],
                            blocked_reason="frozen auto SAM3 object mask has no valid frame",
                        ))
                else:
                    profile_data = {
                        "object_segmentation": "sam2", "hand_source": "taco_oracle_hand",
                    }
                    strict_oracle_run = (
                        output / "episodes" / episode_id / "profiles"
                        / "known_mesh_oracle_depth" / "run"
                    )
                    hand_gt = Path(json.loads(
                        gt_manifest.read_text(encoding="utf-8"),
                    )["artifacts"]["hand"]["path"])
                    if camera_proxy is None:
                        stages = [_reuse_stage(
                            paper_run, run_dir, ["segmentation", "mesh_proposals"],
                        )]
                    else:
                        stages = [_segmentation_stage(
                            type("SourceSpec", (), {"data": original_spec})(), repo,
                            episode_id, profile_data, run_dir, hand_gt,
                        ), _reuse_stage(paper_run, run_dir, ["mesh_proposals"])]
                    stages.append(_reuse_stage(
                        strict_oracle_run, run_dir, ["rendered_gt_proxy", "depth"],
                    ))
                stages.extend([
                    _foundationpose_stage(repo, original_spec, run_dir),
                    _evaluation_stage(run_dir, profile_root, gt_manifest),
                ])
                inputs = {
                    "oracle_depth_quality_label": "rendered_gt_proxy",
                    "only_foundationpose_depth_replaced": profile == "known_mesh_oracle_depth",
                }
            plan = {
                "schema_version": SCHEMA_VERSION, "ablation": spec.name,
                "episode_id": episode_id, "profile": profile,
                "source_run": auto_plan["source_run"], "run_dir": str(run_dir),
                "ground_truth_manifest": str(gt_manifest),
                "camera_quality_label": (
                    "derived_calibration_proxy" if camera_proxy else "dataset_calibration_ground_truth"
                ),
                "rendered_depth_quality_label": (
                    "rendered_gt_proxy" if "oracle_depth" in profile else None
                ),
                "controlled_inputs": inputs, "stages": stages,
            }
            plan_path = profile_root / "execution_plan.json"
            plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            profile_records.append({
                "profile": profile, "run_dir": str(run_dir),
                "execution_plan": artifact_record(plan_path),
                "blocked_stage_count": sum(
                    stage["status"] == "blocked_config" for stage in stages
                ),
            })
        episode_records.append({"episode_id": episode_id, "profiles": profile_records})
    manifest_path = output / "taco_oracle_depth_ablation_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "name": spec.name,
        "spec": artifact_record(spec.path),
        "source_ablation_manifest": artifact_record(source_path),
        "profile_order": list(PROFILE_ORDER), "episode_count": len(episode_records),
        "episodes": episode_records,
        "proxy_policy": {
            "rendered_depth": "rendered_gt_proxy; never manual GT",
            "skim_camera": "derived_calibration_proxy; not independent GT",
        },
    }, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _relative_improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return float((baseline - candidate) / baseline)


def summarize_taco_oracle_depth_ablation(
    manifest_path: str | Path, output_path: str | Path,
) -> Path:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    spec = yaml.safe_load(Path(manifest["spec"]["path"]).read_text(encoding="utf-8"))
    by_episode: dict[str, dict[str, Any]] = {}
    values: dict[str, dict[str, list[float]]] = {}
    for episode in manifest["episodes"]:
        episode_metrics = {}
        for profile in episode["profiles"]:
            report_path = Path(profile["execution_plan"]["path"]).parent / "evaluation_3_1/offline_3_1_metrics.json"
            if not report_path.is_file():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            metrics = report["modalities"]["object_trajectory"].get("metrics", {})
            compact = {
                "translation_mean_m": metrics.get("translation_error_m", {}).get("mean"),
                "translation_p95_m": metrics.get("translation_error_m", {}).get("p95"),
                "rotation_mean_rad": metrics.get("rotation_geodesic_rad", {}).get("mean"),
                "rotation_p95_rad": metrics.get("rotation_geodesic_rad", {}).get("p95"),
                "add_s_mean_m": metrics.get("add_s_m", {}).get("mean"),
                "valid_rate": metrics.get("valid_rate"),
            }
            episode_metrics[profile["profile"]] = compact
            for name, value in compact.items():
                if isinstance(value, (int, float)):
                    values.setdefault(profile["profile"], {}).setdefault(name, []).append(float(value))
        by_episode[episode["episode_id"]] = episode_metrics
    aggregate = {
        profile: {name: float(sum(items) / len(items)) for name, items in metrics.items()}
        for profile, metrics in values.items()
    }
    episode_count = len(manifest["episodes"])
    profile_coverage = {
        profile: sum(
            isinstance(metrics.get(profile, {}).get("translation_mean_m"), (int, float))
            for metrics in by_episode.values()
        )
        for profile in PROFILE_ORDER
    }
    strict_pair_episode_ids = [
        episode_id for episode_id, metrics in by_episode.items()
        if isinstance(
            metrics.get("known_mesh_mono_depth", {}).get("translation_mean_m"),
            (int, float),
        ) and isinstance(
            metrics.get("known_mesh_oracle_depth", {}).get("translation_mean_m"),
            (int, float),
        )
    ]
    paired_values: dict[str, dict[str, list[float]]] = {
        "known_mesh_mono_depth": {}, "known_mesh_oracle_depth": {},
    }
    for episode_id in strict_pair_episode_ids:
        for profile in paired_values:
            for name, value in by_episode[episode_id][profile].items():
                if isinstance(value, (int, float)):
                    paired_values[profile].setdefault(name, []).append(float(value))
    paired_aggregate = {
        profile: {name: float(sum(items) / len(items)) for name, items in metrics.items()}
        for profile, metrics in paired_values.items()
    }
    mono = paired_aggregate.get("known_mesh_mono_depth", {})
    oracle = paired_aggregate.get("known_mesh_oracle_depth", {})
    threshold = spec["thresholds"]
    rotation_improvement = _relative_improvement(
        mono.get("rotation_mean_rad"), oracle.get("rotation_mean_rad"),
    )
    translation_improvement = _relative_improvement(
        mono.get("translation_mean_m"), oracle.get("translation_mean_m"),
    )
    adds_improvement = _relative_improvement(mono.get("add_s_mean_m"), oracle.get("add_s_mean_m"))
    oracle_p95_values = [
        float(by_episode[episode_id]["known_mesh_oracle_depth"]["translation_p95_m"])
        for episode_id in strict_pair_episode_ids
        if isinstance(
            by_episode[episode_id]["known_mesh_oracle_depth"].get("translation_p95_m"),
            (int, float),
        )
    ]
    max_oracle_p95 = max(oracle_p95_values) if oracle_p95_values else None
    rotation_outlier_episode_ids = [
        episode_id for episode_id in strict_pair_episode_ids
        if isinstance(
            by_episode[episode_id]["known_mesh_oracle_depth"].get("rotation_p95_rad"),
            (int, float),
        ) and float(
            by_episode[episode_id]["known_mesh_oracle_depth"]["rotation_p95_rad"]
        ) >= 2.5
    ]
    gates = {
        "strict_depth_only_pair_coverage_complete": len(strict_pair_episode_ids) == episode_count,
        "translation_mean_below_0.08_m": oracle.get("translation_mean_m", float("inf")) < float(
            threshold["oracle_translation_mean_m"]
        ),
        "no_0.5_m_scale_translation_anomaly": (
            max_oracle_p95 is not None and max_oracle_p95 < float(
                threshold["anomalous_translation_p95_m"]
            )
        ),
        "rotation_significantly_improved": (
            rotation_improvement is not None and rotation_improvement >= float(
                threshold["minimum_relative_rotation_improvement"]
            )
        ),
        "add_s_significantly_improved": (
            adds_improvement is not None and adds_improvement >= float(
                threshold["minimum_relative_add_s_improvement"]
            )
        ),
    }
    if not strict_pair_episode_ids:
        conditional_diagnosis = "inconclusive_no_strict_depth_only_pair"
    elif gates["translation_mean_below_0.08_m"]:
        if rotation_outlier_episode_ids:
            conditional_diagnosis = (
                "monocular_metric_scale_is_primary_bottleneck_but_rotation_axis_symmetry_or_"
                "initialization_issue_remains"
            )
        else:
            conditional_diagnosis = (
                "monocular_metric_scale_is_primary_bottleneck"
                if gates["rotation_significantly_improved"]
                else "translation_recovers_but_rotation_requires_axis_symmetry_or_initialization_audit"
            )
    else:
        conditional_diagnosis = "coordinate_extrinsic_or_mesh_canonical_frame_bug_remains"
    diagnosis = (
        conditional_diagnosis if gates["strict_depth_only_pair_coverage_complete"]
        else "inconclusive_partial_strict_pair_coverage"
    )
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "source_manifest": artifact_record(manifest_path),
        "by_episode": by_episode, "aggregate_equal_episode_weight": aggregate,
        "profile_metric_coverage": {
            "expected_episode_count": episode_count, "available_episode_count": profile_coverage,
        },
        "strict_depth_only_pairs": {
            "episode_ids": strict_pair_episode_ids,
            "count": len(strict_pair_episode_ids),
            "aggregate_equal_episode_weight": paired_aggregate,
        },
        "relative_improvement_vs_known_mesh_mono": {
            "translation": translation_improvement,
            "rotation": rotation_improvement, "add_s": adds_improvement,
        },
        "max_oracle_translation_p95_m": max_oracle_p95,
        "rotation_tail_diagnostic": {
            "threshold_rad": 2.5,
            "oracle_episode_ids_at_or_above_threshold": rotation_outlier_episode_ids,
            "is_acceptance_gate": False,
        },
        "acceptance_gates": gates, "diagnosis": diagnosis,
        "conditional_diagnosis_on_available_strict_pairs": conditional_diagnosis,
        "proxy_disclosure": manifest["proxy_policy"],
    }, indent=2) + "\n", encoding="utf-8")
    return destination


def execute_taco_oracle_depth_ablation(
    manifest_path: str | Path, *, force: bool = False, gpu: int | None = 6,
    progress: Any = None,
) -> Path:
    execution = execute_taco_ablation(
        manifest_path, force=force, gpu=gpu, progress=progress,
    )
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for episode in manifest["episodes"]:
        for profile in episode["profiles"]:
            plan_path = Path(profile["execution_plan"]["path"])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            profile["execution_plan"] = artifact_record(plan_path)
            profile["blocked_stage_count"] = sum(
                stage["status"] == "blocked_config" for stage in plan["stages"]
            )
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = summarize_taco_oracle_depth_ablation(
        manifest_file, manifest_file.parent / "p1_oracle_depth_results.json",
    )
    payload = json.loads(execution.read_text(encoding="utf-8"))
    payload["source_manifest"] = artifact_record(manifest_file)
    payload["p1_oracle_depth_results"] = artifact_record(summary)
    execution.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return execution
