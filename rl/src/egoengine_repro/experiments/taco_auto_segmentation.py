"""Freeze the isolated TACO ``auto_segmentation_v2`` profile."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from video_to_spider.schemas import SCHEMA_VERSION

from ..artifacts import artifact_record
from .taco_ablation import _branch_run, _cmd, _model_path, _stage, execute_taco_ablation


@dataclass(frozen=True)
class TacoAutoSegmentationSpec:
    path: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.data["name"])


def bundled_taco_auto_segmentation_path() -> Path:
    return Path(str(files("egoengine_repro").joinpath(
        "configs", "taco_auto_segmentation_v2_dev4.yaml",
    )))


def load_taco_auto_segmentation_spec(
    path: str | Path | None = None,
) -> TacoAutoSegmentationSpec:
    candidate = Path(path) if path is not None else bundled_taco_auto_segmentation_path()
    data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or str(data.get("schema_version")) != "1.0":
        raise ValueError("TACO auto-segmentation spec must use schema version 1.0")
    policy = data.get("inference_policy", {})
    required = {
        "object_prompt_source": "instruction_text_only",
        "manual_point_allowed": False,
        "ground_truth_object_name_allowed": False,
        "oracle_hand_allowed": False,
        "hand_source": "sam3_text_prompt",
        "default_profile_replaced": False,
    }
    if policy != required:
        raise ValueError("auto_segmentation_v2 inference policy cannot enable oracle inputs")
    return TacoAutoSegmentationSpec(candidate.resolve(), data)


def _auto_profile_lookup(source_manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    result = []
    for episode in source_manifest["episodes"]:
        matches = [
            profile for profile in episode["profiles"]
            if profile["profile"] == "auto_current"
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one auto_current source for {episode['episode_id']}")
        plan = json.loads(Path(matches[0]["execution_plan"]["path"]).read_text(encoding="utf-8"))
        result.append((str(episode["episode_id"]), plan))
    return result


def prepare_taco_auto_segmentation_v2(
    spec: TacoAutoSegmentationSpec, source_ablation_manifest: str | Path,
    output_dir: str | Path, *, force: bool = False,
) -> Path:
    repo = Path(__file__).resolve().parents[2]
    source_path = Path(source_ablation_manifest).resolve()
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, runtime = spec.data["model_inputs"], spec.data["runtime"]
    episode_records = []
    for episode_id, source_plan in _auto_profile_lookup(source_manifest):
        profile_root = output / "episodes" / episode_id / "profiles" / "auto_segmentation_v2"
        run_dir = profile_root / "run"
        source_run = Path(source_plan["source_run"]).resolve()
        _branch_run(source_run, run_dir, "auto_segmentation_v2", force=force)
        outputs = [
            run_dir / "segmentation/object_masks.npz",
            run_dir / "segmentation/hand_masks.npz",
            run_dir / "segmentation/metadata.json",
        ]
        stage = _stage("segmentation_sam3_auto_v2", "v2s-sam3", _cmd(
            repo / "scripts/run_model_adapter.sh", "v2s-sam3",
            "video_to_spider.adapters.sam3", "--run-dir", run_dir,
            "--checkpoint", _model_path(repo, model["sam3_checkpoint"]),
            "--selection-profile", "auto_segmentation_v2",
            "--max-candidates", runtime["max_instruction_candidates"],
            "--max-instances", runtime["max_instances"],
            "--probe-anchor-count", runtime["probe_anchor_count"],
            "--max-propagated-candidates", runtime["max_propagated_candidates"],
            "--overwrite",
        ), outputs)
        plan = {
            "schema_version": SCHEMA_VERSION, "ablation": spec.name,
            "episode_id": episode_id, "profile": "auto_segmentation_v2",
            "source_run": str(source_run), "run_dir": str(run_dir),
            "uses_ground_truth_in_inference": False,
            "inference_policy": spec.data["inference_policy"],
            "same_interval_contract": artifact_record(run_dir / "frames/frame_index.json"),
            "stages": [stage],
        }
        plan_path = profile_root / "execution_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        episode_records.append({
            "episode_id": episode_id,
            "profiles": [{
                "profile": "auto_segmentation_v2", "run_dir": str(run_dir),
                "execution_plan": artifact_record(plan_path), "blocked_stage_count": 0,
            }],
        })
    expected = int(spec.data["acceptance"]["expected_episode_count"])
    if len(episode_records) != expected:
        raise ValueError(f"expected {expected} TACO episodes, found {len(episode_records)}")
    manifest_path = output / "taco_auto_segmentation_v2_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "name": spec.name,
        "spec": artifact_record(spec.path),
        "source_ablation_manifest": artifact_record(source_path),
        "profile": "auto_segmentation_v2", "episode_count": len(episode_records),
        "uses_ground_truth_in_inference": False,
        "inference_policy": spec.data["inference_policy"],
        "episodes": episode_records,
    }, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def summarize_taco_auto_segmentation_v2(
    manifest_path: str | Path, output_path: str | Path,
) -> Path:
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    spec = yaml.safe_load(Path(manifest["spec"]["path"]).read_text(encoding="utf-8"))
    by_episode: dict[str, Any] = {}
    for episode in manifest["episodes"]:
        profile = episode["profiles"][0]
        run_dir = Path(profile["run_dir"])
        segmentation = run_dir / "segmentation"
        metadata_path = segmentation / "metadata.json"
        mask_path = segmentation / "object_masks.npz"
        hand_path = segmentation / "hand_masks.npz"
        if not (metadata_path.is_file() and mask_path.is_file() and hand_path.is_file()):
            by_episode[episode["episode_id"]] = {"status": "missing"}
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(mask_path, allow_pickle=False) as artifact:
            masks = np.asarray(artifact["masks"], dtype=bool)
            valid = np.asarray(artifact["valid"], dtype=bool)
            confidence = np.asarray(artifact["confidence"], dtype=np.float64)
        valid &= masks.reshape(len(masks), -1).any(axis=1)
        valid &= confidence > 0.0
        policy = metadata.get("automatic_input_policy", {})
        by_episode[episode["episode_id"]] = {
            "status": "available", "frame_count": len(valid),
            "valid_frame_count": int(np.count_nonzero(valid)),
            "valid_rate": float(np.mean(valid)),
            "nonempty": bool(valid.any()),
            "selected_prompt": metadata.get("selected_prompt"),
            "selected_prompt_role": metadata.get("selected_prompt_role"),
            "manual_point_used": policy.get("manual_point_used"),
            "ground_truth_object_name_used": policy.get("ground_truth_object_name_used"),
            "oracle_hand_used": policy.get("oracle_hand_used"),
            "hand_source": policy.get("hand_source"),
            "artifacts": {
                "object_masks": artifact_record(mask_path),
                "hand_masks": artifact_record(hand_path),
                "metadata": artifact_record(metadata_path),
            },
        }
    available = [item for item in by_episode.values() if item["status"] == "available"]
    expected = int(spec["acceptance"]["expected_episode_count"])
    target_rate = float(spec["acceptance"]["target_minimum_valid_rate"])
    policy_clean = all(
        item["manual_point_used"] is False
        and item["ground_truth_object_name_used"] is False
        and item["oracle_hand_used"] is False
        and item["hand_source"] == "sam3_text_prompt"
        for item in available
    )
    acceptance = {
        "four_episode_artifact_coverage": len(available) == expected,
        "all_episodes_nonempty": len(available) == expected and all(item["nonempty"] for item in available),
        "all_episode_valid_rate_at_least_0.95": (
            len(available) == expected and all(item["valid_rate"] >= target_rate for item in available)
        ),
        "automatic_input_policy_clean": len(available) == expected and policy_clean,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "source_manifest": artifact_record(manifest_file),
        "profile": "auto_segmentation_v2",
        "frozen_for_downstream_depth_experiments": True,
        "default_profile_replaced": False,
        "by_episode": by_episode,
        "acceptance": acceptance,
    }, indent=2) + "\n", encoding="utf-8")
    return destination


def execute_taco_auto_segmentation_v2(
    manifest_path: str | Path, *, force: bool = False, gpu: int | None = 6,
    progress: Any = None,
) -> Path:
    execution = execute_taco_ablation(
        manifest_path, force=force, gpu=gpu, progress=progress,
    )
    result = summarize_taco_auto_segmentation_v2(
        manifest_path, Path(manifest_path).resolve().parent / "auto_segmentation_v2_results.json",
    )
    payload = json.loads(execution.read_text(encoding="utf-8"))
    payload["auto_segmentation_v2_results"] = artifact_record(result)
    execution.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return execution
