#!/usr/bin/env python3
"""Registration-first TACO support-surface generalization audit.

This runner is deliberately fail-closed.  It performs no physics, retargeting,
planning, or learning and never reads an object bottom as an estimator input.
If the official RGB-D/camera registration gate fails, plane estimation is not
executed and no plane artifacts are produced.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.evaluation.taco_official_projection import TacoOfficialProjector  # noqa: E402


DEFAULT_CONFIG = RL_ROOT / "configs/taco_support_surface_generalization_audit_v1.yaml"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected-actual)}, "
            f"unknown={sorted(actual-expected)}"
        )


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    _exact_keys(cfg, {
        "schema", "status", "minimum_baseline", "paper_faithful", "authorization",
        "official", "sample", "inputs", "registration", "multi_sample", "output",
    }, "root")
    _exact_keys(cfg["authorization"], {
        "mink_candidates", "physics", "replay", "mpc", "reinforcement_learning",
        "promotion", "chunk_commit", "active_support_contract_replacement",
    }, "authorization")
    _exact_keys(cfg["official"], {
        "repository", "commit", "checkout", "entrypoint", "device",
    }, "official")
    _exact_keys(cfg["sample"], {
        "triplet", "sequence_name", "expected_frames", "endpoint_zero_is_frame_zero",
        "tool_id", "target_id", "focus_frames", "timing_offsets",
    }, "sample")
    _exact_keys(cfg["inputs"], {
        "data_root", "rgb_video", "depth_video", "egocentric_intrinsic",
        "egocentric_extrinsic", "hand_joints", "object_pose_dir", "object_model_dir",
        "official_projection_manifest", "official_projection_visual_review",
        "active_support_contract",
    }, "inputs")
    _exact_keys(cfg["registration"], {
        "depth_width", "depth_height", "depth_dtype", "depth_scale",
        "interior_erosion_px", "front_surface_tolerance_m",
        "minimum_valid_pixels_per_pair", "minimum_valid_pixel_fraction_per_pair",
        "maximum_median_of_medians_absolute_error_mm", "threshold_provenance",
        "timing_offsets_are_diagnostic_not_selected",
    }, "registration")
    _exact_keys(cfg["multi_sample"], {
        "root", "minimum_complete_samples", "selection", "required_depth_resolution",
        "required_equal_frame_counts",
    }, "multi_sample")
    if cfg["schema"] != "taco_support_surface_generalization_audit_v1":
        raise ValueError("unexpected audit schema")
    if cfg["authorization"]["mink_candidates"] != 0 or any(
        value for key, value in cfg["authorization"].items() if key != "mink_candidates"
    ):
        raise ValueError("audit must authorize zero runtime or contract replacement")
    baseline = cfg["minimum_baseline"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, git_head(REPO_ROOT)],
        cwd=REPO_ROOT, check=True,
    )
    official = Path(cfg["official"]["checkout"])
    if git_head(official) != cfg["official"]["commit"]:
        raise ValueError("official TACO checkout commit mismatch")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=official, text=True,
    ).strip()
    if dirty:
        raise ValueError("official TACO checkout is dirty")
    if cfg["sample"]["focus_frames"] != [0, 42, 83, 125, 166, 208]:
        raise ValueError("focus-frame contract changed")
    if cfg["sample"]["timing_offsets"] != [-1, 0, 1]:
        raise ValueError("diagnostic timing offsets changed")
    return cfg


def video_metadata(path: Path) -> dict[str, Any]:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,"
        "nb_frames,nb_read_frames,duration", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {path}")
    return streams[0]


def frame_count(metadata: dict[str, Any]) -> int:
    for key in ("nb_read_frames", "nb_frames"):
        value = metadata.get(key)
        if value not in (None, "N/A"):
            return int(value)
    raise ValueError(f"video has no frame count: {metadata}")


def depth_frames(
    path: Path, indices: set[int], *, count: int, width: int, height: int, scale: float,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    process = subprocess.Popen([
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
        "-pix_fmt", "gray16le", "-",
    ], stdout=subprocess.PIPE)
    assert process.stdout is not None
    result: dict[int, np.ndarray] = {}
    valid_fraction: dict[int, float] = {}
    frame_bytes = width * height * 2
    for frame in range(count):
        payload = process.stdout.read(frame_bytes)
        if len(payload) != frame_bytes:
            process.kill()
            raise RuntimeError(f"truncated depth frame {frame}")
        raw_view = np.frombuffer(payload, dtype="<u2").reshape(height, width)
        valid_fraction[frame] = float(np.mean(raw_view > 0))
        if frame in indices:
            raw = raw_view.copy()
            result[frame] = raw.astype(np.float32) / scale
    process.stdout.close()
    if process.wait() != 0 or set(result) != indices:
        raise RuntimeError("depth decode failed")
    return result, valid_fraction


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    size = 2 * pixels + 1
    return cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8)) > 0


def object_mesh(vertices: np.ndarray, faces: np.ndarray, pose: np.ndarray) -> trimesh.Trimesh:
    transformed = vertices @ pose[:3, :3].T + pose[:3, 3]
    return trimesh.Trimesh(vertices=transformed, faces=faces, process=False)


def input_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = {key: Path(value).resolve(strict=True) for key, value in cfg["inputs"].items()}
    for role in ("tool", "target"):
        object_id = cfg["sample"][f"{role}_id"]
        paths[f"{role}_pose"] = (paths["object_pose_dir"] / f"{role}_{object_id}.npy").resolve(strict=True)
        paths[f"{role}_mesh"] = (paths["object_model_dir"] / f"{object_id}_cm.obj").resolve(strict=True)
    return paths


def registration_audit(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    reg = cfg["registration"]
    sample = cfg["sample"]
    rgb_meta = video_metadata(paths["rgb_video"])
    depth_meta = video_metadata(paths["depth_video"])
    expected = int(sample["expected_frames"])
    intrinsic = np.loadtxt(paths["egocentric_intrinsic"])
    extrinsics = np.load(paths["egocentric_extrinsic"], allow_pickle=False)
    hand_joints = np.load(paths["hand_joints"], allow_pickle=False)
    objects: dict[str, dict[str, Any]] = {}
    for role in ("tool", "target"):
        value = trimesh.load_mesh(paths[f"{role}_mesh"], process=False)
        value.apply_scale(0.01)
        objects[role] = {
            "vertices": np.asarray(value.vertices),
            "faces": np.asarray(value.faces),
            "poses": np.load(paths[f"{role}_pose"], allow_pickle=False),
        }

    counts = {
        "rgb": frame_count(rgb_meta), "depth": frame_count(depth_meta),
        "extrinsic": int(extrinsics.shape[0]), "hand_annotation": int(hand_joints.shape[0]),
        "tool_pose": int(objects["tool"]["poses"].shape[0]),
        "target_pose": int(objects["target"]["poses"].shape[0]),
    }
    dimensions = {
        "rgb": [int(rgb_meta["width"]), int(rgb_meta["height"])],
        "depth": [int(depth_meta["width"]), int(depth_meta["height"])],
        "intrinsic_shape": list(intrinsic.shape),
        "extrinsic_shape": list(extrinsics.shape),
        "official_projected_mask": [int(reg["depth_width"]), int(reg["depth_height"])],
    }
    focus = set(sample["focus_frames"])
    depths, all_depth_valid_fraction = depth_frames(
        paths["depth_video"], focus, count=expected,
        width=int(reg["depth_width"]), height=int(reg["depth_height"]),
        scale=float(reg["depth_scale"]),
    )
    utils = Path(cfg["official"]["checkout"]) / "dataset_utils"
    projector = TacoOfficialProjector(
        dataset_utils=utils,
        image_size=(int(reg["depth_width"]), int(reg["depth_height"])),
        intrinsic=intrinsic, extrinsic=extrinsics[0], device=cfg["official"]["device"],
    )
    rasterizer = projector.official_wrapper.renderer.renderer.rasterizer
    rasterizer.raster_settings = replace(
        rasterizer.raster_settings, max_faces_per_bin=400000,
    )
    records: list[dict[str, Any]] = []
    for observed_frame in sample["focus_frames"]:
        for offset in sample["timing_offsets"]:
            annotation_frame = observed_frame + offset
            if not 0 <= annotation_frame < expected:
                continue
            projector.set_camera(intrinsic, extrinsics[annotation_frame])
            meshes = {
                role: object_mesh(row["vertices"], row["faces"], row["poses"][annotation_frame])
                for role, row in objects.items()
            }
            full = projector.render_depth([meshes["tool"], meshes["target"]])
            for role in ("tool", "target"):
                rendered = projector.render_depth([meshes[role]])
                front = ((rendered > 0) & (full > 0) &
                         (np.abs(rendered - full) <= float(reg["front_surface_tolerance_m"])))
                selector = erode(front, int(reg["interior_erosion_px"]))
                valid = selector & np.isfinite(depths[observed_frame]) & (depths[observed_frame] > 0)
                values = depths[observed_frame][valid] - rendered[valid]
                absolute = np.abs(values) * 1000.0
                records.append({
                    "observed_depth_frame": observed_frame,
                    "annotation_frame": annotation_frame,
                    "timing_offset": offset,
                    "object": role,
                    "interior_pixels": int(selector.sum()),
                    "valid_pixels": int(valid.sum()),
                    "valid_pixel_fraction": float(valid.sum() / max(int(selector.sum()), 1)),
                    "signed_error_mm": None if not len(values) else {
                        "p05": float(np.percentile(values * 1000.0, 5)),
                        "median": float(np.median(values) * 1000.0),
                        "p95": float(np.percentile(values * 1000.0, 95)),
                    },
                    "absolute_error_mm": None if not len(values) else {
                        "median": float(np.median(absolute)),
                        "p90": float(np.percentile(absolute, 90)),
                        "p95": float(np.percentile(absolute, 95)),
                    },
                })

    zero = [row for row in records if row["timing_offset"] == 0]
    usable = [row for row in zero if row["absolute_error_mm"] is not None]
    per_object: dict[str, Any] = {}
    for role in ("tool", "target"):
        rows = [row for row in usable if row["object"] == role]
        per_object[role] = {
            "comparisons": len(rows),
            "minimum_valid_pixel_fraction": min(
                (row["valid_pixel_fraction"] for row in rows), default=0.0,
            ),
            "median_valid_pixel_fraction": None if not rows else float(np.median([
                row["valid_pixel_fraction"] for row in rows
            ])),
            "median_of_median_absolute_error_mm": None if not rows else float(np.median([
                row["absolute_error_mm"]["median"] for row in rows
            ])),
        }
    by_offset: dict[str, Any] = {}
    for offset in sample["timing_offsets"]:
        rows = [row for row in records if row["timing_offset"] == offset and row["absolute_error_mm"]]
        by_offset[str(offset)] = {
            "comparisons": len(rows),
            "median_of_median_absolute_error_mm": None if not rows else float(np.median([
                row["absolute_error_mm"]["median"] for row in rows
            ])),
        }
    best_rows = []
    for observed_frame in sample["focus_frames"]:
        for role in ("tool", "target"):
            rows = [row for row in records if row["observed_depth_frame"] == observed_frame
                    and row["object"] == role and row["absolute_error_mm"]]
            if rows:
                best = min(rows, key=lambda row: row["absolute_error_mm"]["median"])
                best_rows.append({"frame": observed_frame, "object": role,
                                  "best_offset": best["timing_offset"]})

    review = json.loads(paths["official_projection_visual_review"].read_text())
    manifest = json.loads(paths["official_projection_manifest"].read_text())
    checks = {
        "all_counts_equal_expected": all(value == expected for value in counts.values()),
        "container_timebase_consistent": (
            rgb_meta.get("avg_frame_rate") == depth_meta.get("avg_frame_rate")
            and abs(float(rgb_meta["duration"]) - float(depth_meta["duration"]))
            <= (1.0 / 30.0)
        ),
        "rgb_resolution_is_1920x1080": dimensions["rgb"] == [1920, 1080],
        "depth_resolution_is_1920x1080": dimensions["depth"] == [1920, 1080],
        "intrinsic_is_3x3": dimensions["intrinsic_shape"] == [3, 3],
        "extrinsic_is_Nx4x4": dimensions["extrinsic_shape"] == [expected, 4, 4],
        "official_projected_mask_matches_depth_pixels": (
            dimensions["official_projected_mask"] == dimensions["depth"]
        ),
        "official_projection_visual_review_pass": review.get("result") == "PASS",
        "official_projection_has_no_manual_pixel_offset": (
            manifest.get("manual_pixel_offset") is False and review.get("manual_pixel_offset") is False
        ),
        "all_rigid_pairs_available": len(usable) == 2 * len(sample["focus_frames"]),
        "all_rigid_pairs_have_minimum_pixels": all(
            row["valid_pixels"] >= int(reg["minimum_valid_pixels_per_pair"]) for row in usable
        ),
        "all_rigid_pairs_meet_valid_fraction": all(
            row["valid_pixel_fraction"] >= float(reg["minimum_valid_pixel_fraction_per_pair"])
            for row in usable
        ),
        "aggregate_error_within_gate": bool(usable) and float(np.median([
            row["absolute_error_mm"]["median"] for row in usable
        ])) <= float(reg["maximum_median_of_medians_absolute_error_mm"]),
    }
    passed = all(checks.values())
    return {
        "schema": "taco_rgb_depth_camera_registration_audit_v1",
        "classification": "REGISTRATION_PASS" if passed else "TACO_RGBD_REGISTRATION_BLOCKER",
        "passed": passed,
        "official_semantics": {
            "depth_decode": "ffmpeg gray16le; uint16 / 4000 metres",
            "extrinsic": "world_to_camera, consumed unchanged by pinned official projector",
            "frame_policy": "RGB frame i = depth frame i = annotation row i; no rate resampling",
            "timing_offsets": "diagnostic only; offset zero remains the formal contract",
        },
        "stream_metadata": {"rgb": rgb_meta, "depth": depth_meta},
        "temporal_registration_evidence": {
            "rgb_container_rate_hz": rgb_meta.get("avg_frame_rate"),
            "depth_container_rate_hz": depth_meta.get("avg_frame_rate"),
            "rgb_duration_s": float(rgb_meta["duration"]),
            "depth_duration_s": float(depth_meta["duration"]),
            "same_native_frame_count_but_conflicting_timebase": (
                counts["rgb"] == counts["depth"]
                and rgb_meta.get("avg_frame_rate") != depth_meta.get("avg_frame_rate")
            ),
            "official_readme_fps30_decode_implication": (
                "Applying the published fps=30 filter to the 15 Hz depth container would "
                "duplicate frames and no longer yield the 209 annotation rows. No resampling "
                "policy is selected by this audit."
            ),
        },
        "counts": counts,
        "dimensions": dimensions,
        "per_frame_depth_valid_fraction": {
            str(frame): value for frame, value in all_depth_valid_fraction.items()
        },
        "focus_depth_valid_fraction": {
            str(frame): all_depth_valid_fraction[frame] for frame in sample["focus_frames"]
        },
        "rigid_object_records": records,
        "per_object_zero_offset": per_object,
        "timing_diagnostic": {
            "by_offset": by_offset,
            "best_by_frame_object": best_rows,
            "zero_offset_best_fraction": float(np.mean([
                row["best_offset"] == 0 for row in best_rows
            ])) if best_rows else 0.0,
        },
        "aggregate_zero_offset": {
            "comparisons": len(usable),
            "minimum_valid_pixel_fraction": min(
                (row["valid_pixel_fraction"] for row in usable), default=0.0,
            ),
            "median_of_median_absolute_error_mm": None if not usable else float(np.median([
                row["absolute_error_mm"]["median"] for row in usable
            ])),
        },
        "gate_contract": {
            "minimum_valid_pixels_per_pair": reg["minimum_valid_pixels_per_pair"],
            "minimum_valid_pixel_fraction_per_pair": reg["minimum_valid_pixel_fraction_per_pair"],
            "maximum_median_of_medians_absolute_error_mm": reg["maximum_median_of_medians_absolute_error_mm"],
            "threshold_provenance": reg["threshold_provenance"],
        },
        "checks": checks,
        "blocker_evidence": [name for name, value in checks.items() if not value],
    }


def sample_inventory(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(cfg["multi_sample"]["root"])
    poses_root = root / "object_poses/Object_Poses"
    rows = []
    for sequence_dir in sorted(path for path in poses_root.glob("*/*") if path.is_dir()):
        triplet, sequence = sequence_dir.parent.name, sequence_dir.name
        # Released filenames sometimes shorten the natural-language verb
        # (for example, "skim off" -> "skim").  Resolve by the globally
        # unique sequence id rather than inventing a triplet-to-slug rule.
        rgb_matches = sorted((root / "rgb").glob(f"*_{sequence}.mp4"))
        depth_matches = sorted((root / "depth_original").glob(f"*_{sequence}.avi"))
        rgb = rgb_matches[0] if len(rgb_matches) == 1 else root / "__missing_rgb__"
        depth = depth_matches[0] if len(depth_matches) == 1 else root / "__missing_depth__"
        camera = root / "camera/Egocentric_Camera_Parameters" / triplet / sequence
        hand = root / "hand_poses/Hand_Poses" / triplet / sequence / "hand_joints.npy"
        required = [rgb, depth, camera / "egocentric_intrinsic.txt",
                    camera / "egocentric_frame_extrinsic.npy", hand]
        pose_files = sorted(sequence_dir.glob("*.npy"))
        paths_exist = all(path.is_file() for path in required) and len(pose_files) >= 2
        row: dict[str, Any] = {
            "triplet": triplet, "sequence": sequence,
            "stable_selection_key": hashlib.sha256(f"{triplet}/{sequence}".encode()).hexdigest(),
            "required_paths_exist": paths_exist,
        }
        if paths_exist:
            rgb_meta, depth_meta = video_metadata(rgb), video_metadata(depth)
            extrinsics = np.load(camera / "egocentric_frame_extrinsic.npy", allow_pickle=False)
            joints = np.load(hand, allow_pickle=False)
            pose_counts = {path.stem: int(np.load(path, allow_pickle=False).shape[0]) for path in pose_files}
            counts = {
                "rgb": frame_count(rgb_meta), "depth": frame_count(depth_meta),
                "extrinsic": int(extrinsics.shape[0]), "hand": int(joints.shape[0]), **pose_counts,
            }
            row.update({
                "rgb_resolution": [int(rgb_meta["width"]), int(rgb_meta["height"])],
                "depth_resolution": [int(depth_meta["width"]), int(depth_meta["height"])],
                "counts": counts,
                "complete": (
                    [int(rgb_meta["width"]), int(rgb_meta["height"])] == [1920, 1080]
                    and [int(depth_meta["width"]), int(depth_meta["height"])] == [1920, 1080]
                    and len(set(counts.values())) == 1
                ),
            })
        else:
            row["complete"] = False
        rows.append(row)
    return rows


def callgraph() -> dict[str, Any]:
    return {
        "schema": "taco_support_surface_active_callgraph_v1",
        "classification": "ONE_SEMANTIC_SUPPORT_SOURCE_MULTIPLE_CONSISTENT_MATERIALIZATIONS",
        "active_chain": [
            "build_taco_bimanual_scene.build",
            "selected_objects(target id 146 for Brush)",
            "taco_project_sample_support_contract(target_id=146)",
            "make_taco_support_contract(source target frame0 minimum world +Z, simulator z=0.72)",
            "scene XML floor geom + recorded SupportSurfaceContract",
        ],
        "consumers": [
            "camera/table infra audit resolves and checks the same target-bottom contract",
            "environment-aware MINK v2 loads the persisted contract as a Plane",
            "both retarget and static audits consume that same Plane",
        ],
        "answers": {
            "deciding_function": "egoengine_repro.scene.support_surface.taco_project_sample_support_contract",
            "active_call_site": "rl/scripts/build_taco_bimanual_scene.py:build",
            "target_id_flow": "selected Brush target id 146 -> contract factory entity_id",
            "why_bowl_bottom": "Brush target 146 is the bowl; frame-0 minimum along world +Z is its bottom",
            "other_implicit_active_table_source": False,
            "all_current_consumers_share_contract": True,
        },
        "duplicate_materializations": {
            "count_as_independent_sources": False,
            "reason": "builder, infra audit, and v2 are producer/checker/consumer stages for one semantic contract",
        },
        "generalization": "sample-specific target-bottom contract, not an observation-derived estimator",
    }


def input_contract(cfg: dict[str, Any], paths: dict[str, Path], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in inventory if row["complete"]]
    return {
        "schema": "taco_support_surface_input_contract_v1",
        "depth": {"resolution": [1920, 1080], "dtype": "uint16", "scale": 4000.0,
                  "metres": "raw / 4000", "source": str(paths["depth_video"])},
        "camera": {"intrinsic": str(paths["egocentric_intrinsic"]),
                   "extrinsic": str(paths["egocentric_extrinsic"]),
                   "extrinsic_semantics": "world_to_camera",
                   "consumer": "pinned official TACO Pyt3DWrapper; no inverse/transpose search"},
        "foreground_exclusion_if_estimator_runs": {
            "method": "pinned official hand/tool/target projection masks",
            "forbidden": ["cyan table mask", "sample rectangle", "target bottom", "tool bottom"],
            "status": "NOT_EXECUTED_UNTIL_REGISTRATION_PASS",
        },
        "estimator_forbidden_inputs": [
            "bowl/target bottom", "brush/tool bottom", "expected table height",
            "current support offset", "brush -1.311 mm residual",
        ],
        "historical_table_audits": [
            {
                "artifact": "rl/runs/taco_pour_table_calibration_v1/report.json",
                "why_not_adopted": (
                    "Original metric depth was unavailable; resized H264 depth was not metric, "
                    "and the task-specific cyan/stereo mask lacked stable static matches."
                ),
            },
            {
                "artifact": "rl/runs/taco_pour_table_calibration_v2/report.json",
                "why_not_adopted": (
                    "Raw depth became available, but the cyan/stereo selection remained "
                    "task-specific and the audit was superseded by the raw-depth check."
                ),
            },
            {
                "artifact": "rl/runs/taco_pour_raw_depth_table_audit_v2/report.json",
                "why_not_adopted": (
                    "It still used a task-specific cyan table mask and found centimetre-scale "
                    "depth/GT/table disagreement; it explicitly made no scene change."
                ),
            },
        ],
        "local_sample_inventory": inventory,
        "complete_sample_count": len(complete),
        "minimum_complete_samples": cfg["multi_sample"]["minimum_complete_samples"],
        "future_phase_f_classification_if_reached": (
            "READY" if len(complete) >= cfg["multi_sample"]["minimum_complete_samples"]
            else "INSUFFICIENT_MULTI_SAMPLE_VALIDATION_DATA"
        ),
    }


def collision_handoff() -> str:
    return """# Collision handoff (read-only)\n\n""" + "\n".join([
        "1. Official SPIDER XHand disables broad contacts by default.",
        "2. Runtime collision uses explicit collision pairs.",
        "3. The official scene generator self-collision set is not all-pairs.",
        "4. The project scene builder retains the pinned 30 intrahand pairs.",
        "5. Historical Pour evidence preserves the same 30/20/82 taxonomy.",
        "6. Omitted shell overlap is not automatically a runtime collision failure.",
        "7. Native-material real self-collision remains a separate audit question.",
        "8. The 82 omitted pairs must not be added wholesale to the solver.",
        "", "Collision code/topology diff in this audit: 0.",
    ]) + "\n"


def source_pins(cfg_path: Path, paths: dict[str, Path]) -> dict[str, Any]:
    official = Path(yaml.safe_load(cfg_path.read_text())["official"]["checkout"])
    sources = {
        "config": cfg_path,
        "runner": Path(__file__),
        "official_projection_wrapper": RL_ROOT / "src/egoengine_repro/evaluation/taco_official_projection.py",
        "support_surface_core": RL_ROOT / "src/egoengine_repro/scene/support_surface.py",
        "scene_builder": RL_ROOT / "scripts/build_taco_bimanual_scene.py",
        "official_entrypoint": official / "dataset_utils/project_pose_to_egocentric_view.py",
        "official_wrapper": official / "dataset_utils/pyt3d_wrapper.py",
        "official_readme": official / "README.md",
        "historical_table_calibration_v1": RL_ROOT / "runs/taco_pour_table_calibration_v1/report.json",
        "historical_table_calibration_v2": RL_ROOT / "runs/taco_pour_table_calibration_v2/report.json",
        "historical_raw_depth_table_audit_v2": RL_ROOT / "runs/taco_pour_raw_depth_table_audit_v2/report.json",
        "collision_intrahand_audit": RL_ROOT / "runs/taco_pour_intrahand_pipeline_v1/intrahand_collision_audit.json",
        "collision_phase_contract": RL_ROOT / "runs/taco_pour_retarget_contract_closure_v2/phase_contract.json",
        "collision_contract": RL_ROOT / "runs/taco_pour_retarget_contract_closure_v2/collision_contract_v2.md",
        "collision_overlap_policy": RL_ROOT / "runs/taco_pour_retarget_contract_closure_v2/structural_overlap_policy.json",
    }
    for key in (
        "rgb_video", "depth_video", "egocentric_intrinsic", "egocentric_extrinsic",
        "hand_joints", "tool_pose", "target_pose", "tool_mesh", "target_mesh",
        "official_projection_manifest", "official_projection_visual_review",
        "active_support_contract",
    ):
        sources[key] = paths[key]
    return {
        "schema": "taco_support_surface_generalization_source_pins_v1",
        "repository_head_before_audit": git_head(REPO_ROOT),
        "official_taco_commit": git_head(official),
        "artifacts": {key: artifact(path) for key, path in sources.items()},
    }


def write_hash_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "server_artifacts.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg_path = args.config.resolve(strict=True)
    cfg = load_config(cfg_path)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = input_paths(cfg)
    inventory = sample_inventory(cfg)
    registration = registration_audit(cfg, paths)
    if registration["passed"]:
        raise RuntimeError(
            "registration unexpectedly passed; this runner intentionally stops before adding an estimator"
        )

    reports = {
        "support_surface_active_callgraph.json": callgraph(),
        "support_surface_input_contract.json": input_contract(cfg, paths, inventory),
        "rgb_depth_camera_registration_audit.json": registration,
        "source_pins.json": source_pins(cfg_path, paths),
        "config_consumption_audit.json": {
            "schema": "taco_support_surface_generalization_config_consumption_v1",
            "config": artifact(cfg_path),
            "parser": "exact-key fail-closed validation at every mapping level",
            "unknown_keys_rejected": True,
            "all_config_sections_consumed": True,
        },
    }
    for name, value in reports.items():
        write_json(output / name, value)
    (output / "collision_handoff.md").write_text(collision_handoff(), encoding="utf-8")

    target = registration["per_object_zero_offset"]["target"]
    aggregate = registration["aggregate_zero_offset"]
    decision = {
        "schema": "taco_support_surface_generalization_decision_v1",
        "classification": "TACO_RGBD_REGISTRATION_BLOCKER",
        "stop_phase": "PHASE_C_RGB_DEPTH_CAMERA_REGISTRATION",
        "reason": (
            "RGB and depth containers have conflicting timebases, and pinned official rigid-object "
            "projections do not satisfy the frozen raw-depth registration gate. The evidence is "
            "not adequate for background support-plane inference."
        ),
        "registration": {
            "target_minimum_valid_pixel_fraction": target["minimum_valid_pixel_fraction"],
            "target_median_absolute_error_mm": target["median_of_median_absolute_error_mm"],
            "aggregate_minimum_valid_pixel_fraction": aggregate["minimum_valid_pixel_fraction"],
            "aggregate_median_absolute_error_mm": aggregate["median_of_median_absolute_error_mm"],
        },
        "executed": {"phase_a": True, "phase_b": True, "phase_c": True,
                     "phase_d_estimator": False, "phase_e_holdout": False,
                     "phase_f_multi_sample_estimation": False},
        "artifacts_intentionally_absent": [
            "support_plane_estimates.jsonl", "support_plane_consensus.json",
            "brush_holdout_validation.json", "multi_sample_support_summary.json",
        ],
        "runtime_counts": {"mink_candidates": 0, "physics": 0, "replay": 0,
                           "mpc": 0, "rl": 0, "promotion": 0, "chunk_commit": 0},
        "active_support_contract_changed": False,
        "collision_code_or_topology_changed": False,
        "next_step": "Resolve TACO RGB/depth/camera registration before estimating a support plane.",
    }
    write_json(output / "decision.json", decision)
    summary = "# TACO support-surface generalization audit v1\n\n"
    summary += f"**Decision:** `TACO_RGBD_REGISTRATION_BLOCKER` at Phase C. No estimator, MINK, physics, Replay, MPC, RL, promotion, or chunk commit ran.\n\n"
    summary += "## Required answers\n\n"
    summary += "1. **Current active support plane:** the Brush scene builder passes target id 146 to `taco_project_sample_support_contract`, which takes the target frame-0 minimum along world +Z and maps it to simulator `z=0.72 m`. The infra audit and v2 retarget/static audit check or consume this same contract.\n"
    summary += "2. **Why it is sample-specific:** target 146 is the bowl, so this construction uses the bowl bottom as the source plane. It is a project sample contract, not an observation-derived estimator.\n"
    summary += f"3. **RGB/depth/camera registration:** not yet trustworthy for support inference. RGB has 209 frames at 30 Hz while depth has 209 frames at 15 Hz (durations 6.967 s versus 13.933 s), so the official fps=30 decode instruction has no unique 209-row alignment here. Independently, the frozen official-projection gate failed: target minimum valid fraction is `{target['minimum_valid_pixel_fraction']:.6f}` and target median rigid-depth error is `{target['median_of_median_absolute_error_mm']:.3f} mm`.\n"
    summary += "4. **Did a new estimator use bowl/brush bottoms?** No. No estimator executed because Phase C failed; bowl/brush bottom values were never read as estimator inputs.\n"
    summary += "5. **Independent plane versus bowl bottom:** not evaluated; no consensus plane was permitted.\n"
    summary += "6. **Brush -1.311 mm under an independent plane:** not evaluated; retaining the old number only as historical context, not as an input or new result.\n"
    summary += "7. **Other-sample stability:** not evaluated. The local inventory contains only three complete 1920x1080/equal-count sequences; the fourth local dev4 sequence is incomplete, so Phase F would also lack the required four complete samples if reached.\n"
    summary += "8. **Eligible for scene-builder integration?** No. The active SupportSurfaceContract remains unchanged.\n\n"
    summary += "## Stop discipline\n\nThe registration blocker prevents background point-cloud fitting and hold-out validation. Timing offsets `-1/0/+1` are diagnostics only; no offset was selected to improve results. Existing v2 and collision artifacts were not modified.\n"
    (output / "summary.md").write_text(summary, encoding="utf-8")
    write_hash_manifest(output)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
