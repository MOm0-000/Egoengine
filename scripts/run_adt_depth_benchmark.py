#!/usr/bin/env python3
"""Batch-run the ADT P1 depth benchmark across downloaded sequences.

This is the operational driver for the P1 stage of
``docs/ADT_STEREO_P0.md``.  It does, per sequence:

1. resolve the object instance id from ``instances.json`` (prototype or
   instance name),
2. pick a short frame window where that object is visible in the left SLAM
   stream (using GT segmentation),
3. prepare rectified stereo,
4. ingest into the standard run directory,
5. run unmodified FoundationStereo (``v2s-depth`` env),
6. extract ADT GT depth/segmentation in the same rectified reference,
7. compute the stereo-depth metrics table.

Nothing here fits scale or shift to GT.  FoundationStereo remains the
unchanged pinned checkout, and GT is only used for evaluation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from video_to_spider.benchmark import adt_gt
from video_to_spider.ingest import adt_stereo


V2S_CORE_PY = Path("/home/zzx/miniconda3/envs/v2s-core/bin/python")
V2S_FS_PY = Path("/home/zzx/miniconda3/envs/v2s-sam3d/bin/python")


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("  + " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd or _REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        raise RuntimeError(f"command failed rc={proc.returncode}: {cmd[0]}")


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_object_uids(instances_path: Path, prototypes: list[str], prefer_dynamic: bool = False) -> dict[str, int]:
    payload = _json_load(instances_path)
    matches: dict[str, int] = {}
    wanted = [(p, p.lower()) for p in prototypes]
    for uid, info in payload.items():
        if prefer_dynamic and str(info.get("motion_type", "")).lower() != "dynamic":
            continue
        proto = str(info.get("prototype_name", "")).lower()
        inst = str(info.get("instance_name", "")).lower()
        for original, w in wanted:
            if proto == w or inst == w:
                matches.setdefault(original, int(uid))
    if not matches and prefer_dynamic:
        return resolve_object_uids(instances_path, prototypes, prefer_dynamic=False)
    return matches


def select_object_window(
    *,
    video_vrs: Path,
    segmentation_vrs: Path,
    object_uid: int,
    frames: int,
    stride: int,
    min_pixels: int,
    start_frame: int | None,
    end_frame: int | None,
) -> tuple[int, int, dict[str, Any]]:
    if start_frame is not None and end_frame is not None:
        return start_frame, end_frame, {"mode": "manual"}

    data_provider, _calibration, _mps, sensor_data, _sophus = adt_stereo._load_projectaria()
    video_provider = data_provider.create_vrs_data_provider(str(video_vrs))
    seg_provider = data_provider.create_vrs_data_provider(str(segmentation_vrs))
    left_video_stream = video_provider.get_stream_id_from_label(adt_gt.DEFAULT_LEFT_CAMERA_LABEL)
    left_seg_stream = adt_gt.find_left_gt_stream_id(
        seg_provider, video_provider, adt_gt.DEFAULT_LEFT_CAMERA_LABEL
    )
    video_left_ts = np.asarray(
        video_provider.get_timestamps_ns(left_video_stream, sensor_data.TimeDomain.DEVICE_TIME),
        dtype=np.int64,
    )
    seg_left_ts = np.asarray(
        seg_provider.get_timestamps_ns(left_seg_stream, sensor_data.TimeDomain.DEVICE_TIME),
        dtype=np.int64,
    )
    num_seg = int(seg_provider.get_num_data(left_seg_stream))
    visible_indices: list[int] = []
    for seg_i in range(0, num_seg, max(1, stride)):
        ts = int(seg_left_ts[seg_i])
        video_i = int(np.searchsorted(video_left_ts, ts, side="left"))
        video_i = min(max(video_i, 0), video_left_ts.size - 1)
        image, _record = seg_provider.get_image_data_by_index(left_seg_stream, seg_i)
        arr = np.asarray(image.to_numpy_array())
        if arr.ndim == 3:
            arr = arr[..., 0]
        count = int((arr == int(object_uid)).sum())
        if count >= min_pixels:
            visible_indices.append(video_i)

    if not visible_indices:
        raise RuntimeError(f"object {object_uid} not visible in left SLAM segmentation")
    # Expand the longest sampled-visible run to a contiguous full-resolution window.
    visible_indices = sorted(set(visible_indices))
    best_start = best_end = visible_indices[0]
    cur_start = visible_indices[0]
    prev = visible_indices[0]
    for value in visible_indices[1:]:
        if value - prev > stride * 2:
            if prev - cur_start > best_end - best_start:
                best_start, best_end = cur_start, prev
            cur_start = value
        prev = value
    if prev - cur_start > best_end - best_start:
        best_start, best_end = cur_start, prev

    center = (best_start + best_end) // 2
    start = center - frames // 2
    start = min(max(start, 0), max(0, int(video_left_ts.size) - frames))
    end = start + frames
    audit = {
        "mode": "segmentation_scan",
        "sampled_visible_frames": len(visible_indices),
        "longest_run_start": best_start,
        "longest_run_end": best_end,
        "selected_start": start,
        "selected_end": end,
    }
    return start, end, audit


def _sequence_ready(seq_dir: Path) -> bool:
    return all(
        (seq_dir / rel).is_file()
        for rel in (
            "vrs_files/video.vrs",
            "vrs_files/depth_images.vrs",
            "vrs_files/segmentations.vrs",
            "mps/slam/closed_loop_trajectory.csv",
            "instances.json",
        )
    )


def run_sequence(
    *,
    uid: str,
    prototypes: list[str],
    adt_root: Path,
    runs_root: Path,
    frames: int,
    stride: int,
    min_pixels: int,
    prefer_dynamic: bool,
    start_frame: int | None,
    end_frame: int | None,
    force: bool,
    gpu_index: int,
    gpu_uuid: str,
    fs_repository: Path,
    fs_checkpoint: Path,
    fs_config: Path,
) -> dict[str, Any]:
    seq_dir = adt_root / uid
    if not _sequence_ready(seq_dir):
        raise RuntimeError(f"sequence not fully downloaded: {seq_dir}")
    instances_path = seq_dir / "instances.json"
    matches = resolve_object_uids(instances_path, prototypes, prefer_dynamic=prefer_dynamic)
    if not matches:
        raise RuntimeError(f"no object prototype matched {prototypes} in {instances_path}")
    proto, object_uid = next(iter(matches.items()))

    start, end, window_audit = select_object_window(
        video_vrs=seq_dir / "vrs_files/video.vrs",
        segmentation_vrs=seq_dir / "vrs_files/segmentations.vrs",
        object_uid=object_uid,
        frames=frames,
        stride=stride,
        min_pixels=min_pixels,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    slug = f"adt_{uid}_{proto}_f{start}_{end}"
    prepared_dir = seq_dir / f"stereo_prepared_{proto}_f{start}_{end}"
    run_dir = runs_root / slug

    if not (prepared_dir / "adt_stereo_prepare.json").is_file():
        _run(
            [
                V2S_CORE_PY,
                "scripts/prepare_adt_stereo.py",
                "--video-vrs", seq_dir / "vrs_files/video.vrs",
                "--output-dir", prepared_dir,
                "--closed-loop-trajectory", seq_dir / "mps/slam/closed_loop_trajectory.csv",
                "--start-frame", start,
                "--end-frame", end,
                "--run-ingest",
                "--run-dir", run_dir,
                "--task", "adt",
                "--episode-id", uid,
                "--instruction", "pick up the object",
            ]
        )
    else:
        print(f"  [skip] prepare exists: {prepared_dir}", flush=True)

    fs_out = run_dir / "evaluation/foundationstereo"
    fs_meta = fs_out / "metadata.json"
    if not force and fs_meta.is_file():
        print(f"  [skip] FoundationStereo exists: {fs_meta}", flush=True)
    else:
        env = {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "PYTHONNOUSERSITE": "1",
        }
        _run(
            [
                V2S_FS_PY,
                "-m", "video_to_spider.adapters.depth_foundationstereo",
                "--run-dir", run_dir,
                "--repository", fs_repository,
                "--checkpoint", fs_checkpoint,
                "--config", fs_config,
                "--physical-gpu-index", gpu_index,
                "--gpu-uuid", gpu_uuid,
                "--scale", "1.0",
                "--valid-iters", "32",
                "--low-memory",
                "--output-dir", fs_out,
            ],
            env=env,
        )

    gt_out = run_dir / "evaluation/adt_gt"
    if not force and (gt_out / "adt_gt.zarr").is_dir():
        print(f"  [skip] GT exists: {gt_out}", flush=True)
    else:
        gt_cmd = [
            V2S_CORE_PY,
            "-m", "video_to_spider.benchmark.adt_gt",
            "--video-vrs", seq_dir / "vrs_files/video.vrs",
            "--depth-vrs", seq_dir / "vrs_files/depth_images.vrs",
            "--segmentation-vrs", seq_dir / "vrs_files/segmentations.vrs",
            "--prepared-dir", prepared_dir,
            "--object-uid", object_uid,
            "--output-dir", gt_out,
        ]
        if force:
            gt_cmd.append("--overwrite")
        _run(gt_cmd)

    eval_out = run_dir / "evaluation/adt_stereo_depth"
    if not force and (eval_out / "stereo_depth_metrics.json").is_file():
        print(f"  [skip] eval exists: {eval_out}", flush=True)
    else:
        _run(
            [
                V2S_CORE_PY,
                "-m", "video_to_spider.eval.adt_stereo_depth",
                "--pred-depth-zarr", fs_out / "metric_depth.zarr",
                "--gt-zarr", gt_out / "adt_gt.zarr",
                "--output-dir", eval_out,
                "--prepared-dir", prepared_dir,
            ]
        )

    metrics_path = eval_out / "stereo_depth_metrics.json"
    metrics = _json_load(metrics_path)
    summary = {
        "sequence": uid,
        "prototype": proto,
        "object_uid": object_uid,
        "window": [start, end],
        "window_audit": window_audit,
        "run_dir": str(run_dir),
        "metrics": metrics.get("metrics"),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", dest="sequences", required=True)
    parser.add_argument("--prototype", action="append", dest="prototypes", required=True)
    parser.add_argument("--adt-root", type=Path, default=Path("/data_all/zzx/egoengine/adt_data"))
    parser.add_argument("--runs-root", type=Path, default=_REPO_ROOT / "runs")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--min-pixels", type=int, default=40)
    parser.add_argument("--prefer-dynamic", action="store_true")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=6)
    parser.add_argument("--gpu-uuid", default="GPU-70b5cd84-9fee-40a5-9109-513527bc8fad")
    parser.add_argument(
        "--fs-repository",
        type=Path,
        default=Path(
            "/data_all/zzx/egoengine/experiments/stereo_bakeoff_20260812/foundationstereo/third_party/FoundationStereo"
        ),
    )
    parser.add_argument(
        "--fs-checkpoint",
        type=Path,
        default=Path(
            "/data_all/zzx/egoengine/experiments/stereo_bakeoff_20260812/foundationstereo/checkpoints/23-51-11/model_best_bp2.pth"
        ),
    )
    parser.add_argument(
        "--fs-config",
        type=Path,
        default=Path(
            "/data_all/zzx/egoengine/experiments/stereo_bakeoff_20260812/foundationstereo/checkpoints/23-51-11/cfg.yaml"
        ),
    )
    args = parser.parse_args(argv)

    results: list[dict[str, Any]] = []
    for uid in args.sequences:
        print(f"[sequence] {uid}", flush=True)
        summary = run_sequence(
            uid=uid,
            prototypes=args.prototypes,
            adt_root=args.adt_root,
            runs_root=args.runs_root,
            frames=args.frames,
            stride=args.stride,
            min_pixels=args.min_pixels,
            prefer_dynamic=args.prefer_dynamic,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            force=args.force,
            gpu_index=args.gpu_index,
            gpu_uuid=args.gpu_uuid,
            fs_repository=args.fs_repository,
            fs_checkpoint=args.fs_checkpoint,
            fs_config=args.fs_config,
        )
        results.append(summary)

    out_path = args.runs_root / "adt_depth_benchmark_summary.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"summary -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
