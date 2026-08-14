#!/usr/bin/env python3
"""Prepare an Aria Digital Twin VRS pair and optionally run ``ingest-stereo``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from video_to_spider.ingest.adt_stereo import (
    DEFAULT_LEFT_CAMERA_LABEL,
    DEFAULT_MAX_SYNC_NS,
    DEFAULT_RECTIFIED_FOCAL_LENGTH,
    DEFAULT_RECTIFIED_HEIGHT,
    DEFAULT_RECTIFIED_WIDTH,
    DEFAULT_RIGHT_CAMERA_LABEL,
    prepare_adt_stereo,
)
from video_to_spider.ingest.stereo import ingest_rectified_stereo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-vrs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-camera-label", default=DEFAULT_LEFT_CAMERA_LABEL)
    parser.add_argument("--right-camera-label", default=DEFAULT_RIGHT_CAMERA_LABEL)
    parser.add_argument("--rectified-width", type=int, default=DEFAULT_RECTIFIED_WIDTH)
    parser.add_argument("--rectified-height", type=int, default=DEFAULT_RECTIFIED_HEIGHT)
    parser.add_argument("--focal-length", type=float, default=DEFAULT_RECTIFIED_FOCAL_LENGTH)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--max-sync-ns", type=int, default=DEFAULT_MAX_SYNC_NS)
    parser.add_argument("--closed-loop-trajectory", type=Path)
    parser.add_argument("--static-camera", action="store_true")

    parser.add_argument("--run-ingest", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--episode-id")
    parser.add_argument("--instruction")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args()
    prepared = prepare_adt_stereo(
        video_vrs=args.video_vrs,
        output_dir=args.output_dir,
        left_camera_label=args.left_camera_label,
        right_camera_label=args.right_camera_label,
        rectified_width=args.rectified_width,
        rectified_height=args.rectified_height,
        focal_length=args.focal_length,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_sync_ns=args.max_sync_ns,
        closed_loop_trajectory_path=args.closed_loop_trajectory,
        static_camera=args.static_camera,
    )
    print(json.dumps(json.loads(prepared.metadata_path.read_text(encoding="utf-8")), indent=2))

    if not args.run_ingest:
        print("Prepared files ready; pass them to `video_to_spider.cli ingest-stereo`.")
        return 0

    required = {"run_dir": args.run_dir, "task": args.task, "episode_id": args.episode_id, "instruction": args.instruction}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        print(f"--run-ingest requires: {', '.join('--' + name.replace('_', '-') for name in missing)}", file=sys.stderr)
        return 2
    if args.static_camera:
        camera_poses_path = None
    else:
        camera_poses_path = prepared.camera_poses_path
    result = ingest_rectified_stereo(
        left_dir=prepared.left_dir,
        right_dir=prepared.right_dir,
        intrinsics_path=prepared.intrinsics_path,
        right_intrinsics_path=prepared.right_intrinsics_path,
        common_valid_mask_path=prepared.common_valid_mask_path,
        baseline_m=prepared.baseline_m,
        output_dir=args.run_dir,
        task=args.task,
        episode_id=args.episode_id,
        instruction=args.instruction,
        fps=prepared.fps,
        camera_poses_path=camera_poses_path,
        timestamps_path=prepared.timestamps_path,
        frame_indices_path=prepared.frame_indices_path,
        static_camera=args.static_camera,
    )
    print(f"ingest manifest: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
