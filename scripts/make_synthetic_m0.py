#!/usr/bin/env python3
"""Create a deterministic CPU-only synthetic artifact and SPIDER export for M0."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from video_to_spider.export.spider import export_spider_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("runs/synthetic_m0"))
    parser.add_argument("--spider-package-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame_count, fps = 61, 30.0
    timestamps = np.arange(frame_count, dtype=np.float64) / fps
    phase = np.linspace(0, 2 * np.pi, frame_count)
    object_transform = np.repeat(np.eye(4)[None], frame_count, axis=0)
    object_transform[:, 0, 3] = 0.008 * np.sin(phase)
    object_transform[:, 1, 3] = -0.06
    object_transform[:, 2, 3] = 0.04
    wrist_transform = np.repeat(np.eye(4)[None], frame_count, axis=0)
    wrist_transform[:, :3, 3] = [-0.20, 0.20, 0.10]
    wrist_transform[:, 0, 3] += 0.004 * np.sin(phase)
    fingertip_base = np.array([
        [-0.13, 0.17, 0.135], [-0.12, 0.17, 0.110], [-0.115, 0.19, 0.090],
        [-0.12, 0.205, 0.072], [-0.135, 0.205, 0.060],
    ])
    fingertips = np.repeat(fingertip_base[None, None], frame_count, axis=0)
    fingertips[:, :, :, 0] += (0.004 * np.sin(phase))[:, None, None]
    mano_pose = np.broadcast_to(np.eye(3), (frame_count, 1, 15, 3, 3)).copy()
    aligned_path = artifact_dir / "aligned_trajectory.npz"
    np.savez_compressed(
        aligned_path, frame_indices=np.arange(frame_count), timestamps_s=timestamps,
        T_sim_object=object_transform[:, None], T_sim_wrist=wrist_transform[:, None],
        fingertips_sim=fingertips, mano_pose=mano_pose, mano_betas=np.zeros((1, 10)),
        object_scale_to_m=np.ones(1), valid_object=np.ones((frame_count, 1), bool),
        valid_hand=np.ones((frame_count, 1), bool), confidence_object=np.ones((frame_count, 1)),
        confidence_hand=np.ones((frame_count, 1)),
    )
    contact_path = artifact_dir / "contact.npz"
    np.savez_compressed(
        contact_path, frame_indices=np.arange(frame_count), timestamps_s=timestamps,
        contact=np.zeros((frame_count, 1, 5), dtype=np.float32),
        contact_pos_object_local=np.zeros((1, 5, 3), dtype=np.float32),
    )
    mesh_path = artifact_dir / "visual.obj"
    trimesh.creation.icosphere(subdivisions=2, radius=0.04).export(mesh_path)
    result = export_spider_dataset(
        aligned_path=aligned_path, contact_path=contact_path, visual_mesh_path=mesh_path,
        dataset_root=root / "dataset", task="synthetic_m0", data_id=0,
        source_run_id="synthetic_m0", hand_sides=["right"],
        spider_package_root=args.spider_package_root, embodiment_type="right", robot_type="xhand",
    )
    for name, path in result.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()

