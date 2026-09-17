"""Render TACO object-only metric depth as an isolated GT proxy.

The output is intentionally labeled ``rendered_gt_proxy``.  It is suitable
only for the oracle-depth ablation and must never be presented as manually
captured or annotated depth ground truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
import zarr

from video_to_spider.schemas import SCHEMA_VERSION

from ..artifacts import artifact_record


PROXY_LABEL = "rendered_gt_proxy"


def camera_coordinate_sign(T_camera_object: np.ndarray) -> np.ndarray:
    """Return the projective xyz sign needed to make object depth positive.

    A negative value is a homogeneous/projective repair only.  It must not be
    interpreted as a rigid camera transform because ``-I`` is not in SO(3).
    """
    poses = np.asarray(T_camera_object, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("T_camera_object must have shape (T,4,4)")
    return np.where(poses[:, 2, 3] < 0.0, -1.0, 1.0).astype(np.int8)


class _CudaRasterizer:
    def __init__(self, mesh: trimesh.Trimesh, K: np.ndarray, image_shape: tuple[int, int]):
        import torch
        import nvdiffrast.torch as dr

        self.torch = torch
        self.dr = dr
        self.vertices = torch.as_tensor(
            np.asarray(mesh.vertices, dtype=np.float32), device="cuda",
        )
        self.faces = torch.as_tensor(
            np.asarray(mesh.faces, dtype=np.int32), device="cuda",
        )
        self.K = np.asarray(K, dtype=np.float64)
        self.height, self.width = image_shape
        self.context = dr.RasterizeCudaContext()

    def render(self, pose: np.ndarray, coordinate_sign: int) -> tuple[np.ndarray, np.ndarray]:
        torch, dr = self.torch, self.dr
        transform = torch.as_tensor(np.asarray(pose, np.float32), device="cuda")
        camera = self.vertices @ transform[:3, :3].T + transform[:3, 3]
        camera = camera * float(coordinate_sign)
        z = camera[:, 2]
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        clip = torch.stack([
            (2.0 * fx / self.width) * camera[:, 0]
            + (2.0 * cx / self.width - 1.0) * z,
            (-2.0 * fy / self.height) * camera[:, 1]
            + (1.0 - 2.0 * cy / self.height) * z,
            0.5 * z,
            z,
        ], dim=1)
        rast, _ = dr.rasterize(
            self.context, clip[None], self.faces,
            resolution=(self.height, self.width),
        )
        depth, _ = dr.interpolate(camera[:, 2:3][None].contiguous(), rast, self.faces)
        mask = rast[0, ..., 3] > 0
        depth = depth[0, ..., 0]
        # nvdiffrast returns raster rows in OpenGL bottom-up order; the run RGB
        # and OpenCV intrinsics use top-down image rows.
        mask = torch.flip(mask, dims=[0])
        depth = torch.flip(depth, dims=[0])
        depth = torch.where(mask & torch.isfinite(depth) & (depth > 0), depth, 0.0)
        return (
            depth.detach().cpu().numpy().astype(np.float32),
            mask.detach().cpu().numpy().astype(bool),
        )


def _frame_rows(run_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((run_dir / "frames/frame_index.json").read_text(encoding="utf-8"))
    rows = payload.get("frames", [])
    if not rows:
        raise ValueError("oracle-depth rendering requires RGB frames")
    return rows


def render_taco_oracle_metric_depth(
    run_dir: str | Path, ground_truth_manifest: str | Path, *, overwrite: bool = False,
) -> Path:
    root = Path(run_dir).resolve()
    gt_path = Path(ground_truth_manifest).resolve()
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    if gt.get("dataset") != "TACO V1" or gt.get("scope") != "evaluation_only":
        raise ValueError("oracle depth requires an evaluation-only TACO bundle")
    mesh_path = Path(gt["artifacts"]["mesh"]["path"]).resolve()
    trajectory_path = Path(gt["artifacts"]["object_trajectory"]["path"]).resolve()
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"invalid TACO mesh: {mesh_path}")
    with np.load(trajectory_path, allow_pickle=False) as artifact:
        gt_frames = np.asarray(artifact["frame_indices"], dtype=np.int64)
        timestamps = np.asarray(artifact["timestamps_s"], dtype=np.float64)
        poses = np.asarray(artifact["T_camera_object"], dtype=np.float64)
    rows = _frame_rows(root)
    run_frames = np.asarray([row["source_frame_index"] for row in rows], dtype=np.int64)
    if not np.array_equal(run_frames, gt_frames):
        raise ValueError("run and oracle object trajectory frame indices differ")
    first = cv2.imread(str(root / rows[0]["rgb_path"]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError("cannot read first RGB frame")
    height, width = first.shape[:2]
    K = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    signs = camera_coordinate_sign(poses)

    proxy_dir = root / PROXY_LABEL
    proxy_zarr = proxy_dir / "object_metric_depth.zarr"
    depth_dir = root / "depth"
    depth_link = depth_dir / "metric_depth.zarr"
    metadata_path = proxy_dir / "metadata.json"
    if any(path.exists() or path.is_symlink() for path in (proxy_zarr, depth_link, metadata_path)):
        if not overwrite:
            raise FileExistsError("oracle-depth output exists; pass --overwrite")
        if proxy_zarr.exists():
            shutil.rmtree(proxy_zarr)
        if depth_link.is_symlink() or depth_link.is_file():
            depth_link.unlink()
        elif depth_link.exists():
            shutil.rmtree(depth_link)
        if metadata_path.exists():
            metadata_path.unlink()
    proxy_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(proxy_zarr), mode="w")
    chunks = (1, min(height, 270), min(width, 480))
    depth_array = group.create_dataset(
        "depth_m", shape=(len(rows), height, width), chunks=chunks, dtype="f4",
    )
    valid_array = group.create_dataset(
        "valid", shape=(len(rows), height, width), chunks=chunks, dtype="bool",
    )
    mask_array = group.create_dataset(
        "object_geometry_mask", shape=(len(rows), height, width), chunks=chunks, dtype="bool",
    )
    group.create_dataset("frame_indices", data=run_frames, dtype="i8")
    group.create_dataset("timestamps_s", data=timestamps, dtype="f8")
    group.create_dataset("camera_coordinate_sign", data=signs, dtype="i1")
    group.attrs.update({
        "schema_version": SCHEMA_VERSION,
        "units": "meter",
        "pixel_mapping": "original RGB resolution",
        "quality_label": PROXY_LABEL,
        "scope": "isolated_oracle_ablation_only",
        "outside_object_depth": "zero_and_invalid",
    })
    rasterizer = _CudaRasterizer(mesh, K, (height, width))
    areas: list[float] = []
    depth_medians: list[float] = []
    for index, (pose, sign) in enumerate(zip(poses, signs)):
        depth, mask = rasterizer.render(pose, int(sign))
        depth_array[index] = depth
        valid_array[index] = mask
        mask_array[index] = mask
        areas.append(float(mask.mean()))
        if mask.any():
            depth_medians.append(float(np.median(depth[mask])))
    depth_link.symlink_to(proxy_zarr, target_is_directory=True)
    depth_metadata = {
        "schema_version": SCHEMA_VERSION,
        "quality_label": PROXY_LABEL,
        "description": "object-only metric depth rendered from TACO mesh, object pose, and camera",
        "scope": "isolated_oracle_ablation_only",
        "uses_ground_truth": True,
        "manual_ground_truth": False,
        "complete_scene_depth": False,
        "object_geometry_mask_in_zarr": "object_geometry_mask",
        "outside_object_depth": "zero_and_invalid",
        "projective_sign_repair_frame_count": int(np.count_nonzero(signs < 0)),
        "projective_sign_repair_is_se3": False,
        "frame_count": len(rows),
        "image_size": [width, height],
        "mean_object_area_ratio": float(np.mean(areas)),
        "median_object_depth_m": float(np.median(depth_medians)) if depth_medians else None,
        "inputs": {
            "ground_truth_manifest": artifact_record(gt_path),
            "mesh": artifact_record(mesh_path),
            "object_trajectory": artifact_record(trajectory_path),
            "intrinsics": artifact_record(root / "calibration/intrinsics.npy"),
        },
    }
    metadata_path.write_text(json.dumps(depth_metadata, indent=2) + "\n", encoding="utf-8")
    (depth_dir / "metadata.json").write_text(
        json.dumps({**depth_metadata, "proxy_zarr": artifact_record(proxy_zarr)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(render_taco_oracle_metric_depth(
        args.run_dir, args.ground_truth_manifest, overwrite=args.overwrite,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
