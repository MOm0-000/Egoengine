"""Measure the Pour tabletop from the released original uint16 depth."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/taco_v1/pour_bowl_plate"
SEQUENCE = "(pour in some, bowl, plate)/20230927_017"
EPISODE = "taco_pour_bowl_plate_20230927_017"
N = 198
HEIGHT, WIDTH = 1080, 1920
STEP = 4
DEPTH_SCALE = 4000.0


def stream_info(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,bits_per_raw_sample,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def rgb_masks(path: Path) -> list[np.ndarray]:
    masks = []
    cap = cv2.VideoCapture(str(path))
    for _ in range(N):
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError("RGB frame decode failed")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array([75, 55, 35], np.uint8),
            np.array([110, 255, 255], np.uint8),
        )[::STEP, ::STEP] > 0
        border = max(4, 8 // (STEP // 2))
        mask[:border] = False
        mask[-border:] = False
        mask[:, :border] = False
        mask[:, -border:] = False
        masks.append(mask)
    cap.release()
    return masks


def decode_depth(path: Path, masks: list[np.ndarray], extrinsics: np.ndarray,
                 intrinsic: np.ndarray) -> tuple[dict, dict]:
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = np.arange(0, WIDTH, STEP)
    v = np.arange(0, HEIGHT, STEP)
    U, V = np.meshgrid(u, v)
    stores = {"same_index": [], "half_rate": []}
    raw_stats = []
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
        stdout=subprocess.PIPE,
    )
    frame_bytes = HEIGHT * WIDTH * 2
    for depth_index in range(N):
        payload = proc.stdout.read(frame_bytes)
        if len(payload) != frame_bytes:
            raise RuntimeError(f"depth frame {depth_index} is truncated")
        raw = np.frombuffer(payload, dtype="<u2").reshape(HEIGHT, WIDTH)
        sampled = raw[::STEP, ::STEP]
        raw_stats.append(
            dict(
                depth_index=depth_index,
                valid_fraction=float(np.mean(sampled > 0)),
                raw_percentiles=[
                    float(x) for x in np.percentile(sampled[sampled > 0], [1, 50, 99])
                ] if np.any(sampled > 0) else [],
            )
        )
        mappings = [("same_index", [depth_index])]
        mappings.append(("half_rate", [2 * depth_index, 2 * depth_index + 1]))
        for mode, rows in mappings:
            for row in rows:
                if row >= N:
                    continue
                good = masks[row] & (sampled >= 1000) & (sampled <= 10000)
                depth_m = sampled[good].astype(float) / DEPTH_SCALE
                if depth_m.size < 100:
                    continue
                camera = np.column_stack(
                    ((U[good] - cx) * depth_m / fx,
                     (V[good] - cy) * depth_m / fy,
                     depth_m)
                )
                # TACO's released extrinsic maps world points to camera points.
                world = (camera - extrinsics[row, :3, 3]) @ extrinsics[row, :3, :3]
                stores[mode].append(world)
    proc.stdout.close()
    if proc.wait() != 0:
        raise RuntimeError("depth decoder failed")
    return stores, {"per_frame": raw_stats}


def fit_plane(chunks: list[np.ndarray]) -> dict:
    points = np.concatenate(chunks, axis=0)
    keep = np.ones(len(points), dtype=bool)
    for _ in range(8):
        design = np.column_stack(
            (points[keep, 0], points[keep, 1], np.ones(int(keep.sum())))
        )
        coeff = np.linalg.lstsq(design, points[keep, 2], rcond=None)[0]
        residual = points[:, 2] - (points[:, :2] @ coeff[:2] + coeff[2])
        median = np.median(residual[keep])
        mad = 1.4826 * np.median(np.abs(residual[keep] - median))
        keep = np.abs(residual - median) <= max(0.003, 4.0 * mad)
    residual = points[:, 2] - (points[:, :2] @ coeff[:2] + coeff[2])
    normal = np.array([-coeff[0], -coeff[1], 1.0])
    normal /= np.linalg.norm(normal)
    return dict(
        points=int(len(points)),
        inliers=int(keep.sum()),
        coefficients_m=coeff.tolist(),
        normal=normal.tolist(),
        tilt_deg=float(np.rad2deg(np.arccos(np.clip(normal[2], -1, 1)))),
        absolute_residual_mm=[
            float(x) for x in np.percentile(np.abs(residual[keep]) * 1000, [50, 90, 95, 99, 100])
        ],
        world_z_percentiles_m=[
            float(x) for x in np.percentile(points[keep, 2], [1, 50, 99])
        ],
        median_xy_m=[float(x) for x in np.median(points[keep, :2], axis=0)],
    )


def temporal_plane_stability(chunks: list[np.ndarray], reference_xy: np.ndarray) -> dict:
    frames = [fit_plane([points]) for points in chunks]
    coefficients = np.asarray([frame["coefficients_m"] for frame in frames])
    normals = np.asarray([frame["normal"] for frame in frames])
    mean_normal = normals.mean(axis=0)
    mean_normal /= np.linalg.norm(mean_normal)
    normal_delta = np.rad2deg(
        np.arccos(np.clip(normals @ mean_normal, -1.0, 1.0))
    )
    z = coefficients[:, :2] @ reference_xy + coefficients[:, 2]
    median_residual = np.asarray(
        [frame["absolute_residual_mm"][0] for frame in frames]
    )
    p95_residual = np.asarray(
        [frame["absolute_residual_mm"][2] for frame in frames]
    )

    def percentiles(values):
        return [float(x) for x in np.percentile(values, [0, 1, 50, 99, 100])]

    return dict(
        frames=len(frames),
        reference_xy_m=[float(x) for x in reference_xy],
        plane_z_at_reference_xy_m_percentiles=percentiles(z),
        plane_z_p99_minus_p1_mm=float((np.percentile(z, 99) - np.percentile(z, 1)) * 1000),
        normal_delta_from_mean_deg_percentiles=percentiles(normal_delta),
        median_absolute_residual_mm_percentiles=percentiles(median_residual),
        p95_absolute_residual_mm_percentiles=percentiles(p95_residual),
        worst_normal_source_row_zero_based=int(np.argmax(normal_delta)),
    )


def object_clearance(vertices: np.ndarray, poses: np.ndarray, coeff: np.ndarray) -> dict:
    values = []
    for pose in poses:
        world = vertices @ pose[:3, :3].T + pose[:3, 3]
        table = world[:, :2] @ coeff[:2] + coeff[2]
        values.append(float(np.min(world[:, 2] - table)))
    values = np.asarray(values)
    return dict(
        initial_m=float(values[0]),
        minimum_m=float(values.min()),
        minimum_row=int(values.argmin()),
        percentiles_m=[float(x) for x in np.percentile(values, [1, 50, 99])],
    )


def audit() -> dict:
    camera_dir = DATA / "camera/Egocentric_Camera_Parameters" / SEQUENCE
    intrinsic = np.loadtxt(camera_dir / "egocentric_intrinsic.txt")
    extrinsics = np.load(camera_dir / "egocentric_frame_extrinsic.npy").astype(float)
    rgb = DATA / "rgb" / f"{EPISODE}.mp4"
    depth = DATA / "depth_original" / f"{EPISODE}.avi"
    masks = rgb_masks(rgb)
    stores, depth_stats = decode_depth(depth, masks, extrinsics, intrinsic)
    planes = {name: fit_plane(chunks) for name, chunks in stores.items()}
    same_coeff = np.asarray(planes["same_index"]["coefficients_m"], dtype=float)
    center_xy = np.asarray(planes["same_index"]["median_xy_m"])
    temporal = temporal_plane_stability(stores["same_index"], center_xy)

    objects = {}
    for role, object_id in (("tool", "022"), ("target", "135")):
        mesh = trimesh.load_mesh(
            DATA / "object_models/object_models_released" / f"{object_id}_cm.obj",
            process=False,
        )
        mesh.apply_scale(0.01)
        poses = np.load(
            DATA / "object_poses/Object_Poses" / SEQUENCE / f"{role}_{object_id}.npy",
            allow_pickle=False,
        ).astype(float)
        objects[role] = object_clearance(mesh.vertices, poses, same_coeff)

    with np.load(ROOT / "runs/taco_pour_bimanual_gt_v1/human_reference.npz") as reference:
        sim_transform = reference["T_sim_world"]
    sim_table_z = float(planes["same_index"]["coefficients_m"][2] + sim_transform[2, 3])
    source_assumed_z = float(0.72 - sim_transform[2, 3])
    depth_plane_at_center = float(
        same_coeff[0] * center_xy[0] + same_coeff[1] * center_xy[1] + same_coeff[2]
    )
    return dict(
        status="raw_depth_table_measured_no_scene_change",
        source={
            "depth_path": str(depth.resolve()),
            "depth_stream": stream_info(depth),
            "official_scale": DEPTH_SCALE,
            "decode_dtype": "uint16_little_endian",
            "valid_rule": "raw > 0",
            "rgb_frames": N,
            "rgb_fps": 30.0,
            "official_decoder_note": "TACO README requests fps=30; source container advertises 15 fps",
        },
        depth_statistics=depth_stats,
        timing_comparison=planes,
        same_index_temporal_stability=temporal,
        tabletop={
            "selection": "RGB cyan table mask, border removed; no GT object/hand pixels used",
            "mask_pixels_median": float(np.median([m.sum() for m in masks])),
            "plane_z_at_median_xy_m": depth_plane_at_center,
            "plane_tilt_deg": planes["same_index"]["tilt_deg"],
            "sim_plane_z_at_median_xy_m": float(depth_plane_at_center + sim_transform[2, 3]),
            "paper_sim_table_z_m": 0.72,
            "sim_plane_minus_paper_table_m": float(depth_plane_at_center + sim_transform[2, 3] - 0.72),
            "source_assumed_table_z_m": source_assumed_z,
            "plane_intercept_m": float(same_coeff[2]),
        },
        object_clearance_to_measured_plane_m=objects,
        interpretation=(
            "The original depth supports an axial metric-depth reading and a stable table plane. "
            "The measured plane is above the released object bottoms by roughly centimeter scale; "
            "this is a depth/GT/table registration discrepancy, not evidence to move the table "
            "until temporal and camera registration are independently certified."
        ),
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "timing": report["timing_comparison"],
        "tabletop": report["tabletop"],
        "object_clearance": report["object_clearance_to_measured_plane_m"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
