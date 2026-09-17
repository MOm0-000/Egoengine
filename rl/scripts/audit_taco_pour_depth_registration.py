"""Compare Pour's original metric depth with released mesh/MANO surfaces."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from egoengine_repro.evaluation.taco_surface import reconstruct_taco_mano
from egoengine_repro.retarget.paper_audit import artifact

DATA = ROOT / "data/taco_v1/pour_bowl_plate"
SEQUENCE = "(pour in some, bowl, plate)/20230927_017"
EPISODE = "taco_pour_bowl_plate_20230927_017"
MANO = ROOT / "data/taco_v1/hand_poses_v1/mano_v1_2/models"
ROWS = (0, 150, 180, 197)
HEIGHT, WIDTH = 1080, 1920
DEPTH_SCALE = 4000.0


def decode_selected_depth(path: Path) -> dict[int, np.ndarray]:
    selected = {}
    process = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "gray16le", "-"],
        stdout=subprocess.PIPE,
    )
    frame_bytes = HEIGHT * WIDTH * 2
    assert process.stdout is not None
    for row in range(198):
        payload = process.stdout.read(frame_bytes)
        if len(payload) != frame_bytes:
            raise RuntimeError(f"depth frame {row} is truncated")
        if row in ROWS:
            raw = np.frombuffer(payload, dtype="<u2").reshape(HEIGHT, WIDTH)
            selected[row] = raw.copy().astype(np.float32) / DEPTH_SCALE
    process.stdout.close()
    if process.wait() != 0 or tuple(selected) != ROWS:
        raise RuntimeError("depth decoder failed")
    return selected


def load_selected_rgb(path: Path) -> dict[int, np.ndarray]:
    selected = {}
    capture = cv2.VideoCapture(str(path))
    for row in range(198):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"RGB frame {row} is truncated")
        if row in ROWS:
            selected[row] = frame
    capture.release()
    return selected


def raycast_camera_depth(vertices_world: np.ndarray, faces: np.ndarray,
                         extrinsic: np.ndarray, intrinsic: np.ndarray):
    camera = vertices_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    projected = camera @ intrinsic.T
    projected = projected[:, :2] / projected[:, 2:]
    lower = np.maximum(np.floor(projected.min(axis=0) - 3).astype(int), 0)
    upper = np.minimum(
        np.ceil(projected.max(axis=0) + 3).astype(int), [WIDTH - 1, HEIGHT - 1]
    )
    x0, y0 = lower
    x1, y1 = upper
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(camera.astype(np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    directions = np.stack(
        ((xx - intrinsic[0, 2]) / intrinsic[0, 0],
         (yy - intrinsic[1, 2]) / intrinsic[1, 1], np.ones_like(xx)), axis=-1,
    ).astype(np.float32)
    rays = np.concatenate((np.zeros_like(directions), directions), axis=-1)
    rendered = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return rendered, (slice(y0, y1 + 1), slice(x0, x1 + 1))


def residual_record(observed: np.ndarray, rendered: np.ndarray,
                    selector: np.ndarray | None = None):
    geometry = np.isfinite(rendered)
    interior = cv2.erode(geometry.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    valid = interior & np.isfinite(observed) & (observed > 0)
    if selector is not None:
        valid &= selector
    residual = observed[valid] - rendered[valid]
    near = residual[np.abs(residual) < 0.05]
    if len(near) < 100:
        raise ValueError("insufficient corresponding depth samples")
    return dict(
        rendered_interior_pixels=int(interior.sum()),
        valid_pixels=int(len(residual)),
        within_50mm_fraction=float(len(near) / len(residual)),
        raw_minus_rendered_depth_mm_percentiles=[
            float(x) for x in np.percentile(near * 1000, [10, 25, 50, 75, 90])
        ],
    ), near


def boundary_alignment(observed: np.ndarray, rgb: np.ndarray,
                       rendered: np.ndarray) -> dict:
    geometry = np.isfinite(rendered).astype(np.uint8)
    boundary = cv2.morphologyEx(geometry, cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8)) > 0
    dx = np.abs(np.diff(observed, axis=1, prepend=observed[:, :1]))
    dy = np.abs(np.diff(observed, axis=0, prepend=observed[:1]))
    depth_edges = (np.maximum(dx, dy) > 0.008) & (observed > 0)
    rgb_edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), 50, 120) > 0
    result = {}
    for name, edges in (("raw_depth", depth_edges), ("rgb", rgb_edges)):
        distance = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 5)
        result[name + "_edge_distance_px_percentiles"] = [
            float(x) for x in np.percentile(distance[boundary], [50, 90])
        ]
    return result


def inspect() -> dict:
    camera_dir = DATA / "camera/Egocentric_Camera_Parameters" / SEQUENCE
    intrinsic = np.loadtxt(camera_dir / "egocentric_intrinsic.txt")
    extrinsics = np.load(camera_dir / "egocentric_frame_extrinsic.npy").astype(float)
    rgb_path = DATA / "rgb" / f"{EPISODE}.mp4"
    depth_path = DATA / "depth_original" / f"{EPISODE}.avi"
    depths = decode_selected_depth(depth_path)
    rgbs = load_selected_rgb(rgb_path)
    object_records, hand_records = {}, {}
    object_residuals, hand_residuals = [], []
    inputs = [depth_path, rgb_path, camera_dir / "egocentric_intrinsic.txt",
              camera_dir / "egocentric_frame_extrinsic.npy"]

    for role, object_id, label in (("tool", "022", "bowl"),
                                   ("target", "135", "plate")):
        mesh_path = DATA / "object_models/object_models_released" / f"{object_id}_cm.obj"
        pose_path = DATA / "object_poses/Object_Poses" / SEQUENCE / f"{role}_{object_id}.npy"
        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh.apply_scale(0.01)
        poses = np.load(pose_path, allow_pickle=False).astype(float)
        inputs += [mesh_path, pose_path]
        rows = []
        for row in ROWS:
            world = mesh.vertices @ poses[row, :3, :3].T + poses[row, :3, 3]
            rendered, box = raycast_camera_depth(world, mesh.faces, extrinsics[row], intrinsic)
            record, residual = residual_record(depths[row][box], rendered)
            record.update(source_row_zero_based=row,
                          boundary_alignment=boundary_alignment(
                              depths[row][box], rgbs[row][box], rendered))
            rows.append(record)
            object_residuals.append(residual)
        object_records[label] = dict(role=role, object_id=object_id, frames=rows)

    hands_dir = DATA / "hand_poses/Hand_Poses" / SEQUENCE
    for side in ("right", "left"):
        pose_path = hands_dir / f"{side}_hand.pkl"
        shape_path = hands_dir / f"{side}_hand_shape.pkl"
        model_path = MANO / f"MANO_{side.upper()}.pkl"
        vertices, _, faces, _, _ = reconstruct_taco_mano(
            pose_path, shape_path, model_path, side=side)
        inputs += [pose_path, shape_path, model_path]
        rows = []
        for row in ROWS:
            rendered, box = raycast_camera_depth(
                vertices[row], faces, extrinsics[row], intrinsic)
            ycrcb = cv2.cvtColor(rgbs[row][box], cv2.COLOR_BGR2YCrCb)
            skin = ((ycrcb[..., 1] > 130) & (ycrcb[..., 1] < 180)
                    & (ycrcb[..., 2] > 75) & (ycrcb[..., 2] < 140))
            record, residual = residual_record(depths[row][box], rendered, skin)
            record["source_row_zero_based"] = row
            rows.append(record)
            hand_residuals.append(residual)
        hand_records[side] = dict(frames=rows)

    object_median = float(np.median(np.concatenate(object_residuals)) * 1000)
    hand_median = float(np.median(np.concatenate(hand_residuals)) * 1000)
    opposite = object_median < 0 < hand_median
    return dict(
        status="depth_registration_audited_no_scene_change",
        inputs=[artifact(path) for path in inputs],
        method=dict(
            rows=list(ROWS), depth_conversion="raw_uint16 / 4000.0",
            rendered_depth="Open3D ray casting of released metric meshes in the released RGB camera",
            residual="raw depth minus rendered annotation depth along the RGB optical axis",
            correspondence_filter="eroded rendered interior; abs residual < 50 mm",
            hand_filter="additional broad YCrCb skin-color mask",
            rows_selected="visually unobstructed initial/returned-object frames; disclosed Pour-only diagnostic",
        ),
        objects=object_records,
        hands=hand_records,
        aggregate=dict(object_raw_minus_rendered_median_mm=object_median,
                       hand_raw_minus_rendered_median_mm=hand_median,
                       opposite_signed_offsets=opposite),
        interpretation=(
            "Projected object boundaries are near both RGB and raw-depth edges, so a gross "
            "extrinsic direction or pixel-registration failure is not supported. Raw depth is "
            "closer than the object meshes but farther than MANO skin in these filtered samples. "
            "A common additive optical-depth offset cannot zero both medians. This does not "
            "rule out a rigid registration correction or establish annotation, surface or "
            "material effects as the cause; temporal and spatial calibration remain unresolved."
        ),
        scene_or_gt_modified=False,
        global_depth_correction_applied=False,
        code=artifact(Path(__file__)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
