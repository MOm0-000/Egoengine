"""CPU-only diagnostic videos generated from saved pipeline artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from .schemas import SCHEMA_VERSION
from .video import FFmpegVideoWriter, VIDEO_ENCODING


MESH_VIDEO = "05_mesh_proposals.mp4"
FOUNDATIONPOSE_VIDEO = "06_foundationpose.mp4"
OPTIMIZATION_VIDEO = "07_raw_vs_aligned_contact.mp4"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _frame_rows(root: Path) -> tuple[dict[int, tuple[Path, int]], float]:
    rows = _json(root / "frames/frame_index.json")["frames"]
    source = _json(root / "input/source.json")
    lookup = {
        int(row["source_frame_index"]): (root / row["rgb_path"], int(row["frame_index"]))
        for row in rows
    }
    return lookup, float(source["video"]["fps"])


def _writer(path: Path, fps: float, size: tuple[int, int], overwrite: bool) -> FFmpegVideoWriter:
    return FFmpegVideoWriter(path, fps, size, overwrite=overwrite)


def _record_output(root: Path, output: Path, sources: list[str]) -> None:
    manifest_path = root / "visualization" / "visualization_manifest.json"
    if manifest_path.exists():
        payload = _json(manifest_path)
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "diagnostic_only": True,
            "ground_truth_consumed": False,
            "invalid_gaps_preserved": True,
            "outputs": [],
            "sources": [],
        }
    relative_output = str(output.relative_to(root))
    payload["outputs"] = sorted(set(payload.get("outputs", [])) | {relative_output})
    payload["sources"] = sorted(set(payload.get("sources", [])) | set(sources))
    payload["video_encoding"] = VIDEO_ENCODING
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fit_frame(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width)) if max_side > 0 else 1.0
    target_width = max(2, int(round(width * scale)) // 2 * 2)
    target_height = max(2, int(round(height * scale)) // 2 * 2)
    if (target_width, target_height) == (width, height):
        return image, 1.0, 1.0
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, target_width / width, target_height / height


def _scaled_intrinsics(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    result = np.asarray(K, dtype=np.float64).copy()
    result[0] *= scale_x
    result[1] *= scale_y
    return result


def _tint_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not mask.any():
        return
    color_array = np.asarray(color, dtype=np.float32)
    image[mask] = np.clip(
        image[mask].astype(np.float32) * (1.0 - alpha) + color_array * alpha, 0, 255
    ).astype(np.uint8)


def _draw_contour(
    image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 2,
) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def _label(
    image: np.ndarray, text: str, origin: tuple[int, int], *,
    color: tuple[int, int, int] = (245, 245, 245), scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _mesh_mask(
    mesh: trimesh.Trimesh, K: np.ndarray, pose: np.ndarray,
    image_shape: tuple[int, int], scale_m: float,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale_m)
    camera = vertices @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    mask = np.zeros(image_shape, dtype=np.uint8)
    if np.count_nonzero(depth > 1e-5) < 3:
        return mask.astype(bool)
    homogeneous = camera @ K.T
    pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-8)
    height, width = image_shape
    for face in np.asarray(mesh.faces):
        if np.any(depth[face] <= 1e-5):
            continue
        polygon = np.rint(pixels[face]).astype(np.int32)
        if polygon[:, 0].max() < 0 or polygon[:, 0].min() >= width:
            continue
        if polygon[:, 1].max() < 0 or polygon[:, 1].min() >= height:
            continue
        cv2.fillConvexPoly(mask, polygon, 1)
    return mask.astype(bool)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError(f"file does not contain a triangle mesh: {path}")
    return loaded


def _turntable_panel(
    size: tuple[int, int], mesh: trimesh.Trimesh, angle: float, *, selected: bool,
) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), (27, 30, 34), dtype=np.uint8)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = vertices - mesh.bounds.mean(axis=0)
    vertices /= max(float(np.max(mesh.extents)), 1e-8)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation_y = np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    tilt = math.radians(-18.0)
    rotation_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(tilt), -math.sin(tilt)],
        [0.0, math.sin(tilt), math.cos(tilt)],
    ])
    rotated = vertices @ (rotation_x @ rotation_y).T
    render_scale = min(width, height) * 0.36
    pixels = np.column_stack([
        width * 0.5 + rotated[:, 0] * render_scale,
        height * 0.54 - rotated[:, 1] * render_scale,
    ])
    faces = np.asarray(mesh.faces)
    order = np.argsort(rotated[faces, 2].mean(axis=1))
    light = np.array([0.25, -0.35, 0.90])
    light /= np.linalg.norm(light)
    base = np.array((205, 142, 62) if selected else (95, 161, 198), dtype=np.float64)
    for face_index in order:
        face = faces[face_index]
        triangle = rotated[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        shade = 0.35 + 0.65 * abs(float(normal @ light) / norm)
        color = tuple(int(value) for value in np.clip(base * shade, 0, 255))
        cv2.fillConvexPoly(panel, np.rint(pixels[face]).astype(np.int32), color, cv2.LINE_AA)
    border = (55, 205, 255) if selected else (77, 83, 91)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), border, 2)
    return panel


def render_mesh_proposals(
    run_dir: str | Path, *, overwrite: bool = False, duration_s: float = 4.0,
    output_size: tuple[int, int] = (1280, 720),
) -> Path:
    """Render every qualified SAM 3D proposal in a deterministic turntable video."""
    root = Path(run_dir).resolve()
    ranking = _json(root / "mesh_proposals/mesh_ranking.json")
    selected_path = root / "object_tracking/selected_mesh.json"
    selected_id = _json(selected_path).get("proposal_id") if selected_path.exists() else None
    records = [record for record in ranking["proposals"] if record.get("qualified")]
    if not records:
        raise ValueError("mesh ranking contains no qualified proposal")
    meshes = [
        _load_mesh(root / "mesh_proposals" / record["visual_mesh"])
        for record in records
    ]
    _, fps = _frame_rows(root)
    output = root / "visualization" / MESH_VIDEO
    writer = _writer(output, fps, output_size, overwrite)
    count = max(2, int(round(fps * duration_s)))
    columns = min(3, len(records))
    rows = int(math.ceil(len(records) / columns))
    header_height = 58
    panel_width = output_size[0] // columns
    panel_height = (output_size[1] - header_height) // rows
    try:
        for frame_index in range(count):
            canvas = np.full((output_size[1], output_size[0], 3), (17, 19, 22), dtype=np.uint8)
            _label(canvas, "SAM 3D mesh proposals | automatic ranking and tracking selection", (18, 35), scale=0.72)
            angle = 2.0 * math.pi * frame_index / count
            for index, (record, mesh) in enumerate(zip(records, meshes)):
                row, column = divmod(index, columns)
                x0, y0 = column * panel_width, header_height + row * panel_height
                panel = _turntable_panel(
                    (panel_width, panel_height), mesh, angle,
                    selected=record["proposal_id"] == selected_id,
                )
                canvas[y0:y0 + panel_height, x0:x0 + panel_width] = panel
                status = "SELECTED" if record["proposal_id"] == selected_id else "candidate"
                _label(canvas, f"{record['proposal_id']} | {status}", (x0 + 14, y0 + 27), scale=0.50)
                _label(
                    canvas,
                    f"static rank {record['rank']}  score {float(record['static_score']):.3f}",
                    (x0 + 14, y0 + 50), color=(205, 210, 215), scale=0.44,
                )
            writer.write(canvas)
    finally:
        writer.release()
    _record_output(
        root, output,
        ["mesh_proposals/mesh_ranking.json", "mesh_proposals/*/visual.obj"],
    )
    return output


def render_foundationpose(
    run_dir: str | Path, *, overwrite: bool = False, max_side: int = 960,
) -> Path:
    """Overlay the selected mesh pose, SAM mask, invalid gaps, and registrations on RGB."""
    root = Path(run_dir).resolve()
    selected = _json(root / "object_tracking/selected_mesh.json")
    mesh = _load_mesh(root / selected["canonical_visual_mesh"])
    tracking = _npz(root / "object_tracking/foundationpose_raw.npz")
    segmentation = _npz(root / "segmentation/object_masks.npz")
    rows, fps = _frame_rows(root)
    K = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    first = cv2.imread(str(rows[int(tracking["frame_indices"][0])][0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError("cannot read first RGB frame")
    fitted, scale_x, scale_y = _fit_frame(first, max_side)
    size = (fitted.shape[1], fitted.shape[0])
    scaled_K = _scaled_intrinsics(K, scale_x, scale_y)
    output = root / "visualization" / FOUNDATIONPOSE_VIDEO
    writer = _writer(output, fps, size, overwrite)
    mask_lookup = {int(frame): index for index, frame in enumerate(segmentation["frame_indices"])}
    try:
        for index, raw_frame_index in enumerate(tracking["frame_indices"]):
            frame_index = int(raw_frame_index)
            image = cv2.imread(str(rows[frame_index][0]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read RGB frame {rows[frame_index][0]}")
            image, _, _ = _fit_frame(image, max_side)
            mask_at = mask_lookup[frame_index]
            sam_mask = cv2.resize(
                segmentation["masks"][mask_at].astype(np.uint8), size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            _tint_mask(image, sam_mask, (55, 180, 75), 0.24)
            _draw_contour(image, sam_mask, (70, 235, 95), 2)
            valid = bool(tracking["valid"][index])
            if valid:
                rendered = _mesh_mask(
                    mesh, scaled_K, tracking["T_camera_object"][index],
                    (size[1], size[0]), float(selected["scale_to_m"]),
                )
                _tint_mask(image, rendered, (220, 145, 45), 0.34)
                _draw_contour(image, rendered, (255, 195, 70), 2)
            cv2.rectangle(image, (0, 0), (size[0], 58), (18, 20, 23), -1)
            _label(
                image,
                f"FoundationPose | frame {frame_index:06d} | IoU {float(tracking['mask_iou'][index]):.3f} | "
                f"confidence {float(tracking['confidence'][index]):.3f}",
                (14, 24), scale=0.52,
            )
            segment_id = int(tracking.get("segment_id", np.full(len(tracking["valid"]), -1))[index])
            status = f"segment {segment_id}"
            if bool(tracking["registration_frame"][index]):
                status += " | REGISTER"
            if not valid:
                status += " | INVALID GAP (preserved)"
            _label(
                image, status, (14, 49),
                color=(60, 95, 255) if not valid else ((65, 190, 255) if "REGISTER" in status else (205, 210, 215)),
                scale=0.48,
            )
            writer.write(image)
    finally:
        writer.release()
    _record_output(
        root, output,
        [
            "object_tracking/foundationpose_raw.npz",
            "object_tracking/selected_mesh.json",
            "segmentation/object_masks.npz",
        ],
    )
    return output


def _project_points(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = points @ K.T
    return homogeneous[..., :2] / np.maximum(homogeneous[..., 2:3], 1e-8)


def _pose_centers(K: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return _project_points(K, poses[:, :3, 3])


def _draw_tail(
    image: np.ndarray, centers: np.ndarray, index: int, valid: np.ndarray,
    color: tuple[int, int, int], length: int = 18,
) -> None:
    start = max(0, index - length + 1)
    for current in range(start + 1, index + 1):
        if not (valid[current - 1] and valid[current]):
            continue
        points = np.rint(centers[current - 1:current + 1]).astype(np.int32)
        if np.isfinite(centers[current - 1:current + 1]).all():
            cv2.line(image, tuple(points[0]), tuple(points[1]), color, 2, cv2.LINE_AA)


def render_optimization(
    run_dir: str | Path, *, overwrite: bool = False, panel_width: int = 640,
) -> Path:
    """Render raw-vs-aligned object poses and aligned fingertip contact states."""
    root = Path(run_dir).resolve()
    selected = _json(root / "object_tracking/selected_mesh.json")
    metrics = _json(root / "optimization/optimization_metrics.json")
    mesh = _load_mesh(root / selected["canonical_visual_mesh"])
    raw = _npz(root / "object_tracking/foundationpose_raw.npz")
    aligned = _npz(root / "optimization/aligned_trajectory.npz")
    contact = _npz(root / "optimization/contact.npz")
    masks = _npz(root / "segmentation/object_masks.npz")
    rows, fps = _frame_rows(root)
    K = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    T_world_camera = np.load(root / "calibration/T_world_camera.npy").astype(np.float64)
    first = cv2.imread(str(rows[int(raw["frame_indices"][0])][0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError("cannot read first RGB frame")
    source_height, source_width = first.shape[:2]
    panel_height = max(2, int(round(source_height * panel_width / source_width)) // 2 * 2)
    scale_x, scale_y = panel_width / source_width, panel_height / source_height
    scaled_K = _scaled_intrinsics(K, scale_x, scale_y)
    header_height = 78
    output_size = (panel_width * 2, panel_height + header_height)
    output = root / "visualization" / OPTIMIZATION_VIDEO
    writer = _writer(output, fps, output_size, overwrite)
    T_sim_world = np.asarray(metrics["T_sim_world"], dtype=np.float64)
    calibration_indices = np.asarray([rows[int(frame)][1] for frame in aligned["frame_indices"]])
    T_sim_camera = np.einsum("ij,tjk->tik", T_sim_world, T_world_camera[calibration_indices])
    aligned_camera_pose = np.einsum(
        "tij,tjk->tik", np.linalg.inv(T_sim_camera), aligned["T_sim_object"][:, 0]
    )
    inverse_sim_camera = np.linalg.inv(T_sim_camera)
    fingertips_h = np.concatenate([
        aligned["fingertips_sim"], np.ones((*aligned["fingertips_sim"].shape[:-1], 1))
    ], axis=-1)
    fingertips_camera = np.einsum("tij,thfj->thfi", inverse_sim_camera, fingertips_h)[..., :3]
    fingertip_pixels = _project_points(scaled_K, fingertips_camera)
    raw_centers = _pose_centers(scaled_K, raw["T_camera_object"])
    aligned_centers = _pose_centers(scaled_K, aligned_camera_pose)
    mask_lookup = {int(frame): index for index, frame in enumerate(masks["frame_indices"])}
    raw_valid = raw["valid"].astype(bool)
    aligned_valid = aligned["valid_object"][:, 0].astype(bool)
    comparison = metrics["raw_vs_aligned"]
    initial_scale = float(selected["scale_to_m"])
    optimized_scale = float(aligned["object_scale_to_m"][0])
    hand_count = aligned["valid_hand"].shape[1]
    default_hand_order = ["left", "right"][:hand_count]
    artifact_hand_order = metrics.get("hands", {}).get("artifact_hand_order", default_hand_order)
    if len(artifact_hand_order) != hand_count:
        raise ValueError("optimization hand order does not match aligned hand dimension")
    hand_colors = {"left": (255, 180, 45), "right": (70, 220, 255)}
    try:
        for index, raw_frame_index in enumerate(aligned["frame_indices"]):
            frame_index = int(raw_frame_index)
            source = cv2.imread(str(rows[frame_index][0]), cv2.IMREAD_COLOR)
            if source is None:
                raise RuntimeError(f"cannot read RGB frame {rows[frame_index][0]}")
            source = cv2.resize(source, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
            raw_panel, aligned_panel = source.copy(), source.copy()
            mask = cv2.resize(
                masks["masks"][mask_lookup[frame_index]].astype(np.uint8),
                (panel_width, panel_height), interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for panel in (raw_panel, aligned_panel):
                _tint_mask(panel, mask, (50, 175, 70), 0.20)
                _draw_contour(panel, mask, (65, 230, 90), 1)
            if raw_valid[index]:
                raw_render = _mesh_mask(
                    mesh, scaled_K, raw["T_camera_object"][index],
                    (panel_height, panel_width), initial_scale,
                )
                _tint_mask(raw_panel, raw_render, (170, 70, 205), 0.36)
                _draw_contour(raw_panel, raw_render, (220, 100, 245), 2)
            if aligned_valid[index]:
                aligned_render = _mesh_mask(
                    mesh, scaled_K, aligned_camera_pose[index],
                    (panel_height, panel_width), optimized_scale,
                )
                _tint_mask(aligned_panel, aligned_render, (220, 145, 45), 0.36)
                _draw_contour(aligned_panel, aligned_render, (255, 195, 70), 2)
            _draw_tail(raw_panel, raw_centers, index, raw_valid, (220, 100, 245))
            _draw_tail(aligned_panel, aligned_centers, index, aligned_valid, (255, 195, 70))
            for hand_index, side in enumerate(artifact_hand_order):
                hand_color = hand_colors[side]
                if not bool(aligned["valid_hand"][index, hand_index]):
                    continue
                for fingertip_index, point in enumerate(fingertip_pixels[index, hand_index]):
                    if not np.isfinite(point).all() or fingertips_camera[index, hand_index, fingertip_index, 2] <= 0:
                        continue
                    center = tuple(np.rint(point).astype(int))
                    active = bool(contact["contact"][index, hand_index, fingertip_index] >= 0.5)
                    cv2.circle(aligned_panel, center, 7 if active else 4, (45, 65, 255) if active else hand_color, -1, cv2.LINE_AA)
                    if active:
                        cv2.circle(aligned_panel, center, 9, (245, 245, 245), 1, cv2.LINE_AA)
            canvas = np.full((output_size[1], output_size[0], 3), (17, 19, 22), dtype=np.uint8)
            canvas[header_height:, :panel_width] = raw_panel
            canvas[header_height:, panel_width:] = aligned_panel
            _label(canvas, f"RAW FoundationPose | frame {frame_index:06d}", (14, 26), scale=0.58)
            _label(canvas, "ALIGNED + CONTACT", (panel_width + 14, 26), scale=0.58)
            _label(
                canvas,
                f"jitter {float(comparison['object_translation_acceleration_jitter_raw_m_s2']):.2f} -> "
                f"{float(comparison['object_translation_acceleration_jitter_aligned_m_s2']):.2f} m/s2    "
                f"reprojection {float(comparison['object_mask_centroid_reprojection_raw_px']):.2f} -> "
                f"{float(comparison['object_mask_centroid_reprojection_aligned_px']):.2f} px",
                (14, 53), color=(205, 210, 215), scale=0.48,
            )
            active_count = int(np.count_nonzero(contact["contact"][index] >= 0.5))
            _label(
                canvas,
                f"global scale {initial_scale:.4f} -> {optimized_scale:.4f} m | "
                f"active fingertips {active_count}/{hand_count * 5}",
                (14, 72), color=(205, 210, 215), scale=0.43,
            )
            if not aligned_valid[index]:
                _label(canvas, "INVALID OBJECT GAP (preserved)", (panel_width + 14, 55), color=(60, 95, 255), scale=0.48)
            writer.write(canvas)
    finally:
        writer.release()
    _record_output(
        root, output,
        [
            "object_tracking/foundationpose_raw.npz",
            "object_tracking/selected_mesh.json",
            "optimization/aligned_trajectory.npz",
            "optimization/contact.npz",
            "segmentation/object_masks.npz",
        ],
    )
    return output


def render_run_visualizations(
    run_dir: str | Path, *, overwrite: bool = False, max_side: int = 960,
    mesh_duration_s: float = 4.0,
) -> Path:
    """Generate all three diagnostic videos and a machine-readable index."""
    root = Path(run_dir).resolve()
    outputs = [
        render_mesh_proposals(root, overwrite=overwrite, duration_s=mesh_duration_s),
        render_foundationpose(root, overwrite=overwrite, max_side=max_side),
        render_optimization(root, overwrite=overwrite),
    ]
    manifest_path = root / "visualization" / "visualization_manifest.json"
    if not all(path.exists() and path.stat().st_size > 0 for path in outputs):
        raise RuntimeError("one or more visualization videos are missing or empty")
    return manifest_path
