"""Generate a disclosed CoACD candidate without overwriting source geometry."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import coacd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def weld_exact_vertices(mesh):
    """STL repeats triangle corners: share identical vertices without moving them."""
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    result = trimesh.Trimesh(vertices=vertices, faces=inverse[mesh.faces], process=False)
    if not np.array_equal(mesh.triangles, result.triangles):
        raise ValueError("exact welding changed source triangles")
    return result


def clip_existing_parts(directory, output):
    """Intersect convex pieces with the native convex envelope, never shrink it.

    This removes preprocessing overreach outside the old single-hull collider.
    It cannot repair holes in the source; all-surface coverage is audited later.
    """
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    path = directory / "provenance.json"
    record = json.loads(path.read_text())
    source = Path(record["source"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != record["source_sha256"]:
        raise ValueError("native source changed")
    native = trimesh.load_mesh(source, process=True)
    native.apply_scale(record["unit_scale"])
    envelope = native.convex_hull
    parts = []
    max_excess = 0.0
    for entry in record["artifacts"]:
        part_path = Path(entry["path"])
        if hashlib.sha256(part_path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("input decomposition changed")
        original = trimesh.load_mesh(part_path, process=True)
        clipped = trimesh.boolean.intersection([original, envelope], engine="manifold")
        if not clipped.is_volume:
            raise ValueError("clipping did not preserve a positive closed piece")
        # Convex vertices inside every source half-space certify the entire
        # convex result is inside, up to the disclosed numerical tolerance.
        for container in (original.convex_hull, envelope):
            n = container.face_normals
            offsets = np.einsum("ij,ij->i", n, container.triangles_center)
            excess = float((clipped.vertices @ n.T - offsets).max())
            max_excess = max(max_excess, excess)
            if excess > 1e-7:
                raise ValueError("clipped vertices exceed input hull by over 0.1 micrometre")
        parts.append(clipped)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for i, part in enumerate(parts):
        part_path = output / f"{i}.obj"
        part.export(part_path)
        artifacts.append(dict(path=str(part_path.resolve()), sha256=hashlib.sha256(part_path.read_bytes()).hexdigest(),
                              metric_extents=part.extents.tolist()))
    record.update(artifacts=artifacts, parts=len(parts),
        postprocess="Manifold convex intersection with unchanged native convex envelope; local numerical correction",
        postprocess_input=dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        containment_halfspace_max_excess_m=max_excess, containment_numeric_tolerance_m=1e-7,
        postprocess_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), parts=len(parts), max_excess_m=max_excess)), flush=True)


def subtract_empty_box(directory, output, bounds):
    """Remove only a CAD-certified empty cavity from convex approximations.

    Six plane splits keep every exported piece convex. This is local geometry
    engineering, not an EgoEngine recipe or an exemption for real interference.
    """
    import manifold3d

    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    bounds = np.asarray(bounds, dtype=float).reshape(-1, 2, 3)
    if not len(bounds) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("finite ordered cavity box bounds required")
    path = directory / "provenance.json"
    record = json.loads(path.read_text())
    source = Path(record["source"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != record["source_sha256"]:
        raise ValueError("native source changed")
    native = trimesh.load_mesh(source, process=True)
    native.apply_scale(record["unit_scale"] * 1000)
    if not native.is_volume:
        raise ValueError("empty-cavity certification requires a closed native solid")
    for cut in bounds:
        box = trimesh.creation.box(extents=(cut[1] - cut[0]) * 1000,
            transform=trimesh.transformations.translation_matrix(cut.mean(0) * 1000))
        occupied = trimesh.boolean.intersection([native, box], engine="manifold")
        if len(occupied.faces):
            raise ValueError(f"cavity box removes native material: {occupied.volume} mm3")
    parts, removed, zero_volume_fragments = [], 0.0, 0
    for entry in record["artifacts"]:
        part_path = Path(entry["path"])
        if hashlib.sha256(part_path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("input decomposition changed")
        original = trimesh.load_mesh(part_path, process=True)
        original.apply_scale(1000)
        pieces = [manifold3d.Manifold(manifold3d.Mesh(
            np.asarray(original.vertices, dtype=np.float32), np.asarray(original.faces, dtype=np.uint32)))]
        for cut in bounds:
            outside = []
            for remaining in pieces:
                for axis in range(3):
                    normal = np.eye(3)[axis]
                    remaining, low = remaining.split_by_plane(normal, cut[0, axis] * 1000)
                    high, remaining = remaining.split_by_plane(normal, cut[1, axis] * 1000)
                    outside.extend(p for p in (low, high) if not p.is_empty())
                removed += max(0.0, remaining.volume())
            pieces = outside
        for part in pieces:
            raw = part.to_mesh()
            mesh = trimesh.Trimesh(vertices=np.asarray(raw.vert_properties)[:, :3].astype(float),
                                   faces=np.asarray(raw.tri_verts), process=True)
            hull = mesh.convex_hull
            if mesh.volume == 0 and hull.volume == 0:
                zero_volume_fragments += 1
                continue
            if not mesh.is_volume or abs(hull.volume - mesh.volume) > max(1e-5, abs(mesh.volume) * 1e-6):
                raise ValueError(f"plane split invalid: closed={mesh.is_watertight}, volume={mesh.volume} mm3, hull={hull.volume} mm3")
            mesh.apply_scale(.001)
            parts.append(mesh)
    if not parts:
        raise ValueError("cavity subtraction unexpectedly removed every piece")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for i, part in enumerate(parts):
        destination = output / f"{i}.obj"
        part.export(destination)
        artifacts.append(dict(path=str(destination.resolve()), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                              metric_extents=part.extents.tolist()))
    record.update(artifacts=artifacts, parts=len(parts),
        postprocess="convex plane splits subtract only native-solid-certified empty box; millimetre arithmetic",
        postprocess_input=dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        empty_box_bounds_m=bounds.tolist(), native_intersection_faces=0,
        removed_piece_volume_sum_mm3=removed,
        discarded_exact_zero_volume_fragments=zero_volume_fragments,
        postprocess_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), parts=len(parts), removed_piece_volume_sum_mm3=removed)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/taco_v1/dev4/object_models/object_models_released/146_cm.obj")
    parser.add_argument("--output", type=Path, default=ROOT / "models/taco_xhand/assets/objects/146/convex_m")
    parser.add_argument("--unit-scale", type=float, default=0.01)
    parser.add_argument("--max-convex-hull", type=int, default=32,
                        help="-1 removes forced merging to a fixed part count")
    parser.add_argument("--preprocess-mode", choices=("auto", "off", "on"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="normalized CoACD concavity, NOT metres or a task-success threshold")
    parser.add_argument("--real-metric", action="store_true",
                        help="use CoACD's metric mode: threshold is in scaled mesh units (metres here)")
    parser.add_argument("--clip-existing-parts", type=Path,
                        help="intersect an existing decomposition with its source hull, into a NEW output")
    parser.add_argument("--subtract-empty-box", type=float, nargs=6, action="append",
                        help="with --clip-existing-parts: subtract a native-certified empty box, min xyz then max xyz, metres")
    args = parser.parse_args()
    if args.clip_existing_parts:
        if args.subtract_empty_box is not None:
            subtract_empty_box(args.clip_existing_parts, args.output, args.subtract_empty_box)
        else:
            clip_existing_parts(args.clip_existing_parts, args.output)
        return
    if args.subtract_empty_box is not None:
        parser.error("--subtract-empty-box requires --clip-existing-parts")
    source, output = args.source.resolve(strict=True), args.output
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not np.isfinite(args.unit_scale) or args.unit_scale <= 0:
        raise ValueError("positive finite source unit scale required")
    if args.max_convex_hull != -1 and args.max_convex_hull < 1:
        raise ValueError("part limit must be positive or -1")
    if not np.isfinite(args.threshold) or (args.threshold <= 0 if args.real_metric else not .01 <= args.threshold <= 1):
        raise ValueError("threshold must be positive in metric mode, or between .01 and 1 in normalized mode")
    mesh = trimesh.load_mesh(source, process=False)
    if source.suffix.lower() == ".stl":
        mesh = weld_exact_vertices(mesh)
    mesh.apply_scale(args.unit_scale)
    if args.preprocess_mode == "off" and not mesh.is_volume:
        raise ValueError("preprocessing off requires a closed positive-volume source; no hole filling applied")
    parameters = dict(threshold=args.threshold, max_convex_hull=args.max_convex_hull, seed=0,
                      preprocess_mode=args.preprocess_mode,
                      mcts_iterations=100, mcts_max_depth=3, max_ch_vertex=64)
    if args.real_metric:
        parameters["real_metric"] = True
    print(json.dumps(dict(source=str(source), output=str(output), parameters=parameters)), flush=True)
    coacd.set_log_level("warn")
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), **parameters)
    if not parts:
        raise ValueError("CoACD produced no collision parts")
    validated = []
    for index, (vertices, faces) in enumerate(parts):
        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if not np.isfinite(part.vertices).all() or part.volume <= 0:
            raise ValueError(f"invalid convex part {index}")
        validated.append(part)
    # A failed decomposition must not leave an apparently usable partial asset.
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for index, part in enumerate(validated):
        path = output / f"{index}.obj"
        part.export(path)
        artifacts.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              metric_extents=part.extents.tolist()))
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  unit_scale=args.unit_scale, parameters=parameters, parts=len(parts),
                  metric_extents=mesh.extents.tolist(), coacd_version=version("coacd"),
                  artifacts=artifacts,
                  parameter_provenance="local settings; disclosed threshold/part-cap/preprocess/unit choices; not published EgoEngine parameters",
                  source_is_watertight=bool(mesh.is_watertight),
                  source_is_positive_volume=bool(mesh.is_volume),
                  exact_stl_vertex_welding=source.suffix.lower() == ".stl",
                  requested_threshold_certified=False,
                  status="collision_approximation_requires_contact_audit")
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
