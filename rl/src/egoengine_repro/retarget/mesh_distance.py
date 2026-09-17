"""Closed-solid audit distances; positive inside, negative outside, in mesh units.

Open3D's odd multi-ray vote avoids Trimesh's random ambiguous-ray fallback.
Neither backend certifies arbitrary self-intersecting meshes. Critical probes
are independently checked with double-precision solid angles by the caller.
This is an offline diagnostic, not a replacement for MuJoCo collision forces.
"""

import numpy as np


def _validate(mesh, points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("finite (N,3) query points required")
    if (not len(mesh.faces) or not np.isfinite(mesh.vertices).all()
            or not mesh.is_watertight or not mesh.is_winding_consistent):
        raise ValueError("closed consistently oriented finite triangle mesh required")
    return points


def closed_mesh_signed_distance(mesh, points):
    """Return positive-inside distances; no sign assertion for surface points."""
    points = _validate(mesh, points)
    if not len(points):
        return np.empty(0)
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32))
    # Open3D uses the opposite convention: negative inside. Eleven rays is a
    # local numerical safeguard, not a paper/task success threshold.
    result = -scene.compute_signed_distance(
        o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32),
        nthreads=4, nsamples=11).numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite native surface distance")
    return result


def mesh_surface_distance(mesh, points):
    """Unsigned surface distance, including open meshes and overlapping pieces.

    Trimesh closest_point lost nanometre-scale plane distances on small clipped
    triangles in metre units. Open3D agrees with independent convex projection.
    No inside/outside claim is made by this function.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("finite (N,3) query points required")
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("finite nonempty triangle mesh required")
    if not len(points):
        return np.empty(0)
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32))
    result = scene.compute_distance(o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32),
                                    nthreads=4).numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite surface distance")
    return result


def solid_angle_winding(mesh, points):
    """Independent float64 check: abs(w)≈1 inside, ≈0 outside, away from surface.

    Sum signed triangle solid angles. Linear in all faces per probe; use for
    selected regression/extreme probes rather than millions of surface samples.
    """
    points = _validate(mesh, points)
    values = []
    for point in points:
        a, b, c = (mesh.triangles - point).transpose(1, 0, 2)
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc
                       + np.einsum("ij,ij->i", b, c) * la
                       + np.einsum("ij,ij->i", c, a) * lb)
        values.append(np.arctan2(numerator, denominator).sum() / (2 * np.pi))
    return np.asarray(values)
