"""TACO MANO surface reconstruction and contact-geometry helpers.

TACO V1 stores MANO pose/shape parameters in Python pickles.  The released
MANO model pickles refer to the historical :mod:`chumpy` package, which is no
longer maintained for the project's supported Python version.  This module
loads only the numeric model arrays needed by modern ``smplx`` and never
executes a chumpy graph.  The resulting vertices are exactly the same MANO
surface used by the TACO visualizer when ``flat_hand_mean=True``.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np


MANO21_SOURCE = np.asarray(
    (
        0, 13, 14, 15, 16,
        1, 2, 3, 17,
        4, 5, 6, 18,
        10, 11, 12, 19,
        7, 8, 9, 20,
    ),
    dtype=np.int64,
)
"""MANO raw joints/tips reordered to TACO's published MANO21 convention."""

MANO_TIP_VERTICES = {
    "right": np.asarray((745, 317, 444, 556, 673), dtype=np.int64),
    "left": np.asarray((745, 317, 445, 556, 673), dtype=np.int64),
}

SURFACE_REGION_ORDER = ("palm", "thumb", "index", "middle", "ring", "pinky")

# The 16 MANO skinning joints are in the raw order produced by the model.  A
# vertex's largest skinning weight gives a stable, anatomy-aware surface group.
_RAW_JOINT_TO_REGION = np.asarray(
    (0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1),
    dtype=np.int8,
)


class _ChumpyPlaceholder:
    """Minimal pickle target for obsolete chumpy graph nodes."""

    def __new__(cls, *args: object, **kwargs: object) -> "_ChumpyPlaceholder":
        return object.__new__(cls)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)


class _ManoUnpickler(pickle.Unpickler):
    """Load numeric MANO arrays without importing the obsolete chumpy runtime."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("chumpy."):
            return _ChumpyPlaceholder
        if module == "__builtin__" and name == "set":
            return set
        return super().find_class(module, name)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 identity of one small, declared source artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value: object, *, name: str) -> np.ndarray:
    """Convert a released CPU tensor or ndarray to finite float32 data."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"TACO {name} contains non-finite values")
    return array


def load_taco_mano_sequence(
    pose_path: str | Path, shape_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load ordered TACO MANO pose, translation, shape and source frame keys."""
    with Path(shape_path).open("rb") as handle:
        shape_data = pickle.load(handle)
    if not isinstance(shape_data, dict) or "hand_shape" not in shape_data:
        raise ValueError("TACO MANO shape pickle lacks hand_shape")
    betas = _as_numpy(shape_data["hand_shape"], name="hand_shape").reshape(-1)
    if betas.shape != (10,):
        raise ValueError("TACO MANO shape must contain exactly 10 coefficients")

    with Path(pose_path).open("rb") as handle:
        pose_data = pickle.load(handle)
    if not isinstance(pose_data, dict) or not pose_data:
        raise ValueError("TACO MANO pose pickle is empty or malformed")
    keys = tuple(sorted(str(key) for key in pose_data))
    poses, translations = [], []
    for key in keys:
        entry = pose_data[key]
        if not isinstance(entry, dict) or {"hand_pose", "hand_trans"} - entry.keys():
            raise ValueError(f"TACO MANO pose entry {key!r} is malformed")
        pose = _as_numpy(entry["hand_pose"], name="hand_pose").reshape(-1)
        translation = _as_numpy(entry["hand_trans"], name="hand_trans").reshape(-1)
        if pose.shape != (48,) or translation.shape != (3,):
            raise ValueError(f"TACO MANO pose entry {key!r} has an unexpected shape")
        poses.append(pose)
        translations.append(translation)
    return (
        np.stack(poses, axis=0), np.stack(translations, axis=0), betas, keys,
    )


def _load_model_data(model_path: str | Path) -> dict[str, object]:
    """Extract the numeric MANO model dictionary needed by ``smplx.MANO``."""
    with Path(model_path).open("rb") as handle:
        data = _ManoUnpickler(handle, encoding="latin1").load()
    if not isinstance(data, dict):
        raise ValueError("MANO model pickle is malformed")
    shapedirs = data.get("shapedirs")
    if isinstance(shapedirs, _ChumpyPlaceholder):
        source = getattr(getattr(shapedirs, "a", None), "x", None)
        indices = getattr(shapedirs, "idxs", None)
        if source is None or indices is None:
            raise ValueError("MANO shapedirs chumpy selector is malformed")
        source_array = np.asarray(source)
        index_array = np.asarray(indices, dtype=np.int64)
        if source_array.shape != (778, 3, 20) or index_array.shape != (778 * 3 * 10,):
            raise ValueError("MANO shapedirs has an unexpected source layout")
        shapedirs = source_array.reshape(-1)[index_array].reshape(778, 3, 10)
    shapedirs_array = np.asarray(shapedirs, dtype=np.float32)
    if shapedirs_array.shape != (778, 3, 10):
        raise ValueError("MANO shapedirs must have shape (778, 3, 10)")
    data = dict(data)
    data["shapedirs"] = shapedirs_array
    required = {
        "hands_components", "f", "J_regressor", "kintree_table", "weights",
        "posedirs", "hands_mean", "v_template", "shapedirs",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"MANO model misses required fields: {missing}")
    return data


def reconstruct_taco_mano(
    pose_path: str | Path, shape_path: str | Path, model_path: str | Path, *, side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Recover world-frame TACO MANO vertices and published MANO21 joints.

    Returns vertices in metres ``(T,778,3)``, MANO21 joints in metres
    ``(T,21,3)``, static faces, a per-vertex anatomical region id, and the
    original ordered TACO frame keys.
    """
    if side not in MANO_TIP_VERTICES:
        raise ValueError(f"unsupported MANO side {side!r}")
    try:
        import torch
        import smplx
        from smplx.utils import Struct
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "surface TACO GT needs the smplx package; install the declared "
            "MANO reconstruction dependency before running this stage",
        ) from error

    poses, translations, betas, keys = load_taco_mano_sequence(pose_path, shape_path)
    model_data = _load_model_data(model_path)
    model = smplx.MANO(
        "unused-model-path",
        data_struct=Struct(**model_data),
        is_rhand=side == "right",
        use_pca=False,
        # TACO's own hand_pose_loader leaves this at Manopth's default.
        flat_hand_mean=True,
        create_transl=False,
        batch_size=len(poses),
    )
    with torch.no_grad():
        result = model(
            global_orient=torch.from_numpy(poses[:, :3]),
            hand_pose=torch.from_numpy(poses[:, 3:]),
            betas=torch.from_numpy(np.repeat(betas[None], len(poses), axis=0)),
            return_verts=True,
        )
    raw_vertices = np.asarray(result.vertices.detach().cpu().numpy(), dtype=np.float32)
    raw_joints = np.asarray(result.joints.detach().cpu().numpy(), dtype=np.float32)
    if raw_vertices.shape != (len(poses), 778, 3) or raw_joints.shape != (len(poses), 16, 3):
        raise ValueError("MANO reconstruction produced an unexpected vertex/joint layout")
    # The TACO writer stores ``hand_trans`` as the world wrist coordinate after
    # its Manopth center_idx=0 normalization.  Preserve that convention here.
    root = raw_joints[:, :1]
    vertices = raw_vertices - root + translations[:, None]
    raw_all_joints = np.concatenate(
        (raw_joints, raw_vertices[:, MANO_TIP_VERTICES[side]]), axis=1,
    )
    joints21 = (raw_all_joints - root + translations[:, None])[:, MANO21_SOURCE]
    weights = np.asarray(model_data["weights"], dtype=np.float32)
    if weights.shape != (778, 16):
        raise ValueError("MANO skinning weights have an unexpected layout")
    vertex_regions = _RAW_JOINT_TO_REGION[np.argmax(weights, axis=1)]
    faces = np.asarray(model_data["f"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("MANO mesh faces are malformed")
    return vertices, joints21, faces, vertex_regions, keys


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Assign each triangle's area equally to its three surface vertices."""
    verts = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if verts.shape != (778, 3) or tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("MANO rest surface arrays are malformed")
    face_vertices = verts[tri]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0]),
        axis=1,
    )
    result = np.zeros(len(verts), dtype=np.float64)
    for column in range(3):
        np.add.at(result, tri[:, column], face_areas / 3.0)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("MANO vertex area computation failed")
    return result
