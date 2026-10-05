"""Thin access layer for TACO's pinned official PyTorch3D projection code.

This module deliberately contains no camera or projection mathematics.  It
loads the exact official ``Pyt3DWrapper`` used by
``dataset_utils/project_pose_to_egocentric_view.py`` and exposes its camera,
renderer and rasterizer for local evidence analysis.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


@contextmanager
def _official_import_path(dataset_utils: Path):
    value = str(dataset_utils)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == value:
            sys.path.pop(0)


class TacoOfficialProjector:
    """Use the pinned official camera and renderer without reimplementing them."""

    def __init__(
        self,
        *,
        dataset_utils: str | Path,
        image_size: tuple[int, int],
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        device: str,
    ) -> None:
        self.dataset_utils = Path(dataset_utils).resolve(strict=True)
        with _official_import_path(self.dataset_utils):
            from pyt3d_wrapper import Pyt3DWrapper  # type: ignore
            from pytorch3d.renderer import PointLights

        self._torch = __import__("torch")
        self._wrapper = Pyt3DWrapper(
            image_size=image_size,
            use_fixed_cameras=True,
            intrin=np.asarray(intrinsic),
            extrin=np.asarray(extrinsic),
            device=device,
            lights=PointLights(device=device, location=[[0.0, 0.0, 2.0]]),
        )

    @property
    def official_wrapper(self):
        return self._wrapper

    def set_camera(self, intrinsic: np.ndarray, extrinsic: np.ndarray) -> None:
        self._wrapper.setup_intrin_extrin(
            intrin=np.asarray(intrinsic), extrin=np.asarray(extrinsic),
        )

    def render_rgb(self, meshes: Iterable[object]) -> np.ndarray:
        return np.asarray(self._wrapper.render_meshes(list(meshes))[0])

    def render_mask(self, meshes: Iterable[object]) -> np.ndarray:
        values = list(meshes)
        prepared = self._wrapper.prepare_render(
            values, self._wrapper.colors[: len(values)], None,
        )
        _, mask = self._wrapper.renderer.render(
            prepared, self._wrapper.cameras[0], ret_mask=True,
        )
        return np.asarray(mask, dtype=bool)

    def render_depth(self, meshes: Iterable[object]) -> np.ndarray:
        values = list(meshes)
        prepared = self._wrapper.prepare_render(
            values, self._wrapper.colors[: len(values)], None,
        )
        fragments = self._wrapper.renderer.renderer.rasterizer(
            prepared, cameras=self._wrapper.cameras[0],
        )
        return fragments.zbuf[0, ..., 0].detach().cpu().numpy().astype(np.float32)

    def project_points(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = self._torch.as_tensor(
            np.asarray(points_world), dtype=self._torch.float32,
            device=self._wrapper.device,
        ).unsqueeze(0)
        screen = self._wrapper.cameras[0].transform_points_screen(points)[0]
        values = screen.detach().cpu().numpy()
        return values[:, :2], values[:, 2]


def load_official_hand_sequence(
    *,
    dataset_utils: str | Path,
    pose_path: str | Path,
    shape_path: str | Path,
    side: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Call TACO's unmodified official MANO loader and expose skin weights."""
    dataset_utils = Path(dataset_utils).resolve(strict=True)
    with _official_import_path(dataset_utils):
        import pickle
        from hand_pose_loader import mano_params_to_hand_info  # type: ignore
        from manopth.manopth.manolayer import ManoLayer  # type: ignore

        with Path(shape_path).open("rb") as stream:
            beta = pickle.load(stream)["hand_shape"].reshape(10).detach().cpu().numpy()
        vertices, joints, faces = mano_params_to_hand_info(
            str(pose_path), mano_beta=beta, side=side, max_cnt=None,
            return_pose=False, return_faces=True, device=device,
        )
        layer = ManoLayer(
            mano_root=str(dataset_utils / "manopth/mano/models"),
            use_pca=False, ncomps=45, side=side, center_idx=0,
        )
        weights = layer.th_weights.detach().cpu().numpy()
    return (
        np.asarray(vertices), np.asarray(joints), np.asarray(faces),
        np.asarray(weights),
    )


def official_overlay(
    *, dataset_utils: str | Path, rgb: np.ndarray, render: np.ndarray,
) -> np.ndarray:
    """Call the exact overlay helper used by the official egocentric script."""
    dataset_utils = Path(dataset_utils).resolve(strict=True)
    with _official_import_path(dataset_utils):
        from video_utils import overlay_two_imgs  # type: ignore
    return np.asarray(overlay_two_imgs(rgb, render))
