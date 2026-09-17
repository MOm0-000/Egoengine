"""Minimal re-implementation of the old SPIDER 720x480 side-by-side renderer.

This deliberately keeps only the two functions that define the old video style:
``setup_renderer`` and ``render_image``. The old ``viewers/__init__.py`` version
also imports rerun/viser/``Config``, which are unrelated to offline MP4 output.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np


def setup_renderer(model: mujoco.MjModel) -> mujoco.Renderer:
    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 480
    return mujoco.Renderer(model, height=480, width=720)


def render_image(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data_sim: mujoco.MjData,
    data_ref: mujoco.MjData,
) -> np.ndarray:
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    mujoco.mj_forward(model, data_sim)
    try:
        renderer.update_scene(data_sim, "front", options)
    except Exception:
        renderer.update_scene(data_sim, 0, options)
    sim_image = renderer.render()
    cv2.putText(
        sim_image,
        "sim",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (128, 128, 128),
        2,
    )

    mujoco.mj_forward(model, data_ref)
    try:
        renderer.update_scene(data_ref, "front")
    except Exception:
        renderer.update_scene(data_ref, 0)
    ref_image = renderer.render()
    cv2.putText(
        ref_image,
        "ref",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (128, 128, 128),
        2,
    )

    return np.concatenate([ref_image, sim_image], axis=1)
