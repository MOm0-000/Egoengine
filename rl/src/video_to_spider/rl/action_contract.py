"""Fail-closed residual-action profiles for Replay -> RL experiments."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import yaml

from .h2s2r import XHandResidualPolicySpec


def load_residual_action_profile(path: Path) -> tuple[XHandResidualPolicySpec, dict]:
    raw = path.read_bytes()
    profile = yaml.safe_load(raw)
    if profile.get("schema_version") != 1:
        raise ValueError("unsupported residual-action profile schema")
    if profile.get("status") not in {"frozen_historical", "resolved_local_unpublished"}:
        raise ValueError("residual-action profile is not run-ready")
    if profile.get("paper_faithful") is not False:
        raise ValueError("the local residual-action scale must not be marked paper-faithful")

    output = profile.get("policy_output", {})
    if output.get("lower") != -1.0 or output.get("upper") != 1.0:
        raise ValueError("PPO policy output must remain normalized to [-1, 1]")
    mapping = profile.get("mapping", {})
    scale = float(mapping.get("residual_scale", math.nan))
    clip = float(mapping.get("residual_clip_rad", math.nan))
    kind = mapping.get("kind")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("residual_scale must be finite and positive")
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError("residual_clip_rad must be finite and positive")
    if kind not in {"direct_clip_legacy", "linear_scaled_candidate"}:
        raise ValueError("unsupported residual-action mapping kind")
    if kind == "linear_scaled_candidate" and not math.isclose(scale, clip):
        raise ValueError("the scaled candidate must map normalized +/-1 exactly to the safety limit")

    spec = XHandResidualPolicySpec(residual_scale=scale, residual_clip=clip)
    spec.validate()
    report = {
        "profile_id": profile.get("action_profile_id"),
        "status": profile["status"],
        "paper_faithful": False,
        "provenance": profile.get("provenance"),
        "profile_path": str(path.resolve()),
        "profile_sha256": hashlib.sha256(raw).hexdigest(),
        "policy_output_range": [-1.0, 1.0],
        "mapping_kind": kind,
        "residual_scale": scale,
        "residual_clip_rad": clip,
        "equation": f"delta_a = clip({scale} * u, -{clip}, +{clip}) rad",
        "author_recovered_scale": mapping.get("author_recovered") is True,
        "controlled_experiment": profile.get("controlled_experiment"),
    }
    if report["author_recovered_scale"]:
        raise ValueError("EgoEngine does not publish the residual action scale")
    return spec, report
