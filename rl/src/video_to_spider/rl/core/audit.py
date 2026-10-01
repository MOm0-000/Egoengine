"""Passive, detached audit records for the direct core runtime."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def detached_snapshot(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in values.items():
        if torch.is_tensor(value):
            result[name] = value.detach().cpu().clone()
        elif isinstance(value, np.ndarray):
            result[name] = value.copy()
        elif isinstance(value, (str, int, float, bool, type(None))):
            result[name] = value
        else:
            raise TypeError(f"audit field {name} retains unsupported mutable value {type(value)}")
    return result


def distribution(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    array = (
        values.detach().cpu().double().numpy()
        if torch.is_tensor(values)
        else np.asarray(values, dtype=np.float64)
    ).reshape(-1)
    return {
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def state_dict_max_abs_error(
    observed: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]
) -> tuple[float, dict[str, float]]:
    if set(observed) != set(expected):
        raise ValueError("state dictionaries have different keys")
    rows = {
        name: float((observed[name].detach().cpu() - expected[name].detach().cpu()).abs().max())
        for name in observed
    }
    return max(rows.values(), default=0.0), rows
