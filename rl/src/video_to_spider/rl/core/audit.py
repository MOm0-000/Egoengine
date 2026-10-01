"""Passive, detached audit records for the direct core runtime."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

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


def json_ready(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(name): json_ready(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(f"value of type {type(value)} is not JSON serializable")


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(json_ready(row), sort_keys=True) + "\n")


def append_rollout_npz(path: str | Path, *, epoch: int, batch: Any) -> None:
    """Append one epoch as distinct NPY members in a single resumable NPZ."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"epoch_{int(epoch):04d}/"
    mode = "a" if target.exists() else "w"
    with zipfile.ZipFile(target, mode=mode, compression=zipfile.ZIP_DEFLATED) as archive:
        if any(name.startswith(prefix) for name in archive.namelist()):
            raise ValueError(f"batch archive already contains epoch {epoch}")
        for name, value in vars(batch).items():
            if torch.is_tensor(value):
                arrays = ((name, value.detach().cpu().numpy()),)
            elif isinstance(value, tuple) and all(torch.is_tensor(item) for item in value):
                arrays = tuple(
                    (f"{name}_{index}", item.detach().cpu().numpy())
                    for index, item in enumerate(value)
                )
            elif isinstance(value, (int, float, bool)):
                arrays = ((name, np.asarray(value)),)
            else:
                continue
            for member, array in arrays:
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
                archive.writestr(prefix + member + ".npy", buffer.getvalue())
