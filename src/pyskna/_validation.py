"""Shared input validation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_1d_finite(values: ArrayLike, name: str = "signal") -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_fs(fs: float) -> float:
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive finite sampling frequency")
    return fs


def resolve_time(time: ArrayLike | None, size: int, fs: float) -> NDArray[np.float64]:
    if time is None:
        return np.arange(size, dtype=float) / fs
    result = as_1d_finite(time, "time")
    if result.size != size:
        raise ValueError("time and signal must have the same length")
    if np.any(np.diff(result) <= 0):
        raise ValueError("time must be strictly increasing")
    return result


def sampling_frequency(time: ArrayLike, rtol: float = 0.01) -> float:
    """Infer sampling frequency and reject materially irregular timestamps."""
    values = as_1d_finite(time, "time")
    differences = np.diff(values)
    if np.any(differences <= 0):
        raise ValueError("time must be strictly increasing")
    median_dt = float(np.median(differences))
    if not np.allclose(differences, median_dt, rtol=rtol, atol=0.0):
        raise ValueError(
            "time is not uniformly sampled; resample before applying digital filters"
        )
    return 1.0 / median_dt

