"""Loading helpers for LVM and ordinary delimited recordings."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


_NUMBER = re.compile(r"^[\s,;\t]*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def load_delimited(
    path: str | Path,
    *,
    delimiter: str | None = None,
    skip_header: int = 0,
    time_column: int = 0,
    signal_column: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load time and signal columns from CSV, TSV, TXT, or numeric LVM data."""
    data = np.genfromtxt(
        Path(path), delimiter=delimiter, skip_header=skip_header, dtype=float, invalid_raise=False
    )
    if data.ndim != 2 or data.shape[1] <= max(time_column, signal_column):
        raise ValueError("file does not contain the requested numeric columns")
    selected = data[:, [time_column, signal_column]]
    selected = selected[np.all(np.isfinite(selected), axis=1)]
    if selected.shape[0] < 2:
        raise ValueError("file contains fewer than two valid numeric rows")
    return selected[:, 0], selected[:, 1]


def load_lvm(
    path: str | Path,
    *,
    time_column: int = 0,
    signal_column: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load a LabVIEW Measurement (LVM) file with or without a text header.

    The first numeric row is detected automatically. Comma, tab, semicolon, and
    whitespace separators are supported.
    """
    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    first_numeric = next((i for i, line in enumerate(lines) if _NUMBER.match(line)), None)
    if first_numeric is None:
        raise ValueError("no numeric data rows found in LVM file")
    sample = lines[first_numeric]
    delimiter = "\t" if "\t" in sample else "," if "," in sample else ";" if ";" in sample else None
    return load_delimited(
        source,
        delimiter=delimiter,
        skip_header=first_numeric,
        time_column=time_column,
        signal_column=signal_column,
    )

