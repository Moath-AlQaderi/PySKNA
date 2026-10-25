"""Result containers returned by high-level PySKNA workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class SKNAAnalysis:
    """All stages of an SKNA processing pipeline."""

    time: FloatArray
    raw: FloatArray
    detrended: FloatArray
    filtered: FloatArray
    rskna: FloatArray
    iskna: FloatArray
    askna_time: FloatArray
    askna: FloatArray
    fs: float
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ECGAnalysis:
    """Filtered ECG, R peaks, heart rate, and standard time-domain HRV."""

    time: FloatArray
    raw: FloatArray
    detrended: FloatArray
    filtered: FloatArray
    peak_indices: IntArray
    peak_times: FloatArray
    hr_time: FloatArray
    heart_rate: FloatArray
    binned_hr_time: FloatArray
    binned_heart_rate: FloatArray
    hrv: dict[str, float]
    fs: float


@dataclass(frozen=True)
class PhaseStatistics:
    """Summary statistics for one experimental phase."""

    start: float
    stop: float
    count: int
    mean: float
    standard_deviation: float
    standard_error: float
    fraction_above_threshold: float


@dataclass(frozen=True)
class CPTAnalysis:
    """SKNA and optional ECG results for a cold pressor test."""

    skna: SKNAAnalysis
    phases: dict[str, PhaseStatistics]
    cpt_start: float
    cpt_stop: float
    threshold: float
    overdrive_rate: float
    ecg: ECGAnalysis | None = None
    heart_rate_increase: float | None = None


@dataclass(frozen=True)
class Burst:
    """One threshold-crossing sympathetic burst."""

    start_time: float
    peak_time: float
    stop_time: float
    duration: float
    peak_amplitude: float
    area_above_threshold: float


@dataclass(frozen=True)
class BurstAnalysis:
    """Detected bursts plus conventional SNA summary metrics."""

    bursts: tuple[Burst, ...]
    threshold: float
    duration_minutes: float
    burst_frequency_per_minute: float
    mean_peak_amplitude: float
    mean_duration: float
    total_area_above_threshold: float

