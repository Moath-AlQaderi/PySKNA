"""Compatibility workflow translated from ``SKNA MATLAB Code.txt``."""

from __future__ import annotations

from numpy.typing import ArrayLike

from ._validation import sampling_frequency
from .ecg import analyze_ecg
from .models import ECGAnalysis, SKNAAnalysis
from .signal import (
    average_skna,
    burst_threshold,
    filter_skna,
    integrate_skna,
    polynomial_detrend,
    rectify_skna,
)


def analyze_matlab_workflow(
    time: ArrayLike,
    skna: ArrayLike,
    ecg: ArrayLike,
    *,
    baseline: tuple[float, float] | None = None,
    skna_unit_scale: float = 1000.0,
    skna_highcut: float = 999.0,
    ecg_peak_height: float | None = 0.08,
) -> tuple[SKNAAnalysis, ECGAnalysis]:
    """Translate the complete, executable portion of the supplied MATLAB script.

    This compatibility path intentionally preserves its choices: cubic
    detrending; 500–999 Hz SKNA; a 10-ms leaky envelope; 5-s aSKNA calculated
    from rSKNA; 5–20 Hz ECG; and R-peak spacing of at least 0.5 s. Set
    ``skna_unit_scale=1`` when SKNA is already in microvolts.
    """
    import numpy as np

    timestamps = np.asarray(time, dtype=float)
    skna_values = np.asarray(skna, dtype=float) * float(skna_unit_scale)
    fs = sampling_frequency(timestamps)
    detrended = polynomial_detrend(skna_values, 3, time=timestamps, keep_mean=True)
    filtered = filter_skna(detrended, fs, lowcut=500.0, highcut=skna_highcut, order=3)
    rskna = rectify_skna(filtered)
    iskna = integrate_skna(rskna, fs, time_constant=0.01, method="leaky")
    # The source MATLAB averages rSKNA, even though the paper averages iSKNA.
    askna_time, askna = average_skna(rskna, fs, window_seconds=5.0, time=timestamps)
    threshold = None
    if baseline is not None:
        mask = (askna_time >= baseline[0]) & (askna_time <= baseline[1])
        if mask.sum() < 2:
            raise ValueError("baseline must include at least two complete 5-s windows")
        threshold = burst_threshold(askna[mask])
    skna_result = SKNAAnalysis(
        time=timestamps,
        raw=skna_values,
        detrended=detrended,
        filtered=filtered,
        rskna=rskna,
        iskna=iskna,
        askna_time=askna_time,
        askna=askna,
        fs=fs,
        threshold=threshold,
        metadata={
            "compatibility": "SKNA MATLAB Code.txt",
            "bandpass_hz": (500.0, float(skna_highcut)),
            "integration_time_seconds": 0.01,
            "askna_source": "rSKNA",
        },
    )
    ecg_result = analyze_ecg(
        ecg,
        fs,
        time=timestamps,
        detrend_degree=3,
        lowcut=5.0,
        highcut=20.0,
        filter_order=3,
        notch=None,
        peak_height=ecg_peak_height,
        min_peak_distance=0.5,
        hr_window=5.0,
    )
    return skna_result, ecg_result

