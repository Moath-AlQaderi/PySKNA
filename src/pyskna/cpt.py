"""Cold pressor test and simultaneous SKNA/ECG analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .ecg import analyze_ecg
from .models import CPTAnalysis, PhaseStatistics
from .signal import analyze_skna, burst_threshold, overdrive_rate


def _phase_statistics(
    time: np.ndarray,
    values: np.ndarray,
    start: float,
    stop: float,
    threshold: float,
) -> PhaseStatistics:
    selected = values[(time >= start) & (time < stop)]
    if selected.size == 0:
        raise ValueError(f"phase {start:g}–{stop:g} s has no complete aSKNA windows")
    sd = float(np.std(selected, ddof=1)) if selected.size > 1 else float("nan")
    sem = sd / np.sqrt(selected.size) if selected.size > 1 else float("nan")
    return PhaseStatistics(
        start=float(start),
        stop=float(stop),
        count=int(selected.size),
        mean=float(np.mean(selected)),
        standard_deviation=sd,
        standard_error=float(sem),
        fraction_above_threshold=float(np.mean(selected > threshold)),
    )


def analyze_cpt(
    skna: ArrayLike,
    fs: float,
    *,
    cpt_start: float,
    cpt_stop: float,
    time: ArrayLike | None = None,
    ecg: ArrayLike | None = None,
    analysis_start: float | None = None,
    analysis_stop: float | None = None,
    skna_options: dict[str, Any] | None = None,
    ecg_options: dict[str, Any] | None = None,
) -> CPTAnalysis:
    """Analyze SKNA before, during, and after a cold pressor test.

    The pre-CPT phase defines the individualized burst threshold. The function
    returns phase statistics, the paper's overdrive rate, and—when ECG is
    supplied—the simultaneous heart-rate response.
    """
    skna_options = dict(skna_options or {})
    ecg_options = dict(ecg_options or {})
    if not cpt_start < cpt_stop:
        raise ValueError("cpt_stop must be greater than cpt_start")
    result = analyze_skna(skna, fs, time=time, baseline=None, **skna_options)
    start = float(result.time[0] if analysis_start is None else analysis_start)
    stop = float(result.time[-1] + 1.0 / result.fs if analysis_stop is None else analysis_stop)
    if not start < cpt_start < cpt_stop < stop:
        raise ValueError("require analysis_start < cpt_start < cpt_stop < analysis_stop")
    baseline_mask = (result.askna_time >= start) & (result.askna_time < cpt_start)
    if baseline_mask.sum() < 2:
        raise ValueError("pre-CPT baseline must contain at least two aSKNA windows")
    threshold = burst_threshold(result.askna[baseline_mask])
    # Preserve immutability while attaching the CPT-derived threshold.
    result = result.__class__(**{**result.__dict__, "threshold": threshold})
    phases = {
        "before": _phase_statistics(
            result.askna_time, result.askna, start, cpt_start, threshold
        ),
        "during": _phase_statistics(
            result.askna_time, result.askna, cpt_start, cpt_stop, threshold
        ),
        "after": _phase_statistics(result.askna_time, result.askna, cpt_stop, stop, threshold),
    }
    ecg_result = None
    heart_rate_increase = None
    if ecg is not None:
        ecg_result = analyze_ecg(ecg, fs, time=time, **ecg_options)
        before_mask = (ecg_result.binned_hr_time >= start) & (
            ecg_result.binned_hr_time < cpt_start
        )
        during_mask = (ecg_result.binned_hr_time >= cpt_start) & (
            ecg_result.binned_hr_time < cpt_stop
        )
        before_hr = ecg_result.binned_heart_rate[before_mask]
        during_hr = ecg_result.binned_heart_rate[during_mask]
        before_mean = np.nanmean(before_hr)
        during_mean = np.nanmean(during_hr)
        if np.isfinite(before_mean) and before_mean != 0 and np.isfinite(during_mean):
            heart_rate_increase = float((during_mean - before_mean) / before_mean * 100)
    return CPTAnalysis(
        skna=result,
        phases=phases,
        cpt_start=float(cpt_start),
        cpt_stop=float(cpt_stop),
        threshold=threshold,
        overdrive_rate=overdrive_rate(phases["during"].mean, threshold),
        ecg=ecg_result,
        heart_rate_increase=heart_rate_increase,
    )


def analyze_simultaneous(
    skna: ArrayLike,
    ecg: ArrayLike,
    fs: float,
    *,
    time: ArrayLike | None = None,
    skna_options: dict[str, Any] | None = None,
    ecg_options: dict[str, Any] | None = None,
):
    """Analyze synchronous SKNA and ECG outside a CPT protocol."""
    skna_result = analyze_skna(skna, fs, time=time, **dict(skna_options or {}))
    ecg_result = analyze_ecg(ecg, fs, time=time, **dict(ecg_options or {}))
    return skna_result, ecg_result

