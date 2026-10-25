"""ECG processing used for simultaneous SKNA/ECG recordings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks, iirnotch, sosfiltfilt, tf2sos

from ._validation import as_1d_finite, resolve_time, validate_fs
from .models import ECGAnalysis
from .signal import bandpass_filter, polynomial_detrend


def notch_filter(
    signal: ArrayLike, fs: float, *, frequency: float = 60.0, quality_factor: float = 30.0
) -> NDArray[np.float64]:
    """Apply a zero-phase IIR power-line notch filter."""
    values = as_1d_finite(signal)
    fs = validate_fs(fs)
    if not 0 < frequency < fs / 2:
        raise ValueError("frequency must lie between 0 and fs/2")
    if quality_factor <= 0:
        raise ValueError("quality_factor must be positive")
    b, a = iirnotch(frequency, quality_factor, fs)
    return sosfiltfilt(tf2sos(b, a), values)


def filter_ecg(
    signal: ArrayLike,
    fs: float,
    *,
    lowcut: float = 0.5,
    highcut: float = 35.0,
    order: int = 3,
    notch: float | None = 60.0,
    notch_quality: float = 30.0,
) -> NDArray[np.float64]:
    """Band-pass an ECG and optionally suppress power-line noise.

    The paper uses 0.05–35 Hz; 0.5 Hz is the safer default for digital
    filtering of short records.
    """
    result = bandpass_filter(signal, fs, lowcut, highcut, order=order)
    if notch is not None and notch < fs / 2:
        result = notch_filter(result, fs, frequency=notch, quality_factor=notch_quality)
    return result


def detect_r_peaks(
    ecg: ArrayLike,
    fs: float,
    *,
    min_distance: float = 0.5,
    height: float | None = None,
    prominence: float | None = None,
) -> NDArray[np.int_]:
    """Detect ECG R peaks with configurable amplitude and prominence criteria."""
    values = as_1d_finite(ecg, "ecg")
    fs = validate_fs(fs)
    distance = max(1, int(round(float(min_distance) * fs)))
    if prominence is None and height is None:
        # A scale-adaptive default works across volts, millivolts, and normalized data.
        prominence = max(float(np.std(values)) * 0.5, np.finfo(float).eps)
    peaks, _ = find_peaks(values, distance=distance, height=height, prominence=prominence)
    return peaks.astype(int)


def instantaneous_heart_rate(
    peak_times: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert R-peak timestamps to instantaneous heart rate in bpm."""
    peaks = as_1d_finite(peak_times, "peak_times")
    rr = np.diff(peaks)
    if np.any(rr <= 0):
        raise ValueError("peak_times must be strictly increasing")
    return peaks[1:], 60.0 / rr


def bin_heart_rate(
    hr_time: ArrayLike,
    heart_rate: ArrayLike,
    *,
    start: float,
    stop: float,
    window_seconds: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average instantaneous heart rate in fixed-width time bins."""
    times = as_1d_finite(hr_time, "hr_time")
    values = as_1d_finite(heart_rate, "heart_rate")
    if times.size != values.size:
        raise ValueError("hr_time and heart_rate must have the same length")
    if stop <= start or window_seconds <= 0:
        raise ValueError("require stop > start and a positive window_seconds")
    edges = np.arange(start, stop + window_seconds, window_seconds, dtype=float)
    if edges[-1] < stop:
        edges = np.append(edges, stop)
    indices = np.digitize(times, edges, right=False) - 1
    means = np.full(edges.size - 1, np.nan)
    for index in range(means.size):
        selected = values[indices == index]
        if selected.size:
            means[index] = np.mean(selected)
    return (edges[:-1] + edges[1:]) / 2.0, means


def hrv_time_domain(peak_times: ArrayLike) -> dict[str, float]:
    """Compute mean RR, SDNN, RMSSD, pNN50, and mean heart rate."""
    peaks = as_1d_finite(peak_times, "peak_times")
    rr = np.diff(peaks)
    if np.any(rr <= 0):
        raise ValueError("peak_times must be strictly increasing")
    rr_diff = np.diff(rr)
    return {
        "mean_rr_s": float(np.mean(rr)),
        "sdnn_s": float(np.std(rr, ddof=1)) if rr.size > 1 else float("nan"),
        "rmssd_s": float(np.sqrt(np.mean(rr_diff**2))) if rr_diff.size else float("nan"),
        "pnn50_percent": float(np.mean(np.abs(rr_diff) > 0.05) * 100)
        if rr_diff.size
        else float("nan"),
        "mean_hr_bpm": float(60.0 / np.mean(rr)),
    }


def analyze_ecg(
    signal: ArrayLike,
    fs: float,
    *,
    time: ArrayLike | None = None,
    detrend_degree: int = 3,
    lowcut: float = 0.5,
    highcut: float = 35.0,
    filter_order: int = 3,
    notch: float | None = 60.0,
    peak_height: float | None = None,
    peak_prominence: float | None = None,
    min_peak_distance: float = 0.5,
    hr_window: float = 5.0,
) -> ECGAnalysis:
    """Run ECG filtering, peak detection, HR, and 5-s averaging."""
    raw = as_1d_finite(signal, "ecg")
    fs = validate_fs(fs)
    timestamps = resolve_time(time, raw.size, fs)
    detrended = polynomial_detrend(raw, detrend_degree, time=timestamps, keep_mean=True)
    filtered = filter_ecg(
        detrended,
        fs,
        lowcut=lowcut,
        highcut=highcut,
        order=filter_order,
        notch=notch,
    )
    peaks = detect_r_peaks(
        filtered,
        fs,
        min_distance=min_peak_distance,
        height=peak_height,
        prominence=peak_prominence,
    )
    if peaks.size < 2:
        raise ValueError("fewer than two R peaks were detected; adjust peak parameters")
    peak_times = timestamps[peaks]
    hr_time, heart_rate = instantaneous_heart_rate(peak_times)
    binned_time, binned_hr = bin_heart_rate(
        hr_time,
        heart_rate,
        start=float(timestamps[0]),
        stop=float(timestamps[-1] + 1.0 / fs),
        window_seconds=hr_window,
    )
    return ECGAnalysis(
        time=timestamps,
        raw=raw,
        detrended=detrended,
        filtered=filtered,
        peak_indices=peaks,
        peak_times=peak_times,
        hr_time=hr_time,
        heart_rate=heart_rate,
        binned_hr_time=binned_time,
        binned_heart_rate=binned_hr,
        hrv=hrv_time_domain(peak_times),
        fs=fs,
    )

