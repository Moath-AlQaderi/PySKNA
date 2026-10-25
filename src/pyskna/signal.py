"""Core SKNA preprocessing and feature extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, lfilter, sosfiltfilt

from ._validation import as_1d_finite, resolve_time, validate_fs
from .models import Burst, BurstAnalysis, SKNAAnalysis


def polynomial_detrend(
    signal: ArrayLike, degree: int = 3, *, time: ArrayLike | None = None, keep_mean: bool = True
) -> NDArray[np.float64]:
    """Remove a polynomial trend, matching the supplied MATLAB workflow.

    Parameters
    ----------
    signal:
        One-dimensional signal.
    degree:
        Polynomial degree. The supplied MATLAB script uses 3.
    time:
        Optional time coordinates. Sample indices are used otherwise.
    keep_mean:
        Add the original mean back after detrending, as in the MATLAB script.
    """
    values = as_1d_finite(signal)
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a non-negative integer")
    if degree >= values.size:
        raise ValueError("degree must be smaller than the number of samples")
    x = np.arange(values.size, dtype=float) if time is None else as_1d_finite(time, "time")
    if x.size != values.size:
        raise ValueError("time and signal must have the same length")
    # Scaling x improves conditioning for recordings with large absolute timestamps.
    scaled = (x - x.mean()) / (np.ptp(x) / 2.0 if np.ptp(x) else 1.0)
    trend = np.polynomial.polynomial.polyval(
        scaled, np.polynomial.polynomial.polyfit(scaled, values, degree)
    )
    result = values - trend
    if keep_mean:
        result = result + values.mean()
    return result


def bandpass_filter(
    signal: ArrayLike,
    fs: float,
    lowcut: float,
    highcut: float,
    *,
    order: int = 3,
) -> NDArray[np.float64]:
    """Apply a zero-phase Butterworth band-pass filter."""
    values = as_1d_finite(signal)
    fs = validate_fs(fs)
    lowcut, highcut = float(lowcut), float(highcut)
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError("cutoffs must satisfy 0 < lowcut < highcut < fs/2")
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, values)
    except ValueError as exc:
        raise ValueError("signal is too short for zero-phase filtering") from exc


def filter_skna(
    signal: ArrayLike,
    fs: float,
    *,
    lowcut: float = 500.0,
    highcut: float = 1000.0,
    order: int = 3,
) -> NDArray[np.float64]:
    """Extract the standard 500–1000 Hz SKNA band used in the paper."""
    return bandpass_filter(signal, fs, lowcut, highcut, order=order)


def rectify_skna(signal: ArrayLike) -> NDArray[np.float64]:
    """Compute full-wave rectified SKNA (rSKNA)."""
    return np.abs(as_1d_finite(signal))


def integrate_skna(
    rskna: ArrayLike,
    fs: float,
    *,
    time_constant: float = 0.1,
    method: str = "leaky",
) -> NDArray[np.float64]:
    """Compute iSKNA using a causal leaky envelope or moving integral.

    ``method="leaky"`` implements the standard leaky integrator used by SNA
    workflows and retains microvolt units. ``method="moving_integral"`` returns
    a trailing 100-ms (by default) Riemann integral in microvolt-seconds.
    """
    values = as_1d_finite(rskna, "rskna")
    fs = validate_fs(fs)
    time_constant = float(time_constant)
    if not np.isfinite(time_constant) or time_constant <= 0:
        raise ValueError("time_constant must be positive")
    if method == "leaky":
        dt = 1.0 / fs
        alpha = dt / (time_constant + dt)
        # Initialize at the first value instead of introducing a startup ramp.
        return lfilter([alpha], [1.0, -(1.0 - alpha)], values, zi=[(1 - alpha) * values[0]])[0]
    if method == "moving_integral":
        samples = max(1, int(round(time_constant * fs)))
        cumulative = np.concatenate(([0.0], np.cumsum(values)))
        starts = np.maximum(0, np.arange(values.size) + 1 - samples)
        sums = cumulative[np.arange(1, values.size + 1)] - cumulative[starts]
        return sums / fs
    raise ValueError("method must be 'leaky' or 'moving_integral'")


def average_skna(
    iskna: ArrayLike,
    fs: float,
    *,
    window_seconds: float = 5.0,
    time: ArrayLike | None = None,
    include_partial: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average iSKNA in non-overlapping windows to obtain aSKNA."""
    values = as_1d_finite(iskna, "iskna")
    fs = validate_fs(fs)
    timestamps = resolve_time(time, values.size, fs)
    window_seconds = float(window_seconds)
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    window = max(1, int(round(window_seconds * fs)))
    stop = values.size if include_partial else values.size - values.size % window
    if stop == 0:
        raise ValueError("signal is shorter than one aSKNA window")
    means: list[float] = []
    centers: list[float] = []
    for start in range(0, stop, window):
        end = min(start + window, values.size)
        means.append(float(np.mean(values[start:end])))
        centers.append(float((timestamps[start] + timestamps[end - 1]) / 2.0))
    return np.asarray(centers), np.asarray(means)


def burst_threshold(values: ArrayLike, *, sigma: float = 3.0, ddof: int = 1) -> float:
    """Return the individualized burst threshold: mean + ``sigma`` × SD."""
    array = as_1d_finite(values, "baseline aSKNA")
    if array.size <= ddof:
        raise ValueError("baseline must contain more samples than ddof")
    return float(np.mean(array) + float(sigma) * np.std(array, ddof=ddof))


def overdrive_rate(during_mean: float, threshold: float) -> float:
    """Compute the paper's aSKNA CPT overdrive rate in percent."""
    during_mean, threshold = float(during_mean), float(threshold)
    if not np.isfinite(during_mean) or not np.isfinite(threshold):
        raise ValueError("during_mean and threshold must be finite")
    if threshold == 0:
        raise ValueError("threshold must be non-zero")
    return (during_mean - threshold) / threshold * 100.0


def rms_envelope(signal: ArrayLike, fs: float, *, window_seconds: float = 0.2) -> NDArray[np.float64]:
    """Return a moving RMS neurogram, a common SNA-toolbox representation."""
    values = as_1d_finite(signal)
    fs = validate_fs(fs)
    window = max(1, int(round(float(window_seconds) * fs)))
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(values**2, kernel, mode="same"))


def artifact_mask(
    signal: ArrayLike,
    *,
    z_threshold: float = 8.0,
    padding_samples: int = 0,
) -> NDArray[np.bool_]:
    """Flag extreme-amplitude samples using a robust median/MAD z-score."""
    values = as_1d_finite(signal)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        # In near-constant recordings, any departure from the median is visible
        # even though the conventional MAD scale collapses to zero.
        mask = values != median
    else:
        robust_z = 0.6744897501960817 * (values - median) / mad
        mask = np.abs(robust_z) > float(z_threshold)
    padding_samples = int(padding_samples)
    if padding_samples < 0:
        raise ValueError("padding_samples must be non-negative")
    if padding_samples and mask.any():
        mask = np.convolve(mask.astype(int), np.ones(2 * padding_samples + 1), mode="same") > 0
    return mask


def detect_bursts(
    signal: ArrayLike,
    fs: float,
    threshold: float,
    *,
    time: ArrayLike | None = None,
    min_duration: float = 0.0,
) -> BurstAnalysis:
    """Detect threshold-crossing bursts and compute conventional SNA metrics."""
    values = as_1d_finite(signal)
    fs = validate_fs(fs)
    timestamps = resolve_time(time, values.size, fs)
    threshold = float(threshold)
    above = values > threshold
    changes = np.diff(np.pad(above.astype(int), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    bursts: list[Burst] = []
    for start, stop in zip(starts, stops, strict=True):
        duration = (stop - start) / fs
        if duration < min_duration:
            continue
        segment = values[start:stop]
        peak_offset = int(np.argmax(segment))
        area = float(np.trapezoid(np.maximum(segment - threshold, 0.0), dx=1.0 / fs))
        bursts.append(
            Burst(
                start_time=float(timestamps[start]),
                peak_time=float(timestamps[start + peak_offset]),
                stop_time=float(timestamps[min(stop - 1, timestamps.size - 1)]),
                duration=float(duration),
                peak_amplitude=float(segment[peak_offset]),
                area_above_threshold=area,
            )
        )
    duration_minutes = (timestamps[-1] - timestamps[0] + 1.0 / fs) / 60.0
    peak_values = [burst.peak_amplitude for burst in bursts]
    durations = [burst.duration for burst in bursts]
    return BurstAnalysis(
        bursts=tuple(bursts),
        threshold=threshold,
        duration_minutes=float(duration_minutes),
        burst_frequency_per_minute=len(bursts) / duration_minutes,
        mean_peak_amplitude=float(np.mean(peak_values)) if bursts else float("nan"),
        mean_duration=float(np.mean(durations)) if bursts else float("nan"),
        total_area_above_threshold=float(sum(b.area_above_threshold for b in bursts)),
    )


def analyze_skna(
    signal: ArrayLike,
    fs: float,
    *,
    time: ArrayLike | None = None,
    unit_scale: float = 1.0,
    detrend_degree: int = 3,
    lowcut: float = 500.0,
    highcut: float = 1000.0,
    filter_order: int = 3,
    integration_time: float = 0.1,
    integration_method: str = "leaky",
    averaging_window: float = 5.0,
    baseline: tuple[float, float] | None = None,
) -> SKNAAnalysis:
    """Run the complete SKNA → rSKNA → iSKNA → aSKNA workflow.

    ``unit_scale=1000`` converts millivolts to microvolts, matching the supplied
    MATLAB script. Inputs already in microvolts should keep the default 1.
    """
    raw = as_1d_finite(signal) * float(unit_scale)
    fs = validate_fs(fs)
    timestamps = resolve_time(time, raw.size, fs)
    detrended = polynomial_detrend(raw, detrend_degree, time=timestamps, keep_mean=True)
    filtered = filter_skna(
        detrended, fs, lowcut=lowcut, highcut=highcut, order=filter_order
    )
    rskna = rectify_skna(filtered)
    iskna = integrate_skna(
        rskna, fs, time_constant=integration_time, method=integration_method
    )
    askna_time, askna = average_skna(
        iskna, fs, window_seconds=averaging_window, time=timestamps
    )
    threshold = None
    if baseline is not None:
        baseline_mask = (askna_time >= baseline[0]) & (askna_time < baseline[1])
        if baseline_mask.sum() < 2:
            raise ValueError("baseline must include at least two complete aSKNA windows")
        threshold = burst_threshold(askna[baseline_mask])
    return SKNAAnalysis(
        time=timestamps,
        raw=raw,
        detrended=detrended,
        filtered=filtered,
        rskna=rskna,
        iskna=iskna,
        askna_time=askna_time,
        askna=askna,
        fs=fs,
        threshold=threshold,
        metadata={
            "unit_scale": float(unit_scale),
            "detrend_degree": detrend_degree,
            "bandpass_hz": (float(lowcut), float(highcut)),
            "filter_order": filter_order,
            "integration_time_seconds": float(integration_time),
            "integration_method": integration_method,
            "averaging_window_seconds": float(averaging_window),
        },
    )

