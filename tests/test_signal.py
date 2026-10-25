import numpy as np
import pytest

from pyskna import (
    analyze_skna,
    artifact_mask,
    average_skna,
    burst_threshold,
    detect_bursts,
    filter_skna,
    integrate_skna,
    overdrive_rate,
    rectify_skna,
    rms_envelope,
)


def test_filter_skna_selects_high_frequency_band():
    fs = 4000.0
    time = np.arange(int(2 * fs)) / fs
    wanted = np.sin(2 * np.pi * 700 * time)
    unwanted = np.sin(2 * np.pi * 50 * time)
    result = filter_skna(wanted + unwanted, fs)
    assert np.corrcoef(result[500:-500], wanted[500:-500])[0, 1] > 0.99


def test_rectify_integrate_and_average():
    rectified = rectify_skna([-2, 1, -3, 4])
    np.testing.assert_array_equal(rectified, [2, 1, 3, 4])
    integrated = integrate_skna(np.ones(1000), 1000, time_constant=0.1)
    np.testing.assert_allclose(integrated, 1.0)
    centers, averages = average_skna(np.arange(20), 2, window_seconds=5)
    np.testing.assert_allclose(centers, [2.25, 7.25])
    np.testing.assert_allclose(averages, [4.5, 14.5])


def test_threshold_and_overdrive_formula():
    baseline = np.array([1.0, 2.0, 3.0])
    expected = baseline.mean() + 3 * baseline.std(ddof=1)
    assert burst_threshold(baseline) == pytest.approx(expected)
    assert overdrive_rate(2.0, 1.6) == pytest.approx(25.0)


def test_rms_artifacts_and_bursts():
    signal = np.zeros(1000)
    signal[100:200] = 2.0
    signal[500:700] = 3.0
    rms = rms_envelope(signal, 1000, window_seconds=0.1)
    assert rms.shape == signal.shape
    bursts = detect_bursts(signal, 1000, 1.0, min_duration=0.05)
    assert len(bursts.bursts) == 2
    assert bursts.bursts[1].peak_amplitude == 3.0
    contaminated = np.zeros(100)
    contaminated[50] = 100
    assert artifact_mask(contaminated, padding_samples=2).sum() == 5


def test_analyze_skna_shapes_and_threshold():
    fs = 4000.0
    time = np.arange(int(20 * fs)) / fs
    amplitude = 1 + 0.1 * np.sin(2 * np.pi * 0.2 * time)
    signal = amplitude * np.sin(2 * np.pi * 700 * time) + 0.001 * time**2
    result = analyze_skna(signal, fs, time=time, baseline=(0, 10))
    assert result.filtered.shape == signal.shape
    assert result.rskna.shape == signal.shape
    assert result.iskna.shape == signal.shape
    assert result.askna.size == 4
    assert result.threshold is not None


def test_filter_rejects_insufficient_sampling_rate():
    with pytest.raises(ValueError, match="cutoffs"):
        filter_skna(np.ones(100), 2000)

