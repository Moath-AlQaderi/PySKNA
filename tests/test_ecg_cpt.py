import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pyskna import (
    analyze_cpt,
    analyze_ecg,
    plot_cpt_analysis,
    plot_processing_steps,
)


def synthetic_ecg(time, bpm=72):
    result = np.zeros_like(time)
    period = 60.0 / bpm
    for beat in np.arange(0.5, time[-1], period):
        result += np.exp(-0.5 * ((time - beat) / 0.012) ** 2)
    return result


def test_ecg_analysis_detects_expected_rate():
    fs = 2500.0
    time = np.arange(int(12 * fs)) / fs
    ecg = synthetic_ecg(time, 75)
    result = analyze_ecg(ecg, fs, time=time, notch=None)
    assert result.peak_indices.size >= 13
    assert result.hrv["mean_hr_bpm"] == pytest.approx(75, rel=0.02)
    assert np.nanmean(result.binned_heart_rate) == pytest.approx(75, rel=0.02)


def test_cpt_workflow_and_visualizations():
    fs = 2500.0
    time = np.arange(int(40 * fs)) / fs
    amplitude = np.where(time < 20, 1.0 + 0.05 * np.sin(2 * np.pi * 0.1 * time), 1.0)
    amplitude = np.where((time >= 20) & (time < 30), 2.0, amplitude)
    amplitude = np.where(time >= 30, 1.2, amplitude)
    skna = amplitude * np.sin(2 * np.pi * 700 * time)
    ecg = synthetic_ecg(time, 72)
    result = analyze_cpt(
        skna,
        fs,
        time=time,
        ecg=ecg,
        cpt_start=20,
        cpt_stop=30,
        ecg_options={"notch": None},
    )
    assert result.phases["during"].mean > result.phases["before"].mean
    assert result.overdrive_rate > 0
    assert result.ecg is not None
    fig1, _ = plot_cpt_analysis(result)
    fig2, _ = plot_processing_steps(result.skna)
    fig1.canvas.draw()
    fig2.canvas.draw()

