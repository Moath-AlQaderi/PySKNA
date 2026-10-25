"""Matplotlib visualizations for each PySKNA analysis stage."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .models import CPTAnalysis, ECGAnalysis, SKNAAnalysis


def _phase_background(axes: Sequence[Axes], start: float, stop: float) -> None:
    for axis in axes:
        axis.axvspan(start, stop, color="#8b5cf6", alpha=0.12, label="CPT")
        axis.axvline(start, color="#7c3aed", linestyle="--", linewidth=1)
        axis.axvline(stop, color="#7c3aed", linestyle="--", linewidth=1)


def plot_processing_steps(
    result: SKNAAnalysis, *, figsize: tuple[float, float] = (12, 12)
) -> tuple[Figure, np.ndarray]:
    """Plot raw, detrended, filtered, rSKNA, iSKNA, and aSKNA signals."""
    fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=False, constrained_layout=True)
    stages = (
        (result.time, result.raw, "Raw signal"),
        (result.time, result.detrended, "Polynomial-detrended signal"),
        (result.time, result.filtered, "Filtered SKNA"),
        (result.time, result.rskna, "rSKNA (full-wave rectified)"),
        (result.time, result.iskna, "iSKNA (integrated envelope)"),
        (result.askna_time, result.askna, "aSKNA (window average)"),
    )
    for axis, (time, values, title) in zip(axes, stages, strict=True):
        axis.plot(time, values, color="#111827", linewidth=0.8)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.2)
    if result.threshold is not None:
        axes[-1].axhline(
            result.threshold, color="#dc2626", linestyle="--", label="Burst threshold"
        )
        axes[-1].legend(frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("PySKNA processing stages", fontweight="bold")
    return fig, axes


def plot_askna_distribution(
    result: SKNAAnalysis, *, bins: int | str = "auto", figsize: tuple[float, float] = (8, 4.5)
) -> tuple[Figure, Axes]:
    """Plot the relative-frequency distribution of aSKNA."""
    fig, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    weights = np.ones(result.askna.size) * 100.0 / result.askna.size
    axis.hist(result.askna, bins=bins, weights=weights, color="#a855f7", edgecolor="white")
    axis.axvline(np.mean(result.askna), color="#2563eb", linestyle="--", label="Mean")
    if result.threshold is not None:
        axis.axvline(result.threshold, color="#dc2626", linestyle="--", label="Threshold")
    axis.set(xlabel="aSKNA amplitude", ylabel="Relative frequency (%)", title="aSKNA distribution")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    return fig, axis


def plot_ecg_analysis(
    result: ECGAnalysis, *, figsize: tuple[float, float] = (12, 6)
) -> tuple[Figure, np.ndarray]:
    """Plot filtered ECG with R peaks and 5-s heart rate."""
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=True)
    axes[0].plot(result.time, result.filtered, color="#111827", linewidth=0.8)
    axes[0].scatter(
        result.peak_times,
        result.filtered[result.peak_indices],
        s=18,
        color="#dc2626",
        label="R peaks",
        zorder=3,
    )
    axes[0].set_ylabel("ECG amplitude")
    axes[0].legend(frameon=False)
    axes[1].plot(result.binned_hr_time, result.binned_heart_rate, color="#ef4444", marker="o")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].set(xlabel="Time (s)", ylabel="Heart rate (bpm)")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("ECG and heart-rate analysis", fontweight="bold")
    return fig, axes


def plot_simultaneous(
    skna: SKNAAnalysis,
    ecg: ECGAnalysis,
    *,
    figsize: tuple[float, float] = (12, 9),
) -> tuple[Figure, np.ndarray]:
    """Plot aligned filtered SKNA, aSKNA, ECG, and heart rate."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)
    axes[0].plot(skna.time, skna.filtered, color="#1d4ed8", linewidth=0.6)
    axes[0].set_ylabel("SKNA")
    axes[1].plot(skna.askna_time, skna.askna, color="#a855f7", marker="o", markersize=3)
    if skna.threshold is not None:
        axes[1].axhline(skna.threshold, color="#dc2626", linestyle="--")
    axes[1].set_ylabel("aSKNA")
    axes[2].plot(ecg.time, ecg.filtered, color="#111827", linewidth=0.7)
    axes[2].scatter(
        ecg.peak_times, ecg.filtered[ecg.peak_indices], color="#ef4444", s=10, zorder=3
    )
    axes[2].set_ylabel("ECG")
    axes[3].plot(ecg.binned_hr_time, ecg.binned_heart_rate, color="#ef4444", marker="o")
    axes[3].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[3].set(xlabel="Time (s)", ylabel="HR (bpm)")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Simultaneous SKNA and ECG", fontweight="bold")
    return fig, axes


def plot_cpt_analysis(
    result: CPTAnalysis, *, figsize: tuple[float, float] = (13, 11)
) -> tuple[Figure, np.ndarray]:
    """Visualize before, during, and after CPT, including optional ECG/HR."""
    rows = 4 if result.ecg is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=figsize, sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(result.skna.time, result.skna.filtered, color="#1d4ed8", linewidth=0.5)
    axes[0].set_ylabel("SKNA")
    axes[1].plot(
        result.skna.askna_time, result.skna.askna, color="#a855f7", marker="o", markersize=3
    )
    axes[1].axhline(
        result.threshold, color="#dc2626", linestyle="--", label="Baseline mean + 3 SD"
    )
    axes[1].set_ylabel("aSKNA")
    axes[1].legend(frameon=False)
    if result.ecg is not None:
        axes[2].plot(result.ecg.time, result.ecg.filtered, color="#111827", linewidth=0.6)
        axes[2].scatter(
            result.ecg.peak_times,
            result.ecg.filtered[result.ecg.peak_indices],
            color="#ef4444",
            s=9,
            zorder=3,
        )
        axes[2].set_ylabel("ECG")
        axes[3].plot(
            result.ecg.binned_hr_time,
            result.ecg.binned_heart_rate,
            color="#ef4444",
            marker="o",
        )
        axes[3].ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[3].set_ylabel("HR (bpm)")
    _phase_background(axes, result.cpt_start, result.cpt_stop)
    axes[-1].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.2)
    subtitle = f"aSKNA overdrive: {result.overdrive_rate:.2f}%"
    if result.heart_rate_increase is not None:
        subtitle += f" | HR increase: {result.heart_rate_increase:.2f}%"
    fig.suptitle(f"Cold pressor test analysis | {subtitle}", fontweight="bold")
    return fig, axes


def plot_cpt_phase_summary(
    result: CPTAnalysis, *, figsize: tuple[float, float] = (7, 4.5)
) -> tuple[Figure, Axes]:
    """Plot mean ± SEM aSKNA before, during, and after CPT."""
    labels = ["Before", "During", "After"]
    stats = [result.phases[key] for key in ("before", "during", "after")]
    means = [item.mean for item in stats]
    errors = [item.standard_error for item in stats]
    fig, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    axis.bar(labels, means, yerr=errors, capsize=5, color=["#c4b5fd", "#8b5cf6", "#ddd6fe"])
    axis.axhline(result.threshold, color="#dc2626", linestyle="--", label="Burst threshold")
    axis.set(ylabel="Mean aSKNA", title="aSKNA response to cold pressor test")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    return fig, axis

