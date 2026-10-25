# PySKNA

PySKNA is an open-source Python library for reproducible analysis of skin sympathetic nerve activity (SKNA), capacitive SKNA (cSKNA), and simultaneous SKNA/ECG recordings. It provides a paper-guided processing pipeline, a direct compatibility translation of the supplied MATLAB workflow, cold pressor test (CPT) analysis, sympathetic-burst metrics, heart-rate/HRV extraction, file loading, and publication-ready visualizations.

> **Research software notice:** PySKNA is not a medical device and is not intended for diagnosis or clinical decision-making. Validate acquisition settings and outputs for your hardware, population, and protocol.

## What the library does

The main SKNA workflow is:

```text
raw recording → detrend → 500–1000 Hz band-pass → rSKNA
             → 100 ms integrated envelope (iSKNA)
             → 5 s average (aSKNA)
             → baseline mean + 3 SD burst threshold
```

These defaults follow Hsu et al. (2025), *Flexible Electronic Sensor for Noncontact Simultaneous Measurement of Capacitive Skin Sympathetic Nerve Activity (cSKNA) and cECG*, IEEE Transactions on Instrumentation and Measurement, DOI [10.1109/TIM.2025.3544747](https://doi.org/10.1109/TIM.2025.3544747). The paper sampled at 10 kHz, separated cSKNA with a third-order 500–1000 Hz Butterworth filter, rectified it, integrated it over 100 ms, and averaged it over 5 s. It defined the burst threshold as `mean + 3 × SD` of baseline aSKNA and the CPT overdrive rate as `(during-CPT mean − threshold) / threshold × 100%`.

The supplied MATLAB code is also represented explicitly by `analyze_matlab_workflow()`. That function preserves the script’s 500–999 Hz SKNA filter, 10 ms leaky envelope, 5 s averaging of rSKNA, 5–20 Hz ECG filter, 0.5 s R-peak spacing, and optional 0.08 peak-height threshold.

## Installation

From a cloned repository:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

Python 3.10 or newer is required. The core dependencies are NumPy, SciPy, and Matplotlib.

## Data requirements

- SKNA analysis at 500–1000 Hz requires a sampling frequency **strictly greater than 2000 Hz**. A substantially higher rate, such as the paper’s 10 kHz, provides transition-band headroom.
- Supply SKNA in microvolts by default. Use `unit_scale=1000` for millivolt input.
- Supply uniformly sampled, finite one-dimensional arrays.
- When both SKNA and ECG are provided, they must share timestamps and sampling frequency.
- Motion and EMG can contaminate the SKNA band. Inspect every processing stage and use protocol-appropriate artifact handling.

## Quick start: SKNA only

```python
import matplotlib.pyplot as plt
import numpy as np
import pyskna

time, skna_mv = pyskna.load_lvm("SKNA.lvm")
fs = pyskna.sampling_frequency(time)

result = pyskna.analyze_skna(
    skna_mv,
    fs,
    time=time,
    unit_scale=1000,          # mV → µV
    baseline=(0, 120),        # seconds, stop-exclusive
)

print(result.threshold)
pyskna.plot_processing_steps(result)
pyskna.plot_askna_distribution(result)
plt.show()
```

Each intermediate array is available as `result.raw`, `result.detrended`, `result.filtered`, `result.rskna`, `result.iskna`, and `result.askna`. Their time axes are `result.time` and `result.askna_time`.

## Cold pressor test

The paper’s CPT protocol used 120 s before, 90 s during, and 120 s after hand immersion. Any phase durations can be passed explicitly:

```python
result = pyskna.analyze_cpt(
    skna,
    fs,
    time=time,
    ecg=ecg,                  # optional
    cpt_start=120,
    cpt_stop=210,
    analysis_start=0,
    analysis_stop=330,
    skna_options={"unit_scale": 1},
    ecg_options={"lowcut": 0.5, "highcut": 35},
)

print(result.threshold)
print(result.overdrive_rate)
print(result.phases["before"])
print(result.phases["during"])
print(result.phases["after"])
print(result.heart_rate_increase)  # present when ECG was supplied

pyskna.plot_cpt_analysis(result)
pyskna.plot_cpt_phase_summary(result)
```

The time-series figure shades the CPT interval and shows the baseline-derived threshold. The summary figure compares mean ± SEM aSKNA before, during, and after CPT.

## Simultaneous SKNA and ECG

```python
skna_result, ecg_result = pyskna.analyze_simultaneous(
    skna,
    ecg,
    fs,
    time=time,
    skna_options={"baseline": (0, 120)},
    ecg_options={"lowcut": 0.5, "highcut": 35, "notch": 60},
)

print(ecg_result.hrv)
pyskna.plot_simultaneous(skna_result, ecg_result)
```

For the exact choices encoded in the supplied MATLAB script:

```python
skna_result, ecg_result = pyskna.analyze_matlab_workflow(
    time,
    skna_mv,
    ecg_mv,
    baseline=(11.5, 71.5),
    skna_unit_scale=1000,
    ecg_peak_height=0.08,
)
```

The source MATLAB file ends partway through its HR phase-indexing section. PySKNA ports every complete operation and supplies a finished, tested before/during/after CPT summary through `analyze_cpt()`.

## Choosing the iSKNA definition

`integrate_skna()` supports two explicit definitions because published and legacy workflows use the word “integrate” differently:

- `method="leaky"` (default) applies a causal exponential envelope with a 100 ms time constant and retains amplitude units. This matches common SNA processing and the structure of the supplied MATLAB recurrence.
- `method="moving_integral"` calculates a trailing Riemann integral over 100 ms and returns amplitude-seconds, a literal implementation of summed/integrated rSKNA.

Always report the selected method and time constant. The MATLAB compatibility function intentionally uses the source script’s 10 ms value, even though its comment says 100 ms.

## Function reference

### SKNA processing

| Function | Purpose |
|---|---|
| `polynomial_detrend(signal, degree=3, time=None, keep_mean=True)` | Removes slow polynomial drift. Cubic detrending and mean restoration match the MATLAB script. |
| `bandpass_filter(signal, fs, lowcut, highcut, order=3)` | General zero-phase Butterworth band-pass filter with cutoff and Nyquist validation. |
| `filter_skna(signal, fs, lowcut=500, highcut=1000, order=3)` | Extracts the standard high-frequency SKNA band. |
| `rectify_skna(signal)` | Performs full-wave rectification to produce rSKNA. |
| `integrate_skna(rskna, fs, time_constant=0.1, method="leaky")` | Produces iSKNA with a leaky envelope or moving Riemann integral. |
| `average_skna(iskna, fs, window_seconds=5, time=None)` | Computes non-overlapping aSKNA windows and their center times. |
| `burst_threshold(values, sigma=3, ddof=1)` | Calculates baseline mean + sigma × sample SD; the paper uses sigma=3. |
| `overdrive_rate(during_mean, threshold)` | Implements the paper’s percentage CPT overdrive formula. |
| `analyze_skna(...)` | Runs detrending, filtering, rSKNA, iSKNA, aSKNA, and optional baseline thresholding in one call. |

### SNA-inspired utilities

| Function | Purpose |
|---|---|
| `rms_envelope(signal, fs, window_seconds=0.2)` | Produces a moving RMS neurogram used in multi-unit SNA inspection. |
| `artifact_mask(signal, z_threshold=8, padding_samples=0)` | Flags extreme-amplitude samples with a robust median/MAD score and optional padding. |
| `detect_bursts(signal, fs, threshold, time=None, min_duration=0)` | Detects contiguous threshold crossings and reports burst count, frequency/min, peak, duration, and area above threshold. |

These additions are inspired by conventional capabilities described for the open Microneurography Toolbox: RMS/integrated neurograms, threshold-and-duration burst detection, and conventional burst amplitude, duration, area, and frequency summaries. They are implemented independently in Python and should not be treated as validated replacements for microneurography-specific software.

### ECG and heart rate

| Function | Purpose |
|---|---|
| `notch_filter(signal, fs, frequency=60, quality_factor=30)` | Suppresses power-line interference. |
| `filter_ecg(signal, fs, lowcut=0.5, highcut=35, ...)` | Applies ECG band-pass and optional notch filtering. Use 5–20 Hz for MATLAB compatibility or 0.05–35 Hz for long paper-style records. |
| `detect_r_peaks(ecg, fs, min_distance=0.5, height=None, prominence=None)` | Detects R peaks using SciPy peak criteria. |
| `instantaneous_heart_rate(peak_times)` | Converts RR intervals to bpm. |
| `bin_heart_rate(hr_time, heart_rate, start, stop, window_seconds=5)` | Averages HR into fixed bins aligned with aSKNA. Empty bins are `NaN`. |
| `hrv_time_domain(peak_times)` | Returns mean RR, SDNN, RMSSD, pNN50, and mean HR. |
| `analyze_ecg(...)` | Runs detrending, ECG filtering, R-peak detection, HR, 5 s binning, and HRV. |

### Experiments, compatibility, and I/O

| Function | Purpose |
|---|---|
| `analyze_cpt(...)` | Calculates baseline threshold, before/during/after aSKNA statistics, overdrive rate, and optional simultaneous HR response. |
| `analyze_simultaneous(skna, ecg, fs, ...)` | Processes synchronous SKNA and ECG arrays together. |
| `analyze_matlab_workflow(time, skna, ecg, ...)` | Reproduces the complete processing choices from the supplied MATLAB script. |
| `load_delimited(path, ...)` | Loads time and signal columns from CSV, TSV, TXT, or numeric LVM data. |
| `load_lvm(path, ...)` | Detects an LVM header and loads the chosen time and signal columns. |
| `sampling_frequency(time)` | Infers `fs` and rejects materially irregular timestamps. |

### Visualization

All plotting functions return `(figure, axes)` and do not call `plt.show()`, so callers can restyle, save, embed, or test them.

| Function | Purpose |
|---|---|
| `plot_processing_steps(result)` | Six aligned panels for raw, detrended, filtered, rSKNA, iSKNA, and aSKNA. |
| `plot_askna_distribution(result)` | Relative-frequency histogram with mean and optional threshold. |
| `plot_ecg_analysis(result)` | Filtered ECG/R peaks and binned heart rate. |
| `plot_simultaneous(skna_result, ecg_result)` | Aligned SKNA, aSKNA, ECG, and HR panels. |
| `plot_cpt_analysis(result)` | Before/during/after CPT time series, threshold, ECG, and HR when available. |
| `plot_cpt_phase_summary(result)` | Mean ± SEM aSKNA across CPT phases. |

## Saving results and figures

Result objects are frozen dataclasses, which makes analysis settings and outputs explicit. Arrays can be saved with NumPy, and figures with Matplotlib:

```python
np.savez(
    "analysis.npz",
    askna_time=result.askna_time,
    askna=result.askna,
    threshold=result.threshold,
)

fig, _ = pyskna.plot_processing_steps(result)
fig.savefig("processing-steps.png", dpi=300, bbox_inches="tight")
```

## Methodological cautions

- A 1000 Hz upper cutoff is invalid at exactly 2000 Hz sampling because it equals Nyquist. Use a lower cutoff (the MATLAB script uses 999 Hz) or, preferably, acquire faster.
- `filtfilt`-style zero-phase filtering is noncausal and intended for offline analysis.
- Short records can produce unstable low-frequency ECG filtering and do not support reliable HRV.
- R-peak height is unit-dependent. Prefer prominence or set a height appropriate to your acquisition chain.
- The threshold depends on baseline length, artifacts, averaging window, and sample-standard-deviation convention (`ddof=1`).
- A positive overdrive rate means the **mean** during-CPT aSKNA exceeds the baseline burst threshold; it is not the fraction of windows above threshold. That fraction is also reported in each `PhaseStatistics` object.
- High-frequency ECG, EMG, motion, poor electrode coupling, mains interference, and saturation can bias SKNA. Visual review and protocol-specific quality control are essential.

## Development and verification

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

The test suite covers filter behavior, rectification/integration, aSKNA windows, threshold and overdrive equations, burst metrics, ECG/HR extraction, CPT summaries, MATLAB compatibility, and plotting smoke tests. See [`examples/basic_analysis.py`](examples/basic_analysis.py) for a complete synthetic example.

## References

1. Hsu D-Y, et al. Flexible Electronic Sensor for Noncontact Simultaneous Measurement of Capacitive Skin Sympathetic Nerve Activity (cSKNA) and cECG. *IEEE Transactions on Instrumentation and Measurement*. 2025;74. DOI: [10.1109/TIM.2025.3544747](https://doi.org/10.1109/TIM.2025.3544747).
2. Doytchinova A, et al. Simultaneous noninvasive recording of skin sympathetic nerve activity and electrocardiogram. *Heart Rhythm*. 2017;14(1):25–33. DOI: [10.1016/j.hrthm.2016.09.019](https://doi.org/10.1016/j.hrthm.2016.09.019).
3. D’Anna C, et al. An open computational toolbox to analyze multi- and single-unit sympathetic nerve activity in microneurography. *Scientific Reports*. 2024. [Open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11184970/).

## Contributing and license

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md). PySKNA is released under the [MIT License](LICENSE).

