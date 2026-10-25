"""End-to-end PySKNA example with synthetic SKNA and ECG data."""

import matplotlib.pyplot as plt
import numpy as np

import pyskna


fs = 4000.0
time = np.arange(int(40 * fs)) / fs

# A 700-Hz carrier with more sympathetic activity during a simulated CPT.
amplitude = np.where((time >= 20) & (time < 30), 2.0, 1.0)
skna = amplitude * np.sin(2 * np.pi * 700 * time)

# Synthetic ECG with CPT-related acceleration (replace both arrays with recordings).
ecg = np.zeros_like(time)
beat_times = np.concatenate(
    [
        np.arange(0.5, 20, 60 / 72),
        np.arange(20.5, 30, 60 / 84),
        np.arange(30.5, 40, 60 / 78),
    ]
)
for beat in beat_times:
    ecg += np.exp(-0.5 * ((time - beat) / 0.012) ** 2)

result = pyskna.analyze_cpt(
    skna,
    fs,
    time=time,
    ecg=ecg,
    cpt_start=20,
    cpt_stop=30,
    ecg_options={"notch": None},
)

print("aSKNA overdrive rate:", result.overdrive_rate)
print("HR increase:", result.heart_rate_increase)
pyskna.plot_cpt_analysis(result)
plt.show()

