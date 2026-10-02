# make_synthetic_data.py
# makes a small synthetic data set with the same format as our (confidential) well data,
# so that the notebooks can be run and tested by anyone.
#
#   example_data/synthetic_well_trace.csv          one trace, 201 samples
#   example_data/synthetic_section_amplitude.npy   (20, 201) seismic section
#   example_data/synthetic_section_impedance.npy   (20, 201) "inverted" impedance section
#
# the numbers are NOT real data. density from a random layered model, porosity from density
# (carbonate matrix 2.71 g/cc, fluid 1.0 g/cc), velocity from Gardner, seismic = reflectivity * ricker
#
# run:  python make_synthetic_data.py

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 201
DT = 0.002          # 2 ms
os.makedirs("example_data", exist_ok=True)


def ricker(f=30.0, dt=DT, n=61):
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def layered_density(n, shift=0.0):
    # blocky layers + small noise
    rho = np.zeros(n)
    i = 0
    while i < n:
        th = rng.integers(5, 25)
        rho[i:i + th] = rng.uniform(2.35, 2.62)
        i += th
    # a low density (porous) zone in the middle, moves a bit along the section
    c = int(110 + shift)
    rho[c - 8:c + 8] -= 0.12
    rho += rng.normal(0, 0.01, n)
    return np.convolve(rho, np.ones(3) / 3, mode="same")


def make_trace(shift=0.0):
    rho = layered_density(N, shift)
    rho[0], rho[-1] = rho[1], rho[-2]
    phi = np.clip((2.71 - rho) / (2.71 - 1.0), 0.01, 0.35)
    vp = (rho / 0.31) ** 4                      # gardner, m/s
    ai = vp * rho
    r = np.zeros(N)
    r[1:] = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1])
    seis = np.convolve(r, ricker(), mode="same") * 1e4
    seis += rng.normal(0, 0.05 * seis.std(), N)
    # "inverted" impedance = smoothed true impedance + noise (like an inversion result)
    ai_inv = np.convolve(ai, np.ones(5) / 5, mode="same")
    ai_inv[:2], ai_inv[-2:] = ai[:2], ai[-2:]
    ai_inv += rng.normal(0, 0.01 * ai.std(), N)
    return seis, ai_inv, rho, phi


seis, ai_inv, rho, phi = make_trace()
df = pd.DataFrame({
    "Time_ms": 1700 + np.arange(N) * 2,
    "Seismic_Amplitude": seis,
    "Inverted_Impedance_HPSO": ai_inv,
    "Density": rho,
    "Porosity": phi,
})
df.to_csv("example_data/synthetic_well_trace.csv", index=False, float_format="%.6f")

amp_sec, imp_sec = [], []
for k in range(20):
    s, a, _, _ = make_trace(shift=8 * np.sin(k / 4))
    amp_sec.append(s); imp_sec.append(a)
np.save("example_data/synthetic_section_amplitude.npy", np.array(amp_sec, dtype=np.float32))
np.save("example_data/synthetic_section_impedance.npy", np.array(imp_sec, dtype=np.float32))

print(df.describe().round(3))
print("written to example_data/")
