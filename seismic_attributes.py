# seismic_attributes.py
# helper functions to compute trace based seismic attributes
# (hilbert / complex trace attributes, spectral freq, ormsby bandpass, amplitude integration)
# used by all the notebooks in this repo
#
# Soutik Bhakta, IIT Bombay

import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import tukey
from scipy.ndimage import uniform_filter1d

# NOTE: attributes are computed with a nominal 1 ms sample interval (this is what we used
# in the paper). Our traces are actually 2 ms, so every frequency in Hz in the attribute
# names (filter bands, spectral frequency, inst. frequency) is 2x the true frequency,
# e.g. "Filter_135_140_145_150" = 67.5-75 Hz band of the real data.
# Keep 1000 to reproduce the paper results (changing it rescales some attributes).
SAMPLE_RATE_HZ = 1000


def _pad_reflect(x, pad_len=None):
    if pad_len is None:
        pad_len = max(256, len(x) // 2)
    return np.pad(x, pad_len, mode='reflect'), pad_len


def complex_trace(x):
    # analytic signal -> envelope, inst. phase, inst. freq, quadrature
    # padding is done to reduce edge effect of hilbert transform
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    xp, pl = _pad_reflect(x)
    z = hilbert(xp)[pl:pl + n]

    env = np.abs(z)
    quad = np.imag(z)
    phase = np.unwrap(np.angle(z))
    dt = 1.0 / SAMPLE_RATE_HZ
    freq = np.gradient(phase, dt) / (2.0 * np.pi)

    # where envelope is ~0 the inst. freq is garbage, so interpolate there
    thr = 1e-6 * np.max(env) if np.max(env) > 0 else 1e-12
    bad = env < thr
    if np.any(bad):
        good = ~bad
        if np.any(good):
            freq[bad] = np.interp(np.where(bad)[0], np.where(good)[0], freq[good])
        else:
            freq[:] = 0.0
    nyq = SAMPLE_RATE_HZ / 2.0
    freq = np.clip(freq, -nyq, nyq)
    return env, phase, freq, quad


def freq_derivative(x, order=1):
    # derivative in frequency domain, with a gaussian taper near nyquist
    n = len(x)
    dt = 1.0 / SAMPLE_RATE_HZ
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=dt)
    nyq = SAMPLE_RATE_HZ / 2.0
    op = (1j * 2 * np.pi * f) ** order
    taper = np.exp(-0.5 * ((f - 0.8 * nyq) / (0.1 * nyq)) ** 2)
    taper[f <= 0.8 * nyq] = 1.0
    return np.fft.irfft(X * op * taper, n=n)


SMOOTHER_LENGTHS = [200, 100, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5,
                    4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2]


def integrate_trace(x, smoother_length, use_abs=False):
    # running sum of the trace (or abs trace) minus a running mean trend
    dt = 1.0 / SAMPLE_RATE_HZ
    s = np.abs(x) if use_abs else x
    c = np.cumsum(s) * dt
    if smoother_length < 1.0:
        w = max(1, int(round(smoother_length * len(x))))
    else:
        w = max(1, int(round(smoother_length)))
    w = min(w, len(x))
    if w >= len(x):
        return c - np.mean(c)
    return c - uniform_filter1d(c, size=w, mode='reflect')


# window / step / taper (in samples) for the short time fourier transform
STFT_CONFIGS = [
    (256, 128, 64), (128, 64, 32), (64, 32, 16), (32, 16, 16), (32, 16, 8),
    (16, 8, 8), (16, 8, 4), (8, 4, 4), (4, 2, 2), (4, 1, 1), (2, 1, 1),
]


def spectral_frequency(x, win, step, taper, mode="average"):
    # mean (average) or dominant frequency in a sliding tukey window
    n = len(x)
    dt = 1.0 / SAMPLE_RATE_HZ
    win = max(2, min(win, n))
    step = max(1, step)
    window = tukey(win, alpha=min(1.0, 2.0 * taper / win))
    pad_l = (win - 1) // 2
    pad_r = (win - 1) - pad_l
    xp = np.pad(x, (pad_l, pad_r), mode='reflect')

    starts = np.arange(0, n, step)
    freqs = np.fft.rfftfreq(win, d=dt)
    centers = np.clip(starts + win // 2 - pad_l, 0, n - 1).astype(np.float64)
    vals = np.zeros(len(starts))
    for i, s0 in enumerate(starts):
        seg = xp[s0:s0 + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        spec = np.abs(np.fft.rfft(seg * window, n=win)) ** 2
        p = np.sum(spec)
        if p < 1e-30:
            vals[i] = 0.0
        elif mode == "average":
            vals[i] = np.sum(freqs * spec) / p
        else:
            vals[i] = freqs[np.argmax(spec)]
    if len(starts) == 1:
        return np.full(n, vals[0])
    return np.interp(np.arange(n), centers, vals)


# ormsby corner frequencies f1-f2-f3-f4 (Hz), 5 Hz to 250 Hz
FILTER_SPECS = [(f, f + 5, f + 10, f + 15) for f in [5] + list(range(15, 240, 10))]


def ormsby_filter(x, f1, f2, f3, f4):
    n = len(x)
    dt = 1.0 / SAMPLE_RATE_HZ
    xp, pl = _pad_reflect(x, pad_len=max(128, n // 4))
    N = len(xp)
    X = np.fft.rfft(xp)
    f = np.fft.rfftfreq(N, d=dt)
    H = np.zeros_like(f)
    up = (f > f1) & (f < f2)
    H[up] = (f[up] - f1) / (f2 - f1)
    H[(f >= f2) & (f <= f3)] = 1.0
    dn = (f > f3) & (f < f4)
    H[dn] = (f4 - f[dn]) / (f4 - f3)
    return np.fft.irfft(X * H, n=N)[pl:pl + n]


def all_attributes(x, prefix=""):
    """returns dict {name: array} with 100 attributes of one trace"""
    x = np.asarray(x, dtype=np.float64)
    dt = 1.0 / SAMPLE_RATE_HZ
    env, phase, freq, quad = complex_trace(x)
    p = prefix
    a = {}
    a[p + "Amplitude"] = x.copy()
    a[p + "Amplitude_Envelope"] = env
    a[p + "Inst_Phase"] = phase
    a[p + "Cosine_Inst_Phase"] = np.cos(phase)
    a[p + "Inst_Frequency"] = freq
    a[p + "Quadrature_Trace"] = quad
    a[p + "Apparent_Polarity"] = np.sign(x) * env
    a[p + "Amp_Weighted_Phase"] = env * phase
    a[p + "Amp_Weighted_Cosine_Phase"] = env * np.cos(phase)
    a[p + "Amp_Weighted_Frequency"] = env * freq
    a[p + "Deriv_Inst_Amplitude"] = np.gradient(env, dt)
    a[p + "2nd_Deriv_Inst_Amplitude"] = np.gradient(np.gradient(env, dt), dt)
    a[p + "Derivative"] = freq_derivative(x, 1)
    a[p + "Second_Derivative"] = freq_derivative(x, 2)

    for (w, s, t) in STFT_CONFIGS:
        a[p + f"AvgFreq_w{w}_s{s}_t{t}"] = spectral_frequency(x, w, s, t, "average")
    for (w, s, t) in STFT_CONFIGS:
        a[p + f"DomFreq_w{w}_s{s}_t{t}"] = spectral_frequency(x, w, s, t, "dominant")
    for (f1, f2, f3, f4) in FILTER_SPECS:
        a[p + f"Filter_{f1}_{f2}_{f3}_{f4}"] = ormsby_filter(x, f1, f2, f3, f4)
    for sl in SMOOTHER_LENGTHS:
        a[p + "Integrate_sm" + str(sl).replace('.', 'p')] = integrate_trace(x, sl, False)
    for sl in SMOOTHER_LENGTHS:
        a[p + "IntAbsAmp_sm" + str(sl).replace('.', 'p')] = integrate_trace(x, sl, True)
    return a


def attribute_matrix(X_raw, n_samples_trace):
    """
    X_raw : (n_total, n_channels) array, traces stacked one after another
    every channel gets the full attribute set (prefix Ch1_, Ch2_ ...)
    returns (n_total, n_feat) matrix and list of names
    """
    n_total, n_ch = X_raw.shape
    n_tr = n_total // n_samples_trace
    blocks = []
    names = None
    for t in range(n_tr):
        s, e = t * n_samples_trace, (t + 1) * n_samples_trace
        d = {}
        for c in range(n_ch):
            d[f"RAW_Channel_{c+1}"] = X_raw[s:e, c]
        for c in range(n_ch):
            d.update(all_attributes(X_raw[s:e, c], prefix=f"Ch{c+1}_"))
        if names is None:
            names = list(d.keys())
        blocks.append(np.column_stack([d[k] for k in names]))
    return np.vstack(blocks).astype(np.float32), names


# names used in the paper for some attributes (for plots)
def nice_name(key):
    k = key.split("_", 1)[1] if key.startswith("Ch") else key
    fixed = {
        "Amplitude": "Seismic Amplitude",
        "Amplitude_Envelope": "Instantaneous Amplitude",
        "Inst_Phase": "Instantaneous Phase",
        "Cosine_Inst_Phase": "Cosine of Instantaneous Phase",
        "Inst_Frequency": "Instantaneous Frequency",
        "Quadrature_Trace": "Quadrature Trace",
        "Apparent_Polarity": "Apparent Polarity",
        "Amp_Weighted_Phase": "Amplitude Weighted Phase",
        "Amp_Weighted_Cosine_Phase": "Amplitude Weighted Cosine Phase",
        "Amp_Weighted_Frequency": "Amplitude Weighted Frequency",
        "Deriv_Inst_Amplitude": "First Derivative of Envelope",
        "2nd_Deriv_Inst_Amplitude": "Second Derivative of Envelope",
        "Derivative": "Derivative",
        "Second_Derivative": "Second Derivative",
    }
    if k in fixed:
        return fixed[k]
    if k.startswith("AvgFreq_w") or k.startswith("DomFreq_w"):
        w, s, t = k.split("_w")[1].replace("_s", " ").replace("_t", " ").split()
        lab = "Mean" if k.startswith("Avg") else "Dominant"
        return f"{lab} Spectral Frequency (win={w}, step={s}, taper={t})"
    if k.startswith("Filter_"):
        f = k.split("_")[1:]
        return f"Bandpass Filtered Amplitude ({f[0]}-{f[3]} Hz)"
    if k.startswith("Integrate_sm"):
        return "Amplitude Integration (smoother=%s)" % k[12:].replace("p", ".")
    if k.startswith("IntAbsAmp_sm"):
        return "Absolute Amplitude Integration (smoother=%s)" % k[12:].replace("p", ".")
    return k
