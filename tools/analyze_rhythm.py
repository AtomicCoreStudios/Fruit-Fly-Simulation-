"""Rhythmicity of probe traces (tools/ CPG test, Pugliese et al. 2025 protocol): per group, the dominant frequency in
2-30 Hz and a rhythmicity score = fraction of 1-50 Hz power within +-1 Hz of that peak (0 = flat, ~1 = pure sine).
usage: analyze_rhythm.py <probe csv> [t_start_ms]"""
import sys
import numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
d = d[d.t_ms > t0]
t = d.t_ms.values / 1000.0; fs = 1.0 / np.median(np.diff(t))
rows = []
for k in d.columns:
    if k in ("t_ms", "threat"):
        continue
    x = d[k].values.astype(float); m = x.mean()
    if m < 0.5:
        rows.append((k, m, np.nan, 0.0)); continue
    x = (x - m) * np.hanning(len(x))
    p = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / fs)
    band = (f >= 1) & (f <= 50); sel = (f >= 2) & (f <= 30)
    fp = f[sel][np.argmax(p[sel])]
    score = p[(f >= fp - 1) & (f <= fp + 1)].sum() / max(p[band].sum(), 1e-9)
    rows.append((k, m, fp, score))
r = pd.DataFrame(rows, columns=["group", "mean_hz", "peak_hz", "rhythmicity"])
print(f"sampling {fs:.0f} Hz, {len(d)} samples")
print(r.sort_values("rhythmicity", ascending=False).round(2).to_string(index=False))
