"""Per-motor-neuron recruitment and phase from a probe CSV with columns '<type>|<action>|<index>' (live) or
'mn|<type>|<action>|<index>' (offline): active MNs, the common rhythm frequency (strongest active MN), and each
active MN's phase at that frequency. usage: analyze_mn_phase.py <csv> [t_start_ms]"""
import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); d = d[d.t_ms > (float(sys.argv[2]) if len(sys.argv) > 2 else 1000)]
t = d.t_ms.values / 1000; fs = 1 / np.median(np.diff(t))
cols = [c for c in d.columns if c.count("|") >= 2]
act = [c for c in cols if d[c].mean() >= 0.3]
best = (0, 0)
for c in act:
    x = d[c].values; y = (x - x.mean()) * np.hanning(len(x)); p = np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(y), 1 / fs)
    sel = (f >= 3) & (f <= 30); best = max(best, (p[sel].max(), f[sel][np.argmax(p[sel])]))
f0 = best[1]; rows = []
for c in act:
    x = d[c].values; z = np.sum((x - x.mean()) * np.exp(-2j * np.pi * f0 * t)) / len(x)
    k = c.split("|"); rows.append((k[-2], k[-3], round(x.mean(), 1), round(2 * abs(z), 1), round(np.degrees(np.angle(z)))))
r = pd.DataFrame(rows, columns=["action", "type", "mean_hz", "mod_hz", "phase"])
print(f"{len(act)} of {len(cols)} MNs active; rhythm {f0:.1f} Hz")
print(r.sort_values("phase").to_string(index=False))
